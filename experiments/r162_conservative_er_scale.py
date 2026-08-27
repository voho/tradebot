#!/usr/bin/env python
"""R-162 CONSERVATIVE branch: post-vote SCALE multiplier using Kaufman's
(1995) Efficiency Ratio (ER). Direction, citations, non-duplication
argument, kill switches, and the pre-registered decision rule all live in
`experiments/r162_shared.py`'s module docstring (read there first -- this
file does not repeat that reasoning and does not edit that module, which is
frozen/read-only for both branches).

THE MECHANISM, exactly: `r162_shared.build_conservative_target(df, gamma)`
multiplies v4's own unmodified `frac*scale` (`v4_raw_desired`) by
`1 + gamma*(ER_t - ER_ref_t)` (clipped to [0.5, 1.5]), then applies v4's own
unmodified 10% deadband -- the ONLY change from v4 anywhere in this file.
Every number below comes from `r162_shared`'s frozen primitives; this file
only sequences the pre-registered steps and prints the results.

This is the CONSERVATIVE branch only. The NOVEL branch (per-anchor ER vote
weighting) is a sibling agent's disjoint file; `delayed_flip_diagnostic` is
NOVEL-branch-only and is not run here.

Run with: python experiments/r162_conservative_er_scale.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r162_shared import (  # noqa: E402
    CONST_CAP_R2_THRESH,
    ETH_SLICE_NAME,
    FEE_TIER,
    FUTURES,
    GRID,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    PRIMARY,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TargetStrategy,
    build_conservative_target,
    causal_truncation_probe_series,
    clears_bar,
    compare,
    er_scale_multiplier,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_target,
)
from scripts.experiment import ev  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

CONFIGS_EVALUATED = 0  # incremented as each build_target(df) call is scored


# ================================================================== STEP 1
# Sanity / causality kill switches (A1 + causal truncation probe).
# ==================================================================

def step1_sanity_checks(btc_full: pd.DataFrame) -> dict:
    print("\n" + "=" * 88)
    print("STEP 1 -- Sanity / causality kill switches")
    print("=" * 88)
    # A1: gamma=0.0 must reproduce v4_target bit-for-bit, on a subset for speed.
    sub = btc_full.iloc[:150_000]  # ~520 days: covers the 80d ER window and
                                    # most of the 365d reference window, fast.
    cand0 = build_conservative_target(sub, 0.0)
    ctrl0 = v4_target(sub)
    identical = np.array_equal(cand0, ctrl0)
    max_abs_diff = float(np.max(np.abs(cand0 - ctrl0)))
    r2_identity = r_squared(cand0, ctrl0)
    print(f"A1 identity check (gamma=0.0 vs v4_target, n={len(sub)} bars):")
    print(f"    bit-for-bit equal : {identical}")
    print(f"    max |diff|        : {max_abs_diff:.3e}")
    print(f"    R^2               : {r2_identity:.10f}")
    assert identical, "A1 KILL SWITCH FAILED: gamma=0.0 does not reproduce v4_target bit-for-bit"

    # Causal truncation probe at PRIMARY gamma.
    t0 = time.time()
    probe_ok = causal_truncation_probe_series(
        lambda d: build_conservative_target(d, PRIMARY), sub)
    print(f"Causal truncation probe (gamma={PRIMARY}): PASS={probe_ok} "
          f"[{time.time() - t0:.1f}s]")
    assert probe_ok, "CAUSALITY PROBE FAILED at PRIMARY gamma"

    return {"a1_identical": identical, "a1_max_abs_diff": max_abs_diff,
            "truncation_probe_ok": probe_ok}


# ================================================================== STEP 2
# A2 kill switch: non-inertness / non-collinearity with v4's own exposure.
# ==================================================================

def step2_a2_kill_switch(btc_full: pd.DataFrame) -> dict:
    print("\n" + "=" * 88)
    print("STEP 2 -- A2 kill switch (non-collinearity with v4's own exposure)")
    print("=" * 88)
    train = btc_full[(btc_full.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc_full.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))]
    raw = v4_raw_desired(train)
    results = {}
    for gamma in GRID:
        mult = er_scale_multiplier(train, gamma)
        r2 = r_squared(raw * mult, raw)
        results[gamma] = r2
        flag = "OK (below cap)" if r2 < CONST_CAP_R2_THRESH else "FAIL (>= cap -- relabeling)"
        print(f"    gamma={gamma:<4} R^2(candidate raw exposure vs v4 raw exposure) "
              f"= {r2:.6f}  [{flag}]")

    mult_primary = er_scale_multiplier(train, PRIMARY)
    print(f"\n    ER-scale multiplier at PRIMARY (gamma={PRIMARY}) on inner-train:")
    print(f"        min={mult_primary.min():.4f} max={mult_primary.max():.4f} "
          f"mean={mult_primary.mean():.4f} std={mult_primary.std():.4f}")

    a2_pass = results[PRIMARY] < CONST_CAP_R2_THRESH
    print(f"\n    A2 kill switch (PRIMARY gamma={PRIMARY}): "
          f"R^2={results[PRIMARY]:.6f} < {CONST_CAP_R2_THRESH} -> {'PASS' if a2_pass else 'FAIL'}")
    if not a2_pass:
        print("    ==> A2 FAILS. Per pre-registration, this construction never meaningfully")
        print("        moves exposure relative to v4's own raw path at the primary setting --")
        print("        it is a relabeling of v4, not a tested mechanism. Continuing through the")
        print("        remaining pre-registered steps regardless (a null result is still reported")
        print("        in full), but no Sharpe/growth number below should be read as a real effect.")
    return {"r2_by_gamma": results, "a2_pass": a2_pass}


# ================================================================== STEP 3
# Sweep: 4 gammas x 2 markets x 3 slices = 24 cells.
# ==================================================================

def step3_sweep(btc_full: pd.DataFrame, eth_full: pd.DataFrame) -> list[dict]:
    global CONFIGS_EVALUATED
    print("\n" + "=" * 88)
    print("STEP 3 -- Sweep: gamma in GRID x {spot, futures_5x} x "
          "{inner_train, inner_val, eth_replication}")
    print("=" * 88)
    all_rows: list[dict] = []
    for gamma in GRID:
        t0 = time.time()
        rows = compare(lambda df, g=gamma: build_conservative_target(df, g),
                        label=f"r162_cons_g{gamma}", btc=btc_full, eth=eth_full)
        CONFIGS_EVALUATED += len(rows)
        print(f"\n-- gamma={gamma} [{time.time() - t0:.0f}s, {len(rows)} cells] --")
        print_rows(rows)
        all_rows.extend(rows)
    return all_rows


# ================================================================== STEP 4
# Fee-tier robustness: PRIMARY gamma at FEE_TIER=0.40% on both markets,
# inner-validation slice only -- 2 cells (matches the pre-registration's
# "+2 cells" fee-tier re-run of the finalist config).
# ==================================================================

def step4_fee_tier(btc_full: pd.DataFrame, gamma: float = PRIMARY) -> list[dict]:
    global CONFIGS_EVALUATED
    print("\n" + "=" * 88)
    print(f"STEP 4 -- Fee-tier robustness: gamma={gamma} @ FEE_TIER={FEE_TIER:.2%}, "
          "inner_val, both markets (2 cells)")
    print("=" * 88)
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    cand = TargetStrategy(lambda df: build_conservative_target(df, gamma),
                           name=f"r162_cons_g{gamma}_fee{FEE_TIER:.2%}")
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    rows = []
    for market in fee_markets:
        a = run_slice(cand, btc_full, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        b = run_slice(ctrl, btc_full, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        pr = paired_diff(a.daily, b.daily)
        exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                     if b.mean_abs_exposure else float("nan"))
        vol_ratio = (a.realized_vol / b.realized_vol if b.realized_vol else float("nan"))
        rows.append({
            "label": f"r162_cons_g{gamma}_fee{FEE_TIER:.2%}", "slice": "inner_val",
            "market": market.name,
            "cand_final": a.final_balance, "ctrl_final": b.final_balance,
            "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
            "d_log_growth": a.log_growth - b.log_growth,
            "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
            "d_sharpe": a.sharpe - b.sharpe,
            "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
            "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
            "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
            "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
            "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                            if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
            "boot_d_loggrowth": pr.diff.point, "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
            "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        })
    CONFIGS_EVALUATED += len(rows)
    print_rows(rows)
    return rows


# ================================================================== STEP 5
# Decision rule, applied exactly as written in r162_shared's module
# docstring.
# ==================================================================

def apply_decision_rule(sweep_rows: list[dict]) -> dict:
    print("\n" + "=" * 88)
    print("STEP 5 -- Decision rule (r162_shared.py's frozen table)")
    print("=" * 88)

    inner_val_rows = [r for r in sweep_rows if r["slice"] == "inner_val"]
    eth_rows = [r for r in sweep_rows if r["slice"] == ETH_SLICE_NAME]

    def gamma_of(row: dict) -> float:
        # label is "r162_cons_g<gamma>"
        return float(row["label"].rsplit("_g", 1)[1])

    clear_by_market: dict[str, list[tuple[float, dict]]] = {"spot": [], "futures_5x": []}
    for row in inner_val_rows:
        if clears_bar(row):
            clear_by_market[row["market"]].append((gamma_of(row), row))

    clear_spot = len(clear_by_market["spot"]) > 0
    clear_futures = len(clear_by_market["futures_5x"]) > 0
    print(f"CLEAR(spot)        = {clear_spot}  "
          f"(clearing gammas: {[g for g, _ in clear_by_market['spot']]})")
    print(f"CLEAR(futures_5x)  = {clear_futures}  "
          f"(clearing gammas: {[g for g, _ in clear_by_market['futures_5x']]})")

    # ETH sign-replication: for whichever market cleared (if any), the SAME
    # gamma/market run on eth_replication must show the SAME sign of
    # d_log_growth vs v4.
    sign_ok = True
    plateau_ok = True
    replication_notes = []

    def eth_row_for(gamma: float, market: str) -> dict | None:
        for r in eth_rows:
            if r["market"] == market and abs(gamma_of(r) - gamma) < 1e-9:
                return r
        return None

    cleared_markets = [m for m, ok in (("spot", clear_spot), ("futures_5x", clear_futures)) if ok]
    for market in cleared_markets:
        for gamma, row in clear_by_market[market]:
            eth_row = eth_row_for(gamma, market)
            if eth_row is None:
                replication_notes.append(f"no ETH row for gamma={gamma}/{market}")
                continue
            btc_sign = np.sign(row["d_log_growth"])
            eth_sign = np.sign(eth_row["d_log_growth"])
            note = (f"market={market} gamma={gamma}: BTC d_log_growth="
                    f"{row['d_log_growth']:+.4f} (sign {btc_sign:+.0f}), "
                    f"ETH d_log_growth={eth_row['d_log_growth']:+.4f} (sign {eth_sign:+.0f})")
            replication_notes.append(note)
            if btc_sign != 0 and eth_sign != 0 and btc_sign != eth_sign:
                sign_ok = False

    # Plateau check: non-zero gamma values that clear (on any market) must
    # share the same sign of d_log_growth on inner-validation.
    nonzero_clearing_signs = set()
    for market in ("spot", "futures_5x"):
        for gamma, row in clear_by_market[market]:
            if gamma != 0.0:
                nonzero_clearing_signs.add(np.sign(row["d_log_growth"]))
    if len(nonzero_clearing_signs) > 1:
        plateau_ok = False

    gate_ok = sign_ok and plateau_ok
    print(f"\nETH sign-replication notes:")
    for n in replication_notes:
        print(f"    {n}")
    print(f"\nsign_ok (no inversion on the market that cleared) = {sign_ok}")
    print(f"plateau_ok (clearing non-zero gammas share one sign)  = {plateau_ok}")
    print(f"GATE_OK = sign_ok AND plateau_ok = {gate_ok}")

    if not gate_ok:
        verdict = "REJECT"
    elif not clear_spot and not clear_futures:
        verdict = "REJECT"
    elif clear_spot and clear_futures:
        verdict = "PROMOTE"
    else:
        verdict = "PARTIAL"

    print(f"\nDecision table row matched:")
    print(f"    | GATE_OK={gate_ok!s:<5} | CLEAR(spot)={clear_spot!s:<5} "
          f"| CLEAR(futures)={clear_futures!s:<5} | Verdict = {verdict} |")

    return {
        "clear_spot": clear_spot, "clear_futures": clear_futures,
        "sign_ok": sign_ok, "plateau_ok": plateau_ok, "gate_ok": gate_ok,
        "verdict": verdict,
    }


# ================================================================== STEP 6
# Holdout -- ONLY if GATE_OK and at least one market clears.
# ==================================================================

def step6_holdout_if_warranted(decision: dict) -> list | None:
    print("\n" + "=" * 88)
    print("STEP 6 -- Holdout")
    print("=" * 88)
    if decision["verdict"] == "REJECT":
        print("Gate did NOT clear (verdict=REJECT at inner-validation). Per the pre-registered")
        print("protocol, the holdout is NOT read. Reporting NEGATIVE at the inner-validation")
        print("stage and stopping here.")
        return None

    print(f"Verdict={decision['verdict']} (PARTIAL or PROMOTE) -- reading the holdout ONCE at "
          f"the frozen PRIMARY=1.0 config, no further tuning.")
    strat = TargetStrategy(lambda df: build_conservative_target(df, PRIMARY),
                            name="r162_conservative")
    results = []
    for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        print(f"\n-- holdout on {market_name} --")
        m_cand = ev(strat, market=market, start=OOS_START, tag=f"r162_conservative[{market_name}]")
        m_bh = ev(get_strategy("buy_and_hold"), market=market, start=OOS_START,
                  tag=f"buy_and_hold[{market_name}]")
        m_v4 = ev(get_strategy("kelly_regime_v4"), market=market, start=OOS_START,
                  tag=f"kelly_regime_v4[{market_name}]")
        results.append((market_name, m_cand, m_bh, m_v4))
    return results


# ================================================================== main

if __name__ == "__main__":
    t_start = time.time()
    print("R-162 CONSERVATIVE branch (post-vote ER SCALE multiplier) -- full run")
    print(f"GRID={GRID}  PRIMARY={PRIMARY}  FEE_TIER={FEE_TIER:.2%}  "
          f"SHARPE_NOISE_FLOOR={SHARPE_NOISE_FLOOR}  CONST_CAP_R2_THRESH={CONST_CAP_R2_THRESH}")

    btc_full = load_btc()
    eth_full = load_eth()
    print(f"BTC pre-holdout bars: {len(btc_full)} ({btc_full.index[0]} .. {btc_full.index[-1]})")
    print(f"ETH replication bars: {len(eth_full)} ({eth_full.index[0]} .. {eth_full.index[-1]})")

    step1_sanity_checks(btc_full)
    a2 = step2_a2_kill_switch(btc_full)
    sweep_rows = step3_sweep(btc_full, eth_full)
    fee_rows = step4_fee_tier(btc_full, gamma=PRIMARY)

    print("\n" + "=" * 88)
    print("FULL RESULT TABLE (sweep + fee-tier robustness)")
    print("=" * 88)
    print_rows(sweep_rows + fee_rows)

    decision = apply_decision_rule(sweep_rows)
    holdout = step6_holdout_if_warranted(decision)

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"Configs evaluated: {CONFIGS_EVALUATED} (expected 26: 4 gammas x 2 markets x 3 "
          f"slices = 24, + 2 fee-tier cells)")
    print(f"A2 kill switch (PRIMARY gamma={PRIMARY}): "
          f"R^2={a2['r2_by_gamma'][PRIMARY]:.6f} vs cap {CONST_CAP_R2_THRESH} -> "
          f"{'PASS' if a2['a2_pass'] else 'FAIL -- relabeling of v4, not a tested mechanism'}")
    print(f"Decision rule verdict (inner-validation): {decision['verdict']}")
    print(f"  GATE_OK={decision['gate_ok']}  CLEAR(spot)={decision['clear_spot']}  "
          f"CLEAR(futures_5x)={decision['clear_futures']}")
    if not a2["a2_pass"]:
        print("NOTE: A2 failed at PRIMARY -- per pre-registration, any apparent Sharpe/growth")
        print("      effect above is not attributable to this mechanism moving exposure; it is")
        print("      indistinguishable from v4 itself at this setting. This does not change the")
        print("      mechanical decision-rule verdict above, but it is the honest headline.")
    print(f"Holdout read: {'YES' if holdout is not None else 'NO'}")
    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")
