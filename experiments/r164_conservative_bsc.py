#!/usr/bin/env python
"""R-164 CONSERVATIVE branch: literal Barroso & Santa-Clara (2015)
risk-managed-momentum overlay on `kelly_regime_v4`. Direction, citations,
non-duplication argument, kill switches, and the pre-registered decision
rule all live in `experiments/r164_shared.py`'s module docstring (read
there first -- this file does not repeat that reasoning and does not edit
that module, which is frozen/read-only for both branches).

THE MECHANISM, exactly: `r164_shared.build_conservative_target(df,
window_days)` multiplies v4's own unmodified `frac*scale`
(`v4_raw_desired`) by `sqrt(target_variance / realized_variance)` of the
strategy's OWN payoff (`vote_only_daily_return`), clips to v4's own
max_leverage envelope, then applies v4's own unmodified 10% deadband --
the ONLY change from v4 anywhere in this file. Every number below comes
from `r164_shared`'s frozen primitives; this file only sequences the
pre-registered steps and prints the results.

This is the CONSERVATIVE branch only. The NOVEL branch (Daniel & Moskowitz
2016 panic/calm-bull compound multiplier) is a sibling agent's disjoint
file; `dm_multiplier` / `build_novel_target` / `dm_panic_calm_score` are
NOVEL-branch-only and are not used here.

Run with: python experiments/r164_conservative_bsc.py
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

from experiments.r164_shared import (  # noqa: E402
    BSC_MIN_TARGET_DAYS,
    BSC_MULT_CLIP,
    BSC_PRIMARY,
    BSC_WINDOW_GRID,
    CONST_CAP_R2_THRESH,
    ETH_SLICE_NAME,
    FEE_TIER,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    SPOT,
    V4_MAX_LEVERAGE,
    broadcast_daily_lambda,
    bsc_scale_multiplier,
    build_conservative_target,
    causal_truncation_probe_series,
    clears_bar,
    compare,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    v4_raw_desired,
    v4_target,
)

CONFIGS_EVALUATED = 0  # incremented per compare()-cell (Step 3: 6/grid value; Step 4: 2 fee cells)


# ================================================================== STEP 1
# Sanity / causality kill switches (A1 + causal truncation probe) plus the
# disclosed BSC-multiplier diagnostic (failure mode (5): variance-ratio
# degeneracy).
# ==================================================================

def step1_sanity_checks(btc_full: pd.DataFrame) -> dict:
    print("\n" + "=" * 88)
    print("STEP 1 -- Sanity / causality kill switches + multiplier diagnostic")
    print("=" * 88)
    # A1: window_days=0 must reproduce v4_target bit-for-bit, on a subset for speed.
    sub = btc_full.iloc[:150_000]  # ~520 days: matches R-162/R-163's own convention.
    cand0 = build_conservative_target(sub, 0)
    ctrl0 = v4_target(sub)
    identical = np.array_equal(cand0, ctrl0)
    max_abs_diff = float(np.max(np.abs(cand0 - ctrl0)))
    r2_identity = r_squared(cand0, ctrl0)
    print(f"A1 identity check (window_days=0 vs v4_target, n={len(sub)} bars):")
    print(f"    bit-for-bit equal : {identical}")
    print(f"    max |diff|        : {max_abs_diff:.3e}")
    print(f"    R^2               : {r2_identity:.10f}")
    assert identical, "A1 KILL SWITCH FAILED: window_days=0 does not reproduce v4_target bit-for-bit"
    assert r2_identity == 1.0, "A1 KILL SWITCH FAILED: R^2 != 1.0"
    assert max_abs_diff == 0.0, "A1 KILL SWITCH FAILED: max abs diff != 0.0"

    # Causal truncation probe at PRIMARY window_days, on the FULL BTC
    # pre-holdout frame.
    t0 = time.time()
    probe_ok = causal_truncation_probe_series(
        lambda d: build_conservative_target(d, BSC_PRIMARY), btc_full)
    probe_dt = time.time() - t0
    print(f"\nCausal truncation probe (window_days={BSC_PRIMARY}), "
          f"full BTC pre-holdout frame (n={len(btc_full)} bars): PASS={probe_ok} "
          f"[{probe_dt:.1f}s]")
    assert probe_ok, "CAUSALITY PROBE FAILED at PRIMARY window_days"

    # BSC multiplier diagnostic (failure mode (5)) -- reported BEFORE any
    # Sharpe number, on the full BTC pre-holdout frame, at PRIMARY.
    mult = bsc_scale_multiplier(btc_full, BSC_PRIMARY).dropna()
    n = len(mult)
    frac_lo_clip = float(np.mean(np.isclose(mult.to_numpy(), BSC_MULT_CLIP[0])))
    frac_hi_clip = float(np.mean(np.isclose(mult.to_numpy(), BSC_MULT_CLIP[1])))
    print(f"\nBSC scale-multiplier diagnostic (window_days={BSC_PRIMARY}, "
          f"full BTC pre-holdout frame, n_defined_days={n}):")
    print(f"    min={mult.min():.4f} max={mult.max():.4f} mean={mult.mean():.4f} "
          f"median={mult.median():.4f} std={mult.std():.4f}")
    print(f"    fraction at lower clip ({BSC_MULT_CLIP[0]}) = {frac_lo_clip:.4f}")
    print(f"    fraction at upper clip ({BSC_MULT_CLIP[1]}) = {frac_hi_clip:.4f}")

    return {
        "a1_identical": identical, "a1_max_abs_diff": max_abs_diff,
        "a1_r2": r2_identity,
        "truncation_probe_ok": probe_ok, "truncation_probe_seconds": probe_dt,
        "mult_diagnostic": {
            "n_defined_days": n, "min": float(mult.min()), "max": float(mult.max()),
            "mean": float(mult.mean()), "median": float(mult.median()),
            "std": float(mult.std()), "frac_lo_clip": frac_lo_clip, "frac_hi_clip": frac_hi_clip,
        },
    }


# ================================================================== STEP 2
# A2 kill switch: non-inertness / non-collinearity with v4's own raw
# exposure. Reproduces build_conservative_target's own pre-deadband
# computation (frozen r164_shared.py is never edited).
# ==================================================================

def conservative_raw_exposure(df: pd.DataFrame, window_days: int) -> np.ndarray:
    """v4's own raw_desired multiplied by the BSC scale factor, clipped to
    v4's own max_leverage envelope -- the exact pre-deadband computation
    inside `build_conservative_target`, reproduced here (not edited there)
    so Step 2 can compare it directly against `v4_raw_desired` unclipped."""
    raw = v4_raw_desired(df)
    if window_days <= 0:
        return raw
    mult_daily = bsc_scale_multiplier(df, window_days)
    mult_bars = broadcast_daily_lambda(mult_daily, df.index)
    return np.clip(raw * mult_bars, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)


def step2_a2_kill_switch(btc_full: pd.DataFrame) -> dict:
    print("\n" + "=" * 88)
    print("STEP 2 -- A2 kill switch (non-collinearity with v4's own raw exposure)")
    print("=" * 88)
    train = btc_full[(btc_full.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc_full.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))]
    raw_v4 = v4_raw_desired(train)
    results = {}
    for wd in BSC_WINDOW_GRID:
        cand_raw = conservative_raw_exposure(train, wd)
        r2 = r_squared(cand_raw, raw_v4)
        results[wd] = r2
        flag = "OK (below cap)" if r2 < CONST_CAP_R2_THRESH else "FAIL (>= cap -- relabeling)"
        print(f"    window_days={wd:<4} R^2(candidate raw exposure vs v4 raw exposure) "
              f"= {r2:.6f}  [{flag}]")

    cand_primary = conservative_raw_exposure(train, BSC_PRIMARY)
    diff = cand_primary - raw_v4
    print(f"\n    (candidate_raw - v4_raw) at PRIMARY (window_days={BSC_PRIMARY}) "
          f"on inner-train:")
    print(f"        min={diff.min():.4f} max={diff.max():.4f} mean={diff.mean():.4f} "
          f"std={diff.std():.4f} fraction_nonzero={float(np.mean(np.abs(diff) > 1e-9)):.4f}")

    a2_pass = results[BSC_PRIMARY] < CONST_CAP_R2_THRESH
    print(f"\n    A2 kill switch (PRIMARY window_days={BSC_PRIMARY}): "
          f"R^2={results[BSC_PRIMARY]:.6f} < {CONST_CAP_R2_THRESH} -> "
          f"{'PASS (candidate is NOT a relabeling of v4)' if a2_pass else 'FAIL'}")
    assert a2_pass, (
        f"A2 KILL SWITCH FAILED: R^2={results[BSC_PRIMARY]:.6f} >= {CONST_CAP_R2_THRESH} "
        f"at PRIMARY window_days={BSC_PRIMARY} -- candidate is a relabeling of v4's own scale")
    return {"r2_by_window": results, "a2_pass": a2_pass}


# ================================================================== STEP 3
# Sweep: 4 window_days values x 2 markets x 3 slices = 24 cells.
# ==================================================================

def step3_sweep(btc_full: pd.DataFrame, eth_full: pd.DataFrame) -> list[dict]:
    global CONFIGS_EVALUATED
    print("\n" + "=" * 88)
    print("STEP 3 -- Sweep: window_days in BSC_WINDOW_GRID x {spot, futures_5x} x "
          "{inner_train, inner_val, eth_replication}")
    print("=" * 88)
    all_rows: list[dict] = []
    for wd in BSC_WINDOW_GRID:
        t0 = time.time()
        rows = compare(lambda df, w=wd: build_conservative_target(df, w),
                        label=f"bsc_w{wd}", btc=btc_full, eth=eth_full,
                        include_eth=True)
        CONFIGS_EVALUATED += len(rows)  # 6 cells/grid value
        print(f"\n-- window_days={wd} [{time.time() - t0:.0f}s, {len(rows)} cells] --")
        print_rows(rows)
        all_rows.extend(rows)
    return all_rows


# ================================================================== STEP 4
# Fee-tier robustness: PRIMARY window_days at FEE_TIER=0.40% on both
# markets, inner-validation slice only -- 2 cells (matches the
# pre-registration's "+2 cells" fee-tier re-run of the finalist config).
# ==================================================================

def step4_fee_tier(btc_full: pd.DataFrame, window_days: int = BSC_PRIMARY) -> list[dict]:
    global CONFIGS_EVALUATED
    print("\n" + "=" * 88)
    print(f"STEP 4 -- Fee-tier robustness: window_days={window_days} @ FEE_TIER={FEE_TIER:.2%}, "
          "inner_val, both markets (2 cells)")
    print("=" * 88)
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    t0 = time.time()
    rows_all = compare(lambda df, w=window_days: build_conservative_target(df, w),
                        label=f"bsc_w{window_days}_fee{FEE_TIER:.2%}", btc=btc_full,
                        markets=fee_markets, include_eth=False)
    rows = [r for r in rows_all if r["slice"] == "inner_val"]
    CONFIGS_EVALUATED += len(rows)
    print(f"[{time.time() - t0:.0f}s, {len(rows_all)} cells computed "
          f"(inner_train + inner_val x 2 markets), {len(rows)} kept (inner_val only)]")
    print_rows(rows)
    return rows


# ================================================================== STEP 5
# Decision rule, applied exactly as written in r164_shared's module
# docstring, plus the pre-registered A3 exposure-match disclosure for
# EVERY cell (not just PRIMARY -- per instructions).
# ==================================================================

def apply_decision_rule(sweep_rows: list[dict]) -> dict:
    print("\n" + "=" * 88)
    print("STEP 5 -- Decision rule (r164_shared.py's frozen table)")
    print("=" * 88)

    inner_val_rows = [r for r in sweep_rows if r["slice"] == "inner_val"]
    eth_rows = [r for r in sweep_rows if r["slice"] == ETH_SLICE_NAME]

    def wd_of(row: dict) -> int:
        # label is "bsc_w<window_days>"
        return int(row["label"].rsplit("w", 1)[1])

    clear_by_market: dict[str, list[tuple[int, dict]]] = {"spot": [], "futures_5x": []}
    for row in inner_val_rows:
        wd = wd_of(row)
        if wd == 0:
            continue  # only non-zero grid values count toward CLEAR(m)
        if clears_bar(row):
            clear_by_market[row["market"]].append((wd, row))

    clear_spot = len(clear_by_market["spot"]) > 0
    clear_futures = len(clear_by_market["futures_5x"]) > 0
    print(f"CLEAR(spot)        = {clear_spot}  "
          f"(clearing window_days: {[w for w, _ in clear_by_market['spot']]})")
    print(f"CLEAR(futures_5x)  = {clear_futures}  "
          f"(clearing window_days: {[w for w, _ in clear_by_market['futures_5x']]})")

    # ETH sign-replication falsification test, at PRIMARY only, on whichever
    # market cleared (pre-registration's falsification test is stated for
    # the PRIMARY config specifically).
    sign_ok = True
    plateau_ok = True
    replication_notes = []

    def eth_row_for(wd: int, market: str) -> dict | None:
        for r in eth_rows:
            if r["market"] == market and wd_of(r) == wd:
                return r
        return None

    cleared_markets = [m for m, ok in (("spot", clear_spot), ("futures_5x", clear_futures)) if ok]
    for market in cleared_markets:
        primary_row = next((r for w, r in clear_by_market[market] if w == BSC_PRIMARY), None)
        if primary_row is None:
            replication_notes.append(
                f"market={market}: PRIMARY (window_days={BSC_PRIMARY}) did not itself "
                f"clear on this market (a different non-zero window_days did) -- falsification "
                f"test is pre-registered specifically for the PRIMARY config, so this market "
                f"contributes no sign-replication check.")
            continue
        eth_row = eth_row_for(BSC_PRIMARY, market)
        if eth_row is None:
            replication_notes.append(f"no ETH row for window_days={BSC_PRIMARY}/{market}")
            continue
        btc_sign = np.sign(primary_row["d_log_growth"])
        eth_sign = np.sign(eth_row["d_log_growth"])
        note = (f"market={market} window_days={BSC_PRIMARY}: BTC d_log_growth="
                f"{primary_row['d_log_growth']:+.4f} (sign {btc_sign:+.0f}), "
                f"ETH d_log_growth={eth_row['d_log_growth']:+.4f} (sign {eth_sign:+.0f})")
        replication_notes.append(note)
        if btc_sign != 0 and eth_sign != 0 and btc_sign != eth_sign:
            sign_ok = False

    # Plateau check: non-zero window_days values that clear (on any market)
    # must share the same sign of d_log_growth on inner-validation.
    nonzero_clearing_signs = set()
    for market in ("spot", "futures_5x"):
        for wd, row in clear_by_market[market]:
            nonzero_clearing_signs.add(np.sign(row["d_log_growth"]))
    if len(nonzero_clearing_signs) > 1:
        plateau_ok = False

    gate_ok = sign_ok and plateau_ok
    print("\nETH sign-replication notes (PRIMARY config only, per pre-registration):")
    for n in replication_notes:
        print(f"    {n}")
    print(f"\nsign_ok (no inversion on the market that cleared, at PRIMARY) = {sign_ok}")
    print(f"plateau_ok (clearing non-zero window_days values share one sign) = {plateau_ok}")
    print(f"GATE_OK = sign_ok AND plateau_ok = {gate_ok}")

    if not gate_ok:
        verdict = "REJECT"
    elif not clear_spot and not clear_futures:
        verdict = "REJECT"
    elif clear_spot and clear_futures:
        verdict = "PROMOTE"
    else:
        verdict = "PARTIAL"

    print("\nDecision table row matched:")
    print(f"    | GATE_OK={gate_ok!s:<5} | CLEAR(spot)={clear_spot!s:<5} "
          f"| CLEAR(futures)={clear_futures!s:<5} | Verdict = {verdict} |")

    # A3 exposure-match disclosure (NOT a kill switch -- disclosed, not
    # calibrated per pre-registration) for EVERY inner-validation cell, not
    # just PRIMARY.
    print("\nA3 exposure/vol-ratio disclosure, EVERY inner-validation cell "
          "(disclosed, not calibrated -- no predicted direction pre-registered):")
    a3_notes = []
    for row in sorted(inner_val_rows, key=lambda r: (wd_of(r), r["market"])):
        note = (f"    window_days={wd_of(row):<4} market={row['market']:<11} "
                f"exposure_ratio={row['exposure_ratio']:.4f} vol_ratio={row['vol_ratio']:.4f} "
                f"risk_matched={row['risk_matched']} d_log_growth={row['d_log_growth']:+.4f} "
                f"d_sharpe={row['d_sharpe']:+.3f} d_dd={row['d_dd']:+.2f}")
        print(note)
        a3_notes.append(row)

    return {
        "clear_spot": clear_spot, "clear_futures": clear_futures,
        "sign_ok": sign_ok, "plateau_ok": plateau_ok, "gate_ok": gate_ok,
        "verdict": verdict, "a3_rows": a3_notes,
    }


# ================================================================== main

if __name__ == "__main__":
    t_start = time.time()
    print("R-164 CONSERVATIVE branch (Barroso & Santa-Clara 2015 risk-managed momentum) -- full run")
    print(f"BSC_WINDOW_GRID={BSC_WINDOW_GRID}  BSC_PRIMARY={BSC_PRIMARY}  "
          f"BSC_MIN_TARGET_DAYS={BSC_MIN_TARGET_DAYS}  BSC_MULT_CLIP={BSC_MULT_CLIP}  "
          f"FEE_TIER={FEE_TIER:.2%}  SHARPE_NOISE_FLOOR={SHARPE_NOISE_FLOOR}  "
          f"CONST_CAP_R2_THRESH={CONST_CAP_R2_THRESH}")

    btc_full = load_btc()
    eth_full = load_eth()
    print(f"BTC pre-holdout bars: {len(btc_full)} ({btc_full.index[0]} .. {btc_full.index[-1]})")
    print(f"ETH replication bars: {len(eth_full)} ({eth_full.index[0]} .. {eth_full.index[-1]})")
    assert pd.Timestamp(btc_full.index[-1]) < pd.Timestamp(OOS_START, tz="UTC"), (
        "btc_full reaches OOS_START -- holdout must not be touched by this branch")
    assert pd.Timestamp(eth_full.index[-1]) < pd.Timestamp(OOS_START, tz="UTC"), (
        "eth_full reaches OOS_START -- holdout must not be touched by this branch")

    step1 = step1_sanity_checks(btc_full)
    a2 = step2_a2_kill_switch(btc_full)
    sweep_rows = step3_sweep(btc_full, eth_full)
    fee_rows = step4_fee_tier(btc_full, window_days=BSC_PRIMARY)

    print("\n" + "=" * 88)
    print("FULL RESULT TABLE (sweep + fee-tier robustness)")
    print("=" * 88)
    print_rows(sweep_rows + fee_rows)

    decision = apply_decision_rule(sweep_rows)

    print("\n" + "=" * 88)
    print("STEP 6 -- Holdout")
    print("=" * 88)
    print("NOT READ. Per instructions, this branch stops after Step 5 regardless of verdict --")
    print(f"the holdout (start={OOS_START}) is left for a human/operator to authorize separately,")
    print(f"even though this run's verdict is {decision['verdict']!r}.")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"Configs evaluated: {CONFIGS_EVALUATED} (expected 26: 4 window_days values x 2 "
          f"markets x 3 slices = 24, + 2 fee-tier cells)")
    print(f"A1 kill switch: bit-for-bit={step1['a1_identical']}  R^2={step1['a1_r2']:.10f}  "
          f"max_abs_diff={step1['a1_max_abs_diff']:.3e} -> "
          f"{'PASS' if step1['a1_identical'] and step1['a1_r2'] == 1.0 else 'FAIL'}")
    print(f"A2 kill switch (PRIMARY window_days={BSC_PRIMARY}): "
          f"R^2={a2['r2_by_window'][BSC_PRIMARY]:.6f} vs cap {CONST_CAP_R2_THRESH} -> "
          f"{'PASS' if a2['a2_pass'] else 'FAIL -- relabeling of v4, not a tested mechanism'}")
    print(f"Decision rule verdict (inner-validation): {decision['verdict']}")
    print(f"  GATE_OK={decision['gate_ok']}  CLEAR(spot)={decision['clear_spot']}  "
          f"CLEAR(futures_5x)={decision['clear_futures']}")
    print("Holdout read: NO (not authorized at this stage)")
    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")
