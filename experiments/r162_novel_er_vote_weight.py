"""R-162 NOVEL branch: per-anchor Kaufman Efficiency Ratio conviction
weighting of `kelly_regime_v4`'s three-anchor VOTE combination, replacing
the plain mean with an ER-conviction-weighted mean -- SCALE is never
touched. All mechanism, constants and decision rule are frozen in
``experiments/r162_shared.py`` (read-only, not edited by this file); this
file only drives that pre-registration end to end and reports the result.

CONSTRUCTION UNDER TEST: ``build_novel_target(df, beta)`` /
``er_vote_weighted_frac(df, beta)`` (r162_shared.py). beta=0.0 is the
identity check (equal weights -> bit-for-bit v4). beta in GRID sweeps the
conviction strength; PRIMARY=1.0 is the falsification-test / holdout config.

STEPS (mirrors the dispatch instructions verbatim):
  1. Sanity/causality: beta=0.0 reproduces v4_target bit-for-bit (A1);
     causal_truncation_probe_series passes at beta=1.0.
  2. A2 kill switch: R^2 of er_vote_weighted_frac(PRIMARY)*v4_scale vs
     v4_raw_desired on inner-train, must be < CONST_CAP_R2_THRESH (0.98).
  3. Sweep: GRID x 2 markets x 3 slices via compare() = 24 cells.
  4. Fee-tier robustness: PRIMARY beta, FEE_TIER=0.40%, both markets = 2
     more cells. Total 26.
  5. Delayed-flip diagnostic on inner-train and inner-val (disclosed, not
     folded into headline Sharpe).
  6. Decision rule from r162_shared.py's own module docstring, applied to
     the inner-validation rows.
  7. Holdout, read once, ONLY if GATE_OK and >=1 market clears.

USAGE
-----
    python experiments/r162_novel_er_vote_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r162_shared import (  # noqa: E402
    CONST_CAP_R2_THRESH,
    FEE_TIER,
    FUTURES,
    GRID,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    PRIMARY,
    SPOT,
    TargetStrategy,
    build_novel_target,
    causal_truncation_probe_series,
    clears_bar,
    compare,
    delayed_flip_diagnostic,
    er_vote_weighted_frac,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
)

CONFIGS_EVALUATED = 26  # 4 betas x 2 markets x 3 slices (24) + fee-tier re-run x 2 markets (2)


# ================================================================== (1)
# Step 1: sanity / causality kill switch A1.
# ==================================================================

def step1_identity_and_causality() -> dict:
    btc_full = load_btc()
    # A subset for speed -- long enough to clear every anchor's warmup
    # (80-day slowest anchor + 365-day ER reference window) many times over.
    n = min(len(btc_full), 400_000)
    btc = btc_full.iloc[:n].copy()

    novel_b0 = np.asarray(build_novel_target(btc, 0.0), dtype=float)
    v4 = np.asarray(v4_target(btc), dtype=float)
    identical = bool(np.array_equal(novel_b0, v4))
    max_abs_diff = float(np.max(np.abs(novel_b0 - v4))) if len(novel_b0) else float("nan")

    causal_ok = causal_truncation_probe_series(
        lambda df: build_novel_target(df, PRIMARY), btc)

    return dict(identical=identical, max_abs_diff=max_abs_diff, causal_ok=causal_ok, n_bars=n)


# ================================================================== (2)
# Step 2: A2 non-collinearity kill switch, on inner-train.
# ==================================================================

def step2_a2_kill_switch() -> dict:
    btc = load_btc()
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))

    cand_raw = er_vote_weighted_frac(btc, PRIMARY).to_numpy() * v4_scale(btc)
    ctrl_raw = v4_raw_desired(btc)
    r_sq = r_squared(cand_raw[mask], ctrl_raw[mask])
    passes = bool(np.isfinite(r_sq) and r_sq < CONST_CAP_R2_THRESH)
    return dict(r_sq=r_sq, passes=passes, n_bars=int(mask.sum()))


# ================================================================== (3)+(4)
# Sweep + fee-tier robustness, both via the shared compare().
# ==================================================================

def step3_sweep(btc: pd.DataFrame, eth: pd.DataFrame) -> list[dict]:
    rows = []
    for beta in GRID:
        rows.extend(compare(lambda df, b=beta: build_novel_target(df, b),
                            label=f"r162_novel_b{beta}", btc=btc, eth=eth))
    return rows


def step4_fee_tier(btc: pd.DataFrame) -> list[dict]:
    """Re-run the PRIMARY beta at FEE_TIER=0.40% on both markets, BTC
    inner-validation ONLY (the pre-registered "2 more cells" -- matching the
    r105/r106-family convention of a fee-tier check at the primary cell on
    the promotion-relevant slice, not a repeat of every slice). Builds rows
    with the identical schema to compare()'s own, by replicating its row
    construction directly (compare() itself always runs both SLICES, so it
    cannot be restricted to one slice without duplicating this logic)."""
    label = f"r162_novel_b{PRIMARY}_fee{FEE_TIER:.4f}"
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    cand = TargetStrategy(lambda df: build_novel_target(df, PRIMARY), name=f"r162_{label}")
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")

    rows = []
    for market in fee_markets:
        a = run_slice(cand, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        b = run_slice(ctrl, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        pr = paired_diff(a.daily, b.daily)
        exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                    if b.mean_abs_exposure else float("nan"))
        vol_ratio = (a.realized_vol / b.realized_vol if b.realized_vol else float("nan"))
        rows.append({
            "label": label, "slice": "inner_val", "market": market.name,
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
            "boot_d_loggrowth": pr.diff.point,
            "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
            "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        })
    return rows


# ================================================================== (5)
# Delayed-flip diagnostic (disclosed, not a kill switch).
# ==================================================================

def step5_delayed_flip(btc: pd.DataFrame) -> dict:
    def slice_of(start, end):
        m = np.asarray((btc.index >= pd.Timestamp(start, tz="UTC")) &
                        (btc.index <= pd.Timestamp(end, tz="UTC")))
        return btc.loc[m]

    train = slice_of(INNER_TRAIN_START, INNER_TRAIN_END)
    val = slice_of(INNER_VAL_START, INNER_VAL_END)
    return {
        "inner_train": delayed_flip_diagnostic(train, PRIMARY),
        "inner_val": delayed_flip_diagnostic(val, PRIMARY),
    }


# ================================================================== (6)
# Decision rule, exactly as frozen in r162_shared.py's module docstring.
# ==================================================================

def step6_decision_rule(sweep_rows: list[dict]) -> dict:
    val_rows = [r for r in sweep_rows if r["slice"] == "inner_val"]
    eth_rows = [r for r in sweep_rows if r["slice"] == "eth_replication"]

    # label -> beta, parsed back out of compare()'s own label convention.
    def beta_of(row) -> float:
        return float(row["label"].split("_b", 1)[1])

    clearing = [r for r in val_rows if beta_of(r) != 0.0 and clears_bar(r)]
    clear_spot = any(r["market"] == "spot" for r in clearing)
    clear_futures = any(r["market"] == "futures_5x" for r in clearing)

    # ETH sign-replication falsification test: for whichever market a
    # clearing config cleared on, the SAME beta/market's eth_replication row
    # must show the SAME sign of d_log_growth.
    def eth_row_for(beta: float, market: str):
        for r in eth_rows:
            if beta_of(r) == beta and r["market"] == market:
                return r
        return None

    eth_checks = []
    eth_pass = True
    for r in clearing:
        beta = beta_of(r)
        eth_r = eth_row_for(beta, r["market"])
        same_sign = (eth_r is not None and
                     np.sign(eth_r["d_log_growth"]) == np.sign(r["d_log_growth"]) and
                     np.sign(r["d_log_growth"]) != 0)
        eth_checks.append(dict(beta=beta, market=r["market"],
                               btc_d_log_growth=r["d_log_growth"],
                               eth_d_log_growth=eth_r["d_log_growth"] if eth_r is not None else None,
                               same_sign=bool(same_sign)))
        if not same_sign:
            eth_pass = False

    # Plateau check: non-zero grid values that clear (on ANY market) share
    # the same sign of d_log_growth on inner-validation.
    nonzero_clear_signs = {np.sign(r["d_log_growth"]) for r in clearing if r["d_log_growth"] != 0}
    plateau_ok = len(nonzero_clear_signs) <= 1

    clear_any = clear_spot or clear_futures
    gate_ok = bool((eth_pass if clear_any else True) and plateau_ok)
    # GATE_OK is only meaningful (and only gates anything) once at least one
    # market clears; if nothing clears, GATE_OK's truth value is moot and
    # the table's first row (false -> REJECT) already applies, but we still
    # report eth_pass/plateau_ok honestly rather than defaulting them True
    # in a way that could be misread as "the gate passed."
    if not clear_any:
        gate_ok = False

    if not gate_ok:
        verdict = "REJECT"
    elif not clear_spot and not clear_futures:
        verdict = "REJECT"
    elif clear_spot and clear_futures:
        verdict = "PROMOTE"
    else:
        verdict = "PARTIAL"

    return dict(
        val_rows=val_rows, clearing=clearing,
        clear_spot=clear_spot, clear_futures=clear_futures, clear_any=clear_any,
        eth_checks=eth_checks, eth_pass=eth_pass, plateau_ok=plateau_ok,
        gate_ok=gate_ok, verdict=verdict,
    )


# ================================================================== (7)
# Holdout -- only if GATE_OK and at least one market clears.
# ==================================================================

def step7_holdout() -> list[str]:
    from scripts.experiment import ev
    from tradebot.registry import get_strategy

    lines = []

    def run(strategy, market, tag):
        m = ev(strategy, market=market, start=OOS_START, tag=tag)
        lines.append(f"{tag:28s} {market.name:11s} final=${m.final_balance:>13,.0f} "
                     f"({m.profit_pct:>+9.1f}%) trades={m.num_trades:>5d} "
                     f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f}")

    novel_strategy = TargetStrategy(lambda df: build_novel_target(df, PRIMARY), name="r162_novel")

    for market, mname in ((SPOT, "spot"), (FUTURES, "futures_5x")):
        run(novel_strategy, market, f"r162_novel[HOLDOUT-{mname}]")
        run(get_strategy("kelly_regime_v4"), market, f"kelly_regime_v4[HOLDOUT-{mname}]")
        run(get_strategy("buy_and_hold"), market, f"buy_and_hold[HOLDOUT-{mname}]")

    return lines


# ================================================================== main
# ==================================================================

def main() -> None:
    print("=" * 100)
    print("R-162 NOVEL: per-anchor ER conviction-weighted VOTE (kelly_regime_v4)")
    print("=" * 100)

    print("\n--- Step 1: A1 identity + causality kill switches ---")
    s1 = step1_identity_and_causality()
    print(f"  beta=0.0 bit-for-bit v4_target: {s1['identical']} "
          f"(max abs diff={s1['max_abs_diff']:.3e}, n_bars={s1['n_bars']:,})")
    print(f"  causal_truncation_probe_series(build_novel_target, beta={PRIMARY}): "
          f"{'PASS' if s1['causal_ok'] else 'FAIL'}")
    a1_ok = s1["identical"] and s1["causal_ok"]
    if not a1_ok:
        print("  A1/causality FAILED -- stopping. This branch cannot be trusted further.")
        return

    print("\n--- Step 2: A2 non-collinearity kill switch (inner-train) ---")
    s2 = step2_a2_kill_switch()
    print(f"  R^2(er_vote_weighted_frac(PRIMARY)*v4_scale, v4_raw_desired) = {s2['r_sq']:.6f} "
          f"(n_bars={s2['n_bars']:,}); threshold < {CONST_CAP_R2_THRESH}")
    print(f"  A2 kill switch: {'PASS (non-collinear)' if s2['passes'] else 'FAIL (relabeling of v4)'}")
    if not s2["passes"]:
        print("  NOTE: proceeding to run and report the sweep honestly per instructions, "
              "but any Sharpe/growth differences below must NOT be interpreted as a real "
              "mechanism effect -- A2 failure means the candidate's exposure path is a "
              "near-exact rescale of v4's own.")

    print("\n--- Step 3: sweep (GRID x 2 markets x 3 slices = 24 cells) ---")
    btc = load_btc()
    eth = load_eth()
    sweep_rows = step3_sweep(btc, eth)
    print_rows(sweep_rows)

    print(f"\n--- Step 4: fee-tier robustness (beta={PRIMARY}, FEE_TIER={FEE_TIER:.2%}, "
          f"inner_val only, 2 cells) ---")
    fee_rows = step4_fee_tier(btc)
    print_rows(fee_rows)

    all_rows = sweep_rows + fee_rows
    print(f"\nTotal configurations evaluated: {len(all_rows)} "
          f"(pre-registered count: {CONFIGS_EVALUATED})")
    assert len(all_rows) == CONFIGS_EVALUATED, (
        f"config count mismatch: got {len(all_rows)}, expected {CONFIGS_EVALUATED}")

    print("\n--- Step 5: delayed-flip diagnostic (disclosed, not headline) ---")
    flip = step5_delayed_flip(btc)
    for slice_name, d in flip.items():
        print(f"  {slice_name:12s} v4_flips={d['v4_flip_count']:4d} "
              f"novel_flips={d['novel_flip_count']:4d} "
              f"delayed(no-match-within-30bars)={d['flips_with_no_match_within_30bars']:4d} "
              f"delayed_fraction={d['delayed_fraction']:.3f}")

    print("\n--- Step 6: decision rule (r162_shared.py's own frozen table) ---")
    dr = step6_decision_rule(sweep_rows)
    print(f"  CLEAR(spot)    = {dr['clear_spot']}")
    print(f"  CLEAR(futures) = {dr['clear_futures']}")
    print(f"  clearing inner-val cells (non-zero beta, clears_bar()==True):")
    if dr["clearing"]:
        for r in dr["clearing"]:
            print(f"    label={r['label']:24s} market={r['market']:11s} "
                  f"d_sharpe={r['d_sharpe']:+.3f} d_log_growth={r['d_log_growth']:+.4f} "
                  f"excludes_zero={r['excludes_zero']} risk_matched={r['risk_matched']}")
    else:
        print("    (none)")
    print(f"  plateau_ok (clearing non-zero-beta cells share one sign of d_log_growth) = "
          f"{dr['plateau_ok']}")
    print(f"  ETH sign-replication checks (for each clearing cell, same beta/market's "
          f"eth_replication row):")
    if dr["eth_checks"]:
        for c in dr["eth_checks"]:
            print(f"    beta={c['beta']:<4} market={c['market']:11s} "
                  f"btc_d_log_growth={c['btc_d_log_growth']:+.4f} "
                  f"eth_d_log_growth={c['eth_d_log_growth']:+.4f} same_sign={c['same_sign']}")
    else:
        print("    (n/a -- nothing cleared)")
    print(f"  eth_pass (no sign inversion on any market that cleared) = {dr['eth_pass']}")
    print(f"  GATE_OK = {dr['gate_ok']}")
    print(f"  VERDICT = {dr['verdict']}")

    holdout_lines: list[str] = []
    if dr["verdict"] in ("PARTIAL", "PROMOTE"):
        print("\n--- Step 7: holdout (GATE_OK and >=1 market cleared -- reading once) ---")
        holdout_lines = step7_holdout()
        for line in holdout_lines:
            print("  " + line)
    else:
        print("\n--- Step 7: holdout NOT read (gate did not clear; verdict is REJECT) ---")

    print("\n" + "=" * 100)
    print(f"FINAL VERDICT: {dr['verdict']}  |  A2 kill switch: "
          f"{'PASS' if s2['passes'] else 'FAIL (relabeling risk -- see note above)'}  |  "
          f"configs evaluated: {len(all_rows)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
