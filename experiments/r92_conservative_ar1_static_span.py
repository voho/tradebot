#!/usr/bin/env python
"""R-92 CONSERVATIVE branch: derive `kelly_regime_v4`'s 20/40/80-day anchor
ladder analytically instead of empirically, via a SINGLE, FROZEN, inner-
train-only AR(1) fit of Sepp & Lucic (2026)'s closed-form trend-following
Sharpe. The full citation trail, the round's direction, the not-a-duplicate
reasoning, the disclosed simplifications (kurtosis-free Eq. 5.12; EWMA-span
matched to v4's SMA anchors by first-moment mean lag), the shared A0/A1'/A2/A3
gates and the B1-B5 promotion bar all live in `experiments/r92_shared.py`'s
module docstring; this file does not repeat that reasoning and does not edit
that module.

MECHANISM (one sentence, frozen, no deviation):

    Fit phi (lag-1 autocorrelation) and mu (drift) ONCE on inner-train-only,
    vol-normalized BTC daily log returns; find the SR(nu)-maximizing span S*
    on a fine closed-form grid (5..200 days); freeze the doubling ladder
    (S*//2, S*, S*x2) in place of v4's shipped (20, 40, 80), everything else
    (vote architecture, scale factor, 10% deadband) identical to v4.

A0 KILL SWITCH -- run FIRST, before any strategy is built or backtested. Per
`r92_shared`'s pre-registration: if phi <= 0, or the SR(nu) closed-form
optimum sits within `edge_tolerance` grid steps of either edge of the
feasible span grid (no interior optimum), the branch is disqualified BY
PRE-REGISTRATION and this script stops immediately -- no strategy build, no
backtest, no sweep. This is a complete, correct outcome, not an unfinished
one: it means Sepp & Lucic's own precondition for a profitable trend system
(positive, well-scaled autocorrelation) does not hold on BTC's own inner-
train series.

This file never reads a bar at or after OOS_START (2023-01-01): every load
goes through `r92_shared`'s truncating, asserting loaders, and the max
timestamp actually read anywhere in this run is tracked and printed at the
end of main().

Run: source .venv/bin/activate && python3 experiments/r92_conservative_ar1_static_span.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r92_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    OOS_START,
    SPOT,
    Z_VOL_WINDOW_DAYS,
    assert_no_holdout,
    causal_truncation_probe,
    compare,
    daily_log_returns,
    derive_optimal_span,
    fee_at,
    fit_ar1,
    kill_switch_a0,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    span_ladder_target,
    v4_target,
    vol_normalized_returns,
)

# ---------------------------------------------------------------- frozen bar
R2_CEILING = 0.98      # frozen A2 bar
SHARPE_FLOOR = 0.2     # frozen B2 bar (R-20 noise floor)
HIGH_FEE = 0.0040      # frozen B5 taker fee: 0.40%
EDGE_TOLERANCE = 3     # frozen A0 grid-boundary tolerance (r92_shared default)


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(rows: list[dict], slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["slice"] == slice_name and r["market"] == market:
            return r
    return None


# ------------------------------------------------------------- the mechanism

def derive_phi_mu_span(df_train: pd.DataFrame) -> tuple[float, float, int, np.ndarray, np.ndarray]:
    """The one frozen fit + derivation this whole branch depends on. Pure
    function of the inner-train frame it is handed, so calling it twice must
    return identical numbers (A1' determinism)."""
    dr = daily_log_returns(df_train)
    z = vol_normalized_returns(dr, window_days=Z_VOL_WINDOW_DAYS)
    phi, mu = fit_ar1(z)
    span, sr_curve, grid = derive_optimal_span(phi, mu)
    return phi, mu, span, sr_curve, grid


def ladder_of(span: int) -> tuple[int, int, int]:
    lo = max(1, int(span // 2))
    hi = max(1, int(span * 2))
    mid = max(1, int(span))
    return (lo, mid, hi)


def build_candidate(df: pd.DataFrame, horizons: tuple[int, int, int]) -> np.ndarray:
    """Pure function of the bars it is handed -- what compare()/
    causal_truncation_probe call. Frozen mechanism: v4's own architecture
    (vote x scale, then deadband) with the derived horizons in place of
    (20, 40, 80)."""
    return span_ladder_target(df, horizons)


def make_builder(horizons: tuple[int, int, int]):
    return lambda d: build_candidate(d, horizons)


# ------------------------------------------------------------------- runner

def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []
    n_configs_backtested = 0

    hr("R-92 CONSERVATIVE -- single frozen AR(1)-derived anchor span, "
       "in place of v4's shipped 20/40/80 doubling ladder.\nDefault verdict: "
       "NEGATIVE. A0 kill switch checked FIRST, before any strategy is built.")

    df_full = load_btc()
    max_ts_seen.append(df_full.index.max())
    assert_no_holdout(df_full, "BTC full")
    df_train = df_full.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(df_train, "BTC inner-train")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(df_full):,} bars, "
          f"{df_full.index[0]} -> {df_full.index[-1]}")
    print(f"inner-train frame: {len(df_train):,} bars, {df_train.index[0]} -> "
          f"{df_train.index[-1]}")

    # ------------------------------------------------------- fit + derive
    hr("STEP 1 -- fit AR(1) on inner-train-only, vol-normalized BTC daily "
       "log returns; derive the SR(nu)-maximizing span")
    phi, mu, span, sr_curve, grid = derive_phi_mu_span(df_train)
    dr = daily_log_returns(df_train)
    z = vol_normalized_returns(dr, window_days=Z_VOL_WINDOW_DAYS)
    print(f"\n    daily log returns (inner-train): {len(dr)} days, "
          f"{dr.index[0].date()} -> {dr.index[-1].date()}")
    print(f"    vol-normalized z_t (20-day causal window): {len(z)} days")
    print(f"    fitted phi (lag-1 autocorrelation of z_t) = {phi:.6f}")
    print(f"    fitted mu  (mean of z_t)                  = {mu:.6f}")
    print(f"    SPAN_GRID_DAYS = [{grid[0]}, {grid[-1]}], step 1 day, "
          f"{len(grid)} points")
    if span == -1:
        print(f"    derived span = -1 (SR(nu) curve fully degenerate: no "
              f"finite points)")
    else:
        print(f"    derived SR(nu)-maximizing span = {span} days   "
              f"(SR at optimum = {sr_curve[int(np.where(grid == span)[0][0])]:.4f})")

    # ------------------------------------------------------- A0 kill switch
    hr("A0 KILL SWITCH (pre-registered, checked BEFORE any strategy build "
       "or backtest)")
    a0_pass, a0_reason = kill_switch_a0(phi, sr_curve, grid, edge_tolerance=EDGE_TOLERANCE)
    print(f"\n    A0: {'PASSES' if a0_pass else 'FAILS'}")
    print(f"    reason: {a0_reason}")

    if not a0_pass:
        hr("VERDICT")
        print(f"\n    phi   = {phi:.6f}")
        print(f"    mu    = {mu:.6f}")
        print(f"    derived span = {span}")
        print(f"    A0 kill switch: FAILED -- {a0_reason}")
        print(f"\n    VERDICT: NEGATIVE (A0 kill switch fired, pre-registered "
              f"and binding). Per pre-registration this branch STOPS HERE: "
              f"no strategy is built, no backtest is run, no sweep is "
              f"performed. Sepp & Lucic's own precondition for a profitable "
              f"European trend system (positive, well-scaled AR(1) "
              f"autocorrelation with an interior SR(nu) optimum) does not "
              f"hold on BTC's own causal inner-train series.")
        hr("BOOKKEEPING")
        print(f"    Closed-form SR(nu) grid evaluations (NOT counted as "
              f"backtests): {len(grid)}")
        print(f"    Configurations actually BACKTESTED: 0")
        print(f"    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
              f"(OOS_START = {OOS_START}; strictly earlier: "
              f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")
        return

    # ==================================================================
    # A0 passed -- proceed to build the candidate and run the full gate.
    # ==================================================================
    horizons = ladder_of(span)
    hr(f"A0 PASSED -- proceeding. Derived doubling ladder (from span={span}): "
       f"{horizons}")

    # ------------------------------------------------------- A1' determinism
    hr("A1' -- reproducibility: re-run the fit + derivation twice from a "
       "clean call")
    phi2, mu2, span2, sr_curve2, grid2 = derive_phi_mu_span(df_train)
    a1_pass = (phi == phi2) and (mu == mu2) and (span == span2)
    print(f"\n    run 1: phi={phi:.10f} mu={mu:.10f} span={span}")
    print(f"    run 2: phi={phi2:.10f} mu={mu2:.10f} span={span2}")
    print(f"    A1' determinism: {'PASS' if a1_pass else 'FAIL'}")
    if not a1_pass:
        raise AssertionError("A1' determinism FAILED -- the fit is not a pure "
                             "function of df_train.")

    # ------------------------------------------------------- A2 non-inertness
    hr("A2 -- non-inertness: R^2 of candidate's exposure path vs v4's own, "
       "on inner-train, must be < " + str(R2_CEILING))
    cand_path_train = build_candidate(df_train, horizons)
    v4_path_train = v4_target(df_train)
    rsq = r_squared(cand_path_train, v4_path_train)
    a2_pass = rsq < R2_CEILING
    print(f"\n    R^2(candidate, v4) on inner-train = {rsq:.6f}")
    a2_verdict = "PASS" if a2_pass else "FAIL (INERT -- derived ladder rounds back to effectively v4)"
    print(f"    A2 non-inertness: {a2_verdict}")

    # ------------------------------------------------------- A3 causality
    hr("A3 -- causality: causal_truncation_probe at cuts (0.55, 0.80)")
    a3_pass = causal_truncation_probe(make_builder(horizons), df_full)
    print(f"\n    A3 causality (full BTC frame, cuts 0.55/0.80): "
          f"{'PASS' if a3_pass else 'FAIL'}")
    if not a3_pass:
        raise AssertionError("A3 causality FAILED.")

    # ------------------------------------------------------- Step B: compare
    hr("STEP B -- compare() vs v4 on all four (slice x market) cells")
    rows = compare(make_builder(horizons), df_full,
                   label="r92_conservative_ar1_static_span",
                   markets=(SPOT, FUTURES),
                   slice_names=("inner_train", "inner_val"))
    n_configs_backtested += 1 * 2 * 2  # 1 config x 2 markets x 2 slices
    print()
    print_rows(rows)

    print("\nRISK MATCH (exposure_ratio / vol_ratio, candidate / v4, from "
          "compare()):")
    for r in rows:
        print(f"    {r['slice']:11s} {r['market']:11s} expR={r['exposure_ratio']:.3f} "
              f"volR={r['vol_ratio']:.3f} risk_matched={r['risk_matched']}")

    # ------------------------------------------------------- promotion bar
    hr("PROMOTION BAR (default REJECT, all B1-B5 must hold)")

    # ---- B1 ------------------------------------------------------------
    pts = [r["d_loggrowth"] for r in rows]
    excl = [r["excludes_zero"] for r in rows]
    b1_pos_all = all(p > 0 for p in pts)
    b1_excl_any = any(excl)
    b1 = bool(b1_pos_all and b1_excl_any)
    print("\n--- B1  paired bootstrap excludes zero in >=1 of four cells AND "
          "point estimate positive in all four")
    for r in rows:
        print(f"      {r['slice']:11s} {r['market']:11s} dlogG={r['d_loggrowth']:+7.3f} "
              f"[{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]  excludes_zero="
              f"{'YES' if r['excludes_zero'] else 'no'}")
    print(f"      point estimate positive in all four: {b1_pos_all}; "
          f"excludes zero in >=1: {b1_excl_any}")
    print(f"    B1: {'PASS' if b1 else 'FAIL'}")

    # ---- B2 ------------------------------------------------------------
    v_f = cell(rows, "inner_val", FUTURES.name)
    v_s = cell(rows, "inner_val", SPOT.name)
    sharpe_leg = (v_f["d_sharpe"] > SHARPE_FLOOR) and (v_s["d_sharpe"] > SHARPE_FLOOR)
    dd_leg = (v_f["d_dd"] < 0 and v_f["risk_matched"] and
              v_s["d_dd"] < 0 and v_s["risk_matched"])
    b2 = bool(sharpe_leg or dd_leg)
    print("\n--- B2  dSharpe > +0.2 on inner-validation on BOTH markets, OR a "
          "max-drawdown improvement on BOTH where risk_matched is True for "
          "both")
    print(f"      inner_val futures_5x: dSharpe={v_f['d_sharpe']:+.3f}  "
          f"dMaxDD={v_f['d_dd']:+.2f}pp  risk_matched={v_f['risk_matched']}  "
          f"(cand {v_f['cand_dd']:.1f}% vs v4 {v_f['ctrl_dd']:.1f}%)")
    print(f"      inner_val spot      : dSharpe={v_s['d_sharpe']:+.3f}  "
          f"dMaxDD={v_s['d_dd']:+.2f}pp  risk_matched={v_s['risk_matched']}  "
          f"(cand {v_s['cand_dd']:.1f}% vs v4 {v_s['ctrl_dd']:.1f}%)")
    print(f"      Sharpe leg: {sharpe_leg};  risk-matched drawdown leg: {dd_leg}")
    print(f"    B2: {'PASS' if b2 else 'FAIL'}")

    # ---- B3 -- plateau not peak (theory's own internal consistency) ----
    hr_idx = int(np.where(grid == span)[0][0])
    print("\n--- B3  plateau not peak: SR(nu) one grid step either side of "
          "the derived optimum")
    lo_idx, hi_idx = hr_idx - 1, hr_idx + 1
    sr_here = sr_curve[hr_idx]
    sr_lo = sr_curve[lo_idx] if lo_idx >= 0 else float("nan")
    sr_hi = sr_curve[hi_idx] if hi_idx < len(grid) else float("nan")
    print(f"      span={grid[lo_idx] if lo_idx >= 0 else '-':>5}  SR={sr_lo:.6f}  "
          f"(one step below)")
    print(f"      span={span:>5}  SR={sr_here:.6f}  (derived optimum)")
    print(f"      span={grid[hi_idx] if hi_idx < len(grid) else '-':>5}  SR={sr_hi:.6f}  "
          f"(one step above)")
    b3 = bool(np.isfinite(sr_lo) and np.isfinite(sr_hi)
              and sr_lo <= sr_here and sr_hi <= sr_here)
    print(f"      both neighbours <= the optimum (a local max, not a cliff "
          f"edge or a spike the closed form itself disowns): {b3}")
    print(f"    B3: {'PASS' if b3 else 'FAIL'}")

    # ---- B4  ETH replication --------------------------------------------
    print("\n--- B4  falsification: ETH replication (Bitfinex ETH, pre-2023, "
          "inner-train only)")
    eth_full = load_eth()
    max_ts_seen.append(eth_full.index.max())
    print(f"      ETH frame: {len(eth_full):,} bars, {eth_full.index[0]} -> "
          f"{eth_full.index[-1]}")
    eth_train_avail = eth_full.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    print(f"      ETH inner-train slice: {len(eth_train_avail):,} bars "
          f"({eth_train_avail.index[0] if len(eth_train_avail) else 'n/a'} -> "
          f"{eth_train_avail.index[-1] if len(eth_train_avail) else 'n/a'})")
    eth_rows = compare(make_builder(horizons), eth_full,
                       label=f"ETH_r92_conservative", markets=(SPOT, FUTURES),
                       slice_names=("inner_train",))
    n_configs_backtested += 1 * 2 * 1  # finalist x markets x inner_train
    print()
    print_rows(eth_rows)
    btc_train_pts = {m.name: cell(rows, "inner_train", m.name)["d_loggrowth"]
                     for m in (SPOT, FUTURES)}
    eth_pts = {r["market"]: r["d_loggrowth"] for r in eth_rows}
    same_sign = {m: bool(np.sign(eth_pts[m]) == np.sign(btc_train_pts[m])
                        and eth_pts[m] != 0)
                 for m in eth_pts}
    b4 = bool(eth_rows) and all(same_sign.values())
    for m in eth_pts:
        print(f"      BTC inner-train {m:11s} dlogG = {btc_train_pts[m]:+7.3f}   "
              f"ETH inner-train {m:11s} dlogG = {eth_pts[m]:+7.3f}   "
              f"same sign: {same_sign[m]}")
    print(f"    B4: {'PASS' if b4 else 'FAIL'}")

    # ---- B5  0.40% taker --------------------------------------------------
    print("\n--- B5  cost robustness: BTC inner-validation re-run at a 0.40% "
          "taker fee (via r92_shared.fee_at)")
    spot40 = fee_at(SPOT, HIGH_FEE)
    fut40 = fee_at(FUTURES, HIGH_FEE)
    print(f"      market specs (fee_at, same MarketSpec fields, fee_rate "
          f"swapped to {HIGH_FEE:.4f}):\n      {spot40}\n      {fut40}")
    rows40 = compare(make_builder(horizons), df_full, label="fee40_r92_conservative",
                     markets=(spot40, fut40), slice_names=("inner_val",))
    n_configs_backtested += 1 * 2 * 1  # finalist x fee markets x inner_val
    print()
    print_rows(rows40)
    base_spot = cell(rows, "inner_val", SPOT.name)
    base_fut = cell(rows, "inner_val", FUTURES.name)
    r40_spot = cell(rows40, "inner_val", spot40.name)
    r40_fut = cell(rows40, "inner_val", fut40.name)
    spot_keeps = bool(np.sign(r40_spot["d_loggrowth"]) == np.sign(base_spot["d_loggrowth"]))
    fut_keeps = bool(np.sign(r40_fut["d_loggrowth"]) == np.sign(base_fut["d_loggrowth"]))
    print(f"      SPOT (decisive per project convention): dlogG at 0.10% "
          f"taker = {base_spot['d_loggrowth']:+7.3f}   at 0.40% taker = "
          f"{r40_spot['d_loggrowth']:+7.3f}   sign preserved: {spot_keeps}")
    print(f"      FUTURES (reported for completeness):   dlogG at 0.05% "
          f"taker = {base_fut['d_loggrowth']:+7.3f}   at 0.40% taker = "
          f"{r40_fut['d_loggrowth']:+7.3f}   sign preserved: {fut_keeps}")
    b5 = spot_keeps  # SPOT at 0.40% taker is the decisive cell per project convention
    print(f"    B5: {'PASS' if b5 else 'FAIL'}   (decisive cell = SPOT@0.40%; "
          f"futures reported for completeness only, per project convention)")

    # ------------------------------------------------------------- verdict
    hr("VERDICT")
    print(f"\n    phi = {phi:.6f}   mu = {mu:.6f}   derived span = {span}   "
          f"ladder = {horizons}")
    print(f"    A0: PASSES ({a0_reason})")
    clauses = {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5}
    for k, v in clauses.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    promote = all(clauses.values())
    verdict = "CANDIDATE FOR HOLDOUT" if promote else "NEGATIVE"
    print(f"\n    VERDICT: {verdict}")
    if verdict == "NEGATIVE":
        failed_b = [k for k, v in clauses.items() if not v]
        print(f"    Reason(s): B-bar clauses failed: {', '.join(failed_b)}")
    print("\n    (The decision rule above is exactly the one frozen in "
          "r92_shared.py's docstring and\n    this file's own docstring "
          "before any number was read. No threshold was moved. The\n    "
          "holdout itself is NOT read or touched by this script, win or "
          "lose -- that decision belongs\n    to the operator.)")

    # ---------------------------------------------------------- bookkeeping
    hr("BOOKKEEPING")
    print(f"    Closed-form SR(nu) grid evaluations (NOT counted as "
          f"backtests): {len(grid)} x 2 (run 1 + A1' run 2) = {2 * len(grid)}")
    print(f"    Configurations actually BACKTESTED:")
    print(f"      Step B (BTC, 2 markets x 2 slices)              : 4 cells "
          f"(1 config)")
    print(f"      B4 ETH replication (2 markets x 1 slice)         : 2 cells "
          f"(same 1 config)")
    print(f"      B5 fee robustness (2 fee-markets x 1 slice)      : 2 cells "
          f"(same 1 config)")
    print(f"    TOTAL DISTINCT CONFIGURATIONS BACKTESTED: 1 (the derived "
          f"ladder {horizons}), across 8 measured (slice x market) cells")
    print(f"\n    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")


if __name__ == "__main__":
    main()
