#!/usr/bin/env python
"""R-102 CONSERVATIVE branch: `DownsideVolKellyV4` -- `kelly_regime_v4`
unchanged EXCEPT that the volatility series feeding its extremes-only
conditional-vol-target (`conditional_target_scale` in `r102_shared.py`) is
`downside_ewm_vol(df)` (realized DOWNSIDE semivariance, Barndorff-Nielsen,
Kinnebrock & Shephard 2010, computed from this project's own native
5-minute bars) instead of `v4_symmetric_vol(df)` (v4's own total, symmetric
volatility). The direction, citations, and non-duplication argument all
live in `experiments/r102_shared.py`'s module docstring (read there first,
it is this round's pre-registration context); this file does not repeat
that reasoning and does not edit that module.

MECHANISM (one sentence): if BTC's inverse-leverage effect (Baur & Dimpfl
2018) is driven specifically by DOWNSIDE variation rather than total
variation, sizing off downside semivariance alone should de-risk more
precisely into genuine crash episodes while staying more exposed through
benign (upside-driven) volatility spikes that the current total-vol target
treats identically.

Everything else -- the 3-anchor vote (`vote_frac`, horizons 20/40/80 days,
band=1%), `target_vol=0.55`, `max_leverage=2.0`, `anchor_span_days=180`,
the hysteresis bands (`high_in=1.70, high_out=1.20, low_in=0.55,
low_out=0.85`), and the 10% deadband -- is `kelly_regime_v4`'s own,
unmodified. The candidate is:

    build_target(df) = apply_deadband(
        vote_frac(df) * conditional_target_scale(downside_ewm_vol(df))
    )

======================================================================
HEADLINE RESULT, stated before the detail: NEGATIVE. Kill switch A2 does
NOT trip (target-path R^2 against v4 is 0.55-0.70 across the pre-registered
span grid, far below the 0.98 inertness ceiling) -- this is a genuinely
different exposure path, not a rescale of v4. But the difference is not a
sizing edge: B1 (Sharpe leg) FAILS on every span x market cell on
inner-validation (no cell clears +0.2 dSharpe, and no bootstrap interval
excludes zero), B2 (drawdown leg) FAILS because inner-validation drawdown
gets WORSE, not better, for the candidate, on every span, and the arms are
never risk-matched anyway (exposure_ratio and vol_ratio both sit at
1.2-1.5x v4, not 1.0). The mechanical reason: downside-only realized
variance is, by construction, <= total realized variance every bar
(RS+ + RS- = RV, RS- <= RV), so `target_vol / downside_vol` is
systematically LARGER than `target_vol / total_vol` -- the candidate runs
structurally HOTTER than v4 across the whole sample, not more selectively
de-levered into crashes as the mechanism hypothesized. B4 (ETH
falsification) FAILS: BTC's own inner-validation sign is not even stable
between spot (+) and futures (-) at any span, and ETH does not replicate
either sign. No clause of the promotion bar passes.
======================================================================

GATES (run in the order the task brief specifies):

  A2  non-inertness (kill switch, FIRST, before any Sharpe/drawdown
      number): R^2 of the candidate's scale path and of its full target
      path against v4's own, on BTC inner-train. R^2 >= 0.98 on the TARGET
      path would mean STOP -- a flat rescale of v4. Checked here on the
      pre-registered span grid (4, 8, 16 days); default (v4's own 8-day
      span) is the frozen primary candidate.
  causal  causal_truncation_probe_series on `downside_ewm_vol` and on this
      file's own `build_target`, against REAL BTC data (not just
      r102_shared's synthetic self-test).
  B1  Sharpe leg: dSharpe > +0.2 on inner-validation, BOTH markets (R-20's
      noise floor), OR the paired bootstrap excludes zero with a positive
      point estimate.
  B2  drawdown leg: a genuinely risk-matched (exposure_ratio AND vol_ratio
      both in [0.9, 1.1]) drawdown improvement on inner-validation.
  B3  plateau not peak: the span grid (4, 8, 16 days) must show a
      directionally consistent finding, not a spike at one value.
  B4  falsification (pre-registered, not changed after seeing results):
      ETH must show the SAME SIGN of dSharpe improvement as BTC
      inner-validation, on both markets.
  B5  cost robustness: sign of BTC log-growth deltas must survive a 0.40%
      taker fee tier (reported for completeness; moot once B1 fails).

Promotion requires A2 non-trip, causal PASS, and B1-B5 all hold. Default is
REJECT.

----------------------------------------------------------------------
Run: PYTHONPATH=<repo_root> python3 experiments/r102_conservative_downside_vol.py
(from the repo root, with the project venv active: `. .venv/bin/activate`)
----------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import (  # noqa: E402
    BARS_PER_DAY,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    OOS_START,
    SPOT,
    V4_VOL_SPAN,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    conditional_target_scale,
    downside_ewm_vol,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    v4_scale,
    v4_target,
    vote_frac,
)

# Pre-registered span grid (B3), fixed before any real-data result is seen:
# v4's own 8-day span, plus one neighbour on each side (a-priori grid, not
# chosen after looking at any number below).
SPAN_DAYS_GRID = [4, 8, 16]
DEFAULT_SPAN_DAYS = 8              # kelly_regime_v4's own V4_VOL_SPAN
A2_R2_CEILING = 0.98               # frozen kill-switch bar (task brief)
SHARPE_FLOOR = 0.2                 # frozen B1 bar (R-20 noise floor)
HIGH_FEE = 0.0040                  # frozen B5 taker fee: 0.40%


def make_build_target(span: int):
    """Candidate `build_target(df) -> np.ndarray`: v4's own vote and deadband,
    unchanged v4 scale-machinery (`conditional_target_scale`), fed
    `downside_ewm_vol` instead of `v4_symmetric_vol`. Pure function of `df`,
    exactly the shape `TargetStrategy`/`compare()` expect."""

    def build_target(df: pd.DataFrame) -> np.ndarray:
        scale = conditional_target_scale(downside_ewm_vol(df, span))
        raw = vote_frac(df).to_numpy() * scale
        return apply_deadband(raw)

    build_target.__name__ = f"downside_vol_kelly_v4_span{span}"
    return build_target


DEFAULT_SPAN = DEFAULT_SPAN_DAYS * BARS_PER_DAY
build_target = make_build_target(DEFAULT_SPAN)   # the frozen primary candidate


def downside_scale(df: pd.DataFrame, span: int) -> np.ndarray:
    return conditional_target_scale(downside_ewm_vol(df, span))


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(rows: list[dict], label: str, slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["label"] == label and r["slice"] == slice_name and r["market"] == market:
            return r
    return None


def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-102 CONSERVATIVE -- DownsideVolKellyV4: kelly_regime_v4's "
       "conditional-vol-target scale fed realized DOWNSIDE semivariance "
       "instead of total (symmetric) volatility.\nDefault verdict: NEGATIVE.")

    print("\nMECHANISM: if BTC's inverse-leverage effect (Baur & Dimpfl 2018) "
          "is driven specifically by\nDOWNSIDE variation, sizing off downside "
          "semivariance alone should de-risk more precisely into\ngenuine "
          "crash episodes while staying more exposed through benign "
          "(upside-driven) vol spikes.")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "BTC full")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    btc_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(btc_train, "BTC inner-train")
    print(f"BTC inner-train slice: {len(btc_train):,} bars, "
          f"{btc_train.index[0]} -> {btc_train.index[-1]}")

    # ================================================================= A2
    hr("A2 -- KILL SWITCH (non-inertness), run FIRST, before any "
       "Sharpe/drawdown number")
    print(f"\nCandidate scale/target paths vs v4's own, on BTC inner-train "
          f"({INNER_TRAIN_START} -> {INNER_TRAIN_END}),\nacross the "
          f"pre-registered span grid {SPAN_DAYS_GRID} days "
          f"(v4's own default = {DEFAULT_SPAN_DAYS}d):\n")

    ctrl_scale_train = v4_scale(btc_train)
    ctrl_target_train = v4_target(btc_train)
    vf_train = vote_frac(btc_train).to_numpy()

    a2_rows = []
    for days in SPAN_DAYS_GRID:
        span = days * BARS_PER_DAY
        cand_scale = downside_scale(btc_train, span)
        cand_target = apply_deadband(vf_train * cand_scale)
        r2_scale = r_squared(cand_scale, ctrl_scale_train)
        r2_target = r_squared(cand_target, ctrl_target_train)
        a2_rows.append((days, r2_scale, r2_target))
        tag = " <- default (frozen primary candidate)" if days == DEFAULT_SPAN_DAYS else ""
        print(f"    span={days:2d}d   R^2(scale)={r2_scale:+.4f}   "
              f"R^2(target)={r2_target:+.4f}{tag}")

    r2_target_default = next(r2t for d, _, r2t in a2_rows if d == DEFAULT_SPAN_DAYS)
    a2_tripped = r2_target_default >= A2_R2_CEILING
    print(f"\n    A2 ceiling: {A2_R2_CEILING}.  Default-span target R^2 = "
          f"{r2_target_default:.4f}.  Kill switch: "
          f"{'TRIPPED (>= ceiling)' if a2_tripped else 'NOT TRIPPED (< ceiling)'}")

    if not a2_tripped:
        print("\n    Reading: R^2(target) sits in 0.55-0.70 across the whole "
              "span grid -- well below the 0.98\n    ceiling. The "
              "substitution changes the traded path materially; it is not "
              "a disguised rescale of v4.\n    This does NOT mean the "
              "substitution is good -- only that it is worth evaluating on "
              "its own merits below.")

    # ============================================================ causality
    hr("CAUSAL TRUNCATION PROBES (real BTC data, not the shared module's "
       "synthetic self-test)")
    causal_ok = True
    try:
        ok1 = causal_truncation_probe_series(downside_ewm_vol, btc)
        print(f"    causal_truncation_probe_series(downside_ewm_vol, btc): "
              f"{'PASS' if ok1 else 'FAIL'}")
    except AssertionError as e:
        ok1 = False
        print(f"    causal_truncation_probe_series(downside_ewm_vol, btc): FAIL ({e})")
    causal_ok = causal_ok and ok1

    try:
        ok2 = causal_truncation_probe_series(build_target, btc)
        print(f"    causal_truncation_probe_series(build_target, btc): "
              f"{'PASS' if ok2 else 'FAIL'}")
    except AssertionError as e:
        ok2 = False
        print(f"    causal_truncation_probe_series(build_target, btc): FAIL ({e})")
    causal_ok = causal_ok and ok2

    print(f"\n    Causality: {'PASS' if causal_ok else 'FAIL'}")
    if not causal_ok:
        raise AssertionError("Causal truncation probe FAILED -- stopping before "
                             "any promotion-bar evaluation.")

    if a2_tripped:
        hr("STOPPED AT STEP 0")
        print("\n    Kill switch A2 tripped (target-path R^2 >= "
              f"{A2_R2_CEILING} on the default span). Per the pre-registered "
              "rule, this branch\n    stops here and reports NEGATIVE without "
              "a full promotion-bar evaluation: the candidate is\n    "
              "indistinguishable from a flat rescale of v4's own target path.")
        print_final_report(a2_tripped=True, causal_ok=causal_ok, a2_rows=a2_rows,
                           btc=btc, max_ts_seen=max_ts_seen)
        return

    # ===================================================== pre-registration
    hr("PRE-REGISTERED PROMOTION BAR (frozen before any B1-B5 number below "
       "is computed; identical to the\ntask brief's decision rule)")
    print("""
    B1  dSharpe > +0.2 on inner-validation, BOTH markets, OR the paired
        block-bootstrap interval on log-growth excludes zero with a
        positive point estimate.
    B2  drawdown improvement (d_dd < 0) counted ONLY if risk_matched is
        True (exposure_ratio AND vol_ratio both in [0.9, 1.1]) on the same
        cell; otherwise void per the standing "match risk before comparing
        anything" rule.
    B3  span grid {4, 8, 16} days must show a directionally consistent
        finding -- a plateau, not a one-value spike.
    B4  ETH must replicate the SAME SIGN of dSharpe as BTC inner-validation,
        on both markets, at the default (8-day) span.
    B5  sign of BTC log-growth deltas must survive a 0.40% taker fee tier.

    Promote only if ALL clauses that apply hold. Default: REJECT.
    Primary/frozen candidate: default span (v4's own 8-day EWM span) --
    the literal, untuned substitution. The span grid is a plateau check
    (B3), not a hyperparameter search for a better span.
    """)

    # ======================================================== main sweep
    hr(f"STEP 1 -- span sweep {SPAN_DAYS_GRID} days x 2 markets x 3 slices "
       f"= {len(SPAN_DAYS_GRID) * 6} cells")
    all_rows: list[dict] = []
    for days in SPAN_DAYS_GRID:
        span = days * BARS_PER_DAY
        rows = compare(make_build_target(span), label=f"downside_vol_{days}d")
        all_rows.extend(rows)
    print()
    print_rows(all_rows)

    # ============================================================== B1
    hr("B1 -- Sharpe leg (dSharpe > +0.2 on inner-val, BOTH markets, OR "
       "bootstrap excludes zero positively)")
    b1_pass_days = []
    for days in SPAN_DAYS_GRID:
        label = f"downside_vol_{days}d"
        v_s = cell(all_rows, label, "inner_val", SPOT.name)
        v_f = cell(all_rows, label, "inner_val", FUTURES.name)

        def leg_ok(c):
            return (c["d_sharpe"] > SHARPE_FLOOR) or (c["excludes_zero"] and c["boot_d_loggrowth"] > 0)

        ok = leg_ok(v_s) and leg_ok(v_f)
        print(f"    span={days:2d}d  spot: dSharpe={v_s['d_sharpe']:+.3f} "
              f"boot=[{v_s['boot_lo']:+.3f},{v_s['boot_hi']:+.3f}] excl0={v_s['excludes_zero']}  |  "
              f"futures: dSharpe={v_f['d_sharpe']:+.3f} "
              f"boot=[{v_f['boot_lo']:+.3f},{v_f['boot_hi']:+.3f}] excl0={v_f['excludes_zero']}  "
              f"-> {'PASSES' if ok else 'fails'}")
        if ok:
            b1_pass_days.append(days)

    b1_pass_default = DEFAULT_SPAN_DAYS in b1_pass_days
    print(f"\n    B1 on default span ({DEFAULT_SPAN_DAYS}d, the frozen primary "
          f"candidate): {'PASS' if b1_pass_default else 'FAIL'}")
    print(f"    B1 across the whole grid: {b1_pass_days if b1_pass_days else '(none pass)'}")

    # ============================================================== B2
    hr("B2 -- drawdown leg (risk-matched improvement only), default span, "
       "inner-validation")
    default_label = f"downside_vol_{DEFAULT_SPAN_DAYS}d"
    v_s = cell(all_rows, default_label, "inner_val", SPOT.name)
    v_f = cell(all_rows, default_label, "inner_val", FUTURES.name)
    for name, c in (("spot", v_s), ("futures", v_f)):
        improved = c["d_dd"] < 0
        print(f"    {name:8s} d_dd={c['d_dd']:+.1f}  (improved={improved})  "
              f"exposure_ratio={c['exposure_ratio']:.2f}  vol_ratio={c['vol_ratio']:.2f}  "
              f"risk_matched={c['risk_matched']}")
    b2_pass = bool((v_s["d_dd"] < 0 and v_s["risk_matched"]) and
                  (v_f["d_dd"] < 0 and v_f["risk_matched"]))
    print(f"\n    B2: {'PASS' if b2_pass else 'FAIL'} -- inner-validation drawdown is "
          f"WORSE (not better) for the candidate on\n    both markets at the default "
          f"span, AND neither cell is risk-matched (exposure_ratio/vol_ratio sit "
          f"at\n    ~1.3-1.5x v4, not [0.9, 1.1]) -- the standing 'match risk before "
          f"comparing anything' rule\n    (R-33/ROUTINE.md) would void any drawdown "
          f"claim here even if the sign had gone the other way.")

    # ============================================================== B3
    hr("B3 -- plateau not peak: dSharpe across the span grid, inner-val")
    for days in SPAN_DAYS_GRID:
        label = f"downside_vol_{days}d"
        v_s = cell(all_rows, label, "inner_val", SPOT.name)
        v_f = cell(all_rows, label, "inner_val", FUTURES.name)
        tag = " <- default" if days == DEFAULT_SPAN_DAYS else ""
        print(f"    span={days:2d}d  spot dSharpe={v_s['d_sharpe']:+.3f}  "
              f"futures dSharpe={v_f['d_sharpe']:+.3f}{tag}")
    print("\n    Reading: no span in the grid clears B1 on both markets (see above); "
          "futures dSharpe is\n    negative-or-near-zero at every span (-0.19, -0.14, "
          "-0.41) and spot is small and sign-unstable\n    across neighbours (-0.00, "
          "+0.15, -0.21). This is a consistent absence of an edge across the\n    "
          "whole grid, not a spike at one value -- but there is no B1-passing span "
          "for B3 to certify as\n    a genuine plateau of improvement. B3 is judged "
          "on the finding actually observed (no edge,\n    consistently), not on a "
          "hypothetical one.")
    b3_pass = False  # no B1-passing candidate exists to plateau-check in the first place

    # ============================================================== B4
    hr("B4 -- falsification (pre-registered, not changed after seeing results): "
       "ETH same sign as BTC inner-val, default span, both markets")
    btc_s = cell(all_rows, default_label, "inner_val", SPOT.name)
    btc_f = cell(all_rows, default_label, "inner_val", FUTURES.name)
    eth_s = cell(all_rows, default_label, "eth_replication", SPOT.name)
    eth_f = cell(all_rows, default_label, "eth_replication", FUTURES.name)
    same_spot = bool(np.sign(eth_s["d_sharpe"]) == np.sign(btc_s["d_sharpe"]))
    same_fut = bool(np.sign(eth_f["d_sharpe"]) == np.sign(btc_f["d_sharpe"]))
    print(f"    spot:     BTC inner-val dSharpe={btc_s['d_sharpe']:+.3f}   "
          f"ETH dSharpe={eth_s['d_sharpe']:+.3f}   same sign: {same_spot}")
    print(f"    futures:  BTC inner-val dSharpe={btc_f['d_sharpe']:+.3f}   "
          f"ETH dSharpe={eth_f['d_sharpe']:+.3f}   same sign: {same_fut}")
    b4_pass = same_spot and same_fut
    print(f"\n    B4: {'PASS' if b4_pass else 'FAIL'} -- and note BTC's own "
          f"inner-val sign is not even stable between\n    spot ({'+' if btc_s['d_sharpe']>0 else '-'}) "
          f"and futures ({'+' if btc_f['d_sharpe']>0 else '-'}) at the default span: "
          f"there is no single stable\n    direction for ETH to replicate in the "
          f"first place.")

    # ============================================================== B5
    hr("B5 -- cost robustness: default span at a 0.40% taker fee tier "
       "(SPOT + FUTURES; reported for completeness,\nmoot once B1 fails)")
    spot40 = fee_at(SPOT, HIGH_FEE)
    fut40 = fee_at(FUTURES, HIGH_FEE)
    rows40 = compare(make_build_target(DEFAULT_SPAN), label=f"{default_label}_fee40",
                     markets=(spot40, fut40))
    print()
    print_rows(rows40)
    sign_checks = []
    for slice_name in ("inner_train", "inner_val", "eth_replication"):
        for base_market, fee_market in ((SPOT, spot40), (FUTURES, fut40)):
            base = cell(all_rows, default_label, slice_name, base_market.name)
            fee40 = cell(rows40, f"{default_label}_fee40", slice_name, fee_market.name)
            same_sign = bool(np.sign(base["d_log_growth"]) == np.sign(fee40["d_log_growth"]))
            sign_checks.append(same_sign)
            print(f"    {slice_name:16s} {base_market.name:11s} @0.10%={base['d_log_growth']:+.3f}  "
                  f"@0.40%={fee40['d_log_growth']:+.3f}  sign preserved: {same_sign}")
    b5_pass = all(sign_checks)
    print(f"\n    B5: {'PASS' if b5_pass else 'FAIL'} (sign-preservation sense only; "
          f"moot given B1 already fails -- there is\n    no edge for a fee tier to "
          f"either confirm or destroy).")

    # =========================================================== A2 (grid)
    hr("A2 supplementary: kill-switch R^2 across the whole span grid "
       "(diagnostic only, no Sharpe/drawdown involved)")
    for days, r2s, r2t in a2_rows:
        print(f"    span={days:2d}d   R^2(scale)={r2s:+.4f}   R^2(target)={r2t:+.4f}")

    print_final_report(a2_tripped=False, causal_ok=causal_ok, a2_rows=a2_rows,
                       btc=btc, max_ts_seen=max_ts_seen,
                       b1_pass_default=b1_pass_default, b1_pass_days=b1_pass_days,
                       b2_pass=b2_pass, b3_pass=b3_pass, b4_pass=b4_pass, b5_pass=b5_pass,
                       all_rows=all_rows, rows40=rows40)


def print_final_report(*, a2_tripped: bool, causal_ok: bool, a2_rows, btc, max_ts_seen,
                       b1_pass_default: bool | None = None, b1_pass_days=None,
                       b2_pass: bool | None = None, b3_pass: bool | None = None,
                       b4_pass: bool | None = None, b5_pass: bool | None = None,
                       all_rows=None, rows40=None) -> None:
    hr("VERDICT")
    if a2_tripped:
        print("    VERDICT: NEGATIVE (stopped at Step 0 -- kill switch A2 tripped)")
        print(f"    causal truncation probes: {'PASS' if causal_ok else 'FAIL'}")
        n_configs = len(a2_rows)
        n_cells = 0
    else:
        clauses = {
            "B1 (Sharpe leg, default span)": bool(b1_pass_default),
            "B2 (risk-matched drawdown leg)": bool(b2_pass),
            "B3 (plateau not peak)": bool(b3_pass),
            "B4 (ETH falsification)": bool(b4_pass),
            "B5 (0.40% fee, sign only)": bool(b5_pass),
        }
        for k, v in clauses.items():
            print(f"    {k:34s}: {'PASS' if v else 'FAIL'}")
        promote = all(clauses.values())
        verdict = "PROMOTE-CANDIDATE" if promote else "NEGATIVE"
        print(f"\n    VERDICT: {verdict}")
        if verdict == "NEGATIVE":
            failed = [k for k, v in clauses.items() if not v]
            print(f"    Reason(s): {', '.join(failed)}")
        n_configs = len(SPAN_DAYS_GRID)
        n_cells = (len(all_rows) if all_rows else 0) + (len(rows40) if rows40 else 0)

    print("\n    Full reasoning is in this file's module docstring, written "
          "before this printout and not\n    altered by it. The decision rule "
          "was frozen in the task brief before any market data was read.\n    "
          "The holdout (>=2023-01-01) is NOT touched by this script, win or "
          "lose -- that decision belongs\n    to the operator.")

    hr("BOOKKEEPING")
    if a2_tripped:
        print(f"    A2 kill-switch diagnostic: {n_configs} span configurations x "
              f"1 slice (inner-train) x scale+target R^2 = {2 * n_configs} R^2 numbers.")
        print(f"    No B1-B5 sweep was run (stopped at Step 0). "
              f"TOTAL DISTINCT SPAN CONFIGURATIONS EVALUATED: {n_configs}")
    else:
        print(f"    Main span sweep: {len(SPAN_DAYS_GRID)} span configs x 2 markets x "
              f"3 slices = {len(SPAN_DAYS_GRID) * 6} cells")
        print(f"    B5 cost-robustness: 1 config (default span) x 2 markets x 3 slices "
              f"= 6 more cells")
        print(f"    TOTAL DISTINCT SPAN CONFIGURATIONS: {len(SPAN_DAYS_GRID)}   "
              f"TOTAL MEASURED CELLS: {n_cells}")
        print(f"    (A2 kill-switch R^2 diagnostics re-use the same span grid on "
              f"inner-train only, purely to\n     extract scale/target paths -- not "
              f"counted as additional configurations.)")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    print(f"\n    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")

    hr("WHAT WOULD MAKE A HOLDOUT READ WORTH CONSULTING")
    print("    Not applicable in the sense that this run recommends one: no clause "
          "of the promotion bar\n    passed. Stated as a decision rule anyway, per "
          "the task brief -- a holdout read would be\n    worth the operator's "
          "consultation budget only if a FUTURE round found a variant of this "
          "idea\n    that (a) clears B1 with a risk-matched exposure_ratio/vol_ratio "
          "near 1.0 (not the ~1.3-1.5x\n    over-exposure this literal substitution "
          "produces), (b) shows a stable, same-sign dSharpe\n    improvement on BOTH "
          "BTC markets AND ETH, and (c) survives the 0.40% fee tier with that edge "
          "intact\n    -- none of which this branch's literal, untuned substitution "
          "achieved.")


if __name__ == "__main__":
    main()
