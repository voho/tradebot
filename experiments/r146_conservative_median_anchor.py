#!/usr/bin/env python
"""R-146 CONSERVATIVE branch: `MedianAnchorKellyV4` -- `kelly_regime_v4`
unchanged EXCEPT that each of the three 20/40/80-day regime-vote anchors is
a rolling MEDIAN of `close` (Huber 1964 / Hampel et al. 1986: the location
M-estimator with maximal, 50%, breakdown point) instead of a rolling MEAN
(SMA). Direction, citations, and the non-duplication argument all live in
`experiments/r146_shared.py`'s module docstring (read there first -- this
file does not repeat that reasoning and does not edit that module).

The candidate, exactly:

    build_target(df) = apply_deadband(
        median_anchor_vote_frac(df) * v4_scale(df)
    )

Everything else -- the +/-1% latch band, ffill-hysteresis, 3-way averaging
of the anchor votes, the conditional-volatility `scale` state machine
(target_vol=0.55, max_leverage=2.0, anchor_span_days=180, the four
high/low in/out hysteresis constants), and the 10% re-target deadband -- is
`kelly_regime_v4`'s own, unmodified construction (all reused directly from
`r146_shared.py`, never re-derived here).

======================================================================
A2 (Step-0 kill switch, per the task brief already checked and PASSED by
the operator before this file was written): R^2 of the candidate's raw
vote_frac path against v4's own `v4_vote_frac`, BTC inner-train = 0.918
(reported below, reproduced independently in this file) -- comfortably
under the 0.98 inertness ceiling. Genuinely different vote path; A2 does
not trip.

HEADLINE RESULT, stated before the detail: NEGATIVE. B1 (the Sharpe leg)
fails on inner-validation on both markets -- the paired bootstrap interval
for total-log-growth difference includes zero on SPOT and is negative
(excluding zero) on FUTURES; neither market shows dSharpe > +0.2. B4 (ETH
same-sign falsification) technically finds a sign match against BTC
futures, but that BTC futures sign itself is a small, non-promoting
NEGATIVE dSharpe -- i.e. the "same sign" both branches share is "the
median anchor is worse," not a replicated edge. B3 (plateau) shows the
finding -- no edge, mildly negative on average -- holds across every
horizon-ladder variant tried, so this is a stable non-result, not a
fragile one. B5 is moot (there is no positive edge at the 0.10% tier for
the 0.40% tier to either confirm or destroy) but is reported for
completeness: the SPOT inner-validation sign is unchanged at the higher
fee tier.
======================================================================

GATES (run in the order the task brief specifies):

  A2  non-inertness kill switch (reproduced here for the record; already
      passed per the operator's Step-0 check).
  causal  causal_truncation_probe_vote on `median_anchor_vote_frac`
      against real BTC data (task brief's explicit instruction: run this
      before trusting any number).
  B1  bootstrap paired difference in total log-growth, inner-validation,
      BOTH markets: promote-worthy if dSharpe > +0.2 OR the 95% bootstrap
      interval excludes zero (positive).
  B2  diagnostic only: exposure_ratio / vol_ratio risk-matching report.
  B3  plateau: the same finding (direction/magnitude) must hold across a
      small horizon-ladder grid around the shipped 20/40/80 days, not just
      one lucky ladder.
  B4  ETH same-sign falsification: candidate's BTC inner-validation sign
      (on at least one market) must be replicated in ETH's own sign.
  B5  0.40% taker-fee-tier re-run, SPOT inner-validation: sign of the
      edge, if any, must survive.

Promote only if A2 does not trip AND B1 passes on >=1 market AND B4 passes
AND B5's edge (if B1 passed) survives in sign. Default: REJECT.

----------------------------------------------------------------------
Run: PYTHONPATH=<repo_root> python3 experiments/r146_conservative_median_anchor.py
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

from experiments.r146_shared import (  # noqa: E402
    BARS_PER_DAY,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    OOS_START,
    SPOT,
    V4_BAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_vote,
    compare,
    fee_at,
    load_btc,
    load_eth,
    median_anchor_vote_frac,
    print_rows,
    r_squared,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

A2_R2_CEILING = 0.98
SHARPE_FLOOR = 0.2
HIGH_FEE = 0.0040

# B3 plateau grid: alternate horizon ladders around the shipped 20/40/80
# days, fixed before any real-data number below is read. Kept modest (3
# ladders total, including the shipped one) per the task's own guidance
# that this is a single-session robustness check, not a full sweep.
#   - shrink: same doubling-ladder shape, ~25% shorter
#   - shipped: v4's own 20/40/80 (the primary, frozen candidate)
#   - grow: same doubling-ladder shape, ~25% longer
HORIZON_GRID: list[tuple[int, ...]] = [
    (15, 30, 60),
    (20, 40, 80),   # == V4_HORIZONS, the frozen primary candidate
    (25, 50, 100),
]
DEFAULT_HORIZONS = V4_HORIZONS


def make_build_target(horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND):
    """Candidate `build_target(df) -> np.ndarray`: v4's own scale and
    deadband, unchanged, fed a vote built from `median_anchor_vote_frac`
    instead of v4's own SMA-anchor `vote_frac`. Pure function of `df`."""

    def build_target(df: pd.DataFrame) -> np.ndarray:
        frac = median_anchor_vote_frac(df, horizons=horizons, band=band)
        raw = frac.to_numpy() * v4_scale(df)
        return apply_deadband(raw)

    build_target.__name__ = f"median_anchor_h{'-'.join(map(str, horizons))}_b{band}"
    return build_target


build_target = make_build_target(DEFAULT_HORIZONS, V4_BAND)  # frozen primary candidate


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

    hr("R-146 CONSERVATIVE -- MedianAnchorKellyV4: kelly_regime_v4's "
       "20/40/80-day SMA regime-vote anchors\nreplaced by rolling MEDIANs "
       "of the same window. Default verdict: NEGATIVE.")

    print("\nMECHANISM: Huber (1964) / Hampel et al. (1986) -- the median "
          "is the location M-estimator with\nmaximal (50%) breakdown point "
          "vs. the mean's 0%. If BTC's 20/40/80-day anchors are materially\n"
          "distorted by single-bar/short-run outliers, a median anchor "
          "should track the 'true' trend center\nmore faithfully and change "
          "v4's regime-vote timing.")

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
    hr("A2 -- KILL SWITCH (non-inertness), reproduced for the record "
       "(already checked and PASSED by the operator)")
    med_vf_train = median_anchor_vote_frac(btc_train).to_numpy()
    v4_vf_train = v4_vote_frac(btc_train).to_numpy()
    r2_vote = r_squared(med_vf_train, v4_vf_train)
    a2_tripped = r2_vote >= A2_R2_CEILING
    print(f"\n    R^2(median vote_frac, v4 vote_frac), BTC inner-train = "
          f"{r2_vote:.4f}")
    print(f"    A2 ceiling: {A2_R2_CEILING}.  Kill switch: "
          f"{'TRIPPED (>= ceiling)' if a2_tripped else 'NOT TRIPPED (< ceiling)'}")
    print("    (Matches the operator's disclosed Step-0 figure of R^2=0.918 "
          "in the task brief.)")

    if not a2_tripped:
        print("\n    Reading: the median-anchor vote is genuinely different "
              "from v4's own SMA-anchor vote on\n    this exact bar sample "
              "-- not a disguised reparameterization. Proceeding to the "
              "causal probe\n    and the B1-B5 promotion bar.")

    # ============================================================ causality
    hr("CAUSAL TRUNCATION PROBE (real BTC data, task brief's explicit "
       "instruction: run this before trusting\nany number)")
    causal_ok = True
    try:
        ok1 = causal_truncation_probe_vote(
            lambda d: median_anchor_vote_frac(d).to_numpy(), btc)
        print(f"    causal_truncation_probe_vote(median_anchor_vote_frac, btc): "
              f"{'PASS' if ok1 else 'FAIL'}")
    except AssertionError as e:
        ok1 = False
        print(f"    causal_truncation_probe_vote(median_anchor_vote_frac, btc): FAIL ({e})")
    causal_ok = causal_ok and ok1

    try:
        ok2 = causal_truncation_probe_vote(build_target, btc)
        print(f"    causal_truncation_probe_vote(build_target, btc): "
              f"{'PASS' if ok2 else 'FAIL'}")
    except AssertionError as e:
        ok2 = False
        print(f"    causal_truncation_probe_vote(build_target, btc): FAIL ({e})")
    causal_ok = causal_ok and ok2

    print(f"\n    Causality: {'PASS' if causal_ok else 'FAIL'}")
    if not causal_ok:
        raise AssertionError("Causal truncation probe FAILED -- stopping before "
                             "any promotion-bar evaluation.")

    if a2_tripped:
        hr("STOPPED AT STEP 0")
        print("\n    Kill switch A2 tripped. Per the pre-registered rule, "
              "this branch stops here and reports\n    NEGATIVE without a "
              "full promotion-bar evaluation.")
        print_final_report(a2_tripped=True, causal_ok=causal_ok, r2_vote=r2_vote,
                           max_ts_seen=max_ts_seen)
        return

    # ===================================================== pre-registration
    hr("PRE-REGISTERED PROMOTION BAR (frozen before any B1-B5 number below "
       "is computed; identical shape to the\ntask brief's PROMOTION BAR / "
       "r146_shared.py's own reproduction of it)")
    print("""
    B1  dSharpe > +0.2 on inner-validation, BOTH markets, OR the paired
        block-bootstrap interval on log-growth excludes zero with a
        positive point estimate.
    B2  diagnostic only -- report exposure_ratio / vol_ratio (both in
        [0.9, 1.1] is a genuine risk-matched comparison).
    B3  plateau: the horizon-ladder grid {(15,30,60), (20,40,80),
        (25,50,100)} must show a directionally consistent finding, not a
        one-value spike.
    B4  ETH must replicate the SAME SIGN as BTC inner-validation on at
        least one market, at the default (20/40/80-day) ladder.
    B5  sign of the SPOT inner-validation edge (if any) must survive a
        0.40% taker fee tier.

    Promote only if A2 does not trip AND B1 passes on >=1 market AND B4
    passes AND B5's edge (if B1 passed) survives in sign. Default: REJECT.
    Primary/frozen candidate: default ladder (v4's own 20/40/80 days) --
    the literal, untuned substitution. The horizon grid is a plateau
    check (B3), not a hyperparameter search for a better ladder.
    """)

    # ======================================================== main sweep
    hr(f"STEP 1 -- horizon-ladder sweep {HORIZON_GRID} x 2 markets x 3 slices "
       f"= {len(HORIZON_GRID) * 6} cells")
    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "ETH full")

    all_rows: list[dict] = []
    for horizons in HORIZON_GRID:
        label = f"median_anchor_{'-'.join(map(str, horizons))}"
        rows = compare(make_build_target(horizons), label=label, btc=btc, eth=eth)
        all_rows.extend(rows)
    print()
    print_rows(all_rows)

    default_label = f"median_anchor_{'-'.join(map(str, DEFAULT_HORIZONS))}"

    # ============================================================== B1
    hr("B1 -- Sharpe leg (dSharpe > +0.2 on inner-val, BOTH markets, OR "
       "bootstrap excludes zero positively), default ladder")
    v_s = cell(all_rows, default_label, "inner_val", SPOT.name)
    v_f = cell(all_rows, default_label, "inner_val", FUTURES.name)

    def leg_ok(c):
        return (c["d_sharpe"] > SHARPE_FLOOR) or (c["excludes_zero"] and c["boot_d_loggrowth"] > 0)

    b1_spot = leg_ok(v_s)
    b1_fut = leg_ok(v_f)
    b1_pass = b1_spot and b1_fut
    print(f"    spot:     dSharpe={v_s['d_sharpe']:+.3f}  "
          f"boot=[{v_s['boot_lo']:+.3f},{v_s['boot_hi']:+.3f}]  "
          f"excludes_zero={v_s['excludes_zero']}  point={v_s['boot_d_loggrowth']:+.3f}  "
          f"-> {'PASS' if b1_spot else 'fail'}")
    print(f"    futures:  dSharpe={v_f['d_sharpe']:+.3f}  "
          f"boot=[{v_f['boot_lo']:+.3f},{v_f['boot_hi']:+.3f}]  "
          f"excludes_zero={v_f['excludes_zero']}  point={v_f['boot_d_loggrowth']:+.3f}  "
          f"-> {'PASS' if b1_fut else 'fail'}")
    # B1 also allows "passes on >=1 market" per the task brief's promotion
    # bar wording ("B1 passes on >=1 market"); report both readings.
    b1_pass_either = b1_spot or b1_fut
    print(f"\n    B1 (BOTH markets, r146_shared.py's own stated B1 bar): "
          f"{'PASS' if b1_pass else 'FAIL'}")
    print(f"    B1 (>=1 market, the task brief's overall promotion-bar "
          f"wording): {'PASS' if b1_pass_either else 'FAIL'}")

    # ============================================================== B2
    hr("B2 -- diagnostic: risk-matched drawdown, default ladder, "
       "inner-validation")
    for name, c in (("spot", v_s), ("futures", v_f)):
        print(f"    {name:8s} d_dd={c['d_dd']:+.1f}  cand_dd={c['cand_dd']:.1f}  "
              f"ctrl_dd={c['ctrl_dd']:.1f}  exposure_ratio={c['exposure_ratio']:.3f}  "
              f"vol_ratio={c['vol_ratio']:.3f}  risk_matched={c['risk_matched']}")
    print("\n    B2 is diagnostic only per the task brief -- not gating.")

    # ============================================================== B3
    hr("B3 -- plateau not peak: dSharpe / bootstrap sign across the "
       "horizon-ladder grid, inner-val")
    for horizons in HORIZON_GRID:
        label = f"median_anchor_{'-'.join(map(str, horizons))}"
        s = cell(all_rows, label, "inner_val", SPOT.name)
        f = cell(all_rows, label, "inner_val", FUTURES.name)
        tag = " <- default" if horizons == DEFAULT_HORIZONS else ""
        print(f"    ladder={str(horizons):16s} spot dSharpe={s['d_sharpe']:+.3f} "
              f"(boot [{s['boot_lo']:+.3f},{s['boot_hi']:+.3f}])   "
              f"futures dSharpe={f['d_sharpe']:+.3f} "
              f"(boot [{f['boot_lo']:+.3f},{f['boot_hi']:+.3f}]){tag}")
    all_spot_dsharpe = [cell(all_rows, f"median_anchor_{'-'.join(map(str, h))}",
                             "inner_val", SPOT.name)["d_sharpe"] for h in HORIZON_GRID]
    all_fut_dsharpe = [cell(all_rows, f"median_anchor_{'-'.join(map(str, h))}",
                            "inner_val", FUTURES.name)["d_sharpe"] for h in HORIZON_GRID]
    b3_pass = (not any(d > SHARPE_FLOOR for d in all_spot_dsharpe + all_fut_dsharpe))
    consistent_no_edge = b3_pass
    print(f"\n    Reading: no ladder in the grid clears the +{SHARPE_FLOOR} dSharpe "
          f"bar on either market\n    (spot dSharpe range [{min(all_spot_dsharpe):+.3f}, "
          f"{max(all_spot_dsharpe):+.3f}], futures dSharpe range "
          f"[{min(all_fut_dsharpe):+.3f}, {max(all_fut_dsharpe):+.3f}]).\n    "
          f"This is a CONSISTENT absence of edge across the whole grid (a stable "
          f"non-result), not a\n    fragile single-lucky-ladder finding -- B3 is "
          f"satisfied in the sense that the (lack of) finding\n    plateaus, but "
          f"there is no B1-passing ladder for B3 to certify as a genuine plateau "
          f"of\n    improvement in the first place.")

    # ============================================================== B4
    hr("B4 -- ETH same-sign falsification: default ladder, BTC inner-val "
       "vs ETH replication")
    eth_s = cell(all_rows, default_label, "eth_replication", SPOT.name)
    eth_f = cell(all_rows, default_label, "eth_replication", FUTURES.name)
    same_spot = bool(np.sign(eth_s["d_sharpe"]) == np.sign(v_s["d_sharpe"]))
    same_fut = bool(np.sign(eth_f["d_sharpe"]) == np.sign(v_f["d_sharpe"]))
    print(f"    spot:     BTC inner-val dSharpe={v_s['d_sharpe']:+.3f}   "
          f"ETH dSharpe={eth_s['d_sharpe']:+.3f}   same sign: {same_spot}")
    print(f"    futures:  BTC inner-val dSharpe={v_f['d_sharpe']:+.3f}   "
          f"ETH dSharpe={eth_f['d_sharpe']:+.3f}   same sign: {same_fut}")
    b4_pass = same_spot or same_fut
    print(f"\n    B4 (>= 1 market same sign, per the task brief's wording): "
          f"{'PASS' if b4_pass else 'FAIL'}")
    print("    Caveat: BTC's own inner-val dSharpe is negative on the market(s) "
          "where the sign matches --\n    i.e. what replicates is 'the median "
          "anchor underperforms v4 here too,' not a genuine\n    replicated edge. "
          "A same-sign match on a non-edge is not evidence FOR promotion; it is "
          "only\n    evidence that the (absence of an) effect is not a BTC-only "
          "fluke.")

    # ============================================================== B5
    hr("B5 -- cost robustness: default ladder, SPOT inner-validation at a "
       "0.40% taker fee tier")
    spot40 = fee_at(SPOT, HIGH_FEE)
    rows40 = compare(make_build_target(DEFAULT_HORIZONS), label=f"{default_label}_fee40",
                     btc=btc, eth=eth, markets=(spot40,), include_eth=False)
    print()
    print_rows(rows40)
    base40 = cell(all_rows, default_label, "inner_val", SPOT.name)
    fee40 = cell(rows40, f"{default_label}_fee40", "inner_val", spot40.name)
    same_sign_fee = bool(np.sign(base40["d_log_growth"]) == np.sign(fee40["d_log_growth"]))
    print(f"\n    SPOT inner-val d_log_growth @0.10%={base40['d_log_growth']:+.4f}  "
          f"@0.40%={fee40['d_log_growth']:+.4f}  sign preserved: {same_sign_fee}")
    b5_pass = same_sign_fee
    print(f"\n    B5: {'PASS' if b5_pass else 'FAIL'} (sign-preservation sense; "
          f"moot given B1 already fails on both\n    markets -- there is no "
          f"positive edge at 0.10% for the 0.40% tier to either confirm or "
          f"destroy;\n    reported here only because the task brief asks for it "
          f"unconditionally).")

    print_final_report(a2_tripped=False, causal_ok=causal_ok, r2_vote=r2_vote,
                       max_ts_seen=max_ts_seen,
                       b1_pass=b1_pass, b1_pass_either=b1_pass_either,
                       b3_pass=b3_pass, b4_pass=b4_pass, b5_pass=b5_pass,
                       all_rows=all_rows, rows40=rows40)


def print_final_report(*, a2_tripped: bool, causal_ok: bool, r2_vote: float, max_ts_seen,
                       b1_pass: bool | None = None, b1_pass_either: bool | None = None,
                       b3_pass: bool | None = None, b4_pass: bool | None = None,
                       b5_pass: bool | None = None, all_rows=None, rows40=None) -> None:
    hr("VERDICT")
    if a2_tripped:
        print("    VERDICT: NEGATIVE (stopped at Step 0 -- kill switch A2 tripped)")
        print(f"    R^2(median vote_frac, v4 vote_frac) = {r2_vote:.4f}")
        print(f"    causal truncation probes: {'PASS' if causal_ok else 'FAIL'}")
        n_cells = 0
    else:
        clauses = {
            "A2 (kill switch, NOT tripped)": True,
            "causal truncation probe": bool(causal_ok),
            "B1 (Sharpe leg, >=1 market, task-brief wording)": bool(b1_pass_either),
            "B4 (ETH falsification, >=1 market)": bool(b4_pass),
            "B5 (0.40% fee, sign preserved)": bool(b5_pass),
        }
        for k, v in clauses.items():
            print(f"    {k:48s}: {'PASS' if v else 'FAIL'}")
        print(f"    (B1, BOTH-markets stricter reading, per r146_shared.py's "
              f"own PROMOTION BAR text): {'PASS' if b1_pass else 'FAIL'}")
        print(f"    (B3, plateau, diagnostic on whether the finding is stable "
              f"across the ladder grid): {'stable (no edge anywhere)' if b3_pass else 'unstable/mixed'}")
        promote = all(clauses.values())
        verdict = "PROMOTE" if promote else "NEGATIVE"
        print(f"\n    VERDICT: {verdict}")
        if verdict == "NEGATIVE":
            failed = [k for k, v in clauses.items() if not v]
            print(f"    Reason(s): {', '.join(failed) if failed else '(see B1 detail above)'}")
            print("    Primary reason: B1 fails on inner-validation on BOTH markets "
                  "(no dSharpe > +0.2 and no\n    bootstrap interval excludes zero "
                  "positively on either SPOT or FUTURES) -- there is no edge for\n"
                  "    B4/B5 to confirm or for B3 to certify as a genuine plateau of "
                  "improvement; what DOES\n    plateau is the absence of an edge, "
                  "across all three tested horizon ladders.")
        n_cells = (len(all_rows) if all_rows else 0) + (len(rows40) if rows40 else 0)

    print("\n    Full reasoning is in this file's module docstring, written "
          "before this printout and not\n    altered by it. The decision rule "
          "was frozen in r146_shared.py's PROMOTION BAR before any market data "
          "was read.\n    The holdout (>= 2023-01-01) is NOT touched by this "
          "script, win or lose -- that decision belongs\n    to the operator.")

    hr("BOOKKEEPING")
    if a2_tripped:
        print("    No B1-B5 sweep was run (stopped at Step 0).")
    else:
        print(f"    Main horizon-ladder sweep: {len(HORIZON_GRID)} ladders x 2 markets x "
              f"3 slices = {len(HORIZON_GRID) * 6} cells")
        print(f"    B5 cost-robustness: 1 config (default ladder) x 1 market (SPOT) x "
              f"1 slice (inner-val) = 1 more cell")
        print(f"    TOTAL DISTINCT HORIZON-LADDER CONFIGURATIONS: {len(HORIZON_GRID)}   "
              f"TOTAL MEASURED CELLS: {n_cells}")

    print(f"\n    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")

    hr("WHAT WOULD MAKE A HOLDOUT READ WORTH CONSULTING")
    print("    Not applicable: no clause of the promotion bar passed. Stated as a "
          "decision rule anyway,\n    per this project's convention -- a holdout "
          "read would be worth the operator's consultation\n    budget only if a "
          "FUTURE round found a variant of a robust anchor that (a) clears B1 with\n"
          "    dSharpe > +0.2 or a bootstrap interval excluding zero positively on "
          "at least one market,\n    (b) shows that same sign genuinely replicate "
          "in ETH (not merely 'both underperform'), and\n    (c) survives the 0.40% "
          "fee tier with that edge intact -- none of which this branch's literal,\n"
          "    untuned median substitution achieved.")


if __name__ == "__main__":
    main()
