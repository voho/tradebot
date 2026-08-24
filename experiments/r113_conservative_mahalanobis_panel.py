"""R-113 CONSERVATIVE BRANCH: Mahalanobis-distance panel-level novelty brake
on `xsmom_entry_band`.

This file is the executable companion to the pre-registration frozen in
`experiments/r113_shared.py` (read that file's docstring first -- it is this
round's literature, failure modes, Step-0 grid resolution and holdout-read
convention, none of which are restated here). This file does exactly one
thing: run `model="mahalanobis"` at the frozen operating point
`(PRIMARY_THRESH, PRIMARY_MAXD) = (0.90, 1.0)` through the pre-registered
D1/D2/D3/D5/SCRAMBLE battery and print/save the real numbers, with full
honesty. It does not edit `r113_shared.py` and does not touch anything with
"knn" or "novel" in its name -- that is the parallel branch's file.

=====================================================================
WHICH COMPARISON THE D1-D5 GATES ARE KEYED TO -- READ THIS FIRST
=====================================================================

R-113's own question is "does adding the novelty brake help the ALREADY
REGISTERED, undiscounted `xsmom_entry_band` construction" -- not "does
xsmom beat buy-and-hold" (settled rounds ago) and not "does xsmom beat a
risk-matched passive basket" (R-65's own question, also settled). So the
PRIMARY comparator for every D1/D2/D3/D5 cell below is:

    candidate = build_r113_targets(aligned, "mahalanobis", 0.90, 1.0)
    bench     = frozen_targets(aligned)     # the undiscounted construction

i.e. candidate (discounted) vs frozen (undiscounted), on identical price
paths. This is a direct A/B on the brake itself, not a benchmark-relative
statement, and it needs no risk-matching: candidate's exposure is `(1 -
discount) * frozen`'s by construction (discount in [0, 1]), so the two arms
are never independently-sized strategies whose volatility could be
compared to the wrong basket -- they are the same strategy with and without
one multiplicative adjustment applied.

`volmatched_hold_equity`, `matched_hold_targets` and `static_hold_equity`
are ALSO reported below, on both frozen and candidate, strictly as
secondary context (continuity with R-63 onward's convention of reporting
multiple comparator cells) -- they are never used to decide D1/D2/D3/D5.

=====================================================================
WARMUP DEVIATION FROM THE R-63...R-111 CONVENTION, DISCLOSED
=====================================================================

Every prior multi-asset branch aligns each evaluation window with
`warm_window(window)`, which backs the window start off by `WARM_DAYS = 91`
days -- enough for R-63's score, which is this round's frozen `score`
component too. It is NOT enough for this round's OWN feature:
`BASELINE_WINDOW_DAYS = 365` with `MIN_REF_DAYS = 90` needs up to 365 days
of panel history before the Mahalanobis reference distribution is fully
warmed. Backing a `W_VAL` or `W_HOLD` cell off by only 91 days would leave
much of the early evaluation window running on an under-warmed (or, before
`MIN_REF_DAYS`, entirely absent -> zero-discount) reference, understating
whatever effect this round is trying to measure -- not a subtle bug, a
structural mismatch between this round's own constant and the inherited
convention.

Fix, applied uniformly below: every cell in this file aligns from the
EARLIEST available bar in the relevant universe (`align_frames(sub, (None,
window[1]))`) through the window's own end, computes `frozen_targets` /
`build_r113_targets` over that full span (both are causal end-to-end, and
`check_causality` re-confirms it live below), and ONLY THEN slices the
result down to the evaluation window. This never reads past `window[1]`,
so it introduces no lookahead and no holdout leakage (`W_VAL`'s own align
call is capped at `W_VAL[1] = "2022-12-31"`, `W_HOLD` is the one cell
explicitly allowed to see through the last committed bar). It simply gives
the panel model the same amount of real history a live deployment would
have had, which is the only way to score this round's own construction
under its own stated constants rather than under a different round's
warmup budget.

Run: `python experiments/r113_conservative_mahalanobis_panel.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r113_shared import (  # noqa: E402
    BOOT_KW,  # noqa: F401
    OUT_DIR,
    PRIMARY_MAXD,
    PRIMARY_THRESH,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_FREE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_HOLD,
    W_TRAIN,
    W_VAL,
    align_frames,
    build_r113_targets,
    check_causality,
    compare,
    config_count,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    excludes_zero,
    frontier_row,
    frozen_targets,
    further_work as r113_further_work,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    scramble_fixed_perm,
    simulate_portfolio,
    static_hold_equity,
    step0_gate,
    print_step0_report,
    volmatched_hold_equity,
    warm_window,
)
from tradebot.inference import daily_returns, total_log_return  # noqa: E402

MODEL = "mahalanobis"
RESULTS_CSV = OUT_DIR / "conservative_mahalanobis_results.csv"
SCRAMBLE_CSV = OUT_DIR / "conservative_mahalanobis_scramble.csv"
REPORT_MD = OUT_DIR / "conservative_mahalanobis.md"


def _pt(label, val):
    print(f"  {label:46s} {val}")


def _fmt_cmp(cmp: dict) -> str:
    return (f"growth_diff={cmp['growth_diff']:.6f} "
            f"[{cmp['growth_lo']:.6f}, {cmp['growth_hi']:.6f}]  "
            f"dd_diff={cmp['dd_diff']:.6f} [{cmp['dd_lo']:.6f}, {cmp['dd_hi']:.6f}]  "
            f"n_days={cmp['n_days']}")


def build_cell(frames: dict, universe, window, model: str, thresh: float, maxd: float):
    """Aligned prices + frozen/candidate targets, warmed from the EARLIEST
    available bar (see module docstring) through `window[1]`, sliced to the
    evaluation window only for scoring. Never reads past `window[1]`."""
    sub = {t: frames[t] for t in universe}
    aligned = align_frames(sub, (None, window[1]))
    frozen = frozen_targets(aligned)
    candidate = build_r113_targets(aligned, model, thresh, maxd)
    idx = aligned[universe[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    aligned_eval = {t: df.loc[idx] for t, df in aligned.items()}
    return aligned_eval, frozen.loc[idx], candidate.loc[idx]


def cell_cmp(frozen_t, cand_t, aligned_eval, market):
    """Run both arms through the identical simulator on the identical price
    path and return `compare(candidate, frozen)`, plus both equity curves."""
    frozen_eq = simulate_portfolio(frozen_t, aligned_eval, market)
    cand_eq = simulate_portfolio(cand_t, aligned_eval, market)
    return compare(cand_eq, frozen_eq), frozen_eq, cand_eq


def main():
    print("=" * 78)
    print("R-113 CONSERVATIVE BRANCH: Mahalanobis panel-novelty brake on xsmom_entry_band")
    print(f"model={MODEL}  PRIMARY_THRESH={PRIMARY_THRESH}  PRIMARY_MAXD={PRIMARY_MAXD}")
    print("=" * 78)

    configs_before = config_count()

    # ------------------------------------------------------------- STEP 0
    print("\n--- STEP 0: reconfirm the Step-0 gate live, W_TRAIN, U8 ---")
    frames = load_universe(UNIVERSE_8)
    aligned_train = align_frames(frames, warm_window(W_TRAIN))

    causality_ok = check_causality(
        lambda al: build_r113_targets(al, MODEL, PRIMARY_THRESH, PRIMARY_MAXD),
        aligned_train, cut_from_end=20_000,
    )
    _pt("check_causality (build_r113_targets, mahalanobis)", causality_ok)

    step0 = step0_gate(aligned_train, MODEL, PRIMARY_THRESH, PRIMARY_MAXD)
    print_step0_report(f"W_TRAIN, U8, {MODEL} @ ({PRIMARY_THRESH}, {PRIMARY_MAXD})", step0)

    if not causality_ok:
        print("\n!!! CAUSALITY CHECK FAILED -- ABORTING BEFORE ANY DECISIVE NUMBER !!!")
        return
    if not step0["passed"]:
        print("\n!!! STEP-0 GATE FAILED LIVE (docstring claimed PASS) -- ABORTING, "
              "REPORT THIS DISCREPANCY !!!")
        return

    # ------------------------------------------------------------- STEP 3: D1/D2/D5
    print("\n--- STEP 3: D1/D2/D5 -- W_FULL6, U6, candidate vs FROZEN (undiscounted) ---")
    aligned_full6, frozen_full6, cand_full6 = build_cell(
        frames, UNIVERSE_6, W_FULL6, MODEL, PRIMARY_THRESH, PRIMARY_MAXD
    )
    params = {"model": MODEL, "thresh": PRIMARY_THRESH, "max_discount": PRIMARY_MAXD}

    primary_net_base, frozen_eq_base, cand_eq_base = cell_cmp(
        frozen_full6, cand_full6, aligned_full6, SPOT_BASE
    )
    primary_net_real, frozen_eq_real, cand_eq_real = cell_cmp(
        frozen_full6, cand_full6, aligned_full6, SPOT_REAL
    )
    primary_gross, frozen_eq_gross, cand_eq_gross = cell_cmp(
        frozen_full6, cand_full6, aligned_full6, SPOT_FREE
    )

    row_base = frontier_row(
        "r113_mahalanobis_vs_frozen", params, cand_full6,
        primary_net_base, primary_gross, "FROZEN_XSMOM(undiscounted)",
        "W_FULL6", "U6", fee_tier="SPOT_BASE",
    )
    row_real = frontier_row(
        "r113_mahalanobis_vs_frozen", params, cand_full6,
        primary_net_real, primary_gross, "FROZEN_XSMOM(undiscounted)",
        "W_FULL6", "U6", fee_tier="SPOT_REAL",
    )

    d1 = d1_pass(row_base)
    d2 = d2_pass(row_base)
    d5 = d5_pass(row_base)
    d1_real = d1_pass(row_real)
    d2_real = d2_pass(row_real)

    print("  [SPOT_BASE, 0.10%] candidate vs frozen:", _fmt_cmp(primary_net_base))
    print("  [SPOT_REAL, 0.40%] candidate vs frozen:", _fmt_cmp(primary_net_real))
    print("  [SPOT_FREE, 0.00%] candidate vs frozen (gross, for D5):", _fmt_cmp(primary_gross))
    _pt("D1 PASS (SPOT_BASE, primary)", d1)
    _pt("D2 PASS (SPOT_BASE, primary)", d2)
    _pt("D5 PASS (gross_growth_diff >= D5_BAR_R68)", d5)
    _pt("D1 PASS (SPOT_REAL, robustness only)", d1_real)
    _pt("D2 PASS (SPOT_REAL, robustness only)", d2_real)
    print("  NOTE: D5_BAR_R68 was calibrated for a candidate-vs-VOLMATCH_HOLD gross")
    print("  comparison in R-65/R-68. Applied here to candidate-vs-FROZEN per this")
    print("  round's own primary question, it is a structurally harder bar: the brake")
    print("  can only ever REDUCE exposure relative to frozen, so gross_growth_diff")
    print("  here is bounded well below the scale D5_BAR_R68 was set against. Applied")
    print("  literally as pre-registered/instructed, not adjusted after seeing the number.")

    # ------------------------------------------------------- secondary context
    print("\n--- SECONDARY CONTEXT (SPOT_BASE, W_FULL6, U6) -- NOT scored, informational only ---")
    vh_frozen_eq, c_f, vol_f, matched_f = volmatched_hold_equity(
        frozen_eq_base, aligned_full6, UNIVERSE_6, SPOT_BASE
    )
    if matched_f:
        frozen_vs_volmatch = compare(frozen_eq_base, vh_frozen_eq)
        print("  frozen vs VOLMATCH_HOLD:", _fmt_cmp(frozen_vs_volmatch), f"(c={c_f:.4f}, vol={vol_f:.4f})")
    else:
        frozen_vs_volmatch = None
        print(f"  frozen vs VOLMATCH_HOLD: VOIDED (not matched, c={c_f:.4f}, vol={vol_f:.4f})")

    vh_cand_eq, c_c, vol_c, matched_c = volmatched_hold_equity(
        cand_eq_base, aligned_full6, UNIVERSE_6, SPOT_BASE
    )
    if matched_c:
        cand_vs_volmatch = compare(cand_eq_base, vh_cand_eq)
        print("  candidate vs VOLMATCH_HOLD:", _fmt_cmp(cand_vs_volmatch), f"(c={c_c:.4f}, vol={vol_c:.4f})")
    else:
        cand_vs_volmatch = None
        print(f"  candidate vs VOLMATCH_HOLD: VOIDED (not matched, c={c_c:.4f}, vol={vol_c:.4f})")

    idx_full6 = aligned_full6[UNIVERSE_6[0]].index
    mh_targets = matched_hold_targets(idx_full6, UNIVERSE_6, mean_total_notional(cand_full6))
    mh_eq = simulate_portfolio(mh_targets, aligned_full6, SPOT_BASE)
    cand_vs_matchedhold = compare(cand_eq_base, mh_eq)
    print("  candidate vs MATCHED_HOLD (own mean notional):", _fmt_cmp(cand_vs_matchedhold))

    static_eq = static_hold_equity(aligned_full6, UNIVERSE_6, SPOT_BASE)
    cand_vs_statichold = compare(cand_eq_base, static_eq)
    print("  candidate vs STATIC_HOLD (buy & hold, equal-weight U6):", _fmt_cmp(cand_vs_statichold))

    # ------------------------------------------------------------------- D3
    print("\n--- D3: W_VAL, U8, candidate vs FROZEN, directional only ---")
    aligned_val, frozen_val, cand_val = build_cell(
        frames, UNIVERSE_8, W_VAL, MODEL, PRIMARY_THRESH, PRIMARY_MAXD
    )
    net_cmp_val_base, _, _ = cell_cmp(frozen_val, cand_val, aligned_val, SPOT_BASE)
    net_cmp_val_real, _, _ = cell_cmp(frozen_val, cand_val, aligned_val, SPOT_REAL)
    gross_cmp_val, _, _ = cell_cmp(frozen_val, cand_val, aligned_val, SPOT_FREE)

    row_val_base = frontier_row(
        "r113_mahalanobis_vs_frozen", params, cand_val,
        net_cmp_val_base, gross_cmp_val, "FROZEN_XSMOM(undiscounted)",
        "W_VAL", "U8", fee_tier="SPOT_BASE",
    )
    row_val_real = frontier_row(
        "r113_mahalanobis_vs_frozen", params, cand_val,
        net_cmp_val_real, gross_cmp_val, "FROZEN_XSMOM(undiscounted)",
        "W_VAL", "U8", fee_tier="SPOT_REAL",
    )
    d3 = d3_pass(row_val_base)
    d3_real = d3_pass(row_val_real)

    print("  [SPOT_BASE] candidate vs frozen:", _fmt_cmp(net_cmp_val_base))
    print("  [SPOT_REAL] candidate vs frozen:", _fmt_cmp(net_cmp_val_real))
    _pt("D3 PASS (SPOT_BASE, primary)", d3)
    _pt("D3 PASS (SPOT_REAL, robustness only)", d3_real)

    # --------------------------------------------------------------- SCRAMBLE
    print("\n--- SCRAMBLE (falsification): candidate vs its own scrambled counterpart, W_FULL6, SPOT_BASE ---")
    cand_log_return = total_log_return(daily_returns(cand_eq_base).to_numpy(dtype=float))
    scramble_points = []
    for seed in SCRAMBLE_SEEDS:
        scrambled_targets = scramble_fixed_perm(cand_full6, seed)
        scrambled_eq = simulate_portfolio(scrambled_targets, aligned_full6, SPOT_BASE)
        scrambled_log_return = total_log_return(daily_returns(scrambled_eq).to_numpy(dtype=float))
        scramble_points.append(scrambled_log_return)
        _pt(f"scramble seed={seed} (scrambled candidate log-return)", f"{scrambled_log_return:.6f}")

    scramble_points = np.array(scramble_points, dtype=float)
    scramble_p90 = float(np.percentile(scramble_points, 90))
    scramble_survived = bool(cand_log_return > scramble_p90)
    _pt("candidate (real assignment) log-return", f"{cand_log_return:.6f}")
    _pt("scrambled 90th percentile", f"{scramble_p90:.6f}")
    _pt("SCRAMBLE SURVIVED (candidate beats scrambled p90)", scramble_survived)

    # ------------------------------------------------------------- FURTHER-WORK
    print("\n--- FURTHER-WORK BAR (r113_shared.further_work) ---")
    fw = r113_further_work(d1, d2, d3, d5, scramble_survived)
    _pt("(d1 or d2)", d1 or d2)
    _pt("d3", d3)
    _pt("d5", d5)
    _pt("scramble_survived", scramble_survived)
    _pt("FURTHER-WORK BAR CLEARED", fw)

    # ------------------------------------------------------------- HOLDOUT
    holdout_read = False
    net_cmp_hold_base = net_cmp_hold_real = gross_cmp_hold = None
    row_hold = None
    if fw:
        print("\n--- HOLDOUT: W_HOLD, U8, frozen 0.90/1.0 config (+1 holdout consultation) ---")
        holdout_read = True
        aligned_hold, frozen_hold, cand_hold = build_cell(
            frames, UNIVERSE_8, W_HOLD, MODEL, PRIMARY_THRESH, PRIMARY_MAXD
        )
        net_cmp_hold_base, _, _ = cell_cmp(frozen_hold, cand_hold, aligned_hold, SPOT_BASE)
        net_cmp_hold_real, _, _ = cell_cmp(frozen_hold, cand_hold, aligned_hold, SPOT_REAL)
        gross_cmp_hold, _, _ = cell_cmp(frozen_hold, cand_hold, aligned_hold, SPOT_FREE)

        row_hold = frontier_row(
            "r113_mahalanobis_vs_frozen", params, cand_hold,
            net_cmp_hold_base, gross_cmp_hold, "FROZEN_XSMOM(undiscounted)",
            "W_HOLD", "U8", fee_tier="SPOT_BASE",
        )
        print("  [SPOT_BASE] candidate vs frozen:", _fmt_cmp(net_cmp_hold_base))
        print("  [SPOT_REAL] candidate vs frozen:", _fmt_cmp(net_cmp_hold_real))
        print("  [SPOT_FREE, gross] candidate vs frozen:", _fmt_cmp(gross_cmp_hold))
        _pt("HOLDOUT D1-style pass (SPOT_BASE)", d1_pass(row_hold))
        _pt("HOLDOUT D2-style pass (SPOT_BASE)", d2_pass(row_hold))
        print("+1 holdout consultation.")
    else:
        print("\nholdout not read (further-work bar not cleared -- reading W_HOLD now would be")
        print("an undisclosed extra consultation).")

    # ------------------------------------------------------------- config count
    configs_after = config_count()
    trials_delta = configs_after - configs_before
    print(f"\n--- CONFIG COUNT: {configs_before} -> {configs_after} (delta = {trials_delta}) ---")

    # ------------------------------------------------------------- save CSVs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_base, row_real, row_val_base, row_val_real]
    if row_hold is not None:
        rows.append(row_hold)
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(rows)} rows to {RESULTS_CSV}")

    pd.DataFrame({"seed": list(SCRAMBLE_SEEDS), "scrambled_log_return": scramble_points}).to_csv(
        SCRAMBLE_CSV, index=False
    )
    print(f"Saved scramble results to {SCRAMBLE_CSV}")

    # ------------------------------------------------------------- write report
    write_report(
        causality_ok=causality_ok, step0=step0,
        primary_net_base=primary_net_base, primary_net_real=primary_net_real,
        primary_gross=primary_gross, d1=d1, d2=d2, d5=d5, d1_real=d1_real, d2_real=d2_real,
        frozen_vs_volmatch=frozen_vs_volmatch, cand_vs_volmatch=cand_vs_volmatch,
        cand_vs_matchedhold=cand_vs_matchedhold, cand_vs_statichold=cand_vs_statichold,
        net_cmp_val_base=net_cmp_val_base, net_cmp_val_real=net_cmp_val_real,
        gross_cmp_val=gross_cmp_val, d3=d3, d3_real=d3_real,
        cand_log_return=cand_log_return, scramble_p90=scramble_p90,
        scramble_survived=scramble_survived, scramble_points=scramble_points,
        fw=fw, holdout_read=holdout_read,
        net_cmp_hold_base=net_cmp_hold_base, net_cmp_hold_real=net_cmp_hold_real,
        gross_cmp_hold=gross_cmp_hold, trials_delta=trials_delta,
    )

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"causality_ok={causality_ok}  step0_passed={step0['passed']}")
    print(f"D1={d1}  D2={d2}  D3={d3}  D5={d5}  scramble_survived={scramble_survived}")
    print(f"FURTHER-WORK BAR: {fw}   holdout_read={holdout_read}")
    print(f"config_count delta (this branch): {trials_delta}")


def write_report(**kw):
    causality_ok = kw["causality_ok"]
    step0 = kw["step0"]
    d1, d2, d5 = kw["d1"], kw["d2"], kw["d5"]
    d1_real, d2_real = kw["d1_real"], kw["d2_real"]
    d3, d3_real = kw["d3"], kw["d3_real"]
    fw = kw["fw"]
    holdout_read = kw["holdout_read"]
    trials_delta = kw["trials_delta"]
    pn_base, pn_real, pg = kw["primary_net_base"], kw["primary_net_real"], kw["primary_gross"]
    nvb, nvr, gvb = kw["net_cmp_val_base"], kw["net_cmp_val_real"], kw["gross_cmp_val"]

    def cmp_row(label, c):
        if c is None:
            return f"| {label} | VOIDED | | | |"
        return (f"| {label} | {c['growth_diff']:.6f} | [{c['growth_lo']:.6f}, {c['growth_hi']:.6f}] "
                f"| {c['dd_diff']:.6f} | [{c['dd_lo']:.6f}, {c['dd_hi']:.6f}] |")

    verdict = "FURTHER-WORK CLEARED (holdout read authorized and consumed)" if fw else "NEGATIVE (further-work bar not cleared)"

    lines = []
    lines.append("# R-113 conservative branch: Mahalanobis panel-novelty brake")
    lines.append("")
    lines.append("## Pre-registration summary")
    lines.append("")
    lines.append(
        "This branch runs `model=\"mahalanobis\"` from `experiments/r113_shared.py` at the "
        "round's frozen, pre-registered operating point `PRIMARY_THRESH=0.90, "
        "PRIMARY_MAXD=1.0` -- the same cell both models were confirmed to pass Step-0 at "
        "before either branch was dispatched. The construction discounts "
        "`xsmom_entry_band`'s TOTAL notional (R-63's score, R-68's ENTRY_ONLY timing, "
        "R-107/R-110's k=1 equal-weight allocation, all frozen and unmodified) by a "
        "Mahalanobis-distance novelty statistic computed over three panel-level features "
        "(cross-sectional dispersion, mean pairwise correlation, eligible-count anomaly) "
        "reused verbatim from R-109. This round's own question is whether adding that "
        "brake helps the already-registered, undiscounted construction, so every D1-D5 "
        "gate below is keyed to **candidate (discounted) vs frozen (undiscounted)** on "
        "identical price paths, not to a passive benchmark; VOLMATCH_HOLD, MATCHED_HOLD "
        "and STATIC_HOLD are reported separately as secondary context only, matching "
        "R-63 onward's convention of reporting multiple comparator cells."
    )
    lines.append("")
    lines.append("## Step-0 gate (live reconfirmation, W_TRAIN, U8)")
    lines.append("")
    lines.append(f"- `check_causality` on `build_r113_targets(..., \"mahalanobis\", 0.90, 1.0)`: **{causality_ok}**")
    lines.append(f"- `bind_frac` = {step0['bind_frac']:.4f} (kill <= 0.01) -> {'ok' if step0['bind_ok'] else 'KILL'}")
    lines.append(f"- mean discount on bound bars = {step0['mean_bound_discount']:.4f} (kill < 0.05) -> "
                  f"{'ok' if step0['not_trivial_discount'] else 'KILL'}")
    lines.append(f"- R^2 vs basket realized vol = {step0['r2_vs_basketvol']:.4f} (kill >= 0.90) -> "
                  f"{'ok' if step0['not_vol_rescale'] else 'KILL (relabelled vol rescale)'}")
    lines.append(f"- state CoV = {step0['state_cv']:.4f} (kill < 0.05) -> "
                  f"{'ok' if step0['non_degenerate'] else 'KILL (degenerate)'}")
    lines.append(f"- **STEP-0 VERDICT: {'PASS' if step0['passed'] else 'FAIL'}** "
                  "(matches the shared module docstring's claim, reconfirmed live rather than trusted blindly)")
    lines.append("")
    lines.append("## Decisive battery: candidate vs FROZEN (primary comparator)")
    lines.append("")
    lines.append("### D1/D2 -- W_FULL6, U6")
    lines.append("")
    lines.append("| fee tier | growth_diff | 95% CI | dd_diff | 95% CI |")
    lines.append("|---|---|---|---|---|")
    lines.append(cmp_row("SPOT_BASE (0.10%, primary)", pn_base))
    lines.append(cmp_row("SPOT_REAL (0.40%, robustness)", pn_real))
    lines.append(cmp_row("SPOT_FREE (0.00%, gross, D5 input)", pg))
    lines.append("")
    lines.append(f"- D1 PASS (SPOT_BASE, primary): **{d1}**")
    lines.append(f"- D2 PASS (SPOT_BASE, primary): **{d2}**")
    lines.append(f"- D1 PASS (SPOT_REAL, robustness only): {d1_real}")
    lines.append(f"- D2 PASS (SPOT_REAL, robustness only): {d2_real}")
    lines.append(f"- D5 PASS (gross_growth_diff={pg['growth_diff']:.6f} >= D5_BAR_R68=+0.342): **{d5}**")
    lines.append("")
    lines.append(
        "> D5_BAR_R68 was calibrated in R-65/R-68 for a candidate-vs-VOLMATCH_HOLD gross "
        "comparison. Applied here to candidate-vs-FROZEN, it is a structurally harder bar: "
        "the brake can only ever reduce exposure relative to frozen, so gross_growth_diff "
        "here is bounded well below the scale D5_BAR_R68 was set against. Applied literally "
        "as instructed, not adjusted after seeing the number."
    )
    lines.append("")
    lines.append("### D3 -- W_VAL, U8 (directional only, no CI required)")
    lines.append("")
    lines.append("| fee tier | growth_diff | 95% CI | dd_diff | 95% CI |")
    lines.append("|---|---|---|---|---|")
    lines.append(cmp_row("SPOT_BASE (primary)", nvb))
    lines.append(cmp_row("SPOT_REAL (robustness)", nvr))
    lines.append(cmp_row("SPOT_FREE (gross)", gvb))
    lines.append("")
    lines.append(f"- D3 PASS (SPOT_BASE, primary): **{d3}**")
    lines.append(f"- D3 PASS (SPOT_REAL, robustness only): {d3_real}")
    lines.append("")
    lines.append("### Scramble control (falsification, W_FULL6, SPOT_BASE)")
    lines.append("")
    lines.append(f"- candidate (real asset assignment) log-return: {kw['cand_log_return']:.6f}")
    lines.append(f"- scrambled counterpart, 90th percentile across {len(kw['scramble_points'])} seeds: "
                  f"{kw['scramble_p90']:.6f}")
    lines.append(f"- **SCRAMBLE SURVIVED: {kw['scramble_survived']}**")
    lines.append("")
    lines.append("## Secondary context (SPOT_BASE, W_FULL6, U6) -- not scored")
    lines.append("")
    lines.append("| comparison | growth_diff | 95% CI | dd_diff | 95% CI |")
    lines.append("|---|---|---|---|---|")
    lines.append(cmp_row("frozen vs VOLMATCH_HOLD", kw["frozen_vs_volmatch"]))
    lines.append(cmp_row("candidate vs VOLMATCH_HOLD", kw["cand_vs_volmatch"]))
    lines.append(cmp_row("candidate vs MATCHED_HOLD (own mean notional)", kw["cand_vs_matchedhold"]))
    lines.append(cmp_row("candidate vs STATIC_HOLD (buy & hold)", kw["cand_vs_statichold"]))
    lines.append("")
    lines.append("## Further-work verdict")
    lines.append("")
    lines.append(f"`further_work = (D1 or D2) and D3 and D5 and scramble_survived` = **{fw}**")
    lines.append("")
    lines.append(f"Overall verdict for this branch: **{verdict}**")
    lines.append("")
    lines.append("## Holdout (W_HOLD, U8)")
    lines.append("")
    if holdout_read:
        nhb, nhr, ghc = kw["net_cmp_hold_base"], kw["net_cmp_hold_real"], kw["gross_cmp_hold"]
        lines.append(
            "Further-work bar cleared -> exactly ONE holdout read authorized and taken. "
            "**+1 holdout consultation** recorded. Reported honestly regardless of outcome."
        )
        lines.append("")
        lines.append("| fee tier | growth_diff | 95% CI | dd_diff | 95% CI |")
        lines.append("|---|---|---|---|---|")
        lines.append(cmp_row("SPOT_BASE", nhb))
        lines.append(cmp_row("SPOT_REAL", nhr))
        lines.append(cmp_row("SPOT_FREE (gross)", ghc))
    else:
        lines.append(
            "Further-work bar NOT cleared -> `W_HOLD` was **not read**. Reading it here would "
            "have been an undisclosed extra holdout consultation, which ROUTINE.md treats as a "
            "serious methodological violation. This is disclosed as a data point that was never "
            "collected, not as a negative holdout result."
        )
    lines.append("")
    lines.append("## Configs evaluated")
    lines.append("")
    lines.append(f"This branch alone ran **{trials_delta}** portfolio-backtest configurations "
                  "(each `simulate_portfolio` / `static_hold_equity` call), per `config_count()`. "
                  "The round's trials count for deflated Sharpe is this number summed with the "
                  "parallel kNN branch's own delta, per ROUTINE.md's parallelism rule.")
    lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved report to {REPORT_MD}")


if __name__ == "__main__":
    main()
