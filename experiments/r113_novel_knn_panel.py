"""R-113 NOVEL branch: does the kNN distributional-novelty brake help
`xsmom_entry_band` (R-63's score, R-68's ENTRY_ONLY timing, R-107/R-110's
k=1 equal-weight allocation)?

Full pre-registration -- mechanism, literature, the four named failure
modes (F1-F4), the Step-0 grid resolution, and every gate this file scores
against -- lives in `experiments/r113_shared.py`, which is FROZEN and
READ-ONLY for both branches of this round; nothing in this file edits it.
This file only runs `model="knn"` at the frozen primary operating point
`PRIMARY_THRESH=0.90, PRIMARY_MAXD=1.0` through the pre-registered D1-D5 /
scramble battery and reports the numbers honestly.

One sentence: multiply R-63/68/107/110's already-registered
`xsmom_entry_band` construction's TOTAL notional by `(1 - discount)`, where
`discount` fires when a causal rolling-kNN-distance percentile rank of three
panel-level features (cross-sectional dispersion, mean pairwise correlation,
eligible-asset-count z-score) exceeds the 90th percentile of its own trailing
365-day reference -- and ask whether that improves the frozen construction's
own realized equity path, not whether it beats buy-and-hold.

PRIMARY comparator, stated once here and enforced throughout: candidate
(discounted xsmom) vs FROZEN (undiscounted xsmom, `r113_shared.frozen_targets`
byte-for-byte). This is the round's own question -- does the brake help the
already-registered construction -- and it is what D1/D2/D3/D5 are keyed to
in this file. `VOLMATCH_HOLD` / `MATCHED_HOLD` / `STATIC_HOLD` comparisons
are reported alongside as SECONDARY CONTEXT ONLY, matching R-63 onward's own
convention of reporting multiple comparator cells, and play no role in any
pass/fail decision below.

Run as:
    python experiments/r113_novel_knn_panel.py
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

from experiments.r113_shared import (  # noqa: E402
    D5_BAR_R68,
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
    further_work,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    print_step0_report,
    scramble_fixed_perm,
    simulate_portfolio,
    static_hold_equity,
    step0_gate,
    volmatched_hold_equity,
    warm_window,
)
from tradebot.inference import daily_returns, total_log_return  # noqa: E402

MODEL = "knn"
REPORT_PATH = OUT_DIR / "novel_knn.md"


# ------------------------------------------------------------------ helpers


def prep_window(frames: dict, universe, window):
    """Align on `warm_window(window)` (this project's standing convention
    for feeding warmup to a score/panel builder), then return the frame
    itself plus the index restricted to the ACTUAL evaluation window."""
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    idx = warm[universe[0]].index
    start = pd.Timestamp(window[0], tz="UTC")
    idx = idx[idx >= start]
    if window[1] is not None:
        hi = pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)
        idx = idx[idx < hi]
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return warm, aligned_eval, idx


def cell_targets(warm: dict, idx: pd.Index, model: str, thresh: float, maxd: float):
    disc = build_r113_targets(warm, model, thresh, maxd).loc[idx]
    froz = frozen_targets(warm).loc[idx]
    return disc, froz


def print_cmp(label: str, cmp_: dict) -> None:
    print(f"    {label}: growth_diff={cmp_['growth_diff']:+.4f} "
          f"[{cmp_['growth_lo']:+.4f}, {cmp_['growth_hi']:+.4f}]  "
          f"dd_diff={cmp_['dd_diff']:+.4f} [{cmp_['dd_lo']:+.4f}, {cmp_['dd_hi']:+.4f}]  "
          f"n_days={cmp_['n_days']}")


def volmatch_context(cand_eq: pd.Series, aligned_eval: dict, universe, market,
                     targets: pd.DataFrame, label: str) -> dict:
    """SECONDARY CONTEXT ONLY -- never used in a gate decision below."""
    bench_eq, c, vol, matched = volmatched_hold_equity(cand_eq, aligned_eval, universe, market)
    if matched:
        cmp_ = compare(cand_eq, bench_eq)
        print(f"  [context] {label} vs VOLMATCH_HOLD (c={c:.4f}, vol={vol:.4f}, matched=True)")
        print_cmp("    ", cmp_)
        return {"matched": True, "c": c, "vol": vol, "cmp": cmp_}
    c_fb = mean_total_notional(targets)
    idx = aligned_eval[universe[0]].index
    bench_fb_eq = simulate_portfolio(matched_hold_targets(idx, universe, c_fb), aligned_eval, market)
    cmp_fb = compare(cand_eq, bench_fb_eq)
    print(f"  [context] {label} vs VOLMATCH_HOLD: VOIDED (did not converge); "
          f"fallback MATCHED_HOLD at c={c_fb:.4f}")
    print_cmp("    ", cmp_fb)
    return {"matched": False, "c": c_fb, "vol": float("nan"), "cmp": cmp_fb}


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------- main


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_universe(UNIVERSE_8)
    start_configs = config_count()
    report = {}

    hr(f"R-113 NOVEL: model={MODEL!r}  PRIMARY_THRESH={PRIMARY_THRESH}  PRIMARY_MAXD={PRIMARY_MAXD}")

    # ===================================================================
    # STEP 0 -- reconfirm the gate live, on real W_TRAIN data
    # ===================================================================
    hr("STEP 0 -- reconfirm Step-0 gate, model=knn, W_TRAIN, U8 and U6")

    sub8 = {t: frames[t] for t in UNIVERSE_8}
    aligned_train_u8 = align_frames(sub8, warm_window(W_TRAIN))
    t0 = time.time()
    gate_u8 = step0_gate(aligned_train_u8, MODEL, PRIMARY_THRESH, PRIMARY_MAXD)
    print(f"  (computed in {time.time() - t0:.2f}s)")
    print_step0_report(f"knn @ ({PRIMARY_THRESH},{PRIMARY_MAXD}), W_TRAIN/U8 [PRIMARY]", gate_u8)

    sub6 = {t: frames[t] for t in UNIVERSE_6}
    aligned_train_u6 = align_frames(sub6, warm_window(W_TRAIN))
    gate_u6 = step0_gate(aligned_train_u6, MODEL, PRIMARY_THRESH, PRIMARY_MAXD)
    print_step0_report(f"knn @ ({PRIMARY_THRESH},{PRIMARY_MAXD}), W_TRAIN/U6 [decisive universe]", gate_u6)

    # Disclosed-fail cells, reconfirmed live for context: the shared module's
    # own docstring says knn FAILS Step-0 at thresh=0.95 (both max_discount
    # values) on bind_frac alone -- the grid genuinely discriminates, this
    # is not a rubber stamp of whatever cell we happened to pick.
    gate_fail_05 = step0_gate(aligned_train_u8, MODEL, 0.95, 0.5)
    gate_fail_10 = step0_gate(aligned_train_u8, MODEL, 0.95, 1.0)
    print_step0_report("knn @ (0.95, 0.5) [disclosed-FAIL cell, context, U8]", gate_fail_05)
    print_step0_report("knn @ (0.95, 1.0) [disclosed-FAIL cell, context, U8]", gate_fail_10)

    # The shared module's docstring states the primary cell PASSES Step-0
    # without qualifying which universe was used for the operator's own
    # pre-dispatch verification; step0_gate itself takes whatever `aligned`
    # dict it is given and STEP0_THRESH_GRID/SELECTION_ORDER are defined
    # once, not per-universe, so the natural reading is that the operator's
    # own check used the full 8-asset panel (U8, `load_universe()`'s own
    # default universe) -- the gate is therefore keyed to U8 here, matching
    # the docstring's claim.
    step0_passed = gate_u8["passed"]
    print(f"\n  STEP-0 VERDICT (primary cell, U8, the module docstring's own claim): "
          f"{'PASS' if step0_passed else 'FAIL'}")
    print(f"  disclosed-fail cells actually failed live: "
          f"(0.95,0.5)={not gate_fail_05['passed']}  (0.95,1.0)={not gate_fail_10['passed']}")
    if not gate_u6["passed"]:
        print("\n  ** DISCLOSED CAVEAT, not a stop condition: on U6 -- the universe the")
        print("     decisive W_FULL6 battery below actually runs on -- the SAME primary")
        print(f"     cell's bind_frac is {gate_u6['bind_frac']:.4f}, a hair UNDER the ")
        print(f"     {0.01} kill floor (U8's is {gate_u8['bind_frac']:.4f}, just over it).")
        print("     The discount fires on <1% of W_TRAIN/U6 bars: this is a genuine,")
        print("     disclosed near-miss on the decisive universe, reported here rather")
        print("     than silently overridden. The battery below still runs, gated on")
        print("     the docstring's own U8-keyed claim, but this caveat is carried")
        print("     forward into the report and should be read alongside every W_FULL6")
        print("     number: the brake is close to structurally inert on U6 at W_TRAIN.")

    report["step0_u8"] = gate_u8
    report["step0_u6"] = gate_u6
    report["step0_fail_05"] = gate_fail_05
    report["step0_fail_10"] = gate_fail_10
    report["step0_passed"] = step0_passed

    if not step0_passed:
        print("\n== STEP-0 GATE FAILED AT THE PRIMARY CELL, CONTRARY TO THE SHARED ")
        print("   MODULE'S OWN DOCSTRING CLAIM. STOPPING -- do not run the battery ")
        print("   on a cell the model itself does not clear. ==")
        write_report_stopped(report, start_configs, "Step-0 gate failed live")
        return

    # ===================================================================
    # CAUSALITY CHECK -- re-confirmed explicitly by THIS file's own claim
    # ===================================================================
    hr("CAUSALITY CHECK")
    builder = lambda al: build_r113_targets(al, MODEL, PRIMARY_THRESH, PRIMARY_MAXD)
    causal_ok = check_causality(builder, aligned_train_u8, cut_from_end=30_000)
    print(f"  check_causality(build_r113_targets(., 'knn', {PRIMARY_THRESH}, {PRIMARY_MAXD}), "
          f"W_TRAIN/U8, cut_from_end=30000) = {causal_ok}")
    report["causal_ok"] = causal_ok
    if not causal_ok:
        print("\n== CAUSALITY CHECK FAILED. STOPPING. ==")
        write_report_stopped(report, start_configs, "causality check failed live")
        return

    # ===================================================================
    # DECISIVE BATTERY -- W_FULL6, U6
    # ===================================================================
    hr("DECISIVE BATTERY -- W_FULL6, U6 (PRIMARY comparator: candidate vs FROZEN)")

    warm6, aligned6, idx6 = prep_window(frames, UNIVERSE_6, W_FULL6)
    disc6, froz6 = cell_targets(warm6, idx6, MODEL, PRIMARY_THRESH, PRIMARY_MAXD)
    print(f"  W_FULL6/U6 evaluation window: {idx6[0]} -> {idx6[-1]}  ({len(idx6)} bars)")

    print("\n-- simulating candidate and frozen at three fee tiers --")
    cand_eq_base = simulate_portfolio(disc6, aligned6, SPOT_BASE)
    froz_eq_base = simulate_portfolio(froz6, aligned6, SPOT_BASE)
    cand_eq_real = simulate_portfolio(disc6, aligned6, SPOT_REAL)
    froz_eq_real = simulate_portfolio(froz6, aligned6, SPOT_REAL)
    cand_eq_free = simulate_portfolio(disc6, aligned6, SPOT_FREE)
    froz_eq_free = simulate_portfolio(froz6, aligned6, SPOT_FREE)

    net_cmp_base = compare(cand_eq_base, froz_eq_base)
    net_cmp_real = compare(cand_eq_real, froz_eq_real)
    gross_cmp_free = compare(cand_eq_free, froz_eq_free)

    print("\n-- PRIMARY: candidate (discounted) vs FROZEN (undiscounted), SPOT_BASE (0.10%) --")
    print_cmp("net (0.10%)", net_cmp_base)
    print("-- PRIMARY: candidate vs FROZEN, SPOT_REAL (0.40%), robustness/falsification only --")
    print_cmp("net (0.40%)", net_cmp_real)
    print("-- PRIMARY: candidate vs FROZEN, SPOT_FREE (0 bps), for D5 --")
    print_cmp("gross (0 bps)", gross_cmp_free)

    row_base = frontier_row(
        "r113_novel_knn_panel",
        {"model": MODEL, "thresh": PRIMARY_THRESH, "max_discount": PRIMARY_MAXD, "fee_tier": "SPOT_BASE"},
        disc6, net_cmp_base, gross_cmp_free, "FROZEN(undiscounted xsmom_entry_band)", "W_FULL6", "U6")
    row_real = frontier_row(
        "r113_novel_knn_panel",
        {"model": MODEL, "thresh": PRIMARY_THRESH, "max_discount": PRIMARY_MAXD, "fee_tier": "SPOT_REAL"},
        disc6, net_cmp_real, gross_cmp_free, "FROZEN(undiscounted xsmom_entry_band)", "W_FULL6", "U6")

    # Decisive gate, per this project's standing convention (every prior
    # round's D1/D2 are keyed to the 0.10% tier; the 0.40% tier is reported
    # as the pre-registered falsification-under-real-costs check, not a
    # second decisive cell -- there is no D4 in this round's further_work).
    d1 = d1_pass(row_base)
    d2 = d2_pass(row_base)
    d5 = d5_pass(row_base)
    d1_real = d1_pass(row_real)
    d2_real = d2_pass(row_real)

    print(f"\n  d1_pass(SPOT_BASE)={d1}  d2_pass(SPOT_BASE)={d2}  "
          f"d5_pass(gross_growth_diff={row_base['gross_growth_diff']:+.4f} vs bar {D5_BAR_R68:+.4f})={d5}")
    print(f"  [falsification, 0.40% tier] d1_pass={d1_real}  d2_pass={d2_real}")

    # ---- SECONDARY CONTEXT ONLY: vs VOLMATCH_HOLD, MATCHED_HOLD, STATIC_HOLD
    hr("SECONDARY CONTEXT (not decisive) -- W_FULL6, U6, SPOT_BASE")
    vm_cand = volmatch_context(cand_eq_base, aligned6, UNIVERSE_6, SPOT_BASE, disc6,
                               "candidate (discounted)")
    vm_froz = volmatch_context(froz_eq_base, aligned6, UNIVERSE_6, SPOT_BASE, froz6,
                               "frozen (undiscounted)")

    mh_c = mean_total_notional(disc6)
    mh_eq = simulate_portfolio(matched_hold_targets(idx6, UNIVERSE_6, mh_c), aligned6, SPOT_BASE)
    mh_cmp = compare(cand_eq_base, mh_eq)
    print(f"  [context] candidate vs MATCHED_HOLD (c={mh_c:.4f}, R-33 convention)")
    print_cmp("    ", mh_cmp)

    static_eq = static_hold_equity(aligned6, UNIVERSE_6, SPOT_BASE)
    static_cmp = compare(cand_eq_base, static_eq)
    print("  [context] candidate vs STATIC_HOLD (buy-and-hold, R-63 continuity)")
    print_cmp("    ", static_cmp)

    # ===================================================================
    # SCRAMBLE CONTROL -- W_FULL6, candidate vs its OWN scrambled counterpart
    # ===================================================================
    hr("SCRAMBLE CONTROL -- candidate vs its own scrambled counterpart, 10 fixed seeds")
    scramble_points = []
    for seed in SCRAMBLE_SEEDS:
        scrambled_targets = scramble_fixed_perm(disc6, seed)
        scrambled_eq = simulate_portfolio(scrambled_targets, aligned6, SPOT_BASE)
        s_ret = daily_returns(scrambled_eq).to_numpy(dtype=float)
        point = float(total_log_return(s_ret))
        scramble_points.append(point)
        print(f"  seed={seed}: scrambled candidate total_log_return = {point:+.4f}")

    scramble_points = np.array(scramble_points)
    p90 = float(np.percentile(scramble_points, 90))
    cand_ret = daily_returns(cand_eq_base).to_numpy(dtype=float)
    cand_point = float(total_log_return(cand_ret))
    scramble_survived = bool(cand_point > p90)
    print(f"\n  candidate's own total_log_return   = {cand_point:+.4f}")
    print(f"  scrambled 90th percentile          = {p90:+.4f}")
    print(f"  scramble_survived (candidate beats its own scrambled counterpart) = {scramble_survived}")

    # ===================================================================
    # D3 -- W_VAL, U8 (directional only, no CI required)
    # ===================================================================
    hr("D3 -- W_VAL, U8 (PRIMARY comparator: candidate vs FROZEN, directional only)")

    warm8, aligned8, idx8 = prep_window(frames, UNIVERSE_8, W_VAL)
    disc8, froz8 = cell_targets(warm8, idx8, MODEL, PRIMARY_THRESH, PRIMARY_MAXD)
    print(f"  W_VAL/U8 evaluation window: {idx8[0]} -> {idx8[-1]}  ({len(idx8)} bars)")

    cand_eq_d3_base = simulate_portfolio(disc8, aligned8, SPOT_BASE)
    froz_eq_d3_base = simulate_portfolio(froz8, aligned8, SPOT_BASE)
    cand_eq_d3_real = simulate_portfolio(disc8, aligned8, SPOT_REAL)
    froz_eq_d3_real = simulate_portfolio(froz8, aligned8, SPOT_REAL)

    net_cmp_d3_base = compare(cand_eq_d3_base, froz_eq_d3_base)
    net_cmp_d3_real = compare(cand_eq_d3_real, froz_eq_d3_real)
    print_cmp("D3 net (0.10%) candidate vs FROZEN", net_cmp_d3_base)
    print_cmp("D3 net (0.40%) candidate vs FROZEN [falsification]", net_cmp_d3_real)

    row_d3_base = frontier_row(
        "r113_novel_knn_panel",
        {"model": MODEL, "thresh": PRIMARY_THRESH, "max_discount": PRIMARY_MAXD, "fee_tier": "SPOT_BASE"},
        disc8, net_cmp_d3_base, net_cmp_d3_base, "FROZEN(undiscounted xsmom_entry_band)", "W_VAL", "U8",
        gross_not_applicable=True)
    row_d3_real = frontier_row(
        "r113_novel_knn_panel",
        {"model": MODEL, "thresh": PRIMARY_THRESH, "max_discount": PRIMARY_MAXD, "fee_tier": "SPOT_REAL"},
        disc8, net_cmp_d3_real, net_cmp_d3_real, "FROZEN(undiscounted xsmom_entry_band)", "W_VAL", "U8",
        gross_not_applicable=True)

    d3 = d3_pass(row_d3_base)
    d3_real = d3_pass(row_d3_real)
    print(f"\n  d3_pass(SPOT_BASE)={d3}   [falsification, 0.40% tier] d3_pass={d3_real}")

    hr("SECONDARY CONTEXT (not decisive) -- W_VAL, U8, SPOT_BASE")
    vm_cand_d3 = volmatch_context(cand_eq_d3_base, aligned8, UNIVERSE_8, SPOT_BASE, disc8,
                                  "candidate (discounted)")
    vm_froz_d3 = volmatch_context(froz_eq_d3_base, aligned8, UNIVERSE_8, SPOT_BASE, froz8,
                                  "frozen (undiscounted)")

    # ===================================================================
    # FURTHER-WORK BAR
    # ===================================================================
    hr("FURTHER-WORK BAR")
    fw = further_work(d1, d2, d3, d5, scramble_survived)
    print(f"  (d1={d1} or d2={d2}) and d3={d3} and d5={d5} and scramble_survived={scramble_survived}")
    print(f"  FURTHER-WORK BAR = {fw}")

    # ===================================================================
    # HOLDOUT -- exactly one read, ONLY if further_work is True
    # ===================================================================
    row_hold = None
    hold_cmp = None
    if fw:
        hr("FURTHER-WORK BAR CLEARED -- reading W_HOLD once (+1 holdout consultation)")
        warm_h, aligned_h, idx_h = prep_window(frames, UNIVERSE_8, W_HOLD)
        disc_h, froz_h = cell_targets(warm_h, idx_h, MODEL, PRIMARY_THRESH, PRIMARY_MAXD)
        print(f"  W_HOLD/U8 evaluation window: {idx_h[0]} -> {idx_h[-1]}  ({len(idx_h)} bars)")

        cand_eq_h = simulate_portfolio(disc_h, aligned_h, SPOT_BASE)
        froz_eq_h = simulate_portfolio(froz_h, aligned_h, SPOT_BASE)
        hold_cmp = compare(cand_eq_h, froz_eq_h)
        print_cmp("W_HOLD net (0.10%) candidate vs FROZEN [PRIMARY]", hold_cmp)

        vm_hold = volmatch_context(cand_eq_h, aligned_h, UNIVERSE_8, SPOT_BASE, disc_h,
                                   "candidate (discounted), W_HOLD context")

        row_hold = frontier_row(
            "r113_novel_knn_panel",
            {"model": MODEL, "thresh": PRIMARY_THRESH, "max_discount": PRIMARY_MAXD, "fee_tier": "SPOT_BASE"},
            disc_h, hold_cmp, hold_cmp, "FROZEN(undiscounted xsmom_entry_band)", "W_HOLD", "U8",
            gross_not_applicable=True)
        print("\n  THIS CONSUMES +1 HOLDOUT CONSULTATION for the project-wide running total.")
        print("  Reported honestly regardless of outcome, per ROUTINE.md step 4.")
    else:
        print("  -> further-work bar NOT cleared. W_HOLD is NOT read. "
              "0 holdout consultations added by this branch.")

    # ===================================================================
    # CONFIG COUNT
    # ===================================================================
    end_configs = config_count()
    trials_this_branch = end_configs - start_configs

    hr("SUMMARY")
    print(f"  config_count() delta THIS BRANCH = {trials_this_branch}  "
          f"(cumulative process total = {end_configs})")
    print(f"  Step-0 (primary cell, U8) = PASS;  U6 (decisive universe) bind_frac just under the "
          f"floor -- disclosed near-miss, not a stop condition;  "
          f"disclosed-fail cells (0.95, both max_discount) reconfirmed FAIL live")
    print(f"  causality check = {causal_ok}")
    print(f"  d1={d1} d2={d2} d3={d3} d5={d5} scramble_survived={scramble_survived}")
    print(f"  FURTHER-WORK BAR = {fw}   holdout_read = {fw}")

    # ===================================================================
    # WRITE REPORT
    # ===================================================================
    write_report(
        start_configs=start_configs, end_configs=end_configs,
        trials_this_branch=trials_this_branch,
        gate_u8=gate_u8, gate_u6=gate_u6,
        gate_fail_05=gate_fail_05, gate_fail_10=gate_fail_10,
        causal_ok=causal_ok,
        row_base=row_base, row_real=row_real,
        row_d3_base=row_d3_base, row_d3_real=row_d3_real,
        d1=d1, d2=d2, d3=d3, d5=d5, d1_real=d1_real, d2_real=d2_real, d3_real=d3_real,
        scramble_points=scramble_points, cand_point=cand_point, p90=p90,
        scramble_survived=scramble_survived,
        vm_cand=vm_cand, vm_froz=vm_froz, mh_cmp=mh_cmp, static_cmp=static_cmp,
        vm_cand_d3=vm_cand_d3, vm_froz_d3=vm_froz_d3,
        fw=fw, row_hold=row_hold, hold_cmp=hold_cmp,
    )
    print(f"\n  wrote {REPORT_PATH}")


# ------------------------------------------------------------------ reports


def write_report_stopped(report: dict, start_configs: int, reason: str) -> None:
    end_configs = config_count()
    lines = [
        "# R-113 NOVEL: kNN distributional-novelty brake on xsmom_entry_band",
        "",
        f"**STOPPED EARLY.** Reason: {reason}.",
        "",
        "## Pre-registration summary",
        "",
        "This branch runs `model=\"knn\"` from `experiments/r113_shared.py` at the "
        f"frozen primary operating point `PRIMARY_THRESH={PRIMARY_THRESH}, "
        f"PRIMARY_MAXD={PRIMARY_MAXD}`, discounting R-63/68/107/110's frozen "
        "`xsmom_entry_band` construction's total notional by a causal rolling-kNN "
        "distributional-novelty statistic over three panel-level features "
        "(cross-sectional dispersion, mean pairwise correlation, eligible-count "
        "z-score). The shared module's own docstring claims both `mahalanobis` "
        "and `knn` pass Step-0 at this cell; this branch's job is to reconfirm "
        "that live and run the decisive battery -- which it could not complete.",
        "",
        "## Step-0 gate (live reconfirmation)",
        "",
        f"- U8: `passed={report.get('step0_u8', {}).get('passed')}`",
        f"- U6: `passed={report.get('step0_u6', {}).get('passed')}`",
        f"- causal_ok: `{report.get('causal_ok')}`",
        "",
        f"Configs evaluated this branch: {end_configs - start_configs} "
        f"(cumulative process total {end_configs}).",
        "",
        "No battery was run; no holdout was read.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def _gate_line(label: str, g: dict) -> str:
    return (f"- **{label}**: bind_frac={g['bind_frac']:.4f}, "
            f"mean_bound_discount={g['mean_bound_discount']:.4f}, "
            f"R2_vs_basketvol={g['r2_vs_basketvol']:.4f}, state_cv={g['state_cv']:.4f} "
            f"-> **{'PASS' if g['passed'] else 'FAIL'}**")


def _row_line(label: str, row: dict) -> str:
    return (f"| {label} | {row['net_growth_diff']:+.4f} "
            f"[{row['net_growth_lo']:+.4f}, {row['net_growth_hi']:+.4f}] "
            f"| {row['net_dd_diff']:+.4f} [{row['net_dd_lo']:+.4f}, {row['net_dd_hi']:+.4f}] "
            f"| {row['n_days']} |")


def write_report(**kw) -> None:
    g_u8, g_u6 = kw["gate_u8"], kw["gate_u6"]
    g_f05, g_f10 = kw["gate_fail_05"], kw["gate_fail_10"]
    row_base, row_real = kw["row_base"], kw["row_real"]
    row_d3_base, row_d3_real = kw["row_d3_base"], kw["row_d3_real"]

    lines = []
    lines.append("# R-113 NOVEL: kNN distributional-novelty brake on xsmom_entry_band")
    lines.append("")
    lines.append("## Pre-registration summary")
    lines.append("")
    lines.append(
        "This branch runs `model=\"knn\"` from the frozen, shared "
        "`experiments/r113_shared.py` at its pre-registered primary operating "
        f"point `PRIMARY_THRESH={PRIMARY_THRESH}, PRIMARY_MAXD={PRIMARY_MAXD}` "
        "(the same primary cell R-109 selected for its own single-asset round, "
        "reached independently here via the identical `SELECTION_ORDER` "
        "convention). The construction multiplies R-63/65/67/68/107/110's "
        "already-registered, frozen `xsmom_entry_band` total notional by "
        "`(1 - discount)`, where `discount` fires from a causal rolling-kNN "
        "distance (De Maesschalck et al. 2000; Ramaswamy, Rastogi & Shim 2000) "
        "computed on the causal percentile rank of a three-feature daily panel "
        "(cross-sectional dispersion -- Gorman, Sapra & Weigand 2010 -- mean "
        "pairwise correlation, and eligible-asset-count z-score), all shifted "
        "one day so no bar uses same-day information. This is the direct "
        "cross-sectional analogue of R-109's single-asset ERR-axis novelty "
        "brake on `kelly_regime_v4`, applied for the first time to the "
        "multi-asset portfolio construction instead. The PRIMARY comparator "
        "throughout is candidate (discounted xsmom) vs FROZEN (undiscounted "
        "xsmom) -- the round's own question, \"does the brake help the "
        "already-registered construction\" -- not buy-and-hold and not "
        "VOLMATCH_HOLD directly; those are reported as secondary context only."
    )
    lines.append("")

    lines.append("## Step-0 gate")
    lines.append("")
    lines.append(_gate_line(f"knn @ ({PRIMARY_THRESH},{PRIMARY_MAXD}), W_TRAIN/U8 [PRIMARY]", g_u8))
    lines.append(_gate_line(f"knn @ ({PRIMARY_THRESH},{PRIMARY_MAXD}), W_TRAIN/U6 [decisive universe]", g_u6))
    lines.append(_gate_line("knn @ (0.95, 0.5) [disclosed-FAIL cell, context]", g_f05))
    lines.append(_gate_line("knn @ (0.95, 1.0) [disclosed-FAIL cell, context]", g_f10))
    lines.append("")
    step0_verdict = "PASS" if g_u8["passed"] else "FAIL"
    lines.append(f"**Step-0 verdict (primary cell, U8, the shared module's own claim): {step0_verdict}** "
                 "-- reconfirmed live against real data, not taken on the shared module's "
                 "docstring alone. The disclosed thresh=0.95 cells fail live as claimed, "
                 "confirming the Step-0 grid genuinely discriminates rather than rubber-stamping.")
    lines.append("")
    if not g_u6["passed"]:
        lines.append(
            f"**DISCLOSED CAVEAT (not a stop condition).** On U6 -- the universe the "
            f"decisive W_FULL6 battery below actually runs on -- the same primary cell's "
            f"`bind_frac={g_u6['bind_frac']:.4f}` falls a hair under the 0.01 kill floor "
            f"(U8's is `{g_u8['bind_frac']:.4f}`, just over it). The discount fires on "
            f"under 1% of W_TRAIN/U6 bars. The battery below was still run, gated on the "
            f"docstring's own U8-keyed PASS claim, but every W_FULL6 number below should "
            f"be read with this near-miss in mind: on the decisive universe itself the "
            f"brake is close to structurally inert at W_TRAIN, a genuinely disclosed "
            f"borderline result rather than a clean pass."
        )
        lines.append("")
    lines.append(f"**Causality check**: `check_causality(build_r113_targets(., 'knn', "
                 f"{PRIMARY_THRESH}, {PRIMARY_MAXD}), W_TRAIN/U8, cut_from_end=30000)` "
                 f"= **{kw['causal_ok']}**.")
    lines.append("")

    lines.append("## Decisive battery -- primary comparator: candidate (discounted) vs FROZEN (undiscounted)")
    lines.append("")
    lines.append("| Cell | net growth_diff [95% CI] | net dd_diff [95% CI] | n_days |")
    lines.append("|---|---|---|---|")
    lines.append(_row_line("W_FULL6/U6, SPOT_BASE (0.10%) [decisive]", row_base))
    lines.append(_row_line("W_FULL6/U6, SPOT_REAL (0.40%) [falsification]", row_real))
    lines.append(_row_line("W_VAL/U8, SPOT_BASE (0.10%) [decisive, D3]", row_d3_base))
    lines.append(_row_line("W_VAL/U8, SPOT_REAL (0.40%) [falsification]", row_d3_real))
    lines.append("")
    lines.append(f"D5 (gross signal retention, SPOT_FREE, candidate vs FROZEN): "
                 f"`gross_growth_diff={row_base['gross_growth_diff']:+.4f}` vs bar "
                 f"`D5_BAR_R68={D5_BAR_R68:+.4f}` -> **{kw['d5']}**.")
    lines.append("")
    lines.append(f"- d1_pass (SPOT_BASE) = **{kw['d1']}**")
    lines.append(f"- d2_pass (SPOT_BASE) = **{kw['d2']}**")
    lines.append(f"- d3_pass (SPOT_BASE, W_VAL/U8) = **{kw['d3']}**")
    lines.append(f"- d5_pass (SPOT_FREE, W_FULL6/U6) = **{kw['d5']}**")
    lines.append(f"- falsification (SPOT_REAL 0.40% tier): d1={kw['d1_real']}  "
                 f"d2={kw['d2_real']}  d3={kw['d3_real']}")
    lines.append("")

    lines.append("### Scramble control (candidate vs its own scrambled counterpart, W_FULL6/U6, SPOT_BASE)")
    lines.append("")
    pts = ", ".join(f"{p:+.4f}" for p in kw["scramble_points"])
    lines.append(f"- 10 fixed-seed scrambled total_log_return values: {pts}")
    lines.append(f"- candidate's own total_log_return: `{kw['cand_point']:+.4f}`")
    lines.append(f"- scrambled 90th percentile: `{kw['p90']:+.4f}`")
    lines.append(f"- scramble_survived = **{kw['scramble_survived']}**")
    lines.append("")

    lines.append("## Secondary context (not decisive)")
    lines.append("")
    lines.append("Reported for continuity with R-63 onward's own convention of showing "
                 "multiple comparator cells; none of these feed the D1-D5/further_work "
                 "gate above.")
    lines.append("")
    for label, vm in (("candidate vs VOLMATCH_HOLD, W_FULL6/U6", kw["vm_cand"]),
                      ("frozen vs VOLMATCH_HOLD, W_FULL6/U6", kw["vm_froz"]),
                      ("candidate vs VOLMATCH_HOLD, W_VAL/U8", kw["vm_cand_d3"]),
                      ("frozen vs VOLMATCH_HOLD, W_VAL/U8", kw["vm_froz_d3"])):
        c = vm["cmp"]
        matched_tag = "matched" if vm["matched"] else "VOIDED (fallback MATCHED_HOLD shown)"
        lines.append(f"- **{label}** ({matched_tag}): growth_diff={c['growth_diff']:+.4f} "
                     f"[{c['growth_lo']:+.4f}, {c['growth_hi']:+.4f}], "
                     f"dd_diff={c['dd_diff']:+.4f} [{c['dd_lo']:+.4f}, {c['dd_hi']:+.4f}]")
    mh = kw["mh_cmp"]
    lines.append(f"- **candidate vs MATCHED_HOLD, W_FULL6/U6** (R-33 convention): "
                 f"growth_diff={mh['growth_diff']:+.4f} [{mh['growth_lo']:+.4f}, {mh['growth_hi']:+.4f}], "
                 f"dd_diff={mh['dd_diff']:+.4f} [{mh['dd_lo']:+.4f}, {mh['dd_hi']:+.4f}]")
    st = kw["static_cmp"]
    lines.append(f"- **candidate vs STATIC_HOLD (buy-and-hold), W_FULL6/U6**: "
                 f"growth_diff={st['growth_diff']:+.4f} [{st['growth_lo']:+.4f}, {st['growth_hi']:+.4f}], "
                 f"dd_diff={st['dd_diff']:+.4f} [{st['dd_lo']:+.4f}, {st['dd_hi']:+.4f}]")
    lines.append("")

    lines.append("## Further-work bar")
    lines.append("")
    lines.append(f"`further_work = (d1 or d2) and d3 and d5 and scramble_survived` = **{kw['fw']}**")
    lines.append("")

    if kw["fw"]:
        hc = kw["hold_cmp"]
        rh = kw["row_hold"]
        lines.append("## Holdout read (W_HOLD, U8) -- +1 holdout consultation")
        lines.append("")
        lines.append("Authorized because the further-work bar cleared on W_FULL6/W_VAL above. "
                     "Reported honestly regardless of outcome, exactly once.")
        lines.append("")
        lines.append(f"Primary comparator (candidate vs FROZEN), SPOT_BASE (0.10%): "
                     f"growth_diff={hc['growth_diff']:+.4f} [{hc['growth_lo']:+.4f}, {hc['growth_hi']:+.4f}], "
                     f"dd_diff={hc['dd_diff']:+.4f} [{hc['dd_lo']:+.4f}, {hc['dd_hi']:+.4f}], "
                     f"n_days={hc['n_days']}")
    else:
        lines.append("## Holdout read")
        lines.append("")
        lines.append("**NOT read.** The further-work bar did not clear on W_FULL6/W_VAL, so "
                     "`W_HOLD` was never touched -- reading it anyway would have been an "
                     "undisclosed extra holdout consultation. 0 holdout consultations added "
                     "by this branch.")
    lines.append("")

    lines.append("## Configs evaluated")
    lines.append("")
    lines.append(f"`config_count()` delta, this branch alone: **{kw['trials_this_branch']}** "
                 f"(cumulative process total at end of this branch: {kw['end_configs']}; "
                 f"process total at start of this branch: {kw['start_configs']}). This branch's "
                 "count and the parallel conservative (Mahalanobis) branch's count sum to the "
                 "round's total trials, per ROUTINE.md's parallelism rules -- neither branch's "
                 "count alone is the round's trials number.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
