"""R-111 NOVEL -- Da/Gurun/Warachka (2014) "frog in the pan" path-consistency,
Kim (2025/26, SSRN #6889877)'s crypto extension, as a cross-sectional score
for R-63's own eight-asset panel.

See `r111_shared.py` for the full pre-registration (mechanism, literature,
the three named failure modes F1/F2/F3, the zero-new-fitted-parameter
threshold-transfer rule, the recentering fix, and the D1/D2/D3/D5/scramble
battery). This file does not redefine any of that machinery -- it only runs
it against `novel_score` and reports the numbers.

One sentence: `novel_score` multiplies R-63's own per-horizon magnitude term
(`close/anchor - 1`) by a causal [0,1] daily-sign-consistency factor, so a
smooth trend keeps its old-score value while a jumpy one is discounted
toward zero.

Pre-registered headline to watch for (F3): pooled Spearman rank correlation
against the OLD score on W_TRAIN already measured at 0.971, argmax
agreement 93% -- i.e. this score may simply be a noisy rescaling of the old
one. That is not treated as a bug; it is reported as the predicted outcome.

Run as:
    python experiments/r111_novel_path_consistency.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r111_shared import (  # noqa: E402
    DELTA_OUT_FIXED,
    D5_BAR_R68,
    HOLD_DAYS_FIXED,
    K_FIXED,
    NEIGHBOUR_MULTIPLIERS,
    OUT_DIR,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_FREE,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_HOLD,
    W_TRAIN,
    W_VAL,
    align_frames,
    build_targets_from_score,
    check_band_selection_matches_r68,
    check_causality,
    compare,
    config_count,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    frontier_row,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    novel_score,
    r63_cross_sectional_score,
    rank_agreement,
    scramble_fixed_perm,
    simulate_portfolio,
    transferred_thresholds,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)
from tradebot.inference import daily_returns, total_log_return  # noqa: E402

R68_REBALANCES_PER_DAY = 0.102  # R-68 conservative band-decomposition winner, W_TRAIN


# --------------------------------------------------------------- cell helpers


def align_and_score(frames, universe, window, center):
    """Align FRESH for (universe, window) (per the shared file's own
    warning -- `align_frames` depends on the window) and compute the
    RECENTERED `novel_score` once. Callers that need the same (universe,
    window) at several delta_in/buffer multipliers should call this once
    and reuse the result -- `novel_score` is the slow part, band selection
    is cheap."""
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    t0 = time.time()
    raw = novel_score(warm)
    dt = time.time() - t0
    centered = raw - center
    return warm, centered, dt


def build_cell_from_score(warm, centered, window, delta_in, buffer,
                          k=K_FIXED, hold_days=HOLD_DAYS_FIXED,
                          delta_out=DELTA_OUT_FIXED):
    targets, ev, ev_bars = build_targets_from_score(
        warm, centered, k, buffer, hold_days, delta_in, delta_out)
    universe = list(centered.columns)
    idx = warm[universe[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], ev


def volmatch_or_void(cand_eq, aligned_eval, universe, market, targets):
    """VOLMATCH_HOLD, plus the standing-rule fallback context number
    (MATCHED_HOLD at the candidate's own mean total notional, R-33's
    convention) reported whenever the match fails."""
    bench_eq, c, vol, matched = volmatched_hold_equity(cand_eq, aligned_eval, universe, market)
    fallback = None
    if not matched:
        idx = aligned_eval[universe[0]].index
        c_fb = mean_total_notional(targets)
        bench_fb_eq = simulate_portfolio(matched_hold_targets(idx, universe, c_fb), aligned_eval, market)
        fallback = {"c": c_fb, "cmp": compare(cand_eq, bench_fb_eq)}
    return {"bench_eq": bench_eq, "c": c, "vol": vol, "matched": matched, "fallback": fallback}


def print_cmp(label, cmp_):
    print(f"    {label}: growth_diff={cmp_['growth_diff']:+.4f} "
          f"[{cmp_['growth_lo']:+.4f}, {cmp_['growth_hi']:+.4f}]  "
          f"dd_diff={cmp_['dd_diff']:+.4f} [{cmp_['dd_lo']:+.4f}, {cmp_['dd_hi']:+.4f}]  "
          f"n_days={cmp_['n_days']}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_universe(UNIVERSE_8)
    start_configs = config_count()
    csv_rows = []

    # ===================================================================
    # STEP 1 -- freeze thresholds on W_TRAIN, U8
    # ===================================================================
    print("=" * 78)
    print("STEP 1 -- freeze thresholds on W_TRAIN, U8")
    print("=" * 78)
    aligned_train = align_frames(frames, warm_window(W_TRAIN))
    t0 = time.time()
    raw_score_train = novel_score(aligned_train)
    print(f"  novel_score(W_TRAIN, U8) computed in {time.time() - t0:.2f}s")
    th = transferred_thresholds(raw_score_train)
    center = th["center"]
    sigma = th["sigma"]
    delta_out = DELTA_OUT_FIXED
    k = K_FIXED
    hold_days = HOLD_DAYS_FIXED
    print(f"  center (pooled W_TRAIN mean)  = {center:+.6f}")
    print(f"  sigma  (pooled W_TRAIN std)   = {sigma:.6f}")
    for m in NEIGHBOUR_MULTIPLIERS:
        print(f"  {m}x: delta_in={th[m]['delta_in']:.6f}  buffer={th[m]['buffer']:.6f}")
    delta_in_1x = th[1.0]["delta_in"]
    buffer_1x = th[1.0]["buffer"]
    print(f"  DECISIVE (1.0x): delta_in={delta_in_1x:.6f}  buffer={buffer_1x:.6f}  "
          f"delta_out={delta_out}  k={k}  hold_days={hold_days}")

    # ===================================================================
    # STEP 2 -- sanity checks
    # ===================================================================
    print("\n" + "=" * 78)
    print("STEP 2 -- sanity checks")
    print("=" * 78)
    causal_ok = check_causality(novel_score, aligned_train, cut_from_end=20_000)
    print(f"  check_causality(novel_score, W_TRAIN/U8, cut_from_end=20000) = {causal_ok}")

    band_ok, band_err = check_band_selection_matches_r68(frames=frames)
    print(f"  check_band_selection_matches_r68() = {band_ok}  (max abs err = {band_err:.3e})")

    ra = rank_agreement(raw_score_train, r63_cross_sectional_score(aligned_train), window=W_TRAIN)
    print(f"  rank_agreement(novel_score, old score, W_TRAIN): "
          f"spearman_rho={ra['spearman_rho']:.4f}  "
          f"argmax_agreement={ra['argmax_agreement']:.4f}  n_bars={ra['n_bars']}")
    print("  (pre-measured reference: rho=0.971, argmax_agreement=0.93 -- F3 predicts near-duplication)")

    if not (causal_ok and band_ok):
        print("\n== SANITY CHECKS FAILED. r111_shared.py machinery is broken. STOP. ==")
        return

    # ===================================================================
    # STEP 3 -- decisive battery
    # ===================================================================
    print("\n" + "=" * 78)
    print("STEP 3 -- decisive battery")
    print("=" * 78)

    # ---- D1 / D2 cell: W_FULL6, U6, SPOT_BASE vs VOLMATCH_HOLD ----------
    print("\n-- D1/D2 cell: W_FULL6, U6, spot 0.10% vs VOLMATCH_HOLD (decisive 1.0x) --")
    warm_d1, centered_d1, dt_d1 = align_and_score(frames, UNIVERSE_6, W_FULL6, center)
    print(f"  novel_score(W_FULL6, U6) computed in {dt_d1:.2f}s")
    aligned_eval_d1, targets_d1, ev_d1 = build_cell_from_score(
        warm_d1, centered_d1, W_FULL6, delta_in_1x, buffer_1x)
    print(f"  event ledger: {ev_d1}")

    cand_eq_net = simulate_portfolio(targets_d1, aligned_eval_d1, SPOT_BASE)
    vm_net = volmatch_or_void(cand_eq_net, aligned_eval_d1, UNIVERSE_6, SPOT_BASE, targets_d1)
    print(f"  VOLMATCH_HOLD (net, SPOT_BASE): c={vm_net['c']:.4f}  vol={vm_net['vol']:.4f}  "
          f"matched={vm_net['matched']}")

    cand_eq_gross = simulate_portfolio(targets_d1, aligned_eval_d1, SPOT_FREE)
    vm_gross = volmatch_or_void(cand_eq_gross, aligned_eval_d1, UNIVERSE_6, SPOT_FREE, targets_d1)
    print(f"  VOLMATCH_HOLD (gross, SPOT_FREE): c={vm_gross['c']:.4f}  vol={vm_gross['vol']:.4f}  "
          f"matched={vm_gross['matched']}")

    d1_voided = not vm_net["matched"]
    d5_voided = not vm_gross["matched"]

    row_d1 = None
    net_cmp = None
    gross_cmp = None
    d1 = d2 = d5 = False
    if not d1_voided:
        net_cmp = compare(cand_eq_net, vm_net["bench_eq"])
        print_cmp("D1/D2 net (SPOT_BASE) vs VOLMATCH_HOLD", net_cmp)
    else:
        print("  ** D1/D2 CELL VOIDED (VOLMATCH_HOLD did not converge within tolerance) **")
        print_cmp("  fallback MATCHED_HOLD (net) context only", vm_net["fallback"]["cmp"])

    if not d5_voided:
        gross_cmp = compare(cand_eq_gross, vm_gross["bench_eq"])
        print_cmp("D5 gross (SPOT_FREE) vs VOLMATCH_HOLD", gross_cmp)
    else:
        print("  ** D5 CELL VOIDED (VOLMATCH_HOLD did not converge within tolerance, SPOT_FREE) **")
        print_cmp("  fallback MATCHED_HOLD (gross) context only", vm_gross["fallback"]["cmp"])

    if net_cmp is not None and gross_cmp is not None:
        row_d1 = frontier_row(
            "novel_path_consistency",
            {"delta_in": delta_in_1x, "buffer": buffer_1x, "multiplier": 1.0,
             "center": center, "sigma": sigma},
            targets_d1, net_cmp, gross_cmp, "VOLMATCH_HOLD", "W_FULL6", "U6")
        d1 = d1_pass(row_d1)
        d2 = d2_pass(row_d1)
        d5 = d5_pass(row_d1)
        csv_rows.append(row_d1)
        print(f"  d1_pass={d1}  d2_pass={d2}  d5_pass={d5} "
              f"(gross_growth_diff={row_d1['gross_growth_diff']:+.4f} vs bar {D5_BAR_R68:+.4f})")
    else:
        print("  d1/d2/d5 cannot be scored: cell(s) voided above.")

    # ---- D3 cell: W_VAL, U8, SPOT_BASE vs VOLMATCH_HOLD ------------------
    print("\n-- D3 cell: W_VAL, U8, spot 0.10% vs VOLMATCH_HOLD (directional only) --")
    warm_d3, centered_d3, dt_d3 = align_and_score(frames, UNIVERSE_8, W_VAL, center)
    print(f"  novel_score(W_VAL, U8) computed in {dt_d3:.2f}s")
    aligned_eval_d3, targets_d3, ev_d3 = build_cell_from_score(
        warm_d3, centered_d3, W_VAL, delta_in_1x, buffer_1x)
    print(f"  event ledger: {ev_d3}")

    cand_eq_d3 = simulate_portfolio(targets_d3, aligned_eval_d3, SPOT_BASE)
    vm_d3 = volmatch_or_void(cand_eq_d3, aligned_eval_d3, UNIVERSE_8, SPOT_BASE, targets_d3)
    print(f"  VOLMATCH_HOLD (D3): c={vm_d3['c']:.4f}  vol={vm_d3['vol']:.4f}  matched={vm_d3['matched']}")

    d3 = False
    row_d3 = None
    if vm_d3["matched"]:
        net_cmp_d3 = compare(cand_eq_d3, vm_d3["bench_eq"])
        print_cmp("D3 net vs VOLMATCH_HOLD", net_cmp_d3)
        # gross_cmp not decisionally required for D3; frontier_row needs one,
        # so pass net_cmp twice under an explicit gross-not-applicable note.
        row_d3 = frontier_row(
            "novel_path_consistency",
            {"delta_in": delta_in_1x, "buffer": buffer_1x, "multiplier": 1.0,
             "center": center, "sigma": sigma},
            targets_d3, net_cmp_d3, net_cmp_d3, "VOLMATCH_HOLD", "W_VAL", "U8",
            gross_not_applicable=True)
        d3 = d3_pass(row_d3)
        csv_rows.append(row_d3)
        print(f"  d3_pass={d3}")
    else:
        print("  ** D3 CELL VOIDED **")
        print_cmp("  fallback MATCHED_HOLD context only", vm_d3["fallback"]["cmp"])

    # ---- SCRAMBLE (falsification) ----------------------------------------
    print("\n-- SCRAMBLE falsification test (10 fixed seeds, on the D1 cell) --")
    scramble_points = []
    bench_eq_for_scramble = vm_net["bench_eq"]
    bench_ret = daily_returns(bench_eq_for_scramble).to_numpy(dtype=float)
    bench_stat = total_log_return(bench_ret)
    for seed in SCRAMBLE_SEEDS:
        scrambled_targets = scramble_fixed_perm(targets_d1, seed)
        scrambled_eq = simulate_portfolio(scrambled_targets, aligned_eval_d1, SPOT_BASE)
        s_ret = daily_returns(scrambled_eq).to_numpy(dtype=float)
        n = min(len(s_ret), len(bench_ret))
        point = float(total_log_return(s_ret[:n]) - total_log_return(bench_ret[:n]))
        scramble_points.append(point)
        print(f"  seed={seed}: scrambled growth_diff vs VOLMATCH_HOLD = {point:+.4f}")

    scramble_points = np.array(scramble_points)
    p90 = float(np.percentile(scramble_points, 90))
    cand_point = net_cmp["growth_diff"] if net_cmp is not None else float("nan")
    scramble_survived = bool(np.isfinite(cand_point) and cand_point > p90)
    print(f"  candidate D1 point estimate = {cand_point:+.4f}")
    print(f"  scrambled 90th percentile   = {p90:+.4f}")
    print(f"  scramble_survived = {scramble_survived}")

    # ---- NEIGHBOUR CHECK (plateau, not re-selection) ---------------------
    print("\n-- NEIGHBOUR CHECK: delta_in/buffer at 0.5x / 1.0x(decisive) / 1.5x, W_FULL6/U6 --")
    neighbour_results = {}
    for m in NEIGHBOUR_MULTIPLIERS:
        if m == 1.0:
            neighbour_results[m] = {
                "delta_in": delta_in_1x, "buffer": buffer_1x,
                "cmp": net_cmp if net_cmp is not None else vm_net["fallback"]["cmp"],
                "voided": d1_voided,
            }
            continue
        d_in_m = th[m]["delta_in"]
        buf_m = th[m]["buffer"]
        aligned_eval_m, targets_m, ev_m = build_cell_from_score(
            warm_d1, centered_d1, W_FULL6, d_in_m, buf_m)
        cand_eq_m = simulate_portfolio(targets_m, aligned_eval_m, SPOT_BASE)
        vm_m = volmatch_or_void(cand_eq_m, aligned_eval_m, UNIVERSE_6, SPOT_BASE, targets_m)
        if vm_m["matched"]:
            cmp_m = compare(cand_eq_m, vm_m["bench_eq"])
            voided_m = False
        else:
            cmp_m = vm_m["fallback"]["cmp"]
            voided_m = True
        neighbour_results[m] = {"delta_in": d_in_m, "buffer": buf_m, "cmp": cmp_m, "voided": voided_m}

    for m in NEIGHBOUR_MULTIPLIERS:
        r = neighbour_results[m]
        tag = " (DECISIVE)" if m == 1.0 else ""
        voided_tag = " [VOIDED, fallback shown]" if r["voided"] else ""
        print(f"  {m}x{tag}{voided_tag}: delta_in={r['delta_in']:.6f} buffer={r['buffer']:.6f}")
        print_cmp("    ", r["cmp"])

    # ---- FURTHER-WORK BAR --------------------------------------------------
    print("\n" + "=" * 78)
    print("FURTHER-WORK BAR")
    print("=" * 78)
    fw = bool((d1 or d2) and d3 and d5 and scramble_survived)
    print(f"  (d1={d1} or d2={d2}) and d3={d3} and d5={d5} and scramble_survived={scramble_survived}")
    print(f"  FURTHER-WORK BAR = {fw}")

    row_hold = None
    if fw:
        print("\n-- FURTHER-WORK BAR CLEARED: reading W_HOLD once (+1 holdout consultation) --")
        warm_hold, centered_hold, dt_hold = align_and_score(frames, UNIVERSE_8, W_HOLD, center)
        print(f"  novel_score(W_HOLD, U8) computed in {dt_hold:.2f}s")
        aligned_eval_hold, targets_hold, ev_hold = build_cell_from_score(
            warm_hold, centered_hold, W_HOLD, delta_in_1x, buffer_1x)
        cand_eq_hold = simulate_portfolio(targets_hold, aligned_eval_hold, SPOT_BASE)
        vm_hold = volmatch_or_void(cand_eq_hold, aligned_eval_hold, UNIVERSE_8, SPOT_BASE, targets_hold)
        print(f"  VOLMATCH_HOLD (W_HOLD): c={vm_hold['c']:.4f}  vol={vm_hold['vol']:.4f}  "
              f"matched={vm_hold['matched']}")
        if vm_hold["matched"]:
            hold_cmp = compare(cand_eq_hold, vm_hold["bench_eq"])
        else:
            hold_cmp = vm_hold["fallback"]["cmp"]
            print("  ** W_HOLD CELL VOIDED, fallback MATCHED_HOLD context shown **")
        print_cmp("W_HOLD net vs VOLMATCH_HOLD", hold_cmp)
        row_hold = frontier_row(
            "novel_path_consistency",
            {"delta_in": delta_in_1x, "buffer": buffer_1x, "multiplier": 1.0,
             "center": center, "sigma": sigma},
            targets_hold, hold_cmp, hold_cmp, "VOLMATCH_HOLD", "W_HOLD", "U8")
        csv_rows.append(row_hold)
        print("  +1 holdout consultation")
    else:
        print("  -> further-work bar NOT cleared. W_HOLD is NOT read. holdout not read.")

    # ===================================================================
    # STEP 4 -- config count
    # ===================================================================
    end_configs = config_count()
    trials_this_branch = end_configs - start_configs

    # ===================================================================
    # STEP 5 -- turnover
    # ===================================================================
    print("\n" + "=" * 78)
    print("STEP 5 -- turnover, decisive D1 cell")
    print("=" * 78)
    tstats = turnover_stats(targets_d1)
    print(f"  rebalances_per_day = {tstats['rebalances_per_day']:.4f}  "
          f"(R-68's own published: {R68_REBALANCES_PER_DAY:.3f}/day)")
    print(f"  turnover_per_day (notional) = {tstats['turnover_per_day']:.4f}")
    print(f"  implied_log_drag_total = {tstats['implied_log_drag_total']:.4f}")

    # ------------------------------------------------------------- CSV
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(OUT_DIR / "novel_path_consistency.csv", index=False)
        print(f"\n  wrote {OUT_DIR / 'novel_path_consistency.csv'}")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  config_count() delta this branch = {trials_this_branch}  "
          f"(cumulative process total = {end_configs})")
    print(f"  causality/correctness: check_causality={causal_ok}  "
          f"check_band_selection_matches_r68={band_ok} (err={band_err:.3e})")
    print(f"  F1 rank agreement (novel vs old, W_TRAIN): "
          f"spearman_rho={ra['spearman_rho']:.4f}  argmax_agreement={ra['argmax_agreement']:.4f}")
    print(f"  frozen config: center={center:+.6f} sigma={sigma:.6f} "
          f"delta_in(1.0x)={delta_in_1x:.6f} buffer(1.0x)={buffer_1x:.6f}")
    print(f"  d1={d1} d2={d2} d3={d3} d5={d5} scramble_survived={scramble_survived}")
    print(f"  FURTHER-WORK BAR = {fw}   holdout_read = {fw}")


if __name__ == "__main__":
    main()
