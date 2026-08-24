"""R-111 CONSERVATIVE BRANCH: George & Hwang (2004) 52-week-high proximity as
the cross-sectional score, run through the shared R-111 battery.

This file is the executable companion to the pre-registration and the two
score formulas committed in `experiments/r111_shared.py` (read that file's
docstring first -- it is this round's decision rules, failure modes and
transfer rule, none of which are restated here). This file does exactly one
thing: run `conservative_score` through the frozen D1/D2/D3/D5/SCRAMBLE/
NEIGHBOUR battery and print/save the real numbers. It does not edit
`r111_shared.py` and does not touch anything with "novel" in its name.

Run: `python experiments/r111_conservative_52w_high.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r111_shared import (  # noqa: E402
    DELTA_OUT_FIXED,
    HOLD_DAYS_FIXED,
    K_FIXED,
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
    conservative_score,
    config_count,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    frontier_row,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
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

RESULTS_CSV = OUT_DIR / "conservative_52w_high_results.csv"


def _pt(label, val):
    print(f"  {label:40s} {val}")


def build_cell(frames, universe, window, delta_in, buffer, center, k, hold_days, delta_out):
    """Fresh-aligned cell builder, per the pre-registered protocol. Aligns
    for THIS specific window (never reuses W_TRAIN's alignment), builds the
    score, subtracts the FROZEN center (never re-measured per-window), then
    builds targets and slices everything to the evaluation window."""
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    raw = conservative_score(warm)
    centered = raw - center
    targets, ev, ev_bars = build_targets_from_score(
        warm, centered, k, buffer, hold_days, delta_in, delta_out
    )
    idx = warm[universe[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], ev


def main():
    results_rows = []

    print("=" * 78)
    print("R-111 CONSERVATIVE BRANCH: 52-week-high proximity cross-sectional score")
    print("=" * 78)

    # ----------------------------------------------------------- Step 1
    print("\n--- STEP 1: freeze thresholds on W_TRAIN, U8 ---")
    frames = load_universe(UNIVERSE_8)
    aligned_train = align_frames(frames, warm_window(W_TRAIN))
    raw_score_train = conservative_score(aligned_train)
    th = transferred_thresholds(raw_score_train)
    center = th["center"]
    sigma = th["sigma"]
    delta_out = DELTA_OUT_FIXED
    k = K_FIXED
    hold_days = HOLD_DAYS_FIXED

    _pt("center (pooled W_TRAIN mean)", f"{center:.6f}")
    _pt("sigma (pooled W_TRAIN std)", f"{sigma:.6f}")
    for m in (0.5, 1.0, 1.5):
        _pt(f"delta_in @ {m}x", f"{th[m]['delta_in']:.6f}")
        _pt(f"buffer   @ {m}x", f"{th[m]['buffer']:.6f}")
    _pt("delta_out (fixed)", delta_out)
    _pt("k (fixed)", k)
    _pt("hold_days (fixed)", hold_days)

    # ----------------------------------------------------------- Step 2
    print("\n--- STEP 2: sanity checks ---")
    configs_before = config_count()

    causality_ok = check_causality(conservative_score, aligned_train, cut_from_end=20_000)
    _pt("check_causality", causality_ok)

    band_ok, band_err = check_band_selection_matches_r68()
    _pt("check_band_selection_matches_r68", f"{band_ok} (err={band_err:.3e})")

    ra = rank_agreement(raw_score_train, r63_cross_sectional_score(aligned_train))
    _pt("F1 rank_agreement (this run)", ra)
    _pt("F1 reference (pre-measured by operator)",
        "spearman=0.588, argmax_agreement=0.44")

    if not causality_ok:
        print("\n!!! CAUSALITY CHECK FAILED -- ABORTING BEFORE ANY DECISIVE NUMBER !!!")
        return
    if not band_ok:
        print("\n!!! BAND-SELECTION CORRECTNESS CHECK FAILED -- ABORTING !!!")
        return

    # ----------------------------------------------------------- Step 3: D1/D2/D5
    print("\n--- STEP 3: D1/D2/D5 -- W_FULL6, U6, decisive (1.0x) ---")
    delta_in_1 = th[1.0]["delta_in"]
    buffer_1 = th[1.0]["buffer"]

    aligned_eval, targets_d1, ev_d1 = build_cell(
        frames, UNIVERSE_6, W_FULL6, delta_in_1, buffer_1, center, k, hold_days, delta_out
    )
    print("  band_selection event ledger (D1 cell):", ev_d1)

    # --- net (SPOT_BASE) ---
    cand_eq_net = simulate_portfolio(targets_d1, aligned_eval, SPOT_BASE)
    bench_eq_net, c_net, vol_net, matched_net = volmatched_hold_equity(
        cand_eq_net, aligned_eval, UNIVERSE_6, SPOT_BASE
    )
    _pt("VOLMATCH_HOLD matched (net, SPOT_BASE)", f"{matched_net} (c={c_net:.4f}, vol={vol_net:.4f})")

    d1_cell_voided = not matched_net
    matched_hold_context_net = None
    if d1_cell_voided:
        mean_notional = mean_total_notional(targets_d1)
        mh_idx = aligned_eval[UNIVERSE_6[0]].index
        mh_targets = matched_hold_targets(mh_idx, UNIVERSE_6, mean_notional)
        mh_eq = simulate_portfolio(mh_targets, aligned_eval, SPOT_BASE)
        matched_hold_context_net = compare(cand_eq_net, mh_eq)
        print("  !!! D1/D2 CELL VOIDED: volmatched_hold_equity did not match within tolerance !!!")
        print("  MATCHED_HOLD fallback context (net):", matched_hold_context_net)
        net_cmp = compare(cand_eq_net, bench_eq_net)  # still computed, reported as context only
    else:
        net_cmp = compare(cand_eq_net, bench_eq_net)

    # --- gross (SPOT_FREE), D5 ---
    cand_eq_gross = simulate_portfolio(targets_d1, aligned_eval, SPOT_FREE)
    bench_eq_gross, c_gross, vol_gross, matched_gross = volmatched_hold_equity(
        cand_eq_gross, aligned_eval, UNIVERSE_6, SPOT_FREE
    )
    _pt("VOLMATCH_HOLD matched (gross, SPOT_FREE)",
        f"{matched_gross} (c={c_gross:.4f}, vol={vol_gross:.4f})")
    gross_cmp = compare(cand_eq_gross, bench_eq_gross)

    row_d1 = frontier_row(
        "conservative_52w_high",
        {"delta_in": delta_in_1, "buffer": buffer_1, "k": k, "hold_days": hold_days,
         "delta_out": delta_out, "multiplier": 1.0, "center": center, "sigma": sigma},
        targets_d1, net_cmp, gross_cmp, "VOLMATCH_HOLD", "W_FULL6", "U6",
        matched_net=matched_net, matched_gross=matched_gross,
    )
    results_rows.append(row_d1)

    d1 = d1_pass(row_d1) if not d1_cell_voided else False
    d2 = d2_pass(row_d1) if not d1_cell_voided else False
    d5 = d5_pass(row_d1)

    print("\n  D1 cell result (net_cmp, vs VOLMATCH_HOLD, SPOT_BASE):")
    for kk, vv in net_cmp.items():
        _pt(kk, vv)
    print("  D5 cell result (gross_cmp, vs VOLMATCH_HOLD, SPOT_FREE):")
    for kk, vv in gross_cmp.items():
        _pt(kk, vv)

    _pt("D1 PASS" + (" (VOIDED)" if d1_cell_voided else ""), d1)
    _pt("D2 PASS" + (" (VOIDED)" if d1_cell_voided else ""), d2)
    _pt("D5 PASS (gross_growth_diff >= D5_BAR_R68)", d5)

    # ----------------------------------------------------------- D3
    print("\n--- D3: W_VAL, U8, directional only ---")
    aligned_val, targets_d3, ev_d3 = build_cell(
        frames, UNIVERSE_8, W_VAL, delta_in_1, buffer_1, center, k, hold_days, delta_out
    )
    print("  band_selection event ledger (D3 cell):", ev_d3)

    cand_eq_d3 = simulate_portfolio(targets_d3, aligned_val, SPOT_BASE)
    bench_eq_d3, c_d3, vol_d3, matched_d3 = volmatched_hold_equity(
        cand_eq_d3, aligned_val, UNIVERSE_8, SPOT_BASE
    )
    _pt("VOLMATCH_HOLD matched (D3)", f"{matched_d3} (c={c_d3:.4f}, vol={vol_d3:.4f})")

    net_cmp_d3 = compare(cand_eq_d3, bench_eq_d3)
    cand_eq_d3_gross = simulate_portfolio(targets_d3, aligned_val, SPOT_FREE)
    bench_eq_d3_gross, _, _, matched_d3_gross = volmatched_hold_equity(
        cand_eq_d3_gross, aligned_val, UNIVERSE_8, SPOT_FREE
    )
    gross_cmp_d3 = compare(cand_eq_d3_gross, bench_eq_d3_gross)

    row_d3 = frontier_row(
        "conservative_52w_high",
        {"delta_in": delta_in_1, "buffer": buffer_1, "k": k, "hold_days": hold_days,
         "delta_out": delta_out, "multiplier": 1.0, "center": center, "sigma": sigma},
        targets_d3, net_cmp_d3, gross_cmp_d3, "VOLMATCH_HOLD", "W_VAL", "U8",
        matched_net=matched_d3, matched_gross=matched_d3_gross,
    )
    results_rows.append(row_d3)
    d3 = d3_pass(row_d3)

    print("  D3 cell result (net_cmp):")
    for kk, vv in net_cmp_d3.items():
        _pt(kk, vv)
    _pt("D3 PASS", d3)

    # ----------------------------------------------------------- SCRAMBLE
    print("\n--- SCRAMBLE (falsification, on the D1 cell's own targets) ---")
    d1_point = net_cmp["growth_diff"]
    scramble_points = []
    for seed in SCRAMBLE_SEEDS:
        scrambled_targets = scramble_fixed_perm(targets_d1, seed)
        scrambled_eq = simulate_portfolio(scrambled_targets, aligned_eval, SPOT_BASE)
        s_ret = daily_returns(scrambled_eq).to_numpy(dtype=float)
        b_ret = daily_returns(bench_eq_net).to_numpy(dtype=float)
        n = min(len(s_ret), len(b_ret))
        point = total_log_return(s_ret[:n]) - total_log_return(b_ret[:n])
        scramble_points.append(point)
        _pt(f"scramble seed={seed}", f"{point:.6f}")

    scramble_points = np.array(scramble_points, dtype=float)
    scramble_p90 = float(np.percentile(scramble_points, 90))
    scramble_survived = bool(d1_point > scramble_p90)
    _pt("D1 candidate point estimate", f"{d1_point:.6f}")
    _pt("scramble 90th percentile", f"{scramble_p90:.6f}")
    _pt("SCRAMBLE SURVIVED", scramble_survived)

    # ----------------------------------------------------------- NEIGHBOUR CHECK
    print("\n--- NEIGHBOUR CHECK (plateau, not a re-selection) ---")
    neighbour_rows = []
    for m in (0.5, 1.0, 1.5):
        d_in = th[m]["delta_in"]
        buf = th[m]["buffer"]
        aligned_m, targets_m, ev_m = build_cell(
            frames, UNIVERSE_6, W_FULL6, d_in, buf, center, k, hold_days, delta_out
        )
        cand_eq_m = simulate_portfolio(targets_m, aligned_m, SPOT_BASE)
        bench_eq_m, c_m, vol_m, matched_m = volmatched_hold_equity(
            cand_eq_m, aligned_m, UNIVERSE_6, SPOT_BASE
        )
        if matched_m:
            cmp_m = compare(cand_eq_m, bench_eq_m)
            point_m = cmp_m["growth_diff"]
            lo_m, hi_m = cmp_m["growth_lo"], cmp_m["growth_hi"]
        else:
            point_m = float("nan")
            lo_m = hi_m = float("nan")
        neighbour_rows.append({
            "multiplier": m, "delta_in": d_in, "buffer": buf,
            "matched": matched_m, "growth_diff": point_m,
            "growth_lo": lo_m, "growth_hi": hi_m,
        })
        _pt(f"neighbour {m}x: delta_in={d_in:.4f} buffer={buf:.4f} matched={matched_m}",
            f"growth_diff={point_m:.6f} [{lo_m:.6f}, {hi_m:.6f}]" if matched_m else "VOIDED")

    # ----------------------------------------------------------- FURTHER-WORK BAR
    print("\n--- FURTHER-WORK BAR ---")
    further_work_bar = (d1 or d2) and d3 and d5 and scramble_survived
    _pt("(d1 or d2)", d1 or d2)
    _pt("d3", d3)
    _pt("d5", d5)
    _pt("scramble_survived", scramble_survived)
    _pt("FURTHER-WORK BAR CLEARED", further_work_bar)

    holdout_read = False
    row_hold = None
    if further_work_bar:
        print("\n--- HOLDOUT: W_HOLD, U8, frozen 1.0x config (+1 holdout consultation) ---")
        holdout_read = True
        aligned_hold, targets_hold, ev_hold = build_cell(
            frames, UNIVERSE_8, W_HOLD, delta_in_1, buffer_1, center, k, hold_days, delta_out
        )
        cand_eq_hold = simulate_portfolio(targets_hold, aligned_hold, SPOT_BASE)
        bench_eq_hold, c_hold, vol_hold, matched_hold = volmatched_hold_equity(
            cand_eq_hold, aligned_hold, UNIVERSE_8, SPOT_BASE
        )
        _pt("VOLMATCH_HOLD matched (holdout)", f"{matched_hold} (c={c_hold:.4f}, vol={vol_hold:.4f})")
        if matched_hold:
            net_cmp_hold = compare(cand_eq_hold, bench_eq_hold)
            print("  HOLDOUT result (net_cmp vs VOLMATCH_HOLD, SPOT_BASE):")
            for kk, vv in net_cmp_hold.items():
                _pt(kk, vv)
            cand_eq_hold_gross = simulate_portfolio(targets_hold, aligned_hold, SPOT_FREE)
            bench_eq_hold_gross, _, _, matched_hold_gross = volmatched_hold_equity(
                cand_eq_hold_gross, aligned_hold, UNIVERSE_8, SPOT_FREE
            )
            gross_cmp_hold = compare(cand_eq_hold_gross, bench_eq_hold_gross)
            row_hold = frontier_row(
                "conservative_52w_high",
                {"delta_in": delta_in_1, "buffer": buffer_1, "k": k, "hold_days": hold_days,
                 "delta_out": delta_out, "multiplier": 1.0, "center": center, "sigma": sigma},
                targets_hold, net_cmp_hold, gross_cmp_hold, "VOLMATCH_HOLD", "W_HOLD", "U8",
                matched_net=matched_hold, matched_gross=matched_hold_gross,
            )
            results_rows.append(row_hold)
        else:
            print("  !!! HOLDOUT CELL VOIDED: volmatched_hold_equity did not match within tolerance !!!")
        print("+1 holdout consultation.")
    else:
        print("holdout not read.")

    # ----------------------------------------------------------- config count
    configs_after = config_count()
    trials_delta = configs_after - configs_before
    print(f"\n--- CONFIG COUNT: {configs_before} -> {configs_after} (delta = {trials_delta}) ---")

    # ----------------------------------------------------------- turnover
    print("\n--- TURNOVER (D1 cell) ---")
    tstats = turnover_stats(targets_d1)
    for kk, vv in tstats.items():
        _pt(kk, vv)
    r68_reference = 0.102
    _pt("R-68's own published rebalances_per_day", r68_reference)
    _pt("HIGHER than R-68's reference (F2 predicted this)",
        tstats["rebalances_per_day"] > r68_reference)

    # ----------------------------------------------------------- save CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(df_out)} rows to {RESULTS_CSV}")

    neighbours_csv = OUT_DIR / "conservative_52w_high_neighbours.csv"
    pd.DataFrame(neighbour_rows).to_csv(neighbours_csv, index=False)
    print(f"Saved neighbour check to {neighbours_csv}")

    scramble_csv = OUT_DIR / "conservative_52w_high_scramble.csv"
    pd.DataFrame({"seed": list(SCRAMBLE_SEEDS), "point_estimate": scramble_points}).to_csv(
        scramble_csv, index=False
    )
    print(f"Saved scramble results to {scramble_csv}")

    # ----------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"causality_ok={causality_ok}  band_selection_ok={band_ok} (err={band_err:.3e})")
    print(f"F1 spearman_rho={ra['spearman_rho']:.4f}  argmax_agreement={ra['argmax_agreement']:.4f}"
          f"  (n_bars={ra['n_bars']})")
    print(f"D1 cell voided: {d1_cell_voided}")
    print(f"D1={d1}  D2={d2}  D3={d3}  D5={d5}  scramble_survived={scramble_survived}")
    print(f"FURTHER-WORK BAR: {further_work_bar}   holdout_read={holdout_read}")
    print(f"config_count delta (this branch): {trials_delta}")


if __name__ == "__main__":
    main()
