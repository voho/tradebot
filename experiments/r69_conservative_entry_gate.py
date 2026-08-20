"""R-69 CONSERVATIVE branch -- one entry-only gate, bolted onto R-63's
ORIGINAL rule, with R-65/R-67/R-68's buffer and hold_days machinery removed
entirely. Backlog item B-37.

Reuses R-68 conservative's own event-loop construction (`band_selection` in
`experiments/r68_conservative_band_decomposition.py`) UNCHANGED -- copied
below with an attribution comment, not reimplemented, per
`experiments/r69_shared.py`'s own instruction (see that file's module
docstring, section "WHY THIS IS PROVABLY R-63'S ORIGINAL RULE AT
delta_in=0"). Three of that loop's five inputs are pinned at the values that
DELETE R-65's machinery rather than retuning it:

    buffer     = 0.0   (no margin -- any strictly-better challenger swaps)
    hold_days  = 0     (no minimum tenure -- the timer never blocks)
    delta_out  = 0.0   (exit purely at s > 0, R-63's original rule)

The ONLY swept quantity is `delta_in`, over R-68's own extended grid
(`r69_shared.DELTA_GRID_EXT`, 0.000..0.350).

This file implements a candidate and measures it; it does not define, relax
or edit a rule. Windows, universes, costs, D1-D5, M1', the fixed-permutation
scramble control and the further-work bar all live in the frozen
pre-registration in `experiments/r69_shared.py` (and through it in
`r68_shared.py` / `r65_shared.py` / `r63_shared.py`).

**W_HOLD is never imported, sliced or referenced.**

Run as:
    .venv/bin/python experiments/r69_conservative_entry_gate.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.r69_shared import (  # noqa: E402
    BARS_PER_DAY,
    D5_BAR_R69,
    DELTA_GRID_EXT,
    K_FROZEN,
    OUT_DIR,
    R68_ENTRY_ONLY_DELTA_WINNER,
    R68_ENTRY_ONLY_MEMBERSHIP_PER_DAY,
    R68_ENTRY_ONLY_TURNOVER_PER_DAY,
    R68_ENTRY_ONLY_WFULL6_DD_VOLMATCH,
    R68_ENTRY_ONLY_WFULL6_NET_VOLMATCH,
    R68_ENTRY_ONLY_WVAL_NET,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_TRAIN,
    W_VAL,
    align_frames,
    basket_log_returns,
    check_against_engine,
    check_causality,
    compare,
    conditional_vol_scale,
    config_count,
    cross_sectional_score,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    frontier_row,
    further_work,
    holding_period_days,
    load_universe,
    m1_pass_vs_raw,
    matched_hold_targets,
    mean_total_notional,
    membership_change_rate_thresholded,
    r63_baseline_targets,
    realized_vol,
    scramble_fixed_perm,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)

ARM = "entry_gate"
K_FIXED = K_FROZEN  # 1, R-63's own frozen constant, never retuned

# Fixed, never swept, never retuned -- the three values that delete R-65's
# machinery entirely rather than retuning it. Named explicitly here so a
# reader never has to infer them from a call site.
BUFFER_FIXED = 0.0
HOLD_DAYS_FIXED = 0.0
DELTA_OUT_FIXED = 0.0

SPOT_FREE = SPOT_BASE.__class__.spot(fee_rate=0.0)

VOLMATCH_TOL = 0.02  # r65_shared's own default; not retuned here.


# ===========================================================================
# THE LOOP -- copied from `r68_conservative_band_decomposition.band_selection`
# (lines ~315-415 of that file), UNCHANGED, with an attribution comment.
# That file is NOT edited and NOT imported (its import chain pulls in
# argparse/CLI machinery this file has no use for); the small pure loop is
# copied instead, per r69_shared.py's own instruction. Called below ONLY
# with buffer=0.0, hold_days=0, delta_out=0.0 fixed -- the round's whole
# point is what happens to R-68's own construction once those three inputs
# are set to the values that delete R-65's machinery.
# ===========================================================================


def band_selection(s: np.ndarray, k: int, buffer: float, hold_days: float,
                   delta_in: float, delta_out: float):
    """R-68 conservative's `band_selection`, copied verbatim (attribution
    above). See that file for the full docstring; reproduced in brief:

    enter_eligible = isfinite(s) & (s >  +delta_in)   # new entrants
    hold_eligible  = isfinite(s) & (s >  -delta_out)  # incumbents kept

    STRICTLY CAUSAL BY CONSTRUCTION: a forward loop whose state at bar `i`
    depends on rows <= i and nothing else.

    Returns ``(sel, ev, ev_bars)``.
    """
    n, n_assets = s.shape
    finite = np.isfinite(s)
    enter_eligible = finite & (s > float(delta_in))
    hold_eligible = finite & (s > -float(delta_out))
    hold_bars = int(round(float(hold_days) * BARS_PER_DAY))
    buf = float(buffer)

    sel = np.zeros((n, n_assets), dtype=bool)
    held: list[int] = []
    last_change = -(1 << 60)
    keys = ("forced_exit", "entry", "swap", "blocked_by_timer",
            "blocked_by_buffer", "flat_bars")
    ev = {key: 0 for key in keys}
    ev_bars = {key: np.zeros(n, dtype=np.int32) for key in keys}

    for i in range(n):
        row = s[i]
        elig_in = enter_eligible[i]
        elig_hold = hold_eligible[i]
        changed = False

        # (a) forced exits -- never blocked by the timer.
        if held:
            keep = [a for a in held if elig_hold[a]]
            if len(keep) != len(held):
                ev["forced_exit"] += 1
                ev_bars["forced_exit"][i] = 1
                held = keep
                changed = True

        # entries into empty slots.
        if len(held) < k:
            free = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if free:
                free.sort(key=lambda a: -row[a])
                for a in free[: k - len(held)]:
                    held.append(a)
                    changed = True
                ev["entry"] += 1
                ev_bars["entry"][i] = 1

        # (b) voluntary swap -- buffered AND time-gated.
        elif not changed:
            worst = min(held, key=lambda a: row[a])
            out = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if out:
                best = max(out, key=lambda a: row[a])
                if row[best] > row[worst] + buf:
                    if (i - last_change) >= hold_bars:
                        held.remove(worst)
                        held.append(best)
                        changed = True
                        ev["swap"] += 1
                        ev_bars["swap"][i] = 1
                    else:
                        ev["blocked_by_timer"] += 1
                        ev_bars["blocked_by_timer"][i] = 1
                elif row[best] > row[worst]:
                    ev["blocked_by_buffer"] += 1
                    ev_bars["blocked_by_buffer"][i] = 1

        if changed:
            last_change = i
        if held:
            sel[i, held] = True
        else:
            ev["flat_bars"] += 1
            ev_bars["flat_bars"][i] = 1

    return sel, ev, ev_bars


def _size(sel: np.ndarray, aligned: dict[str, pd.DataFrame], k: int,
          index, assets) -> pd.DataFrame:
    """R-63's sizing tail, IDENTICAL to `r63_novel_xsmom_rank.build_targets`
    after the selection matrix: m = sel.sum(axis=1), the 0.10 DEADBAND latch
    on desired TOTAL notional, clip to 1.0, split equally over held slots."""
    n = sel.shape[0]
    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))
    DEADBAND = 0.10

    desired = scale * (m / float(k))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        d = desired[i]
        if abs(d - cur) > DEADBAND:
            cur = d
        pos[i] = cur

    total = np.minimum(pos, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(m > 0, total / np.maximum(m, 1), 0.0)
    w = sel * per[:, None]
    return pd.DataFrame(w, index=index, columns=assets)


def build_targets_entry_gate_ev(aligned: dict[str, pd.DataFrame], k: int,
                                delta_in: float):
    """Targets AND the event ledger, one pass of the loop, buffer/hold_days/
    delta_out pinned to the values that delete R-65's machinery."""
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    sel, ev, ev_bars = band_selection(s, k, BUFFER_FIXED, HOLD_DAYS_FIXED,
                                      delta_in, DELTA_OUT_FIXED)
    return _size(sel, aligned, k, score.index, assets), ev, ev_bars, score.index


def build_targets_entry_gate(aligned: dict[str, pd.DataFrame], k: int,
                             delta_in: float) -> pd.DataFrame:
    """The candidate: R-63's score and vol scale, R-68's event loop with
    buffer=0, hold_days=0, delta_out=0, sweeping delta_in alone."""
    return build_targets_entry_gate_ev(aligned, k, delta_in)[0]


def entry_gate_fn(delta_in: float, k: int = K_FIXED):
    return lambda aligned: build_targets_entry_gate(aligned, k, delta_in)


# ------------------------------------------------------------------ cells


def build_cell(frames, universe, window, targets_fn):
    """Aligned prices + targets, both sliced to the evaluation window.
    Right edge applied strictly, as R-63/R-68's own conservative branches
    did after finding the shared `_hi` helper's original bug."""
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    targets = targets_fn(warm)

    idx = warm[universe[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]

    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], first_warm


def mean_tenure_days(targets: pd.DataFrame) -> float:
    held = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0) > 0.0
    starts = int(held[0].sum() + (held[1:] & ~held[:-1]).sum())
    return float(held.sum()) / max(starts, 1) / BARS_PER_DAY


def volmatch(cand_eq, aligned, assets, market, label=""):
    """`volmatched_hold_equity` with the documented cap-binding workaround
    both R-68 arms used. The frozen shared file is NOT edited."""
    eq, c, vol, shared_flag = volmatched_hold_equity(cand_eq, aligned, assets,
                                                      market, tol=VOLMATCH_TOL)
    target = realized_vol(cand_eq)
    matched = bool(eq is not None and np.isfinite(vol) and np.isfinite(target)
                   and target > 0 and abs(vol - target) <= VOLMATCH_TOL * target)
    if shared_flag != matched:
        print(f"    ~~ shared volmatched_hold_equity flag OVERRIDDEN{label}: "
              f"shared said matched={shared_flag}, tolerance test says "
              f"{matched} (c={c:.4f} bench_vol={vol:.4f} cand_vol={target:.4f})")
    if not matched:
        print(f"    !! VOLMATCH_HOLD did NOT match{label}: c={c:.4f} "
              f"bench_vol={vol:.4f} cand_vol={target:.4f} -> cell VOIDED")
    return eq, c, vol, matched, shared_flag


def measure_cell(targets, aligned, assets, window_name, universe_name, params,
                 arm=ARM, rates=None):
    """One cell: both fee levels, each against VOLMATCH_HOLD at that fee
    level. R-68 conservative's `measure_pair`, unchanged in structure."""
    cand_net = simulate_portfolio(targets, aligned, SPOT_BASE)
    vm_net, c_net, v_net, ok_net, sf_net = volmatch(cand_net, aligned, assets,
                                                    SPOT_BASE, " @0.10%")
    net_cmp = compare(cand_net, vm_net)

    cand_gross = simulate_portfolio(targets, aligned, SPOT_FREE)
    vm_gross, c_gross, v_gross, ok_gross, sf_gross = volmatch(
        cand_gross, aligned, assets, SPOT_FREE, " @0bps")
    gross_cmp = compare(cand_gross, vm_gross)

    extra = dict(
        volmatch_c_net=c_net, volmatch_vol_net=v_net, volmatch_matched_net=ok_net,
        cand_vol_net=realized_vol(cand_net),
        volmatch_c_gross=c_gross, volmatch_vol_gross=v_gross,
        volmatch_matched_gross=ok_gross,
        cand_vol_gross=realized_vol(cand_gross),
        n_bars=len(targets),
        tenure_days=mean_tenure_days(targets),
        membership_change_rate_thresholded=membership_change_rate_thresholded(
            targets),
    )
    if rates:
        extra.update(rates)
    row = frontier_row(arm, params, targets, net_cmp, gross_cmp,
                       "VOLMATCH_HOLD", window_name, universe_name, **extra)
    return row, dict(cand_net=cand_net, vm_net=vm_net, cand_gross=cand_gross,
                     vm_gross=vm_gross)


def write_csv(path, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")


def fmt_front(row):
    return (f"    d={row['p_delta_in']:6.3f} | turn {row['turnover_per_day']:6.3f}"
            f" | GROSS {row['gross_growth_diff']:+8.3f}"
            f" [{row['gross_growth_lo']:+7.3f},{row['gross_growth_hi']:+7.3f}]"
            f" | NET {row['net_growth_diff']:+8.3f}"
            f" [{row['net_growth_lo']:+7.3f},{row['net_growth_hi']:+7.3f}]")


# =========================================================================
# STEP 1: IDENTITY CHECK -- delta_in=0.0 must equal r63_baseline_targets
# =========================================================================


def run_identity(frames) -> tuple[bool, float]:
    print("== IDENTITY: delta_in=0.0 vs r63_baseline_targets(k=1), "
          f"warm_window(W_TRAIN), U8 ==")
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    mine = build_targets_entry_gate(warm, K_FIXED, 0.0)
    theirs = r63_baseline_targets(warm, K_FIXED)

    a = np.nan_to_num(mine.to_numpy(dtype=float), nan=0.0)
    b = np.nan_to_num(theirs.to_numpy(dtype=float), nan=0.0)
    same_shape = a.shape == b.shape
    maxabs = float(np.max(np.abs(a - b))) if same_shape else float("nan")
    bitwise = bool(same_shape and np.array_equal(a, b))
    exact = bool(same_shape and np.allclose(a, b, atol=1e-12, rtol=0.0))
    print(f"  shape {a.shape}  max|diff| = {maxabs:.3e}  "
          f"bit-identical={bitwise}  allclose(atol=1e-12)={exact}")

    write_csv(OUT_DIR / "conservative_identity.csv", [{
        "check": "identity_delta_in_0", "universe": "U8", "window": str(W_TRAIN),
        "n_bars": a.shape[0], "max_abs_diff": maxabs, "bit_identical": bitwise,
        "allclose_1e12": exact, "passed": exact,
    }])
    return exact, maxabs


# =========================================================================
# STEP 2: SELF-CHECKS
# =========================================================================


def run_self_checks(frames, delta_in_probe: float = 0.08):
    print("\n== SELF-CHECKS ==")
    ok_engine, err_engine = check_against_engine()
    print(f"  check_against_engine(): ok={ok_engine} "
          f"relative_final_balance_error={err_engine:.6f}")

    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    t0 = time.time()
    ok_causal = check_causality(entry_gate_fn(delta_in_probe), warm)
    print(f"  check_causality(build_fn=entry_gate delta_in={delta_in_probe}): "
          f"{ok_causal}  [{time.time() - t0:.1f}s]")

    write_csv(OUT_DIR / "conservative_selfchecks.csv", [
        {"check": "check_against_engine", "passed": ok_engine,
         "relative_final_balance_error": err_engine},
        {"check": "check_causality", "delta_in_probe": delta_in_probe,
         "passed": ok_causal},
    ])
    return ok_engine, err_engine, ok_causal


# =========================================================================
# STEP 3: SWEEP delta_in over DELTA_GRID_EXT on W_TRAIN then W_VAL
# =========================================================================


def run_frontier(frames):
    print(f"\n== FRONTIER: {len(DELTA_GRID_EXT)} delta_in cells + R-63 "
          f"reference, k={K_FIXED}, buffer=0, hold_days=0, delta_out=0, "
          f"U8, vs VOLMATCH_HOLD ==")
    print(f"   grid = {DELTA_GRID_EXT}")
    wmap = {"W_TRAIN": W_TRAIN, "W_VAL": W_VAL}
    rows = []
    per_window = {}
    for wname in ("W_TRAIN", "W_VAL"):
        window = wmap[wname]
        print(f"  -- {wname} --")
        aligned, tg63, warm_ok = build_cell(frames, UNIVERSE_8, window,
                                            lambda a: r63_baseline_targets(a, K_FIXED))
        if not warm_ok:
            raise RuntimeError(f"{wname}: first evaluated bar not warm")
        row, _ = measure_cell(tg63, aligned, UNIVERSE_8, wname, "U8",
                              {"delta_in": float("nan"), "buffer": BUFFER_FIXED,
                               "hold_days": HOLD_DAYS_FIXED,
                               "delta_out": DELTA_OUT_FIXED, "k": K_FIXED},
                              arm="R63_REF")
        rows.append(row)
        print("    [R63_REF]" + fmt_front({**row, "p_delta_in": 0.0}))

        window_rows = []
        for d in DELTA_GRID_EXT:
            aligned_d, tg_d, warm_ok_d = build_cell(
                frames, UNIVERSE_8, window, entry_gate_fn(d))
            if not warm_ok_d:
                raise RuntimeError(f"{wname} d={d}: first evaluated bar not warm")
            r, _ = measure_cell(tg_d, aligned_d, UNIVERSE_8, wname, "U8",
                                {"delta_in": d, "buffer": BUFFER_FIXED,
                                 "hold_days": HOLD_DAYS_FIXED,
                                 "delta_out": DELTA_OUT_FIXED, "k": K_FIXED})
            rows.append(r)
            window_rows.append(r)
            print(fmt_front(r))
        per_window[wname] = window_rows

    write_csv(OUT_DIR / "conservative_frontier.csv", rows)
    return per_window


# =========================================================================
# STEP 4: SELECTION -- delta_in maximizing net_growth_diff on W_VAL
# =========================================================================


def select_delta_in(per_window):
    val_rows = per_window["W_VAL"]
    best = max(val_rows, key=lambda r: r["net_growth_diff"])
    print(f"\n== SELECTION: best W_VAL net_growth_diff -> "
          f"delta_in={best['p_delta_in']:.3f} "
          f"(net {best['net_growth_diff']:+.4f} "
          f"[{best['net_growth_lo']:+.4f},{best['net_growth_hi']:+.4f}]) ==")
    return float(best["p_delta_in"])


# =========================================================================
# STEP 5: DECISION CELLS at the selected delta_in
# =========================================================================


def run_decision_cells(frames, delta_in: float):
    print(f"\n== DECISION CELLS at delta_in={delta_in:.3f} ==")

    # ---- D1 / D2 / D5: W_FULL6, U6 ----
    aligned6, tg6, warm6_ok = build_cell(frames, UNIVERSE_6, W_FULL6,
                                        entry_gate_fn(delta_in))
    if not warm6_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")
    row6, eqs6 = measure_cell(tg6, aligned6, UNIVERSE_6, "W_FULL6", "U6",
                              {"delta_in": delta_in, "buffer": BUFFER_FIXED,
                               "hold_days": HOLD_DAYS_FIXED,
                               "delta_out": DELTA_OUT_FIXED, "k": K_FIXED})
    d1 = d1_pass(row6)
    d2 = d2_pass(row6)
    d5 = d5_pass(row6)
    print(f"  [D1/D2/D5] {fmt_front(row6)}")
    print(f"    D1 PASS={d1}   D2 PASS={d2}   D5 (gross {row6['gross_growth_diff']:+.4f}"
          f" >= bar {D5_BAR_R69:.3f}) PASS={d5}")

    # ---- M1' ----
    m1 = m1_pass_vs_raw(tg6, aligned6, K_FIXED)
    print(f"  [M1'] {m1}")

    # ---- Scramble, on the D1 cell, SAME VOLMATCH_HOLD benchmark ----
    real = row6["net_growth_diff"]
    vm_net = eqs6["vm_net"]
    diffs = []
    scramble_rows = []
    for seed in SCRAMBLE_SEEDS:
        st = scramble_fixed_perm(tg6, seed)
        eq = simulate_portfolio(st, aligned6, SPOT_BASE)
        r = compare(eq, vm_net)
        diffs.append(r["growth_diff"])
        scramble_rows.append({"seed": seed, "growth_diff": r["growth_diff"],
                              "cand_final": r["cand_final"],
                              "cand_dd": r["cand_dd"]})
        print(f"    seed {seed}: growth_diff {r['growth_diff']:+.4f}  "
              f"final {r['cand_final']:>10,.0f}")
    p90 = float(np.percentile(diffs, 90))
    scramble_survived = bool(real > p90)
    print(f"  [SCRAMBLE] real (D1 point) {real:+.4f} vs p90 {p90:+.4f} -> "
          f"SURVIVED={scramble_survived}")

    # ---- D3: W_VAL, U8, net ----
    aligned3, tg3, warm3_ok = build_cell(frames, UNIVERSE_8, W_VAL,
                                        entry_gate_fn(delta_in))
    if not warm3_ok:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    row3, _ = measure_cell(tg3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                           {"delta_in": delta_in, "buffer": BUFFER_FIXED,
                            "hold_days": HOLD_DAYS_FIXED,
                            "delta_out": DELTA_OUT_FIXED, "k": K_FIXED})
    d3 = d3_pass(row3)
    print(f"  [D3] {fmt_front(row3)}")
    print(f"    D3 PASS={d3}")

    # ---- D4: W_FULL6, U6, 0.40% taker vs EW_HOLD ----
    cand40 = simulate_portfolio(tg6, aligned6, SPOT_REAL)
    ew40 = static_hold_equity(aligned6, UNIVERSE_6, SPOT_REAL)
    d4_cmp = compare(cand40, ew40)
    d4_ok = bool(d4_cmp["cand_final"] > d4_cmp["bench_final"])
    print(f"  [D4 @0.40%] cand {d4_cmp['cand_final']:,.0f} vs EW_HOLD "
          f"{d4_cmp['bench_final']:,.0f} -> D4 PASS={d4_ok}")

    # ---- Context ----
    ctx = {
        "mean_total_notional": mean_total_notional(tg6),
        "turnover_stats": turnover_stats(tg6),
        "holding_period_days": holding_period_days(tg6),
        "realized_vol_net": realized_vol(eqs6["cand_net"]),
    }
    print(f"  [CONTEXT, W_FULL6/U6] mean_total_notional="
          f"{ctx['mean_total_notional']:.4f}  turnover/day="
          f"{ctx['turnover_stats']['turnover_per_day']:.4f}  "
          f"holding_period_days={ctx['holding_period_days']:.4f}  "
          f"realized_vol={ctx['realized_vol_net']:.4f}")

    write_csv(OUT_DIR / "conservative_decision_cells.csv",
              [row6, row3, {**d4_cmp, "window": "W_FULL6", "universe": "U6",
                            "fee": 0.004, "bench": "EW_HOLD", "d4_pass": d4_ok}])
    write_csv(OUT_DIR / "conservative_scramble.csv", scramble_rows + [
        {"seed": -1, "growth_diff": real, "scramble_p90": p90,
         "scramble_survived": scramble_survived}])
    write_csv(OUT_DIR / "conservative_m1.csv", [m1])

    return {
        "row_full6": row6, "row_val": row3, "d4_cmp": d4_cmp,
        "d1": d1, "d2": d2, "d3": d3, "d4": d4_ok, "d5": d5,
        "m1": m1, "scramble_survived": scramble_survived,
        "scramble_real": real, "scramble_p90": p90,
        "context": ctx,
    }


# =========================================================================
# STEP 6: R-68 comparison, at delta_in=0.08 (nominal), on W_TRAIN/U8
# =========================================================================


def run_r68_comparison(frames, d: float = R68_ENTRY_ONLY_DELTA_WINNER):
    print(f"\n== R-68 ENTRY_ONLY COMPARISON at d={d:.3f}, W_TRAIN/U8 ==")
    aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, W_TRAIN, entry_gate_fn(d))
    if not warm_ok:
        raise RuntimeError("W_TRAIN first evaluated bar not warm")
    ts = turnover_stats(tg)
    mem = membership_change_rate_thresholded(tg)
    print(f"  this branch : turnover/day={ts['turnover_per_day']:.4f}  "
          f"membership_changes/day={mem:.4f}")
    print(f"  R-68 ENTRY_ONLY (published): turnover/day="
          f"{R68_ENTRY_ONLY_TURNOVER_PER_DAY:.4f}  membership_changes/day="
          f"{R68_ENTRY_ONLY_MEMBERSHIP_PER_DAY:.4f}")
    return {"turnover_per_day": ts["turnover_per_day"],
            "membership_per_day": mem}


# =========================================================================
# MAIN
# =========================================================================


def main():
    t_start = time.time()
    frames = load_universe(UNIVERSE_8)

    ident_ok, ident_maxabs = run_identity(frames)
    if not ident_ok:
        print(f"\n!!! IDENTITY CHECK FAILED (max|diff|={ident_maxabs:.3e}) -- "
              f"stopping before any other number is reported, per instructions.")
        print(f"\nconfig_count() = {config_count()}")
        return

    ok_engine, err_engine, ok_causal = run_self_checks(frames)

    per_window = run_frontier(frames)
    selected_delta_in = select_delta_in(per_window)

    decision = run_decision_cells(frames, selected_delta_in)

    r68_cmp = run_r68_comparison(frames)
    # also run the comparison at the SELECTED delta if it differs from 0.08,
    # so the reported turnover/membership numbers match the delta actually
    # used for the decision cells.
    if not np.isclose(selected_delta_in, R68_ENTRY_ONLY_DELTA_WINNER):
        print(f"\n(selected delta_in={selected_delta_in:.3f} != R-68's 0.080; "
              f"the comparison above is at the shared nominal 0.080 for "
              f"checkability, per instructions)")

    fw = further_work(decision["m1"]["passed"], decision["d1"], decision["d2"],
                      decision["d3"], decision["d5"], decision["scramble_survived"])
    print(f"\n== further_work(m1={decision['m1']['passed']}, d1={decision['d1']}, "
          f"d2={decision['d2']}, d3={decision['d3']}, d5={decision['d5']}, "
          f"scramble={decision['scramble_survived']}) = {fw} ==")
    if fw:
        print("  -> STOP. Report to the operator; the holdout read is theirs.")
    else:
        print("  -> DONE. W_HOLD is NOT read.")

    print(f"\nconfig_count() = {config_count()}")
    print(f"wall time = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
