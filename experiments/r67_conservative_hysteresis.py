"""R-67 CONSERVATIVE branch -- asymmetric hysteresis on the long/flat gate.

R-65's conservative arm bought a 3.9x turnover cut essentially for free and
still missed its own break-even by 1.38:1 (needed turnover <= 0.641/day,
reached 0.900). It named the reason precisely, and B-31 files it: R-63's frozen
selection rule is `eligible = isfinite(s) & (s > 0.0)` -- "hold only
positive-scoring assets, flat otherwise" -- and **every crossing of that exact
boundary forces an exit that neither the rank buffer nor the minimum-hold timer
is allowed to block**. R-65 measured that channel at **0.386 forced exits/day,
invariant across all 20 cells** of its buffer x hold_days grid, while voluntary
swaps fell 16-fold (0.270 -> 0.017/day). Neither of its two free parameters
governs the channel, so neither could move it.

This branch attacks that channel directly and literally, exactly as B-31 and
`experiments/r67_shared.py` name it.

=====================================================================
THE RULE (frozen before any number was read)
=====================================================================

Start from `r65_conservative_rank_buffer.build_buffered_targets`
**byte-for-byte** at R-65's own frozen winner -- `k=1, buffer=0.05,
hold_days=1`, NOT re-selected here -- and change ONLY the eligibility test.
R-65 applies one symmetric test to incumbents and challengers alike:

    eligible = isfinite(s) & (s > 0.0)                     # R-65 / R-63

This branch splits it in two, asymmetrically, around zero:

    ENTRY.      a new entrant (an empty slot's filler, or a challenger in a
                voluntary swap) requires  score >  +delta
    RETENTION.  an asset already held stays eligible until its score falls to
                or below -delta, i.e. it is retained while  score >  -delta

`delta` is the ONE free parameter. Everything below the selection -- R-63's
composite cross-sectional score, its conditional volatility scale, its 0.10
deadband on desired TOTAL notional, equal weighting among held slots, the 1.0
long-only unlevered cap -- is R-63's, untouched, imported and never copied.
`buffer` stays at 0.05 and `hold_days` at 1: R-65's own selection on that axis,
inherited, not re-run.

**delta = 0.0 must reproduce R-65's frozen winner EXACTLY**, and that identity
is verified numerically (`identity` subcommand) before anything else is
reported. That is why both halves of the test are written with a STRICT `>`:
at delta = 0 they collapse to the identical `s > 0.0` predicate and the target
matrix is bit-identical. The literal phrasing "stays eligible until score <
-delta" would put retention at `s >= -delta`, which differs from R-65 on the
measure-zero event `s == 0.0` exactly; the strict form is chosen so that the
pre-registered identity requirement holds bit-for-bit, and the number of bars
at which any score lands exactly on a grid boundary is COUNTED and reported
rather than assumed to be zero.

The score is in raw units of `close/anchor - 1`, so delta = 0.01 is one
percentage point. Nothing here is normalized, standardized or ranked over
time, so no whole-series statistic exists to leak; the selection is a forward
loop whose state at bar i depends on rows <= i and nothing else, and the 60%
truncation probe verifies that rather than this docstring.

GRID: delta in {0.000, 0.005, 0.010, 0.020, 0.040, 0.080}
      = 6 parameter-search cells. Swept on W_TRAIN, selected on W_VAL.

=====================================================================
SELECTION CRITERION -- DECLARED HERE BEFORE THE SWEEP WAS RUN
=====================================================================

The frozen configuration is the grid cell with the highest **W_VAL net growth
difference versus VOLMATCH_HOLD at 0.10%** -- the D1 decision statistic
evaluated on the selection window, which is R-63's and R-65's own convention.
Tie-break: the more negative W_VAL net drawdown difference.

**No filter on the gross column is applied at selection time.** The gross
column is D5's diagnostic and selecting on it would be selecting on the
falsification test. This criterion is honoured as written; if it selects a
cell that then fails D5, that is a reported finding, not a reason to reselect.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- the shared pre-registration's F1 and F2
=====================================================================

**(F1)** R-66 found twice, from opposite directions, that reaching exactly flat
was not what cost R-64's arm growth, and that softening the trip to flat either
failed to help or actively cost more. Widening the exit threshold here means
holding a genuinely declining asset longer before de-risking, in a long-only
universe where "declining" often means "about to keep declining". If the forced
exits are informative rather than degenerate, this trades turnover for drawdown
at unfavourable odds.

**(F2)** The 0.386/day floor may be a property of the composite score's own
sign-change frequency rather than of the gate's threshold, in which case no
downstream softening reduces it -- it only relabels fast exits as slow fades at
the same underlying rate. **M1 is what tells the two apart**, and this branch
reports the M1 rate for EVERY grid cell, not only the selected one: if M1
passes only at a delta extreme enough to gut D5, F2 is confirmed.

Windows, universes, costs, benchmarks, D1/D2/D3/D5, D4, M1 and the further-work
bar all live in the frozen pre-registration in `experiments/r67_shared.py` (and,
through it, `r65_shared.py` / `r63_shared.py`). This file implements a candidate
and measures it; it does not define, relax or edit a rule. **W_HOLD is never
sliced, imported or referenced.**

Run as:
    python3 experiments/r67_conservative_hysteresis.py identity
    python3 experiments/r67_conservative_hysteresis.py checks
    python3 experiments/r67_conservative_hysteresis.py m1
    python3 experiments/r67_conservative_hysteresis.py frontier
    python3 experiments/r67_conservative_hysteresis.py select
    python3 experiments/r67_conservative_hysteresis.py run
    python3 experiments/r67_conservative_hysteresis.py scramble
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.r67_shared import (  # noqa: E402
    BARS_PER_DAY,
    D5_BAR,
    DEADBAND,
    M1_MIN_REDUCTION,
    OUT_DIR,
    R65_BUFFER,
    R65_FORCED_EXIT_PER_DAY,
    R65_HOLD_DAYS,
    R65_K,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_FREE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_TRAIN,
    W_VAL,
    align_frames,
    basket_log_returns,
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
    m1_pass,
    matched_hold_targets,
    mean_total_notional,
    mean_total_notional as _mtn,
    membership_change_rate,
    r63_baseline_targets,
    r65_winner_targets,
    realized_vol,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)

ARM = "gate_hysteresis"

# R-65's frozen winner, inherited. NOT re-selected in this round.
K_FIXED = R65_K            # 1
BUFFER_FIXED = R65_BUFFER  # 0.05
HOLD_FIXED = R65_HOLD_DAYS  # 1

DELTA_GRID = (0.000, 0.005, 0.010, 0.020, 0.040, 0.080)

# ---------------------------------------------------------------------------
# FROZEN CONFIGURATION.
#
# To be set from `conservative_frontier.csv` by `cmd_select`, on **W_VAL
# only**, on the criterion declared in this module's docstring above, which was
# fixed before the sweep was run -- and set BEFORE any D-cell is computed. The
# W_VAL ordering as measured, the W_TRAIN ordering, the neighbourhood and the
# cross-window rank correlation are all recorded here at freezing time (and
# printed by `cmd_select`), rather than discovered afterwards.
#
# W_VAL ordering AS MEASURED (net growth diff vs VOLMATCH_HOLD @0.10%, best
# first); full table in reports/r67_gate/conservative_frontier.csv:
#   delta=0.080  net +0.5194  gross +0.5436  dd -16.47   <-- SELECTED
#   delta=0.020  net +0.4223  gross +0.4930  dd -10.51
#   delta=0.010  net +0.3673  gross +0.4771  dd -11.64
#   delta=0.040  net +0.3514  gross +0.3990  dd -12.27
#   delta=0.005  net +0.2900  gross +0.4415  dd  -9.03
#   delta=0.000  net +0.1425  gross +0.5699  dd  -1.82   (= R-65's own winner,
#                                                         reproduced exactly)
# All six W_VAL cells are net-positive; none has an interval excluding zero.
#
# THE NEIGHBOURHOOD, recorded at freezing time rather than discovered
# afterwards, and it is a MIXED verdict that is reported as such:
#   - the winner is a GRID CORNER. delta=0.080 is the largest value swept, so
#     it has only one immediate neighbour (delta=0.040, W_VAL net +0.3514 --
#     worse). Whether the curve has turned over past 0.080 is UNTESTED, and
#     the shared pre-registration's (F3) -- "the frontier's far end is a hold"
#     -- is the specific risk a corner selection carries. Not re-gridded after
#     seeing the number: the grid was declared before the sweep and extending
#     it now would be moving the goalposts.
#   - the winner ranks 1 of 6 on W_TRAIN as well (net -0.9672, the best of the
#     six there too), so the two selection windows AGREE on the winner.
#   - Spearman rank correlation of net growth between W_TRAIN and W_VAL over
#     the 6 cells is +0.486. The ordering TRANSFERS -- the opposite of R-65's
#     holding-period axis, which anti-transferred at -0.316. Both windows
#     agree on both endpoints (delta=0.000 worst, delta=0.080 best); the
#     disagreement is confined to the interior.
# The selection rule was fixed before the sweep and is honoured exactly as
# written. No filter on the gross column was applied.
FROZEN_DELTA: float | None = 0.080
# ---------------------------------------------------------------------------


# ------------------------------------------------------------------ targets


def hysteresis_selection(s: np.ndarray, k: int, buffer: float,
                         hold_days: float, delta: float):
    """R-65's `buffered_selection`, byte-for-byte, with ONE change: the single
    symmetric eligibility test is split into an asymmetric pair.

    R-65:   eligible     = isfinite(s) & (s >  0.0)     for everyone
    Here:   enter_elig   = isfinite(s) & (s >  +delta)  new entrants
            hold_elig    = isfinite(s) & (s >  -delta)  incumbents

    At delta = 0.0 the two predicates are the identical expression and this
    function is R-65's function. Everything else -- the loop, the event
    ledger, the ordering of the three cases, the timer's exemption for forced
    exits -- is unchanged.

    STRICTLY CAUSAL BY CONSTRUCTION: a forward loop whose state at bar ``i``
    depends on rows <= i and nothing else. No mean, std, quantile, scaler or
    time-series rank is taken anywhere; `delta` and `buffer` are in RAW score
    units precisely so that no normalization is needed.

    Returns ``(sel, ev, ev_bars)``. ``ev`` is R-65's own aggregate event
    ledger; ``ev_bars`` is the same events as per-bar 0/1 arrays so a rate can
    be computed on an evaluation SLICE rather than on the warm-up-inclusive
    range (see :func:`event_rates`).
    """
    n, n_assets = s.shape
    finite = np.isfinite(s)
    enter_eligible = finite & (s > float(delta))
    hold_eligible = finite & (s > -float(delta))
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

        # (a) forced exits -- never blocked by the timer. An incumbent leaves
        #     only once its score is no longer above -delta.
        if held:
            keep = [a for a in held if elig_hold[a]]
            if len(keep) != len(held):
                ev["forced_exit"] += 1
                ev_bars["forced_exit"][i] = 1
                held = keep
                changed = True

        # entries into empty slots (including refilling a slot a forced exit
        # just freed, and re-entering from flat). Allowed immediately, as in
        # R-65. A new entrant must clear +delta.
        if len(held) < k:
            free = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if free:
                free.sort(key=lambda a: -row[a])
                for a in free[: k - len(held)]:
                    held.append(a)
                    changed = True
                ev["entry"] += 1
                ev_bars["entry"][i] = 1

        # (b) voluntary swap -- buffered AND time-gated. The challenger is a
        #     new entrant and must clear +delta.
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


def build_hysteresis_targets(aligned: dict[str, pd.DataFrame], k: int,
                             buffer: float, hold_days: float,
                             delta: float) -> pd.DataFrame:
    """R-65's `build_buffered_targets` with :func:`hysteresis_selection` in
    place of `buffered_selection`. The sizing block below the selection is
    R-65's / R-63's, copied byte-for-byte and unmodified."""
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    n = s.shape[0]

    sel, _, _ = hysteresis_selection(s, k, buffer, hold_days, delta)

    # ---- sizing: R-63's, untouched -------------------------------------
    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))

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
    return pd.DataFrame(w, index=score.index, columns=assets)


def hyst_fn(delta: float, k: int = K_FIXED, buffer: float = BUFFER_FIXED,
            hold_days: float = HOLD_FIXED):
    return lambda aligned: build_hysteresis_targets(aligned, k, buffer,
                                                    hold_days, delta)


def r63_fn(k: int):
    return lambda aligned: r63_baseline_targets(aligned, k)


# ------------------------------------------------------------------ cells


def build_cell(frames, universe, window, targets_fn):
    """Aligned prices + targets, both sliced to the evaluation window.

    The right edge is applied STRICTLY (``idx < end + 1 day``), independently
    of the shared `_hi` helper -- the guard R-63's conservative branch added
    after finding that helper admitted one bar of the reserved holdout.
    """
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
    """Mean length of one asset's CONTINUOUS holding spell, in days.

    R-65's own diagnostic: `holding_period_days` measures time between changes
    to the held SET, which for a long/flat arm counts a flat spell as a
    holding. This measures how long an asset is actually owned once bought.
    """
    held = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0) > 0.0
    starts = int(held[0].sum() + (held[1:] & ~held[:-1]).sum())
    return float(held.sum()) / max(starts, 1) / BARS_PER_DAY


def raw_turnover(targets: pd.DataFrame) -> dict:
    """R-63's OWN turnover convention (no deadband), for comparability with
    the published 3.44/day and 2.86 changes/day."""
    w = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0)
    dw = np.abs(np.diff(w, axis=0)).sum(axis=1)
    days = (targets.index[-1] - targets.index[0]).total_seconds() / 86400.0
    sel = w > 0
    chg = int((np.abs(np.diff(sel.astype(int), axis=0)).sum(axis=1) > 0).sum())
    return {"raw_turnover_per_day": float(dw.sum() / days),
            "membership_changes_per_day": chg / days,
            "span_days": days}


VOLMATCH_TOL = 0.02  # the shared file's own default; not retuned here.


def volmatch(cand_eq, aligned, assets, market, label=""):
    """`volmatched_hold_equity`, with R-65's documented WORKAROUND for the
    cap-binding early return. The frozen shared file is NOT edited.

    r65_shared.py now carries the amended version (the early return tests the
    tolerance), so this should be a no-op here -- but the boolean is recomputed
    from the function's own final-line criterion anyway, no threshold is
    invented, no equity curve is altered, and BOTH flags are written to every
    CSV row (`volmatch_matched_*` is what scores the cell,
    `volmatch_shared_flag_*` is what the shared function returned) so any
    divergence is auditable rather than assumed absent.
    """
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
              f"bench_vol={vol:.4f} cand_vol={target:.4f} "
              f"(gap {abs(vol - target) / max(target, 1e-12):.3%}) "
              f"-> cell is VOIDED, not scored")
    return eq, c, vol, matched, shared_flag


def measure_pair(targets, aligned, assets, window_name, universe_name, params,
                 arm=ARM):
    """One grid cell: both fee levels, each against VOLMATCH_HOLD computed at
    that fee level. R-65's `measure_pair`, unchanged in structure."""
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
        volmatch_shared_flag_net=sf_net,
        cand_vol_net=realized_vol(cand_net),
        volmatch_c_gross=c_gross, volmatch_vol_gross=v_gross,
        volmatch_matched_gross=ok_gross,
        volmatch_shared_flag_gross=sf_gross,
        cand_vol_gross=realized_vol(cand_gross),
        cand_final_gross=float(cand_gross.iloc[-1]),
        bench_final_gross=float(vm_gross.iloc[-1]),
        n_bars=len(targets),
        tenure_days=mean_tenure_days(targets),
        membership_change_rate=membership_change_rate(targets),
        **raw_turnover(targets),
    )
    row = frontier_row(arm, params, targets, net_cmp, gross_cmp,
                       "VOLMATCH_HOLD", window_name, universe_name, **extra)
    return row, dict(cand_net=cand_net, vm_net=vm_net, cand_gross=cand_gross,
                     vm_gross=vm_gross, net_cmp=net_cmp, gross_cmp=gross_cmp,
                     matched_net=ok_net, matched_gross=ok_gross)


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


def log_configs(cmd: str):
    """Append this process's `config_count()` to an owned CSV.

    `config_count` is process-wide, and the subcommands are run as separate
    processes, so no single call can report the round's total. The branch total
    is the SUM of the rows in this file, and it is reported that way.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "conservative_configcount.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["cmd", "config_count", "utc"])
        wr.writerow([cmd, config_count(), pd.Timestamp.now("UTC").isoformat()])
    print(f"\nconfig_count() = {config_count()}   (appended to {path.name})")


def fmt_front(row):
    return (f"    set {row['hold_days']:6.2f}d ten {row['tenure_days']:6.2f}d"
            f" | turn {row['turnover_per_day']:6.3f}"
            f" (raw {row['raw_turnover_per_day']:6.3f}) | chg/d "
            f"{row['membership_changes_per_day']:6.3f} | mtn "
            f"{row['mean_notional']:.3f} | GROSS {row['gross_growth_diff']:+8.3f}"
            f" [{row['gross_growth_lo']:+7.3f},{row['gross_growth_hi']:+7.3f}]"
            f" | NET {row['net_growth_diff']:+8.3f}"
            f" [{row['net_growth_lo']:+7.3f},{row['net_growth_hi']:+7.3f}]")


# ---------------------------------------------------------------- identity


def cmd_identity(frames):
    """GATE ZERO. delta = 0.0 must reproduce R-65's frozen winner EXACTLY.

    Compared against `r67_shared.r65_winner_targets`, which is R-65's own
    committed `build_buffered_targets` imported (not reconstructed), on the
    full warm-inclusive target matrix for both universes.
    """
    print("== IDENTITY: delta=0.0 vs r67_shared.r65_winner_targets ==")
    rows, ok = [], True
    for uname, uni, window in (("U8", UNIVERSE_8, W_TRAIN),
                               ("U6", UNIVERSE_6, W_FULL6)):
        warm = align_frames({t: frames[t] for t in uni}, warm_window(window))
        mine = build_hysteresis_targets(warm, K_FIXED, BUFFER_FIXED,
                                        HOLD_FIXED, 0.0)
        theirs = r65_winner_targets(warm)
        a = np.nan_to_num(mine.to_numpy(dtype=float), nan=0.0)
        b = np.nan_to_num(theirs.to_numpy(dtype=float), nan=0.0)
        same_shape = a.shape == b.shape
        maxabs = float(np.max(np.abs(a - b))) if same_shape else float("nan")
        bitwise = bool(same_shape and np.array_equal(a, b))
        exact = bool(same_shape and np.allclose(a, b, atol=1e-12, rtol=0.0))
        print(f"  {uname} {window}: shape {a.shape} vs {b.shape}  "
              f"max|diff| = {maxabs:.3e}  bit-identical={bitwise}  "
              f"allclose(atol=1e-12)={exact}")
        ok &= exact
        rows.append({"check": "identity_delta0", "universe": uname,
                     "window": str(window), "n_bars": a.shape[0],
                     "max_abs_diff": maxabs, "bit_identical": bitwise,
                     "allclose_1e12": exact})

    # How close does any score ever come to a grid boundary? The retention
    # test is strict (`s > -delta`); the literal "until s < -delta" reading
    # would differ only where a score lands EXACTLY on -delta. Counted, not
    # assumed.
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    s = cross_sectional_score(warm).to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    for d in DELTA_GRID:
        n_hi = int((s == float(d)).sum())
        n_lo = int((s == -float(d)).sum())
        print(f"  boundary ties at delta={d:.3f}: s == +delta -> {n_hi}, "
              f"s == -delta -> {n_lo}  (of {len(s):,} finite score cells)")
        rows.append({"check": "boundary_ties", "universe": "U8",
                     "window": str(W_TRAIN), "delta": d,
                     "n_eq_plus_delta": n_hi, "n_eq_minus_delta": n_lo,
                     "n_finite_scores": len(s)})

    write_csv(OUT_DIR / "conservative_identity.csv", rows)
    print(f"  IDENTITY HOLDS: {ok}")
    return ok


# ------------------------------------------------------------------ checks


def truncation_probe(frames, delta, frac=0.6) -> tuple[bool, float]:
    """Build targets on the first 60% of bars and on 100%; the first 60% of
    rows must agree EXACTLY (atol=1e-12)."""
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * frac)
    full = build_hysteresis_targets(warm, K_FIXED, BUFFER_FIXED, HOLD_FIXED,
                                    delta).to_numpy()
    trunc = build_hysteresis_targets({t: df.iloc[:cut] for t, df in warm.items()},
                                     K_FIXED, BUFFER_FIXED, HOLD_FIXED,
                                     delta).to_numpy()
    a = np.nan_to_num(full[:cut], nan=0.0)
    b = np.nan_to_num(trunc[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0)), float(np.max(np.abs(a - b)))


def tail_perturbation_probe(frames, delta, frac_tail=0.4) -> tuple[bool, float]:
    """R-63's complementary probe: multiply the TAIL of every price series by
    10 and require the EARLY rows to be bit-identical. Truncation removes the
    tail; this corrupts it."""
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * (1.0 - frac_tail))
    bad = {}
    for t, df in warm.items():
        d = df.copy()
        for col in ("open", "high", "low", "close"):
            v = d[col].to_numpy(dtype=float).copy()
            v[cut:] *= 10.0
            d[col] = v
        bad[t] = d
    a = np.nan_to_num(build_hysteresis_targets(warm, K_FIXED, BUFFER_FIXED,
                                               HOLD_FIXED, delta).to_numpy()[:cut],
                      nan=0.0)
    b = np.nan_to_num(build_hysteresis_targets(bad, K_FIXED, BUFFER_FIXED,
                                               HOLD_FIXED, delta).to_numpy()[:cut],
                      nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0)), float(np.max(np.abs(a - b)))


def cmd_checks(frames, delta=None):
    d = FROZEN_DELTA if delta is None else float(delta)
    if d is None:
        d = 0.020  # mid-grid, only when nothing is frozen yet
        print(f"  (no frozen delta yet -- probing at mid-grid delta={d})")
    print(f"== correctness gates (delta={d:.3f}) ==")
    ok = True
    rows = []

    t0 = time.time()
    warm8 = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    c1 = check_causality(hyst_fn(d), warm8)
    print(f"  (a) r67_shared.check_causality (truncation probe, delta={d:.3f}):"
          f" {c1}  [{time.time() - t0:.1f}s]")
    ok &= c1
    rows.append({"check": "check_causality", "delta": d, "passed": c1})

    for dd in sorted({0.0, d, 0.080}):
        p, mx = truncation_probe(frames, dd)
        print(f"  (b) 60% truncation probe (delta={dd:.3f}): {p}  "
              f"max|diff| = {mx:.3e}")
        ok &= p
        rows.append({"check": "truncation_60pct", "delta": dd, "passed": p,
                     "max_abs_diff": mx})

    p2, mx2 = tail_perturbation_probe(frames, d)
    print(f"  (extra) tail x10 perturbation probe (delta={d:.3f}): {p2}  "
          f"max|diff| = {mx2:.3e}")
    ok &= p2
    rows.append({"check": "tail_x10", "delta": d, "passed": p2,
                 "max_abs_diff": mx2})

    # index hygiene: no bar dated 2023-01-01 or later may reach a selection
    # window cell. (W_FULL6 ends at the last bar by R-63's own convention and
    # is not a selection window; see the report's caveats and B-33.)
    for wname, window, uni in (("W_TRAIN", W_TRAIN, UNIVERSE_8),
                               ("W_VAL", W_VAL, UNIVERSE_8)):
        _, tg, warm_ok = build_cell(frames, uni, window, hyst_fn(d))
        bad = int((tg.index >= pd.Timestamp("2023-01-01", tz="UTC")).sum())
        print(f"  (c) {wname}: {len(tg):,} bars {tg.index[0]} -> {tg.index[-1]}"
              f"  bars in reserved holdout: {bad}  first_bar_warm={warm_ok}")
        ok &= (bad == 0) and warm_ok
        rows.append({"check": f"index_{wname}", "delta": d, "n_bars": len(tg),
                     "first": str(tg.index[0]), "last": str(tg.index[-1]),
                     "bars_in_holdout": bad, "first_bar_warm": warm_ok,
                     "passed": (bad == 0) and warm_ok})

    write_csv(OUT_DIR / "conservative_checks.csv", rows)
    print(f"  ALL GATES PASS: {ok}")
    return ok


# ---------------------------------------------------------------------- M1


def event_rates(frames, delta, window=W_TRAIN, universe=UNIVERSE_8):
    """R-65's event ledger for one cell, on the evaluation slice of `window`.

    Two conventions are reported because R-65's own number (0.386 forced
    exits/day) was computed the first way:
      * `*_per_day_r65`  events over the WARM-INCLUSIVE range divided by the
                         EVALUATION window's days -- R-65's `cmd_checks`
                         convention, reproduced here so the two are
                         commensurable;
      * `*_per_day_eval` events falling INSIDE the evaluation window divided
                         by the same days -- the arithmetically clean number.
    """
    warm = align_frames({t: frames[t] for t in universe}, warm_window(window))
    score = cross_sectional_score(warm)
    s = score.to_numpy(dtype=float)
    _, ev, ev_bars = hysteresis_selection(s, K_FIXED, BUFFER_FIXED, HOLD_FIXED,
                                          delta)
    lo = pd.Timestamp(window[0], tz="UTC")
    keep = score.index >= lo
    if window[1] is not None:
        keep &= score.index < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)
    days = int(keep.sum()) / BARS_PER_DAY
    out = {"delta": delta, "window": str(window), "eval_bars": int(keep.sum()),
           "eval_days": days}
    for key, arr in ev_bars.items():
        out[f"{key}_total_warm"] = int(ev[key])
        out[f"{key}_total_eval"] = int(arr[keep].sum())
        out[f"{key}_per_day_r65"] = ev[key] / days
        out[f"{key}_per_day_eval"] = float(arr[keep].sum()) / days
    return out


def cmd_m1(frames):
    """M1 for EVERY grid cell on W_TRAIN, plus the forced-exit ledger.

    No simulation runs here -- M1 is a property of the target matrix -- so this
    costs nothing against the config count and every cell is reported, not just
    the selected one. That is the whole point: if M1 passes only where D5 is
    gutted, the shared pre-registration's (F2) is confirmed.
    """
    print("== M1 (mechanism gate) on W_TRAIN / U8, ALL grid cells ==")
    _, base_tg, _ = build_cell(frames, UNIVERSE_8, W_TRAIN,
                               lambda a: r65_winner_targets(a))
    base_rate = membership_change_rate(base_tg)
    base_ev = event_rates(frames, 0.0)
    print(f"  baseline = R-65's frozen winner (k=1, buffer=0.05, hold_days=1)")
    print(f"    membership changes/day = {base_rate:.4f}   "
          f"forced exits/day = {base_ev['forced_exit_per_day_r65']:.4f} "
          f"(R-65 convention)  /  {base_ev['forced_exit_per_day_eval']:.4f} "
          f"(eval-window)   [R-65 published {R65_FORCED_EXIT_PER_DAY}]")
    print(f"  M1 bar: >= {M1_MIN_REDUCTION:.0%} fewer membership changes/day\n")
    print("   delta |  chg/d  reduction  M1 || forced_exit/d  entry/d  swap/d "
          " blk_timer/d  blk_buf/d | set(d) tenure(d) turn/d")
    rows = []
    for d in DELTA_GRID:
        _, tg, warm_ok = build_cell(frames, UNIVERSE_8, W_TRAIN, hyst_fn(d))
        if not warm_ok:
            raise RuntimeError(f"delta={d}: first evaluated bar not warm")
        m1 = m1_pass(tg, base_tg)
        ev = event_rates(frames, d)
        ts = turnover_stats(tg)
        row = {"arm": ARM, "window": "W_TRAIN", "universe": "U8", "p_delta": d,
               "p_buffer": BUFFER_FIXED, "p_hold_days": HOLD_FIXED,
               "p_k": K_FIXED,
               "m1_cand_rate_per_day": m1["cand_rate_per_day"],
               "m1_baseline_rate_per_day": m1["baseline_rate_per_day"],
               "m1_reduction": m1["reduction"], "m1_passed": m1["passed"],
               "m1_bar": M1_MIN_REDUCTION,
               "set_period_days": holding_period_days(tg),
               "tenure_days": mean_tenure_days(tg),
               "turnover_per_day": ts["turnover_per_day"],
               "mean_notional": mean_total_notional(tg),
               "r65_published_forced_exit_per_day": R65_FORCED_EXIT_PER_DAY,
               **{k2: v for k2, v in ev.items() if k2 != "window"},
               **raw_turnover(tg)}
        rows.append(row)
        print(f"  {d:6.3f} | {m1['cand_rate_per_day']:6.3f}  "
              f"{m1['reduction']:+8.2%}  {'PASS' if m1['passed'] else 'fail'} ||"
              f" {ev['forced_exit_per_day_r65']:12.3f} "
              f"{ev['entry_per_day_r65']:8.3f} {ev['swap_per_day_r65']:7.3f} "
              f"{ev['blocked_by_timer_per_day_r65']:11.3f} "
              f"{ev['blocked_by_buffer_per_day_r65']:10.3f} |"
              f" {row['set_period_days']:6.3f} {row['tenure_days']:8.3f} "
              f"{ts['turnover_per_day']:6.3f}")

    write_csv(OUT_DIR / "conservative_m1.csv", rows)
    return rows


# ------------------------------------------------------------------ frontier


def cmd_frontier(frames, windows=("W_TRAIN", "W_VAL")):
    """The round's deliverable: every grid cell, both fee levels, on the two
    SELECTION windows. No decision-window read happens here."""
    print(f"== FRONTIER: {len(DELTA_GRID)} delta cells + R-63 reference, "
          f"k={K_FIXED} buffer={BUFFER_FIXED} hold_days={HOLD_FIXED}, U8, "
          f"vs VOLMATCH_HOLD ==")
    wmap = {"W_TRAIN": W_TRAIN, "W_VAL": W_VAL}
    rows = []
    for wname in windows:
        window = wmap[wname]
        print(f"  -- {wname} --")
        t0 = time.time()
        aligned, tg63, warm_ok = build_cell(frames, UNIVERSE_8, window,
                                            r63_fn(K_FIXED))
        if not warm_ok:
            raise RuntimeError(f"{wname}: first evaluated bar not warm")
        row, _ = measure_pair(tg63, aligned, UNIVERSE_8, wname, "U8",
                              {"delta": float("nan"), "buffer": float("nan"),
                               "hold_days": 0, "k": K_FIXED},
                              arm="r63_reference")
        row["config_kind"] = "reference"
        rows.append(row)
        print("  [R-63 reference k=1, no buffer, no hysteresis]")
        print(fmt_front(row))

        for d in DELTA_GRID:
            aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, window,
                                              hyst_fn(d))
            if not warm_ok:
                raise RuntimeError(f"{wname} delta={d}: first bar not warm")
            row, _ = measure_pair(tg, aligned, UNIVERSE_8, wname, "U8",
                                  {"delta": d, "buffer": BUFFER_FIXED,
                                   "hold_days": HOLD_FIXED, "k": K_FIXED})
            row["config_kind"] = "grid"
            rows.append(row)
            print(f"  [delta={d:.3f}"
                  f"{'  (= R-65 frozen winner)' if d == 0.0 else ''}]")
            print(fmt_front(row))
        print(f"  {wname} done in {time.time() - t0:.0f}s "
              f"(config_count={config_count()})")

    write_csv(OUT_DIR / "conservative_frontier.csv", rows)
    return rows


def cmd_select(frames=None, rows=None):
    """Report the selection, both orderings, the neighbourhood and the
    cross-window rank correlation. Reads the frontier CSV; runs nothing."""
    df = pd.read_csv(OUT_DIR / "conservative_frontier.csv")
    df["cost"] = df["gross_growth_diff"] - df["net_growth_diff"]

    out = []
    for wname in ("W_TRAIN", "W_VAL"):
        ref = df[(df["window"] == wname) & (df["config_kind"] == "reference")]
        sub = df[(df["window"] == wname) & (df["config_kind"] == "grid")] \
            .sort_values("net_growth_diff", ascending=False)
        print(f"== {wname}: value and cost vs VOLMATCH_HOLD "
              f"(ordered by net growth diff, best first) ==")
        print("    delta | turn/d  ten(d) |    GROSS |    COST |      NET "
              "| net 95% interval        | dd_diff")
        for _, r in pd.concat([ref, sub]).iterrows():
            tag = ("R-63 ref" if r["config_kind"] == "reference"
                   else f"   {r['p_delta']:.3f}")
            print(f"  {tag} | {r['turnover_per_day']:6.3f} "
                  f"{r['tenure_days']:6.2f}"
                  f" | {r['gross_growth_diff']:+8.3f} | {r['cost']:7.3f} |"
                  f" {r['net_growth_diff']:+8.3f} | "
                  f"[{r['net_growth_lo']:+7.3f},{r['net_growth_hi']:+7.3f}] |"
                  f" {r['net_dd_diff']:+7.2f}")
        n = len(sub)
        n_pos = int((sub["net_growth_diff"] > 0).sum())
        n_sig = int(((sub["net_growth_diff"] > 0)
                     & (sub["net_growth_lo"] > 0)).sum())
        n_d5 = int((sub["gross_growth_diff"] >= D5_BAR).sum())
        print(f"  -> best net {sub['net_growth_diff'].max():+.4f}; net>0: "
              f"{n_pos}/{n}; net>0 with interval excluding zero: {n_sig}/{n}; "
              f"clearing the D5 bar (+{D5_BAR:.3f} gross): {n_d5}/{n}")
        out.append(sub)

    tr, va = out
    merged = tr[["p_delta", "net_growth_diff"]].merge(
        va[["p_delta", "net_growth_diff"]], on="p_delta",
        suffixes=("_train", "_val"))
    # Spearman by hand: scipy is not installed; rank-then-Pearson is identical.
    rho = merged["net_growth_diff_train"].rank().corr(
        merged["net_growth_diff_val"].rank())
    print(f"\n  Spearman rank correlation W_TRAIN vs W_VAL net growth over the "
          f"{len(merged)} cells: {rho:+.3f}")

    best = va.iloc[0]
    bd = float(best["p_delta"])
    print(f"  W_VAL WINNER (declared criterion): delta={bd:.3f}  "
          f"net={best['net_growth_diff']:+.4f}  "
          f"gross={best['gross_growth_diff']:+.4f}  "
          f"dd={best['net_dd_diff']:+.3f}")
    tr_rank = list(tr["p_delta"]).index(bd) + 1
    print(f"  that cell's W_TRAIN rank: {tr_rank} of {len(tr)}")

    grid = list(DELTA_GRID)
    i = grid.index(bd)
    for j, lab in ((i - 1, "lower"), (i + 1, "upper")):
        if 0 <= j < len(grid):
            nb = va[va["p_delta"] == grid[j]].iloc[0]
            print(f"  neighbour ({lab}) delta={grid[j]:.3f}: W_VAL net "
                  f"{nb['net_growth_diff']:+.4f} -> "
                  f"{'BETTER' if nb['net_growth_diff'] > best['net_growth_diff'] else 'worse'}"
                  f" than the winner")
        else:
            print(f"  neighbour ({lab}): none -- the winner is a GRID CORNER")
    return best, rho


# ------------------------------------------------------------------ run


def _frozen(delta=None):
    d = FROZEN_DELTA if delta is None else delta
    if d is None:
        raise SystemExit("configuration is not frozen yet: run `frontier`, "
                         "then `select`, then set FROZEN_DELTA")
    return float(d)


def cmd_run(frames, delta=None):
    d = _frozen(delta)
    print(f"== D-CELLS: frozen delta={d:.3f} (k={K_FIXED}, "
          f"buffer={BUFFER_FIXED}, hold_days={HOLD_FIXED}) ==")
    rows = []

    # ---- D1 / D2 / D5 : W_FULL6, U6, vs VOLMATCH_HOLD -----------------
    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6,
                                           hyst_fn(d))
    if not warm_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")
    print(f"  W_FULL6 bars {len(targets):,}  {targets.index[0]} -> "
          f"{targets.index[-1]}  first_bar_warm={warm_ok}")

    row, st = measure_pair(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                           {"delta": d, "buffer": BUFFER_FIXED,
                            "hold_days": HOLD_FIXED, "k": K_FIXED})
    row["config_kind"] = "decision"
    d1 = d1_pass(row) and st["matched_net"]
    d2 = d2_pass(row) and st["matched_net"]
    d5 = d5_pass(row) and st["matched_gross"]
    row.update({"d1_pass": d1, "d2_pass": d2, "d5_pass": d5, "d5_bar": D5_BAR,
                "note": "D1/D2/D5 primary vs VOLMATCH_HOLD"})
    rows.append(row)
    print(fmt_front(row))
    print(f"    cand_vol {row['cand_vol_net']:.3f} vs volmatch_vol "
          f"{row['volmatch_vol_net']:.3f} at c={row['volmatch_c_net']:.3f}  "
          f"matched={st['matched_net']} (shared flag "
          f"{row['volmatch_shared_flag_net']})")
    print(f"    net dd_diff {row['net_dd_diff']:+.3f}pp "
          f"[{row['net_dd_lo']:+.3f}, {row['net_dd_hi']:+.3f}]")
    print(f"    D1 PASS={d1}  D2 PASS={d2}  D5 PASS={d5} (bar {D5_BAR:+.3f}, "
          f"gross {row['gross_growth_diff']:+.4f})")

    # ---- M1 at the frozen delta, on W_TRAIN --------------------------
    _, tg_tr, _ = build_cell(frames, UNIVERSE_8, W_TRAIN, hyst_fn(d))
    _, base_tr, _ = build_cell(frames, UNIVERSE_8, W_TRAIN,
                               lambda a: r65_winner_targets(a))
    m1 = m1_pass(tg_tr, base_tr)
    ev = event_rates(frames, d)
    rows.append({"arm": ARM, "window": "W_TRAIN", "universe": "U8",
                 "config_kind": "decision", "bench": "R65_WINNER",
                 "p_delta": d, "p_buffer": BUFFER_FIXED,
                 "p_hold_days": HOLD_FIXED, "p_k": K_FIXED,
                 "m1_cand_rate_per_day": m1["cand_rate_per_day"],
                 "m1_baseline_rate_per_day": m1["baseline_rate_per_day"],
                 "m1_reduction": m1["reduction"], "m1_passed": m1["passed"],
                 "m1_bar": M1_MIN_REDUCTION,
                 **{k2: v for k2, v in ev.items() if k2 != "window"},
                 "note": "M1 mechanism gate at the frozen configuration"})
    print(f"  [M1 W_TRAIN U8] cand {m1['cand_rate_per_day']:.4f}/day vs "
          f"baseline {m1['baseline_rate_per_day']:.4f}/day -> reduction "
          f"{m1['reduction']:+.2%}  bar {M1_MIN_REDUCTION:.0%}  "
          f"M1 PASS={m1['passed']}")

    # ---- continuity: MATCHED_HOLD / EW_HOLD / BTC_HOLD ----------------
    c = mean_total_notional(targets)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                            aligned, SPOT_BASE)
    r_mh = compare(st["cand_net"], mh)
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "MATCHED_HOLD", "config_kind": "context",
                 "p_delta": d, "p_buffer": BUFFER_FIXED,
                 "p_hold_days": HOLD_FIXED, "p_k": K_FIXED,
                 "mean_notional": c, "matched_hold_vol": realized_vol(mh),
                 "cand_vol_net": row["cand_vol_net"],
                 "net_growth_diff": r_mh["growth_diff"],
                 "net_growth_lo": r_mh["growth_lo"],
                 "net_growth_hi": r_mh["growth_hi"],
                 "net_dd_diff": r_mh["dd_diff"], "net_dd_lo": r_mh["dd_lo"],
                 "net_dd_hi": r_mh["dd_hi"], "cand_final": r_mh["cand_final"],
                 "bench_final": r_mh["bench_final"], "cand_dd": r_mh["cand_dd"],
                 "bench_dd": r_mh["bench_dd"], "n_days": r_mh["n_days"],
                 "note": "continuity: R-63's benchmark (notional-matched)"})
    print(f"  [continuity vs MATCHED_HOLD c={c:.3f}] growth "
          f"{r_mh['growth_diff']:+.4f} [{r_mh['growth_lo']:+.3f}, "
          f"{r_mh['growth_hi']:+.3f}]  dd {r_mh['dd_diff']:+.2f}pp  "
          f"(matched_hold vol {realized_vol(mh):.3f} vs cand "
          f"{row['cand_vol_net']:.3f})")

    ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)
    r_ew = compare(st["cand_net"], ew)
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "EW_HOLD", "config_kind": "context", "p_delta": d,
                 "net_growth_diff": r_ew["growth_diff"],
                 "net_growth_lo": r_ew["growth_lo"],
                 "net_growth_hi": r_ew["growth_hi"],
                 "net_dd_diff": r_ew["dd_diff"], "cand_final": r_ew["cand_final"],
                 "bench_final": r_ew["bench_final"], "cand_dd": r_ew["cand_dd"],
                 "bench_dd": r_ew["bench_dd"], "n_days": r_ew["n_days"],
                 "note": "context: vs EW_HOLD @0.10%"})
    print(f"  [context vs EW_HOLD @0.10%] {r_ew['cand_final']:,.2f} vs "
          f"{r_ew['bench_final']:,.2f}  growth {r_ew['growth_diff']:+.4f}")

    btc = frames["BTC"]
    btc_on = btc.reindex(btc.index.union(targets.index)).ffill().reindex(targets.index)
    btc_eq = static_hold_equity({"BTC": btc_on}, ["BTC"], SPOT_BASE)
    r_btc = compare(st["cand_net"], btc_eq)
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "BTC_HOLD", "config_kind": "context", "p_delta": d,
                 "net_growth_diff": r_btc["growth_diff"],
                 "net_growth_lo": r_btc["growth_lo"],
                 "net_growth_hi": r_btc["growth_hi"],
                 "net_dd_diff": r_btc["dd_diff"],
                 "cand_final": r_btc["cand_final"],
                 "bench_final": r_btc["bench_final"],
                 "cand_dd": r_btc["cand_dd"], "bench_dd": r_btc["bench_dd"],
                 "n_days": r_btc["n_days"],
                 "note": "context: vs BTC buy-and-hold (ffilled onto U6 grid)"})
    print(f"  [context vs BTC_HOLD] {r_btc['cand_final']:,.2f} vs "
          f"{r_btc['bench_final']:,.2f}  growth {r_btc['growth_diff']:+.4f}")

    # ---- D4 : W_FULL6, 0.40% SPOT_REAL, vs EW_HOLD --------------------
    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    r40 = compare(cand40, ew40)
    d4 = bool(r40["cand_final"] > r40["bench_final"])
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "EW_HOLD", "config_kind": "decision", "p_delta": d,
                 "fee": 0.004, "d4_pass": d4,
                 "net_growth_diff": r40["growth_diff"],
                 "net_growth_lo": r40["growth_lo"],
                 "net_growth_hi": r40["growth_hi"],
                 "net_dd_diff": r40["dd_diff"], "cand_final": r40["cand_final"],
                 "bench_final": r40["bench_final"], "cand_dd": r40["cand_dd"],
                 "bench_dd": r40["bench_dd"], "n_days": r40["n_days"],
                 "note": "D4 cost tier 0.40% (SPOT_REAL) vs EW_HOLD"})
    print(f"  [D4 @0.40%] cand {r40['cand_final']:,.2f} vs EW_HOLD "
          f"{r40['bench_final']:,.2f} -> D4 PASS={d4}")

    # ---- D3 : W_VAL, U8, 0.10% ---------------------------------------
    aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL, hyst_fn(d))
    if not warm3:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    row3, st3 = measure_pair(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                             {"delta": d, "buffer": BUFFER_FIXED,
                              "hold_days": HOLD_FIXED, "k": K_FIXED})
    row3["config_kind"] = "decision"
    d3 = d3_pass(row3) and st3["matched_net"]
    row3["d3_pass"] = d3
    row3["note"] = "D3 inner-validation vs VOLMATCH_HOLD"
    rows.append(row3)
    print("  [D3 W_VAL U8]")
    print(fmt_front(row3))
    print(f"    D3 PASS={d3}  (growth {row3['net_growth_diff']:+.4f}, dd "
          f"{row3['net_dd_diff']:+.3f}pp, matched={st3['matched_net']})")

    # continuity: MATCHED_HOLD on the W_VAL cell too
    c3 = mean_total_notional(targets3)
    mh3 = simulate_portfolio(matched_hold_targets(targets3.index, UNIVERSE_8, c3),
                             aligned3, SPOT_BASE)
    r_mh3 = compare(st3["cand_net"], mh3)
    rows.append({"arm": ARM, "window": "W_VAL", "universe": "U8",
                 "bench": "MATCHED_HOLD", "config_kind": "context",
                 "p_delta": d, "mean_notional": c3,
                 "matched_hold_vol": realized_vol(mh3),
                 "net_growth_diff": r_mh3["growth_diff"],
                 "net_growth_lo": r_mh3["growth_lo"],
                 "net_growth_hi": r_mh3["growth_hi"],
                 "net_dd_diff": r_mh3["dd_diff"],
                 "cand_final": r_mh3["cand_final"],
                 "bench_final": r_mh3["bench_final"],
                 "cand_dd": r_mh3["cand_dd"], "bench_dd": r_mh3["bench_dd"],
                 "n_days": r_mh3["n_days"],
                 "note": "continuity: MATCHED_HOLD on the D3 cell"})
    print(f"  [continuity vs MATCHED_HOLD on W_VAL c={c3:.3f}] growth "
          f"{r_mh3['growth_diff']:+.4f}  dd {r_mh3['dd_diff']:+.2f}pp")

    write_csv(OUT_DIR / "conservative_cells.csv", rows)
    return {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5,
            "m1": bool(m1["passed"]), "row": row, "targets": targets,
            "aligned": aligned, "vm": st["vm_net"],
            "real": row["net_growth_diff"], "matched_net": st["matched_net"],
            "matched_gross": st["matched_gross"], "matched_net_d3":
            st3["matched_net"], "delta": d}


# ------------------------------------------------------------------ scramble


def cmd_scramble(frames, delta=None, state=None):
    d = _frozen(delta)
    print(f"== FALSIFICATION: cross-section scramble, seeds 0..9, D1 cell, "
          f"delta={d:.3f} ==")
    if state is None:
        aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6,
                                               hyst_fn(d))
        if not warm_ok:
            raise RuntimeError("W_FULL6 first evaluated bar not warm")
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        vm, _, _, ok, _ = volmatch(cand, aligned, UNIVERSE_6, SPOT_BASE)
        real = compare(cand, vm)["growth_diff"]
    else:
        aligned, targets, vm = state["aligned"], state["targets"], state["vm"]
        real = state["real"]

    cand_turn = turnover_stats(targets)["turnover_per_day"]
    rows, diffs = [], []

    # ---- FORM A: the pre-registered `scramble_targets` (redraw per change)
    for seed in SCRAMBLE_SEEDS:
        stg = scramble_targets(targets, seed)
        eq = simulate_portfolio(stg, aligned, SPOT_BASE)
        r = compare(eq, vm)
        diffs.append(r["growth_diff"])
        rows.append({"arm": f"{ARM}_scrambled", "form": "preregistered_redraw",
                     "seed": seed, "window": "W_FULL6", "universe": "U6",
                     "bench": "VOLMATCH_HOLD", "fee": 0.001, "p_delta": d,
                     "mean_notional": mean_total_notional(stg),
                     "turnover_per_day": turnover_stats(stg)["turnover_per_day"],
                     **{key: r[key] for key in
                        ("cand_final", "bench_final", "cand_dd", "bench_dd",
                         "growth_diff", "growth_lo", "growth_hi", "dd_diff",
                         "dd_lo", "dd_hi", "n_days")}})
        print(f"  [A] seed {seed}: growth_diff {r['growth_diff']:+.4f}  final "
              f"{r['cand_final']:>12,.4f}  turnover "
              f"{rows[-1]['turnover_per_day']:7.3f}/d")

    p90 = float(np.percentile(diffs, 90))
    survived = bool(real > p90)
    mean_turn_a = float(np.mean([r["turnover_per_day"] for r in rows]))
    rows.append({"arm": ARM, "form": "preregistered_redraw", "seed": -1,
                 "window": "W_FULL6", "universe": "U6",
                 "bench": "VOLMATCH_HOLD", "fee": 0.001, "p_delta": d,
                 "growth_diff": real, "scramble_p90": p90,
                 "scramble_survived": survived,
                 "turnover_per_day": cand_turn,
                 "scramble_mean_turnover_per_day": mean_turn_a,
                 "mean_notional": mean_total_notional(targets),
                 "n_better": int(sum(x >= real for x in diffs))})
    print(f"  [A] real {real:+.4f} vs p90 {p90:+.4f} -> SURVIVED={survived}"
          f"  ({sum(x >= real for x in diffs)} of 10 better)")
    print(f"      candidate turnover {cand_turn:.3f}/d vs scramble mean "
          f"{mean_turn_a:.3f}/d  (ratio {mean_turn_a / max(cand_turn, 1e-12):.2f}x)")

    # ---- FORM B: R-65's fixed-permutation control (turnover-preserving)
    # One fixed asset relabeling held for the whole run. A column permutation
    # is an L1 isometry, so turnover and total notional are preserved
    # bar-for-bar; only the asset->signal assignment is destroyed.
    print("  -- [B] fixed-permutation control (R-65's correction) --")
    fdiffs, fixed_rows = [], []
    cols = list(targets.columns)
    for seed in SCRAMBLE_SEEDS:
        perm = np.random.default_rng(1000 + seed).permutation(len(cols))
        stg = pd.DataFrame(targets.to_numpy()[:, perm], index=targets.index,
                           columns=cols)
        eq = simulate_portfolio(stg, aligned, SPOT_BASE)
        r = compare(eq, vm)
        fdiffs.append(r["growth_diff"])
        ident = bool(np.array_equal(perm, np.arange(len(cols))))
        fixed_rows.append({"arm": f"{ARM}_fixedperm", "form": "fixed_permutation",
                           "seed": seed, "window": "W_FULL6", "universe": "U6",
                           "bench": "VOLMATCH_HOLD", "fee": 0.001, "p_delta": d,
                           "perm": ";".join(map(str, perm)),
                           "identity_perm": ident,
                           "mean_notional": mean_total_notional(stg),
                           "turnover_per_day":
                               turnover_stats(stg)["turnover_per_day"],
                           **{key: r[key] for key in
                              ("cand_final", "bench_final", "cand_dd",
                               "bench_dd", "growth_diff", "growth_lo",
                               "growth_hi", "dd_diff", "dd_lo", "dd_hi",
                               "n_days")}})
        print(f"  [B] seed {seed} perm {perm}: growth_diff "
              f"{r['growth_diff']:+.4f}  turnover "
              f"{fixed_rows[-1]['turnover_per_day']:7.3f}/d"
              f"{'  (IDENTITY)' if ident else ''}")
    rows.extend(fixed_rows)
    fp90 = float(np.percentile(fdiffs, 90))
    fsurv = bool(real > fp90)
    mean_turn_b = float(np.mean([r["turnover_per_day"] for r in fixed_rows]))
    rows.append({"arm": ARM, "form": "fixed_permutation", "seed": -2,
                 "window": "W_FULL6", "universe": "U6",
                 "bench": "VOLMATCH_HOLD", "fee": 0.001, "p_delta": d,
                 "growth_diff": real, "scramble_p90": fp90,
                 "scramble_survived": fsurv, "turnover_per_day": cand_turn,
                 "scramble_mean_turnover_per_day": mean_turn_b,
                 "n_better": int(sum(x >= real for x in fdiffs))})
    print(f"  [B] real {real:+.4f} vs fixed-perm p90 {fp90:+.4f} -> "
          f"SURVIVED={fsurv}  ({sum(x >= real for x in fdiffs)} of 10 better)")
    print(f"      candidate turnover {cand_turn:.3f}/d vs fixed-perm mean "
          f"{mean_turn_b:.3f}/d  (ratio "
          f"{mean_turn_b / max(cand_turn, 1e-12):.2f}x)")

    write_csv(OUT_DIR / "conservative_scramble.csv", rows)
    return {"preregistered": survived, "fixed_perm": fsurv}


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["identity", "checks", "m1", "frontier",
                                    "select", "run", "scramble", "all"])
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--windows", default="W_TRAIN,W_VAL")
    args = ap.parse_args()

    if args.cmd == "select":
        cmd_select()
        return

    frames = load_universe(UNIVERSE_8)

    if args.cmd == "identity":
        cmd_identity(frames)
    elif args.cmd == "checks":
        cmd_checks(frames, args.delta)
    elif args.cmd == "m1":
        cmd_m1(frames)
    elif args.cmd == "frontier":
        cmd_frontier(frames, tuple(args.windows.split(",")))
    elif args.cmd == "run":
        cmd_run(frames, args.delta)
    elif args.cmd == "scramble":
        cmd_scramble(frames, args.delta)
    else:
        cmd_identity(frames)
        cmd_checks(frames, args.delta)
        cmd_m1(frames)
        cmd_frontier(frames)
        cmd_select()
        st = cmd_run(frames, args.delta)
        sc = cmd_scramble(frames, st["delta"], st)
        fw = further_work(st["m1"], st["d1"], st["d2"], st["d3"], st["d5"],
                          sc["fixed_perm"])
        print(f"\n== further_work(m1={st['m1']}, d1={st['d1']}, d2={st['d2']}, "
              f"d3={st['d3']}, d5={st['d5']}, scramble={sc['fixed_perm']}) "
              f"= {fw} ==")
        print("  -> STOP. Report to the operator; the holdout read is theirs."
              if fw else "  -> DONE. The reserved holdout is NOT read.")

    log_configs(args.cmd)


if __name__ == "__main__":
    main()
