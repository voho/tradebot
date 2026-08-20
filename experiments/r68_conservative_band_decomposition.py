"""R-68 CONSERVATIVE branch -- decompose R-67's band; extend the grid to its cap.

R-67's conservative arm bought a 30-fold cut in the forced-exit channel with
ONE new parameter, and left two questions open that `experiments/r68_shared.py`
(this round's frozen pre-registration, backlog item **B-34**) names as the
whole reason this round exists:

1. **Which half of the band does the work?** R-67's rule moves the entry
   threshold up to `+delta` and the exit threshold down to `-delta` in a single
   step, so it cannot attribute its result to either. That is the identical
   confound R-64 and R-66 each died of on the single-asset axis.
2. **Does the curve turn over past R-67's edge?** R-67's winner, `delta=0.080`,
   is the top corner of a grid that stopped there. Whether the frontier keeps
   improving, flattens or degrades beyond it is simply untested.

This branch answers both, on one extended grid, and reports every cell of every
sub-arm whatever the verdict.

=====================================================================
THE RULE (frozen before any number was read)
=====================================================================

Start from R-65's frozen conservative winner byte-for-byte -- `k=1`,
`buffer=0.05`, `hold_days=1`, R-63's composite cross-sectional score, R-63's
conditional volatility scale, R-63's 0.10 deadband on desired TOTAL notional,
equal weighting among held slots, long-only spot, unlevered -- and change ONLY
the eligibility test. R-67 couples two thresholds into one parameter:

    enter_eligible = isfinite(s) & (s >  +delta)      # R-67
    hold_eligible  = isfinite(s) & (s >  -delta)      # R-67

This branch DECOUPLES them into two free thresholds:

    enter_eligible = isfinite(s) & (s >  +delta_in)   # a new entrant must clear
    hold_eligible  = isfinite(s) & (s >  -delta_out)  # an incumbent is kept

and sweeps THREE named sub-arms of that plane:

    EXIT_ONLY    delta_in = 0.0,  delta_out = d    hold through the crossing;
                                                   enter exactly as R-63 does
    ENTRY_ONLY   delta_in = d,    delta_out = 0.0  demand a stronger entrant;
                                                   exit exactly as R-63 does
    COUPLED      delta_in = delta_out = d          R-67's own arm, the diagonal

Both predicates use a STRICT `>` for the same reason R-67 chose it: at d = 0
all three sub-arms collapse to the identical `s > 0.0` expression and the
target matrix must be bit-identical to R-65's frozen winner. Boundary ties
(`s` landing exactly on +d or -d) are COUNTED and reported rather than assumed
absent.

The score is in raw units of `close/anchor - 1`; nothing here is normalized,
standardized or ranked over time, so no whole-series statistic exists to leak.
`delta_in` and `delta_out` are CONSTANTS -- the selection is a forward loop
whose state at bar `i` depends on rows <= i and nothing else -- and that is
verified by a 60% truncation probe and a tail-x10 perturbation probe rather
than asserted by this docstring.

GRID: `r68_shared.DELTA_GRID_EXT`, frozen in the shared pre-registration before
either branch was written:

    (0.000, 0.005, 0.010, 0.020, 0.040, 0.080,   <- R-67's grid, DELTA_GRID_R67
     0.120, 0.160, 0.220, 0.280, 0.350)          <- the extension

11 values x 3 sub-arms = 33 grid cells per window. The cap is de Lataillade &
Chaouki (2020), arXiv:2003.04646, Eq. (11) -- the optimal tolerance around zero
saturates at ~1.6 sigma_signal; with the R-63 score's pooled W_TRAIN sigma of
0.2295 that is 0.367, and the grid stops just under it at 0.350. The cap is
computed from a whole-window standard deviation and is used ONLY to bound a
grid of constants; it never enters a per-bar decision.

Note by construction: `d = 0.000` is the SAME configuration in all three
sub-arms (and is R-65's frozen winner), so the 33 grid cells per window are 31
distinct configurations. Both counts are reported.

=====================================================================
SELECTION CRITERION -- DECLARED HERE BEFORE THE SWEEP WAS RUN
=====================================================================

Swept on W_TRAIN, selected on W_VAL. EXACTLY R-65's and R-67's criterion, not
a new one:

    The frozen configuration is the ONE (sub-arm, d) cell with the highest
    **W_VAL net growth difference versus VOLMATCH_HOLD at the 0.10% tier** --
    the D1 decision statistic evaluated on the selection window.

    Tie-break 1: the more negative W_VAL net drawdown difference.
    Tie-break 2 (only reachable at d = 0.000, where the three sub-arms are
    literally the same configuration): the sub-arm earliest in the fixed
    order EXIT_ONLY, ENTRY_ONLY, COUPLED.

**No filter on the gross column is applied at selection time.** The gross
column is D5's diagnostic and selecting on it would be selecting on the
falsification test. This criterion is honoured as written; if it selects a cell
that then fails D5 or M1', that is a reported finding, not a reason to
reselect. The grid is NOT extended a second time whatever the winner's
position: `r68_shared`'s (F1) pre-registers a second corner as an artifact of
the selection criterion, not a discovery.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- the shared pre-registration's F1, F2, F3
=====================================================================

**(F1)** B-35 measured the fee saving asymptoting (+1.044 / +1.062 / +1.074 at
delta 0.080 / 0.120 / 0.160) while the grace span keeps growing (0.17 -> 0.29
-> 0.39 days). The honest expectation is therefore a FLAT extended frontier. A
materially better cell at 0.350 is more likely a second grid-edge artifact than
a discovery, and is reported as one.

**(F2)** As `d -> large` both eligibility tests stop firing and the arm
converges on a concentrated buy-and-hold of whatever was strongest in 2020-04.
D5, at its CORRECTED +0.342 bar, is what catches that, and at the cap it should.

**(F3)** Neither half alone may reproduce the coupled result -- the effect may
require both, and the round then learns only that the confound cannot be
separated on this data. That is a real outcome, it is pre-registered as an
acceptable answer, and it is reported as such rather than dressed up.

Windows, universes, costs, benchmarks, D1/D2/D3/D4/D5, M1', the fixed-
permutation scramble control and the further-work bar all live in the frozen
pre-registration in `experiments/r68_shared.py` (and through it in
`r67_shared.py` / `r65_shared.py` / `r63_shared.py`). This file implements a
candidate and measures it; it does not define, relax or edit a rule.
**W_HOLD is never imported, sliced or referenced.**

Run as:
    .venv/bin/python experiments/r68_conservative_band_decomposition.py identity
    .venv/bin/python experiments/r68_conservative_band_decomposition.py checks
    .venv/bin/python experiments/r68_conservative_band_decomposition.py m1
    .venv/bin/python experiments/r68_conservative_band_decomposition.py frontier
    .venv/bin/python experiments/r68_conservative_band_decomposition.py select
    .venv/bin/python experiments/r68_conservative_band_decomposition.py run
    .venv/bin/python experiments/r68_conservative_band_decomposition.py scramble
    .venv/bin/python experiments/r68_conservative_band_decomposition.py all
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

from experiments.r68_shared import (  # noqa: E402
    BARS_PER_DAY,
    D5_BAR_R68,
    DEADBAND,
    DELTA_GRID_EXT,
    DELTA_GRID_R67,
    DLC_SATURATION,
    M1_MIN_REDUCTION,
    MEMBERSHIP_WEIGHT_FLOOR,
    OUT_DIR,
    R65_BUFFER,
    R65_HOLD_DAYS,
    R65_K,
    R67_DELTA_WINNER,
    SCRAMBLE_SEEDS,
    SIGMA_SCORE_W_TRAIN,
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
    membership_change_rate_thresholded,
    r63_baseline_targets,
    r65_winner_targets,
    realized_vol,
    scramble_fixed_perm,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)
# Read-only imports of two R-67 constants/diagnostics, so the M1 column this
# branch prints is commensurable with R-67's published one. `r67_shared.py` is
# NOT edited, and R-67's `hysteresis_selection` is deliberately NOT imported --
# its loop is copied and generalised below, per this round's instructions.
from experiments.r67_shared import (  # noqa: E402
    R65_FORCED_EXIT_PER_DAY,
    membership_change_rate as membership_change_rate_r67,
)

ARM = "band_decomposition"

# R-65's frozen winner, inherited. NOT re-selected in this round.
K_FIXED = R65_K            # 1
BUFFER_FIXED = R65_BUFFER  # 0.05
HOLD_FIXED = R65_HOLD_DAYS  # 1

# The three named sub-arms: d -> (delta_in, delta_out). Order is fixed here and
# is the criterion's tie-break 2.
SUB_ARMS: dict[str, callable] = {
    "EXIT_ONLY": lambda d: (0.0, float(d)),
    "ENTRY_ONLY": lambda d: (float(d), 0.0),
    "COUPLED": lambda d: (float(d), float(d)),
}
SUB_ARM_ORDER = ("EXIT_ONLY", "ENTRY_ONLY", "COUPLED")

# ---------------------------------------------------------------------------
# FROZEN CONFIGURATION.
#
# To be set from `conservative_frontier.csv` by `cmd_select`, on **W_VAL only**,
# on the criterion declared in this module's docstring above, which was fixed
# and committed before the sweep was run -- and set BEFORE any D-cell is
# computed. The W_VAL ordering as measured, the W_TRAIN ordering, the winner's
# immediate neighbourhood and the per-sub-arm cross-window rank correlation are
# recorded here at freezing time (and printed by `cmd_select`).
#
# ---- MEASURED AND RECORDED AT FREEZING TIME, from `cmd_select`, AFTER the
# ---- sweep and BEFORE any D-cell was computed. (This block was left empty
# ---- until the sweep had run; nothing was written into it in advance.)
#
# W_VAL ordering AS MEASURED (net growth diff vs VOLMATCH_HOLD @0.10%, best
# first; ties broken by the more negative net dd, as declared). Full table in
# reports/r68_band/conservative_frontier.csv:
#   1. ENTRY_ONLY d=0.080  net +0.8480 [-0.132,+1.843] gross +0.8708 dd -21.31 <-SELECTED
#   2. ENTRY_ONLY d=0.120  net +0.8040 [-0.148,+1.781] gross +0.8210 dd -17.50
#   3. ENTRY_ONLY d=0.160  net +0.7084 [-0.238,+1.689] gross +0.7217 dd -14.62
#   4. ENTRY_ONLY d=0.350  net +0.6790 [-0.047,+1.441] gross +0.6817 dd -17.28
#   5. COUPLED    d=0.160  net +0.5893 [-0.393,+1.601] gross +0.6013 dd -21.03
#   6. COUPLED    d=0.120  net +0.5745 [-0.489,+1.658] gross +0.5893 dd -14.64
#   7. EXIT_ONLY  d=0.160  net +0.5501 [-0.519,+1.499] gross +0.5865 dd  -1.25
#   8. ENTRY_ONLY d=0.280  net +0.5421 [-0.436,+1.470] gross +0.5551 dd  -3.27
#   9. COUPLED    d=0.080  net +0.5194 [-0.516,+1.566] gross +0.5436 dd -16.46
#                          (= R-67's published cell, reproduced to 4 dp)
#  10. COUPLED    d=0.280  net +0.4811   11. EXIT_ONLY  d=0.020  net +0.4697
#  12. ENTRY_ONLY d=0.220  net +0.4687   13. ENTRY_ONLY d=0.040  net +0.4563
#  14. EXIT_ONLY  d=0.010  net +0.4423   15. EXIT_ONLY  d=0.220  net +0.4262
#  16. COUPLED    d=0.020  net +0.4220   17. EXIT_ONLY  d=0.080  net +0.4146
#  18. EXIT_ONLY  d=0.040  net +0.4006   19. COUPLED    d=0.010  net +0.3668
#  20. EXIT_ONLY  d=0.005  net +0.3599   21. EXIT_ONLY  d=0.120  net +0.3551
#  22. COUPLED    d=0.040  net +0.3514   23. ENTRY_ONLY d=0.010  net +0.3453
#  24. ENTRY_ONLY d=0.020  net +0.3137   25. COUPLED    d=0.005  net +0.2899
#  26. ENTRY_ONLY d=0.005  net +0.2698   27. EXIT_ONLY  d=0.280  net +0.2600
#  28. COUPLED    d=0.220  net +0.2118
#  29-31. EXIT_ONLY / ENTRY_ONLY / COUPLED d=0.000  net +0.1425 (one identical
#         configuration listed three times = R-65's frozen winner, reproduced)
#  32. COUPLED    d=0.350  net +0.1403   33. EXIT_ONLY  d=0.350  net +0.0846
# All 33 W_VAL cells are net-positive; NOT ONE has an interval excluding zero.
# 29 of 33 clear the corrected D5 bar (+0.342 gross) on W_VAL.
#
# W_TRAIN ordering (net growth diff, best first). EVERY cell is net-NEGATIVE
# there, and 0 of 33 clear the D5 bar on W_TRAIN (as was already true of R-63's
# own arm, which scores -2.097 gross there; D5 is a statement about W_FULL6):
#   1. ENTRY_ONLY d=0.120  -0.5881    2. ENTRY_ONLY d=0.160  -0.6682
#   3. ENTRY_ONLY d=0.220  -0.7610    4. COUPLED    d=0.160  -0.8219
#   5. COUPLED    d=0.220  -0.8371    6. COUPLED    d=0.120  -0.8698
#   7. ENTRY_ONLY d=0.080  -0.8951 <- the SELECTED cell, W_TRAIN rank 7 of 33
#   8. COUPLED    d=0.080  -0.9672 (R-67's cell, reproduced)
#   ... 29-31. d=0.000 (x3) -1.4197  32. ENTRY_ONLY d=0.350 -1.5062
#   33. COUPLED d=0.350 -1.5331
#
# THE NEIGHBOURHOOD, recorded at freezing time rather than discovered after:
#   - the winner is NOT a grid corner and NOT a knife-edge. ENTRY_ONLY d=0.080
#     is an INTERIOR cell with both neighbours present and both worse but
#     close: d=0.040 -> +0.4563 (below), d=0.120 -> +0.8040 (above). The top of
#     the ENTRY_ONLY curve is a broad plateau, +0.848/+0.804/+0.708 at
#     0.080/0.120/0.160, spanning far less than one bootstrap standard error.
#   - the top corner is NOT selected in any sub-arm. d=0.350 ranks 4th within
#     ENTRY_ONLY, 11th (last) within EXIT_ONLY and 10th within COUPLED on
#     W_VAL. r68_shared's (F1) therefore does NOT fire: the extension does not
#     produce a second corner, and the grid is not extended again.
#   - the winner ranks 7 of 33 on W_TRAIN; the top W_TRAIN cell is the same
#     sub-arm one grid step higher (ENTRY_ONLY d=0.120, which is W_VAL rank 2).
#     The two windows agree on the SUB-ARM and on the region, not on the cell.
#   - Spearman rank correlation of net growth between W_TRAIN and W_VAL, per
#     sub-arm over its 11 cells: EXIT_ONLY -0.227, ENTRY_ONLY +0.500,
#     COUPLED +0.573; pooled over all 33 cells +0.357. The two halves of the
#     band do NOT transfer equally: the exit threshold's ordering
#     ANTI-transfers, the entry threshold's transfers.
# The selection rule was fixed before the sweep and is honoured exactly as
# written. No filter on the gross column was applied, and the criterion was not
# revisited after seeing that it selects a different sub-arm from the one the
# round's own framing (a wider EXIT band, B-31/B-34) expected.
FROZEN_SUBARM: str | None = "ENTRY_ONLY"
FROZEN_D: float | None = 0.080
# ---------------------------------------------------------------------------


def thresholds(subarm: str, d: float) -> tuple[float, float]:
    return SUB_ARMS[subarm](d)


# ------------------------------------------------------------------ targets


def band_selection(s: np.ndarray, k: int, buffer: float, hold_days: float,
                   delta_in: float, delta_out: float):
    """R-67's `hysteresis_selection`, copied, with ONE generalisation: its
    single coupled `delta` becomes two independent thresholds.

    R-65:   eligible     = isfinite(s) & (s >  0.0)         for everyone
    R-67:   enter_elig   = isfinite(s) & (s >  +delta)      new entrants
            hold_elig    = isfinite(s) & (s >  -delta)      incumbents
    Here:   enter_elig   = isfinite(s) & (s >  +delta_in)   new entrants
            hold_elig    = isfinite(s) & (s >  -delta_out)  incumbents

    At `delta_in == delta_out` this function IS R-67's function; at
    `delta_in == delta_out == 0.0` it is R-65's. Everything else -- the loop,
    the event ledger, the ordering of the three cases, the timer's exemption
    for forced exits -- is unchanged from R-67, which took it unchanged from
    R-65.

    STRICTLY CAUSAL BY CONSTRUCTION: a forward loop whose state at bar ``i``
    depends on rows <= i and nothing else. No mean, std, quantile, scaler or
    time-series rank is taken anywhere; both thresholds and `buffer` are in RAW
    score units precisely so that no normalization is needed.

    Returns ``(sel, ev, ev_bars)``. ``ev`` is R-65's own aggregate event
    ledger; ``ev_bars`` is the same events as per-bar 0/1 arrays so a rate can
    be computed on an evaluation SLICE rather than on the warm-up-inclusive
    range.
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

        # (a) forced exits -- never blocked by the timer. An incumbent leaves
        #     only once its score is no longer above -delta_out.
        if held:
            keep = [a for a in held if elig_hold[a]]
            if len(keep) != len(held):
                ev["forced_exit"] += 1
                ev_bars["forced_exit"][i] = 1
                held = keep
                changed = True

        # entries into empty slots (including refilling a slot a forced exit
        # just freed, and re-entering from flat). Allowed immediately, as in
        # R-65. A new entrant must clear +delta_in.
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
        #     new entrant and must clear +delta_in.
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
    """R-63's sizing block, copied byte-for-byte from R-65/R-67 and unmodified."""
    n = sel.shape[0]
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
    return pd.DataFrame(w, index=index, columns=assets)


def build_band_targets_ev(aligned: dict[str, pd.DataFrame], k: int,
                          buffer: float, hold_days: float, delta_in: float,
                          delta_out: float):
    """Targets AND the event ledger, from one pass of the selection loop."""
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    sel, ev, ev_bars = band_selection(s, k, buffer, hold_days, delta_in,
                                      delta_out)
    return _size(sel, aligned, k, score.index, assets), ev, ev_bars, score.index


def build_band_targets(aligned: dict[str, pd.DataFrame], k: int, buffer: float,
                       hold_days: float, delta_in: float,
                       delta_out: float) -> pd.DataFrame:
    return build_band_targets_ev(aligned, k, buffer, hold_days, delta_in,
                                 delta_out)[0]


def band_fn(delta_in: float, delta_out: float, k: int = K_FIXED,
            buffer: float = BUFFER_FIXED, hold_days: float = HOLD_FIXED):
    return lambda aligned: build_band_targets(aligned, k, buffer, hold_days,
                                              delta_in, delta_out)


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


def build_cell_band(frames, universe, window, delta_in, delta_out):
    """`build_cell` for a band configuration, returning the event ledger too,
    from the SAME single pass of the selection loop (the loop is the expensive
    part; running it twice would double the sweep's cost for nothing)."""
    sub = {t: frames[t] for t in universe}
    warm = align_frames(sub, warm_window(window))
    targets, ev, ev_bars, index = build_band_targets_ev(
        warm, K_FIXED, BUFFER_FIXED, HOLD_FIXED, delta_in, delta_out)

    idx = warm[universe[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]

    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}

    keep = index.isin(idx)
    days = max(int(keep.sum()) / BARS_PER_DAY, 1e-9)
    rates = {"eval_bars": int(keep.sum()), "eval_days": days}
    for key, arr in ev_bars.items():
        rates[f"{key}_total_warm"] = int(ev[key])
        rates[f"{key}_total_eval"] = int(arr[keep].sum())
        rates[f"{key}_per_day_r65"] = ev[key] / days
        rates[f"{key}_per_day_eval"] = float(arr[keep].sum()) / days
    rates["flat_bar_frac_eval"] = (float(ev_bars["flat_bars"][keep].sum())
                                   / max(int(keep.sum()), 1))
    return aligned_eval, targets.loc[idx], first_warm, rates


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

    Both flags are written to every CSV row (`volmatch_matched_*` is what
    scores the cell, `volmatch_shared_flag_*` is what the shared function
    returned) so any divergence is auditable rather than assumed absent.
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
                 arm=ARM, rates=None):
    """One grid cell: both fee levels, each against VOLMATCH_HOLD computed at
    that fee level. R-65's / R-67's `measure_pair`, unchanged in structure."""
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
        membership_change_rate=membership_change_rate_r67(targets),
        membership_change_rate_thresholded=membership_change_rate_thresholded(
            targets),
        **raw_turnover(targets),
    )
    if rates:
        extra.update(rates)
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
    processes, so no single call can report the branch's total. The branch
    total is the SUM of the rows in this file, and it is reported that way.
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
            f" | chgT/d {row.get('membership_change_rate_thresholded', float('nan')):6.3f}"
            f" | mtn {row['mean_notional']:.3f}"
            f" | GROSS {row['gross_growth_diff']:+8.3f}"
            f" [{row['gross_growth_lo']:+7.3f},{row['gross_growth_hi']:+7.3f}]"
            f" | NET {row['net_growth_diff']:+8.3f}"
            f" [{row['net_growth_lo']:+7.3f},{row['net_growth_hi']:+7.3f}]")


# ---------------------------------------------------------------- identity


def cmd_identity(frames):
    """GATE ZERO, both halves. Nothing else is reported until both pass.

    (1) d = 0.0 in ALL THREE sub-arms must reproduce R-65's frozen winner
        exactly, compared against `r68_shared.r65_winner_targets` (R-65's own
        committed `build_buffered_targets`, imported, not reconstructed).
    (2) COUPLED at d = 0.080 must reproduce R-67's published W_TRAIN frontier
        row, compared against the committed
        `reports/r67_gate/conservative_frontier.csv` -- which is read, never
        written. This is a stronger check than a code comparison: it shares no
        code path with R-67 at all.
    """
    print("== IDENTITY (1): d=0.0, all three sub-arms, vs r65_winner_targets ==")
    rows, ok = [], True
    for uname, uni, window in (("U8", UNIVERSE_8, W_TRAIN),
                               ("U6", UNIVERSE_6, W_FULL6)):
        warm = align_frames({t: frames[t] for t in uni}, warm_window(window))
        theirs = r65_winner_targets(warm)
        b = np.nan_to_num(theirs.to_numpy(dtype=float), nan=0.0)
        for sa in SUB_ARM_ORDER:
            din, dout = thresholds(sa, 0.0)
            mine = build_band_targets(warm, K_FIXED, BUFFER_FIXED, HOLD_FIXED,
                                      din, dout)
            a = np.nan_to_num(mine.to_numpy(dtype=float), nan=0.0)
            same_shape = a.shape == b.shape
            maxabs = float(np.max(np.abs(a - b))) if same_shape else float("nan")
            bitwise = bool(same_shape and np.array_equal(a, b))
            exact = bool(same_shape and np.allclose(a, b, atol=1e-12, rtol=0.0))
            print(f"  {uname} {window} {sa:10s} (din={din}, dout={dout}): "
                  f"shape {a.shape}  max|diff| = {maxabs:.3e}  "
                  f"bit-identical={bitwise}  allclose(atol=1e-12)={exact}")
            ok &= exact
            rows.append({"check": "identity_d0", "subarm": sa,
                         "universe": uname, "window": str(window),
                         "n_bars": a.shape[0], "max_abs_diff": maxabs,
                         "bit_identical": bitwise, "allclose_1e12": exact,
                         "passed": exact})

    # How close does any score ever come to a grid boundary? Both predicates
    # are strict (`>`); the literal "until s < -d" reading would differ only
    # where a score lands EXACTLY on a boundary. Counted, not assumed.
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    s_all = cross_sectional_score(warm).to_numpy(dtype=float)
    s_fin = s_all[np.isfinite(s_all)]
    n_ties = 0
    for d in DELTA_GRID_EXT:
        n_hi = int((s_fin == float(d)).sum())
        n_lo = int((s_fin == -float(d)).sum())
        n_ties += n_hi + n_lo
        rows.append({"check": "boundary_ties", "universe": "U8",
                     "window": str(W_TRAIN), "d": d,
                     "n_eq_plus_d": n_hi, "n_eq_minus_d": n_lo,
                     "n_finite_scores": len(s_fin), "passed": True})
    print(f"  boundary ties over all 11 grid values (s == +d or s == -d): "
          f"{n_ties} of {len(s_fin):,} finite score cells")

    # ---- IDENTITY (2): COUPLED d=0.080 vs R-67's PUBLISHED W_TRAIN row ----
    print(f"\n== IDENTITY (2): COUPLED d={R67_DELTA_WINNER:.3f} vs R-67's "
          f"published W_TRAIN frontier row ==")
    ref_path = Path(__file__).resolve().parents[1] / "reports" / "r67_gate" \
        / "conservative_frontier.csv"
    ref = pd.read_csv(ref_path)
    ref = ref[(ref["window"] == "W_TRAIN") & (ref["arm"] == "gate_hysteresis")
              & (np.isclose(ref["p_delta"], R67_DELTA_WINNER))]
    if len(ref) != 1:
        raise RuntimeError(f"expected exactly one R-67 reference row, got {len(ref)}")
    ref = ref.iloc[0]

    din, dout = thresholds("COUPLED", R67_DELTA_WINNER)
    aligned, tg, warm_ok, rates = build_cell_band(frames, UNIVERSE_8, W_TRAIN,
                                                  din, dout)
    if not warm_ok:
        raise RuntimeError("W_TRAIN first evaluated bar not warm")
    row, _ = measure_pair(tg, aligned, UNIVERSE_8, "W_TRAIN", "U8",
                          {"subarm": "COUPLED", "d": R67_DELTA_WINNER,
                           "delta_in": din, "delta_out": dout,
                           "buffer": BUFFER_FIXED, "hold_days": HOLD_FIXED,
                           "k": K_FIXED}, rates=rates)
    fields = ("turnover_per_day", "hold_days", "tenure_days", "mean_notional",
              "gross_growth_diff", "gross_growth_lo", "gross_growth_hi",
              "net_growth_diff", "net_growth_lo", "net_growth_hi",
              "net_dd_diff", "cand_final", "bench_final",
              "membership_changes_per_day", "raw_turnover_per_day", "n_days")
    ok2 = True
    print(f"  {'field':<28} {'R-67 published':>18} {'this branch':>18} "
          f"{'|diff|':>12}")
    for f in fields:
        a, b = float(ref[f]), float(row[f])
        diff = abs(a - b)
        same = bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))
        ok2 &= same
        print(f"  {f:<28} {a:>18.9f} {b:>18.9f} {diff:>12.3e}"
              f"{'' if same else '   <-- MISMATCH'}")
        rows.append({"check": "identity_r67_coupled_0080", "field": f,
                     "r67_published": a, "this_branch": b, "abs_diff": diff,
                     "passed": same})
    print(f"  R-67 CELL REPRODUCED: {ok2}")

    ok = bool(ok and ok2)
    write_csv(OUT_DIR / "conservative_identity.csv", rows)
    print(f"\n  BOTH IDENTITIES HOLD: {ok}")
    return ok


# ------------------------------------------------------------------ checks


def truncation_probe(frames, din, dout, frac=0.6) -> tuple[bool, float]:
    """Build targets on the first 60% of bars and on 100%; the first 60% of
    rows must agree EXACTLY (atol=1e-12)."""
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * frac)
    full = build_band_targets(warm, K_FIXED, BUFFER_FIXED, HOLD_FIXED,
                              din, dout).to_numpy()
    trunc = build_band_targets({t: df.iloc[:cut] for t, df in warm.items()},
                               K_FIXED, BUFFER_FIXED, HOLD_FIXED,
                               din, dout).to_numpy()
    a = np.nan_to_num(full[:cut], nan=0.0)
    b = np.nan_to_num(trunc[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0)), float(np.max(np.abs(a - b)))


def tail_perturbation_probe(frames, din, dout, frac_tail=0.4) -> tuple[bool, float]:
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
    a = np.nan_to_num(build_band_targets(warm, K_FIXED, BUFFER_FIXED,
                                         HOLD_FIXED, din, dout).to_numpy()[:cut],
                      nan=0.0)
    b = np.nan_to_num(build_band_targets(bad, K_FIXED, BUFFER_FIXED,
                                         HOLD_FIXED, din, dout).to_numpy()[:cut],
                      nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0)), float(np.max(np.abs(a - b)))


def cmd_checks(frames, subarm=None, d=None):
    sa = FROZEN_SUBARM if subarm is None else subarm
    dd = FROZEN_D if d is None else float(d)
    if sa is None or dd is None:
        sa, dd = "COUPLED", 0.080  # mid-grid, only when nothing is frozen yet
        print(f"  (nothing frozen yet -- probing at {sa} d={dd})")
    din, dout = thresholds(sa, dd)
    print(f"== correctness gates ({sa} d={dd:.3f} -> delta_in={din:.3f}, "
          f"delta_out={dout:.3f}) ==")
    ok = True
    rows = []

    t0 = time.time()
    warm8 = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    c1 = check_causality(band_fn(din, dout), warm8)
    print(f"  (a) r63_shared.check_causality (truncation probe): {c1}  "
          f"[{time.time() - t0:.1f}s]")
    ok &= c1
    rows.append({"check": "check_causality", "subarm": sa, "d": dd,
                 "delta_in": din, "delta_out": dout, "passed": c1})

    # 60% truncation on the frozen cell, on the identity cell, and on one cell
    # of EACH sub-arm at the extended grid's top.
    probes = [(sa, dd)] + [(x, 0.0) for x in ("COUPLED",)] \
        + [(x, DELTA_GRID_EXT[-1]) for x in SUB_ARM_ORDER]
    seen = set()
    for psa, pd_ in probes:
        pin, pout = thresholds(psa, pd_)
        if (pin, pout) in seen:
            continue
        seen.add((pin, pout))
        p, mx = truncation_probe(frames, pin, pout)
        print(f"  (b) 60% truncation probe ({psa} d={pd_:.3f}): {p}  "
              f"max|diff| = {mx:.3e}")
        ok &= p
        rows.append({"check": "truncation_60pct", "subarm": psa, "d": pd_,
                     "delta_in": pin, "delta_out": pout, "passed": p,
                     "max_abs_diff": mx})

    p2, mx2 = tail_perturbation_probe(frames, din, dout)
    print(f"  (extra) tail x10 perturbation probe ({sa} d={dd:.3f}): {p2}  "
          f"max|diff| = {mx2:.3e}")
    ok &= p2
    rows.append({"check": "tail_x10", "subarm": sa, "d": dd, "delta_in": din,
                 "delta_out": dout, "passed": p2, "max_abs_diff": mx2})

    # index hygiene: no bar dated 2023-01-01 or later may reach a selection
    # window cell. (W_FULL6 ends at the last bar by R-63's own convention and
    # is not a selection window; see B-33.)
    for wname, window, uni in (("W_TRAIN", W_TRAIN, UNIVERSE_8),
                               ("W_VAL", W_VAL, UNIVERSE_8)):
        _, tg, warm_ok, _ = build_cell_band(frames, uni, window, din, dout)
        bad = int((tg.index >= pd.Timestamp("2023-01-01", tz="UTC")).sum())
        print(f"  (c) {wname}: {len(tg):,} bars {tg.index[0]} -> {tg.index[-1]}"
              f"  bars in reserved holdout: {bad}  first_bar_warm={warm_ok}")
        ok &= (bad == 0) and warm_ok
        rows.append({"check": f"index_{wname}", "subarm": sa, "d": dd,
                     "n_bars": len(tg), "first": str(tg.index[0]),
                     "last": str(tg.index[-1]), "bars_in_holdout": bad,
                     "first_bar_warm": warm_ok,
                     "passed": (bad == 0) and warm_ok})

    # Fold the identity results in, so `conservative_checks.csv` carries the
    # identity/probe record the round asked for in one file.
    ident_path = OUT_DIR / "conservative_identity.csv"
    merged = rows
    if ident_path.exists():
        ident = pd.read_csv(ident_path).to_dict("records")
        merged = ident + rows
        print(f"  (folded {len(ident)} identity rows from "
              f"{ident_path.name} into conservative_checks.csv)")
    write_csv(OUT_DIR / "conservative_checks.csv", merged)
    print(f"  ALL GATES PASS: {ok}")
    return ok


# --------------------------------------------------------------------- M1'


def cmd_m1(frames):
    """M1' for EVERY grid cell of EVERY sub-arm on W_TRAIN, plus the event
    ledger. No simulation runs here -- M1' is a property of the target matrix
    -- so this costs nothing against the config count and every cell is
    reported, not just the selected one."""
    print("== M1' (mechanism gate, r68_shared) on W_TRAIN / U8, ALL cells ==")
    _, base_tg, _ = build_cell(frames, UNIVERSE_8, W_TRAIN,
                               lambda a: r65_winner_targets(a))
    base_turn = turnover_stats(base_tg)["turnover_per_day"]
    _, _, _, base_rates = build_cell_band(frames, UNIVERSE_8, W_TRAIN, 0.0, 0.0)
    print("  baseline = R-65's frozen winner (k=1, buffer=0.05, hold_days=1)")
    print(f"    thresholded membership changes/day = "
          f"{membership_change_rate_thresholded(base_tg):.4f}   "
          f"turnover/day = {base_turn:.4f}   forced exits/day = "
          f"{base_rates['forced_exit_per_day_r65']:.4f} (R-65 convention) / "
          f"{base_rates['forced_exit_per_day_eval']:.4f} (eval)   "
          f"[R-65 published {R65_FORCED_EXIT_PER_DAY}]")
    print(f"  M1' bar: >= {M1_MIN_REDUCTION:.0%} fewer thresholded membership "
          f"changes/day AND >= {M1_MIN_REDUCTION:.0%} lower turnover\n")
    print("  subarm         d |  chgT/d   red_mem |  turn/d  red_turn | M1' ||"
          " fexit/d  entry/d   swap/d  flat_frac | ten(d)")
    rows = []
    for sa in SUB_ARM_ORDER:
        for d in DELTA_GRID_EXT:
            din, dout = thresholds(sa, d)
            _, tg, warm_ok, rates = build_cell_band(frames, UNIVERSE_8,
                                                    W_TRAIN, din, dout)
            if not warm_ok:
                raise RuntimeError(f"{sa} d={d}: first evaluated bar not warm")
            ts = turnover_stats(tg)
            m1 = m1_pass(tg, base_tg, ts["turnover_per_day"], base_turn)
            row = {"arm": ARM, "window": "W_TRAIN", "universe": "U8",
                   "p_subarm": sa, "p_d": d, "p_delta_in": din,
                   "p_delta_out": dout, "p_buffer": BUFFER_FIXED,
                   "p_hold_days": HOLD_FIXED, "p_k": K_FIXED,
                   "m1_bar": M1_MIN_REDUCTION,
                   "m1_floor": MEMBERSHIP_WEIGHT_FLOOR,
                   **{f"m1_{k2}": v for k2, v in m1.items()},
                   "set_period_days": holding_period_days(tg),
                   "tenure_days": mean_tenure_days(tg),
                   "turnover_per_day": ts["turnover_per_day"],
                   "mean_notional": mean_total_notional(tg),
                   "membership_change_rate_r67": membership_change_rate_r67(tg),
                   "r65_published_forced_exit_per_day": R65_FORCED_EXIT_PER_DAY,
                   **rates, **raw_turnover(tg)}
            rows.append(row)
            print(f"  {sa:10s} {d:6.3f} | {m1['cand_membership_per_day']:6.3f} "
                  f"{m1['membership_reduction']:+8.2%} | "
                  f"{ts['turnover_per_day']:6.3f} "
                  f"{m1['turnover_reduction']:+8.2%} | "
                  f"{'PASS' if m1['passed'] else 'fail'} ||"
                  f" {rates['forced_exit_per_day_eval']:7.3f} "
                  f"{rates['entry_per_day_eval']:8.3f} "
                  f"{rates['swap_per_day_eval']:8.3f} "
                  f"{rates['flat_bar_frac_eval']:10.4f} |"
                  f" {row['tenure_days']:6.2f}")

    write_csv(OUT_DIR / "conservative_m1.csv", rows)
    return rows


# ------------------------------------------------------------------ frontier


def cmd_frontier(frames, windows=("W_TRAIN", "W_VAL")):
    """The round's deliverable: every cell of every sub-arm, both fee levels,
    on the two SELECTION windows. No decision-window read happens here."""
    print(f"== FRONTIER: {len(SUB_ARM_ORDER)} sub-arms x {len(DELTA_GRID_EXT)} "
          f"d-cells + R-63 reference, k={K_FIXED} buffer={BUFFER_FIXED} "
          f"hold_days={HOLD_FIXED}, U8, vs VOLMATCH_HOLD ==")
    print(f"   grid = {DELTA_GRID_EXT}  (cap 1.6*sigma = {DLC_SATURATION:.4f}, "
          f"sigma_W_TRAIN = {SIGMA_SCORE_W_TRAIN})")
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
                              {"subarm": "R63_REF", "d": float("nan"),
                               "delta_in": float("nan"),
                               "delta_out": float("nan"),
                               "buffer": float("nan"), "hold_days": 0,
                               "k": K_FIXED}, arm="r63_reference")
        row["config_kind"] = "reference"
        rows.append(row)
        print("  [R-63 reference k=1, no buffer, no band]")
        print(fmt_front(row))

        for sa in SUB_ARM_ORDER:
            for d in DELTA_GRID_EXT:
                din, dout = thresholds(sa, d)
                aligned, tg, warm_ok, rates = build_cell_band(
                    frames, UNIVERSE_8, window, din, dout)
                if not warm_ok:
                    raise RuntimeError(f"{wname} {sa} d={d}: first bar not warm")
                row, _ = measure_pair(tg, aligned, UNIVERSE_8, wname, "U8",
                                      {"subarm": sa, "d": d, "delta_in": din,
                                       "delta_out": dout,
                                       "buffer": BUFFER_FIXED,
                                       "hold_days": HOLD_FIXED, "k": K_FIXED},
                                      rates=rates)
                row["config_kind"] = "grid"
                rows.append(row)
                tag = ""
                if d == 0.0:
                    tag = "  (= R-65 frozen winner)"
                elif sa == "COUPLED" and d == R67_DELTA_WINNER:
                    tag = "  (= R-67's published winner)"
                print(f"  [{sa} d={d:.3f}]{tag}")
                print(fmt_front(row))
        print(f"  {wname} done in {time.time() - t0:.0f}s "
              f"(config_count={config_count()})")

    write_csv(OUT_DIR / "conservative_frontier.csv", rows)
    return rows


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman by hand: scipy is not installed; rank-then-Pearson is
    identical."""
    return float(a.rank().corr(b.rank()))


def cmd_select(frames=None, rows=None):
    """Report the selection, both orderings, the neighbourhood and the
    per-sub-arm cross-window rank correlation. Reads the frontier CSV; runs
    nothing."""
    df = pd.read_csv(OUT_DIR / "conservative_frontier.csv")
    df["cost"] = df["gross_growth_diff"] - df["net_growth_diff"]

    out = {}
    for wname in ("W_TRAIN", "W_VAL"):
        ref = df[(df["window"] == wname) & (df["config_kind"] == "reference")]
        sub = df[(df["window"] == wname) & (df["config_kind"] == "grid")] \
            .sort_values(["net_growth_diff", "net_dd_diff"],
                         ascending=[False, True])
        print(f"== {wname}: value and cost vs VOLMATCH_HOLD "
              f"(ordered by net growth diff, best first) ==")
        print("   rk  subarm          d | turn/d chgT/d ten(d) fexit/d swap/d "
              "flat% |    GROSS |   COST |      NET | net 95% interval       "
              "| dd_diff")
        for i, (_, r) in enumerate(pd.concat([ref, sub]).iterrows()):
            if r["config_kind"] == "reference":
                tag, rk = "R-63 reference   ", "  -"
            else:
                tag, rk = f"{r['p_subarm']:<11s} {r['p_d']:6.3f}", f"{i:3d}"
            print(f"  {rk} {tag} | {r['turnover_per_day']:6.3f} "
                  f"{r['membership_change_rate_thresholded']:6.3f} "
                  f"{r['tenure_days']:6.2f} "
                  f"{r.get('forced_exit_per_day_eval', float('nan')):7.3f} "
                  f"{r.get('swap_per_day_eval', float('nan')):6.3f} "
                  f"{100 * r.get('flat_bar_frac_eval', float('nan')):5.1f}"
                  f" | {r['gross_growth_diff']:+8.3f} | {r['cost']:6.3f} |"
                  f" {r['net_growth_diff']:+8.3f} | "
                  f"[{r['net_growth_lo']:+7.3f},{r['net_growth_hi']:+7.3f}] |"
                  f" {r['net_dd_diff']:+7.2f}")
        n = len(sub)
        n_pos = int((sub["net_growth_diff"] > 0).sum())
        n_sig = int(((sub["net_growth_diff"] > 0)
                     & (sub["net_growth_lo"] > 0)).sum())
        n_d5 = int((sub["gross_growth_diff"] >= D5_BAR_R68).sum())
        print(f"  -> best net {sub['net_growth_diff'].max():+.4f}; net>0: "
              f"{n_pos}/{n}; net>0 with interval excluding zero: {n_sig}/{n}; "
              f"clearing the D5 bar (+{D5_BAR_R68:.3f} gross): {n_d5}/{n}")
        out[wname] = sub

    tr, va = out["W_TRAIN"], out["W_VAL"]
    merged = tr[["p_subarm", "p_d", "net_growth_diff"]].merge(
        va[["p_subarm", "p_d", "net_growth_diff"]], on=["p_subarm", "p_d"],
        suffixes=("_train", "_val"))
    print("\n  Spearman rank correlation of net growth, W_TRAIN vs W_VAL:")
    for sa in SUB_ARM_ORDER:
        m = merged[merged["p_subarm"] == sa]
        print(f"    {sa:<11s} over its {len(m):2d} cells: "
              f"{_spearman(m['net_growth_diff_train'], m['net_growth_diff_val']):+.3f}")
    print(f"    {'POOLED':<11s} over all {len(merged):2d} cells: "
          f"{_spearman(merged['net_growth_diff_train'], merged['net_growth_diff_val']):+.3f}")

    best = va.iloc[0]
    bsa, bd = str(best["p_subarm"]), float(best["p_d"])
    print(f"\n  W_VAL WINNER (declared criterion): {bsa} d={bd:.3f}  "
          f"net={best['net_growth_diff']:+.4f} "
          f"[{best['net_growth_lo']:+.4f},{best['net_growth_hi']:+.4f}]  "
          f"gross={best['gross_growth_diff']:+.4f}  "
          f"dd={best['net_dd_diff']:+.3f}")
    key = list(zip(tr["p_subarm"], tr["p_d"]))
    print(f"  that cell's W_TRAIN rank: {key.index((bsa, bd)) + 1} of {len(tr)}")

    grid = list(DELTA_GRID_EXT)
    i = grid.index(bd)
    for j, lab in ((i - 1, "lower"), (i + 1, "upper")):
        if 0 <= j < len(grid):
            nb = va[(va["p_subarm"] == bsa)
                    & (np.isclose(va["p_d"], grid[j]))].iloc[0]
            better = nb["net_growth_diff"] > best["net_growth_diff"]
            print(f"  neighbour ({lab}) {bsa} d={grid[j]:.3f}: W_VAL net "
                  f"{nb['net_growth_diff']:+.4f} -> "
                  f"{'BETTER' if better else 'worse'} than the winner")
        else:
            print(f"  neighbour ({lab}): none -- the winner is a GRID CORNER")

    # the extension's own question: do cells past R-67's edge improve?
    print("\n  EXTENSION (F1): W_VAL net growth by sub-arm across the grid")
    for sa in SUB_ARM_ORDER:
        m = va[va["p_subarm"] == sa].sort_values("p_d")
        cells = "  ".join(f"{d:.3f}:{v:+.3f}"
                          for d, v in zip(m["p_d"], m["net_growth_diff"]))
        print(f"    {sa:<11s} {cells}")
    return best


# ------------------------------------------------------------------ run


def _frozen(subarm=None, d=None):
    sa = FROZEN_SUBARM if subarm is None else subarm
    dd = FROZEN_D if d is None else d
    if sa is None or dd is None:
        raise SystemExit("configuration is not frozen yet: run `frontier`, "
                         "then `select`, then set FROZEN_SUBARM / FROZEN_D")
    return str(sa), float(dd)


def cmd_run(frames, subarm=None, d=None):
    sa, dd = _frozen(subarm, d)
    din, dout = thresholds(sa, dd)
    print(f"== D-CELLS: frozen {sa} d={dd:.3f} (delta_in={din:.3f}, "
          f"delta_out={dout:.3f}; k={K_FIXED}, buffer={BUFFER_FIXED}, "
          f"hold_days={HOLD_FIXED}) ==")
    rows = []
    params = {"subarm": sa, "d": dd, "delta_in": din, "delta_out": dout,
              "buffer": BUFFER_FIXED, "hold_days": HOLD_FIXED, "k": K_FIXED}

    # ---- D1 / D2 / D5 : W_FULL6, U6, vs VOLMATCH_HOLD -----------------
    aligned, targets, warm_ok, rates = build_cell_band(frames, UNIVERSE_6,
                                                       W_FULL6, din, dout)
    if not warm_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")
    print(f"  W_FULL6 bars {len(targets):,}  {targets.index[0]} -> "
          f"{targets.index[-1]}  first_bar_warm={warm_ok}")

    row, st = measure_pair(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                           params, rates=rates)
    row["config_kind"] = "decision"
    d1 = d1_pass(row) and st["matched_net"]
    d2 = d2_pass(row) and st["matched_net"]
    d5 = d5_pass(row) and st["matched_gross"]
    row.update({"d1_pass": d1, "d2_pass": d2, "d5_pass": d5,
                "d5_bar": D5_BAR_R68,
                "note": "D1/D2/D5 primary vs VOLMATCH_HOLD"})
    rows.append(row)
    print(fmt_front(row))
    print(f"    cand_vol {row['cand_vol_net']:.3f} vs volmatch_vol "
          f"{row['volmatch_vol_net']:.3f} at c={row['volmatch_c_net']:.3f}  "
          f"matched={st['matched_net']} (shared flag "
          f"{row['volmatch_shared_flag_net']})")
    print(f"    net dd_diff {row['net_dd_diff']:+.3f}pp "
          f"[{row['net_dd_lo']:+.3f}, {row['net_dd_hi']:+.3f}]")
    print(f"    D1 PASS={d1}  D2 PASS={d2}  D5 PASS={d5} "
          f"(CORRECTED bar {D5_BAR_R68:+.3f}, gross "
          f"{row['gross_growth_diff']:+.4f})")

    # ---- M1' at the frozen configuration, on W_TRAIN -------------------
    _, tg_tr, _, ev_tr = build_cell_band(frames, UNIVERSE_8, W_TRAIN, din, dout)
    _, base_tr, _ = build_cell(frames, UNIVERSE_8, W_TRAIN,
                               lambda a: r65_winner_targets(a))
    m1 = m1_pass(tg_tr, base_tr, turnover_stats(tg_tr)["turnover_per_day"],
                 turnover_stats(base_tr)["turnover_per_day"])
    rows.append({"arm": ARM, "window": "W_TRAIN", "universe": "U8",
                 "config_kind": "decision", "bench": "R65_WINNER",
                 **{f"p_{k2}": v for k2, v in params.items()},
                 "m1_bar": M1_MIN_REDUCTION, "m1_floor": MEMBERSHIP_WEIGHT_FLOOR,
                 **{f"m1_{k2}": v for k2, v in m1.items()}, **ev_tr,
                 "note": "M1' mechanism gate at the frozen configuration"})
    print(f"  [M1' W_TRAIN U8] membership {m1['cand_membership_per_day']:.4f}/day "
          f"vs {m1['baseline_membership_per_day']:.4f} -> "
          f"{m1['membership_reduction']:+.2%} ({m1['membership_passed']}); "
          f"turnover {m1['cand_turnover_per_day']:.4f}/day vs "
          f"{m1['baseline_turnover_per_day']:.4f} -> "
          f"{m1['turnover_reduction']:+.2%} ({m1['turnover_passed']}); "
          f"bar {M1_MIN_REDUCTION:.0%}  M1' PASS={m1['passed']}")

    # ---- continuity: MATCHED_HOLD / EW_HOLD / BTC_HOLD ----------------
    c = mean_total_notional(targets)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                            aligned, SPOT_BASE)
    r_mh = compare(st["cand_net"], mh)
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "MATCHED_HOLD", "config_kind": "context",
                 **{f"p_{k2}": v for k2, v in params.items()},
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
                 "bench": "EW_HOLD", "config_kind": "context",
                 **{f"p_{k2}": v for k2, v in params.items()},
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
                 "bench": "BTC_HOLD", "config_kind": "context",
                 **{f"p_{k2}": v for k2, v in params.items()},
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
    # R-67's D4 cell construction, copied.
    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    r40 = compare(cand40, ew40)
    d4 = bool(r40["cand_final"] > r40["bench_final"])
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "EW_HOLD", "config_kind": "decision",
                 **{f"p_{k2}": v for k2, v in params.items()},
                 "fee": 0.004, "d4_pass": d4,
                 "net_growth_diff": r40["growth_diff"],
                 "net_growth_lo": r40["growth_lo"],
                 "net_growth_hi": r40["growth_hi"],
                 "net_dd_diff": r40["dd_diff"], "cand_final": r40["cand_final"],
                 "bench_final": r40["bench_final"], "cand_dd": r40["cand_dd"],
                 "bench_dd": r40["bench_dd"], "n_days": r40["n_days"],
                 "note": "D4 cost tier 0.40% (SPOT_REAL) vs EW_HOLD"})
    print(f"  [D4 @0.40%] cand {r40['cand_final']:,.2f} vs EW_HOLD "
          f"{r40['bench_final']:,.2f} -> D4 PASS={d4}  (growth "
          f"{r40['growth_diff']:+.4f} [{r40['growth_lo']:+.3f}, "
          f"{r40['growth_hi']:+.3f}])")

    # ---- D3 : W_VAL, U8, 0.10% ---------------------------------------
    aligned3, targets3, warm3, rates3 = build_cell_band(frames, UNIVERSE_8,
                                                        W_VAL, din, dout)
    if not warm3:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    row3, st3 = measure_pair(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                             params, rates=rates3)
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
                 **{f"p_{k2}": v for k2, v in params.items()},
                 "mean_notional": c3, "matched_hold_vol": realized_vol(mh3),
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
            "matched_gross": st["matched_gross"],
            "matched_net_d3": st3["matched_net"], "subarm": sa, "d": dd}


# ------------------------------------------------------------------ scramble


def cmd_scramble(frames, subarm=None, d=None, state=None):
    """The round's SINGLE shared control: `r68_shared.scramble_fixed_perm`.

    One fixed column permutation for the whole series. A column permutation is
    an L1 isometry, so the candidate's total notional path, turnover and
    holding periods are preserved EXACTLY -- only the asset->signal assignment
    is destroyed. R-63's redraw-per-change form is deliberately not run: the
    shared file withdrew it (R-65 and R-67 both measured it as a large turnover
    over-charge, a bias running in the candidate's favour).
    """
    sa, dd = _frozen(subarm, d)
    din, dout = thresholds(sa, dd)
    print(f"== FALSIFICATION: fixed-permutation cross-section scramble, seeds "
          f"{list(SCRAMBLE_SEEDS)}, D1 cell, {sa} d={dd:.3f} ==")
    if state is None:
        aligned, targets, warm_ok, _ = build_cell_band(frames, UNIVERSE_6,
                                                       W_FULL6, din, dout)
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
    for seed in SCRAMBLE_SEEDS:
        stg = scramble_fixed_perm(targets, seed)
        ident = bool(np.array_equal(
            np.nan_to_num(stg.to_numpy(dtype=float), nan=0.0),
            np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0)))
        eq = simulate_portfolio(stg, aligned, SPOT_BASE)
        r = compare(eq, vm)
        diffs.append(r["growth_diff"])
        rows.append({"arm": f"{ARM}_fixedperm", "form": "fixed_permutation",
                     "seed": seed, "window": "W_FULL6", "universe": "U6",
                     "bench": "VOLMATCH_HOLD", "fee": 0.001, "p_subarm": sa,
                     "p_d": dd, "p_delta_in": din, "p_delta_out": dout,
                     "identity_perm": ident,
                     "mean_notional": mean_total_notional(stg),
                     "turnover_per_day": turnover_stats(stg)["turnover_per_day"],
                     **{key: r[key] for key in
                        ("cand_final", "bench_final", "cand_dd", "bench_dd",
                         "growth_diff", "growth_lo", "growth_hi", "dd_diff",
                         "dd_lo", "dd_hi", "n_days")}})
        print(f"  seed {seed}: growth_diff {r['growth_diff']:+.4f}  final "
              f"{r['cand_final']:>12,.4f}  turnover "
              f"{rows[-1]['turnover_per_day']:7.3f}/d"
              f"{'  (IDENTITY PERM)' if ident else ''}")

    p90 = float(np.percentile(diffs, 90))
    survived = bool(real > p90)
    mean_turn = float(np.mean([r["turnover_per_day"] for r in rows]))
    rows.append({"arm": ARM, "form": "fixed_permutation", "seed": -1,
                 "window": "W_FULL6", "universe": "U6",
                 "bench": "VOLMATCH_HOLD", "fee": 0.001, "p_subarm": sa,
                 "p_d": dd, "p_delta_in": din, "p_delta_out": dout,
                 "growth_diff": real, "scramble_p90": p90,
                 "scramble_survived": survived,
                 "turnover_per_day": cand_turn,
                 "scramble_mean_turnover_per_day": mean_turn,
                 "mean_notional": mean_total_notional(targets),
                 "n_better": int(sum(x >= real for x in diffs))})
    print(f"  real {real:+.4f} vs p90 {p90:+.4f} -> SURVIVED={survived}"
          f"  ({sum(x >= real for x in diffs)} of {len(diffs)} better)")
    print(f"      candidate turnover {cand_turn:.3f}/d vs scramble mean "
          f"{mean_turn:.3f}/d  (ratio {mean_turn / max(cand_turn, 1e-12):.2f}x)")

    write_csv(OUT_DIR / "conservative_scramble.csv", rows)
    return {"fixed_perm": survived}


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["identity", "checks", "m1", "frontier",
                                    "select", "run", "scramble", "all"])
    ap.add_argument("--subarm", default=None, choices=[None] + list(SUB_ARM_ORDER))
    ap.add_argument("--d", type=float, default=None)
    ap.add_argument("--windows", default="W_TRAIN,W_VAL")
    args = ap.parse_args()

    if args.cmd == "select":
        cmd_select()
        return

    frames = load_universe(UNIVERSE_8)

    if args.cmd == "identity":
        cmd_identity(frames)
    elif args.cmd == "checks":
        cmd_checks(frames, args.subarm, args.d)
    elif args.cmd == "m1":
        cmd_m1(frames)
    elif args.cmd == "frontier":
        cmd_frontier(frames, tuple(args.windows.split(",")))
    elif args.cmd == "run":
        cmd_run(frames, args.subarm, args.d)
    elif args.cmd == "scramble":
        cmd_scramble(frames, args.subarm, args.d)
    else:
        cmd_identity(frames)
        cmd_checks(frames, args.subarm, args.d)
        cmd_m1(frames)
        cmd_frontier(frames)
        cmd_select()
        st = cmd_run(frames, args.subarm, args.d)
        sc = cmd_scramble(frames, st["subarm"], st["d"], st)
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
