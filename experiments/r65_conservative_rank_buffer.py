"""R-65 CONSERVATIVE branch -- rank buffering plus a minimum holding period.

R-63 priced this project's one genuinely-real signal at exactly one point on
the holding-period axis: at k=1 the cross-sectional rank leader changes 2.86
times per day, the arm turns over 3.44x equity/day, and the 0.00344 log/day
drag that implies is 8.02 log units over 2,332 days -- against a frictionless
edge of +0.480. A 16-to-1 deficit. This branch asks the standard practitioner
question: **if you stop churning on rank noise, does the edge survive the
saving?**

The mitigation is not invented here. It is the one the transaction-cost
literature endorses most consistently for exactly this problem:

  - Novy-Marx, R., & Velikov, M. (2016), "A Taxonomy of Anomalies and Their
    Trading Costs," *Review of Financial Studies* 29(1), 104-147. Of the
    mitigation techniques they test across 23 anomalies, **buffering / banding
    a cross-sectional selection rule** is the one that most reliably preserves
    net alpha, because it removes the trades that carry the least information
    per unit of cost -- the marginal rank flips.
  - Leland, H. E. (2000), "Optimal Portfolio Implementation with Transactions
    Costs and Capital Gains Taxes," Haas School working paper. The no-trade
    region: with proportional costs the optimal policy never tracks the
    frictionless target, it tolerates a band around it.
  - This repo's own L-05 / L-06 derived a no-trade band -- but for a
    **continuous single-asset exposure fraction**, never for a **discrete
    cross-sectional selection**. R-63's turnover bill is entirely the discrete
    "which asset" decision: its exposure magnitude already carried v4's shipped
    0.10 deadband and it still turned over 3.44x/day.

=====================================================================
THE RULE (frozen before any number was read)
=====================================================================

Start from R-63's novel arm BYTE-FOR-BYTE -- `cross_sectional_score`,
`conditional_vol_scale` driven by the equal-weight all-N basket, the 0.10
deadband on desired TOTAL notional, long-only unlevered, equal weighting among
holdings, all imported from `experiments/r65_shared.py` rather than copied --
and change ONLY the selection rule's stickiness:

  RETENTION.  An asset currently held is retained unless
                (a) its score goes non-positive or non-finite  -> FORCED EXIT
                (b) a challenger's score exceeds the weakest incumbent's by
                    more than `buffer` (raw score units; the score is
                    `close/anchor - 1`, so buffer=0.02 is 2 percentage points)
  MINIMUM HOLD.  No VOLUNTARY swap (case b) may occur until `hold_days` days
                 have elapsed since the last membership change.
  FORCED EXITS ARE NEVER BLOCKED by the timer, and the slot they free may be
                 refilled on the same bar. This is deliberate and is part of
                 the hypothesis: a rule that holds a decaying asset for 30 days
                 regardless is a different and worse hypothesis, and holding a
                 non-positive-score asset is exactly what R-63's "flat when
                 nothing is positive" rule exists to prevent.

`k` is FIXED AT 1 -- R-63's own frozen selection -- so this arm extends R-63's
exact endpoint rather than confounding the holding-period axis with a
concentration axis. (A k=3 diagnostic is available via `--k` and is reported,
clearly labelled, as an EXTRA, not as the primary.)

GRID: buffer in {0.00, 0.02, 0.05, 0.10} x hold_days in {1, 3, 7, 14, 30}
      = 20 parameter-search trials. Swept on W_TRAIN, selected on W_VAL.

SELECTION CRITERION, declared here BEFORE the sweep was run: the frozen
configuration is the grid cell with the highest **W_VAL net growth difference
versus VOLMATCH_HOLD at 0.10%** -- the D1 decision statistic, on the
selection window, which is R-63's own convention. Tie-break: the more negative
W_VAL net drawdown difference. No filter on the gross column is applied at
selection time; the gross column is D5's diagnostic and using it to select
would be selecting on the falsification test.

Note that `buffer=0.00, hold_days=1` is NOT identical to R-63's arm -- it still
imposes one day of stickiness -- so `r63_baseline_targets(aligned, 1)` is run
through the identical frontier reporting as the leftmost reference point.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- the shared pre-registration's (F2)
=====================================================================

Buffering trades signal for turnover at worse than 1:1. The selection IS the
signal here; a margin wide enough to cut 2.86 leader-changes/day to 0.03 may
discard most of the information along with the trades, in which case the value
curve and the cost curve never cross and the frontier is monotone. Novy-Marx &
Velikov's result is that mitigation *reduces* the cost of an anomaly, not that
it rescues one whose gross alpha is 16x smaller than its gross cost.

Windows, universes, costs, D1-D5, the scramble control and the further-work bar
all live in the frozen pre-registration in `experiments/r65_shared.py`. This
file implements a candidate and measures it; it does not define or relax a
rule, and it does not edit that file.

Run as:
    python3 experiments/r65_conservative_rank_buffer.py checks
    python3 experiments/r65_conservative_rank_buffer.py repro
    python3 experiments/r65_conservative_rank_buffer.py frontier
    python3 experiments/r65_conservative_rank_buffer.py run
    python3 experiments/r65_conservative_rank_buffer.py scramble
    python3 experiments/r65_conservative_rank_buffer.py all
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

from experiments.r65_shared import (  # noqa: E402
    BARS_PER_DAY,
    D5_BAR,
    DEADBAND,
    OUT_DIR,
    R63_NET_D1,
    R63_TURNOVER_PER_DAY,
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
    matched_hold_targets,
    mean_total_notional,
    r63_baseline_targets,
    realized_vol,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)
from experiments.r63_shared import check_causality  # noqa: E402

ARM = "rank_buffer"
K_FIXED = 1  # R-63's own frozen selection. Not a free parameter here.

BUFFER_GRID = (0.00, 0.02, 0.05, 0.10)
HOLD_GRID = (1, 3, 7, 14, 30)

# ---------------------------------------------------------------------------
# FROZEN CONFIGURATION.
#
# Selected on **W_VAL only** (2022-01-01 -> 2022-12-31, U8, spot 0.10%) on the
# D1 decision statistic -- net growth difference versus VOLMATCH_HOLD -- by the
# criterion declared in this module's docstring before the sweep was run.
# Tie-break was the net drawdown difference; it was not needed.
#
# Set from `conservative_frontier.csv` BEFORE any D-cell was computed. The
# W_TRAIN ordering, the neighbourhood, and whether the ranking inverts between
# the two selection windows are all reported in the branch report; see
# `cmd_select` which prints them.
#
# W_VAL top of the ordering (net growth diff vs VOLMATCH_HOLD, best first);
# full table in reports/r65_holding_period/conservative_frontier.csv:
#   buffer=0.05 H= 1  net +0.1425  gross +0.5699   <-- SELECTED
#   buffer=0.10 H= 1  net +0.0163  gross +0.4487
#   buffer=0.00 H= 1  net -0.0074  gross +0.4540
#   buffer=0.02 H= 1  net -0.0752  gross +0.3736
#   buffer=0.05 H= 3  net -0.0970  gross +0.3271
#   ... monotonically worse through H=7, 14, 30; worst is buffer=0.10 H=14
#       at net -0.7297.
#
# THE NEIGHBOURHOOD IS A PEAK, NOT A PLATEAU, and the axis inverts between the
# two selection windows. Recorded here, at freezing time, rather than
# discovered afterwards:
#   - every immediate neighbour of the winner is worse AND negative
#     (buffer 0.02/0.10 at H=1: -0.075 / +0.016; hold_days 3 at buffer 0.05:
#     -0.097). The winner is the only cell in its neighbourhood above zero.
#   - the winner ranks 19th of 20 on W_TRAIN (net -1.4205, against the
#     W_TRAIN best of -0.4253 at buffer=0.00 H=14).
#   - Spearman rank correlation of net growth between W_TRAIN and W_VAL over
#     the 20 cells is -0.316: the ordering does not merely fail to transfer,
#     it ANTI-transfers. This is R-63's `k`-axis signature reproduced on the
#     holding-period axis, and it is evidence that the axis is noise.
# The selection rule was fixed before the sweep and is honoured as written.
FROZEN_BUFFER: float | None = 0.05
FROZEN_HOLD_DAYS: int | None = 1
# ---------------------------------------------------------------------------


# ------------------------------------------------------------------ targets


def buffered_selection(s: np.ndarray, k: int, buffer: float,
                       hold_days: float) -> tuple[np.ndarray, dict]:
    """The selection rule itself: a boolean (n x A) membership matrix.

    STRICTLY CAUSAL BY CONSTRUCTION -- a forward loop whose state at bar ``i``
    depends on rows ``<= i`` and nothing else. No mean, std, quantile, scaler
    or `.rank(pct=True)` is taken over the whole series anywhere here; the
    buffer is in RAW score units precisely so that no normalization is needed.
    The 60% truncation probe in :func:`cmd_checks` verifies this rather than
    this docstring.

    Also returns an event ledger, because the round's question is *which*
    channel carries the turnover:
      forced_exit  incumbent's score went non-positive (never timer-blocked)
      entry        an empty slot filled, including a slot a forced exit just
                   freed and a re-entry from flat
      swap         a VOLUNTARY, buffered, time-gated replacement
      blocked_*    voluntary swaps the buffer or the timer refused
    """
    n, n_assets = s.shape
    eligible = np.isfinite(s) & (s > 0.0)
    hold_bars = int(round(float(hold_days) * BARS_PER_DAY))
    buf = float(buffer)

    sel = np.zeros((n, n_assets), dtype=bool)
    held: list[int] = []
    last_change = -(1 << 60)
    ev = {"forced_exit": 0, "entry": 0, "swap": 0,
          "blocked_by_timer": 0, "blocked_by_buffer": 0, "flat_bars": 0}

    for i in range(n):
        row = s[i]
        elig = eligible[i]
        changed = False

        # (a) forced exits -- never blocked by the timer.
        if held:
            keep = [a for a in held if elig[a]]
            if len(keep) != len(held):
                ev["forced_exit"] += 1
                held = keep
                changed = True

        # entries into empty slots (including refilling a slot a forced exit
        # just freed, and re-entering from flat). Allowed immediately: sitting
        # flat while a positive-score asset is available is not the hypothesis
        # under test, and it is not R-63's rule either.
        if len(held) < k:
            free = [a for a in range(n_assets) if elig[a] and a not in held]
            if free:
                free.sort(key=lambda a: -row[a])
                for a in free[: k - len(held)]:
                    held.append(a)
                    changed = True
                ev["entry"] += 1

        # (b) voluntary swap -- buffered AND time-gated.
        elif not changed:
            worst = min(held, key=lambda a: row[a])
            out = [a for a in range(n_assets) if elig[a] and a not in held]
            if out:
                best = max(out, key=lambda a: row[a])
                if row[best] > row[worst] + buf:
                    if (i - last_change) >= hold_bars:
                        held.remove(worst)
                        held.append(best)
                        changed = True
                        ev["swap"] += 1
                    else:
                        ev["blocked_by_timer"] += 1
                elif row[best] > row[worst]:
                    ev["blocked_by_buffer"] += 1

        if changed:
            last_change = i
        if held:
            sel[i, held] = True
        else:
            ev["flat_bars"] += 1

    return sel, ev


def build_buffered_targets(aligned: dict[str, pd.DataFrame], k: int,
                           buffer: float, hold_days: float) -> pd.DataFrame:
    """R-63's novel arm with :func:`buffered_selection` in place of its
    recompute-every-bar top-k rule. Everything below the selection -- the
    score, the vol scale, the 0.10 deadband, the equal weighting, the 1.0
    cap -- is R-63's, untouched."""
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    n = s.shape[0]

    sel, _ = buffered_selection(s, k, buffer, hold_days)

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


def buffered_fn(k: int, buffer: float, hold_days: float):
    return lambda aligned: build_buffered_targets(aligned, k, buffer, hold_days)


def r63_fn(k: int):
    return lambda aligned: r63_baseline_targets(aligned, k)


# ------------------------------------------------------------------ cells


def build_cell(frames, universe, window, targets_fn):
    """Aligned prices + targets, both sliced to the evaluation window.

    The right edge is applied STRICTLY (``idx < end + 1 day``), independently
    of the shared `_hi` helper, which is the guard R-63's conservative branch
    added after finding that helper admitted one bar of the reserved holdout.
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

    `holding_period_days` in the shared file measures the time between changes
    to the held SET, which for a long/flat arm counts a flat spell as a
    "holding". This measures the thing the `hold_days` parameter is actually
    supposed to lengthen: once you own an asset, how long do you own it.
    """
    held = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0) > 0.0
    starts = int(held[0].sum() + (held[1:] & ~held[:-1]).sum())
    return float(held.sum()) / max(starts, 1) / BARS_PER_DAY


def raw_turnover(targets: pd.DataFrame) -> dict:
    """R-63's OWN turnover convention (no deadband), so the frontier's
    reference row is comparable to the published 3.44/day and 2.86 changes/day.

    `turnover_stats` in the shared file applies the 5% band and is the number
    the simulator actually pays; both are reported.
    """
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
    """`volmatched_hold_equity`, with a documented WORKAROUND for a bug in the
    frozen shared file. The shared file is NOT edited.

    BUG (r65_shared.py lines 358-368). The iteration's cap-binding early
    return is evaluated in the wrong order:

        c = clip(c * (target/vol), 1e-3, 1.0)
        eq = simulate(...); vol = realized_vol(eq)
        if c >= 1.0 and vol < target_vol:      # <-- fires FIRST
            return eq, c, vol, False
        # loop head's `abs(vol-target) <= tol*target` never sees this vol

    So whenever the long-only cap binds, the freshly computed `vol` is
    reported as UNMATCHED without ever being tested against the tolerance --
    even when it is inside it. Observed on the R-63 reference row on W_TRAIN:
    candidate vol 0.9533, benchmark vol at c=1.0 of 0.9451, a gap of 0.86%
    against a 2% tolerance, returned matched=False. That is a spurious VOID of
    a cell that is in fact matched, and it would have thrown away the
    frontier's own reference point.

    The workaround applies the shared function's OWN final-line criterion,
    `abs(vol - target_vol) <= tol * target_vol`, to the (eq, c, vol) it
    returned. No threshold is invented and no equity curve is altered: only
    the boolean is recomputed. BOTH flags are written to every CSV row
    (`volmatch_matched_*` is the corrected one used for scoring,
    `volmatch_shared_flag_*` is what the frozen function returned) so the
    operator can see exactly where the two differ.
    """
    eq, c, vol, shared_flag = volmatched_hold_equity(cand_eq, aligned, assets,
                                                     market, tol=VOLMATCH_TOL)
    target = realized_vol(cand_eq)
    matched = bool(eq is not None and np.isfinite(vol) and np.isfinite(target)
                   and target > 0 and abs(vol - target) <= VOLMATCH_TOL * target)
    if shared_flag != matched:
        print(f"    ~~ shared volmatched_hold_equity flag OVERRIDDEN{label}: "
              f"shared said matched={shared_flag}, tolerance test says "
              f"{matched} (c={c:.4f} bench_vol={vol:.4f} cand_vol={target:.4f}, "
              f"gap {abs(vol - target) / target:.3%} vs tol {VOLMATCH_TOL:.0%})")
    if not matched:
        print(f"    !! VOLMATCH_HOLD did NOT match{label}: c={c:.4f} "
              f"bench_vol={vol:.4f} cand_vol={target:.4f} "
              f"(gap {abs(vol - target) / max(target, 1e-12):.3%}) "
              f"-> cell is VOIDED, not scored")
    return eq, c, vol, matched, shared_flag


def measure_pair(targets, aligned, assets, window_name, universe_name, params,
                 arm=ARM):
    """The frontier's unit of work: one grid cell, both fee levels, both
    against VOLMATCH_HOLD computed at that fee level."""
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


def fmt_front(row):
    return (f"    set {row['hold_days']:6.2f}d ten {row['tenure_days']:6.2f}d"
            f" | turn {row['turnover_per_day']:6.3f}"
            f" (raw {row['raw_turnover_per_day']:6.3f}) | chg/d "
            f"{row['membership_changes_per_day']:6.3f} | mtn "
            f"{row['mean_notional']:.3f} | GROSS {row['gross_growth_diff']:+8.3f}"
            f" [{row['gross_growth_lo']:+7.3f},{row['gross_growth_hi']:+7.3f}]"
            f" | NET {row['net_growth_diff']:+8.3f}"
            f" [{row['net_growth_lo']:+7.3f},{row['net_growth_hi']:+7.3f}]")


# ------------------------------------------------------------------ checks


def truncation_probe(frames, k, buffer, hold_days, frac=0.6) -> bool:
    """The instructed gate: build targets on the first 60% of bars and on
    100%, and require the first 60% of rows to agree EXACTLY.

    Catches any whole-series statistic applied to early rows, and (because the
    selection here is a stateful forward loop) any accidental peek at bar t+1.
    """
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    n = len(next(iter(warm.values())))
    cut = int(n * frac)
    full = build_buffered_targets(warm, k, buffer, hold_days).to_numpy()
    trunc = build_buffered_targets({t: df.iloc[:cut] for t, df in warm.items()},
                                   k, buffer, hold_days).to_numpy()
    a = np.nan_to_num(full[:cut], nan=0.0)
    b = np.nan_to_num(trunc[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0))


def tail_perturbation_probe(frames, k, buffer, hold_days, frac_tail=0.4) -> bool:
    """R-63's complementary probe: multiply the TAIL of every price series by
    10 and require the EARLY rows to be bit-identical. Truncation removes the
    tail; this corrupts it, and a whole-series statistic that ignores length
    but not content is only caught by the second."""
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
    a = np.nan_to_num(build_buffered_targets(warm, k, buffer, hold_days)
                      .to_numpy()[:cut], nan=0.0)
    b = np.nan_to_num(build_buffered_targets(bad, k, buffer, hold_days)
                      .to_numpy()[:cut], nan=0.0)
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0))


def cmd_checks(frames, k=K_FIXED):
    print("== correctness gates ==")
    ok = True

    t0 = time.time()
    c1 = check_causality(lambda a: build_buffered_targets(a, k, 0.05, 7),
                         align_frames({t: frames[t] for t in UNIVERSE_8},
                                      warm_window(W_TRAIN)))
    print(f"  check_causality (r63 truncation probe, buffer=0.05 H=7): {c1}"
          f"  [{time.time() - t0:.1f}s]")
    ok &= c1

    for (b, h) in ((0.00, 1), (0.10, 30)):
        p = truncation_probe(frames, k, b, h)
        print(f"  truncation_probe 60% (buffer={b:.2f} H={h}): {p}")
        ok &= p

    p2 = tail_perturbation_probe(frames, k, 0.05, 7)
    print(f"  tail x10 perturbation probe (buffer=0.05 H=7): {p2}")
    ok &= p2

    # Gate 4: does the rule respond to its parameters at all?
    print("  holding-period response (W_TRAIN, U8, full grid corners + mid):")
    rows = []
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(W_TRAIN))
    score = cross_sectional_score(warm)
    s_all = score.to_numpy(dtype=float)
    lo = pd.Timestamp(W_TRAIN[0], tz="UTC")
    hi = pd.Timestamp(W_TRAIN[1], tz="UTC") + pd.Timedelta(days=1)
    keep = (score.index >= lo) & (score.index < hi)
    for (b, h) in ((0.00, 1), (0.00, 7), (0.00, 30),
                   (0.10, 1), (0.10, 7), (0.10, 30)):
        sel, ev = buffered_selection(s_all, k, b, h)
        tg = build_buffered_targets(warm, k, b, h).loc[keep]
        days = len(tg) / BARS_PER_DAY
        hp = holding_period_days(tg)
        ten = mean_tenure_days(tg)
        rt = raw_turnover(tg)
        ts = turnover_stats(tg)
        print(f"    buffer={b:.2f} H={h:2d}: set_period={hp:6.2f}d  "
              f"tenure={ten:6.2f}d  raw_turn={rt['raw_turnover_per_day']:.3f}/d"
              f"  banded_turn={ts['turnover_per_day']:.3f}/d  || events/day:"
              f" swap={ev['swap'] / days:.3f} forced_exit="
              f"{ev['forced_exit'] / days:.3f} entry={ev['entry'] / days:.3f}"
              f" blocked_timer={ev['blocked_by_timer'] / days:.3f}"
              f" blocked_buffer={ev['blocked_by_buffer'] / days:.3f}")
        rows.append({"check": "holding_period_response", "k": k, "buffer": b,
                     "hold_days": h, "measured_set_period_days": hp,
                     "measured_tenure_days": ten,
                     **{f"ev_{key}": v for key, v in ev.items()}, **rt, **ts})

    # The instructed gate is "does the measured holding period RESPOND to the
    # parameter". Two responses are required and neither threshold is tuned
    # after the fact: voluntary swaps must fall by at least 3x from H=1 to
    # H=30, and mean tenure must be strictly monotone in hold_days at both
    # buffer corners.
    swap1, swap30 = rows[3]["ev_swap"], rows[5]["ev_swap"]
    resp_swap = swap30 < swap1 / 3.0
    ten0 = [rows[0]["measured_tenure_days"], rows[1]["measured_tenure_days"],
            rows[2]["measured_tenure_days"]]
    ten1 = [rows[3]["measured_tenure_days"], rows[4]["measured_tenure_days"],
            rows[5]["measured_tenure_days"]]
    resp_ten = all(b > a for a, b in zip(ten0, ten0[1:])) and \
        all(b > a for a, b in zip(ten1, ten1[1:]))
    print(f"  voluntary swaps respond to hold_days (H=30 < H=1/3): {resp_swap}"
          f"  [{swap1} -> {swap30} over the window]")
    print(f"  mean tenure strictly monotone in hold_days at both buffers: "
          f"{resp_ten}  [{ten0[0]:.2f}->{ten0[-1]:.2f}d, "
          f"{ten1[0]:.2f}->{ten1[-1]:.2f}d]")

    # SATURATION, reported loudly rather than tuned away. The parameters can
    # only reach the VOLUNTARY channel. R-63's own "hold only positive-scoring
    # assets, flat otherwise" rule -- frozen for this round, inherited
    # byte-for-byte -- forces an exit every time the incumbent's score crosses
    # zero, and that channel is invariant across the whole grid. It is what
    # caps tenure near 2 days no matter how large `hold_days` is.
    floor = rows[5]["ev_forced_exit"] / (len(tg) / BARS_PER_DAY)
    print(f"  !! SATURATION: forced exits are {floor:.3f}/day and are FLAT "
          f"across the grid ({rows[0]['ev_forced_exit']} .. "
          f"{rows[5]['ev_forced_exit']} events). The buffer and the timer "
          f"cannot reach that channel; it is R-63's frozen positive-score "
          f"long/flat gate, and it sets a hard floor on this arm's turnover.")
    ok &= resp_swap and resp_ten

    # index hygiene: no W_HOLD bar can reach a selection-window cell.
    for wname, window, uni in (("W_TRAIN", W_TRAIN, UNIVERSE_8),
                               ("W_VAL", W_VAL, UNIVERSE_8)):
        _, tg, warm_ok = build_cell(frames, uni, window, buffered_fn(k, 0.05, 7))
        bad = int((tg.index >= pd.Timestamp("2023-01-01", tz="UTC")).sum())
        print(f"  {wname}: {len(tg):,} bars {tg.index[0]} -> {tg.index[-1]}"
              f"  bars dated >=2023-01-01: {bad}  first_bar_warm={warm_ok}")
        rows.append({"check": f"index_{wname}", "n_bars": len(tg),
                     "first": str(tg.index[0]), "last": str(tg.index[-1]),
                     "bars_in_holdout": bad, "first_bar_warm": warm_ok})
        ok &= (bad == 0) and warm_ok

    write_csv(OUT_DIR / "conservative_checks.csv", rows)
    print(f"  ALL GATES PASS: {ok}")
    return ok


# ------------------------------------------------------------------ repro


def cmd_repro(frames, k=K_FIXED):
    """Reproduce R-63's published reference point on W_FULL6 / U6 / 0.10%.

    Against MATCHED_HOLD, which is what R-63 published, so the numbers are
    directly comparable. A mismatch means the substrate drifted and the
    round's comparability is gone.
    """
    print("== R-63 reference reproduction: W_FULL6, U6, k=1, 0.10% ==")
    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, r63_fn(k))
    rt = raw_turnover(targets)
    ts = turnover_stats(targets)
    c = mean_total_notional(targets)
    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                            aligned, SPOT_BASE)
    r = compare(cand, mh)
    cand0 = simulate_portfolio(targets, aligned, SPOT_FREE)
    mh0 = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                             aligned, SPOT_FREE)
    r0 = compare(cand0, mh0)

    print(f"  bars {len(targets):,}  {targets.index[0]} -> {targets.index[-1]}"
          f"  first_bar_warm={warm_ok}")
    print(f"  raw turnover/day    {rt['raw_turnover_per_day']:.3f}   "
          f"(R-63 published {R63_TURNOVER_PER_DAY})")
    print(f"  membership chg/day  {rt['membership_changes_per_day']:.3f}   "
          f"(R-63 published 2.86)")
    print(f"  banded turnover/day {ts['turnover_per_day']:.3f}   "
          f"(deadband-aware; R-63 published the raw number)")
    print(f"  net growth vs MATCHED_HOLD  {r['growth_diff']:+.4f} "
          f"[{r['growth_lo']:+.3f}, {r['growth_hi']:+.3f}]   "
          f"(R-63 published {R63_NET_D1})")
    print(f"  gross growth vs MATCHED_HOLD {r0['growth_diff']:+.4f} "
          f"[{r0['growth_lo']:+.3f}, {r0['growth_hi']:+.3f}]   "
          f"(R-63 published +0.480 [-2.58, +3.65])")
    print(f"  mean total notional {c:.4f}  (R-63 published 0.525)")
    print(f"  cand final {r['cand_final']:,.4f} vs matched {r['bench_final']:,.2f}"
          f"  (R-63 published $1.44)")

    row = {"arm": "r63_reference", "window": "W_FULL6", "universe": "U6",
           "k": k, "bench": "MATCHED_HOLD", "n_bars": len(targets),
           "mean_notional": c, **rt, **ts,
           "net_growth_diff": r["growth_diff"], "net_growth_lo": r["growth_lo"],
           "net_growth_hi": r["growth_hi"], "net_dd_diff": r["dd_diff"],
           "gross_growth_diff": r0["growth_diff"],
           "gross_growth_lo": r0["growth_lo"], "gross_growth_hi": r0["growth_hi"],
           "cand_final": r["cand_final"], "bench_final": r["bench_final"],
           "cand_dd": r["cand_dd"], "bench_dd": r["bench_dd"],
           "published_turnover_per_day": R63_TURNOVER_PER_DAY,
           "published_net_d1": R63_NET_D1, "published_gross": 0.480,
           "turnover_reproduced": abs(rt["raw_turnover_per_day"]
                                      - R63_TURNOVER_PER_DAY) < 0.10,
           "net_d1_reproduced": abs(r["growth_diff"] - R63_NET_D1) < 0.10}
    print(f"  turnover reproduced: {row['turnover_reproduced']}   "
          f"net D1 reproduced: {row['net_d1_reproduced']}")

    # ---- Is the D5 bar attainable by the signal it was calibrated from? ----
    # D5 is stated as "gross growth vs VOLMATCH_HOLD >= +0.240", but +0.480 --
    # the number it is half of -- was measured by R-63 against MATCHED_HOLD.
    # VOLMATCH_HOLD is a strictly HARDER benchmark for a concentrated arm: it
    # holds MORE notional (c up to 1.0) in order to reach the candidate's own
    # realized volatility. So the two are not the same yardstick, and whether
    # R-63's own arm clears +0.240 against the new one is a fact about the
    # BAR, not about this branch's candidate. Measured here, on the reference
    # arm, before any candidate D-cell is read.
    ref_row, ref_st = measure_pair(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                                   {"buffer": float("nan"),
                                    "hold_days_param": 0, "k": k},
                                   arm="r63_reference")
    ref_row["config_kind"] = "reference_vs_volmatch"
    ref_row["d5_bar"] = D5_BAR
    ref_row["d5_pass"] = d5_pass(ref_row) and ref_st["matched_gross"]
    ref_row["note"] = ("D5-bar attainability: R-63's own arm scored against "
                       "VOLMATCH_HOLD, the benchmark D5 is stated against")
    print("  [R-63 reference vs VOLMATCH_HOLD on the same W_FULL6 cell]")
    print(fmt_front(ref_row))
    print(f"    R-63's own arm vs the D5 bar (+{D5_BAR:.3f}): gross "
          f"{ref_row['gross_growth_diff']:+.4f} -> D5 would "
          f"{'PASS' if ref_row['d5_pass'] else 'FAIL'}")
    print(f"    (+0.480 was measured against MATCHED_HOLD at c=0.525; "
          f"VOLMATCH_HOLD sits at c={ref_row['volmatch_c_gross']:.3f})")

    write_csv(OUT_DIR / "conservative_repro.csv", [row, ref_row])
    return row


# ------------------------------------------------------------------ frontier


def cmd_frontier(frames, k=K_FIXED, windows=("W_TRAIN", "W_VAL")):
    """The round's deliverable: every grid cell, both fee levels, on the two
    SELECTION windows. No decision-window read happens here."""
    print(f"== FRONTIER: {len(BUFFER_GRID)}x{len(HOLD_GRID)} grid + R-63 "
          f"reference, k={k}, U8, vs VOLMATCH_HOLD ==")
    wmap = {"W_TRAIN": W_TRAIN, "W_VAL": W_VAL}
    rows = []
    for wname in windows:
        window = wmap[wname]
        print(f"  -- {wname} --")
        t0 = time.time()
        aligned, tg63, warm_ok = build_cell(frames, UNIVERSE_8, window, r63_fn(k))
        if not warm_ok:
            raise RuntimeError(f"{wname}: first evaluated bar not warm")
        row, _ = measure_pair(tg63, aligned, UNIVERSE_8, wname, "U8",
                              {"buffer": float("nan"), "hold_days_param": 0,
                               "k": k}, arm="r63_reference")
        row["config_kind"] = "reference"
        rows.append(row)
        print(f"  [R-63 reference k={k}]")
        print(fmt_front(row))

        for hd in HOLD_GRID:
            for bf in BUFFER_GRID:
                aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, window,
                                                  buffered_fn(k, bf, hd))
                if not warm_ok:
                    raise RuntimeError(f"{wname} b={bf} H={hd}: not warm")
                row, _ = measure_pair(tg, aligned, UNIVERSE_8, wname, "U8",
                                      {"buffer": bf, "hold_days_param": hd,
                                       "k": k})
                row["config_kind"] = "grid"
                rows.append(row)
                print(f"  [buffer={bf:.2f} H={hd:2d}]")
                print(fmt_front(row))
        print(f"  {wname} done in {time.time() - t0:.0f}s "
              f"(config_count={config_count()})")

    write_csv(OUT_DIR / "conservative_frontier.csv", rows)
    return rows


def cmd_select(frames=None, rows=None):
    """Report the selection, the W_TRAIN ordering, and the neighbourhood.

    Reads the frontier CSV; runs nothing. The criterion is the one declared in
    the module docstring and was fixed before the sweep ran.
    """
    path = OUT_DIR / "conservative_frontier.csv"
    df = pd.read_csv(path)
    grid = df[df["config_kind"] == "grid"].copy()

    # The frontier's own question: as turnover falls, does GROSS edge fall
    # faster or slower than COST? `cost` is the log units the fee schedule
    # takes from the candidate RELATIVE to the same benchmark, i.e.
    # gross_diff - net_diff, measured on the identical pair of equity curves.
    df["cost"] = df["gross_growth_diff"] - df["net_growth_diff"]

    out = []
    for wname in ("W_TRAIN", "W_VAL"):
        ref = df[(df["window"] == wname) & (df["config_kind"] == "reference")]
        sub = df[(df["window"] == wname) & (df["config_kind"] == "grid")] \
            .sort_values("net_growth_diff", ascending=False)
        print(f"== {wname}: value and cost vs VOLMATCH_HOLD "
              f"(ordered by net growth diff) ==")
        print("   buffer   H | turn/d  ten(d) |    GROSS |    COST |      NET "
              "| net 95% interval        | dd_diff")
        for _, r in pd.concat([ref, sub]).iterrows():
            tag = ("R-63 ref " if r["config_kind"] == "reference"
                   else f"  {r['p_buffer']:.2f} {int(r['p_hold_days_param']):3d}")
            print(f"  {tag} | {r['turnover_per_day']:6.3f} {r['tenure_days']:6.2f}"
                  f" | {r['gross_growth_diff']:+8.3f} | {r['cost']:7.3f} |"
                  f" {r['net_growth_diff']:+8.3f} | "
                  f"[{r['net_growth_lo']:+7.3f},{r['net_growth_hi']:+7.3f}] |"
                  f" {r['net_dd_diff']:+7.2f}")
        best_net = sub["net_growth_diff"].max()
        n_pos = int((sub["net_growth_diff"] > 0).sum())
        n_sig = int(((sub["net_growth_diff"] > 0) & (sub["net_growth_lo"] > 0)).sum())
        n_d5 = int((sub["gross_growth_diff"] >= D5_BAR).sum())
        print(f"  -> best net {best_net:+.4f}; cells with net>0: {n_pos}/20; "
              f"with net>0 AND interval excluding zero: {n_sig}/20; "
              f"cells clearing the D5 bar (+{D5_BAR:.3f} gross): {n_d5}/20")
        out.append(sub)

    tr, va = out
    key = ["p_buffer", "p_hold_days_param"]
    merged = tr[key + ["net_growth_diff"]].merge(
        va[key + ["net_growth_diff"]], on=key, suffixes=("_train", "_val"))
    # Spearman by hand: pandas' `method="spearman"` needs scipy, which is not
    # installed here. Rank then Pearson is the identical statistic.
    rho = merged["net_growth_diff_train"].rank().corr(
        merged["net_growth_diff_val"].rank())
    print(f"\n  Spearman rank correlation W_TRAIN vs W_VAL net growth: {rho:+.3f}")

    best = va.iloc[0]
    print(f"  W_VAL WINNER: buffer={best['p_buffer']:.2f} "
          f"H={int(best['p_hold_days_param'])}  net={best['net_growth_diff']:+.4f}"
          f"  gross={best['gross_growth_diff']:+.4f}")
    tr_rank = list(zip(tr["p_buffer"], tr["p_hold_days_param"])).index(
        (best["p_buffer"], best["p_hold_days_param"])) + 1
    print(f"  that cell's W_TRAIN rank: {tr_rank} of {len(tr)}")
    return best, rho


# ------------------------------------------------------------------ run


def _frozen(buffer=None, hold_days=None):
    b = FROZEN_BUFFER if buffer is None else buffer
    h = FROZEN_HOLD_DAYS if hold_days is None else hold_days
    if b is None or h is None:
        raise SystemExit("configuration is not frozen yet: run `frontier`, then "
                         "`select`, then set FROZEN_BUFFER / FROZEN_HOLD_DAYS")
    return float(b), int(h)


def cmd_run(frames, k=K_FIXED, buffer=None, hold_days=None):
    bf, hd = _frozen(buffer, hold_days)
    print(f"== D-CELLS: frozen buffer={bf:.2f} hold_days={hd} k={k} ==")
    rows = []

    # ---- D1 / D2 / D5 : W_FULL6, U6 ----------------------------------
    aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6,
                                           buffered_fn(k, bf, hd))
    if not warm_ok:
        raise RuntimeError("W_FULL6 first evaluated bar not warm")
    print(f"  W_FULL6 bars {len(targets):,}  {targets.index[0]} -> "
          f"{targets.index[-1]}  first_bar_warm={warm_ok}")

    row, st = measure_pair(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                           {"buffer": bf, "hold_days_param": hd, "k": k})
    row["config_kind"] = "decision"
    d1 = d1_pass(row) and st["matched_net"]
    d2 = d2_pass(row) and st["matched_net"]
    d5 = d5_pass(row) and st["matched_gross"]
    row["d1_pass"] = d1
    row["d2_pass"] = d2
    row["d5_pass"] = d5
    row["d5_bar"] = D5_BAR
    row["note"] = "D1/D2/D5 primary vs VOLMATCH_HOLD"
    rows.append(row)
    print(fmt_front(row))
    print(f"    cand_vol {row['cand_vol_net']:.3f} vs volmatch_vol "
          f"{row['volmatch_vol_net']:.3f} at c={row['volmatch_c_net']:.3f}  "
          f"matched={st['matched_net']}")
    print(f"    D1 PASS={d1}  D2 PASS={d2}  D5 PASS={d5} (bar {D5_BAR:+.3f}, "
          f"gross {row['gross_growth_diff']:+.4f})")
    if not st["matched_net"]:
        print("    !! D1/D2 VOIDED (vol match failed)")
    if not st["matched_gross"]:
        print("    !! D5 VOIDED (vol match failed)")

    # continuity: MATCHED_HOLD, EW_HOLD, BTC_HOLD on the same cell
    c = mean_total_notional(targets)
    mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                            aligned, SPOT_BASE)
    r_mh = compare(st["cand_net"], mh)
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "MATCHED_HOLD", "config_kind": "context",
                 "p_buffer": bf, "p_hold_days_param": hd, "p_k": k,
                 "mean_notional": c, "matched_hold_vol": realized_vol(mh),
                 "cand_vol_net": row["cand_vol_net"],
                 "net_growth_diff": r_mh["growth_diff"],
                 "net_growth_lo": r_mh["growth_lo"],
                 "net_growth_hi": r_mh["growth_hi"],
                 "net_dd_diff": r_mh["dd_diff"], "net_dd_lo": r_mh["dd_lo"],
                 "net_dd_hi": r_mh["dd_hi"], "cand_final": r_mh["cand_final"],
                 "bench_final": r_mh["bench_final"], "cand_dd": r_mh["cand_dd"],
                 "bench_dd": r_mh["bench_dd"], "n_days": r_mh["n_days"],
                 "note": "continuity: R-63's benchmark"})
    print(f"  [continuity vs MATCHED_HOLD c={c:.3f}] growth "
          f"{r_mh['growth_diff']:+.4f} [{r_mh['growth_lo']:+.3f}, "
          f"{r_mh['growth_hi']:+.3f}]  dd {r_mh['dd_diff']:+.2f}  "
          f"(matched_hold vol {realized_vol(mh):.3f} vs cand "
          f"{row['cand_vol_net']:.3f})")

    ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)
    r_ew = compare(st["cand_net"], ew)
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "EW_HOLD", "config_kind": "context",
                 "p_buffer": bf, "p_hold_days_param": hd, "p_k": k,
                 "net_growth_diff": r_ew["growth_diff"],
                 "net_growth_lo": r_ew["growth_lo"],
                 "net_growth_hi": r_ew["growth_hi"],
                 "net_dd_diff": r_ew["dd_diff"], "cand_final": r_ew["cand_final"],
                 "bench_final": r_ew["bench_final"], "cand_dd": r_ew["cand_dd"],
                 "bench_dd": r_ew["bench_dd"], "n_days": r_ew["n_days"],
                 "note": "context: vs EW_HOLD @0.10%"})
    print(f"  [context vs EW_HOLD] {r_ew['cand_final']:,.2f} vs "
          f"{r_ew['bench_final']:,.2f}  growth {r_ew['growth_diff']:+.4f}")

    btc = frames["BTC"]
    btc_on = btc.reindex(btc.index.union(targets.index)).ffill().reindex(targets.index)
    btc_eq = static_hold_equity({"BTC": btc_on}, ["BTC"], SPOT_BASE)
    r_btc = compare(st["cand_net"], btc_eq)
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "BTC_HOLD", "config_kind": "context",
                 "p_buffer": bf, "p_hold_days_param": hd, "p_k": k,
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

    # ---- D4 : W_FULL6, 0.40% vs EW_HOLD ------------------------------
    cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
    ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
    r40 = compare(cand40, ew40)
    d4 = r40["cand_final"] > r40["bench_final"]
    rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                 "bench": "EW_HOLD", "config_kind": "decision",
                 "p_buffer": bf, "p_hold_days_param": hd, "p_k": k,
                 "fee": 0.004, "d4_pass": d4,
                 "net_growth_diff": r40["growth_diff"],
                 "net_growth_lo": r40["growth_lo"],
                 "net_growth_hi": r40["growth_hi"],
                 "net_dd_diff": r40["dd_diff"], "cand_final": r40["cand_final"],
                 "bench_final": r40["bench_final"], "cand_dd": r40["cand_dd"],
                 "bench_dd": r40["bench_dd"], "n_days": r40["n_days"],
                 "note": "D4 cost tier 0.40% vs EW_HOLD"})
    print(f"  [D4 @0.40%] cand {r40['cand_final']:,.2f} vs EW_HOLD "
          f"{r40['bench_final']:,.2f} -> D4 PASS={d4}")

    # ---- D3 : W_VAL, U8, 0.10% ---------------------------------------
    aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL,
                                           buffered_fn(k, bf, hd))
    if not warm3:
        raise RuntimeError("W_VAL first evaluated bar not warm")
    row3, st3 = measure_pair(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                             {"buffer": bf, "hold_days_param": hd, "k": k})
    row3["config_kind"] = "decision"
    d3 = d3_pass(row3) and st3["matched_net"]
    row3["d3_pass"] = d3
    row3["note"] = "D3 inner-validation vs VOLMATCH_HOLD"
    rows.append(row3)
    print("  [D3 W_VAL U8]")
    print(fmt_front(row3))
    print(f"    D3 PASS={d3}  (matched={st3['matched_net']})")

    write_csv(OUT_DIR / "conservative_cells.csv", rows)
    return {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5,
            "row": row, "targets": targets, "aligned": aligned,
            "vm": st["vm_net"], "real": row["net_growth_diff"],
            "matched_net": st["matched_net"], "matched_gross": st["matched_gross"],
            "buffer": bf, "hold_days": hd, "k": k}


# ------------------------------------------------------------------ scramble


def cmd_scramble(frames, k=K_FIXED, buffer=None, hold_days=None, state=None):
    bf, hd = _frozen(buffer, hold_days)
    print(f"== FALSIFICATION: cross-section scramble, seeds 0..9, D1 cell, "
          f"buffer={bf:.2f} H={hd} ==")
    if state is None:
        aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6,
                                               buffered_fn(k, bf, hd))
        if not warm_ok:
            raise RuntimeError("W_FULL6 first evaluated bar not warm")
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        vm, _, _, ok, _ = volmatch(cand, aligned, UNIVERSE_6, SPOT_BASE)
        real = compare(cand, vm)["growth_diff"]
    else:
        aligned, targets, vm = state["aligned"], state["targets"], state["vm"]
        real = state["real"]

    rows, diffs = [], []
    for seed in SCRAMBLE_SEEDS:
        stg = scramble_targets(targets, seed)
        eq = simulate_portfolio(stg, aligned, SPOT_BASE)
        r = compare(eq, vm)
        diffs.append(r["growth_diff"])
        rows.append({"arm": f"{ARM}_scrambled", "seed": seed, "window": "W_FULL6",
                     "universe": "U6", "bench": "VOLMATCH_HOLD", "fee": 0.001,
                     "p_buffer": bf, "p_hold_days_param": hd, "p_k": k,
                     "mean_notional": mean_total_notional(stg),
                     **{key: r[key] for key in
                        ("cand_final", "bench_final", "cand_dd", "bench_dd",
                         "growth_diff", "growth_lo", "growth_hi", "dd_diff",
                         "dd_lo", "dd_hi", "n_days")}})
        print(f"  seed {seed}: growth_diff {r['growth_diff']:+.4f}  final "
              f"{r['cand_final']:>12,.4f}  dd {r['cand_dd']:5.1f}%")

    p90 = float(np.percentile(diffs, 90))
    survived = bool(real > p90)
    rows.append({"arm": ARM, "seed": -1, "window": "W_FULL6", "universe": "U6",
                 "bench": "VOLMATCH_HOLD", "fee": 0.001, "p_buffer": bf,
                 "p_hold_days_param": hd, "p_k": k, "growth_diff": real,
                 "mean_notional": mean_total_notional(targets),
                 "scramble_p90": p90, "scramble_survived": survived,
                 "n_better": int(sum(d >= real for d in diffs))})
    print(f"  real growth_diff {real:+.4f} vs scramble p90 {p90:+.4f} -> "
          f"SURVIVED={survived}  ({sum(d >= real for d in diffs)} of 10 "
          f"scrambles did better)")
    write_csv(OUT_DIR / "conservative_scramble.csv", rows)
    return survived


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "repro", "frontier", "select",
                                    "run", "scramble", "all"])
    ap.add_argument("--k", type=int, default=K_FIXED)
    ap.add_argument("--buffer", type=float, default=None)
    ap.add_argument("--hold-days", type=int, default=None)
    ap.add_argument("--windows", default="W_TRAIN,W_VAL")
    args = ap.parse_args()

    if args.cmd == "select":
        cmd_select()
        return

    frames = load_universe(UNIVERSE_8)

    if args.cmd == "checks":
        cmd_checks(frames, args.k)
    elif args.cmd == "repro":
        cmd_repro(frames, args.k)
    elif args.cmd == "frontier":
        cmd_frontier(frames, args.k, tuple(args.windows.split(",")))
    elif args.cmd == "run":
        cmd_run(frames, args.k, args.buffer, args.hold_days)
    elif args.cmd == "scramble":
        cmd_scramble(frames, args.k, args.buffer, args.hold_days)
    else:
        cmd_checks(frames, args.k)
        cmd_repro(frames, args.k)
        st = cmd_run(frames, args.k, args.buffer, args.hold_days)
        surv = cmd_scramble(frames, args.k, st["buffer"], st["hold_days"], st)
        fw = further_work(st["d1"], st["d2"], st["d3"], st["d5"], surv)
        print(f"\n== further_work(d1={st['d1']}, d2={st['d2']}, d3={st['d3']}, "
              f"d5={st['d5']}, scramble={surv}) = {fw} ==")
        print("  -> STOP. Report to the operator; the holdout read is theirs."
              if fw else "  -> DONE. W_HOLD is NOT read.")

    print(f"\nconfig_count() = {config_count()}")


if __name__ == "__main__":
    main()
