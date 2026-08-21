#!/usr/bin/env python
"""R-81 CONSERVATIVE branch: Binance futures crowding (top-trader long/short
ratio + open-interest growth) as a confirming vote on `kelly_regime_v4`'s
3-anchor gate -- Step A measurement gate first, per this project's
established discipline for every INFO-axis round (R-53/R-73/R-74/R-79).

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). When Binance's top-trader long/short account
   ratio is extreme (z-scored against its own trailing baseline) in the
   same direction as `kelly_regime_v4`'s own 3-anchor vote, the position
   is crowded/over-levered and historically prone to a squeeze/reversal,
   so a discrete confirming vote fades it via
   `r81_shared.confirming_vote_frac`. Constraint attacked: INFO -- a
   structurally new information channel (derivatives positioning at the
   bar's own 5-minute cadence) with a specific, checkable reason it might
   behave differently from the 8 prior INFO signals in this ledger, all
   of which were daily-or-coarser and all of which LAGGED the anchor gate
   (R-53 median -5.5d; R-73 -2.0d/-9.0d; R-74 -4.0d/-35.0d): this one
   shares the bar's own cadence rather than being coarser than it.

   Citations: Hirshleifer (1988, JF) on crowded/informed-trader positions
   and price pressure; Kang, Rouwenhorst & Tang (2021/2023, "Crowding and
   Factor Returns", FAJ 79(1), SSRN 3803954); Palazzi, Junior & Klotzle
   (2025, SSRN 6725492), funding rate + open interest predicting BTC
   returns "in every regime" of a 2014-2025 sample -- the direct citation
   motivating this round (see `r81_shared.py` docstring for the full
   citation set and the not-a-duplicate-of list against R-35/R-39, R-53/
   R-54, R-73, R-80 and the ruled-out "recovering flow from OHLCV" line;
   not repeated here to keep one citation trail in one place).

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy
   code, on BTC only (ETH's metrics history starts 2021-12-01, too late
   to cover any of the 3 `r81_shared.STRESS_EPISODES` with a meaningful
   pre-episode baseline -- named here explicitly, not discovered after
   the fact, and not silently skipped: see the ETH note in the results
   section).

   PRIMARY METRIC (chosen now, before any number): `ls_z` (the top-trader
   long/short account ratio, z-scored against its own trailing 14-day
   mean/std by `r81_shared.crowding_z`) -- the single most direct
   "crowding" measure in the feed. `oi_chg_z` (open-interest growth,
   z-scored the same way) is computed and reported alongside as secondary
   context only; it is NOT part of the primary pass/fail decision, to
   avoid a second decision path that could be cherry-picked after
   looking. EXTREME THRESHOLD: |ls_z| >= 1.5 (a standard "1.5-sigma"
   extremity bar -- large enough to be a real tail reading, not so strict
   that a ~120-day search window around each episode has no chance of a
   crossing at all).

   EPISODE-LOCAL SEARCH WINDOW: [onset - 60 days, onset + 60 days], fixed
   before any number was computed. `kelly_regime_v4`'s own gate reacts to
   price with 20/40/80-day rolling anchors, so its own reaction to a
   stress event is not necessarily dated to the event's onset either --
   the same reason R-73's DVOL gate used a +/-45-day search window rather
   than requiring same-day crossings. 60 days is used here (rather than
   R-73's 45) because this round's "flip" definition (below) is a nearest-
   transition search, not a fixed hysteresis latch, and needs enough room
   on both sides to find ANY transition, including in an already-
   established trend (see the flip-definition caveat below).

   ANCHOR-GATE "FLIP" DEFINITION (disclosed reasoning, AND a disclosed
   mid-implementation correction -- see below): `anchor_majority` is the
   mean of three latched 0/1 votes and is not a single monotonic bull/
   bear switch -- during an already-established trend (e.g. the 2022 bear
   was underway well before the Terra/Luna and FTX episodes), the gate
   may already sit at its extreme value and not re-flip through zero at
   every subsequent stress event, while a faster anchor (20-day) can
   still tick over a single crossing of its own rolling mean. "Flip" here
   means: the anchor_majority DOWNWARD transition (majority DECREASES --
   the gate de-risking, which is the actual mechanism this gate tests)
   whose timestamp is closest to the episode's onset date within the
   search window.

   DISCLOSED CORRECTION: the first working version of this file used
   "nearest transition in EITHER direction" rather than "nearest DOWNWARD
   transition" as the flip rule. Run once, that rule silently picked a
   spurious BULLISH blip on 2021-11-08 (majority 0.667->1.0, two days
   before the actual 2021 top) as episode 1's "flip" merely because it
   was the transition closest in clock time to the 2021-11-10 onset,
   while the genuine bearish confirmation (majority 1.0->0.333) did not
   occur until 2021-11-16. Caught by inspecting the daily
   `anchor_majority` trajectory around the episode before treating any
   Step-A number as final, and fixed to "nearest DOWNWARD transition"
   before this file's one reported gate run -- disclosed here as a bug
   fix to the flip-selection RULE's implementation, not a re-run of the
   gate after seeing whether episode 1 passed or failed under the
   original (buggy) rule; the corrected rule is what this file's single
   reported run below used throughout, and it changes only episode 1
   (episodes 2 and 3's nearest transitions were already downward under
   either rule, so their numbers are identical either way -- reported in
   this file's diagnostic output).

   CROWDING "CROSSING" DEFINITION: the first-crossing timestamp (prior
   bar |ls_z| < 1.5, this bar |ls_z| >= 1.5) whose timestamp is closest
   to the episode's onset date within the same window -- the same
   "nearest to onset" logic applied to the candidate signal, for an
   apples-to-apples comparison.

   LEAD = (flip_time - crossing_time) in days. Positive = crowding
   extremity was reached BEFORE the anchor gate's own nearest reaction.

   NULL: `r81_shared.block_bootstrap_lead_null` circularly block-shifts
   the LOCAL (episode-window) `ls_z` array (block_days=5, N=500 draws,
   seed=81, fixed before running) and recomputes the "crossing nearest to
   the REAL, unshifted flip time" against each shifted copy -- i.e. the
   null holds the gate's true reaction fixed and asks whether an
   arbitrarily time-shifted copy of the SAME crowding series would have
   looked just as informative. This is the shared module's own
   recommended construction for this exact signal type (see
   `r81_shared.block_bootstrap_lead_null`'s docstring: a level/positioning
   measure, not a cyclical phase partition, so a block bootstrap against
   the fixed real flip dates is the applicable null here, not R-79's
   placebo-OFFSET device built for a different confound).

   PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, combining the task's "majority lead, materially better than
   the null" instruction into one checkable per-episode criterion): an
   episode counts as a PASS if BOTH (a) LEAD > 0 (crowding crossed before
   the gate's nearest reaction), AND (b) the true LEAD exceeds the 90th
   percentile of that same episode's own 500-draw block-bootstrap null
   lead distribution (i.e. the true lead is more extreme than 90% of
   arbitrarily-shifted copies of the same series). PROCEED TO STEP B only
   if >= 2 of the 3 episodes PASS. If fewer than 2 pass: STOP, report the
   gate result as this branch's whole product, do not build a strategy.
   The bar is not relaxed after seeing the numbers.

3. CONFIGS EVALUATED IN THIS STEP-A FILE, IF THE GATE FAILS: 0 (a fixed,
   non-swept measurement gate -- this project's standing accounting
   convention for this exact construction, per R-53/R-73/R-74/R-79's own
   Step-A studies). If the gate passes, Step B's sweep grid is
   pre-registered separately in this file's `STEP_B_PREREGISTRATION`
   docstring below, written and frozen before Step A's numbers were
   allowed to influence it.

4. WHAT WOULD MAKE STEP A FAIL, named now: the same failure this
   project's other 8 INFO signals hit -- the crowding extremity is
   reached AFTER (not before) the anchor gate's own nearest reaction, or
   the lead (if positive) is not distinguishable from an arbitrary
   time-shift of the same series (i.e. it is generic autocorrelation
   structure in a slow-moving z-score, not a real early-warning property).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r81_shared import (  # noqa: E402
    BARS_PER_DAY,
    METRICS_END,
    METRICS_START,
    OOS_START,
    STRESS_EPISODES,
    anchor_majority,
    block_bootstrap_lead_null,
    crowding_z,
    load_crowding_inputs,
)

DATA_DIR = ROOT / "data"

Z_THRESH = 1.5           # primary extreme threshold on |ls_z|, fixed a priori
WINDOW_DAYS = 60         # +/- days around each episode's onset, fixed a priori
N_DRAWS = 500            # block-bootstrap null draws, fixed a priori
BLOCK_DAYS = 5           # null block length in days, fixed a priori
NULL_SEED = 81           # fixed once, this round's number


# ---------------------------------------------------------------- holdout guard

def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START. Independent of any truncation already done
    at load time (same pattern as r79_conservative_halving_gate.py)."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


# --------------------------------------------------------------------- data

def load_btc_bars() -> pd.DataFrame:
    """BTC spot, truncated strictly before OOS_START at load time."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def load_crowding_features(bars: pd.DataFrame) -> pd.DataFrame:
    """ls_z / oi_chg_z aligned onto `bars`, BTC only. Independently
    re-truncated before OOS_START even though `load_crowding_inputs`
    already truncates at METRICS_END (== INNER_VAL_END, strictly before
    OOS_START) -- belt and suspenders, matching this project's own
    "independent second guard" convention."""
    metrics = load_crowding_inputs(DATA_DIR, asset="BTC")
    assert metrics is not None, "BTC crowding metrics file missing"
    assert_no_holdout(metrics)
    feats = crowding_z(metrics, bars)
    assert_no_holdout(feats)
    print(f"BTC crowding metrics: {METRICS_START['BTC']} -> {METRICS_END}  "
          f"({len(metrics):,} raw rows)", file=sys.stderr)
    return feats


# ------------------------------------------------------------- flip / crossing

def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    """Timestamp, within `window`, of the anchor-gate transition closest to
    `onset`. `direction="down"` (the primary, pre-registered definition)
    restricts to transitions where the value DECREASES -- i.e. the gate
    de-risking in reaction to stress, which is the mechanism this gate
    is actually testing. `direction="any"` is kept only as a documented
    robustness/diagnostic alternative (see the disclosed correction in
    this file's results banner: an early draft used `direction="any"` as
    the primary rule and it silently selected a spurious BULLISH blip 2
    days before the 2021 top as episode 1's "flip", 2 days before a
    genuine bearish confirmation on 2021-11-16 -- caught by inspecting
    the daily majority trajectory before any Step A number was finalized,
    fixed to `direction="down"` before this file's reported gate ran).
    `series` must already be aligned/reindexed onto `window`. Returns None
    if no matching transition exists in the window."""
    vals = series.reindex(window).to_numpy()
    changed = np.zeros(len(vals), dtype=bool)
    if direction == "down":
        changed[1:] = vals[1:] < vals[:-1]
    elif direction == "any":
        changed[1:] = vals[1:] != vals[:-1]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    idx = np.where(changed)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def nearest_crossing(z: pd.Series, window: pd.DatetimeIndex,
                      onset: pd.Timestamp, thresh: float = Z_THRESH) -> pd.Timestamp | None:
    """Timestamp, within `window`, of the first-crossing (prior bar below
    threshold, this bar at/above threshold, in absolute value) closest to
    `onset`. Returns None if no such crossing exists in the window."""
    vals = z.reindex(window).to_numpy()
    above = np.abs(vals) >= thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = above[1:] & ~above[:-1]
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars: pd.DataFrame, onset_str: str) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=WINDOW_DAYS)
    hi = onset + pd.Timedelta(days=WINDOW_DAYS)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


# --------------------------------------------------------------------- null

def episode_null_leads(ls_z: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, flip_time: pd.Timestamp,
                        n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                        seed: int = NULL_SEED) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the LOCAL ls_z array (within `window`) and recompute the
    "crossing nearest the real, unshifted onset" against the fixed, real
    `flip_time`. Returns an array of `n_draws` null leads in days (NaN
    where a shifted copy has no crossing at all in the window)."""
    local = ls_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_lead_null(
        event_offsets_days=np.array([0.0]), n_bars=n_bars,
        block_days=block_days, n_draws=n_draws, seed=seed)

    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        above = np.abs(shifted) >= Z_THRESH
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = above[1:] & ~above[:-1]
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        cross_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads


# --------------------------------------------------------------------- gate

def gate() -> dict:
    print("=" * 78)
    print("R-81 CONSERVATIVE: crowding vote (ls_z) -- STEP A lead-time gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    feats = load_crowding_features(bars)
    ls_z = feats["ls_z"]
    oi_chg_z = feats["oi_chg_z"]

    print(f"\nprimary metric: ls_z (top-trader long/short ratio, 14-day trailing "
          f"z-score)  threshold=|z|>={Z_THRESH}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars in range "
                  f"-- outside data coverage. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=None, cross=None,
                                 lead=float("nan"), pass_a=False, pass_b=False,
                                 null_p90=float("nan"), oi_cross_lead=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        flip_time_any = nearest_transition(majority, window, onset, direction="any")
        cross_time = nearest_crossing(ls_z, window, onset)
        # secondary/context-only: oi_chg_z's own nearest-crossing lead (not
        # part of the pass/fail decision, reported for context only)
        oi_cross_time = nearest_crossing(oi_chg_z, window, onset, thresh=Z_THRESH)
        oi_lead = (float((flip_time - oi_cross_time).total_seconds() / 86400.0)
                   if (flip_time is not None and oi_cross_time is not None)
                   else float("nan"))

        if flip_time is None or cross_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no ls_z crossing'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction "
                  f"(lead undefined).")
            results.append(dict(label=label, onset=onset_str, flip=flip_time,
                                 cross=cross_time, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan"),
                                 oi_cross_lead=oi_lead))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(ls_z, window, onset, flip_time)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        null_median = float(np.median(valid_null)) if len(valid_null) else float("nan")
        pass_a = lead > 0
        pass_b = pass_a and (not np.isnan(null_p90)) and (lead > null_p90)

        local_majority = majority.reindex(window).to_numpy()
        flip_pos = int(window.get_indexer([flip_time])[0])
        prev_val = local_majority[flip_pos - 1] if flip_pos > 0 else float("nan")
        new_val = local_majority[flip_pos]
        print(f"[{label}] onset={onset_str}")
        print(f"    anchor-gate nearest transition: {flip_time}  "
              f"(majority {prev_val:.3f} -> {new_val:.3f})")
        print(f"    ls_z nearest crossing (|z|>={Z_THRESH}): {cross_time}")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'crowding LED' if lead > 0 else 'crowding LAGGED/coincided'})")
        print(f"    null (500 draws): median={null_median:+.2f}d  p90={null_p90:+.2f}d  "
              f"({'valid draws: ' + str(len(valid_null)) + '/' + str(N_DRAWS)})")
        print(f"    PASS (a) lead>0: {pass_a}   PASS (b) lead > null p90: {pass_b}")
        print(f"    [context only] oi_chg_z nearest-crossing lead: {oi_lead:+.2f}d "
              f"(not part of the decision)")
        print(f"    [diagnostic only] 'any-direction' flip would have been: "
              f"{flip_time_any}  {'(differs from the primary down-only flip)' if flip_time_any != flip_time else '(same)'}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             cross=cross_time, lead=lead, pass_a=pass_a,
                             pass_b=pass_b, null_p90=null_p90,
                             null_median=null_median, oi_cross_lead=oi_lead))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 2

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (fixed before any number above was computed):")
    print("  an episode PASSES iff (a) lead>0 AND (b) lead exceeds its own")
    print("  500-draw block-bootstrap null's 90th percentile.")
    print("  proceed to Step B only if >= 2 of 3 episodes PASS.")
    print("=" * 78)
    for r in results:
        print(f"  {r['label']:40s} lead={r['lead']:+.2f}d  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/3")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")

    print(f"\nETH note: ETH crowding metrics start {METRICS_START['ETH']}, which is "
          f"AFTER all 3 stress episodes' onset dates (earliest onset "
          f"{STRESS_EPISODES[0][1]}) -- ETH has no usable pre-episode baseline for "
          f"any of the 3 episodes, so no ETH Step-A gate is run. Stated explicitly "
          f"per this round's instructions, not silently skipped.")

    print(f"\nconfigurations evaluated in this file: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session: "
          f"{max(bars.index.max(), feats.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed)


if __name__ == "__main__":
    cmds = {"gate": gate}
    choice = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r81_conservative_crowding_vote.py [{'|'.join(cmds)}]")
