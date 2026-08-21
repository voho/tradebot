#!/usr/bin/env python
"""R-88 CONSERVATIVE branch: Binance taker buy/sell volume-ratio order flow
(`sum_taker_long_short_vol_ratio`) as a discrete CONFIRMING VOTE on
`kelly_regime_v4`'s 3-anchor gate, via R-53/R-55's already-validated
`confirming_vote_frac` combination rule -- Step A measurement gate first,
this project's established discipline for every INFO-axis round since
R-53 (R-53/R-73/R-74/R-79/R-81/R-84).

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). Binance's aggregated taker buy/sell volume
   ratio -- the direction currently-EXECUTING aggressive trades are
   leaning, reported directly by the venue rather than reconstructed from
   price -- is a genuine order-flow-imbalance signal, so when it is
   extreme (z-scored against its own trailing baseline) in a direction,
   that reading should on average LEAD `kelly_regime_v4`'s own
   price-anchor reaction to the same regime shift, and, if it does, can
   serve as a discrete confirming vote via `r88_shared.confirming_vote_frac`
   that raises exposure when flow confirms the anchors' own bullish
   consensus and lowers it when flow confirms bearish consensus (the flow
   itself is directional/signed, unlike R-84's raw-volume magnitude, so
   this is a same-direction CONFIRM, not R-81's contrarian fade).
   Constraint attacked: INFO. Full citation trail and the "not a
   duplicate of" argument against R-81 (crowding/positioning, a STOCK),
   R-84 (raw volume, a magnitude-only signal), L-14/L-15/L-16 (BVC/VPIN
   flow RECONSTRUCTED from price, ruled out), R-53/R-55 (the reused
   combination architecture) and B-24/R-77 (volatility-driven execution
   timing, a different mechanism) are in `experiments/r88_shared.py`'s
   module docstring -- not repeated here to keep one citation trail in
   one place, per that file's own stated convention.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy
   code, on BTC only, over `r88_shared.STRESS_EPISODES` (the same 3
   episodes as R-81's table). ETH note (named here, not discovered after
   the fact): ETH's metrics history starts 2021-12-01, so only the
   2022-05 Terra/Luna and 2022-11 FTX episodes have any pre-episode data
   at all on ETH, and the 2021-top episode (onset 2021-11-10) predates
   ETH coverage entirely -- no ETH Step-A gate is run in this file; ETH
   is used only in Step B's falsification test, contingent on reaching it.

   PRIMARY METRIC (chosen now, before any number): `tv_z` --
   `r88_shared.taker_flow_z(metrics, bars, window_days=14)`, the causal
   z-score of the taker buy/sell volume ratio against its own trailing
   14-day mean/std. 14 days, not some other window, for direct continuity
   with R-81's `ls_z` (also a 14-day trailing z-score on the same metrics
   feed) -- the two rounds' Step-A gates are then read on the same
   baseline horizon, isolating "which metric" as the only thing that
   differs between R-81's result and this one.

   EXTREME THRESHOLD: bidirectional, `|tv_z| >= 1.5`. Flow imbalance is
   naturally two-sided (extreme buy-side or extreme sell-side aggression
   are both informative events, unlike R-84's volume magnitude, which had
   no direction of its own), so this is a two-sided rule, matching R-81's
   `ls_z` convention exactly. 1.5-sigma matches this project's standing
   "extreme" bar (R-81/R-84) -- large enough to be a real tail reading,
   loose enough that a +/-60-day window has a realistic chance of a
   crossing.

   EPISODE-LOCAL SEARCH WINDOW: [onset - 60 days, onset + 60 days], fixed
   before any number was computed, identical to R-81's window and for the
   identical reason: `kelly_regime_v4`'s own anchors react to price with a
   lag (20/40/80-day rolling means), so its own reaction to a stress event
   is not necessarily dated to the event's onset, and the "nearest
   transition" search needs room on both sides.

   ANCHOR-GATE "FLIP" DEFINITION: reused VERBATIM from R-81's disclosed,
   bug-fixed convention -- "flip" means the `anchor_majority` DOWNWARD
   transition (majority DECREASES, the gate de-risking, the mechanism
   this gate actually tests) whose timestamp is closest to the episode's
   onset within the search window. R-81 disclosed that an early draft
   used "nearest transition in EITHER direction" and it silently picked a
   spurious bullish blip two days before the 2021 top as episode 1's
   "flip" -- the lesson this file inherits directly by using "down-only"
   as the PRIMARY rule from the first line of code, never trying
   "any-direction" as a candidate rule at all (it is still computed and
   printed as a diagnostic, exactly as R-81/R-84 do, for transparency).

   TAKER-FLOW "CROSSING" DEFINITION: the first bidirectional crossing
   (prior bar |tv_z| < 1.5, this bar |tv_z| >= 1.5) whose timestamp is
   closest to the episode's onset within the same window -- the same
   "nearest to onset" logic applied to the candidate signal, for an
   apples-to-apples comparison with the flip search.

   DATA-COVERAGE HANDLING (pre-registered here, before running, per
   `r88_shared.py`'s disclosed caveat): BTC `sum_taker_long_short_vol_ratio`
   has a raw-value gap 2022-01-31 -> 2022-05-09 that ends essentially
   exactly at the Terra/Luna episode's onset (2022-05-09). Concretely:
   for each episode, before searching for any crossing, this file checks
   whether `tv_z` has ANY non-NaN value in the PRE-onset half of the
   window (`onset - 60d` through `onset`, inclusive) -- if it has zero
   valid pre-onset observations, the episode is a construction-forced
   FAIL, reported as such, and no crossing search (which could otherwise
   spuriously "succeed" on data that only exists strictly after the
   onset, which is not evidence of any lead) is attempted. This is named
   now, not discovered after running: it is expected to force-fail the
   Terra/Luna episode specifically. That forced FAIL counts toward the
   denominator of 3 -- it is NOT dropped from the episode table and the
   bar is NOT recomputed over N=2 valid episodes. Concretely, this means
   the pre-registered ">= 2 of 3" bar below can only be cleared if BOTH
   of the two data-complete episodes (2021-top, FTX) independently pass;
   Terra/Luna cannot contribute a pass under any circumstance. This is a
   deliberately strict reading, chosen for continuity with R-81's own
   ">= 2 of 3" bar on the identical episode table rather than inventing a
   looser one now that a data gap is known.

   LEAD = (flip_time - crossing_time) in days. Positive = flow extremity
   was reached BEFORE the anchor gate's own nearest reaction.

   NULL: `r88_shared.block_bootstrap_lead_null(n_bars, block_days=5,
   n_draws=500, seed=88)` circularly block-shifts the LOCAL (episode-
   window) `tv_z` array and recomputes the "crossing nearest the REAL,
   unshifted onset" against each shifted copy, compared to the fixed,
   real flip time -- the null holds the gate's true reaction fixed and
   asks whether an arbitrarily time-shifted copy of the SAME flow series
   would have looked just as informative. `block_days=5` matches R-81/
   R-84's primary block length; `seed=88` is this round's own number.

   PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed): an episode counts as a PASS if BOTH (a) LEAD > 0 (flow
   extremity crossed before the gate's nearest reaction), AND (b) the
   true LEAD exceeds the 90th percentile of that same episode's own
   500-draw block-bootstrap null lead distribution. PROCEED TO STEP B
   only if >= 2 of the 3 episodes PASS -- the same bar R-81 used on the
   same episode table, for continuity, with the Terra/Luna handling
   above already folded in. If fewer than 2 pass: STOP, report the gate
   result as this branch's whole product, do not build a strategy. The
   bar is not relaxed after seeing the numbers.

3. WHAT WOULD MAKE STEP A FAIL, named now: the same failure this
   project's other 10 INFO signals have hit -- the flow extremity is
   reached AFTER (not before) the anchor gate's own nearest reaction, or
   a positive lead (where it occurs) is not distinguishable from an
   arbitrary time-shift of the same series (i.e. it is generic
   autocorrelation/regime-persistence structure in a slow-moving z-score,
   not a real early-warning property). Given the base rate on this exact
   construction (0 of 9 prior INFO signals led before R-84; R-81's own
   native-cadence positioning signal, the structurally closest precedent
   -- same feed, same 14-day z-score, same episode table -- also lagged
   on 3 of 3 tested episodes; R-84's raw volume also failed), the modal
   outcome pre-registered here IS failure, and a clean negative is this
   round's fully successful, complete product if that is what happens.

4. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step
   A's numbers exist; only executed if Step A's stop rule passes).

   CONFIRMING-VOTE CONSTRUCTION. Unlike R-84's raw volume (magnitude
   only, direction borrowed from an anchor), `tv_z` is itself signed, so
   the meta-vote's direction comes from the flow reading directly: a
   discrete latch, `meta_vote[i] = 1` on any bar where `tv_z[i] >=
   Z_THRESH` ("confirmed bullish"), `meta_vote[i] = 0` on any bar where
   `tv_z[i] <= -Z_THRESH` ("confirmed bearish"), and on bars with neither,
   `meta_vote` carries forward its last confirmed value (a latch, exactly
   R-53/54/55's hysteresis pattern, keyed on a flow-extremity gate instead
   of a threshold-band on the signal's own level). Before the first
   confirmed bar, `meta_vote` TRACKS the fastest anchor's (20-day) own
   then-current 0/1 vote each bar (no dilution in either direction while
   unconfirmed -- the same default rule R-84 used for its own
   direction-borrowing case, applied here even though this signal has its
   own sign, for architectural consistency across confirming-vote rounds).

   frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
        = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity check,
   run first, before any swept configuration).

   SWEEP GRID (fixed a priori, not tuned to any inner-validation number):
   - `weight` in {0.5, 1.0, 2.0, 4.0} x `window_days` (tv_z's own trailing
     baseline) in {7, 14, 28} -- 12 configurations. 14 is the Step-A
     primary; 7/28 bracket it 2x on each side, matching R-84's bracketing
     convention.
   - `Z_THRESH` sensitivity at the pre-registered primary point
     (weight=1.0, window_days=14) over {1.0, 2.0} -- 2 configurations
     (1.5 is already covered by the main grid's weight=1.0/window=14 cell).
   - identity check (weight=0): 1 configuration.
   Total Step B configurations, if reached: 15.

   PRE-REGISTERED DECISION RULE for recommending a holdout consultation
   to the operator (fixed now, contingent on reaching Step B and
   everything below clearing -- this branch does NOT have authority to
   read the holdout itself, only to recommend the operator do so):
   recommend holdout consultation ONLY IF ALL of (a) the pre-registered
   primary configuration's (weight=1.0, window_days=14, Z_THRESH=1.5)
   inner-validation Sharpe improvement over `kelly_regime_v4` exceeds the
   +/-0.2 Sharpe noise floor (R-20) on at least one market with the other
   not materially worse, OR shows a matched-risk (comparable realized
   vol/exposure) drawdown improvement -- per docs/ROUTINE.md's standing
   "match risk before comparing anything" rule; AND (b) the improving
   region is a genuine parameter PLATEAU (neighbouring grid cells agree
   in sign), not an isolated peak; AND (c) the frozen primary config
   passes the falsification test below. If ANY of these fail, this
   branch reports NEGATIVE and does not recommend a holdout read -- an
   honest negative at any stage is this project's own definition of a
   complete, successful piece of work.

   FALSIFICATION TEST: run the FROZEN primary config (weight=1.0,
   window_days=14, Z_THRESH=1.5) on ETH, over whatever inner-period ETH's
   metrics coverage allows -- 2021-12-01 through INNER_VAL_END
   (2022-12-31), still strictly inside the training period, never the
   holdout -- and check that the same QUALITATIVE sign/direction of edge
   over `kelly_regime_v4` replicates (does not decisively reverse). A
   reversal, or a result indistinguishable from noise on ETH's much
   shorter window, falsifies the mechanism as cross-asset rather than a
   BTC-period-specific artifact.

   MANDATORY: the causal-truncation probe
   (`r88_shared.truncation_causality_probe`) is run on the frozen primary
   candidate's constructed target series regardless of whether the
   decision rule above ultimately recommends a holdout read, per this
   project's standing discipline (R-80/R-84).

5. CONFIGS EVALUATED IN STEP A: 0 (a fixed, non-swept measurement gate,
   this project's standing accounting convention for this exact
   construction -- R-53/R-73/R-74/R-79/R-81/R-84's own Step-A studies).
   Step B's count, if reached, is 15 as itemized above.

6. HOLDOUT DISCIPLINE. This file asserts, at every load point, that no
   bar with timestamp >= OOS_START (2023-01-01) is ever read. This branch
   has no authority to consult the 2023+ holdout: that decision is made
   centrally by the operator after both this round's branches report. If
   the decision rule above says "recommend a holdout read", this file
   only prints that recommendation -- it never acts on it.
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

from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r88_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    METRICS_END,
    METRICS_START,
    OOS_START,
    STRESS_EPISODES,
    V4_BAND,
    V4_HORIZONS,
    anchor_majority,
    anchor_votes,
    block_bootstrap_lead_null,
    confirming_vote_frac,
    load_flow_inputs,
    taker_flow_z,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
Z_THRESH = 1.5
WINDOW_DAYS = 60          # episode-local search window, +/- days
TV_WINDOW_DAYS = 14       # tv_z trailing baseline, matches R-81's ls_z
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 88
MIN_EPISODES_PASS = 2     # of 3, matching R-81's bar on the same table


# ---------------------------------------------------------------- holdout guard

def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START. Independent of any truncation already done
    at load time (same pattern as r81/r84's conservative branches)."""
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


def load_btc_flow(bars: pd.DataFrame, window_days: int = TV_WINDOW_DAYS) -> tuple[pd.Series, pd.Series]:
    """tv_z aligned onto `bars`, and the RAW (unaligned, unffilled) metric
    for the data-coverage guard, BTC only. Independently re-truncated
    before OOS_START even though `load_flow_inputs` already truncates at
    METRICS_END (== INNER_VAL_END, strictly before OOS_START) -- belt and
    suspenders, matching this project's own convention."""
    metrics = load_flow_inputs(DATA_DIR, asset="BTC")
    assert metrics is not None, "BTC taker-flow metrics file missing"
    assert_no_holdout(metrics)
    tv_z = taker_flow_z(metrics, bars, window_days=window_days)
    assert_no_holdout(tv_z.to_frame())
    print(f"BTC taker-flow metrics: {METRICS_START['BTC']} -> {METRICS_END}  "
          f"({len(metrics):,} raw rows)", file=sys.stderr)
    return tv_z, metrics["sum_taker_long_short_vol_ratio"]


# ------------------------------------------------------------- flip / crossing

def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    """Reused, byte-for-byte convention, from r81_conservative_crowding_vote.py
    (see banner item 2 for the disclosed down-only rationale)."""
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
    """Bidirectional: prior bar |z| < thresh, this bar |z| >= thresh."""
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


def has_preonset_coverage(raw_metric: pd.Series, window: pd.DatetimeIndex,
                           onset: pd.Timestamp) -> tuple[bool, int]:
    """Pre-registered data-coverage guard (banner item 2): does the RAW,
    un-aligned, un-forward-filled `sum_taker_long_short_vol_ratio` column
    have ANY non-NaN value in the PRE-onset half of the window? Returns
    (has_coverage, n_valid_preonset).

    Deliberately checks the RAW metric, not `tv_z`: `align_metrics_causal`
    forward-fills across gaps (by design, so a momentary feed outage does
    not NaN out mid-series), and `taker_flow_z`'s rolling z-score of a
    perfectly flat, forward-filled series degenerates to (0-0)/tiny_std =
    0.0 -- NOT NaN. A first working version of this check tested `tv_z`
    for NaN and, on the Terra/Luna window, found 17,281/17,281 "valid"
    observations, because the entire 60-day pre-onset window sits inside
    the raw 2022-01-31->2022-05-09 gap and every one of those bars reads
    an artifact tv_z of exactly 0.0 from the ffilled-flat input -- not
    real information, and never able to cross |z|>=1.5 in either
    direction, so it silently could never register a false PASS, but it
    would have been dishonestly reported as "measured, non-degenerate
    data". Caught by cross-checking against `r88_shared.py`'s disclosed
    caveat before treating any Step-A number as final, and fixed to check
    the raw column directly, which correctly reads 0/17,281 valid
    pre-onset observations for Terra/Luna (and 17,235/17,281 and
    17,280/17,281 for the other two episodes, confirming they are
    genuinely covered)."""
    pre = window[window <= onset]
    vals = raw_metric.reindex(pre).to_numpy()
    n_valid = int(np.sum(~np.isnan(vals)))
    return n_valid > 0, n_valid


# --------------------------------------------------------------------- null

def episode_null_leads(tv_z: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, flip_time: pd.Timestamp,
                        n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                        seed: int = NULL_SEED) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the LOCAL tv_z array (within `window`) and recompute the
    "crossing nearest the real, unshifted onset" against the fixed, real
    `flip_time`."""
    local = tv_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_lead_null(n_bars=n_bars, block_days=block_days,
                                        n_draws=n_draws, seed=seed)

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
    print("R-88 CONSERVATIVE: taker-flow confirming vote (tv_z) -- STEP A lead-time gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    tv_z, raw_metric = load_btc_flow(bars)

    print(f"\nprimary metric: tv_z (taker buy/sell volume ratio, {TV_WINDOW_DAYS}-day "
          f"trailing z-score)  threshold=|z|>={Z_THRESH}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars in range "
                  f"-- outside data coverage. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=None, cross=None,
                                 lead=float("nan"), pass_a=False, pass_b=False,
                                 null_p90=float("nan"), reason="no bars in window"))
            continue

        has_cov, n_valid = has_preonset_coverage(raw_metric, window, onset)
        if not has_cov:
            print(f"[{label}] onset={onset_str}: ZERO valid RAW taker-flow observations "
                  f"in the PRE-onset half of the +/-{WINDOW_DAYS}d window (data-coverage gap). "
                  f"Construction-forced FAIL per pre-registration -- no crossing search "
                  f"attempted (a post-gap-only crossing would not be evidence of a lead).")
            results.append(dict(label=label, onset=onset_str, flip=None, cross=None,
                                 lead=float("nan"), pass_a=False, pass_b=False,
                                 null_p90=float("nan"),
                                 reason="no pre-onset tv_z coverage (data gap)"))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        flip_time_any = nearest_transition(majority, window, onset, direction="any")
        cross_time = nearest_crossing(tv_z, window, onset)

        if flip_time is None or cross_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no tv_z crossing'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction "
                  f"(lead undefined).")
            results.append(dict(label=label, onset=onset_str, flip=flip_time,
                                 cross=cross_time, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan"),
                                 reason="no transition/crossing in window"))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(tv_z, window, onset, flip_time)
        valid_null = null_leads[~np.isnan(null_leads)]
        null_p90 = float(np.percentile(valid_null, 90)) if len(valid_null) else float("nan")
        null_median = float(np.median(valid_null)) if len(valid_null) else float("nan")
        pass_a = lead > 0
        pass_b = pass_a and (not np.isnan(null_p90)) and (lead > null_p90)

        local_majority = majority.reindex(window).to_numpy()
        flip_pos = int(window.get_indexer([flip_time])[0])
        prev_val = local_majority[flip_pos - 1] if flip_pos > 0 else float("nan")
        new_val = local_majority[flip_pos]
        print(f"[{label}] onset={onset_str}  (pre-onset valid tv_z bars: {n_valid})")
        print(f"    anchor-gate nearest transition: {flip_time}  "
              f"(majority {prev_val:.3f} -> {new_val:.3f})")
        print(f"    tv_z nearest crossing (|z|>={Z_THRESH}): {cross_time}")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'flow LED' if lead > 0 else 'flow LAGGED/coincided'})")
        print(f"    null ({N_DRAWS} draws): median={null_median:+.2f}d  p90={null_p90:+.2f}d  "
              f"(valid draws: {len(valid_null)}/{N_DRAWS})")
        print(f"    PASS (a) lead>0: {pass_a}   PASS (b) lead > null p90: {pass_b}")
        print(f"    [diagnostic only] 'any-direction' flip would have been: "
              f"{flip_time_any}  {'(differs from the primary down-only flip)' if flip_time_any != flip_time else '(same)'}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             cross=cross_time, lead=lead, pass_a=pass_a,
                             pass_b=pass_b, null_p90=null_p90,
                             null_median=null_median, reason=None))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= MIN_EPISODES_PASS

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (fixed before any number above was computed):")
    print("  an episode PASSES iff (a) lead>0 AND (b) lead exceeds its own")
    print(f"  {N_DRAWS}-draw block-bootstrap null's 90th percentile.")
    print(f"  Terra/Luna is a construction-forced FAIL (counted in the denominator).")
    print(f"  proceed to Step B only if >= {MIN_EPISODES_PASS} of 3 episodes PASS.")
    print("=" * 78)
    for r in results:
        lead_str = f"{r['lead']:+.2f}d" if np.isfinite(r["lead"]) else "undefined"
        reason = f"  ({r['reason']})" if r.get("reason") else ""
        print(f"  {r['label']:40s} lead={lead_str:>10s}  PASS={r['pass_b']}{reason}")
    print(f"\nEpisodes passing: {n_pass}/3")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")

    print(f"\nETH note: ETH taker-flow metrics start {METRICS_START['ETH']}, which is "
          f"AFTER the 2021-top episode's onset ({STRESS_EPISODES[0][1]}) -- only the "
          f"Terra/Luna and FTX episodes have any pre-episode data on ETH at all. No ETH "
          f"Step-A gate is run in this file (stated explicitly, not silently skipped); "
          f"ETH is used only in Step B's falsification test, contingent on reaching it.")

    print(f"\nconfigurations evaluated in this file's Step A: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session so far: "
          f"{bars.index.max()}  (< {OOS_START})")

    # Causal-truncation probe on this branch's own constructed target series
    # (task item 7): run unconditionally, on the primary tv_z construction
    # itself, whatever Step A's verdict is -- if the gate fails, tv_z (not
    # a Step-B strategy target, which is never built) is the "final
    # constructed target series" this branch produced.
    print("\n" + "=" * 78)
    print("CAUSAL-TRUNCATION PROBE on the primary tv_z construction")
    print("=" * 78)
    metrics = load_flow_inputs(DATA_DIR, asset="BTC")
    assert_no_holdout(metrics)

    def build_tv_z(d: pd.DataFrame) -> np.ndarray:
        return taker_flow_z(metrics, d, window_days=TV_WINDOW_DAYS).to_numpy()

    probe_results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(build_tv_z, bars, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        probe_results.append(ok)

    return dict(results=results, n_pass=n_pass, passed=passed, causality_probe=probe_results)


# ==========================================================================
# STEP B -- built only if the gate above passes (banner item 4).
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, metrics: pd.DataFrame, window_days: int,
                       z_thresh: float, horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """Signed, flow-confirmed discrete latch (banner item 4):
    `meta_vote[i] = 1` on a bar where `tv_z[i] >= z_thresh` (confirmed
    bullish), `= 0` where `tv_z[i] <= -z_thresh` (confirmed bearish),
    otherwise carries forward the last confirmed value. Before the first
    confirmation, tracks the fastest (20-day) anchor's own then-current
    vote each bar (no dilution while unconfirmed, R-84's convention).

    Causal: `tv_z` and each anchor vote are both causal (rolling/ffill
    constructions, row i depends only on rows <= i); the latch update at
    i depends only on values at <= i.
    """
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    tv_z = taker_flow_z(metrics, df, window_days=window_days).to_numpy()
    bullish = tv_z >= z_thresh
    bearish = tv_z <= -z_thresh

    n = len(df)
    meta = np.empty(n)
    last = fast_vote[0]
    confirmed_ever = False
    for i in range(n):
        if bullish[i]:
            last = 1.0
            confirmed_ever = True
        elif bearish[i]:
            last = 0.0
            confirmed_ever = True
        elif not confirmed_ever:
            last = fast_vote[i]
        meta[i] = last
    return meta


def build_target_primary(df: pd.DataFrame, metrics: pd.DataFrame) -> np.ndarray:
    """Target-construction function for the truncation causality probe,
    frozen at the pre-registered primary candidate."""
    return TakerFlowConfirmKelly(weight=1.0, window_days=TV_WINDOW_DAYS,
                                  z_thresh=Z_THRESH, metrics=metrics).prepare(df.copy())["target"].to_numpy()


class TakerFlowConfirmKelly(Strategy):
    """kelly_regime_v4 + a taker-flow-confirmed signed vote (R-88
    conservative, unregistered). Structurally v3/v4's own prepare(), with
    the plain 3-anchor average `frac = anchor_sum/3` replaced by
    `confirming_vote_frac(anchor_sum, meta_vote, weight)`. `weight=0` must
    recover v4 bit-for-bit. Not `@register`ed -- stays in experiments/ per
    docs/ROUTINE.md. `metrics` (the raw taker-flow DataFrame) must be
    passed at construction; it is asset-specific and not derivable from
    `df` alone.
    """

    name = "r88_conservative_taker_flow_vote"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, metrics: pd.DataFrame, weight: float = 1.0,
                 window_days: int = TV_WINDOW_DAYS, z_thresh: float = Z_THRESH,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.metrics = metrics
        self.weight = weight
        self.window_days = window_days
        self.z_thresh = z_thresh
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = anchor_votes(df, horizons=self.horizons, band=self.band)
        anchor_sum = sum(v.to_numpy() for v in votes)

        meta_vote = compute_meta_vote(df, self.metrics, self.window_days, self.z_thresh,
                                       horizons=self.horizons, band=self.band)
        frac = confirming_vote_frac(anchor_sum, meta_vote, self.weight)

        # Identical conditional-volatility-targeting scale to kelly_regime_v3/_v4.
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


# --------------------------------------------------------------------- checks

def run_identity_check(df_full: pd.DataFrame, metrics: pd.DataFrame) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = TakerFlowConfirmKelly(metrics=metrics, weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe(df_full: pd.DataFrame, metrics: pd.DataFrame) -> list[bool]:
    df = df_full.loc[:INNER_TRAIN_END].copy()
    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(lambda d: build_target_primary(d, metrics), df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, metrics: pd.DataFrame, weight: float, window_days: int,
                 z_thresh: float, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = TakerFlowConfirmKelly(metrics=metrics, weight=weight,
                                           window_days=window_days, z_thresh=z_thresh)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES, metrics: pd.DataFrame) -> dict:
    results = {}
    for weight in (0.5, 1.0, 2.0, 4.0):
        for window_days in (7, 14, 28):
            tag = f"w{weight} win{window_days} z{Z_THRESH}"
            results[("main", weight, window_days, Z_THRESH)] = eval_config(
                ev, SPOT, FUTURES, metrics, weight, window_days, Z_THRESH, tag)
    for z_thresh in (1.0, 2.0):
        tag = f"w1.0 win{TV_WINDOW_DAYS} z{z_thresh}"
        results[("zsens", 1.0, TV_WINDOW_DAYS, z_thresh)] = eval_config(
            ev, SPOT, FUTURES, metrics, 1.0, TV_WINDOW_DAYS, z_thresh, tag)
    return results


def run_eth_falsification(ev, weight: float = 1.0, window_days: int = TV_WINDOW_DAYS,
                           z_thresh: float = Z_THRESH) -> dict:
    """ETH falsification (banner item 4): frozen primary config on ETH,
    over ETH's own metrics coverage window, strictly inside training."""
    from tradebot.broker import MarketSpec

    spot = MarketSpec.spot()
    eth_bars = load_ohlcv_csv(DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz")
    eth_bars = eth_bars.loc[eth_bars.index < pd.Timestamp(OOS_START, tz=eth_bars.index.tz)].copy()
    assert_no_holdout(eth_bars)

    eth_metrics = load_flow_inputs(DATA_DIR, asset="ETH")
    assert eth_metrics is not None, "ETH taker-flow metrics file missing"
    assert_no_holdout(eth_metrics)

    eth_start = pd.Timestamp(METRICS_START["ETH"], tz="UTC")
    print(f"ETH bars: {len(eth_bars):,}  {eth_bars.index[0]} -> {eth_bars.index[-1]}")
    print(f"ETH metrics coverage: {METRICS_START['ETH']} -> {METRICS_END} "
          f"({len(eth_metrics):,} raw rows)")

    v4 = get_strategy("kelly_regime_v4")
    cand = TakerFlowConfirmKelly(metrics=eth_metrics, weight=weight,
                                  window_days=window_days, z_thresh=z_thresh)
    out = {}
    m_v4 = ev(v4, df=eth_bars, market=spot, start=METRICS_START["ETH"], end=INNER_VAL_END,
              tag="ETH falsification: v4")
    m_cand = ev(cand, df=eth_bars, market=spot, start=METRICS_START["ETH"], end=INNER_VAL_END,
                tag="ETH falsification: candidate")
    delta = m_cand.sharpe - m_v4.sharpe
    print(f"  ETH ({METRICS_START['ETH']} -> {INNER_VAL_END}): v4 sharpe={m_v4.sharpe:.2f}  "
          f"candidate sharpe={m_cand.sharpe:.2f}  delta={delta:+.2f}")
    out["ETH"] = dict(v4=m_v4.sharpe, cand=m_cand.sharpe, delta=delta)
    return out


def run_step_b() -> None:
    from scripts.experiment import DF, FUTURES, SPOT, ev

    print("\n" + "=" * 78)
    print("STEP B (gate passed): sweep + mandatory checks")
    print("=" * 78)

    metrics = load_flow_inputs(DATA_DIR, asset="BTC")
    assert_no_holdout(metrics)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(DF, metrics)
    n_configs += 1

    print("\n=== causality probe ===")
    run_causality_probe(DF, metrics)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=INNER_TRAIN_END)),
                                ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)

    print("\n=== sweep ===")
    sweep_results = run_sweep(ev, SPOT, FUTURES, metrics)
    n_configs += len(sweep_results)

    print("\n=== ETH falsification ===")
    run_eth_falsification(ev)

    print(f"\nTotal Step B configurations evaluated: {n_configs}")


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    gate_result = gate()
    if gate_result["passed"]:
        run_step_b()
    else:
        print("\nSTEP A FAILED the pre-registered stop rule. Per this file's own "
              "pre-registration, no strategy is built and no Step-B code runs. "
              "This gate result is this branch's whole product.")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmds = {"gate": gate, "main": main}
    choice = sys.argv[1] if len(sys.argv) > 1 else "main"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r88_conservative_taker_flow_vote.py [{'|'.join(cmds)}]")
