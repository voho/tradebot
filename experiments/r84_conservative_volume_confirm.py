#!/usr/bin/env python
"""R-84 CONSERVATIVE branch: raw traded volume (the OHLCV file's own
`volume` column, unsigned, not BVC-classified) as a confirming vote on
`kelly_regime_v4`'s 3-anchor gate, via R-53/R-55's already-validated
`confirming_vote_frac` combination rule -- Step A measurement gate first,
this project's established discipline for every INFO-axis round since
R-53 (R-53/R-73/R-74/R-79/R-81).

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). A price move crossing one of `kelly_regime_
   v4`'s anchors is more likely to be information-driven, and therefore a
   more trustworthy confirming vote, when it is accompanied by unusually
   high traded volume than when it happens on ordinary or thin
   participation, because the Mixture-of-Distributions Hypothesis (Clark
   1973, Econometrica 41(1)) models volume as a proxy for the latent rate
   of information arrival driving the same subordinated return process,
   Easley & O'Hara (1992, J. Finance 47(2)) show informed trading
   concentrates volume around genuine information events in a
   sequential-trade model, and Llorente, Michaely, Saar & Wang (2002, Rev.
   Fin. Studies 15(4)) find high-volume price moves carry a different
   continuation/reversal signature than low-volume ones depending on
   whether the volume behind the move is information- or liquidity-
   motivated. Full citation set and not-a-duplicate-of argument (vs. the
   9 prior INFO-axis signals, the 4 failed bounded-brake attempts, and
   L-14/L-15/L-16's signed-BVC family) is in `experiments/r84_shared.py`'s
   module docstring -- not repeated here to keep one citation trail in one
   place, per this project's own convention (R-81's file does the same).

   Constraint attacked: INFO. Not a duplicate of any of the 9 prior
   INFO-axis rounds (R-44, R-53/R-54, R-54/R-55/R-58, R-73, R-74, R-75,
   R-79, R-81, R-76) or of L-14/L-15/L-16 (signed BVC classification of
   volume, ruled out as "a price transform, not order flow" -- this round
   uses raw volume MAGNITUDE only, never classifies a side). Not a
   duplicate of the never-increase-only bounded-brake family (R-34, R-41,
   R-53-conservative, R-73-conservative, 4-for-4 failed regardless of
   signal): this branch reuses R-53/R-55's validated CONFIRMING-VOTE
   architecture instead, per `r84_shared.py`'s own disclosed reasoning.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy
   code, on the FULL R-82/R-83 six-episode table (`r84_shared.
   STRESS_EPISODES`) -- usable in full here, for the first time on an
   INFO-axis round, because raw volume is the file's own sixth OHLCV
   column with zero external coverage-start caveat across the whole
   2017-01 -> 2026-08 committed BTC history.

   PRIMARY FEATURE (chosen now, before any number): `volume_z` --
   `r84_shared.volume_z(df, window_days=20)`, the causal log-volume
   z-score against its own trailing 20-day mean/std. The 20-day window is
   not arbitrary: it matches `kelly_regime_v4`'s own FASTEST anchor
   (20 days), so "unusual participation" is measured on the same
   timescale as the anchor this round ultimately ties the confirming vote
   to in Step B (see below) -- one consistent horizon choice, stated
   before any number, not picked after comparing windows.

   THRESHOLD: ONE-SIDED, `volume_z >= 1.5`. Unlike R-81's `ls_z`
   (positioning), which is naturally bidirectional (extreme LONG or
   extreme SHORT are both informative), the mechanism here is about
   participation INTENSITY, not sign: MDH and Easley & O'Hara predict
   that elevated volume (more information arrival) makes a co-occurring
   price move more trustworthy, but say nothing that would make
   UNUSUALLY LOW volume a signal of anything in this vote-confirmation
   sense. A two-sided rule would test a different, unmotivated
   hypothesis. 1.5-sigma matches this project's own standing "extreme"
   convention (R-81) -- large enough to be a real tail reading, loose
   enough that a +/-60-day window has a realistic chance of a crossing.

   EPISODE-LOCAL SEARCH WINDOW: [onset - 60 days, onset + 60 days], fixed
   before any number was computed -- identical to R-81's window and for
   the identical reason: `kelly_regime_v4`'s own anchors react to price
   with a lag (20/40/80-day rolling means), so its own reaction to a
   stress event is not necessarily dated to the event's onset, and the
   "nearest transition" search (below) needs room on both sides.

   ANCHOR-GATE "FLIP" DEFINITION: reused VERBATIM from R-81's disclosed,
   bug-fixed convention (`r81_conservative_crowding_vote.py`'s
   `nearest_transition`, `direction="down"`) -- "flip" means the
   `anchor_majority` DOWNWARD transition (majority DECREASES, i.e. the
   gate de-risking, the mechanism this gate actually tests) whose
   timestamp is closest to the episode's onset date within the search
   window. R-81 disclosed that an early draft used "nearest transition in
   EITHER direction" and it silently picked a spurious bullish blip two
   days before the 2021 top as episode 1's "flip" -- the lesson this
   file inherits directly by reusing "down-only" as the PRIMARY rule from
   the first line of code, never trying "any-direction" as a candidate at
   all.

   VOLUME "CROSSING" DEFINITION: the first ONE-SIDED upward crossing
   (prior bar `volume_z < 1.5`, this bar `volume_z >= 1.5`) whose
   timestamp is closest to the episode's onset date within the same
   window -- the same "nearest to onset" logic R-81 applied to its own
   candidate signal, for an apples-to-apples comparison.

   LEAD = (flip_time - crossing_time) in days. Positive = volume
   extremity was reached BEFORE the anchor gate's own nearest reaction.

   NULL: `r84_shared.block_bootstrap_shifts(n_bars, block_days=5,
   n_draws=500, seed=84)` circularly block-shifts the LOCAL (episode-
   window) `volume_z` array and recomputes the "crossing nearest to the
   REAL, unshifted onset" against each shifted copy, compared to the
   fixed, real flip time -- the null holds the gate's true reaction fixed
   and asks whether an arbitrarily time-shifted copy of the SAME volume
   series would have looked just as informative. `block_days=5` matches
   R-81's primary block length; `seed=84` is this round's own number
   (distinct from R-81's 81 and R-82's implicit seed).

   PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed): an episode counts as a PASS if BOTH (a) LEAD > 0 (volume
   crossed before the gate's nearest reaction), AND (b) the true LEAD
   exceeds the 90th percentile of that same episode's own 500-draw
   block-bootstrap null lead distribution. PROCEED TO STEP B only if
   a MAJORITY of episodes PASS -- i.e. >= 4 of 6, matching R-82/R-83's
   bar on this identical six-episode table. If fewer than 4 pass: STOP,
   report the gate result as this branch's whole product, do not build a
   strategy. The bar is not relaxed after seeing the numbers.

3. WHAT WOULD MAKE STEP A FAIL, named now: the same failure every one of
   the 9 prior INFO-axis signals in this ledger hit -- the volume
   extremity is reached AFTER (not before) the anchor gate's own nearest
   reaction, or the lead (if positive in a minority of episodes) is not
   distinguishable from an arbitrary time-shift of the same series (i.e.
   it is generic autocorrelation/regime-persistence structure in a
   slow-moving z-score, not a real early-warning property). Given the
   base rate on this exact construction (0 of 9 prior INFO signals led;
   R-81's own native-cadence positioning signal, structurally the closest
   precedent, also lagged on 3 of 3 tested episodes), the modal outcome
   pre-registered here IS failure, and a clean negative is this round's
   fully successful, complete product if that is what happens.

4. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step
   A's numbers exist, exactly as R-80's file froze its sweep grid before
   any inner-validation number; only executed if Step A's stop rule
   passes).

   CONFIRMING-VOTE CONSTRUCTION. `confirming_vote_frac` (imported
   unchanged from `r84_shared.py`) requires a DISCRETE {0,1} `meta_vote`
   -- R-80's hard-won lesson, preserved here: a continuous meta_vote
   costs `kelly_regime_v4` its ability to reach exactly flat on unanimous
   bearish anchor consensus (R-80's diagnosed formula defect), so this
   round follows R-53/54/55/56's ORIGINAL discrete-latch pattern, not
   R-80's continuous generalization. Volume itself carries no direction,
   so the vote's direction has to come from one of v4's own anchors, per
   the mechanism statement ("a price move relative to one of v4's
   anchors"): `meta_vote` tracks the FASTEST anchor's own 0/1 vote
   (20-day, matching `volume_z`'s own window, the single most reactive
   anchor v4 has, and the timescale MDH/Easley&O'Hara's information-event
   concentration argument fits best), but only UPDATES on a bar where
   `volume_z >= Z_THRESH` (the vote is "confirmed" that bar); on bars
   without volume confirmation, `meta_vote` carries forward its last
   confirmed value (a latch, exactly like R-53's `_macro_vote`/R-54's
   `_stable_vote`/R-55's `stable_vote` hysteresis pattern, just keyed on a
   volume gate instead of a threshold-band on the signal's own level).
   Before any bar is ever confirmed, `meta_vote` defaults to the fast
   anchor's own then-current value (no dilution in either direction while
   unconfirmed -- the same "absence causes no dilution" default R-54/R-55
   used for missing stablecoin data).

   frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
        = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity check,
   run first, before any swept configuration).

   SWEEP GRID (fixed a priori, not tuned to any inner-validation number):
   - `weight` in {0.5, 1.0, 2.0, 4.0} x `window_days` (volume_z's own
     trailing baseline) in {10, 20, 40} -- 12 configurations. 20 is the
     Step-A primary; 10/40 bracket it 2x on each side.
   - `Z_THRESH` sensitivity at the pre-registered primary point
     (weight=1.0, window_days=20) over {1.0, 2.0} -- 2 configurations
     (1.5 is already covered by the main grid's weight=1.0/window=20
     cell).
   - identity check (weight=0): 1 configuration.
   Total Step B configurations, if reached: 15.

   MANDATORY CHECKS (this project's standing discipline for every
   confirming-vote round that reaches Step B): (i) exposure-artifact R^2
   -- regress the candidate's `target` series against a mean-notional-
   matched flat rescale of `kelly_regime_v4`'s own `target` series on
   inner-validation, both markets; R^2 > 0.95 = "just a rescale", fail;
   (ii) ETH falsification + pre-2020 BTC control, using the standing
   Bitfinex pair (`btcusd_bitfinex_5m.csv.gz` / `ethusd_bitfinex_5m.csv.gz`,
   both 2016-01/2016-03 -> 2019-12-31, R-77's established falsification
   pair -- chosen over Coinbase ETH because it gives a genuine pre-2020
   BTC CONTROL on a wholly disjoint period from every stress episode's
   FITTING window, not just a cross-instrument check); (iii) the
   mandatory truncation causality probe
   (`r84_shared.truncation_causality_probe`) on the volume-confirmed
   meta-vote construction.

   PRE-REGISTERED HOLDOUT DECISION RULE (fixed now, contingent on
   reaching Step B and everything above clearing): read the 2023+ holdout
   ONLY IF ALL of (a) the pre-registered primary configuration's
   inner-validation Sharpe improvement over `kelly_regime_v4` exceeds the
   +/-0.2 noise floor (R-20) on BOTH markets, on a genuine parameter
   PLATEAU (neighbouring grid cells, not an isolated peak); (b) exposure-
   artifact R^2 <= 0.95 on both markets; (c) the ETH falsification
   replicates the same sign of edge (not decisively reversed); (d) the
   pre-2020 BTC control is not decisively worse than `kelly_regime_v4`
   itself. If ANY of these fail, this branch reports NEGATIVE and the
   holdout is never read -- an honest negative at any stage is this
   project's own definition of a complete, successful piece of work.

5. CONFIGS EVALUATED IN STEP A: 0 (a fixed, non-swept measurement gate,
   this project's standing accounting convention for this exact
   construction -- R-53/R-73/R-74/R-79/R-81's own Step-A studies). Step
   B's count, if reached, is 15 as itemized above.
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

from experiments.r84_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    STRESS_EPISODES,
    V4_BAND,
    V4_HORIZONS,
    anchor_majority,
    anchor_votes,
    assert_no_holdout,
    block_bootstrap_shifts,
    confirming_vote_frac,
    truncation_causality_probe,
    volume_z,
)

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
Z_THRESH = 1.5
WINDOW_DAYS = 60
VOL_BASELINE_DAYS = 20
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 84
MIN_EPISODES_PASS = 4  # of 6, majority


# --------------------------------------------------------------------- data

def load_btc_bars() -> pd.DataFrame:
    """BTC spot, truncated strictly before OOS_START at load time."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


# ------------------------------------------------------------- flip / crossing
# (reused, byte-for-byte convention, from r81_conservative_crowding_vote.py's
# nearest_transition/nearest_crossing -- see banner item 2 for the disclosed
# down-only rationale.)

def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
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
    """One-sided: prior bar below threshold, this bar at/above (banner item 2)."""
    vals = z.reindex(window).to_numpy()
    above = vals >= thresh
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

def episode_null_leads(vz: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, flip_time: pd.Timestamp,
                        n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                        seed: int = NULL_SEED) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the LOCAL volume_z array (within `window`) and recompute the
    "crossing nearest the real, unshifted onset" against the fixed, real
    `flip_time`."""
    local = vz.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)

    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        above = shifted >= Z_THRESH
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
    print("R-84 CONSERVATIVE: raw volume confirming vote -- STEP A lead-time gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    vz = volume_z(bars, window_days=VOL_BASELINE_DAYS)

    print(f"\nprimary feature: volume_z (log-volume z-score, {VOL_BASELINE_DAYS}-day "
          f"trailing baseline)  threshold: one-sided volume_z>={Z_THRESH}  "
          f"search window=+/-{WINDOW_DAYS}d  null: {N_DRAWS} draws, "
          f"block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars in range "
                  f"-- outside data coverage. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=None, cross=None,
                                 lead=float("nan"), pass_a=False, pass_b=False,
                                 null_p90=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        flip_time_any = nearest_transition(majority, window, onset, direction="any")
        cross_time = nearest_crossing(vz, window, onset)

        if flip_time is None or cross_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no volume_z crossing'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction "
                  f"(lead undefined).")
            results.append(dict(label=label, onset=onset_str, flip=flip_time,
                                 cross=cross_time, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan")))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(vz, window, onset, flip_time)
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
        print(f"    volume_z nearest crossing (>={Z_THRESH}): {cross_time}")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'volume LED' if lead > 0 else 'volume LAGGED/coincided'})")
        print(f"    null ({N_DRAWS} draws): median={null_median:+.2f}d  p90={null_p90:+.2f}d  "
              f"(valid draws: {len(valid_null)}/{N_DRAWS})")
        print(f"    PASS (a) lead>0: {pass_a}   PASS (b) lead > null p90: {pass_b}")
        print(f"    [diagnostic only] 'any-direction' flip would have been: "
              f"{flip_time_any}  {'(differs from the primary down-only flip)' if flip_time_any != flip_time else '(same)'}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             cross=cross_time, lead=lead, pass_a=pass_a,
                             pass_b=pass_b, null_p90=null_p90,
                             null_median=null_median))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= MIN_EPISODES_PASS

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (fixed before any number above was computed):")
    print("  an episode PASSES iff (a) lead>0 AND (b) lead exceeds its own")
    print(f"  {N_DRAWS}-draw block-bootstrap null's 90th percentile.")
    print(f"  proceed to Step B only if >= {MIN_EPISODES_PASS} of 6 episodes PASS.")
    print("=" * 78)
    for r in results:
        lead_str = f"{r['lead']:+.2f}d" if np.isfinite(r["lead"]) else "undefined"
        print(f"  {r['label']:42s} lead={lead_str:>10s}  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/6")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")

    print(f"\nconfigurations evaluated in this file's Step A: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session so far: "
          f"{bars.index.max()}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed)


# ==========================================================================
# STEP B -- built only if the gate above passes (banner item 4).
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, window_days: int, z_thresh: float,
                       horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """Volume-confirmed latch on the FASTEST anchor's own 0/1 vote (banner
    item 4): `meta_vote[i] = fast_anchor_vote[i]` on any bar where
    `volume_z[i] >= z_thresh` ("confirmed"); otherwise `meta_vote[i] =
    meta_vote[i-1]` (carries forward, exactly R-53/R-54/R-55's hysteresis-
    latch pattern, keyed on a volume gate rather than a level threshold-
    band). Before the first confirmed bar, defaults to the fast anchor's
    own then-current value (no dilution while unconfirmed).

    Causal: `volume_z` and each anchor vote are both causal (rolling/ffill
    constructions, row i depends only on rows <= i); the latch update at i
    depends only on values at <= i.
    """
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    vz = volume_z(df, window_days=window_days).to_numpy()
    confirmed = vz >= z_thresh

    n = len(df)
    meta = np.empty(n)
    meta[0] = fast_vote[0]
    for i in range(1, n):
        meta[i] = fast_vote[i] if confirmed[i] else meta[i - 1]
    return meta


def build_target_primary(df: pd.DataFrame) -> np.ndarray:
    """Target-construction function for the truncation causality probe,
    frozen at the pre-registered primary candidate."""
    return VolumeConfirmKelly(weight=1.0, window_days=VOL_BASELINE_DAYS,
                               z_thresh=Z_THRESH).prepare(df.copy())["target"].to_numpy()


class VolumeConfirmKelly(Strategy):
    """kelly_regime_v4 + a raw-volume-confirmed fast-anchor vote (R-84
    conservative, unregistered). Structurally v3/v4's own prepare(), with
    the plain 3-anchor average `frac = anchor_sum/3` replaced by
    `confirming_vote_frac(anchor_sum, meta_vote, weight)`. `weight=0` must
    recover v4 bit-for-bit. Not `@register`ed -- stays in experiments/ per
    docs/ROUTINE.md.
    """

    name = "r84_conservative_volume_confirm"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, weight: float = 1.0, window_days: int = VOL_BASELINE_DAYS,
                 z_thresh: float = Z_THRESH,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
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

        meta_vote = compute_meta_vote(df, self.window_days, self.z_thresh,
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

def run_identity_check(df_full: pd.DataFrame) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = VolumeConfirmKelly(weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe(df_full: pd.DataFrame) -> list[bool]:
    df = df_full.loc[:INNER_TRAIN_END].copy()
    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(build_target_primary, df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, weight: float, window_days: int, z_thresh: float,
                 tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = VolumeConfirmKelly(weight=weight, window_days=window_days,
                                        z_thresh=z_thresh)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES) -> dict:
    results = {}
    for weight in (0.5, 1.0, 2.0, 4.0):
        for window_days in (10, 20, 40):
            tag = f"w{weight} win{window_days} z{Z_THRESH}"
            results[("main", weight, window_days, Z_THRESH)] = eval_config(
                ev, SPOT, FUTURES, weight, window_days, Z_THRESH, tag)
    for z_thresh in (1.0, 2.0):
        tag = f"w1.0 win{VOL_BASELINE_DAYS} z{z_thresh}"
        results[("zsens", 1.0, VOL_BASELINE_DAYS, z_thresh)] = eval_config(
            ev, SPOT, FUTURES, 1.0, VOL_BASELINE_DAYS, z_thresh, tag)
    return results


def exposure_artifact_check(ev, DF, SPOT, FUTURES, weight: float = 1.0,
                             window_days: int = VOL_BASELINE_DAYS,
                             z_thresh: float = Z_THRESH) -> dict:
    """Diagnostic: regress the candidate's target series against a
    mean-notional-matched flat rescale of v4's own target series, on
    inner-validation, both markets. R^2 > 0.95 -> "just a rescale"."""
    v4 = get_strategy("kelly_regime_v4")
    cand = VolumeConfirmKelly(weight=weight, window_days=window_days, z_thresh=z_thresh)
    out = {}
    print(f"\nexposure-artifact check (weight={weight}, window_days={window_days}, "
          f"z_thresh={z_thresh}):")
    for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
        lo = int(DF.index.searchsorted(INNER_VAL_START))
        hi = int(DF.index.searchsorted(INNER_VAL_END, side="right"))
        prefix = min(lo, max(cand.warmup, v4.warmup))
        frame = DF.iloc[lo - prefix:hi]

        v4_prepared = v4.prepare(frame.copy())
        cand_prepared = cand.prepare(frame.copy())
        v4_t = v4_prepared["target"].to_numpy()[prefix:]
        cand_t = cand_prepared["target"].to_numpy()[prefix:]

        mean_abs_v4 = np.mean(np.abs(v4_t))
        mean_abs_cand = np.mean(np.abs(cand_t))
        alpha = mean_abs_cand / mean_abs_v4 if mean_abs_v4 > 0 else 0.0
        rescaled = alpha * v4_t
        ss_res = np.sum((cand_t - rescaled) ** 2)
        ss_tot = np.sum((cand_t - cand_t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {mkt_name:9s} mean|v4|={mean_abs_v4:.3f} mean|cand|={mean_abs_cand:.3f} "
              f"alpha={alpha:.3f}  R^2(cand vs alpha*v4)={r2:.4f}  "
              f"{'JUST A RESCALE' if r2 > 0.95 else 'genuinely different exposure shape'}")
        out[mkt_name] = r2
    return out


def run_eth_btc_falsification(ev, weight: float = 1.0,
                               window_days: int = VOL_BASELINE_DAYS,
                               z_thresh: float = Z_THRESH) -> dict:
    """ETH falsification + pre-2020 BTC control on the Bitfinex pair
    (R-77's established convention): both series' own full pre-2020
    history, both well before OOS_START, so no further truncation risk."""
    from tradebot.broker import MarketSpec

    spot = MarketSpec.spot()
    out = {}
    for name, path in (("BTC-control (bitfinex, pre-2020)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (bitfinex, pre-2020)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(DATA_DIR / path)
        assert_no_holdout(df)
        cand = VolumeConfirmKelly(weight=weight, window_days=window_days, z_thresh=z_thresh)
        v4 = get_strategy("kelly_regime_v4")
        m_v4 = ev(v4, df=df, market=spot, tag=f"{name}: v4")
        m_cand = ev(cand, df=df, market=spot, tag=f"{name}: candidate")
        delta = m_cand.sharpe - m_v4.sharpe
        print(f"  {name}: v4 sharpe={m_v4.sharpe:.2f}  candidate sharpe={m_cand.sharpe:.2f}  "
              f"delta={delta:+.2f}")
        out[name] = dict(v4=m_v4.sharpe, cand=m_cand.sharpe, delta=delta)
    return out


def run_step_b() -> None:
    from scripts.experiment import DF, FUTURES, SPOT, ev

    print("\n" + "=" * 78)
    print("STEP B (gate passed): sweep + mandatory checks")
    print("=" * 78)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(DF)
    n_configs += 1

    print("\n=== causality probe ===")
    run_causality_probe(DF)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=INNER_TRAIN_END)),
                                ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)

    print("\n=== sweep ===")
    sweep_results = run_sweep(ev, SPOT, FUTURES)
    n_configs += len(sweep_results)

    print("\n=== exposure-artifact check (primary) ===")
    exposure_artifact_check(ev, DF, SPOT, FUTURES)

    print("\n=== ETH falsification / pre-2020 BTC control ===")
    run_eth_btc_falsification(ev)

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
        print(f"usage: python experiments/r84_conservative_volume_confirm.py [{'|'.join(cmds)}]")
