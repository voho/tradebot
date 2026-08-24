#!/usr/bin/env python
"""R-120 CONSERVATIVE branch: Deribit calendar (quarterly-future) basis LEVEL,
z-scored against its own trailing baseline, as a confirming vote on
`kelly_regime_v4`'s 3-anchor gate -- Step A measurement gate first, this
project's established discipline for every INFO-axis round since R-53
(R-53/R-73/R-74/R-79/R-81/R-84).

Shared infrastructure (loader, causal front-quarter selector, anchor-vote
duplication, confirming-vote rule, stress table, null generator, causality
probe) lives in `experiments/r120_shared.py`, written and frozen by the
operator before dispatch. This file does not edit that module.

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). An extreme calendar basis reading -- deep
   contango (longs paying up for leveraged length via the quarterly) or
   backwardation (shorts/hedgers dominating) -- reflects a levered
   speculative or hedging position that often unwinds before slow
   price-based anchors (20/40/80-day rolling means) catch up, since it is
   priced by cash-and-carry arbitrageurs and term traders reacting in real
   time to the same information.

   Citations: Schmeling, Schrimpf & Todorov (2023, rev. 2025), "Crypto
   carry", BIS Working Papers No. 1087; Chi et al. (2023), "An empirical
   investigation on risk factors in cryptocurrency futures", Journal of
   Futures Markets (basis is one of three persistent crypto-futures
   factors, adapting Erb & Harvey 2006, Financial Analysts Journal 62(2));
   Zhang, T. (2026), "Funding Rate Mechanism in Perpetual Futures", SSRN
   6185958 (the structural argument for why a fixed-expiry calendar basis
   is a different statistical object from R-41's funding-reset spot-vs-
   perpetual basis, not a re-parameterization of it). Full citation set
   and the coverage-caveat measurement is in `r120_shared.py`'s module
   docstring -- not repeated here to keep one citation trail in one place
   (R-81/R-84's own convention).

   CONSTRAINT ATTACKED: INFO (one price series) -- Deribit's dated
   quarterly future is a genuinely new, independently-transacted
   instrument, not a transform of price already in this project's OHLCV.

   NOT A DUPLICATE OF: R-41/`kelly_regime_v9_basis_lead` (spot-vs-
   PERPETUAL basis -- different instrument, funding-reset every 8h vs a
   fixed quarterly expiry); B-05/R-35/R-39 (raw Binance funding rate, a
   COST-axis flat gate, not a term-structure quantity); R-73 (DVOL,
   implied volatility, not a forward price); R-81 (OI + top-trader
   long/short crowding, a positioning stock, not a priced term-structure
   quantity); R-63/R-76 (cross-COIN spot pairs -- this pairs the SAME coin
   across two MATURITIES, spot vs. its own dated future). Grepped
   docs/LEDGER.md for "term structure", "quarterly future", "roll yield",
   "contango", "backwardation", "calendar future", "dated future",
   "calendar basis": zero hits before this round (confirmed again in this
   file's own dispatch, see grep note in the results section below).

   THIS BRANCH TESTS THE BASIS'S OWN LEVEL (z-scored against its own
   trailing baseline), mirroring R-81's `ls_z` LEVEL construction and
   R-84's `volume_z` LEVEL construction. The sibling NOVEL branch tests
   the basis's own MOMENTUM/rate-of-change instead, a different,
   non-overlapping statistic -- this file does not test momentum.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy code,
   BTC only, on `r120_shared.USABLE_EPISODES_BTC` (4 of the full 6-episode
   table -- 2018 episodes unreachable, no Deribit quarterly contract
   existed before 2019; see `r120_shared.py`'s disclosed coverage
   measurement).

   PRIMARY FEATURE (chosen now, before any number): `basis_z` -- `ann_basis`
   (the annualized front-quarter calendar basis) z-scored against its own
   trailing `WINDOW_DAYS=20` mean/std, causal rolling (row i depends only
   on rows <= i). The 20-day window matches `kelly_regime_v4`'s own
   FASTEST anchor (20 days) -- the same reasoning R-84 used for
   `volume_z`'s window choice, and the timescale this round's Step B would
   ultimately tie the confirming vote to if reached.

   THRESHOLD: BIDIRECTIONAL, `|basis_z| >= 1.5`. Unlike R-84's `volume_z`
   (a one-sided participation-intensity rule -- only unusually HIGH volume
   is motivated by MDH/Easley&O'Hara), the basis can signal in either
   direction: deep contango (longs crowded) and deep backwardation
   (shorts/hedgers dominating) are both informative extremes, matching
   R-81's bidirectional `ls_z` reasoning exactly. 1.5-sigma matches this
   project's own standing "extreme" convention (R-81/R-84).

   EPISODE-LOCAL SEARCH WINDOW: [onset - 60 days, onset + 60 days], fixed
   before any number was computed -- identical to R-81/R-84's window and
   for the identical reason: `kelly_regime_v4`'s own anchors react to
   price with a lag (20/40/80-day rolling means), so its own reaction to a
   stress event is not necessarily dated to the event's onset, and the
   "nearest transition" search needs room on both sides.

   ANCHOR-GATE "FLIP" DEFINITION: reused verbatim from R-81/R-84's
   disclosed, bug-fixed convention (`r120_shared.nearest_transition`,
   `direction="down"`) -- all 4 usable episodes are bearish transitions,
   so "down" (majority DECREASING, the gate de-risking) is the relevant
   direction; "either direction" was disclosed by R-81 to silently pick a
   spurious blip and is not used here at all, not even as a candidate.

   BASIS "CROSSING" DEFINITION: the first BIDIRECTIONAL crossing (prior
   bar `|basis_z| < 1.5`, this bar `|basis_z| >= 1.5`) whose timestamp is
   nearest the episode's onset within the same window -- the same
   "nearest to onset" logic R-81/R-84 applied to their own candidate
   signal, for an apples-to-apples comparison.

   LEAD = (flip_time - crossing_time) in days. Positive = basis extremity
   reached before the gate's own nearest reaction.

   NULL: `r120_shared.block_bootstrap_lead_null(n_bars=<local window
   length>, block_days=5, n_draws=500, seed=120)` circularly block-shifts
   the LOCAL (episode-window) `basis_z` array and recomputes the "crossing
   nearest the REAL, fixed onset" against each shifted copy, compared to
   the fixed, real flip time -- identical construction to R-84's null.
   `seed=120` is this round's own number.

   PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed): an episode counts as a PASS if BOTH (a) LEAD > 0 (basis
   crossed before the gate's nearest reaction), AND (b) the true LEAD
   exceeds the 90th percentile of that same episode's own 500-draw
   block-bootstrap null lead distribution. PROCEED TO STEP B only if
   `>= MIN_EPISODES_PASS_BTC` (3 of 4, majority) episodes PASS. If fewer
   than 3 pass: STOP, report NEGATIVE with the full 4-episode table, do
   not write any Step B strategy code. The bar is not relaxed after
   seeing the numbers.

3. WHAT WOULD MAKE STEP A FAIL, named now: the same failure every one of
   the 17 prior INFO-axis signals in this ledger hit -- the basis
   extremity is reached AFTER (not before) the anchor gate's own nearest
   reaction, or a positive lead is not distinguishable from an arbitrary
   time-shift of the same series (i.e. generic autocorrelation, not a real
   early-warning property). The modal, fully-expected outcome is FAILURE,
   and a clean, well-documented negative is this branch's fully successful
   product if that is what happens -- this file does not force a positive
   result.

4. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step
   A's numbers exist; only executed if Step A's stop rule passes).

   CONFIRMING-VOTE CONSTRUCTION. `confirming_vote_frac` (imported
   unchanged from `r120_shared.py`) requires a DISCRETE {0,1} `meta_vote`
   (R-80's lesson, preserved by every confirming-vote round since).
   `meta_vote` tracks v4's FASTEST (20-day) anchor's own 0/1 vote, but
   only UPDATES on a bar where `|basis_z| >= Z_THRESH` (the vote is
   "confirmed" that bar); on bars without basis confirmation, `meta_vote`
   carries forward its last confirmed value -- the identical hysteresis-
   latch pattern R-84's `volume`-confirmed vote used, keyed on a basis gate
   instead of a volume gate. Before the first confirmed bar, `meta_vote`
   defaults to the fast anchor's own then-current value (no dilution in
   either direction while unconfirmed).

   frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
        = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity check,
   run first, before any swept configuration).

   SWEEP GRID (fixed a priori, not tuned to any inner-validation number):
   - `weight` in {0.5, 1.0, 2.0, 4.0} x `window_days` (basis_z's own
     trailing baseline) in {10, 20, 40} -- 12 configurations. 20 is the
     Step-A primary; 10/40 bracket it 2x on each side.
   - `Z_THRESH` sensitivity at the pre-registered primary point
     (weight=1.0, window_days=20) over {1.0, 2.0} -- 2 configurations
     (1.5 is already covered by the main grid's weight=1.0/window=20
     cell).
   - identity check (weight=0): 1 configuration.
   Total Step B configurations, if reached: 15. Evaluated on train
   (2019-01-01 -> 2020-12-31 -- basis coverage only starts 2019-01-01, NOT
   the full 2017 inner-train start) and inner-validation (2021-01-01 ->
   2022-12-31), both spot and futures_5x markets.

   MANDATORY CHECKS (this project's standing discipline for every
   confirming-vote round that reaches Step B): (i) exposure-artifact R^2
   -- regress the candidate's `target` series against a mean-notional-
   matched flat rescale of `kelly_regime_v4`'s own `target` series on
   inner-validation, both markets; R^2 > 0.95 = "just a rescale", fail;
   (ii) ETH falsification (`data/ethusd_deribit_quarterly_5m.csv.gz` +
   `load_coinbase_eth_spot`, `USABLE_EPISODES_ETH`,
   `ETH_BASIS_COVERAGE_START` from `r120_shared` -- same sign of edge must
   replicate); (iii) `truncation_causality_probe` on this basis-confirmed
   meta-vote construction; (iv) the 0.40% fee tier via
   `scripts/fee_study.py`'s convention if this round reaches this stage.

   PRE-REGISTERED HOLDOUT DECISION RULE (fixed now, contingent on reaching
   Step B and everything above clearing): read the 2023+ holdout ONLY IF
   ALL of (a) the primary configuration's inner-validation Sharpe
   improvement over `kelly_regime_v4` exceeds the +/-0.2 noise floor
   (R-20) on BOTH markets, on a genuine parameter PLATEAU (neighbouring
   grid cells, not an isolated peak); (b) exposure-artifact R^2 <= 0.95
   both markets; (c) the ETH falsification replicates the same sign of
   edge; (d) the causality probe passes. If ANY of these fail, this branch
   reports NEGATIVE and the holdout is never read -- an honest negative at
   any stage is this project's own definition of a complete, successful
   piece of work.

5. CONFIGS EVALUATED IN STEP A: 0 (a fixed, non-swept measurement gate,
   this project's standing accounting convention for this exact
   construction -- R-53/R-73/R-74/R-79/R-81/R-84's own Step-A studies).
   Step B's count, if reached, is 15 as itemized above.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r120_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BTC_BASIS_COVERAGE_START,
    ETH_BASIS_COVERAGE_START,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_EPISODES_PASS_BTC,
    OOS_START,
    USABLE_EPISODES_BTC,
    USABLE_EPISODES_ETH,
    V4_BAND,
    V4_HORIZONS,
    anchor_majority,
    anchor_votes,
    block_bootstrap_lead_null,
    confirming_vote_frac,
    front_quarter_basis,
    load_deribit_quarterly,
    nearest_transition,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
Z_THRESH = 1.5
SEARCH_WINDOW_DAYS = 60
BASIS_BASELINE_DAYS = 20
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 120

# This round's train split starts at basis coverage, not the project's
# usual 2017-01-01 inner-train start (banner item 4).
BASIS_TRAIN_START = BTC_BASIS_COVERAGE_START


# ---------------------------------------------------------------- holdout guard

def assert_no_holdout(df: pd.DataFrame) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before OOS_START. Independent of any truncation already done
    at load time (R-79/R-81/R-84's own convention)."""
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


def compute_basis_z(bars: pd.DataFrame, quarterly: pd.DataFrame,
                     window_days: int = BASIS_BASELINE_DAYS) -> pd.Series:
    """`ann_basis` z-scored against its own trailing `window_days` mean/std.

    Causal: `front_quarter_basis` is causal by construction (merge_asof
    direction="backward"); `.rolling(window)` is trailing (row i depends
    only on rows <= i), matching `anchor_votes`' own rolling-mean
    construction with the same "full window required" default
    (min_periods == window, no partial-window leakage into the z-score's
    early bars).
    """
    ann_basis = front_quarter_basis(bars, quarterly)["ann_basis"]
    win_bars = int(window_days * BARS_PER_DAY)
    roll_mean = ann_basis.rolling(win_bars).mean()
    roll_std = ann_basis.rolling(win_bars).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (ann_basis - roll_mean) / roll_std
    return z.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------- flip / crossing

def nearest_crossing_bidirectional(z: np.ndarray, window: pd.DatetimeIndex,
                                    onset: pd.Timestamp,
                                    thresh: float = Z_THRESH) -> pd.Timestamp | None:
    """First BIDIRECTIONAL crossing (prior bar |z|<thresh, this bar
    |z|>=thresh) whose timestamp is closest to `onset` in `window`. NaN
    entries are treated as "not above threshold" (never trigger or clear a
    crossing on their own)."""
    above = np.abs(z) >= thresh
    above = np.where(np.isnan(z), False, above)
    cross = np.zeros(len(z), dtype=bool)
    cross[1:] = above[1:] & ~above[:-1]
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars_index: pd.DatetimeIndex, onset_str: str,
                    window_days: int = SEARCH_WINDOW_DAYS) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars_index[(bars_index >= lo) & (bars_index <= hi)]
    return onset, window


# --------------------------------------------------------------------- null

def episode_null_leads(basis_z: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, flip_time: pd.Timestamp,
                        n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                        seed: int = NULL_SEED) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the LOCAL basis_z array (within `window`) and recompute the
    "crossing nearest the real, unshifted onset" against the fixed, real
    `flip_time`."""
    local = basis_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_lead_null(n_bars=n_bars, block_days=block_days,
                                        n_draws=n_draws, seed=seed)

    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        cross_time = nearest_crossing_bidirectional(shifted, window, onset)
        if cross_time is None:
            continue
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads


# --------------------------------------------------------------------- gate

def gate() -> dict:
    print("=" * 78)
    print("R-120 CONSERVATIVE: calendar-basis LEVEL confirming vote -- STEP A lead-time gate")
    print("=" * 78)

    bars = load_btc_bars()
    quarterly = load_deribit_quarterly(DATA_DIR, asset="BTC")
    assert quarterly is not None, "BTC Deribit quarterly file missing"
    # Truncate the raw contract file's own rows to strictly before OOS_START
    # too (not just `bars`) -- the committed file's last contract expires
    # 2023-03-31, so its own price rows extend past the holdout even though
    # `front_quarter_basis`'s causal merge_asof would never actually use a
    # row after `bars`' own max timestamp. Truncating removes that
    # never-used tail explicitly rather than relying on the merge alone.
    quarterly = quarterly.loc[quarterly.index < pd.Timestamp(OOS_START, tz=quarterly.index.tz)].copy()
    assert_no_holdout(quarterly)
    majority = anchor_majority(bars)
    majority_arr = majority.to_numpy()
    basis_z = compute_basis_z(bars, quarterly, window_days=BASIS_BASELINE_DAYS)

    n_nonnan = int(basis_z.notna().sum())
    print(f"\nBTC quarterly: {len(quarterly):,} raw rows, "
          f"{quarterly['instrument'].nunique()} contracts")
    print(f"basis_z: {n_nonnan:,}/{len(basis_z):,} bars non-NaN "
          f"({basis_z.first_valid_index()} -> {basis_z.last_valid_index()})")
    print(f"\nprimary feature: basis_z (ann_basis z-scored, {BASIS_BASELINE_DAYS}-day "
          f"trailing baseline)  threshold: BIDIRECTIONAL |basis_z|>={Z_THRESH}  "
          f"search window=+/-{SEARCH_WINDOW_DAYS}d  null: {N_DRAWS} draws, "
          f"block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in USABLE_EPISODES_BTC:
        onset, window = episode_window(bars.index, onset_str)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars in range "
                  f"-- outside data coverage. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=None, cross=None,
                                 lead=float("nan"), pass_a=False, pass_b=False,
                                 null_p90=float("nan")))
            continue

        flip_time = nearest_transition(majority_arr, bars.index, onset,
                                        SEARCH_WINDOW_DAYS, direction="down")
        local_bz = basis_z.reindex(window).to_numpy()
        cross_time = nearest_crossing_bidirectional(local_bz, window, onset)

        if flip_time is None or cross_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no basis_z crossing'} "
                  f"found in +/-{SEARCH_WINDOW_DAYS}d window. FAIL by construction "
                  f"(lead undefined).")
            results.append(dict(label=label, onset=onset_str, flip=flip_time,
                                 cross=cross_time, lead=float("nan"),
                                 pass_a=False, pass_b=False, null_p90=float("nan")))
            continue

        lead = (flip_time - cross_time).total_seconds() / 86400.0
        null_leads = episode_null_leads(basis_z, window, onset, flip_time)
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
        print(f"    basis_z nearest crossing (|z|>={Z_THRESH}): {cross_time}")
        print(f"    LEAD = {lead:+.2f} days  "
              f"({'basis LED' if lead > 0 else 'basis LAGGED/coincided'})")
        print(f"    null ({N_DRAWS} draws): median={null_median:+.2f}d  p90={null_p90:+.2f}d  "
              f"(valid draws: {len(valid_null)}/{N_DRAWS})")
        print(f"    PASS (a) lead>0: {pass_a}   PASS (b) lead > null p90: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=str(flip_time),
                             cross=str(cross_time), lead=lead, pass_a=pass_a,
                             pass_b=pass_b, null_p90=null_p90,
                             null_median=null_median))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= MIN_EPISODES_PASS_BTC

    print("\n" + "=" * 78)
    print("PRE-REGISTERED STOP RULE (fixed before any number above was computed):")
    print("  an episode PASSES iff (a) lead>0 AND (b) lead exceeds its own")
    print(f"  {N_DRAWS}-draw block-bootstrap null's 90th percentile.")
    print(f"  proceed to Step B only if >= {MIN_EPISODES_PASS_BTC} of "
          f"{len(USABLE_EPISODES_BTC)} episodes PASS.")
    print("=" * 78)
    for r in results:
        lead_str = f"{r['lead']:+.2f}d" if np.isfinite(r["lead"]) else "undefined"
        print(f"  {r['label']:42s} lead={lead_str:>10s}  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/{len(USABLE_EPISODES_BTC)}")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")

    print(f"\nconfigurations evaluated in this file's Step A: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session so far: "
          f"{max(bars.index.max(), quarterly.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed)


# ==========================================================================
# STEP B -- built only if the gate above passes (banner item 4).
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, quarterly: pd.DataFrame, window_days: int,
                       z_thresh: float, horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """Basis-confirmed latch on the FASTEST anchor's own 0/1 vote (banner
    item 4): `meta_vote[i] = fast_anchor_vote[i]` on any bar where
    `|basis_z[i]| >= z_thresh` ("confirmed"); otherwise `meta_vote[i] =
    meta_vote[i-1]` (carries forward, R-53/R-84's hysteresis-latch pattern,
    keyed on a basis gate instead of a volume gate).

    Causal: `basis_z` and each anchor vote are both causal (rolling/ffill
    constructions, row i depends only on rows <= i); the latch update at i
    depends only on values at <= i.
    """
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    bz = compute_basis_z(df, quarterly, window_days=window_days).to_numpy()
    confirmed = np.abs(bz) >= z_thresh

    n = len(df)
    meta = np.empty(n)
    meta[0] = fast_vote[0]
    for i in range(1, n):
        meta[i] = fast_vote[i] if confirmed[i] else meta[i - 1]
    return meta


class BasisLevelConfirmKelly(Strategy):
    """kelly_regime_v4 + a Deribit-calendar-basis-LEVEL-confirmed fast-anchor
    vote (R-120 conservative, unregistered). Structurally v3/v4's own
    prepare(), with the plain 3-anchor average `frac = anchor_sum/3`
    replaced by `confirming_vote_frac(anchor_sum, meta_vote, weight)`.
    `weight=0` must recover v4 bit-for-bit. Not `@register`ed -- stays in
    experiments/ per docs/ROUTINE.md.
    """

    name = "r120_conservative_basis_level_confirm"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, quarterly: pd.DataFrame, weight: float = 1.0,
                 window_days: int = BASIS_BASELINE_DAYS, z_thresh: float = Z_THRESH,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.quarterly = quarterly
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

        meta_vote = compute_meta_vote(df, self.quarterly, self.window_days,
                                       self.z_thresh, horizons=self.horizons,
                                       band=self.band)
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

def run_identity_check(df_full: pd.DataFrame, quarterly: pd.DataFrame) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = BasisLevelConfirmKelly(quarterly, weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe(df_full: pd.DataFrame, quarterly: pd.DataFrame) -> list[bool]:
    df = df_full.loc[:INNER_TRAIN_END].copy()

    def build_target(frame: pd.DataFrame) -> np.ndarray:
        q = quarterly.loc[quarterly.index <= frame.index.max()]
        return BasisLevelConfirmKelly(q, weight=1.0, window_days=BASIS_BASELINE_DAYS,
                                       z_thresh=Z_THRESH).prepare(frame.copy())["target"].to_numpy()

    results = []
    for check_at in (150_000, 250_000, 350_000):
        if check_at >= len(df):
            continue
        ok = truncation_causality_probe(build_target, df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, quarterly, weight: float, window_days: int,
                 z_thresh: float, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(start=BASIS_TRAIN_START, end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
            strat = BasisLevelConfirmKelly(quarterly, weight=weight, window_days=window_days,
                                            z_thresh=z_thresh)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES, quarterly) -> dict:
    results = {}
    for weight in (0.5, 1.0, 2.0, 4.0):
        for window_days in (10, 20, 40):
            tag = f"w{weight} win{window_days} z{Z_THRESH}"
            results[("main", weight, window_days, Z_THRESH)] = eval_config(
                ev, SPOT, FUTURES, quarterly, weight, window_days, Z_THRESH, tag)
    for z_thresh in (1.0, 2.0):
        tag = f"w1.0 win{BASIS_BASELINE_DAYS} z{z_thresh}"
        results[("zsens", 1.0, BASIS_BASELINE_DAYS, z_thresh)] = eval_config(
            ev, SPOT, FUTURES, quarterly, 1.0, BASIS_BASELINE_DAYS, z_thresh, tag)
    return results


def exposure_artifact_check(ev, DF, SPOT, FUTURES, quarterly, weight: float = 1.0,
                             window_days: int = BASIS_BASELINE_DAYS,
                             z_thresh: float = Z_THRESH) -> dict:
    """Diagnostic: regress the candidate's target series against a
    mean-notional-matched flat rescale of v4's own target series, on
    inner-validation, both markets. R^2 > 0.95 -> "just a rescale"."""
    v4 = get_strategy("kelly_regime_v4")
    cand = BasisLevelConfirmKelly(quarterly, weight=weight, window_days=window_days,
                                   z_thresh=z_thresh)
    out = {}
    print(f"\nexposure-artifact check (weight={weight}, window_days={window_days}, "
          f"z_thresh={z_thresh}):")
    for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
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


def run_eth_falsification(ev, weight: float = 1.0,
                           window_days: int = BASIS_BASELINE_DAYS,
                           z_thresh: float = Z_THRESH) -> dict:
    """ETH falsification using `load_coinbase_eth_spot` +
    `ethusd_deribit_quarterly_5m.csv.gz` (banner item 4)."""
    from tradebot.broker import MarketSpec

    spot_spec = MarketSpec.spot()
    eth_spot = load_coinbase_eth_spot(DATA_DIR)
    assert eth_spot is not None, "ETH Coinbase spot file missing"
    eth_spot = eth_spot.loc[eth_spot.index < pd.Timestamp(OOS_START, tz=eth_spot.index.tz)].copy()
    assert_no_holdout(eth_spot)
    eth_quarterly = load_deribit_quarterly(DATA_DIR, asset="ETH")
    assert eth_quarterly is not None, "ETH Deribit quarterly file missing"
    eth_quarterly = eth_quarterly.loc[
        eth_quarterly.index < pd.Timestamp(OOS_START, tz=eth_quarterly.index.tz)].copy()
    assert_no_holdout(eth_quarterly)

    v4 = get_strategy("kelly_regime_v4")
    cand = BasisLevelConfirmKelly(eth_quarterly, weight=weight, window_days=window_days,
                                   z_thresh=z_thresh)
    m_v4 = ev(v4, df=eth_spot, market=spot_spec, tag="ETH: v4",
              start=ETH_BASIS_COVERAGE_START, end=INNER_VAL_END)
    m_cand = ev(cand, df=eth_spot, market=spot_spec, tag="ETH: candidate",
                start=ETH_BASIS_COVERAGE_START, end=INNER_VAL_END)
    delta = m_cand.sharpe - m_v4.sharpe
    print(f"  ETH (Coinbase spot + Deribit quarterly, {ETH_BASIS_COVERAGE_START} -> "
          f"{INNER_VAL_END}): v4 sharpe={m_v4.sharpe:.2f}  "
          f"candidate sharpe={m_cand.sharpe:.2f}  delta={delta:+.2f}")
    return dict(v4=m_v4.sharpe, cand=m_cand.sharpe, delta=delta)


def run_step_b(gate_result: dict) -> dict:
    from scripts.experiment import DF, FUTURES, SPOT, ev

    print("\n" + "=" * 78)
    print("STEP B (gate passed): sweep + mandatory checks")
    print("=" * 78)

    quarterly = load_deribit_quarterly(DATA_DIR, asset="BTC")
    quarterly = quarterly.loc[quarterly.index < pd.Timestamp(OOS_START, tz=quarterly.index.tz)].copy()
    assert_no_holdout(quarterly)

    step_b = {"n_configs": 0}

    print("\n=== identity check ===")
    step_b["identity_max_diff"] = run_identity_check(DF, quarterly)
    step_b["n_configs"] += 1

    print("\n=== causality probe ===")
    step_b["causality_probe"] = run_causality_probe(DF, quarterly)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (
            ("train", dict(start=BASIS_TRAIN_START, end=INNER_TRAIN_END)),
            ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
        ):
            for mkt_name, mkt in (("spot", SPOT), ("futures_5x", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)

    print("\n=== sweep (15 configs, incl. identity) ===")
    sweep_results = run_sweep(ev, SPOT, FUTURES, quarterly)
    step_b["n_configs"] += len(sweep_results)

    print("\n=== exposure-artifact check (primary) ===")
    step_b["exposure_r2"] = exposure_artifact_check(ev, DF, SPOT, FUTURES, quarterly)

    print("\n=== ETH falsification ===")
    step_b["eth_falsification"] = run_eth_falsification(ev)

    print(f"\nTotal Step B configurations evaluated: {step_b['n_configs']}")
    return step_b


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    gate_result = gate()
    out = {"branch": "r120_conservative_basis_level", "gate": gate_result}
    if gate_result["passed"]:
        out["step_b"] = run_step_b(gate_result)
    else:
        print("\nSTEP A FAILED the pre-registered stop rule. Per this file's own "
              "pre-registration, no strategy is built and no Step-B code runs. "
              "This gate result is this branch's whole product.")

    results_path = ROOT / "experiments" / "r120_conservative_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {results_path}")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmds = {"gate": gate, "main": main}
    choice = sys.argv[1] if len(sys.argv) > 1 else "main"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r120_conservative_basis_level.py [{'|'.join(cmds)}]")
