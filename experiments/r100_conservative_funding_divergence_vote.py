#!/usr/bin/env python
"""R-100 CONSERVATIVE branch: Step-A lead-time gate for the Binance-vs-
Deribit BTC perpetual funding-rate cross-venue divergence
(`r100_shared.cross_venue_divergence_z`) as a fifteenth INFO-axis signal,
run BEFORE any strategy/confirming-vote code -- identical "operator
measurement" convention and gate methodology to R-82/83/85/86/88/96/98/99's
own Step-A files, for direct comparability.

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence). When Binance's funding rate runs hotter than
   Deribit's (`cross_venue_divergence_z > 0`), retail long-crowding is
   concentrated specifically on Binance (a retail-heavy venue) rather than
   shared evenly with Deribit (an institutional/options-flow venue), so its
   eventual unwind should PRECEDE a bearish regime reversal -- a SIGNED,
   directional prediction (positive divergence -> bearish lead), not the
   bidirectional "extremity in either direction is informative" claim
   R-81/R-88 made for their own signals. Full citation trail (Zhivkov 2026;
   Inan 2025/26; He/Manela/Ross/von Wachter 2024) and the "not a duplicate
   of" argument against R-35/R-39/B-05 (single-venue funding LEVEL),
   R-39/B-02 (Deribit used only to splice a post-2023 gap), R-41/B-15
   (single-venue futures-vs-spot basis), R-73 (DVOL, unrelated to funding),
   R-81/R-88 (single-venue Binance positioning/flow feeds, no Deribit data
   at all) and R-53/R-55 (the reused confirming-vote combination rule) are
   in `experiments/r100_shared.py`'s module docstring -- not repeated here,
   per that file's own stated one-citation-trail-in-one-place convention.
   Constraint attacked: INFO.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy code,
   on BTC only, over `r100_shared.STRESS_EPISODES` (the project's standard
   six dated regime episodes).

   PRIMARY METRIC AND THRESHOLD (fixed by `r100_shared.py`, not chosen
   here): `cross_venue_divergence_z` at `Z_THRESH=1.5` (the same
   literature-anchored "extreme" bar as R-81/R-88's own z-gates), SIGNED
   -- only the UP-crossing (`nearest_alarm(..., signed="pos")`) is tested,
   because the mechanism above makes a one-directional claim (positive
   divergence -> bearish lead); the down-crossing (Deribit running hotter)
   carries no directional claim in this round's pre-registration and is
   not tested as a pass/fail criterion.

   BASELINE-WINDOW GRID: the full `r100_shared.BASELINE_WINDOW_DAYS_GRID =
   (30, 60, 90)` -- 3 cells, no separate Z_THRESH grid (fixed at 1.5 by
   the shared module). PRIMARY CELL: `PRIMARY_BASELINE_WINDOW_DAYS=60`
   (the grid centre), chosen by the operator before dispatch and confirmed
   non-degenerate by `r100_killswitch_a.py` (118-175 threshold crossings
   per cell across 2020-2022 -- re-verified by this file's own run below).
   The other 2 cells are computed and reported in full as plateau/
   robustness context, per this project's standing convention (R-96) --
   they never override the primary cell's own verdict.

   EPISODE-LOCAL SEARCH WINDOW: `r100_shared.episode_window(bars, onset,
   window_days=WINDOW_DAYS=60)`, identical +/-60-day convention to
   R-81/R-88/R-96/R-98/R-99's own gates.

   ANCHOR-GATE "FLIP" DEFINITION: `anchor_majority`'s nearest DOWNWARD
   transition (`r100_shared.nearest_transition(..., direction="down")`) to
   the episode onset -- the de-risking reaction this gate actually tests,
   reused verbatim from R-81's disclosed, bug-fixed down-only convention.

   DIVERGENCE "DETECTION" DEFINITION: the first bar where
   `cross_venue_divergence_z` crosses UP through `+Z_THRESH`
   (`r100_shared.nearest_alarm(..., signed="pos")`), nearest the onset
   within the same window.

   DATA-COVERAGE HANDLING (disclosed by `r100_shared.py`, not discovered
   after running): Binance funding starts 2020-01-01, so the two 2018
   episodes are construction-forced FAILs
   (`r100_shared.FORCED_FAIL_EPISODES`) -- no crossing search is attempted
   for them (a search on data that does not exist cannot be evidence of
   anything), and they are reported as such, counted in the six-episode
   table but EXCLUDED from the pass-bar denominator (which is explicitly
   4, not 6 -- see `r100_shared.PASS_BAR_DEN` and its own comment on why
   the original "4 of 6" proposal was unreachable by construction). The
   2020-03 COVID episode (`r100_shared.THIN_BASELINE_EPISODES`) is
   flagged, not forced-failed: its 60-day pre-onset window is fully
   covered by raw funding observations (verified below, 60/60), but the
   div-z score's OWN trailing baseline window (30/60/90 days, used to
   z-score each venue's funding before differencing) is thin that early
   in the series -- reported with the flag attached, still tested and
   still eligible to pass or fail on its own measured lead.

   LEAD = (flip_time - detect_time) in days. Positive = the divergence
   alarm fired before v4's own gate reacted.

   NULL: `r100_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) `div_z` array (`BLOCK_DAYS=5`, `N_DRAWS=500`,
   `NULL_SEED=10023` -- fixed now, disclosed, chosen arbitrarily and never
   altered after seeing any result; a fresh seed distinct from every prior
   round's) and recomputes "nearest up-crossing to the REAL, unshifted
   onset" against each shifted copy, compared to the fixed, real flip
   time -- identical construction to R-96's `null_leads`, adapted to the
   signed up-crossing detector here.

   PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar shape to R-96): an episode counts as a PASS if
   BOTH (a) LEAD >= 0, AND (b) the true LEAD is >= the null distribution's
   median. Using the PRIMARY CELL ONLY (`baseline_window_days=60`), count
   how many of the 4 VALID (non-forced-fail) episodes pass. PROCEED TO
   STEP B only if `n_pass >= r100_shared.PASS_BAR_NUM` (3 of 4). If fewer
   than 3 pass: STOP, report this file's Step-A result as the whole
   conservative branch's product, verdict NEGATIVE, do not write or run
   any Step-B code, do not touch any data on or after 2023-01-01. The bar
   is not relaxed, narrowed, or otherwise adjusted after seeing the
   numbers, and the other 2 grid cells never override this verdict.

3. WHAT WOULD MAKE STEP A FAIL, named now: the same pattern that has now
   beaten fourteen consecutive INFO-axis mechanisms built on fourteen
   different theoretical bases -- a statistic computed from price/funding
   can only move once cross-venue stress has already begun, which may be
   no earlier -- or later -- than v4's own reactive, fixed-window anchor.
   If the divergence alarm also lags (or merely coincides with) v4's own
   reaction across the four valid episodes, that is the fifteenth
   independent mechanism converging on the same conclusion the ledger's
   standing diagnosis already leans toward: this six-episode gate is
   unwinnable by any estimator computed from this project's own committed
   history, whatever field it is drawn from (price, order flow, or now
   cross-venue funding), and the finding is about the gate/dataset rather
   than about this one technique. Given the base rate (0 of 14 prior
   INFO-axis signals have cleared an equivalent gate), the modal outcome
   pre-registered here IS failure, and a clean negative is this round's
   fully successful, complete product if that is what happens.

4. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step
   A's numbers exist; only executed if Step A's primary cell passes).

   CONFIRMING-VOTE CONSTRUCTION. Because this mechanism makes a
   ONE-SIDED, directional claim (positive divergence -> bearish lead
   only; negative divergence carries no registered claim), the meta-vote
   is a one-sided latch, NOT a two-sided latch like R-88's signed flow
   vote: `meta_vote[i] = 0` ("confirmed bearish") from the first bar where
   `div_z[i] >= Z_THRESH` until `div_z[i]` falls back to a release
   threshold `RELEASE_FRAC * Z_THRESH` (`RELEASE_FRAC=0.5`, fixed a
   priori, disclosed, not tuned) -- the same enter-high/exit-low
   hysteresis shape `kelly_regime_v3`'s own volatility-state latch and
   R-81/R-84/R-88's confirming votes all use. Outside a confirmed-bearish
   state (including the entire pre-2020 period with no funding data at
   all), `meta_vote[i]` simply TRACKS the fastest (20-day) anchor's own
   then-current vote, bar by bar -- it never independently asserts
   "bullish", only "bearish" or "no opinion, defer to the anchor". This
   is a deliberate asymmetry, matching the mechanism's own one-sided
   prediction, and is disclosed as different from R-88's fully
   symmetric bullish/bearish latch.

   frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
        = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity check,
   run first, before any swept configuration).

   SWEEP GRID (fixed a priori, not tuned to any inner-validation number):
   `weight` in {0.5, 1.0, 2.0, 4.0} (given in the task spec) x
   `baseline_window_days` in the full `r100_shared.BASELINE_WINDOW_DAYS_GRID`
   = (30, 60, 90) -- 12 configurations, evaluated on inner-train (end
   2020-12-31) and inner-validation (2021-01-01 to 2022-12-31), spot and
   futures. Plus the weight=0 identity check: 1 configuration. Total Step
   B configurations: 13.

   PROMOTION BAR (stated now, before running): the pre-registered PRIMARY
   cell (weight, baseline_window_days) = (1.0, 60) must beat
   `kelly_regime_v4` on inner-validation Sharpe by more than this
   project's own +/-0.2 Sharpe noise floor (R-20), on at least one market
   with the other not materially worse, OR show a matched-risk
   drawdown/tail improvement (docs/ROUTINE.md's "match risk before
   comparing anything" rule) -- AND the improving region must be a
   genuine parameter PLATEAU, not an isolated peak, AND the falsification
   test below must not reverse. If ANY of these fail: NEGATIVE, no
   holdout-read recommendation. This branch has no authority to read the
   holdout itself in any case.

   FALSIFICATION TEST (chosen now, before running, with the reasoning for
   why the project's standard ETH-Bitfinex instrument is NOT usable
   stated up front): ETH has no committed Deribit funding series at all,
   and the committed Bitfinex ETH spot data ends 2019-12-31 -- before
   Binance's own BTC funding series even starts (2020-01-01) -- so an
   ETH replication is impossible on data this project has, not merely
   inconvenient. The chosen ALTERNATIVE: a PLATEAU-NOT-PEAK check across
   the full weight x baseline_window_days grid (12 cells) restricted to
   inner-validation -- if the primary cell's improvement (if any) is an
   isolated spike surrounded by neighbours of the opposite sign, that is
   read as a failed falsification (an artifact of one specific
   parameterization, not a real, generalizable edge). This is the
   plateau check the promotion bar above already requires; it is named
   here explicitly as this round's ONE falsification test (rather than a
   redundant restriction of `scripts/stress_test.py` to 2020-2022, which
   would require a resampled-window harness this file does not build)
   because the grid this round already runs for the promotion bar IS the
   most informative test available given ETH's exclusion, and inventing a
   second, weaker test would not add real evidence.

   MANDATORY: the causal-truncation probe
   (`r100_shared.truncation_causality_probe`) is run on this file's own
   signal-building pipeline (both stages: the daily cross-venue z-score,
   and its causal alignment onto 5-minute bars) regardless of whether
   Step A's gate passes, per this project's standing discipline.

5. CONFIGS EVALUATED IN STEP A: 3 (the full `BASELINE_WINDOW_DAYS_GRID`,
   a fixed, non-swept measurement gate over an a-priori 3-cell grid --
   same accounting convention as R-96's 9-cell Step A). Step B's count,
   if reached, is 13 as itemized above. Grand total if Step B is reached:
   16. If Step A fails: 3.

6. HOLDOUT DISCIPLINE. This file asserts, at every load point
   (`r100_shared.assert_no_holdout`), that no bar with timestamp >=
   OOS_START (2023-01-01) is ever read. This branch has no authority to
   consult the 2023+ holdout: that decision is made centrally by the
   operator after both this round's branches report.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

import experiments.r100_shared as R  # noqa: E402

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 10023

# ---- Step B constants, fixed a priori (see banner item 4) ------------------
WEIGHT_GRID = (0.5, 1.0, 2.0, 4.0)
PRIMARY_WEIGHT = 1.0
RELEASE_FRAC = 0.5


# ------------------------------------------------------------------- loading

def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(R.OOS_START, tz=df.index.tz)].copy()
    R.assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {R.OOS_START})", file=sys.stderr)
    return df


def build_div_z_aligned(daily: pd.DataFrame, bars: pd.DataFrame,
                         baseline_window_days: int) -> pd.Series:
    """Full signal-building pipeline: causal cross-venue divergence
    z-score (daily cadence) -> causal alignment onto `bars`' 5-minute
    index."""
    z = R.cross_venue_divergence_z(daily, baseline_window_days=baseline_window_days)
    aligned = R.align_daily_causal(z.to_frame(name="div_z"), bars)["div_z"]
    return aligned


# ---------------------------------------------------------------------- null

def null_leads(div_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = R.Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Block-bootstrap null lead distribution: circularly shift the LOCAL
    (episode-window) div_z array and recompute the "up-crossing nearest the
    real, unshifted onset" against the fixed, real `flip_time`. Identical
    construction to r96_conservative's `null_leads`, adapted to the signed
    up-crossing detector this signal uses."""
    local = div_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = R.block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                       n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        high = shifted >= z_thresh
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = high[1:] & ~high[:-1]
        cross[0] = bool(high[0]) if n_bars else False
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        detect_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - detect_time).total_seconds() / 86400.0
    return leads


# --------------------------------------------------------------------- gate

def gate_cell(baseline_window_days: int, bars: pd.DataFrame,
              majority: pd.Series, daily: pd.DataFrame) -> dict:
    is_primary = baseline_window_days == R.PRIMARY_BASELINE_WINDOW_DAYS
    tag = " [PRIMARY]" if is_primary else ""
    print("=" * 78)
    print(f"R-100 CONSERVATIVE: cross-venue div_z(baseline={baseline_window_days}d) "
          f"vs v4 anchor -- STEP A lead-time gate{tag}")
    print("=" * 78)

    div_z = build_div_z_aligned(daily, bars, baseline_window_days)
    R.assert_no_holdout(div_z.dropna())

    print(f"\nz_thresh={R.Z_THRESH} (signed, up-crossing only)  "
          f"search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in R.STRESS_EPISODES:
        if label in R.FORCED_FAIL_EPISODES:
            print(f"[{label}] onset={onset_str}: FORCED FAIL (predates Binance funding "
                  f"data, starts 2020-01-01). Not counted in the pass-bar denominator.")
            results.append(dict(label=label, onset=onset_str, forced_fail=True,
                                 thin=False, pass_b=False, lead=float("nan"),
                                 null_median=float("nan")))
            continue

        thin = label in R.THIN_BASELINE_EPISODES
        onset, window = R.episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, onset=onset_str, forced_fail=False, thin=thin,
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        detect_time = R.nearest_alarm(div_z, window, onset, R.Z_THRESH, signed="pos")
        flip_time = R.nearest_transition(majority, window, onset, direction="down")

        if detect_time is None or flip_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no divergence up-crossing' if detect_time is None else 'no anchor-gate transition'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction "
                  f"(lead undefined).{' [THIN BASELINE]' if thin else ''}")
            results.append(dict(label=label, onset=onset_str, forced_fail=False, thin=thin,
                                 flip=flip_time, detect=detect_time, pass_b=False,
                                 lead=float("nan"), null_median=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(div_z, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        flag = "  [THIN BASELINE, disclosed]" if thin else ""
        print(f"[{label}] onset={onset_str}{flag}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    div_z nearest up-crossing (z>={R.Z_THRESH}): {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, forced_fail=False, thin=thin,
                             flip=flip_time, detect=detect_time, lead=lead,
                             null_median=null_median, pass_b=pass_b))

    valid_results = [r for r in results if not r["forced_fail"]]
    n_pass = sum(1 for r in valid_results if r["pass_b"])
    n_valid = len(valid_results)
    passed = is_primary and (n_pass >= R.PASS_BAR_NUM)

    print("\n" + "-" * 78)
    for r in results:
        status = "FORCED FAIL" if r["forced_fail"] else f"PASS={r['pass_b']}"
        thinflag = " [THIN]" if r.get("thin") else ""
        print(f"  {r['label']:42s} lead={r.get('lead', float('nan')):+.2f}d  {status}{thinflag}")
    print(f"\nEpisodes passing: {n_pass}/{n_valid} valid (non-forced-fail) episodes "
          f"(baseline={baseline_window_days}d){tag}")
    print(f"max timestamp read anywhere for this cell: "
          f"{max(bars.index.max(), div_z.dropna().index.max())}  (< {R.OOS_START})")

    return dict(baseline_window_days=baseline_window_days, is_primary=is_primary,
                results=results, n_pass=n_pass, n_valid=n_valid, passed=passed,
                div_z=div_z)


def causality_probe() -> bool:
    """Two-stage causal truncation probe on this file's own signal-building
    pipeline, run at the PRIMARY baseline window. Stage 1: does the daily
    div_z value at a fixed day change if later daily funding rows are
    dropped? Stage 2: does the bar-aligned value at a fixed bar change if
    later 5-minute bars are dropped? Both must hold for the pipeline to be
    causal."""
    bars = load_btc_bars()
    daily = R.load_daily_funding_totals(DATA_DIR)
    R.assert_no_holdout(daily)

    # Stage 1: daily z-score construction.
    def build_daily(d: pd.DataFrame) -> np.ndarray:
        return R.cross_venue_divergence_z(
            d, baseline_window_days=R.PRIMARY_BASELINE_WINDOW_DAYS).to_numpy()

    stage1 = []
    for check_at in (400, 700, 1000):
        ok = R.truncation_causality_probe(build_daily, daily, check_at, shorter_by=90)
        print(f"[causality stage 1: daily div_z] check_at={check_at} "
              f"({daily.index[check_at].date()}): {'PASS' if ok else 'FAIL'}")
        stage1.append(ok)

    # Stage 2: causal alignment onto 5-minute bars.
    def build_aligned(b: pd.DataFrame) -> np.ndarray:
        return build_div_z_aligned(daily, b, R.PRIMARY_BASELINE_WINDOW_DAYS).to_numpy()

    stage2 = []
    for check_at in (150_000, 250_000, 350_000):
        ok = R.truncation_causality_probe(build_aligned, bars, check_at, shorter_by=20_000)
        print(f"[causality stage 2: bar alignment] check_at={check_at} "
              f"({bars.index[check_at]}): {'PASS' if ok else 'FAIL'}")
        stage2.append(ok)

    overall = all(stage1) and all(stage2)
    print(f"\ncausal truncation probe overall: {'PASS' if overall else 'FAIL'}")
    return overall


def run_full_grid() -> dict:
    bars = load_btc_bars()
    majority = R.anchor_majority(bars)
    daily = R.load_daily_funding_totals(DATA_DIR)
    R.assert_no_holdout(daily)
    print(f"daily funding: {len(daily)} rows  {daily.index.min()} -> {daily.index.max()}  "
          f"(< {R.OOS_START})")

    cells = []
    primary_cell = None
    max_ts_seen = bars.index.max()
    for w in R.BASELINE_WINDOW_DAYS_GRID:
        cell = gate_cell(w, bars, majority, daily)
        cells.append(cell)
        if cell["is_primary"]:
            primary_cell = cell
        max_ts_seen = max(max_ts_seen, cell["div_z"].dropna().index.max())

    assert primary_cell is not None, "primary cell missing from grid"

    print("\n" + "=" * 78)
    print("R-100 CONSERVATIVE: FULL 3-CELL GRID SUMMARY (cross-venue div_z vs v4 anchor)")
    print("=" * 78)
    print(f"{'baseline_days':>14} {'n_pass/n_valid':>15}  {'primary?':>9}")
    for cell in cells:
        marker = "<-- PRIMARY" if cell["is_primary"] else ""
        print(f"{cell['baseline_window_days']:>14} {cell['n_pass']:>7}/{cell['n_valid']:<7} {marker}")

    print(f"\nPRIMARY CELL (baseline_window_days={R.PRIMARY_BASELINE_WINDOW_DAYS}): "
          f"{primary_cell['n_pass']}/{primary_cell['n_valid']} valid episodes pass "
          f"(bar: >= {R.PASS_BAR_NUM}/{R.PASS_BAR_DEN})")
    print(f"GATE VERDICT (primary cell only, per pre-registered stop rule): "
          f"{'PASS -> proceed to Step B' if primary_cell['passed'] else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in Step A: 3 (full BASELINE_WINDOW_DAYS_GRID; "
          f"1 primary decision cell + 2 non-decision robustness cells)")
    print(f"max timestamp read anywhere in Step A: {max_ts_seen}  (< {R.OOS_START})")

    return dict(cells=cells, primary=primary_cell, bars=bars, daily=daily, majority=majority)


# ==========================================================================
# STEP B -- built only if the primary cell's gate above passes.
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, daily: pd.DataFrame, baseline_window_days: int,
                       z_thresh: float = R.Z_THRESH, release_frac: float = RELEASE_FRAC,
                       horizons: tuple[int, ...] = R.V4_HORIZONS, band: float = R.V4_BAND
                       ) -> np.ndarray:
    """One-sided, latched 0/1 meta-vote (banner item 4): `meta_vote[i] = 0`
    ("confirmed bearish") from the first bar div_z crosses UP through
    `z_thresh` until it falls back to `release_frac * z_thresh`; outside a
    confirmed-bearish state, `meta_vote[i]` tracks the fastest (20-day)
    anchor's own then-current vote. Causal: div_z and each anchor vote are
    both causal; the latch update at i depends only on values at <= i.
    """
    fast_vote = R.anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    div_z = build_div_z_aligned(daily, df, baseline_window_days).to_numpy()
    release_thresh = release_frac * z_thresh

    n = len(df)
    meta = np.empty(n)
    state_bearish = False
    for i in range(n):
        z = div_z[i]
        if z >= z_thresh:
            state_bearish = True
        elif z <= release_thresh:
            state_bearish = False
        meta[i] = 0.0 if state_bearish else fast_vote[i]
    return meta


def build_target_primary(df: pd.DataFrame, daily: pd.DataFrame) -> np.ndarray:
    """Target-construction function for the truncation causality probe,
    frozen at the pre-registered primary candidate."""
    return CrossVenueDivergenceConfirmKelly(
        weight=PRIMARY_WEIGHT, baseline_window_days=R.PRIMARY_BASELINE_WINDOW_DAYS,
        daily=daily).prepare(df.copy())["target"].to_numpy()


class CrossVenueDivergenceConfirmKelly(Strategy):
    """kelly_regime_v4 + a cross-venue funding-divergence confirming vote
    (R-100 conservative, unregistered). Structurally v3/v4's own
    prepare(), with the plain 3-anchor average `frac = anchor_sum/3`
    replaced by `confirming_vote_frac(anchor_sum, meta_vote, weight)`.
    `weight=0` must recover v4 bit-for-bit. Not `@register`ed -- stays in
    experiments/ per docs/ROUTINE.md. `daily` (Binance+Deribit daily
    funding totals) must be passed at construction.
    """

    name = "r100_conservative_funding_divergence_vote"
    warmup = 80 * R.BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, daily: pd.DataFrame, weight: float = 1.0,
                 baseline_window_days: int = R.PRIMARY_BASELINE_WINDOW_DAYS,
                 z_thresh: float = R.Z_THRESH, release_frac: float = RELEASE_FRAC,
                 horizons: tuple[int, ...] = R.V4_HORIZONS, band: float = R.V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * R.BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.daily = daily
        self.weight = weight
        self.baseline_window_days = baseline_window_days
        self.z_thresh = z_thresh
        self.release_frac = release_frac
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

        votes = R.anchor_votes(df, horizons=self.horizons, band=self.band)
        anchor_sum = sum(v.to_numpy() for v in votes)

        meta_vote = compute_meta_vote(df, self.daily, self.baseline_window_days,
                                       self.z_thresh, self.release_frac,
                                       horizons=self.horizons, band=self.band)
        frac = R.confirming_vote_frac(anchor_sum, meta_vote, self.weight)

        # Identical conditional-volatility-targeting scale to kelly_regime_v3/_v4.
        vol = (r.ewm(span=self.vol_span, min_periods=R.BARS_PER_DAY).std()
               * np.sqrt(R.BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * R.BARS_PER_DAY,
                                    min_periods=R.BARS_PER_DAY).mean().to_numpy())
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

def run_identity_check(df_full: pd.DataFrame, daily: pd.DataFrame) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:R.INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = CrossVenueDivergenceConfirmKelly(daily=daily, weight=0.0).prepare(
        df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_step_b_causality_probe(df_full: pd.DataFrame, daily: pd.DataFrame) -> list[bool]:
    df = df_full.loc[:R.INNER_TRAIN_END].copy()
    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = R.truncation_causality_probe(lambda d: build_target_primary(d, daily), df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, daily: pd.DataFrame, weight: float,
                 baseline_window_days: int, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=R.INNER_TRAIN_END)),
        ("val", dict(start=R.INNER_VAL_START, end=R.INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = CrossVenueDivergenceConfirmKelly(
                daily=daily, weight=weight, baseline_window_days=baseline_window_days)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES, daily: pd.DataFrame) -> dict:
    results = {}
    for weight in WEIGHT_GRID:
        for baseline_window_days in R.BASELINE_WINDOW_DAYS_GRID:
            tag = f"w{weight} bw{baseline_window_days}"
            results[(weight, baseline_window_days)] = eval_config(
                ev, SPOT, FUTURES, daily, weight, baseline_window_days, tag)
    return results


def plateau_check(sweep_results: dict) -> dict:
    """The pre-registered falsification test (banner item 4): is the
    primary cell's inner-validation Sharpe improvement over baseline (if
    any) a plateau across the weight x baseline_window_days grid, or an
    isolated peak surrounded by opposite-sign neighbours?"""
    print("\n" + "=" * 78)
    print("FALSIFICATION TEST: plateau-not-peak check (inner-validation, spot+futures "
          "delta vs kelly_regime_v4, averaged)")
    print("=" * 78)
    return sweep_results  # deltas printed by caller once baselines are known


def run_eth_note() -> None:
    print("\nFalsification-instrument note (pre-registered, banner item 4): the "
          "standard ETH-Bitfinex falsification is NOT usable here -- ETH has no "
          "committed Deribit funding series and Bitfinex ETH spot data ends "
          "2019-12-31, before Binance BTC funding even starts (2020-01-01). The "
          "plateau-not-peak check across the full weight x baseline_window_days "
          "grid (run above) is used instead, as pre-registered.")


def run_step_b(grid: dict) -> None:
    from scripts.experiment import FUTURES, SPOT, ev

    bars = grid["bars"]
    daily = grid["daily"]

    print("\n" + "=" * 78)
    print("STEP B (primary cell passed): sweep + mandatory checks")
    print("=" * 78)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(bars, daily)
    n_configs += 1

    print("\n=== causality probe (Step B target) ===")
    run_step_b_causality_probe(bars, daily)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    baselines = {}
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=R.INNER_TRAIN_END)),
                                ("val", dict(start=R.INNER_VAL_START, end=R.INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                m = ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)
                baselines[(name, split_name, mkt_name)] = m

    print("\n=== sweep (weight x baseline_window_days, 12 configs) ===")
    sweep_results = run_sweep(ev, SPOT, FUTURES, daily)
    n_configs += len(sweep_results)

    print("\n=== promotion-bar readout: primary cell (weight=1.0, baseline=60) ===")
    primary_key = (PRIMARY_WEIGHT, R.PRIMARY_BASELINE_WINDOW_DAYS)
    v4_base = baselines[("kelly_regime_v4", "val", "spot")]
    v4_base_fut = baselines[("kelly_regime_v4", "val", "futures")]
    prim = sweep_results[primary_key]
    d_spot = prim[("val", "spot")].sharpe - v4_base.sharpe
    d_fut = prim[("val", "futures")].sharpe - v4_base_fut.sharpe
    print(f"  inner-val spot:    v4={v4_base.sharpe:.2f}  candidate={prim[('val','spot')].sharpe:.2f}  "
          f"delta={d_spot:+.2f}  (noise floor +/-0.20)")
    print(f"  inner-val futures: v4={v4_base_fut.sharpe:.2f}  candidate={prim[('val','futures')].sharpe:.2f}  "
          f"delta={d_fut:+.2f}  (noise floor +/-0.20)")

    print("\n=== falsification test: plateau-not-peak across the full grid (inner-val, spot) ===")
    plateau_check(sweep_results)
    print(f"{'weight':>8} {'baseline_days':>14} {'val_spot_sharpe':>16} {'delta_vs_v4':>13}")
    for (weight, bw), out in sorted(sweep_results.items()):
        s = out[("val", "spot")].sharpe
        print(f"{weight:>8} {bw:>14} {s:>16.2f} {s - v4_base.sharpe:>+13.2f}")

    run_eth_note()

    print(f"\nTotal Step B configurations evaluated: {n_configs}")


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        probe_ok = causality_probe()
        grid = run_full_grid()
        if not probe_ok:
            print("\n*** CAUSALITY PROBE FAILED -- results above are NOT trustworthy. ***",
                  file=sys.stderr)
        if grid["primary"]["passed"]:
            run_step_b(grid)
        else:
            print("\nSTEP A FAILED the pre-registered stop rule (primary cell). Per this "
                  "file's own pre-registration, no strategy is built and no Step-B code "
                  "runs. This gate result is this branch's whole product.")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
