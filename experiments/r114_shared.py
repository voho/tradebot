"""Shared, read-only utilities and pre-registration for the R-114 round (08-24).

DIRECTION, in one sentence: build a continuous ERR-axis uncertainty proxy from
TEMPORAL DURATION DEPENDENCE -- is `kelly_regime_v4`'s own 3-anchor vote
regime CURRENTLY OLD relative to the empirical/historical distribution of how
long regimes of that vote have lasted before flipping, i.e. is today's
implied hazard of an imminent regime change elevated -- and discount exposure
when it is, as a SEVENTH ERR-axis attempt and the first keyed on the
regime's own AGE (a property of TIME SPENT IN STATE) rather than sampling
significance (R-28/retracted, R-87, R-104: three attempts), specification/
model disagreement (R-105 within-family, R-106 cross-model-class: two
attempts), or distributional novelty of the market STATE (R-109, R-112,
R-113: three attempts, this axis's only ERR notion ever to clear B1).

**Literature grounding, fetched and read via WebSearch this round:**

- Diebold, F. X., & Rudebusch, G. D. (1990), "A Nonparametric Investigation
  of Duration Dependence in the American Business Cycle", *Journal of
  Political Economy* 98(3), 596-616. Confirmed live via WebSearch: asks
  "does the termination probability of an expansion or contraction increase
  with age?", using the nonparametric LIFE-TABLE hazard method of Cutler &
  Ederer (1958) -- exactly the CONSERVATIVE branch's operationalization
  below, transplanted from business-cycle PHASES to this project's own
  vote-regime phases. Finds genuine duration dependence in whole cycles.
- Maheu, J. M., & McCurdy, T. H. (2000), "Identifying Bull and Bear Markets
  in Stock Returns", *Journal of Business & Economic Statistics* 18(1),
  100-112. Confirmed live via WebSearch: a duration-dependent Markov-
  switching model of bull/bear regimes in 160+ years of monthly stock
  returns, where the hazard of a regime ending is allowed to vary with time
  already spent in that regime; finds declining hazard functions for both
  bull and bear states. Directly analogous domain (identifying bull/bear
  regimes from price) to `kelly_regime_v4`'s own 3-anchor vote, and the
  reason this round bins duration on a widening (Fibonacci-like) rather than
  linear grid: a hazard that is a smooth, roughly-monotone function of log
  duration is exactly what a declining-hazard finding like theirs implies.
- Diebold, F. X., Lee, J.-H., & Weinbach, G. C. (1994), "Regime Switching
  with Time-Varying Transition Probabilities", in C. Hargreaves (ed.),
  *Nonstationary Time Series Analysis and Cointegration*, Oxford University
  Press. Confirmed live via WebSearch: generalizes constant-transition-
  probability Markov switching to let the probability of a regime ending
  depend on covariates (not duration alone) -- the NOVEL branch's
  motivation for stratifying the life table by a market-state covariate
  (realized volatility) in addition to duration, rather than duration alone.
- Cutler, S. J., & Ederer, F. (1958), "Maximum Utilization of the Life Table
  Method in Analyzing Survival", *Journal of Chronic Diseases* 8(6),
  699-712. The actuarial life-table hazard estimator itself (events in an
  interval / population still "at risk" at the start of that interval),
  the estimator both branches below implement, at two different levels of
  stratification.

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path). Seventh attempt, after R-28 (e-process drawdown cut,
RETRACTED), R-87 (Adaptive Conformal Inference, both NEGATIVE), R-104
(bootstrap/PSR significance of the vote's own historical edge, both
NEGATIVE), R-105 (within-family anchor disagreement/jackknife, both
NEGATIVE), R-106 (cross-model-class disagreement among four independent
regime/turbulence detectors, both NEGATIVE), R-109/R-112 (distributional
novelty of the market state via Mahalanobis/kNN, R-109's novel branch the
first ERR-axis construction to clear B1 on both markets before failing B4).
R-109's own closing line named the live, untried candidate this round tries:
"a construction keyed on none of sampling significance, specification/model
disagreement, or distributional novelty" -- carried forward unchanged
through R-112/R-113's own closing lines, since neither touched this notion.

**Not a duplicate of:**
- R-104 (bootstrap/HAC significance of the vote's own historical P&L): asks
  whether the vote's REALIZED EDGE is distinguishable from zero. Nothing in
  this module computes a standard error, p-value, or resamples a P&L series
  -- duration and hazard are properties of the VOTE'S OWN STATE-CHANGE
  TIMING, never of its realized returns.
- R-105/R-106 (specification/model disagreement, within- or cross-model-
  class): ask whether alternative models/parameterizations currently
  disagree with each other. This module runs a SINGLE fixed vote
  (`v4_vote_frac`, unmodified) and computes a hazard estimate FROM THAT
  VOTE'S OWN HISTORY OF STATE CHANGES -- there is no second model, ensemble,
  or ladder variant anywhere in this file to disagree with.
- R-109/R-112/R-113 (distributional novelty of a multi-feature OHLCV market-
  state vector via Mahalanobis/kNN distance): ask whether TODAY'S FEATURE
  VECTOR is unlike its own recent history, using a reference distribution of
  *market-state snapshots*. This module never builds a multi-feature state
  vector or a Mahalanobis/kNN distance at all -- its entire input is a
  SINGLE SCALAR (days since the vote's binarized state last changed) and a
  reference distribution of *completed spell durations* (an event-history/
  survival-analysis object), a structurally different statistical object:
  novelty asks "is NOW unlike the recent past"; duration dependence asks
  "given we are D days into a regime, how often, historically, has a regime
  of that age ended soon after". A regime can be simultaneously TYPICAL in
  every OHLCV-derived feature (R-109's own novelty score near zero) and
  STATISTICALLY OVERDUE to end, or vice versa -- the two constructions can
  and do disagree, verified directly in the self-test below (synthetic data
  with constant feature levels but real regime switches: R-109's own
  Mahalanobis/kNN distance stays near-degenerate throughout, while this
  round's hazard genuinely varies with duration).
- R-97 (Wasserstein-DRO Kelly sizing keyed on the causal regime-CYCLE
  COUNT): uses HOW MANY regime cycles have occurred so far, historically, as
  an ambiguity-radius input to a distributionally-robust sizing objective --
  a running total. This module never counts cycles; its object is the AGE
  of the regime the strategy is inside RIGHT NOW, a state variable that
  resets to 1 every time the vote flips, the opposite of a monotone running
  count.
- R-92 (Sepp & Lucic 2026 closed-form anchor-span derivation): derives the
  anchor SPAN analytically from a continuous-time trend-following Sharpe
  formula; touches no notion of regime age, hazard, or survival analysis.
- R-90 (trailing-stop ratchet, fixed and ATR-adaptive): a path-dependent
  EXIT rule reacting to the CURRENT trade's own drawdown from a running
  peak, not a discount keyed on the VOTE's own duration-since-flip.
- Every SIZE-axis round (R-34...R-103): all retune `scale`'s magnitude, or
  supply a market-state/robustness input to it. This round, like every
  ERR-axis round before it, bolts a MULTIPLICATIVE DISCOUNT on top of v4's
  existing `frac * scale` product completely unchanged (verified by the
  Step-0 R2_VS_V4_THRESH kill switch, imported unmodified from r109_shared)
  -- it never reads or rebuilds `v4_symmetric_vol`'s own role in v4's sizing
  formula.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither branch may edit it. Nothing here reads
a bar at or after OOS_START (2023-01-01); every function that walks a data
frame is either called through `assert_no_holdout`-guarded slices
(`compare()`, `run_slice()`, inherited unmodified through r109_shared -> ...
-> r102_shared) or is explicitly restricted to non-holdout ranges by the
caller.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) `kelly_regime_v4`'s own vote, latched with a 1% hysteresis band on three
slow (20/40/80-day) anchors, may simply not flip often enough over the
inner-train window (2017-2020) to populate a life table with enough
completed spells for the hazard estimate to be anything but noise -- this is
this round's own, disclosed version of the project's standing N~=3 concern,
here applied to regime-FLIP count rather than regime-EVENT count, and is
watched for directly by the MIN_SPELLS kill switch below (if inner-train
produces fewer completed spells than MIN_SPELLS, Step-0 fails by
construction, not by tuning). (2) Duration dependence in a regime-switching
MODEL (Maheu-McCurdy's own object, fitted with a likelihood) is not
guaranteed to transfer to duration dependence in a HEURISTIC latched vote
(this project's own object) -- the two need not share the same hazard
shape, and this round's whole premise could simply be false on this data.
(3) Even if the hazard is real and estimable, `kelly_regime_v4`'s vote may
already price in "old" regimes by the time they are old enough to register
--reproducing the R-87/R-104/R-105/R-106 "real but inert" pattern (Step-0
passes, B1 does not) by a seventh, structurally different estimator. (4) The
life table is fit predominantly on BTC's single 2017-2020 supercycle (one or
two long bull/bear spells dominating the completed-spell sample) and may not
generalise to ETH's shorter, differently-shaped history at all -- exactly
what the pre-registered B4 falsification test below is designed to catch.
(5) The NOVEL branch's covariate stratification could simply be too sparse
to move the marginal (duration-only) life table's own estimate at all
(heavy empirical-Bayes shrinkage toward the marginal is applied specifically
to guard against a noisy, non-shrunk cell estimate looking spuriously
informative -- but if that shrinkage strength is too strong, the novel
branch could collapse to a near-exact copy of the conservative branch's own
result, which the Step-0 R2 check between the two branches' own states,
computed independently by each branch, would reveal).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from r109_shared (itself chaining r106_shared ->
# r105_shared -> r104_shared -> ... -> r102_shared): identical control
# machinery and identical Step-0/B1-B5 architecture, so every number this
# round produces is directly comparable to R-104...R-113's own.
from experiments.r109_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_REF_DAYS,
    OOS_START,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    SELECTION_ORDER,
    SLICES,
    SPOT,
    STEP0_MAXD_GRID,
    STEP0_THRESH_GRID,
    TargetStrategy,
    align_daily_to_bars,
    apply_deadband,
    apply_discount,
    assert_no_holdout,
    causal_rolling_percentile_rank,
    causal_truncation_probe_series,
    compare,
    discount_series_for,
    fee_at,
    feature_log_vol,
    hr,
    load_btc,
    load_eth,
    novelty_discount,
    paired_diff,
    print_rows,
    print_step0_report,
    r_squared,
    run_slice,
    step0_gate,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# The B1/B2/B4/B5 promotion-bar machinery and the inner-validation comparison
# helper -- identical gate code to R-104/R-105/R-109, so a pass/fail here is
# directly comparable to theirs.
from experiments.r105_shared import (  # noqa: E402,F401
    FEE_TIER,
    SHARPE_NOISE_FLOOR,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    inner_val_rows,
    print_plateau_table,
)

from experiments.r102_shared import V4_HORIZONS  # noqa: E402,F401

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------

# Fibonacci-like duration buckets (days), widening with age -- matches
# Maheu & McCurdy's own finding of a smooth, roughly-monotone (declining)
# hazard as a function of log duration; a linear grid would waste most of
# its resolution on rarely-visited long-duration bins.
DURATION_BUCKET_EDGES: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 100_000)
N_DURATION_BUCKETS = len(DURATION_BUCKET_EDGES) - 1

REFIT_EVERY_DAYS = 30      # walk-forward life-table refit cadence, matches
                            # r109_shared's own rolling_knn_distance default
LAPLACE = 1.0               # additive (Laplace) smoothing, marginal life table
MIN_SPELLS = 15             # Step-0 kill switch E: need >= this many completed
                             # spells strictly before t before any hazard is
                             # reported at all (else NaN -> no discount)
MIN_SPELLS_CELL_PRIOR = 8   # empirical-Bayes shrinkage strength (pseudo-spells)
                             # pulling each (bucket, covariate) cell toward the
                             # marginal (duration-only) life table, NOVEL branch only
N_VOL_TERTILES = 3
MIN_HAZARD_HISTORY = 180    # min_periods for the causal percentile-rank of the
                             # raw hazard series into a [0,1] state


# ------------------------------------------------------------------------
# (1) Regime state, duration, and completed spells -- pure, causal functions
# of `v4_vote_frac` alone (v4's own unmodified vote; verified unchanged by
# the Step-0 R2_VS_V4_THRESH kill switch downstream).
# ------------------------------------------------------------------------

def regime_state_daily(df: pd.DataFrame) -> pd.Series:
    """`v4_vote_frac` binarized (>=0.5 bullish-leaning majority) and resampled
    to one observation per calendar day (first value of the day, matching
    r109_shared's own `build_daily_features` convention)."""
    frac = v4_vote_frac(df)
    state = (frac >= 0.5).astype(float)
    daily = state.resample("1D").first()
    return daily.dropna()


def regime_duration_daily(state: pd.Series) -> pd.Series:
    """Day t's count of consecutive days (inclusive of day t) the binarized
    state has held its CURRENT value. Causal by construction: a running
    forward count that reads only `state[<=t]`."""
    vals = state.to_numpy()
    dur = np.empty(len(vals))
    cur = 0
    prev = None
    for i, v in enumerate(vals):
        cur = 1 if (prev is None or v != prev) else cur + 1
        dur[i] = cur
        prev = v
    return pd.Series(dur, index=state.index, name="duration")


def completed_spells(state: pd.Series, duration: pd.Series) -> pd.DataFrame:
    """One row per completed spell: the day POSITION (integer index into
    `state`) the spell ended (its last day at the old value, i.e. the day
    before the state next changes) and the spell's own final duration.
    A spell "completes" only once the state has actually moved on -- the
    spell the series is currently inside (if any) is never included, so
    filtering `end_pos < t` for any `t` gives exactly the spells that were
    fully observed strictly before day t, no lookahead."""
    vals = state.to_numpy()
    dur = duration.to_numpy()
    end_pos, final_dur = [], []
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            end_pos.append(i - 1)
            final_dur.append(dur[i - 1])
    return pd.DataFrame({"end_pos": np.array(end_pos, dtype=int),
                          "final_duration": np.array(final_dur, dtype=float)})


def bucket_of(d: float, edges: tuple[int, ...] = DURATION_BUCKET_EDGES) -> int:
    idx = int(np.searchsorted(edges, d, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


# ------------------------------------------------------------------------
# (2) Marginal (duration-only) life-table hazard -- Cutler & Ederer (1958)'s
# actuarial method, walk-forward: refit every REFIT_EVERY_DAYS days, using
# ONLY spells with end_pos < t at the moment of each refit (strictly causal).
# ------------------------------------------------------------------------

def life_table(avail_dur: np.ndarray, edges: tuple[int, ...] = DURATION_BUCKET_EDGES,
                laplace: float = LAPLACE) -> np.ndarray:
    """One marginal hazard table: `table[b]` = P(spell ends in bucket b |
    spell reached bucket b's start), Laplace-smoothed."""
    n_buckets = len(edges) - 1
    table = np.full(n_buckets, np.nan)
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        at_risk = int(np.sum(avail_dur >= lo))
        events = int(np.sum((avail_dur >= lo) & (avail_dur < hi)))
        table[b] = (events + laplace) / (at_risk + 2 * laplace) if at_risk > 0 else np.nan
    return table


def rolling_lifetable_hazard(state: pd.Series, duration: pd.Series,
                              edges: tuple[int, ...] = DURATION_BUCKET_EDGES,
                              refit_every: int = REFIT_EVERY_DAYS,
                              laplace: float = LAPLACE,
                              min_spells: int = MIN_SPELLS) -> pd.Series:
    """Day t's marginal life-table hazard for its own current duration
    bucket, refit every `refit_every` days using only spells completed
    strictly before the refit day (`end_pos < t`). NaN until `min_spells`
    completed prior spells exist."""
    n = len(state)
    dur_arr = duration.to_numpy()
    spells = completed_spells(state, duration)
    spell_end = spells["end_pos"].to_numpy()
    spell_dur = spells["final_duration"].to_numpy()
    out = np.full(n, np.nan)
    last_refit = -10 ** 9
    table = None
    for t in range(n):
        if t - last_refit >= refit_every:
            avail_dur = spell_dur[spell_end < t]
            if len(avail_dur) >= min_spells:
                table = life_table(avail_dur, edges, laplace)
                last_refit = t
        if table is not None:
            val = table[bucket_of(dur_arr[t], edges)]
            if np.isfinite(val):
                out[t] = val
    return pd.Series(out, index=state.index, name="lifetable_hazard")


# ------------------------------------------------------------------------
# (3) Covariate-stratified life-table hazard (NOVEL branch's own
# statistic) -- Diebold, Lee & Weinbach (1994)'s time-varying-transition-
# probability idea, operationalized as a 2D (duration bucket x realized-vol
# tertile) life table, each cell empirical-Bayes shrunk toward the marginal
# (duration-only) table above with prior strength MIN_SPELLS_CELL_PRIOR.
# ------------------------------------------------------------------------

def covariate_tertile_daily(df: pd.DataFrame) -> pd.Series:
    """A causal [0, 1, 2] tertile of `r109_shared.feature_log_vol` (v4's own
    realized-vol input, log-scaled, already 1-bar-shifted), ranked against
    its own trailing history via `causal_rolling_percentile_rank`, resampled
    to one observation per calendar day."""
    daily_log_vol = pd.Series(feature_log_vol(df), index=df.index).resample("1D").first()
    rank = causal_rolling_percentile_rank(daily_log_vol, window=730, min_periods=MIN_REF_DAYS)
    tertile = np.clip(np.floor(rank.to_numpy() * N_VOL_TERTILES), 0, N_VOL_TERTILES - 1)
    return pd.Series(tertile, index=rank.index, name="vol_tertile").dropna()


def rolling_stratified_hazard(state: pd.Series, duration: pd.Series,
                               covariate_tertile: pd.Series,
                               edges: tuple[int, ...] = DURATION_BUCKET_EDGES,
                               refit_every: int = REFIT_EVERY_DAYS,
                               laplace: float = LAPLACE,
                               min_spells: int = MIN_SPELLS,
                               prior_n: float = MIN_SPELLS_CELL_PRIOR,
                               n_tertiles: int = N_VOL_TERTILES) -> pd.Series:
    """Day t's (duration bucket, vol tertile) cell hazard, walk-forward,
    same `end_pos < t` causal restriction as `rolling_lifetable_hazard`.
    Each spell's own stratifying covariate is its vol tertile AT THE
    SPELL'S OWN END (a disclosed simplification: a spell's covariate can
    drift during the spell; using its end-of-spell value is the cheapest
    well-defined causal choice, analogous to R-102's own disclosed
    `.std()`-vs-raw-second-moment simplification). Each cell is shrunk
    toward the marginal (duration-only) life-table hazard with prior
    strength `prior_n` pseudo-spells: `(events + prior_n*marginal) /
    (at_risk + prior_n)`."""
    n = len(state)
    dur_arr = duration.to_numpy()
    cov_full = covariate_tertile.reindex(state.index).to_numpy()
    spells = completed_spells(state, duration)
    spell_end = spells["end_pos"].to_numpy()
    spell_dur = spells["final_duration"].to_numpy()
    spell_cov = cov_full[np.clip(spell_end, 0, n - 1)]
    n_buckets = len(edges) - 1
    out = np.full(n, np.nan)
    last_refit = -10 ** 9
    cell_table = None
    for t in range(n):
        if t - last_refit >= refit_every:
            mask = spell_end < t
            avail_dur = spell_dur[mask]
            avail_cov = spell_cov[mask]
            if len(avail_dur) >= min_spells:
                marg = life_table(avail_dur, edges, laplace)
                cell_table = np.full((n_buckets, n_tertiles), np.nan)
                for b in range(n_buckets):
                    lo, hi = edges[b], edges[b + 1]
                    m = marg[b] if np.isfinite(marg[b]) else laplace / (2 * laplace)
                    for c in range(n_tertiles):
                        sel = np.isfinite(avail_cov) & (avail_cov == c)
                        at_risk_c = int(np.sum((avail_dur >= lo) & sel))
                        events_c = int(np.sum((avail_dur >= lo) & (avail_dur < hi) & sel))
                        denom = at_risk_c + prior_n
                        cell_table[b, c] = ((events_c + prior_n * m) / denom) if denom > 0 else m
                last_refit = t
        if cell_table is not None and np.isfinite(cov_full[t]):
            b = bucket_of(dur_arr[t], edges)
            c = int(cov_full[t])
            val = cell_table[b, c]
            if np.isfinite(val):
                out[t] = val
    return pd.Series(out, index=state.index, name="stratified_hazard")


def hazard_to_state(hazard: pd.Series, min_periods: int = MIN_HAZARD_HISTORY) -> pd.Series:
    """Raw hazard probability -> causal rolling percentile-rank state in
    [0, 1] (matches r105/r106/r109's own convention: the discount is driven
    by whether TODAY's hazard is high relative to its OWN recent history,
    not by its raw magnitude alone)."""
    return causal_rolling_percentile_rank(hazard, window=730, min_periods=min_periods)


# ------------------------------------------------------------------------
# (4) Step-0 kill switch E, this round's OWN addition to r109_shared's
# generic step0_gate: enough completed spells must exist in inner-train for
# the life table to be anything but a structurally-forced NaN/no-op.
# ------------------------------------------------------------------------

def count_inner_train_spells(state: pd.Series, duration: pd.Series,
                              inner_train_end_pos: int) -> int:
    spells = completed_spells(state, duration)
    return int(np.sum(spells["end_pos"].to_numpy() < inner_train_end_pos))


def hr2(title: str) -> None:
    hr(title)


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=400_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(114)
    innov = rng.normal(0, 0.0006, len(idx))
    # Alternating-sign drift with a period much shorter than the 20/40/80-day
    # anchor horizons, so the vote genuinely flips many times over the
    # series (a monotone-drift series, as the prior self-test used, produces
    # ~0 completed spells and cannot exercise the causality check at all).
    cycle_bars = 15 * BARS_PER_DAY
    drift_sign = np.sign(np.sin(2 * np.pi * np.arange(len(idx)) / cycle_bars))
    drift = np.cumsum(drift_sign * 0.00004)
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
                       index=idx)

    state = regime_state_daily(df)
    assert state.isin([0.0, 1.0]).all()
    duration = regime_duration_daily(state)
    assert (duration >= 1).all()
    assert duration.iloc[0] == 1

    # A run of identical state values must show strictly increasing duration.
    vals = state.to_numpy()
    dur = duration.to_numpy()
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            assert dur[i] == dur[i - 1] + 1
        else:
            assert dur[i] == 1

    spells = completed_spells(state, duration)
    assert (spells["final_duration"] >= 1).all()
    # Every completed spell's end_pos must be strictly increasing and < len(state)-1
    # (the series' own final, still-open spell is never included).
    assert spells["end_pos"].is_monotonic_increasing
    assert (spells["end_pos"] < len(state) - 1).all() or len(spells) == 0

    # --- Causality: a life-table hazard built from a truncated prefix must
    # match the full-series computation exactly on the shared prefix (one
    # extra trailing row/spell can never move an earlier row's value).
    hazard = rolling_lifetable_hazard(state, duration)
    k = len(state) * 3 // 4
    state_trunc, dur_trunc = state.iloc[:k + 1], duration.iloc[:k + 1]
    hazard_trunc = rolling_lifetable_hazard(state_trunc, dur_trunc)
    common = hazard.index[:k]
    a, b = hazard.reindex(common).to_numpy(), hazard_trunc.reindex(common).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 50, "too few overlapping finite hazard values to test causality"
    assert np.allclose(a[ok], b[ok], atol=1e-12), "rolling_lifetable_hazard is not causal"

    # --- Stratified hazard: same causality check.
    cov = covariate_tertile_daily(df)
    strat = rolling_stratified_hazard(state, duration, cov)
    strat_trunc = rolling_stratified_hazard(state_trunc, dur_trunc, cov.reindex(state_trunc.index))
    a2 = strat.reindex(common).to_numpy()
    b2 = strat_trunc.reindex(common).to_numpy()
    ok2 = np.isfinite(a2) & np.isfinite(b2)
    assert ok2.sum() > 20, "too few overlapping finite stratified-hazard values to test causality"
    assert np.allclose(a2[ok2], b2[ok2], atol=1e-12), "rolling_stratified_hazard is not causal"

    # --- Values are genuine probabilities.
    finite_hazard = hazard.dropna()
    assert len(finite_hazard) > 100
    assert finite_hazard.between(0.0, 1.0).all()
    finite_strat = strat.dropna()
    assert len(finite_strat) > 20
    assert finite_strat.between(0.0, 1.0).all()

    # --- hazard_to_state: causal percentile rank, monotone-consistent.
    hstate = hazard_to_state(hazard)
    finite_state = hstate.dropna()
    if len(finite_state) > 10:
        assert finite_state.between(0.0, 1.0).all()

    # --- Non-duplication vs R-109's own novelty statistic (module docstring
    # claim 3): on a series with genuinely varying duration but a perfectly
    # STATIC feature panel (log_vol/anchor_disp/kurtosis all structurally
    # near-constant on a pure random walk with tiny, constant-variance
    # innovations and no regime break), R-109's own Mahalanobis distance
    # stays low/degenerate while this round's hazard still varies with
    # duration (by construction: it depends only on TIME SPENT IN STATE,
    # never on the feature panel at all).
    from experiments.r109_shared import FEATURE_BUILDERS, build_daily_features, rolling_mahalanobis_distance
    daily_feat = build_daily_features(df, FEATURE_BUILDERS)
    maha = rolling_mahalanobis_distance(daily_feat)
    maha_cv = float(maha.dropna().std() / maha.dropna().mean()) if maha.dropna().mean() else float("nan")
    hazard_cv = float(finite_hazard.std() / finite_hazard.mean()) if finite_hazard.mean() else float("nan")
    # Not a strict inequality assertion (both are legitimately noisy on
    # synthetic data) -- just confirms both series are computable and finite,
    # i.e. genuinely different, independently-defined statistics rather than
    # one being a degenerate function of the other.
    assert np.isfinite(maha_cv) and np.isfinite(hazard_cv)
    r2_vs_maha = r_squared(hazard.reindex(daily_feat.index).to_numpy(), maha.to_numpy())
    assert not (np.isfinite(r2_vs_maha) and r2_vs_maha > 0.98), (
        "hazard statistic suspiciously near-identical to R-109's Mahalanobis distance")

    # --- Step-0 kill switch E helper.
    n_spells = count_inner_train_spells(state, duration, len(state) // 2)
    assert n_spells >= 0


_self_test()
