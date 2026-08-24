"""Shared, read-only utilities and pre-registration for the R-106 round (08-24).

DIRECTION, in one sentence: build a continuous ERR-axis uncertainty proxy
from CROSS-MODEL-CLASS DISAGREEMENT among this project's own four already-
built, structurally-independent causal regime/turbulence estimators --
BOCPD (R-82), a Kalman local-linear-trend filter (R-83), critical-slowing-
down fluctuation statistics (R-85), and a self-exciting Hawkes point
process (R-96) -- and use that disagreement to discount `kelly_regime_v4`'s
exposure, as a fifth attempt on the ERR axis and the first to measure
disagreement ACROSS four theoretically disjoint model classes rather than
within one model family or one fixed estimator's own sampling uncertainty.

**Literature grounding:**

- Zarnowitz, V., & Lambros, L. A. (1987), "Consensus and Uncertainty in
  Economic Prediction", *Journal of Political Economy* 95(3), 591-621.
  Establishes that cross-sectional DISAGREEMENT among independent
  forecasters is empirically related to, and a usable proxy for, genuine
  forecast uncertainty -- distinct from any one forecaster's own stated
  confidence.
- Bomberger, W. A. (1996), "Disagreement as a Measure of Uncertainty",
  *Journal of Money, Credit and Banking* 28(3), 381-392. The specific
  operationalization this round borrows: disagreement among a small panel
  of independent forecasts/models, measured as the CROSS-SECTIONAL STANDARD
  DEVIATION of their point estimates at each date, is itself a valid,
  literature-standard uncertainty proxy. Both papers study human forecaster
  panels (survey point-forecasts of inflation/output); this round is, as
  far as this project's ledger records, the first application of that
  disagreement-as-uncertainty idea to an ENSEMBLE OF REGIME/TURBULENCE
  DETECTORS (BOCPD, Kalman, CSD, Hawkes) rather than to survey point-
  forecasts -- the panel members here are pre-existing causal model outputs,
  not elicited human forecasts, so the transplant is disclosed explicitly
  rather than treated as a direct replication.

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path). Prior ERR attempts: R-28 (e-process drawdown cut, RETRACTED),
R-87 (Adaptive Conformal Inference on the vote's own confidence, NEGATIVE),
R-104 (bootstrap/HAC significance of the vote's own realized edge, mixed/
NEGATIVE), R-105 (disagreement among alternative ANCHOR-LADDER
specifications and leave-one-anchor-out jackknife of the vote's OWN THREE
COMPONENTS -- disagreement WITHIN one model family, the 20/40/80 moving-
average vote, at a single point in time). This round is the fifth ERR
attempt and the first whose disagreement panel is built from four
STRUCTURALLY DIFFERENT mathematical objects (a Bayesian run-length
posterior, a linear-Gaussian state-space filter, a dynamical-systems
fluctuation/trend test, and a conditional point-process intensity) that
share no estimation machinery with each other or with the vote's own
20/40/80 moving-average construction at all.

**Not a duplicate of:**
- R-104 (bootstrap/HAC significance of the vote's own FIXED historical
  edge): a SAMPLING-uncertainty construction over a P&L time series (is the
  realized return distinguishable from zero). Nothing in this module ever
  reads or resamples a P&L series, computes a standard error, or computes a
  p-value; the Step-0 gate below never touches `kelly_regime_v4`'s exposure
  or returns at all -- only the four detectors' own native outputs.
- R-105 (disagreement among alternative anchor-ladder parameterizations of
  the SAME 20/40/80 moving-average mechanism, and a leave-one-anchor-out
  jackknife of that mechanism's own three components): disagreement WITHIN
  one model family (every alternative is still a moving-average-crossover
  vote, differing only in which horizons are used). This round's four
  panel members share no mathematical machinery with the moving-average
  vote or with each other -- a Bayesian changepoint posterior, a Kalman
  filter, a fluctuation-statistics trend test, and a point-process
  intensity are four different formal objects, not four parameterizations
  of one object. R-105's own module never reads BOCPD, Kalman, CSD, or
  Hawkes outputs; this module never reads alternative anchor ladders.
- R-82/R-83/R-85/R-96 (BOCPD / Kalman / CSD / Hawkes, each on its own):
  every one of these four prior rounds asked "does THIS ONE detector alarm
  before v4's own fixed-anchor gate does, around the six dated stress
  episodes" (a detection-LAG race against the vote) -- all four closed with
  the SAME negative Step-A finding (no estimator computed from this
  project's own committed BTC price history leads the vote's own reaction
  around 2020-2022's sudden shocks; at best matches it around 2018's slow
  build-up). This round asks a different question entirely: not "does any
  one of these four models alarm early", but "does the DISPERSION across
  all four simultaneously carry information a single model's own alarm
  level does not" -- a second-order statistic over the panel, never
  computed or asked by any of R-82/83/85/96's own modules, each of which
  only ever inspects its own single detector's crossing time against one
  onset date.
- R-86 (transfer entropy) / R-98 (bipower-variation jump alarm) / R-99
  (BV jump forward-loss) / R-84 (vote-latch modulation): further single-
  estimator regime/timing rounds on the same detection-lag question R-82/
  83/85/96 asked; none is included in this round's four-member panel (the
  panel is fixed a priori to the four DIRECTION-agent-specified detectors
  named above) and none is duplicated by this module, which never computes
  a detection lag against a stress-episode onset for any individual
  detector at all -- only the four-way normalized-state correlation matrix
  and cross-sectional dispersion statistic.
- Every SIZE-axis round (R-34...R-103): all retune `scale`'s magnitude or
  supply an exogenous/endogenous market-state variable to it; none builds
  a cross-model disagreement statistic over BOCPD/Kalman/CSD/Hawkes.

This module is READ-ONLY infrastructure for the conservative/novel branches
that come after. It contains ONLY: (1) thin per-model "raw alarm scalar"
extractors on top of the four already-committed detector modules
(r82/r83/r85/r96_shared, imported, never reimplemented); (2) ONE shared
causal normalization primitive mapping each raw alarm scalar onto a
comparable [0,1] state; (3) the Step-0 falsification-gate computations
(pairwise correlation, cross-sectional dispersion); (4) a `_self_test()`.
No conservative- or novel-branch-specific exposure-discounting logic
appears anywhere in this file.

**Normalization -- a disclosed, explicit deviation from a naive "four
signed states" framing.** Inspecting the four detectors' own native output
semantics (r82_shared.bocpd_daily's docstring; r83_novel_kalman_shared.
kalman_llt_filter's docstring; r85_shared.csd_trend_zscore/
csd_daily_causal_signals; r96_shared.hawkes_intensity_zscore) shows THREE
of the four are natively TURBULENCE/ALARM levels with no directional sign
(BOCPD's `p_recent_cp`, a changepoint-recency posterior probability in
[0,1]; CSD's `csd_var_z`/`csd_autocorr_z`, trend z-scores of variance/
autocorrelation that rise -- regardless of price direction -- as a
bifurcation approaches; Hawkes's intensity z-score, which rises with
self-exciting jump clustering regardless of jump sign), while only the
Kalman filter's `slope` is a naturally SIGNED directional state (the
local-linear trend's own estimated drift). Forcing all four onto a common
signed convention would require inventing an artificial sign for BOCPD/
CSD/Hawkes that none of their own theory supplies (a changepoint, a rising
autocorrelation trend, or a jump cluster is not "bullish" or "bearish" --
it is "turbulent", full stop). This module therefore normalizes all four
onto a common UNSIGNED alarm/uncertainty scale instead:

1. **BOCPD**: raw alarm = `bocpd_p_recent_cp` directly (already an alarm
   probability in [0,1]; nothing to unsign).
2. **Kalman**: raw alarm = `abs(kalman_slope)` -- the MAGNITUDE of the
   filtered trend estimate, converting the one naturally signed member into
   an unsigned "how strongly is a trend currently asserting itself"
   turbulence proxy, on the same unsigned footing as the other three
   (an unusually large |slope|, whichever direction, is the Kalman
   analogue of "this regime looks unlike calm/range-bound conditions").
3. **CSD**: raw alarm = `mean(csd_var_z, csd_autocorr_z)` -- CSD theory
   (Dakos et al. 2012, cited in r85_shared.py) predicts BOTH indicators
   trend upward together as a critical transition approaches, so averaging
   the two rising-trend z-scores gives one combined alarm level rather than
   arbitrarily picking one of the two and discarding the other.
4. **Hawkes**: raw alarm = `hawkes_intensity_zscore` directly (already an
   unsigned alarm-style z-score of clustering intensity against its own
   trailing baseline).

Each raw alarm scalar is then passed through ONE shared normalization
primitive, `causal_rolling_percentile_rank` (below): a causal ROLLING
(not expanding) percentile rank against the trailing `BASELINE_WINDOW_DAYS
= 730` calendar days of that SAME model's own history (identical window to
r85/r86/r96_shared's own `BASELINE_WINDOW_DAYS` baseline convention, reused
here for comparability rather than introducing a new nuisance parameter).
Rolling rather than expanding is a deliberate choice: an expanding
percentile rank compares today's alarm level against the ENTIRE historical
record including years-old, decreasingly relevant regimes, and would grow
steadily less sensitive to recent conditions over 2017-2020; a rolling
730-day window keeps every model's [0,1] state a comparable, roughly
stationary "how unusual is this model's current reading relative to its
own recent history" quantity, matching how every other rolling-baseline
z-score in this project's own detector modules (R-85/86/96) is already
defined. Percentile-ranking ALL FOUR raw alarms with the SAME transform --
even BOCPD's, which is already bounded in [0,1] natively -- is deliberate:
a monotonic percentile-rank transform preserves each model's own alarm
ordering while forcing all four onto a genuinely comparable (uniform
marginal, not just same-bounded) distribution, so the pairwise-correlation
and dispersion statistics below are not distorted by one model's raw
output happening to sit in a different part of its own range than another.

**Step-0 falsification gate (pre-registered, before any Sharpe/PnL
number, before any conservative/novel branch exists):**
(a) The four normalized [0,1] states, restricted to
`INNER_TRAIN_START..INNER_TRAIN_END` for the correlation check in (b).
(b) `mean_pairwise_abs_corr`: mean of the six pairwise |Pearson
correlation| values among the four normalized states, over inner-train.
KILL if `>= 0.5` -- cite R-85's own AND-gate-collapse finding (r85_shared/
r85_conservative's own documented result that near-duplicate indicators
collapse an ensemble's information content instead of adding to it): if
the four detectors are mostly re-deriving the same underlying price
feature, their "disagreement" would just be shared noise around one common
signal, not a genuine multi-model uncertainty proxy.
(c) Disagreement statistic: `cross_sectional_std`, the STANDARD DEVIATION
across the four normalized states at each bar (chosen over range/IQR
because it is literally Bomberger (1996)'s own definition of cross-panel
disagreement, uses all four members' values rather than only the extreme
two, and is the standard spread-skill statistic this literature reports).
Checked for genuine non-degenerate dispersion at this project's six
standard historical stress episodes (identical dates and `episode_window`
convention to r82/r85/r96_shared.py's own `STRESS_EPISODES` table -- copied
verbatim below, not re-derived): compute the mean `cross_sectional_std`
within each of the six 60-day episode windows, then the coefficient of
variation (CoV = std/mean) ACROSS those six episode means, plus the CoV of
`cross_sectional_std` over the whole inner-train span. Episode windows
after `INNER_TRAIN_END` (2021-11 top, 2022-05 Terra/Luna, 2022-11 FTX all
fall in `INNER_VAL_START..INNER_VAL_END`) are read from `load_btc()`'s full
pre-holdout frame -- data through `OOS_START` (2023-01-01, exclusive) is
never holdout, so this is within this project's standing walk-forward
convention (identical to how r82/85/96_shared's own Step-A gates read
episode windows that extend past `INNER_TRAIN_END`); the Step-0
CORRELATION check in (b) alone is restricted to inner-train only, per this
round's own pre-registration.
KILL if `mean_pairwise_abs_corr >= 0.5`, OR the disagreement statistic is
degenerate: CoV across the six episode-window means `< 5%`, OR CoV over
the whole inner-train span `< 5%` (both read as "the four models never
meaningfully diverge from each other, anywhere, regardless of market
state" -- a constant disagreement statistic could not possibly discount
exposure selectively).

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) all four detectors are, at bottom, functions of the same underlying
BTC daily-return series filtered at broadly similar 1-3 month horizons
(BOCPD's hazard-implied expected run length, Kalman's slope smoothness,
CSD's 15-day sub-window, Hawkes's 3-14 day half-life grid all sit in
roughly the same 1-12 week band v4's own 20-80 day anchors also occupy) --
if that shared horizon band dominates, the four normalized alarms could
move together tightly (mean |rho| >= 0.5) even though they are
mathematically unrelated constructions, the same "near-duplicate collapse"
R-85's AND-gate finding already flagged for a different pair of indicators.
(2) Even if the four are weakly correlated, their DISAGREEMENT might simply
track one dominant driver (e.g. Hawkes's own jump-clustering z-score
swinging far more than the other three ever do) -- a disagreement
statistic that is really just "one noisy member's own variance" rather
than genuine multi-model uncertainty would still pass the CoV check
mechanically while carrying none of the ensemble-uncertainty interpretation
Bomberger's own literature claims for it; this Step-0 gate as specified
cannot distinguish that failure mode from genuine dispersion, and is
flagged here as a known limitation for the branch-design agents to inherit
and address in a later step, not resolved in this module.
(3) R-82/83/85/96's own shared finding -- every detector computed from this
project's own committed BTC price history lags 2020-2022's sudden shocks
and at best matches the 2018 slow build-up -- makes it plausible that ALL
FOUR normalized alarms rise together (not apart) exactly at the moments
this project's whole `kelly_regime` family most needs protection, which
would produce LOW disagreement (not high) at the riskiest episodes even if
the gate above passes on its own narrow terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from r105_shared (itself chaining r104_shared ->
# r103_shared -> r102_shared): identical control machinery, so every
# number this round produces is directly comparable to R-101...R-105's own.
from experiments.r105_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    compare,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# The four already-built causal detector modules this round's panel is
# drawn from -- imported and used exactly as committed, never reimplemented.
from experiments.r82_shared import (  # noqa: E402,F401
    bocpd_daily_causal_signals,
    episode_window,
    STRESS_EPISODES,
)
from experiments.r83_novel_kalman_shared import (  # noqa: E402,F401
    kalman_daily_causal_signals,
)
from experiments.r85_shared import (  # noqa: E402,F401
    csd_daily_causal_signals,
)
from experiments.r96_shared import (  # noqa: E402,F401
    intraday_relative_jump,
    hawkes_intensity_daily,
    hawkes_intensity_zscore,
)

# STRESS_EPISODES is imported from r82_shared.py above (copied verbatim,
# not re-derived, there and here) -- sanity-check it matches the table
# quoted in this module's own docstring and in r85/r96_shared.py.
assert [onset for _, onset in STRESS_EPISODES] == [
    "2018-01-17", "2018-12-15", "2020-03-12",
    "2021-11-10", "2022-05-09", "2022-11-08",
], STRESS_EPISODES

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before any real-data number was
# computed.
# ------------------------------------------------------------------------
BASELINE_WINDOW_DAYS = 730   # rolling percentile-rank window; identical to
                              # r85/r86/r96_shared's own BASELINE_WINDOW_DAYS
MIN_PERIODS_DAYS = 90        # matches DETECTION_WINDOW_DAYS across r85/r96;
                              # the shortest window any percentile rank is
                              # reported from
PRIMARY_HAWKES_N = 0.5           # r96_conservative's own PRIMARY_N
PRIMARY_HAWKES_HALFLIFE_DAYS = 7  # r96_conservative's own PRIMARY_HALFLIFE_DAYS

CORR_KILL_THRESH = 0.5       # Step-0 kill switch (b): mean pairwise |rho|
CV_KILL_THRESH = 0.05        # Step-0 kill switch (c): CoV degeneracy floor
EPISODE_WINDOW_DAYS = 60     # matches r82/r85/r96_shared's own episode_window default

MODEL_NAMES = ("bocpd", "kalman", "csd", "hawkes")


# ------------------------------------------------------------------------
# Shared causal normalization primitive
# ------------------------------------------------------------------------

def causal_rolling_percentile_rank(x: pd.Series, window: int = BASELINE_WINDOW_DAYS,
                                    min_periods: int = MIN_PERIODS_DAYS) -> pd.Series:
    """Causal rolling percentile rank of `x` against its own trailing
    `window`-length history, mapped onto [0, 1].

    Row t = the fraction of `x[t-window+1 .. t]` (inclusive, i.e. INCLUDING
    x[t] itself) that is `<= x[t]`. Strictly causal by construction (each
    row is a rolling-window computation reading only rows `<= t`); does not
    require, and never uses, any row after `t`. NaN wherever `x` itself is
    NaN or fewer than `min_periods` valid observations are available in the
    trailing window.
    """
    def _pct_rank(arr: np.ndarray) -> float:
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.mean(valid <= last))

    return x.rolling(window, min_periods=min_periods).apply(_pct_rank, raw=True)


# ------------------------------------------------------------------------
# Per-model raw alarm scalar + normalized [0,1] state
# ------------------------------------------------------------------------

def bocpd_raw_alarm(df: pd.DataFrame) -> pd.Series:
    """Raw BOCPD alarm: `p_recent_cp` (already an alarm probability in
    [0,1] -- see r82_shared.bocpd_daily's own docstring), daily-cadence,
    causally aligned onto `df`'s 5-minute index."""
    sig = bocpd_daily_causal_signals(df)
    return sig["bocpd_p_recent_cp"].rename("bocpd_raw_alarm")


def kalman_raw_alarm(df: pd.DataFrame) -> pd.Series:
    """Raw Kalman alarm: `abs(kalman_slope)` -- the magnitude of the
    filtered local-linear trend estimate, unsigning the one naturally
    directional panel member (see module docstring, point 2)."""
    sig = kalman_daily_causal_signals(df)
    return sig["kalman_slope"].abs().rename("kalman_raw_alarm")


def csd_raw_alarm(df: pd.DataFrame) -> pd.Series:
    """Raw CSD alarm: mean of the variance and autocorrelation trend
    z-scores (see module docstring, point 3)."""
    sig = csd_daily_causal_signals(df)
    return sig[["csd_var_z", "csd_autocorr_z"]].mean(axis=1).rename("csd_raw_alarm")


def hawkes_raw_alarm(df: pd.DataFrame, n: float = PRIMARY_HAWKES_N,
                      halflife_days: float = PRIMARY_HAWKES_HALFLIFE_DAYS) -> pd.Series:
    """Raw Hawkes alarm: the intensity z-score at r96_conservative's own
    pre-registered PRIMARY (n, halflife_days) cell, causally aligned onto
    `df`'s 5-minute index via the same full-calendar-day shift every other
    daily-cadence signal in this project uses."""
    from tradebot.data import align_onchain_causal

    event_flag = intraday_relative_jump(df)
    lam = hawkes_intensity_daily(event_flag, n=n, halflife_days=halflife_days)
    z = hawkes_intensity_zscore(lam).rename("hawkes_raw_alarm")
    daily = z.to_frame()
    return align_onchain_causal(daily, df)["hawkes_raw_alarm"]


def build_normalized_states(df: pd.DataFrame,
                             window: int = BASELINE_WINDOW_DAYS,
                             min_periods: int = MIN_PERIODS_DAYS) -> pd.DataFrame:
    """The four raw alarm scalars, each passed through the SAME causal
    rolling-percentile-rank normalization, aligned onto `df`'s own index.
    Columns: `bocpd`, `kalman`, `csd`, `hawkes`, each in [0, 1] (or NaN
    during warmup)."""
    raw = {
        "bocpd": bocpd_raw_alarm(df),
        "kalman": kalman_raw_alarm(df),
        "csd": csd_raw_alarm(df),
        "hawkes": hawkes_raw_alarm(df),
    }
    # Normalize on the DAILY-cadence series (before the 5-minute causal
    # forward-fill) so the rolling window is genuinely `window` CALENDAR
    # DAYS of distinct daily observations, not `window` days' worth of
    # forward-filled 5-minute bars (which would just repeat the identical
    # value ~288x per day and not change the percentile-rank semantics,
    # but is needlessly slow) -- resample back to daily first, normalize,
    # then reindex onto df exactly like the underlying signals already do.
    out = {}
    for name, series in raw.items():
        daily = series.resample("1D").first().dropna()
        ranked = causal_rolling_percentile_rank(daily, window=window, min_periods=min_periods)
        out[name] = ranked.reindex(ranked.index.union(df.index)).sort_index().ffill().reindex(df.index)
    return pd.DataFrame(out, index=df.index)[list(MODEL_NAMES)]


# ------------------------------------------------------------------------
# Step-0 gate computations
# ------------------------------------------------------------------------

def pairwise_abs_corr_matrix(states: pd.DataFrame) -> pd.DataFrame:
    """4x4 matrix of pairwise |Pearson correlation| among the normalized
    states (NaN rows dropped pairwise, standard `pd.DataFrame.corr`
    semantics)."""
    return states.corr().abs()


def mean_pairwise_abs_corr(states: pd.DataFrame) -> float:
    """Mean of the six OFF-DIAGONAL entries of `pairwise_abs_corr_matrix`."""
    mat = pairwise_abs_corr_matrix(states).to_numpy()
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.nanmean(mat[iu]))


def cross_sectional_std(states: pd.DataFrame) -> pd.Series:
    """The disagreement statistic: standard deviation across the four
    normalized states at each bar (Bomberger 1996's own cross-panel
    disagreement definition; see module docstring)."""
    return states.std(axis=1, skipna=False).rename("cross_sectional_std")


def episode_disagreement_summary(bars: pd.DataFrame, disagreement: pd.Series,
                                  window_days: int = EPISODE_WINDOW_DAYS) -> pd.DataFrame:
    """Mean/std of `disagreement` within each of the six STRESS_EPISODES
    windows (identical dates/convention to r82/r85/r96_shared.py's own
    `episode_window`)."""
    rows = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, window_days)
        vals = disagreement.reindex(window).dropna()
        rows.append(dict(
            episode=label, onset=onset_str, n_bars=int(len(vals)),
            mean=float(vals.mean()) if len(vals) else float("nan"),
            std=float(vals.std()) if len(vals) else float("nan"),
        ))
    return pd.DataFrame(rows)


def step0_gate(states: pd.DataFrame, bars: pd.DataFrame,
               corr_kill_thresh: float = CORR_KILL_THRESH,
               cv_kill_thresh: float = CV_KILL_THRESH,
               episode_window_days: int = EPISODE_WINDOW_DAYS,
               states_for_episodes: pd.DataFrame | None = None) -> dict:
    """The full pre-registered Step-0 falsification gate. `states` MUST
    already be restricted to whatever span the caller wants the
    correlation check (b), AND the "~constant over the whole inner-train
    span" degeneracy check, computed over (inner-train, per
    pre-registration). `bars` is used to build the six episode windows for
    (c) and may span a wider (still non-holdout) range than `states` --
    three of the six STRESS_EPISODES onsets (2021-11, 2022-05, 2022-11)
    fall after INNER_TRAIN_END, so the PER-EPISODE disagreement check in
    (c) is built from `states_for_episodes` (a wider, still non-holdout
    range; defaults to `states` itself if not given) rather than being
    silently restricted to inner-train and returning empty windows for
    those three episodes.

    Returns a dict with the correlation matrix, mean |rho|, the
    disagreement statistic's overall (inner-train) and per-episode
    (wider-range) summary, both CoV checks, and the final PASS/FAIL
    verdict.
    """
    corr_matrix = pairwise_abs_corr_matrix(states)
    mean_rho = mean_pairwise_abs_corr(states)
    corr_kill = mean_rho >= corr_kill_thresh

    # "~constant over the whole inner-train span" (part of the pre-
    # registered degeneracy kill condition) is checked on `states` itself
    # -- the SAME span the correlation check (b) uses -- not on the wider
    # `states_for_episodes` range.
    disagreement_train = cross_sectional_std(states)
    overall_mean = float(disagreement_train.mean())
    overall_std = float(disagreement_train.std())
    overall_cv = (overall_std / overall_mean) if overall_mean else float("nan")

    # The six-episode dispersion check (c) needs data through 2022-11 FTX
    # (three of six onsets fall after INNER_TRAIN_END), so it is built from
    # `states_for_episodes` (a wider, still non-holdout range) instead.
    episode_states = states if states_for_episodes is None else states_for_episodes
    disagreement_episodes = cross_sectional_std(episode_states)
    ep_summary = episode_disagreement_summary(bars, disagreement_episodes, episode_window_days)
    ep_means = ep_summary["mean"].to_numpy()
    ep_mean_of_means = float(np.nanmean(ep_means))
    ep_std_of_means = float(np.nanstd(ep_means))
    ep_cv = (ep_std_of_means / ep_mean_of_means) if ep_mean_of_means else float("nan")

    degenerate = (bool(ep_cv < cv_kill_thresh) if np.isfinite(ep_cv) else True) or \
                 (bool(overall_cv < cv_kill_thresh) if np.isfinite(overall_cv) else True)

    passed = (not corr_kill) and (not degenerate)

    return dict(
        corr_matrix=corr_matrix,
        mean_pairwise_abs_corr=mean_rho,
        corr_kill=corr_kill,
        disagreement_overall_mean=overall_mean,
        disagreement_overall_std=overall_std,
        disagreement_overall_cv=overall_cv,
        episode_summary=ep_summary,
        episode_mean_of_means=ep_mean_of_means,
        episode_std_of_means=ep_std_of_means,
        episode_cv=ep_cv,
        degenerate=degenerate,
        passed=passed,
    )


def print_step0_report(gate: dict) -> None:
    print("4x4 pairwise |correlation| matrix (normalized states):")
    print(gate["corr_matrix"].round(3).to_string())
    print(f"\nmean pairwise |rho| = {gate['mean_pairwise_abs_corr']:.4f}"
          f"  (kill >= {CORR_KILL_THRESH})  -> "
          f"{'KILL' if gate['corr_kill'] else 'ok'}")
    print(f"\ncross_sectional_std overall: mean={gate['disagreement_overall_mean']:.4f} "
          f"std={gate['disagreement_overall_std']:.4f} "
          f"CoV={gate['disagreement_overall_cv']:.4f}  (kill < {CV_KILL_THRESH})")
    print("\nper-episode cross_sectional_std summary:")
    print(gate["episode_summary"].to_string(index=False))
    print(f"\nacross-episode CoV of episode means = {gate['episode_cv']:.4f}"
          f"  (kill < {CV_KILL_THRESH})")
    print(f"\ndegenerate dispersion: {gate['degenerate']}")
    verdict = "PASS" if gate["passed"] else "FAIL"
    print(f"\nSTEP-0 GATE VERDICT: {verdict}")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    # causal_rolling_percentile_rank: strictly causal, bounded [0,1],
    # correctly ranks a known monotone sequence.
    idx = pd.date_range("2017-01-01", periods=50, freq="1D", tz="UTC")
    x = pd.Series(np.arange(50, dtype=float), index=idx)
    ranked = causal_rolling_percentile_rank(x, window=10, min_periods=5)
    valid = ranked.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()
    # A strictly increasing series' rolling percentile rank should always
    # be 1.0 (the current value is always the max of its trailing window).
    assert np.allclose(valid.to_numpy(), 1.0), valid.to_numpy()

    # causal_rolling_percentile_rank is unaffected by extending the series
    # into the future -- an earlier row's value must not change (a direct
    # no-lookahead probe, same spirit as `causal_truncation_probe_series`
    # elsewhere in this project).
    x2 = pd.concat([x, pd.Series([1000.0, -1000.0],
                                  index=pd.date_range("2017-02-20", periods=2, freq="1D", tz="UTC"))])
    ranked2 = causal_rolling_percentile_rank(x2, window=10, min_periods=5)
    common = ranked.index
    assert np.allclose(ranked.reindex(common).to_numpy(), ranked2.reindex(common).to_numpy(),
                        equal_nan=True), "percentile rank must be causal (no lookahead)"

    # mean_pairwise_abs_corr / cross_sectional_std: known-input sanity.
    n = 500
    rng = np.random.default_rng(106)
    idx2 = pd.date_range("2018-01-01", periods=n, freq="1D", tz="UTC")
    base = rng.normal(size=n)
    # Two states are the SAME series (rho=1 => contributes 1.0 to the mean);
    # two states are independent noise.
    states = pd.DataFrame({
        "bocpd": base,
        "kalman": base,
        "csd": rng.normal(size=n),
        "hawkes": rng.normal(size=n),
    }, index=idx2)
    mat = pairwise_abs_corr_matrix(states)
    assert abs(mat.loc["bocpd", "kalman"] - 1.0) < 1e-9
    assert np.allclose(np.diag(mat.to_numpy()), 1.0)
    mean_rho = mean_pairwise_abs_corr(states)
    assert 0.0 < mean_rho < 1.0

    # cross_sectional_std of four IDENTICAL columns must be exactly zero
    # (degenerate-disagreement sanity check, mirroring the Step-0 kill
    # condition this statistic exists to detect).
    same = pd.DataFrame({m: base for m in MODEL_NAMES}, index=idx2)
    disagreement_same = cross_sectional_std(same)
    assert np.allclose(disagreement_same.to_numpy(), 0.0)

    # step0_gate: four identical states must be caught by BOTH kill
    # conditions (corr = 1.0 >= thresh, and disagreement is uniformly zero
    # so both CoV checks read as degenerate).
    fake_bars = pd.DataFrame({"close": np.exp(np.cumsum(rng.normal(0, 0.001, n)))}, index=idx2)
    gate_degenerate = step0_gate(same, fake_bars)
    assert gate_degenerate["corr_kill"] is True
    assert gate_degenerate["passed"] is False

    # Four independent-noise states, scaled so dispersion genuinely varies
    # across episodes, should NOT trip the kill switches.
    indep = pd.DataFrame({m: rng.normal(size=n) for m in MODEL_NAMES}, index=idx2)
    gate_indep = step0_gate(indep, fake_bars)
    assert gate_indep["mean_pairwise_abs_corr"] < CORR_KILL_THRESH

    # STRESS_EPISODES sanity: exactly six, matching r82/85/96_shared's own
    # table verbatim (already asserted at import time above, re-checked
    # here for the self-test's own defense-in-depth).
    assert len(STRESS_EPISODES) == 6


_self_test()
