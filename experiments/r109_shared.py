"""Shared, read-only utilities and pre-registration for the R-109 round (08-24).

DIRECTION, in one sentence: build a continuous ERR-axis uncertainty proxy from
DISTRIBUTIONAL NOVELTY / DATASET SHIFT -- is the CURRENT multivariate market-
state feature vector unlike anything `kelly_regime_v4`'s own vote has recently
been calibrated against, regardless of whether any model disagrees with any
other model or whether the vote's historical edge is statistically significant
-- and discount exposure when the current state looks novel relative to its
own trailing history, as a SIXTH ERR-axis attempt and the first keyed on
epistemic/novelty uncertainty (is THIS bar unlike recent bars at all) rather
than sampling uncertainty (R-28/retracted, R-87, R-104: three attempts, is the
vote's realized edge distinguishable from zero) or specification/model
disagreement (R-105: within one model family, alternative anchor ladders and a
leave-one-anchor jackknife; R-106: across four structurally different
regime/turbulence detectors -- BOCPD/Kalman/CSD/Hawkes).

**Literature grounding, fetched and read via WebSearch this round:**

- Rabanser, S., Gunnemann, S., & Lipton, Z. (2019), "Failing Loudly: An
  Empirical Study of Methods for Detecting Dataset Shift", *NeurIPS 2019*.
  The general framing this round rests on: a model trained/calibrated on one
  data distribution degrades, silently and often severely, once deployed
  data drifts away from that distribution -- and the shift itself is directly
  measurable, via a distance/novelty statistic over the model's own INPUT
  FEATURES, without ever touching the model's output, its residuals, or any
  P&L series. Every prior ERR-axis round in this ledger (R-28, R-87, R-104,
  R-105, R-106) measured uncertainty as a property of the VOTE's own edge,
  confidence, or cross-model spread; none measured whether the market STATE
  ITSELF looks unlike anything recently seen, independent of any model output.
- De Maesschalck, R., Jouan-Rimbaud, D., & Massart, D. L. (2000), "The
  Mahalanobis Distance", *Chemometrics and Intelligent Laboratory Systems*
  50(1), 1-18. The CONSERVATIVE branch's exact operationalization: a
  covariance-weighted distance of a new observation from a reference
  distribution's mean, the textbook-standard parametric novelty/outlier
  statistic, assuming the reference distribution is (locally) elliptical.
- Ramaswamy, S., Rastogi, R., & Shim, K. (2000), "Efficient Algorithms for
  Mining Outliers from Large Data Sets", *ACM SIGMOD*; Breunig, M. M.,
  Kriegel, H.-P., Ng, R. T., & Sander, J. (2000), "LOF: Identifying
  Density-Based Local Outliers", *ACM SIGMOD*. The NOVEL branch's family:
  nonparametric, distance-to-k-nearest-neighbours outlier/novelty scoring,
  which makes no elliptical/single-Gaussian assumption about the reference
  distribution's shape -- the literature's standard alternative to Mahalanobis
  distance precisely when a reference distribution may be multimodal (e.g.
  genuinely different market regimes forming separate clusters rather than
  one stretched Gaussian blob), a scenario Mahalanobis distance instead
  averages over. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008), "Isolation
  Forest", *ICDM 2008*, is the closely-related tree-ensemble alternative
  cited as the branch's own motivation for choosing a genuinely different
  ALGORITHM CLASS (nonparametric/ensemble) from the conservative branch's
  single fitted Gaussian, not merely a different parameter grid on the same
  statistic.

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path). Sixth attempt, after R-28 (e-process drawdown cut, RETRACTED),
R-87 (Adaptive Conformal Inference on the vote's own confidence and the
Kelly-scale's dispersion, both NEGATIVE), R-104 (periodic bootstrap / HAC-PSR
significance discount of the vote's own historical edge, both NEGATIVE),
R-105 (leave-one-anchor jackknife and alternative-ladder ensemble
disagreement, both NEGATIVE), R-106 (cross-model-class disagreement among
BOCPD/Kalman/CSD/Hawkes, both NEGATIVE). R-106's own closing line named the
live, untried candidate this round tries: "an ERR-axis construction keyed on
none of sampling significance, within-family specification disagreement, or
cross-model-class disagreement."

**Not a duplicate of:**
- R-104 (bootstrap/HAC significance of the vote's own historical P&L): a
  SAMPLING-uncertainty construction over a P&L time series. Nothing in this
  module ever reads `kelly_regime_v4`'s returns, exposure, or P&L to build
  its novelty statistic -- the reference distribution and the distance
  statistic below are built ENTIRELY from OHLCV-derived market-state
  features (realized vol, trend-anchor dispersion, return kurtosis), never
  from the strategy's own realized edge.
- R-105 (disagreement among alternative anchor-ladder specifications of the
  SAME 20/40/80 vote, and a leave-one-anchor jackknife of that vote's own
  three components): a SPECIFICATION-disagreement construction over
  alternative parameterizations of one fixed mechanism (the moving-average
  vote). This module never constructs an alternative anchor ladder, never
  reads `vote_frac` with any horizon set other than V4_HORIZONS, and computes
  no jackknife or ensemble-spread statistic over model variants at all --
  its distance statistic is a property of ONE feature vector (today's market
  state) against ONE reference distribution (that same feature's own
  trailing history), never a comparison across multiple models or multiple
  parameterizations of anything.
- R-106 (cross-model-class disagreement among BOCPD/Kalman/CSD/Hawkes): this
  module imports NONE of r82_shared/r83_novel_kalman_shared/r85_shared/
  r96_shared and computes no cross-sectional dispersion across multiple
  detector outputs. R-106 asks "do four independent MODELS disagree with each
  other right now"; this round asks "is the market's own STATE, this round's
  novelty features, unlike its own recent past" -- a first-order statement
  about the DATA, computable from a single reference distribution with no
  second model to disagree with at all.
- Every regime-timing round (R-01 HMM, R-82 BOCPD, R-83 Kalman LLT, R-85 CSD,
  R-86 transfer entropy, R-96 Hawkes, R-98 POT/GPD tail-shape, R-99 BNS
  bipower-variation jump split): every one of these asked "does THIS
  estimator's own alarm CROSS a threshold before v4's own anchor gate reacts,
  around six dated historical stress-episode ONSETS" -- a detection-LAG race
  against a calendar of named events, scored by episode-window hit-rate. This
  module computes no detection lag, races against no episode onset calendar,
  and produces a CONTINUOUS distance statistic at every bar rather than a
  binary alarm-crossed/not-crossed state -- Step-0 below checks the
  statistic's own dispersion and degeneracy, never a hit-rate against
  STRESS_EPISODES.
- Every SIZE-axis round (R-34...R-103, including R-99's magnitude/jump split
  and R-102's downside/upside semivariance sign split of v4's OWN realized-
  variance object): all retune `scale`'s magnitude directly, or replace one
  of its two inputs (vote or vol) outright. This round, like R-87/R-104/
  R-105/R-106 before it, bolts a MULTIPLICATIVE DISCOUNT on top of v4's
  existing `frac * scale` product completely unchanged (verified by the
  Step-0 R2_VS_V4_THRESH kill switch below) -- it never reads or rebuilds
  `v4_symmetric_vol`'s OWN role in v4's sizing formula; its three novelty
  features are used ONLY to build a reference-distance statistic, and the
  Step-0 R2_VS_VOL_THRESH kill switch specifically guards against this
  statistic secretly collapsing into a relabelled realized-volatility
  rescale, the exact failure mode 26 prior SIZE-axis attempts have hit.

This module is written by the operator BEFORE the branches are dispatched and
is READ-ONLY for both -- neither branch may edit it. Nothing here reads a bar
at or after OOS_START (2023-01-01); every function that walks a data frame is
either called through `assert_no_holdout`-guarded slices (`compare()`,
`run_slice()`, inherited unmodified through r102_shared -> ... -> r106_shared)
or is explicitly restricted to non-holdout ranges by the caller.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) The three novelty features (realized vol, anchor-ladder dispersion, return
kurtosis) may simply co-move with `v4_symmetric_vol` closely enough that the
Mahalanobis/kNN distance is, in substance, a relabelled volatility rescale --
guarded by the R2_VS_VOL_THRESH Step-0 kill switch, but if that check is too
lax the branches would silently reproduce the 26-attempt SIZE-axis collapse
R-102's own closing line named ("a SIZE-axis construction whose state
variable does not collapse toward... a near-constant exposure fraction
relative to v4").
(2) `kelly_regime_v4`'s own reactive, latched vote may already price in
"unusual" conditions by the time they are unusual enough to register as
distributionally novel -- reproducing the R-87/R-104/R-105/R-106 "real but
inert" pattern (Step-0 passes, B1 does not) by a sixth, structurally
different estimator.
(3) A reference distribution built from a ROLLING 730-day window is itself
adaptive -- if the market state drifts slowly (e.g. a multi-year volatility
regime), the reference distribution drifts with it and yesterday's "novel"
conditions become tomorrow's new normal within months, so the novelty score
may only ever fire briefly at the SPEED of a genuine regime break rather than
persisting through the stress period the strategy most needs protecting
through -- the same class of concern R-77 named for the execution-urgency
brake and R-90 named for the trailing-stop ratchet.
(4) The reference distribution is fit on inner-train/inner-validation data
that is itself dominated by BTC's single 2017-2020 bull-to-bear supercycle
(the same concentration R-63's breadth analysis and R-14's own single-cycle
caveat both name); a novelty statistic calibrated mostly against one cycle
may not generalise to ETH's shorter, differently-shaped history at all,
which is exactly what the pre-registered B4 falsification test below is
designed to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from r106_shared (itself chaining r105_shared ->
# r104_shared -> r103_shared -> r102_shared): identical control machinery, so
# every number this round produces is directly comparable to R-101...R-106's.
from experiments.r106_shared import (  # noqa: E402,F401
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
    causal_rolling_percentile_rank,
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

# The B1/B2/B4/B5 promotion-bar machinery and the inner-validation comparison
# helper: NOT re-exported past r105_shared (r106_shared never reached B1-B5,
# per its own Step-0-only closing verdict), imported directly from there --
# identical gate code to R-104/R-105, so a pass/fail here is directly
# comparable to theirs.
from experiments.r105_shared import (  # noqa: E402,F401
    B3_MIN_DAYS_GRID as _UNUSED_B3_GRID,  # not reused (this round defines its own B3 grid)
    FEE_TIER,
    SHARPE_NOISE_FLOOR,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    inner_val_rows,
    print_plateau_table,
)

# Generic causal truncation probe (df -> array builder), reused verbatim for
# this round's own self-test rather than reimplemented.
from experiments.r102_shared import (  # noqa: E402,F401
    V4_HORIZONS,
    causal_truncation_probe_series,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
BASELINE_WINDOW_DAYS = 730     # rolling reference window; identical to
                                # r85/r86/r96/r106_shared's own convention
MIN_REF_DAYS = 180             # minimum trailing days before a reference
                                # distribution/distance is reported at all
KURT_WINDOW_BARS = BARS_PER_DAY  # 1-day trailing window for return kurtosis
ANCHOR_HORIZONS = V4_HORIZONS   # reuse v4's own 20/40/80 anchors for the
                                 # dispersion feature -- no new nuisance
                                 # horizon parameter introduced
RIDGE_EPS = 1e-6                # covariance ridge regularisation

BIND_FRAC_THRESH = 0.01         # Step-0 kill switch A: must bind >1% of inner-train
R2_VS_V4_THRESH = 0.98          # Step-0 kill switch B: discount path must not be a
                                 # near-exact rescale of v4's own final target (the
                                 # generic non-inertness check every prior round uses)
R2_VS_VOL_THRESH = 0.90         # Step-0 kill switch C, THIS ROUND'S OWN: the discount
                                 # path must not be a near-exact rescale of v4's own
                                 # realized-vol scale input -- guards specifically
                                 # against reproducing the 26-attempt SIZE-axis
                                 # "just relabelled volatility" collapse
CV_KILL_THRESH = 0.05           # Step-0 kill switch D: novelty state must not be
                                 # ~constant (reuses r106_shared's own CoV convention)

MODEL_NAMES = ("mahalanobis", "knn")  # the two branches' own primary statistic name

# Step-0 selection grid, shared by both branches (mirrors r105_shared's own
# STEP0_FLOOR_GRID/SELECTION_ORDER convention): swept BEFORE any inner-
# validation Sharpe/PnL number is read, primary cell picked from where the
# grid genuinely passes Step-0 (verified against real BTC inner-train data
# by the operator while building this shared module -- see the smoke-test
# note above), not from the inner-validation result either branch produces.
STEP0_THRESH_GRID = (0.80, 0.90, 0.95)
STEP0_MAXD_GRID = (0.5, 1.0)
PRIMARY_THRESH = 0.90
PRIMARY_MAXD = 1.0
SELECTION_ORDER = ((0.90, 1.0), (0.95, 1.0), (0.80, 0.5), (0.90, 0.5), (0.95, 0.5), (0.80, 1.0))


# ------------------------------------------------------------------------
# (1) Causal novelty features -- pure functions of OHLCV, no new data
# channel, no strategy P&L or exposure read anywhere.
# ------------------------------------------------------------------------

def feature_log_vol(df: pd.DataFrame) -> np.ndarray:
    """log(v4's own realized EWM volatility input), already shifted by 1 bar
    inside `v4_symmetric_vol` -- vol-clustering feature."""
    from experiments.r102_shared import v4_symmetric_vol
    vol = np.asarray(v4_symmetric_vol(df), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(vol > 0, np.log(vol), np.nan)
    return out


def feature_anchor_dispersion(df: pd.DataFrame, horizons=ANCHOR_HORIZONS) -> np.ndarray:
    """Mean absolute pairwise %% difference among the SMA(20)/SMA(40)/SMA(80)
    anchors v4's own vote is built from, normalized by close and shifted by 1
    bar -- a continuous "how far apart do short/med/long trend estimates
    currently sit" feature, distinct from the vote's own discrete +/-1/0
    sign. Uses the SAME anchor horizons the shipped vote uses; introduces no
    new nuisance parameter."""
    close = df["close"]
    smas = [close.rolling(h * BARS_PER_DAY, min_periods=h * BARS_PER_DAY).mean()
            for h in horizons]
    n = len(smas)
    pair_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_diffs.append((smas[i] - smas[j]).abs() / close)
    disp = pd.concat(pair_diffs, axis=1).mean(axis=1)
    return disp.shift(1).to_numpy()


def feature_kurtosis(df: pd.DataFrame, window: int = KURT_WINDOW_BARS) -> np.ndarray:
    """Rolling excess kurtosis of 5-minute log returns over a trailing
    1-day window, shifted by 1 bar -- a higher (4th) -moment tail-fatness
    feature; distinct from every prior SIZE-axis round's 2nd-moment
    (variance/semivariance) decompositions."""
    r = np.log(df["close"]).diff()
    k = r.rolling(window, min_periods=window).kurt()
    return k.shift(1).to_numpy()


FEATURE_BUILDERS = {
    "log_vol": feature_log_vol,
    "anchor_disp": feature_anchor_dispersion,
    "kurtosis": feature_kurtosis,
}
# Richer panel for the novel (kNN) branch only -- two additional OHLCV-only
# features, still zero new data channels.

def feature_volume_z(df: pd.DataFrame, window: int = BARS_PER_DAY * 30) -> np.ndarray:
    """Rolling z-score of bar volume against its own trailing 30-day mean/std,
    shifted by 1 bar."""
    vol = df["volume"].astype(float)
    mu = vol.rolling(window, min_periods=window).mean()
    sd = vol.rolling(window, min_periods=window).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (vol - mu) / sd
    return z.shift(1).to_numpy()


def feature_skew(df: pd.DataFrame, window: int = KURT_WINDOW_BARS) -> np.ndarray:
    """Rolling skewness of 5-minute log returns over the same trailing
    1-day window as `feature_kurtosis`, shifted by 1 bar."""
    r = np.log(df["close"]).diff()
    s = r.rolling(window, min_periods=window).skew()
    return s.shift(1).to_numpy()


NOVEL_FEATURE_BUILDERS = dict(FEATURE_BUILDERS, volume_z=feature_volume_z, skew=feature_skew)


def build_daily_features(df: pd.DataFrame, builders: dict[str, callable] | None = None) -> pd.DataFrame:
    """Each feature builder evaluated on the full 5-minute frame, resampled to
    ONE observation per calendar day (first non-NaN value of the day -- every
    builder already only changes once per day or slower for the SMA/vol
    features, and the intraday kurtosis/skew/volume-z features are themselves
    already point-in-time at each bar, so the daily resample is a
    once-per-day snapshot rather than an aggregation), matching the daily
    cadence r85/r96/r106_shared's own detector modules already use."""
    builders = builders or FEATURE_BUILDERS
    cols = {}
    for name, fn in builders.items():
        s = pd.Series(fn(df), index=df.index, name=name)
        cols[name] = s.resample("1D").first()
    return pd.DataFrame(cols).dropna(how="all")


# ------------------------------------------------------------------------
# (2) Reference-distance statistics -- CAUSAL by construction: the
# reference distribution/set at day t is built ONLY from days strictly
# BEFORE t (an explicit `.shift(1)` on the daily panel before any rolling
# window is taken), so day t's own feature vector never contributes to its
# own reference.
# ------------------------------------------------------------------------

def rolling_mahalanobis_distance(daily: pd.DataFrame, window: int = BASELINE_WINDOW_DAYS,
                                  min_periods: int = MIN_REF_DAYS,
                                  ridge_eps: float = RIDGE_EPS) -> pd.Series:
    """Day t's Mahalanobis distance of `daily.loc[t]` from the mean/covariance
    of `daily.loc[t-window : t-1]` (strictly prior days only). NaN until
    `min_periods` prior days are available."""
    arr = daily.to_numpy(dtype=float)
    n, k = arr.shape
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window)
        ref = arr[lo:t]  # strictly prior to t -- t itself excluded
        valid = ref[np.all(np.isfinite(ref), axis=1)]
        x = arr[t]
        if len(valid) < min_periods or not np.all(np.isfinite(x)):
            continue
        mu = valid.mean(axis=0)
        cov = np.cov(valid, rowvar=False)
        cov = cov + ridge_eps * np.eye(k) * (np.trace(cov) / k if np.trace(cov) > 0 else 1.0)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        d = x - mu
        out[t] = float(np.sqrt(max(d @ inv @ d, 0.0)))
    return pd.Series(out, index=daily.index, name="mahalanobis")


def rolling_knn_distance(daily: pd.DataFrame, window: int = BASELINE_WINDOW_DAYS,
                          min_periods: int = MIN_REF_DAYS, k: int = 10,
                          refit_every: int = 30) -> pd.Series:
    """Day t's mean Euclidean distance (in per-feature standardized space,
    z-scored against the SAME reference window used to fit) to its `k`
    nearest neighbours in the trailing `window`-day reference SET (strictly
    prior days only), nonparametric density-based novelty (Ramaswamy et al.
    2000 / Breunig et al. 2000's own kNN-distance family) rather than a
    single-Gaussian assumption. The reference set and its standardization are
    refit only every `refit_every` days (walk-forward, still strictly causal:
    a refit at day t uses only days < t) and held fixed between refits, both
    for speed and because it is the more realistic deployable construction
    (re-fitting a nonparametric reference on every single bar is not
    something a live system would do either)."""
    arr = daily.to_numpy(dtype=float)
    n, kdim = arr.shape
    out = np.full(n, np.nan)
    ref_valid = None
    mu = sd = None
    last_refit = -10 ** 9
    for t in range(n):
        if t - last_refit >= refit_every:
            lo = max(0, t - window)
            ref = arr[lo:t]  # strictly prior to t
            valid = ref[np.all(np.isfinite(ref), axis=1)]
            if len(valid) >= min_periods:
                mu = valid.mean(axis=0)
                sd = valid.std(axis=0)
                sd = np.where(sd > 0, sd, 1.0)
                ref_valid = (valid - mu) / sd
                last_refit = t
        x = arr[t]
        if ref_valid is None or not np.all(np.isfinite(x)):
            continue
        xz = (x - mu) / sd
        dists = np.sqrt(((ref_valid - xz) ** 2).sum(axis=1))
        kk = min(k, len(dists))
        nearest = np.partition(dists, kk - 1)[:kk]
        out[t] = float(nearest.mean())
    return pd.Series(out, index=daily.index, name="knn")


def align_daily_to_bars(daily: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Forward-fill a once-a-day statistic onto `df`'s 5-minute index --
    identical convention to r82/85/96/106_shared's own daily-signal
    alignment (`tradebot.data.align_onchain_causal`-style forward fill)."""
    return daily.reindex(daily.index.union(df.index)).sort_index().ffill().reindex(df.index)


# ------------------------------------------------------------------------
# (3) Discount construction -- shared by both branches: a threshold/scale
# pair maps the [0,1] novelty state onto a multiplicative exposure discount.
# ------------------------------------------------------------------------

def novelty_discount(state: pd.Series, thresh: float, max_discount: float) -> pd.Series:
    """`0` below `thresh`; ramps linearly to `max_discount` as `state` -> 1."""
    x = (state - thresh) / max(1e-9, (1.0 - thresh))
    return x.clip(lower=0.0, upper=1.0) * max_discount


def discount_series_for(df: pd.DataFrame, state: pd.Series, thresh: float,
                         max_discount: float) -> pd.Series:
    """The bar-aligned discount FRACTION (in [0,1]) implied by a DAILY
    `state` series (e.g. `causal_rolling_percentile_rank` of a daily
    distance series). Aligned via `align_daily_to_bars` -- a proper
    union-then-forward-fill, NOT a bare `.reindex` (a bare reindex only
    matches the handful of 5-minute bars whose timestamp happens to equal a
    daily label exactly, silently leaving >85% of bars NaN; caught by this
    round's own smoke test against real BTC data before either branch was
    dispatched, see the module changelog note in the docstring above)."""
    aligned = align_daily_to_bars(state, df).fillna(0.0)
    return novelty_discount(aligned, thresh, max_discount)


def apply_discount(df: pd.DataFrame, state: pd.Series, thresh: float,
                    max_discount: float) -> np.ndarray:
    """v4's own final target, multiplicatively discounted by this round's
    novelty state -- the SAME architecture R-87/R-104/R-105/R-106 all use
    (a discount bolted onto `v4_target` unchanged, never a replacement of
    `frac` or `scale`)."""
    discount = discount_series_for(df, state, thresh, max_discount)
    return v4_target(df) * (1.0 - discount.to_numpy())


# ------------------------------------------------------------------------
# (4) Step-0 gate
# ------------------------------------------------------------------------

def step0_gate(df: pd.DataFrame, state: pd.Series, thresh: float, max_discount: float,
               bind_frac_thresh: float = BIND_FRAC_THRESH,
               r2_v4_thresh: float = R2_VS_V4_THRESH,
               r2_vol_thresh: float = R2_VS_VOL_THRESH,
               cv_kill_thresh: float = CV_KILL_THRESH) -> dict:
    """Pre-registered Step-0 falsification gate for one (thresh, max_discount)
    grid cell, restricted to whatever span `df`/`state` are already
    restricted to by the caller (inner-train, per this round's own
    pre-registration -- callers must slice `df` themselves before calling
    this, exactly as every prior round's Step-0 gate does).

    (a) bind_frac: fraction of BARS where the discount FRACTION itself
    (not the resulting exposure path -- exposure is nonzero most of the
    time regardless of any discount) is > 0.
    (b) r2_vs_v4: R^2 of the resulting DISCOUNTED exposure path against
    v4's own unmodified target -- must not be a near-exact rescale (i.e.
    the discount must be doing SOMETHING, not nothing).
    (c) r2_vs_vol: R^2 of the discount FRACTION itself against v4's own
    realized-vol input -- must not be a relabelled volatility rescale
    (this round's own named failure risk, see module docstring point 1).
    (d) state_cv: the underlying [0,1] novelty state must have genuine
    (non-degenerate) dispersion.
    """
    discount = discount_series_for(df, state, thresh, max_discount)
    disc_arr = discount.to_numpy(dtype=float)
    finite = np.isfinite(disc_arr)
    bind_frac = float(np.mean(disc_arr[finite] > 1e-9)) if finite.any() else 0.0
    bind_ok = bind_frac > bind_frac_thresh

    candidate_path = v4_target(df) * (1.0 - disc_arr)
    v4_path = v4_target(df)
    r2_v4 = r_squared(candidate_path[finite], v4_path[finite]) if finite.any() else 1.0
    not_v4_rescale = r2_v4 < r2_v4_thresh

    from experiments.r102_shared import v4_symmetric_vol
    vol_path = np.asarray(v4_symmetric_vol(df), dtype=float)
    both_finite = finite & np.isfinite(vol_path)
    r2_vol = r_squared(disc_arr[both_finite], vol_path[both_finite]) if both_finite.any() else 1.0
    not_vol_rescale = r2_vol < r2_vol_thresh

    s = state.to_numpy(dtype=float)
    s_finite = s[np.isfinite(s)]
    cv = float(s_finite.std() / s_finite.mean()) if len(s_finite) and s_finite.mean() else float("nan")
    non_degenerate = np.isfinite(cv) and cv >= cv_kill_thresh

    passed = bind_ok and not_v4_rescale and not_vol_rescale and non_degenerate
    return dict(bind_frac=bind_frac, bind_ok=bind_ok, r2_vs_v4=r2_v4,
                not_v4_rescale=not_v4_rescale, r2_vs_vol=r2_vol,
                not_vol_rescale=not_vol_rescale, state_cv=cv,
                non_degenerate=non_degenerate, passed=passed)


def print_step0_report(label: str, gate: dict) -> None:
    print(f"\n--- Step-0 gate: {label} ---")
    print(f"bind_frac={gate['bind_frac']:.4f} (kill <= {BIND_FRAC_THRESH}) -> "
          f"{'ok' if gate['bind_ok'] else 'KILL'}")
    print(f"R^2 vs v4_target={gate['r2_vs_v4']:.4f} (kill >= {R2_VS_V4_THRESH}) -> "
          f"{'ok' if gate['not_v4_rescale'] else 'KILL (near-exact v4 rescale)'}")
    print(f"R^2 vs v4 realized vol={gate['r2_vs_vol']:.4f} (kill >= {R2_VS_VOL_THRESH}) -> "
          f"{'ok' if gate['not_vol_rescale'] else 'KILL (relabelled vol rescale)'}")
    print(f"state CoV={gate['state_cv']:.4f} (kill < {CV_KILL_THRESH}) -> "
          f"{'ok' if gate['non_degenerate'] else 'KILL (degenerate)'}")
    verdict = "PASS" if gate["passed"] else "FAIL"
    print(f"STEP-0 GATE VERDICT ({label}): {verdict}")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=250_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(107)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
                       index=idx)

    # Every feature builder must be strictly causal.
    for name, fn in NOVEL_FEATURE_BUILDERS.items():
        assert causal_truncation_probe_series(fn, df), f"{name} failed causal truncation probe"

    daily = build_daily_features(df)
    assert daily.shape[1] == 3
    assert len(daily) > 100

    # Mahalanobis distance: a reference built from strictly-prior days only
    # (t itself never in its own reference) -- verify via truncation: values
    # up to k must be identical whether computed on the full frame or on a
    # frame truncated at k+1 (one extra row can never move an earlier row).
    dist = rolling_mahalanobis_distance(daily)
    assert (dist.dropna() >= 0).all()
    k = len(daily) // 2
    daily_trunc = daily.iloc[:k + 1]
    dist_trunc = rolling_mahalanobis_distance(daily_trunc)
    common = dist.index[:k]
    a, b = dist.reindex(common).to_numpy(), dist_trunc.reindex(common).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 10
    assert np.allclose(a[ok], b[ok], atol=1e-8), "Mahalanobis distance is not causal"

    # kNN distance: identical causality check.
    knn = rolling_knn_distance(daily, refit_every=10)
    assert (knn.dropna() >= 0).all()
    knn_trunc = rolling_knn_distance(daily_trunc, refit_every=10)
    a2, b2 = knn.reindex(common).to_numpy(), knn_trunc.reindex(common).to_numpy()
    ok2 = np.isfinite(a2) & np.isfinite(b2)
    assert ok2.sum() > 10
    assert np.allclose(a2[ok2], b2[ok2], atol=1e-8), "kNN distance is not causal"

    # A degenerate (constant) feature panel must produce near-zero
    # Mahalanobis distance throughout (sanity: covariance of a constant
    # column is ridge-only, so distance stays small and finite, never NaN
    # or inf, and the Step-0 CV kill switch on it must fire).
    const_daily = pd.DataFrame({"a": np.full(400, 1.0), "b": np.full(400, 2.0),
                                 "c": np.full(400, 3.0)},
                                index=pd.date_range("2017-01-01", periods=400, freq="1D", tz="UTC"))
    dist_const = rolling_mahalanobis_distance(const_daily)
    valid_const = dist_const.dropna()
    assert len(valid_const) > 0
    assert np.all(np.isfinite(valid_const))
    state_const = causal_rolling_percentile_rank(dist_const, window=BASELINE_WINDOW_DAYS,
                                                   min_periods=MIN_REF_DAYS)
    cv_const = (state_const.std() / state_const.mean()) if state_const.mean() else float("nan")
    assert (not np.isfinite(cv_const)) or cv_const < CV_KILL_THRESH

    # novelty_discount: monotone, bounded, zero below threshold.
    s = pd.Series(np.linspace(0, 1, 101))
    d = novelty_discount(s, thresh=0.9, max_discount=0.5)
    assert (d[s <= 0.9] == 0).all()
    assert np.isclose(d.iloc[-1], 0.5)
    assert (d.diff().dropna() >= -1e-12).all()  # monotone non-decreasing


_self_test()
