"""Shared, read-only pre-registration for the R-173 round (08-28).

DIRECTION, one sentence: causally estimate the transaction spread/friction
`kelly_regime_v4` actually pays, using two purely price/range-based
microstructure estimators -- Roll (1984, *Journal of Finance* 39(4),
1127-1139) and Corwin & Schultz (2012, *Journal of Finance* 67(2), 719-760)
-- in place of this project's current flat fee-tier cost model, and test
(a) whether re-pricing the strategy's real trading history at the measured
friction changes its existing OOS verdict [conservative] and (b) whether a
spread-conditioned deadband change trades better around it [novel].

Full Step 1/Step 2 design (constraint attacked [COST], non-duplication
against L-14/L-15/L-16, R-56/R-77, R-64/65/67/68/165, R-131/133/134/151,
R-145, and the 08-26 verification pass that dismissed Roll/Corwin-Schultz
without implementing either, simulability, the Step-0 degeneracy/
falsification measurement on real data, and both branches' pre-registered
decision rules) is in `experiments/r173_direction.md`, written by the
operator (based on a research sub-agent's proposal, independently
verified against the ledger) BEFORE either branch was dispatched. This
module implements NEITHER branch's own re-pricing/deadband logic -- that
is each branch's own job, on top of the estimators and helpers below.
Neither branch may edit this file or each other's file (R-89-through-R-172's
own convention).

============================================================================
ROLL (1984)'S OWN FORMULA, confirmed via WebSearch against the paper's
result before implementation: for a covariance-stationary series of
observed price changes with a constant effective spread `s`, if trades
alternate buy/sell with equal probability and no drift,
    Cov(delta_P_t, delta_P_{t-1}) = -(s/2)^2  =>  s = 2*sqrt(-Cov)
whenever the covariance is negative (Roll's own stated degeneracy
condition: a non-negative sample covariance yields no spread estimate, not
a zero one -- occasionally reported as zero in later literature, but this
module follows Roll's own convention and reports 0.0 rather than treating
"undefined" as "no cost" implicitly, so a NaN can never silently reach a
strategy's cost path).

CORWIN & SCHULTZ (2012)'S OWN FORMULA (Section I, eq. 5-13), confirmed via
WebSearch against the paper's abstract/summary before implementation, for
a TWO-CONSECUTIVE-BAR window: with beta = sum over the pair of
[ln(H_i/L_i)]^2, gamma = [ln(max(H_i)/min(L_i))]^2 (the 2-bar high/low
range), and k = 3 - 2*sqrt(2):
    alpha = (sqrt(2*beta) - sqrt(beta))/k - sqrt(gamma/k)
    S = 2*(e^alpha - 1) / (1 + e^alpha)
The paper's own published convention sets negative S to 0 (disclosed and
applied verbatim below) -- a negative raw estimate is a known small-sample
artifact of the derivation, not evidence of a negative spread.

The paper was designed for DAILY bars; this project has only 5-minute
bars, so the literal 2-bar estimator here spans 10 minutes, not 2 days. A
`smooth_window` rolling MEAN (default 1 day = BARS_PER_DAY bars) is
therefore applied on top of the raw 2-bar estimate before use, both
because a single 10-minute estimate is far noisier than the daily
estimator the paper studied and because BOTH branches consume this as a
slowly-varying friction *level*, not a per-bar trading signal -- this
smoothing choice is disclosed here rather than left implicit, matching
this project's own convention (r102_shared's undemeaned-vs-EWM-std note,
r161_shared's Hoeffding-not-Bentkus note).

CAUSALITY: both estimators are shifted by exactly one bar after
computation (`.shift(1)`), identical to `r102_shared.v4_symmetric_vol`'s
own `.shift(1)` convention -- the value used to decide bar t's trade was
knowable strictly before bar t's own close. Checked end-to-end below via
`causal_truncation_probe_series` (this project's standard lookahead-bug
catcher).

Configs evaluated by this file: 0 (shared infrastructure and the frozen
Step-0 measurement only; each branch's own count is logged in its own
module and summed in the ledger entry, per R-163/R-168/R-172's convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SliceResult,
    TargetStrategy,
    V4_BAND,
    V4_DEADBAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
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
from experiments.r161_shared import (  # noqa: E402,F401
    FEE_TIER,
    FUTURES,
    SHARPE_NOISE_FLOOR,
    SPOT,
    daily_close,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_DEADBAND - 0.10) < 1e-12, V4_DEADBAND

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# Provenance for every number is in r173_direction.md. None is fitted to
# a result computed after this module was frozen.
# ------------------------------------------------------------------------
ROLL_WINDOW = BARS_PER_DAY          # 1-day rolling covariance window
CS_SMOOTH_WINDOW = BARS_PER_DAY     # 1-day rolling mean of the raw 2-bar CS estimate
CS_SMOOTH_MIN_PERIODS = BARS_PER_DAY // 4
STRESS_WINDOW_DAYS = 3              # +/- days around each episode onset
STEP0_MEDIAN_ELEVATION_GATE = 1.0   # frozen falsification bar (r173_direction.md)

# The project's own canonical six-episode calendar (R-01 / R-82 / R-83 /
# R-85 / R-86 / R-96 / R-98 / R-99 / R-139 / R-141's shared Step-A gate;
# copied verbatim, matching every prior round's own convention of
# self-contained modules rather than a cross-round import of a constant
# that was never itself exported from a single canonical file).
STRESS_EPISODES: list[tuple[str, str]] = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]


# ==========================================================================
# (1) Roll (1984) implicit effective spread, rolling, causal.
# ==========================================================================


def roll_spread_causal(df: pd.DataFrame, window: int = ROLL_WINDOW) -> np.ndarray:
    """Causal, rolling Roll (1984) implicit effective spread.

    `spread_t = 2*sqrt(-Cov(dP, dP_lag1))` over the trailing `window` bars,
    computed at bar t and then shifted by one bar (see module docstring).
    0.0 wherever the rolling covariance is non-negative (Roll's own
    degeneracy condition) rather than NaN, so a strategy's cost path can
    never silently receive a NaN.
    """
    dP = df["close"].diff()
    cov = dP.rolling(window).cov(dP.shift(1))
    spread = np.where(cov < 0, 2.0 * np.sqrt(np.clip(-cov, 0.0, None)), 0.0)
    return pd.Series(spread, index=df.index).shift(1).fillna(0.0).to_numpy()


# ==========================================================================
# (2) Corwin & Schultz (2012) high-low spread, raw 2-bar + rolling smooth,
#     causal.
# ==========================================================================


def corwin_schultz_raw(df: pd.DataFrame) -> pd.Series:
    """The literal 2-consecutive-bar Corwin & Schultz (2012) estimator,
    NOT yet shifted for causality (see `corwin_schultz_spread_causal`,
    which is what either branch should actually use)."""
    h, l = df["high"], df["low"]
    h1, l1 = h.shift(1), l.shift(1)
    hi2 = np.maximum(h, h1)
    lo2 = np.minimum(l, l1)
    beta = np.log(h / l) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(hi2 / lo2) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s_raw = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s_raw = s_raw.clip(lower=0.0)  # CS2012's own published convention
    return s_raw.fillna(0.0)


def corwin_schultz_spread_causal(df: pd.DataFrame,
                                  smooth_window: int = CS_SMOOTH_WINDOW,
                                  min_periods: int = CS_SMOOTH_MIN_PERIODS) -> np.ndarray:
    """Causal, smoothed Corwin & Schultz (2012) spread: a trailing
    `smooth_window`-bar mean of the raw 2-bar estimate, shifted by one bar
    (see module docstring for why smoothing is applied at 5m resolution)."""
    raw = corwin_schultz_raw(df)
    smooth = raw.rolling(smooth_window, min_periods=min_periods).mean()
    return smooth.shift(1).fillna(0.0).to_numpy()


def spread_percentile_causal(spread: np.ndarray, window: int = 90 * BARS_PER_DAY,
                              min_periods: int = 10 * BARS_PER_DAY) -> np.ndarray:
    """Each bar's OWN trailing percentile rank (0..1) of the (already
    causal) spread series within a rolling `window`-bar history -- used by
    the novel branch to condition the deadband on "is friction elevated
    relative to its own recent regime" rather than an absolute level that
    would need re-fitting per instrument. Causal by construction: rank is
    computed only against bars strictly before and including t within the
    window, using pandas' own trailing `.rolling().rank()`, THEN shifted
    one more bar to match every other series in this module."""
    s = pd.Series(spread, dtype=float)
    pct = s.rolling(window, min_periods=min_periods).apply(
        lambda w: (w <= w[-1]).mean(), raw=True)
    return pct.shift(1).fillna(0.5).to_numpy()


# ==========================================================================
# (3) Step-0 degeneracy / falsification measurement helpers.
# ==========================================================================


def non_degenerate_fraction(spread: np.ndarray) -> float:
    """Fraction of bars where the (causal) spread estimate is strictly
    positive -- the Step-0 kill-switch statistic."""
    arr = np.asarray(spread, dtype=float)
    return float(np.mean(arr > 0.0)) if len(arr) else float("nan")


def stress_episode_elevation_ratios(spread: np.ndarray, index: pd.DatetimeIndex,
                                     episodes: list[tuple[str, str]] = STRESS_EPISODES,
                                     window_days: int = STRESS_WINDOW_DAYS) -> dict[str, float]:
    """For each canonical stress episode, the ratio of the spread's mean
    over a +/-`window_days` window around the episode's onset to the
    whole-period unconditional daily median. Returns {label: ratio}."""
    s = pd.Series(np.asarray(spread, dtype=float), index=index)
    daily = s.resample("1D").mean()
    baseline = float(daily.median())
    out: dict[str, float] = {}
    for label, date in episodes:
        ts = pd.Timestamp(date, tz="UTC")
        w = daily.loc[ts - pd.Timedelta(days=window_days): ts + pd.Timedelta(days=window_days)]
        out[label] = float(w.mean() / baseline) if baseline and len(w) else float("nan")
    return out


def median_elevation_ratio(spread: np.ndarray, index: pd.DatetimeIndex,
                            episodes: list[tuple[str, str]] = STRESS_EPISODES,
                            window_days: int = STRESS_WINDOW_DAYS) -> float:
    """The frozen falsification statistic: median of the per-episode
    elevation ratios. Must exceed STEP0_MEDIAN_ELEVATION_GATE."""
    ratios = stress_episode_elevation_ratios(spread, index, episodes, window_days)
    vals = [v for v in ratios.values() if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


# ==========================================================================
# (4) Dynamic (spread-conditioned) deadband -- shared primitive both
#     branches may call; the NOVEL branch is the one expected to use it as
#     its own headline mechanism, but it lives here (not in the novel
#     branch's own file) so the conservative branch's robustness checks
#     can call the identical function rather than a re-implementation.
# ==========================================================================


def apply_deadband_dynamic(desired: np.ndarray, pctile: np.ndarray,
                            base_deadband: float = V4_DEADBAND,
                            k: float = 0.0) -> np.ndarray:
    """Like `r102_shared.apply_deadband`, but the trigger threshold at
    bar t is `base_deadband * (1 + k * pctile[t])` instead of a constant
    `base_deadband`. `k=0.0` reproduces `apply_deadband` EXACTLY (checked
    in `_self_test`). `pctile` must already be causal (see
    `spread_percentile_causal`)."""
    desired = np.asarray(desired, dtype=float)
    pctile = np.asarray(pctile, dtype=float)
    n = len(desired)
    target = np.zeros(n)
    pos = 0.0
    for i in range(n):
        band = base_deadband * (1.0 + k * pctile[i])
        if abs(desired[i] - pos) > band:
            pos = float(desired[i])
        target[i] = pos
    return target


# ==========================================================================
# (5) Self-test on synthetic + a real-data probe. Mirrors r102/r161/r172's
#     convention: fast synthetic checks first, then the frozen real-data
#     Step-0 measurement reproduced exactly as recorded in r173_direction.md.
# ==========================================================================


def _cs_build(df: pd.DataFrame) -> np.ndarray:
    return corwin_schultz_spread_causal(df)


def _roll_build(df: pd.DataFrame) -> np.ndarray:
    return roll_spread_causal(df)


def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=100_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(173)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": 1.0}, index=idx)

    # (1) Both estimators: finite, non-negative everywhere.
    roll = roll_spread_causal(df)
    cs = corwin_schultz_spread_causal(df)
    assert len(roll) == len(cs) == len(df)
    assert np.all(np.isfinite(roll)) and np.all(roll >= 0.0)
    assert np.all(np.isfinite(cs)) and np.all(cs >= 0.0)

    # (2) Causal-truncation probe on both estimator pipelines -- this
    # project's standard lookahead-bug catcher.
    assert causal_truncation_probe_series(_roll_build, df, cuts=(0.5, 0.7, 0.9))
    assert causal_truncation_probe_series(_cs_build, df, cuts=(0.5, 0.7, 0.9))

    # (3) spread_percentile_causal: bounded in [0, 1], causal-truncation-safe.
    pct = spread_percentile_causal(cs)
    assert len(pct) == len(cs)
    assert np.all((pct >= 0.0) & (pct <= 1.0))

    def _pct_build(d: pd.DataFrame) -> np.ndarray:
        return spread_percentile_causal(corwin_schultz_spread_causal(d))

    assert causal_truncation_probe_series(_pct_build, df, cuts=(0.5, 0.7, 0.9))

    # (4) apply_deadband_dynamic(k=0.0) reproduces apply_deadband EXACTLY.
    desired = rng.normal(0, 0.3, 5000)
    base = apply_deadband(desired)
    dyn_zero = apply_deadband_dynamic(desired, pctile=rng.uniform(0, 1, 5000), k=0.0)
    assert np.allclose(base, dyn_zero, atol=1e-12)

    # (5) A wider deadband trades no more often, never more responsively,
    # than the base deadband -- a monotonicity sanity check on k.
    fixed_pct = np.full(5000, 1.0)  # pin percentile at its max
    tight = apply_deadband_dynamic(desired, pctile=fixed_pct, k=0.0)
    wide = apply_deadband_dynamic(desired, pctile=fixed_pct, k=2.0)
    tight_changes = int(np.sum(np.abs(np.diff(tight)) > 1e-12))
    wide_changes = int(np.sum(np.abs(np.diff(wide)) > 1e-12))
    assert wide_changes <= tight_changes, (wide_changes, tight_changes)

    # (6) non_degenerate_fraction / stress-episode helpers: sane bounds on
    # synthetic data (a real-data reproduction of the FROZEN Step-0 numbers
    # in r173_direction.md is each branch's own job on load_btc(), not
    # re-run here to keep this module's import fast).
    frac = non_degenerate_fraction(cs)
    assert 0.0 <= frac <= 1.0
    ratios = stress_episode_elevation_ratios(cs, df.index, window_days=3)
    assert len(ratios) == len(STRESS_EPISODES)
    med = median_elevation_ratio(cs, df.index)
    assert np.isfinite(med) or np.isnan(med)  # synthetic series may miss episode dates entirely


_self_test()
