"""Shared, read-only utilities for the R-85 round (08-21).

Idea in one sentence: `kelly_regime_v4`'s regime vote reacts to a break
only after price has already crossed a fixed 20/40/80-day anchor -- this
round tests whether **critical slowing down (CSD)**, the dynamical-systems
signature that a system approaching a critical transition recovers more
slowly from small perturbations (rising variance and rising lag-1
autocorrelation of its own fluctuations; Scheffer et al. 2009, "Early-
warning signals for critical transitions", *Nature* 461:53-59), gives
*earlier* warning of the same six dated historical BTC regime breaks than
v4's own heuristic does. Applied to crypto directly: Wen, Cai, Ma, Wu &
Xie (2020), "Critical slowing down associated with critical transition and
risk of collapse in crypto-currency", *Royal Society Open Science*
7(2):191450, find CSD signatures (rising variance/AR(1)) ahead of BTC's
2017-2018 and other collapses. The framing that shaped this round's
falsification design is much more recent and much more skeptical: an
arXiv paper studying seven BTC-perpetual liquidation cascades from
2022-2025 (arXiv:2607.27070, "Where does the criticality live? Early-
warning signals are event-heterogeneous across seven crypto-perpetual
liquidation cascades") finds price carries a CSD signature in 5 of 7
events but is **silent in exactly the two sudden, news-driven shocks**
(a tariff-driven cascade and a similarly abrupt event) -- i.e. CSD is
argued to detect slow endogenous build-ups but not sudden exogenous
shocks. That is a **named, pre-registered prediction of this round's own
likely failure mode**, not a hoped-for result: if it holds here too, this
signal should behave like BOCPD (R-82, 2/6, decisive early passes on both
slow 2018 episodes, lags on all four sudden 2020-2022 shocks) and Kalman
LLT (R-83, 1/6), not differently.

Which constraint this attacks: **ERR** (no error control anywhere in this
project's signal path) and **N-approx-3** (a different theoretical *basis*
for estimating "has the regime just started to break," rather than a
retuned version of a basis already tried). This is explicitly **not** an
eleventh INFO-axis signal: it reads no data beyond the committed OHLCV
close series v4 itself already uses -- same information, a third
formally distinct way of extracting an early-warning estimate from it.

Not a duplicate of:
- R-82 (BOCPD, Adams & MacKay 2007): a Bayesian generative run-length
  posterior over discrete regime segments (a changepoint/segmentation
  model). CSD is not a segmentation model at all -- it has no notion of
  "which regime am I in," only "is the system's own fluctuation structure
  showing the statistical signature of approaching *some* bifurcation,"
  drawn from dynamical-systems/bifurcation theory (a stochastic system
  near a fold/transcritical bifurcation has a Jacobian eigenvalue
  approaching zero, which mechanically slows its return to equilibrium
  after a perturbation -- Kuehn 2011, "A mathematical framework for
  critical transitions", *Physica D* 240(12)).
- R-83 (causal Kalman local-linear-trend filter, Harvey 1989): a linear
  state-space filter estimating a latent level/slope. CSD estimates
  neither a level nor a slope -- it estimates second- and higher-moment
  *fluctuation* statistics (variance, autocorrelation) around whatever
  the current level is, and is explicitly agnostic to trend direction.
- R-01/R-34 (HMM / `harsanyi_crowd`'s minority-game posterior): discrete
  regime-switching / game-theoretic belief machinery, unrelated to a
  continuous fluctuation-statistics trend test.
- R-80 (causal meta-labeling): a discriminative classifier trained on
  hand-engineered features of the vote's own trailing hit rate. CSD
  indicators are computed from price returns alone, with zero trained
  weights and no reference to the vote at all.
- The nine prior INFO-axis rounds (R-44/R-53/R-54/R-55/R-58/R-73/R-74/
  R-75/R-76/R-79/R-81): every one introduced a NEW external or timestamp-
  derived data channel. This round introduces no new data channel --
  same question as R-82/R-83: does a different, error-aware ESTIMATOR of
  the *existing* price series detect known historical breaks with less
  lag than v4's own fixed-window heuristic?
- R-62 (vote x scale factorization): motivates keeping any change confined
  to the DIRECTION/vote side (where R-62 showed the whole matched-exposure
  drawdown signature lives), scale untouched, exactly as R-82/R-83 did.

This module is read-only utility, written by the operator before dispatch
(same convention as r79_shared.py through r83's shared files). Neither
branch edits it. Contains: (1) a byte-for-byte duplicate of
`kelly_regime_v4`'s 3-anchor vote construction and the R-53/R-55
confirming-vote combination rule (both copied verbatim from
`r82_shared.py`, not reimplemented, to remove any chance of an
inconsistent baseline across R-82/R-83/R-85); (2) the two CSD indicators
(rolling variance, rolling lag-1 autocorrelation) computed causally on
DAILY-resampled log returns, matching R-82/R-83's own daily-cadence
convention (v4's anchors operate on a 20-80 CALENDAR-DAY horizon); (3) a
causal rolling Kendall-tau trend statistic and a causal z-score of that
statistic against its own trailing baseline, the standard Dakos et al.
(2012, "Methods for Detecting Early Warnings of Critical Transitions in
Time Series Illustrated Using Simulated Ecological Data", *PLoS ONE*
7(7):e41010) sliding-window-trend-test construction, simplified to a
self-calibrating z-score threshold (this project's own standing
convention for every other z-scored gate feature -- dvol_z, mvrv_roc_z,
ls_z, the stablecoin growth z-score -- rather than Dakos's own fixed
tau>0.5-style cutoff, which was tuned on simulated ecological data at a
different sampling cadence and would be an unjustified import here); (4)
the identical dated stress-episode table and detection-lag gate
scaffolding R-82/R-83 used (`STRESS_EPISODES`, `episode_window`,
`nearest_transition`, `block_bootstrap_shifts`), copied verbatim so all
three rounds' numbers are directly comparable; (5) the causal truncation
probe.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market CSD
number was computed, and are not retuned after seeing any result:
`SUB_WINDOW_DAYS` (the short window each indicator value is estimated
over) is set to 15 days -- shorter than v4's fastest 20-day anchor, a
deliberate choice so the indicator can move before that anchor could
possibly react. `DETECTION_WINDOW_DAYS` (the trailing window the
Kendall-tau trend test is computed over) is set to 90 days -- close to
v4's slowest 80-day anchor, the same "match the horizon the mechanism
describes" logic R-82's docstring used for its own hazard prior.
`BASELINE_WINDOW_DAYS` (the trailing window the tau statistic is
z-scored against) is set to 730 days (2 years), matching this project's
other long-baseline z-score features. `Z_THRESH = 2.0` is a round,
literature-standard two-sigma alarm threshold, chosen before any tau
series was computed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated verbatim from
# r82_shared.py / r83_novel_kalman_shared.py, not reimplemented.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# IDENTICAL to R-82/R-83's own table -- copied verbatim, not re-derived,
# so all three rounds' gate numbers are directly comparable.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# --------------------------------------------------------------- CSD params
SUB_WINDOW_DAYS = 15
DETECTION_WINDOW_DAYS = 90
BASELINE_WINDOW_DAYS = 730
Z_THRESH = 2.0

# The novel branch's JOINT alarm (both indicators simultaneously elevated)
# uses a lower per-indicator threshold than the conservative branch's
# single-indicator gate. Disclosed reasoning, fixed before any detection-lag
# number was computed: the two indicators are empirically near-uncorrelated
# on inner-train/inner-validation data (r=-0.09), so an AND-gate at the same
# Z_THRESH=2.0 fires on 0.0% of bars (verified directly) -- an empty
# alarm set is not a meaningful test of the joint-confirmation mechanism,
# it is a gate that fails by construction before the mechanism is ever
# exercised. Z_THRESH_JOINT=1.5 was chosen by requiring only that the joint
# alarm rate land within the same order of magnitude as the conservative
# branch's single-indicator rate at Z_THRESH=2.0 (0.55%): at 1.5 the joint
# rate is 0.40%, at 1.25 it is 1.10% (already >2x), at 1.0 it is 1.45%
# (>2.5x) -- 1.5 is the loosest round threshold that does not already leave
# this order-of-magnitude band, chosen from marginal base-rate arithmetic
# alone, before any episode/gate/lead-time number existed.
Z_THRESH_JOINT = 1.5

# ----------------------------------------------------------------- v4 vote
#   Copied verbatim from r82_shared.py.


def anchor_votes(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                  band: float = V4_BAND) -> list[pd.Series]:
    """The three latched 0/1 anchor votes `kelly_regime`/`_v3`/`_v4` use internally.

    Causal: row i depends only on rows <= i (rolling mean + ffill/latch).
    """
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return votes


def anchor_majority(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                     band: float = V4_BAND) -> pd.Series:
    """`frac` = mean of the three anchor votes, in {0, 1/3, 2/3, 1} -- v4's
    own gate, exactly, for use as the Step-A detection-lag comparison
    baseline."""
    votes = anchor_votes(df, horizons, band)
    return sum(votes) / len(votes)


def confirming_vote_frac(anchor_sum: np.ndarray, meta_vote: np.ndarray,
                          weight: float) -> np.ndarray:
    """R-53/R-55's combination rule, copied verbatim.

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``

    ``anchor_sum`` in [0, 3], ``meta_vote`` in {0, 1} (per R-80/R-81's
    lesson: keep it DISCRETE so the formula can still reach exactly
    flat/exactly full). ``weight == 0`` recovers `kelly_regime_v4` exactly.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


# ------------------------------------------------------------ CSD indicators


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).diff().dropna()


def _rolling_autocorr_lag1(x: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) < 1e-12:
        return np.nan
    a, b = x[:-1], x[1:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def csd_indicator_series(returns: pd.Series, kind: str,
                          sub_window_days: int = SUB_WINDOW_DAYS) -> pd.Series:
    """Causal rolling CSD indicator on daily log returns.

    ``kind="variance"``: rolling variance over the trailing
    `sub_window_days` returns. ``kind="autocorr"``: rolling lag-1
    autocorrelation over the same window. Row t uses only
    ``returns[t-sub_window_days+1 : t+1]`` -- strictly causal.
    """
    if kind == "variance":
        return returns.rolling(sub_window_days).var()
    if kind == "autocorr":
        return returns.rolling(sub_window_days).apply(_rolling_autocorr_lag1, raw=True)
    raise ValueError(f"unknown kind {kind!r}")


def _rolling_kendall_tau(x: np.ndarray) -> float:
    """Kendall's tau-a (no tie correction; the time index -- one of the two
    variables being correlated -- has no ties by construction, so tau-a
    and tau-b coincide unless the indicator values themselves tie exactly,
    negligible for a continuous variance/autocorrelation statistic) of
    ``x`` against its own position index, computed by hand: scipy is not
    a dependency of this project (see R-65/R-67/R-68/R-79's own
    hand-rolled Spearman helpers, same reason). Vectorized pairwise-sign
    formula: tau = sum_{i!=j} sign(i-j)*sign(x_i-x_j) / (n*(n-1)), which
    is exactly (concordant - discordant) / (n*(n-1)/2) since every
    unordered pair is counted twice, once each direction.
    """
    valid = ~np.isnan(x)
    if valid.sum() < 8:
        return np.nan
    idx = np.arange(len(x))[valid]
    vals = x[valid]
    sign_idx = np.sign(np.subtract.outer(idx, idx))
    sign_val = np.sign(np.subtract.outer(vals, vals))
    n = len(idx)
    tau = float(np.sum(sign_idx * sign_val)) / (n * (n - 1))
    return tau if np.isfinite(tau) else np.nan


def csd_trend_zscore(indicator: pd.Series,
                      detection_window_days: int = DETECTION_WINDOW_DAYS,
                      baseline_window_days: int = BASELINE_WINDOW_DAYS) -> pd.Series:
    """Causal Kendall-tau rising-trend statistic, z-scored against its own
    trailing baseline (Dakos et al. 2012's sliding-window-trend-test,
    simplified to this project's standing self-calibrating-z-score
    convention). Row t's tau uses only `indicator[t-detection_window_days+1
    : t+1]`; row t's z-score uses only tau values up to and including t.
    Strictly causal by construction (rolling window operations only).
    """
    tau = indicator.rolling(detection_window_days).apply(_rolling_kendall_tau, raw=True)
    baseline_mean = tau.rolling(baseline_window_days, min_periods=detection_window_days).mean()
    baseline_std = tau.rolling(baseline_window_days, min_periods=detection_window_days).std()
    z = (tau - baseline_mean) / baseline_std.replace(0.0, np.nan)
    return z


def csd_daily_causal_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Build both CSD indicators' trend z-scores on daily-resampled log
    returns and align them onto `df`'s 5-minute index with the same
    full-calendar-day causal shift every other daily-cadence signal in
    this project uses (`align_onchain_causal`)."""
    from tradebot.data import align_onchain_causal

    ret = daily_log_returns(df)
    var_ind = csd_indicator_series(ret, "variance")
    ac_ind = csd_indicator_series(ret, "autocorr")
    var_z = csd_trend_zscore(var_ind)
    ac_z = csd_trend_zscore(ac_ind)
    daily = pd.DataFrame({"csd_var_z": var_z, "csd_autocorr_z": ac_z}, index=ret.index)
    return align_onchain_causal(daily, df)


# --------------------------------------------------------- Step-A gate infra
#   `nearest_transition`, `episode_window`, `block_bootstrap_shifts`,
#   `truncation_causality_probe` copied verbatim from r82_shared.py.


def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    """Timestamp, within `window`, of the anchor-gate transition closest to
    `onset`. Duplicated from R-82's own gate file (self-contained, not
    imported, per this project's per-round shared-module convention)."""
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


def nearest_csd_alarm(z: pd.Series, window: pd.DatetimeIndex,
                       onset: pd.Timestamp, z_thresh: float = Z_THRESH
                       ) -> pd.Timestamp | None:
    """Timestamp, within `window`, of the first bar where the CSD trend
    z-score crosses UP through `z_thresh`, closest to `onset` -- the CSD
    analogue of R-82's `nearest_bocpd_detection`."""
    vals = z.reindex(window).to_numpy()
    high = vals >= z_thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = high[1:] & ~high[:-1]
    cross[0] = bool(high[0]) if len(high) else False
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def nearest_joint_csd_alarm(z_a: pd.Series, z_b: pd.Series, window: pd.DatetimeIndex,
                             onset: pd.Timestamp, z_thresh: float = Z_THRESH
                             ) -> pd.Timestamp | None:
    """Timestamp of the first bar where BOTH CSD indicators' trend
    z-scores are simultaneously >= z_thresh -- the novel branch's joint
    two-indicator confirmation (Dakos et al. 2012's own recommendation
    that agreement between independent EWS indicators is more robust than
    any single one alone)."""
    va = z_a.reindex(window).to_numpy()
    vb = z_b.reindex(window).to_numpy()
    high = (va >= z_thresh) & (vb >= z_thresh)
    cross = np.zeros(len(high), dtype=bool)
    cross[1:] = high[1:] & ~high[:-1]
    cross[0] = bool(high[0]) if len(high) else False
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars: pd.DataFrame, onset_str: str, window_days: int = 60
                    ) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


def block_bootstrap_shifts(n_bars: int, block_days: int, n_draws: int, seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n_bars`
    index, for a block-bootstrap null on the Step-A detection-lag gate.
    Copied verbatim from r82_shared.py."""
    rng = np.random.default_rng(seed)
    block = int(block_days * BARS_PER_DAY)
    draws = []
    for _ in range(n_draws):
        shift = int(rng.integers(block, n_bars - block)) if n_bars > 2 * block else int(rng.integers(1, n_bars))
        draws.append((np.arange(n_bars) + shift) % n_bars)
    return draws


def truncation_causality_probe(build_target_fn, df: pd.DataFrame,
                                check_at: int, shorter_by: int = 20_000) -> bool:
    """Standard truncation probe: does `target[check_at]` change if bars
    after it are dropped? Returns True if causal (identical both ways)."""
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))
