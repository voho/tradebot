"""Shared, read-only utilities for the R-86 round (08-21).

Idea in one sentence: transfer entropy (Schreiber 2000, "Measuring
Information Transfer", *Phys. Rev. Lett.* 85(2):461) -- a model-free,
non-parametric measure of DIRECTED information flow between two time
series -- gives a fifth, structurally distinct theoretical basis for
early-warning / regime-timing detection, tested against the identical
six dated historical BTC regime transitions R-82 (BOCPD), R-83 (Kalman
LLT) and R-85 (critical slowing down) used. Applied to crypto directly:
Garcia-Medina & Hernandez C. (2020), "Network Analysis of Multivariate
Transfer Entropy of Cryptocurrencies in Times of Turbulence", *Entropy*
22(7):760, show that the total (network) transfer entropy among a panel
of cryptocurrencies rises sharply around the March 2020 COVID crash --
information-flow complexity increases as a market approaches or enters
turbulence, mechanistically distinct from a rising-variance/autocorrelation
fluctuation statistic (CSD, R-85), a generative changepoint posterior
(BOCPD, R-82) or a linear state-space filter (Kalman LLT, R-83).

Which constraint this attacks: **ERR** (no error control anywhere in this
project's signal path) and **N-approx-3** (a fifth theoretical *basis* for
"has the regime just started to break", not a retuned version of a basis
already tried). The CONSERVATIVE branch is explicitly **not** an eleventh
INFO-axis signal: it reads no data beyond the committed OHLCV close/volume
series v4 itself already uses. The NOVEL branch reads one already-committed
external series (Coinbase ETH-USD spot, `load_coinbase_eth_spot`, already
used by R-47/R-76/R-57 -- no new fetch, no new coverage-gap risk).

Not a duplicate of:
- R-82 (BOCPD): a Bayesian generative run-length posterior over discrete
  regime segments. TE has no notion of "which regime am I in" and no
  generative model at all -- it is a model-free functional of the joint
  empirical distribution of two series.
- R-83 (causal Kalman LLT): a linear-Gaussian state-space filter
  estimating a latent level/slope. TE makes no linearity or Gaussianity
  assumption and estimates directed *information flow* between two
  series, not a latent trend of one.
- R-85 (CSD): a univariate fluctuation statistic (variance,
  autocorrelation) of ONE series' own returns. TE is fundamentally
  BIVARIATE -- it measures whether one series' history reduces
  uncertainty about another series' future beyond what that second
  series' own history already explains, which no prior mechanism in this
  ledger has tested in any form.
- R-01/R-34 (HMM / harsanyi_crowd): discrete regime-switching /
  game-theoretic belief machinery, unrelated to an information-theoretic
  directed-flow estimator.
- R-80 (causal meta-labeling): a trained discriminative classifier on the
  vote's own trailing hit rate. TE has zero trained weights and no
  reference to the vote at all -- purely a function of price/volume (or
  price/price) history.
- The ten prior INFO-axis rounds (R-44/R-53/R-54/R-55/R-58/R-73/R-74/
  R-75/R-79/R-81/R-84): every one introduced a NEW external or
  timestamp-derived data channel used as a directional vote or brake. The
  CONSERVATIVE branch here introduces no new data channel at all (same
  posture as R-82/R-83/R-85); the NOVEL branch reuses an already-committed
  series (ETH spot) in a structurally new way -- as the second leg of a
  bivariate information-flow estimator, not as a directional signal or a
  cointegration/distance-method pairs trade (R-76, which tested price
  co-movement, not information flow).
- R-62 (vote x scale factorization): motivates keeping any change confined
  to the DIRECTION/vote side (where R-62 showed the whole matched-exposure
  drawdown signature lives), scale untouched, exactly as R-82/R-83/R-85 did.

This module is read-only utility, written by the operator before dispatch
(same convention as r82_shared.py through r85_shared.py). Neither branch
edits it. Contains: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s
3-anchor vote construction and the R-53/R-55 confirming-vote combination
rule (copied verbatim from `r85_shared.py`, itself copied verbatim from
`r82_shared.py`, not reimplemented, so all four rounds' baselines are
identical); (2) a dependency-free (numpy only -- scipy/sklearn are not
project dependencies, same reason R-65/R-67/R-68/R-79/R-85 hand-rolled
their own statistics) discretized transfer-entropy estimator, lag-1,
quantile-binned per window (fully causal: bin edges are computed only
from the trailing window itself, never from future data); (3) the
IDENTICAL csd-style causal rolling-trend-z-score construction R-85 used
(a causal Kendall-tau trend statistic of the raw indicator, z-scored
against its own trailing baseline), applied here to the TE indicator
series instead of variance/autocorrelation -- reusing R-85's own
z-score scaffold rather than inventing a new one, to minimize the surface
for a new statistical bug and keep the alarm-threshold semantics
identical across all four mechanism rounds; (4) the identical dated
stress-episode table and detection-lag gate scaffolding R-82/R-83/R-85
used (`STRESS_EPISODES`, `episode_window`, `nearest_transition`,
`block_bootstrap_shifts`), copied verbatim so all four rounds' numbers
are directly comparable; (5) the causal truncation probe.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market TE number
was computed, and are not retuned after seeing any result.

`N_BINS = 3`: the standard low bin-count used in short-sample transfer
entropy estimation (3 discretized states -- low/mid/high -- keeps the
joint alphabet at `N_BINS**3 = 27` cells manageable at daily cadence).
`TE_SUB_WINDOW_DAYS = 30`: the trailing window each single TE value is
estimated over. Chosen, before any real number was computed, as
`ceil(N_BINS**3 / N_BINS) = 9` samples per joint (y_t, y_lag) cell at
minimum and roughly one expected sample per full joint cell on average
(30 obs / 27 cells ~= 1.1) -- thin, but this project's own CSD precedent
(a 15-day window for a 2-moment statistic) already accepted comparably
thin windows, and TE's higher data requirement relative to a variance/
autocorrelation statistic is compensated by doubling that window rather
than by adding a tunable choice. `DETECTION_WINDOW_DAYS = 90` and
`BASELINE_WINDOW_DAYS = 730` are copied verbatim from R-85 -- close to
v4's slowest 80-day anchor, and a 2-year trend baseline, respectively,
the same "match the horizon the mechanism describes" logic R-82's hazard
prior and R-85's baseline both used. `Z_THRESH = 2.0` is the same round,
literature-standard two-sigma alarm threshold R-82/R-83/R-85 all used,
chosen before any TE series was computed.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
the same pattern that has now beaten four consecutive mechanisms built on
four different theoretical bases -- an estimator computed FROM price (and,
for the novel branch, volume or a second price series) fluctuations can
only rise once those fluctuations have already become unusual, which is
exactly the moment v4's own fixed-window anchor is also starting to react.
If TE also lags every sudden 2020-2022 shock and only leads the slow 2018
build-up (or fails to lead anything at all, as R-84's conservative volume
branch did), that is the fifth independent mechanism converging on the
same conclusion the ledger's standing diagnosis is already leaning toward:
this six-episode gate is unwinnable by any estimator computed from this
project's own committed price/volume history, sudden or not, and the
finding is about the gate/dataset rather than about any one technique.
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
# r85_shared.py / r82_shared.py, not reimplemented.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# IDENTICAL to R-82/R-83/R-85's own table -- copied verbatim, not
# re-derived, so all four rounds' gate numbers are directly comparable.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ------------------------------------------------------------------ TE params
N_BINS = 3
TE_SUB_WINDOW_DAYS = 30
DETECTION_WINDOW_DAYS = 90
BASELINE_WINDOW_DAYS = 730
Z_THRESH = 2.0

# ----------------------------------------------------------------- v4 vote
#   Copied verbatim from r85_shared.py / r82_shared.py.


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
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


# --------------------------------------------------------- transfer entropy


def _digitize_window(x: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Discretize `x` into `n_bins` quantile bins, edges computed ONLY from
    `x` itself (the trailing window passed in) -- never from future data,
    so a rolling caller stays causal by construction."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.zeros(len(x), dtype=int)
    inner = edges[1:-1]
    bins = np.digitize(x, inner, right=True)
    return np.clip(bins, 0, n_bins - 1)


def transfer_entropy(x: np.ndarray, y: np.ndarray, n_bins: int = N_BINS,
                      eps: float = 1e-9) -> float:
    """Discretized, lag-1 transfer entropy TE_{X->Y} (Schreiber 2000), in
    bits: how much `x`'s immediate past reduces uncertainty about `y`'s
    present, beyond what `y`'s own immediate past already explains.

    ``TE = sum p(y_t, y_lag, x_lag) * log2[ p(y_t|y_lag,x_lag) / p(y_t|y_lag) ]``

    `x` and `y` must be equal-length 1-D arrays (same trailing window,
    same timestamps). Laplace-smoothed (`eps` added to every joint count)
    to avoid zero-count log singularities in a thin window -- this is a
    biased estimator (all short-window TE estimators are; see Schreiber
    2000's own discussion), which is why it is used only as the raw input
    to a self-calibrating trend z-score (`te_trend_zscore` below) rather
    than read as an absolute level.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 8 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    y_t = y[1:]
    y_lag = y[:-1]
    x_lag = x[:-1]
    by_t = _digitize_window(y_t, n_bins)
    by_lag = _digitize_window(y_lag, n_bins)
    bx_lag = _digitize_window(x_lag, n_bins)

    idx_xyz = by_t * n_bins * n_bins + by_lag * n_bins + bx_lag
    counts_xyz = np.bincount(idx_xyz, minlength=n_bins ** 3).reshape(n_bins, n_bins, n_bins).astype(float)
    counts_xyz += eps

    p_xyz = counts_xyz / counts_xyz.sum()
    p_yz = p_xyz.sum(axis=0)              # p(y_lag, x_lag)
    p_y_ylag = p_xyz.sum(axis=2)          # p(y_t, y_lag)
    p_ylag = p_y_ylag.sum(axis=0)         # p(y_lag)

    te = 0.0
    for a in range(n_bins):       # y_t
        for b in range(n_bins):   # y_lag
            for c in range(n_bins):  # x_lag
                num = p_xyz[a, b, c] * p_ylag[b]
                den = p_yz[b, c] * p_y_ylag[a, b]
                te += p_xyz[a, b, c] * np.log2(num / den)
    return float(te)


def rolling_transfer_entropy(x: pd.Series, y: pd.Series,
                              sub_window_days: int = TE_SUB_WINDOW_DAYS,
                              n_bins: int = N_BINS) -> pd.Series:
    """Causal rolling TE_{X->Y}: row t uses only `x[t-sub_window_days+1:t+1]`
    and `y[t-sub_window_days+1:t+1]` -- strictly causal, matching
    `csd_indicator_series`'s rolling convention."""
    both = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    out = np.full(len(both), np.nan)
    xv = both["x"].to_numpy()
    yv = both["y"].to_numpy()
    for i in range(sub_window_days - 1, len(both)):
        lo = i - sub_window_days + 1
        out[i] = transfer_entropy(xv[lo:i + 1], yv[lo:i + 1], n_bins=n_bins)
    return pd.Series(out, index=both.index)


def _rolling_kendall_tau(x: np.ndarray) -> float:
    """Copied verbatim from r85_shared.py -- causal Kendall's tau-a of `x`
    against its own position index, hand-rolled (scipy is not a project
    dependency)."""
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


def trend_zscore(indicator: pd.Series,
                  detection_window_days: int = DETECTION_WINDOW_DAYS,
                  baseline_window_days: int = BASELINE_WINDOW_DAYS) -> pd.Series:
    """Causal Kendall-tau rising-trend statistic, z-scored against its own
    trailing baseline -- IDENTICAL construction to r85_shared.csd_trend_zscore,
    generalized to any indicator series (here: rolling TE instead of
    variance/autocorrelation). Row t's tau uses only
    `indicator[t-detection_window_days+1:t+1]`; row t's z-score uses only
    tau values up to and including t. Strictly causal by construction."""
    tau = indicator.rolling(detection_window_days).apply(_rolling_kendall_tau, raw=True)
    baseline_mean = tau.rolling(baseline_window_days, min_periods=detection_window_days).mean()
    baseline_std = tau.rolling(baseline_window_days, min_periods=detection_window_days).std()
    z = (tau - baseline_mean) / baseline_std.replace(0.0, np.nan)
    return z


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).diff().dropna()


def daily_log_volume_change(df: pd.DataFrame) -> pd.Series:
    daily_vol = df["volume"].resample("1D").sum()
    daily_vol = daily_vol.replace(0.0, np.nan)
    return np.log(daily_vol).diff().dropna()


def align_daily_causal(daily: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex a daily-cadence signal onto `bars`' 5-minute index, causally
    -- IDENTICAL shift convention to `tradebot.data.align_onchain_causal`
    (a bar at time T may only see the row for the most recent day that
    closed strictly before T's own day)."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


# --------------------------------------------------------- Step-A gate infra
#   `nearest_transition`, `episode_window`, `block_bootstrap_shifts`,
#   `truncation_causality_probe` copied verbatim from r85_shared.py /
#   r82_shared.py.


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


def nearest_te_alarm(z: pd.Series, window: pd.DatetimeIndex,
                      onset: pd.Timestamp, z_thresh: float = Z_THRESH
                      ) -> pd.Timestamp | None:
    """Timestamp of the first bar where the TE trend z-score crosses UP
    through `z_thresh`, closest to `onset` -- the TE analogue of
    R-85's `nearest_csd_alarm`."""
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


def episode_window(bars: pd.DataFrame, onset_str: str, window_days: int = 60
                    ) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


def block_bootstrap_shifts(n_bars: int, block_days: int, n_draws: int, seed: int) -> list[np.ndarray]:
    """Copied verbatim from r85_shared.py / r82_shared.py."""
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
