"""Shared, read-only utilities for the R-99 round (08-23).

Idea in one sentence: a causal, rolling Generalized Hurst Exponent (GHE) of
BTC's own daily log-price series -- a scaling-law/self-similarity measure of
whether recent price moves are persistent (trending, H>0.5), anti-persistent
(mean-reverting, H<0.5) or a random walk (H=0.5) -- gives a NINTH structurally
distinct theoretical basis for regime-timing (the conservative branch's alarm
role, tested against the identical six-episode gate R-82/83/84/85/86/96/98
all used), and, separately, a genuinely new COMBINATION role for the novel
branch: using the same estimator to widen or narrow `kelly_regime_v4`'s
rebalance no-trade band (the COST axis, not another regime alarm) on the
theory that a locally rough/anti-persistent (H<0.5) market has a worse
short-horizon signal-to-noise ratio, so a rebalance triggered inside it is
more likely to be reacting to noise than to real drift.

Literature grounding, both fetched and read before being relied on:
- Hurst, H. E. (1951), "Long-term storage capacity of reservoirs",
  *Transactions of the American Society of Civil Engineers* 116, 770-808 --
  the original rescaled-range statistic and its scaling exponent H, the
  foundational definition of what "persistent" vs. "anti-persistent" means
  for a time series' own self-similarity.
- Mandelbrot, B. B., & Van Ness, J. W. (1968), "Fractional Brownian
  Motions, Fractional Noises and Applications", *SIAM Review* 10(4),
  422-437 -- the formal generative model (fBm) a Hurst exponent
  characterizes: H=0.5 is ordinary Brownian motion (no memory), H>0.5 is
  positively autocorrelated (trending) noise, H<0.5 is negatively
  autocorrelated (mean-reverting) noise. A fundamentally different
  mathematical object from a hidden discrete state (HMM/R-01), a Bayesian
  run-length posterior (BOCPD/R-82), a linear-Gaussian state-space filter
  (Kalman LLT/R-83), the vote's own latch dynamics (R-84), a
  fluctuation-statistics trend test on the LEVEL of variance/autocorrelation
  (CSD/R-85), an information-theoretic directed-flow functional (transfer
  entropy/R-86), a conditional event-rate/self-excitation model
  (Hawkes/R-96), or an asymptotic tail-exceedance theorem (GPD/POT/R-98) --
  it is a SCALING EXPONENT describing how the variance of price increments
  grows with the aggregation horizon, not a level, a state, a rate, or a
  tail shape. (Distinction from CSD/R-85 specifically, since both use
  "variance": CSD asks whether the LEVEL of variance is rising over
  calendar time; GHE asks how variance SCALES across aggregation lags
  within one fixed window -- the two are computed from different axes of
  the data and can move independently.)
- Barabasi, A.-L., & Vicsek, T. (1991), "Multifractality of self-affine
  fractals", *Physical Review A* 44(4), 2730-2733 -- the generalized Hurst
  exponent (GHE) estimator used here: H(q) from the scaling of the
  q-th-order structure function `K(q,tau) = <|X(t+tau)-X(t)|^q>` against
  lag `tau` on log-log axes. q=1 is used below (the standard, least
  fat-tail-sensitive choice; Di Matteo 2007's own recommended default for
  financial series).
- Di Matteo, T. (2007), "Multi-scaling in Finance", *Quantitative Finance*
  7(1), 21-36 -- established the GHE(q=1) methodology as a standard
  market-development/regime diagnostic across asset classes; the estimator
  implemented below (`rolling_ghe_signal`) follows this construction.
- Bariviera, A. F. (2017), "The inefficiency of Bitcoin revisited: a
  dynamic approach", *Economics Letters* 161, 1-4 (arXiv:1709.08090) --
  applies a ROLLING (dynamic) Hurst exponent specifically to Bitcoin daily
  returns, finding H is time-varying and regime-dependent rather than a
  fixed constant -- confirmed live via WebSearch this round, the direct
  precedent for treating H as a live, rolling signal on this exact
  instrument rather than a single full-sample diagnostic.
- Takaishi, T. (2018), "Statistical properties and multifractality of
  Bitcoin", *Physica A* 506, 507-519, and the companion multiscaling study
  "Bitcoin market route to maturity? Evidence from return fluctuations,
  temporal correlations and multiscaling effects" (arXiv:1804.05916) --
  confirmed live via WebSearch this round: both apply exactly this
  fluctuation-scaling machinery to BTC specifically (not a generic-equity
  import) and document regime-dependent, time-varying Hurst behavior
  around BTC's major bull/bear transitions -- i.e. this is a mainstream,
  asset-specific tool for exactly this question on exactly this instrument.

Attacks **ERR** (no error control anywhere in this project's signal path --
GHE is a formal scaling-law estimator, the same posture R-85/86/96/98's
mechanisms used) and **N-approx-3** (a ninth theoretical basis for "has the
regime already turned", not a retune of a basis already tried) for the
conservative branch; the novel branch additionally attacks **COST** (costs
scale with the signal), the one constraint this project's R-65/67/68 line
showed can actually move, by deriving a rebalance-worthiness threshold from
local roughness rather than trading it on a fixed calendar/percentage band.

Not a duplicate of:
- R-01 (HMM), R-82 (BOCPD), R-83 (Kalman LLT), R-84 (vote-latch
  modulation), R-85 (CSD), R-86 (transfer entropy), R-96 (Hawkes), R-98
  (GPD/POT): eight regime-timing mechanisms drawn from eight different
  fields. GHE is a ninth, sharing no mathematical machinery with any of the
  eight -- see the CSD distinction spelled out above specifically, since
  that is the pair most easily mistaken for one another.
- R-65/R-67/R-68 (Garleanu-Pedersen partial-adjustment trading rate) and
  `kelly_regime_ev`/`kelly_regime_ev_fast` (the fee-derived no-trade band):
  both derive a trading RATE or BAND from the signal's own economics
  (decay rate, or the growth-given-up-vs-fee tradeoff). This round's novel
  branch derives its band from a structurally different quantity -- the
  scaling exponent of the price process itself, a property of the MARKET,
  not of the strategy's own signal decay or fee structure -- and does not
  touch either of those two existing derivations; it multiplies
  `kelly_regime_v4`'s existing fixed 10% deadband by a GHE-derived factor,
  never replacing it with the EV-derived band from `kelly_regime_ev`.
- R-93 (Grossman-Zhou drawdown-constrained SIZE) and R-97
  (Wasserstein-DRO SIZE): both replace v4's `scale` factor directly. R-62
  isolated `scale` as carrying NONE of v4's signature (four independent
  confirmations: R-38/R-46/R-59/R-60/R-87). This round does not touch
  `scale` at all -- the conservative branch is a regime-timing ALARM fed
  into the vote (R-53/R-55's validated confirming-vote architecture,
  exactly as R-82/83/84/85/86/96/98 tested their own alarms), and the novel
  branch modulates the rebalance BAND WIDTH (a COST-axis trading-frequency
  control), never the sizing scale.
- The fourteen INFO-axis rounds: every one introduced a NEW external data
  channel. Neither branch here reads any data beyond the already-committed
  BTC OHLCV close series `kelly_regime_v4` itself already consumes -- GHE
  is a function of BTC's own daily close history alone, same posture as
  R-82/83/84/85/86/96/98.
- R-62 (vote x scale factorization): motivates keeping the conservative
  branch confined to a regime-timing ALARM role (fed additively into the
  vote via the confirming-vote weight, never a `scale` retune).

This module is read-only utility, written by the operator before dispatch
(same convention as r82_shared.py through r98_shared.py). Neither branch
edits it. Contains: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s
3-anchor vote construction and the confirming-vote combination rule (copied
verbatim, ultimately from r82_shared.py); (2) a dependency-free (numpy only)
causal rolling GHE(q=1) estimator on daily close-to-close log prices; (3)
the IDENTICAL dated stress-episode table and detection-lag gate scaffolding
R-82/83/84/85/86/96/98 used, copied verbatim so all eight rounds' numbers
stay directly comparable; (4) the causal truncation probe; (5) the holdout
guard.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market GHE number
was computed, and are not retuned after seeing any result.

`FIT_WINDOW_DAYS_GRID = (90, 180, 365)`: the trailing window feeding each
day's GHE fit -- 3/6/12 months, spanning from short enough to be locally
responsive to long enough to match this project's own BASELINE_WINDOW_DAYS
convention (R-85/86/96/98) at the grid's upper end. `LAG_GRID_DAYS =
(1, 2, 4, 8, 16, 32)`: the aggregation lags `tau` the structure function is
fit across -- a geometric ladder spanning from 1 day to ~1 month, the same
doubling-ladder logic `kelly_regime_v4`'s own 20/40/80 anchors use, chosen
for structure rather than fit. `MIN_LAGS_FOR_FIT = 4`: at least 4 of the 6
lag points must have >=30 valid pairs for the log-log regression to run
(Di Matteo's own small-sample caution), else NaN. `Z_THRESH = 2.0`,
`DETECTION_WINDOW_DAYS = 90`, `BASELINE_WINDOW_DAYS = 730` are the same
round, literature-standard values R-82/83/85/86/96/98 all used, copied
verbatim for comparability.

**Kill Switch A (degeneracy check, run once by the operator before any
per-episode number, same posture as R-97/R-98's own Kill Switch A):** does
the alarm z-score (`ghe_signal_zscore`, smoothed+baselined per grid cell)
actually cross `Z_THRESH=2.0` at least once across the full 2017-2022
pre-holdout history, for at least one of the 3 grid cells? This is a
sanity gate on the estimator's own non-degeneracy, checked before any
episode-level lead-time number is computed for any cell.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
the same pattern that has now beaten eight consecutive mechanisms built on
eight different theoretical bases -- an estimator computed FROM price can
only shift once a move has already happened, and the six episodes are
dominated by sudden, discontinuous shocks rather than slow scaling-law
drift. If GHE also lags every sudden 2020-2022 shock and only (at best)
leads the slow 2018 build-up, that is the ninth independent mechanism
converging on the conclusion the ledger's standing diagnosis already leans
toward. For the novel branch specifically: if the GHE-derived band
multiplier does not measurably cut inner-train turnover/whipsaw relative to
v4's fixed 10% band without degrading inner-train Sharpe below the +/-0.2
noise floor, or if it degenerates to a near-constant multiplier (the
same collapse-to-constant failure 23 prior SIZE-axis attempts have hit),
the branch stops at its own pre-registered Step-0 gate before any holdout
read.
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
# r82_shared.py / r85_shared.py / r86_shared.py / r96_shared.py / r98_shared.py.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# IDENTICAL to R-82/83/84/85/86/96/98's own table -- copied verbatim, not
# re-derived, so all eight rounds' gate numbers are directly comparable.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ------------------------------------------------------------- GHE grid
FIT_WINDOW_DAYS_GRID = (90, 180, 365)
LAG_GRID_DAYS = (1, 2, 4, 8, 16, 32)
MIN_LAGS_FOR_FIT = 4
MIN_PAIRS_PER_LAG = 30
DETECTION_WINDOW_DAYS = 90
BASELINE_WINDOW_DAYS = 730
Z_THRESH = 2.0

# Primary decision cell, a-priori: the grid's centre window, matching
# R-85/86/96/98's own BASELINE_WINDOW_DAYS convention as closely as GHE's
# own grid allows. Kill Switch A (see docstring) may override this -- run
# by the operator before any episode-level number, per R-97/98's own
# convention -- and if it does, the override and the reason are recorded
# in the branch that discovers it, not silently substituted here.
PRIMARY_FIT_WINDOW_DAYS = 180


# ----------------------------------------------------------------- v4 vote
#   Copied verbatim from r82_shared.py / r85_shared.py / r86_shared.py /
#   r96_shared.py / r98_shared.py.


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


# --------------------------------------------------------- daily returns


def daily_log_prices(df: pd.DataFrame) -> pd.Series:
    """Daily close, log, from this project's own 5-minute bars resampled to
    daily close (last bar of each UTC day). Entirely causal by construction:
    day t's value depends only on bars dated on or before day t."""
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).rename("log_close")


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily close-to-close log return. Causal, same construction as
    r82-r98_shared.py's identical helper."""
    r = daily_log_prices(df).diff().dropna()
    return r.rename("daily_log_ret")


# ------------------------------------------------------------- GHE core


def ghe_fit_single_window(log_prices: np.ndarray, lags: tuple[int, ...] = LAG_GRID_DAYS,
                           min_lags: int = MIN_LAGS_FOR_FIT,
                           min_pairs: int = MIN_PAIRS_PER_LAG) -> float:
    """Generalized Hurst Exponent, q=1 (Barabasi & Vicsek 1991; Di Matteo
    2007), fit to ONE fixed window of log prices via ordinary least squares
    on ``log(K(tau)) = H*log(tau) + c``, where
    ``K(tau) = mean(|log_prices[i+tau] - log_prices[i]|)`` over all valid
    pairs `i` inside the window.

    Returns NaN if fewer than `min_lags` of the lag points have at least
    `min_pairs` valid pairs (Di Matteo's own small-sample caution), or the
    regression is otherwise degenerate.
    """
    x = np.asarray(log_prices, dtype=float)
    n = len(x)
    log_tau, log_k = [], []
    for tau in lags:
        if tau >= n:
            continue
        diffs = np.abs(x[tau:] - x[:-tau])
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) < min_pairs:
            continue
        k_tau = float(np.mean(diffs))
        if k_tau <= 0 or not np.isfinite(k_tau):
            continue
        log_tau.append(np.log(tau))
        log_k.append(np.log(k_tau))
    if len(log_tau) < min_lags:
        return float("nan")
    slope, _ = np.polyfit(np.asarray(log_tau), np.asarray(log_k), 1)
    if not np.isfinite(slope):
        return float("nan")
    return float(slope)


def rolling_ghe_signal(log_prices: pd.Series, fit_window_days: int,
                        lags: tuple[int, ...] = LAG_GRID_DAYS) -> pd.Series:
    """Causal rolling GHE(q=1), one value per day.

    At day t, uses ONLY log-price observations at days in
    ``[t - fit_window_days, t]`` (inclusive of t itself -- the fit at day t
    characterizes the trailing window ENDING at t, exactly like
    `kelly_regime`'s own anchor means; the +1-day shift that keeps this out
    of the bar it is later joined onto happens in `align_daily_causal`
    below, the same two-step causal discipline every prior daily-cadence
    estimator in this project's INFO/regime-timing rounds uses).
    """
    x = log_prices.to_numpy(dtype=float)
    idx = log_prices.index
    n = len(x)
    out = np.full(n, np.nan)
    min_window = max(int(fit_window_days * 0.5), max(lags) + MIN_PAIRS_PER_LAG)
    for i in range(n):
        lo = max(0, i - fit_window_days + 1)
        window = x[lo:i + 1]
        if len(window) < min_window:
            continue
        out[i] = ghe_fit_single_window(window, lags)
    return pd.Series(out, index=idx, name="ghe")


def ghe_signal_zscore(ghe: pd.Series,
                       detection_window_days: int = DETECTION_WINDOW_DAYS,
                       baseline_window_days: int = BASELINE_WINDOW_DAYS
                       ) -> pd.Series:
    """Causal z-score of a smoothed (detection-window mean) GHE against its
    own trailing baseline -- same "smooth, then z-score against a trailing
    window" shape as R-85/86/96/98's own alarm construction, applied here
    to the Hurst scaling exponent instead of variance/autocorrelation/TE/
    Hawkes intensity/tail shape."""
    smoothed = ghe.rolling(detection_window_days, min_periods=detection_window_days // 3).mean()
    baseline_mean = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).mean()
    baseline_std = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).std()
    z = (smoothed - baseline_mean) / baseline_std.replace(0.0, np.nan)
    return z


def align_daily_causal(daily: pd.Series | pd.DataFrame, bars: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reindex a daily-cadence signal onto `bars`' 5-minute index, causally
    -- a bar at time T may only see the row for the most recent day that
    closed strictly before T's own day (IDENTICAL shift convention to
    `tradebot.data.align_onchain_causal` / r85/86/96/98_shared's own
    helper)."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


def nearest_alarm(z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                   z_thresh: float = Z_THRESH) -> pd.Timestamp | None:
    """Timestamp of the first bar where `z` crosses UP through `z_thresh`,
    closest to `onset` -- the GHE analogue of R-85/86/96/98's own
    `nearest_csd_alarm` / `nearest_te_alarm` / `nearest_hawkes_alarm` /
    `nearest_alarm`."""
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


# --------------------------------------------------------- Step-A gate infra
#   `episode_window`, `block_bootstrap_shifts`, `truncation_causality_probe`
#   copied verbatim from r82/85/86/96/98_shared.py.


def episode_window(bars: pd.DataFrame, onset_str: str, window_days: int = 60
                    ) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


def block_bootstrap_shifts(n_bars: int, block_days: int, n_draws: int, seed: int) -> list[np.ndarray]:
    """Copied verbatim from r82/85/86/96/98_shared.py."""
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


# ---------------------------------------------------------------- holdout guard


def assert_no_holdout(obj) -> None:
    """Hard guard, same pattern as r81/r86/r88/r96/r98: the max timestamp
    anywhere this file touches must be strictly before OOS_START."""
    idx = obj.index if hasattr(obj, "index") else obj
    if len(idx) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz="UTC")
    max_ts = pd.Timestamp(idx.max())
    if max_ts.tzinfo is None:
        max_ts = max_ts.tz_localize("UTC")
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")
