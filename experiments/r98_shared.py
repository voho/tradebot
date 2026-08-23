"""Shared, read-only utilities for the R-98 round (08-23).

Idea in one sentence: a causal, rolling Peaks-Over-Threshold (POT) fit of
the Generalized Pareto Distribution (GPD) to BTC's own daily return
exceedances -- the shape parameter `xi` (tail heaviness) and the model's
own implied extreme quantile (VaR-style) -- gives an EIGHTH structurally
distinct theoretical basis for regime-timing / detection-lag, tested
against the identical six dated historical BTC regime transitions R-82
(BOCPD), R-83 (Kalman LLT), R-84 (vote-latch modulation), R-85 (critical
slowing down), R-86 (transfer entropy) and R-96 (Hawkes point process) all
used, and, separately, as a genuinely new combination role for the novel
branch: a live tail-quantile BREACH used as a Step-0 kill-switch premise
test (does gating exposure off immediately after a fitted-VaR breach
reduce forward realized loss?), rather than another lead-time alarm.

Literature grounding, both fetched and read before being relied on:
- Pickands, J. (1975), "Statistical inference using extreme order
  statistics", *Annals of Statistics* 3(1), 119-131 -- the foundational
  Peaks-Over-Threshold theorem: exceedances over a sufficiently high
  threshold converge in distribution to a Generalized Pareto Distribution,
  independent of the parent distribution's own shape. A fundamentally
  different mathematical object from a hidden discrete state (HMM/R-01), a
  Bayesian run-length posterior (BOCPD/R-82), a linear-Gaussian
  state-space filter (Kalman LLT/R-83), the vote's own latch dynamics
  (R-84), a fluctuation-statistics trend test (CSD/R-85), an
  information-theoretic directed-flow functional (transfer entropy/R-86),
  or a self-exciting conditional event-rate (Hawkes/R-96) -- it is an
  asymptotic THEOREM ABOUT TAIL SHAPE, with no notion of hidden state,
  segmentation, linearity, information flow, or point-process
  self-excitation at all.
- Hosking, J. R. M., & Wallis, J. R. (1987), "Parameter and Quantile
  Estimation for the Generalized Pareto Distribution", *Technometrics*
  29(3), 339-349 -- the closed-form probability-weighted-moments (PWM)
  estimator used here (`gpd_pwm_fit`), chosen over full MLE for the same
  reason this project's other hand-rolled estimators (Kendall-tau,
  discretized TE, moment-matched Hawkes) avoid iterative optimizers:
  dependency-free (numpy only; scipy is not a project dependency) and
  numerically stable on small tail samples, at the disclosed cost of being
  less efficient asymptotically than MLE.
- McNeil, A. J., & Frey, R. (2000), "Estimation of tail-related risk
  measures for heteroscedastic financial time series: an extreme value
  approach", *Journal of Empirical Finance* 7(3-4), 271-300 -- the
  standard dynamic-POT recipe for financial return series (rolling
  threshold + GPD tail fit refreshed through time) and the closed-form
  POT quantile (VaR) estimator `gpd_var` below.
- Ke, R., Yang, L., & Tan, C. (2022), "Forecasting tail risk for Bitcoin:
  A dynamic peak over threshold approach", *Finance Research Letters* --
  confirmed live via WebSearch this round: applies exactly this
  rolling-POT/GPD machinery to BTC daily returns specifically (not a
  generic-equity import) and finds the dynamic PoT model's lower-tail VaR
  forecasts outperform GARCH-EVT on out-of-sample backtests -- i.e. this
  is a mainstream, asset-specific 2022-2026 tool for exactly this
  question on exactly this instrument, not an idea imported from an
  unrelated domain the way Grossman-Zhou/CPPI (R-93) or
  Goulding-Harvey-Mazzoleni (R-91) were flagged as edge cases.

Attacks **ERR** (no error control anywhere in this project's signal path
-- POT/GPD is a formal asymptotic tail-inference device, the same
justification R-87's conformal wrapper and R-97's Wasserstein-DRO ball
used) and **N-approx-3** (an eighth theoretical basis for "has the tail
regime already broken", not a retune of a basis already tried).

Not a duplicate of:
- R-01 (HMM), R-82 (BOCPD), R-83 (Kalman LLT), R-84 (vote-latch
  modulation), R-85 (CSD), R-86 (transfer entropy), R-96 (Hawkes): seven
  regime-timing mechanisms drawn from seven different fields. GPD/POT is
  an eighth, sharing no mathematical machinery with any of the seven: no
  hidden state, no segmentation, no linearity assumption, no trend/
  variance functional, no information-theoretic functional of two series,
  and no conditional event-rate/self-excitation -- it is an asymptotic
  distributional theorem about EXCEEDANCES OVER A THRESHOLD, a strictly
  different formal object.
- R-93 (Grossman-Zhou drawdown-constrained SIZE) and R-97
  (Wasserstein-DRO SIZE): both replace v4's `scale` factor directly. R-62
  isolated `scale` as carrying NONE of v4's signature (four independent
  confirmations, R-38/R-46/R-59/R-60/R-87), and both R-93 and R-97
  reproduced that exact failure mode. This round does not touch `scale`
  at all -- the conservative branch is a regime-timing ALARM (fed into
  the vote, R-53/R-55's validated confirming-vote architecture, exactly
  as R-82/83/84/85/86/96 tested their own alarms), and the novel branch
  is a discrete kill-switch on TOTAL exposure triggered by a live
  quantile breach, structurally closer to R-90's stop/ratchet family than
  to any scale retune -- disclosed explicitly in the novel branch's own
  pre-registration as the risk this shares with B-41's closed finding
  (whipsaw on tight re-entry), which is why the novel branch's own Step-0
  gate is designed to catch that failure mode before any strategy code
  is written, not merely to hope it does not appear.
- The fourteen INFO-axis rounds (R-44/R-53/R-54/R-55/R-58/R-73/R-74/
  R-75/R-79/R-81/R-84's INFO half/R-88's conservative half/R-94/R-95):
  every one introduced a NEW external data channel used as a directional
  vote or brake. Neither branch here reads any data beyond the
  already-committed BTC OHLCV close series `kelly_regime_v4` itself
  already consumes -- GPD tail shape is a function of BTC's own daily
  return history alone, same posture as R-82/83/84/85/86/96.
- R-62 (vote x scale factorization): motivates keeping the conservative
  branch confined to a regime-timing ALARM role (fed additively into the
  vote via the confirming-vote weight, never a `scale` retune) and the
  novel branch confined to a discrete exposure override (never a
  continuous scale multiplier), exactly the same posture R-96 held.

This module is read-only utility, written by the operator before dispatch
(same convention as r82_shared.py through r97_shared.py). Neither branch
edits it. Contains: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s
3-anchor vote construction and the confirming-vote combination rule
(copied verbatim, ultimately from r82_shared.py, so every regime-timing
round's baseline stays identical); (2) a dependency-free (numpy only)
causal rolling-threshold POT event extractor operating on DAILY
close-to-close log returns (a deliberately simpler event definition than
R-96's intraday relative-jump statistic -- POT's own literature, including
Ke-Yang-Tan 2022, standardly works on the daily return series directly,
and using a different cadence than R-96's intraday jump flag keeps this
round's event definition genuinely distinct rather than a re-skin); (3) a
dependency-free Hosking-Wallis PWM GPD fit and the McNeil-Frey POT
quantile (VaR) formula; (4) the IDENTICAL dated stress-episode table and
detection-lag gate scaffolding R-82/83/84/85/86/96 used (`STRESS_EPISODES`,
`episode_window`, `nearest_transition`, `block_bootstrap_shifts`), copied
verbatim so all seven rounds' numbers stay directly comparable; (5) the
causal truncation probe.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market GPD number
was computed, and are not retuned after seeing any result.

`THRESH_QUANTILE_GRID = (0.90, 0.95, 0.975)`: the POT threshold `u`, the
fraction of trailing days counted as "normal" rather than "peak" --
spanning the standard POT literature's usual 90th-97.5th percentile range
(McNeil & Frey 2000 use 90th; Ke-Yang-Tan's own robustness range extends
to the mid-97th). `FIT_WINDOW_DAYS_GRID = (365, 730, 1095)`: 1/2/3 years
of trailing history feeding the GPD fit, matching this project's own
BASELINE_WINDOW_DAYS=730 convention (R-85/86/96) at the grid's centre.
`MIN_EXCEEDANCES = 15`: Hosking & Wallis's own stated small-sample
guidance for a stable PWM fit; a threshold/window combination producing
fewer exceedances than this returns NaN for that day rather than an
unstable fit (disclosed, not silently degraded -- the 0.975/365-day corner
of the grid is expected to be mostly NaN by construction, ~9 expected
exceedances against a floor of 15; this is reported as a grid-corner
limitation, not treated as a result). `Z_THRESH = 2.0`, `DETECTION_WINDOW_
DAYS = 90`, `BASELINE_WINDOW_DAYS = 730` are the same round, literature-
standard values R-82/83/85/86/96 all used, copied verbatim for
comparability. `VAR_PROB = 0.99`: the POT quantile probability used for
the novel branch's kill-switch trigger, the standard regulatory/risk-
management VaR level (Basel-style), fixed here rather than swept.

**Kill Switch A (degeneracy check, run once by the operator before any
per-episode number, the same posture as R-97's own Kill Switch A/B):**
does the alarm z-score (`gpd_signal_zscore`, smoothed+baselined per grid
cell) actually cross `Z_THRESH=2.0` at least once across the full
2017-2022 pre-holdout history? Checked for all 9 grid cells before
choosing which is PRIMARY (this is a sanity gate on the estimator's own
non-degeneracy, not a search over episode-level performance -- no episode,
lead time, or gate pass/fail number was computed before this check ran).
Result: the grid-CENTRE cell (quantile=0.95, fit_window_days=730), the
combination that would otherwise have been the natural a-priori choice by
analogy to R-85/86/96's own BASELINE_WINDOW_DAYS=730 convention, never
crosses 2.0 in six years (max z=1.81) -- a real, disclosed property of
this cell (the 90-day-smoothed, 2-year-fit tail-shape estimate is itself
too inertial to produce a single 2-sigma excursion against its own
730-day baseline on this series), not a bug: the 0.90/365 and 0.90/730
neighbours both fire 3 times, so the pipeline is not degenerate overall.
**PRIMARY is therefore set to quantile=0.90, fit_window_days=730** --
still a literature-standard value (McNeil & Frey's own primary threshold
choice) and still the grid's central fit-window, the smallest possible
deviation from the original a-priori choice that clears Kill Switch A,
chosen for non-degeneracy alone and BEFORE any episode-level lead number
existed. The 0.95/730 cell is retained and reported as a full grid
member (context, not the decision cell).

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
the same pattern that has now beaten seven consecutive mechanisms built on
seven different theoretical bases -- an estimator computed FROM price
(here: the empirical tail shape of recent daily returns) can only shift
once a heavy-tailed move has already happened, which is exactly the
moment v4's own fixed-window anchor is also starting to react. If GPD tail
shape also lags every sudden 2020-2022 shock and only (at best) leads the
slow 2018 build-up, that is the eighth independent mechanism converging on
the same conclusion the ledger's standing diagnosis already leans toward:
this six-episode gate is unwinnable by any estimator computed from this
project's own committed price history, whatever field it is drawn from,
and the finding is about the gate/dataset rather than about any one
technique. For the novel branch specifically: if forward realized loss in
the bars immediately following a live POT/VaR breach is NOT significantly
worse than the unconditional baseline on inner-train, a tail-breach
kill-switch buys nothing and the branch must stop at that pre-registered
Step-0 gate before any kill-switch strategy is built or backtested.
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
# r82_shared.py / r85_shared.py / r86_shared.py / r96_shared.py.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# IDENTICAL to R-82/83/84/85/86/96's own table -- copied verbatim, not
# re-derived, so all seven rounds' gate numbers are directly comparable.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ------------------------------------------------------------- POT/GPD grid
THRESH_QUANTILE_GRID = (0.90, 0.95, 0.975)
FIT_WINDOW_DAYS_GRID = (365, 730, 1095)
MIN_EXCEEDANCES = 15
DETECTION_WINDOW_DAYS = 90
BASELINE_WINDOW_DAYS = 730
Z_THRESH = 2.0
VAR_PROB = 0.99

# Primary decision cell -- see "Kill Switch A" note above: chosen for
# non-degeneracy (>=1 alarm in 2017-2022) alone, before any episode-level
# number was computed.
PRIMARY_THRESH_QUANTILE = 0.90
PRIMARY_FIT_WINDOW_DAYS = 730


# ----------------------------------------------------------------- v4 vote
#   Copied verbatim from r82_shared.py / r85_shared.py / r86_shared.py /
#   r96_shared.py.


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


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily close-to-close log return, from this project's own 5-minute
    bars resampled to daily close (last bar of each UTC day). Entirely
    causal by construction: day t's value depends only on bars dated on or
    before day t."""
    daily_close = df["close"].resample("1D").last().dropna()
    r = np.log(daily_close).diff().dropna()
    return r.rename("daily_log_ret")


# ------------------------------------------------------------- GPD core


def gpd_pwm_fit(exceedances: np.ndarray) -> tuple[float, float]:
    """Hosking & Wallis (1987) probability-weighted-moments estimator for
    GPD(sigma, xi) fit to POSITIVE exceedances over a threshold (i.e.
    already ``x - u`` for x > u).

    Closed form, no iterative optimizer (dependency-free -- scipy/sklearn
    are not project dependencies, same reason R-65/67/68/79/85/86/96
    hand-rolled their own statistics): with sorted exceedances
    ``y_(1) <= ... <= y_(n)``,

        b0 = mean(y)
        b1 = mean( ((n - rank) / (n - 1)) * y )   (rank = 1..n)
        xi    = 2 - b0 / (b0 - 2*b1)
        sigma = 2 * b0 * b1 / (b0 - 2*b1)

    (Hosking-Wallis's own kappa = -xi in this project's sign convention;
    xi > 0 is a heavy (Pareto-type) tail, xi = 0 exponential, xi < 0
    bounded.) Returns ``(nan, nan)`` if fewer than 2 exceedances, the
    denominator is degenerate, or the fitted scale is non-positive
    (a well-known PWM failure mode on small/ill-conditioned samples,
    caught rather than silently propagated).
    """
    y = np.sort(np.asarray(exceedances, dtype=float))
    y = y[np.isfinite(y) & (y > 0)]
    n = len(y)
    if n < 2:
        return float("nan"), float("nan")
    ranks = np.arange(1, n + 1, dtype=float)
    b0 = float(np.mean(y))
    b1 = float(np.mean(((n - ranks) / (n - 1.0)) * y))
    denom = b0 - 2.0 * b1
    if not np.isfinite(denom) or abs(denom) < 1e-15:
        return float("nan"), float("nan")
    xi = 2.0 - b0 / denom
    sigma = 2.0 * b0 * b1 / denom
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan"), float("nan")
    return float(xi), float(sigma)


def gpd_var(u: float, sigma: float, xi: float, n_total: int, n_exceed: int,
            p: float = VAR_PROB) -> float:
    """McNeil & Frey (2000) POT quantile (VaR) estimator at probability
    `p` (e.g. 0.99 = the 99th-percentile loss magnitude), given a GPD fit
    to exceedances over threshold `u` from a sample of `n_total` points of
    which `n_exceed` exceeded `u`.

    ``VaR_p = u + (sigma/xi) * [ ((n_total/n_exceed)*(1-p))^{-xi} - 1 ]``
    for xi != 0; the xi -> 0 (exponential-tail) limit substitutes a log.
    Returns NaN if the fit is degenerate or `p` is not inside the range
    the exceedance count can support (ratio <= 0).
    """
    if not (np.isfinite(sigma) and np.isfinite(xi)) or sigma <= 0 or n_exceed <= 0:
        return float("nan")
    ratio = (n_total / n_exceed) * (1.0 - p)
    if ratio <= 0:
        return float("nan")
    if abs(xi) < 1e-8:
        return float(u - sigma * np.log(ratio))
    return float(u + (sigma / xi) * (ratio ** (-xi) - 1.0))


def rolling_gpd_signal(daily_ret: pd.Series, thresh_quantile: float,
                        fit_window_days: int,
                        min_exceedances: int = MIN_EXCEEDANCES) -> pd.DataFrame:
    """Causal rolling POT/GPD tail fit on |daily return|.

    At each day t, using ONLY days strictly before t (the window
    ``vals[lo:i]`` excludes index i itself, so day t's own return never
    enters its own threshold or fit -- the same "causal expanding/rolling,
    shift by one" discipline as `r96_shared.hawkes_intensity_daily`'s
    `mu_t`):

    - threshold ``u_t`` = the `thresh_quantile` quantile of ``|return|``
      over the trailing `fit_window_days` days (days [t-fit_window_days,
      t) ).
    - exceedances = ``|return| - u_t`` for trailing-window days where
      ``|return| > u_t``.
    - ``(xi_t, sigma_t) = gpd_pwm_fit(exceedances)`` if the exceedance
      count clears `min_exceedances`, else NaN.
    - ``var_t`` = `gpd_var(u_t, sigma_t, xi_t, ...)` at `VAR_PROB`.
    - ``breach_t`` = 1.0 if TODAY's own ``|return[t]|`` exceeds
      ``var_{t}`` (the quantile fit from data strictly before t) else 0.0
      -- a live, causal breach flag: whether today's realized move is
      larger than yesterday's model would have called a `VAR_PROB`-tail
      event.

    Returns a DataFrame indexed like `daily_ret` with columns
    ``['xi', 'sigma', 'threshold', 'n_exceed', 'var', 'breach']``.
    """
    abs_ret = daily_ret.abs().to_numpy(dtype=float)
    idx = daily_ret.index
    n = len(abs_ret)
    xi_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    thresh_arr = np.full(n, np.nan)
    nexc_arr = np.full(n, np.nan)
    var_arr = np.full(n, np.nan)
    breach_arr = np.zeros(n)

    min_window = max(60, fit_window_days // 4)
    for i in range(n):
        lo = max(0, i - fit_window_days)
        window = abs_ret[lo:i]  # strictly earlier days only
        if len(window) < min_window:
            continue
        u = float(np.quantile(window, thresh_quantile))
        exc = window[window > u] - u
        n_exceed = len(exc)
        thresh_arr[i] = u
        nexc_arr[i] = n_exceed
        if n_exceed >= min_exceedances:
            xi, sigma = gpd_pwm_fit(exc)
            xi_arr[i] = xi
            sigma_arr[i] = sigma
            var_t = gpd_var(u, sigma, xi, len(window), n_exceed, VAR_PROB)
            var_arr[i] = var_t
            if np.isfinite(var_t) and abs_ret[i] > var_t:
                breach_arr[i] = 1.0

    return pd.DataFrame(
        {"xi": xi_arr, "sigma": sigma_arr, "threshold": thresh_arr,
         "n_exceed": nexc_arr, "var": var_arr, "breach": breach_arr},
        index=idx,
    )


def gpd_signal_zscore(xi: pd.Series,
                       detection_window_days: int = DETECTION_WINDOW_DAYS,
                       baseline_window_days: int = BASELINE_WINDOW_DAYS
                       ) -> pd.Series:
    """Causal z-score of a smoothed (detection-window mean) GPD shape `xi`
    against its own trailing baseline -- same "smooth, then z-score
    against a trailing window" shape as R-85/86/96's own alarm
    construction, applied here to tail shape instead of variance/
    autocorrelation/TE/Hawkes intensity."""
    smoothed = xi.rolling(detection_window_days, min_periods=detection_window_days // 3).mean()
    baseline_mean = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).mean()
    baseline_std = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).std()
    z = (smoothed - baseline_mean) / baseline_std.replace(0.0, np.nan)
    return z


def align_daily_causal(daily: pd.Series | pd.DataFrame, bars: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reindex a daily-cadence signal onto `bars`' 5-minute index, causally
    -- a bar at time T may only see the row for the most recent day that
    closed strictly before T's own day (IDENTICAL shift convention to
    `tradebot.data.align_onchain_causal` / r85/86/96_shared's own
    helper)."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


def nearest_alarm(z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                   z_thresh: float = Z_THRESH) -> pd.Timestamp | None:
    """Timestamp of the first bar where `z` crosses UP through `z_thresh`,
    closest to `onset` -- the GPD analogue of R-85/86/96's own
    `nearest_csd_alarm` / `nearest_te_alarm` / `nearest_hawkes_alarm`."""
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
#   `nearest_transition`, `episode_window`, `block_bootstrap_shifts`,
#   `truncation_causality_probe` copied verbatim from r82/85/86/96_shared.py.


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


def episode_window(bars: pd.DataFrame, onset_str: str, window_days: int = 60
                    ) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


def block_bootstrap_shifts(n_bars: int, block_days: int, n_draws: int, seed: int) -> list[np.ndarray]:
    """Copied verbatim from r82/85/86/96_shared.py."""
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
    """Hard guard, same pattern as r81/r86/r88/r96: the max timestamp
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
