"""Shared, read-only utilities for the R-83 NOVEL round (08-21).

Idea in one sentence: `kelly_regime_v4`'s regime vote is a fixed 20/40/80-
day EMA-crossing heuristic; this round replaces (or confirms) it with the
FILTERED state of a causal Kalman filter for Harvey's (1989, "Forecasting,
Structural Time Series Models and the Kalman Filter", Cambridge UP, ch. 2
"the local linear trend model") local-linear-trend (LLT) structural
time-series model: a two-state model with a stochastic LEVEL and a
stochastic SLOPE, both updated online by the textbook Kalman prediction/
update recursions on daily-resampled log(close).

Which constraint this attacks: **ERR** (no error control anywhere in this
project's signal path -- v4's vote is a deterministic 0/1 latch; the
Kalman filter instead carries an explicit state-covariance `P_t` at every
step, a genuine uncertainty estimate the anchor ladder has no analogue of)
and **N-approx-3** (a continuously-updated ESTIMATE of the current trend's
sign and rate, rather than a fixed 20/40/80-day window asserted a priori).
This is explicitly NOT a tenth INFO-axis signal: it consumes no data
beyond the committed OHLCV close series v4 itself already reads -- same
information, a third structurally distinct way of extracting a regime/
trend estimate from it.

Not a duplicate of:
- R-01 (Hamilton 1989 Markov-switching / HMM, REJECTED ON READING):
  Hamilton's model has a DISCRETE hidden state (bull/bear/chop) that the
  filter assigns probability mass to and which can flip abruptly between
  neighbouring bars once the posterior crosses 0.5; R-01's own verdict was
  that the causal FILTERED state is legitimate but "reported rapid
  switching is fatal at a 0.1% round trip." The LLT model used here has NO
  discrete state at all -- `nu_t` (the slope) is a single CONTINUOUS
  real-valued random walk, so there is no latent regime label to flip
  between; whatever switching behaviour it exhibits is a *sign change of a
  continuous number*, governed by two named process-noise variances
  (`SIGMA_ETA`, `SIGMA_ZETA` below), not a discrete-state transition
  matrix. Section "WHAT WOULD MAKE THIS FAIL" in `r83_novel_kalman_gate.py`
  names directly why this structural difference might, or might not, avoid
  R-01's failure mode in practice -- it is a hypothesis, not an assumption.
- R-82 (Adams & MacKay 2007 BOCPD, NEGATIVE at the gate, 2/6): BOCPD
  maintains a discrete PROBABILITY DISTRIBUTION over "days since the last
  regime break" (a run-length posterior) and requires enough evidence to
  accumulate against a strong prior of persistence (hazard_lambda=250)
  before its MAP run length drops -- an evidence-accumulation step that
  R-82 found fast on slow 2018-style bear onsets and slow on sudden 2020-
  2022 shocks. The LLT filter carries no run-length posterior and no
  changepoint prior at all: its slope is a running estimate that updates
  by a fixed Kalman GAIN every single bar, proportionally to the size of
  the day's surprise -- there is no explicit "how many days has this
  persisted" quantity anywhere in the model. Whether that structural
  difference actually produces different lag behaviour on the same six
  episodes is exactly this round's Step-A question, not asserted here.
- The nine prior INFO-axis rounds (R-53/54/58/73/74/75/76/79/81): none of
  them apply -- this round introduces no new data channel, only a third
  ESTIMATOR of regime state from the same close series v4 already reads.
- R-62 (factored v4 into vote x scale; the vote alone carries the whole
  matched-exposure drawdown signature, the scale factor none of it):
  motivates confining any change to the direction/vote side and leaving
  the conditional-vol-targeting scale factor untouched, exactly as R-80/
  R-81/R-82 did.

This module is read-only utility, written by the operator before any
strategy code is built (same convention as r79_shared.py through
r82_shared.py). Per this project's established per-round convention
("duplicated, not imported" -- R-54/R-55), the pieces this file needs from
r82_shared.py (STRESS_EPISODES, anchor_votes/anchor_majority,
nearest_transition, block_bootstrap_shifts, episode_window,
confirming_vote_frac, the inner-train/validation date constants, the
truncation-causality probe) are duplicated here byte-for-byte rather than
imported, so this round's files are self-contained and cannot be silently
broken by an edit to a sibling round's module.

Contains: (1) the v4 anchor-vote duplicate and the R-53/R-55 confirming-
vote formula (unchanged, duplicated from r82_shared); (2) the causal
Harvey (1989) local-linear-trend Kalman filter itself -- FILTERED state
only, `x_{t|t}`, never smoothed (`x_{t|T}`, which would need the RTS
backward pass and future bars -- exactly the R-01 "smoothed state is not
causal" distinction, so the smoother is not implemented anywhere in this
file at all, not merely unused); (3) the causal daily-to-bar alignment
(reuses `tradebot.data.align_onchain_causal`, the same causal contract
R-82 used); (4) the dated stress-episode table and detection-lag gate
infrastructure (STRESS_EPISODES, episode_window, nearest_transition,
nearest_kalman_detection, block_bootstrap_shifts); (5) shared date
constants and the causality truncation probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_onchain_causal  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated, not imported.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by this file in Step A, and only inner-train/inner-validation are
# read in Step B.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# Dated, PUBLICLY KNOWN historical BTC regime transitions -- byte-for-byte
# identical to r82_shared.STRESS_EPISODES, fixed before any number in this
# round was computed, so the Step-A comparison is apples-to-apples with
# R-82's own BOCPD gate on the identical six episodes.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ----------------------------------------------------------------- v4 vote


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
    own gate, exactly, for use as the Step-A detection-lag gate baseline."""
    votes = anchor_votes(df, horizons, band)
    return sum(votes) / len(votes)


def confirming_vote_frac(anchor_sum: np.ndarray, meta_vote: np.ndarray,
                          weight: float) -> np.ndarray:
    """R-53/R-55's combination rule, duplicated unchanged.

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``

    ``anchor_sum`` in [0, 3], ``meta_vote`` in {0, 1} (per R-80/R-81's
    lesson: keep it DISCRETE so the formula can still reach exactly
    flat/exactly full). ``weight == 0`` recovers `kelly_regime_v4` exactly.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


# ------------------------------------------------------------ Kalman LLT
#
# Harvey (1989), ch. 2: local linear trend model.
#
#   observation:  y_t     = mu_t + eps_t,               eps_t ~ N(0, H)
#   level:        mu_t    = mu_{t-1} + nu_{t-1} + eta_t, eta_t ~ N(0, Q_eta)
#   slope:        nu_t    = nu_{t-1} + zeta_t,           zeta_t ~ N(0, Q_zeta)
#
# eta_t and zeta_t are independent (the standard Harvey 1989 formulation;
# a nonzero level/slope process-noise COVARIANCE is a documented extension
# this file does not use, to keep the model to exactly the two named
# hyperparameters the task asks to reason about explicitly). The coupling
# between level and slope lives in the TRANSITION matrix (mu_t depends on
# nu_{t-1}), not in a noise covariance -- level and slope are correlated
# STATE VARIABLES evolving as a joint (vector) random walk, which is the
# sense in which they are "correlated random walks": nu is itself a random
# walk, and mu is a random walk driven forward at whatever rate nu
# currently estimates, so a shift in nu propagates into mu at every
# subsequent step even though the two INNOVATION terms are independent.
#
# `y_t` is DAILY log(close) (not log-return): the level state directly
# tracks a smoothed log-price and the slope state IS the filter's
# continuously-updated estimate of the current daily log-return trend --
# the quantity this round's mechanism claim is about.

MU0_MODE = "first_obs"   # mu_0 initialized to the series' own first observation
NU0 = 0.0                # no prior belief about initial trend direction
P0_MU = 1.0              # diffuse-ish initial level uncertainty (log-price units^2)
P0_NU = 1.0              # diffuse-ish initial slope uncertainty

# Process/observation noise (log-price units, daily cadence). Fixed BEFORE
# any real BTC number was computed, calibrated ONLY on synthetic data (see
# r83_novel_kalman_gate.py's module docstring, section 2, for the full
# calibration procedure and the numbers that motivated this exact triple)
# and never retuned against the real Step-A gate result.
SIGMA_EPS = 0.030    # observation noise std (idiosyncratic daily price noise
                      # around the smooth trend; smaller than BOCPD's own
                      # ~5.7%/day prior because part of daily variance here
                      # is explained by the trend/slope state itself rather
                      # than treated as pure regime-mean noise)
SIGMA_ETA = 0.0010   # level process-noise std (how much the level can drift
                      # beyond what the current slope already predicts)
SIGMA_ZETA = 0.00007 # slope process-noise std -- THE speed/smoothness knob:
                      # larger values let the slope react faster to a new
                      # trend at the cost of reacting to noise; see the gate
                      # file for the calibration sweep and the false-flip
                      # tradeoff this value was chosen against.


def kalman_llt_filter(y: np.ndarray, sigma_eps: float = SIGMA_EPS,
                       sigma_eta: float = SIGMA_ETA, sigma_zeta: float = SIGMA_ZETA,
                       nu0: float = NU0, p0_mu: float = P0_MU, p0_nu: float = P0_NU
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Causal Harvey (1989) local-linear-trend Kalman filter.

    Returns the FILTERED state only -- ``(level, slope)``, each
    ``x_{t|t}``, i.e. the estimate using observations ``y[0..t]`` inclusive
    and NOTHING after ``t``. This is a deliberate, disclosed choice, not an
    oversight: R-01 (docs/LEDGER.md) rejected Hamilton's HMM regime
    detector on exactly the distinction between the causal FILTERED state
    (legitimate) and the SMOOTHED state most textbook/tutorial code plots
    (`x_{t|T}`, computed by a backward Rauch-Tung-Striebel pass that uses
    every observation up to the END of the series, T, including bars AFTER
    t -- not causal, and not implementable in a real-time backtest at all).
    No RTS smoother, and no other backward pass, exists anywhere in this
    file: only the forward predict/update recursion below. Row/step t's
    output depends only on ``y[0], ..., y[t]`` by construction (each loop
    iteration reads exactly one new observation and the state carried
    forward from the previous iteration; nothing else is in scope), which
    is what `truncation_causality_probe` (below) verifies empirically as a
    second, independent check before this filter ever touches real data.

    Standard Kalman recursion, one bar at a time (T = transition matrix,
    Q = process-noise covariance, Z = observation matrix, H = observation
    noise variance):

        predict:  x_pred = T @ x_prev          P_pred = T @ P_prev @ T.T + Q
        update:   v = y_t - Z @ x_pred          F = Z @ P_pred @ Z.T + H
                  K = P_pred @ Z.T / F          x_t = x_pred + K * v
                  P_t = P_pred - K @ (Z @ P_pred)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    if n == 0:
        return level, slope

    T = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[sigma_eta ** 2, 0.0], [0.0, sigma_zeta ** 2]])
    Z = np.array([1.0, 0.0])
    H = sigma_eps ** 2

    x = np.array([y[0], nu0])
    P = np.array([[p0_mu, 0.0], [0.0, p0_nu]])

    for t in range(n):
        if t > 0:
            x_pred = T @ x
            P_pred = T @ P @ T.T + Q
        else:
            # First observation: no prior transition, just the initial state.
            x_pred = x
            P_pred = P
        yt = y[t]
        if np.isfinite(yt):
            v = yt - Z @ x_pred
            F = float(Z @ P_pred @ Z.T + H)
            K = (P_pred @ Z.T) / F
            x = x_pred + K * v
            P = P_pred - np.outer(K, Z @ P_pred)
        else:
            # Missing observation: propagate the prediction only (still
            # causal -- no information from any future bar is used).
            x = x_pred
            P = P_pred
        level[t] = x[0]
        slope[t] = x[1]

    return level, slope


def kalman_daily_causal_signals(df: pd.DataFrame, sigma_eps: float = SIGMA_EPS,
                                 sigma_eta: float = SIGMA_ETA, sigma_zeta: float = SIGMA_ZETA
                                 ) -> pd.DataFrame:
    """Resample ``df["close"]`` to daily, run the causal LLT filter on
    daily log(close), and align the result onto ``df``'s own 5-minute
    index with a full-calendar-day causal shift
    (`tradebot.data.align_onchain_causal` -- day D's filtered state, which
    depends on day D's own close, only becomes visible to bars starting
    2026-01-02T00:00 UTC given a day dated 2026-01-01; the identical
    causal contract R-82's BOCPD signal used, and every other daily-cadence
    signal in this project).

    Daily rather than 5-minute cadence, for the SAME two reasons R-82 gave
    for BOCPD: (1) v4's own anchors already operate on a 20-80 CALENDAR-DAY
    horizon, so daily granularity matches the horizon the mechanism is
    meant to describe; (2) it keeps the Step-A gate an apples-to-apples
    comparison against R-82's own daily-cadence BOCPD signal on the
    identical six episodes. Unlike BOCPD, the Kalman recursion's per-step
    cost is O(1) (two fixed 2x2 matrix updates), not O(t) growing
    hypothesis pruning, so 5-minute cadence would be computationally cheap
    here -- this is a horizon/comparability choice, not a performance one.

    Returns a DataFrame indexed like ``df`` with columns
    ``kalman_level``, ``kalman_slope``.
    """
    daily_close = df["close"].resample("1D").last().dropna()
    log_close = np.log(daily_close.to_numpy())
    level, slope = kalman_llt_filter(log_close, sigma_eps, sigma_eta, sigma_zeta)
    daily = pd.DataFrame(
        {"kalman_level": level, "kalman_slope": slope},
        index=daily_close.index,
    )
    return align_onchain_causal(daily, df)


# --------------------------------------------------------- Step-A gate infra


def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    """Timestamp, within `window`, of the anchor-gate transition closest to
    `onset`. Duplicated from r82_shared.py (self-contained, not imported,
    per this project's per-round shared-module convention)."""
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


def nearest_kalman_detection(slope: pd.Series, window: pd.DatetimeIndex,
                              onset: pd.Timestamp) -> pd.Timestamp | None:
    """Timestamp, within `window`, of the first bar where the filtered
    Kalman slope crosses DOWN through zero (the sign flips from >= 0 to
    < 0), closest to `onset` -- the LLT analogue of `nearest_transition`
    and of R-82's `nearest_bocpd_detection`.

    Threshold justification: zero is the natural, parameter-free threshold
    for a SLOPE -- "is the estimated trend currently rising or falling" --
    and mirrors exactly the comparison already being made against v4's own
    anchor gate (`nearest_transition(..., direction="down")`, itself a
    downward CROSSING of a comparable quantity, the anchor-vote average).
    Using the same crossing convention on both sides of the comparison
    (down-crossing vs down-crossing) avoids introducing an extra, untested
    degree of freedom (e.g. a band around zero) at Step-A gate time; a
    banded variant is reported ONLY as a post-hoc robustness diagnostic in
    the gate file, never substituted for the frozen primary rule.
    """
    vals = slope.reindex(window).to_numpy()
    neg = vals < 0.0
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = neg[1:] & ~neg[:-1]
    cross[0] = bool(neg[0])
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
    Byte-for-byte duplicate of r82_shared.block_bootstrap_shifts."""
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
