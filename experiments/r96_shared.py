"""Shared, read-only utilities for the R-96 round (08-22).

Idea in one sentence: a self-exciting Hawkes point process (Hawkes 1971,
"Spectra of some self-exciting and mutually exciting point processes",
*Biometrika* 58(1):83-90) fit to `kelly_regime_v4`'s own price series --
the conditional intensity of further "jump" events rising immediately
after a jump has just occurred -- gives a SEVENTH structurally distinct
theoretical basis for regime-timing / detection-lag, tested against the
identical six dated historical BTC regime transitions R-82 (BOCPD), R-83
(Kalman LLT), R-84 (vote-latch modulation), R-85 (critical slowing down)
and R-86 (transfer entropy) all used, and, separately, as a genuinely new
combination role: an execution-timing brake keyed on cluster intensity
rather than volatility (R-77/B-24) or order-flow direction (R-88).

Literature grounding, both fetched and read before being relied on:
- Hawkes, A. G. (1971), *Biometrika* 58(1):83-90 -- the exponential-kernel
  self-exciting point process itself: lambda(t) = mu + sum_{t_i<t}
  alpha*exp(-beta*(t-t_i)), a fundamentally different mathematical object
  from a hidden discrete state (HMM/R-01), a Bayesian run-length posterior
  over segments (BOCPD/R-82), a linear-Gaussian state-space filter
  (Kalman LLT/R-83), the vote's own latch/confirmation dynamics (R-84), a
  fluctuation-statistics trend test (CSD/R-85), or an information-theoretic
  directed-flow functional (transfer entropy/R-86) -- it is a CONDITIONAL
  RATE of future point events given the timing of past ones, with no
  notion of hidden state, segmentation, linearity, or information flow at
  all.
- Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2015), "Hawkes Processes in
  Finance", *Market Microstructure and Liquidity* 1(1):1550005 -- the
  standard finance review: self-excitation captures the empirically
  well-documented fact that large price moves cluster in time (volatility
  clustering as *causally generated* by a jump raising the conditional
  probability of a further jump, not merely correlated variance).
- Février 2025 arXiv/quant-fin surveys (fetched via WebSearch this round)
  confirm Hawkes processes are an active, standard 2024-2026 tool for
  crypto regime/turbulence and jump-clustering study (e.g. the Markov-
  modulated Hawkes regime-detection line, arXiv:2502.04027; multivariate
  Hawkes BTC/LOB work, Springer DEF 2026), i.e. this is not an idea
  imported from an unrelated domain the way Grossman-Zhou/CPPI (R-93) or
  Goulding-Harvey-Mazzoleni (R-91) were flagged as edge cases -- it is a
  mainstream tool for exactly this question on exactly this asset class.
- Barndorff-Nielsen, O. E., & Shephard, N. (2004, 2006), power/bipower
  variation and jump detection in the presence of stochastic volatility,
  *J. Financial Econometrics* -- used here ONLY to define the point
  process's own EVENT TIMES (which days count as "a jump happened"), via
  the standard relative-jump statistic RJ_t = max(0, RV_t - BV_t)/RV_t
  (Huang & Tauchen 2005; Andersen, Bollerslev & Diebold 2007), computed
  from this project's own 5-minute intraday bars grouped by calendar day
  -- INTRADAY realized variance (RV) and bipower variation (BV), not a
  daily-return proxy. This is a jump-robust volatility estimator used as
  an event-time filter, not a SIZE-axis scale (R-62's finding -- the scale
  factor carries none of v4's signature, confirmed four independent ways
  by R-38/R-46/R-59/R-60/R-87 -- is the reason this round does not spend a
  branch retuning `scale` with it).

Attacks **ERR** (no error control anywhere in this project's signal path)
and **N-approx-3** (a seventh theoretical basis for "has a regime just
started to break", not a retuned version of a basis already tried, and
a fundamentally different combination role for the novel branch -- an
execution brake keyed on self-excitation rather than an alarm/vote).

Not a duplicate of:
- R-01 (HMM), R-82 (BOCPD), R-83 (Kalman LLT), R-84 (vote-latch
  modulation), R-85 (CSD), R-86 (transfer entropy): six regime-timing
  mechanisms drawn from six different fields (discrete-state switching,
  Bayesian changepoint estimation, linear state-space filtering, vote
  confirmation-dynamics modulation, dynamical-systems fluctuation
  statistics, information-theoretic directed flow). A self-exciting point
  process is a seventh, sharing no mathematical machinery with any of the
  six: it has no hidden state, no segmentation, no linearity assumption,
  no trend/variance functional of one series, and no informationtheoretic
  functional of two series -- it is a conditional EVENT RATE, a strictly
  different formal object.
- R-88 (taker-flow execution delay): reuses the bounded-delay-then-force
  architecture (same shape: postpone a scheduled rebalance up to K bars,
  force through at the deadline) but keys it on Hawkes cluster intensity
  (a purely price-derived, univariate signal with no notion of trade
  direction) rather than order-flow direction (a bivariate buy/sell
  imbalance reported by the venue) -- a different signal driving an
  already-validated architecture, exactly the precedent R-88 itself set
  reusing R-77's delay shape with a new driver.
- R-77/B-24 (volatility-driven execution timing, closed NEGATIVE): both
  condition on a VOLATILITY LEVEL (patient-limit/regime-adaptive urgency
  keyed to realized vol). This round's novel branch conditions on
  CLUSTERING INTENSITY -- the conditional rate of further jumps given
  recent jump timing, a property no volatility estimator captures (two
  periods with identical realized volatility can have very different
  Hawkes branching ratios depending on whether that volatility arose from
  one isolated jump or several temporally clustered ones).
- R-93 (Grossman-Zhou drawdown-constrained SIZE): a function of the
  strategy's OWN realized drawdown, replacing `scale`. This round touches
  neither `scale` nor drawdown at all -- it is a regime-timing alarm
  (conservative) and an execution-timing brake (novel), not a sizing rule.
- The thirteen INFO-axis rounds (R-44/R-53/R-54/R-55/R-58/R-73/R-74/
  R-75/R-79/R-81/R-84's INFO half/R-88's conservative half/R-94/R-95):
  every one introduced a NEW external data channel used as a directional
  vote or brake. This round's conservative branch reads no data beyond
  the committed OHLCV close series `kelly_regime_v4` itself already
  consumes (same posture as R-82/83/84/85/86); the novel branch reads no
  new channel either -- Hawkes intensity is a function of BTC's own price
  history alone.
- R-62 (vote x scale factorization): motivates keeping the conservative
  branch's mechanism confined to a regime-timing ALARM role (tested
  against the vote's own transition timing, exactly as R-82/83/84/85/86
  did), never a `scale` retune.

This module is read-only utility, written by the operator before dispatch
(same convention as r82_shared.py through r88_shared.py). Neither branch
edits it. Contains: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s
3-anchor vote construction (copied verbatim, ultimately from
r82_shared.py, so every regime-timing round's baseline is identical);
(2) a dependency-free (numpy only -- scipy/sklearn are not project
dependencies, same reason R-65/R-67/R-68/R-79/R-85/R-86 hand-rolled their
own statistics) intraday relative-jump event detector built from this
project's own 5-minute bars; (3) a dependency-free exponential-kernel
Hawkes intensity evaluator, using a MOMENT-MATCHED (not full-MLE)
parametrization -- disclosed plainly, not hidden: `mu_t = rate_t*(1-n)`,
`alpha = n*beta` from the standard stationary first-moment identity for a
sub-critical exponential Hawkes process (`E[rate] = mu/(1-n)`), where
`rate_t` is a CAUSAL EXPANDING mean of the event flag over strictly
earlier days only (never a single whole-series constant, which would
leak future event frequency into early rows), swept over a small
a-priori grid of the branching ratio `n` and decay half-life, rather than
fit by full likelihood maximization -- a simplification in the same
spirit as this project's other hand-rolled estimators (Kendall-tau,
discretized TE), NOT a claim of a state-of-the-art Hawkes fit; (4) the
IDENTICAL dated stress-episode table and detection-lag gate scaffolding
R-82/83/84/85/86 used (`STRESS_EPISODES`, `episode_window`,
`nearest_transition`, `block_bootstrap_shifts`), copied verbatim so all
six rounds' numbers stay directly comparable; (5) the causal truncation
probe.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market Hawkes
number was computed, and are not retuned after seeing any result.

`RJ_THRESH = 0.5`: a day is a "jump day" (an event for the point process)
if more than half its intraday quadratic variation (RV, sum of squared
5-minute log returns) is attributable to the jump component (RV - BV,
where BV is the intraday bipower variation) -- the standard Huang-Tauchen/
Andersen-Bollerslev-Diebold relative-jump statistic, thresholded at the
simplest defensible round number rather than an asymptotic chi-square
critical value (which needs realized tripower quarticity this project has
no prior use for and would add a second novel dependency-free estimator
to a single event-definition step). `N_GRID = (0.3, 0.5, 0.7)`: sub-
critical branching ratios spanning weak/moderate/strong self-excitation
(n must be < 1 for a stationary process; 0.7 is already close to the
"nearly critical, long memory" regime finance Hawkes fits commonly find).
`HALFLIFE_DAYS_GRID = (3, 7, 14)`: short/medium/long cluster memory,
matching the same "span a decade of horizons, cheaply" logic R-82's
hazard grid and R-85/R-86's Z_THRESH choice used. `Z_THRESH = 2.0` is the
same round, literature-standard two-sigma alarm threshold R-82/83/85/86
all used. `DETECTION_WINDOW_DAYS = 90` / `BASELINE_WINDOW_DAYS = 730` are
copied verbatim from R-85/R-86 -- close to v4's slowest 80-day anchor, and
a 2-year baseline, respectively.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
the same pattern that has now beaten six consecutive mechanisms built on
six different theoretical bases -- an estimator computed FROM price (here:
jump event timing) can only rise once a jump has already happened, which
is exactly the moment v4's own fixed-window anchor is also starting to
react. If Hawkes intensity also lags every sudden 2020-2022 shock and only
(at best) leads the slow 2018 build-up, that is the seventh independent
mechanism converging on the same conclusion the ledger's standing
diagnosis already leans toward: this six-episode gate is unwinnable by any
estimator computed from this project's own committed price history,
whatever field it is drawn from, and the finding is about the gate/dataset
rather than about any one technique. For the novel branch specifically:
if realized volatility/whipsaw frequency in the bars immediately following
a Hawkes-intensity spike is NOT significantly elevated relative to the
unconditional baseline on inner-train, delaying execution during a cluster
buys nothing and the branch must stop at that pre-registered gate before
any delay mechanism is built or backtested.
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
# r82_shared.py / r85_shared.py / r86_shared.py, not reimplemented.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# IDENTICAL to R-82/83/84/85/86's own table -- copied verbatim, not
# re-derived, so all six rounds' gate numbers are directly comparable.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ------------------------------------------------------------- jump params
RJ_THRESH = 0.5

# ------------------------------------------------------------ Hawkes grid
N_GRID = (0.3, 0.5, 0.7)
HALFLIFE_DAYS_GRID = (3, 7, 14)
DETECTION_WINDOW_DAYS = 90
BASELINE_WINDOW_DAYS = 730
Z_THRESH = 2.0


# ----------------------------------------------------------------- v4 vote
#   Copied verbatim from r82_shared.py / r85_shared.py / r86_shared.py.


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


# ---------------------------------------------------------- jump events


def intraday_relative_jump(df: pd.DataFrame, rj_thresh: float = RJ_THRESH) -> pd.Series:
    """Daily 0/1 jump-event flag from this project's own 5-minute bars.

    For each calendar day: `RV = sum(r_i^2)` (realized variance, all
    intraday 5-minute log returns); `BV = (pi/2) * sum(|r_i| * |r_{i-1}|)`
    (bipower variation, Barndorff-Nielsen & Shephard 2004/2006 -- robust to
    jumps, estimates the continuous component only); the relative-jump
    statistic `RJ = max(0, RV - BV) / RV` (Huang & Tauchen 2005). A day is
    flagged a jump event if `RJ > rj_thresh`. Entirely within-day (no
    cross-day lookahead): day t's flag depends only on bars dated t.
    Returns a daily-indexed 0/1 Series (UTC midnight timestamps).
    """
    close = df["close"]
    r = np.log(close).diff()
    day = df.index.floor("D")
    frame = pd.DataFrame({"r": r.to_numpy(), "day": day})
    frame = frame.dropna(subset=["r"])

    def _day_stat(g: pd.DataFrame) -> float:
        rv = float(np.sum(g["r"].to_numpy() ** 2))
        absr = np.abs(g["r"].to_numpy())
        if len(absr) < 2 or rv <= 0.0:
            return 0.0
        bv = (np.pi / 2.0) * float(np.sum(absr[1:] * absr[:-1]))
        rj = max(0.0, rv - bv) / rv
        return rj

    rj_by_day = frame.groupby("day").apply(_day_stat, include_groups=False)
    flag = (rj_by_day > rj_thresh).astype(float)
    flag.index = pd.DatetimeIndex(flag.index, tz="UTC")
    return flag.rename("jump_flag")


# ------------------------------------------------------------- Hawkes core


def hawkes_decay_params(n: float, halflife_days: float) -> tuple[float, float]:
    """`(alpha, beta)` from the branching ratio and decay half-life.

    `beta = ln(2) / halflife_days`. `alpha = n * beta` (branching ratio
    `n = alpha/beta` for kernel `phi(t) = alpha * exp(-beta*t)`). `n` must
    be < 1; the a-priori grid (`N_GRID`) never includes 1 or above.
    """
    assert 0.0 <= n < 1.0, f"branching ratio must be sub-critical, got {n}"
    beta = np.log(2.0) / float(halflife_days)
    alpha = n * beta
    return alpha, beta


def hawkes_intensity_daily(event_flag: pd.Series, n: float, halflife_days: float,
                            min_days: int = DETECTION_WINDOW_DAYS) -> pd.Series:
    """Causal daily Hawkes intensity `lambda(t)`, moment-matched with a
    CAUSAL EXPANDING baseline rate rather than a single whole-series
    constant -- using the full-series event rate to set `mu` would leak
    the series' own future event frequency into early rows (exactly the
    class of bug `truncation_causality_probe` exists to catch). Instead,
    at each day `t`, `mu_t = rate_t * (1 - n)` where `rate_t` is the mean
    event rate over strictly earlier days only
    (`event_flag[:t].expanding().mean()`, shifted by one day so day t's
    own flag is never included in its own mu). The first `min_days` rows
    are NaN (insufficient history for a stable rate estimate -- same
    warmup convention as `hawkes_intensity_zscore`'s baseline window).

    Recursive evaluation (Ozaki 1979) of the excitation term: maintain
    `R`, the decayed sum of past excitation; at each day,
    `lambda[t] = mu_t + alpha * R` using R accumulated from days < t,
    then (if day t was itself a jump event) update
    `R <- R * exp(-beta) + 1` for the next day's evaluation. O(n) in the
    number of days, fully causal by construction (each row depends only
    on strictly earlier rows -- both the excitation state R and the
    baseline rate mu_t).
    """
    alpha, beta = hawkes_decay_params(n, halflife_days)
    idx = event_flag.index
    flags = event_flag.to_numpy(dtype=float)
    n_days = len(idx)
    lam = np.full(n_days, np.nan)
    r_state = 0.0
    cum_events = 0.0
    for i in range(n_days):
        if i >= min_days:
            rate_t = cum_events / i  # events in days [0, i-1] / i days observed
            mu_t = rate_t * (1.0 - n)
            lam[i] = mu_t + alpha * r_state
        r_state = r_state * np.exp(-beta) + flags[i]
        cum_events += flags[i]
    return pd.Series(lam, index=idx, name="hawkes_intensity")


def hawkes_intensity_zscore(lam: pd.Series,
                             detection_window_days: int = DETECTION_WINDOW_DAYS,
                             baseline_window_days: int = BASELINE_WINDOW_DAYS
                             ) -> pd.Series:
    """Causal z-score of a smoothed (detection-window mean) Hawkes
    intensity against its own trailing baseline -- same "smooth, then
    z-score against a trailing window" shape as R-85/R-86's trend_zscore,
    generalized to a level statistic (intensity itself already IS the
    regime-timing signal here; no separate trend transform is needed the
    way CSD/TE required one of variance/autocorrelation/TE)."""
    smoothed = lam.rolling(detection_window_days, min_periods=detection_window_days // 3).mean()
    baseline_mean = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).mean()
    baseline_std = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).std()
    z = (smoothed - baseline_mean) / baseline_std.replace(0.0, np.nan)
    return z


def align_daily_causal(daily: pd.Series, bars: pd.DataFrame) -> pd.Series:
    """Reindex a daily-cadence signal onto `bars`' 5-minute index, causally
    -- a bar at time T may only see the row for the most recent day that
    closed strictly before T's own day (IDENTICAL shift convention to
    `tradebot.data.align_onchain_causal` / r86_shared's own helper)."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


# --------------------------------------------------------- Step-A gate infra
#   `nearest_transition`, `episode_window`, `block_bootstrap_shifts`,
#   `truncation_causality_probe` copied verbatim from r82/85/86_shared.py.


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


def nearest_hawkes_alarm(z: pd.Series, window: pd.DatetimeIndex,
                          onset: pd.Timestamp, z_thresh: float = Z_THRESH
                          ) -> pd.Timestamp | None:
    """Timestamp of the first bar where the Hawkes intensity z-score
    crosses UP through `z_thresh`, closest to `onset` -- the Hawkes
    analogue of R-85/R-86's `nearest_csd_alarm` / `nearest_te_alarm`."""
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
    """Copied verbatim from r82/85/86_shared.py."""
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
    """Hard guard, same pattern as r81/r86/r88: the max timestamp anywhere
    this file touches must be strictly before OOS_START."""
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
