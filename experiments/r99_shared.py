"""Shared, read-only utilities for the R-99 round (08-23).

Idea in one sentence: a causal, rolling Barndorff-Nielsen & Shephard
bipower-variation decomposition of BTC's own realized quadratic variation
into a continuous and a discontinuous (jump) component, computed from this
project's own NATIVE 5-MINUTE bars -- the relative jump measure `RJ_t`
(Huang & Tauchen 2005) -- gives a NINTH structurally distinct theoretical
basis for regime-timing / detection-lag, tested against the identical six
dated historical BTC regime transitions R-82 (BOCPD), R-83 (Kalman LLT),
R-84 (vote-latch modulation), R-85 (critical slowing down), R-86 (transfer
entropy), R-96 (Hawkes) and R-98 (POT/GPD) all used, and, separately, as a
Step-0 sub-claim test for the novel branch (does a large-jump day predict
elevated forward realized loss?), the same architecture R-96's and R-98's
own novel branches used.

**The one genuine methodological first this round adds to the axis:**
every one of the eight prior regime-timing mechanisms (HMM, BOCPD, Kalman
LLT, vote-latch/volume, CSD, transfer entropy, Hawkes, POT/GPD) was
computed from DAILY-RESAMPLED close-to-close returns -- a choice, not a
requirement, for all eight. Bipower variation is not resample-able this
way: it is *defined* as a sum over the many intraday increments inside one
day, and is undefined (indistinguishable from realized variance, i.e.
always reports zero jump component) on a series sampled once per day. This
is therefore the first regime-timing construction on this axis that
requires this project's own native 5-minute cadence to exist as a
statistic at all, rather than choosing that cadence for convenience.

Literature grounding, fetched and read (or, where paywalled, corroborated
via multiple independent secondary sources quoting the same finding)
before being relied on:

- Barndorff-Nielsen, O. E., & Shephard, N. (2004), "Power and Bipower
  Variation with Stochastic Volatility and Jumps", *Journal of Financial
  Econometrics* 2(1), 1-37 -- introduces bipower variation `BV` as a
  jump-robust estimator of the continuous (integrated-variance) part of
  quadratic variation, converging to the SAME probability limit as
  realized variance `RV` when there are no jumps, but staying finite and
  jump-free when there are: the foundational result that makes
  `RV - BV -> jump quadratic variation` a valid decomposition.
- Barndorff-Nielsen, O. E., & Shephard, N. (2006), "Econometrics of Testing
  for Jumps in Financial Economics Using Bipower Variation", *Journal of
  Financial Econometrics* 4(1), 1-30 -- the formal jump-test framework:
  under the null of no jumps, `RV_t - BV_t` is asymptotically
  distributionally pinned at zero (up to estimation noise); a
  significantly positive gap is evidence of a realized jump on day `t`.
  Confirmed live via WebSearch this round (search: "Barndorff-Nielsen
  Shephard bipower variation jump test realized variance 2004 2006"):
  "The BNS test is based on the difference between realized variance and
  the corresponding bipower variation... The decision regarding the
  occurrence of a price jump can be based on testing whether RV - BV is
  significantly larger than zero" -- exactly the object `daily_jump_component`
  below computes, floored at zero per BNS's own convention (a negative
  finite-sample RV-BV gap is treated as no detected jump, not a negative
  quantity).
- Huang, X., & Tauchen, G. (2005), "The Relative Contribution of Jumps to
  Total Price Variance", *Journal of Financial Econometrics* 3(4), 456-499
  -- defines the RELATIVE jump measure `RJ_t = (RV_t - BV_t) / RV_t` used
  as this round's base statistic (bounded in [0, 1), scale-free across
  BTC's own decade of wildly varying volatility regimes, which a raw
  dollar/log-variance jump measure is not) and popularizes the BV
  finite-sample bias correction (`(pi/2) * M/(M-1)`) used in
  `daily_bipower_variation` below.
- Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007), "Roughing It
  Up: Including Jump Components in the Measurement, Modeling, and
  Forecasting of Return Volatility", *Review of Economics and Statistics*
  89(4), 701-720 -- confirmed live via WebSearch this round: finds the
  CONTINUOUS component of realized variance is materially more persistent
  (better forecastable from its own past) than the JUMP component, which
  behaves close to i.i.d. This is the pre-registered, literature-supplied
  reason to EXPECT the novel branch's Step-0 test to fail (named now, in
  section "WHAT WOULD MAKE THIS FAIL" below, before any number on this
  round's own data was computed) -- if jumps are themselves largely
  unforecastable and only weakly autocorrelated with their own recent
  past, there is no strong prior reason a single jump day should predict
  DAMAGE over the following 1-10 days either, independent of whether jump
  ACTIVITY clusters near regime transitions (the separate claim the
  conservative branch's Step-A gate tests).
- Shen, D., Urquhart, A., & Wang, P. (2020), "Forecasting the Volatility
  of Bitcoin: The Importance of Jumps and Structural Breaks", *European
  Financial Management* 26(5), 1294-1323 -- confirmed live via WebSearch
  this round (author/year/journal/volume/pages independently verified via
  the paper's own SSRN listing, abstract_id=3449756, posted 2019-09-07):
  applies exactly this jump-detection machinery to BTC specifically (not a
  generic-equity import) and finds jumps and structural breaks both matter
  for BTC's own volatility dynamics -- the same "BTC-specific, not an
  unrelated-domain import" role Ke, Yang & Tan (2022) played for R-98's
  GPD/POT round. A second, converging BTC-specific corroboration (found
  the same session, secondary-source confirmed rather than read directly
  behind its own paywall): a 2022 Finance Research Letters-family study on
  Bitcoin volatility predictability via jumps and regimes reports "jump
  intensity rising sharply prior to major market stress episodes" -- the
  DIRECT motivating claim for this round's conservative branch, read here
  as a hypothesis to TEST against this project's own six dated episodes
  and specific detection-lag bar, not as a result already established on
  this project's own data or holdout-honest methodology.

Attacks **N-approx-3** (a ninth theoretical basis for "has the regime
already broken", not a retune of a basis already tried) and, for the
novel branch specifically, **ERR** (a formal semimartingale decomposition
with a known asymptotic null, the same class of justification R-87's
conformal wrapper, R-97's Wasserstein-DRO ball and R-98's GPD/VaR used).

Not a duplicate of:
- R-01 (HMM), R-82 (BOCPD), R-83 (Kalman LLT), R-84 (vote-latch
  modulation), R-85 (CSD), R-86 (transfer entropy), R-96 (Hawkes), R-98
  (POT/GPD): eight regime-timing mechanisms from eight different fields.
  Bipower-variation jump/continuous decomposition is a ninth, sharing no
  mathematical machinery with any of them: no hidden discrete state, no
  Bayesian run-length posterior, no linear-Gaussian filter, no latch/width
  modulation, no fluctuation-statistics trend test, no information-
  theoretic directed-flow functional, no self-exciting conditional
  event-rate, and no asymptotic extreme-value tail theorem -- it is a
  semimartingale QUADRATIC-VARIATION DECOMPOSITION into continuous and
  discontinuous parts, a strictly different formal object, and (as noted
  above) the first of the nine that structurally REQUIRES intraday data
  rather than merely using it for convenience.
- R-93 (Grossman-Zhou) and R-97 (Wasserstein-DRO): both replace v4's
  `scale` factor directly. R-62 isolated `scale` as carrying NONE of v4's
  signature (four independent confirmations). This round does not touch
  `scale` at all: the conservative branch is a regime-timing ALARM fed
  additively into the vote (R-53/R-55's validated confirming-vote
  architecture, exactly as R-82/83/84/85/86/96/98 tested their own
  alarms), and the novel branch is a Step-0 measurement gate with no
  strategy code contingent on it passing -- had it passed, the
  contingent construction named in its own pre-registration is a
  discrete kill-switch/de-risking overlay (R-90/R-98's family), never a
  continuous scale multiplier.
- The fourteen INFO-axis rounds (R-44/R-53/R-54/R-55/R-58/R-73/R-74/R-75/
  R-79/R-81/R-84's INFO half/R-88/R-94/R-95): every one introduced a NEW
  EXTERNAL data channel used as a directional vote or brake. Neither
  branch here reads any data beyond the already-committed BTC OHLCV
  file's own `close` column at its own native 5-minute cadence -- bipower
  variation is a function of BTC's own intraday price path alone, same
  posture as R-82/83/84/85/86/96/98.
- R-62 (vote x scale factorization): motivates keeping the conservative
  branch confined to a regime-timing ALARM role (fed additively into the
  vote via the confirming-vote weight, never a `scale` retune), exactly
  the same posture R-96/R-98 held.
- R-84's conservative branch (raw log-volume z-score, unsigned, magnitude
  only) and R-88 (Binance taker buy/sell volume RATIO, a directional
  order-flow-imbalance quantity reported by the exchange): both are
  volume-derived. This round reads no volume column at all -- RV/BV/RJ
  are functions of the `close` price path's own squared and absolute
  log-returns, computed identically whether that day's volume was record
  weekend-thin or all-time-high.

This module is read-only utility, written by the operator before dispatch
(same convention as r82_shared.py through r98_shared.py). Neither branch
edits it. Contains: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s
3-anchor vote construction and the confirming-vote combination rule
(copied verbatim, ultimately from r82_shared.py); (2) a dependency-free
(numpy only) causal computation of daily realized variance, bias-corrected
bipower variation, the jump component and the relative jump measure `RJ`,
all built from NATIVE 5-minute log-returns grouped by UTC calendar day
(never resampled to daily closes -- that would discard exactly the
intraday increments BV needs); (3) the causal smoothed-z-score alarm
construction identical in shape to R-85/86/96/98's own; (4) the IDENTICAL
dated stress-episode table and detection-lag gate scaffolding
R-82/83/84/85/86/96/98 used (`STRESS_EPISODES`, `episode_window`,
`nearest_transition`, `block_bootstrap_shifts`), copied verbatim so all
eight rounds' numbers stay directly comparable; (5) the causal truncation
probe.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market RJ number
was computed, and are not retuned after seeing any result.

`DETECTION_WINDOW_DAYS_GRID = (30, 90, 180)`: how many days of daily `RJ`
values are averaged before z-scoring -- fast/medium/slow smoothing,
spanning v4's own fastest (20d) to slowest (80d) anchor horizons on either
side, the same "grid a construction-specific smoothing choice across a
literature-defensible range" logic R-98 used for its threshold quantile.
`BASELINE_WINDOW_DAYS_GRID = (365, 730, 1095)`: the trailing window the
z-score's own mean/std are computed over -- 1/2/3 years, IDENTICAL grid
and identical values to R-98's `FIT_WINDOW_DAYS_GRID`, chosen for direct
comparability across the axis rather than re-derived. `MIN_BARS_PER_DAY =
200`: out of a possible 288 5-minute bars, a UTC calendar day with fewer
than this many observed bars has its RV/BV/RJ set to NaN rather than
computed on a truncated, non-comparable day (a data-completeness floor,
disclosed, not swept -- this project's OHLCV file has essentially no
missing-bar gaps on BTC, so this is expected to bind rarely; any day it
does bind is reported, not silently patched). `Z_THRESH = 2.0`: identical
literature-standard value R-82/83/85/86/96/98 all used.

**Kill Switch A (degeneracy check, run once by the operator before any
per-episode number, the same posture as R-96's/R-97's/R-98's own Kill
Switch A):** does the alarm z-score actually cross `Z_THRESH=2.0` at
least once across the full 2017-2022 pre-holdout history, for each of the
9 grid cells, before any cell is chosen as PRIMARY? Run via
`experiments/r99_killswitch_a.py`. Result (2,191 calendar days, RJ mean
0.0927 / median 0.0694 / p95 0.2819 / max 0.7293): 8 of 9 cells fire at
least once; only the slowest corner (detection=180d, baseline=1095d,
max_z=1.86) is degenerate. The a-priori natural grid-CENTRE cell
(detection=90d, baseline=730d -- matching every predecessor's own
BASELINE_WINDOW_DAYS=730 convention and this round's own grid centre)
fires cleanly (max_z=3.45, 19 bars >= 2.0 across six years), so **no
substitution was needed** -- unlike R-98, where the natural centre cell
was itself the degenerate one. PRIMARY is therefore
`PRIMARY_DETECTION_WINDOW_DAYS=90, PRIMARY_BASELINE_WINDOW_DAYS=730`,
chosen for non-degeneracy alone, before any episode-level lead number
existed.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
the identical pattern that has now beaten eight consecutive mechanisms
built on eight different theoretical bases -- a statistic computed FROM
price (here: how much of today's realized variance is attributable to
discontinuous jumps, relative to a smooth trailing baseline) can only
shift once a large discontinuous move has already printed, which is
exactly the moment v4's own fixed-window anchor is also starting to
react, or later (a jump, by definition, resolves within a single day or
a handful of bars -- it has no structural reason to lead a slower
multi-week regime transition by weeks, the way a genuinely anticipatory
signal would have to). If bipower-variation jump activity also lags every
sudden 2020-2022 shock and only (at best) fires near the one slow 2018
build-up, that is the ninth independent mechanism converging on the same
conclusion the ledger's standing diagnosis already leans toward: this
six-episode gate is unwinnable by any estimator computed from this
project's own committed price history, whatever field it is drawn from,
and the finding is about the gate/dataset rather than about any one
technique. For the novel branch specifically: Andersen-Bollerslev-Diebold
(2007) themselves found the jump component of realized variance is close
to i.i.d. and far less persistent than the continuous component -- if
that finding holds here too, a single large-jump day should show LITTLE
OR NO significantly elevated forward N-day loss relative to the
unconditional baseline, and the branch must stop at its own pre-registered
Step-0 gate before any kill-switch/de-risking strategy is built or
backtested.
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

# ------------------------------------------------------- bipower/jump grid
DETECTION_WINDOW_DAYS_GRID = (30, 90, 180)
BASELINE_WINDOW_DAYS_GRID = (365, 730, 1095)
MIN_BARS_PER_DAY = 200
Z_THRESH = 2.0
# Rare-jump-day quantile for the novel branch's event flag (analogous role
# to R-98's VAR_PROB=0.99, fixed here rather than swept).
JUMP_EVENT_QUANTILE = 0.95

# Primary decision cell -- set by r99_killswitch_a.py's disclosed,
# pre-episode-number degeneracy check (see module docstring "Kill Switch A"
# above): the natural grid-centre cell fires cleanly, so it is PRIMARY
# unchanged.
PRIMARY_DETECTION_WINDOW_DAYS: int = 90
PRIMARY_BASELINE_WINDOW_DAYS: int = 730


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


# ------------------------------------------------- bipower variation core


def intraday_log_returns(df: pd.DataFrame) -> pd.Series:
    """Native 5-minute log return of `close`, NOT resampled -- the object
    bipower variation is defined over. Entirely causal by construction
    (row i depends only on rows <= i)."""
    r = np.log(df["close"]).diff()
    return r.rename("log_ret_5m")


def daily_rv_bv_jump(df: pd.DataFrame,
                      min_bars_per_day: int = MIN_BARS_PER_DAY) -> pd.DataFrame:
    """Daily realized variance (RV), bias-corrected bipower variation (BV),
    the jump component and the relative jump measure (RJ), from NATIVE
    5-minute log returns grouped by UTC calendar day.

    For day t with M observed 5-minute log returns r_1..r_M:
        RV_t = sum(r_i^2)                                    (Andersen &
                                                                Bollerslev 1998)
        BV_t = (pi/2) * (M/(M-1)) * sum_{i=2}^{M} |r_i| |r_{i-1}|
                                                              (Barndorff-
                                                                Nielsen &
                                                                Shephard
                                                                2004, the
                                                                mu_1^-2 = pi/2
                                                                finite-sample
                                                                bias
                                                                correction
                                                                and M/(M-1)
                                                                degrees-of-
                                                                freedom
                                                                adjustment
                                                                also used by
                                                                Huang &
                                                                Tauchen 2005
                                                                eq 2.4)
        J_t  = max(RV_t - BV_t, 0)                           (Barndorff-
                                                                Nielsen &
                                                                Shephard
                                                                2004/2006 --
                                                                floored at
                                                                zero, a
                                                                negative
                                                                finite-sample
                                                                gap is "no
                                                                detected
                                                                jump", not a
                                                                negative
                                                                quantity)
        RJ_t = J_t / RV_t                                    (Huang &
                                                                Tauchen
                                                                2005, bounded
                                                                in [0, 1))

    Days with fewer than `min_bars_per_day` observed 5-minute returns (a
    data-completeness floor, not swept) get NaN for every column rather
    than a truncated, non-comparable estimate.

    Entirely causal: day t's values depend only on bars dated on day t
    itself (which have all already closed by day t's own end), never on
    any later day.
    """
    r = intraday_log_returns(df).dropna()
    day = r.index.tz_convert("UTC").normalize() if r.index.tz is not None else r.index.normalize()
    r_abs = r.abs()

    grp = r.groupby(day)
    grp_abs = r_abs.groupby(day)

    m = grp.count()
    rv = grp.apply(lambda x: float(np.sum(x.to_numpy() ** 2)))

    def _bv(x: pd.Series) -> float:
        vals = x.to_numpy()
        n = len(vals)
        if n < 3:
            return float("nan")
        prod = np.abs(vals[1:]) * np.abs(vals[:-1])
        return float((np.pi / 2.0) * (n / (n - 1.0)) * np.sum(prod))

    bv = grp.apply(_bv)

    out = pd.DataFrame({"n_obs": m, "rv": rv, "bv": bv})
    bad = out["n_obs"] < min_bars_per_day
    out.loc[bad, ["rv", "bv"]] = np.nan

    jump = (out["rv"] - out["bv"]).clip(lower=0.0)
    rj = jump / out["rv"]
    out["jump"] = jump
    out["rj"] = rj
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    return out.rename_axis("date")


def rj_signal_zscore(rj: pd.Series, detection_window_days: int,
                      baseline_window_days: int) -> pd.Series:
    """Causal z-score of a smoothed (detection-window mean) relative jump
    measure `RJ` against its own trailing baseline -- same "smooth, then
    z-score against a trailing window" shape as R-85/86/96/98's own alarm
    construction, applied here to the bipower-variation jump share instead
    of variance/autocorrelation/TE/Hawkes-intensity/tail-shape."""
    smoothed = rj.rolling(detection_window_days, min_periods=max(3, detection_window_days // 3)).mean()
    baseline_mean = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).mean()
    baseline_std = smoothed.rolling(baseline_window_days, min_periods=detection_window_days).std()
    z = (smoothed - baseline_mean) / baseline_std.replace(0.0, np.nan)
    return z


def align_daily_causal(daily: pd.Series | pd.DataFrame, bars: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reindex a daily-cadence signal onto `bars`' 5-minute index, causally
    -- a bar at time T may only see the row for the most recent day that
    closed strictly before T's own day (IDENTICAL shift convention to
    `tradebot.data.align_onchain_causal` / r85/86/96/98_shared's own
    helper). Day t's RV/BV/RJ use ONLY bars dated on day t, all of which
    have closed by day t's own 23:55 bar -- shifting the whole day's value
    forward by one calendar day is a deliberately conservative (not
    minimal) causality margin, identical to every prior daily-signal round
    on this axis."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


def nearest_alarm(z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                   z_thresh: float = Z_THRESH) -> pd.Timestamp | None:
    """Timestamp of the first bar where `z` crosses UP through `z_thresh`,
    closest to `onset` -- the RJ analogue of R-85/86/96/98's own
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
#   `nearest_transition`, `episode_window`, `block_bootstrap_shifts`,
#   `truncation_causality_probe` copied verbatim from r82/85/86/96/98_shared.py.


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
    """Copied verbatim from r82/85/86/96/98_shared.py."""
    rng = np.random.default_rng(seed)
    block = int(block_days * BARS_PER_DAY)
    draws = []
    for _ in range(n_draws):
        shift = int(rng.integers(block, n_bars - block)) if n_bars > 2 * block else int(rng.integers(1, n_bars))
        draws.append((np.arange(n_bars) + shift) % n_bars)
    return draws


def block_bootstrap_shifts_daily(n_days: int, block_days: int, n_draws: int, seed: int) -> list[np.ndarray]:
    """Daily-cadence analogue of `block_bootstrap_shifts`, for the novel
    branch's Step-0 forward-loss test (which operates on the daily RV/BV/RJ
    frame directly, not on 5-minute bars)."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        shift = int(rng.integers(block_days, max(block_days + 1, n_days - block_days))) if n_days > 2 * block_days else int(rng.integers(1, max(2, n_days)))
        draws.append((np.arange(n_days) + shift) % n_days)
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
