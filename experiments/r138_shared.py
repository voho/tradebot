"""R-138: a formally-justified small-N significance test for this project's
own headline diagnosis -- that `kelly_regime_v4`'s edge over a risk-matched
hold concentrates in a handful of regime-transition episodes (L-01/R-62's
"roughly three sudden regime transitions"; the project's own N approx 3
constraint). Shared, frozen infrastructure for a two-branch parallel round.
Per ROUTINE.md's parallelism rules this file is neutral ground: both
branches import from it, NEITHER BRANCH EDITS IT, and it does not itself
compute a verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **N approx 3** (primary) -- a formally justified,
non-asymptotic significance test of the project's own "edge concentrates
in ~3 events" claim, which to date has only ever been asserted narratively
(L-01) or checked with ad hoc, informally-designed placebo controls
(R-127's and R-137's own bespoke random-day draws). **ERR** (secondary) --
this is a sixth, structurally distinct notion of statistical uncertainty
control, different from the five already in this project's toolkit:
e-processes (R-28/R-31), conformal prediction (R-87), anytime-valid
sequential testing (R-71/R-78/R-83), the stationary block bootstrap used
throughout `tradebot/inference.py`, and Wasserstein/DRO bounds (R-97).

Literature:
- Nguyen, P.A. & Wolf, M. (2026), "The permutation test for event studies
  with a small number of events," Empirical Economics 70 (also SSRN
  5804142; UZH working-paper version). Constructs a nonparametric
  permutation test for average/cumulative abnormal returns (AAR/CAAR) that
  stays valid when the number of events is as small as two, by permuting
  EVENT DATES across the eligible calendar rather than leaning on a CLT
  over the event count -- precisely this project's own N approx 3
  situation, named in the paper's own motivation.
- MacKinlay, A.C. (1997), "Event studies in economics and finance," J.
  Economic Literature 35(1), 13-39. The standard event-window convention
  (a short pre-event window plus a longer post-event window) this round's
  WINDOW_PRE_DAYS / WINDOW_POST_DAYS are drawn from, not fit to this data.
- Page, E.S. (1954), "Continuous inspection schemes," Biometrika 41(1/2).
  Hawkins, D.M. & Olwell, D.H. (1998), *Cumulative Sum Charts and Charting
  for Quality Improvement*. The two-sided CUSUM changepoint detector and
  its standard textbook constants (k = 0.5 sigma, h = 5 sigma), reused here
  at the SAME frozen values R-137 used for a different (cross-asset
  spread) CUSUM, for continuity and because they were textbook defaults
  there too, never swept against this data.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r138_conservative_stress_permutation.py`): apply the
  Nguyen-Wolf permutation test off the shelf to the already-frozen,
  narrative-selected `STRESS_EPISODES` list below (used verbatim, unedited,
  across R-82 through R-131 -- not re-selected for this round) as the event
  set, on `kelly_regime_v4` vs. its own realized-volatility-matched
  constant-exposure hold (`experiments/matched_hold.py`), 5x futures.

- NOVEL (`r138_novel_cusum_permutation.py`): identical permutation-test
  machinery, but the event set is produced by a CAUSAL, hindsight-free
  two-sided CUSUM changepoint detector run on BTC's own daily log-return
  MEAN (not the BTC/ETH spread R-137's own CUSUM watched -- a different
  series, answering a different question: single-asset drift regime
  breaks, not cross-asset divergence), at the frozen textbook constants
  above -- testing whether the edge-concentration finding survives when
  "which dates count as an event" is handed to an algorithm blind to
  narrative history and blind to v4's own performance, rather than to the
  operator's memory of what mattered.

**Not a duplicate of:**
- R-45 (purged-fold minimax model *selection*, not a hypothesis test on a
  fixed event set).
- R-118 (synthetic-path robust *calibration* of a sizing rule, not
  inference on regime episodes).
- R-71 / R-78 / R-83 (anytime-valid sequential betting-score accumulation
  over FORWARD time for B-06; this round is a fixed-sample, retrospective
  test on the existing backtest, not a stopping rule over accruing days).
- R-57 / R-63 (cross-sectional panel breadth across independent price
  PATHS across instruments; this round tests independent EVENTS in time on
  one instrument).
- R-101 (delete-one-episode jackknife feeding a sizing multiplier, not a
  hypothesis test).
- R-127 / R-137 (informal, bespoke random-day placebo controls built
  specifically for the ETH-idiosyncratic-event-excision question on five
  ETH constructions; neither implements the peer-reviewed Nguyen-Wolf
  construction, and neither is applied to `kelly_regime_v4`'s own primary
  edge-concentration claim at all -- R-137's own CUSUM detector watches the
  BTC/ETH SPREAD, a different series from this round's single-asset one).

**Is it simulable here?** Yes. Entirely computable from already-existing
backtest equity curves (`kelly_regime_v4`, and a constant-exposure hold
from `experiments/matched_hold.py`) over the training period
(`<= INNER_VAL_END`). No order book, no new data channel, no proxying.

**What would make this fail, named now, before any code:**

(a) The observed CAAR at the six `STRESS_EPISODES` may simply not clear
    the permutation null at conventional significance -- a legitimate
    NEGATIVE/METHOD outcome, not a bug. With only 6 events the permutation
    p-value's own achievable resolution is `1/(N_PERM+1)`; a result read as
    "significant" that is in fact sitting at that floor (e.g. literally
    zero permutation draws exceeded the observed statistic) is not
    informative about how far below alpha the true p-value actually sits.
    Guarded by fixing `N_PERM = 20000` up front (resolution `~5e-5`, far
    below `ALPHA = 0.05`) and by requiring (decision rule below) that the
    observed statistic beat MORE than `2` permutation draws, not zero.
(b) The test's own Type-I error may be miscalibrated on this data, because
    5m-bar-derived BTC daily returns are autocorrelated and
    volatility-clustered, which can violate the exchangeability-under-the-
    null the permutation argument leans on. Guarded by
    `empirical_type1_rate` below: apply the IDENTICAL test procedure to
    `N_CALIBRATION_TRIALS` independently-drawn random 6-date pseudo-event
    sets (each compared against its OWN freshly-built permutation null,
    same `N_PERM`), and require the empirical false-positive rate at
    `ALPHA = 0.05` to land in `[0.02, 0.09]` -- the two-sided band implied
    by binomial sampling noise at `N_CALIBRATION_TRIALS = 200`
    (`SE = sqrt(0.05*0.95/200) ~= 0.0154`, `0.05 +/- 2*SE`). A branch whose
    calibration check falls outside that band is VOIDED (not scored as
    negative -- a broken instrument, per ROUTINE.md's void-don't-score
    rule), and must report that plainly rather than trusting its own
    p-value.
(c) NOVEL branch specifically: the CUSUM detector may reproduce R-137's own
    finding of near-zero overlap with the narrative `STRESS_EPISODES` set,
    meaning its event set could be dominated by volatility-clustering false
    breaks rather than genuine drift-regime transitions. Guarded by
    reporting the CUSUM event count and its date overlap against
    `STRESS_EPISODES` explicitly (diagnostic, does not gate promotion by
    itself), and by running the SAME calibration check as (b) on the
    CUSUM-derived event COUNT (if the detector flags a different number of
    events than 6, the calibration check uses that actual count).

**Falsification test, pre-registered:** the standard B4 test this
project's SIZE/ERR/COST programme has used since R-59 -- does the sign and
significance of the finding replicate on ETH? Concretely: re-run the
IDENTICAL permutation-test machinery on ETH's own `kelly_regime_v4` vs.
matched-hold edge (conservative: same `STRESS_EPISODES` calendar dates,
restricted to ETH's available history from 2019-03; novel: the CUSUM
detector re-run on ETH's own daily log-return series, its own frozen
constants, producing ETH's own event dates).

**Decision rule, pre-registered verbatim, evaluated identically by both
branches on the training period (`<= INNER_VAL_END`); NO bar at or after
`OOS_START = 2023-01-01` may be read by either branch during Step 3:**

A branch's finding is promotable (worth writing up as a genuine
methodological result, not a NEGATIVE) only if ALL of:

1. **C1 (calibration).** `empirical_type1_rate(...)` at `ALPHA=0.05` lands
   in `[0.02, 0.09]` on BTC. If not, the branch is VOIDED for that market
   and must say so rather than trust its p-value there.
2. **C2 (significance, resolution-aware).** The two-sided permutation
   p-value for the observed CAAR is `< 0.05`, AND the observed statistic
   is beaten by (i.e. is more extreme than) more than 2 of the `N_PERM`
   permutation draws (guards against reading a floor-resolution `p < 1/
   (N_PERM+1)` as more informative than it is).
3. **C3 (ETH replication).** The identical procedure on ETH passes its own
   C1 and produces a p-value `< 0.10` with the SAME SIGN as BTC's CAAR
   (a looser bar than BTC's, matching this project's convention of a wider
   ETH tolerance given its shorter/noisier history -- e.g. R-68's own
   asymmetric thresholds).

Anything else is NEGATIVE or METHOD (a genuine, informative finding about
this project's own small-N claim or about the test's applicability here,
even without a promotion) -- per ROUTINE.md's standing culture, exactly
like R-131 through R-137. **This round produces no strategy code change
regardless of outcome.** If, and only if, a branch clears the decision
rule above, the reusable permutation-test function is proposed for
addition to `tradebot/inference.py` as new evidence-generating
infrastructure (parallel to how `deflated_sharpe_ratio` and
`paired_bootstrap` already live there) -- a METHOD result, not a
PROMOTED strategy, matching R-134's precedent (`broker.REBALANCE_DEADBAND`
became a `MarketSpec` field on a "fix adopted" verdict with no promoted
strategy attached).

No bar at or after `OOS_START = 2023-01-01` may be read by either branch
during Step 3. Step 4 (holdout) is run ONLY if a branch clears the
decision rule above on the training period, per ROUTINE.md step 4's
pre-registration discipline -- and even then only as a confirmatory,
already-pre-registered check, never as a second chance to find
significance a training-period run did not.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round: inner-train / inner-
# validation only. The holdout (>= OOS_START) is never read during Step 3.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# The primary market: 5x futures, this project's own headline comparison
# for kelly_regime_v4 ("current leader ... $156.2K", STRATEGIES.md).
PRIMARY_MARKET = FUTURES

# ----------------------------------------------------------------------
# STRESS_EPISODES -- copied VERBATIM from experiments/r131_shared.py
# (itself unchanged since R-82). Not re-selected for this round: reusing a
# list already frozen and used dozens of times is what keeps the
# conservative branch's event set free of this round's own hindsight.
# ----------------------------------------------------------------------
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# Event-window convention (MacKinlay 1997's standard shape: a short
# pre-event window, a longer post-event window), fixed before any real
# statistic was computed.
WINDOW_PRE_DAYS = 5
WINDOW_POST_DAYS = 20

# Permutation / calibration budget, fixed before any real statistic.
N_PERM = 20_000
N_CALIBRATION_TRIALS = 200
ALPHA = 0.05
CALIBRATION_BAND = (0.02, 0.09)

# CUSUM constants (Page 1954; Hawkins & Olwell 1998), textbook defaults,
# identical values to R-137's (different) spread CUSUM, never swept
# against this round's data.
CUSUM_TRAIL_DAYS = 90
CUSUM_K_MULT = 0.5
CUSUM_H_MULT = 5.0


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_btc_train():
    df, label = load_dataset(ROOT / "data", "spot")
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


def load_eth_train():
    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


# ----------------------------------------------------------------------
# Vol-matched constant-exposure hold (experiments/matched_hold.py),
# matched once on the WHOLE training period (2017-2022) -- a single,
# stable match, not a per-25-day-window one (too few observations in a
# 25-day window to solve `c` stably).
# ----------------------------------------------------------------------

def _run(strategy, df, market, start=None, end=None, label=""):
    return run_period(strategy, df, start=start, end=end, market=market,
                      start_balance=1_000.0, data_label=label)


def realized_vol_daily(equity: pd.Series) -> float:
    r = daily_returns(equity).to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(365.25))


def solve_matched_c(target_vol: float, df: pd.DataFrame, market: MarketSpec,
                    start=None, end=None, label="", tol: float = 0.02,
                    max_iter: int = 8) -> tuple[float, float]:
    """Solve for the constant exposure `c` whose realized daily volatility
    matches `target_vol`, on `df[start:end]`. Returns `(c, achieved_vol)`;
    `achieved_vol` is returned even when the tolerance is not met, so the
    caller can decide whether to void the cell (never silently pin)."""
    c_max = market.leverage
    c = min(0.5, c_max)
    for _ in range(max_iter):
        res = _run(ConstantExposureHold(c, static=False), df, market,
                   start, end, label)
        vol = realized_vol_daily(res.equity)
        if not np.isfinite(vol) or vol <= 0:
            return c, vol
        if abs(vol - target_vol) <= tol * target_vol:
            return c, vol
        c = float(np.clip(c * (target_vol / vol), 1e-3, c_max))
        if c >= c_max and vol < target_vol:
            return c, vol
    return c, vol


def candidate_and_matched_daily_logret(df: pd.DataFrame, market: MarketSpec,
                                       label: str = "") -> tuple[pd.Series, pd.Series, float, float]:
    """`kelly_regime_v4` vs. its own vol-matched constant-exposure hold,
    over the FULL training period. Returns `(cand_log_ret, matched_log_ret,
    c, achieved_vol)` -- daily log-return series, aligned, plus the solved
    match diagnostics so a caller can check `matched` before trusting AR."""
    cand_res = _run(get_strategy("kelly_regime_v4"), df, market,
                    INNER_TRAIN_START, INNER_VAL_END, label)
    target_vol = realized_vol_daily(cand_res.equity)
    c, achieved_vol = solve_matched_c(target_vol, df, market,
                                      INNER_TRAIN_START, INNER_VAL_END, label)
    matched_res = _run(ConstantExposureHold(c, static=False), df, market,
                       INNER_TRAIN_START, INNER_VAL_END, label)
    cand_simple = daily_returns(cand_res.equity)
    matched_simple = daily_returns(matched_res.equity)
    idx = cand_simple.index.intersection(matched_simple.index)
    cand_log = np.log1p(cand_simple.loc[idx].clip(lower=-0.999))
    matched_log = np.log1p(matched_simple.loc[idx].clip(lower=-0.999))
    return cand_log, matched_log, c, achieved_vol


# ----------------------------------------------------------------------
# CAAR statistic and Nguyen-Wolf (2026) permutation test.
#
# Implemented over a prefix-sum of `ar` plus `np.searchsorted`, rather than
# repeated `pandas.Series.loc` boolean masking, so that `N_PERM = 20000`
# draws (and the `N_CALIBRATION_TRIALS x N_PERM` draws C1 needs) run in
# seconds rather than hours. This is a performance choice only -- the
# statistic and the permutation procedure it computes are unchanged from
# the mechanism described above and in the module docstring.
# ----------------------------------------------------------------------

_NS_PER_DAY = np.int64(86_400_000_000_000)


def _window_bounds(ar_index: pd.DatetimeIndex, event_date, pre: int, post: int):
    ts = pd.Timestamp(event_date)
    if ts.tzinfo is None and ar_index.tz is not None:
        ts = ts.tz_localize(ar_index.tz)
    lo = ts - pd.Timedelta(days=pre)
    hi = ts + pd.Timedelta(days=post)
    return lo, hi


def _prefix_sum(ar: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """`(t, cs)`: sorted int64-ns timestamps and a length-(n+1) cumulative
    sum with `cs[0] = 0`, so the sum of `ar` over half-open index range
    `[i, j)` is `cs[j] - cs[i]`."""
    t = ar.index.values.astype("datetime64[ns]").astype(np.int64)
    order = np.argsort(t)
    t = t[order]
    vals = ar.to_numpy(dtype=float)[order]
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    return t, cs


def _car_batch(t: np.ndarray, cs: np.ndarray, event_dates_ns: np.ndarray,
               pre: int, post: int) -> np.ndarray:
    """Vectorized CAR over an array of event dates (any shape), inclusive
    of both window edges -- matches the semantics `car_for_event` used."""
    lo = event_dates_ns - np.int64(pre) * _NS_PER_DAY
    hi = event_dates_ns + np.int64(post) * _NS_PER_DAY
    left = np.searchsorted(t, lo, side="left")
    right = np.searchsorted(t, hi, side="right")
    car = cs[right] - cs[left]
    return np.where(right > left, car, np.nan)


def car_for_event(ar: pd.Series, event_date, pre: int = WINDOW_PRE_DAYS,
                  post: int = WINDOW_POST_DAYS) -> float:
    """Cumulative abnormal return in `[event_date - pre, event_date + post]`."""
    t, cs = _prefix_sum(ar)
    lo, hi = _window_bounds(ar.index, event_date, pre, post)
    val = _car_batch(t, cs, np.array([lo.value], dtype=np.int64), pre, post)[0]
    return float(val) if np.isfinite(val) else float("nan")


def caar_statistic(ar: pd.Series, event_dates, pre: int = WINDOW_PRE_DAYS,
                   post: int = WINDOW_POST_DAYS) -> float:
    """Cumulative AVERAGE abnormal return across `event_dates`."""
    t, cs = _prefix_sum(ar)
    ns = np.array([pd.Timestamp(d).tz_localize(ar.index.tz).value
                   if pd.Timestamp(d).tzinfo is None and ar.index.tz is not None
                   else pd.Timestamp(d).value for d in event_dates], dtype=np.int64)
    cars = _car_batch(t, cs, ns, pre, post)
    cars = cars[np.isfinite(cars)]
    if len(cars) == 0:
        return float("nan")
    return float(np.mean(cars))


def eligible_pseudo_dates(ar: pd.Series, real_events, pre: int, post: int,
                          buffer_days: int = 3) -> pd.DatetimeIndex:
    """Calendar days eligible to be drawn as a pseudo-event: far enough
    from either series edge for a full window, and not overlapping any
    real event's own window (plus a small buffer) -- so the null is built
    from genuinely different dates, not near-duplicates of the real ones."""
    idx = ar.index
    lo_bound = idx.min() + pd.Timedelta(days=pre + buffer_days)
    hi_bound = idx.max() - pd.Timedelta(days=post + buffer_days)
    eligible = idx[(idx >= lo_bound) & (idx <= hi_bound)]
    exclude = pd.DatetimeIndex([])
    for d in real_events:
        lo, hi = _window_bounds(idx, d, pre + buffer_days, post + buffer_days)
        exclude = exclude.union(eligible[(eligible >= lo) & (eligible <= hi)])
    return eligible.difference(exclude)


def permutation_test(ar: pd.Series, event_dates, *, pre: int = WINDOW_PRE_DAYS,
                     post: int = WINDOW_POST_DAYS, n_perm: int = N_PERM,
                     seed: int = 138, exclude: pd.DatetimeIndex | None = None
                     ) -> dict:
    """Nguyen & Wolf (2026): permute event DATES across the eligible
    calendar, recompute CAAR under each permutation, and read the two-sided
    p-value off the resulting empirical null. `exclude` lets a caller (the
    calibration check) draw pseudo-"real" events from a calendar that
    already excludes a different trial's events, if needed; ordinarily
    left `None` and computed from `event_dates` itself.
    """
    observed = caar_statistic(ar, event_dates, pre, post)
    real_events = list(event_dates) if exclude is None else list(exclude)
    pool = eligible_pseudo_dates(ar, real_events, pre, post)
    n_events = len(list(event_dates))
    rng = np.random.default_rng(seed)
    if len(pool) < n_events:
        return {"observed": observed, "pvalue": float("nan"), "n_perm": 0,
                "n_exceed": 0, "pool_size": len(pool)}
    t, cs = _prefix_sum(ar)
    pool_ns = pool.values.astype("datetime64[ns]").astype(np.int64)
    draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        sample_ns = rng.choice(pool_ns, size=n_events, replace=False)
        cars = _car_batch(t, cs, sample_ns, pre, post)
        cars = cars[np.isfinite(cars)]
        draws[i] = float(np.mean(cars)) if len(cars) else float("nan")
    draws = draws[np.isfinite(draws)]
    n_exceed = int(np.sum(np.abs(draws) >= abs(observed)))
    pvalue = (1.0 + n_exceed) / (1.0 + len(draws))
    return {"observed": observed, "pvalue": float(pvalue), "n_perm": len(draws),
            "n_exceed": n_exceed, "pool_size": len(pool)}


def empirical_type1_rate(ar: pd.Series, n_events: int, *,
                         pre: int = WINDOW_PRE_DAYS, post: int = WINDOW_POST_DAYS,
                         n_perm: int = N_PERM,
                         n_trials: int = N_CALIBRATION_TRIALS,
                         alpha: float = ALPHA, seed: int = 2138) -> dict:
    """C1: empirical false-positive rate of the permutation test on THIS
    series' own dependence structure. Draws `n_trials` independent random
    `n_events`-date pseudo-"observed" sets (none of them the real events --
    there are none here, this checks the test in isolation), runs the full
    permutation procedure on each against its own freshly-built null, and
    reports the fraction with `pvalue < alpha`. A well-calibrated test
    should reject at ~`alpha`; this repo requires landing in
    `CALIBRATION_BAND`.
    """
    rng = np.random.default_rng(seed)
    pool = eligible_pseudo_dates(ar, [], pre, post)
    pool_arr = pool.to_numpy()
    if len(pool_arr) < n_events * 2:
        return {"rate": float("nan"), "n_trials": 0}
    rejects = 0
    valid = 0
    for t in range(n_trials):
        fake_events = rng.choice(pool_arr, size=n_events, replace=False)
        result = permutation_test(ar, fake_events, pre=pre, post=post,
                                  n_perm=n_perm, seed=int(rng.integers(0, 2**31 - 1)))
        if np.isfinite(result["pvalue"]):
            valid += 1
            if result["pvalue"] < alpha:
                rejects += 1
    rate = rejects / valid if valid else float("nan")
    return {"rate": rate, "n_trials": valid, "rejects": rejects}


def price_daily_logret(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Daily log returns of a raw OHLCV price series (not an equity curve) --
    what the NOVEL branch's CUSUM detector watches, as distinct from the AR
    series (candidate minus matched hold) the permutation test watches."""
    simple = daily_returns(df[price_col])
    return np.log1p(simple.clip(lower=-0.999))


# ----------------------------------------------------------------------
# NOVEL branch only: causal two-sided CUSUM on a single series' own daily
# log-return MEAN (distinct from R-137's cross-asset spread CUSUM).
# ----------------------------------------------------------------------

def causal_cusum_breaks(daily_logret: pd.Series, *, trail_days: int = CUSUM_TRAIL_DAYS,
                        k_mult: float = CUSUM_K_MULT, h_mult: float = CUSUM_H_MULT
                        ) -> list:
    """Two-sided CUSUM (Page 1954) on `daily_logret`'s own mean, using only
    a TRAILING `trail_days` window at each point to estimate the reference
    mean/sigma (causal: bar t's flag uses only data up to and including t).
    Returns a list of `pd.Timestamp` flagged as a break. `k`/`h` are in
    units of the trailing sigma, recomputed at every bar (a standard
    "self-normalizing" CUSUM, matching R-137's own convention).
    """
    x = daily_logret.dropna()
    breaks = []
    pos = 0.0
    neg = 0.0
    last_reset = 0
    for i in range(trail_days, len(x)):
        trail = x.iloc[max(0, i - trail_days):i]
        mu = float(trail.mean())
        sigma = float(trail.std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        k = k_mult * sigma
        h = h_mult * sigma
        val = float(x.iloc[i]) - mu
        pos = max(0.0, pos + val - k)
        neg = min(0.0, neg + val + k)
        if pos > h or -neg > h:
            breaks.append(x.index[i])
            pos = 0.0
            neg = 0.0
            last_reset = i
    return breaks
