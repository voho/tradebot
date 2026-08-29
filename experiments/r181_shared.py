"""Shared, read-only pre-registration for the R-181 round (08-29).

DIRECTION, one sentence: R-174 diagnosed (and named, in its own closing
"next step" line) exactly why gating `kelly_regime_v4`'s exposure
INCREASES behind a sequential evidence test (Wald SPRT / GROW e-value)
never binds within a deadband episode's short lifetime -- BTC's raw
per-bar drift/volatility ratio is so small that the implied sample size
(1e4-1e5 bars) is two to three orders of magnitude longer than an episode
survives -- and it proposed the fix without building it: anchor the test
to "a statistic with a shorter natural resolution time than raw per-bar
drift-vs-volatility". This round implements that fix, two ways, reusing
R-174's own validated, statistically-neutral `run_asymmetric_gate` engine
unchanged (imported below, not copied) so both branches differ ONLY in
which statistic accumulates evidence, never in the gate mechanics that
were already tested.

- Conservative: resolve the gate against the CAUSAL, ONE-DAY-LAGGED daily
  aggregate log return instead of the raw 5-minute bar return, on the
  hypothesis that naive 5-minute realized variance is inflated by market-
  microstructure noise (Zhang, Mykland & Ait-Sahalia 2005; Ait-Sahalia,
  Mykland & Zhang 2005) relative to the true process, so the SPRT has been
  fighting noise it did not need to.
- Novel: keep 5-minute granularity (discard nothing) but feed the
  sequential test a two-time-scale, noise-bias-corrected variance
  estimate (TSRV, Zhang-Mykland-Ait-Sahalia 2005) and a GROW/safe-testing
  mixture alternative (Grunwald, de Heide & Koolen 2024) instead of
  R-174's fixed-TAU point alternative.

Full Step 1/Step 2 design (constraint attacked [ERR primary], the
citations, non-duplication against R-160/R-161/R-165/R-167/R-172/R-174/
R-179/R-180, simulability, named failure modes, and the pre-registered
decision rule for each branch, frozen BEFORE either branch is dispatched)
is in `experiments/r181_direction.md`.

This module is operator-authored and READ-ONLY to both branches (R-89
through R-180's own convention -- neither branch may edit this file or
each other's file). It provides two genuinely new, causal, self-tested
primitives specific to this round (a lagged daily-return broadcast for
the conservative branch, a two-time-scale realized-variance estimator for
the novel branch) in ADDITION to re-exporting R-174's neutral gate engine
and R-102's common inner-train/inner-validation/ETH-replication harness,
so neither branch has to re-derive a causal daily-broadcast (the exact
bug class R-172 found live) or a nontrivial rolling-window estimator from
scratch -- both are the single highest-risk part of this round and are
therefore built once, here, and self-tested against a truncation probe
before either branch touches them.

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r174_shared import (  # noqa: E402,F401
    ALPHA_GRID,
    ALPHA_PRIMARY,
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    GATE_MIN_DELAYS,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MU1 as MU1_PER_BAR,
    OOS_START,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SLICES,
    SPOT,
    TAU as TAU_PER_BAR,
    TargetStrategy,
    V4_BAND,
    V4_DEADBAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_bar_returns,
    causal_bar_sigma,
    causal_truncation_probe_series,
    compare,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_asymmetric_gate,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_symmetric_vol,
    v4_target,
    v4_vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND
assert abs(V4_DEADBAND - 0.10) < 1e-12, V4_DEADBAND

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch touches real
# data, DERIVED (not hand-copied) from inner-train only, per this
# project's own convention (e.g. r174_shared's own MU1).
# ------------------------------------------------------------------------
_btc_full = load_btc()
_train = _btc_full.loc[INNER_TRAIN_START:INNER_TRAIN_END]
_daily_close = _train["close"].resample("1D").last().dropna()
_daily_r = np.diff(np.log(_daily_close.to_numpy()))
MU1_DAILY = float(_daily_r.mean())              # inner-train mean daily log-return
TAU_DAILY = MU1_DAILY                           # novel branch's mixture-prior scale (daily units)
assert 1e-5 < abs(MU1_DAILY) < 1e-1 or MU1_DAILY == 0.0, MU1_DAILY  # sanity
del _btc_full, _train, _daily_close, _daily_r    # do not leak inner-train frames

# R-181's own two falsification-precheck thresholds (Step 1's "what would
# make it fail?", fixed now, before any code beyond this shared module
# runs) -- see r181_direction.md for the full argument behind each.
KAPPA_MIN = 1.10          # conservative precheck 1: noise-inflation ratio must exceed this
TSRV_MIN_REDUCTION = 0.10  # novel precheck 1: TSRV must read >=10% below naive RV


# ================================================================== (1)
# Conservative branch primitive: a CAUSAL, one-calendar-day-lagged daily
# aggregate log-return, broadcast onto every intraday bar. This is the
# exact daily-to-bar broadcast pattern R-172 found live with a one-day
# lookahead bug (a same-day statistic broadcast onto that same day's own
# bars); the discipline that fixes it is used here from the start: bar i,
# occurring on calendar day D, only ever sees day D-1's own (fully
# closed) return, never day D's.
# ==================================================================

def causal_daily_log_return_broadcast(df: pd.DataFrame) -> np.ndarray:
    """Bar i's evidence unit for the conservative branch: the log return
    of the most recently FULLY CLOSED calendar day as of bar i, held
    constant through the current day and updating only when the day
    rolls over. Never uses the current (possibly incomplete) day's own
    return. First calendar day's bars are NaN (no prior day)."""
    close = df["close"]
    day = close.index.floor("1D")
    daily_close = close.groupby(day).last()
    daily_ret = np.log(daily_close).diff()          # day D's own return
    daily_ret_lagged = daily_ret.shift(1)            # usable ON day D: D-1's return
    return daily_ret_lagged.reindex(day).to_numpy()


def causal_daily_log_sigma_broadcast(df: pd.DataFrame, span_days: int = 30) -> np.ndarray:
    """A causal per-DAY standard deviation of daily log returns (EWM,
    span in days), lagged and broadcast the same way as
    `causal_daily_log_return_broadcast` -- the Gaussian-model nuisance
    parameter the conservative branch's SPRT needs, at the SAME
    (daily) cadence as its evidence unit, rather than v4's own per-bar
    sigma (a unit mismatch the conservative branch must not make)."""
    close = df["close"]
    day = close.index.floor("1D")
    daily_close = close.groupby(day).last()
    daily_r = np.log(daily_close).diff()
    daily_sigma = daily_r.ewm(span=span_days, min_periods=5).std()
    daily_sigma_lagged = daily_sigma.shift(1)
    return daily_sigma_lagged.reindex(day).to_numpy()


# ================================================================== (2)
# Novel branch primitive: Zhang, Mykland & Ait-Sahalia (2005) two-scale
# realized variance (TSRV), a bias-corrected estimator of the true
# quadratic variation in the presence of market-microstructure noise.
# Computed causally over a trailing rolling window (no centred/future
# bars), K=5 non-overlapping 25-minute (5-bar) subsampling grids.
# ==================================================================

def _tsrv_window(log_close: np.ndarray, k: int = 5) -> float:
    """TSRV of one fixed window of log-prices (length n1+1 -> n1 returns).
    RV1 = sum of squared 1-bar (finest-scale) log returns.
    RVk = average, over the k possible offsets, of the sum of squared
          k-bar log returns on that offset's non-overlapping subsample.
    nbar = average number of observations across the k subsample grids.
    TSRV = RVk - (nbar/n1) * RV1  (the standard bias correction)."""
    r1 = np.diff(log_close)
    n1 = len(r1)
    if n1 < 2 * k:
        return float("nan")
    rv1 = float(np.sum(r1 * r1))
    rvk_list = []
    n_list = []
    for offset in range(k):
        sub = log_close[offset::k]
        if len(sub) < 2:
            continue
        rk = np.diff(sub)
        rvk_list.append(float(np.sum(rk * rk)))
        n_list.append(len(rk))
    if not rvk_list:
        return float("nan")
    rvk = float(np.mean(rvk_list))
    nbar = float(np.mean(n_list))
    tsrv = rvk - (nbar / n1) * rv1
    return tsrv


def two_scale_realized_variance(df: pd.DataFrame, window_bars: int = BARS_PER_DAY,
                                 k: int = 5) -> np.ndarray:
    """Causal, trailing-window TSRV, one estimate per bar, using ONLY the
    `window_bars` bars strictly before bar i (the window [i-window_bars, i),
    never including bar i itself) -- a rolling analogue of computing TSRV
    once per calendar day, but updated every bar so it feeds a per-bar
    sequential test at the same cadence as `causal_bar_sigma`. Returns the
    TOTAL variance realized over the window (not yet annualized or
    per-bar); callers divide by `window_bars` for a per-bar variance."""
    log_close = np.log(df["close"].to_numpy())
    n = len(log_close)
    out = np.full(n, np.nan)
    for i in range(window_bars, n):
        out[i] = _tsrv_window(log_close[i - window_bars:i + 1], k=k)
    return out


def two_scale_realized_variance_naive(df: pd.DataFrame,
                                      window_bars: int = BARS_PER_DAY) -> np.ndarray:
    """The K=1 (naive, uncorrected) realized variance over the identical
    trailing window -- the novel branch's own precheck baseline (does
    TSRV read measurably below this on real data?)."""
    log_close = np.log(df["close"].to_numpy())
    n = len(log_close)
    out = np.full(n, np.nan)
    for i in range(window_bars, n):
        r = np.diff(log_close[i - window_bars:i + 1])
        out[i] = float(np.sum(r * r))
    return out


def causal_bar_sigma_tsrv(df: pd.DataFrame, window_bars: int = BARS_PER_DAY,
                          k: int = 5) -> np.ndarray:
    """TSRV converted to a per-bar sigma, same units as `causal_bar_sigma`
    (a per-bar standard deviation, not annualized) -- a drop-in
    bias-corrected replacement for the novel branch's Gaussian-model
    nuisance parameter."""
    total_var = two_scale_realized_variance(df, window_bars=window_bars, k=k)
    per_bar_var = total_var / float(window_bars)
    per_bar_var = np.where(np.isfinite(per_bar_var) & (per_bar_var > 0), per_bar_var, np.nan)
    return np.sqrt(per_bar_var)


# ================================================================== (3)
# Historical rebalance-gap measurement -- conservative precheck 2 needs
# the MEASURED median gap between v4's own deadband-triggered exposure
# INCREASES on inner-train, not an assumed number.
# ==================================================================

def median_increase_episode_gap_days(df: pd.DataFrame) -> float:
    """Median number of days between consecutive bars where v4's own
    `apply_deadband(v4_raw_desired(df))` output INCREASES (a fresh
    episode start, in `run_asymmetric_gate`'s own sense), measured on the
    given frame. Used only as a diagnostic bound for the conservative
    branch's reachability precheck, not as a promotion criterion."""
    desired = v4_raw_desired(df)
    gated = apply_deadband(desired)
    is_increase = np.zeros(len(gated), dtype=bool)
    pos = 0.0
    for i, d in enumerate(gated):
        if d > pos + 1e-12:
            is_increase[i] = True
        pos = d
    idx = df.index.to_numpy()[is_increase]
    if len(idx) < 2:
        return float("nan")
    gaps_days = np.diff(idx).astype("timedelta64[s]").astype(float) / 86400.0
    return float(np.median(gaps_days))


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    # (1) Daily broadcast: causal (truncation probe) and correctly lagged
    # (a controlled two-day synthetic series: day 2's broadcast value must
    # equal day 1's own return, never day 2's).
    idx = pd.date_range("2020-01-01", periods=BARS_PER_DAY * 4, freq="5min", tz="UTC")
    rng = np.random.default_rng(181)
    innov = rng.normal(0.00002, 0.0005, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0003, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0003, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    assert causal_truncation_probe_series(causal_daily_log_return_broadcast, df)
    assert causal_truncation_probe_series(
        lambda d: causal_daily_log_sigma_broadcast(d, span_days=5), df)

    broadcast = causal_daily_log_return_broadcast(df)
    day = df.index.floor("1D")
    daily_close = df["close"].groupby(day).last()
    true_daily_r = np.log(daily_close).diff()
    for d_i, d in enumerate(daily_close.index):
        if d_i == 0:
            continue  # no prior day, broadcast must be NaN
        bar_mask = (day == d)
        vals = broadcast[bar_mask]
        expected = true_daily_r.iloc[d_i - 1]  # PRIOR day's own return, never this day's
        assert np.allclose(vals[np.isfinite(vals)], expected, atol=1e-10), (
            f"day {d}: broadcast leaks same-day or wrong-day return")
    assert np.all(np.isnan(broadcast[day == daily_close.index[0]])), (
        "first day must broadcast NaN (no prior day)")

    # (2) TSRV: causal (truncation probe) and mechanically does what it
    # claims -- on a synthetic price WITH added iid microstructure noise,
    # TSRV must read closer to the TRUE (noise-free) variance than the
    # naive full-resolution RV does, which is inflated by the noise.
    n = BARS_PER_DAY * 20
    idx2 = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    rng2 = np.random.default_rng(182)
    true_sigma = 0.0006
    true_log = np.cumsum(rng2.normal(0.0, true_sigma, n))
    noise = rng2.normal(0.0, true_sigma * 1.5, n)     # heavy microstructure noise
    noisy_close = np.exp(true_log + noise) * 10_000
    df2 = pd.DataFrame({"open": noisy_close, "high": noisy_close * 1.0002,
                        "low": noisy_close * 0.9998, "close": noisy_close,
                        "volume": 1.0}, index=idx2)

    assert causal_truncation_probe_series(
        lambda d: two_scale_realized_variance(d, window_bars=BARS_PER_DAY), df2)

    true_window_var = true_sigma ** 2 * BARS_PER_DAY
    tsrv = two_scale_realized_variance(df2, window_bars=BARS_PER_DAY)
    naive = two_scale_realized_variance_naive(df2, window_bars=BARS_PER_DAY)
    tail_tsrv = tsrv[-BARS_PER_DAY * 5:]
    tail_naive = naive[-BARS_PER_DAY * 5:]
    m = np.isfinite(tail_tsrv) & np.isfinite(tail_naive)
    assert m.sum() > 100, "not enough finite TSRV windows in self-test"
    mean_tsrv_err = float(np.mean(np.abs(tail_tsrv[m] - true_window_var)))
    mean_naive_err = float(np.mean(np.abs(tail_naive[m] - true_window_var)))
    assert mean_tsrv_err < mean_naive_err, (
        f"TSRV bias correction did not help on synthetic noisy data: "
        f"tsrv_err={mean_tsrv_err:.6g} naive_err={mean_naive_err:.6g}")

    sigma_tsrv = causal_bar_sigma_tsrv(df2, window_bars=BARS_PER_DAY)
    assert np.all(sigma_tsrv[np.isfinite(sigma_tsrv)] > 0), "TSRV sigma must be positive"

    # (3) rebalance-gap measurement runs and returns a plausible value on
    # a slice of real BTC inner-train data.
    btc = load_btc()
    sample = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    gap = median_increase_episode_gap_days(sample)
    assert np.isfinite(gap) and 0.0 < gap < 365.0, gap

    print(f"r181_shared self-test OK. MU1_DAILY={MU1_DAILY:.6g} "
          f"median_increase_gap_days(inner-train)={gap:.2f}")


if __name__ == "__main__":
    _self_test()
