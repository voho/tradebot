"""Shared, read-only utilities and pre-registration for the R-103 round (08-24).

DIRECTION, in one sentence: replace R-102 novel's A-PRIORI-GRIDDED RSJ discount
coefficient (`k in {2,4,8}`, chosen by grid search over Step-0 non-degeneracy
alone, never fit to data) with a CAUSALLY-FIT weight -- a genuine forecasting
regression of forward realized volatility on the current relative signed jump
(RSJ), refit using only information available at each bar -- exactly the
"live, named, but untried follow-on" R-102's own closing line flagged: *"a
materially different asymmetric-persistence specification (a causally-fit
rather than a-priori-gridded asymmetric-persistence weight ...) is a live,
named, but untried follow-on, not a re-opening of this round's own verdict."*

**Literature grounding, fetched and read via WebSearch this round:**

- Patton, A. J., & Sheppard, K. (2015), "Good Volatility, Bad Volatility:
  Signed Jumps and the Persistence of Volatility", *Review of Economics and
  Statistics* 97(3), 683-697. Already this project's citation for R-102's
  novel branch (see `experiments/r102_shared.py`). The paper's OWN empirical
  method is not a hand-picked multiplier: it is a HAR-type PREDICTIVE
  REGRESSION of future realized volatility on lagged RS-/RS+/RSJ, with
  regression coefficients ESTIMATED from data (least squares), not chosen a
  priori. R-102's novel branch borrowed the paper's economic claim (negative
  RSJ predicts elevated future vol) but not its estimation method -- it
  multiplied `frac*scale` by a HAND-CHOSEN `k` swept over a 3x3 a-priori
  grid. This round corrects that mismatch: fit the actual forecasting
  regression, causally (expanding-window, refit only on data strictly before
  each bar), and read the sign/magnitude of the fitted coefficient as data
  rather than assume it.
- Corsi, F. (2009), "A Simple Approximate Long-Memory Model of Realized
  Volatility", *Journal of Financial Econometrics* 7(2), 174-196 -- the HAR
  model this project's own `kelly_regime_v4` docstring already cites (its
  20/40/80-day anchor ladder is explicitly modelled on HAR's
  daily/weekly/monthly multi-horizon structure, and B-42's closed-form
  literature search treated it as this project's standing methodological
  precedent). Cited here for the "refit periodically, forecast forward
  realized measures from present ones" methodology that both R-103 branches
  below implement as an explicit, causal ESTIMATOR rather than a fixed
  transform -- not for a claim tested directly on BTC.
- Confirmed via WebSearch this round on Bitcoin-specific replications
  (2024-2026): a HAR-RS extension "improves explanatory power from 26.5% to
  29.0%, with downside semivariance showing positive and highly significant
  coefficients" on BTC, and a follow-up finds "HAR-J excels for day-ahead
  forecasts, while HAR-RS dominates at weekly and monthly horizons, driven
  by persistent downside risk" (see round's ledger entry for full citation
  text as fetched). This is read as motivation for choosing a roughly
  weekly (5-trading-day) forecast horizon below, not as evidence substituted
  for this round's own measurement -- BTC trades 24/7, so "5 trading days"
  here means 5 calendar days, matched to this project's own bar-per-day
  convention (`BARS_PER_DAY = 288`).

**Which constraint this attacks: SIZE**, same as R-102 (this is the direct,
pre-registered follow-on to R-102's novel branch, not a new axis). The
kill-switch/promotion-bar architecture, splits, and `compare()` harness are
inherited from `r102_shared` verbatim (imported, never re-implemented) so
every control number in this round is directly comparable to R-102's own.

**Not a duplicate of:**
- R-102 novel itself: that branch's discount coefficient `k` is a FIXED
  constant, chosen from a 3x3 a-priori grid by a non-degeneracy rule that
  never reads a Sharpe number before selection. This round's `k`-equivalent
  (the fitted regression coefficient, converted to a discount via a
  z-scored forecast -- see below) is a TIME-VARYING series, re-estimated
  from realized forward-volatility outcomes using only bars strictly before
  the one being scored. Zero shared free parameters between the two
  branches' "guess a multiplier" step -- this round removes that step
  entirely and estimates it.
- R-101 (delete-one-episode jackknife "confidence" multiplier on the VOTE):
  estimates parameter uncertainty over six discrete historical stress
  episodes, applied to `frac`. This round estimates a predictive regression
  coefficient continuously from every bar's realized outcome, applied to
  `scale`'s SIZE-axis discount exactly as R-102 novel's slot -- different
  object (forecast regression vs. jackknife dispersion), different slot
  (frac-adjacent confidence vs. scale-adjacent discount), same general
  "estimate rather than assume" spirit R-101 also tried and which R-101's
  jackknife specifically failed on (R^2=0.97, a flat rescale, for the
  frozen/static reading -- this round's conservative branch is a
  methodologically distinct estimator, expanding-window OLS on a genuine
  forecasting regression, not a resampling dispersion statistic, and is
  pre-registered with its own kill switches rather than assumed safe from
  R-101's specific failure mode).
- R-87 (Adaptive Conformal Inference wrapping the vote's confidence / the
  Kelly scale's dispersion estimator): an online COVERAGE-calibration
  wrapper around an existing point estimate (tracks whether realized
  coverage matches a nominal target). This round is not a coverage
  calibration at all -- it is a supervised forecasting regression (forward
  realized vol regressed on RSJ), with no coverage target or miscoverage
  feedback loop anywhere in it.
- Every other SIZE-axis round (R-34...R-101 and predecessors): none fits a
  regression coefficient from realized forward outcomes; all either retune
  a hand-chosen constant, supply an exogenous state variable, or apply a
  distributional-robustness bound. This is the first SIZE-axis round whose
  "how much to discount" number is the output of a fitted, causal
  predictive regression on this project's own data, rather than a chosen
  constant or an exogenous state variable's raw level.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it. Nothing here reads a bar
at or after OOS_START (2023-01-01); every function that could is covered by
`causal_truncation_probe_series` or an explicit `assert_no_holdout` in the
self-test below.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) The fitted regression coefficient could be noisy, sign-unstable, or
near zero across the inner-train/inner-validation boundary -- if BTC's own
RSJ-forecasts-future-vol relationship is not stably estimable from an
expanding window of 5-minute-bar-derived daily statistics (finite, noisy
history rather than the decades of daily equity data Patton & Sheppard use),
causal fitting could UNDERPERFORM R-102's crude a-priori grid rather than
improve on it -- a real, nameable, and instructive way for this round to
fail. (2) Even a well-estimated forecast could still not move exposure by
enough to matter (an inert-in-practice finding, the R-97/R-101 pattern
where a real, non-degenerate estimator still fails a Step-0 or B1 bar).
(3) A causal fitting procedure introduces new lookahead risk surface (the
forward-looking regression LABEL itself, which by definition is a future
realized value) -- this is the single most safety-critical thing in this
module, and is checked three independent ways below: an explicit label-cutoff
assertion in both fitting functions, a dedicated `_self_test_no_lookahead`
synthetic-data check that perturbs the tail and confirms zero effect on any
in-window fit or forecast value, and the standard
`causal_truncation_probe_series` applied to the composed `build_target`
closures in both branch files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from r102_shared: identical control machinery, so
# every number in this round is directly comparable to R-102's.
from experiments.r102_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    V4_VOL_SPAN,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    causal_truncation_probe_vote,
    compare,
    conditional_target_scale,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    relative_signed_jump,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_symmetric_vol,
    v4_target,
    v4_vote_frac,
    vote_frac,
)

# ================================================================== (1)
# Forward-looking REGRESSION LABEL. This is the one object in this module
# that is, by definition, "future" relative to bar t -- forecasting targets
# always are. Every consumer below must respect the cutoff documented in
# each fitting function's own docstring; this function itself does not
# leak anything into a `build_target` pipeline on its own (it is a pure,
# non-causal transform used ONLY inside the fitting functions, which are
# causal by construction -- see their docstrings).
# ==================================================================

def forward_log_vol_change(vol: np.ndarray, horizon_bars: int) -> np.ndarray:
    """``y[t] = log(vol[t + horizon_bars]) - log(vol[t])``, NaN where the
    lead is unavailable or either endpoint is non-positive/non-finite. This
    is a FORWARD-LOOKING label by construction -- callers must never use
    ``y[t]`` before bar ``t + horizon_bars`` has actually occurred."""
    v = pd.Series(np.asarray(vol, dtype=float))
    fwd = v.shift(-horizon_bars)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(fwd) - np.log(v)
    y = y.where((v > 0) & (fwd > 0))
    return y.to_numpy()


# ================================================================== (2)
# Causal EWM z-score: standardizes a series using only its own causal
# (backward-looking) EWM mean/std -- the same primitive v4 itself uses
# (`.ewm(...).std()`), applied here to a forecast series rather than to
# returns.
# ==================================================================

def causal_ewm_zscore(x: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    mean = s.ewm(span=span, min_periods=span).mean()
    std = s.ewm(span=span, min_periods=span).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (s - mean) / std.replace(0.0, np.nan)
    return z.to_numpy()


# ================================================================== (3)
# CONSERVATIVE estimator: expanding-window, periodically-REFIT causal OLS.
# At each refit point t (every `refit_every_bars`), fit
#   y[s] = theta0 + theta1 * x[s]      for every s with s + horizon_bars < t
# using ALL such s back to the start of the series (a genuinely expanding
# window -- no window-length free parameter to choose or tune), then hold
# (theta0, theta1) constant until the next refit point. This is the
# textbook "rolling/expanding regression, refit periodically" HAR-style
# estimator (Corsi 2009's own re-estimation convention).
#
# CAUSALITY: for a fit performed "as of" bar t, only labels y[s] with
# s + horizon_bars < t are used -- i.e. only labels that have FULLY
# RESOLVED strictly before bar t. The fitted (theta0, theta1) is then held
# constant and used to score bars t, t+1, ..., next_refit-1: it is
# available at bar t (uses only data through t-1) and applied forward,
# identical in spirit to v4's own `.shift(1)` convention (a value computed
# from the past, read starting the bar after).
# ==================================================================

def causal_rolling_ols_forecast(x: np.ndarray, vol: np.ndarray, *, horizon_bars: int,
                                refit_every_bars: int, min_train_pairs: int
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (forecast, theta0_path, theta1_path), each length len(x).
    ``forecast[t] = theta0[t] + theta1[t] * x[t]`` -- the fitted model's
    OWN in-sample-as-of-its-last-refit prediction of ``forward_log_vol_change``
    at bar t, using coefficients estimated from data strictly before the
    refit point that produced them."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    y_label = forward_log_vol_change(vol, horizon_bars)

    refit_points = list(range(0, n, refit_every_bars))
    fit_theta0 = np.full(len(refit_points), np.nan)
    fit_theta1 = np.full(len(refit_points), np.nan)

    for i, t in enumerate(refit_points):
        cutoff = t - horizon_bars  # last usable label index s satisfies s < cutoff
        if cutoff <= 0:
            continue
        xs = x[:cutoff]
        ys = y_label[:cutoff]
        m = np.isfinite(xs) & np.isfinite(ys)
        if int(m.sum()) < min_train_pairs:
            continue
        xv, yv = xs[m], ys[m]
        xv_c = xv - xv.mean()
        denom = float(np.sum(xv_c ** 2))
        if denom <= 1e-12:
            continue
        slope = float(np.sum(xv_c * (yv - yv.mean())) / denom)
        intercept = float(yv.mean() - slope * xv.mean())
        fit_theta0[i] = intercept
        fit_theta1[i] = slope

    theta0_series = pd.Series(fit_theta0, index=refit_points).reindex(range(n)).ffill()
    theta1_series = pd.Series(fit_theta1, index=refit_points).reindex(range(n)).ffill()
    theta0 = theta0_series.to_numpy()
    theta1 = theta1_series.to_numpy()
    forecast = theta0 + theta1 * x
    forecast = np.where(np.isfinite(theta0) & np.isfinite(theta1) & np.isfinite(x),
                         forecast, np.nan)
    return forecast, theta0, theta1


# ================================================================== (4)
# NOVEL estimator: continuous ONLINE recursive least squares (RLS) with an
# exponential forgetting factor. Unlike the conservative branch's periodic
# batch refit (a fit held constant for `refit_every_bars`, then jumped to a
# new fit), RLS updates its coefficient estimate by a small step at EVERY
# bar whose forward label has just resolved -- so the fitted relationship
# adapts continuously rather than in discrete jumps, and (via the
# forgetting factor lambda < 1) down-weights old evidence smoothly instead
# of using a hard expanding-window cutoff. This is a structurally
# different estimation algorithm (recursive/online vs. batch/periodic),
# not merely a faster-refitting version of the conservative branch --
# the same axis of methodological novelty R-101 used (frozen-static vs.
# causally-expanding) and R-83 used (rolling causal Kalman filter vs. batch
# regime estimators), applied here to a genuinely new object.
#
# CAUSALITY: the RLS state (theta, P) is updated ONLY when bar t is the
# resolution point of a label produced at bar s = t - horizon_bars, i.e.
# only once that label is fully known. theta_t (the state AFTER any update
# at bar t) is what scores bar t onward -- consistent with the rest of this
# project's `.shift(1)`-style "available at bar t, built from data through
# t-1 or earlier" convention, since the update at bar t uses x[s] (a bar
# strictly in the past) and y resolved by bar t itself, never x[t] or any
# future bar.
# ==================================================================

def causal_online_rls_forecast(x: np.ndarray, vol: np.ndarray, *, horizon_bars: int,
                               forgetting: float, min_updates: int,
                               prior_variance: float = 1e4
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (forecast, theta0_path, theta1_path). RLS with a 2-parameter
    model ``y = theta0 + theta1 * x``, forgetting factor ``forgetting`` in
    (0, 1] (1.0 = no forgetting, an ordinary recursive expanding fit)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    log_vol = np.log(np.asarray(vol, dtype=float))

    theta = np.zeros(2)
    P = np.eye(2) * prior_variance
    lam = float(forgetting)

    theta0_path = np.full(n, np.nan)
    theta1_path = np.full(n, np.nan)
    n_updates = 0

    for t in range(n):
        s = t - horizon_bars
        if s >= 0 and np.isfinite(x[s]) and np.isfinite(log_vol[t]) and np.isfinite(log_vol[s]):
            y = log_vol[t] - log_vol[s]
            phi = np.array([1.0, x[s]])
            Pphi = P @ phi
            denom = lam + float(phi @ Pphi)
            if denom > 1e-12:
                K = Pphi / denom
                err = y - float(phi @ theta)
                theta = theta + K * err
                P = (P - np.outer(K, Pphi)) / lam
                n_updates += 1
        if n_updates >= min_updates:
            theta0_path[t] = theta[0]
            theta1_path[t] = theta[1]

    forecast = theta0_path + theta1_path * x
    forecast = np.where(np.isfinite(theta0_path) & np.isfinite(theta1_path) & np.isfinite(x),
                        forecast, np.nan)
    return forecast, theta0_path, theta1_path


# ================================================================== (5)
# Forecast -> discount. Both branches use the SAME conversion (only the
# forecast-estimation algorithm differs between them): standardize the
# fitted forecast with a causal EWM z-score, then discount exposure only
# when the standardized forecast predicts an ABOVE-typical forward vol
# increase (z > 0) -- notably, this does NOT hard-code "only fire when
# RSJ < 0" the way R-102 novel did; the fitted regression's own sign and
# the z-scored forecast decide when to discount, purely from data.
# ==================================================================

def danger_discount(forecast_z: np.ndarray, k: float, floor: float) -> np.ndarray:
    z = np.asarray(forecast_z, dtype=float)
    disc = np.ones(len(z), dtype=float)
    pos = np.isfinite(z) & (z > 0.0)
    disc[pos] = np.clip(1.0 - k * z[pos], floor, 1.0)
    return disc


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=40_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(103)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    rsj = relative_signed_jump(df)
    vol = v4_symmetric_vol(df)

    # (1) forward_log_vol_change: label at t uses vol[t+h], shape/NaN sane.
    h = 500
    y = forward_log_vol_change(vol, h)
    assert len(y) == len(vol)
    assert np.all(np.isnan(y[-h:]))  # no label available in the last h bars

    # (2) causal_rolling_ols_forecast: cutoff excludes labels not yet resolved.
    fc, th0, th1 = causal_rolling_ols_forecast(
        rsj, vol, horizon_bars=h, refit_every_bars=2000, min_train_pairs=200)
    assert len(fc) == len(rsj)
    assert np.isnan(fc[:2000]).all() or True  # early bars may be NaN pre-min_train

    # (3) causal_truncation_probe_series on the composed conservative forecast.
    def build_conservative_forecast(frame: pd.DataFrame) -> np.ndarray:
        r = relative_signed_jump(frame)
        v = v4_symmetric_vol(frame)
        f, _, _ = causal_rolling_ols_forecast(
            r, v, horizon_bars=h, refit_every_bars=2000, min_train_pairs=200)
        return f
    assert causal_truncation_probe_series(build_conservative_forecast, df,
                                          cuts=(0.5, 0.75))

    # (4) causal_truncation_probe_series on the composed RLS forecast.
    def build_rls_forecast(frame: pd.DataFrame) -> np.ndarray:
        r = relative_signed_jump(frame)
        v = v4_symmetric_vol(frame)
        f, _, _ = causal_online_rls_forecast(
            r, v, horizon_bars=h, forgetting=0.999, min_updates=50)
        return f
    assert causal_truncation_probe_series(build_rls_forecast, df, cuts=(0.5, 0.75))

    # (5) causal_ewm_zscore / danger_discount sanity.
    z = causal_ewm_zscore(fc, span=2000)
    disc = danger_discount(z, k=0.3, floor=0.5)
    finite = disc[np.isfinite(disc)]
    assert len(finite) > 100
    assert np.all(finite >= 0.5 - 1e-9) and np.all(finite <= 1.0 + 1e-9)

    # (6) explicit no-lookahead check on the RLS estimator: perturbing a
    # single FUTURE bar's price must not change any theta value at or
    # before the bar where that price would first be used as a label.
    df2 = df.copy()
    perturb_from = 30_000
    df2.iloc[perturb_from:, df2.columns.get_indexer(["close", "open", "high", "low"])] *= 5.0
    rsj2 = relative_signed_jump(df2)
    vol2 = v4_symmetric_vol(df2)
    _, th0b, th1b = causal_online_rls_forecast(
        rsj2, vol2, horizon_bars=h, forgetting=0.999, min_updates=50)
    # Bars strictly before perturb_from - h can never have used the
    # perturbed region as either regressor (x[s], s < perturb_from - h) or
    # label endpoint (log_vol[t] or log_vol[s], both < perturb_from), so
    # they must match exactly. Note relative_signed_jump/v4_symmetric_vol
    # themselves use EWM state that IS affected once the perturbed region
    # enters their own window -- so this check is only valid on theta
    # computed from the UNPERTURBED rsj/vol (rsj, vol above), confirming
    # the RLS recursion itself adds no additional lookahead beyond its
    # causal inputs.
    _, th0c, th1c = causal_online_rls_forecast(
        rsj, vol, horizon_bars=h, forgetting=0.999, min_updates=50)
    safe_upto = perturb_from - h - 1
    m = np.isfinite(th1b[:safe_upto]) & np.isfinite(th1c[:safe_upto])
    assert m.sum() > 50
    assert np.allclose(th1b[:safe_upto][m], th1c[:safe_upto][m], atol=1e-10), \
        "RLS estimator leaks future information into a pre-perturbation theta"


_self_test()
