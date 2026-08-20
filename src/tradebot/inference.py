"""Trials-aware inference: bootstrap intervals, deflated Sharpe, purged CV.

Every headline in this repo is a **point estimate selected from a search**.
The comparison table ranks 25 strategies on one path; the fee study ran 32
configurations; the e-process round ran 24. A number chosen that way is
biased upward by construction, and reporting it without an interval or a
trials adjustment overstates it — which is exactly how 28-of-32 in-sample
winners became 0-of-28 out-of-sample (R-12 in ``docs/LEDGER.md``).

This module holds the three corrections, as plain functions over an
**equity curve** rather than over a strategy, so they apply to anything
the framework can back-test:

1. :func:`stationary_bootstrap_indices` and the helpers built on it —
   block-bootstrap confidence intervals for Sharpe, drawdown, and the
   *paired* difference between two strategies on identical resamples.
   Paired matters: two strategies on the same market share most of their
   variance, and comparing independent intervals throws that away.
2. :func:`ledoit_wolf_sharpe_diff` and :func:`bootstrap_studentized_sharpe_diff`
   — Ledoit & Wolf (2008). ``paired_bootstrap`` above answers "is the
   difference between two strategies' *log growth* distinguishable from
   zero"; these answer the same question for the *Sharpe ratio*, and do it
   by **studentizing** the difference by an estimate of its own standard
   error before doing inference, rather than treating a raw percentile
   interval as already pivotal — the correction that gives L&W's test its
   improved finite-sample coverage on autocorrelated, heavy-tailed returns.
   Two independent estimators of that standard error are provided —
   ``ledoit_wolf_sharpe_diff``'s Parzen-kernel HAC estimate (the paper's own
   construction) and ``bootstrap_studentized_sharpe_diff``'s nonparametric
   stationary-bootstrap estimate, at this module's own 30-day-block
   convention — because R-70 (``docs/LEDGER.md``) found they do not always
   agree on this project's data, and that disagreement is itself the honest
   answer on a cell where it occurs, not a discrepancy to paper over by
   picking whichever one says "significant".
3. :func:`deflated_sharpe_ratio` — Bailey & López de Prado (2014). How
   high a Sharpe the *best of N trials* reaches by luck alone when the
   true Sharpe is zero, and whether the observed one clears it.
4. :func:`cpcv_splits` with :func:`purged_train_mask` — combinatorially
   purged cross-validation (López de Prado 2018), which turns one
   walk-forward number into a distribution of out-of-fold paths.

**Everything here works on daily returns, not 5m bars.** A million
autocorrelated bars is not a million observations; a block bootstrap over
them is both intractable and dishonest about the effective sample size.
Resampling daily with a 30-day mean block is what produced the ±0.2
Sharpe noise floor (R-20), and it is the convention the whole module
follows — so a Sharpe reported here is a *daily* Sharpe and will differ
slightly from the per-bar figure in the comparison table.

References
----------
Politis & Romano (1994), "The stationary bootstrap", JASA 89(428).
Ledoit & Wolf (2008), "Robust performance hypothesis testing with the
Sharpe ratio", J. Empirical Finance 15(5), 850-859.
Newey & West (1994), "Automatic lag selection in covariance matrix
estimation", Review of Economic Studies 61(4), 631-653.
Bailey & López de Prado (2014), "The deflated Sharpe ratio", J. Portfolio
Management 40(5).
López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 & 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e as EULER_E
from math import erf, sqrt

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------- basic stats

def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple daily returns from a bar-frequency equity curve.

    The last equity value of each UTC day is taken as that day's close. A
    liquidated account floors at zero; once there, returns are zero rather
    than NaN, so the curve stays usable and the drawdown stays 100%.
    """
    daily = equity.resample("1D").last().dropna()
    prev = daily.shift(1)
    rets = (daily - prev) / prev.where(prev > 0)
    return rets.iloc[1:].fillna(0.0)


def _scalarize(out: np.ndarray, ndim: int):
    """Return a float for a 1-D input, an array for a stack of resamples."""
    return float(out) if ndim == 1 else out


def annualized_sharpe(rets: np.ndarray,
                      periods_per_year: float = DAYS_PER_YEAR):
    """Annualized Sharpe of a return series (rf = 0).

    Accepts a 1-D series or an ``(n_boot, n)`` stack of bootstrap
    resamples, and reduces along the last axis — the bootstrap loops in
    this module apply every statistic to the whole stack at once, so each
    one has to be axis-aware.
    """
    rets = np.asarray(rets, dtype=float)
    if rets.shape[-1] < 3:
        return _scalarize(np.zeros(rets.shape[:-1]), rets.ndim)
    sd = rets.std(axis=-1, ddof=1)
    safe = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
    out = np.where((sd > 0) & np.isfinite(sd),
                   rets.mean(axis=-1) / safe * np.sqrt(periods_per_year), 0.0)
    return _scalarize(out, rets.ndim)


def max_drawdown_from_returns(rets: np.ndarray):
    """Largest peak-to-trough drop of the compounded path, in percent."""
    rets = np.asarray(rets, dtype=float)
    if rets.shape[-1] == 0:
        return _scalarize(np.zeros(rets.shape[:-1]), rets.ndim)
    equity = np.cumprod(1.0 + np.clip(rets, -1.0, None), axis=-1)
    peaks = np.maximum.accumulate(equity, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    return _scalarize(np.nanmax(dd, axis=-1) * 100.0, rets.ndim)


def total_log_return(rets: np.ndarray):
    """Compounded log return of a return series, robust to a wipeout."""
    rets = np.asarray(rets, dtype=float)
    if rets.shape[-1] == 0:
        return _scalarize(np.zeros(rets.shape[:-1]), rets.ndim)
    growth = np.clip(1.0 + rets, 1e-12, None)
    return _scalarize(np.log(growth).sum(axis=-1), rets.ndim)


# ---------------------------------------------------------------- bootstrap

def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """``(n_boot, n)`` index matrix for the stationary bootstrap.

    Politis & Romano (1994): blocks of geometric length with mean
    ``mean_block``, wrapping at the end of the series, so the resample is
    stationary and preserves dependence up to roughly the block length.
    Returning indices (rather than resampled values) is what makes the
    *paired* comparison possible — the same resample is applied to two
    strategies, keeping their co-movement intact.
    """
    if n <= 0:
        raise ValueError("empty series")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    p = 1.0 / float(mean_block)
    starts = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    keep = rng.random((n_boot, n)) >= p  # continue the current block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        prev = idx[:, t - 1] + 1
        prev[prev >= n] = 0
        idx[:, t] = np.where(keep[:, t], prev, starts[:, t])
    return idx


@dataclass
class Interval:
    """A point estimate with a bootstrap interval."""

    point: float
    lo: float
    hi: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.point:.2f} [{self.lo:.2f}, {self.hi:.2f}]"


def _percentile_interval(point: float, draws: np.ndarray,
                         level: float) -> Interval:
    tail = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(draws, [100 * tail, 100 * (1 - tail)])
    return Interval(float(point), float(lo), float(hi), level)


def _indices(n: int, mean_block: float, n_boot: int, seed: int,
             indices: np.ndarray | None) -> np.ndarray:
    """Reuse a caller-supplied index matrix, or build one.

    Callers comparing many strategies over the same period should build
    the matrix once and pass it in: it is the expensive part, and sharing
    it means every strategy is scored on the *identical* resamples.
    """
    if indices is None:
        return stationary_bootstrap_indices(n, mean_block, n_boot,
                                            np.random.default_rng(seed))
    if indices.shape[1] != n:
        raise ValueError(f"indices are for n={indices.shape[1]}, series has {n}")
    return indices


def bootstrap_interval(rets: np.ndarray, stat, *, mean_block: float = 30.0,
                       n_boot: int = 2_000, level: float = 0.95,
                       seed: int = 7,
                       indices: np.ndarray | None = None) -> Interval:
    """Block-bootstrap interval for ``stat`` applied to one return series."""
    rets = np.asarray(rets, dtype=float)
    idx = _indices(len(rets), mean_block, n_boot, seed, indices)
    draws = np.asarray(stat(rets[idx]), dtype=float)
    return _percentile_interval(stat(rets), draws, level)


@dataclass
class PairedResult:
    """Paired block-bootstrap comparison of two strategies."""

    stat_a: float
    stat_b: float
    diff: Interval
    p_positive: float

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero."""
        return self.diff.lo > 0.0 or self.diff.hi < 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, stat, *,
                     mean_block: float = 30.0, n_boot: int = 2_000,
                     level: float = 0.95, seed: int = 7,
                     indices: np.ndarray | None = None) -> PairedResult:
    """Compare ``stat(a) - stat(b)`` on identical resamples of both series.

    ``a`` and ``b`` must be aligned, equal-length return series from the
    same period — the whole point is that a resample which happens to draw
    the 2022 bear draws it for both.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"unaligned series: {len(a)} vs {len(b)}")
    idx = _indices(len(a), mean_block, n_boot, seed, indices)
    draws = np.asarray(stat(a[idx]), dtype=float) - np.asarray(stat(b[idx]), dtype=float)
    interval = _percentile_interval(stat(a) - stat(b), draws, level)
    return PairedResult(stat_a=float(stat(a)), stat_b=float(stat(b)),
                        diff=interval,
                        p_positive=float(np.mean(draws > 0.0)))


# ------------------------------------------- Ledoit-Wolf Sharpe-diff tests

@dataclass
class LedoitWolfResult:
    """HAC-studentized Sharpe-ratio-difference test, Ledoit & Wolf (2008).

    Studentizes ``SR(a) - SR(b)`` by the delta-method standard error of a
    Parzen-kernel HAC estimate of the long-run covariance of the four
    return moments the Sharpe ratio is a function of, with an automatic
    Newey-West (1994) bandwidth. See :func:`ledoit_wolf_sharpe_diff`.
    """

    sr_a: float
    sr_b: float
    diff: Interval          # point = sr_a - sr_b, HAC-studentized 95% CI
    se: float
    tstat: float
    pvalue: float
    bandwidth: float        # S* (Parzen kernel automatic bandwidth, in bars)
    alpha_hat: float        # Newey-West (1994) plug-in used to derive S*
    n: int

    @property
    def significant(self) -> bool:
        return self.diff.lo > 0.0 or self.diff.hi < 0.0


def _sharpe_moments(x: np.ndarray) -> tuple[float, float, float]:
    mu = float(np.mean(x))
    gamma = float(np.mean(x ** 2))
    denom = gamma - mu * mu
    sr = mu / sqrt(denom) if denom > 0 else 0.0
    return mu, gamma, sr


def _lw_gradient(mu1: float, gamma1: float, mu2: float, gamma2: float) -> np.ndarray:
    """Gradient of ``SR_1 - SR_2`` w.r.t. ``(mu1, mu2, gamma1, gamma2)``.

    ``SR_1 = mu1 * (gamma1 - mu1**2)**-0.5``, so
    ``d(SR_1)/d(mu1) = gamma1 / (gamma1-mu1**2)**1.5`` and
    ``d(SR_1)/d(gamma1) = -0.5*mu1 / (gamma1-mu1**2)**1.5``, symmetrically
    for the ``b`` leg (with a sign flip, since the statistic is ``a - b``).
    Verified by finite differences in ``tests/test_ledoit_wolf.py``.
    """
    s1 = (gamma1 - mu1 * mu1) ** 1.5
    s2 = (gamma2 - mu2 * mu2) ** 1.5
    return np.array([gamma1 / s1, -gamma2 / s2, -0.5 * mu1 / s1, 0.5 * mu2 / s2])


def _parzen_kernel(x: float) -> float:
    ax = abs(x)
    if ax <= 0.5:
        return 1.0 - 6.0 * ax * ax + 6.0 * ax ** 3
    if ax <= 1.0:
        return 2.0 * (1.0 - ax) ** 3
    return 0.0


def _nw_bandwidth(V: np.ndarray) -> tuple[float, float]:
    """Newey & West (1994) automatic Parzen-kernel bandwidth: an AR(1)
    plug-in fit to each of ``V``'s columns, combined with equal weights (the
    standard multivariate generalisation used by e.g. R's
    ``sandwich::bwNeweyWest`` and the ``PeerPerformance`` package's
    ``sharpeTesting()``). Returns ``(S_star, alpha_hat)``."""
    T, k = V.shape
    num, den = 0.0, 0.0
    for j in range(k):
        x = V[:, j]
        x0, x1 = x[:-1], x[1:]
        denom_ols = float(np.sum(x0 * x0))
        rho = float(np.sum(x0 * x1) / denom_ols) if denom_ols > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))  # numerical guard only
        resid = x1 - rho * x0
        sigma2 = float(np.var(resid, ddof=1)) if len(resid) > 1 else float(np.var(resid))
        num += 4.0 * rho * rho * sigma2 * sigma2 / (((1 - rho) ** 6) * ((1 + rho) ** 2))
        den += sigma2 * sigma2 / ((1 - rho) ** 4)
    alpha_hat = num / den if den > 0 else 0.0
    s_star = 2.6614 * (alpha_hat * T) ** 0.2
    return float(s_star), float(alpha_hat)


def _hac_long_run_cov(V: np.ndarray, s_star: float) -> np.ndarray:
    """Parzen-kernel HAC estimate of ``V_t``'s long-run covariance,
    degrees-of-freedom corrected by ``T/(T-k)`` for the ``k`` estimated
    moments."""
    T, k = V.shape
    gamma0 = V.T @ V / T
    psi = gamma0.copy()
    s_int = int(np.floor(s_star))
    for j in range(1, s_int + 1):
        w = _parzen_kernel(j / s_star)
        if w == 0.0:
            continue
        gj = V[j:].T @ V[:-j] / T
        psi += w * (gj + gj.T)
    return psi * (T / (T - k))


def ledoit_wolf_sharpe_diff(a: np.ndarray, b: np.ndarray,
                            level: float = 0.95) -> LedoitWolfResult:
    """The literal Ledoit & Wolf (2008) HAC-studentized Sharpe-ratio
    difference test between two aligned, equal-length return series.

    ``a`` and ``b`` must be aligned (same dates, same length) — as in
    :func:`paired_bootstrap`, the point is that the two series' shared
    variance is what makes the difference well resolved. Unlike
    :func:`paired_bootstrap`, this needs no resampling: the standard error
    comes from a closed-form delta-method/HAC estimate, so the result is
    deterministic given ``a`` and ``b``. Only the 95% two-sided CI
    (``z=1.959964``) is implemented.

    See :func:`bootstrap_studentized_sharpe_diff` for a second, independent
    estimator of the same studentized statistic's standard error (a
    nonparametric stationary bootstrap in place of this function's kernel
    HAC) — R-70 (``docs/LEDGER.md``) found the two do not always agree, and
    recommends reporting both rather than treating either as authoritative.
    Note the Sharpe ratio here uses the population std (``ddof=0``, the
    only form available to the delta method over the two smooth sample
    means ``mu`` and ``gamma=E[X**2]``) rather than this module's usual
    ``ddof=1`` (:func:`annualized_sharpe`); the difference is the standard
    ``sqrt(T/(T-1))`` small-sample correction, negligible at this project's
    sample sizes.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"unaligned series: {len(a)} vs {len(b)}")
    if level != 0.95:
        raise NotImplementedError("only the 95% CI (z=1.959964) is implemented")
    T = len(a)

    mu1, gamma1, sr1 = _sharpe_moments(a)
    mu2, gamma2, sr2 = _sharpe_moments(b)
    diff = sr1 - sr2

    V = np.column_stack([a - mu1, b - mu2, a ** 2 - gamma1, b ** 2 - gamma2])
    s_star, alpha_hat = _nw_bandwidth(V)
    psi = _hac_long_run_cov(V, s_star)
    grad = _lw_gradient(mu1, gamma1, mu2, gamma2)

    var = float(grad @ psi @ grad) / T
    se = sqrt(var) if var > 0 else 0.0
    tstat = diff / se if se > 0 else 0.0
    pvalue = 2.0 * (1.0 - norm_cdf(abs(tstat))) if se > 0 else 1.0

    z95 = 1.959964
    ci = Interval(diff, diff - z95 * se, diff + z95 * se, level)
    return LedoitWolfResult(sr_a=sr1, sr_b=sr2, diff=ci, se=se, tstat=tstat,
                            pvalue=pvalue, bandwidth=s_star, alpha_hat=alpha_hat, n=T)


@dataclass
class StudentizedSharpeDiff:
    """Bootstrap-studentized Sharpe-ratio-difference test.

    Same statistic as :class:`LedoitWolfResult`, with the standard error
    estimated by a paired stationary bootstrap (:func:`stationary_bootstrap_indices`)
    instead of a kernel HAC estimate. ``studentized_ci`` is a single-bootstrap
    studentized-t interval (not the fully general double bootstrap — each
    replicate's own standard error is approximated by the one ``se_boot``
    computed on the original sample, the standard practical compromise).
    """

    sr_a: float
    sr_b: float
    diff: float
    se_boot: float
    tstat: float
    normal_ci: Interval
    studentized_ci: Interval
    n_boot: int
    mean_block: float
    n: int

    @property
    def significant_normal(self) -> bool:
        return self.normal_ci.lo > 0.0 or self.normal_ci.hi < 0.0

    @property
    def significant_studentized(self) -> bool:
        return self.studentized_ci.lo > 0.0 or self.studentized_ci.hi < 0.0


def bootstrap_studentized_sharpe_diff(
    a: np.ndarray, b: np.ndarray, *,
    mean_block: float = 30.0, n_boot: int = 2_000,
    level: float = 0.95, seed: int = 7,
    indices: np.ndarray | None = None, ddof: int = 1,
) -> StudentizedSharpeDiff:
    """Ledoit-Wolf (2008)-style studentized Sharpe-difference test, with the
    standard error estimated via this module's own stationary-bootstrap
    convention (Politis & Romano 1994) instead of a kernel HAC estimate.

    ``a`` and ``b`` must be aligned, equal-length return series from the
    same period — exactly :func:`paired_bootstrap`'s own precondition: the
    resample has to draw the same days from both series so their shared
    variance cancels in the difference. See :func:`ledoit_wolf_sharpe_diff`
    for the analytic HAC alternative this is meant to be checked against.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"unaligned series: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 3:
        raise ValueError("need at least 3 paired observations")

    idx = _indices(n, mean_block, n_boot, seed, indices)

    sr_a = annualized_sharpe(a, periods_per_year=1.0)
    sr_b = annualized_sharpe(b, periods_per_year=1.0)
    diff = float(sr_a - sr_b)

    boot_diff = (annualized_sharpe(a[idx], periods_per_year=1.0)
                - annualized_sharpe(b[idx], periods_per_year=1.0))
    se_boot = float(np.std(boot_diff, ddof=1))

    tail = (1.0 - level) / 2.0
    if se_boot <= 0.0 or not np.isfinite(se_boot):
        tstat = float("nan")
        normal_ci = Interval(diff, float("nan"), float("nan"), level)
        studentized_ci = Interval(diff, float("nan"), float("nan"), level)
    else:
        tstat = diff / se_boot
        z = norm_ppf(1.0 - tail)
        normal_ci = Interval(diff, diff - z * se_boot, diff + z * se_boot, level)

        studentized = (boot_diff - diff) / se_boot
        q_lo, q_hi = np.percentile(studentized, [100 * tail, 100 * (1 - tail)])
        studentized_ci = Interval(
            diff, diff - se_boot * q_hi, diff - se_boot * q_lo, level)

    return StudentizedSharpeDiff(
        sr_a=float(sr_a), sr_b=float(sr_b), diff=diff,
        se_boot=se_boot, tstat=float(tstat),
        normal_ci=normal_ci, studentized_ci=studentized_ci,
        n_boot=int(idx.shape[0]), mean_block=float(mean_block), n=n)


# ---------------------------------------------------------- deflated Sharpe

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the normal quantile (~1e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def moments(rets: np.ndarray) -> tuple[float, float]:
    """Skewness and (non-excess) kurtosis of a return series."""
    rets = np.asarray(rets, dtype=float)
    sd = rets.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0, 3.0
    z = (rets - rets.mean()) / sd
    return float((z ** 3).mean()), float((z ** 4).mean())


def probabilistic_sharpe_ratio(sharpe: float, n_obs: int, skew: float,
                               kurtosis: float, benchmark: float = 0.0,
                               periods_per_year: float = DAYS_PER_YEAR) -> float:
    """P(true Sharpe > ``benchmark``), correcting for skew and fat tails.

    ``sharpe`` and ``benchmark`` are annualized; ``n_obs`` counts the
    return observations they were computed from. Bailey & López de Prado
    (2012): non-normality matters, and a negatively skewed, fat-tailed
    equity curve needs a longer record to prove the same Sharpe.
    """
    if n_obs < 3:
        return float("nan")
    scale = sqrt(periods_per_year)
    sr, bench = sharpe / scale, benchmark / scale
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return norm_cdf((sr - bench) * sqrt(n_obs - 1) / sqrt(denom))


def expected_max_sharpe(n_trials: int, sd_trials: float,
                        periods_per_year: float = DAYS_PER_YEAR) -> float:
    """SR* — the Sharpe the best of ``n_trials`` reaches by luck alone.

    Both ``sd_trials`` (the spread of Sharpes across the trials searched)
    and the return value are annualized. With a true Sharpe of zero for
    every trial, the maximum of N draws is not zero: it grows like the
    expected maximum of N normals, which is what this returns.
    """
    if n_trials < 2 or sd_trials <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z = ((1 - g) * norm_ppf(1 - 1.0 / n_trials)
         + g * norm_ppf(1 - 1.0 / (n_trials * EULER_E)))
    return float(sd_trials * z)


def deflated_sharpe_ratio(sharpe: float, n_obs: int, skew: float,
                          kurtosis: float, n_trials: int, sd_trials: float,
                          periods_per_year: float = DAYS_PER_YEAR) -> float:
    """P(true Sharpe > SR*) — the probabilistic Sharpe against the trials bar.

    The conventional bar is 0.95. Below it, the observed Sharpe is not
    distinguishable from the best of ``n_trials`` coin flips.
    """
    sr_star = expected_max_sharpe(n_trials, sd_trials, periods_per_year)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis,
                                      benchmark=sr_star,
                                      periods_per_year=periods_per_year)


def deflation_breakeven_sd(sharpe: float, n_obs: int, skew: float,
                           kurtosis: float, n_trials: int,
                           target: float = 0.95, hi: float = 20.0,
                           periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Largest trial dispersion at which this Sharpe still clears ``target``.

    The deflated Sharpe depends on two things the analyst must supply: how
    many configurations were searched, and **how spread out their Sharpes
    were**. The second is the one nobody can pin down after the fact — a
    narrow sweep around one idea and a scan across twenty-five unrelated
    strategies give wildly different answers from the same data.

    So rather than defend one estimate, invert the question: at what trial
    dispersion does the claim stop surviving? That number can be compared
    against searches this project actually ran. Returns 0.0 when the
    Sharpe fails even against a zero-dispersion search.
    """
    if deflated_sharpe_ratio(sharpe, n_obs, skew, kurtosis, n_trials, 0.0,
                             periods_per_year) < target:
        return 0.0
    lo = 0.0
    if deflated_sharpe_ratio(sharpe, n_obs, skew, kurtosis, n_trials, hi,
                             periods_per_year) >= target:
        return float(hi)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if deflated_sharpe_ratio(sharpe, n_obs, skew, kurtosis, n_trials, mid,
                                 periods_per_year) >= target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def min_track_record_length(sharpe: float, skew: float, kurtosis: float,
                            benchmark: float = 0.0, confidence: float = 0.95,
                            periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Observations needed before ``sharpe`` beats ``benchmark`` at ``confidence``.

    Returns ``inf`` when the observed Sharpe does not exceed the benchmark
    at all — no amount of track record fixes that.
    """
    scale = sqrt(periods_per_year)
    sr, bench = sharpe / scale, benchmark / scale
    if sr <= bench:
        return float("inf")
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    return 1.0 + denom * (norm_ppf(confidence) / (sr - bench)) ** 2


# ------------------------------------------------------- purged CV / CPCV

def group_bounds(n: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal ``[start, end)`` bounds splitting ``n`` points."""
    if n_groups < 2 or n_groups > n:
        raise ValueError(f"n_groups must be in [2, {n}]")
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def purged_train_mask(n: int, bounds: list[tuple[int, int]],
                      test_groups: tuple[int, ...], purge: int = 0,
                      embargo: int = 0) -> np.ndarray:
    """Boolean mask of usable training points for a given test set.

    Everything inside a test group is excluded, plus ``purge`` points
    immediately before it and ``embargo`` points immediately after. The
    purge kills the overlap that makes a neighbouring training point a
    near-copy of a test point; the embargo kills the reverse leak through
    serial correlation (López de Prado 2018, ch. 7).
    """
    mask = np.ones(n, dtype=bool)
    for g in test_groups:
        lo, hi = bounds[g]
        mask[max(0, lo - purge):min(n, hi + embargo)] = False
    return mask


def fold_mask(n: int, bounds: list[tuple[int, int]],
              test_groups: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of the test points for a given group selection."""
    mask = np.zeros(n, dtype=bool)
    for g in test_groups:
        lo, hi = bounds[g]
        mask[lo:hi] = True
    return mask


def cpcv_splits(n_groups: int, k_test: int) -> list[tuple[int, ...]]:
    """Every combination of ``k_test`` groups out of ``n_groups``.

    ``C(N, k)`` splits rather than the ``N`` of a plain k-fold, which is
    the point: it yields a *distribution* of out-of-fold outcomes instead
    of one walk-forward number, and each test set is a different mix of
    market regimes.
    """
    if not 1 <= k_test < n_groups:
        raise ValueError("k_test must be in [1, n_groups)")
    return list(combinations(range(n_groups), k_test))
