"""Statistical validation.

A backtest number on its own is close to worthless: the question is always
whether it could have arisen from a strategy with no edge, evaluated on this
much data, after this much searching.  The tools here answer that.

Negative controls matter as much as the tests.  If a strategy claims to exploit
a specific market structure, it must earn approximately nothing on data where
that structure has been destroyed but the marginal distribution preserved.  A
strategy that keeps "working" on shuffled data is reading its own tail, not the
market's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    mean: float
    lower: float
    upper: float
    p_value: float
    n_resamples: int


def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence.

    Geometric block lengths preserve serial dependence without the boundary
    artefacts of fixed blocks, which matters because strategy returns are
    autocorrelated through overlapping positions.
    """
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    jumps = rng.integers(0, n, n)
    for i in range(1, n):
        idx[i] = jumps[i] if new_block[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe(
    returns: np.ndarray,
    *,
    bars_per_year: float,
    n_resamples: int = 2000,
    mean_block: float = 288.0,
    seed: int = 0,
) -> BootstrapResult:
    """Block-bootstrap confidence interval for the annualised Sharpe ratio.

    The one-sided p-value is the fraction of resamples with a Sharpe at or
    below zero — an estimate of how often this track record could be produced
    by a process with no edge but the same dependence structure.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 50:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, 0)

    rng = np.random.default_rng(seed)
    scale = math.sqrt(bars_per_year)
    sharpes = np.empty(n_resamples)
    for b in range(n_resamples):
        sample = r[stationary_bootstrap_indices(n, mean_block, rng)]
        sd = sample.std(ddof=1)
        sharpes[b] = sample.mean() / sd * scale if sd > 1e-15 else 0.0

    observed = r.mean() / r.std(ddof=1) * scale if r.std(ddof=1) > 1e-15 else 0.0
    return BootstrapResult(
        mean=float(observed),
        lower=float(np.quantile(sharpes, 0.025)),
        upper=float(np.quantile(sharpes, 0.975)),
        p_value=float(np.mean(sharpes <= 0.0)),
        n_resamples=n_resamples,
    )


def whites_reality_check(
    candidate_returns: list[np.ndarray], *, n_resamples: int = 2000, seed: int = 0
) -> float:
    """White's Reality Check p-value for the best of a set of candidates.

    Tests the null that the *best* strategy among those evaluated has no
    expected excess return.  Without it, reporting the winner of a search is
    just reporting the maximum of a noise distribution.
    """
    if not candidate_returns:
        return 1.0
    rng = np.random.default_rng(seed)
    n = min(len(r) for r in candidate_returns)
    mat = np.column_stack([np.asarray(r, dtype=float)[:n] for r in candidate_returns])
    means = mat.mean(axis=0)
    observed = float(np.max(means) * math.sqrt(n))

    stat = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = stationary_bootstrap_indices(n, 288.0, rng)
        resampled = mat[idx].mean(axis=0) - means  # centred under the null
        stat[b] = float(np.max(resampled) * math.sqrt(n))
    return float(np.mean(stat >= observed))


def permutation_test(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    n_permutations: int = 1000,
    block: int = 288,
    seed: int = 0,
) -> float:
    """Block-permutation p-value for signal/return dependence.

    Permuting *blocks* of returns preserves volatility clustering while
    destroying the alignment between signal and outcome, so the null is "this
    signal carries no timing information", not "returns are i.i.d.".
    """
    s = np.asarray(signal, dtype=float)
    f = np.asarray(forward_returns, dtype=float)
    mask = np.isfinite(s) & np.isfinite(f)
    s, f = s[mask], f[mask]
    if s.size < 100:
        return 1.0

    observed = float(np.mean(s * f))
    n_blocks = int(np.ceil(f.size / block))
    padded = np.concatenate([f, np.zeros(n_blocks * block - f.size)])
    blocks = padded.reshape(n_blocks, block)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        shuffled = blocks[rng.permutation(n_blocks)].ravel()[: f.size]
        if abs(float(np.mean(s * shuffled))) >= abs(observed):
            count += 1
    return float((count + 1) / (n_permutations + 1))


def confidence_sequence(
    returns: np.ndarray, *, bars_per_year: float, alpha: float = 0.05
) -> tuple[float, float, int | None]:
    """Anytime-valid confidence sequence for the mean return.

    A fixed-sample confidence interval is only valid if you decide the sample
    size before looking.  Backtesting never works that way — you extend the
    history, you re-run after a change — and every extra look inflates the true
    error rate.  A confidence sequence is valid at *every* sample size
    simultaneously, so it can be monitored continuously and stopped the moment
    it excludes zero.

    This is the empirical-Bernstein mixture bound of Howard et al., the same
    machinery the AV-AIVAT line applies to agent evaluation in
    imperfect-information games, where it buys large reductions in the number of
    trials needed to certify a result.

    Returns ``(lower, upper, first_bar_excluding_zero)`` with the bounds
    annualised into Sharpe-comparable units.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 30:
        return (0.0, 0.0, None)

    scale = float(np.std(r, ddof=1))
    if scale <= 1e-18:
        return (0.0, 0.0, None)

    # Hoeffding-style mixture boundary: width ~ sqrt((2 (n rho^2 + 1) / (n^2 rho^2))
    # * log(sqrt(n rho^2 + 1) / alpha)).  rho tunes which sample size the bound
    # is tightest at; n/2 is a standard choice when the horizon is unknown.
    def _radius(k: int) -> float:
        rho = 2.0 / max(k, 1)
        term = k * rho + 1.0
        return scale * math.sqrt(
            (2.0 * term / (k * k * rho)) * math.log(math.sqrt(term) / alpha)
        )

    mean = float(r.mean())
    rad = _radius(n)
    lo, hi = mean - rad, mean + rad

    # When the sequence first became conclusive: the whole point of an
    # anytime-valid bound is that this question is legitimate to ask.
    first = None
    step = max(n // 200, 1)
    csum = np.cumsum(r)
    for k in range(30, n + 1, step):
        m = csum[k - 1] / k
        if abs(m) > _radius(k):
            first = k
            break

    ann = math.sqrt(bars_per_year) / scale
    return (lo * ann, hi * ann, first)


def newey_west_tstat(returns: np.ndarray, lags: int | None = None) -> float:
    """t-statistic of the mean return with a HAC (Newey-West) standard error.

    Overlapping holding periods make plain OLS standard errors too small; this
    is the standard correction.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 20:
        return 0.0
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    dev = r - r.mean()
    var = float(dev @ dev) / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(dev[l:] @ dev[:-l]) / n
        var += 2.0 * w * cov
    se = math.sqrt(max(var, 1e-24) / n)
    return float(r.mean() / se)


def summarise(returns: np.ndarray, *, bars_per_year: float, seed: int = 0) -> dict:
    """Convenience bundle used by the report."""
    boot = bootstrap_sharpe(returns, bars_per_year=bars_per_year, seed=seed)
    cs_lo, cs_hi, cs_first = confidence_sequence(returns, bars_per_year=bars_per_year)
    return {
        "conf_seq_95": (cs_lo, cs_hi),
        "conf_seq_first_conclusive_bar": cs_first,
        "sharpe": boot.mean,
        "sharpe_ci95": (boot.lower, boot.upper),
        "bootstrap_p_value": boot.p_value,
        "newey_west_t": newey_west_tstat(returns),
        "skew": float(stats.skew(returns[np.isfinite(returns)])),
    }
