"""Performance metrics.

Beyond the usual ratios this module carries the two statistics that matter when
a strategy has been selected from a set of candidates: the Probabilistic Sharpe
Ratio and the Deflated Sharpe Ratio (Bailey & Lopez de Prado).  A Sharpe ratio
reported without them is not a claim about the strategy, only about the search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class Metrics:
    n_bars: int
    years: float
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    profit_factor: float
    skew: float
    kurtosis: float
    turnover_annual: float
    cost_drag_annual: float
    n_trades: int
    psr: float
    dsr: float
    t_stat: float

    def to_dict(self) -> dict:
        return asdict(self)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def sharpe_ratio(returns: np.ndarray, bars_per_year: float) -> float:
    if returns.size < 2:
        return 0.0
    sd = returns.std(ddof=1)
    if sd <= 1e-15:
        return 0.0
    return float(returns.mean() / sd * math.sqrt(bars_per_year))


def sortino_ratio(returns: np.ndarray, bars_per_year: float) -> float:
    downside = returns[returns < 0]
    if downside.size < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if dd <= 1e-15:
        return 0.0
    return float(returns.mean() / dd * math.sqrt(bars_per_year))


def probabilistic_sharpe(observed_sr: float, n: int, skew: float, kurt: float, benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), corrected for skew and kurtosis.

    ``observed_sr`` and ``benchmark`` are per-observation (not annualised).
    """
    if n < 3:
        return 0.5
    denom = 1.0 - skew * observed_sr + 0.25 * (kurt - 1.0) * observed_sr**2
    if denom <= 1e-12:
        return 0.5
    z = (observed_sr - benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe(
    observed_sr: float, n: int, skew: float, kurt: float, n_trials: int, trial_sr_std: float
) -> float:
    """Probabilistic Sharpe against the Sharpe a *lucky* trial would produce.

    ``n_trials`` is how many strategy configurations were evaluated and
    ``trial_sr_std`` the dispersion of their Sharpe ratios.  The benchmark is
    the expected maximum of ``n_trials`` draws — so a strategy only clears the
    bar if it beats what the search itself would have thrown up by chance.
    """
    if n_trials <= 1 or trial_sr_std <= 0:
        return probabilistic_sharpe(observed_sr, n, skew, kurt, 0.0)
    euler = 0.5772156649015329
    e_max = trial_sr_std * (
        (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    return probabilistic_sharpe(observed_sr, n, skew, kurt, e_max)


def compute(
    returns: np.ndarray,
    equity: np.ndarray,
    position: np.ndarray,
    costs: np.ndarray,
    *,
    bars_per_year: float,
    n_trades: int = 0,
    n_trials: int = 1,
    trial_sr_std: float = 0.0,
) -> Metrics:
    """Full metric set for one equity path."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    years = n / bars_per_year if bars_per_year else 0.0

    total = float(equity[-1] / equity[0] - 1.0) if equity.size else 0.0
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else 0.0
    ann_vol = float(r.std(ddof=1) * math.sqrt(bars_per_year)) if n > 1 else 0.0
    sr = sharpe_ratio(r, bars_per_year)
    mdd = max_drawdown(equity) if equity.size else 0.0

    wins = r[r > 0]
    losses = r[r < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else float("inf")

    turnover = float(np.abs(np.diff(position)).sum() / years) if years > 0 else 0.0
    cost_drag = float(costs.sum() / years) if years > 0 else 0.0

    # A strategy that never traded has a constant return series; skew and
    # kurtosis are undefined there and would poison every downstream ratio
    # with NaN rather than reporting the true answer, which is "no track record".
    degenerate = n < 4 or r.std(ddof=1) <= 1e-15
    sk = 0.0 if degenerate else float(stats.skew(r))
    ku = 3.0 if degenerate else float(stats.kurtosis(r, fisher=False))
    sr_per_obs = sr / math.sqrt(bars_per_year) if bars_per_year else 0.0

    return Metrics(
        n_bars=n,
        years=years,
        total_return=total,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sr,
        sortino=sortino_ratio(r, bars_per_year),
        max_drawdown=mdd,
        calmar=float(cagr / mdd) if mdd > 1e-9 else 0.0,
        hit_rate=float(wins.size / max(wins.size + losses.size, 1)),
        profit_factor=pf,
        skew=sk,
        kurtosis=ku,
        turnover_annual=turnover,
        cost_drag_annual=cost_drag,
        n_trades=n_trades,
        psr=0.5 if degenerate else probabilistic_sharpe(sr_per_obs, n, sk, ku),
        dsr=0.0 if degenerate else deflated_sharpe(sr_per_obs, n, sk, ku, n_trials, trial_sr_std),
        t_stat=float(sr * math.sqrt(years)) if years > 0 else 0.0,
    )
