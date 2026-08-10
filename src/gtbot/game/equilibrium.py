"""Robust position sizing as a two-player zero-sum game.

Point estimates of edge are unreliable, and sizing on them directly is how
strategies blow up.  Instead we play a game against an adversarial "nature"
that chooses the return distribution from an ambiguity set:

* we choose a mixed strategy over candidate position sizes;
* nature chooses a mixed strategy over scenarios (edge realised, edge absent,
  edge inverted, adverse tail);
* the payoff is our utility of the resulting P&L net of costs.

The maximin value of this game is a size that is optimal against the *worst*
plausible model rather than the point estimate — Gilboa-Schmeidler maxmin
expected utility, solved concretely.  It degrades gracefully: when the edge is
well supported the ambiguity set is tight and the size approaches Kelly; when
the edge is marginal the adversary's scenarios dominate and the size collapses
toward zero.

The solver is fictitious play, which converges to the value of a finite
zero-sum game (Robinson, 1951) and needs no LP dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fictitious_play(
    payoff: np.ndarray, iters: int = 400, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a zero-sum game where the row player maximises ``payoff``.

    Returns ``(row_strategy, col_strategy, value)``.  Both strategies are the
    empirical frequencies of best responses, which for a zero-sum game converge
    to a Nash equilibrium.
    """
    n_rows, n_cols = payoff.shape
    row_counts = np.zeros(n_rows)
    col_counts = np.zeros(n_cols)

    # Seed with one arbitrary but deterministic pure strategy each.
    rng = np.random.default_rng(seed)
    row_counts[rng.integers(n_rows)] += 1.0
    col_counts[rng.integers(n_cols)] += 1.0

    for _ in range(iters):
        col_mix = col_counts / col_counts.sum()
        best_row = int(np.argmax(payoff @ col_mix))
        row_counts[best_row] += 1.0

        row_mix = row_counts / row_counts.sum()
        best_col = int(np.argmin(row_mix @ payoff))  # column player minimises
        col_counts[best_col] += 1.0

    p = row_counts / row_counts.sum()
    q = col_counts / col_counts.sum()
    return p, q, float(p @ payoff @ q)


@dataclass(frozen=True)
class AmbiguityConfig:
    """Nature's ambiguity set: a confidence interval around the estimated edge.

    An earlier version let nature reallocate probability across a fixed set of
    "edge realised / halved / absent / inverted" scenarios.  That embeds a
    constant ~40% haircut on the point estimate no matter how much data backs
    it, which vetoes every thin-but-real microstructure edge in existence — the
    adversary should get weaker as evidence accumulates, not stay equally
    pessimistic forever.

    Here nature instead picks the true edge from ``[mu - k*SE, mu]``, the
    one-sided confidence interval implied by the estimator's own standard
    error, and a fixed ``model_haircut`` covers the structural gap between any
    backtest and live trading.  The ambiguity set therefore contracts at the
    statistically correct rate as the sample grows.
    """

    #: Standard errors of downside nature may claim.
    k_sigma: float = 0.5
    #: Points nature's interval is discretised into (for the matrix game).
    n_scenarios: int = 5
    #: Structural allowance for backtest-to-live decay, applied to every edge.
    model_haircut: float = 0.90
    #: Position sizes considered, as a fraction of the maximum position.
    size_grid: tuple[float, ...] = tuple(np.round(np.linspace(0.0, 1.0, 41), 4))
    #: Quadratic risk penalty.  Size reaches 1 when the robust per-trade net
    #: Sharpe reaches ``risk_aversion``.
    risk_aversion: float = 0.06
    iters: int = 200


def robust_size(
    edge: float,
    se: float,
    vol: float,
    cost: float,
    *,
    cfg: AmbiguityConfig = AmbiguityConfig(),
) -> tuple[float, float]:
    """Solve the sizing game for the robust position size.

    Parameters
    ----------
    edge:
        Point estimate of the expected return over the holding period.
    se:
        Standard error of that estimate, same units.
    vol:
        Forecast volatility of the return over the holding period.
    cost:
        Round-trip transaction cost as a return.

    Returns ``(size, game_value)`` with ``size`` in ``[0, 1]``; direction is
    applied by the caller.
    """
    vol = max(float(vol), 1e-12)
    sizes = np.asarray(cfg.size_grid, dtype=float)
    # Nature's action set: the discretised one-sided confidence interval.
    scenarios = edge - np.linspace(0.0, cfg.k_sigma * max(se, 0.0), cfg.n_scenarios)
    net = (scenarios * cfg.model_haircut - cost) / vol
    payoff = sizes[:, None] * net[None, :] - 0.5 * cfg.risk_aversion * sizes[:, None] ** 2

    p, _, value = fictitious_play(payoff, iters=cfg.iters)
    return float(p @ sizes), value


def fast_robust_size(
    edge: float, se: float, vol: float, cost: float, cfg: AmbiguityConfig
) -> float:
    """Closed form of :func:`robust_size`.

    Nature's optimum is simply the low end of the interval, so the game
    collapses to a one-line formula.  ``tests/test_equilibrium.py`` asserts the
    two agree.
    """
    vol = max(float(vol), 1e-12)
    robust_edge = (edge - cfg.k_sigma * max(se, 0.0)) * cfg.model_haircut
    return float(np.clip((robust_edge - cost) / (vol * cfg.risk_aversion), 0.0, 1.0))


def nash_size_table(
    edges: np.ndarray, se: float, vol: float, cost: float, *, cfg: AmbiguityConfig = AmbiguityConfig()
) -> np.ndarray:
    """Pre-solve the sizing game on a grid of edges.

    The game is small but solving it every bar is still wasteful, and the map
    from edge to size is monotone and smooth.  The backtester solves it once on
    a grid and interpolates, which is exact to within the grid resolution and
    roughly two orders of magnitude faster.
    """
    return np.array([robust_size(float(e), se, vol, cost, cfg=cfg)[0] for e in edges])
