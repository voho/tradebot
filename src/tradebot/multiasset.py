"""Compose independent single-asset backtests into one portfolio result.

**What this is not.** It does not change how any one instrument is
simulated, filled, liquidated or funded — ``engine.py``, ``strategy.py``
and ``broker.py`` are untouched and stay the only place fill/liquidation
logic lives. It does not let a strategy see more than one instrument's
live state inside a single decision: each leg runs to completion in
total isolation before any curve is combined. That is a real limit, not
an oversight — see below.

**What it is.** ``run_multi_backtest`` runs N legs — each an
already-registered, ordinary single-asset ``Strategy`` on its own data —
through the existing, unmodified :func:`tradebot.window.run_period`, then
sums their equity curves at a fixed, pre-decided capital split and wraps
the sum in a synthetic :class:`~tradebot.engine.BacktestResult` so the
also-unmodified :func:`tradebot.metrics.compute_metrics` can derive
portfolio-level Sharpe/drawdown/final-balance exactly as it would for any
single-asset run. The entire correctness burden for fills, fees,
liquidation and funding stays inside the already-audited
``run_backtest``/``PaperBroker`` path; this module only sums curves.

**The honest limit.** ``weights`` must sum to 1.0 — a split fixed *before*
the run. This design cannot express a strategy that needs a shared risk
or leverage budget decided *during* the run (one leg lending idle margin
to another, or a live cross-asset covariance allocator reallocating
capital bar-by-bar): no leg's ``on_bar`` can see another leg's state,
because each leg is run to completion as an independent call before any
combination happens. That family needs a native multi-instrument engine
(one shared bar clock, one ``Context`` reading N brokers at once) — a
materially larger and riskier change to this project's most heavily
causality-tested files. Build that only once a specific strategy needs
it and has already earned the risk by clearing inner-validation as a
fixed-split prototype first.

**A concrete consequence of that limit, found by R-76 (docs/LEDGER.md):**
because each leg's ``PaperBroker`` is margined in complete isolation, a
leveraged long/short pair (e.g. a market-neutral spread trade, one leg
long and one short on ``MarketSpec.futures()``) can have ONE leg
liquidated by its own outright price move even while the pair's spread
itself is calm or moving favourably — there is no shared margin call
across legs to draw on. R-76's literal pairs trade lost 97-99% of its
capital to exactly this on 2 of 3 position-size fractions tested, before
any mean-reversion thesis was ever tested. This is not a bug: it is the
same "no shared risk budget" limit above, restated as a warning for
anyone sizing a *leveraged, direction-hedged* multi-leg strategy through
this module rather than a simple weighted portfolio of independent
directional bets.

Registering a multi-asset strategy for real (into ``tradebot run`` /
the README table / CI) additionally needs: an asset-aware
``tradebot.data`` load path (today's loaders are implicitly one asset
per market kind), a second code path in ``run.py``'s ``run_matrix`` that
detects a strategy declaring ``instruments`` and routes it through
``run_multi_backtest`` instead of ``run_backtest``, a table convention
for which instruments/weights a row represents, and a per-strategy
(rather than global) bootstrap-interval requirement in
``tests/test_evidence.py``. None of that is implemented here — this
module is the composition primitive those pieces would call, built and
tested in isolation first, per ``docs/ROUTINE.md``'s "build the missing
capability and record that" convention (B-17 in ``docs/LEDGER.md``).

Origin: R-49, generalized from the ad hoc composition
``experiments/kelly_regime_dual_fixed.py`` (R-42/R-43) already used once.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.engine import BacktestResult
from tradebot.metrics import Metrics, compute_metrics
from tradebot.registry import available_strategies
from tradebot.strategy import Strategy
from tradebot.window import run_period


@dataclass
class MultiAssetSpec:
    """One instrument leg of a multi-asset portfolio.

    Deliberately thin: an already-instantiated, fresh ``Strategy`` (the
    caller decides whether every leg runs the same strategy class or a
    different one per asset — this module does not care), that
    strategy's own full OHLCV frame, and the ``MarketSpec`` it trades in.
    """

    label: str
    strategy: Strategy
    df: pd.DataFrame
    market: MarketSpec


@dataclass
class MultiBacktestResult:
    """The output of composing N independent single-asset backtests.

    ``leg_results``/``leg_metrics`` are exactly what
    :func:`tradebot.window.run_period` and
    :func:`tradebot.metrics.compute_metrics` produce for each instrument
    run alone. ``portfolio``/``metrics`` are the synthetic, summed-equity
    combination — see :func:`combine_equity_curves`.

    ``metrics.num_trades``/``win_rate_pct``/``best_trade``/etc. are a
    plain concatenation of every leg's own trades (mixed currencies, no
    cross-leg netting) — read as "every trade any leg made", not a single
    book's history. ``final_balance``/``sharpe``/``max_drawdown_pct`` ARE
    genuinely portfolio-level, computed off the summed equity curve.
    """

    leg_labels: list[str]
    weights: list[float]
    leg_results: list[BacktestResult]
    leg_metrics: list[Metrics]
    portfolio: BacktestResult
    metrics: Metrics


def combine_equity_curves(leg_results: list[BacktestResult]) -> pd.Series:
    """Sum N independent equity curves onto their union bar grid.

    Forward-fills each leg between its own bar closes (equity is
    piecewise-constant there by construction) and backfills the start of
    the union grid with that leg's own ``start_balance`` — never with a
    forward-looking value, so a leg that starts later than another (ETH
    vs. BTC, here) contributes its flat starting balance until its own
    data begins, not a peek at its first real value.
    """
    if not leg_results:
        raise ValueError("need at least one leg")
    idx = leg_results[0].equity.index
    for r in leg_results[1:]:
        idx = idx.union(r.equity.index)
    total = pd.Series(0.0, index=idx)
    for r in leg_results:
        e = r.equity.reindex(idx).ffill().fillna(r.start_balance)
        total = total + e
    return total.rename("equity")


def _synthetic_portfolio_result(leg_results: list[BacktestResult], equity: pd.Series,
                                start_balance: float) -> BacktestResult:
    """Wrap the combined equity curve so the unmodified ``compute_metrics``
    can derive portfolio Sharpe/drawdown/final-balance.

    ``df`` is an empty placeholder — ``compute_metrics`` never reads
    ``result.df``. ``market`` is informational only, taken from the first
    leg: a portfolio whose legs trade different markets has no single
    honest market label yet (see this module's docstring).
    """
    names = sorted({r.strategy_name for r in leg_results})
    return BacktestResult(
        strategy_name="+".join(names),
        market=leg_results[0].market,
        start_balance=start_balance,
        data_label="+".join(r.data_label for r in leg_results),
        equity=equity,
        fills=[f for r in leg_results for f in r.fills],
        trades=[t for r in leg_results for t in r.trades],
        df=pd.DataFrame(index=equity.index),
        liquidated=any(r.liquidated for r in leg_results),
        fees_paid=sum(r.fees_paid for r in leg_results),
        funding_paid=sum(r.funding_paid for r in leg_results),
    )


def run_multi_backtest(
    specs: list[MultiAssetSpec],
    weights: list[float],
    start_balance: float,
    start: object | None = None,
    end: object | None = None,
    slippage_bps: float = 0.0,
) -> MultiBacktestResult:
    """Run N single-asset legs independently, then compose them.

    Each leg is executed via :func:`tradebot.window.run_period` — the
    same fair-warmup, date-bounded wrapper around
    :func:`tradebot.engine.run_backtest` every experiment in this repo
    uses — with its own fresh ``Strategy`` instance, its own isolated
    ``PaperBroker``, on its own data, capitalized at
    ``start_balance * weight``. No leg can see any other leg's state at
    any point during its own run.

    ``weights`` must sum to 1.0: a fixed capital split decided *before*
    the run, whether a hardcoded constant or a rule computed causally
    from a training-only slice. This is enforced, not a convention —
    relaxing it (e.g. summing to more than 1.0 to express shared
    leverage) is exactly the "shared risk budget" capability this design
    does not have.
    """
    if len(specs) != len(weights):
        raise ValueError(f"{len(specs)} specs but {len(weights)} weights")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1.0 (a fixed split decided in advance); got {sum(weights)!r}. "
            "This module cannot express a shared/overlapping risk budget across legs.")

    leg_results: list[BacktestResult] = []
    for spec, w in zip(specs, weights):
        leg_balance = start_balance * w
        result = run_period(
            spec.strategy, spec.df, start, end,
            market=spec.market, start_balance=leg_balance,
            slippage_bps=slippage_bps, data_label=spec.label,
        )
        leg_results.append(result)

    equity = combine_equity_curves(leg_results)
    portfolio = _synthetic_portfolio_result(leg_results, equity, start_balance)
    leg_metrics = [compute_metrics(r) for r in leg_results]
    metrics = compute_metrics(portfolio)

    return MultiBacktestResult(
        leg_labels=[s.label for s in specs],
        weights=list(weights),
        leg_results=leg_results,
        leg_metrics=leg_metrics,
        portfolio=portfolio,
        metrics=metrics,
    )


def available_multi_asset_strategies() -> dict[str, type[Strategy]]:
    """Registered strategies that declare a non-empty ``instruments``
    class attribute — the convention a future multi-asset strategy would
    use to opt in (see this module's docstring). No strategy declares one
    today; this is the filter ``run.py`` would iterate once one does.
    """
    return {n: c for n, c in available_strategies().items()
            if getattr(c, "instruments", ())}
