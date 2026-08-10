"""Walk-forward evaluation with purging and embargo.

The strategy learns online, so there is no separate "fit" step to leak through
— but there are still three ways to fool yourself, and this module closes all
three:

*Warm-up leakage.*  The first weeks of any fold are spent filling trailing
windows and letting the meta-learner converge.  Scoring them mixes learning
cost into the performance estimate, so each fold has an explicit warm-up
segment that is simulated but not scored.

*Overlap leakage.*  A trade opened near the end of a fold resolves inside the
next one.  Folds are therefore *purged*: observations within one holding period
of a boundary are dropped from scoring.

*Serial-correlation leakage.*  Adjacent observations share volatility state.
An *embargo* after each test fold discards a further stretch so that the next
fold does not begin inside the tail of the previous one.

The result is a set of out-of-sample segments whose scores can be pooled, and a
dispersion across folds that says whether the edge is persistent or the artefact
of one lucky stretch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.schema import BTCUSD_5M, BarSpec
from ..engine.backtest import BacktestResult, run_backtest
from ..engine.broker import CostModel, ExecutionConfig
from . import metrics


@dataclass
class Fold:
    index: int
    start: int
    warmup_end: int
    end: int

    @property
    def scored_bars(self) -> int:
        return self.end - self.warmup_end


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    fold_metrics: list[metrics.Metrics]
    pooled: metrics.Metrics
    pooled_returns: np.ndarray
    equity: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": f.index,
                    "bars": f.scored_bars,
                    "sharpe": m.sharpe,
                    "cagr": m.cagr,
                    "vol": m.ann_vol,
                    "max_dd": m.max_drawdown,
                    "trades": m.n_trades,
                }
                for f, m in zip(self.folds, self.fold_metrics)
            ]
        )


def make_folds(
    n_bars: int, *, n_folds: int, warmup: int, embargo: int, purge: int
) -> list[Fold]:
    """Split ``n_bars`` into sequential folds with warm-up, purge and embargo."""
    usable = n_bars - embargo * (n_folds - 1)
    span = usable // n_folds
    if span <= warmup + purge + 10:
        raise ValueError("not enough bars for the requested fold structure")

    folds: list[Fold] = []
    cursor = 0
    for i in range(n_folds):
        start = cursor
        end = min(start + span, n_bars)
        folds.append(Fold(index=i, start=start, warmup_end=start + warmup, end=end - purge))
        cursor = end + embargo
        if cursor >= n_bars:
            break
    return folds


def run_walkforward(
    bars: pd.DataFrame,
    strategy_factory,
    *,
    n_folds: int = 6,
    warmup: int | None = None,
    embargo: int = 288,
    purge: int = 12,
    costs: CostModel | None = None,
    execution: ExecutionConfig | None = None,
    spec: BarSpec = BTCUSD_5M,
    max_leverage: float = 2.0,
) -> WalkForwardResult:
    """Run an independent strategy instance on each fold and pool the results.

    ``strategy_factory`` must return a *fresh* strategy: reusing one across
    folds would carry learned weights forward and quietly turn the later folds
    into in-sample evaluations.
    """
    n = len(bars)
    probe = strategy_factory()
    warmup = warmup if warmup is not None else probe.cfg.features.warmup + 2016
    folds = make_folds(n, n_folds=n_folds, warmup=warmup, embargo=embargo, purge=purge)

    fold_metrics: list[metrics.Metrics] = []
    pooled: list[np.ndarray] = []
    results: list[BacktestResult] = []

    for fold in folds:
        segment = bars.iloc[fold.start : fold.end].reset_index(drop=True)
        if len(segment) <= warmup + 50:
            continue
        res = run_backtest(
            segment,
            strategy_factory(),
            costs=costs,
            execution=execution,
            spec=spec,
            max_leverage=max_leverage,
        )
        results.append(res)

        # Score only the post-warm-up part of the fold.
        offset = fold.warmup_end - fold.start
        r = res.returns[offset:]
        eq = res.equity[offset:]
        if r.size < 50:
            continue
        eq = eq / eq[0]
        pooled.append(r)
        fold_metrics.append(
            metrics.compute(
                r,
                eq,
                res.position[offset:],
                res.costs[offset:],
                bars_per_year=spec.bars_per_year,
                n_trades=sum(1 for f in res.fills if f.bar >= offset),
            )
        )

    all_returns = np.concatenate(pooled) if pooled else np.zeros(0)
    equity = np.cumprod(1.0 + all_returns) if all_returns.size else np.ones(1)
    pooled_metrics = metrics.compute(
        all_returns,
        equity,
        np.concatenate([r.position for r in results]) if results else np.zeros(1),
        np.concatenate([r.costs for r in results]) if results else np.zeros(1),
        bars_per_year=spec.bars_per_year,
        n_trades=sum(m.n_trades for m in fold_metrics),
    )
    return WalkForwardResult(
        folds=folds[: len(fold_metrics)],
        fold_metrics=fold_metrics,
        pooled=pooled_metrics,
        pooled_returns=all_returns,
        equity=equity,
    )
