"""Event-driven backtester.

Timing contract (the thing that makes or breaks a backtest):

1. Bar ``t`` closes.  Every feature and every expert signal at index ``t`` is a
   function of bars ``0..t`` only.
2. The strategy's online learner is updated with the payoff realised *over* bar
   ``t``, i.e. from the decision it made at ``t-1``.
3. The strategy emits a target position for bar ``t+1``.
4. Execution happens during bar ``t+1``: at the open for a taker order, or at a
   resting limit price if and when the bar trades through it.
5. P&L for bar ``t+1`` is split at the fill price, so the pre-fill portion is
   earned on the old position and the post-fill portion on the new one.

The same :class:`~gtbot.strategy.GameTheoreticStrategy` object drives this loop
and the paper/live loop in :mod:`gtbot.engine.paper`, so there is exactly one
implementation of the decision logic and no chance of backtest/live drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.schema import BTCUSD_5M, BarSpec
from .broker import Broker, CostModel, ExecutionConfig, Fill


@dataclass
class BacktestResult:
    ts: np.ndarray
    equity: np.ndarray
    position: np.ndarray
    returns: np.ndarray  # per-bar net returns on equity
    gross_returns: np.ndarray
    costs: np.ndarray  # per-bar cost drag as a return
    fills: list[Fill] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    spec: BarSpec = BTCUSD_5M

    @property
    def n_trades(self) -> int:
        return len(self.fills)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts": self.ts,
                "equity": self.equity,
                "position": self.position,
                "ret": self.returns,
                "gross_ret": self.gross_returns,
                "cost": self.costs,
            }
        )


def run_backtest(
    bars: pd.DataFrame,
    strategy,
    *,
    costs: CostModel | None = None,
    execution: ExecutionConfig | None = None,
    spec: BarSpec = BTCUSD_5M,
    initial_equity: float = 100_000.0,
    max_leverage: float = 1.0,
) -> BacktestResult:
    """Run ``strategy`` over ``bars`` and return the realised equity path."""
    costs = costs or CostModel()
    execution = execution or ExecutionConfig()
    broker = Broker(costs, execution)

    strategy.prepare(bars)

    ts = bars["ts"].to_numpy(dtype="int64")
    op = bars["open"].to_numpy(dtype=float)
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    cl = bars["close"].to_numpy(dtype=float)
    qv = bars["quote_volume"].to_numpy(dtype=float)
    n = ts.size

    atr_frac = strategy.atr_fraction()
    bar_vol_bp = atr_frac * 1e4

    equity = np.full(n, initial_equity)
    position = np.zeros(n)
    net_ret = np.zeros(n)
    gross_ret = np.zeros(n)
    cost_ret = np.zeros(n)
    fills: list[Fill] = []

    bars_per_8h = 8 * 3_600_000 / spec.interval_ms
    warmup = strategy.warmup

    pos = 0.0
    eq = initial_equity
    pending_age = 0  # bars the current unfilled limit order has been resting

    for t in range(warmup, n - 1):
        # 1. Learn from what the previous decision actually earned.
        strategy.observe(t)

        # 2. Decide the target for the next bar.
        target = float(np.clip(strategy.decide(t), -max_leverage, max_leverage))

        # 3. Execute during bar t+1.
        e = t + 1
        force_taker = execution.cross_after_ttl and pending_age >= execution.ttl_bars
        fill = broker.try_execute(
            bar=e,
            target=target,
            position=pos,
            open_=op[e],
            high=hi[e],
            low=lo[e],
            atr_frac=float(atr_frac[t]),
            quote_volume=float(qv[e]),
            bar_vol_bp=float(bar_vol_bp[t]),
            equity=eq,
            force_taker=force_taker,
        )

        # 4. Attribute P&L across the bar, split at the fill price.
        if fill is None:
            g = pos * (cl[e] / cl[t] - 1.0)
            c = 0.0
            wanted = abs(target - pos) >= execution.min_trade
            pending_age = pending_age + 1 if wanted else 0
        else:
            g = pos * (fill.price / cl[t] - 1.0) + target * (cl[e] / fill.price - 1.0)
            c = abs(fill.delta) * fill.cost_bp * 1e-4
            pos = target
            fills.append(fill)
            pending_age = 0

        g += broker.funding_cost(pos, bars_per_8h)
        r = g - c
        eq *= 1.0 + r

        equity[e] = eq
        position[e] = pos
        gross_ret[e] = g
        cost_ret[e] = c
        net_ret[e] = r

        strategy.record(t, target=target, realized_position=pos, equity=eq)

    equity[: warmup + 1] = initial_equity
    return BacktestResult(
        ts=ts,
        equity=equity,
        position=position,
        returns=net_ret,
        gross_returns=gross_ret,
        costs=cost_ret,
        fills=fills,
        diagnostics=strategy.diagnostics(),
        spec=spec,
    )
