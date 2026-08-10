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
from .broker import Broker, CostModel, ExecutionConfig, Fill, MarginConfig


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
    #: Bar index at which the account was liquidated, if it was.
    liquidated_at: int | None = None
    #: Worst intrabar adverse excursion seen, as a fraction of the liquidation
    #: distance.  1.0 means the account touched liquidation.
    worst_margin_use: float = 0.0

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
    margin: MarginConfig | None = None,
) -> BacktestResult:
    """Run ``strategy`` over ``bars`` and return the realised equity path."""
    costs = costs or CostModel()
    execution = execution or ExecutionConfig()
    margin = margin or MarginConfig()
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
    liquidated_at: int | None = None
    worst_margin_use = 0.0

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
        pos_before = pos
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

        # 5. Liquidation, checked against the bar's *adverse extreme* rather
        #    than its close: the exchange closes the position the moment the
        #    price is touched, and a bar that recovered by its close would not
        #    have saved the account.
        #
        #    Both legs of the bar are checked, each anchored to the price it was
        #    actually opened at.  Checking only the post-fill position misses a
        #    bar whose fill flattened the book after the old position had
        #    already been wiped out; anchoring the new position to the previous
        #    close charges it a move it was never exposed to.  Bar data cannot
        #    resolve intrabar ordering, so each leg is conservatively assumed to
        #    have seen the whole bar's range.
        entry_price = fill.price if fill is not None else cl[t]
        use = max(
            margin.margin_use(pos_before, cl[t], lo[e], hi[e]),
            margin.margin_use(pos, entry_price, lo[e], hi[e]),
        )
        worst_margin_use = max(worst_margin_use, use)

        liquidated = use >= 1.0
        if not liquidated:
            eq_after = eq * (1.0 + r)
            # A leveraged account cannot go through zero; if the arithmetic says
            # it did, the position was liquidated on the way.
            liquidated = eq_after <= 0.0
        if liquidated:
            # What survives a forced close is at most the maintenance margin,
            # not the equity the account started the bar with.
            residual = margin.maintenance_margin_rate * abs(pos_before or pos) * eq
            eq_new = max(min(residual, eq), 0.0) * margin.liquidation_recovery
            net_ret[e] = eq_new / eq - 1.0 if eq > 0 else -1.0
            cost_ret[e] = c
            gross_ret[e] = net_ret[e] + c
            eq = eq_new
            liquidated_at = e
            pos = 0.0
            equity[e:] = eq
            position[e:] = 0.0
            strategy.record(t, target=0.0, realized_position=0.0, equity=eq)
            break

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
        liquidated_at=liquidated_at,
        worst_margin_use=worst_margin_use,
    )
