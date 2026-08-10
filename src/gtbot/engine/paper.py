"""Paper and live trading loop.

The point of this module is that there is **no second implementation of the
decision logic**.  ``run_backtest`` and ``PaperTrader`` drive the same
:class:`~gtbot.strategy.GameTheoreticStrategy` object through the same
``prepare / observe / decide / record`` contract.  The only thing that changes
is where bars come from and where orders go — which is exactly the boundary
that should differ between simulation and production, and nothing else.

A run keeps a rolling window of history because every feature is a trailing
window computation; the window must be at least the feature warm-up or the
first live decisions would be made on half-formed features.

Nothing here places real orders.  :class:`BrokerAdapter` defines the interface a
live venue must implement, and :class:`PaperBroker` is a fill simulator that
marks against subsequent bars.  Connecting a real exchange means implementing
one small class, not rewriting the strategy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..data.schema import BTCUSD_5M, BarSpec, validate
from ..risk import RiskConfig
from ..strategy import GameTheoreticStrategy


@dataclass
class PaperOrder:
    ts: int
    side: str  # "buy" | "sell"
    qty: float  # signed change in position, as a fraction of equity
    price: float
    is_maker: bool
    fee_bp: float


class BrokerAdapter(Protocol):
    """What a venue must provide for the paper/live loop to run against it."""

    def position(self) -> float: ...

    def equity(self) -> float: ...

    def submit(self, target: float, reference_price: float, ts: int) -> PaperOrder | None: ...


@dataclass
class PaperBroker:
    """Fill simulator that behaves like a venue for the paper loop.

    Orders are filled at the next observed price with a configurable cost, so a
    paper session's P&L is directly comparable with a backtest over the same
    bars.
    """

    starting_equity: float = 100_000.0
    taker_fee_bp: float = 4.5
    half_spread_bp: float = 0.35
    min_trade: float = 0.02

    _position: float = field(default=0.0, init=False)
    _equity: float = field(default=0.0, init=False)
    _last_price: float = field(default=0.0, init=False)
    orders: list[PaperOrder] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._equity = self.starting_equity

    def position(self) -> float:
        return self._position

    def equity(self) -> float:
        return self._equity

    def mark(self, price: float) -> None:
        """Mark the open position to a new price."""
        if self._last_price > 0 and self._position != 0.0:
            self._equity *= 1.0 + self._position * (price / self._last_price - 1.0)
        self._last_price = price

    def submit(self, target: float, reference_price: float, ts: int) -> PaperOrder | None:
        delta = target - self._position
        if abs(delta) < self.min_trade:
            return None
        fee_bp = self.taker_fee_bp + self.half_spread_bp
        self._equity *= 1.0 - abs(delta) * fee_bp * 1e-4
        self._position = target
        order = PaperOrder(
            ts=ts,
            side="buy" if delta > 0 else "sell",
            qty=delta,
            price=reference_price,
            is_maker=False,
            fee_bp=fee_bp,
        )
        self.orders.append(order)
        return order


@dataclass
class PaperSession:
    """State a paper run reports and persists."""

    started_ts: int
    bars_seen: int = 0
    decisions: int = 0
    orders: int = 0
    equity: float = 0.0
    position: float = 0.0
    last_signal: float = 0.0
    last_edge_bp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class PaperTrader:
    """Drives the strategy over a live or replayed stream of bars.

    Usage::

        trader = PaperTrader(strategy, broker)
        trader.warm_up(history)          # a DataFrame of past bars
        for bar in stream:               # dicts in the canonical schema
            trader.on_bar(bar)
    """

    def __init__(
        self,
        strategy: GameTheoreticStrategy,
        broker: BrokerAdapter | None = None,
        *,
        spec: BarSpec = BTCUSD_5M,
        history_bars: int | None = None,
        state_path: str | Path | None = None,
        max_leverage: float = 2.0,
    ):
        self.strategy = strategy
        self.broker = broker or PaperBroker()
        self.spec = spec
        self.max_leverage = max_leverage
        self.state_path = Path(state_path) if state_path else None
        # Keep enough history for every trailing window plus the learner's own
        # standardisation window; anything less and early live decisions are
        # made on features that have not converged.
        self.history_bars = history_bars or (strategy.cfg.features.warmup + 2016 + 64)
        #: History is append-only during a session.  A rolling window would
        #: shift every index each bar, and the strategy's per-bar arrays (its
        #: signal history, its edge-estimator bookkeeping) are indexed by bar
        #: position — shifting them silently corrupts the online state.
        self.max_history = max(self.history_bars * 4, 200_000)
        self._history: pd.DataFrame | None = None
        self._prepared = False
        self.session = PaperSession(started_ts=int(time.time() * 1000))

    # ------------------------------------------------------------------
    def warm_up(self, bars: pd.DataFrame) -> None:
        """Seed the strategy with history so it starts converged, not cold."""
        frame = validate(bars, self.spec, strict=False)
        self._history = frame.tail(self.history_bars).reset_index(drop=True)
        self._replay()
        self._prepared = True

    def _replay(self) -> None:
        """Re-run the online learner over the retained history."""
        assert self._history is not None
        self.strategy.prepare(self._history)
        n = len(self._history)
        for t in range(self.strategy.warmup, n - 1):
            self.strategy.observe(t)
            self.strategy.decide(t)
            self.strategy.record(
                t,
                target=0.0,
                realized_position=self.broker.position(),
                equity=self.broker.equity(),
            )

    # ------------------------------------------------------------------
    def on_bar(self, bar: dict) -> PaperOrder | None:
        """Process one closed bar and act on it.

        ``bar`` must be a mapping in the canonical schema.  Returns the order
        submitted, if any.
        """
        if self._history is None:
            raise RuntimeError("call warm_up() before streaming bars")

        row = pd.DataFrame([bar])
        self._history = pd.concat([self._history, row], ignore_index=True)
        if len(self._history) > self.max_history:
            # Trimming shifts indices, so the online state has to be rebuilt.
            self._history = self._history.tail(self.history_bars).reset_index(drop=True)
            self._replay()
        self.session.bars_seen += 1

        # Recompute features over the retained window.  This is the same code
        # path the backtester uses, so a paper decision is bit-for-bit the
        # decision the backtest would have made on the same window.
        self.strategy.prepare(self._history, preserve_state=self._prepared)
        t = len(self._history) - 1
        if t <= self.strategy.warmup:
            return None

        price = float(self._history["close"].iloc[-1])
        self.broker.mark(price)

        self.strategy.observe(t)
        target = float(np.clip(self.strategy.decide(t), -self.max_leverage, self.max_leverage))
        self.session.decisions += 1

        order = self.broker.submit(target, price, int(self._history["ts"].iloc[-1]))
        self.strategy.record(
            t,
            target=target,
            realized_position=self.broker.position(),
            equity=self.broker.equity(),
        )

        self.session.orders += 1 if order else 0
        self.session.equity = self.broker.equity()
        self.session.position = self.broker.position()
        self.session.last_signal = float(self.strategy._z[t])
        self.session.last_edge_bp = 1e4 * self.strategy._edge_estimate()
        self._persist()
        return order

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.session.to_dict(), indent=2))


def replay_paper(
    bars: pd.DataFrame,
    strategy: GameTheoreticStrategy,
    *,
    warmup_bars: int | None = None,
    spec: BarSpec = BTCUSD_5M,
    max_leverage: float = 2.0,
) -> tuple[PaperSession, PaperBroker]:
    """Replay historical bars through the paper loop.

    This is the cross-check that the paper path and the backtest path agree:
    run both over the same bars and compare the equity curves.
    """
    broker = PaperBroker()
    trader = PaperTrader(strategy, broker, spec=spec, max_leverage=max_leverage)
    split = warmup_bars or (strategy.cfg.features.warmup + 2016)
    trader.warm_up(bars.iloc[:split])
    for _, row in bars.iloc[split:].iterrows():
        trader.on_bar(row.to_dict())
    return trader.session, broker
