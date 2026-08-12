"""Strategy base class and the per-bar Context handed to strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tradebot.orders import Order, Side

if TYPE_CHECKING:
    from tradebot.broker import MarketSpec, PaperBroker


class Context:
    """What a strategy sees on one bar, and how it places orders.

    The engine calls ``strategy.on_bar(ctx)`` at the close of every bar.
    Orders queued here are executed at the *next* bar's open, so nothing
    the strategy does can use future data.
    """

    def __init__(self, df: pd.DataFrame, i: int, broker: "PaperBroker") -> None:
        self._df = df
        self.i = i
        self._broker = broker
        self.orders: list[Order] = []

    # ------------------------------------------------------------- market data

    @property
    def ts(self) -> pd.Timestamp:
        return self._df.index[self.i]

    @property
    def bar(self) -> pd.Series:
        """Current (just closed) bar, including any prepare() columns."""
        return self._df.iloc[self.i]

    @property
    def prev(self) -> pd.Series | None:
        """Previous bar, or None on the first bar."""
        return self._df.iloc[self.i - 1] if self.i > 0 else None

    @property
    def close(self) -> float:
        return float(self._df["close"].iloc[self.i])

    def history(self, n: int | None = None) -> pd.DataFrame:
        """All closed bars up to and including the current one (last n rows)."""
        end = self.i + 1
        start = 0 if n is None else max(0, end - n)
        return self._df.iloc[start:end]

    # ------------------------------------------------------------ account state

    @property
    def market(self) -> "MarketSpec":
        return self._broker.market

    @property
    def can_short(self) -> bool:
        return self._broker.market.allow_short

    @property
    def position(self) -> float:
        """Signed base-asset position."""
        return self._broker.pos

    @property
    def in_market(self) -> bool:
        return self._broker.pos != 0.0

    @property
    def equity(self) -> float:
        return self._broker.equity(self.close)

    # ----------------------------------------------------------------- orders

    def submit(self, order: Order) -> Order:
        self.orders.append(order)
        return order

    def order_target(self, fraction: float) -> Order:
        """Move the position to ``fraction`` of max notional (equity x leverage).

        1.0 = fully long, 0.0 = flat, -1.0 = fully short (futures only;
        clamped to 0 on spot).
        """
        return self.submit(Order(target=fraction))

    def buy(self, qty: float) -> Order:
        return self.submit(Order(side=Side.BUY, qty=qty))

    def sell(self, qty: float) -> Order:
        return self.submit(Order(side=Side.SELL, qty=qty))

    def close_position(self) -> Order:
        return self.order_target(0.0)


class Strategy:
    """Base class for strategies.

    Subclass, set ``name`` (unique), optionally ``warmup`` (bars to skip
    before the first ``on_bar`` call), and implement:

    - ``prepare(df)``: called once with the full OHLCV frame; add indicator
      columns and return the frame. MUST be causal: row i may only depend
      on rows <= i (rolling / ewm / shift are fine). This is verified by a
      framework test for every registered strategy.
    - ``on_bar(ctx)``: called at each bar close; place orders via ctx.

    Keep strategies parameterizable via __init__ keyword arguments with
    defaults, so the registry can instantiate them with no arguments.
    """

    name: str = "base"
    warmup: int = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx: Context) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        """One-line description used in reports."""
        return (self.__doc__ or "").strip().splitlines()[0] if self.__doc__ else ""
