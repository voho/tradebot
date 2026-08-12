"""Order types shared by strategies, the engine and the broker."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """A market order.

    Exactly one of ``qty`` or ``target`` is set:

    - ``qty``: absolute base-asset quantity to buy/sell.
    - ``target``: signed fraction of the maximum allowed notional
      (equity x leverage) the position should end up at. The broker
      translates it into a concrete quantity at fill time.
    """

    side: Side | None = None
    qty: float | None = None
    target: float | None = None

    def __post_init__(self) -> None:
        if (self.qty is None) == (self.target is None):
            raise ValueError("set exactly one of qty or target")
        if self.qty is not None:
            if self.side is None:
                raise ValueError("qty orders need an explicit side")
            if self.qty <= 0:
                raise ValueError("qty must be positive")


@dataclass
class Fill:
    """An executed order (one broker transaction)."""

    ts: object  # pandas.Timestamp
    side: Side
    qty: float
    price: float
    fee: float
    realized_pnl: float  # realized PnL of the closed part, before fees
    kind: str = "order"  # "order" or "liquidation"
