"""An offline Exchange that replays committed data through the real API shape.

This exists so the live path can be tested without network access or an
account. It serves candles under the SAME constraints a venue imposes -
a hard page limit, ``end_ms`` paging, and never returning the forming
bar - so a test that passes here exercises the pagination, stitching and
warmup logic the real adapters rely on.

It also simulates a spot wallet, so a full bot loop (fetch -> decide ->
order -> balance changes) can be driven end to end deterministically.
"""

from __future__ import annotations

import pandas as pd

from tradebot.exchanges.base import Balance, Exchange, OrderResult

BARS_PER_DAY = 288


class ReplayExchange(Exchange):
    """Serve a historical frame as if it were a live venue."""

    name = "replay"

    def __init__(self, df: pd.DataFrame, max_candles_per_request: int = 1000,
                 taker_fee: float = 0.001, base: float = 0.0,
                 quote: float = 1_000.0) -> None:
        self._df = df
        self.max_candles_per_request = max_candles_per_request
        self.taker_fee = taker_fee
        self._base = base
        self._quote = quote
        self.orders: list[OrderResult] = []
        #: index of the most recent CLOSED bar the venue will admit
        self._cursor = len(df) - 1

    # -------------------------------------------------------------- test clock

    def set_now(self, index: int) -> None:
        """Pretend 'now' is just after bar ``index`` closed."""
        self._cursor = index

    @property
    def last_close(self) -> float:
        return float(self._df["close"].iloc[self._cursor])

    # ------------------------------------------------------------ market data

    def fetch_candles(self, symbol: str = "BTCUSD", minutes: int = 5,
                      limit: int = 1000, end_ms: int | None = None) -> pd.DataFrame:
        visible = self._df.iloc[: self._cursor + 1]
        if end_ms is not None:
            visible = visible[visible.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]
        limit = min(limit, self.max_candles_per_request)
        return visible.iloc[-limit:].copy()

    # ---------------------------------------------------------------- account

    def fetch_balance(self, symbol: str = "BTCUSD") -> Balance:
        return Balance(base=self._base, quote=self._quote)

    def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        price = self.last_close
        notional = qty * price
        fee = notional * self.taker_fee
        if side.lower() == "buy":
            self._quote -= notional + fee
            self._base += qty
        else:
            self._quote += notional - fee
            self._base -= qty
        result = OrderResult(side=side.lower(), qty=qty, price=price, dry_run=False)
        self.orders.append(result)
        return result
