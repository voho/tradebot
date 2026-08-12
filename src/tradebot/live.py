"""Run a strategy outside the backtester — in a live/paper trading bot.

Strategies are pure decision functions over (candle history, account
state): nothing in the Strategy API references the backtest engine, so
the exact same class that was ranked in the comparison table can drive a
live bot. This module is the extraction point:

    from tradebot.live import LiveAccount, compute_signal

    strategy = get_strategy("macd_rsi")
    account = LiveAccount(position=current_btc_position,
                          equity_quote=account_equity_usd,
                          market=MarketSpec.spot())
    orders = compute_signal(strategy, candles, account)   # on every closed bar

``candles`` is the same OHLCV DataFrame shape the backtester uses
(UTC-indexed ``open/high/low/close/volume``, oldest first, ONLY closed
bars — never include the forming candle). Feed at least
``strategy.warmup + 50`` bars for stable indicators. The returned
:class:`~tradebot.orders.Order` objects are venue-agnostic; the adapter
translates them:

- **Binance / Bitstamp (REST or websocket loop)**: for ``order.target``
  f, desired notional = ``account.equity_quote * market.leverage * f``;
  delta qty = (desired - current position notional) / price -> place a
  market order for the delta. ``order.qty`` orders map 1:1.
- **3Commas (signal bots via webhook)**: ``target > 0`` -> send the
  bot's *start/long* signal, ``target == 0`` -> *close* signal,
  ``target < 0`` -> *short* signal; sizing is configured in the bot.

Timing contract (identical to the backtest): decide on a bar's CLOSE,
execute at the NEXT bar's open — i.e. call ``compute_signal`` right
after a candle closes and act immediately. A parity test in
``tests/test_live.py`` verifies the signals reproduce the backtester's
fills bar for bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.engine import validate_ohlcv
from tradebot.orders import Order
from tradebot.strategy import Context, Strategy


@dataclass
class LiveAccount:
    """Minimal account view a live adapter supplies to the strategy.

    Duck-type compatible with the paper broker's surface that
    :class:`~tradebot.strategy.Context` reads (``pos``, ``equity()``,
    ``market``) — keep those three meaningful and any Strategy works.
    """

    position: float  # signed base-asset position (BTC), 0 when flat
    equity_quote: float  # current account equity in quote currency (USD)
    market: MarketSpec  # venue parameters (spot / futures leverage, fees)

    @property
    def pos(self) -> float:
        return self.position

    def equity(self, price: float) -> float:  # noqa: ARG002 - already marked
        return self.equity_quote


def compute_signal(strategy: Strategy, candles: pd.DataFrame,
                   account: LiveAccount) -> list[Order]:
    """Ask ``strategy`` for orders as of the latest CLOSED candle.

    Returns an empty list while the window is shorter than the
    strategy's warmup. Orders use the same semantics as the backtest
    (``target`` = fraction of equity x leverage; ``qty`` = base units).
    """
    validate_ohlcv(candles)
    prepared = strategy.prepare(candles.copy())
    if prepared is None or len(prepared) != len(candles):
        raise ValueError(f"{strategy.name}.prepare() must keep the same rows")
    i = len(prepared) - 1
    if i < strategy.warmup:
        return []
    ctx = Context(prepared, i, account)
    strategy.on_bar(ctx)
    return ctx.orders
