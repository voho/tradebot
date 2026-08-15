"""Backtest engine: bar-close signals, next-open execution.

Per-bar sequence (no lookahead by construction):

1. Fill orders queued on the previous bar at this bar's OPEN.
2. Check liquidation against this bar's extremes.
3. Record equity at this bar's CLOSE.
4. Call ``strategy.on_bar`` with history up to this bar; queue its orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from tradebot.broker import MarketSpec, PaperBroker
from tradebot.orders import Fill
from tradebot.strategy import Context, Strategy

OHLCV = ("open", "high", "low", "close", "volume")


@dataclass
class Trade:
    """One round trip: position opened from flat until back to flat."""

    direction: str  # "long" or "short"
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    qty: float  # max absolute position size during the episode
    pnl: float  # realized PnL after all fees of the episode
    fees: float
    liquidated: bool = False
    open_at_end: bool = False


@dataclass
class BacktestResult:
    strategy_name: str
    market: MarketSpec
    start_balance: float
    data_label: str
    equity: pd.Series  # equity at each bar close
    fills: list[Fill]
    trades: list[Trade]
    df: pd.DataFrame  # the (prepared) OHLCV frame the run used
    liquidated: bool
    fees_paid: float

    @property
    def final_balance(self) -> float:
        return float(self.equity.iloc[-1])


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("data index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("data index must be sorted ascending")
    if df.index.has_duplicates:
        raise ValueError("data index has duplicate timestamps")
    if len(df) and ((df[["open", "high", "low", "close"]] <= 0).any().any()):
        raise ValueError("prices must be positive")


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    market: MarketSpec,
    start_balance: float,
    slippage_bps: float = 0.0,
    data_label: str = "",
    trade_start: int = 0,
) -> BacktestResult:
    """Backtest ``strategy`` over ``df``.

    ``trade_start`` separates *warming up* from *trading*: bars before it
    feed the strategy (``on_bar`` still runs, so any internal state warms
    normally) but the orders they produce are discarded, so the account is
    untouched — flat, at ``start_balance`` — until that bar. Window
    resampling needs this: without it a leveraged strategy can be
    liquidated inside the data prefix and the window then measures a
    corpse rather than a fresh account.
    """
    validate_ohlcv(df)
    prepared = strategy.prepare(df.copy())
    if prepared is None:
        raise ValueError(f"{strategy.name}.prepare() returned None; return the DataFrame")
    if len(prepared) != len(df) or not prepared.index.equals(df.index):
        raise ValueError(f"{strategy.name}.prepare() must keep the same rows/index")

    broker = PaperBroker(market=market, start_balance=start_balance, slippage_bps=slippage_bps)

    cols = {c: prepared[c].to_numpy() for c in prepared.columns}
    opens = prepared["open"].to_numpy(dtype=float)
    highs = prepared["high"].to_numpy(dtype=float)
    lows = prepared["low"].to_numpy(dtype=float)
    closes = prepared["close"].to_numpy(dtype=float)
    index = prepared.index

    equity = [0.0] * len(prepared)
    fills: list[Fill] = []
    pending = []

    for i in range(len(prepared)):
        ts = index[i]
        # A gap can put the account past its liquidation price at the open,
        # before any queued order gets to trade — check the open first so a
        # close order cannot realize more loss than the account holds.
        liq = broker.check_liquidation(ts, opens[i], opens[i], opens[i])
        if liq is not None:
            fills.append(liq)
            pending = []

        if pending and not broker.dead:
            for order in pending:
                fills.extend(broker.execute(order, ts, opens[i]))
        pending = []

        liq = broker.check_liquidation(ts, opens[i], highs[i], lows[i])
        if liq is not None:
            fills.append(liq)

        equity[i] = broker.equity(closes[i])
        if not math.isfinite(equity[i]):
            # Fail loudly: an all-NaN curve otherwise reports 0% drawdown and
            # 0.00 Sharpe, which reads like a valid (if flat) result.
            raise ValueError(
                f"{strategy.name}: equity became non-finite at bar {i} "
                f"({index[i]}) - a NaN entered the account, check the strategy's "
                "target/qty for NaN")

        last_bar = i == len(prepared) - 1
        if not broker.dead and not last_bar and i >= strategy.warmup:
            ctx = Context(prepared, i, broker, cols=cols)
            strategy.on_bar(ctx)
            # Before trade_start the strategy is only warming its state; its
            # orders are dropped so the account stays flat at start_balance.
            # The first surviving order is queued AT trade_start and fills at
            # the next open, so equity[trade_start] == start_balance exactly.
            if i >= trade_start:
                pending = ctx.orders

    trades = build_trades(fills, end_price=closes[-1] if len(closes) else None,
                          broker=broker)

    return BacktestResult(
        strategy_name=strategy.name,
        market=market,
        start_balance=start_balance,
        data_label=data_label,
        equity=pd.Series(equity, index=index, name="equity"),
        fills=fills,
        trades=trades,
        df=prepared,
        liquidated=broker.dead,
        fees_paid=broker.fees_paid,
    )


def build_trades(fills: list[Fill], end_price, broker: PaperBroker) -> list[Trade]:
    """Group fills into round-trip trades.

    The broker executes sign flips as close-then-open, so the running
    position only ever moves toward or away from zero within one episode.
    An episode still open at the end of the data keeps ``exit_ts=None``,
    is marked ``open_at_end`` and its PnL includes the unrealized part
    marked at the last close.
    """
    trades: list[Trade] = []
    pos = 0.0
    episode_fills: list[Fill] = []

    def flush(exit_ts, exit_price, closed: bool, liquidated: bool) -> None:
        if not episode_fills:
            return
        first = episode_fills[0]
        direction = "long" if first.side.name == "BUY" else "short"
        fees = sum(f.fee for f in episode_fills)
        pnl = sum(f.realized_pnl for f in episode_fills) - fees
        # signed max exposure reached during the episode
        run = 0.0
        peak = 0.0
        entry_notional = 0.0
        entry_qty = 0.0
        for f in episode_fills:
            signed = f.qty if f.side.name == "BUY" else -f.qty
            increasing = abs(run + signed) > abs(run)
            run += signed
            peak = max(peak, abs(run))
            if increasing:
                entry_notional += f.qty * f.price
                entry_qty += f.qty
        entry_price = entry_notional / entry_qty if entry_qty else first.price
        if not closed and end_price is not None:
            pnl += (end_price - broker.entry) * broker.pos
        exits = [f for f in episode_fills if
                 (f.side.name == "SELL") == (direction == "long")]
        exit_px = (sum(f.qty * f.price for f in exits) / sum(f.qty for f in exits)) if exits else None
        trades.append(Trade(
            direction=direction,
            entry_ts=first.ts,
            exit_ts=exit_ts,
            entry_price=entry_price,
            exit_price=exit_px if closed else exit_price,
            qty=peak,
            pnl=pnl,
            fees=fees,
            liquidated=liquidated,
            open_at_end=not closed,
        ))

    for f in fills:
        signed = f.qty if f.side.name == "BUY" else -f.qty
        episode_fills.append(f)
        pos += signed
        if abs(pos) < 1e-12:
            pos = 0.0
            flush(f.ts, None, closed=True, liquidated=(f.kind == "liquidation"))
            episode_fills = []

    if episode_fills:
        flush(None, end_price, closed=False, liquidated=False)
    return trades
