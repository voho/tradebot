"""The live trading loop: fetch closed candles, decide, place the delta order.

One function, :func:`step`, does a single iteration and is what a
scheduler should call just after each candle closes. It is deliberately
stateless — the position lives in the exchange balance, not in memory —
so a restarted bot picks up exactly where it left off.

Spot only: targets are clamped to [0, 1] of equity, so the bot is long
or flat and can never be liquidated.

    from tradebot.bot import BotConfig, step
    from tradebot.exchanges.binance import BinanceSpot
    from tradebot.registry import get_strategy

    ex = BinanceSpot(api_key, api_secret, dry_run=True)   # start in dry run
    cfg = BotConfig(symbol="BTCUSDT", strategy="kelly_regime_v3")
    step(ex, cfg, get_strategy(cfg.strategy))
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from tradebot.broker import MarketSpec
from tradebot.exchanges.base import Exchange
from tradebot.live import LiveAccount, compute_signal
from tradebot.strategy import Strategy


@dataclass
class BotConfig:
    symbol: str = "BTCUSDT"
    strategy: str = "kelly_regime_v3"
    minutes: int = 5
    #: extra bars fetched beyond the strategy's warmup, for indicator stability
    warmup_slack: int = 500
    #: don't trade unless the position needs to move by this fraction of equity
    min_rebalance: float = 0.10
    #: refuse orders below the venue's minimum notional
    min_notional: float = 10.0
    verbose: bool = True


@dataclass
class StepResult:
    target: float  # desired fraction of equity in base asset, [0, 1]
    current: float  # current fraction
    equity: float
    price: float
    order: object | None = None
    reason: str = ""


def step(exchange: Exchange, config: BotConfig, strategy: Strategy) -> StepResult:
    """Run one decision cycle. Returns what was decided and why."""
    bars = strategy.warmup + config.warmup_slack
    candles = exchange.fetch_history(config.symbol, bars=bars,
                                     minutes=config.minutes,
                                     progress=config.verbose)
    if len(candles) < strategy.warmup:
        return StepResult(0.0, 0.0, 0.0, 0.0, None,
                          f"insufficient history: {len(candles)}/{strategy.warmup} bars")

    price = float(candles["close"].iloc[-1])
    balance = exchange.fetch_balance(config.symbol)
    equity = balance.equity(price)
    if equity <= 0:
        return StepResult(0.0, 0.0, equity, price, None, "no equity")
    current = balance.base * price / equity

    # The strategy sees the same Context the backtester gives it. Spot means
    # 1x, long-only; the account view carries the real balances.
    account = LiveAccount(position=balance.base, equity_quote=equity,
                          market=MarketSpec.spot(fee_rate=exchange.taker_fee))
    orders = compute_signal(strategy, candles, account)

    target = current
    for order in orders:
        if order.target is None:
            continue  # qty orders are venue-specific; targets are the live API
        # order_notional() already divides by leverage, which is 1.0 on spot
        target = min(1.0, max(0.0, float(order.target)))

    delta_frac = target - current
    delta_notional = delta_frac * equity
    if abs(delta_frac) < config.min_rebalance:
        return StepResult(target, current, equity, price, None,
                          f"inside deadband ({delta_frac:+.3f})")
    if abs(delta_notional) < config.min_notional:
        return StepResult(target, current, equity, price, None,
                          f"below min notional (${abs(delta_notional):.2f})")

    side = "buy" if delta_frac > 0 else "sell"
    qty = abs(delta_notional) / price
    if side == "sell":
        qty = min(qty, balance.base)  # never sell more than we hold
    result = exchange.place_market_order(config.symbol, side, qty)
    if config.verbose:
        tag = "DRY RUN" if getattr(result, "dry_run", False) else "LIVE"
        print(f"[{tag}] {side} {qty:.8f} @ ~{price:,.2f} "
              f"({current:.2f} -> {target:.2f} of equity)", file=sys.stderr)
    return StepResult(target, current, equity, price, result, "traded")
