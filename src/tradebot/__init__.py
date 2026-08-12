"""Paper-testing framework for BTCUSD 5m trading strategies."""

from tradebot.broker import MarketSpec, PaperBroker
from tradebot.engine import BacktestResult, run_backtest
from tradebot.orders import Order, Side
from tradebot.registry import available_strategies, get_strategy, register
from tradebot.strategy import Context, Strategy

__all__ = [
    "MarketSpec",
    "PaperBroker",
    "BacktestResult",
    "run_backtest",
    "Order",
    "Side",
    "available_strategies",
    "get_strategy",
    "register",
    "Context",
    "Strategy",
]
