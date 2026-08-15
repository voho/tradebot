"""Exchange adapters for running strategies live (spot only)."""

from tradebot.exchanges.base import Balance, Exchange, OrderResult
from tradebot.exchanges.binance import BinanceSpot
from tradebot.exchanges.bitstamp import BitstampSpot
from tradebot.exchanges.replay import ReplayExchange

__all__ = ["Balance", "Exchange", "OrderResult", "BinanceSpot", "BitstampSpot",
           "ReplayExchange"]
