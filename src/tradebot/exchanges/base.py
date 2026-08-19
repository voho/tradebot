"""Exchange adapter interface for running a strategy live (spot only).

The framework's strategies are pure decision functions, so going live is
a matter of three things: fetch candles, read the account, place an
order. Any venue that can do those three implements ``Exchange`` and
works with :func:`tradebot.bot.step`.

Design constraints, all deliberate:

- **stdlib only.** No ccxt, no requests - the adapters use urllib, so a
  bot deploys with the same dependencies the backtester needs.
- **Spot only.** Long or flat, no leverage, no shorting. This matches
  the ``MarketSpec.spot()`` the strategies were validated on.
- **Closed candles only.** ``fetch_candles`` must never return the
  forming bar; a strategy that sees a partial candle is reading the
  future. Adapters drop it explicitly.
- **Paginated history.** Exchanges cap a single request (1000 bars on
  both Binance and Bitstamp = ~3.5 days of 5m data), while the leading
  strategies need 80-100 days of warmup. Adapters page backwards until
  the window is filled; see ``docs/LIVE.md`` for the call counts.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd

BARS_PER_DAY = 288


@dataclass(frozen=True)
class Balance:
    """Spot balances for one pair."""

    base: float  # e.g. BTC held
    quote: float  # e.g. USD held

    def equity(self, price: float) -> float:
        return self.quote + self.base * price


@dataclass(frozen=True)
class OrderResult:
    """What the venue did (or, in dry-run, would have done)."""

    side: str  # "buy" | "sell"
    qty: float  # base units
    price: float  # reference price used for sizing
    dry_run: bool
    venue_order_id: str = ""
    raw: dict | None = None


class Exchange(abc.ABC):
    """Minimal spot venue: candles, balances, market orders."""

    name: str = "exchange"
    #: hard cap on candles returned by one API call
    max_candles_per_request: int = 1000
    #: taker fee actually charged, used to sanity-check backtest assumptions
    taker_fee: float = 0.001

    @abc.abstractmethod
    def fetch_candles(self, symbol: str, minutes: int = 5,
                      limit: int = 1000, end_ms: int | None = None) -> pd.DataFrame:
        """Return up to ``limit`` CLOSED candles ending at//before ``end_ms``.

        Must return the framework's OHLCV shape: a UTC DatetimeIndex named
        ``timestamp`` and float columns open/high/low/close/volume, sorted
        ascending, no duplicates, and NOT including the forming bar.
        """

    @abc.abstractmethod
    def fetch_balance(self, symbol: str) -> Balance:
        """Free base and quote balances for ``symbol``."""

    @abc.abstractmethod
    def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """Place a market order for ``qty`` base units."""

    # ---------------------------------------------------------------- helpers

    def fetch_history(self, symbol: str, bars: int, minutes: int = 5,
                      progress: bool = False) -> pd.DataFrame:
        """Page backwards until ``bars`` closed candles are collected.

        Strategies need long warmups (a 100-day regime anchor is 28,800
        five-minute bars) while venues cap one request at
        ``max_candles_per_request``, so a cold start costs
        ``ceil(bars / cap)`` calls. Pages are stitched, de-duplicated and
        returned oldest-first.

        The very first page (``end_ms=None``) is fetched up to *now*, so
        ``fetch_candles`` drops the still-forming bar from it - short of a
        full page even when the venue has plenty more history behind it,
        and real venues can shave off an extra row or two beyond that
        (a duplicate tick right at the boundary, ordinary clock jitter
        between when ``bars`` was sized and when the request lands).
        Every later page's ``end`` is already fixed in the past, so none
        of that applies - a short *historical* page is a reliable "no
        more data" signal, but a short *first* page is not. Treating any
        first-page shortfall as "venue exhausted" would truncate every
        real cold start whose warmup exceeds ``max_candles_per_request``
        bars to a single page - it was never caught because the test
        suite only pages a synthetic/replay venue that has no forming
        candle and no jitter to drop.
        """
        import sys

        chunks: list[pd.DataFrame] = []
        collected = 0
        end_ms: int | None = None
        page = 0
        while collected < bars:
            want = min(self.max_candles_per_request, bars - collected)
            is_live_page = end_ms is None  # up to "now" - may come back short, harmlessly
            chunk = self.fetch_candles(symbol, minutes=minutes, limit=want,
                                       end_ms=end_ms)
            if chunk.empty:
                break
            chunks.append(chunk)
            collected += len(chunk)
            page += 1
            if progress:
                print(f"  {self.name}: page {page}, {collected}/{bars} bars",
                      file=sys.stderr)
            # step strictly before the oldest bar we already hold
            end_ms = int(chunk.index[0].timestamp() * 1000) - 1
            if not is_live_page and len(chunk) < want:
                break  # a historical (fixed end_ms) page came back short: venue has no more history

        if not chunks:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = pd.concat(chunks[::-1])
        out = out[~out.index.duplicated(keep="first")].sort_index()
        return out.iloc[-bars:]


def normalize_candles(rows, tz_unit: str = "ms") -> pd.DataFrame:
    """Build the framework's OHLCV frame from (ts, o, h, l, c, v) tuples."""
    df = pd.DataFrame(list(rows),
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    idx = pd.to_datetime(df["timestamp"], unit=tz_unit, utc=True)
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(idx, name="timestamp")
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out
