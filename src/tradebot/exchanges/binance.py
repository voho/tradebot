"""Binance spot adapter (stdlib only)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from tradebot.exchanges.base import Balance, Exchange, OrderResult, normalize_candles

BASE_URL = "https://api.binance.com"
INTERVALS = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h",
             240: "4h", 1440: "1d"}


class BinanceSpot(Exchange):
    """Binance spot: klines, account balances, market orders.

    Read-only use (candles) needs no credentials. Trading needs an API
    key/secret with **spot trading enabled and withdrawals disabled**.
    ``dry_run=True`` (the default) computes and logs the order without
    sending it — always run that way first.

    Endpoints used: ``GET /api/v3/klines`` (1000 candles max per call),
    ``GET /api/v3/account`` (signed), ``POST /api/v3/order`` (signed).
    """

    name = "binance"
    max_candles_per_request = 1000
    taker_fee = 0.001  # 0.10% standard spot taker

    def __init__(self, api_key: str = "", api_secret: str = "",
                 dry_run: bool = True, timeout: int = 30,
                 base_url: str = BASE_URL) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self.timeout = timeout
        self.base_url = base_url

    # ------------------------------------------------------------------ http

    def _get(self, path: str, params: dict | None = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5_000
            query = urllib.parse.urlencode(params)
            params["signature"] = hmac.new(self.api_secret.encode(), query.encode(),
                                           hashlib.sha256).hexdigest()
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, params: dict):
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5_000
        query = urllib.parse.urlencode(params)
        params["signature"] = hmac.new(self.api_secret.encode(), query.encode(),
                                       hashlib.sha256).hexdigest()
        req = urllib.request.Request(f"{self.base_url}{path}",
                                     data=urllib.parse.urlencode(params).encode(),
                                     headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _headers(self) -> dict:
        h = {"User-Agent": "tradebot/0.1"}
        if self.api_key:
            h["X-MBX-APIKEY"] = self.api_key
        return h

    # ------------------------------------------------------------- market data

    def fetch_candles(self, symbol: str = "BTCUSDT", minutes: int = 5,
                      limit: int = 1000, end_ms: int | None = None) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": INTERVALS[minutes],
                  "limit": min(limit, self.max_candles_per_request)}
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        raw = self._get("/api/v3/klines", params)
        # kline: [openTime, o, h, l, c, v, closeTime, ...]
        rows = [(k[0], k[1], k[2], k[3], k[4], k[5]) for k in raw]
        df = normalize_candles(rows)
        return self._drop_forming(df, minutes)

    @staticmethod
    def _drop_forming(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Remove the still-open candle: a strategy must never see it."""
        if df.empty:
            return df
        now_ms = time.time() * 1000
        bar_ms = minutes * 60 * 1000
        last_open = df.index[-1].timestamp() * 1000
        if last_open + bar_ms > now_ms:
            return df.iloc[:-1]
        return df

    # ---------------------------------------------------------------- account

    def fetch_balance(self, symbol: str = "BTCUSDT") -> Balance:
        base_asset, quote_asset = self._split_symbol(symbol)
        data = self._get("/api/v3/account", signed=True)
        free = {b["asset"]: float(b["free"]) for b in data.get("balances", [])}
        return Balance(base=free.get(base_asset, 0.0), quote=free.get(quote_asset, 0.0))

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        for quote in ("USDT", "FDUSD", "USDC", "BUSD", "USD", "EUR", "BTC"):
            if symbol.endswith(quote):
                return symbol[: -len(quote)], quote
        raise ValueError(f"cannot split {symbol!r} into base/quote")

    def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be buy or sell, got {side!r}")
        if qty <= 0:
            raise ValueError("qty must be positive")
        price = float(self._get("/api/v3/ticker/price", {"symbol": symbol})["price"])
        if self.dry_run:
            return OrderResult(side=side.lower(), qty=qty, price=price, dry_run=True)
        raw = self._post("/api/v3/order", {"symbol": symbol, "side": side,
                                           "type": "MARKET",
                                           "quantity": f"{qty:.8f}".rstrip("0")})
        return OrderResult(side=side.lower(), qty=qty, price=price, dry_run=False,
                           venue_order_id=str(raw.get("orderId", "")), raw=raw)
