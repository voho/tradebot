"""Bitstamp spot adapter (stdlib only)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
import uuid

import pandas as pd

from tradebot.exchanges.base import Balance, Exchange, OrderResult, normalize_candles

BASE_URL = "https://www.bitstamp.net"
STEPS = {1: 60, 3: 180, 5: 300, 15: 900, 30: 1800, 60: 3600, 240: 14400, 1440: 86400}


class BitstampSpot(Exchange):
    """Bitstamp spot: OHLC candles, balances, market orders.

    This is the venue the committed dataset comes from, so backtest and
    live see the same price series - no basis, no venue mismatch.

    Public candles need no credentials. Trading needs an API key/secret
    (Bitstamp's v2 auth: HMAC-SHA256 over a canonical string, with the
    key id, a UUID nonce and a millisecond timestamp).
    ``dry_run=True`` is the default.

    Endpoints: ``GET /api/v2/ohlc/{pair}/`` (1000 candles max, ``step``
    in seconds, ``end`` as a unix second), ``POST /api/v2/balance/``,
    ``POST /api/v2/{buy,sell}/market/{pair}/``.
    """

    name = "bitstamp"
    max_candles_per_request = 1000
    taker_fee = 0.004  # Bitstamp's entry-tier taker fee is materially higher

    def __init__(self, api_key: str = "", api_secret: str = "",
                 dry_run: bool = True, timeout: int = 30,
                 base_url: str = BASE_URL) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self.timeout = timeout
        self.base_url = base_url

    # ------------------------------------------------------------------ http

    def _get(self, path: str, params: dict | None = None):
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "tradebot/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _post_signed(self, path: str, payload: dict | None = None):
        payload = payload or {}
        body = urllib.parse.urlencode(payload)
        nonce = str(uuid.uuid4())
        timestamp = str(int(time.time() * 1000))
        content_type = "application/x-www-form-urlencoded"
        host = self.base_url.split("://", 1)[1]
        message = (f"BITSTAMP {self.api_key}POST{host}{path}"
                   f"{'' if not body else ''}{content_type}{nonce}{timestamp}v2{body}")
        signature = hmac.new(self.api_secret.encode(), message.encode(),
                             hashlib.sha256).hexdigest().upper()
        headers = {
            "X-Auth": f"BITSTAMP {self.api_key}",
            "X-Auth-Signature": signature,
            "X-Auth-Nonce": nonce,
            "X-Auth-Timestamp": timestamp,
            "X-Auth-Version": "v2",
            "Content-Type": content_type,
            "User-Agent": "tradebot/0.1",
        }
        req = urllib.request.Request(f"{self.base_url}{path}", data=body.encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------- market data

    def fetch_candles(self, symbol: str = "btcusd", minutes: int = 5,
                      limit: int = 1000, end_ms: int | None = None) -> pd.DataFrame:
        params = {"step": STEPS[minutes],
                  "limit": min(limit, self.max_candles_per_request)}
        if end_ms is not None:
            params["end"] = int(end_ms // 1000)  # Bitstamp wants unix SECONDS
        data = self._get(f"/api/v2/ohlc/{symbol.lower()}/", params)
        ohlc = data.get("data", {}).get("ohlc", [])
        rows = [(int(c["timestamp"]) * 1000, c["open"], c["high"], c["low"],
                 c["close"], c["volume"]) for c in ohlc]
        df = normalize_candles(rows)
        return self._drop_forming(df, minutes)

    @staticmethod
    def _drop_forming(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        if df.empty:
            return df
        now_ms = time.time() * 1000
        bar_ms = minutes * 60 * 1000
        if df.index[-1].timestamp() * 1000 + bar_ms > now_ms:
            return df.iloc[:-1]
        return df

    # ---------------------------------------------------------------- account

    def fetch_balance(self, symbol: str = "btcusd") -> Balance:
        base_asset, quote_asset = symbol[:3].lower(), symbol[3:].lower()
        data = self._post_signed("/api/v2/balance/")
        return Balance(base=float(data.get(f"{base_asset}_available", 0.0)),
                       quote=float(data.get(f"{quote_asset}_available", 0.0)))

    def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, got {side!r}")
        if qty <= 0:
            raise ValueError("qty must be positive")
        ticker = self._get(f"/api/v2/ticker/{symbol.lower()}/")
        price = float(ticker["last"])
        if self.dry_run:
            return OrderResult(side=side, qty=qty, price=price, dry_run=True)
        raw = self._post_signed(f"/api/v2/{side}/market/{symbol.lower()}/",
                                {"amount": f"{qty:.8f}"})
        return OrderResult(side=side, qty=qty, price=price, dry_run=False,
                           venue_order_id=str(raw.get("id", "")), raw=raw)
