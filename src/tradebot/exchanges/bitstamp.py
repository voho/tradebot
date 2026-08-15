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
CONTENT_TYPE = "application/x-www-form-urlencoded"


def _raise_for_error(data, path: str):
    """Bitstamp reports failures in a 200 body, so they must be read.

    A rejected order otherwise returns as an ``OrderResult`` with an empty
    id, and the bot carries on believing it holds a position it does not.
    """
    if isinstance(data, dict) and (data.get("status") == "error" or "error" in data):
        reason = data.get("reason") or data.get("error")
        code = data.get("code", "")
        raise RuntimeError(f"bitstamp {path} failed: {code} {reason}".strip())
    return data


class BitstampSpot(Exchange):
    """Bitstamp spot: OHLC candles, balances, market orders.

    This is the venue the committed dataset comes from, so backtest and
    live see the same price series - no basis, no venue mismatch.

    Public candles need no credentials. Trading needs an API key/secret
    with the **Trade** permission (Bitstamp's v2 auth: HMAC-SHA256 over a
    canonical string, with the key id, a UUID nonce and a millisecond
    timestamp; the digest is compared verbatim, so it stays lowercase).
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

    def _signed_message(self, path: str, body: str, nonce: str,
                        timestamp: str) -> str:
        """The exact string Bitstamp v2 expects to be HMAC-SHA256 signed.

        Order is fixed and unforgiving::

            "BITSTAMP " + key + method + host + path + query
            + content-type + nonce + timestamp + "v2" + body

        Kept as its own method so a test can pin it without a network call.
        """
        host = self.base_url.split("://", 1)[1]
        return (f"BITSTAMP {self.api_key}POST{host}{path}"
                f"{CONTENT_TYPE}{nonce}{timestamp}v2{body}")

    def _post_signed(self, path: str, payload: dict | None = None):
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                f"{path} needs credentials; construct BitstampSpot(api_key=..., "
                "api_secret=...) with a key that has the Trade permission")
        # Bitstamp rejects a genuinely empty POST body with API0020, so a
        # request that takes no parameters still has to send one.
        body = urllib.parse.urlencode(payload or {"foo": "bar"})
        nonce = str(uuid.uuid4())
        timestamp = str(int(time.time() * 1000))
        message = self._signed_message(path, body, nonce, timestamp)
        # Lowercase hex: Bitstamp compares the digest verbatim, so upper-casing
        # it authenticates as garbage and every signed call fails.
        signature = hmac.new(self.api_secret.encode(), message.encode(),
                             hashlib.sha256).hexdigest()
        headers = {
            "X-Auth": f"BITSTAMP {self.api_key}",
            "X-Auth-Signature": signature,
            "X-Auth-Nonce": nonce,
            "X-Auth-Timestamp": timestamp,
            "X-Auth-Version": "v2",
            "Content-Type": CONTENT_TYPE,
            "User-Agent": "tradebot/0.1",
        }
        req = urllib.request.Request(f"{self.base_url}{path}", data=body.encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return _raise_for_error(data, path)

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
