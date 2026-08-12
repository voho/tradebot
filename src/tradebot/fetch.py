"""Fetch real BTCUSDT 5m data from Binance public endpoints (stdlib only).

Produces the canonical CSVs used by the framework:

- ``data/btcusdt_perp_5m.csv``          (USDT-margined perpetual futures)
- ``data/btcusdt_spot_aligned_5m.csv``  (spot, aligned to the perp index)

Tries the bulk monthly archives on data.binance.vision first (fast), then
falls back to the paginated klines REST API. Needs outbound network access
to binance.com / binance.vision - run it on your own machine if your
environment blocks those hosts.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from tradebot import data as datamod

ARCHIVE = {
    "spot": "https://data.binance.vision/data/spot/monthly/klines/{sym}/5m/{sym}-5m-{ym}.zip",
    "perp": "https://data.binance.vision/data/futures/um/monthly/klines/{sym}/5m/{sym}-5m-{ym}.zip",
}
API = {
    "spot": "https://api.binance.com/api/v3/klines",
    "perp": "https://fapi.binance.com/fapi/v1/klines",
}
API_LIMIT = {"spot": 1000, "perp": 1500}
BAR_MS = 5 * 60 * 1000


def _http_get(url: str, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tradebot/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:  # noqa: BLE001 - network errors vary widely
            last = e
        time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def _norm_ms(t: float) -> int:
    """Normalize an epoch timestamp of unknown resolution to milliseconds."""
    t = float(t)
    if t > 1e14:  # microseconds (newer binance.vision spot files)
        return int(t // 1000)
    if t > 1e11:  # already ms
        return int(t)
    return int(t * 1000)  # seconds


def _rows_from_klines(raw_rows) -> list[tuple[int, float, float, float, float, float]]:
    out = []
    for r in raw_rows:
        out.append((
            _norm_ms(r[0]),
            float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
        ))
    return out


def _parse_archive_csv(payload: bytes) -> list[tuple[int, float, float, float, float, float]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode()
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        try:
            float(row[0])
        except ValueError:
            continue  # header line in newer archives
        rows.append(row)
    return _rows_from_klines(rows)


def _months(start: datetime, end: datetime) -> list[str]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch_klines(kind: str, symbol: str, start: datetime, end: datetime,
                 quiet: bool = False) -> pd.DataFrame:
    """Fetch 5m klines for [start, end) as an OHLCV DataFrame."""
    rows: list[tuple[int, float, float, float, float, float]] = []
    api_from = start
    for ym in _months(start, end):
        url = ARCHIVE[kind].format(sym=symbol, ym=ym)
        try:
            payload = _http_get(url)
        except Exception:
            break  # current/partial month or archive gap: switch to the API
        rows.extend(_parse_archive_csv(payload))
        if not quiet:
            print(f"  archive {kind} {ym}: {len(rows)} rows total", file=sys.stderr)
        api_from = datetime.strptime(ym, "%Y-%m").replace(tzinfo=timezone.utc) + timedelta(days=32)
        api_from = api_from.replace(day=1)

    cursor = int(api_from.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    limit = API_LIMIT[kind]
    while cursor < end_ms:
        url = (f"{API[kind]}?symbol={symbol}&interval=5m"
               f"&startTime={cursor}&endTime={end_ms}&limit={limit}")
        batch = json.loads(_http_get(url))
        if not batch:
            break
        rows.extend(_rows_from_klines(batch))
        cursor = _norm_ms(batch[-1][0]) + BAR_MS
        if not quiet:
            print(f"  api {kind}: {len(rows)} rows total", file=sys.stderr)
        time.sleep(0.15)  # stay well under rate limits

    if not rows:
        raise RuntimeError(f"no {kind} data fetched for {symbol}")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(idx, name="timestamp")
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.loc[(out.index >= start) & (out.index < end)]


def fetch_data(data_dir: str | Path, symbol: str = "BTCUSDT",
               start: str = "2025-01-01", end: str | None = None) -> None:
    data_dir = Path(data_dir)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = (datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
              if end else datetime.now(timezone.utc))

    print(f"Fetching {symbol} 5m perp {start_dt:%Y-%m-%d} -> {end_dt:%Y-%m-%d} ...", file=sys.stderr)
    perp = fetch_klines("perp", symbol, start_dt, end_dt)
    print(f"Fetching {symbol} 5m spot ...", file=sys.stderr)
    spot = fetch_klines("spot", symbol, start_dt, end_dt)

    perp, spot = datamod.align(perp, spot)
    datamod.save_ohlcv_csv(perp, data_dir / datamod.CANONICAL["perp"])
    datamod.save_ohlcv_csv(spot, data_dir / datamod.CANONICAL["spot"])
    print(f"Wrote {len(perp)} aligned bars to {data_dir}/", file=sys.stderr)
