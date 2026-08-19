#!/usr/bin/env python
"""Fetch a real BTC/ETH perpetual 5m OHLCV price series from Deribit (B-15).

This repository has never had a real perpetual price series: the futures
market in every backtest runs on the spot series
(``tradebot.data.load_dataset`` labels it ``spot (perp proxy)``), so the
basis between spot and perp is identically zero by construction. R-39
found that not just funding but the trade-price endpoint is reachable
(Deribit 200s where Binance still 451s at the proxy) and confirmed
``get_tradingview_chart_data`` returns real candles. This script pulls
that series for both ``BTC-PERPETUAL`` and ``ETH-PERPETUAL`` so the basis
becomes a genuinely new, independently-observed data channel rather than
something derived from the existing spot series.

Coverage is NOT the full 2017-2026 span the spot dataset has: Deribit's
BTC-PERPETUAL has no chart data before roughly 2018-08/09 (probed
empirically — 2018-06-01 returns ``no_data``, 2018-09-01 returns data),
and ETH-PERPETUAL was created 2019-03-14. That is a real, stated data
limitation, not an oversight; downstream code must treat bars before
first-available as missing rather than back-filling or proxying them
(the standing rule: never proxy unavailable data out of price).

Usage::

    python scripts/fetch_deribit_perp_price.py --instrument BTC-PERPETUAL \\
        --start 2018-01-01 --end 2026-08-19 \\
        --out data/btcusdt_deribit_perp_5m.csv.gz
    python scripts/fetch_deribit_perp_price.py --instrument ETH-PERPETUAL \\
        --start 2019-01-01 --end 2026-08-19 \\
        --out data/ethusdt_deribit_perp_5m.csv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ENDPOINT = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
RESOLUTION = "5"                 # minutes
CHUNK = timedelta(days=10)       # 10d x 288 bars/day = 2,880 rows/call, well under the ~5,000 cap


def _fetch_chunk(instrument: str, start_ms: int, end_ms: int, retries: int = 5) -> dict:
    url = (f"{ENDPOINT}?instrument_name={instrument}&start_timestamp={start_ms}"
           f"&end_timestamp={end_ms}&resolution={RESOLUTION}")
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=25) as resp:
                payload = json.loads(resp.read())
            return payload["result"]
        except Exception as exc:  # noqa: BLE001 - network flakiness, just retry
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {instrument} {start_ms}-{end_ms}: {last_err}")


def fetch_range(instrument: str, start: datetime, end: datetime) -> list[tuple[datetime, float, float, float, float, float]]:
    rows: list[tuple[datetime, float, float, float, float, float]] = []
    cursor = start
    first_data_seen = False
    while cursor < end:
        chunk_end = min(cursor + CHUNK, end)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)
        result = _fetch_chunk(instrument, start_ms, end_ms)
        status = result.get("status")
        ticks = result.get("ticks", [])
        if status == "ok" and ticks:
            first_data_seen = True
            for t, o, h, l, c, v in zip(ticks, result["open"], result["high"],
                                        result["low"], result["close"], result["volume"]):
                rows.append((datetime.fromtimestamp(t / 1000, tz=timezone.utc), o, h, l, c, v))
        print(f"  {instrument} {cursor.date()} -> {chunk_end.date()}: "
              f"{status} {len(ticks)} rows", file=sys.stderr)
        cursor = chunk_end
    if not first_data_seen:
        raise SystemExit(f"no data ever returned for {instrument} in "
                          f"[{start.date()}, {end.date()})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="BTC-PERPETUAL")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-08-19")
    ap.add_argument("--out", default="data/btcusdt_deribit_perp_5m.csv.gz")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"fetching Deribit {args.instrument} 5m OHLCV {start.date()} -> {end.date()}",
          file=sys.stderr)
    rows = fetch_range(args.instrument, start, end)
    rows.sort(key=lambda r: r[0])

    # dedupe (chunk boundaries can repeat a bar)
    seen: dict[datetime, tuple] = {}
    for r in rows:
        seen[r[0]] = r
    ordered = sorted(seen.items())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for ts, (_, o, h, l, c, v) in ordered:
            f.write(f"{int(ts.timestamp() * 1000)},{o},{h},{l},{c},{v}\n")

    first_ts, last_ts = ordered[0][0], ordered[-1][0]
    print(f"wrote {len(ordered)} 5m bars ({first_ts} -> {last_ts}) to {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
