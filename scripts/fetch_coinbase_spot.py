#!/usr/bin/env python
"""Fetch a real spot 5m OHLCV series from Coinbase Exchange's public candles API.

Built for the B-15 round: computing a real perp/spot basis for ETH needs an
ETH spot series that spans Deribit's ETH-PERPETUAL history (2019-03-14 ->
present); the only ETH spot series already in this repo
(``data/ethusd_bitfinex_5m.csv.gz``) stops in 2019-12, which would leave
under a year of overlap. Coinbase's public candle endpoint is reachable
from this session (Binance, the venue most of this repo's Binance-branded
files reference, still returns 451) and has traded ETH-USD since 2016.

The endpoint caps each request at 300 candles, so at 5-minute granularity
one call covers 300*5min = 25 hours; fetching several years means many
sequential calls, paced to stay under the public rate limit.

Usage::

    python scripts/fetch_coinbase_spot.py --product ETH-USD \\
        --start 2019-03-14 --end 2026-08-19 \\
        --out data/ethusd_coinbase_spot_5m.csv.gz
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

ENDPOINT = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 300                 # seconds = 5 minutes
MAX_CANDLES = 300
CHUNK = timedelta(seconds=GRANULARITY * MAX_CANDLES)
HEADERS = {"User-Agent": "tradebot-research/1.0"}


def _fetch_chunk(product: str, start: datetime, end: datetime, retries: int = 5) -> list[list[float]]:
    url = (f"{ENDPOINT.format(product=product)}?start={start.isoformat()}"
           f"&end={end.isoformat()}&granularity={GRANULARITY}")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {product} {start}-{end}: {last_err}")


def fetch_range(product: str, start: datetime, end: datetime) -> dict[datetime, tuple]:
    rows: dict[datetime, tuple] = {}
    cursor = start
    n_calls = 0
    while cursor < end:
        chunk_end = min(cursor + CHUNK, end)
        batch = _fetch_chunk(product, cursor, chunk_end)
        for t, low, high, open_, close, volume in batch:
            ts = datetime.fromtimestamp(t, tz=timezone.utc)
            rows[ts] = (open_, high, low, close, volume)
        n_calls += 1
        if n_calls % 20 == 0:
            print(f"  {product} {cursor.date()} -> {chunk_end.date()} "
                  f"({len(rows)} bars so far)", file=sys.stderr)
        cursor = chunk_end
        time.sleep(0.15)          # ~6-7 req/s, under Coinbase's public rate limit
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="ETH-USD")
    ap.add_argument("--start", default="2019-03-14")
    ap.add_argument("--end", default="2026-08-19")
    ap.add_argument("--out", default="data/ethusd_coinbase_spot_5m.csv.gz")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"fetching Coinbase {args.product} 5m candles {start.date()} -> {end.date()}",
          file=sys.stderr)
    rows = fetch_range(args.product, start, end)
    if not rows:
        raise SystemExit(f"no data returned for {args.product}")
    ordered = sorted(rows.items())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for ts, (o, h, l, c, v) in ordered:
            f.write(f"{int(ts.timestamp() * 1000)},{o},{h},{l},{c},{v}\n")

    print(f"wrote {len(ordered)} 5m bars ({ordered[0][0]} -> {ordered[-1][0]}) "
          f"to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
