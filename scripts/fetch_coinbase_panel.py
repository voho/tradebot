#!/usr/bin/env python
"""Fetch a PANEL of Coinbase 5m spot series concurrently (R-56).

``scripts/fetch_coinbase_spot.py`` fetches one product with sequential
requests, which is fine for one asset and far too slow for a panel: at the
public candles endpoint's 300-candle cap one request covers 25 hours, so
six and a half years of 5-minute bars is ~2,300 requests per product. From
inside this project's proxied session each request costs ~1.5-2s of
round-trip latency, so a seven-product panel is hours sequentially and
under an hour with a modest amount of concurrency.

This module is the concurrent version, and nothing else: same endpoint,
same granularity, same output format (``timestamp,open,high,low,close,
volume``, ms UTC epoch, gzipped) as ``fetch_coinbase_spot.py``, so the
files it writes load through ``tradebot.data.load_ohlcv_csv`` unchanged.
Requests are issued from a bounded thread pool with a global pacer that
holds the aggregate rate under Coinbase's public limit (~10 req/s per IP);
429s back off and retry rather than silently dropping a chunk.

Usage::

    python scripts/fetch_coinbase_panel.py \\
        --products BCH-USD LTC-USD ETC-USD \\
        --start 2020-01-01 --end 2026-08-20 --out-dir data

The output filename per product is ``<base><quote>_coinbase_spot_5m.csv.gz``
lowercased (BCH-USD -> ``bchusd_coinbase_spot_5m.csv.gz``), matching the
existing ``ethusd_coinbase_spot_5m.csv.gz`` convention.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ENDPOINT = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 300  # seconds = 5 minutes
MAX_CANDLES = 300
CHUNK = timedelta(seconds=GRANULARITY * MAX_CANDLES)
HEADERS = {"User-Agent": "tradebot-research/1.0"}


class Pacer:
    """Global minimum interval between request starts, shared by all threads."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def fetch_chunk(product: str, start: datetime, end: datetime, pacer: Pacer,
                retries: int = 6) -> list[list[float]]:
    url = (f"{ENDPOINT.format(product=product)}?start={start.isoformat()}"
           f"&end={end.isoformat()}&granularity={GRANULARITY}")
    last_err: Exception | None = None
    for attempt in range(retries):
        pacer.wait()
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # 429 / 5xx
            last_err = exc
            time.sleep(1.0 + 2.0 * attempt)
        except Exception as exc:  # noqa: BLE001 - network flakiness
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {product} {start}-{end}: {last_err}")


def fetch_product(product: str, start: datetime, end: datetime, pacer: Pacer,
                  workers: int) -> dict[datetime, tuple]:
    windows = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + CHUNK, end)
        windows.append((cursor, chunk_end))
        cursor = chunk_end

    rows: dict[datetime, tuple] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_chunk, product, a, b, pacer) for a, b in windows]
        for fut in futures:
            batch = fut.result()
            for t, low, high, open_, close, volume in batch:
                ts = datetime.fromtimestamp(t, tz=timezone.utc)
                rows[ts] = (open_, high, low, close, volume)
            done += 1
            if done % 200 == 0:
                print(f"  {product}: {done}/{len(windows)} chunks, {len(rows):,} bars",
                      file=sys.stderr, flush=True)
    return rows


def write_series(rows: dict[datetime, tuple], out_path: Path) -> int:
    ordered = sorted(rows.items())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename, so a fetch interrupted mid-write can
    # never leave a truncated .csv.gz behind under the real name — a
    # half-written gzip loads as a plausible-looking short series, which is
    # the kind of silent data defect this project's own audits keep finding.
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with gzip.open(tmp, "wt") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for ts, (o, h, l, c, v) in ordered:
            f.write(f"{int(ts.timestamp() * 1000)},{o},{h},{l},{c},{v}\n")
    tmp.replace(out_path)
    return len(ordered)


def default_name(product: str) -> str:
    base, quote = product.split("-")
    return f"{base.lower()}{quote.lower()}_coinbase_spot_5m.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", required=True)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-08-20")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--workers", type=int, default=10,
                    help="concurrent in-flight requests per product")
    ap.add_argument("--min-interval", type=float, default=0.13,
                    help="global seconds between request starts (~7.7 req/s)")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pacer = Pacer(args.min_interval)
    out_dir = Path(args.out_dir)

    for product in args.products:
        out_path = out_dir / default_name(product)
        if out_path.exists():
            print(f"{product}: {out_path} exists, skipping", file=sys.stderr)
            continue
        t0 = time.time()
        print(f"fetching {product} {start.date()} -> {end.date()}",
              file=sys.stderr, flush=True)
        rows = fetch_product(product, start, end, pacer, args.workers)
        if not rows:
            print(f"{product}: no data returned, skipping", file=sys.stderr)
            continue
        n = write_series(rows, out_path)
        span = (min(rows), max(rows))
        print(f"{product}: wrote {n:,} bars ({span[0]:%Y-%m-%d} -> {span[1]:%Y-%m-%d}) "
              f"to {out_path} in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
