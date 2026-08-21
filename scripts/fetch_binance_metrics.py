#!/usr/bin/env python
"""Fetch Binance USDⓈ-M futures "metrics" (open interest + long/short
ratios) daily files concurrently (R-81).

Positioning/crowding data: open interest, top-trader and account-wide
long/short ratios, taker buy/sell volume ratio, at the exchange's own
5-minute native cadence -- reachable via the static `data.binance.vision`
history host even when the live `fapi.binance.com` API 451s from this
project's network policy (verified directly before this script was
written). Only daily files exist for this dataset (no monthly archive,
unlike the OHLCV klines this project already fetches elsewhere), so a
multi-year pull is one HTTP request per calendar day -- this is the
concurrent fetcher that makes that tractable, structured exactly like
``fetch_coinbase_panel.py`` (global pacer + thread pool, atomic
temp-file-then-rename writes).

Each daily CSV is genuinely duplicated row-for-row on the wire (every
5-minute timestamp appears twice, byte-identical) -- deduplicated here,
not left for the loader to discover.

Usage::

    python scripts/fetch_binance_metrics.py --symbols BTCUSDT ETHUSDT \\
        --start 2020-09-01 --end 2022-12-31 --out-dir data
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date}.zip"
HEADERS = {"User-Agent": "tradebot-research/1.0"}
COLUMNS = ("create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
           "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
           "count_long_short_ratio", "sum_taker_long_short_vol_ratio")


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


def fetch_day(symbol: str, day: datetime, pacer: Pacer,
              retries: int = 5) -> list[tuple]:
    """One calendar day's rows, deduplicated. Returns [] on 404 (a day the
    venue never published, e.g. a listing gap) rather than raising -- a
    multi-year pull should not die on one missing day."""
    url = BASE.format(symbol=symbol, date=day.strftime("%Y-%m-%d"))
    last_err: Exception | None = None
    for attempt in range(retries):
        pacer.wait()
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    text = f.read().decode()
            rows = []
            seen = set()
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                key = row["create_time"]
                if key in seen:  # the wire format duplicates every row
                    continue
                seen.add(key)
                rows.append(tuple(row[c] for c in COLUMNS))
            return rows
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []  # not published for this day; not a fetch failure
            last_err = exc
            time.sleep(1.0 + 2.0 * attempt)
        except Exception as exc:  # noqa: BLE001 - network flakiness
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {symbol} {day.date()}: {last_err}")


def fetch_symbol(symbol: str, start: datetime, end: datetime, pacer: Pacer,
                 workers: int) -> list[tuple]:
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)

    all_rows: list[tuple] = []
    done = 0
    missing = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_day, symbol, d, pacer): d for d in days}
        for fut in futures:
            d = futures[fut]
            rows = fut.result()
            if not rows:
                missing.append(d)
            all_rows.extend(rows)
            done += 1
            if done % 100 == 0:
                print(f"  {symbol}: {done}/{len(days)} days, {len(all_rows):,} rows so far",
                      file=sys.stderr, flush=True)
    if missing:
        print(f"  {symbol}: {len(missing)} day(s) had no published file "
              f"(first: {missing[0].date()}, last: {missing[-1].date()})",
              file=sys.stderr)
    return all_rows


def write_series(rows: list[tuple], out_path: Path) -> int:
    ordered = sorted(rows, key=lambda r: r[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with gzip.open(tmp, "wt") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in ordered:
            f.write(",".join(row) + "\n")
    tmp.replace(out_path)
    return len(ordered)


def default_name(symbol: str) -> str:
    return f"{symbol.lower()}_perp_metrics_5m.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-interval", type=float, default=0.05,
                    help="global seconds between request starts (~20 req/s)")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pacer = Pacer(args.min_interval)
    out_dir = Path(args.out_dir)

    for symbol in args.symbols:
        out_path = out_dir / default_name(symbol)
        if out_path.exists():
            print(f"{symbol}: {out_path} exists, skipping", file=sys.stderr)
            continue
        t0 = time.time()
        print(f"fetching {symbol} {start.date()} -> {end.date()}",
              file=sys.stderr, flush=True)
        rows = fetch_symbol(symbol, start, end, pacer, args.workers)
        if not rows:
            print(f"{symbol}: no data returned, skipping", file=sys.stderr)
            continue
        n = write_series(rows, out_path)
        span = (rows[0][0], rows[-1][0])
        print(f"{symbol}: wrote {n:,} rows ({span[0]} -> {span[1]}) "
              f"to {out_path} in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
