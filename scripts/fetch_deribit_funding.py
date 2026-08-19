#!/usr/bin/env python
"""Fetch BTC-PERPETUAL funding history from Deribit's public API (R-39 / B-02).

The committed funding series (`data/btcusdt_perp_funding_8h.csv.gz`) is
real Binance BTCUSDT funding but stops in 2023 — Binance has been
unreachable from every session since (HTTP 451 at the proxy). A
connectivity check run alongside R-38 found Deribit's public API
reachable where Binance is not, which is what this script exploits.

Deribit charges funding **continuously** (an `interest_1h` rate applied
every hour) rather than Binance's discrete 8-hourly settlement, so there
is no literal "funding_rate" column to copy. This script sums
`interest_1h` over each UTC-aligned 8-hour bucket [00:00, 08:00, 16:00)
to produce a rate **comparable in magnitude and units** to a per-8h
settlement, and writes it on the bucket's close timestamp. That is a
modelling choice, not a fact about Deribit's mechanism — it is documented
here and in docs/VALIDATION.md rather than silently blended into the
Binance file.

Usage::

    python scripts/fetch_deribit_funding.py --start 2020-01-01 --end 2026-08-19 \
        --out data/btcusdt_deribit_perp_funding_8h.csv.gz
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

ENDPOINT = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
CHUNK = timedelta(days=30)          # API caps a single call near 744 hourly rows
INSTRUMENT = "BTC-PERPETUAL"


def _fetch_chunk(start_ms: int, end_ms: int, retries: int = 4) -> list[dict]:
    url = f"{ENDPOINT}?instrument_name={INSTRUMENT}&start_timestamp={start_ms}&end_timestamp={end_ms}"
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read())
            return payload.get("result", [])
        except Exception as exc:  # noqa: BLE001 - network flakiness, just retry
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {start_ms}-{end_ms}: {last_err}")


def fetch_hourly(start: datetime, end: datetime) -> list[dict]:
    rows: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + CHUNK, end)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)
        batch = _fetch_chunk(start_ms, end_ms)
        rows.extend(batch)
        print(f"  fetched {cursor.date()} -> {chunk_end.date()}: {len(batch)} rows",
              file=sys.stderr)
        cursor = chunk_end
    return rows


def bucket_to_8h(rows: list[dict]) -> list[tuple[datetime, float]]:
    """Sum interest_1h into UTC-aligned [00:00,08:00,16:00) buckets."""
    buckets: dict[datetime, float] = {}
    for row in rows:
        ts = datetime.fromtimestamp(row["timestamp"] / 1000, tz=timezone.utc)
        bucket_hour = (ts.hour // 8) * 8
        bucket_start = ts.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
        buckets.setdefault(bucket_start, 0.0)
        buckets[bucket_start] += float(row["interest_1h"])
    out = [(start + timedelta(hours=8), rate) for start, rate in buckets.items()]
    out.sort(key=lambda pair: pair[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-08-19")
    ap.add_argument("--out", default="data/btcusdt_deribit_perp_funding_8h.csv.gz")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"fetching Deribit {INSTRUMENT} hourly funding {start.date()} -> {end.date()}",
          file=sys.stderr)
    hourly = fetch_hourly(start, end)
    if not hourly:
        raise SystemExit("no data returned")
    bucketed = bucket_to_8h(hourly)
    # drop the first/last bucket if partial (fewer than 8 hourly obs summed)
    counts: dict[datetime, int] = {}
    for row in hourly:
        ts = datetime.fromtimestamp(row["timestamp"] / 1000, tz=timezone.utc)
        bucket_hour = (ts.hour // 8) * 8
        bucket_start = ts.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
        counts[bucket_start] = counts.get(bucket_start, 0) + 1
    complete = [(ts, rate) for ts, rate in bucketed
                if counts.get(ts - timedelta(hours=8), 0) >= 8]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,funding_rate\n")
        for ts, rate in complete:
            f.write(f"{ts.isoformat()},{rate:.8f}\n")

    print(f"wrote {len(complete)} 8h buckets ({complete[0][0]} -> {complete[-1][0]}) "
          f"to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
