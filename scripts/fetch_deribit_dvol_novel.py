#!/usr/bin/env python
"""Fetch daily BTC DVOL (Deribit's official 30-day implied-volatility index)
from Deribit's public API (R-73 NOVEL branch).

The ledger's C-table lists "Options / volatility risk premium" as ruled out
*only* because "no options data, no way to validate here" -- that reason no
longer holds: Deribit serves a genuine historical IV benchmark
(``get_volatility_index_data``), confirmed reachable from this environment.

DVOL is a forward-looking, PRICED market expectation (option writers'
30-day-ahead volatility view), structurally different from every INFO-axis
signal tried before it in this project (on-chain activity/B-07/R-44, VIX/DXY
macro/R-53, stablecoin supply/R-54/R-55/R-58) -- all of which were either
price-derived or a spot/balance-sheet flow proxy, not a market-priced
expectation.

Hard limitation, stated up front and not silently proxied around: DVOL's
history starts ~2021-03-24. There is no way to get BTC options-implied vol
before options markets existed at scale -- this is *why* the row was ruled
out before, and it still constrains coverage to about 2021-04 onward, far
short of this project's usual 2017-> history.

Usage::

    python scripts/fetch_deribit_dvol_novel.py --start 2021-03-24 --end 2026-08-21 \
        --out data/btc_dvol_daily.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ENDPOINT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
CHUNK = timedelta(days=300)  # keep each response well under any page cap


def _fetch_chunk(currency: str, start_ms: int, end_ms: int, retries: int = 4) -> list[list]:
    url = (
        f"{ENDPOINT}?currency={currency}&start_timestamp={start_ms}"
        f"&end_timestamp={end_ms}&resolution=86400"
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read())
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload["result"]["data"]
        except Exception as exc:  # noqa: BLE001 - network flakiness, just retry
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {start_ms}..{end_ms}: {last_err}")


def fetch(currency: str, start: datetime, end: datetime) -> list[list]:
    rows: list[list] = []
    cur = start
    while cur < end:
        nxt = min(cur + CHUNK, end)
        start_ms = int(cur.timestamp() * 1000)
        end_ms = int(nxt.timestamp() * 1000)
        chunk = _fetch_chunk(currency, start_ms, end_ms)
        rows.extend(chunk)
        print(f"  {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}: {len(chunk)} rows", file=sys.stderr)
        cur = nxt
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--start", default="2021-03-24")  # DVOL's real history start
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/btc_dvol_daily.csv.gz")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"{out_path} already exists -- not overwriting (shared file, "
              f"sibling branch may have created it). Delete it explicitly "
              f"if you want to refetch.", file=sys.stderr)
        return

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    print(f"Fetching {args.currency} DVOL, {start:%Y-%m-%d} -> {end:%Y-%m-%d}", file=sys.stderr)
    rows = fetch(args.currency, start, end)

    # de-dupe on overlapping chunk boundaries, keep [ts, open, high, low, close]
    seen = {}
    for r in rows:
        seen[r[0]] = r
    ordered = [seen[k] for k in sorted(seen)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "dvol_open", "dvol_high", "dvol_low", "dvol_close"])
        for ts_ms, o, h, lo, c in ordered:
            date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            writer.writerow([date, o, h, lo, c])
    print(f"  wrote {len(ordered)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
