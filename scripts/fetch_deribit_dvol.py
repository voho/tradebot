#!/usr/bin/env python
"""Fetch BTC DVOL (Deribit's implied-volatility index) daily OHLC (R-73 conservative).

Deribit's public ``get_volatility_index_data`` endpoint serves a genuine
historical implied-volatility index computed from the live BTC options
order book -- the first forward-looking (as opposed to trailing-realized)
volatility series available to this project. Confirmed reachable from this
environment; confirmed the true history starts **2021-03-24** (a single
day earlier than the "~2021-04" figure quoted in this round's brief --
verified directly by paging backward past the first non-empty response
until an empty one comes back, not by trusting a resolution-86400
row-count guess). There is no DVOL data before that date: Deribit's BTC
options market did not carry a computed index further back, and this
script does not synthesize, backfill, or interpolate one.

Modeled on ``scripts/fetch_deribit_funding.py``'s retry/chunking
conventions: fixed-size forward date chunks (comfortably under the
endpoint's observed ~1000-row-per-call cap at daily resolution), a
bounded retry-with-backoff loop per chunk, de-dupe on any overlapping
timestamp, gzip CSV output.

The endpoint's ``continuation`` field pages *backward* (it returns the
most recent ``rows returned`` = up to 1000 candles inside the requested
window, then offers an earlier ``end_timestamp`` to continue toward the
window's start) -- inspected directly against this environment's live
response before writing this script. Rather than lean on that
undocumented behaviour, this script instead requests small enough forward
chunks (350 days, well under the 1000-row cap) that no response is ever
truncated, and asserts that at every chunk boundary.

Also supports ``--currency ETH`` (Deribit publishes an independent ETH
DVOL index, confirmed reachable, same 2021-03-24 start) so the VRP
falsification round (``experiments/_vrp_signal.py``) can test a genuine
asset-specific variance risk premium on ETH rather than proxying it
through BTC's own implied vol.

Usage::

    python scripts/fetch_deribit_dvol.py --start 2021-01-01 --end 2026-08-21 \
        --out data/btc_dvol_daily.csv.gz
    python scripts/fetch_deribit_dvol.py --currency ETH --start 2021-01-01 --end 2026-08-21 \
        --out data/eth_dvol_daily.csv.gz
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

ENDPOINT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
CHUNK = timedelta(days=350)  # well under the observed ~1000-row/call cap at resolution=86400
RESOLUTION = 86400  # 1 day, in seconds
MAX_ROWS_PER_CALL = 1000  # observed cap; used only for the truncation assertion below


def _fetch_chunk(currency: str, start_ms: int, end_ms: int, retries: int = 5) -> list[list[float]]:
    url = (f"{ENDPOINT}?currency={currency}&start_timestamp={start_ms}"
           f"&end_timestamp={end_ms}&resolution={RESOLUTION}")
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
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {start_ms}-{end_ms}: {last_err}")


def fetch_daily(currency: str, start: datetime, end: datetime) -> list[tuple[datetime, float, float, float, float]]:
    """Forward-chunked fetch. Each row is (day, open, high, low, close)."""
    rows: dict[datetime, tuple[float, float, float, float]] = {}
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + CHUNK, end)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)
        batch = _fetch_chunk(currency, start_ms, end_ms)
        assert len(batch) < MAX_ROWS_PER_CALL, (
            f"chunk {cursor.date()}->{chunk_end.date()} returned {len(batch)} rows, "
            f">= the observed {MAX_ROWS_PER_CALL}-row cap -- it may have been silently "
            "truncated; shrink CHUNK and re-run")
        n_new = 0
        for ts_ms, o, h, low, c in batch:
            day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if day not in rows:
                n_new += 1
            rows[day] = (float(o), float(h), float(low), float(c))
        print(f"  fetched {cursor.date()} -> {chunk_end.date()}: "
              f"{len(batch)} rows ({n_new} new)", file=sys.stderr)
        cursor = chunk_end
    return [(day, *ohlc) for day, ohlc in sorted(rows.items())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/btc_dvol_daily.csv.gz")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    print(f"fetching Deribit {args.currency} DVOL daily {start.date()} -> {end.date()}",
          file=sys.stderr)
    rows = fetch_daily(args.currency, start, end)
    if not rows:
        raise SystemExit("no data returned")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,open,high,low,close\n")
        for day, o, h, low, c in rows:
            f.write(f"{day.date().isoformat()},{o:.4f},{h:.4f},{low:.4f},{c:.4f}\n")

    print(f"wrote {len(rows)} daily rows ({rows[0][0].date()} -> {rows[-1][0].date()}) "
          f"to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
