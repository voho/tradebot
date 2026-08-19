#!/usr/bin/env python
"""Fetch daily BTC on-chain metrics from CoinMetrics' free community API (B-07).

B-07 ("on-chain features, sign-corrected") has been `BLOCKED (network)`
since it entered the backlog -- every prior connectivity check only
probed exchange/data venues (Deribit, Kraken, Bitstamp, Coinbase), never
an on-chain data provider. A check run alongside this session found
CoinMetrics' community API (no key required) reachable and serving three
metrics back to 2017 on the free tier: active addresses, transaction
count and hash rate -- the first genuinely orthogonal information channel
this project has had (every existing feature is derived from the same
OHLCV price series).

The community tier caps most metrics behind a paid key; ``AdrActCnt``,
``TxCnt`` and ``HashRate`` are the three price-independent series that
came back unrestricted when probed. Nothing else is fetched.

Also used to fetch the ETH series (same three metrics; ETH's chain has had
proof-of-work hash rate data since genesis and retains it as a historical
column post-Merge) for the ETH falsification test the ROUTINE requires,
since a BTC-derived on-chain signal cannot be tested against ETH price
using BTC chain data -- the analogous test needs ETH's own chain data.

Usage::

    python scripts/fetch_onchain_metrics.py --asset btc --start 2017-01-01 --end 2026-08-19 \
        --out data/btc_onchain_daily.csv.gz
    python scripts/fetch_onchain_metrics.py --asset eth --start 2019-01-01 --end 2026-08-19 \
        --out data/eth_onchain_daily.csv.gz
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

ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["AdrActCnt", "TxCnt", "HashRate"]
CHUNK = timedelta(days=300)  # keep each response well under the community page cap


def _fetch_chunk(asset: str, start: datetime, end: datetime, retries: int = 4) -> list[dict]:
    url = (
        f"{ENDPOINT}?assets={asset}&metrics={','.join(METRICS)}&frequency=1d"
        f"&start_time={start:%Y-%m-%d}&end_time={end:%Y-%m-%d}&page_size=10000"
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read())
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload.get("data", [])
        except Exception as exc:  # noqa: BLE001 - network flakiness, just retry
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {start}..{end}: {last_err}")


def fetch(asset: str, start: datetime, end: datetime) -> list[dict]:
    rows: list[dict] = []
    cur = start
    while cur < end:
        nxt = min(cur + CHUNK, end)
        chunk = _fetch_chunk(asset, cur, nxt)
        rows.extend(chunk)
        print(f"  {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}: {len(chunk)} rows", file=sys.stderr)
        cur = nxt
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="btc", choices=["btc", "eth"])
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/btc_onchain_daily.csv.gz")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"Fetching {METRICS} for {args.asset}, {start:%Y-%m-%d} -> {end:%Y-%m-%d}", file=sys.stderr)
    rows = fetch(args.asset, start, end)

    seen = {}
    for r in rows:
        seen[r["time"]] = r  # de-dupe on overlapping chunk boundaries
    ordered = [seen[k] for k in sorted(seen)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp," + ",".join(METRICS) + "\n")
        for r in ordered:
            ts = r["time"][:10]  # YYYY-MM-DD
            vals = [r.get(m, "") for m in METRICS]
            f.write(",".join([ts] + [str(v) for v in vals]) + "\n")

    print(f"Wrote {len(ordered)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
