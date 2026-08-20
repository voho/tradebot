#!/usr/bin/env python
"""Fetch daily aggregate stablecoin circulating supply from CoinMetrics'
free community API (R-54 NOVEL branch, B-21-adjacent but a genuinely new
signal -- see ``experiments/_stablecoin_signal.py`` for the mechanism).

Modeled directly on ``scripts/fetch_onchain_metrics.py`` (B-07/R-44):
same endpoint family, same chunked date-range fetch with retry/backoff,
same de-dupe-on-overlap and CSV-write pattern. The only difference is the
metric (``SplyCur``, current circulating supply) and the assets (``usdt``,
``usdc``) rather than BTC/ETH chain metrics.

Confirmed reachable from this environment immediately before this branch
started: USDT ``SplyCur`` is served back to 2017-01-01 with no gaps in a
spot-check; USDC ``SplyCur`` is served from 2018-08-03 (all-zero/placeholder
until 2018-09-25, when its real minted supply begins) -- USDC's real
history is therefore roughly 1.5 years shorter than USDT's, which is why
this round's signal module uses USDT alone (see ``_stablecoin_signal.py``'s
module docstring for the explicit reasoning, not silently proxied here).

Usage::

    python scripts/fetch_stablecoin_supply.py --asset usdt --start 2017-01-01 --end 2026-08-20 \
        --out data/stablecoin_supply_daily.csv.gz
    python scripts/fetch_stablecoin_supply.py --asset usdc --start 2018-01-01 --end 2026-08-20 \
        --out /tmp/usdc_check.csv.gz   # exploratory only, not committed by this branch
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
METRICS = ["SplyCur"]
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
    ap.add_argument("--asset", default="usdt", choices=["usdt", "usdc"])
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/stablecoin_supply_daily.csv.gz")
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
    col = f"{args.asset}_SplyCur"
    with gzip.open(out_path, "wt") as f:
        f.write(f"timestamp,{col}\n")
        for r in ordered:
            ts = r["time"][:10]  # YYYY-MM-DD
            val = r.get("SplyCur", "")
            f.write(f"{ts},{val}\n")

    print(f"Wrote {len(ordered)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
