#!/usr/bin/env python
"""Fetch daily MVRV ratio (market value / realized value) from CoinMetrics'
free community API (R-74).

Mahmudov & Puell (2018, "Bitcoin Market-Value-to-Realized-Value (MVRV)
Ratio") define realized value as the aggregate cost basis of the coin
supply -- each coin marked at the price it last moved on-chain, not its
current market price -- building on Nic Carter's realized-cap concept
(Carter & Le Calvez, Honeybadger 2018). MVRV = market cap / realized cap
is therefore a genuinely different information channel from every other
INFO-axis signal this project has tried: it requires per-UTXO last-moved
price history, not derivable from this project's own OHLCV file, and it
measures aggregate holder profit/loss (a valuation-based signal) rather
than a flow (stablecoin supply, R-54), a priced volatility expectation
(DVOL/VRP, R-73), a spillover from the rest of the financial system
(VIX/DXY, R-53), or the traded asset's own network activity
(active-address growth, B-07/R-44).

Modeled directly on ``scripts/fetch_stablecoin_supply.py`` (R-54 NOVEL):
same endpoint family (``community-api.coinmetrics.io/v4/timeseries/asset-metrics``,
no API key), same chunked date-range fetch with retry/backoff, same
de-dupe-on-overlap and gzip CSV-write pattern. The only differences are
the metric (``CapMVRVCur``) and that both ``btc`` and ``eth`` are fetched
(ETH is needed for this round's own falsification test).

Confirmed reachable from this environment immediately before this round
started: catalog query shows ``CapMVRVCur`` served for ``btc`` from
2010-07-18 and (checked separately) for ``eth`` from 2015-08-07, both
through the community tier with no API key, well before this project's
2017-01-01 BTC data start and comfortably covering the pre-2020
BTC-control falsification window every other INFO round uses.

Usage::

    python scripts/fetch_coinmetrics_mvrv.py --asset btc --start 2016-01-01 --end 2026-08-21 \
        --out data/btc_mvrv_daily.csv.gz
    python scripts/fetch_coinmetrics_mvrv.py --asset eth --start 2018-01-01 --end 2026-08-21 \
        --out data/eth_mvrv_daily.csv.gz
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
METRIC = "CapMVRVCur"
CHUNK = timedelta(days=300)  # keep each response well under the community page cap


def _fetch_chunk(asset: str, start: datetime, end: datetime, retries: int = 4) -> list[dict]:
    url = (
        f"{ENDPOINT}?assets={asset}&metrics={METRIC}&frequency=1d"
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
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/btc_mvrv_daily.csv.gz")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"Fetching {METRIC} for {args.asset}, {start:%Y-%m-%d} -> {end:%Y-%m-%d}", file=sys.stderr)
    rows = fetch(args.asset, start, end)

    seen = {}
    for r in rows:
        if METRIC in r and r[METRIC] is not None:
            seen[r["time"]] = r  # de-dupe on overlapping chunk boundaries
    ordered = [seen[k] for k in sorted(seen)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,mvrv\n")
        for r in ordered:
            ts = r["time"][:10]  # YYYY-MM-DD
            f.write(f"{ts},{r[METRIC]}\n")

    print(f"Wrote {len(ordered)} rows ({ordered[0]['time'][:10]} -> {ordered[-1]['time'][:10]}) "
          f"-> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
