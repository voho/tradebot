#!/usr/bin/env python
"""Fetch daily exchange net-flow metrics from CoinMetrics' free community API (R-81).

An eighth structurally distinct INFO signal: CoinMetrics' ``FlowInExNtv``
and ``FlowOutExNtv`` (native-unit BTC/ETH flowing onto and off known
exchange addresses) probe as unrestricted on the free community tier --
found while checking whether any capital-custody / selling-pressure signal
remained untested, alongside B-07/R-44's on-chain activity metrics
(``AdrActCnt``, ``TxCnt``, ``HashRate``), which are a network-activity
signal, not a flow one. Net exchange flow is a distinct economic
mechanism: coins moving onto exchanges are read as increased latent
selling pressure, coins moving off as accumulation/reduced float (Fischer
2019; the mechanism behind CryptoQuant's/Glassnode's netflow products, and
studied academically in Ren, Wu, Liu 2024 (arXiv:2411.06327), which finds
BTC's own net inflow lacks return-forecasting power at intraday horizons
but negatively forecasts volatility -- a caveat worth carrying into this
round's design rather than discovering it mid-round).

Usage::

    python scripts/fetch_exchange_flows.py --asset btc --start 2016-01-01 --end 2026-08-19 \
        --out data/btc_exchange_flow_daily.csv.gz
    python scripts/fetch_exchange_flows.py --asset eth --start 2019-01-01 --end 2026-08-19 \
        --out data/eth_exchange_flow_daily.csv.gz
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
METRICS = ["FlowInExNtv", "FlowOutExNtv"]
CHUNK = timedelta(days=300)


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
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/btc_exchange_flow_daily.csv.gz")
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
