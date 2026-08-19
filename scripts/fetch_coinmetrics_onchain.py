#!/usr/bin/env python
"""Fetch real on-chain valuation metrics from CoinMetrics' free Community API (B-07).

This repository has never used a genuinely non-price data channel: the four
INFO-tagged strategies in the ledger (L-12, L-14, L-15, L-16) all tried to
recover missing information FROM the price series itself (Bayesian regime
posteriors, Bulk Volume Classification) and failed for exactly that reason.
CoinMetrics' community endpoint (free, no key required) exposes
``CapMVRVCur`` -- the MVRV ratio (market cap / realized cap, where realized
cap prices every coin at the block it last moved on-chain) -- computed from
actual blockchain transaction data, not from trade price. It is a genuinely
independent measurement: two assets can trade at the same price with very
different realized cost bases depending on how much of the supply has
recently changed hands.

Literature: Mahmudov & Puell (2018) introduced MVRV; Grobys (2026,
International Review of Financial Analysis, "Using on-chain data to predict
Bitcoin cycles") backtested rule-based MVRV Z-score / NUPL / CVDD strategies
over three complete cycles (2013-12-07 to 2025-04-12, spanning 2015, 2018 and
2022) and found the MVRV Z-score the strongest of the three on risk-adjusted
terms against buy-and-hold and a Monte Carlo random-entry null.

Coverage: BTC from 2010-07-18, ETH from 2015-08-08, both through the current
day -- confirmed by the catalog endpoint, not assumed.

Usage::

    python scripts/fetch_coinmetrics_onchain.py --asset btc \\
        --out data/btcusd_onchain_daily.csv.gz
    python scripts/fetch_coinmetrics_onchain.py --asset eth \\
        --out data/ethusd_onchain_daily.csv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["CapMVRVCur", "AdrActCnt", "SplyCur"]


def _fetch(asset: str, start: str, end: str, retries: int = 5) -> list[dict]:
    url = (f"{ENDPOINT}?assets={asset}&metrics={','.join(METRICS)}"
           f"&start_time={start}&end_time={end}&frequency=1d&page_size=10000")
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read())
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload.get("data", [])
        except Exception as exc:  # noqa: BLE001 - network flakiness, just retry
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {asset}: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["btc", "eth"])
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-08-19")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"fetching CoinMetrics community metrics for {args.asset}: {METRICS}",
          file=sys.stderr)
    rows = _fetch(args.asset, args.start, args.end)
    if not rows:
        raise SystemExit(f"no data returned for {args.asset}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,mvrv,active_addresses,supply\n")
        for r in rows:
            ts = r["time"]
            mvrv = r.get("CapMVRVCur", "")
            adr = r.get("AdrActCnt", "")
            sply = r.get("SplyCur", "")
            f.write(f"{ts},{mvrv},{adr},{sply}\n")

    print(f"wrote {len(rows)} daily rows ({rows[0]['time']} -> {rows[-1]['time']}) "
          f"to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
