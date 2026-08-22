#!/usr/bin/env python
"""Fetch the daily Crypto Fear & Greed Index from alternative.me's free public
API (R-95 round, both branches).

Modeled on ``scripts/fetch_wikipedia_pageviews.py``: a single unauthenticated
GET (the API returns its whole history in one call via ``limit=0``), CSV-gz
write. No API key required.

The index (alternative.me, proprietary methodology, publicly documented
weights: Volatility 25%, Market Momentum/Volume 25%, Social Media 15%,
Surveys 15%, Bitcoin Dominance 10%, Google Trends 10%) is a daily 0-100
composite crowd-sentiment score, "Extreme Fear" (0) to "Extreme Greed" (100).

Confirmed reachable from this environment immediately before this round
started (200 OK, real data). The API's own history starts 2018-02-01 --
after this project's 2017-01-01 dataset start, so (like DVOL and Binance
futures positioning before it) the 2018-01-17 stress episode is a forced
Step-A fail for coverage, not a measured one; every other episode
(2018-12-15 onward) is covered. Two single-day gaps exist in the raw feed
(2018-04-13->17, a 3-day gap; 2024-10-25->27, post-holdout) -- both
disclosed, neither filled by this script (``align_fear_greed_causal``
handles them via forward-fill exactly like every other daily external
signal in ``tradebot.data``).

Usage::

    python scripts/fetch_fear_greed_index.py \
        --out data/btc_fear_greed_index_daily.csv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENDPOINT = "https://api.alternative.me/fng/"
HEADERS = {
    "User-Agent": "tradebot-research/1.0 (github.com/voho/tradebot; research use)"
}


def fetch_all() -> list[dict]:
    url = f"{ENDPOINT}?limit=0&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())
    rows = payload.get("data", [])
    print(f"  fetched {len(rows)} rows", file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/btc_fear_greed_index_daily.csv.gz")
    args = ap.parse_args()

    print("Fetching full Crypto Fear & Greed Index history from alternative.me",
          file=sys.stderr)
    rows = fetch_all()

    seen = {}
    for r in rows:
        ts = int(r["timestamp"])
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        seen[day] = (r["value"], r.get("value_classification", ""))
    ordered = sorted(seen.items())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,value,classification\n")
        for day, (value, cls) in ordered:
            f.write(f"{day},{value},{cls}\n")

    print(f"Wrote {len(ordered)} rows ({ordered[0][0]} -> {ordered[-1][0]}) -> {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
