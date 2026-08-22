#!/usr/bin/env python
"""Fetch daily English Wikipedia pageviews for the "Bitcoin" article from
the Wikimedia Foundation's free public REST API (R-93 round, both branches).

Modeled on ``scripts/fetch_stablecoin_supply.py``/``fetch_onchain_metrics.py``:
a chunked date-range fetch with retry/backoff, de-dupe-on-overlap, CSV-gz
write. No API key required, no rate-limit tier -- the Pageviews API is a
fully public, unauthenticated endpoint.

Confirmed reachable from this environment immediately before this round
started (spot-checked 2017-01-01 -> 2017-01-10, 200 OK, real nonzero
daily view counts). The API's own documented data start is 2015-07-01,
which is before this project's 2017-01-01 dataset start, so -- unlike
DVOL (R-73, coverage from Deribit's launch), MVRV/on-chain metrics, or
Binance futures positioning (R-81, BTC from 2020-09-01 / ETH from
2021-12-01) -- this signal has NO external coverage-start caveat and can
use the project's full six-episode table (R-84's own set, the first
INFO-axis signal after raw volume to have that property).

Usage::

    python scripts/fetch_wikipedia_pageviews.py --article Bitcoin \
        --start 2015-07-01 --end 2026-08-20 \
        --out data/btc_wikipedia_pageviews_daily.csv.gz
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

ENDPOINT = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
# Wikimedia's etiquette policy asks for a descriptive User-Agent identifying
# the project and a contact means; this is a research tool, not a bot.
HEADERS = {
    "User-Agent": "tradebot-research/1.0 (github.com/voho/tradebot; research use)"
}
CHUNK = timedelta(days=3650)  # the API has no page cap on a per-article daily range


def _fetch_chunk(article: str, start: datetime, end: datetime, retries: int = 4) -> list[dict]:
    url = (
        f"{ENDPOINT}/en.wikipedia/all-access/all-agents/{article}/daily/"
        f"{start:%Y%m%d}/{end:%Y%m%d}"
    )
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
            return payload.get("items", [])
        except urllib.error.HTTPError as exc:  # noqa: BLE001
            if exc.code == 404:
                return []  # no data in this window (e.g. before 2015-07-01)
            last_err = exc
            time.sleep(2 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - network flakiness, just retry
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {start}..{end}: {last_err}")


def fetch(article: str, start: datetime, end: datetime) -> list[dict]:
    rows: list[dict] = []
    cur = start
    while cur < end:
        nxt = min(cur + CHUNK, end)
        chunk = _fetch_chunk(article, cur, nxt)
        rows.extend(chunk)
        print(f"  {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}: {len(chunk)} rows", file=sys.stderr)
        cur = nxt
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", default="Bitcoin")
    ap.add_argument("--start", default="2015-07-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/btc_wikipedia_pageviews_daily.csv.gz")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"Fetching '{args.article}' pageviews, {start:%Y-%m-%d} -> {end:%Y-%m-%d}", file=sys.stderr)
    rows = fetch(args.article, start, end)

    seen = {}
    for r in rows:
        seen[r["timestamp"]] = r  # de-dupe on overlapping chunk boundaries
    ordered = [seen[k] for k in sorted(seen)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,views\n")
        for r in ordered:
            ts = r["timestamp"][:8]  # YYYYMMDD
            ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
            f.write(f"{ts},{r['views']}\n")

    print(f"Wrote {len(ordered)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
