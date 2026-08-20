#!/usr/bin/env python
"""Fetch daily macro/cross-asset series from FRED's free public CSV endpoint (R-53).

The ledger's INFO constraint has so far only been attacked with data
*about* the traded asset itself (on-chain metrics, B-07) or derived from
its own price (basis, realized vol). This fetches three series describing
the *rest of the financial system* instead -- the channel the VIX-Bitcoin
spillover literature (Luo, Tsai & Yen 2024/2025, SSRN; IMF WP 2023/213)
and the DXY inverse-correlation literature (multiple 2024-2026 studies)
argue is where crypto risk-off pressure originates before it shows up in
BTC's own price:

- ``SP500``   -- S&P 500 close (equity risk appetite)
- ``VIXCLS``  -- CBOE VIX close (equity-market fear gauge)
- ``DTWEXBGS``-- Fed Trade-Weighted Broad Dollar Index (risk-off dollar strength)

No API key required. FRED's ``SP500`` series only carries a trailing
~10-year window (a FRED platform limit, not a data gap); ``VIXCLS`` and
``DTWEXBGS`` go back decades. All three are published once their
reference day has closed, so the loader in ``tradebot.data`` shifts them
one full day forward before ever handing a bar a value -- the same
convention ``align_onchain_causal`` uses for CoinMetrics data.

Usage::

    python scripts/fetch_macro_data.py --start 2016-06-01 --end 2026-08-20
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import urllib.request
from pathlib import Path

SERIES = {
    "SP500": "spx_daily.csv.gz",
    "VIXCLS": "vix_daily.csv.gz",
    "DTWEXBGS": "dxy_daily.csv.gz",
}
ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_series(series_id: str, start: str, end: str) -> list[tuple[str, str]]:
    url = f"{ENDPOINT}?id={series_id}&cosd={start}&coed={end}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    reader = csv.reader(text.splitlines())
    header = next(reader)
    assert header == ["observation_date", series_id], f"unexpected header: {header}"
    for date, value in reader:
        if value == ".":  # FRED's missing-observation marker
            continue
        rows.append((date, value))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--end", default="2026-08-20")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for series_id, filename in SERIES.items():
        print(f"Fetching {series_id} {args.start} -> {args.end}", file=sys.stderr)
        rows = fetch_series(series_id, args.start, args.end)
        out_path = out_dir / filename
        with gzip.open(out_path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", series_id.lower()])
            writer.writerows(rows)
        print(f"  wrote {len(rows)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
