#!/usr/bin/env python
"""Fetch BTC/USD 5m OHLC directly from Bitstamp's public API for dates
before this project's committed dataset starts (2017-01-01).

``scripts/build_bitstamp_dataset.py`` sources bulk history from a GitHub
mirror (github.com/ff137/bitstamp-btcusd-minute-data), which is blocked
under this project's network policy in some sessions. Bitstamp's own
``/api/v2/ohlc/`` REST endpoint is reachable directly and serves native
5-minute bars (``step=300``) back to 2013, so this script bypasses the
mirror entirely for the pre-2017 extension (R-143).

Usage::

    python scripts/fetch_bitstamp_early.py --start 2013-01-01 --end 2017-01-01 \\
        --out data/btcusd_spot_5m_pre2017.csv.gz

Writes the same schema as the canonical file
(``timestamp,open,high,low,close,volume``, ms epoch) so it can be loaded
with :func:`tradebot.data.load_ohlcv_csv` or concatenated directly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.data import save_ohlcv_csv  # noqa: E402

URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
STEP = 300  # 5 minutes, seconds
LIMIT = 1000  # max bars per call


def fetch_range(start_ts: int, end_ts: int) -> pd.DataFrame:
    rows = []
    cursor = start_ts
    call = 0
    while cursor < end_ts:
        call += 1
        for attempt in range(6):
            try:
                resp = requests.get(
                    URL, params={"step": STEP, "limit": LIMIT, "start": cursor}, timeout=30
                )
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                if attempt == 5:
                    raise
                wait = 2 ** attempt
                print(f"  call {call} failed ({exc}), retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        ohlc = resp.json()["data"]["ohlc"]
        if not ohlc:
            break
        rows.extend(ohlc)
        last_ts = int(ohlc[-1]["timestamp"])
        if last_ts <= cursor:
            break  # no progress; avoid an infinite loop
        cursor = last_ts + STEP
        if call % 20 == 0:
            print(f"  call {call}: at {pd.Timestamp(cursor, unit='s', tz='UTC')}",
                  file=sys.stderr)
        time.sleep(0.15)  # be polite to the public endpoint

    df = pd.DataFrame(rows)
    df = df.astype({"timestamp": "int64", "open": float, "high": float,
                     "low": float, "close": float, "volume": float})
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    lo = pd.Timestamp(start_ts, unit="s", tz="UTC")
    hi = pd.Timestamp(end_ts, unit="s", tz="UTC")
    df = df[(df.index >= lo) & (df.index < hi)]
    return df[["open", "high", "low", "close", "volume"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2017-01-01")
    ap.add_argument("--out", type=Path,
                     default=Path(__file__).resolve().parents[1] / "data"
                     / "btcusd_spot_5m_pre2017.csv.gz")
    args = ap.parse_args()

    start_ts = int(pd.Timestamp(args.start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(args.end, tz="UTC").timestamp())

    print(f"fetching {args.start} -> {args.end} from Bitstamp...", file=sys.stderr)
    bars = fetch_range(start_ts, end_ts)
    bars.index.name = "timestamp"

    # Sanity checks mirroring build_bitstamp_dataset.py
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars[["open", "high", "low", "close"]] > 0).all().all()
    assert bars.index.is_unique and bars.index.is_monotonic_increasing

    expected_bars = (end_ts - start_ts) // STEP
    coverage = len(bars) / expected_bars
    print(f"wrote {len(bars):,} bars ({bars.index[0]} -> {bars.index[-1]}), "
          f"coverage={coverage:.1%} of {expected_bars:,} expected slots",
          file=sys.stderr)

    # Per-year coverage, so a thin early year is visible rather than averaged away.
    by_year = bars.groupby(bars.index.year).size()
    for yr, n in by_year.items():
        yr_start = pd.Timestamp(f"{yr}-01-01", tz="UTC")
        yr_end = min(pd.Timestamp(f"{yr + 1}-01-01", tz="UTC"), pd.Timestamp(args.end, tz="UTC"))
        yr_expected = max(int((yr_end - max(yr_start, pd.Timestamp(args.start, tz="UTC"))).total_seconds() // STEP), 1)
        print(f"  {yr}: {n:,} bars, {n / yr_expected:.1%} coverage", file=sys.stderr)

    save_ohlcv_csv(bars, args.out)


if __name__ == "__main__":
    main()
