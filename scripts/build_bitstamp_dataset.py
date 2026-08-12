#!/usr/bin/env python
"""Rebuild data/btcusd_spot_5m.csv.gz from the Bitstamp minute-data repo.

The source is https://github.com/ff137/bitstamp-btcusd-minute-data — real
BTC/USD 1-minute OHLC from Bitstamp, bulk history since 2012 plus daily
updates. This script resamples it to 5m and writes the canonical committed
dataset. Re-run it whenever you want to refresh the data:

    git clone --depth 1 https://github.com/ff137/bitstamp-btcusd-minute-data /tmp/bitstamp
    python scripts/build_bitstamp_dataset.py --source /tmp/bitstamp

The start date defaults to 2017-01-01 so the span covers the 2017 bull,
2018 bear, 2020 crash and bull, 2021 top, 2022 bear, and the 2023+ cycle
while keeping the file a reasonable size for git.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.data import save_ohlcv_csv  # noqa: E402

BULK = "data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
UPDATES = "data/updates/btcusd_bitstamp_1min_latest.csv"


def build(source: Path, out: Path, start: str) -> pd.DataFrame:
    frames = []
    for rel in (BULK, UPDATES):
        path = source / rel
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"note: {path} not found, skipping", file=sys.stderr)
    if not frames:
        raise SystemExit(f"no source files found under {source}")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = (df.drop_duplicates("ts").set_index("ts").sort_index()
          [["open", "high", "low", "close", "volume"]].loc[start:])

    bars = df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    bars.index.name = "timestamp"

    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars[["open", "high", "low", "close"]] > 0).all().all()
    assert bars.index.is_unique and bars.index.is_monotonic_increasing

    save_ohlcv_csv(bars, out)
    print(f"wrote {len(bars):,} bars ({bars.index[0]:%Y-%m-%d} -> "
          f"{bars.index[-1]:%Y-%m-%d}) to {out}", file=sys.stderr)
    return bars


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True,
                    help="clone of ff137/bitstamp-btcusd-minute-data")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "btcusd_spot_5m.csv.gz")
    ap.add_argument("--start", default="2017-01-01")
    args = ap.parse_args()
    build(args.source, args.out, args.start)
