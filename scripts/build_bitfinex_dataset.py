#!/usr/bin/env python
"""Build ETH/USD and BTC/USD 5m datasets from the Bitfinex minute-data repo.

Source: https://github.com/Zombie-3000/Bitfinex-historical-data — real
1-minute candles, 2016-2019, for several pairs on one venue.

This exists for the **cross-asset falsification test**. Every conclusion
in this repo rests on BTC 2017-2026, which contains roughly three
independent regime events; a filter fitted to those would look identical
to one that works. Running the same strategy on ETH over the same window
and the same venue holds period and venue constant and varies only the
asset, so BTC here doubles as the control.

Note the column order: the Bitfinex API returns
``[timestamp, open, CLOSE, HIGH, LOW, volume]`` — close before high/low,
which is not the usual OHLC order and silently corrupts the data if
assumed.

Usage::

    python scripts/build_bitfinex_dataset.py --source /tmp/bfx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.data import save_ohlcv_csv  # noqa: E402

COLUMNS = ["timestamp", "open", "close", "high", "low", "volume"]
YEARS = (2016, 2017, 2018, 2019)


def build(source: Path, symbol: str, out: Path) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = source / f"{symbol}_{year}.csv"
        if not path.exists():
            print(f"note: {path} missing, skipping", file=sys.stderr)
            continue
        frames.append(pd.read_csv(path, header=None, names=COLUMNS))
    if not frames:
        raise SystemExit(f"no source files for {symbol} under {source}")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (df.drop_duplicates("ts").set_index("ts").sort_index()
            [["open", "high", "low", "close", "volume"]])

    bars = df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"})
    bars = bars.dropna(subset=["open", "high", "low", "close"])

    # A venue outage leaves gaps; forward-filling prices would invent
    # candles, so gaps are simply absent and the engine tolerates them.
    bad = (bars["high"] < bars[["open", "close"]].max(axis=1)) | \
          (bars["low"] > bars[["open", "close"]].min(axis=1))
    if bad.any():
        raise SystemExit(f"{symbol}: {bad.sum()} bars violate high/low bounds - "
                         "check the source column order")

    save_ohlcv_csv(bars, out)
    print(f"{symbol}: {len(bars):,} 5m bars  {bars.index[0]:%Y-%m-%d} -> "
          f"{bars.index[-1]:%Y-%m-%d}  -> {out}", file=sys.stderr)
    return bars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True,
                    help="directory holding <SYMBOL>_<YEAR>.csv files")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()
    for symbol, name in (("ETHUSD", "ethusd_bitfinex_5m.csv.gz"),
                         ("BTCUSD", "btcusd_bitfinex_5m.csv.gz")):
        build(args.source, symbol, args.out_dir / name)


if __name__ == "__main__":
    main()
