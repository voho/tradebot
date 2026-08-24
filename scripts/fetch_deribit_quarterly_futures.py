#!/usr/bin/env python
"""Fetch Deribit dated (quarterly) futures 5m OHLCV for the front-quarter
term-structure/roll-yield INFO-axis round (R-120).

This repo has never had a DATED futures series: R-41/R-53's `basis` column
is spot-vs-PERPETUAL (`btcusdt_deribit_perp_5m.csv.gz`), which settles
funding every 8 hours and, per Zhang (2026, SSRN 6185958), is mechanically
mean-reverting by construction. A calendar (quarterly) future instead
resolves only at a fixed expiry, so its basis is a genuinely different
statistical object — the resolvable "roll yield" cash-and-carry traders
actually price (Schmeling, Schrimpf & Todorov 2023/2025, BIS WP 1087; Chi
et al. 2023, J. Futures Markets, basis/momentum/basis-momentum factors).

Deribit's quarterly contracts follow a fixed, public naming convention
(``{BTC,ETH}-{DD}{MMM}{YY}``, expiry = last Friday of Mar/Jun/Sep/Dec,
08:00 UTC) and a fixed, deterministic listing rule verified empirically
before writing this script (not guessed): BTC-29MAR19 has real chart data
from >= 2018-10-01 and none before 2018-06-01 (~6-9 months of listed
life); the identical get_tradingview_chart_data endpoint
scripts/fetch_deribit_perp_price.py already validated for the perpetual
also serves EXPIRED dated instruments, so no new endpoint or auth is
needed. `get_instruments(expired=true)` was tried first and returns only
the single most-recently-expired contract (a real API limitation, not a
bug in this script) — so contract names are generated from the fixed
expiry-date rule instead of discovered via the API.

Each contract is fetched over its own [expiry - LOOKBACK_DAYS, expiry]
window; `no_data` responses before the contract's real listing date are
expected and handled the same way the perpetual fetcher already does
(skip, not an error, as long as *some* data is eventually seen). Output
is the RAW per-contract, per-bar series (columns: timestamp, instrument,
expiry, open, high, low, close, volume) — front-quarter SELECTION
(picking, at each bar, whichever listed contract expires soonest) is left
to the consuming code, not baked in here, so a later round can pick a
different roll rule (e.g. "next quarter" instead of "front quarter")
without re-fetching.

Usage::

    python scripts/fetch_deribit_quarterly_futures.py --asset BTC \\
        --first-expiry 2019-03-29 --last-expiry 2023-03-31 \\
        --out data/btcusd_deribit_quarterly_5m.csv.gz
    python scripts/fetch_deribit_quarterly_futures.py --asset ETH \\
        --first-expiry 2020-09-25 --last-expiry 2023-03-31 \\
        --out data/ethusd_deribit_quarterly_5m.csv.gz
"""

from __future__ import annotations

import argparse
import calendar
import gzip
import json
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ENDPOINT = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
RESOLUTION = "5"
CHUNK = timedelta(days=10)
LOOKBACK_DAYS = 270  # generous upper bound on a quarterly's listed lifetime


def _last_friday(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 4:
        d = d.replace(day=d.day - 1)
    return d


def quarterly_names(asset: str, first_expiry: date, last_expiry: date) -> list[tuple[str, date]]:
    """(instrument_name, expiry_date) for every quarterly expiry in range."""
    out = []
    year = first_expiry.year
    while True:
        for month in (3, 6, 9, 12):
            d = _last_friday(year, month)
            if d < first_expiry:
                continue
            if d > last_expiry:
                return out
            name = f"{asset}-{d.day:02d}{d.strftime('%b').upper()}{str(d.year)[2:]}"
            out.append((name, d))
        year += 1


def _fetch_chunk(instrument: str, start_ms: int, end_ms: int, retries: int = 5) -> dict:
    url = (f"{ENDPOINT}?instrument_name={instrument}&start_timestamp={start_ms}"
           f"&end_timestamp={end_ms}&resolution={RESOLUTION}")
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=25) as resp:
                payload = json.loads(resp.read())
            return payload["result"]
        except Exception as exc:  # noqa: BLE001 - network flakiness, just retry
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {instrument} {start_ms}-{end_ms}: {last_err}")


def fetch_contract(instrument: str, expiry: date) -> list[tuple[datetime, str, date, float, float, float, float, float]]:
    start = datetime.combine(expiry, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    end = datetime.combine(expiry, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=8)
    rows = []
    cursor = start
    any_data = False
    while cursor < end:
        chunk_end = min(cursor + CHUNK, end)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)
        result = _fetch_chunk(instrument, start_ms, end_ms)
        status = result.get("status")
        ticks = result.get("ticks", [])
        if status == "ok" and ticks:
            any_data = True
            for t, o, h, l, c, v in zip(ticks, result["open"], result["high"],
                                        result["low"], result["close"], result["volume"]):
                rows.append((datetime.fromtimestamp(t / 1000, tz=timezone.utc),
                             instrument, expiry, o, h, l, c, v))
        cursor = chunk_end
    print(f"  {instrument} (expiry {expiry}): {len(rows)} bars"
          + ("" if any_data else "  [NO DATA -- not listed in this window]"),
          file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--first-expiry", required=True)
    ap.add_argument("--last-expiry", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    first_expiry = datetime.strptime(args.first_expiry, "%Y-%m-%d").date()
    last_expiry = datetime.strptime(args.last_expiry, "%Y-%m-%d").date()

    contracts = quarterly_names(args.asset, first_expiry, last_expiry)
    print(f"fetching {len(contracts)} {args.asset} quarterly contracts, "
          f"expiries {contracts[0][1]} -> {contracts[-1][1]}", file=sys.stderr)

    all_rows: list[tuple] = []
    for name, expiry in contracts:
        all_rows.extend(fetch_contract(name, expiry))

    if not all_rows:
        raise SystemExit("no data fetched for any contract")

    all_rows.sort(key=lambda r: (r[1], r[0]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("timestamp,instrument,expiry,open,high,low,close,volume\n")
        for ts, inst, exp, o, h, l, c, v in all_rows:
            f.write(f"{int(ts.timestamp() * 1000)},{inst},{exp},{o},{h},{l},{c},{v}\n")

    n_contracts_with_data = len({r[1] for r in all_rows})
    print(f"wrote {len(all_rows):,} bars across {n_contracts_with_data}/{len(contracts)} "
          f"contracts with data to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
