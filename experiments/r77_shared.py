"""R-77 shared infra: Bitstamp-vs-Coinbase BTC cross-venue divergence.

Frozen before either branch runs (same discipline as ``experiments/r70_shared.py``
and R-76's ``load_leg``): both branches import this module rather than each
writing their own loader, so the causal-join code is written and reviewed
once, not twice.

**The idea.** Every INFO-axis round before this one (R-44 on-chain, R-53
macro, R-54/R-55/R-58 stablecoin, R-73 DVOL, R-74 MVRV, R-75 calendar, R-76
pairs trading) fed the strategy either an *external* data source about a
different quantity, or a relationship between two *different instruments*
on the *same* venue. None has used a second, independent read of the SAME
instrument's price from a different venue. This project's tradable BTC
series (``data/btcusd_spot_5m.csv.gz``, what every registered strategy
actually trades) is Bitstamp-sourced. Coinbase Exchange quotes the same
instrument independently. Persistent cross-venue price deviations in
crypto are real and larger than transaction costs can explain (Makarov &
Schoar 2020, JFE 135(2):293-319); which venue leads price discovery is a
studied, nonzero question (Hasbrouck 1995, J. Finance 50(4); Gonzalo &
Granger 1995, J. Business & Econ. Stat. 13(1); Putninš 2013, J. Financial
Markets — the Information Leadership Share, built to correct IS/CS bias
when the two prices carry unequal microstructure noise; Alexander & Heck
2020, J. Financial Stability 50:100776 — Bitstamp and Coinbase are both in
their regulated-spot panel and do not always lead each other). R-76's own
pre-registration flagged this as an *unaddressed confound* on its BTC leg
("a genuine cross-venue basis could add noise... not corrected for") —
this round is the first to test the confound as a signal in its own right.

**Data.** ``data/btcusd_coinbase_spot_5m.csv.gz`` — fetched fresh this
round via the already-proven ``scripts/fetch_coinbase_spot.py`` (same
endpoint/format R-57's six-asset panel and R-39's ETH basis series used),
``--product BTC-USD --start 2017-01-01``. This is new data, not something
already sitting in the repo — the honest disclosure ROUTINE.md's step 1
Q3 asks for when a round needs to build a missing capability rather than
proxy it out of price.

**Causal join.** Both series are already 5-minute OHLCV on (nominally) the
same wall-clock grid — unlike the on-chain/macro/stablecoin/DVOL/MVRV
loaders, there is no slower series to as-of-shift. ``align()``
(``tradebot.data``) restricts both frames to their exact common
timestamps, which is already a same-bar (not future-bar) comparison: bar
T's Bitstamp close and bar T's Coinbase close both describe the same
closed 5-minute window on each venue's own tape, so ``divergence[T]`` uses
no information not available to both venues by the time bar T closes.
Rows either side does not have (a venue outage, a gap) are dropped by the
intersection, not filled — same discipline as every other causal loader in
this file's family.

``divergence = log(bitstamp_close / coinbase_close)``, positive means
Bitstamp trades above Coinbase for that bar. This is the ONLY quantity
computed here; no scaler, quantile, mean or std is fit over the whole
series (the specific full-series-fit lookahead ROUTINE.md's skeptic
checklist calls out) — any rolling/ewm statistics of ``divergence`` are
each branch's own responsibility, using only causal (``rolling``/``ewm``/
``shift``) operations on this column, exactly like every registered
strategy's own ``prepare()``.

Split convention (unchanged from every other round): inner-train
2017-01-01 -> 2020-12-31, inner-validation 2021-01-01 -> 2022-12-31,
holdout (``OOS_START``) 2023-01-01 -> present. Do not read past
``INNER_VALID[1]`` until your own pre-registered decision rule is frozen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

INNER_TRAIN_END = "2020-12-31"
INNER_VALID = ("2021-01-01", "2022-12-31")
OOS_START = "2023-01-01"

COINBASE_BTC_FILE = "btcusd_coinbase_spot_5m.csv.gz"


def load_crossvenue_bars() -> pd.DataFrame:
    """Bitstamp BTC bars (the tradable series) with a causal ``divergence`` column.

    Returns the full Bitstamp OHLCV frame, restricted to timestamps where
    Coinbase also has a bar, with one new column: ``divergence =
    log(bitstamp_close / coinbase_close)``. Every other column
    (open/high/low/close/volume) is Bitstamp's own, unchanged — this is
    the series every registered strategy already trades, so a strategy
    built on this frame is a strict superset of ``kelly_regime_v4``'s own
    input.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from tradebot.data import load_coinbase_spot, load_dataset

    bitstamp, label = load_dataset(DATA_DIR, "spot")
    if label != "real":
        raise RuntimeError(f"expected real Bitstamp spot data, got label={label!r}")

    coinbase = load_coinbase_spot(DATA_DIR, "BTC")
    if coinbase is None:
        raise RuntimeError(
            f"missing {COINBASE_BTC_FILE} — run scripts/fetch_coinbase_spot.py "
            "--product BTC-USD --start 2017-01-01 --out "
            f"data/{COINBASE_BTC_FILE} first"
        )

    common = bitstamp.index.intersection(coinbase.index)
    bars = bitstamp.loc[common].copy()
    cb_close = coinbase.loc[common, "close"]
    bars["divergence"] = np.log(bars["close"] / cb_close)
    return bars


def coverage_report(bars: pd.DataFrame) -> str:
    """One-line summary of the aligned window, for each branch's own sanity check."""
    return (
        f"{len(bars)} aligned bars, {bars.index.min()} -> {bars.index.max()}, "
        f"divergence: mean={bars['divergence'].mean():.6f} "
        f"std={bars['divergence'].std():.6f} "
        f"abs-median={bars['divergence'].abs().median():.6f}"
    )


if __name__ == "__main__":
    b = load_crossvenue_bars()
    print(coverage_report(b))
    print(f"inner-train:      {b.index.min()} -> {INNER_TRAIN_END}")
    print(f"inner-validation: {INNER_VALID[0]} -> {INNER_VALID[1]}")
    print(f"holdout (DO NOT READ yet): {OOS_START} -> {b.index.max()}")
