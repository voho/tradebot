"""Shared, read-only helper for the R-115 CONSERVATIVE branch (08-24).

DIRECTION, in one sentence: R-112's own closing verdict named a concrete,
undone fix -- "`data/ethusd_coinbase_spot_5m.csv.gz` (the same exchange as
`UNIVERSE_6`) ... spans 2019-03-14 through the present ... overlaps
`UNIVERSE_6`'s 2020-01-02 start for the entire 2020-2022 inner-validation
window, unlike the Bitfinex series R-109/R-112 both used. Swapping
`load_eth()`'s source is a genuine, disclosed change to the falsification
instrument itself" -- and this file does exactly that one thing: a single
loader, swapping ETH's data SOURCE only, nothing else.

WHY THIS WAS NECESSARY, restated from the operator's own brief: R-109 novel
and R-112 novel both called `experiments/r109_shared.py`'s `load_eth()`,
which reads `data/ethusd_bitfinex_5m.csv.gz` -- a series that ends
2019-12-31, before `UNIVERSE_6`'s six committed Coinbase panels (BCH, LTC,
ETC, DASH, LINK, XTZ) even begin (2020-01-02). Zero calendar overlap. Every
refit of R-112 novel's own `rolling_knn_distance_pooled` inside ETH's
2019-2022 evaluation window therefore found every pool instrument's window
empty (`len(pwin) < MIN_REF_DAYS`) and silently fell back to using only the
target's own window -- mechanically identical to R-109's original,
UNPOOLED, single-asset construction. R-112's own B4 numbers for its novel
branch came out bit-for-bit identical to R-109 novel's own B4 numbers,
proving the pool never actually engaged on the falsification instrument.
This file's `load_eth_coinbase()` reads the Coinbase USD spot series
instead -- the SAME exchange `UNIVERSE_6`'s own six panels already use --
so the pool now genuinely overlaps ETH's own evaluation window for the
entire 2020-2022 span. Confirmed by DIRECT READ of both files (not the
README) before any strategy code in this round was written:

    ethusd_coinbase_spot_5m.csv.gz : 2019-03-14 00:00 UTC -> 2026-08-19 00:00 UTC
    UNIVERSE_6 (BCH/LTC/ETC/DASH/LINK/XTZ, Coinbase spot):
        every one of the six starts 2020-01-01 00:00 UTC

So the Coinbase ETH series overlaps UNIVERSE_6's coverage from 2020-01-01
through the end of this round's non-holdout window (2022-12-31) -- roughly
three of the full ~3.8-year `eth_replication` evaluation window's years
(the full pre-`OOS_START` ETH frame, per `compare()`'s own
`ETH_SLICE_NAME` convention, spans 2019-03-14 -> 2022-12-31). PRECISION,
stated plainly: only the 2020-01-01 -> 2022-12-31 portion is genuinely
POOLED (`rolling_knn_distance_pooled` finds non-empty pool windows there);
the leading ~9.5 months (2019-03-14 -> 2019-12-31) predates UNIVERSE_6's
own coverage and falls back to the single-asset (target-only) construction
for that stretch only, exactly as R-109's own construction did throughout.
This is still a fundamentally different, and far larger, overlap than
R-109/R-112's own Bitfinex-sourced ETH series achieved (ZERO overlap,
every single day) -- the pool now genuinely engages for the majority of
the falsification window instead of never engaging at all.

CHANGED, and ONLY this: the ETH data SOURCE (Bitfinex -> Coinbase spot).
NOT changed, at all, from R-112 novel: the 5-feature panel
(`r109_shared.NOVEL_FEATURE_BUILDERS`), `rolling_knn_distance_pooled`'s own
mechanism (k=10, refit_every=30, CORAL-standardized pooling against
`UNIVERSE_6`), the Step-0 grid, the discount architecture, or the B1-B5
promotion bar. Every one of those is imported read-only from
`experiments/r112_shared.py` (which itself chains `r109_shared.py` ->
`r106_shared.py` -> ... -> `r102_shared.py`) by
`experiments/r115_conservative_pooled_eth_coinbase.py`.

Per this project's own established pattern (R-112's own conservative and
novel branches, R-63's own two branches, etc.), this file does NOT edit
`r109_shared.py`, `r112_shared.py`, or `r63_shared.py` -- it is a new,
small, read-only-import module that adds exactly the one function this
round's whole construction change requires.

Nothing here reads a bar at or after `OOS_START` (2023-01-01):
`load_eth_coinbase()` truncates strictly below it and calls
`assert_no_holdout` before returning, identical convention to every prior
round's own ETH loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot.data import load_ohlcv_csv  # noqa: E402

# Re-exported read-only from r112_shared (itself chaining r109_shared ->
# r106_shared -> ... -> r102_shared): NOT redefined here, so OOS_START and
# assert_no_holdout are byte-identical to every other module in this
# round's own ERR-axis sub-line.
from experiments.r112_shared import OOS_START, assert_no_holdout  # noqa: E402,F401

ETH_COINBASE_PATH = ROOT / "data" / "ethusd_coinbase_spot_5m.csv.gz"


def load_eth_coinbase() -> pd.DataFrame:
    """ETH via the Coinbase USD 5m spot series -- the SAME exchange
    `experiments/r63_shared.UNIVERSE_6`'s own six panels already use --
    truncated strictly below `OOS_START`. This is this round's ENTIRE,
    disclosed change to R-112 novel branch's own construction: everything
    else (the 5-feature panel, `rolling_knn_distance_pooled`, the Step-0
    grid, the B1-B5 gate code) is imported unmodified from `r112_shared`."""
    df = load_ohlcv_csv(ETH_COINBASE_PATH)
    out = df[df.index < pd.Timestamp(OOS_START, tz="UTC")]
    assert_no_holdout(out, "ETH (Coinbase spot)")
    return out


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    eth = load_eth_coinbase()
    assert len(eth) > 0
    assert eth.index.max() < pd.Timestamp(OOS_START, tz="UTC")
    # The specific, pre-registered claim this file exists to make true:
    # the Coinbase ETH series' own first bar predates UNIVERSE_6's own
    # 2020-01-02 start (i.e. genuine overlap, not a lucky coincidence of
    # truncation).
    assert eth.index.min() < pd.Timestamp("2020-01-02", tz="UTC"), (
        "Coinbase ETH series does not start before UNIVERSE_6's own "
        "2020-01-02 coverage -- the overlap claim this file rests on is false")
    assert eth.index.min() <= pd.Timestamp("2019-03-15", tz="UTC"), (
        "Coinbase ETH series' own start date has moved since this file was "
        "written -- re-verify the overlap claim in this module's docstring")


_self_test()
