"""No-lookahead checks on real data for strategies with long warmups.

The synthetic series used by ``test_engine.py`` is shorter than some
strategies' warmup (e.g. a 100-day regime anchor needs ~29k bars), so
those strategies pass that test vacuously. This module re-runs the
truncation check on a slice of the committed dataset, which is long
enough to exercise them.
"""

from pathlib import Path

import pandas as pd
import pytest

from tradebot.broker import MarketSpec
from tradebot.data import load_dataset
from tradebot.engine import run_backtest
from tradebot.registry import available_strategies, get_strategy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BARS = 60_000  # must exceed the longest strategy warmup
CUT = 15_000  # bars withheld from the truncated run

LONG_WARMUP = sorted(
    name for name, cls in available_strategies().items() if cls().warmup > 5_000
)


@pytest.fixture(scope="module")
def real_slice():
    df, label = load_dataset(DATA_DIR, "spot")
    if label == "SYNTHETIC":
        pytest.skip("real dataset not present")
    if len(df) < BARS:
        pytest.skip("dataset too short")
    return df.iloc[-BARS:]


@pytest.mark.parametrize("name", LONG_WARMUP)
def test_no_lookahead_on_real_data(name, real_slice):
    """Withholding the final bars must not change any earlier fill."""
    market = MarketSpec.spot()
    keep = len(real_slice) - CUT
    cutoff = real_slice.index[keep - 1]

    full = run_backtest(get_strategy(name), real_slice, market, 10_000.0)
    part = run_backtest(get_strategy(name), real_slice.iloc[:keep], market, 10_000.0)

    def fills(result):
        return [(f.ts, f.side, round(f.qty, 9), round(f.price, 9))
                for f in result.fills if f.ts <= cutoff]

    assert fills(full) == fills(part)


@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_prepare_columns_ignore_future_bars(name, real_slice):
    """Perturbing FUTURE bars must not change any prepared value at bar i.

    Stronger than the truncation check: a strategy that aggregates to a
    coarser timeframe (daily, say) and broadcasts the result back onto
    every bar of the SAME period passes truncation while still leaking -
    a same-day daily signal carries that whole day of future. Multiplying
    the tail of the series and re-running catches it directly.
    """
    df = real_slice.iloc[-40_000:].copy()
    cut = len(df) - 5_000

    strategy = get_strategy(name)
    base = strategy.prepare(df.copy())

    tampered = df.copy()
    for col in ("open", "high", "low", "close"):
        tampered.iloc[cut:, tampered.columns.get_loc(col)] *= 3.0
    tampered.iloc[cut:, tampered.columns.get_loc("volume")] *= 7.0
    after = get_strategy(name).prepare(tampered)

    for col in base.columns:
        if col in ("open", "high", "low", "close", "volume"):
            continue
        a = base[col].to_numpy()[:cut]
        b = after[col].to_numpy()[:cut]
        mismatch = ~(pd.isna(a) & pd.isna(b)) & (a != b)
        assert not mismatch.any(), (
            f"{name}: column {col!r} changed at {int(mismatch.sum())} bars before "
            "the cut when only FUTURE bars were modified - the signal leaks")
