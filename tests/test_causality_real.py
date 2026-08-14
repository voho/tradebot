"""No-lookahead checks on real data for strategies with long warmups.

The synthetic series used by ``test_engine.py`` is shorter than some
strategies' warmup (e.g. a 100-day regime anchor needs ~29k bars), so
those strategies pass that test vacuously. This module re-runs the
truncation check on a slice of the committed dataset, which is long
enough to exercise them.
"""

from pathlib import Path

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
