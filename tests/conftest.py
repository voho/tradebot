import numpy as np
import pandas as pd
import pytest


def make_ohlcv(prices, start="2025-01-01", freq="5min") -> pd.DataFrame:
    """Build a valid OHLCV frame from a list of close prices.

    open = previous close (first open = first close), high/low padded around.
    """
    closes = np.asarray(prices, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.ones(len(closes))},
        index=idx,
    )


@pytest.fixture
def flat_df():
    return make_ohlcv([100.0] * 50)


@pytest.fixture
def trend_df():
    return make_ohlcv(np.linspace(100.0, 200.0, 300))
