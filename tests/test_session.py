"""The UTC-session helpers behind the intraday candidates (R-188) are causal.

They are the one piece of that round that stayed in ``src/``: a per-session
open, previous close, VWAP and time-of-day slot that any future intraday
strategy will need, each of which is easy to get subtly wrong (a
``transform('last')`` is the whole future of the day).
"""

import numpy as np
import pandas as pd

from tradebot.session import (day_id, previous_session_close, session_open,
                              session_vwap, slot)

from conftest import make_ohlcv


def _two_days():
    rng = np.random.default_rng(1)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.2, 2 * 288))
    df = make_ohlcv(closes, start="2025-01-01")
    df["volume"] = rng.uniform(1.0, 5.0, len(df))
    return df


def test_slot_and_day_id_follow_the_utc_clock():
    df = _two_days()
    s, d = slot(df.index), day_id(df.index)
    assert s[0] == 0 and s[287] == 287 and s[288] == 0
    assert d[0] == 0 and d[287] == 0 and d[288] == 1


def test_session_open_is_the_first_open_of_the_day():
    df = _two_days()
    d = day_id(df.index)
    o = session_open(df, d)
    assert (o.iloc[:288] == df["open"].iloc[0]).all()
    assert (o.iloc[288:] == df["open"].iloc[288]).all()


def test_previous_close_is_yesterdays_last_close_and_falls_back_on_day_one():
    df = _two_days()
    d = day_id(df.index)
    p = previous_session_close(df, d)
    assert (p.iloc[:288] == df["open"].iloc[:288]).all()
    assert (p.iloc[288:] == df["close"].iloc[287]).all()


def test_vwap_is_causal_and_resets_each_session():
    df = _two_days()
    d = day_id(df.index)
    v = session_vwap(df, d)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    # first bar of each day: the VWAP is that bar's own typical price
    assert np.isclose(v.iloc[0], typical.iloc[0])
    assert np.isclose(v.iloc[288], typical.iloc[288])
    # tampering with later bars must not move earlier values
    tampered = df.copy()
    tampered.iloc[100:, tampered.columns.get_loc("close")] *= 3.0
    tampered.iloc[100:, tampered.columns.get_loc("volume")] *= 7.0
    v2 = session_vwap(tampered, d)
    pd.testing.assert_series_equal(v.iloc[:100], v2.iloc[:100])
    assert not np.allclose(v.iloc[100:], v2.iloc[100:])
