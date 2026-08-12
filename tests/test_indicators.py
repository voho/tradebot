import numpy as np
import pandas as pd

from tradebot.indicators import crossed_above, crossed_below, ema, macd, rsi


def test_ema_converges_to_constant():
    s = pd.Series([5.0] * 100)
    assert np.allclose(ema(s, 10), 5.0)


def test_ema_is_causal():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(100, 5, 200))
    full = ema(s, 12)
    trunc = ema(s.iloc[:100], 12)
    assert np.allclose(full.iloc[:100], trunc)


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 100))
    down = pd.Series(np.linspace(200, 100, 100))
    r_up, r_dn = rsi(up), rsi(down)
    assert ((r_up >= 0) & (r_up <= 100)).all()
    assert r_up.iloc[-1] > 90  # monotonic gains -> extreme RSI
    assert r_dn.iloc[-1] < 10


def test_rsi_flat_is_neutral():
    r = rsi(pd.Series([100.0] * 50))
    assert (r == 50.0).all()


def test_macd_columns_and_causality():
    rng = np.random.default_rng(1)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    m = macd(s)
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    assert np.allclose(m["macd_hist"], m["macd"] - m["macd_signal"])
    m_trunc = macd(s.iloc[:150])
    assert np.allclose(m.iloc[:150]["macd"], m_trunc["macd"])


def test_crossovers():
    a = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    b = pd.Series([2.0] * 5)
    up = crossed_above(a, b)
    dn = crossed_below(a, b)
    assert list(up) == [False, False, True, False, False]
    assert list(dn) == [False, False, False, False, True]
