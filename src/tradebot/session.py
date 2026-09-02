"""Causal UTC-session helpers shared by the intraday strategies.

Crypto trades around the clock, so "the session" here is the UTC day. Every
helper below is causal: a value at bar ``i`` depends only on bars ``<= i``
(``groupby(...).cumsum`` / ``transform('first')`` / a within-group forward
fill all satisfy that), which is what the framework's perturbation test
checks for every registered strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_DAY = 288  # 5-minute bars


def day_id(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer id of each bar's UTC day (consecutive bars share an id)."""
    return pd.factorize(index.normalize())[0]


def slot(index: pd.DatetimeIndex) -> np.ndarray:
    """Position of each bar inside its UTC day, 0..287 for 5-minute bars."""
    return (index.hour * 12 + index.minute // 5).to_numpy()


def session_open(df: pd.DataFrame, day: np.ndarray) -> pd.Series:
    """The day's first open, broadcast to every bar of that day."""
    return df["open"].groupby(day).transform("first")


def previous_session_close(df: pd.DataFrame, day: np.ndarray) -> pd.Series:
    """The previous day's last close, broadcast to every bar of the day.

    Falls back to the day's own open on the first day of the data.
    """
    close = df["close"]
    first = pd.Series(day, index=df.index).diff().fillna(1.0).ne(0.0)
    prev = close.shift(1).where(first).ffill()
    return prev.fillna(df["open"])


def session_vwap(df: pd.DataFrame, day: np.ndarray) -> pd.Series:
    """Cumulative volume-weighted typical price since the session open."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].clip(lower=0.0)
    pv = (typical * vol).groupby(day).cumsum()
    v = vol.groupby(day).cumsum()
    # a session with zero volume so far falls back to the typical price
    return (pv / v.replace(0.0, np.nan)).fillna(typical)
