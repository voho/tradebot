"""Shared, causality-checked utility: align funding rate onto the 5m bar grid.

Used by the two B-05 branches (``funding_gate_conservative.py`` and
``funding_crowding_novel.py``) so the lookahead-safety of the alignment is
proven once rather than reimplemented twice.

Funding (``tradebot.data.load_funding``) is indexed by 8-hourly settlement
timestamp and is not passed to ``Strategy.prepare(df)`` by the engine - it
is only wired in as a *cost* via ``run_backtest(..., funding=...)``. To use
it as a *signal* inside a strategy, the caller must merge it onto the bar
index itself, before construction, e.g.::

    df = DF.copy()
    df["funding"] = causal_funding_column(df.index, REAL)
    strat = MyStrategy()
    # then run_backtest(strat, df, ...) as usual - prepare() sees df["funding"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_funding_column(index: pd.DatetimeIndex, funding: pd.Series) -> np.ndarray:
    """Strictly-causal funding rate aligned to ``index``.

    For bar ``i``, only settlements with timestamp <= ``index[i]`` are
    visible; the most recently *settled* rate is held constant until the
    next settlement (``pandas.Series.reindex(method="ffill")`` on a sorted
    DatetimeIndex does exactly this - it never looks at a label greater
    than the target). Bars before the first settlement, or after funding
    data ends, get 0.0 (a genuine "no signal" rather than a fabricated
    rate) - so any strategy using this column must be evaluated only over
    the period ``funding.index[0] <= t <= funding.index[-1]``, which the
    experiment code is responsible for enforcing when it slices train /
    inner-validation / holdout windows.
    """
    aligned = funding.reindex(index, method="ffill")
    values = aligned.to_numpy(dtype=float, na_value=0.0)
    # reindex(method="ffill") propagates the LAST known value past the end
    # of `funding`'s own index too, which would fabricate a constant rate
    # for every bar after 2023-12-31 if left uncorrected. Mask the tail
    # back to 0.0 so a strategy built on this column degrades to "funding
    # unknown -> assume zero drag" outside the committed window, rather
    # than carrying a stale rate indefinitely.
    values[index > funding.index[-1]] = 0.0
    return values


def funding_coverage(funding: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The [start, end] settlement timestamps the committed funding file covers."""
    return funding.index[0], funding.index[-1]
