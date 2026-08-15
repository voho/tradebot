"""Evaluate a strategy over a sub-period of a longer series, fairly.

Slicing a frame to a date range and backtesting it looks right and is
not: a strategy with a 100-day warmup cannot trade for the first 100 days
of the slice, while a zero-warmup benchmark trades from day one. On the
2023-2026 out-of-sample split that is **7.6% of the period spent flat** —
a handicap applied to exactly the strategies being evaluated, and one
that reads as a genuine result.

:func:`run_period` fixes it by taking the warmup from the bars *before*
the period and disabling trading until the period actually starts, so
every strategy enters the measured region warm, flat, and with the full
starting balance.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.engine import BacktestResult, run_backtest
from tradebot.strategy import Strategy


def prefix_bars(df: pd.DataFrame, start_pos: int, warmup: int) -> int:
    """How many warmup bars are actually available before ``start_pos``."""
    return int(min(start_pos, max(warmup, 0)))


def run_period(
    strategy: Strategy,
    df: pd.DataFrame,
    start: object | None = None,
    end: object | None = None,
    *,
    market: MarketSpec,
    start_balance: float = 1_000.0,
    slippage_bps: float = 0.0,
    data_label: str = "",
) -> BacktestResult:
    """Backtest ``strategy`` over ``df[start:end]`` with a real warmup prefix.

    ``start``/``end`` are label-based and inclusive, like ``df.loc[a:b]``.
    Bars before ``start`` warm the strategy's indicators and internal
    state but cannot trade. The returned result is trimmed to the measured
    period, so its equity curve begins at ``start_balance`` and its
    metrics describe the period rather than the prefix.

    When there is not enough history before ``start`` (the beginning of
    the dataset), the prefix is as long as the data allows — the run is
    still correct, just cold, which is unavoidable and is why the first
    in-sample split cannot be made fair to a zero-warmup benchmark.
    """
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")

    prefix = prefix_bars(df, lo, strategy.warmup)
    frame = df.iloc[lo - prefix: hi]

    result = run_backtest(strategy, frame, market, start_balance,
                          slippage_bps=slippage_bps, data_label=data_label,
                          trade_start=prefix)
    if prefix == 0:
        return result
    # trade_start guarantees no fill lands before `prefix`, so trimming the
    # curve is all that is needed - fills and trades are already in-period.
    return replace(result,
                   equity=result.equity.iloc[prefix:],
                   df=result.df.iloc[prefix:])
