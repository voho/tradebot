"""Sub-period evaluation must be warm, flat and fairly measured."""

import numpy as np
import pytest

from tradebot.broker import MarketSpec
from tradebot.engine import run_backtest
from tradebot.registry import get_strategy
from tradebot.strategy import Context, Strategy
from tradebot.window import prefix_bars, run_period

from conftest import make_ohlcv


class WarmupHungry(Strategy):
    """Needs 100 bars of warmup, then holds."""

    name = "_test_warmup_hungry"
    warmup = 100

    def on_bar(self, ctx: Context) -> None:
        if not ctx.in_market:
            ctx.order_target(1.0)


def _rising(n=400):
    return make_ohlcv(np.linspace(100.0, 200.0, n))


def test_period_run_trades_from_the_first_bar_of_the_period():
    """Naive slicing costs the strategy its whole warmup; run_period does not."""
    df = _rising()
    start = df.index[200]
    market = MarketSpec.spot(fee_rate=0.0)

    naive = run_backtest(WarmupHungry(), df.loc[start:], market, 1_000.0)
    fair = run_period(WarmupHungry(), df, start, market=market, start_balance=1_000.0)

    # naive cannot act until 100 bars into the period; fair acts immediately
    assert naive.fills[0].ts > fair.fills[0].ts
    assert fair.fills[0].ts == df.index[201]  # signal at `start`, fill next open
    assert fair.final_balance > naive.final_balance


def test_period_equity_starts_at_the_start_balance():
    df = _rising()
    result = run_period(WarmupHungry(), df, df.index[200],
                        market=MarketSpec.spot(), start_balance=1_000.0)
    assert result.equity.iloc[0] == pytest.approx(1_000.0)
    assert result.equity.index[0] == df.index[200]
    assert len(result.equity) == len(result.df)


def test_period_never_reports_a_fill_from_the_prefix():
    df = _rising()
    start = df.index[200]
    result = run_period(WarmupHungry(), df, start, market=MarketSpec.spot())
    assert all(f.ts >= start for f in result.fills)
    assert all(t.entry_ts >= start for t in result.trades)


def test_period_end_is_inclusive():
    df = _rising()
    result = run_period(WarmupHungry(), df, df.index[200], df.index[300],
                        market=MarketSpec.spot())
    assert result.equity.index[0] == df.index[200]
    assert result.equity.index[-1] == df.index[300]


def test_period_without_enough_history_uses_what_there_is():
    df = _rising()
    result = run_period(WarmupHungry(), df, df.index[10], market=MarketSpec.spot())
    assert prefix_bars(df, 10, 100) == 10
    # cold start is unavoidable at the very beginning of a dataset, but the
    # run must still be valid rather than raising or silently empty
    assert len(result.equity) == len(df) - 10


def test_period_matches_a_full_run_when_it_covers_everything():
    df = _rising()
    market = MarketSpec.spot()
    full = run_backtest(WarmupHungry(), df, market, 1_000.0)
    period = run_period(WarmupHungry(), df, market=market, start_balance=1_000.0)
    assert period.final_balance == pytest.approx(full.final_balance)
    assert len(period.equity) == len(full.equity)


def test_period_rejects_an_empty_range():
    df = _rising()
    with pytest.raises(ValueError, match="empty period"):
        run_period(WarmupHungry(), df, df.index[300], df.index[200],
                   market=MarketSpec.spot())


def test_registered_strategy_over_a_period_is_warm_immediately():
    """The real case: a 100-day anchor evaluated on a 2023+ split."""
    df = make_ohlcv(np.linspace(100.0, 300.0, 2_000))
    strategy = get_strategy("macd_cross")
    result = run_period(strategy, df, df.index[1_000],
                        market=MarketSpec.futures(leverage=5.0))
    assert result.equity.index[0] == df.index[1_000]
    assert result.equity.iloc[0] == pytest.approx(1_000.0)
