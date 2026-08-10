"""Backtest engine correctness.

The engine's job is arithmetic, and arithmetic can be checked exactly.  The
anchor test is that a strategy which is always fully long, charged no costs,
reproduces buy-and-hold to floating-point precision.  If that holds, the P&L
attribution across the fill price is right; if it does not, no strategy result
from the engine means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtbot.data.schema import validate
from gtbot.data.synthetic import simulate
from gtbot.engine.backtest import run_backtest
from gtbot.engine.broker import Broker, CostModel, ExecutionConfig
from gtbot.strategy import GameTheoreticStrategy, StrategyConfig

FREE = CostModel(taker_fee_bp=0.0, maker_fee_bp=0.0, half_spread_bp=0.0, impact_coef=0.0)


class AlwaysLong(GameTheoreticStrategy):
    def observe(self, t):  # no learning: keep the test about arithmetic
        pass

    def decide(self, t):
        return 1.0


class AlwaysFlat(GameTheoreticStrategy):
    def observe(self, t):
        pass

    def decide(self, t):
        return 0.0


@pytest.fixture(scope="module")
def bars():
    return validate(simulate(4000, seed=5).bars)


def test_full_long_reproduces_buy_and_hold(bars):
    res = run_backtest(
        bars, AlwaysLong(StrategyConfig()), costs=FREE, execution=ExecutionConfig(entry_mode="taker", exit_mode="taker")
    )
    strat = AlwaysLong(StrategyConfig())
    strat.prepare(bars)
    w = strat.warmup

    close = bars["close"].to_numpy()
    expected = close[-1] / close[w + 1] - 1.0
    actual = res.equity[-1] / res.equity[w + 1] - 1.0
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_flat_strategy_never_moves_equity(bars):
    res = run_backtest(bars, AlwaysFlat(StrategyConfig()), costs=CostModel())
    assert np.allclose(res.equity, res.equity[0])
    assert res.n_trades == 0


def test_costs_only_ever_reduce_equity(bars):
    free = run_backtest(
        bars, AlwaysLong(StrategyConfig()), costs=FREE, execution=ExecutionConfig(entry_mode="taker", exit_mode="taker")
    )
    charged = run_backtest(
        bars,
        AlwaysLong(StrategyConfig()),
        costs=CostModel(),
        execution=ExecutionConfig(entry_mode="taker", exit_mode="taker"),
    )
    assert charged.equity[-1] <= free.equity[-1]
    assert np.all(charged.costs >= 0.0)


def test_position_never_exceeds_leverage_cap(bars):
    res = run_backtest(bars, GameTheoreticStrategy(StrategyConfig()), max_leverage=1.5)
    assert np.max(np.abs(res.position)) <= 1.5 + 1e-12


def test_broker_classifies_entries_and_exits():
    broker = Broker(CostModel(), ExecutionConfig())
    assert broker._is_entry(0.5, 0.0) is True  # open
    assert broker._is_entry(0.8, 0.5) is True  # add
    assert broker._is_entry(0.2, 0.5) is False  # reduce
    assert broker._is_entry(0.0, 0.5) is False  # close
    assert broker._is_entry(-0.3, 0.5) is True  # flip


def test_maker_fill_requires_the_bar_to_trade_through():
    broker = Broker(CostModel(), ExecutionConfig(entry_mode="maker", maker_offset_atr=0.0, queue_ticks=1.0))
    common = dict(
        bar=1, target=1.0, position=0.0, open_=100.0, atr_frac=0.002,
        quote_volume=1e9, bar_vol_bp=20.0, equity=1e5, force_taker=False,
    )
    # Bar never trades below the open: a resting bid cannot have been filled.
    assert broker.try_execute(high=101.0, low=100.0, **common) is None
    # Bar trades through: filled, and as a maker.
    fill = broker.try_execute(high=101.0, low=99.0, **common)
    assert fill is not None and fill.is_maker


def test_taker_always_fills_at_the_open():
    broker = Broker(CostModel(), ExecutionConfig(entry_mode="taker"))
    fill = broker.try_execute(
        bar=1, target=1.0, position=0.0, open_=100.0, high=100.0, low=100.0,
        atr_frac=0.002, quote_volume=1e9, bar_vol_bp=20.0, equity=1e5, force_taker=False,
    )
    assert fill is not None and not fill.is_maker and fill.price == 100.0


def test_slippage_grows_with_participation():
    c = CostModel()
    small = c.slippage_bp(1_000.0, 1e7, 20.0)
    large = c.slippage_bp(1_000_000.0, 1e7, 20.0)
    assert 0.0 < small < large
