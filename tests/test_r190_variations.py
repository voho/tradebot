"""Causality and actual-account execution checks for the frozen R-190 family."""

import numpy as np
import pandas as pd
import pytest

from experiments.r190_variations import CONFIGS, PARENTS, make_strategy
from tradebot.broker import MarketSpec, PaperBroker
from tradebot.engine import run_backtest
from tradebot.registry import available_strategies
from tradebot.strategy import Context

from conftest import make_ohlcv


@pytest.fixture(scope="module")
def regimes():
    t = np.arange(34_000)
    returns = (0.0001 * np.sin(t / 900) + 0.0002 * np.cos(t / 70)
               + np.random.default_rng(29).normal(0, 0.001, len(t)))
    return make_ohlcv(100 * np.exp(np.cumsum(returns)))


@pytest.mark.parametrize("name,parent,band", CONFIGS)
def test_parent_targets_remain_unchanged_bounded_and_causal(name, parent, band, regimes):
    strategy = make_strategy(name)
    cut = 31_200  # A UTC decision slot after every parent's full warmup.
    original = strategy.prepare(regimes.copy())
    modified = regimes.copy()
    modified.iloc[cut + 1:, :4] *= 3
    after = strategy.prepare(modified)
    prefix = make_strategy(name).prepare(regimes.iloc[:cut + 1].copy())
    assert original.r190_decision.iloc[cut] and cut > strategy.warmup
    for column in ("target", "r190_decision"):
        np.testing.assert_array_equal(original[column].iloc[:cut + 1],
                                      after[column].iloc[:cut + 1])
        np.testing.assert_array_equal(original[column].iloc[:cut + 1], prefix[column])
    parents = PARENTS.values() if parent == "blend" else (PARENTS[parent],)
    expected = np.mean([cls().prepare(regimes.copy()).target for cls in parents], axis=0)
    np.testing.assert_array_equal(original.target, expected)
    assert strategy.warmup == max(cls.warmup for cls in strategy.parents)
    assert np.isfinite(original.target).all() and original.target.between(0, 2).all()
    assert np.ptp(original.target.iloc[strategy.warmup:]) > 0.05
    assert name not in available_strategies()


def test_utc_schedule_survives_history_offset_and_timezone(regimes):
    strategy = make_strategy(CONFIGS[0][0])
    frame = regimes.iloc[:1000].copy()
    full = strategy.prepare(frame.copy())
    offset = strategy.prepare(frame.iloc[13:].copy())
    local = strategy.prepare(frame.tz_convert("Europe/Prague"))
    naive = strategy.prepare(frame.tz_localize(None))
    np.testing.assert_array_equal(full.r190_decision.iloc[13:], offset.r190_decision)
    np.testing.assert_array_equal(full.r190_decision, local.r190_decision)
    np.testing.assert_array_equal(full.r190_decision, naive.r190_decision)
    assert full.r190_decision.groupby(full.index.date).sum().max() == 6


@pytest.mark.parametrize("name,parent,band", CONFIGS)
def test_fresh_entry_actual_drift_and_small_zero_target_exit(name, parent, band):
    df = make_ohlcv([100.0] * 50)
    df["target"] = 0.7
    df["r190_decision"] = (df.index.hour % 4 == 0) & (df.index.minute == 0)
    strategy = make_strategy(name)
    broker = PaperBroker(MarketSpec.futures(leverage=5), start_balance=10_000)

    def orders(i=48):
        ctx = Context(df, i, broker)
        strategy.on_bar(ctx)
        return ctx.orders

    # The current target was already held in the signal history, but the
    # account starts fresh; no prior-target-change callback is necessary.
    assert orders(47) == []
    assert len(orders()) == 1 and orders()[0].target == pytest.approx(0.7 / 5)
    broker.entry = 100.0
    broker.pos = (0.7 - band / 2) * 100
    assert orders() == []
    broker.pos = (0.7 - band - 0.01) * 100
    assert len(orders()) == 1 and orders()[0].target == pytest.approx(0.7 / 5)
    # Funding or price drift can change equity while base holdings are
    # unchanged. The callback must use marked equity, not signal differences.
    broker.pos = 70.0
    broker.cash = 5_000
    assert len(orders()) == 1
    df["target"] = 0.0
    broker.pos = 0.001
    assert orders()[0].target == 0.0
    broker.execute(orders()[0], df.index[49], 100.0)
    assert broker.pos == 0.0 and orders() == []
    broker.cash = 0.0
    assert orders() == []


@pytest.mark.parametrize("market", [MarketSpec.spot(), MarketSpec.futures(leverage=5)])
def test_orders_fill_at_next_open_and_native_market_caps_are_retained(market):
    # A controlled parent target isolates execution from indicator fitting.
    strategy = make_strategy("r190_v4_b05")
    strategy.warmup = 0
    df = make_ohlcv([100.0] * 100)
    df["target"] = 2.0
    df["r190_decision"] = (df.index.hour % 4 == 0) & (df.index.minute == 0)
    strategy.prepare = lambda frame: frame
    df.iloc[1, df.columns.get_loc("open")] = 110.0
    df.iloc[1, df.columns.get_loc("high")] = 110.1
    result = run_backtest(strategy, df, market, 1_000, slippage_bps=1.0)
    assert result.fills[0].ts == df.index[1]
    assert result.fills[0].price == pytest.approx(110.0 * 1.0001)
    assert result.fills[0].qty * 110.0 <= 1_000 * min(2.0, market.leverage)
    counts = pd.Series(1, index=[fill.ts for fill in result.fills]).groupby(lambda t: t.date()).sum()
    assert counts.max() <= 6
    # The frozen venue floor still absorbs a 10%-of-equity correction on
    # 5x futures (its 5%-of-max-notional band is 25% of equity).
    broker = PaperBroker(market, start_balance=10_000)
    broker.entry, broker.pos = 100.0, 60.0
    df["target"] = 0.7
    ctx = Context(df, 48, broker)
    strategy.on_bar(ctx)
    assert len(ctx.orders) == 1
    fills = broker.execute(ctx.orders[0], df.index[49], 100.0)
    assert bool(fills) == (market.leverage == 1)
