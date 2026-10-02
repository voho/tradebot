"""Synthetic causality, mechanics and next-open accounting for every R-191 rule."""

import numpy as np
import pandas as pd
import pytest

from experiments.r191_strategies import (
    ALL_NAMES, FAMILIES, LABELS, PARAMS, PRIMARY_NAMES, SIGNAL_COST, WARMUP_DAYS,
    _adaptive_quantiles, _block_weight, _daily_targets, _olmar_weight,
    _proximal_weight, _sticky_targets, make_strategy,
)
from tradebot.broker import MarketSpec, PaperBroker
from tradebot.engine import run_backtest
from tradebot.registry import available_strategies
from tradebot.strategy import Context

from conftest import make_ohlcv


@pytest.fixture(scope="module")
def regimes():
    days = np.arange(500)
    returns = np.select([days < 160, days < 225, days < 315, days < 400],
                        [.008, -.012, .009, -.01], default=.004)
    returns += .002 * np.sin(days / 3)
    returns[[290, 325, 401, 450]] = -.15
    increments = np.repeat(np.log1p(returns) / 288, 288)
    return make_ohlcv(100 * np.exp(np.cumsum(increments)))


def test_frozen_inventory_and_cost_design():
    assert len(PRIMARY_NAMES) == 5 and len(ALL_NAMES) == 15
    assert len(set(ALL_NAMES)) == 15
    assert set(ALL_NAMES) == set(PARAMS) == set(LABELS)
    assert all(len(family) == 3 for family in FAMILIES.values())
    assert SIGNAL_COST == pytest.approx(.0041)
    assert not set(ALL_NAMES) & set(available_strategies())
    with pytest.raises(ValueError, match="Unknown R-191"):
        make_strategy("unknown")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_configuration_is_causal_nonflat_and_actually_orders(name, regimes):
    strategy = make_strategy(name)
    full = strategy.prepare(regimes.copy())
    assert np.isfinite(full.target).all() and full.target.between(0, 1).all()
    assert np.ptp(full.target.iloc[strategy.warmup:]) > .05
    assert not full.r191_decision.iloc[:WARMUP_DAYS * 288].any()
    assert (full.target.iloc[:WARMUP_DAYS * 288] == 0).all()
    assert full.r191_decision.groupby(full.index.date).sum().max() == 1
    # Both a completed close and a partial-day prefix are after warmup.
    for cut in (340 * 288 - 1, 365 * 288 + 137):
        prefix_strategy = make_strategy(name)
        prefix = prefix_strategy.prepare(regimes.iloc[:cut + 1].copy())
        perturbed = regimes.copy()
        perturbed.iloc[cut + 1:, :4] *= 3.
        perturbed.iloc[cut + 1:, 4] *= 7.
        future_strategy = make_strategy(name)
        future = future_strategy.prepare(perturbed)
        for column in ("target", "r191_decision"):
            np.testing.assert_array_equal(full[column].iloc[:cut + 1], prefix[column])
            np.testing.assert_array_equal(full[column].iloc[:cut + 1], future[column].iloc[:cut + 1])
        # Inspect real callbacks with a fresh account; prepared columns alone
        # would not catch a strategy reaching into saved future data.
        times = np.flatnonzero(full.r191_decision.iloc[:cut + 1])[-100:]
        orders = []
        for current_strategy, frame in ((strategy, full), (prefix_strategy, prefix), (future_strategy, future)):
            broker = PaperBroker(MarketSpec.futures(leverage=5), 10_000)
            current = []
            for i in times:
                ctx = Context(frame, int(i), broker)
                current_strategy.on_bar(ctx)
                current.append([order.target for order in ctx.orders])
            orders.append(current)
        assert orders[0] == orders[1] == orders[2]
        assert any(orders[0]), "causality must exercise nonempty real orders"


@pytest.mark.parametrize("name", PRIMARY_NAMES)
def test_partial_day_has_no_new_target_and_timezone_does_not_change_clock(name, regimes):
    frame = regimes.iloc[:360 * 288 + 100]
    normal = make_strategy(name).prepare(frame.copy())
    local = make_strategy(name).prepare(frame.tz_convert("Europe/Prague"))
    naive = make_strategy(name).prepare(frame.tz_localize(None))
    for column in ("target", "r191_decision"):
        np.testing.assert_array_equal(normal[column], local[column])
        np.testing.assert_array_equal(normal[column], naive[column])
    assert (normal.target.iloc[-100:] == normal.target.iloc[-101]).all()
    assert not normal.r191_decision.iloc[-100:].any()


def test_missing_intraday_bar_excludes_whole_day_from_warmup_and_decisions(regimes):
    frame = regimes.iloc[:300 * 288].drop(regimes.index[100])
    prepared = make_strategy(PRIMARY_NAMES[0]).prepare(frame.copy())
    # The first day has a 23:55 close but only287 bars. Its presence must not
    # count as a complete day or advance the common warmup by one day.
    closes = prepared.loc[(prepared.index.hour == 23) & (prepared.index.minute == 55)]
    assert not closes.r191_decision.iloc[:253].any()
    assert closes.r191_decision.iloc[253]
    frame = regimes.iloc[:300 * 288].drop(regimes.index[270 * 288 + 100])
    prepared = make_strategy(PRIMARY_NAMES[0]).prepare(frame.copy())
    bad_day = prepared.index.normalize() == regimes.index[270 * 288].normalize()
    assert not prepared.r191_decision[bad_day].any()
    assert (prepared.target[bad_day] == prepared.target.loc[regimes.index[270 * 288 - 1]]).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_actual_account_drift_fresh_replay_and_zero_target_exit(name):
    frame = make_ohlcv([100.] * 290)
    frame["target"] = .7
    frame["r191_decision"] = False
    frame.iloc[287, frame.columns.get_loc("r191_decision")] = True
    broker = PaperBroker(MarketSpec.futures(leverage=5), 10_000)
    strategy = make_strategy(name)

    def orders(i=287):
        ctx = Context(frame, i, broker)
        strategy.on_bar(ctx)
        return ctx.orders

    assert orders(286) == []
    assert orders()[0].target == pytest.approx(.7 / 5)
    assert make_strategy(name).__dict__ == strategy.__dict__
    broker.entry, broker.pos = 100., 68.
    assert orders() == []
    broker.pos = 60.
    assert orders()[0].target == pytest.approx(.7 / 5)
    broker.pos, broker.cash = 70., 5_000.
    assert len(orders()) == 1  # Same base position, changed actual equity.
    frame["target"] = 0.
    broker.pos = .001
    broker.execute(orders()[0], frame.index[288], 100.)
    assert broker.pos == 0. and orders() == []
    broker.cash = 0.
    assert orders() == []


@pytest.mark.parametrize("market", [MarketSpec.spot(), MarketSpec.futures(leverage=5)])
def test_next_open_fills_one_times_limit_and_native_deadband(market):
    strategy = make_strategy(PRIMARY_NAMES[0])
    strategy.warmup = 0
    frame = make_ohlcv([100.] * 580)
    frame["target"] = 1.
    frame["r191_decision"] = (frame.index.hour == 23) & (frame.index.minute == 55)
    frame.iloc[288, frame.columns.get_loc("open")] = 110.
    frame.iloc[288, frame.columns.get_loc("high")] = 110.1
    strategy.prepare = lambda df: df
    result = run_backtest(strategy, frame, market, 1_000, slippage_bps=1.)
    assert result.fills[0].ts == frame.index[288]
    assert result.fills[0].price == pytest.approx(110. * 1.0001)
    assert result.fills[0].qty * 110. <= 1_000
    assert all(fill.ts.hour == 0 and fill.ts.minute == 0 for fill in result.fills)
    broker = PaperBroker(market, 10_000)
    broker.entry, broker.pos = 100., 60.
    frame["target"] = .7
    ctx = Context(frame, 287, broker)
    strategy.on_bar(ctx)
    assert len(ctx.orders) == 1
    fills = broker.execute(ctx.orders[0], frame.index[288], 100.)
    assert bool(fills) == (market.leverage == 1)


@pytest.mark.parametrize("mean,variance,previous", [
    (.005, .002, .4), (-.005, .002, .4), (.0002, .002, .4),
    (.01, 0., .4), (-.01, 0., .4), (0., 0., .4),
])
def test_proximal_solution_matches_exhaustive_objective(mean, variance, previous):
    chosen = _proximal_weight(mean, variance, previous, 2.)
    grid = np.r_[np.linspace(0., 1., 20_001), previous, chosen]
    objective = 7 * mean * grid - .5 * 4 * 7 * variance * grid ** 2 - 2 * SIGNAL_COST * abs(grid - previous)
    actual = 7 * mean * chosen - .5 * 4 * 7 * variance * chosen ** 2 - 2 * SIGNAL_COST * abs(chosen - previous)
    assert actual >= objective.max() - 1e-12


def test_quantile_scores_previous_forecast_and_includes_only_resolved_losses():
    losses = np.r_[np.nan, np.full(252, .01), .5, .0]
    quantile, alpha = _adaptive_quantiles(losses)
    assert np.isnan(quantile[:252]).all()
    assert quantile[252] == pytest.approx(.01) and alpha[252] == pytest.approx(.1)
    assert alpha[253] == pytest.approx(.091)  # Surprise compared with prior .01.
    assert alpha[254] == pytest.approx(.092)
    assert quantile[253] == pytest.approx(np.quantile(losses[2:254], 1 - .091))
    q2, a2 = _adaptive_quantiles(losses[:254])
    np.testing.assert_array_equal(q2, quantile[:254])
    np.testing.assert_array_equal(a2, alpha[:254])
    rising = pd.Series(np.exp(np.arange(300) * .01))
    target = _daily_targets(rising, PARAMS[PRIMARY_NAMES[1]])
    assert (target[252:] == 1.).all()  # q=0 is uncapped, not a divide-by-zero.


def test_olmar_cash_simplex_update_and_cost_gate():
    assert _olmar_weight(1., .3) == .3
    assert _olmar_weight(1.005, .3) == .3
    assert _olmar_weight(1.02, 0.) == pytest.approx(.5)
    assert _olmar_weight(.98, .8) == 0.
    assert _olmar_weight(1.02, .9) == pytest.approx(.9)


def test_sticky_expert_does_not_reward_a_signal_formed_after_the_return():
    # Only full-long earns the positive return; then lagged full-long loses.
    close = pd.Series([100., 102., 50.])
    returns = close.pct_change(fill_method=None).to_numpy()
    target = _sticky_targets(close, returns, .01)
    np.testing.assert_array_equal(target, [0., 1., 0.])
    # The initial full-long reward is reduced by its opening fee. Without
    # that lagged fee, this just-over-switch-penalty gain would select long.
    close = pd.Series([100., 101.2])
    target = _sticky_targets(close, close.pct_change(fill_method=None).to_numpy(), .01)
    np.testing.assert_array_equal(target, [0., 0.])


def test_robust_growth_uses_block_median_cost_and_least_turnover_ties():
    assert _block_weight(np.zeros(84), .75, 2.) == .75
    assert _block_weight(np.full(84, .01), 0., 2.) == 1.
    assert _block_weight(np.full(84, -.01), 1., 2.) == 0.
    returns = np.r_[np.full(14, .3), np.full(70, -.01)]
    assert returns.mean() > 0
    assert _block_weight(returns, 0., 2.) == 0.
