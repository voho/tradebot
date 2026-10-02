"""Synthetic identity, causality and execution tests for every frozen R-192 update."""

import numpy as np
import pandas as pd
import pytest

from experiments.r192_strategies import (
    ALL_NAMES, FAMILIES, LABELS, PARAMS, PRIMARY_NAMES, _confirmed_anchor,
    _confirm_increases, _factor_events, _latch, _parent_features,
    _tracking_budget, make_strategy,
)
from tradebot.broker import MarketSpec, PaperBroker
from tradebot.engine import run_backtest
from tradebot.registry import available_strategies
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context

from conftest import make_ohlcv


@pytest.fixture(scope="module")
def regimes():
    t = np.arange(54_000)
    drift = np.select([t < 28_000, t < 41_000], [.000045, -.000085], default=.00006)
    noise = np.random.default_rng(192).normal(0, .0008, len(t))
    noise[32_000:32_050] *= 25
    noise[46_000], noise[46_001] = -.2, .2
    return make_ohlcv(100 * np.exp(np.cumsum(drift + noise)))


def test_frozen_inventory_is_complete_and_unregistered():
    assert len(PRIMARY_NAMES) == 5 and len(ALL_NAMES) == len(set(ALL_NAMES)) == 15
    assert set(ALL_NAMES) == set(PARAMS) == set(LABELS)
    assert all(len(family) == 3 for family in FAMILIES.values())
    assert not set(ALL_NAMES) & set(available_strategies())
    assert make_strategy(PRIMARY_NAMES[0]).warmup == KellyRegimeV4.warmup
    with pytest.raises(ValueError, match="Unknown R-192"):
        make_strategy("unknown")
    with pytest.raises(ValueError, match="declared parameter"):
        make_strategy(PRIMARY_NAMES[0], horizons=(10, 20, 40))


def test_shared_reconstruction_matches_native_parent_bit_for_bit(regimes):
    features = _parent_features(regimes)
    target = _latch(features["desired"])
    native = KellyRegimeV4().prepare(regimes.copy()).target.to_numpy()
    np.testing.assert_array_equal(target, native)
    assert np.ptp(native[KellyRegimeV4.warmup:]) > .5


@pytest.mark.parametrize("name,override", [
    ("r192_tracking_budget", {"budget": 0.}),
    ("r192_factor_clock", {"period_hours": 0}),
    ("r192_anchor_confirm", {"confirmation": 0.}),
    ("r192_robust_state", {"clip_sigma": np.inf}),
    ("r192_confirm_increase", {"period_bars": 1}),
])
def test_disabled_mechanism_matches_native_targets_and_orders(name, override, regimes):
    native_strategy = KellyRegimeV4()
    candidate_strategy = make_strategy(name, **override)
    native = native_strategy.prepare(regimes.copy())
    candidate = candidate_strategy.prepare(regimes.copy())
    np.testing.assert_array_equal(candidate.target, native.target)
    times = np.flatnonzero(native.target.diff().fillna(0).to_numpy())
    times = times[times >= native_strategy.warmup]
    assert len(times) > 0
    for i in times:
        broker = PaperBroker(MarketSpec.futures(leverage=5), 10_000)
        a, b = Context(native, int(i), broker), Context(candidate, int(i), broker)
        native_strategy.on_bar(a)
        candidate_strategy.on_bar(b)
        assert [order.target for order in a.orders] == [order.target for order in b.orders]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_configuration_is_causal_bounded_nonflat_and_orders(name, regimes):
    strategy = make_strategy(name)
    original_attributes = dict(strategy.__dict__)
    full = strategy.prepare(regimes.copy())
    assert strategy.__dict__ == original_attributes, "prepare must not stash future state"
    assert np.isfinite(full.target).all() and full.target.between(0, 2).all()
    assert np.ptp(full.target.iloc[strategy.warmup:]) > .1
    for cut in (40_000, 49_999):
        short_strategy = make_strategy(name)
        short = short_strategy.prepare(regimes.iloc[:cut + 1].copy())
        modified = regimes.copy()
        modified.iloc[cut + 1:, :4] *= 3.
        modified.iloc[cut + 1:, 4] *= 7.
        future_strategy = make_strategy(name)
        future = future_strategy.prepare(modified)
        np.testing.assert_array_equal(full.target.iloc[:cut + 1], short.target)
        np.testing.assert_array_equal(full.target.iloc[:cut + 1], future.target.iloc[:cut + 1])
        times = np.flatnonzero(full.target.iloc[:cut + 1].diff().fillna(0).to_numpy())
        times = times[times >= strategy.warmup]
        assert len(times), "actual post-warmup decisions must be exercised"
        times = np.unique(np.r_[times[-40:], cut - 1, cut])
        outcomes = []
        for current_strategy, frame in ((strategy, full), (short_strategy, short),
                                        (future_strategy, future)):
            broker = PaperBroker(MarketSpec.futures(leverage=5), 10_000)
            orders = []
            for i in times:
                ctx = Context(frame, int(i), broker)
                current_strategy.on_bar(ctx)
                orders.append([order.target for order in ctx.orders])
            outcomes.append(orders)
        assert outcomes[0] == outcomes[1] == outcomes[2]
        assert any(outcomes[0])


def test_tracking_budget_accumulates_signed_error_and_has_no_deadband_or_exit_override():
    desired = np.array([.5, .5, 1., 0., 1., 1.])
    target = _tracking_budget(desired, 2 * .5 ** 2 / 288)
    np.testing.assert_array_equal(target, [0., .5, .5, .5, .5, 1.])
    # Zero gap cancels accumulated credit. Returning to .5 needs two fresh bars.
    target = _tracking_budget(np.array([.5, 0., .5, .5]), 2 * .5 ** 2 / 288)
    np.testing.assert_array_equal(target, [0., 0., 0., .5])
    # No ten-percent latch for positive budgets: a five-percent gap can act.
    target = _tracking_budget(np.array([.05, .05, 0.]), .05 ** 2 / 288)
    np.testing.assert_array_equal(target, [.05, .05, 0.])
    target = _tracking_budget(np.array([1., 0.]), 1.5 / 288)
    np.testing.assert_array_equal(target, [0., 0.])
    target = _tracking_budget(np.array([1., 1., 0.]), 1.5 / 288)
    np.testing.assert_array_equal(target, [0., 1., 1.])  # No privileged full exit.


@pytest.mark.parametrize("hours", [12, 24, 48])
def test_factor_clock_is_epoch_aligned_timezone_invariant_and_honors_vote_events(hours):
    index = pd.date_range("2025-01-01", periods=900, freq="5min", tz="UTC")
    frac = np.zeros(len(index))
    frac[13:300] = 1 / 3
    events = _factor_events(index, frac, hours)
    expected_grid = (index - pd.Timestamp("1970-01-01", tz="UTC")) % pd.Timedelta(hours=hours) == pd.Timedelta(0)
    assert events[13] and events[300]
    np.testing.assert_array_equal(events, expected_grid | np.r_[False, np.diff(frac) != 0])
    np.testing.assert_array_equal(events, _factor_events(index.tz_convert("Europe/Prague"), frac, hours))
    np.testing.assert_array_equal(events, _factor_events(index.tz_localize(None), frac, hours))
    np.testing.assert_array_equal(events[101:], _factor_events(index[100:], frac[100:], hours)[1:])
    desired = np.array([.5, .9, .9, 0.])
    np.testing.assert_array_equal(_latch(desired, np.array([True, False, True, False])),
                                  [.5, .5, .9, .9])


def test_anchor_confirmation_requires_consecutive_excursion_and_resets():
    anchor = np.full(9, 100.)
    close = np.array([102., 100., 102., 102., 98., 100., 98., 98., 102.])
    target = _confirmed_anchor(close, anchor, .015 / 288)
    np.testing.assert_array_equal(target, [0., 0., 0., 1., 1., 1., 1., 0., 0.])
    # Exactly on either one-percent boundary is inside the no-change band.
    target = _confirmed_anchor(np.array([101., 102., 99., 98.]), np.full(4, 100.), 0.)
    np.testing.assert_array_equal(target, [0., 1., 1., 0.])
    target = _confirmed_anchor(np.array([200., 200.]), np.array([np.nan, 100.]), 0.)
    np.testing.assert_array_equal(target, [0., 1.])


def test_robust_state_changes_classifier_without_changing_vote_or_raw_risk_budget(regimes):
    native = _parent_features(regimes)
    robust = _parent_features(regimes, clip_sigma=6.)
    for field in ("frac", "vol", "slow", "full", "steady"):
        np.testing.assert_array_equal(native[field], robust[field])
    assert (native["state"][KellyRegimeV4.warmup:] != robust["state"][KellyRegimeV4.warmup:]).any()
    assert not np.array_equal(_latch(native["desired"]), _latch(robust["desired"]))
    frame = make_ohlcv(np.r_[np.full(1000, 100.), np.full(1000, 120.)])
    robust = _parent_features(frame, clip_sigma=6.)
    assert np.isfinite(robust["desired"]).all()
    assert robust["vol"][1001] > 0


def test_increase_confirmation_uses_latest_target_and_never_delays_a_reduction():
    parent = np.array([.5, .6, .8, 1., 1.1, .4, .7, .4, .9, 1., 1.2, 0.])
    actual = _confirm_increases(parent, 3)
    expected = [0., 0., .8, .8, .8, .4, .4, .4, .4, .4, 1.2, 0.]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(_confirm_increases(parent, 1), parent)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_callback_is_native_target_change_only(name):
    frame = make_ohlcv([100.] * 5)
    frame["target"] = [.7, .7, .8, 0., 0.]
    broker = PaperBroker(MarketSpec.futures(leverage=5), 10_000)
    strategy = make_strategy(name)
    for i, expected in ((0, [.7 / 5]), (1, []), (2, [.8 / 5]), (3, [0.]), (4, [])):
        ctx = Context(frame, i, broker)
        strategy.on_bar(ctx)
        assert [order.target for order in ctx.orders] == expected
    broker.entry, broker.pos = 100., 100.
    ctx = Context(frame, 1, broker)
    strategy.on_bar(ctx)
    assert ctx.orders == []


@pytest.mark.parametrize("market", [MarketSpec.spot(), MarketSpec.futures(leverage=5)])
def test_next_open_fills_and_native_market_limits_remain_in_force(market):
    frame = make_ohlcv([100.] * 5)
    frame["target"] = [0., 2., 2., 0., 0.]
    frame.iloc[2, frame.columns.get_loc("open")] = 110.
    frame.iloc[2, frame.columns.get_loc("high")] = 110.1
    strategy = make_strategy(PRIMARY_NAMES[0])
    strategy.warmup = 0
    strategy.prepare = lambda df: df
    result = run_backtest(strategy, frame, market, 1_000, slippage_bps=1.)
    assert result.fills[0].ts == frame.index[2]
    assert result.fills[0].price == pytest.approx(110. * 1.0001)
    assert result.fills[0].qty * 110. <= 1_000 * min(2., market.leverage)
    assert result.fills[-1].ts == frame.index[4]
    broker = PaperBroker(market, 10_000)
    broker.entry, broker.pos = 100., 60.
    frame["target"] = [.5, .7, .7, .7, .7]
    ctx = Context(frame, 1, broker)
    strategy.on_bar(ctx)
    assert len(ctx.orders) == 1
    fills = broker.execute(ctx.orders[0], frame.index[2], 100.)
    assert bool(fills) == (market.leverage == 1)
