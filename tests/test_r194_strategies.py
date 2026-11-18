"""Synthetic identity, causality and mechanics for the R-194 Kelly x transformer hybrids."""

import numpy as np
import pandas as pd
import pytest

import experiments.r194_strategies as r194
from experiments.r193_strategies import HORIZON, MIN_KEYS
from experiments.r194_strategies import (ALL_NAMES, FAMILIES, LABELS, PARAMS, PRIMARY_NAMES,
                                         _complete_days, _encode, _panel_drift, _to_bars,
                                         make_strategy)
from tradebot.broker import MarketSpec, PaperBroker
from tradebot.registry import available_strategies
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context

from conftest import make_ohlcv

DAYS = 760


@pytest.fixture(scope="module")
def regimes():
    days = np.arange(DAYS)
    returns = np.select([days < 160, days < 225, days < 315, days < 400, days < 560],
                        [.008, -.012, .009, -.01, .006], default=-.004)
    returns += .002 * np.sin(days / 3) + .01 * np.sin(days / 17)
    returns[[290, 325, 401, 450, 610]] = -.15
    increments = np.repeat(np.log1p(returns) / 288, 288)
    return make_ohlcv(100 * np.exp(np.cumsum(increments)))


def _panel(frame):
    """Three synthetic instruments on the same daily clock as ``frame``."""
    close = _complete_days(frame)[1]
    rng = np.random.default_rng(11)
    out = [close]
    for k in range(2):
        noise = np.cumsum(rng.normal(0, .02, len(close)))
        out.append(pd.Series(close.to_numpy() * np.exp(noise), index=close.index))
    return out


@pytest.fixture
def fast_es(monkeypatch):
    monkeypatch.setattr(r194, "ES_STEPS", 3)
    monkeypatch.setattr(r194, "ES_PAIRS", 2)


def _make(name, frame):
    return make_strategy(name, panel=_panel(frame))


def test_frozen_inventory():
    assert len(PRIMARY_NAMES) == 3 and len(ALL_NAMES) == 9 == len(set(ALL_NAMES))
    assert set(ALL_NAMES) == set(PARAMS) == set(LABELS)
    assert all(len(f) == 3 for f in FAMILIES.values())
    assert not set(ALL_NAMES) & set(available_strategies())
    assert len(r194.PANEL_FILES) == 9
    with pytest.raises(ValueError, match="Unknown R-194"):
        make_strategy("unknown")


def test_each_component_switched_off_is_kelly_v4_bit_for_bit(regimes, monkeypatch):
    native = KellyRegimeV4().prepare(regimes.copy())["target"].to_numpy()
    monkeypatch.setitem(PARAMS[PRIMARY_NAMES[0]], "weight", 0.)
    off = _make(PRIMARY_NAMES[0], regimes).prepare(regimes.copy())["target"].to_numpy()
    np.testing.assert_array_equal(off, native)
    monkeypatch.setattr(r194, "_episodic_multiplier", lambda close, *a: np.ones(len(close)))
    off = _make(PRIMARY_NAMES[1], regimes).prepare(regimes.copy())["target"].to_numpy()
    np.testing.assert_array_equal(off, native)
    # The untrained residual model is exactly the identity.
    monkeypatch.setattr(r194, "_train", lambda tokens, base, ret, seed:
                        r194._initial_theta(np.random.default_rng(seed)))
    off = _make(PRIMARY_NAMES[2], regimes).prepare(regimes.copy())["target"].to_numpy()
    np.testing.assert_array_equal(off, native)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_configuration_is_causal_bounded_changes_v4_and_orders(name, regimes, fast_es):
    strategy = _make(name, regimes)
    full = strategy.prepare(regimes.copy())
    native = KellyRegimeV4().prepare(regimes.copy())["target"].to_numpy()
    assert np.isfinite(full.target).all() and full.target.between(0, r194.MAX_LEVERAGE).all()
    assert (full.target.to_numpy() != native).any(), "an inert hybrid tests nothing"
    for cut in (560 * 288 - 1, 740 * 288 + 137):
        prefix_frame = regimes.iloc[:cut + 1].copy()
        prefix_strategy = make_strategy(name, panel=[s.loc[:prefix_frame.index[-1]] for s in _panel(regimes)])
        prefix = prefix_strategy.prepare(prefix_frame)
        np.testing.assert_array_equal(full.target.iloc[:cut + 1], prefix.target)
        broker = PaperBroker(MarketSpec.spot(), 10_000)
        a, b = [], []
        changes = np.flatnonzero(np.diff(full.target.to_numpy()[:cut + 1]) != 0) + 1
        for i in changes[changes >= strategy.warmup]:
            for current, frame, sink in ((strategy, full, a), (prefix_strategy, prefix, b)):
                ctx = Context(frame, i, broker)
                current.on_bar(ctx)
                sink.append([o.target for o in ctx.orders])
        assert a == b and any(a)


def test_panel_memory_is_readable_only_after_each_assets_outcome_matures():
    rng = np.random.default_rng(3)
    days = pd.date_range("2020-01-01 23:55", periods=400, freq="D", tz="UTC")
    walk = lambda: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .03, 400))), index=days)  # noqa: E731
    query, other = walk(), walk()
    full = _panel_drift(query, [query, other])
    t = 350
    poisoned = other.copy()
    poisoned.iloc[t + 1:] *= 50.  # prices after t
    np.testing.assert_array_equal(full[:t + 1], _panel_drift(query, [query, poisoned])[:t + 1])
    # A shifted calendar: memories of a later-listed asset become readable later.
    late = other.iloc[200:]
    assert np.isfinite(_panel_drift(query, [late])).sum() < np.isfinite(_panel_drift(query, [other])).sum()
    assert np.isnan(_panel_drift(query, [other])[:120 + HORIZON + MIN_KEYS - 30]).all()


def test_daily_values_apply_from_their_own_close_bar():
    completed = np.array([0, 0, 1, 0, 0, 1, 0], dtype=bool)
    np.testing.assert_array_equal(_to_bars(np.array([.5, np.nan]), completed, 1.),
                                  [1, 1, .5, .5, .5, 1, 1])


def test_episodic_ratio_is_two_sided_and_one_without_information():
    rng = np.random.default_rng(4)
    days = pd.date_range("2020-01-01 23:55", periods=500, freq="D", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(.002, .03, 500))), index=days)
    frac, scale = rng.integers(0, 4, 500) / 3, rng.uniform(.5, 2, 500)
    m = r194._episodic_multiplier(close, frac, scale)
    assert (m[:HORIZON + MIN_KEYS] == 1).all()
    later = m[300:]
    assert later.min() < 1 < later.max() and later.min() >= 0 and later.max() <= 2


def test_residual_head_starts_at_identity_and_learns_a_planted_correction():
    rng = np.random.default_rng(5)
    theta = r194._initial_theta(rng)
    tokens = rng.standard_normal((400, 5, 9))
    np.testing.assert_array_equal(_encode(theta, tokens), 0.)
    base = np.full(400, .5)
    # Leaning in pays when feature 7 (v4's vote) is high, leaning out otherwise.
    next_return = np.where(tokens[:, -1, 7] > 0, .03, -.03)
    trained = r194._train(tokens, base, next_return, seed=1)
    z = _encode(trained, tokens)
    assert z[next_return > 0].mean() - z[next_return < 0].mean() > .5
    assert r194._utility(trained, tokens, base, next_return) > r194._utility(theta, tokens, base, next_return)


def _walk_with_state(n=820, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2017-03-01 23:55", periods=n, freq="D", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(.001, .03, n))), index=index)
    return close, rng.integers(0, 4, n) / 3, rng.uniform(.3, 2, n)


def test_component_outputs_are_causal_at_every_fit_day(fast_es):
    # Pre-latch outputs, so the 10% latch cannot hide a small leak.
    close, frac, scale = _walk_with_state()
    base = np.minimum(frac * scale, 2.)
    residual = lambda c: r194._residual_multiplier(c, frac[:len(c)], scale[:len(c)],  # noqa: E731
                                                   base[:len(c)], 30)
    # The episodic ratio is the identity unless unconditional Kelly is positive,
    # so it gets a positive-drift walk of its own and a non-vacuity check.
    up, up_frac, up_scale = _walk_with_state(seed=9)
    up = up * np.exp(.004 * np.arange(len(up)))
    episodic = lambda c: r194._episodic_multiplier(c, up_frac[:len(c)], up_scale[:len(c)])  # noqa: E731
    full_r, full_e = residual(close), episodic(up)
    assert (full_e != 1).sum() > 200
    first_fit = int(np.argmax(full_r != 1))
    refits = np.flatnonzero(r194._refit_days(close.index))
    cuts = {first_fit - 1, first_fit, first_fit + 1, int(np.argmax(full_e != 1))}
    cuts |= {d + k for d in refits if d > first_fit for k in (-1, 0, 1)}
    for cut in sorted(cuts):
        np.testing.assert_array_equal(residual(close.iloc[:cut + 1]), full_r[:cut + 1], err_msg=f"cut {cut}")
        np.testing.assert_array_equal(episodic(up.iloc[:cut + 1]), full_e[:cut + 1], err_msg=f"cut {cut}")


def test_residual_seed_and_frozen_hyperparameters(monkeypatch):
    assert (r194.ES_PAIRS, r194.ES_SIGMA, r194.ES_STEPS, r194.ES_LR, r194.ES_INIT,
            r194.WEIGHT_DECAY, r194.N_IN, r194.N_PARAMS) == (24, .05, 200, .02, .1, 1e-3, 9, 617)
    seeds = []
    monkeypatch.setattr(r194, "_train", lambda tokens, base, ret, seed:
                        seeds.append(seed) or r194._initial_theta(np.random.default_rng(seed)))
    close, frac, scale = _walk_with_state()
    r194._residual_multiplier(close, frac, scale, np.minimum(frac * scale, 2.), 30)
    assert seeds == [194 + 2018, 194 + 2019]


# --- added from the independent pre-holdout review: the latch and the
# 23:55 cut hid day-start and query-side leaks from the tests above.

@pytest.mark.parametrize("name", PRIMARY_NAMES)
def test_prelatch_desired_is_causal_mid_day_while_long(name, regimes, fast_es, monkeypatch):
    monkeypatch.setattr(r194, "_latch", lambda d: np.asarray(d, float))  # target == desired
    native = KellyRegimeV4().prepare(regimes.copy())["target"].to_numpy()
    full = make_strategy(name, panel=_panel(regimes)).prepare(regimes.copy())["target"].to_numpy()
    cuts = [d * 288 + k for d in (470, 500, 530) for k in (137, 286)]
    assert all(native[c] > 0 for c in cuts)
    for cut in cuts:
        frame = regimes.iloc[:cut + 1].copy()
        panel = [s.loc[:frame.index[-1]] for s in _panel(regimes)]
        prefix = make_strategy(name, panel=panel).prepare(frame)["target"].to_numpy()
        np.testing.assert_array_equal(prefix, full[:cut + 1], err_msg=f"cut {cut}")


def test_panel_drift_is_causal_on_the_query_side():
    rng = np.random.default_rng(3)
    days = pd.date_range("2020-01-01 23:55", periods=400, freq="D", tz="UTC")
    walk = lambda: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .03, 400))), index=days)  # noqa: E731
    query, other = walk(), walk()
    full = _panel_drift(query, [query, other])
    for t in (200, 260, 350):
        cut = [s.iloc[:t + 1] for s in (query, other)]
        np.testing.assert_array_equal(_panel_drift(cut[0], cut), full[:t + 1], err_msg=f"t {t}")
