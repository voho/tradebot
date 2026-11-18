"""Synthetic causality and mechanics for the three R-193 attention allocators."""

import numpy as np
import pandas as pd
import pytest

import experiments.r193_strategies as r193
from experiments.r193_strategies import (
    ALL_NAMES, FAMILIES, HORIZON, LABELS, PARAMS, PRIMARY_NAMES, WARMUP_DAYS,
    _attend, _encode, _head_outputs, _positional, _ridge, _train, _utility,
    _windows, make_strategy,
)
from tradebot.broker import MarketSpec, PaperBroker
from tradebot.registry import available_strategies
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


@pytest.fixture
def fast_es(monkeypatch):
    # Causality does not depend on optimiser length; keep the suite fast.
    monkeypatch.setattr(r193, "ES_STEPS", 3)
    monkeypatch.setattr(r193, "ES_PAIRS", 2)


def test_frozen_inventory():
    assert len(PRIMARY_NAMES) == 3 and len(ALL_NAMES) == 9 == len(set(ALL_NAMES))
    assert set(ALL_NAMES) == set(PARAMS) == set(LABELS)
    assert all(len(family) == 3 for family in FAMILIES.values())
    assert not set(ALL_NAMES) & set(available_strategies())
    with pytest.raises(ValueError, match="Unknown R-193"):
        make_strategy("unknown")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_configuration_is_causal_nonflat_and_actually_orders(name, regimes, fast_es):
    strategy = make_strategy(name)
    full = strategy.prepare(regimes.copy())
    assert np.isfinite(full.target).all() and full.target.between(0, 1).all()
    assert np.ptp(full.target.iloc[strategy.warmup:]) > .05
    assert (full.target.iloc[:WARMUP_DAYS * 288] == 0).all()
    assert not full.r193_decision.iloc[:WARMUP_DAYS * 288].any()
    # A completed close after every family's first fit, and a partial day
    # after the day-730 calendar-year refit.
    for cut in (560 * 288 - 1, 740 * 288 + 137):
        prefix_strategy = make_strategy(name)
        prefix = prefix_strategy.prepare(regimes.iloc[:cut + 1].copy())
        perturbed = regimes.copy()
        perturbed.iloc[cut + 1:, :4] *= 3.
        future_strategy = make_strategy(name)
        future = future_strategy.prepare(perturbed)
        for column in ("target", "r193_decision"):
            np.testing.assert_array_equal(full[column].iloc[:cut + 1], prefix[column])
            np.testing.assert_array_equal(full[column].iloc[:cut + 1], future[column].iloc[:cut + 1])
        times = np.flatnonzero(full.r193_decision.iloc[:cut + 1])
        orders = []
        for current_strategy, frame in ((strategy, full), (prefix_strategy, prefix),
                                        (future_strategy, future)):
            broker = PaperBroker(MarketSpec.spot(), 10_000)
            current = []
            for i in times:
                ctx = Context(frame, int(i), broker)
                current_strategy.on_bar(ctx)
                current.append([order.target for order in ctx.orders])
            orders.append(current)
        assert orders[0] == orders[1] == orders[2]
        assert any(orders[0]), "causality must exercise nonempty real orders"


def test_partial_day_and_timezone_do_not_change_the_clock(regimes):
    frame = regimes.iloc[:360 * 288 + 100]
    normal = make_strategy(PRIMARY_NAMES[0]).prepare(frame.copy())
    local = make_strategy(PRIMARY_NAMES[0]).prepare(frame.tz_convert("Europe/Prague"))
    for column in ("target", "r193_decision"):
        np.testing.assert_array_equal(normal[column], local[column])
    assert (normal.target.iloc[-100:] == normal.target.iloc[-101]).all()
    assert not normal.r193_decision.iloc[-100:].any()


def test_attention_is_a_softmax_over_negative_squared_distance():
    rng = np.random.default_rng(1)
    keys, query = rng.standard_normal((50, 4)), rng.standard_normal(4)
    expected = np.exp(-((keys - query) ** 2).sum(1) / (2 * .5 * 2))
    np.testing.assert_allclose(_attend(query, keys, .5), expected / expected.sum())
    np.testing.assert_allclose(_attend(query, np.ones((5, 4)), 1.), np.full(5, .2))


def test_attention_reads_only_matured_outcomes():
    rng = np.random.default_rng(2)
    x, outcome = rng.standard_normal((300, 7)), rng.standard_normal(300)
    t = 250
    poisoned = outcome.copy()
    poisoned[t - HORIZON + 1:] = 1e6  # outcomes not yet known at t
    heads = {"all": tuple(range(7))}
    np.testing.assert_array_equal(_head_outputs(x, outcome, 1., heads)[:t + 1],
                                  _head_outputs(x, poisoned, 1., heads)[:t + 1])
    assert np.isnan(_head_outputs(x, outcome, 1., heads)[:HORIZON + r193.MIN_KEYS - 1]).all()


def test_ridge_limits():
    rng = np.random.default_rng(3)
    inputs = rng.standard_normal((400, 3))
    y = 2. + inputs @ np.array([.5, -1., 0.]) + .01 * rng.standard_normal(400)
    intercept, mean, sd, beta = _ridge(inputs, y, 1e-9)
    np.testing.assert_allclose(intercept + ((inputs - mean) / sd) @ beta, y, atol=.05)
    _, _, _, shrunk = _ridge(inputs, y, 1.)
    np.testing.assert_allclose(shrunk, beta / 2, rtol=.1, atol=.01)


def _textbook_encode(theta, tokens):
    """Full projected keys/values, the form in Vaswani et al. (2017)."""
    p = r193._unpack(theta)
    batch, length, _ = tokens.shape
    dk = r193.D_MODEL // r193.N_HEADS
    e = tokens @ p["we"] + _positional(length)
    q = (e[:, -1] @ p["wq"]).reshape(batch, r193.N_HEADS, dk)
    k = (e @ p["wk"]).reshape(batch, length, r193.N_HEADS, dk)
    v = (e @ p["wv"]).reshape(batch, length, r193.N_HEADS, dk)
    scores = np.einsum("bhd,blhd->bhl", q, k) / np.sqrt(dk)
    a = np.exp(scores - scores.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    out = np.einsum("bhl,blhd->bhd", a, v).reshape(batch, r193.D_MODEL) @ p["wo"]
    z = r193._layer_norm(e[:, -1] + out)
    z = r193._layer_norm(z + np.maximum(z @ p["w1"] + p["b1"], 0) @ p["w2"] + p["b2"])
    return 1 / (1 + np.exp(-(z @ p["v"] + p["c"])))


def test_folded_encoder_equals_textbook_attention():
    rng = np.random.default_rng(4)
    theta = .5 * rng.standard_normal(r193.N_PARAMS)
    tokens = rng.standard_normal((64, 30, 7))
    np.testing.assert_allclose(_encode(theta, tokens), _textbook_encode(theta, tokens), atol=1e-13)


def test_positional_encoding_and_windows():
    pe = _positional(10)
    assert pe[0, 0] == 0 and pe[0, 1] == 1 and pe[3, 2] == pytest.approx(np.sin(3 / 10000 ** .25))
    x = np.arange(20, dtype=float).reshape(10, 2)
    x[4] = np.nan
    seq, complete = _windows(x, 3)
    np.testing.assert_array_equal(seq[9], x[7:10])
    np.testing.assert_array_equal(complete, [0, 0, 1, 1, 0, 0, 0, 1, 1, 1])


def test_es_learns_a_planted_signal_net_of_cost():
    rng = np.random.default_rng(5)
    tokens = rng.standard_normal((400, 5, 7))
    # Next-day return is +3% when the last token's first feature is positive.
    next_return = np.where(tokens[:, -1, 0] > 0, .03, -.03)
    theta = _train(tokens, next_return, seed=1)
    w = _encode(theta, tokens)
    assert w[next_return > 0].mean() - w[next_return < 0].mean() > .5
    assert _utility(theta, tokens, next_return) > _utility(np.zeros_like(theta), tokens, next_return)


def test_actual_account_drift_and_zero_target_exit():
    frame = make_ohlcv([100.] * 290)
    frame["target"] = .7
    frame["r193_decision"] = False
    frame.iloc[287, frame.columns.get_loc("r193_decision")] = True
    broker = PaperBroker(MarketSpec.futures(leverage=5), 10_000)
    strategy = make_strategy(PRIMARY_NAMES[2])

    def orders(i=287):
        ctx = Context(frame, i, broker)
        strategy.on_bar(ctx)
        return ctx.orders

    assert orders(286) == []
    assert orders()[0].target == pytest.approx(.7 / 5)
    broker.entry, broker.pos = 100., 68.
    assert orders() == []
    frame["target"] = 0.
    broker.pos = .001
    broker.execute(orders()[0], frame.index[288], 100.)
    assert broker.pos == 0. and orders() == []


# --- strict causality at fit days, and pinned arithmetic (added after the
# independent pre-holdout review: the end-to-end test above compares clipped
# targets at two cuts and missed deliberately injected fit-day leaks).

def _daily_walk(n=820, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2017-03-01", periods=n, freq="D", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(.001, .03, n))), index=index)


def _fit_cuts(close, first_fit):
    refits = np.flatnonzero(r193._refit_days(close.index))
    cuts = {first_fit - 1, first_fit, first_fit + 1}
    cuts |= {d + k for d in refits if d > first_fit for k in (-1, 0, 1)}
    return sorted(c for c in cuts if c < len(close) - 1)


def _assert_prefix_equal(fn, close, cuts):
    full = fn(close)
    for cut in cuts:
        np.testing.assert_array_equal(fn(close.iloc[:cut + 1]), full[:cut + 1], err_msg=f"cut {cut}")


@pytest.mark.parametrize("family", ["r193_attention", "r193_multihead"])
def test_drift_forecast_is_causal_at_every_fit_day(family, monkeypatch):
    # Capture the forecast u before proximal sizing absorbs small differences.
    monkeypatch.setattr(r193, "_proximal_path", lambda u, sigma: u)
    close = _daily_walk()
    x, _, outcome = r193._features(close)
    if family == "r193_attention":
        fn = lambda c: r193._attention_targets(c, 1.)  # noqa: E731
        first = int(np.argmax([np.isfinite(x[:max(t - HORIZON + 1, 0)]).all(1).sum() >= r193.MIN_KEYS
                               for t in range(len(x))]))
    else:
        fn = lambda c: r193._multihead_targets(c, 1.)  # noqa: E731
        heads = _head_outputs(x, outcome, 1., r193.HEADS)
        usable = np.isfinite(heads).all(1) & np.isfinite(outcome)
        first = int(np.argmax([usable[:max(t - HORIZON + 1, 0)].sum() >= r193.MIN_FIT
                               for t in range(len(x))]))
    assert np.isfinite(fn(close)[first + 1])
    _assert_prefix_equal(fn, close, _fit_cuts(close, first))


def test_transformer_is_causal_at_every_fit_day(fast_es):
    close = _daily_walk()
    x, _, _ = r193._features(close)
    _, complete = _windows(x, 30)
    next_return = close.pct_change(fill_method=None).shift(-1).to_numpy()
    rows = np.cumsum(complete & np.isfinite(next_return))
    first = int(np.argmax(np.r_[0, rows[:-1]] >= r193.MIN_FIT))  # rows strictly before t
    fn = lambda c: r193._transformer_targets(c, 30)  # noqa: E731
    assert fn(close)[first] > 0 and fn(close)[first - 1] == 0
    _assert_prefix_equal(fn, close, _fit_cuts(close, first))


def test_features_outcome_and_drift_pinned():
    close = _daily_walk(400)
    x, sigma, outcome = r193._features(close)
    lc = np.log(close.to_numpy())
    lr = np.diff(lc, prepend=np.nan)
    t, j = 300, 250
    s60 = np.std(lr[t - 59:t + 1], ddof=1)
    assert sigma[t] == pytest.approx(s60)
    for col, k in enumerate((1, 5, 20, 60, 120)):
        assert x[t, col] == pytest.approx((lc[t] - lc[t - k]) / (s60 * np.sqrt(k)))
    assert x[t, 5] == pytest.approx(np.log(np.std(lr[t - 19:t + 1], ddof=1) / s60))
    assert x[t, 6] == pytest.approx(np.log(s60 / np.std(lr[t - 119:t + 1], ddof=1)))
    assert outcome[j] == pytest.approx((lc[j + 20] - lc[j]) / (sigma[j] * np.sqrt(20)))
    assert np.isnan(outcome[-HORIZON:]).all()
    assert r193._drift(1., .04) == pytest.approx(.04 / np.sqrt(20) + .0008)
    assert r193.PROXIMAL_PENALTY == 2.


def test_transformer_utility_and_frozen_hyperparameters_pinned():
    assert (r193.ES_PAIRS, r193.ES_SIGMA, r193.ES_STEPS, r193.ES_LR, r193.ES_INIT,
            r193.WEIGHT_DECAY, r193.D_MODEL, r193.N_HEADS, r193.D_FF, r193.N_PARAMS) == \
        (24, .05, 200, .02, .1, 1e-3, 8, 2, 16, 601)
    rng = np.random.default_rng(6)
    theta, tokens = rng.standard_normal(r193.N_PARAMS), rng.standard_normal((50, 10, 7))
    ret = rng.normal(0, .03, 50)
    w = _encode(theta, tokens)
    turnover = np.r_[0., np.abs(np.diff(w))]
    expected = np.mean(np.log1p(w * ret) - .0041 * turnover) - 1e-3 * np.mean(theta ** 2)
    assert _utility(theta, tokens, ret) == pytest.approx(expected)


def test_transformer_seed_follows_refit_year(monkeypatch):
    seeds = []
    monkeypatch.setattr(r193, "_train", lambda tokens, ret, seed: seeds.append(seed) or np.zeros(r193.N_PARAMS))
    r193._transformer_targets(_daily_walk(), 30)
    assert seeds == [193 + 2018, 193 + 2019]
