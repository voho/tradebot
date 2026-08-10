"""The no-lookahead invariant.

Every other result in this repository is worthless if a feature can see the
future, so this is tested directly rather than by inspection: build features on
a series, then *change the future* and assert that nothing in the past moved.

This catches the entire class of bugs that centred moving averages, full-sample
standardisation, and `shift(-1)` typos belong to.
"""

from __future__ import annotations

import numpy as np
import pytest

import gtbot.features as F
from gtbot.data.schema import validate
from gtbot.data.synthetic import simulate
from gtbot.features import rolling
from gtbot.game.experts import default_experts, signal_matrix


@pytest.fixture(scope="module")
def bars():
    return validate(simulate(6000, seed=3).bars)


def test_features_ignore_the_future(bars):
    """Perturbing bars after ``cut`` must not change any feature before it."""
    cut = 4000
    base = F.build(bars)

    tampered = bars.copy()
    idx = tampered.index[cut:]
    tampered.loc[idx, "close"] *= 1.25
    tampered.loc[idx, "high"] *= 1.30
    tampered.loc[idx, "low"] *= 1.20
    tampered.loc[idx, "open"] *= 1.25
    tampered.loc[idx, "volume"] *= 3.0
    tampered.loc[idx, "taker_buy_base"] *= 3.0
    tampered.loc[idx, "quote_volume"] *= 3.0
    after = F.build(tampered)

    for name in base.names:
        a = base[name][:cut]
        b = after[name][:cut]
        assert np.allclose(a, b, equal_nan=True), f"feature '{name}' leaks future information"


def test_expert_signals_ignore_the_future(bars):
    cut = 4000
    experts = default_experts()
    base = signal_matrix(experts, F.build(bars))

    tampered = bars.copy()
    idx = tampered.index[cut:]
    for col in ("open", "high", "low", "close"):
        tampered.loc[idx, col] *= 0.8
    after = signal_matrix(experts, F.build(tampered))

    for j, expert in enumerate(experts):
        assert np.allclose(base[:cut, j], after[:cut, j]), f"expert '{expert.name}' leaks"


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: rolling.ewma(x, 10.0),
        lambda x: rolling.rolling_mean(x, 25),
        lambda x: rolling.rolling_std(x, 25),
        lambda x: rolling.rolling_max(x, 25),
        lambda x: rolling.rolling_min(x, 25),
        lambda x: rolling.rolling_rank(x, 50),
        lambda x: rolling.zscore(x, 50),
    ],
)
def test_rolling_primitives_are_causal(fn):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = x.copy()
    y[300:] += 10.0
    assert np.allclose(fn(x)[:300], fn(y)[:300], equal_nan=True)


def test_shift_refuses_to_look_forward():
    with pytest.raises(ValueError):
        rolling.shift(np.arange(10.0), -1)


def test_rolling_max_matches_bruteforce():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    w = 17
    got = rolling.rolling_max(x, w)
    for i in range(300):
        assert got[i] == pytest.approx(x[max(0, i - w + 1) : i + 1].max())
