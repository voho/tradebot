"""R-189's executable reward clock, ten distinct learners and UTC cadence."""

import numpy as np
import pandas as pd
import pytest

from tradebot.broker import MarketSpec, PaperBroker
from tradebot.registry import available_strategies
from tradebot.strategies.intraday_games import (
    BlackwellCouncil, CautiousOptimism, DefensiveForecast, MinimaxCouncil,
    NashCouncil, NormalHedgeCouncil, QRECouncil, SleepingCouncil,
    SquintCouncil, SwapRegretCouncil, _IntradayCouncil,
)
from tradebot.strategy import Context

from conftest import make_ohlcv


CANDIDATES = (CautiousOptimism, SquintCouncil, NormalHedgeCouncil,
              SwapRegretCouncil, BlackwellCouncil, MinimaxCouncil,
              NashCouncil, QRECouncil, SleepingCouncil, DefensiveForecast)


@pytest.fixture(scope="module")
def regimes():
    t = np.arange(27_000)
    rng = np.random.default_rng(29)
    returns = (0.0001 * np.sin(t / 900) + 0.0002 * np.cos(t / 70)
               + rng.normal(0, 0.001, len(t)))
    return make_ohlcv(100 * np.exp(np.cumsum(returns)))


@pytest.mark.parametrize("cls", CANDIDATES, ids=lambda cls: cls.name)
def test_boundary_future_mutation_and_reset_are_causal(cls, regimes):
    # This is a real decision boundary AFTER Kelly's warmup. In particular,
    # changing the next entry open must not change the current decision.
    cut = 24_000
    assert regimes.index[cut].hour % 4 == regimes.index[cut].minute == 0
    strategy = cls()
    original = strategy.prepare(regimes.copy())
    modified = regimes.copy()
    modified.iloc[cut + 1:, :4] *= 3
    after = strategy.prepare(modified)
    prefix = cls().prepare(regimes.iloc[:cut + 1].copy())
    for column in ("target", "game_decision"):
        np.testing.assert_array_equal(original[column].iloc[:cut + 1],
                                      after[column].iloc[:cut + 1])
        np.testing.assert_array_equal(original[column].iloc[:cut + 1], prefix[column])
    target = original.target.to_numpy()
    assert np.isfinite(target).all() and ((target >= 0) & (target <= 1)).all()
    assert np.ptp(target[cls.warmup:]) > 0.05  # Avoid vacuous flat-signal checks.
    changes = np.flatnonzero(np.diff(target) != 0) + 1
    assert original.game_decision.iloc[changes].all()


def test_all_ten_are_registered_and_produce_distinct_paths(regimes):
    registered = available_strategies()
    paths = []
    for cls in CANDIDATES:
        assert registered[cls.name] is cls
        path = cls().prepare(regimes.copy()).target.iloc[cls.warmup:].to_numpy()
        assert all(not np.allclose(path, previous) for previous in paths)
        paths.append(path)


def test_squint_variance_and_normalhedge_time_satisfy_implicit_equations():
    squint, normal = SquintCouncil(), NormalHedgeCouncil()
    squint._reset()
    normal._reset()
    awake = np.ones(6, dtype=bool)
    for gain in np.random.default_rng(19).uniform(-1, 1, size=(24, 6)):
        previous_variance = squint.variance
        centered = (gain - squint.weights @ gain) / 2
        squint._update(gain, squint.weights.copy(), awake)
        increment = squint.variance - previous_variance
        q = squint._integrals(squint.variance, derivative=True)
        assert 0 <= increment <= 1
        assert abs(q @ (increment - centered ** 2)) < 2e-7
        previous_time = normal.time
        normal._update(gain, normal.weights.copy(), awake)
        positive = np.maximum(normal.regret, 0)
        potential = np.exp(positive ** 2 / (2 * normal.time)).sum() / np.sqrt(normal.time)
        assert normal.time >= previous_time
        assert potential == pytest.approx(6.0, rel=1e-7)
        for strategy in (squint, normal):
            assert np.isfinite(strategy.weights).all()
            assert strategy.weights.min() >= 0
            assert strategy.weights.sum() == pytest.approx(1)


def test_swap_regret_allocation_is_stationary_for_its_transition_game():
    strategy = SwapRegretCouncil()
    strategy._reset()
    for gain in np.random.default_rng(31).uniform(-1, 1, size=(24, 6)):
        strategy._update(gain, strategy.weights.copy(), np.ones(6, dtype=bool))
        rates = np.maximum(strategy.swap_regret, 0)
        np.fill_diagonal(rates, 0)
        transition = rates / max(rates.sum(axis=1).max(), 1)
        np.fill_diagonal(transition, 1 - transition.sum(axis=1))
        transition = (1 - 1e-6) * transition + 1e-6 / 6
        np.testing.assert_allclose(strategy.weights @ transition, strategy.weights, atol=1e-12)
        assert strategy.weights.min() >= 0
        assert strategy.weights.sum() == pytest.approx(1)


class _RewardProbe(_IntradayCouncil):
    """Controlled experts make entry timing and turnover observable."""

    def _reset(self):
        super()._reset()
        self.weights = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        self.observed = []

    @staticmethod
    def _experts(df):
        a = np.zeros((len(df), 6))
        a[:, 0] = 0.4
        a[336:, 0] = 0.7
        return a, np.ones_like(a, dtype=bool)

    def _update(self, gain, played, awake):
        self.observed.append(gain.copy())


def test_rewards_begin_after_next_open_and_charge_expert_turnover():
    df = make_ohlcv(np.full(400, 100.0))
    # First decision is row288; open289 is executable, close288 is not.
    df.iloc[289, df.columns.get_loc("open")] = 120.0
    df.iloc[336, df.columns.get_loc("close")] = 121.2
    df.iloc[337, df.columns.get_loc("open")] = 100.0
    df.iloc[384, df.columns.get_loc("close")] = 102.0
    strategy = _RewardProbe()
    strategy.prepare(df.iloc[:336].copy())
    assert not strategy.observed  # The first completed round has not arrived.
    strategy.prepare(df.iloc[:337].copy())
    assert len(strategy.observed) == 1
    assert strategy.observed[0][0] == pytest.approx((0.4 * 0.01 - 0.0011 * 0.4) / 0.02)
    strategy.prepare(df.copy())
    assert len(strategy.observed) == 2
    assert strategy.observed[1][0] == pytest.approx((0.7 * 0.02 - 0.0011 * 0.3) / 0.02)
    assert np.count_nonzero(strategy.observed) == 2


def test_utc_slots_survive_frame_offset_and_timezone(regimes):
    # Price/history signals may depend on retained history, the UTC clock cannot.
    full = CautiousOptimism().prepare(regimes.iloc[:1000].copy())
    offset = CautiousOptimism().prepare(regimes.iloc[13:1000].copy())
    local = regimes.iloc[:1000].tz_convert("Europe/Prague")
    local = CautiousOptimism().prepare(local)
    np.testing.assert_array_equal(full.game_decision.iloc[13:], offset.game_decision)
    np.testing.assert_array_equal(full.game_decision, local.game_decision)
    assert full.game_decision.groupby(full.index.date).sum().max() <= 6


@pytest.mark.parametrize("cls", CANDIDATES, ids=lambda cls: cls.name)
def test_on_bar_fresh_account_enters_held_signal_only_at_slot(cls):
    # A funded subperiod may begin inside an already-positive prepared target.
    df = make_ohlcv([100.0] * 50)
    df["target"] = 0.6
    df["game_decision"] = (df.index.hour % 4 == 0) & (df.index.minute == 0)
    broker = PaperBroker(MarketSpec.futures(leverage=5), start_balance=10_000)
    strategy = cls()
    scheduled = Context(df, 48, broker)
    strategy.on_bar(scheduled)
    assert len(scheduled.orders) == 1
    assert scheduled.orders[0].target == pytest.approx(0.6 / 5)
    nonscheduled = Context(df, 47, broker)
    strategy.on_bar(nonscheduled)
    assert nonscheduled.orders == []
    # Actual holdings inside the band should suppress unneeded rebalances.
    broker.pos = 59.0
    broker.entry = 100.0
    already_held = Context(df, 48, broker)
    strategy.on_bar(already_held)
    assert already_held.orders == []
