"""Causality checks that reach inside ``on_bar``, and anti-vacuity guards.

The other causality tests inspect what ``prepare()`` computes. That leaves
a hole big enough to drive perfect foresight through: a strategy can keep
the frame handed to ``prepare()`` and index ``i + 1`` inside ``on_bar``.
Such a strategy passes truncation, passes the prepared-column perturbation
check, and passes live parity (because most strategies' warmup exceeds the
short synthetic parity fixture, so their ``on_bar`` is never called there).
An audit built exactly that strategy: it returned $3.7e23 from $1,000 at
Sharpe 73 with a fully green suite.

``test_decisions_ignore_every_bar_after_the_decision_bar`` closes it by
comparing the *orders* a strategy queues, not the columns it computes.

The rest of this module exists because a passing test that exercises
nothing is worse than no test: it reports coverage it does not have.
"""

from pathlib import Path

import pytest

from tradebot.broker import MarketSpec, PaperBroker
from tradebot.data import load_dataset
from tradebot.engine import run_backtest
from tradebot.registry import available_strategies, get_strategy
from tradebot.strategy import Context

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BARS = 60_000  # longer than every registered strategy's warmup


@pytest.fixture(scope="module")
def real_slice():
    df, label = load_dataset(DATA_DIR, "spot")
    # fail, never skip: a skipped causality suite is a green run that
    # guarantees nothing, and load_dataset writes a synthetic CSV into
    # data/ as a side effect, so the skip is self-concealing.
    assert label != "SYNTHETIC", (
        "the committed real dataset is missing - every causality guarantee "
        "in this suite would be silently skipped")
    assert len(df) >= BARS, f"dataset too short: {len(df)} < {BARS}"
    return df.iloc[-BARS:]


def _decisions(strategy, df, bars, market):
    """The orders ``strategy`` queues at each bar in ``bars``.

    ``prepare`` is called once, as the engine does, so a strategy that
    stashes the frame there and reads ahead in ``on_bar`` is exercised the
    same way it would be in a real run.
    """
    prepared = strategy.prepare(df.copy())
    broker = PaperBroker(market=market, start_balance=10_000.0)
    out = []
    for i in bars:
        ctx = Context(prepared, i, broker)
        strategy.on_bar(ctx)
        out.append([(o.side, o.qty, o.target) for o in ctx.orders])
    return out


@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_decisions_ignore_every_bar_after_the_decision_bar(name, real_slice):
    """Only bars after the cut differ, so every decision at or before it must match.

    Two *opposite* tampers rather than clean-vs-tampered: a strategy that
    peeks one bar ahead turns long under the up-tamper and short under the
    down-tamper, so a leak is forced to show rather than left to chance.
    """
    df = real_slice.iloc[-40_000:].copy()
    cut = len(df) - 5_000
    # decision bars at and below the cut, so a peek of up to 20 bars ahead
    # still lands inside the tampered region
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    market = MarketSpec.futures(leverage=5.0)
    a = _decisions(get_strategy(name), up, bars, market)
    b = _decisions(get_strategy(name), down, bars, market)

    for bar, oa, ob in zip(bars, a, b):
        assert oa == ob, (
            f"{name}: the order queued at bar {bar} changed when only bars from "
            f"{cut} onward were modified - the strategy reads the future "
            f"({oa} vs {ob})")


@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_every_strategy_actually_trades_on_the_real_slice(name, real_slice):
    """A strategy that never orders makes every test parametrized over it vacuous.

    This is a real risk here, not a hypothetical: the short synthetic
    fixtures leave most strategies inert, so their causality and parity
    assertions reduce to ``[] == []``.
    """
    result = run_backtest(get_strategy(name), real_slice,
                          MarketSpec.futures(leverage=5.0), 1_000.0)
    assert result.fills, (
        f"{name}: placed zero orders over {len(real_slice):,} real bars - "
        "every test parametrized over it is asserting nothing")


def test_the_strict_causality_check_catches_a_real_on_bar_peek():
    """Guard the guard: a deliberate one-bar peek must fail the check above.

    Without this, a refactor that quietly made ``_decisions`` inert would
    leave the whole module green and useless.
    """
    from tradebot.strategy import Strategy

    from conftest import make_ohlcv

    class Peeker(Strategy):
        name = "_test_peeker"
        warmup = 5

        def prepare(self, df):
            self._df = df  # the frame stays reachable from on_bar
            return df

        def on_bar(self, ctx: Context) -> None:
            nxt = float(self._df["close"].iloc[ctx.i + 1])
            ctx.order_target(1.0 if nxt > float(self._df["close"].iloc[ctx.i]) else -1.0)

    df = make_ohlcv([100.0 + (i % 7) for i in range(200)])
    cut = 150
    bars = [cut - 1, cut - 2]
    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0

    market = MarketSpec.futures(leverage=5.0)
    a = _decisions(Peeker(), up, bars, market)
    b = _decisions(Peeker(), down, bars, market)
    assert a != b, "the peek detector no longer detects a peek"
