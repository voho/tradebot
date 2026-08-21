"""B-06 paper-trading recorder: state persistence, idempotency, fee parity
and the inception catch-up path — all offline (no network, no venue).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for `scripts.*`

from conftest import make_ohlcv

from tradebot.broker import MarketSpec, PaperBroker
from tradebot.orders import Order
from tradebot.registry import get_strategy
from tradebot.strategy import Context, Strategy

import scripts.paper_trade as pt


class _FlatGatedStrategy(Strategy):
    """Mimics the 18-strategy gate convention (see
    ``inception_catchup_target``'s docstring) with a target latched well
    before the observed window even starts — the scenario that motivated
    the catch-up path: a plain ``compute_signal`` call emits nothing here,
    because ``on_bar`` only fires on a bar-over-bar CHANGE and this
    fixture never changes.
    """

    name = "_test_flat_gated"
    warmup = 5

    def __init__(self, level: float = 0.7) -> None:
        self.level = level

    def prepare(self, df):
        df = df.copy()
        df["target"] = self.level
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)


class _PositionBasedStrategy(Strategy):
    """No ``target`` column — decides straight from ``ctx.position``,
    like ``macd_rsi``/``rsi_reversion``: nothing for the catch-up path to
    read ahead of time.
    """

    name = "_test_position_based"
    warmup = 0

    def on_bar(self, ctx: Context) -> None:
        if not ctx.in_market:
            ctx.order_target(1.0)


# ------------------------------------------------------- inception_catchup_target

def test_inception_catchup_target_reads_the_raw_column():
    df = make_ohlcv([100.0] * 50)
    assert pt.inception_catchup_target(_FlatGatedStrategy(level=0.7), df) == pytest.approx(0.7)


def test_inception_catchup_target_none_without_a_target_column():
    df = make_ohlcv([100.0] * 50)
    assert pt.inception_catchup_target(_PositionBasedStrategy(), df) is None


def test_run_recorder_catches_up_a_latched_signal_at_inception(tmp_path, monkeypatch):
    """The bug this exists to prevent: without the catch-up, a paper
    account cold-started mid-latch would sit flat forever while
    genuinely believing it tracks the strategy (discovered running this
    recorder live against kelly_regime_v4 on Bitstamp, docs/LEDGER.md).
    """
    strat = _FlatGatedStrategy(level=0.7)
    monkeypatch.setattr(pt, "get_strategy", lambda name: strat)

    df = make_ohlcv([100.0] * 50)
    market = MarketSpec.spot(fee_rate=0.004)
    status = pt.run_recorder("_test_flat_gated", df, market,
                             state_path=tmp_path / "state.json",
                             csv_path=tmp_path / "log.csv",
                             start_equity=1000.0, verbose=False)

    assert status == "inception"
    state, _ = pt.load_state(tmp_path / "state.json", 1000.0)
    assert state.pos > 0.0  # actually entered, not stuck at 0
    assert "INCEPTION CATCH-UP" in (tmp_path / "log.csv").read_text()


def test_run_recorder_never_catches_up_a_genuinely_flat_target(tmp_path, monkeypatch):
    """A strategy that is latched flat at exactly 0 must stay flat —
    the catch-up only fires for a nonzero latched target."""
    strat = _FlatGatedStrategy(level=0.0)
    monkeypatch.setattr(pt, "get_strategy", lambda name: strat)

    df = make_ohlcv([100.0] * 50)
    market = MarketSpec.spot(fee_rate=0.004)
    pt.run_recorder("_test_flat_gated", df, market,
                    state_path=tmp_path / "state.json",
                    csv_path=tmp_path / "log.csv",
                    start_equity=1000.0, verbose=False)

    state, _ = pt.load_state(tmp_path / "state.json", 1000.0)
    assert state.pos == 0.0
    assert "INCEPTION CATCH-UP" not in (tmp_path / "log.csv").read_text()


# --------------------------------------------------------------- run_recorder

def test_run_recorder_inception_fee_matches_paperbroker_directly(tmp_path):
    """The recorder must charge the SAME fee the backtest engine would —
    because it literally reuses PaperBroker.execute(), not a re-derived
    formula. Pin that by executing the identical order on a bare
    PaperBroker and comparing.
    """
    df = make_ohlcv([100.0] * 50)
    market = MarketSpec.spot(fee_rate=0.004)
    status = pt.run_recorder("buy_and_hold", df, market,
                             state_path=tmp_path / "state.json",
                             csv_path=tmp_path / "log.csv",
                             start_equity=1000.0, verbose=False)
    assert status == "inception"

    broker = PaperBroker(market=market, start_balance=1000.0)
    ts, price = df.index[-1], float(df["close"].iloc[-1])
    fills = broker.execute(Order(target=1.0), ts, price)

    state, _ = pt.load_state(tmp_path / "state.json", 1000.0)
    assert state.pos == pytest.approx(broker.pos)
    assert state.cash == pytest.approx(broker.cash)
    assert state.fees_paid == pytest.approx(sum(f.fee for f in fills))
    assert state.fees_paid > 0.0


def test_run_recorder_is_idempotent_on_the_same_latest_candle(tmp_path):
    """Running twice against the same closed candle must not double-count."""
    df = make_ohlcv([100.0] * 50)
    market = MarketSpec.spot(fee_rate=0.004)
    state_path, csv_path = tmp_path / "state.json", tmp_path / "log.csv"

    first = pt.run_recorder("buy_and_hold", df, market, state_path=state_path,
                            csv_path=csv_path, start_equity=1000.0, verbose=False)
    assert first == "inception"
    n_runs = pt.load_state(state_path, 1000.0)[0].n_runs
    rows = csv_path.read_text().count("\n")

    second = pt.run_recorder("buy_and_hold", df, market, state_path=state_path,
                             csv_path=csv_path, start_equity=1000.0, verbose=False)
    assert second == "skipped (no new candle)"
    assert pt.load_state(state_path, 1000.0)[0].n_runs == n_runs
    assert csv_path.read_text().count("\n") == rows


def test_run_recorder_advances_exactly_one_row_per_new_candle(tmp_path):
    market = MarketSpec.spot(fee_rate=0.004)
    state_path, csv_path = tmp_path / "state.json", tmp_path / "log.csv"

    pt.run_recorder("buy_and_hold", make_ohlcv([100.0] * 50), market,
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)
    status = pt.run_recorder("buy_and_hold", make_ohlcv([100.0] * 51), market,
                             state_path=state_path, csv_path=csv_path,
                             start_equity=1000.0, verbose=False)

    # buy_and_hold is already in market by bar 51, so on_bar emits nothing —
    # this is a real decision ("still fully long"), not a skip, and gets
    # its own row.
    assert status == "unchanged"
    assert csv_path.read_text().count("\n") == 3  # header + 2 rows


def test_run_recorder_rejects_an_older_candle_than_the_persisted_state(tmp_path):
    market = MarketSpec.spot(fee_rate=0.004)
    state_path, csv_path = tmp_path / "state.json", tmp_path / "log.csv"

    pt.run_recorder("buy_and_hold", make_ohlcv([100.0] * 51), market,
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)
    with pytest.raises(RuntimeError, match="OLDER"):
        pt.run_recorder("buy_and_hold", make_ohlcv([100.0] * 50), market,
                        state_path=state_path, csv_path=csv_path,
                        start_equity=1000.0, verbose=False)


# ------------------------------------------------------- DEFAULT_STRATEGIES

def test_default_strategies_are_all_registered_and_unique():
    """The multi-strategy default (this recorder's main behavior change:
    a bare invocation now records the whole promoted/registered
    kelly_regime lineage plus buy_and_hold, not just kelly_regime_v4) must
    name only real, currently-registered strategies, each exactly once —
    a typo or a stale/renamed strategy here would silently narrow the
    forward record or crash every scheduled run.
    """
    assert len(pt.DEFAULT_STRATEGIES) == len(set(pt.DEFAULT_STRATEGIES))
    for name in pt.DEFAULT_STRATEGIES:
        get_strategy(name)  # raises KeyError if not registered


def test_default_strategies_includes_the_benchmark():
    assert "buy_and_hold" in pt.DEFAULT_STRATEGIES


def test_kelly_regime_ev_fast_is_its_own_registered_class_not_a_parametrization():
    """Confirms the premise the default list relies on: `kelly_regime_ev`
    and `kelly_regime_ev_fast` are two separately registered classes (both
    defined in kelly_regime_ev.py), not one class instantiated two ways -
    so tracking both by name is meaningful, not a duplicate.
    """
    assert type(get_strategy("kelly_regime_ev")) is not type(get_strategy("kelly_regime_ev_fast"))


# ----------------------------------------------------- level_resync_order (R-78)

class _StepGatedStrategy(Strategy):
    """The 18-strategy edge-triggered gate again, but with a target that
    STEPS at a known bar.

    This is the shape R-78 measured on the live record: the target changes
    exactly once, on one 5-minute candle, and ``on_bar``'s
    ``abs(target[i] - target[i-1]) > 1e-9`` gate is silent on every bar
    either side of it. A recorder that only asks on a slower schedule
    therefore has one chance in ``k`` of ever seeing the change — and no
    second chance, because the target is no longer *changing*.
    """

    name = "_test_step_gated"
    warmup = 5

    def __init__(self, step_at: int = 50, before: float = 0.0,
                 after: float = 0.8) -> None:
        self.step_at, self.before, self.after = step_at, before, after

    def prepare(self, df):
        df = df.copy()
        df["target"] = [self.before if i < self.step_at else self.after
                        for i in range(len(df))]
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)


def _spot() -> MarketSpec:
    return MarketSpec.spot(fee_rate=0.004)


def test_level_resync_none_when_the_account_is_already_in_sync():
    df = make_ohlcv([100.0] * 50)
    strat = _FlatGatedStrategy(level=0.7)
    assert pt.level_resync_order(strat, df, prior_target=0.7,
                                 market=_spot()) is None


def test_level_resync_none_without_a_target_column():
    df = make_ohlcv([100.0] * 50)
    assert pt.level_resync_order(_PositionBasedStrategy(), df,
                                 prior_target=0.0, market=_spot()) is None


def test_level_resync_respects_the_brokers_own_deadband():
    """Below REBALANCE_DEADBAND the broker ignores a same-sign adjustment,
    so emitting one would write a misleading row rather than a trade."""
    df = make_ohlcv([100.0] * 50)
    strat = _FlatGatedStrategy(level=0.7)
    inside = 0.7 - (pt.REBALANCE_DEADBAND / 2.0)
    outside = 0.7 - (pt.REBALANCE_DEADBAND * 2.0)
    assert pt.level_resync_order(strat, df, inside, _spot()) is None
    assert pt.level_resync_order(strat, df, outside, _spot()) is not None


def test_level_resync_always_fires_on_a_move_to_flat():
    """The broker always executes a close, so a strategy that wants out
    must get out even if the step is smaller than the deadband."""
    df = make_ohlcv([100.0] * 50)
    strat = _FlatGatedStrategy(level=0.0)
    tiny = pt.REBALANCE_DEADBAND / 10.0
    order = pt.level_resync_order(strat, df, prior_target=tiny, market=_spot())
    assert order is not None and order.target == 0.0


def test_level_resync_clamps_to_what_the_market_can_actually_hold():
    """A spot account cannot hold 1.55x. Once it is fully long, a strategy
    asking for more is already satisfied and must not emit every run."""
    df = make_ohlcv([100.0] * 50)
    strat = _FlatGatedStrategy(level=1.55)
    assert pt.level_resync_order(strat, df, prior_target=1.0, market=_spot()) is None


def test_run_recorder_resyncs_a_target_change_it_never_saw(tmp_path, monkeypatch):
    """The R-78 bug, end to end.

    The target steps at bar 50. The recorder is invoked at bar 49 and then
    not again until bar 59 — exactly what a cron slower than the bar
    interval does. ``on_bar`` at bar 59 sees no CHANGE (the step is ten
    bars behind it) and emits nothing, so before R-78 the account sat flat
    while the strategy had been fully committed for fifty minutes.
    """
    strat = _StepGatedStrategy(step_at=50, before=0.0, after=0.8)
    monkeypatch.setattr(pt, "get_strategy", lambda name: strat)
    state_path, csv_path = tmp_path / "state.json", tmp_path / "log.csv"
    prices = [100.0] * 60

    pt.run_recorder("_test_step_gated", make_ohlcv(prices[:50]), _spot(),
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)
    assert pt.load_state(state_path, 1000.0)[0].pos == 0.0   # correctly flat

    status = pt.run_recorder("_test_step_gated", make_ohlcv(prices[:60]),
                             _spot(), state_path=state_path, csv_path=csv_path,
                             start_equity=1000.0, verbose=False)

    assert status == "traded"
    assert pt.load_state(state_path, 1000.0)[0].pos > 0.0
    assert "LEVEL RESYNC" in csv_path.read_text()


def test_run_recorder_resync_exits_a_position_the_strategy_abandoned(
        tmp_path, monkeypatch):
    """The dangerous direction of the same bug: the strategy went to cash
    on a candle the schedule skipped, and the account stayed long."""
    strat = _StepGatedStrategy(step_at=50, before=0.8, after=0.0)
    monkeypatch.setattr(pt, "get_strategy", lambda name: strat)
    state_path, csv_path = tmp_path / "state.json", tmp_path / "log.csv"
    prices = [100.0] * 60

    pt.run_recorder("_test_step_gated", make_ohlcv(prices[:50]), _spot(),
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)
    assert pt.load_state(state_path, 1000.0)[0].pos > 0.0    # entered at inception

    pt.run_recorder("_test_step_gated", make_ohlcv(prices[:60]), _spot(),
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)

    assert pt.load_state(state_path, 1000.0)[0].pos == 0.0
    assert "LEVEL RESYNC" in csv_path.read_text()


def test_run_recorder_resync_never_overrides_a_genuine_signal(tmp_path, monkeypatch):
    """When on_bar DOES emit, that order is used unchanged — the resync is
    only ever consulted for an empty signal."""
    strat = _StepGatedStrategy(step_at=59, before=0.0, after=0.8)
    monkeypatch.setattr(pt, "get_strategy", lambda name: strat)
    state_path, csv_path = tmp_path / "state.json", tmp_path / "log.csv"
    prices = [100.0] * 60

    pt.run_recorder("_test_step_gated", make_ohlcv(prices[:50]), _spot(),
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)
    pt.run_recorder("_test_step_gated", make_ohlcv(prices[:60]), _spot(),
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)

    text = csv_path.read_text()
    assert "LEVEL RESYNC" not in text
    assert "target=0.8000" in text
    assert pt.load_state(state_path, 1000.0)[0].pos > 0.0


def test_run_recorder_resync_does_not_fire_when_nothing_changed(tmp_path, monkeypatch):
    """A steady, already-tracked target must still produce a plain
    'unchanged' row — the fix must not manufacture churn."""
    strat = _FlatGatedStrategy(level=0.7)
    monkeypatch.setattr(pt, "get_strategy", lambda name: strat)
    state_path, csv_path = tmp_path / "state.json", tmp_path / "log.csv"

    pt.run_recorder("_test_flat_gated", make_ohlcv([100.0] * 50), _spot(),
                    state_path=state_path, csv_path=csv_path,
                    start_equity=1000.0, verbose=False)
    status = pt.run_recorder("_test_flat_gated", make_ohlcv([100.0] * 51),
                             _spot(), state_path=state_path, csv_path=csv_path,
                             start_equity=1000.0, verbose=False)

    assert status == "unchanged"
    assert "LEVEL RESYNC" not in csv_path.read_text()
