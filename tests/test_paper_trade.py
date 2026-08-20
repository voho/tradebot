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
