"""``tradebot.bot.step`` level-resync fix (R-81/B-39): the same defect
R-78 diagnosed and fixed in ``scripts/paper_trade.py`` also lived here —
``step()`` only used the edge-triggered orders ``compute_signal`` returns,
so any target change landing on a candle a slower-than-bar-interval
scheduler skipped was silently lost. ``raw_desired_target`` reads the
strategy's current stance directly, independent of whether ``on_bar``'s
bar-over-bar change gate happened to fire.
"""

from __future__ import annotations

from conftest import make_ohlcv

from tradebot.bot import BotConfig, raw_desired_target, step
from tradebot.exchanges.base import Balance, Exchange, OrderResult
from tradebot.strategy import Context, Strategy


class _FlatGatedStrategy(Strategy):
    """Mimics the 18-strategy ``kelly_regime``-family convention: emits an
    order only when the current bar's target differs from the previous
    bar's. Latched at a fixed level for the whole fetched window, exactly
    the shape a target change lands on a skipped candle produces — the
    edge gate sees no change inside the window it was actually given.
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
    """No ``target`` column — decides straight from ``ctx.position``, like
    ``macd_rsi``/``rsi_reversion``: nothing for the resync path to read
    ahead of time, so it must fall back to the pre-existing behaviour.
    """

    name = "_test_position_based"
    warmup = 0

    def __init__(self, go_long: bool = True) -> None:
        self.go_long = go_long

    def on_bar(self, ctx: Context) -> None:
        if self.go_long and not ctx.in_market:
            ctx.order_target(1.0)


class _FakeExchange(Exchange):
    name = "fake"
    taker_fee = 0.001

    def __init__(self, candles, base: float, quote: float):
        self._candles = candles
        self._balance = Balance(base=base, quote=quote)
        self.orders: list[tuple[str, float]] = []

    def fetch_candles(self, symbol, minutes=5, limit=1000, end_ms=None):
        return self._candles.iloc[-limit:]

    def fetch_history(self, symbol, bars, minutes=5, progress=False):
        return self._candles.iloc[-bars:]

    def fetch_balance(self, symbol):
        return self._balance

    def place_market_order(self, symbol, side, qty):
        self.orders.append((side, qty))
        price = float(self._candles["close"].iloc[-1])
        if side == "buy":
            self._balance = Balance(self._balance.base + qty,
                                    self._balance.quote - qty * price)
        else:
            self._balance = Balance(self._balance.base - qty,
                                    self._balance.quote + qty * price)
        return OrderResult(side=side, qty=qty, price=price, dry_run=True)


def _df(n=30, level=100.0):
    return make_ohlcv([level] * n)


# ------------------------------------------------------ raw_desired_target


def test_raw_desired_target_reads_the_latched_level_even_with_no_edge():
    df = _df()
    strat = _FlatGatedStrategy(level=0.7)
    assert raw_desired_target(strat, df) == 0.7


def test_raw_desired_target_none_without_a_target_column():
    df = _df()
    assert raw_desired_target(_PositionBasedStrategy(), df) is None


# ------------------------------------------------------------------- step()


def test_step_resyncs_to_a_target_the_edge_gate_never_saw():
    """The scenario R-78/B-39 names: the strategy has wanted 70% long for
    the whole fetched window (the change happened on a candle a slower
    schedule skipped), the account is still flat, and on_bar's edge gate
    fires nothing because bar i and bar i-1 agree inside this window. The
    fixed step() must still move the account toward 0.7, not leave it flat.
    """
    df = _df()
    strat = _FlatGatedStrategy(level=0.7)
    ex = _FakeExchange(df, base=0.0, quote=1_000.0)
    config = BotConfig(strategy=strat.name, min_rebalance=0.10,
                       min_notional=1.0, verbose=False)

    result = step(ex, config, strat)

    assert result.target == 0.7
    assert result.order is not None
    assert ex.orders and ex.orders[0][0] == "buy"


def test_step_resync_exits_a_position_the_strategy_abandoned():
    """Mirror case: the strategy now wants flat (0.0), the edge gate never
    fired because the window never saw the transition, and the account
    still holds the old position — step() must still flatten it.
    """
    df = _df()
    strat = _FlatGatedStrategy(level=0.0)
    ex = _FakeExchange(df, base=7.0, quote=0.0)  # holding 7 units @ $100 = $700
    config = BotConfig(strategy=strat.name, min_rebalance=0.10,
                       min_notional=1.0, verbose=False)

    result = step(ex, config, strat)

    assert result.target == 0.0
    assert result.order is not None
    assert ex.orders and ex.orders[0][0] == "sell"


def test_step_resync_does_not_fire_when_already_in_sync():
    df = _df()
    strat = _FlatGatedStrategy(level=0.7)
    ex = _FakeExchange(df, base=7.0, quote=300.0)  # 7 * 100 = 700 of 1000 equity
    config = BotConfig(strategy=strat.name, min_rebalance=0.10,
                       min_notional=1.0, verbose=False)

    result = step(ex, config, strat)

    assert result.order is None
    assert result.reason.startswith("inside deadband")


def test_step_still_respects_the_min_rebalance_deadband():
    """A tiny drift from the raw target must not trigger a trade — the
    resync is level-triggered, not a reason to ignore the existing
    deadband/min-notional checks."""
    df = _df()
    strat = _FlatGatedStrategy(level=0.71)  # 1pp away from current 0.70
    ex = _FakeExchange(df, base=7.0, quote=300.0)
    config = BotConfig(strategy=strat.name, min_rebalance=0.10,
                       min_notional=1.0, verbose=False)

    result = step(ex, config, strat)

    assert result.order is None


def test_step_falls_back_to_edge_triggered_orders_without_a_target_column():
    """Strategies with no target column (~5 of the roster) are unaffected
    by this fix and keep deciding from compute_signal's orders exactly as
    before."""
    df = _df()
    strat = _PositionBasedStrategy(go_long=True)
    ex = _FakeExchange(df, base=0.0, quote=1_000.0)
    config = BotConfig(strategy=strat.name, min_rebalance=0.10,
                       min_notional=1.0, verbose=False)

    result = step(ex, config, strat)

    assert result.target == 1.0
    assert result.order is not None
