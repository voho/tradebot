import numpy as np
import pandas as pd
import pytest

from tradebot.broker import MarketSpec
from tradebot.engine import run_backtest
from tradebot.registry import available_strategies, get_strategy
from tradebot.strategy import Context, Strategy

from conftest import make_ohlcv


class BuyOnBarTen(Strategy):
    name = "_test_buy_on_ten"

    def on_bar(self, ctx: Context) -> None:
        if ctx.i == 10 and not ctx.in_market:
            ctx.order_target(1.0)


def test_orders_fill_at_next_bar_open(trend_df):
    result = run_backtest(BuyOnBarTen(), trend_df, MarketSpec.spot(), 1_000.0)
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.ts == trend_df.index[11]
    assert fill.price == pytest.approx(trend_df["open"].iloc[11])


def test_equity_curve_shape_and_start(trend_df):
    result = run_backtest(BuyOnBarTen(), trend_df, MarketSpec.spot(), 1_000.0)
    assert len(result.equity) == len(trend_df)
    assert (result.equity.iloc[:11] == 1_000.0).all()  # flat before the fill
    assert result.final_balance > 1_000.0  # uptrend, long


def test_buy_and_hold_matches_price_ratio(trend_df):
    strategy = get_strategy("buy_and_hold")
    result = run_backtest(strategy, trend_df, MarketSpec.spot(fee_rate=0.0), 1_000.0)
    fill_open = trend_df["open"].iloc[1]  # signal at bar 0 close, fill at bar 1 open
    expected = 1_000.0 * trend_df["close"].iloc[-1] / fill_open
    assert result.final_balance == pytest.approx(expected, rel=1e-6)
    assert len(result.trades) == 1
    assert result.trades[0].open_at_end


def test_liquidation_stops_trading():
    closes = [100.0] * 12 + [70.0] * 8  # 30% crash kills a 5x long entered at bar 11
    df = make_ohlcv(closes)
    result = run_backtest(BuyOnBarTen(), df, MarketSpec.futures(leverage=5.0), 1_000.0)
    assert result.liquidated
    liq_fills = [f for f in result.fills if f.kind == "liquidation"]
    assert len(liq_fills) == 1
    # equity stays flat (and tiny) after liquidation
    after = result.equity.iloc[-3:]
    assert (after == after.iloc[0]).all()
    assert result.trades[-1].liquidated


def test_gap_through_bankruptcy_liquidates_at_open_before_orders():
    """A gap past the bankruptcy price must floor the account at zero.

    The queued close order must NOT fill; the open-price liquidation check
    preempts it, and the sum of trade PnLs equals the account's real loss.
    """

    class BuyThenClose(Strategy):
        name = "_test_buy_then_close"

        def on_bar(self, ctx: Context) -> None:
            if ctx.i == 5 and not ctx.in_market:
                ctx.order_target(1.0)
            elif ctx.i == 7:
                ctx.close_position()

    closes = [100.0] * 8 + [40.0] * 4
    df = make_ohlcv(closes)
    df.iloc[8, df.columns.get_loc("open")] = 40.0  # gap straight through p_liq

    result = run_backtest(BuyThenClose(), df, MarketSpec.futures(leverage=5.0), 1_000.0)
    assert result.liquidated
    assert result.fills[-1].kind == "liquidation"
    assert result.final_balance == pytest.approx(0.0)
    assert sum(t.pnl for t in result.trades) == pytest.approx(-1_000.0)


def test_open_trade_has_no_exit_ts_and_counts_last_bar(flat_df):
    result = run_backtest(get_strategy("buy_and_hold"), flat_df, MarketSpec.spot(), 1_000.0)
    trade = result.trades[0]
    assert trade.open_at_end and trade.exit_ts is None
    from tradebot.metrics import compute_metrics

    m = compute_metrics(result)
    # fill at bar 1's open; in market bars 1..49 of 50
    assert m.time_in_market_pct == pytest.approx(98.0)


def test_order_notional_is_leverage_independent(trend_df):
    """order_notional(0.5) must risk the same notional on spot and 5x futures."""

    class HalfNotional(Strategy):
        name = "_test_half_notional"

        def on_bar(self, ctx: Context) -> None:
            if ctx.i == 10 and not ctx.in_market:
                ctx.order_notional(0.5)

    notionals = {}
    for market in (MarketSpec.spot(fee_rate=0.0), MarketSpec.futures(leverage=5.0, fee_rate=0.0)):
        result = run_backtest(HalfNotional(), trend_df, market, 1_000.0)
        fill = result.fills[0]
        notionals[market.name] = fill.qty * fill.price
    assert notionals["spot"] == pytest.approx(notionals["futures_5x"], rel=1e-9)
    assert notionals["spot"] == pytest.approx(500.0, rel=1e-9)


def test_strategy_cannot_change_row_count(trend_df):
    class BadStrategy(Strategy):
        name = "_test_bad"

        def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.iloc[:-5]

        def on_bar(self, ctx):
            pass

    with pytest.raises(ValueError, match="same rows"):
        run_backtest(BadStrategy(), trend_df, MarketSpec.spot(), 1_000.0)


def _synthetic_wave(n=2_000, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    closes = 100.0 + 10.0 * np.sin(t / 60.0) + np.cumsum(rng.normal(0, 0.2, n))
    closes = np.maximum(closes, 5.0)
    return make_ohlcv(closes)


@pytest.mark.parametrize("name", sorted(available_strategies()))
@pytest.mark.parametrize("market", [MarketSpec.spot(), MarketSpec.futures()])
def test_no_lookahead_all_registered_strategies(name, market):
    """Truncating future data must not change past behavior.

    Runs each registered strategy on the full series and on a truncated
    prefix; every fill that happened within the prefix must be identical.
    This catches any non-causal indicator or accidental future access.
    """
    df = _synthetic_wave()
    cut = len(df) - 400

    full = run_backtest(get_strategy(name), df, market, 10_000.0)
    part = run_backtest(get_strategy(name), df.iloc[:cut], market, 10_000.0)

    cutoff_ts = df.index[cut - 1]
    full_fills = [(f.ts, f.side, round(f.qty, 9), round(f.price, 9))
                  for f in full.fills if f.ts <= cutoff_ts]
    part_fills = [(f.ts, f.side, round(f.qty, 9), round(f.price, 9))
                  for f in part.fills if f.ts <= cutoff_ts]
    assert full_fills == part_fills


@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_all_registered_strategies_run_both_markets(name):
    df = _synthetic_wave(seed=11)
    for market in (MarketSpec.spot(), MarketSpec.futures(leverage=5.0)):
        result = run_backtest(get_strategy(name), df, market, 1_000.0)
        assert len(result.equity) == len(df)
        assert np.isfinite(result.equity.to_numpy()).all()
        assert (result.equity >= -1e-6).all()  # equity can hit ~0, never negative
