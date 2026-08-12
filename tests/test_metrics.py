import numpy as np
import pytest

from tradebot.broker import MarketSpec
from tradebot.engine import run_backtest
from tradebot.metrics import compute_metrics, max_drawdown_pct, sharpe_ratio
from tradebot.registry import get_strategy
from tradebot.strategy import Context, Strategy

from conftest import make_ohlcv


def test_max_drawdown_known_series():
    eq = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    # peak 120 -> trough 80 = 33.33%
    assert max_drawdown_pct(eq) == pytest.approx(100 * (120 - 80) / 120)


def test_max_drawdown_monotonic_is_zero():
    assert max_drawdown_pct(np.linspace(1, 2, 50)) == 0.0


def test_sharpe_of_flat_curve_is_zero():
    assert sharpe_ratio(np.full(100, 500.0)) == 0.0


class OneRoundTrip(Strategy):
    name = "_test_one_trip"

    def on_bar(self, ctx: Context) -> None:
        if ctx.i == 5:
            ctx.order_target(1.0)
        elif ctx.i == 20:
            ctx.close_position()


def test_metrics_basic_fields():
    closes = [100.0] * 6 + list(np.linspace(100, 130, 20)) + [130.0] * 10
    df = make_ohlcv(closes)
    result = run_backtest(OneRoundTrip(), df, MarketSpec.spot(fee_rate=0.001), 1_000.0)
    m = compute_metrics(result)

    assert m.num_trades == 1
    assert m.final_balance == pytest.approx(result.equity.iloc[-1])
    assert m.profit == pytest.approx(m.final_balance - 1_000.0)
    assert m.win_rate_pct == 100.0
    assert m.best_trade == m.worst_trade == pytest.approx(m.profit, rel=1e-6)
    assert m.fees_paid > 0
    assert 0 < m.time_in_market_pct < 100
    assert not m.liquidated


def test_trade_pnls_sum_to_total_profit():
    """Sum of per-trade PnL must equal overall profit (closed positions)."""
    rng = np.random.default_rng(5)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1_500)))
    df = make_ohlcv(closes)
    result = run_backtest(get_strategy("rsi_reversion"), df,
                          MarketSpec.futures(leverage=5.0), 1_000.0)
    m = compute_metrics(result)
    if result.trades:
        total = sum(t.pnl for t in result.trades)
        assert total == pytest.approx(m.profit, rel=1e-6, abs=1e-6)
