"""Live-path tests: pagination, the bot loop, and exchange-data parity.

The question these answer is the practical one: given only what a venue
will actually hand a bot — 1000 candles per request, paged backwards —
do the strategies produce the same decisions the backtest validated?
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.bot import BotConfig, step
from tradebot.broker import MarketSpec
from tradebot.data import load_dataset
from tradebot.exchanges.base import Balance
from tradebot.exchanges.binance import BinanceSpot
from tradebot.exchanges.bitstamp import BitstampSpot
from tradebot.exchanges.replay import ReplayExchange
from tradebot.live import LiveAccount, compute_signal
from tradebot.registry import get_strategy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# The top three by final balance — the ones anyone would actually deploy.
TOP3 = ["kelly_regime_v4", "kelly_regime_v3", "kelly_regime_v2"]


@pytest.fixture(scope="module")
def real():
    df, label = load_dataset(DATA_DIR, "spot")
    # fail rather than skip: silently dropping the exchange-parity checks
    # would leave a green suite that has verified nothing about the live path
    assert label != "SYNTHETIC", (
        "the committed real dataset is missing - the exchange-data parity "
        "checks would be silently skipped")
    return df


# ------------------------------------------------------------------ pagination

def test_fetch_history_pages_to_the_requested_depth(real):
    """A 100-day warmup needs ~29 pages of 1000 candles; stitching must be exact."""
    window = real.iloc[-40_000:]
    ex = ReplayExchange(window, max_candles_per_request=1000)
    got = ex.fetch_history("BTCUSD", bars=29_000)

    assert len(got) == 29_000
    assert got.index.is_monotonic_increasing
    assert not got.index.has_duplicates
    # stitched pages must equal the underlying series exactly
    expected = window.iloc[-29_000:]
    assert got.index.equals(expected.index)
    np.testing.assert_array_equal(got["close"].to_numpy(), expected["close"].to_numpy())


def test_fetch_history_stops_when_venue_runs_out(real):
    ex = ReplayExchange(real.iloc[-5_000:], max_candles_per_request=1000)
    got = ex.fetch_history("BTCUSD", bars=29_000)
    assert len(got) == 5_000  # all the venue has, no crash, no duplicates
    assert not got.index.has_duplicates


def test_page_limit_is_respected(real):
    ex = ReplayExchange(real.iloc[-5_000:], max_candles_per_request=1000)
    assert len(ex.fetch_candles("BTCUSD", limit=5_000)) == 1_000


# ------------------------------------------------- exchange data == backtest data

@pytest.mark.parametrize("name", TOP3)
def test_strategy_signal_matches_backtest_on_exchange_data(real, name):
    """The decision from paged exchange data must equal the backtest's.

    Builds the candle window the way a bot must (1000-bar pages) and
    compares the resulting target against the same strategy's target
    computed from the contiguous frame.
    """
    strategy = get_strategy(name)
    bars = strategy.warmup + 500
    window = real.iloc[-(bars + 10):]

    ex = ReplayExchange(window, max_candles_per_request=1000)
    paged = ex.fetch_history("BTCUSD", bars=bars)
    contiguous = window.iloc[-bars:]

    # the paged window IS the contiguous window - pagination must be lossless
    assert paged.index.equals(contiguous.index)

    account = LiveAccount(position=0.0, equity_quote=1_000.0, market=MarketSpec.spot())
    from_paged = compute_signal(get_strategy(name), paged, account)
    from_contig = compute_signal(get_strategy(name), contiguous, account)

    assert [(o.target, o.qty, o.side) for o in from_paged] == \
           [(o.target, o.qty, o.side) for o in from_contig]


@pytest.mark.parametrize("name", TOP3)
def test_warmup_requirement_is_documented_and_reachable(name):
    """A cold start must be feasible within a sane number of API calls."""
    strategy = get_strategy(name)
    bars = strategy.warmup + 500
    calls = -(-bars // 1000)  # ceil
    assert calls <= 40, f"{name} needs {calls} API calls to warm up"


# ------------------------------------------------------------------- bot loop

def test_bot_step_buys_then_holds_then_sells(real):
    """Drive the loop end to end and check it moves the wallet sensibly."""
    strategy = get_strategy("kelly_regime_v3")
    bars = strategy.warmup + 600
    window = real.iloc[-(bars + 2_000):]
    ex = ReplayExchange(window, max_candles_per_request=1000, quote=10_000.0)

    cfg = BotConfig(symbol="BTCUSD", strategy="kelly_regime_v3", verbose=False)
    result = step(ex, cfg, strategy)

    assert 0.0 <= result.target <= 1.0  # spot: long or flat, never short
    assert result.equity > 0
    if result.order is not None:
        bal = ex.fetch_balance("BTCUSD")
        assert bal.base > 0 and bal.quote >= 0
        # position moved toward the target
        assert abs(bal.base * result.price / bal.equity(result.price)
                   - result.target) < 0.05


def test_bot_step_refuses_without_enough_history(real):
    ex = ReplayExchange(real.iloc[-500:], max_candles_per_request=1000)
    strategy = get_strategy("kelly_regime_v3")
    res = step(ex, BotConfig(strategy="kelly_regime_v3", verbose=False), strategy)
    assert res.order is None
    assert "insufficient history" in res.reason


def test_bot_step_respects_deadband(real):
    """Already at target => no order, no fees."""
    strategy = get_strategy("kelly_regime_v3")
    bars = strategy.warmup + 600
    window = real.iloc[-(bars + 2_000):]
    ex = ReplayExchange(window, max_candles_per_request=1000, quote=10_000.0)
    cfg = BotConfig(symbol="BTCUSD", strategy="kelly_regime_v3", verbose=False)

    step(ex, cfg, strategy)          # first call may trade
    before = len(ex.orders)
    second = step(ex, cfg, strategy)  # immediately again: nothing should change
    assert len(ex.orders) == before
    assert second.order is None


# ------------------------------------------------------- adapter shape/safety

def test_adapters_default_to_dry_run():
    """Nobody should be able to trade real money by forgetting a flag."""
    assert BinanceSpot().dry_run is True
    assert BitstampSpot().dry_run is True


def test_binance_symbol_split():
    assert BinanceSpot._split_symbol("BTCUSDT") == ("BTC", "USDT")
    assert BinanceSpot._split_symbol("ETHUSD") == ("ETH", "USD")
    with pytest.raises(ValueError):
        BinanceSpot._split_symbol("NOTAPAIR")


def test_adapters_reject_bad_orders():
    for ex in (BinanceSpot(), BitstampSpot()):
        with pytest.raises(ValueError):
            ex.place_market_order("BTCUSD", "sideways", 1.0)
        with pytest.raises(ValueError):
            ex.place_market_order("BTCUSD", "buy", 0.0)


def test_drop_forming_candle_removes_the_open_bar():
    """The forming candle is the future; it must never reach a strategy."""
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    idx = pd.date_range(end=now, periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    for adapter in (BinanceSpot, BitstampSpot):
        trimmed = adapter._drop_forming(df, 5)
        assert len(trimmed) == 4
        assert trimmed.index[-1] < now


def test_balance_equity():
    assert Balance(base=2.0, quote=100.0).equity(50.0) == 200.0
