"""Backtest <-> live parity: the same strategy, extracted via
tradebot.live, must produce the same decisions the backtester acted on."""

import math

import numpy as np
import pytest

from tradebot.broker import MarketSpec
from tradebot.engine import run_backtest
from tradebot.live import LiveAccount, compute_signal
from tradebot.registry import available_strategies, get_strategy

from conftest import make_ohlcv


def _wave(n=1_200, seed=9):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    closes = 100.0 + 12.0 * np.sin(t / 40.0) + np.cumsum(rng.normal(0, 0.3, n))
    return make_ohlcv(np.maximum(closes, 5.0))


def _engine_decisions(result):
    """(signal_bar_ts -> position sign) from engine fills, up to liquidation.

    A liquidation kills the backtest account, so strategy decisions are
    only comparable before it (a live account would be wiped the same
    way — that's venue behavior, not strategy behavior).
    """
    idx = result.equity.index
    liq_ts = next((f.ts for f in result.fills if f.kind == "liquidation"), None)
    pos = 0.0
    by_bar = {}
    for f in result.fills:
        if liq_ts is not None and f.ts >= liq_ts:
            break
        pos += f.qty if f.side.name == "BUY" else -f.qty
        by_bar[f.ts] = math.copysign(1, pos) if abs(pos) > 1e-9 else 0
    # fills happen at bar i+1's open for a signal at bar i
    return {idx[idx.get_loc(ts) - 1]: sign for ts, sign in by_bar.items()}, liq_ts


@pytest.mark.parametrize("name", sorted(available_strategies()))
@pytest.mark.parametrize("market", [MarketSpec.spot(), MarketSpec.futures(leverage=5.0)])
def test_live_signals_match_backtest_decisions(name, market):
    """Walk the data bar by bar through compute_signal (as a live bot
    would) and compare against the backtester's fills."""
    df = _wave()
    result = run_backtest(get_strategy(name), df, market, 10_000.0)
    expected, liq_ts = _engine_decisions(result)

    live = get_strategy(name)  # fresh instance, like a live process
    pos_sign = 0.0
    got = {}
    end = len(df) - 1  # engine never acts on the last bar's signal
    if liq_ts is not None:
        end = df.index.get_loc(liq_ts) - 1  # account dead from the liq bar on
    for i in range(end):
        window = df.iloc[: i + 1]
        account = LiveAccount(position=pos_sign, equity_quote=10_000.0, market=market)
        orders = compute_signal(live, window, account)
        for order in orders:
            target = order.target
            if target is None:
                continue
            lo = -1.0 if market.allow_short else 0.0
            target = min(1.0, max(lo, target))
            new_sign = math.copysign(1, target) if abs(target) > 1e-9 else 0
            if new_sign != pos_sign:
                got[df.index[i]] = new_sign
                pos_sign = new_sign

    assert got == expected


def test_compute_signal_respects_warmup():
    df = _wave(n=60)
    strategy = get_strategy("macd_cross")  # warmup 150 > 60 bars
    account = LiveAccount(position=0.0, equity_quote=1_000.0, market=MarketSpec.spot())
    assert compute_signal(strategy, df, account) == []


def test_live_account_is_context_compatible():
    account = LiveAccount(position=1.5, equity_quote=5_000.0,
                          market=MarketSpec.futures(leverage=5.0))
    assert account.pos == 1.5
    assert account.equity(123.0) == 5_000.0
    assert account.market.allow_short
