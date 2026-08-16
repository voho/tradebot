"""The expected-profit no-trade band must follow its own derivation."""

import numpy as np
import pytest

from tradebot.broker import MarketSpec
from tradebot.data import load_dataset
from tradebot.engine import run_backtest
from tradebot.registry import get_strategy
from tradebot.strategies.kelly_regime_ev import KellyRegimeEV

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_band_is_proportional_to_fee():
    """Twice the fee, twice the band: the rule is linear in cost."""
    s = KellyRegimeEV()
    assert s._band(0.0005, 0.55) == pytest.approx(2 * s._band(0.00025, 0.55), rel=1e-9)


def test_band_falls_with_the_square_of_volatility():
    """Growth forgone scales with variance, so the band scales with 1/sigma^2."""
    s = KellyRegimeEV()
    assert s._band(0.0002, 0.8) == pytest.approx(s._band(0.0002, 0.4) / 4.0, rel=1e-9)


def test_band_falls_with_a_longer_horizon():
    slow = KellyRegimeEV(horizon_days=30.0)
    fast = KellyRegimeEV(horizon_days=1.0)
    assert slow._band(0.001, 0.55) < fast._band(0.001, 0.55)


def test_band_matches_the_closed_form():
    s = KellyRegimeEV(horizon_days=7.0, min_band=0.0, max_band=99.0)
    fee, vol = 0.001, 0.55
    expected = 2.0 * fee / ((7.0 / 365.25) * vol ** 2)
    assert s._band(fee, vol) == pytest.approx(expected, rel=1e-9)


def test_band_is_clamped_at_both_ends():
    s = KellyRegimeEV(min_band=0.02, max_band=1.0)
    assert s._band(0.0, 0.55) == pytest.approx(0.02)  # free trading still has a floor
    assert s._band(0.5, 0.10) == pytest.approx(1.0)  # absurd fee cannot exceed the cap


def test_a_high_fee_venue_widens_the_band_past_full_exposure():
    """The derivation's own verdict on a 0.40% venue: never rebalance.

    This is the analytic form of what scripts/fee_study.py found by
    brute force, and it is the reason turnover reduction never rescued
    the strategy - the optimum was not a smaller trade, it was no trade.
    """
    s = KellyRegimeEV()
    assert s._band(0.004, 0.55) >= 1.0
    assert s._band(0.001, 0.55) < 1.0


def test_higher_fees_produce_strictly_fewer_fills(real_slice):
    """The whole point, end to end: the strategy responds to the venue."""
    counts = {}
    for fee in (0.0005, 0.002, 0.004):
        result = run_backtest(KellyRegimeEV(), real_slice,
                              MarketSpec.spot(fee_rate=fee), 1_000.0)
        counts[fee] = len(result.fills)
    assert counts[0.0005] > counts[0.002] >= counts[0.004]


def test_it_trades_far_less_than_the_fixed_deadband_version(real_slice):
    market = MarketSpec.spot(fee_rate=0.004)
    ev = run_backtest(get_strategy("kelly_regime_ev"), real_slice, market, 1_000.0)
    v4 = run_backtest(get_strategy("kelly_regime_v4"), real_slice, market, 1_000.0)
    assert len(ev.fills) < len(v4.fills) / 2
    assert ev.fees_paid < v4.fees_paid


def test_it_still_exits_when_the_regime_turns(real_slice):
    """A wide band must never trap the account in a position."""
    result = run_backtest(get_strategy("kelly_regime_ev"), real_slice,
                          MarketSpec.spot(fee_rate=0.004), 1_000.0)
    assert any(f.side.name == "SELL" for f in result.fills), "never exits"
    assert len(result.trades) > 1


def test_registered_variants_differ_only_in_horizon():
    base = get_strategy("kelly_regime_ev")
    fast = get_strategy("kelly_regime_ev_fast")
    assert fast.horizon_days < base.horizon_days
    assert fast._band(0.001, 0.55) > base._band(0.001, 0.55)


@pytest.fixture(scope="module")
def real_slice():
    df, label = load_dataset(DATA_DIR, "spot")
    assert label != "SYNTHETIC"
    return df.iloc[-120_000:]
