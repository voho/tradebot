"""Perpetual funding must be charged like a real venue charges it."""

from pathlib import Path

import pandas as pd
import pytest

from tradebot.broker import MarketSpec, PaperBroker
from tradebot.data import load_funding, load_funding_deribit, load_funding_extended
from tradebot.engine import run_backtest
from tradebot.orders import Order
from tradebot.registry import get_strategy
from tradebot.strategy import Context, Strategy

from conftest import make_ohlcv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class HoldFromBarTwo(Strategy):
    name = "_test_hold_from_two"

    def on_bar(self, ctx: Context) -> None:
        if ctx.i >= 2 and not ctx.in_market:
            ctx.order_target(1.0)


def _funding_at(index, positions, rate):
    return pd.Series([rate] * len(positions), index=[index[p] for p in positions])


def test_long_pays_when_the_rate_is_positive():
    b = PaperBroker(market=MarketSpec.futures(leverage=2.0, fee_rate=0.0),
                    start_balance=1_000.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    notional = abs(b.pos) * 100.0
    flow = b.apply_funding(0.0001, 100.0)
    assert flow == pytest.approx(-0.0001 * notional)
    assert b.funding_paid == pytest.approx(0.0001 * notional)


def test_short_receives_when_the_rate_is_positive():
    b = PaperBroker(market=MarketSpec.futures(leverage=2.0, fee_rate=0.0),
                    start_balance=1_000.0)
    b.execute(Order(target=-1.0), ts=0, price=100.0)
    flow = b.apply_funding(0.0001, 100.0)
    assert flow > 0  # shorts are paid when longs pay
    assert b.funding_paid < 0


def test_spot_never_pays_funding():
    b = PaperBroker(market=MarketSpec.spot(fee_rate=0.0), start_balance=1_000.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    assert b.apply_funding(0.01, 100.0) == 0.0
    assert b.funding_paid == 0.0


def test_flat_account_pays_nothing():
    b = PaperBroker(market=MarketSpec.futures(fee_rate=0.0), start_balance=1_000.0)
    assert b.apply_funding(0.01, 100.0) == 0.0


def test_funding_is_charged_on_notional_so_leverage_multiplies_it():
    """The whole point: a 5x long pays 5x the funding a 1x long pays."""
    paid = {}
    for lev in (1.0, 5.0):
        b = PaperBroker(market=MarketSpec.futures(leverage=lev, fee_rate=0.0),
                        start_balance=1_000.0)
        b.execute(Order(target=1.0), ts=0, price=100.0)
        b.apply_funding(0.0001, 100.0)
        paid[lev] = b.funding_paid
    assert paid[5.0] == pytest.approx(5.0 * paid[1.0], rel=1e-9)


def test_funding_can_liquidate_and_floors_at_zero():
    b = PaperBroker(market=MarketSpec.futures(leverage=5.0, fee_rate=0.0),
                    start_balance=1_000.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    b.apply_funding(0.5, 100.0)  # absurd rate: wipes the margin
    assert b.dead and b.cash == 0.0 and b.pos == 0.0


def test_engine_charges_each_settlement_exactly_once():
    df = make_ohlcv([100.0] * 40)
    rate = 0.001
    funding = _funding_at(df.index, [10, 20, 30], rate)
    market = MarketSpec.futures(leverage=2.0, fee_rate=0.0)

    free = run_backtest(HoldFromBarTwo(), df, market, 1_000.0)
    paid = run_backtest(HoldFromBarTwo(), df, market, 1_000.0, funding=funding)

    assert free.funding_paid == 0.0
    assert paid.funding_paid > 0.0
    # flat price, no fees: the whole difference is three funding payments
    notional = abs(paid.fills[0].qty) * 100.0
    assert paid.funding_paid == pytest.approx(3 * rate * notional, rel=1e-6)
    assert free.final_balance - paid.final_balance == pytest.approx(
        paid.funding_paid, rel=1e-6)


def test_settlements_outside_the_run_are_ignored():
    df = make_ohlcv([100.0] * 30)
    outside = pd.Series([0.01, 0.01],
                        index=[df.index[0] - pd.Timedelta(days=5),
                               df.index[-1] + pd.Timedelta(days=5)])
    result = run_backtest(HoldFromBarTwo(), df,
                          MarketSpec.futures(leverage=2.0, fee_rate=0.0),
                          1_000.0, funding=outside)
    assert result.funding_paid == 0.0


def test_funding_is_ignored_on_spot_even_when_supplied():
    df = make_ohlcv([100.0] * 30)
    funding = _funding_at(df.index, [10, 20], 0.001)
    spot = run_backtest(HoldFromBarTwo(), df, MarketSpec.spot(fee_rate=0.0),
                        1_000.0, funding=funding)
    assert spot.funding_paid == 0.0


def test_a_flat_strategy_pays_no_funding_however_high_the_rate():
    """Funding is a cost of *holding*, so the regime gate should dodge it."""

    class NeverTrades(Strategy):
        name = "_test_never_trades"

        def on_bar(self, ctx: Context) -> None:
            return

    df = make_ohlcv([100.0] * 40)
    funding = _funding_at(df.index, list(range(5, 35)), 0.01)
    result = run_backtest(NeverTrades(), df, MarketSpec.futures(fee_rate=0.0),
                          1_000.0, funding=funding)
    assert result.funding_paid == 0.0
    assert result.final_balance == pytest.approx(1_000.0)


# --------------------------------------------------------------- committed data

def test_committed_funding_data_is_present_and_sane():
    funding = load_funding(DATA_DIR)
    assert funding is not None, "committed funding history is missing"
    assert len(funding) > 4_000
    assert funding.index.is_monotonic_increasing
    assert not funding.index.has_duplicates
    assert funding.index.tz is not None
    # 8-hourly settlements, so ~1,096 a year; allow for venue outages
    span_years = (funding.index[-1] - funding.index[0]).days / 365.25
    per_year = len(funding) / span_years
    assert 1_000 < per_year < 1_150, f"{per_year:.0f} settlements/year is not 8-hourly"
    assert abs(funding).max() < 0.02  # venues cap funding far below 2%/settlement


def test_real_funding_is_positive_on_average_so_longs_pay():
    """The fact that makes this matter: it is a persistent cost, not noise."""
    funding = load_funding(DATA_DIR)
    assert (funding > 0).mean() > 0.7
    annualized = funding.mean() * 3 * 365.25
    assert 0.05 < annualized < 0.40, f"unexpected mean funding: {annualized:.2%}"


def test_funding_materially_reduces_a_leveraged_long(real_ohlcv_slice):
    """End to end on real data: the cost has to actually show up."""
    funding = load_funding(DATA_DIR)
    market = MarketSpec.futures(leverage=5.0, fee_rate=0.0005)
    strategy = get_strategy("kelly_regime_v4")

    free = run_backtest(strategy, real_ohlcv_slice, market, 1_000.0)
    paid = run_backtest(get_strategy("kelly_regime_v4"), real_ohlcv_slice, market,
                        1_000.0, funding=funding)
    assert paid.funding_paid > 0.0
    assert paid.final_balance < free.final_balance


@pytest.fixture(scope="module")
def real_ohlcv_slice():
    from tradebot.data import load_ohlcv_csv

    # Optional fetched files can cover a more recent, unfunded date range.
    df = load_ohlcv_csv(DATA_DIR / "btcusd_spot_5m.csv.gz")
    # a span the committed funding data actually covers
    return df.loc["2021-01-01":"2022-12-31"]


# --- the extended (Binance + Deribit) funding series, added by R-39 ---


def test_deribit_funding_covers_the_gap_the_binance_series_leaves():
    """The whole point of the second series: 2024-2026, which Binance lacks."""
    deribit = load_funding_deribit(DATA_DIR)
    binance = load_funding(DATA_DIR)
    assert deribit is not None
    assert deribit.index.max() > binance.index.max()
    assert deribit.index.max().year >= 2026
    assert deribit.index.is_monotonic_increasing
    assert not deribit.index.has_duplicates
    assert deribit.notna().all()


def test_extended_funding_never_overwrites_a_real_binance_settlement():
    """Deribit fills the post-2023 gap only; the overlap stays Binance.

    The two venues' rates differ materially (r=0.69 daily on the overlap,
    level ratio unstable year to year), so a splice that let Deribit
    override real Binance values inside the overlap would silently change
    every published funding figure in the repo.
    """
    binance = load_funding(DATA_DIR)
    combined, source = load_funding_extended(DATA_DIR)

    overlap = combined.index.intersection(binance.index)
    pd.testing.assert_series_equal(combined.loc[overlap], binance.loc[overlap])
    assert (source.loc[overlap] == "binance").all()
    assert (source[source.index > binance.index.max()] == "deribit").all()
    assert source.index.equals(combined.index)


def test_extended_funding_is_still_a_persistent_cost_after_2023():
    """The premium compressed; it did not invert. Both matter downstream."""
    combined, source = load_funding_extended(DATA_DIR)
    extension = combined[source == "deribit"]
    assert (extension > 0).mean() > 0.6
    annualized = extension.mean() * 3 * 365.25
    assert 0.0 < annualized < 0.20, f"unexpected extension funding: {annualized:.2%}"
