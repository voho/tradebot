import math

import pytest

from tradebot.broker import MarketSpec, PaperBroker
from tradebot.orders import Order, Side


def spot_broker(balance=1_000.0, fee=0.001):
    return PaperBroker(market=MarketSpec.spot(fee_rate=fee), start_balance=balance)


def fut_broker(balance=1_000.0, lev=5.0, fee=0.0005):
    return PaperBroker(market=MarketSpec.futures(leverage=lev, fee_rate=fee),
                       start_balance=balance)


def test_spot_full_buy_respects_cash_and_fees():
    b = spot_broker()
    fills = b.execute(Order(target=1.0), ts=0, price=100.0)
    assert len(fills) == 1
    f = fills[0]
    # spend everything: notional + fee <= starting balance
    assert f.qty * f.price + f.fee <= 1_000.0 + 1e-9
    assert f.qty * f.price >= 1_000.0 * 0.99  # nearly fully invested
    assert b.equity(100.0) == pytest.approx(1_000.0 - f.fee, rel=1e-9)


def test_spot_round_trip_pnl():
    b = spot_broker(fee=0.001)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    b.execute(Order(target=0.0), ts=1, price=110.0)
    assert b.pos == 0.0
    # ~10% gain minus two fees
    assert b.equity(110.0) == pytest.approx(1_000.0 * 1.1 * 0.999 * 0.999, rel=1e-3)


def test_spot_cannot_short():
    b = spot_broker()
    fills = b.execute(Order(target=-1.0), ts=0, price=100.0)
    assert fills == []
    assert b.pos == 0.0
    fills = b.execute(Order(side=Side.SELL, qty=5.0), ts=1, price=100.0)
    assert fills == []


def test_futures_leverage_cap():
    b = fut_broker(balance=1_000.0, lev=5.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    notional = abs(b.pos) * 100.0
    assert notional <= 5_000.0
    assert notional >= 5_000.0 * 0.99  # close to full 5x
    # margin requirement holds after fees
    assert notional <= b.equity(100.0) * 5.0 + 1e-6


def test_futures_short_pnl():
    b = fut_broker(fee=0.0)
    b.execute(Order(target=-1.0), ts=0, price=100.0)
    assert b.pos < 0
    # 10% drop with 5x short => +50%
    assert b.equity(90.0) == pytest.approx(1_500.0, rel=1e-9)
    b.execute(Order(target=0.0), ts=1, price=90.0)
    assert b.pos == 0.0
    assert b.cash == pytest.approx(1_500.0, rel=1e-9)


def test_futures_flip_is_close_then_open():
    b = fut_broker(fee=0.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    fills = b.execute(Order(target=-1.0), ts=1, price=100.0)
    assert len(fills) == 2
    assert fills[0].side is Side.SELL and fills[0].qty == pytest.approx(abs(fills[0].qty))
    assert fills[1].side is Side.SELL
    assert b.pos < 0


def test_target_deadband_prevents_churn():
    b = fut_broker()
    b.execute(Order(target=1.0), ts=0, price=100.0)
    pos = b.pos
    # re-emitting the same target with a tiny price move must not trade
    fills = b.execute(Order(target=1.0), ts=1, price=100.5)
    assert fills == []
    assert b.pos == pos


def test_liquidation_price_long():
    b = fut_broker(balance=1_000.0, lev=5.0, fee=0.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    p_liq = b.liquidation_price()
    # 5x long: ~20% adverse move (mm pushes it slightly up)
    assert 79.0 < p_liq < 81.0
    # equity at p_liq equals maintenance margin
    eq = b.equity(p_liq)
    maint = b.market.maintenance_margin_rate * abs(b.pos) * p_liq
    assert eq == pytest.approx(maint, rel=1e-9)


def test_liquidation_triggers_and_kills_broker():
    b = fut_broker(balance=1_000.0, lev=5.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    p_liq = b.liquidation_price()
    fill = b.check_liquidation(ts=1, o=95.0, h=96.0, l=p_liq - 0.5)
    assert fill is not None and fill.kind == "liquidation"
    assert b.dead and b.pos == 0.0
    assert b.cash >= 0.0
    assert b.execute(Order(target=1.0), ts=2, price=90.0) == []


def test_no_liquidation_when_price_stays_above():
    b = fut_broker(balance=1_000.0, lev=5.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    assert b.check_liquidation(ts=1, o=99.0, h=100.0, l=90.0) is None
    assert not b.dead


def test_spot_never_liquidates():
    b = spot_broker()
    b.execute(Order(target=1.0), ts=0, price=100.0)
    # even a catastrophic bar does not liquidate a spot position
    assert b.check_liquidation(ts=1, o=50.0, h=51.0, l=1.0) is None
    assert not b.dead


def test_qty_order_clamped_to_leverage():
    b = fut_broker(balance=1_000.0, lev=5.0)
    b.execute(Order(side=Side.BUY, qty=1_000.0), ts=0, price=100.0)  # absurd size
    assert abs(b.pos) * 100.0 <= 5_000.0 + 1e-6


def test_slippage_hurts_both_sides():
    b = fut_broker(fee=0.0)
    b.slippage_bps = 10.0
    fills = b.execute(Order(target=1.0), ts=0, price=100.0)
    assert fills[0].price == pytest.approx(100.0 * 1.001)
    fills = b.execute(Order(target=0.0), ts=1, price=100.0)
    assert fills[0].price == pytest.approx(100.0 * 0.999)


def test_qty_order_crossing_zero_splits_fills():
    b = fut_broker(fee=0.0)
    b.execute(Order(side=Side.BUY, qty=1.0), ts=0, price=100.0)
    fills = b.execute(Order(side=Side.SELL, qty=2.0), ts=1, price=100.0)
    assert len(fills) == 2  # close leg + open leg, never one fill through zero
    assert fills[0].qty == pytest.approx(1.0)
    assert b.pos == pytest.approx(-1.0)


def test_qty_flip_respects_leverage_cap():
    b = fut_broker(balance=1_000.0, lev=5.0)
    b.execute(Order(target=-1.0), ts=0, price=100.0)
    # adverse move guts equity; a huge flip order must be margin-checked
    b.execute(Order(side=Side.BUY, qty=99.0), ts=1, price=119.0)
    assert b.pos >= 0
    assert abs(b.pos) * 119.0 <= max(b.equity(119.0), 0.0) * 5.0 + 1e-6


def test_bankrupt_close_floors_cash_at_zero_and_kills_broker():
    b = fut_broker(balance=1_000.0, lev=5.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    # voluntary close far below the bankruptcy price
    fills = b.execute(Order(target=0.0), ts=1, price=40.0)
    assert b.cash == 0.0
    assert b.dead
    # the fill's pnl reflects money actually lost, not more than the account
    assert fills[0].realized_pnl - fills[0].fee >= -1_000.0 - 1e-6


def test_sizing_accounts_for_slippage():
    b = spot_broker(balance=1_000.0, fee=0.001)
    b.slippage_bps = 50.0
    fills = b.execute(Order(target=1.0), ts=0, price=100.0)
    f = fills[0]
    # never spend more than the account holds, even with heavy slippage
    assert f.qty * f.price + f.fee <= 1_000.0 + 1e-9


def test_min_notional_blocks_dust_opens_but_not_closes():
    b = fut_broker(balance=1_000.0, fee=0.0)
    # dust open is skipped
    assert b.execute(Order(side=Side.BUY, qty=0.001), ts=0, price=100.0) == []
    # real open works, then a dust reduce is still allowed
    b.execute(Order(target=1.0), ts=1, price=100.0)
    fills = b.execute(Order(side=Side.SELL, qty=0.001), ts=2, price=100.0)
    assert len(fills) == 1


def test_equity_definition():
    b = fut_broker(fee=0.0)
    b.execute(Order(target=1.0), ts=0, price=100.0)
    qty = b.pos
    assert b.equity(105.0) == pytest.approx(1_000.0 + qty * 5.0)
    assert not math.isnan(b.equity(100.0))
