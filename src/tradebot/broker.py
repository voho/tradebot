"""Paper broker with unified margin-style bookkeeping for spot and futures.

Both market types share one accounting model:

- ``cash`` is the margin/quote balance. Opening a position does not move
  notional out of cash (futures-style); it only pays the fee. Realized
  PnL flows into cash when a position is reduced.
- ``equity(price) = cash + pos * (price - entry)``.

For spot this is economically identical to holding the asset (the equity
path and fees match a real spot account exactly); shorting is disabled
and notional is capped at 1x equity, so liquidation can never trigger.

Futures are USDT-margined cross-perp style: signed position, notional
capped at ``equity * leverage``, and a liquidation check per bar using
the analytic liquidation price with a maintenance-margin rate.

Simplifications (documented in README): no funding rates, taker-only
fees, fills at bar open plus optional slippage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tradebot.orders import Fill, Order, Side


@dataclass(frozen=True)
class MarketSpec:
    """Parameters of the traded market."""

    name: str
    leverage: float = 1.0
    fee_rate: float = 0.001  # taker fee, fraction of notional
    allow_short: bool = False
    maintenance_margin_rate: float = 0.005
    min_notional: float = 5.0  # exchange-style minimum order size (USD)

    @staticmethod
    def spot(fee_rate: float = 0.001) -> "MarketSpec":
        return MarketSpec(name="spot", leverage=1.0, fee_rate=fee_rate, allow_short=False)

    @staticmethod
    def futures(leverage: float = 5.0, fee_rate: float = 0.0005) -> "MarketSpec":
        return MarketSpec(
            name=f"futures_{leverage:g}x",
            leverage=leverage,
            fee_rate=fee_rate,
            allow_short=True,
        )


# Same-sign target adjustments smaller than this fraction of max notional
# are ignored, so strategies may re-emit their target every bar without
# racking up rebalancing churn. Closes and flips always execute.
REBALANCE_DEADBAND = 0.05


@dataclass
class PaperBroker:
    market: MarketSpec
    start_balance: float
    slippage_bps: float = 0.0

    cash: float = field(init=False)
    pos: float = field(init=False, default=0.0)
    entry: float = field(init=False, default=0.0)
    dead: bool = field(init=False, default=False)  # liquidated: trading stops
    fees_paid: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.start_balance <= 0:
            raise ValueError("start_balance must be positive")
        self.cash = float(self.start_balance)

    # ------------------------------------------------------------------ state

    def equity(self, price: float) -> float:
        return self.cash + self.pos * (price - self.entry)

    def notional(self, price: float) -> float:
        return abs(self.pos) * price

    # ------------------------------------------------------------------ fills

    def _slipped(self, price: float, side: Side) -> float:
        slip = self.slippage_bps / 10_000.0
        return price * (1.0 + slip) if side is Side.BUY else price * (1.0 - slip)

    def _transact(self, ts, delta: float, price: float, kind: str = "order") -> Fill | None:
        """Change position by ``delta`` base units at ``price`` (pre-slippage)."""
        if abs(delta) < 1e-12:
            return None
        # exchange-style min order size; reduces are always allowed (close-only)
        increasing = abs(self.pos + delta) > abs(self.pos)
        if increasing and kind == "order" and abs(delta) * price < self.market.min_notional:
            return None
        side = Side.BUY if delta > 0 else Side.SELL
        px = self._slipped(price, side)

        realized = 0.0
        pos0, entry0 = self.pos, self.entry
        closing = 0.0
        if pos0 != 0.0 and math.copysign(1.0, delta) != math.copysign(1.0, pos0):
            closing = min(abs(delta), abs(pos0))
            realized = (px - entry0) * closing * math.copysign(1.0, pos0)

        pos1 = pos0 + delta
        if abs(pos1) < 1e-12:
            pos1 = 0.0
        if pos1 != 0.0 and (pos0 == 0.0 or math.copysign(1.0, pos1) != math.copysign(1.0, pos0)):
            entry1 = px  # freshly opened (or flipped) exposure
        elif pos1 != 0.0 and abs(pos1) > abs(pos0):
            entry1 = (entry0 * abs(pos0) + px * abs(delta)) / abs(pos1)  # scale-in: avg
        elif pos1 != 0.0:
            entry1 = entry0  # partial reduce keeps avg entry
        else:
            entry1 = 0.0

        fee = self.market.fee_rate * abs(delta) * px
        self.cash += realized - fee
        self.fees_paid += fee
        if pos1 == 0.0 and self.cash < 0.0:
            # Bankrupt: an account cannot lose more than it holds. Absorb the
            # shortfall (insurance-fund style) so reported trade PnL matches
            # the money actually lost, and stop trading.
            realized += -self.cash
            self.cash = 0.0
            self.dead = True
        self.pos, self.entry = pos1, entry1
        return Fill(ts=ts, side=side, qty=abs(delta), price=px, fee=fee,
                    realized_pnl=realized, kind=kind)

    # ------------------------------------------------------------------ orders

    def execute(self, order: Order, ts, price: float) -> list[Fill]:
        """Execute a market order at ``price`` (bar open). Returns fills."""
        if self.dead:
            return []
        if order.target is not None:
            return self._execute_target(order.target, ts, price)
        delta = order.qty if order.side is Side.BUY else -order.qty
        fills: list[Fill] = []
        if (self.pos != 0.0 and math.copysign(1.0, delta) != math.copysign(1.0, self.pos)
                and abs(delta) > abs(self.pos)):
            # Crossing zero: close first, so the reopened side is margin-checked
            # against post-close equity and trade episodes never straddle zero.
            remaining = delta + self.pos
            fill = self._transact(ts, -self.pos, price)
            if fill:
                fills.append(fill)
            if self.dead:
                return fills
            delta = remaining
        delta = self._clamp_delta(delta, price)
        fill = self._transact(ts, delta, price)
        if fill:
            fills.append(fill)
        return fills

    def _max_qty(self, price: float) -> float:
        """Largest position size affordable at ``price`` after fees and slippage."""
        eq = self.equity(price)
        if eq <= 0:
            return 0.0
        lev = self.market.leverage
        slip = self.slippage_bps / 10_000.0
        haircut = max(0.0, 1.0 - (self.market.fee_rate + slip) * lev)
        return eq * lev * haircut / price

    def _clamp_delta(self, delta: float, price: float) -> float:
        """Clamp a raw qty delta to short and leverage constraints."""
        pos1 = self.pos + delta
        if not self.market.allow_short and pos1 < 0:
            pos1 = 0.0
        limit = self._max_qty(price)
        if abs(pos1) > abs(self.pos):  # only increases are capped by leverage
            pos1 = math.copysign(min(abs(pos1), max(limit, abs(self.pos))), pos1)
        return pos1 - self.pos

    def _execute_target(self, target: float, ts, price: float) -> list[Fill]:
        lo = -1.0 if self.market.allow_short else 0.0
        target = min(1.0, max(lo, target))

        fills: list[Fill] = []
        # A sign flip is done as close-then-open so the new position is
        # sized on post-close equity (exact, and keeps trade episodes clean).
        if self.pos != 0.0 and target != 0.0 and math.copysign(1.0, target) != math.copysign(1.0, self.pos):
            fill = self._transact(ts, -self.pos, price)
            if fill:
                fills.append(fill)
            if self.dead:
                return fills

        desired = math.copysign(self._max_qty(price) * abs(target), target) if target != 0.0 else 0.0
        delta = desired - self.pos
        max_notional = self.equity(price) * self.market.leverage
        if target != 0.0 and self.pos != 0.0 and max_notional > 0:
            if abs(delta) * price < REBALANCE_DEADBAND * max_notional:
                return fills  # ignore tiny same-sign adjustments
        fill = self._transact(ts, delta, price)
        if fill:
            fills.append(fill)
        return fills

    # ------------------------------------------------------------ liquidation

    def liquidation_price(self) -> float | None:
        """Price at which equity hits maintenance margin, or None if no risk."""
        if self.pos == 0.0:
            return None
        mm = self.market.maintenance_margin_rate
        if self.pos > 0:
            p = (self.entry - self.cash / self.pos) / (1.0 - mm)
            return p if p > 0 else None
        return (self.entry + self.cash / abs(self.pos)) / (1.0 + mm)

    def check_liquidation(self, ts, o: float, h: float, l: float) -> Fill | None:
        """Check if the bar's adverse extreme touches the liquidation price.

        Called after order fills at the open. Fills at the liquidation
        price (or at the open if the bar gapped through it).
        """
        if self.dead or self.pos == 0.0:
            return None
        p_liq = self.liquidation_price()
        if p_liq is None:
            return None
        if self.pos > 0 and l <= p_liq:
            px = min(o, p_liq)
        elif self.pos < 0 and h >= p_liq:
            px = max(o, p_liq)
        else:
            return None
        # _transact absorbs any bankruptcy shortfall into the fill's
        # realized_pnl and floors cash at zero.
        fill = self._transact(ts, -self.pos, px, kind="liquidation")
        self.dead = True
        return fill
