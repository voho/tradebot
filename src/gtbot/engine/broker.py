"""Cost and execution modelling.

The two things that most often turn a profitable backtest into a losing live
system are optimistic fills and understated costs, so both are modelled
explicitly and are the first thing the evaluation stresses.

Costs
-----
``fee`` is the exchange commission (maker and taker differ), ``half_spread`` is
the cost of crossing, and ``impact`` follows the square-root law
``k * sigma * sqrt(Q / V)`` — the standard concave impact model, which matters
once order size is an appreciable fraction of bar volume.

Execution
---------
``taker``
    Cross at the next bar's open.  Simple, pessimistic, always fills.

``maker``
    Post a limit order at the next bar's open on the passive side.  A fill is
    recognised only if the bar trades *through* the limit by ``queue_ticks``
    ticks — a deliberately conservative proxy for queue position, since merely
    touching a price is no guarantee of being filled.  Unfilled orders are
    re-posted for ``ttl_bars`` and then, if ``cross_after_ttl``, crossed.

The maker path carries the fill selection bias that comes with providing
liquidity: a resting bid fills exactly when the market keeps falling.  That
bias is a property of passive execution, not a modelling defect, and it is why
:class:`ExecutionConfig` lets entry and exit choose different modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: Published Binance USDT-M perpetual fee tiers, in basis points.
#:
#: Fee tier is not a detail for a 5-minute strategy — it is the difference
#: between a viable business and a treadmill.  The gross edge available at this
#: frequency is single-digit basis points per trade, so a round trip that costs
#: 6.7bp at the retail tier and 2.1bp at the top tier decides the outcome.
FEE_TIERS: dict[str, tuple[float, float]] = {
    # name: (taker_bp, maker_bp)
    "retail": (4.5, 1.8),
    "vip3": (3.2, 1.4),
    "vip6": (2.5, 1.0),
    "vip9": (1.7, 0.0),
    "market_maker": (2.3, -0.5),  # negative maker fee = rebate
}


@dataclass(frozen=True)
class CostModel:
    """Transaction costs, all in basis points of traded notional."""

    taker_fee_bp: float = 4.5  # Binance USDT-M perp taker, no VIP tier
    maker_fee_bp: float = 1.8  # Binance USDT-M perp maker
    half_spread_bp: float = 0.35  # BTCUSDT perp is 1-2 ticks wide
    #: Square-root impact coefficient, in units of bar volatility.
    impact_coef: float = 0.35
    #: Perpetual funding, charged every 8h on the held position.
    funding_bp_per_8h: float = 0.0
    tick_size: float = 0.1

    @classmethod
    def for_tier(cls, tier: str, **overrides) -> "CostModel":
        """Build a cost model from a published fee tier."""
        if tier not in FEE_TIERS:
            raise KeyError(f"unknown fee tier {tier!r}; known: {sorted(FEE_TIERS)}")
        taker, maker = FEE_TIERS[tier]
        return cls(taker_fee_bp=taker, maker_fee_bp=maker, **overrides)

    def round_trip_bp(self, execution: "ExecutionConfig") -> float:
        """Expected round-trip cost for an execution profile, excluding impact."""
        def leg(mode: str) -> float:
            return (
                self.taker_fee_bp + self.half_spread_bp if mode == "taker" else self.maker_fee_bp
            )

        return leg(execution.entry_mode) + leg(execution.exit_mode)

    def slippage_bp(self, notional: float, bar_quote_volume: float, bar_vol_bp: float) -> float:
        """Square-root market impact for an order of ``notional`` quote units."""
        if notional <= 0 or bar_quote_volume <= 0:
            return 0.0
        participation = min(notional / bar_quote_volume, 1.0)
        return float(self.impact_coef * bar_vol_bp * np.sqrt(participation))


@dataclass(frozen=True)
class ExecutionConfig:
    """How orders reach the market.

    Entry and exit are configured separately because they are not symmetric
    problems.  The edge here is a fast reversion identified at a bar close, so
    a passive entry is badly adversely selected: a resting bid only fills when
    price keeps falling, which is precisely the subset where the trade was
    wrong.  Crossing the spread to get in pays a few basis points and buys the
    whole distribution.  The exit is not time-critical — the position is
    already on and the reversion has largely happened — so it can be worked
    passively and collect the maker fee instead of paying the taker one.
    """

    #: "taker" | "maker", applied when a trade increases exposure.
    entry_mode: str = "taker"
    #: "taker" | "maker", applied when a trade reduces exposure.
    exit_mode: str = "maker"
    #: Limit-order offset from the reference price, in ATR units.
    maker_offset_atr: float = 0.0
    #: Bars a resting limit order stays live before being reconsidered.
    ttl_bars: int = 2
    #: Cross as taker once the TTL expires rather than abandoning the trade.
    cross_after_ttl: bool = True
    #: Ticks the bar must trade *through* the limit before we call it a fill.
    queue_ticks: float = 1.0
    #: Do not bother trading changes smaller than this fraction of equity.
    min_trade: float = 0.02


@dataclass(frozen=True)
class MarginConfig:
    """Margin and liquidation, which only start to matter above ~2x.

    On a cross-margined perpetual the position is closed by the exchange when
    equity falls to the maintenance margin requirement.  For a position of
    ``L`` times equity and a maintenance margin rate ``mmr``, that happens on an
    adverse price move of ``(1 - mmr*L) / L`` — about 19.5% at 5x with a 0.5%
    maintenance rate.

    A 5-minute strategy will not normally see a 19.5% intrabar move, but
    "normally" is doing a lot of work in that sentence: BTC has printed such
    candles.  Modelling it means a leveraged backtest cannot quietly assume an
    account survives what would in reality have been liquidated.
    """

    maintenance_margin_rate: float = 0.005  # Binance BTCUSDT lowest tier
    #: Fraction of remaining equity left after a forced close (fees + slippage
    #: on a liquidation are punitive).
    liquidation_recovery: float = 0.0

    def liquidation_move(self, leverage: float) -> float:
        """Adverse price move, as a fraction of the entry price, that liquidates.

        Maintenance margin is charged on the notional at the *current* mark, not
        at entry, so the condition is ``E(1 - Lx) = mmr * L * E * (1 - x)`` for a
        long, giving ``x = (1 - mmr*L) / (L * (1 - mmr))``.  That matches the
        exchange's published cross-margin liquidation price; dropping the
        ``(1 - mmr)`` denominator liquidates fractionally too early.
        """
        lev = abs(float(leverage))
        if lev <= 1e-9:
            return float("inf")
        denom = lev * (1.0 - self.maintenance_margin_rate)
        if denom <= 1e-12:
            return 0.0
        return max((1.0 - self.maintenance_margin_rate * lev) / denom, 0.0)

    def margin_use(self, position: float, entry_price: float, low: float, high: float) -> float:
        """How far a bar's adverse extreme went, as a fraction of the distance
        to liquidation, for a position opened at ``entry_price``.

        Anchoring to the *entry* price matters: a position opened mid-bar never
        experienced the move between the previous close and its own fill, and
        charging it that move invents liquidations that did not happen.
        """
        if position == 0.0 or entry_price <= 0.0:
            return 0.0
        adverse = (
            (entry_price - low) / entry_price
            if position > 0
            else (high - entry_price) / entry_price
        )
        adverse = max(adverse, 0.0)
        limit = self.liquidation_move(abs(position))
        if limit <= 0.0:
            return float("inf")
        return adverse / limit


@dataclass
class Fill:
    bar: int
    price: float
    delta: float  # change in position, as a fraction of equity
    cost_bp: float
    is_maker: bool


class Broker:
    """Turns a desired position change into fills, prices and costs."""

    def __init__(self, costs: CostModel, execution: ExecutionConfig):
        self.costs = costs
        self.exec = execution

    @staticmethod
    def _is_entry(target: float, position: float) -> bool:
        """True when the trade increases exposure (or flips into a new side)."""
        if position == 0.0:
            return target != 0.0
        if target == 0.0:
            return False  # closing to flat is an exit, not an entry
        if np.sign(target) != np.sign(position):
            return True  # a flip is an exit and an entry; treat as entry
        return abs(target) > abs(position)

    def try_execute(
        self,
        *,
        bar: int,
        target: float,
        position: float,
        open_: float,
        high: float,
        low: float,
        atr_frac: float,
        quote_volume: float,
        bar_vol_bp: float,
        equity: float,
        force_taker: bool,
    ) -> Fill | None:
        """Attempt to move ``position`` toward ``target`` during one bar.

        ``atr_frac`` is ATR as a fraction of price.  Returns ``None`` when
        nothing traded (order too small, or a limit that was not filled).
        """
        delta = target - position
        if abs(delta) < self.exec.min_trade:
            return None

        side = 1.0 if delta > 0 else -1.0
        notional = abs(delta) * equity
        slip_bp = self.costs.slippage_bp(notional, quote_volume, bar_vol_bp)

        mode = self.exec.entry_mode if self._is_entry(target, position) else self.exec.exit_mode
        if force_taker or mode == "taker":
            cost_bp = self.costs.taker_fee_bp + self.costs.half_spread_bp + slip_bp
            return Fill(bar, open_, delta, cost_bp, is_maker=False)

        # --- passive limit on the side that provides liquidity -------------
        offset = self.exec.maker_offset_atr * max(atr_frac, 1e-9) * open_
        limit = open_ - side * offset
        margin = self.exec.queue_ticks * self.costs.tick_size
        filled = (low <= limit - margin) if side > 0 else (high >= limit + margin)
        if not filled:
            return None

        # A maker fill pays no spread and no impact: we were the resting side.
        cost_bp = self.costs.maker_fee_bp
        return Fill(bar, limit, delta, cost_bp, is_maker=True)

    def funding_cost(self, position: float, bars_per_8h: float) -> float:
        """Funding accrued on the held position over a single bar, as a return."""
        if self.costs.funding_bp_per_8h == 0.0:
            return 0.0
        per_bar_bp = self.costs.funding_bp_per_8h / max(bars_per_8h, 1.0)
        # Longs pay funding when it is positive; shorts receive it.
        return -position * per_bar_bp * 1e-4
