"""R-182 shared pre-registration: a core-satellite venue decomposition of
`kelly_regime_v4`'s target exposure.

**Direction (attacks COST).** `kelly_regime_v4`'s `target` column (see
`tradebot/strategies/kelly_regime.py::KellyRegime.prepare`, reused here
unmodified via `r145_shared.compute_target`) is a single causal fraction
of equity notional, currently expressed through ONE venue at a time (all
spot, taker fee 0.10%/0.40%, no funding; or all futures_5x, taker fee
0.05%, funding-bearing). R-145/R-151/R-154 already tried splitting this
SAME target across both venues at once, at a fixed threshold
(`spot_frac = min(target, thr)`, `fut_frac = max(target - thr, 0)`, both
legs always >= 0). That family is now closed: R-154's own fully
bug-fixed harness (B-45 overflow redistribution + B-46 joint-move
gating, both adopted) still failed R-145's own inner-validation gate at
the REAL 0.40% spot tier, with an incremental-turnover/funding-saved
ratio of **0.931** against a <0.50 kill bar -- so close that R-154's own
closing line calls the venue-routing research space on this strategy
"exhausted."

This round's mechanism is a **different split function**, not a
different threshold on the same one: instead of routing by MAGNITUDE
(spot carries the bottom slice, futures the excess -- so BOTH legs
independently chase the SAME frequently-changing signal, which is
exactly why R-145/151/154's two-leg turnover ate the funding saved),
route by TIME-SCALE. A slow-moving BASE (a frozen constant, or a
long-halflife causal EWMA of `target`) is held on unlevered spot and
rebalanced only when equity drift crosses a WIDE deadband -- by
construction it fires far less often than `target` itself changes. The
DEVIATION (`target - BASE`, SIGNED -- it can be negative) is held on
leveraged futures and absorbs essentially all of v4's own turnover, at
futures' lower 0.05% taker fee instead of spot's 0.10%/0.40%. Funding is
paid only on the deviation's own (smaller, mean-reverting-around-zero)
notional rather than on gross exposure. Aggregate exposure
(`spot_frac + fut_frac`) equals v4's own unmodified `target` at every
bar by construction -- this is a pure execution/instrument
decomposition of an UNCHANGED decision, never a new directional bet
(the aggregate is the SAME number `plain_v4_period` already computes;
only which venue carries which slice changes) -- so unlike most of this
ledger's other closed COST mechanisms it should risk-match ("R-33")
close to trivially in principle, and this file's own self-checks
MEASURE that rather than assume it, per R-131's rule that an unmatched
control fails silently.

**Why this is not predictably killed by R-145/R-151/R-154's own
finding.** R-154's decisive number, `0.931`, is
`extra_fees_from_splitting / funding_dollars_saved` at the 0.40% tier.
Their split makes BOTH legs trade on every threshold-crossing of the
SAME fast signal, so `extra_fees` is large relative to what gets saved.
This round's split is asymmetric BY DESIGN: the spot leg's own turnover
is deliberately minimized (a near-static base), so `extra_fees` should
fall toward "occasional spot rebalances only" while the futures leg's
fee bill is roughly UNCHANGED from the plain all-futures baseline (same
signal, same 0.05% rate, on essentially the same number of fills) --
meaning the split's OWN marginal cost is mostly just the spot leg's rare
trades, not a second full copy of v4's own turnover. Whether that
marginal cost clears 50% of whatever funding a demeaned notional saves
is an empirical question this file measures on real data below, not
assumed.

**Citations.** Ackerer, Hugonnier & Jermann (2024/2025 working paper),
"Perpetual Futures Pricing" -- the same financing-decision framing
R-145 used: venue choice is a pure financing decision when the target
exposure itself is unchanged. Dao, C. et al. (2016) and Gârleanu &
Pedersen (2013) -- decay/EWMA-derived rates, the same methodology
R-165's novel branch validated for smoothing a SIGNAL's value; this
round applies the identical "derive a rate from the object's own
measured persistence" logic to a VENUE SPLIT POINT instead, a
genuinely different application (R-165 changed what v4 targets; this
changes nothing about the target, only which venue executes it).
Schmeling, Schrimpf & Todorov (2023, BIS WP 1087) -- funding premium is
volatile and sometimes negative; cited, as R-145 cited it, as a
guardrail that this design may only ever AVOID a cost already being
paid, never harvest a new carry bet.

**Not a duplicate of:**
- R-145/R-151/R-154 (threshold split `min(target,thr)`/`max(target-thr,0)`,
  both legs always long-only, futures leg near-idle -- 11 fills over 2
  years at R-145's own primary threshold; this round's futures leg is
  the ACTIVE leg, trading on nearly every deadband-triggered move,
  SIGNED, and the spot leg is the near-idle one -- the opposite
  allocation, for a different reason: minimizing SPOT turnover, not
  minimizing FUTURES usage);
- R-64 (Gârleanu-Pedersen partial adjustment on the WHOLE product,
  weighted by the 3 VOTE anchors' own decay rates -- single venue
  throughout, no venue split at all; failed because the anchors' decay
  rates are too similar for GP's weight formula to produce
  heterogeneity -- an unrelated failure mode, since this round's split
  point is not a GP weight on anchor decay);
- R-165 (destination/rate axis isolated onto SCALE alone -- a no-trade
  region or EWMA-derived rate applied to v4's `scale` VALUE, changing
  what v4 targets; this round leaves `target` completely unmodified and
  only changes EXECUTION VENUE);
- R-131/R-133 (turnover corridor/shrink throttling an already-decided
  single-venue re-target -- explicitly closed with "a COST-axis attack
  has to change the DECISION, not the order that follows it"; this round
  does not throttle anything -- every bar's aggregate exposure is
  IDENTICAL to v4's own decision, at full size, immediately; only the
  venue changes);
- R-173 (Roll/Corwin-Schultz illiquidity re-pricing and a
  spread-conditioned deadband width -- a single-venue cost AUDIT and
  THROTTLE, no venue split);
- R-176 (dollar-volume activity-clock resampling of the VOTE itself, or
  a dollar-bar crowding gate -- single instrument, no venue split);
- R-178 (a synthetic DVOL-priced options structure ADDED on top of v4's
  existing futures position -- a third, additive position, not a
  re-routing of the SAME position across two venues).

**The one genuinely new simulation capability this needs, and why it is
in scope per ROUTINE.md Step 1 Q3.** R-145/R-151/R-154's own
`HybridBroker` family keeps BOTH legs long-only by construction (their
own `_execute_leg` raises `ValueError` on any negative `frac`, on
either leg) because their split never needs a leg to go negative
(`fut_frac = max(target - thr, 0) >= 0` always). This round's BASE can
sit above the instantaneous `target` (whenever v4's signal dips below
its own recent/typical level), so the deviation leg must be able to go
SHORT even though `kelly_regime_v4` itself never shorts -- a pure
REPLICATION/accounting technique (the aggregate is still exactly v4's
own non-negative `target`), not a new directional bet. `SignedHybridBroker`
below is a small, disjoint generalization of the SAME, already-validated
two-leg mechanics (`_transact_leg`'s fee/PnL arithmetic is reused
verbatim), extending only the futures leg to carry a sign -- not an
order book, not a queue model, nothing this project's standing "no
order book" rule excludes. Every claim about it is self-tested below,
including reproducing `tradebot.broker.PaperBroker`'s own (already
extensively tested) short-side liquidation formula as an oracle.

**Frozen splits, fees, decision rule -- reused verbatim from
R-145/R-151/R-154 wherever the object is the same, so this round's
result is read against the SAME bar rather than a loosened one:**

- Data: BTC spot 5m + BTC funding (`load_funding_extended`) is the
  PRIMARY, dollar-savings evidence, exactly as in R-145. ETH (Coinbase
  spot, funding-free -- no ETH perpetual funding series is committed
  anywhere in this repo, the same ceiling R-145 disclosed) is a
  mechanism/replication check only, never the dollar-savings gate.
- Splits: inner-train `<= 2020-12-31` (build/calibrate BASE only, never
  read for a promotion-relevant number); inner-validation
  `2021-01-01 -> 2022-12-31` (selection, both branches, both fee tiers);
  holdout `>= OOS_START` untouched by this file or by either branch
  until a branch clears the inner-validation gate below.
- Fee tiers: 0.10% spot / 0.05% futures (standard), and 0.40% spot /
  0.05% futures (the real Bitstamp entry tier).
- Futures leverage headroom: 5.0x, R-145's own `futures_5x` convention.

**Inner-validation gate, frozen now, reusing R-145's own thresholds
verbatim except where this round's own new risk (a signed leg) requires
an addition (marked NEW):**

1. `d_sharpe` (hybrid vs plain `kelly_regime_v4` on `futures_5x`,
   identical `target`, identical fee/funding assumptions per leg)
   `>= +0.20` (R-20's noise floor) on BTC at BOTH fee tiers, 95% paired
   block-bootstrap CI (`tradebot.inference.paired_bootstrap`,
   `annualized_sharpe`) excluding zero.
2. incremental two-leg turnover-dollar cost (extra fees paid by the
   hybrid vs. plain futures v4) is `< 50%` of the raw funding dollars
   saved, at BOTH fee tiers -- R-145's own criterion, and the exact
   number (R-154's `0.931` at 0.40%) this round's mechanism is built to
   move.
3. time-in-market and realized volatility matched to within 1% of the
   plain-futures run (R-131's rule; R-33's risk-matching discipline).
4. on ETH, the harness runs with no lookahead/liquidation bug and its
   mechanical routing (`spot_frac + fut_frac == target`, to floating
   tolerance) holds at every bar -- a correctness check, not a dollar
   comparison (ETH ceiling above).
5. **NEW, this round's own added risk control (may only tighten, never
   loosen, per ROUTINE.md's post-freeze rule -- there is no prior bar to
   loosen here since no prior round ever carried a signed leg):** the
   futures leg's `check_liquidation` must never fire on inner-train OR
   inner-validation, for the SAME reason R-33/R-131 require risk-matching
   -- a liquidation event is not "worse Sharpe," it is a different,
   uncomparable object. Trips this clause => NEGATIVE outright,
   regardless of (1)-(4).

Kill bar (reject without holdout, however good BTC looks): (2) fails at
either fee tier, OR any BTC CI in (1) contains zero, OR (5) trips, OR the
harness fails its own self-tests below.

**n-check, done now (R-78's discipline), against THIS round's own
measured noise -- see `reachable_n_report()` below for the real,
computed numbers, not an assumed order of magnitude.** Like R-145's
comparison (and unlike R-78's B-06 problem), both arms trade the
IDENTICAL underlying price path and (for the constant-base branch) an
identical `target`; the paired difference isolates financing/fee cost
alone, settling smoothly (funding ~3x/day, fees on every fill), nothing
like R-78's 3.0%/day whole-strategy common-mode noise. `reachable_n_report()`
computes the actual paired daily-return-difference standard deviation on
real inner-train BTC data for a representative BASE and reports the
number of days needed to resolve a 0.20-Sharpe-equivalent daily mean
difference at that measured noise, compared against inner-validation's
730 days and the holdout's own actual length.

**Configs evaluated (this file, self-tests only): 0** -- this module
freezes the pre-registration and self-tests the new capability; no
promotion-relevant backtest cell is read here. Each implementation
branch declares its own count per ROUTINE.md Step 3.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.inference import annualized_sharpe, daily_returns, paired_bootstrap  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

from r145_shared import (  # noqa: E402
    FUT_FEE,
    FUT_LEVERAGE,
    SPOT_FEE_BASE,
    SPOT_FEE_REAL,
    compute_target,
    fut_market,
    load_btc,
    load_eth,
    plain_v4_period,
    spot_market,
)
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

# ---------------------------------------------------------------- frozen dates
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# --------------------------------------------------------- frozen decision rule
D_SHARPE_FLOOR = 0.20          # R-20's standing +/-0.2 Sharpe noise floor
TURNOVER_SAVINGS_KILL = 0.50   # R-145's own criterion 2
R154_BASELINE_RATIO_040 = 0.931  # measured, R-154, threshold=1.0, 0.40% spot tier
R154_BASELINE_RATIO_010 = 0.161  # measured, R-154, threshold=1.0, 0.10% spot tier
EXPOSURE_MATCH_TOL_PCT = 1.0   # R-131/R-145's rule


# ------------------------------------------------------------------- routing

@dataclass
class RouteFn:
    """A precomputed, per-bar (spot_frac, fut_frac) routing decision.
    `fut_frac` may be NEGATIVE (short deviation) -- the one difference
    from `r145_shared.RouteFn`, which this dataclass otherwise mirrors
    exactly (same precompute-both-arrays-up-front causality discipline).
    """

    spot_frac: np.ndarray
    fut_frac: np.ndarray

    def __call__(self, i: int) -> tuple[float, float]:
        return float(self.spot_frac[i]), float(self.fut_frac[i])


def route_constant_base(target: np.ndarray, base: float) -> RouteFn:
    """Conservative mechanism: a single frozen constant BASE, computed
    once from inner-train only (never re-estimated inside a run), held on
    spot while v4 is in-market at all (`target > 0`); `target - BASE`
    (signed) on futures.

    **Gated by `target > 0`, not unconditional -- found necessary by this
    file's own self-measurement, not assumed.** An unconditional constant
    base creates a synthetic position during v4's genuine FLAT periods
    (28.7% of inner-train bars): spot holds `BASE` long while futures
    holds `-BASE` short, netting to v4's correct zero exposure but as TWO
    offsetting legs rather than true flat -- exactly the delta-neutral
    spot-long/futures-short carry construction B-03 already tried and
    R-39 found NEGATIVE, introduced here as an unintended side effect
    rather than a deliberate bet. Measured directly: the unconditional
    version pushed bar-level time-in-market from the plain baseline's
    66.9% to 98.5% while `target` itself is nonzero only 71.3% of
    inner-train bars. Gating spot to fire only while `target > 0`
    (matching v4's own in/out-of-market state exactly) removes the
    synthetic carry leg during genuine flat periods and keeps every
    other property (spot near-static while v4 IS in the market, futures
    signed whenever `0 < target < BASE`, which is a real "in market but
    below typical size" state, not a synthetic bet).
    """
    in_market = target > 0.0
    spot = np.where(in_market, float(base), 0.0)
    fut = target - spot
    return RouteFn(spot, fut)


def route_ewma_base(target: np.ndarray, halflife_days: float) -> RouteFn:
    """Novel mechanism: BASE is a causal EWMA of `target` (a low-pass
    filter of v4's own signal, gated the same way as
    `route_constant_base` -- see its docstring for why the gate is
    required, not optional), tracking secular shifts across cycles
    instead of freezing one number for the whole run. `target[i]` is
    already causal (`KellyRegimeV4.prepare` only ever looks backward), so
    an EWMA of it at row i uses only `target[0..i]` -- no additional
    `.shift(1)` is needed for causality (unlike v4's own vol estimator,
    which shifts because it wants YESTERDAY's value specifically, not
    because an unshifted EWM would peek ahead).
    """
    halflife_bars = halflife_days * 288.0  # BARS_PER_DAY, 5m bars
    ewma = pd.Series(target).ewm(halflife=halflife_bars, adjust=False).mean().to_numpy()
    ewma = np.clip(ewma, 0.0, None)  # target >= 0 always, so its EWMA is too
    in_market = target > 0.0
    spot = np.where(in_market, ewma, 0.0)
    fut = target - spot
    return RouteFn(spot, fut)


def route_all_futures(target: np.ndarray) -> RouteFn:
    return RouteFn(np.zeros_like(target), target.copy())


def route_all_spot(target: np.ndarray) -> RouteFn:
    return RouteFn(np.clip(target, 0.0, None), np.zeros_like(target))


# --------------------------------------------------------------------- broker

@dataclass
class SignedHybridBroker:
    """`r145_shared.HybridBroker`'s two-leg mechanics (one cash ledger,
    two marked sub-positions), generalized so the FUTURES leg may go
    short. The spot leg stays long-only (unlevered spot cannot short in
    this project's `MarketSpec` convention -- `MarketSpec.spot()` always
    sets `allow_short=False`). `_transact_leg` below is
    `HybridBroker._transact_leg` copied verbatim (it is already sign-
    agnostic -- generalizes correctly for a leg moving through zero in
    either direction, which R-145 never exercised since its `fut_frac`
    never changed sign, but the arithmetic itself does not depend on
    long-only).

    One deliberate departure from `HybridBroker`, a new correctness
    requirement a signed leg introduces:

    1. **Sign-flip close-then-reopen**, mirroring
       `PaperBroker._execute_target`'s own rule exactly (a leg that
       flips sign closes fully first, so the reopened side is
       margin-checked against POST-CLOSE equity) -- `HybridBroker` never
       needed this because its `fut_frac` never changed sign; this
       round's deviation leg crosses zero routinely.

    Liquidation: `check_liquidation` solves `equity(p) == mm*|pos_fut|*p`
    for `p`, generalizing `PaperBroker.liquidation_price()`'s own
    closed form to treat the spot leg's mark as collateral backing the
    futures leg's maintenance margin, for BOTH signs of `pos_fut` (the
    original `HybridBroker.check_liquidation` only ever solved the
    long-futures/downside case, since its `pos_fut` was never negative).
    `_liquidation_price_signed_check()` below verifies the two-branch
    formula reduces EXACTLY to `PaperBroker.liquidation_price()` when
    `pos_spot=0`, for both signs, against real `PaperBroker` instances as
    the oracle -- not merely by algebra in a docstring.

    **Deadband base: per-leg (`eq * market.leverage`), NOT R-151's
    "shared reference" fix (B-44) -- a deliberate choice, not an
    oversight.** B-44 existed because R-145's threshold split makes BOTH
    legs track the SAME fast-changing signal, so a per-leg base (spot's
    own `eq*1.0`, 5x tighter in dollar terms than futures' `eq*5.0`)
    punished the leg that happened to be unlevered. This round's spot
    leg tracks a near-constant BASE regardless of which base convention
    governs its own deadband -- its `delta` from bar to bar is already
    near zero -- so B-44's asymmetry does not bite here, and per-leg
    bases have a real advantage this round's construction needs: they
    make the FUTURES leg's own deadband identical to what plain
    all-futures `kelly_regime_v4` already uses (`eq*5.0`), so the
    dominant, fast-moving leg's trigger behaviour is directly comparable
    to the baseline it is measured against. Verified, not assumed: this
    module's own `_degenerate_equivalence_check` requires BOTH the
    all-futures AND the all-spot routes to reproduce their single-venue
    baselines to `<1e-6` under per-leg bases -- R-145's original harness
    could only get the all-futures side of that (its own docstring never
    ran an all-spot degenerate check at all), and R-151/154's shared-base
    fix breaks the all-spot side outright (measured here at first: 5.5%
    relative divergence) precisely because it deliberately makes spot's
    threshold no longer match spot's own native behaviour. Each
    implementation branch should re-run this file's self-tests before
    trusting any of its own numbers, exactly as this module's own
    `__main__` block does.
    """

    spot: MarketSpec
    fut: MarketSpec
    start_balance: float
    slippage_bps: float = 0.0

    cash: float = field(init=False)
    pos_spot: float = field(init=False, default=0.0)
    entry_spot: float = field(init=False, default=0.0)
    pos_fut: float = field(init=False, default=0.0)   # SIGNED
    entry_fut: float = field(init=False, default=0.0)
    dead: bool = field(init=False, default=False)
    fees_paid: float = field(init=False, default=0.0)
    funding_paid: float = field(init=False, default=0.0)
    fills_spot: int = field(init=False, default=0)
    fills_fut: int = field(init=False, default=0)
    turnover_notional: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.start_balance <= 0:
            raise ValueError("start_balance must be positive")
        if self.spot.allow_short:
            raise ValueError("SignedHybridBroker's spot leg must be long-only")
        if not self.fut.allow_short:
            raise ValueError("SignedHybridBroker's futures leg must allow short")
        self.cash = float(self.start_balance)

    def equity(self, price: float) -> float:
        return (self.cash + self.pos_spot * (price - self.entry_spot)
                + self.pos_fut * (price - self.entry_fut))

    # -------------------------------------------------------------- fills

    def _slipped(self, price: float, buying: bool) -> float:
        slip = self.slippage_bps / 10_000.0
        return price * (1.0 + slip) if buying else price * (1.0 - slip)

    def _transact_leg(self, delta: float, price: float, fee_rate: float,
                       pos: float, entry: float) -> tuple[float, float, float, float]:
        """`r145_shared.HybridBroker._transact_leg`, copied verbatim."""
        buying = delta > 0
        px = self._slipped(price, buying)
        realized = 0.0
        if pos != 0.0 and math.copysign(1.0, delta) != math.copysign(1.0, pos):
            closing = min(abs(delta), abs(pos))
            realized = (px - entry) * closing * math.copysign(1.0, pos)
        pos1 = pos + delta
        if abs(pos1) < 1e-12:
            pos1 = 0.0
        if pos1 != 0.0 and (pos == 0.0 or math.copysign(1.0, pos1) != math.copysign(1.0, pos)):
            entry1 = px
        elif pos1 != 0.0 and abs(pos1) > abs(pos):
            entry1 = (entry * abs(pos) + px * abs(delta)) / abs(pos1)
        elif pos1 != 0.0:
            entry1 = entry
        else:
            entry1 = 0.0
        fee = fee_rate * abs(delta) * px
        self.turnover_notional += abs(delta) * px
        return pos1, entry1, realized - fee, fee

    def _leg_state(self, leg: str) -> tuple[float, float, MarketSpec]:
        return ((self.pos_spot, self.entry_spot, self.spot) if leg == "spot"
                else (self.pos_fut, self.entry_fut, self.fut))

    def _set_leg(self, leg: str, pos: float, entry: float) -> None:
        if leg == "spot":
            self.pos_spot, self.entry_spot = pos, entry
        else:
            self.pos_fut, self.entry_fut = pos, entry

    def _apply_transact(self, leg: str, delta: float, price: float,
                        market: MarketSpec, pos: float, entry: float) -> tuple[float, float]:
        pos1, entry1, cash_delta, fee = self._transact_leg(delta, price, market.fee_rate, pos, entry)
        self.cash += cash_delta
        self.fees_paid += fee
        self._set_leg(leg, pos1, entry1)
        if leg == "spot":
            self.fills_spot += 1
        else:
            self.fills_fut += 1
        if self.equity(price) < 0.0:
            self.cash = 0.0
            self.pos_spot = self.entry_spot = 0.0
            self.pos_fut = self.entry_fut = 0.0
            self.dead = True
        return pos1, entry1

    def _execute_leg(self, leg: str, frac: float, price: float) -> None:
        pos, entry, market = self._leg_state(leg)
        lo = -1.0 if market.allow_short else 0.0
        if lo == 0.0 and frac < 0:
            raise ValueError(f"{leg} leg is long-only; got frac={frac!r}")
        target_equiv = min(1.0, max(lo, frac / max(market.leverage, 1e-9)))

        # Sign-flip close-then-reopen (PaperBroker._execute_target's rule).
        if (pos != 0.0 and target_equiv != 0.0
                and math.copysign(1.0, target_equiv) != math.copysign(1.0, pos)):
            pos, entry = self._apply_transact(leg, -pos, price, market, pos, entry)
            if self.dead:
                return

        eq = self.equity(price)
        slip = self.slippage_bps / 10_000.0
        haircut = max(0.0, 1.0 - (market.fee_rate + slip) * market.leverage)
        max_qty = eq * market.leverage * haircut / price if eq > 0 and price > 0 else 0.0
        desired = max_qty * target_equiv
        delta = desired - pos

        base_notional = eq * market.leverage  # per-leg base -- see class docstring
        if target_equiv != 0.0 and pos != 0.0 and base_notional > 0:
            if abs(delta) * price < market.deadband * base_notional:
                return
        if abs(delta) < 1e-12:
            return
        increasing = abs(pos + delta) > abs(pos)
        if increasing and abs(delta) * price < market.min_notional:
            return
        self._apply_transact(leg, delta, price, market, pos, entry)

    def fill(self, spot_frac: float, fut_frac: float, price: float) -> None:
        if self.dead:
            return
        self._execute_leg("spot", spot_frac, price)
        if self.dead:
            return
        self._execute_leg("fut", fut_frac, price)

    # --------------------------------------------------------------- funding

    def apply_funding(self, rate: float, price: float) -> None:
        """Identical formula to `HybridBroker.apply_funding` -- already
        sign-correct for a signed `pos_fut` with no change: a short
        position (`pos_fut < 0`) RECEIVES funding when `rate > 0` (longs
        pay shorts), which `flow = -rate*pos_fut*price` already gives
        (negative times negative is positive cash flow).
        """
        if self.dead or self.pos_fut == 0.0:
            return
        flow = -rate * self.pos_fut * price
        self.cash += flow
        self.funding_paid -= flow
        if self.cash < 0.0:
            self.cash = 0.0
            self.pos_spot = self.entry_spot = 0.0
            self.pos_fut = self.entry_fut = 0.0
            self.dead = True

    # ----------------------------------------------------------- liquidation

    def liquidation_price(self) -> float | None:
        """Solves `equity(p) == mm*|pos_fut|*p` for `p`, both signs of
        `pos_fut`. See `_liquidation_price_signed_check()` for the
        oracle-verified reduction to `PaperBroker.liquidation_price()`.
        """
        if self.dead or self.pos_fut == 0.0:
            return None
        mm = self.fut.maintenance_margin_rate
        if self.pos_fut > 0:
            denom = self.pos_spot + self.pos_fut * (1.0 - mm)
        else:
            denom = self.pos_spot + self.pos_fut * (1.0 + mm)
        if denom == 0:
            return None
        p_liq = (self.pos_spot * self.entry_spot + self.pos_fut * self.entry_fut
                 - self.cash) / denom
        if p_liq <= 0:
            return None
        return p_liq

    def check_liquidation(self, o: float, h: float, l: float) -> bool:
        """Returns True iff a liquidation fired this bar."""
        if self.dead or self.pos_fut == 0.0:
            return False
        p_liq = self.liquidation_price()
        if p_liq is None:
            return False
        long_side = self.pos_fut > 0
        if long_side and l > p_liq:
            return False
        if not long_side and h < p_liq:
            return False
        px = min(o, p_liq) if long_side else max(o, p_liq)
        pos1, entry1, cash_delta, fee = self._transact_leg(
            -self.pos_fut, px, self.fut.fee_rate, self.pos_fut, self.entry_fut)
        self.cash += cash_delta
        self.fees_paid += fee
        self.pos_fut, self.entry_fut = 0.0, 0.0
        self.fills_fut += 1
        if self.cash < 0.0:
            self.cash = 0.0
        self.dead = True
        return True


@dataclass
class HybridResult:
    equity: pd.Series
    fees_paid: float
    funding_paid: float
    turnover_notional: float
    liquidated: bool
    fills_spot: int
    fills_fut: int

    @property
    def final_balance(self) -> float:
        return float(self.equity.iloc[-1])


def run_signed_hybrid_backtest(
    df: pd.DataFrame,
    route_builder: Callable[[pd.DataFrame], RouteFn],
    spot_mkt: MarketSpec,
    fut_mkt: MarketSpec,
    start_balance: float = 1_000.0,
    funding: pd.Series | None = None,
    start: object | None = None,
    end: object | None = None,
    slippage_bps: float = 0.0,
    warmup: int | None = None,
) -> HybridResult:
    """Same per-bar sequence as `r145_shared.run_hybrid_backtest` (fill
    previous bar's pending order at THIS bar's open, check liquidation
    twice, apply funding, record equity at close, decide next order from
    THIS bar's close) -- reused unchanged for consistency with the
    already-validated harness, rather than re-derived to match
    `engine.py`'s own (slightly different) ordering, which R-145/151/154
    never needed to match either.
    """
    warmup = KellyRegimeV4().warmup if warmup is None else warmup
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")
    prefix = prefix_bars(df, lo, warmup)
    frame = df.iloc[lo - prefix: hi]
    n = len(frame)

    route = route_builder(frame)

    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    index = frame.index

    bar_funding = None
    if funding is not None and len(funding) and fut_mkt.pays_funding:
        rates = funding.sort_index()
        rates = rates[(rates.index >= index[0]) & (rates.index <= index[-1])]
        if len(rates):
            slot = index.searchsorted(rates.index, side="right") - 1
            bar_funding = {}
            for pos_i, rate in zip(slot, rates.to_numpy(dtype=float)):
                if pos_i >= 0:
                    bar_funding[int(pos_i)] = bar_funding.get(int(pos_i), 0.0) + float(rate)

    broker = SignedHybridBroker(spot=spot_mkt, fut=fut_mkt, start_balance=start_balance,
                                slippage_bps=slippage_bps)

    equity = np.zeros(n, dtype=float)
    pending: tuple[float, float] | None = None

    for i in range(n):
        if pending is not None and not broker.dead:
            broker.fill(pending[0], pending[1], opens[i])
        pending = None

        broker.check_liquidation(opens[i], opens[i], opens[i])
        broker.check_liquidation(opens[i], highs[i], lows[i])

        if bar_funding is not None:
            rate = bar_funding.get(i)
            if rate is not None:
                broker.apply_funding(rate, closes[i])

        eq = broker.equity(closes[i])
        if not math.isfinite(eq):
            raise ValueError(f"SignedHybridBroker: equity became non-finite at bar {i} ({index[i]})")
        equity[i] = eq

        last_bar = i == n - 1
        if not broker.dead and not last_bar and i >= prefix:
            changed = (i == 0
                      or abs(route.spot_frac[i] - route.spot_frac[i - 1]) > 1e-9
                      or abs(route.fut_frac[i] - route.fut_frac[i - 1]) > 1e-9)
            if changed:
                pending = route(i)

    eq_series = pd.Series(equity, index=index, name="equity")
    if prefix:
        eq_series = eq_series.iloc[prefix:]

    return HybridResult(
        equity=eq_series,
        fees_paid=broker.fees_paid,
        funding_paid=broker.funding_paid,
        turnover_notional=broker.turnover_notional,
        liquidated=broker.dead,
        fills_spot=broker.fills_spot,
        fills_fut=broker.fills_fut,
    )


# ------------------------------------------------------------- risk-matching

def risk_match_report(hybrid_eq: pd.Series, baseline_eq: pd.Series) -> dict:
    """R-33/R-131's own required comparison: realized volatility and
    time-in-market, hybrid vs the single-venue baseline it must match.
    Uses bar-close returns (matches the equity curves' own native
    frequency; both branches should additionally report the daily-return
    version `daily_returns()` feeds to `annualized_sharpe`).
    """
    hr = hybrid_eq.pct_change().dropna()
    br = baseline_eq.pct_change().dropna()
    vol_h, vol_b = float(hr.std()), float(br.std())
    tim_h = float((hr != 0.0).mean())
    tim_b = float((br != 0.0).mean())
    return {
        "vol_hybrid": vol_h, "vol_baseline": vol_b,
        "vol_ratio": vol_h / vol_b if vol_b else float("nan"),
        "tim_hybrid": tim_h, "tim_baseline": tim_b,
    }


# ----------------------------------------------------------- self-checks: fees

def _degenerate_equivalence_check(df: pd.DataFrame, funding: pd.Series | None,
                                  start: str, end: str) -> None:
    """All-futures (`base=0`) and all-spot (`fut_frac=0`) degenerate
    routes must reproduce plain single-venue `kelly_regime_v4` to
    floating tolerance -- `r145_shared`'s own gate, re-run against
    `SignedHybridBroker` to confirm the new sign-flip/shared-deadband
    code paths did not silently change the long-only-both-legs behaviour
    they must still reproduce exactly.
    """
    fut, spot = fut_market(), spot_market()

    def all_futures(frame: pd.DataFrame) -> RouteFn:
        return route_all_futures(compute_target(frame))

    def all_spot(frame: pd.DataFrame) -> RouteFn:
        return route_all_spot(compute_target(frame))

    hybrid_fut = run_signed_hybrid_backtest(df, all_futures, spot, fut, funding=funding,
                                            start=start, end=end)
    plain_fut = plain_v4_period(df, fut, funding, start, end)
    rel_fut = abs(hybrid_fut.final_balance - plain_fut.final_balance) / max(plain_fut.final_balance, 1.0)

    hybrid_spot = run_signed_hybrid_backtest(df, all_spot, spot, fut, funding=funding,
                                             start=start, end=end)
    plain_spot = plain_v4_period(df, spot, None, start, end)
    rel_spot = abs(hybrid_spot.final_balance - plain_spot.final_balance) / max(plain_spot.final_balance, 1.0)

    print(f"[degenerate-equivalence] all-futures: hybrid=${hybrid_fut.final_balance:,.2f} "
          f"plain=${plain_fut.final_balance:,.2f} rel_diff={rel_fut:.2e}")
    print(f"[degenerate-equivalence] all-spot:    hybrid=${hybrid_spot.final_balance:,.2f} "
          f"plain=${plain_spot.final_balance:,.2f} rel_diff={rel_spot:.2e}")
    assert rel_fut < 1e-6, f"all-futures degenerate route diverges: {rel_fut:.2e}"
    assert rel_spot < 1e-6, f"all-spot degenerate route diverges: {rel_spot:.2e}"


class _SignedTargetStrategy(Strategy):
    """Minimal oracle wrapper: replays a precomputed, possibly-signed
    fraction-of-equity array through the REAL, extensively-tested
    `tradebot.engine.run_backtest` / `PaperBroker` path via
    `ctx.order_notional`, whose fraction/leverage convention is exactly
    `SignedHybridBroker._execute_leg`'s own `target_equiv = frac/leverage`
    -- so a spot_frac==0 run of `SignedHybridBroker` on the SAME array
    must reproduce this oracle bit-for-bit if the new sign-flip and
    shared-deadband code is correct.
    """

    name = "_r182_signed_target_oracle"
    warmup = 0

    def __init__(self, target: np.ndarray) -> None:
        self._target = np.asarray(target, dtype=float)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["target"] = self._target
        return df

    def on_bar(self, ctx) -> None:
        t = float(ctx.bar["target"])
        if ctx.i == 0 or abs(t - float(ctx.prev["target"])) > 1e-9:
            ctx.order_notional(t)


def _signed_futures_oracle_check(df: pd.DataFrame, funding: pd.Series | None,
                                 start: str, end: str) -> None:
    """Builds a SIGNED synthetic fraction array (v4's real target minus a
    mid-range constant, so it swings through zero often, but clipped well
    away from any leverage/liquidation edge) and confirms
    `SignedHybridBroker` (spot_frac==0 throughout) reproduces the real
    engine's `PaperBroker` bit-for-bit -- the decisive validation of the
    two new code paths this round needs (sign-flip close/reopen, signed
    liquidation formula) against the project's own most-trusted broker,
    not a hand-derived expectation.
    """
    from tradebot.engine import run_backtest

    fut = fut_market()
    lo = int(df.index.searchsorted(start))
    hi = int(df.index.searchsorted(end, side="right"))
    warmup = KellyRegimeV4().warmup
    pre = prefix_bars(df, lo, warmup)
    frame = df.iloc[lo - pre: hi]

    raw_target = compute_target(frame)
    synthetic = np.clip(raw_target - 0.35, -0.35, 0.6)  # signed, bounded well inside 5x headroom

    def signed_only(sub_frame: pd.DataFrame) -> RouteFn:
        # sub_frame is `frame` itself (run_signed_hybrid_backtest slices
        # once more internally with the same warmup/prefix convention);
        # recompute the identical synthetic array on that exact slice.
        t = np.clip(compute_target(sub_frame) - 0.35, -0.35, 0.6)
        return RouteFn(np.zeros_like(t), t)

    hybrid = run_signed_hybrid_backtest(df, signed_only, spot_market(), fut, funding=funding,
                                        start=start, end=end)

    strat = _SignedTargetStrategy(synthetic)
    oracle_raw = run_backtest(strat, frame, fut, 1_000.0, trade_start=pre, funding=funding)
    from dataclasses import replace as _replace
    oracle = oracle_raw if pre == 0 else _replace(
        oracle_raw, equity=oracle_raw.equity.iloc[pre:], df=oracle_raw.df.iloc[pre:])

    rel = abs(hybrid.final_balance - oracle.final_balance) / max(oracle.final_balance, 1.0)
    fee_rel = abs(hybrid.fees_paid - oracle.fees_paid) / max(oracle.fees_paid, 1.0)
    fund_rel = abs(hybrid.funding_paid - oracle.funding_paid) / max(abs(oracle.funding_paid), 1.0)
    print(f"[signed-oracle] hybrid=${hybrid.final_balance:,.2f} oracle=${oracle.final_balance:,.2f} "
          f"rel_diff={rel:.2e} fee_rel={fee_rel:.2e} funding_rel={fund_rel:.2e} "
          f"liquidated: hybrid={hybrid.liquidated} oracle={oracle.liquidated}")
    assert rel < 1e-6, f"signed futures leg diverges from PaperBroker oracle: {rel:.2e}"
    assert hybrid.liquidated == oracle.liquidated, "liquidation flag mismatch vs oracle"


def _liquidation_price_signed_check() -> None:
    """`SignedHybridBroker.liquidation_price()` must reduce EXACTLY to
    `PaperBroker.liquidation_price()` when `pos_spot=0`, for both signs of
    `pos_fut` -- checked against real `PaperBroker` instances (the
    oracle), across several hand-picked (cash, pos, entry) fixtures, not
    merely by algebra in a docstring.
    """
    fut = fut_market()
    fixtures = [
        (1_000.0, 2.0, 30_000.0),
        (1_000.0, -2.0, 30_000.0),
        (500.0, 0.05, 60_000.0),
        (500.0, -0.05, 60_000.0),
        (2_000.0, 10.0, 25_000.0),
        (2_000.0, -10.0, 25_000.0),
    ]
    for cash, pos, entry in fixtures:
        oracle = PaperBroker(market=fut, start_balance=cash)
        oracle.cash, oracle.pos, oracle.entry = cash, pos, entry
        p_oracle = oracle.liquidation_price()

        hyb = SignedHybridBroker(spot=spot_market(), fut=fut, start_balance=cash)
        hyb.cash, hyb.pos_spot, hyb.entry_spot = cash, 0.0, 0.0
        hyb.pos_fut, hyb.entry_fut = pos, entry
        p_hybrid = hyb.liquidation_price()

        assert (p_oracle is None) == (p_hybrid is None), (cash, pos, entry, p_oracle, p_hybrid)
        if p_oracle is not None:
            rel = abs(p_hybrid - p_oracle) / p_oracle
            print(f"[liq-formula] pos_fut={pos:+.3f} oracle={p_oracle:,.2f} "
                  f"signed={p_hybrid:,.2f} rel_diff={rel:.2e}")
            assert rel < 1e-9, f"liquidation formula diverges from PaperBroker oracle: {rel:.2e}"


def _causality_truncation_check(df: pd.DataFrame, funding: pd.Series | None,
                                start: str, end: str, base: float) -> None:
    """`r145_shared`'s own truncation-causality style check, run against
    the constant-base route: truncating the tail must not change any bar
    strictly before the cut.
    """
    def route_builder(frame: pd.DataFrame) -> RouteFn:
        return route_constant_base(compute_target(frame), base)

    full = run_signed_hybrid_backtest(df, route_builder, spot_market(), fut_market(),
                                      funding=funding, start=start, end=end)
    truncated_df = df.loc[:df.index[df.index.searchsorted(end, side="right") - 2]]
    trunc = run_signed_hybrid_backtest(truncated_df, route_builder, spot_market(), fut_market(),
                                       funding=funding, start=start)
    overlap = trunc.equity.index
    max_diff = float((full.equity.reindex(overlap) - trunc.equity).abs().max())
    print(f"[causality-truncation] max |diff| over {len(overlap)} shared bars: {max_diff:.2e}")
    assert max_diff < 1e-6, f"truncating the tail changed a bar strictly before it: {max_diff:.2e}"


def _liquidation_safety_scan(df: pd.DataFrame, funding: pd.Series | None) -> None:
    """Runs every candidate BASE (constant) and halflife (EWMA) over
    BOTH inner-train and inner-validation and confirms the futures leg
    never gets liquidated -- clause (5) of the frozen decision rule,
    checked here as a feasibility scan (not itself the promotion gate,
    which each branch re-runs on its own selected config).
    """
    fut, spot = fut_market(), spot_market()
    windows = [("inner-train", "2017-01-01", INNER_TRAIN_END),
               ("inner-validation", INNER_VAL_START, INNER_VAL_END)]
    for label, start, end in windows:
        for base in (0.10, 0.15, 0.25, 0.43):
            def rb(frame, base=base):
                return route_constant_base(compute_target(frame), base)
            r = run_signed_hybrid_backtest(df, rb, spot, fut, funding=funding, start=start, end=end)
            print(f"[liq-scan] {label} base={base:.2f}: liquidated={r.liquidated} "
                  f"fills_spot={r.fills_spot} fills_fut={r.fills_fut} "
                  f"fees=${r.fees_paid:,.2f} funding=${r.funding_paid:,.2f}")
            assert not r.liquidated, f"constant base={base} liquidated during {label}"
        for hl in (60.0, 180.0, 365.0):
            def rb2(frame, hl=hl):
                return route_ewma_base(compute_target(frame), hl)
            r = run_signed_hybrid_backtest(df, rb2, spot, fut, funding=funding, start=start, end=end)
            print(f"[liq-scan] {label} ewma_hl={hl:.0f}d: liquidated={r.liquidated} "
                  f"fills_spot={r.fills_spot} fills_fut={r.fills_fut} "
                  f"fees=${r.fees_paid:,.2f} funding=${r.funding_paid:,.2f}")
            assert not r.liquidated, f"ewma halflife={hl} liquidated during {label}"


# --------------------------------------------------------- reachable-n report

def reachable_n_report(df: pd.DataFrame, funding: pd.Series | None, base: float) -> dict:
    """ROUTINE.md Step 2's mandatory reachable-power check, against the
    ACTUAL decision-rule statistic (`annualized_sharpe`, paired
    block-bootstrapped exactly as `paired_bootstrap` will be used at
    Step 3/4), not a hand-rolled proxy. A naive "std of the daily return
    DIFFERENCE" framing (this file's first draft) overstates the true
    noise: `paired_bootstrap` resamples the SAME blocks for both arms, so
    the common price-path randomness both arms share cancels out of the
    Sharpe-DIFFERENCE statistic far more than a plain difference-series
    std would suggest (R-145's own documented reasoning) -- measured
    directly here, not assumed, exactly as R-67/R-68 measured it on the
    panel's own COST-axis comparison rather than asserting an order of
    magnitude.

    Reports the real `paired_bootstrap` point estimate/CI on inner-train
    (mean_block=30, matching this project's own standard), plus a
    block-bootstrap CI-width scaling projection (half-width ~ 1/sqrt(n)
    under stationarity, the same scaling law R-67/R-68 used) for how many
    days would be needed for the CURRENT point estimate's sign to clear
    significance, so a candidate whose point estimate is already
    unfavourable is reported as genuinely unreachable rather than
    "just needs more data."
    """
    fut = fut_market()

    def rb(frame):
        return route_constant_base(compute_target(frame), base)

    hybrid = run_signed_hybrid_backtest(df, rb, spot_market(), fut, funding=funding,
                                        start="2017-01-01", end=INNER_TRAIN_END)
    baseline = plain_v4_period(df, fut, funding, "2017-01-01", INNER_TRAIN_END)

    r_h = daily_returns(hybrid.equity).to_numpy()
    r_b = daily_returns(baseline.equity).to_numpy()
    n = min(len(r_h), len(r_b))
    r_h, r_b = r_h[:n], r_b[:n]

    result = paired_bootstrap(r_h, r_b, annualized_sharpe, mean_block=30.0, n_boot=2000, seed=182)
    point = result.stat_a - result.stat_b
    half_width = (result.diff.hi - result.diff.lo) / 2.0

    holdout_days = None
    if len(df):
        oos_pos = df.index.searchsorted(OOS_START)
        if oos_pos < len(df):
            holdout_days = float((df.index[-1] - df.index[oos_pos]).days)

    # CI half-width ~ C / sqrt(n) => n_needed = n_measured * (half_width / target_half_width)^2,
    # where target_half_width = |point| (the width at which the CI's near
    # edge just touches zero, keeping the point estimate's own sign).
    if point == 0:
        n_needed = float("inf")
    else:
        target_half_width = abs(point)
        n_needed = n * (half_width / target_half_width) ** 2

    report = {
        "base": base,
        "n_days_measured": n,
        "sharpe_hybrid": result.stat_a,
        "sharpe_baseline": result.stat_b,
        "d_sharpe_point": point,
        "d_sharpe_ci": (result.diff.lo, result.diff.hi),
        "ci_half_width": half_width,
        "sign_favorable": point > 0,
        "n_days_needed_for_significance": n_needed,
        "inner_validation_days": 730,
        "holdout_days_available": holdout_days,
        "reachable_vs_inner_val": (point > 0) and (n_needed <= 730),
        "reachable_vs_holdout": (point > 0) and (holdout_days is not None and n_needed <= holdout_days),
    }
    return report


# --------------------------------------------------------------------- main

if __name__ == "__main__":
    btc, btc_funding, btc_label = load_btc()
    print(f"BTC: {len(btc):,} bars ({btc_label}), funding: "
          f"{'present' if btc_funding is not None else 'MISSING'}\n")

    print("=== liquidation-formula self-check (synthetic fixtures vs PaperBroker oracle) ===")
    _liquidation_price_signed_check()

    print("\n=== degenerate-equivalence self-check (BTC, inner-validation) ===")
    _degenerate_equivalence_check(btc, btc_funding, INNER_VAL_START, INNER_VAL_END)

    print("\n=== signed-futures-leg oracle check (BTC, inner-validation, vs real engine) ===")
    _signed_futures_oracle_check(btc, btc_funding, INNER_VAL_START, INNER_VAL_END)

    print("\n=== causality truncation self-check (BTC, inner-validation, base=0.25) ===")
    _causality_truncation_check(btc, btc_funding, INNER_VAL_START, INNER_VAL_END, base=0.25)

    print("\n=== liquidation safety scan (all BASE/halflife candidates, both inner windows) ===")
    _liquidation_safety_scan(btc, btc_funding)

    print("\n=== cost ratio (R-145 criterion 2: extra fees / funding saved), inner-train ===")
    for base in (0.10, 0.15, 0.25, 0.43):
        for spot_fee, tier_label in ((SPOT_FEE_BASE, "0.10%"), (SPOT_FEE_REAL, "0.40%")):
            spot_mkt = spot_market(spot_fee)
            fut_mkt_ = fut_market()

            def rb(frame, base=base):
                return route_constant_base(compute_target(frame), base)

            hybrid = run_signed_hybrid_backtest(btc, rb, spot_mkt, fut_mkt_, funding=btc_funding,
                                                start="2017-01-01", end=INNER_TRAIN_END)
            baseline = plain_v4_period(btc, fut_mkt_, btc_funding, "2017-01-01", INNER_TRAIN_END)
            extra_fees = hybrid.fees_paid - baseline.fees_paid
            funding_saved = baseline.funding_paid - hybrid.funding_paid
            ratio = extra_fees / funding_saved if funding_saved else float("inf")
            print(f"  base={base:.2f} tier={tier_label}: extra_fees=${extra_fees:,.2f} "
                  f"funding_saved=${funding_saved:,.2f} ratio={ratio:.3f} "
                  f"(R-154 baseline @0.40%: {R154_BASELINE_RATIO_040})")

    print("\n=== reachable-n report (paired_bootstrap Sharpe-diff, inner-train), all BASE candidates ===")
    for base in (0.10, 0.15, 0.25, 0.43):
        rep = reachable_n_report(btc, btc_funding, base=base)
        print(f"\n  --- base={base:.2f} ---")
        for k, v in rep.items():
            print(f"  {k}: {v}")

    print("\nAll self-checks passed.")
