"""R-154 frozen pre-registration: fixing **B-45** (silent exposure
truncation on an over-1.0x spot leg) and **B-46** (per-leg rather than
aggregate throttling) in `HybridBroker`'s venue-routing harness.

**This is a methodology/infrastructure round, not a strategy claim** --
same classification as R-151, which named both defects and is this
round's own "next step." It reads no holdout bar, registers no strategy,
and touches nothing under `src/tradebot/`.

**Direction.** Both backlog rows, verbatim in substance:

- **B-45**: `HybridBrokerV2._execute_leg`'s `target_equiv = min(1.0, frac
  / market.leverage)` caps the unlevered spot leg at 1.0x and the
  overflow (`spot_frac > 1.0`, only reachable when the route's
  `threshold` exceeds 1.0) is dropped, not rerouted to the futures leg
  that could carry it. R-151 measured this at threshold=1.2: `spot_frac >
  1.0` on 2.26% of bars, mean truncated exposure 0.175x.
- **B-46**: each leg's re-target is throttled against the deadband
  **independently**, so a combined move the single-venue baseline would
  execute can be absorbed twice -- once per leg. R-151 measured 12 of 266
  target moves (4.51%) requiring both legs to move, each leg then seeing
  only 67.2% of the aggregate move on those bars, producing systematic
  under-holding (-3.9% at threshold=0.8).

Both were named, measured and explicitly left unfixed by R-151 ("no
choice of base fixes this... That fourth arm was deliberately not built
or scored... an addition after a freeze may only tighten a bar, never
loosen one"), then filed as this project's own next step.

**Attacks COST** (methodology gap, not a market-constraint code) -- the
classification both backlog rows already carry. **Not a duplicate** of
R-145 (diagnosed the base defect, B-44, but could not touch it -- frozen
before either branch ran), or of R-151 (fixed B-44 only; named B-45/B-46
without building a fix for either, by its own freeze discipline). Not a
strategy-improvement round: nine independent 08-26 research passes
(recorded in `docs/LEDGER.md` section D) exhaustively checked the
literature for a new `kelly_regime_v4` mechanism today and found nothing
that survives ROUTINE.md Step 1's four-question filter -- this round
picks up the backlog's one remaining OPEN, LOW, unblocked, non-duplicate
item instead of manufacturing an eleventh redundant literature sweep.

**Where genuine design freedom exists, and where it does not -- stated
before any code, so neither branch invents a difference that is not
really there.** B-45 has exactly one correct fix: the overflow is a
*conservation* requirement (the route's `target` already encodes the
strategy's own intended total exposure; truncating it is a bug, not a
choice), so **both branches apply the identical redistribution**,
`spot_frac' = min(spot_frac, 1.0)`, `fut_frac' = fut_frac + max(0.0,
spot_frac - 1.0)` -- exactly B-45's own row text ("overflow pushed to the
levered leg"). Dressing this up as two competing "philosophies" would be
artificial, and the routine's own discipline is to say so rather than
manufacture a difference. **B-46 is where real freedom exists**: once a
combined re-target is decided as ONE gate rather than two, the gate still
needs a definition of "the size of the combined move," and there are (at
least) two economically distinct candidates that R-145's own
`route_fixed_threshold` can actually separate, because a target crossing
its threshold boundary can make one leg's frac rise while the other's
falls in the same bar:

- **Conservative**: gate on **gross** leg turnover, `sum(|delta_leg| *
  price for leg in gated_legs)` -- the direct, minimal generalization of
  the existing per-leg check to a shared threshold.
- **Novel**: gate on **net** combined exposure change, `abs(sum(delta_leg
  for leg in gated_legs)) * price` -- economically motivated by R-131's
  own rule ("read a metric's definition before dividing by it"): a
  rebalance that shifts notional from one venue to the other without
  changing the strategy's aggregate market exposure is not the same size
  of decision as one that changes it, and gross-sum throttling cannot
  tell the two apart.

Both reduce to the same number whenever only one leg is gated (the
common case), so they can only diverge on bars where BOTH legs move in
the same bar -- exactly the 4.51% R-151 measured, which is also exactly
where B-46 bites.

**Pre-registered falsification test (chosen now):** if the fixed
harness, run with `aggregate_throttle=False` (i.e. every fix disabled),
does not reproduce R-151's own arm-B (`deadband_base="shared"`) numbers
bit-for-bit (final balance, fees, funding, fill counts, at both fee
tiers, threshold=1.0), the refactor itself is broken and neither branch's
numbers can be trusted -- checked first, before any fixed-arm number is
read.

**Decision rule, frozen before either branch runs anything.** Measured
on BTC inner-validation (2021-01-01 -> 2022-12-31) only -- this file
freezes the rule but reads no holdout bar. Both branches build on R-151's
own adopted partial fix (`deadband_base="shared"`, B-44), at all three of
`CONSERVATIVE_THRESHOLDS = (0.8, 1.0, 1.2)` and both fee tiers (0.10%,
0.40%) -- 6 cells each, same grid R-151 used.

1. **Fidelity gate** (both branches must pass before anything else is
   read): `aggregate_throttle=False, redistribute=identity` reproduces
   R-151's arm B to <1e-9 relative on final balance/fees/funding and
   exact fill counts, at every fee tier and threshold. All-futures
   degenerate route reproduces plain futures v4 to <1e-6 under every
   broker variant used below.
2. **Headline**: for each branch, compute the realized-volatility
   mismatch vs. the plain all-futures baseline (R-145's own criterion 3,
   its measurement code imported unchanged) at all 6 cells. A branch
   **ADOPTS** iff all 6 cells are `<= EXPOSURE_MATCH_TOL_PCT = 1.0%` (R-145's
   own tolerance); **PARTIAL** iff every cell improves over R-151's own
   published arm-B mismatch by `>= 25%` relative; **REJECT** iff the
   median improvement over arm B is `< 10%` or a fidelity gate fails.
   (Bar set at 25%/10% rather than R-151's 50%/25%: arm B already closed
   most of the gap at threshold=1.0, so the remaining room to improve is
   structurally smaller -- stated now, not fitted to a result.)
3. **Winner** between branches that ADOPT or PARTIAL: the one with the
   lower median relative-volatility mismatch across the 6 cells. A
   difference `< 0.1` percentage points (comparable to R-151's own
   arm-B/arm-C noise, 0.13pp) is a tie, broken in favor of **conservative**
   (gross-sum is this project's standing turnover convention everywhere
   else fees/turnover are measured, e.g. `Metrics.num_trades` /
   `len(result.fills)`; a novel metric earns adoption only by a real
   margin, not a coin flip).
4. **Secondary** (frozen, matches R-151's own secondary check): does
   either fix flip R-145's own inner-validation gate criteria (1) `d_sharpe
   >= +0.20` CI excluding zero and (2) turnover cost `< 50%` of funding
   saved, at either fee tier, using R-145's own `paired_bootstrap` /
   `annualized_vol` / `paired_daily`, imported unchanged? This is B-47
   (re-run R-145's conservative branch under the corrected harness),
   answered as a byproduct rather than a separate round.

**Configs evaluated (declared before running):** fidelity gate = 2 runs
(one per fee tier, threshold=1.0) x 2 (arm-B reproduction + all-futures
equivalence) = 4. Headline grid = 2 branches x 3 thresholds x 2 fee tiers
= 12. Two diagnostic decomposition arms (`+B45-only`, `+B46-only`, at the
primary threshold/tier only, matching R-151's own arm-C decomposition
style) = 2 x 2 fee tiers = 4. Secondary (B-47) = 2 branches x 2 fee tiers
= 4 (reuses headline-grid equity curves, still counted since it reads a
different statistic). **Total: 24 hybrid runs.** Zero parameters
searched -- every threshold is R-145's own frozen
`CONSERVATIVE_THRESHOLDS`; the two `_aggregate_delta` definitions are
structural, not tuned. **Holdout counter: +0** (inner-validation only).
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from tradebot.window import prefix_bars  # noqa: E402

from r145_shared import (  # noqa: E402
    CONSERVATIVE_THRESHOLDS,
    D_SHARPE_FLOOR,
    EXPOSURE_MATCH_TOL_PCT,
    INNER_VAL_END,
    INNER_VAL_START,
    RouteFn,
    TURNOVER_SAVINGS_KILL,
    compute_target,
    fut_market,
    load_btc,
    plain_v4_period,
    spot_market,
)
from r145_conservative import (  # noqa: E402
    FEE_TIERS,
    PRIMARY_THRESHOLD,
    annualized_vol,
    make_route_builder,
    paired_daily,
    plain_target_slice,
)
from r151_shared import (  # noqa: E402
    DEADBAND_REF_LEVERAGE,
    HybridBrokerV2,
    HybridResultV2,
)

# ---------------------------------------------------------------- constants

ADOPT_TOL_PCT = EXPOSURE_MATCH_TOL_PCT      # 1.0, R-145's own tolerance
PARTIAL_MIN_IMPROVEMENT = 0.25              # >= 25% cut vs R-151's own arm B
REJECT_MAX_MEDIAN_IMPROVEMENT = 0.10        # < 10% median cut -> REJECT
TIE_BAND_PCT = 0.1                          # points; ties default to conservative
FIDELITY_TOL = 1e-9
EQUIV_TOL = 1e-6

#: R-151's own published arm-B ("shared") mismatch table, cell for cell,
#: keyed (tier_label, threshold) -- reused as the baseline the headline
#: rule improves against, not re-derived (avoids a second source of
#: truth for numbers already measured and published).
ARM_B_MISMATCH_PCT = {
    ("base_0.10pct", 0.8): 3.5969,
    ("base_0.10pct", 1.0): 0.1294,
    ("base_0.10pct", 1.2): 1.7903,
    ("real_0.40pct", 0.8): 2.6827,
    ("real_0.40pct", 1.0): 0.7433,
    ("real_0.40pct", 1.2): 0.9392,
}


# ------------------------------------------------------------------ broker

@dataclass
class HybridBrokerV3(HybridBrokerV2):
    """`HybridBrokerV2` (B-44's shared-base fix) with two more switches,
    both defaulting OFF so the class degrades to an exact V2 arm-B
    reproduction until a subclass turns them on:

    - `aggregate_throttle`: if False (default), `fill()` throttles each
      leg against the shared-base deadband independently -- V2's own
      B-46 defect, reproduced bit-for-bit. If True, the combined
      re-target is gated ONCE (`_aggregate_delta`, overridden per
      branch) and both legs fill or both absorb together -- the B-46 fix.
    - `_redistribute`: identity by default (B-45's defect reproduced). A
      fixed subclass overrides it with the one correct redistribution
      (see module docstring).
    """

    aggregate_throttle: bool = False

    def _redistribute(self, spot_frac: float, fut_frac: float) -> tuple[float, float]:
        return spot_frac, fut_frac

    def _aggregate_delta(self, plans: dict, gated: list[str], price: float) -> float:
        """Default = gross leg turnover (conservative's own metric), so a
        plain `HybridBrokerV3(aggregate_throttle=True)` with no further
        override is already the conservative branch.
        """
        return sum(abs(plans[leg]["delta"]) for leg in gated) * price

    # ------------------------------------------------------------ planning

    def _plan_leg(self, leg: str, frac: float, price: float, eq: float) -> dict:
        pos, entry, market = ((self.pos_spot, self.entry_spot, self.spot) if leg == "spot"
                              else (self.pos_fut, self.entry_fut, self.fut))
        frac = max(0.0, frac)
        target_equiv = min(1.0, frac / max(market.leverage, 1e-9))
        slip = self.slippage_bps / 10_000.0
        if self.haircut_base == "leg":
            haircut = max(0.0, 1.0 - (market.fee_rate + slip) * market.leverage)
        else:
            haircut = max(0.0, 1.0 - (self.ref_fee_rate + slip) * self.ref_leverage)
        max_qty = eq * market.leverage * haircut / price if eq > 0 and price > 0 else 0.0
        desired = max_qty * target_equiv
        delta = desired - pos
        return dict(leg=leg, pos=pos, entry=entry, market=market,
                   target_equiv=target_equiv, delta=delta)

    def _base_notional(self, market, eq: float) -> float:
        return eq * market.leverage if self.deadband_base == "leg" else eq * self.ref_leverage

    # -------------------------------------------------------------- gating

    def _execute_leg_sequential(self, leg: str, frac: float, price: float) -> None:
        """`HybridBrokerV2._execute_leg`, copied verbatim (not rebuilt on
        top of `_plan_leg`): it re-fetches `self.equity(price)` fresh for
        EACH leg, so the futures leg's own plan is computed against the
        equity already updated by the spot leg's fill a few lines above.
        That sequential-equity dependency is V2's actual, specified
        behavior (never disclosed as a simplification, so not something
        to quietly change) -- reproducing it bit-for-bit here, rather
        than driving this path through `_plan_leg`'s single shared-eq
        snapshot, is what makes the fidelity gate below exact rather
        than off by the ~1e-5 relative equity drift a snapshot
        introduces once one leg has already paid a fee.
        """
        pos, entry, market = ((self.pos_spot, self.entry_spot, self.spot) if leg == "spot"
                              else (self.pos_fut, self.entry_fut, self.fut))
        eq = self.equity(price)
        target_equiv = min(1.0, max(0.0, frac / max(market.leverage, 1e-9)))
        slip = self.slippage_bps / 10_000.0
        if self.haircut_base == "leg":
            haircut = max(0.0, 1.0 - (market.fee_rate + slip) * market.leverage)
        else:
            haircut = max(0.0, 1.0 - (self.ref_fee_rate + slip) * self.ref_leverage)
        max_qty = eq * market.leverage * haircut / price if eq > 0 and price > 0 else 0.0
        desired = max_qty * target_equiv
        delta = desired - pos
        base = self._base_notional(market, eq)
        if target_equiv != 0.0 and pos != 0.0 and base > 0:
            thr = market.deadband * base
            if leg == "spot":
                self.retargets_spot += 1
                self.threshold_sum_spot += thr
            else:
                self.retargets_fut += 1
                self.threshold_sum_fut += thr
            if abs(delta) * price < thr:
                if leg == "spot":
                    self.absorbed_spot += 1
                    self.absorbed_notional_spot += abs(delta) * price
                else:
                    self.absorbed_fut += 1
                    self.absorbed_notional_fut += abs(delta) * price
                return
        if abs(delta) < 1e-12:
            return
        increasing = abs(pos + delta) > abs(pos)
        if increasing and abs(delta) * price < market.min_notional:
            return
        pos1, entry1, cash_delta, fee = self._transact_leg(delta, price, market.fee_rate, pos, entry)
        self.cash += cash_delta
        self.fees_paid += fee
        if leg == "spot":
            self.pos_spot, self.entry_spot = pos1, entry1
            self.fills_spot += 1
        else:
            self.pos_fut, self.entry_fut = pos1, entry1
            self.fills_fut += 1
        if self.equity(price) < 0.0:
            self.cash = 0.0
            self.pos_spot = self.entry_spot = 0.0
            self.pos_fut = self.entry_fut = 0.0
            self.dead = True

    def _aggregate_gate(self, plans: dict, gated: list[str], price: float, eq: float) -> list[str]:
        if not gated:
            return []
        base = self._base_notional(plans[gated[0]]["market"], eq)
        if base <= 0:
            return []
        deadbands = {plans[leg]["market"].deadband for leg in gated}
        assert len(deadbands) == 1, "aggregate gate assumes one shared deadband %"
        thr = deadbands.pop() * base
        combined = self._aggregate_delta(plans, gated, price)
        for leg in gated:
            if leg == "spot":
                self.retargets_spot += 1
                self.threshold_sum_spot += thr
            else:
                self.retargets_fut += 1
                self.threshold_sum_fut += thr
        if combined < thr:
            for leg in gated:
                p = plans[leg]
                if leg == "spot":
                    self.absorbed_spot += 1
                    self.absorbed_notional_spot += abs(p["delta"]) * price
                else:
                    self.absorbed_fut += 1
                    self.absorbed_notional_fut += abs(p["delta"]) * price
            return list(gated)
        return []

    # ------------------------------------------------------------- commit

    def _commit_leg(self, plan: dict, price: float) -> None:
        leg, pos, entry, market, delta = (plan["leg"], plan["pos"], plan["entry"],
                                          plan["market"], plan["delta"])
        if abs(delta) < 1e-12:
            return
        increasing = abs(pos + delta) > abs(pos)
        if increasing and abs(delta) * price < market.min_notional:
            return
        pos1, entry1, cash_delta, fee = self._transact_leg(delta, price, market.fee_rate, pos, entry)
        self.cash += cash_delta
        self.fees_paid += fee
        if leg == "spot":
            self.pos_spot, self.entry_spot = pos1, entry1
            self.fills_spot += 1
        else:
            self.pos_fut, self.entry_fut = pos1, entry1
            self.fills_fut += 1
        if self.equity(price) < 0.0:
            self.cash = 0.0
            self.pos_spot = self.entry_spot = 0.0
            self.pos_fut = self.entry_fut = 0.0
            self.dead = True

    # ---------------------------------------------------------------- fill

    def fill(self, spot_frac: float, fut_frac: float, price: float) -> None:
        if self.dead:
            return
        spot_frac, fut_frac = self._redistribute(spot_frac, fut_frac)
        if not self.aggregate_throttle:
            # B-46 NOT fixed: exact V2 sequential semantics (see
            # `_execute_leg_sequential`'s own docstring on why this is a
            # verbatim copy rather than routed through `_plan_leg`).
            self._execute_leg_sequential("spot", spot_frac, price)
            if self.dead:
                return
            self._execute_leg_sequential("fut", fut_frac, price)
            return
        # B-46 fixed: one joint decision against a single pre-trade
        # equity snapshot -- deliberately NOT sequential, since gating
        # both legs as one combined move requires planning them against
        # the same starting equity rather than letting the first leg's
        # own fee move the ground the second leg is judged against.
        eq = self.equity(price)
        plans = {"spot": self._plan_leg("spot", spot_frac, price, eq),
                "fut": self._plan_leg("fut", fut_frac, price, eq)}
        gated = [leg for leg, p in plans.items()
                if p["target_equiv"] != 0.0 and p["pos"] != 0.0]
        absorbed = self._aggregate_gate(plans, gated, price, eq)
        for leg in ("spot", "fut"):
            if leg in absorbed:
                continue
            self._commit_leg(plans[leg], price)
            if self.dead:
                return


class HybridBrokerConservative(HybridBrokerV3):
    """B-45 fix (shared) + B-46 fix, gross-leg-turnover gate."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.aggregate_throttle = True

    def _redistribute(self, spot_frac: float, fut_frac: float) -> tuple[float, float]:
        overflow = max(0.0, spot_frac - 1.0)
        return min(spot_frac, 1.0), fut_frac + overflow

    # _aggregate_delta inherited: gross sum (HybridBrokerV3's own default)


class HybridBrokerNovel(HybridBrokerV3):
    """B-45 fix (shared) + B-46 fix, net-combined-exposure gate."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.aggregate_throttle = True

    def _redistribute(self, spot_frac: float, fut_frac: float) -> tuple[float, float]:
        overflow = max(0.0, spot_frac - 1.0)
        return min(spot_frac, 1.0), fut_frac + overflow

    def _aggregate_delta(self, plans: dict, gated: list[str], price: float) -> float:
        net_qty = sum(plans[leg]["delta"] for leg in gated)
        return abs(net_qty) * price


class HybridBrokerB45Only(HybridBrokerV3):
    """Diagnostic decomposition arm: B-45 fixed, B-46 NOT fixed (still
    per-leg throttling). Matches R-151's own arm-C decomposition style --
    never a competing fix, only isolates which defect drives which part
    of the headline number.
    """

    def _redistribute(self, spot_frac: float, fut_frac: float) -> tuple[float, float]:
        overflow = max(0.0, spot_frac - 1.0)
        return min(spot_frac, 1.0), fut_frac + overflow


class HybridBrokerB46Only(HybridBrokerV3):
    """Diagnostic decomposition arm: B-46 fixed (gross-sum gate), B-45
    NOT fixed (overflow still silently dropped).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.aggregate_throttle = True


# ------------------------------------------------------------------ runner

def run_hybrid_backtest_v3(
    df: pd.DataFrame,
    route_builder: Callable[[pd.DataFrame], RouteFn],
    spot_mkt,
    fut_mkt,
    broker_factory: Callable[[], HybridBrokerV3],
    start_balance: float = 1_000.0,
    funding: pd.Series | None = None,
    start: object | None = None,
    end: object | None = None,
    slippage_bps: float = 0.0,
    warmup: int | None = None,
) -> HybridResultV2:
    """`r151_shared.run_hybrid_backtest_v2`'s per-bar loop, copied
    verbatim (fill previous bar's pending order at THIS bar's open, check
    liquidation, apply funding, record equity, decide the NEXT order from
    THIS bar's close), with the broker constructed by `broker_factory`
    instead of hardcoded to `HybridBrokerV2` -- the only change, so this
    function's own fidelity is exactly as trustworthy as V2's for any
    `broker_factory` that returns an unmodified `HybridBrokerV3` (the
    fidelity gate this module's docstring requires before anything else
    is read).
    """
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

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

    broker = broker_factory()

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
            raise ValueError(f"HybridBrokerV3: equity became non-finite at bar {i} ({index[i]})")
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

    return HybridResultV2(
        equity=eq_series,
        fees_paid=broker.fees_paid,
        funding_paid=broker.funding_paid,
        turnover_notional=broker.turnover_notional,
        liquidated=broker.dead,
        fills_spot=broker.fills_spot,
        fills_fut=broker.fills_fut,
        retargets_spot=broker.retargets_spot,
        retargets_fut=broker.retargets_fut,
        absorbed_spot=broker.absorbed_spot,
        absorbed_fut=broker.absorbed_fut,
        absorbed_notional_spot=broker.absorbed_notional_spot,
        absorbed_notional_fut=broker.absorbed_notional_fut,
        mean_threshold_spot=(broker.threshold_sum_spot / broker.retargets_spot
                             if broker.retargets_spot else float("nan")),
        mean_threshold_fut=(broker.threshold_sum_fut / broker.retargets_fut
                            if broker.retargets_fut else float("nan")),
    )


# --------------------------------------------------------------- self-checks

def degenerate_all_futures(frame: pd.DataFrame) -> RouteFn:
    t = compute_target(frame)
    return RouteFn(np.zeros_like(t), t.copy())


__all__ = [
    "ADOPT_TOL_PCT",
    "ARM_B_MISMATCH_PCT",
    "EQUIV_TOL",
    "FIDELITY_TOL",
    "PARTIAL_MIN_IMPROVEMENT",
    "REJECT_MAX_MEDIAN_IMPROVEMENT",
    "TIE_BAND_PCT",
    "HybridBrokerB45Only",
    "HybridBrokerB46Only",
    "HybridBrokerConservative",
    "HybridBrokerNovel",
    "HybridBrokerV3",
    "degenerate_all_futures",
    "run_hybrid_backtest_v3",
]
