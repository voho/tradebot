"""R-151 frozen pre-registration: hold trading precision constant across
legs of different leverage in `HybridBroker` — backlog item **B-44**.

**This is a methodology/infrastructure round, not a strategy claim.** It
reads no holdout bar, registers no strategy, and touches no file under
`src/tradebot/`. Its whole output is: does B-44's named defect reproduce,
does B-44's own proposed fix remove it, and did the defect change any
verdict already recorded?

---

**The backlog item, verbatim in substance (D. Backlog, B-44, filed by
R-145).** `HybridBroker` (`experiments/r145_shared.py`, a two-leg
spot+futures harness sharing one cash ledger) does not hold trading
precision constant across legs of different leverage: its reused per-leg
deadband (`MarketSpec.deadband`) is compared against each leg's OWN
leveraged max notional —

    max_notional = eq * market.leverage          # r145_shared._execute_leg
    if |delta| * price < market.deadband * max_notional: return

— so at a shared equity `eq` the unlevered spot leg ignores re-targets
below `0.05 * eq` while the 5x futures leg ignores everything below
`0.25 * eq`. A route that puts meaningful notional through both legs
therefore re-targets on much smaller absolute moves than either
single-venue baseline does, producing more fills and a measurably
different equity path even when the nominal routed target is bit-identical
every bar. R-145 measured the consequence as a **1.90–4.81% relative
realized-volatility mismatch** against its plain all-futures baseline, at
every one of its six cells, and failed its own pre-registered criterion
(3) (`EXPOSURE_MATCH_TOL_PCT = 1.0`) on it. B-44's own suggested fix, also
verbatim: *"A fix, if ever needed, likely wants the deadband compared
against a SHARED notional base (e.g. combined equity, not each leg's own
max) rather than each leg's own leveraged ceiling."*

**Why this is worth a round rather than a one-line patch.** R-145 was
right not to fix it in place — `r145_shared.py` was frozen before either
branch ran, and editing it after seeing results is the goalpost-move
ROUTINE.md forbids. So `r145_shared.py` **is not modified by this file
either**; it is imported, and the fixed broker is a subclass here. That
also makes the fix testable in the only way that matters: `"leg"` mode
must reproduce the frozen harness bit-for-bit, so the ONLY difference
between the arms below is the deadband base.

This is the same shape of question B-43/R-134 asked one level up
(`broker.REBALANCE_DEADBAND`, single-leg): an *evaluability* defect is
real and worth fixing on its own terms, and separately one must check
whether the verdict it contaminated actually turns on it. R-134's answer
there was "the defect was real, the verdict was not an artifact of it".
That is a possible outcome here too and is an acceptable one.

**Constraint attacked.** COST (methodology gap, not a market-constraint
code) — same classification the backlog row carries. Not a duplicate of
R-134/B-43 (single-leg `broker.py`, one leverage, no cross-leg asymmetry
possible), not of R-145 itself (which measured the defect but could not
touch it), not of R-72 (which asked whether *lowering* the single-venue
deadband helps `kelly_regime_v4`'s economics — a different question
about a different number).

---

**THE THREE ARMS (frozen; only the deadband/haircut base differs).**

    A  "leg"          per-leg base — R-145's frozen behaviour, reproduced.
    B  "shared"       both legs' deadband compared against ONE base,
                      `eq * DEADBAND_REF_LEVERAGE` (= the futures leg's
                      5x, i.e. the plain all-futures baseline's own base).
                      This is B-44's proposed fix.
    C  "shared+hc"    B, plus a shared `_max_qty` fee/slippage haircut.

Arm C exists because there is a SECOND, structurally identical cross-
leverage asymmetry in the same three lines, and it is declared here,
before any number, so that finding it in the residual later is not a
post-hoc story. `_max_qty`'s haircut is `1 - (fee_rate + slip) * leverage`
— `1 - 0.001*1 = 0.9990` on the spot leg against `1 - 0.0005*5 = 0.9975`
on the futures leg — so an identical routed `frac` buys a *different
desired notional* depending on which leg carries it, before any deadband
is consulted. It is ~0.15% (and it flips sign at the 0.40% tier:
`1 - 0.004*1 = 0.9960` vs `0.9975`), an order of magnitude below the
1.90–4.81% R-145 measured, so it is a candidate for the residual and not
for the headline. C is a **diagnostic decomposition arm**, not a
competing fix.

**Choice of shared base, and its disclosed cost.** `DEADBAND_REF_LEVERAGE
= FUT_LEVERAGE` (5.0) rather than `1.0`, because the comparison R-145
failed is against a *plain all-futures* baseline, whose own dollar
deadband is `0.05 * eq * 5`. Matching that base is what makes the two arms
of that comparison equally precise. The price, stated now rather than
discovered later: under `"shared"` the **all-spot degenerate route stops
reproducing plain spot v4**, because plain spot's own base is `0.05 * eq *
1`. That is not a bug, it is the arithmetic of holding precision constant
across two venues that genuinely have different ceilings — but it must be
QUANTIFIED, not waved through (check T3b below), because R-145's harness
kill bar depends on the degenerate checks passing.

---

**PRE-REGISTERED DECISION RULE (frozen before any arm is run).**

Fidelity gates — if either fails, nothing below is trusted and the round
is BLOCKED on a harness bug, not reported as a finding:

- **T3a (equivalence preserved).** Under arm B, the all-futures degenerate
  route reproduces `plain_v4_period(..., fut_market())` to relative
  `< 1e-6` — the same tolerance `r145_shared._degenerate_equivalence_check`
  asserts.
- **T4 (faithful reimplementation).** Arm A reproduces the frozen
  `r145_shared.run_hybrid_backtest` **exactly** — final balance, fees,
  funding, fills, and every bar of the equity curve — at the primary
  threshold, both fee tiers. Tolerance: `0.0` on fills/liquidation and
  `< 1e-9` relative on the three dollar figures and `max|Δequity|`.

Headline rule — the fix is adopted iff, on BTC inner-validation
(2021-01-01 → 2022-12-31), across **all six cells** (3 thresholds
`CONSERVATIVE_THRESHOLDS` × 2 fee tiers):

- **ADOPT** — arm B's realized-volatility mismatch against the plain
  all-futures baseline is `<= EXPOSURE_MATCH_TOL_PCT` (1.0%) in **every**
  cell, i.e. R-145's criterion (3) now passes where it failed 6/6; and
  T3a and T4 both hold.
- **PARTIAL** — T3a/T4 hold and arm B cuts the mismatch by `>= 50%`
  (relative to arm A) in every cell, but at least one cell still exceeds
  1.0%. Then the defect is real and B-44's fix is a genuine improvement
  but not the whole story; arm C's residual decomposition is reported and
  whatever is left is named, not glossed.
- **REJECT** — arm B's median mismatch reduction across the six cells is
  `< 25%`, or T3a/T4 fail. Then B-44's proposed fix is the wrong fix,
  which is a reportable negative and leaves B-44 open with new evidence
  attached rather than closed.

Secondary rule, declared now and scored regardless of the headline —
**did the defect change R-145's verdict?** R-145 rejected on criteria (1)
`d_sharpe >= +0.20` with CI excluding zero, and (2) turnover cost `< 50%`
of funding saved, at the primary threshold. Both are recomputed under arm
B with the identical measurement code (`r145_conservative.run_cell`'s own
`paired_daily` / `annualized_vol` / `inference.paired_bootstrap`,
imported, not reimplemented):

- **VERDICT UNCHANGED** if criteria (1) and (2) still fail under arm B at
  the primary threshold — i.e. the rejection did not turn on the defect.
- **VERDICT CONTAMINATED** if either criterion flips to passing under
  arm B. That does NOT promote anything: it would mean R-145's round must
  be re-run properly under a corrected harness, and this round's job would
  become filing that as a new backlog item, not reading a holdout bar.
  No holdout is read under either outcome — this file contains no code
  path that can slice at or after `OOS_START`.

**What would make this fail (named before any code ran).** (a) The
mismatch is not carried by the deadband at all — it survives arm B
essentially intact, because it comes from the haircut (arm C), from
`min_notional`, or from the fee/funding cash path interacting with a
two-leg ledger; (b) the shared base *breaks* the all-futures equivalence
(T3a) because the deadband is not in fact the only place `leverage` enters
the throttle decision; (c) arm A does not reproduce the frozen harness
(T4), meaning the subclass diverges somewhere and none of the numbers
mean anything.

**n-check (R-78's discipline, done before running).** This is not a
statistical-power question: T3a/T4 and the mismatch percentages are
*deterministic* properties of one fixed data path, not estimates with
sampling error, so no `n` is needed for the headline rule. The only
sampled quantity is the secondary rule's `d_sharpe` CI, which reuses
R-145's own already-powered comparison (730 inner-validation days, paired
on an identical target path) unchanged — no new threshold is being set
against noise here.

**Configs evaluated (declared before running, for the trials count).**
3 arms × 3 thresholds × 2 fee tiers = **18** hybrid runs, plus 1 plain
all-futures baseline, 2 degenerate-equivalence routes × 2 arms = 4, and
2 fidelity (T4) runs = **25 total**. No parameter is searched: every
threshold is R-145's own frozen `CONSERVATIVE_THRESHOLDS`, and the two
new bases are structural switches, not fitted values.

**Holdout counter: +0.** Inner-validation only, by construction.
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

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

from r145_shared import (  # noqa: E402
    FUT_LEVERAGE,
    HybridBroker,
    HybridResult,
    RouteFn,
    compute_target,
    fut_market,
    plain_v4_period,
    spot_market,
)
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

# --------------------------------------------------------------- frozen knobs

#: The shared base's reference leverage. 5.0 == the futures leg == the
#: plain all-futures baseline's own deadband base. See the module
#: docstring on why this and not 1.0, and what it costs.
DEADBAND_REF_LEVERAGE = FUT_LEVERAGE

#: Arm labels, frozen. Order matters only for reporting.
ARMS = ("leg", "shared", "shared+hc")

#: Fidelity tolerances from the decision rule above.
EQUIV_TOL = 1e-6      # T3a: all-futures degenerate route vs plain futures v4
FIDELITY_TOL = 1e-9   # T4: arm A vs the frozen r145 harness

#: Headline rule thresholds from the decision rule above.
ADOPT_TOL_PCT = 1.0        # == r145_shared.EXPOSURE_MATCH_TOL_PCT
PARTIAL_MIN_REDUCTION = 0.50
REJECT_MAX_MEDIAN_REDUCTION = 0.25


# ------------------------------------------------------------------ broker

@dataclass
class HybridBrokerV2(HybridBroker):
    """`r145_shared.HybridBroker` with a configurable trading-precision
    base — the B-44 fix — and per-leg throttle instrumentation.

    Everything except the two `deadband_base`/`haircut_base` switches is
    the parent's own arithmetic, copied verbatim from
    `HybridBroker._execute_leg` (the method must be overridden in full
    because the base enters mid-body; T4 in this module's decision rule
    exists precisely to prove the copy did not drift).

    `deadband_base`:
      - ``"leg"``    — `eq * market.leverage`, the frozen R-145 behaviour.
      - ``"shared"`` — `eq * DEADBAND_REF_LEVERAGE` for BOTH legs, so one
        dollar threshold governs a re-target wherever it is routed.

    `haircut_base`:
      - ``"leg"``    — `1 - (fee_rate + slip) * market.leverage`, frozen.
      - ``"shared"`` — the reference leg's haircut for both, so an
        identical routed `frac` buys an identical desired notional
        wherever it is routed. Diagnostic arm C only.
    """

    deadband_base: str = "leg"
    haircut_base: str = "leg"
    ref_leverage: float = DEADBAND_REF_LEVERAGE
    ref_fee_rate: float | None = None  # defaults to the futures leg's fee

    # -- instrumentation: what the throttle actually did, per leg --------
    retargets_spot: int = field(init=False, default=0)
    retargets_fut: int = field(init=False, default=0)
    absorbed_spot: int = field(init=False, default=0)
    absorbed_fut: int = field(init=False, default=0)
    absorbed_notional_spot: float = field(init=False, default=0.0)
    absorbed_notional_fut: float = field(init=False, default=0.0)
    threshold_sum_spot: float = field(init=False, default=0.0)
    threshold_sum_fut: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.deadband_base not in ("leg", "shared"):
            raise ValueError(f"deadband_base must be 'leg' or 'shared', got {self.deadband_base!r}")
        if self.haircut_base not in ("leg", "shared"):
            raise ValueError(f"haircut_base must be 'leg' or 'shared', got {self.haircut_base!r}")
        if self.ref_fee_rate is None:
            self.ref_fee_rate = self.fut.fee_rate

    # ------------------------------------------------------------------
    def _execute_leg(self, leg: str, frac: float, price: float) -> None:
        if frac < 0:
            raise ValueError(f"HybridBrokerV2 is long-only; got frac={frac!r} for {leg}")
        pos, entry, market = ((self.pos_spot, self.entry_spot, self.spot) if leg == "spot"
                              else (self.pos_fut, self.entry_fut, self.fut))
        eq = self.equity(price)
        target_equiv = min(1.0, max(0.0, frac / max(market.leverage, 1e-9)))
        slip = self.slippage_bps / 10_000.0

        # --- the second cross-leverage asymmetry (arm C) ---------------
        if self.haircut_base == "leg":
            haircut = max(0.0, 1.0 - (market.fee_rate + slip) * market.leverage)
        else:
            haircut = max(0.0, 1.0 - (self.ref_fee_rate + slip) * self.ref_leverage)

        max_qty = eq * market.leverage * haircut / price if eq > 0 and price > 0 else 0.0
        desired = max_qty * target_equiv
        delta = desired - pos

        # --- the B-44 defect, and its fix (arm B) ----------------------
        if self.deadband_base == "leg":
            base_notional = eq * market.leverage
        else:
            base_notional = eq * self.ref_leverage

        if target_equiv != 0.0 and pos != 0.0 and base_notional > 0:
            if leg == "spot":
                self.retargets_spot += 1
                self.threshold_sum_spot += market.deadband * base_notional
            else:
                self.retargets_fut += 1
                self.threshold_sum_fut += market.deadband * base_notional
            if abs(delta) * price < market.deadband * base_notional:
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


@dataclass
class HybridResultV2(HybridResult):
    """`HybridResult` plus the throttle instrumentation, so a future round
    can read absorption directly instead of inferring it from fill counts
    (the R-131 "read a metric's definition before dividing by it" rule:
    fills and re-targets are different units and differ by ~10x here).
    """

    retargets_spot: int = 0
    retargets_fut: int = 0
    absorbed_spot: int = 0
    absorbed_fut: int = 0
    absorbed_notional_spot: float = 0.0
    absorbed_notional_fut: float = 0.0
    mean_threshold_spot: float = float("nan")
    mean_threshold_fut: float = float("nan")
    #: Realized combined notional / equity at each bar close, recorded only
    #: when `run_hybrid_backtest_v2(..., record_exposure=True)`. Read-only
    #: instrumentation added AFTER the freeze to diagnose this round's own
    #: residual; it is off by default, so every scored arm above runs the
    #: identical code path it was frozen with (T4/T3a are re-run to confirm
    #: that, and both still report 0.000e+00).
    exposure: pd.Series | None = None


# ------------------------------------------------------------------ runner

def run_hybrid_backtest_v2(
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
    deadband_base: str = "leg",
    haircut_base: str = "leg",
    record_exposure: bool = False,
) -> HybridResultV2:
    """`r145_shared.run_hybrid_backtest`'s per-bar loop, copied verbatim,
    with `HybridBroker` swapped for `HybridBrokerV2`.

    The copy is deliberate and is the reason T4 exists: the frozen file is
    not edited, and arm A ("leg"/"leg") is asserted equal to the frozen
    runner bit-for-bit before any arm-B number is read. See that function's
    own docstring for why `route_builder` receives the already-sliced
    frame (unbounded-memory EWM) and why the order decision is
    edge-triggered (`KellyRegime.on_bar` parity across a warmup plateau) —
    both properties are preserved here unchanged.
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

    broker = HybridBrokerV2(spot=spot_mkt, fut=fut_mkt, start_balance=start_balance,
                            slippage_bps=slippage_bps, deadband_base=deadband_base,
                            haircut_base=haircut_base)

    equity = np.zeros(n, dtype=float)
    exposure = np.zeros(n, dtype=float) if record_exposure else None
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
            raise ValueError(f"HybridBrokerV2: equity became non-finite at bar {i} ({index[i]})")
        equity[i] = eq
        if exposure is not None:
            exposure[i] = ((broker.pos_spot + broker.pos_fut) * closes[i] / eq
                           if eq > 0 else 0.0)

        last_bar = i == n - 1
        if not broker.dead and not last_bar and i >= prefix:
            changed = (i == 0
                      or abs(route.spot_frac[i] - route.spot_frac[i - 1]) > 1e-9
                      or abs(route.fut_frac[i] - route.fut_frac[i - 1]) > 1e-9)
            if changed:
                pending = route(i)

    eq_series = pd.Series(equity, index=index, name="equity")
    exp_series = (pd.Series(exposure, index=index, name="exposure")
                  if exposure is not None else None)
    if prefix:
        eq_series = eq_series.iloc[prefix:]
        if exp_series is not None:
            exp_series = exp_series.iloc[prefix:]

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
        exposure=exp_series,
    )


def arm_kwargs(arm: str) -> dict:
    """The frozen arm -> (deadband_base, haircut_base) mapping."""
    if arm == "leg":
        return dict(deadband_base="leg", haircut_base="leg")
    if arm == "shared":
        return dict(deadband_base="shared", haircut_base="leg")
    if arm == "shared+hc":
        return dict(deadband_base="shared", haircut_base="shared")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


# --------------------------------------------------------------- self-checks

def degenerate_all_futures(frame: pd.DataFrame) -> RouteFn:
    t = compute_target(frame)
    return RouteFn(np.zeros_like(t), t.copy())


def degenerate_all_spot(frame: pd.DataFrame) -> RouteFn:
    t = compute_target(frame)
    return RouteFn(np.clip(t, 0.0, None), np.zeros_like(t))


__all__ = [
    "ADOPT_TOL_PCT",
    "ARMS",
    "DEADBAND_REF_LEVERAGE",
    "EQUIV_TOL",
    "FIDELITY_TOL",
    "HybridBrokerV2",
    "HybridResultV2",
    "PARTIAL_MIN_REDUCTION",
    "REJECT_MAX_MEDIAN_REDUCTION",
    "arm_kwargs",
    "degenerate_all_futures",
    "degenerate_all_spot",
    "fut_market",
    "plain_v4_period",
    "run_hybrid_backtest_v2",
    "spot_market",
]
