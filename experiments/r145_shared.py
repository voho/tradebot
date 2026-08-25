"""R-145 shared pre-registration: funding-aware venue routing for
`kelly_regime_v4`'s own, unmodified target.

**Direction (attacks COST).** `kelly_regime_v4`'s `target` column (see
`tradebot/strategies/kelly_regime.py::KellyRegime.prepare`) is a single
causal fraction of equity notional the strategy wants to hold, currently
expressed through ONE venue (all spot, or all futures_5x). This round
splits that SAME target, unchanged, across two venues at once: the
`<=1.0x` core through spot (no funding, 0.10%/0.40% taker) and the `>1.0x`
excess (v4 never shorts -- the crowd vote floors at 0, "stands flat rather
than shorting" per its own docstring -- so the excess is always `>=0`)
through leveraged futures (funding-bearing, 0.05% taker). This attacks
COST by instrument choice, not by retiming or throttling the signal --
structurally different from every closed COST family: turnover
corridor/shrink (R-131, R-133), Uziel-El-Yaniv learned mixing (R-130),
Garleanu-Pedersen smoothing (R-64, R-128), Kelly no-trade bands (R-66-69,
R-89, R-90), patient-limit/taker-fallback execution (R-56, R-77, B-24) --
none touches which venue carries the notional. Also not a duplicate of
B-03 (delta-neutral spot-long/futures-short carry harvest, NEGATIVE R-39):
B-03 takes on a NEW zero-net-exposure carry bet; this keeps v4's exact
directional exposure and only changes financing. Not B-05 (funding
gate/flat, R-35/R-39) either: B-05 REDUCES exposure to dodge cost; this
reroutes venue, never shrinks the target.

**Citations.** Ackerer, Hugonnier & Jermann (2024/2025 working paper),
"Perpetual Futures Pricing" -- an arbitrage-free perpetual is priced as a
continuously-refinanced spot replication, with the funding rate as the
financing leg; the formal basis for treating venue choice as a pure
financing decision that leaves the target itself untouched. Schmeling,
Schrimpf & Todorov (2023) measure the crypto funding/carry premium
directly and find it decaying, negative in parts of 2024-2025 -- cited as
a guardrail: this design may only ever AVOID a cost already being paid on
an unchanged position, never harvest a new carry bet (that would
duplicate B-03).

**Step 1 falsification (named before any run):** (a) splitting into two
legs manufactures enough EXTRA turnover (two deadbands, two fee
schedules) to eat the funding saved -- the R-33/B-43 "the fix costs more
than the problem" pattern; (b) the two arms are not risk/exposure-matched
despite an identical source target (a bug, not a finding -- checked by
this file's own self-test below); (c) net d_sharpe misses the standing
+/-0.2 noise floor (R-20) or its CI contains zero; (d) the mechanism does
not replicate on ETH.

**The ETH data ceiling, stated now, before any branch runs anything:** no
ETH perpetual funding-rate series is committed anywhere in this repo
(checked: `btcusdt_perp_funding_8h.csv.gz` and
`btcusdt_deribit_perp_funding_8h.csv.gz` are both BTC-only). Per this
project's standing rule ("never proxy unavailable data out of price"),
ETH's futures leg is run funding-FREE here and every ETH number is an
upper bound, exactly the convention this repo's README already applies to
every uncovered funding case. **Consequently ETH cannot be used for the
dollar-savings promotion gate below** (COST is measured in real funding
dollars, which this project has only ever measured on BTC) -- ETH is
strictly a mechanism/replication check (does the SAME split logic run,
route correctly, avoid a liquidation/lookahead bug, on a second
instrument), matching R-144's own precedent ("branch cannot and does not
test C3 by construction -- BTC-only method finding, ceiling stated before
any run").

**HybridBroker, below, is NOT a change to `tradebot/broker.py`,
`engine.py` or `multiasset.py`** -- all three stay untouched. It is new,
disjoint, experiment-only code (ROUTINE.md step 1 q3: "build the missing
simulation capability and record that"), needed because
`tradebot/multiasset.py`'s own composer explicitly cannot express this:
its `run_multi_backtest` requires a capital split fixed BEFORE the run and
gives each leg an independent, isolated broker (its own docstring names
this the "no shared risk budget" limit). Routing `target`'s `<=1.0x` core
to spot and its excess to futures needs both legs to draw against the
SAME, currently-marked combined equity every bar (spot can hold at most
1x equity; if two independently-capitalized brokers each got a fixed
slice of starting capital, spot could never reach a full 1x-of-COMBINED
notional once any capital was reserved for the futures leg's margin).
Because `target` is a pure, already-causal function of price history
alone (no dependence on realized P&L or any other runtime state -- true
of every `kelly_regime*` strategy), a single shared-ledger broker can
compute both legs' fractions from the one precomputed `target` array with
no leg ever observing the other's live state mid-run, which is exactly
what keeps this within the same causal discipline `tradebot/engine.py`
enforces. `HybridBroker` reuses `PaperBroker`'s own tested per-unit
arithmetic (fee = `fee_rate * |delta| * price`; average-entry accounting
on scale-ins; realized PnL on reduces; the same `_max_qty` /
`_execute_target` clamp-and-haircut math) applied to two position slots
sharing one cash balance instead of one, and its liquidation price is
`PaperBroker.liquidation_price`'s own formula generalized to treat the
spot leg's mark-to-market as part of the collateral backing the futures
leg's maintenance margin -- exactly how a real cross-margined account
works. **Disclosed simplification:** a futures-leg liquidation event marks
the WHOLE hybrid broker dead (stops both legs), matching
`PaperBroker.check_liquidation`'s own unconditional `self.dead = True`
byte-for-byte rather than inventing a softer partial-liquidation rule.

**Frozen pre-registration.**

- Data: BTC spot 5m + BTC funding (`load_funding_extended`: real Binance
  2020-2023, Deribit-extended for the genuine post-2023 gap, B-02) is the
  PRIMARY, dollar-savings evidence. ETH spot 5m (Bitfinex) is the
  mechanism/replication check, funding-free (see ceiling above).
- Splits: inner-train `<= 2020-12-31` (build/debug only, not read for any
  promotion-relevant number); inner-validation `2021-01-01 -> 2022-12-31`
  (all selection, both branches, both fee tiers); holdout
  `OOS_START -> ` untouched until a branch clears the inner-validation
  gate below and is carried to Step 4.
- Fee tiers: 0.10% spot / 0.05% futures (standard) and 0.40% spot / 0.05%
  futures (the real Bitstamp entry tier this repo's README warns every
  strategy must also clear).
- Futures leverage headroom: 5.0x (this repo's standard `futures_5x`),
  comfortably above the `<=2.0x` (default preset) or `<=1.0x` excess a
  `kelly_regime_v4` target ever asks the futures leg to carry.

**Inner-validation gate (Step 3 selection -- decide here whether a branch
is even worth carrying to the holdout; frozen NOW, before either branch
has run anything):**

*Conservative* clears the gate only if, on BTC AND ETH (mechanism check
only on ETH, per the ceiling above), at BOTH fee tiers:
1. `d_sharpe` (hybrid vs plain `kelly_regime_v4` run whole on
   `futures_5x`, identical target, identical fee/funding assumptions per
   leg) `>= +0.20` (R-20's own noise floor) on BTC, with the 95%
   paired block-bootstrap CI (`tradebot.inference.paired_bootstrap`,
   `annualized_sharpe`, this project's standard tool) excluding zero;
2. incremental two-leg turnover-dollar cost (extra fees paid by the
   hybrid vs. plain futures v4) is `< 50%` of the raw funding dollars
   saved (`funding_paid` plain-futures minus `funding_paid` hybrid) --
   the R-65/R-67 "a cost mechanism must earn its own turnover" discipline;
3. time-in-market and realized volatility are matched to within 1% of
   the plain-futures run (R-131's rule: report it for every arm; a
   mismatch here is a bug in the harness, not a result, and must be fixed
   before anything else is trusted);
4. on ETH, the harness runs with no lookahead/liquidation bug and its
   *mechanical* routing (spot vs. futures notional split) tracks
   `min(target,1)` / `max(target-1,0)` to floating tolerance -- a
   correctness check, not a dollar comparison (see the ceiling above).

Kill bar (reject without holdout, however good BTC looks): (2) fails
(turnover eats >=50% of the saving), OR any BTC CI in (1) contains zero,
OR the harness itself fails the degenerate-equivalence self-test below.

*Novel* is scored only once conservative clears (1)-(3) above -- it is a
refinement of conservative's own mechanism, not an independent claim. It
clears its own, smaller bar only if, on BTC inner-validation:
1. `d_sharpe` (novel vs. conservative, same fee tier) `>= +0.05`, CI
   excluding zero;
2. the harvested funding rebate (during trailing-negative-funding
   windows the novel route keeps in futures rather than migrating to
   spot) exceeds the EXTRA switching-fee cost of doing so by `>= 2:1`.

**n-check, done now rather than after a result (R-78's own discipline):**
conservative's comparison is NOT an N~3 rare-episode test -- both arms
track the identical `target` path and price series, so the paired daily
difference isolates financing cost alone; funding settles ~3x/day,
smoothly accruing, nothing like the 3.0%/day common-mode whole-strategy
noise R-78 measured. At v4's own historical mean notional (0.18-0.38x,
R-57/R-62) and BTC funding order 8-15%/yr, the expected daily saving is
O(0.03-0.15%) against a paired-difference noise plausibly an order of
magnitude below full-return noise -- ample power over 730 inner-validation
days to resolve +-0.2 Sharpe, the opposite of R-78's 7-19-year problem.
Novel's comparison is different in kind: BTC funding-negative episodes in
2021-2022 are plausibly a handful of multi-week windows, not a smooth
daily process, so novel's own bar is closer to an N~3 measurement than a
high-power one -- if inner-validation does not contain enough
negative-funding time for a `+0.05` CI to be informative, THAT is the
pre-registered, reportable outcome, not grounds to loosen the bar.

**Configs evaluated (declare before running, sum into deflated Sharpe):**
conservative -- 1 primary (threshold=1.0) + `CONSERVATIVE_THRESHOLDS`
robustness cells = 3 total. Novel -- 1 primary + 2 robustness spans, in
its own file (not duplicated here, since the adaptive mechanism is its
own contribution). **Holdout counter:** increment only by whatever cells
a branch actually reads past the inner-validation gate above -- 0 so far,
this file freezes the rule but reads no holdout bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.data import load_dataset, load_funding_extended, load_ohlcv_csv
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.window import prefix_bars

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------- frozen dates
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# ---------------------------------------------------------------- frozen fees
SPOT_FEE_BASE = 0.001   # 0.10%, this project's standard spot taker tier
SPOT_FEE_REAL = 0.004   # 0.40%, Bitstamp's real entry tier (README warning)
FUT_FEE = 0.0005        # 0.05%, standard futures taker
FUT_LEVERAGE = 5.0
MAINTENANCE_MARGIN = 0.005

# ---------------------------------------------------------------- decision rule
D_SHARPE_FLOOR = 0.20          # R-20's standing +/-0.2 Sharpe noise floor
TURNOVER_SAVINGS_KILL = 0.50   # incremental 2-leg turnover must cost <50% of $ saved
EXPOSURE_MATCH_TOL_PCT = 1.0   # time-in-market / vol match tolerance, R-131's rule
NOVEL_D_SHARPE_FLOOR = 0.05
NOVEL_RATIO_KILL = 2.0         # harvested rebate must be >= 2x its own switching cost

CONSERVATIVE_THRESHOLDS = (0.8, 1.0, 1.2)   # primary=1.0, +/-0.2 robustness cells


def spot_market(fee_rate: float = SPOT_FEE_BASE) -> MarketSpec:
    return MarketSpec.spot(fee_rate=fee_rate)


def fut_market(fee_rate: float = FUT_FEE) -> MarketSpec:
    return MarketSpec.futures(leverage=FUT_LEVERAGE, fee_rate=fee_rate)


# ---------------------------------------------------------------------- data

def load_btc() -> tuple[pd.DataFrame, pd.Series | None, str]:
    df, label = load_dataset(DATA_DIR, "spot")
    rate, source = load_funding_extended(DATA_DIR)
    return df, rate, label


def load_eth() -> tuple[pd.DataFrame, None, str]:
    """ETH spot 5m. Coinbase (2019-03 -> 2026-08), not the Bitfinex pair
    R-138/R-144 used for their event-study replication check: Bitfinex's
    committed file stops at 2019-12-31, before this round's
    inner-validation window even starts (2021-2022), let alone the
    holdout -- checked directly rather than assumed, since a truncated
    series would silently make the ETH check empty. Coinbase spans both.
    Funding is `None` by construction -- see this module's docstring on
    the ETH data ceiling; every ETH futures number this round produces is
    a funding-free upper bound, not a dollar-savings claim.
    """
    df = load_ohlcv_csv(DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz")
    return df, None, "real, no funding series committed (ETH)"


def compute_target(df: pd.DataFrame) -> np.ndarray:
    """v4's own, unmodified, already-causal target column -- reused, not
    reimplemented, so nothing here can silently diverge from the
    registered strategy's real behavior.
    """
    prepared = KellyRegimeV4().prepare(df.copy())
    return prepared["target"].to_numpy(dtype=float)


def plain_v4_period(df: pd.DataFrame, market: MarketSpec, funding: pd.Series | None,
                    start: str, end: str):
    """Plain, single-venue `kelly_regime_v4`, funded and warmup-fair, over
    one sub-period -- the baseline both branches compare against.
    `tradebot.window.run_period` does not accept `funding` (it silently
    runs a funding-free perp), so this instead mirrors
    `scripts/funding_study.py::_period`'s own established pattern exactly:
    a manual warmup prefix plus `run_backtest(..., trade_start=pre,
    funding=funding, ...)`, then the standard `replace`-based trim.
    """
    from dataclasses import replace as _replace

    from tradebot.engine import run_backtest

    strategy = KellyRegimeV4()
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, df.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding)
    return raw if pre == 0 else _replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])


def trailing_funding_ewm(funding: pd.Series | None, index: pd.DatetimeIndex,
                          span_days: float) -> np.ndarray:
    """Causal EWM of the funding rate, bucketed onto `index` and shifted so
    bar i only ever sees settlements strictly before it -- for a branch's
    own adaptive-threshold route. Returns an all-zero array (never
    negative -> never triggers a "harvest" branch) when `funding` is None,
    so a route function built on this is automatically inert on ETH rather
    than needing a special case.
    """
    if funding is None or not len(funding):
        return np.zeros(len(index), dtype=float)
    rates = funding.sort_index()
    rates = rates[(rates.index >= index[0]) & (rates.index <= index[-1])]
    per_bar = pd.Series(0.0, index=index)
    if len(rates):
        slot = index.searchsorted(rates.index, side="right") - 1
        for pos_i, rate in zip(slot, rates.to_numpy(dtype=float)):
            if pos_i >= 0:
                per_bar.iloc[pos_i] += float(rate)
    span_bars = span_days * 288  # BARS_PER_DAY, 5m bars
    ewm = per_bar.ewm(span=span_bars, min_periods=1).mean().shift(1).fillna(0.0)
    return ewm.to_numpy(dtype=float)


def route_fixed_threshold(threshold: float = 1.0) -> Callable[[np.ndarray], "RouteFn"]:
    """The conservative mechanism: split `target` at a fixed multiple.

    ``spot_frac = min(target, threshold)``, ``fut_frac = target -
    spot_frac`` (always ``>= 0``, since `target` never goes negative).
    ``threshold=1.0`` is the literal Ackerer et al. replication split (the
    unlevered core in spot, only true excess leverage in futures);
    `CONSERVATIVE_THRESHOLDS` sweeps it for the plateau check ROUTINE.md's
    promotion bar requires.
    """
    def build(target: np.ndarray) -> "RouteFn":
        spot_frac = np.clip(target, 0.0, threshold)
        fut_frac = target - spot_frac
        return RouteFn(spot_frac, fut_frac)
    return build


@dataclass
class RouteFn:
    """A precomputed, per-bar (spot_frac, fut_frac) routing decision.

    Precomputing both arrays up front (rather than a per-bar callback)
    keeps every route function trivially causal: whatever a branch builds
    `spot_frac[i]`/`fut_frac[i]` from must already be causal by the time
    this object exists, because there is no later hook for it to peek at
    future bars from.
    """

    spot_frac: np.ndarray
    fut_frac: np.ndarray

    def __call__(self, i: int) -> tuple[float, float]:
        return float(self.spot_frac[i]), float(self.fut_frac[i])


# ------------------------------------------------------------------- broker

@dataclass
class HybridBroker:
    """Two long-only marked sub-positions (spot: unlevered, no shorting;
    futures: leveraged, funding-bearing) sharing ONE cash ledger. See this
    module's docstring for the full justification. Reuses
    `PaperBroker`'s own tested per-unit fee/fill arithmetic and
    `_max_qty`/`_execute_target` clamp-and-haircut math verbatim, applied
    to two position slots instead of one.
    """

    spot: MarketSpec
    fut: MarketSpec
    start_balance: float
    slippage_bps: float = 0.0

    cash: float = field(init=False)
    pos_spot: float = field(init=False, default=0.0)
    entry_spot: float = field(init=False, default=0.0)
    pos_fut: float = field(init=False, default=0.0)
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
            raise ValueError("HybridBroker's spot leg must be long-only")
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
        """PaperBroker._transact's realized-PnL / average-entry / fee
        arithmetic, verbatim, generalized to hand cash/fee back to the
        caller instead of mutating a single shared position slot.
        """
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

    def _execute_leg(self, leg: str, frac: float, price: float) -> None:
        if frac < 0:
            raise ValueError(f"HybridBroker is long-only; got frac={frac!r} for {leg}")
        pos, entry, market = ((self.pos_spot, self.entry_spot, self.spot) if leg == "spot"
                              else (self.pos_fut, self.entry_fut, self.fut))
        eq = self.equity(price)
        # Mirrors order_notional's own conversion (fraction of EQUITY ->
        # fraction of max notional) then _execute_target's clamp -- both
        # legs are long-only here, so lo=0 always.
        target_equiv = min(1.0, max(0.0, frac / max(market.leverage, 1e-9)))
        slip = self.slippage_bps / 10_000.0
        haircut = max(0.0, 1.0 - (market.fee_rate + slip) * market.leverage)
        max_qty = eq * market.leverage * haircut / price if eq > 0 and price > 0 else 0.0
        desired = max_qty * target_equiv
        delta = desired - pos
        max_notional = eq * market.leverage
        # Matches `_execute_target` exactly: the deadband only ever throttles
        # a same-direction RE-target, never a full close (`target_equiv ==
        # 0.0`). Missing the `target_equiv != 0.0` guard here silently
        # stuck small positions open (a close whose notional undercut the
        # deadband -- computed against max leveraged notional, not the
        # position's own small size -- was dropped, corrupting every later
        # decision's `pos`), found by this file's own degenerate-equivalence
        # check: 82 fills instead of plain v4's 143 over the same period.
        if (target_equiv != 0.0 and pos != 0.0 and max_notional > 0
                and abs(delta) * price < market.deadband * max_notional):
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
        # Deliberate divergence from PaperBroker._transact's own check
        # (`pos1 == 0.0 and cash < 0.0`): that condition is only a correct
        # bankruptcy test for a single book, where cash IS the equity once
        # the lone position is flat. Here a leg can close to zero while the
        # OTHER leg still carries a large unrealized mark, so raw cash is
        # not equity -- checking combined `self.equity(price)` instead is
        # the direct two-leg generalization of the same "can this ledger
        # still cover what it holds" question, not a laxer or stricter one.
        if self.equity(price) < 0.0:
            # Full wipeout, not just this leg -- a shared ledger that can no
            # longer cover its combined mark cannot support the OTHER leg's
            # position either. Zeroing both (not just cash) keeps every
            # later `equity()` call reading a flat, floored-at-zero account
            # instead of a stale mark on positions that no longer exist.
            self.cash = 0.0
            self.pos_spot = self.entry_spot = 0.0
            self.pos_fut = self.entry_fut = 0.0
            self.dead = True

    def fill(self, spot_frac: float, fut_frac: float, price: float) -> None:
        if self.dead:
            return
        self._execute_leg("spot", spot_frac, price)
        if self.dead:
            return
        self._execute_leg("fut", fut_frac, price)

    # --------------------------------------------------------------- funding

    def apply_funding(self, rate: float, price: float) -> None:
        if self.dead or self.pos_fut == 0.0:
            return
        flow = -rate * self.pos_fut * price
        self.cash += flow
        self.funding_paid -= flow
        if self.cash < 0.0:
            # PaperBroker.apply_funding's own floor+kill, generalized: the
            # shared ledger cannot support either leg once bankrupt.
            self.cash = 0.0
            self.pos_spot = self.entry_spot = 0.0
            self.pos_fut = self.entry_fut = 0.0
            self.dead = True

    # ----------------------------------------------------------- liquidation

    def check_liquidation(self, o: float, h: float, l: float) -> None:
        """PaperBroker.liquidation_price's formula, generalized to treat
        the spot leg's mark-to-market as collateral backing the futures
        leg's maintenance margin -- solving equity(p) = mm*pos_fut*p for p
        with an extra spot-mark term. Long-only by construction (pos_fut
        is never negative in this design), so only the downside branch is
        needed. Disclosed simplification: any futures liquidation marks
        the WHOLE broker dead, matching `PaperBroker.check_liquidation`'s
        own unconditional `self.dead = True` rather than a softer partial
        rule.
        """
        if self.dead or self.pos_fut <= 0.0:
            return
        mm = self.fut.maintenance_margin_rate
        denom = self.pos_spot + self.pos_fut * (1.0 - mm)
        if denom <= 0:
            return
        p_liq = (self.pos_spot * self.entry_spot + self.pos_fut * self.entry_fut
                 - self.cash) / denom
        if p_liq <= 0:
            return
        if l > p_liq:
            return
        px = min(o, p_liq)
        pos1, entry1, cash_delta, fee = self._transact_leg(
            -self.pos_fut, px, self.fut.fee_rate, self.pos_fut, self.entry_fut)
        self.cash += cash_delta
        self.fees_paid += fee
        self.pos_fut, self.entry_fut = 0.0, 0.0
        self.fills_fut += 1
        if self.cash < 0.0:
            self.cash = 0.0
        self.dead = True


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


def run_hybrid_backtest(
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
    """Same per-bar sequence as `tradebot.engine.run_backtest` (fill
    previous bar's pending order at THIS bar's open, check liquidation,
    apply funding, record equity at close, decide the NEXT order from
    THIS bar's close), and the same `run_period` warmup-prefix/trim
    convention (`prefix_bars(df, lo, warmup)` bars of real history before
    `start`, orders discarded until the period itself starts, curve
    trimmed back to it afterward).

    **Why `route_builder` takes the ALREADY-SLICED frame, not the whole
    df:** `KellyRegime.prepare`'s volatility column is an EWM, which has
    unbounded memory -- computing it on the full 1M-bar series gives a
    materially different value at a given calendar date than computing it
    on exactly `run_period`'s own `[lo - prefix : hi]` slice (found by
    this file's own degenerate-equivalence self-check: an 8% final-balance
    gap on the first attempt, which used the full df). `run_period` /
    `run_backtest` only ever see the sliced frame, and every existing
    inner-validation/OOS number in this project's ledger is computed that
    way -- so `route_builder` is handed the identical frame `compute_target`
    (or a branch's own signal construction) must run on to match it,
    rather than the unsliced df.
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

    broker = HybridBroker(spot=spot_mkt, fut=fut_mkt, start_balance=start_balance,
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
            raise ValueError(f"HybridBroker: equity became non-finite at bar {i} ({index[i]})")
        equity[i] = eq

        last_bar = i == n - 1
        # Edge-triggered, matching KellyRegime.on_bar exactly ("if abs(t -
        # prev) > 1e-9: ctx.order_notional(t)"): a route that repeats its
        # previous bar's fractions issues NO order, even though the
        # position is already at that level. This matters here because
        # `target` (and so both fractions) can sit at a PLATEAU straddling
        # `prefix` -- latched during warmup and unchanged for a long
        # stretch afterward -- in which case the real engine leaves the
        # account flat until the signal next actually moves, rather than
        # opening a position the moment trading is allowed. Found by this
        # file's own degenerate-equivalence check: computing a fresh order
        # every bar filled at `prefix` instead of at the next real change,
        # an 8% final-balance gap on the first attempt.
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


# --------------------------------------------------------------- self-checks

def _degenerate_equivalence_check(df: pd.DataFrame, funding: pd.Series | None,
                                  start: str, end: str) -> None:
    """All-futures and all-spot degenerate routes must reproduce plain
    `run_period(KellyRegimeV4(), ..., market=FUT/SPOT)` to floating
    tolerance -- the harness's own correctness gate, run before anything
    else this round is trusted (per this file's kill bar).
    """
    fut, spot = fut_market(), spot_market()

    def all_futures(frame: pd.DataFrame) -> RouteFn:
        t = compute_target(frame)
        return RouteFn(np.zeros_like(t), t.copy())

    def all_spot(frame: pd.DataFrame) -> RouteFn:
        t = compute_target(frame)
        return RouteFn(np.clip(t, 0.0, None), np.zeros_like(t))

    hybrid_fut = run_hybrid_backtest(df, all_futures, spot, fut, funding=funding,
                                     start=start, end=end)
    plain_fut = plain_v4_period(df, fut, funding, start, end)
    diff_fut = abs(hybrid_fut.final_balance - plain_fut.final_balance)
    rel_fut = diff_fut / max(plain_fut.final_balance, 1.0)

    hybrid_spot = run_hybrid_backtest(df, all_spot, spot, fut, funding=funding,
                                      start=start, end=end)
    plain_spot = plain_v4_period(df, spot, None, start, end)  # spot never pays funding
    diff_spot = abs(hybrid_spot.final_balance - plain_spot.final_balance)
    rel_spot = diff_spot / max(plain_spot.final_balance, 1.0)

    print(f"[degenerate-equivalence] all-futures: hybrid=${hybrid_fut.final_balance:,.2f} "
          f"plain=${plain_fut.final_balance:,.2f} rel_diff={rel_fut:.2e}")
    print(f"[degenerate-equivalence] all-spot:    hybrid=${hybrid_spot.final_balance:,.2f} "
          f"plain=${plain_spot.final_balance:,.2f} rel_diff={rel_spot:.2e}")
    assert rel_fut < 1e-6, f"all-futures degenerate route diverges from plain v4: {rel_fut:.2e}"
    assert rel_spot < 1e-6, f"all-spot degenerate route diverges from plain v4: {rel_spot:.2e}"


def _causality_truncation_check(df: pd.DataFrame, funding: pd.Series | None,
                                start: str, end: str) -> None:
    """Truncate the data one bar short of `end` and confirm every fill
    inside the shared overlap is identical -- the same style of check
    `tests/test_causality_strict.py` runs for every registered strategy,
    applied to this harness instead.
    """
    def route_builder(frame: pd.DataFrame) -> RouteFn:
        return route_fixed_threshold(1.0)(compute_target(frame))

    full = run_hybrid_backtest(df, route_builder, spot_market(), fut_market(),
                              funding=funding, start=start, end=end)

    truncated_df = df.loc[:df.index[df.index.searchsorted(end, side="right") - 2]]
    trunc = run_hybrid_backtest(truncated_df, route_builder, spot_market(), fut_market(),
                                funding=funding, start=start)

    overlap = trunc.equity.index
    diffs = (full.equity.reindex(overlap) - trunc.equity).abs()
    max_diff = float(diffs.max())
    print(f"[causality-truncation] max |diff| over {len(overlap)} shared bars: {max_diff:.2e}")
    assert max_diff < 1e-6, f"truncating the tail changed a bar strictly before it: {max_diff:.2e}"


if __name__ == "__main__":
    btc, btc_funding, btc_label = load_btc()
    print(f"BTC: {len(btc):,} bars ({btc_label}), funding: "
          f"{'present' if btc_funding is not None else 'MISSING'}\n")

    print("=== degenerate-equivalence self-check (BTC, inner-validation) ===")
    _degenerate_equivalence_check(btc, btc_funding, INNER_VAL_START, INNER_VAL_END)

    print("\n=== causality truncation self-check (BTC, inner-validation) ===")
    _causality_truncation_check(btc, btc_funding, INNER_VAL_START, INNER_VAL_END)

    print("\nAll self-checks passed.")
