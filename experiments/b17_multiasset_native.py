#!/usr/bin/env python
"""B-17, NOVEL branch: a native multi-instrument backtest context/engine.

Infrastructure round, not a trading-idea round. The question this file
answers is purely: *can the framework be extended, cleanly and safely, to
register a strategy that makes one joint decision across BTC+ETH from a
single shared risk budget?* It does NOT re-test or re-argue whether
dual-asset diversification makes money on this project's data -- R-43
already tested that FINDING on the real 2023+ holdout and it was
REJECTED. Nothing here touches 2023+ data or claims a promotion.

Constraint attacked: none of the four (INFO/N=3/ERR/COST) directly --
this is the same "framework has no multi-asset registration path at all"
gap logged as backlog item **B-17**, discovered while writing R-43's own
pre-registration (see docs/LEDGER.md, "Re-ranked 08-19 after R-43").

Not a duplicate of: the conservative branch (parallel, disjoint files,
not read by this session) builds an ADAPTER that runs the existing
single-asset engine once per instrument and combines the resulting
equity curves after the fact at a fixed capital split. That adapter
cannot express a decision that needs to see both assets' state jointly
in order to size either leg -- e.g. solving a shared Sigma^-1 mu
allocation, where "how much BTC" and "how much ETH" are two components
of ONE simultaneous solve against ONE risk budget, not two independently
-capped sub-accounts. This file builds a genuinely joint context/engine
instead, to see what that costs in code and risk versus the adapter.

Mechanism (of the infra), one sentence: replace ``Context`` (one
instrument, one implicit broker) with ``MultiAssetContext`` (every
instrument's bars, ONE shared broker/equity/position view) and replace
``run_backtest``'s single-array event loop with ``run_multi_backtest_native``,
which steps a SHARED, intersected event-time index across N instrument
frames while preserving the exact same causal contract: bar-close signal,
next-open fill, one bar of latency, no exceptions.

Demonstration strategy: a simplified two-asset Sigma^-1 mu allocator
inspired by ``experiments/kelly_regime_covkelly.py`` (same closed-form
2x2 Kelly weight, same causal trailing-EWM mu/Sigma estimator, same
"clip negative to zero, fractional-Kelly, per-leg cap, total cap"
pipeline) but re-plumbed to run through the NEW native engine bar-by-bar
rather than through covkelly's own "stitch monthly/weekly segments of two
independent single-asset engine runs" pattern -- which is itself a form
of adapter, just a periodic-rebalance one. The point of this file is the
engine underneath, not a new trading claim; the demo strategy exists only
to exercise the joint, shared-budget code path with something an adapter
genuinely cannot express.

Hard rules honored (see prompt / docs/ROUTINE.md)
--------------------------------------------------
- Only this file is touched under experiments/; nothing under
  src/tradebot/ is modified.
- Data is HARD-SLICED to <= 2022-12-31 immediately after loading (see
  ``LOAD_CUTOFF``) -- every frame anywhere in this file is derived from
  that slice, so no path here can reach 2023-01-01+ data. Grep this file
  for "2023" to confirm the only literals are OOS_START/LOAD_CUTOFF and
  this comment.
- Inner-train = 2019-03-14 (ETH's real start) -> 2020-12-31.
  Inner-validation = 2021-01-01 -> 2022-12-31. Holdout (2023-01-01 ->) is
  never read.
- No lookahead: every estimator is ``.ewm(...).shift(1)``; see
  ``causality_check`` for the mechanical truncation proof and the
  deliberate break-it probe.

Data alignment, checked empirically before writing the engine loop
--------------------------------------------------------------------
BTC (Bitstamp spot, committed) is a perfectly regular 5-minute grid over
2019-03-14 -> 2022-12-31 (400,032 bars, one single inter-bar gap value:
exactly 5 minutes, no exceptions). ETH (Coinbase spot, committed) is
*almost* as regular over the same window (399,861 bars) but has a small
number of real gaps -- 171 timestamps BTC has that ETH does not, 0 the
other way. So "do the two series line up in wall-clock time" is "yes,
except for 171 timestamps out of ~400k (0.043%) where ETH is genuinely
missing a bar."

Design decision (documented, not silently swallowed): the shared
event-time index used by ``run_multi_backtest_native`` is the
**intersection** of all instruments' timestamps (an inner join), computed
fresh after each strategy's ``prepare()`` call. A timestamp missing from
even one instrument is dropped from EVERY instrument for that step --
never forward-filled, never synthesized. This is the conservative choice:
it costs 0.043% of the bars here, but it means a strategy can never see a
"pretend" bar for an instrument that did not actually trade at that
minute, which matters more for causal correctness than for coverage. An
instrument that starts later than another (the real BTC-vs-ETH situation
pre-2019-03-14) is handled the same way -- the intersection simply begins
at the later start, which is why this file's TRAIN_START is ETH's start
date rather than BTC's.

Usage::

    python experiments/b17_multiasset_native.py demo         # step 4 demo numbers (this is the report)
    python experiments/b17_multiasset_native.py causality    # mandatory no-lookahead check + break-it probe
    python experiments/b17_multiasset_native.py all           # both, in order
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.engine import validate_ohlcv  # noqa: E402
from tradebot.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402
from tradebot.strategy import Row  # noqa: E402

# --- Data discipline (hard rule) ---------------------------------------
TRAIN_START = "2019-03-14"          # ETH's real start; inner-train begins here
TRAIN_END = "2020-12-31"
VALID_START = "2021-01-01"
VALID_END = "2022-12-31"
OOS_START = "2023-01-01"            # never read in this file
LOAD_CUTOFF = "2022-12-31 23:55:00"  # hard slice applied immediately after load

SPOT = MarketSpec.spot()

N_EVALUATED = 0  # every distinct backtest configuration run (see report step 5)


# =========================================================================
# 1. MultiAssetContext / MultiAssetStrategy -- the new, additive API
# =========================================================================

class MultiAssetContext:
    """What a strategy sees on one shared-event-time step, multi-instrument.

    Analogous to ``tradebot.strategy.Context``, generalized to N
    instruments sharing ONE account: ``mctx.bar(inst)`` /
    ``mctx.prev(inst)`` are per-instrument ``Row`` views (identical class
    to the single-asset engine's, imported unmodified), while
    ``mctx.equity`` and ``mctx.position(inst)`` read ONE shared broker, so
    a strategy can size two legs against a single risk budget rather than
    two independently-capped sub-accounts. Orders are queued as target
    fractions of that ONE shared equity via ``order_target`` and fill at
    the next bar's open, exactly as in the single-asset engine.
    """

    __slots__ = ("_cols", "_dfs", "i", "_broker", "_index", "instruments", "orders")

    def __init__(self, cols: dict[str, dict], dfs: dict[str, pd.DataFrame], i: int,
                 broker: "MultiAssetBroker", index: pd.DatetimeIndex) -> None:
        self._cols = cols
        self._dfs = dfs
        self.i = i
        self._broker = broker
        self._index = index
        self.instruments = list(cols.keys())
        self.orders: dict[str, float] = {}

    @property
    def ts(self) -> pd.Timestamp:
        return self._index[self.i]

    def bar(self, inst: str) -> Row:
        """Current (just closed) bar for ``inst``, including prepare() columns."""
        return Row(self._cols[inst], self.i)

    def prev(self, inst: str) -> Row | None:
        return Row(self._cols[inst], self.i - 1) if self.i > 0 else None

    def history(self, inst: str, n: int | None = None) -> pd.DataFrame:
        """All closed bars of ``inst`` up to and including the current one."""
        end = self.i + 1
        start = 0 if n is None else max(0, end - n)
        return self._dfs[inst].iloc[start:end]

    # ------------------------------------------------------------ account

    def position(self, inst: str) -> float:
        """Signed base-asset position in ``inst``, from the ONE shared broker."""
        return self._broker.pos.get(inst, 0.0)

    @property
    def equity(self) -> float:
        """Total account equity, marked at every instrument's current close."""
        prices = {k: float(self._cols[k]["close"][self.i]) for k in self.instruments}
        return self._broker.equity(prices)

    # ------------------------------------------------------------- orders

    def order_target(self, inst: str, fraction: float) -> None:
        """Queue ``inst`` to move to ``fraction`` of TOTAL shared equity.

        ``fraction`` is a share of the ONE account, not of a per-instrument
        sub-budget: if a strategy submits ``order_target("BTC", 0.7)`` and
        ``order_target("ETH", 0.6)`` in the same bar, the sum (1.3) is
        clamped against the market's shared leverage cap (1.0 for spot) at
        fill time, proportionally across every leg queued that bar -- see
        ``MultiAssetBroker.execute_targets``. This is the property an
        adapter over N independent single-asset engines cannot express: a
        cap on the SUM of exposures rather than N separate caps.
        """
        if inst not in self._cols:
            raise KeyError(f"unknown instrument {inst!r}; have {self.instruments}")
        self.orders[inst] = float(fraction)


class MultiAssetStrategy:
    """Base class for multi-instrument strategies. Analogous to ``Strategy``.

    - ``prepare(dfs)``: called once with the full dict of per-instrument
      OHLCV frames (every instrument together, unlike the single-asset
      ``prepare(df)``, specifically so a joint feature -- e.g. a
      cross-asset covariance matrix -- can be computed once, causally,
      with visibility into every instrument at once). MUST be causal (row
      i of any instrument may only depend on rows <= i of every
      instrument) and MUST return a dict with the same keys, same rows,
      same index per instrument.
    - ``on_bar(mctx)``: called at each shared-event-time bar close; place
      per-instrument orders via ``mctx.order_target``.
    """

    name: str = "base_multi"
    warmup: int = 0

    def prepare(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return dfs

    def on_bar(self, mctx: MultiAssetContext) -> None:
        raise NotImplementedError


# =========================================================================
# 2. MultiAssetBroker -- simplified, but causally correct, shared-cash book
# =========================================================================

# Same convention as tradebot.broker.REBALANCE_DEADBAND: same-direction
# resizes smaller than this fraction of equity are ignored, so a strategy
# may re-emit its target every bar without racking up churn. Opens/closes
# to/from flat always execute regardless of size (subject to the dust
# floor below).
REBALANCE_DEADBAND = 0.05
MIN_TRADE_NOTIONAL = 1.0  # USD; ignore fills smaller than this (float/dust guard)


class MultiAssetBroker:
    """Shared-cash-pool, long-only-by-default paper book for N instruments.

    One ``cash`` balance funds every leg; ``equity(prices) = cash +
    sum_i pos_i * (price_i - entry_i)`` exactly mirrors
    ``tradebot.broker.PaperBroker``'s spot accounting, generalized to N
    legs sharing one pool instead of one leg alone. This is deliberately
    NOT a full reproduction of ``tradebot/broker.py`` (no liquidation
    engine, no cross-margin futures math) -- the task explicitly allows a
    simplified broker here; what it must get right, and does, is: (a) one
    shared budget across legs, computed from ONE equity snapshot per bar
    so legs are never sized against inconsistent, out-of-sync marks, and
    (b) fees charged correctly on every rebalance, per leg, at that leg's
    own execution price.

    Long-only when ``market.allow_short`` is False (the spot case used
    throughout this file's demo); the total-notional cap and per-leg
    scaling logic generalizes to ``market.allow_short=True`` /
    ``leverage>1`` futures markets too (untested here -- see the design
    note's discussion of what a production version would still need:
    liquidation, cross-margin funding, etc.).
    """

    def __init__(self, instruments: list[str], market: MarketSpec, start_balance: float) -> None:
        if start_balance <= 0:
            raise ValueError("start_balance must be positive")
        self.market = market
        self.cash = float(start_balance)
        self.pos: dict[str, float] = {inst: 0.0 for inst in instruments}
        self.entry: dict[str, float] = {inst: 0.0 for inst in instruments}
        self.fees_paid = 0.0
        self.fees_by_inst: dict[str, float] = {inst: 0.0 for inst in instruments}
        self.dead = False

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(self.pos[i] * (prices[i] - self.entry[i]) for i in self.pos)

    def execute_targets(self, targets: dict[str, float], opens: dict[str, float], ts) -> None:
        """Fill every queued target at this bar's OPENS, against ONE equity mark.

        ``targets`` (inst -> signed fraction of equity) are all measured
        against the SAME pre-fill equity snapshot (marked at these opens,
        using each leg's position carried from the previous bar) -- the
        joint, shared-budget property. Margin-style accounting (matches
        ``tradebot.broker.PaperBroker``): opening/growing a position costs
        only its fee, not its notional, so there is nothing for a same-bar
        sell to "free" for a same-bar buy -- sells are still applied first
        purely so realized PnL lands in ``cash`` before any reduce-driven
        edge case reads it, not because buys are cash-constrained.
        """
        if self.dead:
            return
        eq = self.equity(opens)
        if not math.isfinite(eq) or eq <= 0:
            return

        allow_short = self.market.allow_short
        clipped = {inst: (f if allow_short else max(0.0, f)) for inst, f in targets.items()}
        total = sum(abs(v) for v in clipped.values())
        lev = self.market.leverage
        if total > lev and total > 0:
            scale = lev / total
            clipped = {k: v * scale for k, v in clipped.items()}

        deltas: dict[str, float] = {}
        for inst, frac in clipped.items():
            price = opens[inst]
            desired_qty = frac * eq / price if price > 0 else 0.0
            deltas[inst] = desired_qty - self.pos[inst]

        # Sells (delta < 0) first, to free cash for same-bar buys.
        for inst, delta in sorted(deltas.items(), key=lambda kv: kv[1]):
            price = opens[inst]
            pos0 = self.pos[inst]
            if abs(delta) < 1e-12:
                continue
            # Same-sign resize (same convention as tradebot.broker's own
            # REBALANCE_DEADBAND docstring: "same-sign target adjustments
            # smaller than this fraction of max notional are ignored"),
            # regardless of whether it is an increase or a decrease -- a
            # close (target 0) or a flip always executes. Gating only
            # same-direction INCREASES (an earlier version of this
            # broker) let every tiny downtick force a real sell while the
            # matching buy-back stayed under threshold, which bled real
            # fees into pure noise over hundreds of thousands of bars.
            same_sign_resize = (pos0 != 0.0 and clipped.get(inst, 0.0) != 0.0
                               and math.copysign(1.0, clipped[inst]) == math.copysign(1.0, pos0))
            if same_sign_resize and abs(delta) * price < REBALANCE_DEADBAND * eq:
                continue
            if abs(delta) * price < MIN_TRADE_NOTIONAL:
                continue
            self._fill(inst, delta, price, ts)

    def _fill(self, inst: str, delta: float, price: float, ts) -> None:
        """Apply one instrument's fill, margin-style (matches
        ``tradebot.broker.PaperBroker``'s spot accounting exactly): opening
        or growing a position does NOT move notional out of ``cash`` -- only
        the fee does. ``cash`` only otherwise moves when a reduce realizes
        PnL. This is what makes ``equity = cash + pos*(price-entry)``
        correct; an earlier version of this method also subtracted the
        traded notional from cash on every buy, which double-counted it
        against that same equity formula and silently manufactured a
        collapsing, occasionally negative equity curve out of ordinary
        fee-sized noise -- caught by manually inspecting the first 30 bars
        of a plain buy-and-hold run, which should have shown ~flat equity
        near $1000 and instead showed equity swinging negative on bar 1.
        """
        pos0, entry0 = self.pos[inst], self.entry[inst]
        realized = 0.0
        if delta < 0.0 and pos0 > 0.0:
            reduce_qty = min(abs(delta), pos0)
            realized = (price - entry0) * reduce_qty
        fee = self.market.fee_rate * abs(delta) * price
        self.cash += realized - fee
        pos1 = pos0 + delta
        if abs(pos1) < 1e-12:
            pos1 = 0.0
        if pos1 != 0.0 and pos0 == 0.0:
            entry1 = price  # freshly opened
        elif pos1 != 0.0 and abs(pos1) > abs(pos0):
            entry1 = (entry0 * abs(pos0) + price * abs(delta)) / abs(pos1)  # scale-in average
        elif pos1 != 0.0:
            entry1 = entry0  # partial reduce keeps average entry
        else:
            entry1 = 0.0
        self.pos[inst], self.entry[inst] = pos1, entry1
        self.fees_paid += fee
        self.fees_by_inst[inst] += fee
        if self.cash < 0.0:
            if self.cash > -1e-6:
                self.cash = 0.0  # float dust guard, not a real bankruptcy
            else:
                # Long-only spot with leverage<=1 should never actually
                # reach this (there is no borrowing to go bankrupt against);
                # fail loudly rather than let the book go silently negative.
                raise RuntimeError(
                    f"MultiAssetBroker: cash went negative ({self.cash:.6f}) "
                    f"filling {inst} delta={delta} at {ts} -- broker bug")


# =========================================================================
# 3. run_multi_backtest_native -- the new engine loop
# =========================================================================

@dataclass
class MultiBacktestResult:
    strategy_name: str
    market: MarketSpec
    start_balance: float
    equity: pd.Series
    positions: dict[str, pd.Series]
    closes: dict[str, pd.Series]
    fees_paid: float
    fees_by_inst: dict[str, float]
    dropped_bars: dict[str, int] = field(default_factory=dict)

    @property
    def final_balance(self) -> float:
        return float(self.equity.iloc[-1])


def run_multi_backtest_native(
    strategy: MultiAssetStrategy,
    dfs: dict[str, pd.DataFrame],
    market: MarketSpec,
    start_balance: float,
    trade_start: int = 0,
) -> MultiBacktestResult:
    """Backtest ``strategy`` jointly over every instrument in ``dfs``.

    Mirrors ``tradebot.engine.run_backtest``'s per-bar sequence exactly,
    generalized to N instruments stepping a SHARED index:

    1. Fill orders queued on the previous step at THIS step's opens (every
       instrument, against one equity snapshot).
    2. Record equity at this step's closes.
    3. Call ``strategy.on_bar`` with every instrument's history up to this
       step; queue its per-instrument orders.

    No liquidation check (the demo broker is long-only spot, so there is
    nothing to liquidate); everything else about the causal ordering is
    identical to the single-asset engine, which is exactly what
    ``causality_check`` below verifies mechanically rather than by
    inspection.
    """
    if not dfs:
        raise ValueError("need at least one instrument")
    for inst, df in dfs.items():
        validate_ohlcv(df)

    prepared = strategy.prepare({k: v.copy() for k, v in dfs.items()})
    if prepared is None:
        raise ValueError(f"{strategy.name}.prepare() returned None; return the dict")
    for inst, df in dfs.items():
        if inst not in prepared:
            raise ValueError(f"{strategy.name}.prepare() dropped instrument {inst!r}")
        if len(prepared[inst]) != len(df) or not prepared[inst].index.equals(df.index):
            raise ValueError(
                f"{strategy.name}.prepare() must keep the same rows/index for {inst!r}")

    # --- shared event-time alignment: inner join across instruments -----
    common_index = None
    for df in prepared.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    common_index = common_index.sort_values()
    if len(common_index) == 0:
        raise ValueError("instruments share no common timestamps")

    aligned = {inst: df.loc[common_index] for inst, df in prepared.items()}
    cols = {inst: {c: aligned[inst][c].to_numpy() for c in aligned[inst].columns}
            for inst in aligned}
    opens = {inst: cols[inst]["open"] for inst in aligned}
    closes = {inst: cols[inst]["close"] for inst in aligned}

    broker = MultiAssetBroker(list(aligned.keys()), market, start_balance)
    n = len(common_index)
    equity_curve = [0.0] * n
    pos_curves: dict[str, list] = {inst: [0.0] * n for inst in aligned}
    pending: dict[str, float] = {}

    for i in range(n):
        ts = common_index[i]

        if pending and not broker.dead:
            open_prices = {inst: float(opens[inst][i]) for inst in aligned}
            broker.execute_targets(pending, open_prices, ts)
        pending = {}

        close_prices = {inst: float(closes[inst][i]) for inst in aligned}
        eq = broker.equity(close_prices)
        if not math.isfinite(eq):
            raise ValueError(
                f"{strategy.name}: equity became non-finite at bar {i} ({ts})")
        equity_curve[i] = eq
        for inst in aligned:
            pos_curves[inst][i] = broker.pos[inst]

        last_bar = i == n - 1
        if not broker.dead and not last_bar and i >= strategy.warmup:
            mctx = MultiAssetContext(cols, aligned, i, broker, common_index)
            strategy.on_bar(mctx)
            if i >= trade_start:
                pending = dict(mctx.orders)

    return MultiBacktestResult(
        strategy_name=strategy.name,
        market=market,
        start_balance=start_balance,
        equity=pd.Series(equity_curve, index=common_index, name="equity"),
        positions={inst: pd.Series(pos_curves[inst], index=common_index) for inst in aligned},
        closes={inst: pd.Series(closes[inst], index=common_index) for inst in aligned},
        fees_paid=broker.fees_paid,
        fees_by_inst=dict(broker.fees_by_inst),
        dropped_bars={inst: len(dfs[inst]) - n for inst in dfs},
    )


# =========================================================================
# 4. Demo strategies
# =========================================================================

class SingleAssetHold(MultiAssetStrategy):
    """Trivial always-fully-long strategy on exactly one instrument.

    Exists only to demonstrate the N=1 special case: run through the SAME
    native multi-asset engine with a one-key ``dfs`` dict, it degenerates
    to a plain buy-and-hold. Used as an architecture check, not a
    performance baseline for the joint allocator (a joint Sigma^-1 mu
    decision is not separable into two independent single-asset runs --
    that non-separability is the entire point of this file).
    """

    name = "b17_single_hold"
    warmup = 0

    def on_bar(self, mctx: MultiAssetContext) -> None:
        (inst,) = mctx.instruments
        mctx.order_target(inst, 1.0)


def _joint_daily_weights(
    dfs: dict[str, pd.DataFrame],
    instruments: tuple[str, str],
    halflife_days: float = 60.0,
    kelly_frac: float = 0.5,
    max_leg_weight: float = 1.0,
    total_cap: float = 1.0,
    min_periods_days: int = 60,
) -> pd.DataFrame:
    """Causal daily Sigma^-1 mu weight series for exactly two instruments.

    Same closed-form 2x2 Kelly weight, same causal trailing-EWM mu/Sigma
    estimator (``.ewm(...).shift(1)``), same "clip negative to zero,
    fractional-Kelly, per-leg cap, total cap" pipeline as
    ``experiments/kelly_regime_covkelly.py``'s ``build_weight_series`` --
    ported here rather than re-derived, since that file's causal logic is
    already the tested reference for this exact formula. Two differences
    from that file, both documented: (1) this hardcodes N=2 with a
    closed-form inverse (matching this project's actual data: exactly two
    independently-collected instruments) rather than a general N-asset
    ``np.linalg.solve`` -- a mechanical generalization the design note
    below discusses, not attempted here to avoid a second, less-tested
    code path; (2) the estimator-warmup fallback is 0.0/0.0 (sit in cash)
    rather than covkelly's 0.5/0.5 (an active bet) -- more conservative,
    since this file is not re-testing covkelly's own finding and should
    not import its specific warmup assumption uncritically.

    Every value at day D uses data strictly before D: ``daily_log_returns``
    resamples each instrument's OWN 5m closes to one value per UTC day
    using only that day's own bars, the EWM statistics are computed over
    that daily series, and the whole block is shifted by exactly one day
    before anything downstream reads it. See ``causality_check`` for the
    mechanical proof.
    """
    a, b = instruments
    daily = {}
    for inst in (a, b):
        close_d = dfs[inst]["close"].resample("1D").last().ffill()
        daily[inst] = np.log(close_d).diff()
    rets = pd.concat(daily, axis=1)
    rets.columns = [a, b]
    rets = rets.dropna(how="any")

    mp = int(min_periods_days)
    ewm_a = rets[a].ewm(halflife=halflife_days, min_periods=mp)
    ewm_b = rets[b].ewm(halflife=halflife_days, min_periods=mp)

    mu_a = ewm_a.mean().shift(1)
    mu_b = ewm_b.mean().shift(1)
    var_a = ewm_a.var().shift(1)
    var_b = ewm_b.var().shift(1)
    cov_ab = rets[a].ewm(halflife=halflife_days, min_periods=mp).cov(rets[b]).shift(1)

    out = pd.DataFrame({"mu_a": mu_a, "mu_b": mu_b,
                        "var_a": var_a, "var_b": var_b, "cov": cov_ab})

    ma, mb = out["mu_a"].to_numpy(), out["mu_b"].to_numpy()
    va, vb = out["var_a"].to_numpy(), out["var_b"].to_numpy()
    cv = out["cov"].to_numpy()
    w_a = np.zeros(len(out))
    w_b = np.zeros(len(out))
    fallback = np.zeros(len(out), dtype=bool)

    for i in range(len(out)):
        if not (np.isfinite(ma[i]) and np.isfinite(mb[i]) and np.isfinite(va[i])
                and np.isfinite(vb[i]) and np.isfinite(cv[i])):
            fallback[i] = True
            continue  # w_a[i]=w_b[i]=0.0 already: sit in cash during warmup
        Sigma = np.array([[va[i], cv[i]], [cv[i], vb[i]]])
        mu = np.array([ma[i], mb[i]])
        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        trace = Sigma[0, 0] + Sigma[1, 1]
        eps = 1e-8 * max(trace, 1e-12)
        if not np.isfinite(det) or abs(det) < eps:
            Sigma = Sigma + eps * np.eye(2)
            det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        raw_a = (Sigma[1, 1] * mu[0] - Sigma[0, 1] * mu[1]) / det
        raw_b = (Sigma[0, 0] * mu[1] - Sigma[1, 0] * mu[0]) / det
        raw_a = min(max(0.0, raw_a) * kelly_frac, max_leg_weight)
        raw_b = min(max(0.0, raw_b) * kelly_frac, max_leg_weight)
        s = raw_a + raw_b
        if s > total_cap and s > 0:
            scale = total_cap / s
            raw_a *= scale
            raw_b *= scale
        w_a[i], w_b[i] = raw_a, raw_b

    out["w_a"] = w_a
    out["w_b"] = w_b
    out["fallback"] = fallback
    return out


class JointKellyDemo(MultiAssetStrategy):
    """Two-asset Sigma^-1 mu allocator sharing ONE risk budget -- the demo.

    Solves the classical growth-optimal weight ``Sigma^-1 mu`` once per
    day from a causal trailing-EWM mean/covariance of each asset's own raw
    daily log returns (mechanism identical to
    ``kelly_regime_covkelly.py``, see ``_joint_daily_weights``), broadcasts
    that day's weight onto every 5-minute bar of the day (causally valid:
    the value was computed from data strictly before the day started), and
    submits BOTH legs' targets every bar via the shared-budget
    ``mctx.order_target``. The two legs are solved TOGETHER against ONE
    covariance matrix and ONE total-exposure cap -- this is the joint
    decision an adapter over two independent single-asset engines cannot
    express, because there each leg would need its own, separately fixed
    capital split rather than a target that can move fluidly to either
    side of the shared budget as Sigma and mu evolve.
    """

    name = "b17_joint_kelly"
    warmup = 0

    def __init__(self, halflife_days: float = 60.0, kelly_frac: float = 0.5,
                 max_leg_weight: float = 1.0, total_cap: float = 1.0,
                 min_periods_days: int = 60) -> None:
        self.params = dict(halflife_days=halflife_days, kelly_frac=kelly_frac,
                           max_leg_weight=max_leg_weight, total_cap=total_cap,
                           min_periods_days=min_periods_days)
        self._instruments: tuple[str, str] | None = None

    def prepare(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        instruments = tuple(sorted(dfs.keys()))
        if len(instruments) != 2:
            raise ValueError(f"{self.name} is a 2-asset demo; got {instruments}")
        self._instruments = instruments
        a, b = instruments
        weights = _joint_daily_weights(dfs, instruments, **self.params)

        for inst, col in ((a, "w_a"), (b, "w_b")):
            df = dfs[inst]
            day = df.index.floor("D")
            uniq_days = pd.DatetimeIndex(day.unique()).sort_values()
            w_daily = weights[col].reindex(weights.index.union(uniq_days)).sort_index().ffill()
            broadcast = w_daily.reindex(day).to_numpy()
            df["w_target"] = np.nan_to_num(broadcast, nan=0.0)
        return dfs

    def on_bar(self, mctx: MultiAssetContext) -> None:
        a, b = self._instruments
        mctx.order_target(a, float(mctx.bar(a)["w_target"]))
        mctx.order_target(b, float(mctx.bar(b)["w_target"]))


# =========================================================================
# 5. Data loading (hard rule: never read 2023+)
# =========================================================================

def load_assets(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """BTC spot + ETH Coinbase spot, HARD-SLICED to <= 2022-12-31."""
    btc, _ = load_dataset(data_dir, "spot")
    eth = load_coinbase_eth_spot(data_dir)
    if eth is None:
        raise RuntimeError("data/ethusd_coinbase_spot_5m.csv.gz not found")
    btc = btc.loc[:LOAD_CUTOFF].copy()
    eth = eth.loc[:LOAD_CUTOFF].copy()
    return btc, eth


# =========================================================================
# 6. Metrics
# =========================================================================

def multi_metrics(result: MultiBacktestResult) -> dict:
    arr = result.equity.to_numpy(dtype=float)
    return {
        "final_balance": float(arr[-1]) if len(arr) else result.start_balance,
        "sharpe": sharpe_ratio(arr),
        "max_dd_pct": max_drawdown_pct(arr),
        "fees_paid": result.fees_paid,
    }


# =========================================================================
# 7. Causality: truncation proof + deliberate break-it probe
# =========================================================================

def _tamper(dfs: dict[str, pd.DataFrame], cut: pd.Timestamp, factor: float) -> dict[str, pd.DataFrame]:
    out = {}
    for inst, df in dfs.items():
        d = df.copy()
        mask = d.index > cut
        for col in ("open", "high", "low", "close"):
            d.loc[mask, col] = d.loc[mask, col] * factor
        out[inst] = d
    return out


def _align(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, dict, dict]:
    common_index = None
    for df in dfs.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    common_index = common_index.sort_values()
    aligned = {inst: df.loc[common_index] for inst, df in dfs.items()}
    cols = {inst: {c: aligned[inst][c].to_numpy() for c in aligned[inst].columns}
            for inst in aligned}
    return common_index, aligned, cols


def _decisions_multi(strategy: MultiAssetStrategy, dfs: dict[str, pd.DataFrame],
                     bars: list[int], market: MarketSpec) -> list[tuple]:
    """The per-instrument targets ``strategy`` queues at each bar in ``bars``."""
    prepared = strategy.prepare({k: v.copy() for k, v in dfs.items()})
    common_index, aligned, cols = _align(prepared)
    broker = MultiAssetBroker(list(aligned.keys()), market, 10_000.0)
    out = []
    for i in bars:
        mctx = MultiAssetContext(cols, aligned, i, broker, common_index)
        strategy.on_bar(mctx)
        out.append(tuple(sorted(mctx.orders.items())))
    return out


class _PeekProbe(MultiAssetStrategy):
    """Deliberate lookahead for the break-it probe.

    Reads bar ``i + 1``'s close directly off the SAME aligned column
    arrays the real engine builds ``MultiAssetContext`` from, bypassing
    ``mctx.bar`` entirely -- exactly the bug class this file's own
    hand-rolled event loop is at risk of (R-21: an ``i + 1`` index inside
    ``on_bar`` returned $3.7e23 with a fully green suite). Unlike
    ``JointKellyDemo``, whose signal only updates once per calendar day
    (so a 1-bar peek within the same day is invisible -- confirmed by an
    earlier, weaker version of this probe that used index-shifted context
    on the real strategy and correctly failed to disagree), this probe is
    bar-to-bar sensitive by construction, so a 1-bar peek is guaranteed to
    show up whenever the peeked bar has been tampered.
    """

    name = "_probe_peeker"
    warmup = 0

    def __init__(self, cols: dict[str, dict]) -> None:
        self._cols = cols  # bound to the SAME arrays the engine will index with mctx.i

    def on_bar(self, mctx: MultiAssetContext) -> None:
        a, b = sorted(mctx.instruments)
        nxt = float(self._cols[a]["close"][mctx.i + 1])   # <-- the deliberate leak
        cur = float(mctx.bar(a)["close"])
        mctx.order_target(a, 1.0 if nxt > cur else -1.0)
        mctx.order_target(b, 0.0)


def _decisions_peek_probe(dfs: dict[str, pd.DataFrame], bars: list[int],
                          market: MarketSpec) -> list[tuple]:
    common_index, aligned, cols = _align(dfs)
    strategy = _PeekProbe(cols)
    broker = MultiAssetBroker(list(aligned.keys()), market, 10_000.0)
    out = []
    for i in bars:
        mctx = MultiAssetContext(cols, aligned, i, broker, common_index)
        strategy.on_bar(mctx)
        out.append(tuple(sorted(mctx.orders.items())))
    return out


def causality_check(data_dir: str = "data") -> dict:
    """Mandatory no-lookahead proof for the native engine + demo strategy.

    Two *opposite* tampers (up x3, down /3) rather than clean-vs-tampered,
    so a leak is forced to show as a decision DIFFERENCE rather than left
    to chance matching by coincidence -- mirrors
    ``tests/test_causality_strict.py``'s own pattern exactly, generalized
    to per-instrument order dicts instead of single-instrument order
    tuples.

    Then the deliberate break-it probe: the SAME two tampered inputs
    re-run through ``_decisions_peek_probe`` (a strategy with a genuine
    ``i + 1`` peek baked into ``on_bar``), which must now DISAGREE
    (proving the check has teeth) rather than pass.
    """
    btc, eth = load_assets(data_dir)
    dfs = {"BTC": btc, "ETH": eth}
    strategy = JointKellyDemo()

    prepared = strategy.prepare({k: v.copy() for k, v in dfs.items()})
    common_index, _, _ = _align(prepared)
    n = len(common_index)
    cut_pos = n - 5_000
    cut_ts = common_index[cut_pos]
    # k=0 (bar == cut_pos) is included deliberately: it is itself untampered
    # (the tamper mask is `index > cut_ts`, strict), so the real strategy
    # must still match there -- and it is the bar whose i+1 neighbour is the
    # FIRST tampered bar, which is exactly what the break-it probe needs to
    # have any chance of catching a one-bar peek.
    bars = [cut_pos - k for k in (0, 1, 2, 3, 5, 10, 20)]

    up = _tamper(dfs, cut_ts, 3.0)
    down = _tamper(dfs, cut_ts, 1.0 / 3.0)

    market = MarketSpec.spot()
    a = _decisions_multi(JointKellyDemo(), up, bars, market)
    b = _decisions_multi(JointKellyDemo(), down, bars, market)
    real_pass = all(x == y for x, y in zip(a, b))

    a2 = _decisions_peek_probe(up, bars, market)
    b2 = _decisions_peek_probe(down, bars, market)
    probe_disagrees = any(x != y for x, y in zip(a2, b2))

    print(f"causality check: cut={cut_ts}, bars={bars}")
    print(f"  real engine + JointKellyDemo: decisions at/before cut identical "
          f"under opposite post-cut tampers: {real_pass}")
    for k, (bar, oa, ob) in enumerate(zip(bars, a, b)):
        print(f"    bar {bar}: up={oa}  down={ob}  match={oa == ob}")
    print(f"  BREAK-IT PROBE (_PeekProbe, deliberate i+1 index inside on_bar): "
          f"decisions now DISAGREE under opposite tampers: {probe_disagrees} "
          f"(expected True -- proves the check catches a real off-by-one)")
    for k, (bar, oa, ob) in enumerate(zip(bars, a2, b2)):
        print(f"    bar {bar}: up={oa}  down={ob}  match={oa == ob}")

    ok = real_pass and probe_disagrees
    print(f"\nCAUSALITY SUITE RESULT: {'PASS' if ok else 'FAIL'} "
          f"(real engine causal AND probe demonstrated to have teeth)")
    return {"real_pass": real_pass, "probe_disagrees": probe_disagrees, "ok": ok}


# =========================================================================
# 8. Demo: the report numbers
# =========================================================================

def run_demo(data_dir: str = "data") -> dict:
    global N_EVALUATED
    btc, eth = load_assets(data_dir)
    dfs = {"BTC": btc, "ETH": eth}

    results: dict[str, dict] = {}

    print("=== joint 2-asset Sigma^-1 mu allocator (native engine, shared budget) ===")
    for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
        sliced = {k: v.loc[s:e] for k, v in dfs.items()}
        res = run_multi_backtest_native(JointKellyDemo(), sliced, SPOT, 1000.0)
        N_EVALUATED += 1
        m = multi_metrics(res)
        avg_w = {inst: float((res.positions[inst] * res.closes[inst]).div(res.equity).mean())
                 for inst in dfs}
        results[f"joint_{label}"] = {**m, "avg_weight": avg_w, "dropped_bars": res.dropped_bars,
                                     "fees_by_inst": res.fees_by_inst}
        print(f"[{label}] final={m['final_balance']:.1f} sharpe={m['sharpe']:.2f} "
              f"maxDD={m['max_dd_pct']:.1f}% fees={m['fees_paid']:.2f} "
              f"avg_weight(BTC={avg_w['BTC']:.3f}, ETH={avg_w['ETH']:.3f}) "
              f"dropped_bars={res.dropped_bars}")

    print("\n=== per-instrument, N=1 special case (same engine, one instrument) ===")
    for inst in ("BTC", "ETH"):
        for label, (s, e) in (("train", (TRAIN_START, TRAIN_END)), ("valid", (VALID_START, VALID_END))):
            sliced = {inst: dfs[inst].loc[s:e]}
            res = run_multi_backtest_native(SingleAssetHold(), sliced, SPOT, 1000.0)
            N_EVALUATED += 1
            m = multi_metrics(res)
            results[f"solo_{inst}_{label}"] = m
            print(f"[{inst} solo, {label}] final={m['final_balance']:.1f} "
                  f"sharpe={m['sharpe']:.2f} maxDD={m['max_dd_pct']:.1f}% fees={m['fees_paid']:.2f}")

    print(f"\nconfigs evaluated (backtest runs through run_multi_backtest_native): {N_EVALUATED}")
    print("(causality diagnostic runs are NOT counted here -- same convention as "
          "kelly_regime_covkelly.py's own N_EVALUATED, which only counts runs that "
          "could inform a selection decision; this round makes no selection or "
          "promotion claim at all.)")
    return results


# ===================================================================== CLI

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "demo":
        run_demo()
    elif cmd == "causality":
        causality_check()
    elif cmd == "all":
        causality_check()
        print()
        run_demo()
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
