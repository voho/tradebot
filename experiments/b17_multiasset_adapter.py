#!/usr/bin/env python
"""B-17 (infrastructure only): can a genuinely multi-asset strategy be made
registrable at all, without touching the existing single-asset engine?

CONSERVATIVE branch: adapter / composition design.

Not registered, not a strategy, not a trading claim: lives under
``experiments/`` (not auto-discovered) and does not modify anything under
``src/tradebot/``. R-43 already tested the dual-asset BTC+ETH
diversification FINDING on the project's real 2023+ holdout and it was
REJECTED (LEDGER.md row R-43) -- this file does not re-test or re-argue
that finding. The question here is purely mechanical: given that some
future dual-asset finding *does* clear a holdout one day, is there a
clean, additive way to get it into the comparison table? B-17 (open,
OPEN in LEDGER.md D. Backlog) is exactly this gap: the ``Strategy`` base
class, ``registry.py`` and ``tradebot run`` all assume one instrument per
registered class, so even R-42/R-43's genuine dual-asset
``kelly_regime_v4`` branches could never be registered even in principle.

============================================================================
DESIGN NOTE -- what this adapter is, and what it would take to adopt it
============================================================================

The adapter, in one sentence
-----------------------------
Run the UNMODIFIED single-asset engine (``tradebot.engine.run_backtest``,
via ``tradebot.window.run_period`` for fair warmup handling -- both
existing, untouched files) once per instrument, each leg with its own
fresh ``Strategy`` instance, its own isolated ``PaperBroker``/account, on
that instrument's own data and its own slice of starting capital; then
compose the resulting per-leg ``BacktestResult.equity`` curves into one
portfolio equity curve by summation (a fixed, pre-decided capital split),
and feed that combined curve back through the existing, unmodified
``tradebot.metrics.compute_metrics`` by wrapping it in a synthetic
``BacktestResult``. Nothing about how a single instrument is simulated,
feed, filled, liquidated or funded changes even one line.

This is precisely the pattern ``experiments/kelly_regime_dual_fixed.py``
(R-42/R-43) already used ad hoc, generalized into two reusable pieces
(``MultiAssetSpec`` / ``run_multi_backtest``) instead of one-off script
functions, so a future strategy does not have to re-derive
``combine_equity`` from scratch.

1. What would change in ``src/tradebot/registry.py``
------------------------------------------------------
Today: one flat ``_REGISTRY: dict[str, type[Strategy]]``, one decorator
``@register``, ``available_strategies()`` returns all of it, and every
consumer (``run.py``, ``tests/test_causality_strict.py``,
``tests/test_evidence.py``) iterates that single dict assuming each entry
is a plain one-``df`` ``Strategy``.

The additive, lowest-risk change: do NOT fork the registry. Let a
multi-asset strategy declare its instruments as a class attribute,
defaulting to empty (= "ordinary single-asset strategy, unchanged
meaning"):

    class KellyRegimeDualBTCETH(Strategy):
        name = "kelly_regime_dual_50_50"
        instruments = ("BTC", "ETH")          # NEW, defaults to () today
        weights = (0.50, 0.50)                # NEW, or a classmethod that
                                               # computes them causally from
                                               # a training slice, mirroring
                                               # dual_fixed.py's vol_weighted
        def make_leg(self, asset: str) -> Strategy:
            # returns a fresh, per-leg Strategy instance (may be the SAME
            # class run on every leg, as kelly_regime_v4 is here, or a
            # different sub-strategy per asset -- the adapter does not care)
            return KellyRegimeV4()

    @register  # UNCHANGED decorator, UNCHANGED dict
    class KellyRegimeDualBTCETH(...): ...

``available_strategies()`` stays byte-for-byte unchanged (still every
registered class, single- and multi-asset alike -- nothing currently
iterating it breaks). A new, additive helper:

    def available_multi_asset_strategies() -> dict[str, type[Strategy]]:
        return {n: c for n, c in available_strategies().items()
                if getattr(c, "instruments", ())}

is the only new registry-level code. This is easy: it is a pure filter
on an attribute, no new discovery mechanism, no new decorator, no risk to
the 25 existing single-asset registrations.

2. What would change in ``src/tradebot/run.py`` (``run_matrix``)
--------------------------------------------------------------------
Today ``run_matrix`` loads exactly one ``(df, label)`` per market *kind*
("spot" / "perp") ONCE, shared by every strategy in the inner loop, then
calls ``run_backtest(strategy, df, spec, balance)`` once per
(strategy, market, balance). A multi-asset strategy breaks that shared-
dataset assumption directly: it needs a *different* dataset per leg
(BTC spot AND ETH spot are not the same ``df``), which the current
``datasets: dict[str, tuple]`` (keyed only by market kind, not by asset)
cannot express at all -- this is the concrete, structural reason B-17 is
a real gap and not just "add one more strategy".

Two things are needed, both additive:

  (a) ``tradebot.data`` needs an asset-aware load path. Today
      ``CANONICAL``/``load_dataset(data_dir, kind)`` only knows "spot" and
      "perp", both implicitly BTC. ETH already has its own loader
      (``load_coinbase_eth_spot``) but it is a special case, not a case of
      the same interface. A minimal generalization: a
      ``load_dataset(data_dir, kind, asset="BTC")`` parameter (or a
      registry of loaders keyed by ``(asset, kind)``) so ``run_matrix``
      can resolve "the dataset instrument X of strategy Y needs" the same
      way it resolves "spot" today.

  (b) ``run_matrix`` needs a second code path alongside (not replacing)
      today's per-market loop: split ``names`` into single-asset (existing
      path, untouched) and multi-asset (``instruments`` non-empty). For
      each multi-asset strategy: instantiate one leg via
      ``strategy.make_leg(asset)`` per declared instrument, resolve each
      leg's own dataset via (a), build ``MultiAssetSpec`` objects exactly
      as this file's ``demo_50_50_btc_eth_spot`` does below, and call
      ``run_multi_backtest`` in place of ``run_backtest``. The result's
      portfolio-level ``Metrics`` (from ``compute_metrics`` on the
      synthetic combined ``BacktestResult`` -- see below) is appended to
      the SAME ``all_metrics: list[Metrics]`` that already threads through
      to ``comparison_report``/``update_readme`` -- this is the one part
      of the whole adoption path that requires no new code at all, because
      ``run_multi_backtest`` in this file already returns an ordinary
      ``Metrics`` object for the portfolio, indistinguishable in *type*
      from any single-asset strategy's.

  Awkward part, stated plainly: ``run_matrix``'s market loop currently
  decides ONE ``MarketSpec`` per iteration (spot or futures_5x) shared by
  every strategy in it; a multi-asset strategy may reasonably want
  different markets per leg (BTC futures + ETH spot), which has no home
  in that loop structure and would need its own market-resolution rule
  per instrument, not per matrix cell.

3. What would change in ``src/tradebot/report.py`` / the README table
---------------------------------------------------------------------
The table's row key today is effectively (strategy name, market). Because
``run_multi_backtest`` already funnels its portfolio result through the
UNMODIFIED ``compute_metrics``, the portfolio row is a normal ``Metrics``
instance and ``matrix_table``/``markdown_table`` render it with ZERO code
changes -- sorting by final balance, the money/percent formatting, the
evidence columns, all of it already works on any ``Metrics``, however it
was produced. That is the single biggest practical payoff of routing
through ``compute_metrics`` rather than hand-rolling portfolio stats.

What is genuinely awkward, and does need new table design:
  - The table has no column today for "which instruments, at what split".
    That context would have to live in the strategy-NAME string (e.g.
    ``kelly_regime_dual_50_50``, ``kelly_regime_dual_vol_weighted`` as
    two separate registered rows for two weight schemes) -- workable, but
    it turns "one strategy, several weight variants" into several table
    rows the way plain parameter sweeps already do for single-asset
    strategies (nothing new there), rather than one row with a "weights"
    sub-column (which WOULD be new report.py code).
  - The table's ``market`` column has no honest label for "every leg
    traded spot, but on two different underlyings" versus "one leg spot,
    one leg futures" -- a synthetic label such as ``"spot"`` (if uniform)
    or ``"mixed"`` (if not) is needed, and ``evidence.py``'s
    ``(strategy, market)`` lookup key would need to accept it.
  - The table shows the PORTFOLIO row only; the individual BTC-only and
    ETH-only leg numbers (also produced here, as ``leg_metrics``) have no
    place in ``matrix_table`` as it exists -- and showing only the
    portfolio row hides exactly the diversification mechanism the
    strategy exists to demonstrate. This would need either a second,
    per-leg appendix table (``docs/STRATEGIES.md``-style, not
    ``matrix_table``) or a genuinely new nested-row format.

4. What would change in CI (``tests/test_evidence.py`` / bootstrap.csv)
-------------------------------------------------------------------------
Today: ``for name in available_strategies(): for market in ("spot",
"futures_5x"): assert (name, market) in evidence``. Two ways to
generalize, both real code changes, and worth stating honestly rather
than picking one and calling it solved:

  (a) Bootstrap the PORTFOLIO curve only, as one more (strategy, market)
      cell, with a synthetic market label. Cheapest to build --
      ``scripts/inference.py``'s block bootstrap runs on a return series,
      and the synthetic portfolio ``BacktestResult.equity`` this file
      produces IS an ordinary return series, so the bootstrap MACHINERY
      itself needs no changes at all, only a new place to call it from and
      a new market-label convention. Downside: an interval on the
      portfolio number alone says nothing about whether the diversifying
      LEG is pulling its weight versus just being along for the ride
      (exactly the R-33 "match risk before crediting a drawdown edge"
      concern this project already applies everywhere else).
  (b) Bootstrap every leg AND the portfolio (N+1 intervals for an N-asset
      strategy). More honest, more expensive, and it means
      ``test_evidence.py``'s hardcoded ``("spot", "futures_5x")`` tuple
      has to become a per-strategy-declared requirement (how many cells
      does THIS strategy need) rather than a global constant -- a real,
      non-trivial rewrite of that test's assumptions, not a one-line
      change.

  Either way, ``available_multi_asset_strategies()`` (section 1) is the
  right iteration source, and it does not yet exist.

Strengths and weaknesses of the adapter design, stated plainly
------------------------------------------------------------------
Strengths: zero risk to the ~25 existing single-asset registrations, the
CI suite, or ``engine.py``/``strategy.py`` (this project's most heavily
causality-tested files, by a wide margin -- see ``test_causality_strict.py``
and R-21's $3.7e23 cautionary tale). The entire correctness burden for
per-instrument fills, fees, liquidation and funding stays inside the
already-audited ``run_backtest``/``PaperBroker``. The composition logic
itself (``combine_equity_curves``, ~15 lines) is small enough to
causality-test in complete isolation, which section 5 below does. It can
be adopted incrementally -- one strategy at a time -- without touching
``registry.py``, ``run.py`` or ``report.py`` at all until someone actually
wants a table row.

Weaknesses -- the honest, central one first: **this design cannot express
a strategy that needs a SHARED risk or leverage budget decided DURING the
run, only one decided in advance.** Each leg's ``PaperBroker`` and
``Context`` are fully isolated processes; a BTC leg's ``on_bar`` has
*no way to see* the ETH leg's live equity, drawdown or position, and
vice versa, because they are run to completion as two separate calls to
``run_backtest`` before their equity curves are ever combined. Concretely,
this adapter CAN express "BTC gets 50% of starting capital, ETH gets 50%,
fixed for the whole run" or "BTC gets a vol-inverse-weighted share
computed once from training data" (both demonstrated below, mirroring
``kelly_regime_dual_fixed.py``'s ``50_50``/``vol_weighted`` splits) -- but
it CANNOT express "if BTC's leg is drawing down right now, temporarily
lend some of its idle margin to ETH's leg", or any genuinely cross-
sectional Kelly allocator that reallocates capital bar-by-bar based on a
live, shared covariance/mean estimate across legs. That second kind of
strategy is exactly what ``experiments/kelly_regime_covkelly.py``
(R-42/R-43, B-18) already is -- a live Sigma^-1*mu allocator that needs
simultaneous, shared state across BTC and ETH inside one decision -- and
this adapter, honestly, cannot host it, not merely "with more code": no
composition of independently-completed backtests can recover a decision
that depended on both legs' live joint state at the same bar. That family
needs a native engine change: one shared bar clock, one ``MultiAssetContext``
handing a strategy simultaneous read access to N brokers/frames inside a
single ``on_bar`` call. A native redesign is also the only way to get
real cross-margining (currently each leg's leverage cap is strictly
per-leg capital, never portfolio capital) and to remove the very
cadence-inconsistency B-18 flags as possibly a rebalance-ENGINE artifact
of ``kelly_regime_covkelly``'s current per-segment-restart workaround --
which is itself a hint that a native engine is the more durable answer
for that specific open question, at a much larger engineering cost (it
touches ``engine.py``'s per-bar loop, ``strategy.py``'s ``Context``, and
the causality-test suite's core assumptions, with a correspondingly
larger risk of a subtle new lookahead bug living in the SHARED-clock code
itself rather than in any one strategy). The adapter answers "can an
already-independent multi-book strategy reach the table" (yes, cheaply);
it does not and cannot answer "can a strategy that must see all legs
simultaneously be expressed" (no, not even in principle).

Two smaller, secondary weaknesses, worth naming rather than hiding:
  - No true cross-margining: a leg's leverage cap is always relative to
    ITS OWN allocated capital, never to unused capital sitting idle in
    another leg, even briefly.
  - Trade-level portfolio statistics (``num_trades``, ``win_rate_pct``,
    ``best_trade``/``worst_trade``/``avg_trade`` -- everything
    ``compute_metrics`` derives from ``result.trades``) are a plain
    concatenation of each leg's own trades. That is not WRONG, but it
    mixes BTC-dollar and ETH-dollar round trips with no cross-leg
    netting, and it should be read as "every trade any leg made", not as
    a single book's trade history. Sharpe/drawdown/final-balance ARE
    genuinely portfolio-level (computed straight off the summed equity
    curve), so those numbers do not share this caveat.

============================================================================
Data-scope rule (grepped for confirmation before every run below)
============================================================================
Every date literal in this file is at or before 2022-12-31. BTC and ETH
are loaded via the project's own existing loaders (``load_dataset``,
``load_coinbase_eth_spot``, both real, both unchanged) and then
IMMEDIATELY sliced to ``.loc[:"2022-12-31"]`` -- one line below -- before
either frame is used anywhere else in this file, so a full 2017-2026 (BTC)
or 2019-2026 (ETH) file sitting on disk is never mistaken for permission
to read past the holdout boundary. Grep this file for "2023"/"2024"/
"2025"/"2026": the only hits are this docstring's own prose describing
the boundary and citation years in the reference experiment file this one
imports functions from (never executes past its own module-level code).

Inner-train = 2019-03-14 (ETH's real start on the committed Coinbase
file) -> 2020-12-31. Inner-validation = 2021-01-01 -> 2022-12-31 (the
2022 BTC/ETH joint bear) -- identical constants to
``kelly_regime_dual_fixed.py``'s ``TRAIN``/``VALID``, deliberately, so the
sanity-check comparison in section 6 below is apples-to-apples.

Usage
-----
    python experiments/b17_multiasset_adapter.py demo        # the 50/50 BTC+ETH run
    python experiments/b17_multiasset_adapter.py causality   # truncation proof
    python experiments/b17_multiasset_adapter.py sanity      # cross-check vs kelly_regime_dual_fixed.py
    python experiments/b17_multiasset_adapter.py all         # everything, in order
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.engine import BacktestResult  # noqa: E402
from tradebot.metrics import Metrics, compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# --------------------------------------------------------------- data (pre-2023 only)

TRAIN = ("2019-03-14", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
HOLDOUT_BOUNDARY = "2022-12-31"  # inclusive upper bound on every read in this file

_btc_full, BTC_LABEL = load_dataset(ROOT / "data", "spot")
BTC = _btc_full.loc[:HOLDOUT_BOUNDARY].copy()  # <-- HARD RULE: sliced before any use

_eth_full = load_coinbase_eth_spot(ROOT / "data")
if _eth_full is None:
    raise RuntimeError("data/ethusd_coinbase_spot_5m.csv.gz not found -- cannot run this experiment")
ETH = _eth_full.loc[:HOLDOUT_BOUNDARY].copy()  # <-- HARD RULE: sliced before any use
ETH_LABEL = "real (Coinbase spot)"

SPOT = MarketSpec.spot()

INCUMBENT = "kelly_regime_v4"


# =============================================================================
# Reusable infrastructure: MultiAssetSpec / run_multi_backtest / MultiBacktestResult
# =============================================================================


@dataclass
class MultiAssetSpec:
    """One instrument leg of a multi-asset portfolio.

    Deliberately thin: an already-instantiated, fresh ``Strategy`` (so the
    caller controls whether every leg runs the SAME strategy class, as the
    demo below does, or a genuinely different one per asset), that
    strategy's own full OHLCV frame, and the ``MarketSpec`` it trades in.
    Nothing here is engine state -- ``run_multi_backtest`` is the only
    thing that touches ``tradebot.engine``.
    """

    label: str
    strategy: Strategy
    df: pd.DataFrame
    market: MarketSpec


@dataclass
class MultiBacktestResult:
    """The output of composing N independent single-asset backtests.

    ``leg_results``/``leg_metrics`` are exactly what
    ``tradebot.engine.run_backtest`` (via ``run_period``) and
    ``tradebot.metrics.compute_metrics`` would produce for each instrument
    run alone -- nothing about them is adapter-specific. ``portfolio`` is a
    SYNTHETIC ``BacktestResult`` this module builds (summed equity, pooled
    fills/trades, no genuine ``df``) purely so ``compute_metrics`` -- also
    completely unmodified -- can be reused to get portfolio-level
    Sharpe/drawdown/final-balance instead of hand-deriving them.
    """

    leg_labels: list[str]
    weights: list[float]
    leg_results: list[BacktestResult]
    leg_metrics: list[Metrics]
    portfolio: BacktestResult
    metrics: Metrics


def combine_equity_curves(leg_results: list[BacktestResult]) -> pd.Series:
    """Sum N independent equity curves onto their union bar grid.

    Forward-fills each leg between its own bar closes (equity is
    piecewise-constant there by construction) and backfills the very start
    of the union grid with that leg's OWN ``start_balance`` (flat, not yet
    resolved) -- never with a forward-looking value. Generalizes
    ``kelly_regime_dual_fixed.py``'s two-leg ``combine_equity`` to N legs;
    the per-leg logic is unchanged.
    """
    if not leg_results:
        raise ValueError("need at least one leg")
    idx = leg_results[0].equity.index
    for r in leg_results[1:]:
        idx = idx.union(r.equity.index)
    total = pd.Series(0.0, index=idx)
    for r in leg_results:
        e = r.equity.reindex(idx).ffill().fillna(r.start_balance)
        total = total + e
    return total.rename("equity")


def _synthetic_portfolio_result(leg_results: list[BacktestResult], equity: pd.Series,
                                start_balance: float) -> BacktestResult:
    """Wrap the combined equity curve in a ``BacktestResult`` so the
    UNMODIFIED ``compute_metrics`` can compute portfolio Sharpe/drawdown.

    ``df`` is a placeholder (an empty frame on the equity index):
    ``compute_metrics`` never reads ``result.df``. ``market`` is
    informational only, taken from the first leg -- see the design note's
    section 3 for why a portfolio with legs on different markets has no
    single honest market label yet.
    """
    names = sorted({r.strategy_name for r in leg_results})
    return BacktestResult(
        strategy_name="+".join(names),
        market=leg_results[0].market,
        start_balance=start_balance,
        data_label="+".join(r.data_label for r in leg_results),
        equity=equity,
        fills=[f for r in leg_results for f in r.fills],
        trades=[t for r in leg_results for t in r.trades],
        df=pd.DataFrame(index=equity.index),
        liquidated=any(r.liquidated for r in leg_results),
        fees_paid=sum(r.fees_paid for r in leg_results),
        funding_paid=sum(r.funding_paid for r in leg_results),
    )


def run_multi_backtest(
    specs: list[MultiAssetSpec],
    weights: list[float],
    start_balance: float,
    start: str | None = None,
    end: str | None = None,
    slippage_bps: float = 0.0,
) -> MultiBacktestResult:
    """Run N single-asset legs independently through the UNMODIFIED engine,
    then compose them into one portfolio-level result.

    Each leg is executed via ``tradebot.window.run_period`` (itself a thin,
    unmodified fair-warmup wrapper around ``tradebot.engine.run_backtest``
    -- the same call every other experiment file in this repo uses for a
    date-bounded slice) with its OWN fresh ``Strategy`` instance, its own
    isolated ``PaperBroker``, on its own data, capitalized at
    ``start_balance * weight``. No leg can see any other leg's state at any
    point during its own run -- see the design note above for exactly what
    that costs.

    ``weights`` must sum to 1.0 (a fixed capital split decided BEFORE the
    run, whether a hardcoded constant or -- like
    ``kelly_regime_dual_fixed.py``'s ``vol_weighted`` split -- a rule
    computed causally from a training-only slice). This is enforced, not a
    convention: the strategy of relaxing it later (e.g. summing to more
    than 1.0 to express shared leverage) is exactly the "shared risk
    budget" capability this design does not have.
    """
    if len(specs) != len(weights):
        raise ValueError(f"{len(specs)} specs but {len(weights)} weights")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1.0 (a fixed split decided in advance); got {sum(weights)!r}. "
            "This adapter cannot express a shared/overlapping risk budget across legs.")

    leg_results: list[BacktestResult] = []
    for spec, w in zip(specs, weights):
        leg_balance = start_balance * w
        result = run_period(
            spec.strategy, spec.df, start, end,
            market=spec.market, start_balance=leg_balance,
            slippage_bps=slippage_bps, data_label=spec.label,
        )
        leg_results.append(result)

    equity = combine_equity_curves(leg_results)
    portfolio = _synthetic_portfolio_result(leg_results, equity, start_balance)
    leg_metrics = [compute_metrics(r) for r in leg_results]
    metrics = compute_metrics(portfolio)

    return MultiBacktestResult(
        leg_labels=[s.label for s in specs],
        weights=list(weights),
        leg_results=leg_results,
        leg_metrics=leg_metrics,
        portfolio=portfolio,
        metrics=metrics,
    )


# =============================================================================
# 3. Demonstration: kelly_regime_v4 (already-registered, unmodified) x2,
#    BTC + ETH, 50/50 fixed split, spot market, inner-train+inner-validation.
# =============================================================================


def demo_50_50_btc_eth_spot(total_balance: float = 1_000.0) -> MultiBacktestResult:
    specs = [
        MultiAssetSpec("BTC", get_strategy(INCUMBENT), BTC, SPOT),
        MultiAssetSpec("ETH", get_strategy(INCUMBENT), ETH, SPOT),
    ]
    return run_multi_backtest(specs, [0.5, 0.5], total_balance,
                              start=TRAIN[0], end=VALID[1])


def _print_leg(label: str, m: Metrics) -> None:
    print(f"  {label:24s} final=${m.final_balance:>10,.2f}  sharpe={m.sharpe:>6.2f}  "
          f"maxDD={m.max_drawdown_pct:>5.1f}%  trades={m.num_trades}")


def demo() -> MultiBacktestResult:
    print(f"BTC: {len(BTC):,} bars {BTC.index[0]:%Y-%m-%d} -> {BTC.index[-1]:%Y-%m-%d} (data: {BTC_LABEL})")
    print(f"ETH: {len(ETH):,} bars {ETH.index[0]:%Y-%m-%d} -> {ETH.index[-1]:%Y-%m-%d} (data: {ETH_LABEL})")
    print(f"\n=== demo: {INCUMBENT} x2, BTC+ETH, 50/50, spot, {TRAIN[0]} -> {VALID[1]} ===")
    t0 = time.time()
    d = demo_50_50_btc_eth_spot()
    print(f"(ran in {time.time() - t0:.0f}s)\n")
    for label, m in zip(d.leg_labels, d.leg_metrics):
        _print_leg(f"{label} leg (${d.weights[d.leg_labels.index(label)] * 1000:,.0f} start)", m)
    print(f"  {'-' * 70}")
    _print_leg("PORTFOLIO (50/50 combined)", d.metrics)
    return d


# =============================================================================
# 5. Causality check: does the composition itself look ahead?
#
#    kelly_regime_v4's OWN causality is already covered by
#    tests/test_causality_strict.py and re-verified for BTC and ETH
#    individually in kelly_regime_dual_fixed.py's causality() (probes 1-2).
#    What is NEW here, and not covered by either, is whether THIS module's
#    composition step (run_multi_backtest / combine_equity_curves) itself
#    introduces lookahead -- e.g. via reindex/ffill touching a future bar.
#    Mirrors the truncation-test pattern of test_causality_strict.py: run
#    once on data truncated at a cut date, run again on the full span, and
#    require BIT-IDENTICAL portfolio equity for every bar strictly before
#    the cut.
# =============================================================================


def causality_check(cut: str = "2021-06-30") -> tuple[bool, float, int]:
    print(f"\n=== causality check: truncate at {cut} vs full {TRAIN[0]}->{VALID[1]}, "
          "compare portfolio equity before the cut ===")

    specs_full = [
        MultiAssetSpec("BTC", get_strategy(INCUMBENT), BTC, SPOT),
        MultiAssetSpec("ETH", get_strategy(INCUMBENT), ETH, SPOT),
    ]
    full = run_multi_backtest(specs_full, [0.5, 0.5], 1_000.0, start=TRAIN[0], end=VALID[1])

    specs_trunc = [
        MultiAssetSpec("BTC", get_strategy(INCUMBENT), BTC, SPOT),
        MultiAssetSpec("ETH", get_strategy(INCUMBENT), ETH, SPOT),
    ]
    trunc = run_multi_backtest(specs_trunc, [0.5, 0.5], 1_000.0, start=TRAIN[0], end=cut)

    idx = trunc.portfolio.equity.index
    before_cut = idx[idx <= pd.Timestamp(cut, tz="UTC")]
    full_eq = full.portfolio.equity.reindex(before_cut)
    trunc_eq = trunc.portfolio.equity.reindex(before_cut)
    diff = (full_eq - trunc_eq).abs()
    worst = float(diff.max())
    ok = worst < 1e-6
    print(f"  bars compared: {len(before_cut):,}")
    print(f"  max|portfolio equity diff| strictly at/before cut = {worst:.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  ({'no lookahead introduced by the composition step' if ok else 'FAIL -- investigate combine_equity_curves'})")
    return ok, worst, len(before_cut)


# =============================================================================
# 6. Sanity check against kelly_regime_dual_fixed.py's own 50/50 numbers,
#    for the SAME window (TRAIN[0] -> VALID[1]), via ITS OWN code path
#    (run_dual), not a re-derivation. Different code (this file's
#    run_multi_backtest/combine_equity_curves vs their run_dual/
#    combine_equity), same underlying data, same strategy, same split --
#    they need not match exactly, but should be in the same ballpark.
# =============================================================================


def _load_reference_module():
    """Import kelly_regime_dual_fixed.py by path, without executing its
    ``__main__`` block (module-level code only loads data and computes the
    vol_weighted split -- no backtest runs at import time)."""
    path = ROOT / "experiments" / "kelly_regime_dual_fixed.py"
    spec = importlib.util.spec_from_file_location("kelly_regime_dual_fixed_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sanity_check() -> dict:
    print(f"\n=== sanity check vs experiments/kelly_regime_dual_fixed.py's run_dual('50_50', "
          f"{TRAIN[0]!r}, {VALID[1]!r}, SPOT) ===")
    ref = _load_reference_module()
    t0 = time.time()
    ref_d = ref.run_dual("50_50", TRAIN[0], VALID[1], market=ref.SPOT)
    print(f"(reference run in {time.time() - t0:.0f}s)")

    mine = demo_50_50_btc_eth_spot()

    print(f"\n  {'metric':22s} {'this adapter':>16s} {'kelly_regime_dual_fixed.py':>28s} {'diff':>12s}")
    rows = [
        ("final balance ($)", mine.metrics.final_balance, ref_d["final"]),
        ("sharpe", mine.metrics.sharpe, ref_d["sharpe"]),
        ("max drawdown (%)", mine.metrics.max_drawdown_pct, ref_d["max_dd"]),
    ]
    for name, a, b in rows:
        print(f"  {name:22s} {a:>16,.4f} {b:>28,.4f} {a - b:>+12.4f}")
    return {"adapter": mine.metrics, "reference": ref_d}


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "demo":
        demo()
    elif choice == "causality":
        causality_check()
    elif choice == "sanity":
        sanity_check()
    elif choice == "all":
        demo()
        causality_check()
        sanity_check()
    else:
        print("usage: python experiments/b17_multiasset_adapter.py [demo|causality|sanity|all]")
