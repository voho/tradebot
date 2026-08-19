#!/usr/bin/env python
"""Driver for backlog B-05 (conservative) — funding-decile gate on kelly_regime_v4.

See ``experiments/funding_gate_conservative.py`` for the strategy and its
mechanism docstring. This file is the harness: it loads data + real
funding, merges the causal funding column onto the bar frame, sweeps the
gate's two parameters on inner-train, freezes one configuration by a rule
pre-registered below (before the sweep is even run), reports its
inner-validation numbers, then runs the pre-registered falsification test
(a 0.40% taker fee tier) and a funding-cost comparison.

**Hard boundary, enforced in code, not just by convention**: this file
must never evaluate any period reaching 2023-01-01 or later. The 2023+
holdout belongs to a separate, pre-registered evaluation the operator
runs later; reading it here would corrupt that evaluation. ``_period()``
refuses any ``start``/``end`` on or after that date.

Splits (ROUTINE.md step 3 shape, restricted to the funding-covered window
2020-01-01..2023-12-31, further restricted here to 2020-2022 per this
task's explicit instructions)::

    inner-train       2020-01-01 -> 2021-12-31   fit / sweep
    inner-validation  2022-01-01 -> 2022-12-31   report the frozen config

Usage::

    python experiments/run_funding_gate_conservative.py sweep       # step 3
    python experiments/run_funding_gate_conservative.py validate    # step 4
    python experiments/run_funding_gate_conservative.py feetier     # step 5 (falsification)
    python experiments/run_funding_gate_conservative.py fundingcost # step 6
    python experiments/run_funding_gate_conservative.py causality   # step 7
    python experiments/run_funding_gate_conservative.py all
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

from experiments.funding_signal import causal_funding_column  # noqa: E402
from experiments.funding_gate_conservative import FundingGateConservative  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")
if FUNDING is None:
    raise SystemExit("no funding data committed; see docs/VALIDATION.md")

# The funding-merged frame every strategy in this file trades on. Baselines
# (kelly_regime_v4) simply ignore the extra column.
DFF = DF.copy()
DFF["funding"] = causal_funding_column(DFF.index, FUNDING)

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

HOLDOUT_BOUNDARY = "2023-01-01"
TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")
COMBINED = ("2020-01-01", "2022-12-31")

GRID_LOOKBACK = (30, 90, 180)
GRID_THRESHOLD = (0.90, 0.95)
N_CONFIGS_SWEPT = len(GRID_LOOKBACK) * len(GRID_THRESHOLD)  # 6, for deflated-Sharpe bookkeeping

# ---------------------------------------------------------------------------
# PRE-REGISTRATION — written before the sweep numbers below were computed.
#
# Selection rule: of the 6 (lookback_days, threshold) configs, freeze the
# ONE with the best INNER-TRAIN Sharpe on FUTURES (5x). Apply it mechanically
# to whatever the sweep actually prints; do not eyeball the table first.
#
# Falsification test: at the frozen config, re-run inner-train (spot) at the
# Bitstamp entry taker tier (0.40%, vs the 0.10% every other number in this
# file uses) alongside the unmodified kelly_regime_v4 baseline at the same
# fee. If the gate's advantage over the baseline collapses or reverses at
# 0.40%, the mechanism is read as "the gate merely trims turnover, and R-13's
# general finding (the spot edge lives inside the fee margin) applies here
# too" — a fail, not a partial pass to be argued around.
# ---------------------------------------------------------------------------


def _assert_in_bounds(start, end) -> None:
    for d in (start, end):
        if d is not None and str(d) >= HOLDOUT_BOUNDARY:
            raise ValueError(
                f"refusing to evaluate {d!r} - on/after the {HOLDOUT_BOUNDARY} "
                "holdout boundary is out of scope for this file"
            )


N_BACKTESTS = 0  # every distinct (strategy, market, period, fee-tier) backtest actually run


def _period(strategy, market, start=None, end=None, *, funding=None, balance=1_000.0):
    """Warmed sub-period backtest (the run_period pattern), optionally with real funding.

    Reimplements ``tradebot.window.run_period``'s warm-prefix trick by hand
    (rather than calling it) because ``run_period`` does not expose the
    ``funding`` kwarg that ``run_backtest`` does — this is exactly the
    pattern ``scripts/funding_study.py``'s ``_period`` helper uses.
    """
    global N_BACKTESTS
    _assert_in_bounds(start, end)
    lo = 0 if start is None else int(DFF.index.searchsorted(start))
    hi = len(DFF) if end is None else int(DFF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DFF.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    N_BACKTESTS += 1
    return compute_metrics(trimmed), raw.funding_paid


def _fmt(tag: str, market_label: str, m) -> str:
    return (f"{tag:28s} {market_label:8s} final=${m.final_balance:>11,.0f} "
            f"({m.profit_pct:>+8.1f}%) trades={m.num_trades:>4d} "
            f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f}")


# --------------------------------------------------------------------- sweep

def sweep() -> list[dict]:
    """Step 3: 6 configs x 2 markets on inner-train, plus the baseline."""
    print(f"=== step 3: inner-train sweep, {TRAIN[0]} .. {TRAIN[1]} ({N_CONFIGS_SWEPT} configs) ===\n")
    for label, market in MARKETS:
        m, _ = _period(get_strategy("kelly_regime_v4"), market, *TRAIN)
        print(_fmt("baseline kelly_regime_v4", label, m))
    print()
    rows = []
    for lb in GRID_LOOKBACK:
        for th in GRID_THRESHOLD:
            for label, market in MARKETS:
                strat = FundingGateConservative(lookback_days=lb, threshold=th)
                m, _ = _period(strat, market, *TRAIN)
                rows.append(dict(lookback_days=lb, threshold=th, market=label,
                                  final=m.final_balance, sharpe=m.sharpe,
                                  dd=m.max_drawdown_pct, trades=m.num_trades))
                print(_fmt(f"gate lb={lb:>3d}d th={th:.2f}", label, m))
    return rows


def select_frozen(rows: list[dict]) -> tuple[int, float]:
    """Apply the pre-registered rule: best inner-train Sharpe on futures."""
    fut_rows = [r for r in rows if r["market"] == "futures"]
    best = max(fut_rows, key=lambda r: r["sharpe"])
    print(f"\npre-registered selection rule applied: best inner-train futures "
          f"Sharpe -> lookback_days={best['lookback_days']}, "
          f"threshold={best['threshold']:.2f} (Sharpe={best['sharpe']:.2f})")
    return best["lookback_days"], best["threshold"]


# ------------------------------------------------------------ frozen config
#
# Determined by running `sweep()` + `select_frozen()` once (recorded in
# funding_gate_conservative_REPORT.md's results table) and hardcoded here so
# every other command in this file uses the identical, already-frozen
# configuration rather than re-deriving it.

FROZEN_LOOKBACK_DAYS = 180
FROZEN_THRESHOLD = 0.95


def _frozen() -> FundingGateConservative:
    return FundingGateConservative(lookback_days=FROZEN_LOOKBACK_DAYS, threshold=FROZEN_THRESHOLD)


# ----------------------------------------------------------- inner-validation

def validate() -> None:
    """Step 4: report (don't re-select on) the frozen config's inner-validation numbers."""
    print(f"=== step 4: inner-validation, frozen lb={FROZEN_LOOKBACK_DAYS}d "
          f"th={FROZEN_THRESHOLD:.2f}, {VALID[0]} .. {VALID[1]} ===\n")
    for label, market in MARKETS:
        base, _ = _period(get_strategy("kelly_regime_v4"), market, *VALID)
        gated, _ = _period(_frozen(), market, *VALID)
        print(_fmt("baseline kelly_regime_v4", label, base))
        print(_fmt("gate (frozen)", label, gated))
        print()


# --------------------------------------------------------- fee falsification

def feetier() -> None:
    """Step 5: pre-registered falsification, Bitstamp 0.40% taker, inner-train, spot."""
    print("=== step 5: falsification — Bitstamp 0.40% entry taker, inner-train, spot ===\n")
    fee = 0.004
    spot_fee = replace(SPOT, fee_rate=fee)
    base_01, _ = _period(get_strategy("kelly_regime_v4"), SPOT, *TRAIN)
    gate_01, _ = _period(_frozen(), SPOT, *TRAIN)
    base_04, _ = _period(get_strategy("kelly_regime_v4"), spot_fee, *TRAIN)
    gate_04, _ = _period(_frozen(), spot_fee, *TRAIN)
    print(_fmt("baseline @ 0.10%", "spot", base_01))
    print(_fmt("gate     @ 0.10%", "spot", gate_01))
    print(_fmt("baseline @ 0.40%", "spot", base_04))
    print(_fmt("gate     @ 0.40%", "spot", gate_04))
    edge_01 = gate_01.final_balance / base_01.final_balance - 1.0
    edge_04 = gate_04.final_balance / base_04.final_balance - 1.0
    print(f"\ngate advantage over baseline: {edge_01:+.1%} @ 0.10%, {edge_04:+.1%} @ 0.40%")
    print("SURVIVES" if edge_04 > 0 else "DOES NOT SURVIVE", "the 0.40% taker tier")


# ------------------------------------------------------------- funding cost

def fundingcost() -> None:
    """Step 6: does the gate reduce the real DOLLAR funding cost paid?"""
    print(f"=== step 6: real funding charged, {COMBINED[0]} .. {COMBINED[1]}, futures 5x ===\n")
    print(f"{'strategy':26s} {'funding-free':>13s} {'with funding':>13s} "
          f"{'cost':>6s} {'funding paid':>13s}")
    for tag, strat_free, strat_paid in (
        ("baseline kelly_regime_v4", get_strategy("kelly_regime_v4"), get_strategy("kelly_regime_v4")),
        ("gate (frozen)", _frozen(), _frozen()),
    ):
        free, _ = _period(strat_free, FUTURES, *COMBINED)
        paid, cost = _period(strat_paid, FUTURES, *COMBINED, funding=FUNDING)
        print(f"{tag:26s} ${free.final_balance:>12,.0f} ${paid.final_balance:>12,.0f} "
              f"{paid.final_balance / free.final_balance - 1:>5.0%} ${cost:>12,.0f}")


# ------------------------------------------------------------- causality probe

def causality() -> None:
    """Step 7: by-hand lookahead probe (R-28's two-opposite-tampers procedure).

    `tests/test_causality_strict.py` only parametrizes over registered
    strategies, so this unregistered one gets none of that protection.
    Bars strictly after a cut are multiplied by 3 in one copy of the frame
    and divided by 3 in another (prices, volume, AND the funding column);
    every decision at or before the cut must be identical between the two.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    print("=== step 7: by-hand lookahead probe ===\n")
    df = DFF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0
    up.iloc[cut:, up.columns.get_loc("funding")] *= 3.0
    down.iloc[cut:, down.columns.get_loc("funding")] /= 3.0

    def decisions(frame):
        s = _frozen()
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b = decisions(up), decisions(down)
    bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
    print(f"tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("FAIL - reads the future at bars " + str(bad) if bad
          else "PASS - every order decision at or before the cut is unchanged")

    pa = _frozen().prepare(up.copy())
    pb = _frozen().prepare(down.copy())
    for col in ("target", "funding_pct", "funding_gate"):
        va = pa[col].to_numpy()[:cut].astype(float)
        vb = pb[col].to_numpy()[:cut].astype(float)
        worst = float(np.nanmax(np.abs(va - vb)))
        print(f"  column {col:14s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


# ------------------------------------------------------------------------ main

COMMANDS = {"sweep": sweep, "validate": validate, "feetier": feetier,
            "fundingcost": fundingcost, "causality": causality}


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        rows = sweep()
        select_frozen(rows)  # printed for confirmation; FROZEN_* above are hardcoded already
        print()
        validate()
        print()
        feetier()
        print()
        fundingcost()
        print()
        causality()
        print(f"\ntotal distinct backtests run in this invocation: {N_BACKTESTS}")
    elif choice in COMMANDS:
        COMMANDS[choice]()
        print(f"\ndistinct backtests run in this invocation: {N_BACKTESTS}")
    else:
        print(f"usage: python {Path(__file__).name} [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    print(f"{len(DFF):,} bars  {DFF.index[0]:%Y-%m-%d} -> {DFF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    print(f"funding: {len(FUNDING):,} settlements  {FUNDING.index[0]:%Y-%m-%d} -> "
          f"{FUNDING.index[-1]:%Y-%m-%d}\n", file=sys.stderr)
    main()
