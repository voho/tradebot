#!/usr/bin/env python
"""R-33 (B-05) step 4 — the pre-registered 2023 holdout read, run once.

Both branches (``experiments/run_funding_gate_conservative.py``,
``experiments/run_funding_gate_carry_kelly.py``) stayed inside
2020-01-01..2022-12-31 by construction. This script is the orchestrator's
own read of the one year of committed funding data that falls inside the
project's official ``OOS_START = 2023-01-01`` holdout — the pre-registered
decision rule in docs/LEDGER.md (R-33) is evaluated on exactly this
output, run exactly once.

    python experiments/run_funding_gate_holdout.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns, max_drawdown_from_returns, paired_bootstrap, total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

from experiments.funding_gate_conservative import FundingDecileGate  # noqa: E402
from experiments.funding_gate_carry_kelly import CarryAdjustedKelly  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2023-12-31"


def _period(strategy, market, start, end, funding=None):
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    return compute_metrics(trimmed), raw.funding_paid, trimmed.equity


def main() -> None:
    print(f"R-33 holdout, {HOLDOUT_START} .. {HOLDOUT_END}, $1,000 start, "
          f"real funding charged on futures\n")

    hold_m, _, hold_eq = _period(get_strategy("buy_and_hold"), SPOT,
                                 HOLDOUT_START, HOLDOUT_END)
    v4_m, v4_paid, v4_eq = _period(get_strategy("kelly_regime_v4"), FUTURES,
                                   HOLDOUT_START, HOLDOUT_END, funding=REAL)
    gate_m, gate_paid, gate_eq = _period(
        FundingDecileGate(funding=REAL), FUTURES, HOLDOUT_START, HOLDOUT_END, funding=REAL)
    carry_m, carry_paid, carry_eq = _period(
        CarryAdjustedKelly(funding=REAL), FUTURES, HOLDOUT_START, HOLDOUT_END, funding=REAL)

    rows = [
        ("buy_and_hold (spot, no funding)", hold_m, 0.0),
        ("kelly_regime_v4 (futures, funding)", v4_m, v4_paid),
        ("FundingDecileGate (futures, funding)", gate_m, gate_paid),
        ("CarryAdjustedKelly (futures, funding)", carry_m, carry_paid),
    ]
    print(f"{'strategy':40s} {'final':>10s} {'DD%':>7s} {'sharpe':>7s} "
          f"{'trades':>7s} {'funding paid':>13s}")
    for name, m, paid in rows:
        print(f"{name:40s} ${m.final_balance:>9,.0f} {m.max_drawdown_pct:>6.1f}% "
              f"{m.sharpe:>7.2f} {m.num_trades:>7d} ${paid:>12,.0f}")

    print("\nPaired stationary block bootstrap vs kelly_regime_v4, "
          "2023 daily returns, 10-day mean block, 2,000 resamples, seed=7:\n")
    v4_rets = daily_returns(v4_eq).to_numpy()
    for name, eq in (("FundingDecileGate", gate_eq), ("CarryAdjustedKelly", carry_eq)):
        rets = daily_returns(eq).to_numpy()
        n = min(len(rets), len(v4_rets))
        a, b = rets[-n:], v4_rets[-n:]
        growth = paired_bootstrap(a, b, total_log_return, mean_block=10.0, n_boot=2_000, seed=7)
        dd = paired_bootstrap(a, b, max_drawdown_from_returns, mean_block=10.0, n_boot=2_000, seed=7)
        print(f"{name} - kelly_regime_v4:")
        print(f"  Delta log growth   {growth.stat_a - growth.stat_b:+.3f}  "
              f"95% CI [{growth.diff.lo:+.3f}, {growth.diff.hi:+.3f}]  "
              f"P(>0)={growth.p_positive:.2f}  "
              f"{'EXCLUDES ZERO' if growth.significant else 'contains zero'}")
        print(f"  Delta max drawdown {dd.stat_a - dd.stat_b:+.2f}pp  "
              f"95% CI [{dd.diff.lo:+.2f}, {dd.diff.hi:+.2f}]  "
              f"P(>0)={dd.p_positive:.2f}  "
              f"{'EXCLUDES ZERO' if dd.significant else 'contains zero'}")
        print()

    print("Decision rule (R-33, docs/LEDGER.md), default reject:")
    for name, m, paid in ((n, m, p) for n, m, p in rows[2:]):
        p1 = m.final_balance > v4_m.final_balance
        p2 = m.final_balance > hold_m.final_balance
        dd_improve = v4_m.max_drawdown_pct - m.max_drawdown_pct
        print(f"  {name}: P1(beats v4)={p1}  P2(beats hold)={p2}  "
              f"DD improvement over v4={dd_improve:+.1f}pp")


if __name__ == "__main__":
    if REAL is None:
        raise SystemExit("no funding data committed")
    main()
