#!/usr/bin/env python
"""R-33 holdout evaluation — run once, centrally, after both branches froze.

Reads the pre-registered decision rule in docs/LEDGER.md (R-33
pre-registration, committed one commit ahead of this file) and scores
both frozen candidates against funding-charged ``kelly_regime_v4`` on the
2023-01-01 -> 2023-12-31 holdout — the only period inside the committed
funding data (2020-2023) that is also inside the project's OOS_START.

Candidates, frozen on inner-validation (2022) by two independent branches
that never saw this holdout:

- conservative: FundingGateConservative(decile_in=0.90, decile_out=0.75,
  pct_window_days=90) -- experiments/funding_gate_conservative.py
- novel: FundingGateNovel(k=1.0, funding_halflife_days=1.0) --
  experiments/funding_gate_novel.py

This file does not sweep, does not select, and does not touch anything
before 2023-01-01. Its only job is to read the holdout once and apply the
pre-registered P1-P4 rule.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

from funding_gate_conservative import FundingGateConservative  # noqa: E402
from funding_gate_novel import FundingGateNovel  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

HOLDOUT_START = "2023-01-01"
HOLDOUT_END = "2023-12-31"
INNER_VAL_START = "2022-01-01"
INNER_VAL_END = "2022-12-31"


def period(strategy, market, start, end, funding=None):
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                        trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    return compute_metrics(trimmed), raw.funding_paid


def row(label, m, funding_paid):
    liq = " LIQUIDATED" if m.liquidated else ""
    print(f"  {label:34s} final=${m.final_balance:>10,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>6.2f} trades={m.num_trades:>4d} "
          f"funding=${funding_paid:>7,.0f}{liq}")


def main() -> None:
    print(f"R-33 holdout: {HOLDOUT_START} .. {HOLDOUT_END}, real funding, "
          f"$1,000 start, futures 5x\n")

    v4 = get_strategy("kelly_regime_v4")
    hold_spot = get_strategy("buy_and_hold")
    conservative = FundingGateConservative(
        funding=REAL_FUNDING, decile_in=0.90, decile_out=0.75, pct_window_days=90)
    novel = FundingGateNovel(funding=REAL_FUNDING, k=1.0, funding_halflife_days=1.0)

    candidates = {"conservative": conservative, "novel": novel}

    print("=" * 78)
    print("HOLDOUT 2023 (the only read of this window)")
    print("=" * 78)
    m_v4, f_v4 = period(v4, FUTURES, HOLDOUT_START, HOLDOUT_END, funding=REAL_FUNDING)
    row("kelly_regime_v4 (funding-charged)", m_v4, f_v4)
    m_hold, f_hold = period(hold_spot, SPOT, HOLDOUT_START, HOLDOUT_END)
    row("buy_and_hold (spot, standing bar)", m_hold, f_hold)

    results = {}
    for name, strat in candidates.items():
        m, f = period(strat, FUTURES, HOLDOUT_START, HOLDOUT_END, funding=REAL_FUNDING)
        row(f"{name} (funding-charged)", m, f)
        results[name] = (m, f)

    print(f"\ndead_tail check: kelly_regime_v4 liquidated={m_v4.liquidated}, "
          f"conservative liquidated={results['conservative'][0].liquidated}, "
          f"novel liquidated={results['novel'][0].liquidated}")

    print("\n" + "=" * 78)
    print("INNER-VALIDATION 2022 CORROBORATION (already read by each branch;")
    print("re-run here only for a same-script apples-to-apples P2 check)")
    print("=" * 78)
    m_v4_val, f_v4_val = period(v4, FUTURES, INNER_VAL_START, INNER_VAL_END, funding=REAL_FUNDING)
    row("kelly_regime_v4 (funding-charged)", m_v4_val, f_v4_val)
    val_results = {}
    for name, strat in candidates.items():
        m, f = period(strat, FUTURES, INNER_VAL_START, INNER_VAL_END, funding=REAL_FUNDING)
        row(f"{name} (funding-charged)", m, f)
        val_results[name] = (m, f)

    print("\n" + "=" * 78)
    print("PRE-REGISTERED DECISION RULE (R-33) — applied now, mechanically")
    print("=" * 78)
    for name in ("conservative", "novel"):
        m, f = results[name]
        mv, fv = val_results[name]
        p1 = m.final_balance > m_v4.final_balance
        sharpe_edge = m.sharpe - m_v4.sharpe
        dd_edge = m_v4.max_drawdown_pct - m.max_drawdown_pct  # positive = shallower
        sharpe_edge_val = mv.sharpe - m_v4_val.sharpe
        dd_edge_val = m_v4_val.max_drawdown_pct - mv.max_drawdown_pct
        p2_holdout = (abs(sharpe_edge) > 0.2) or (dd_edge >= 10.0)
        p2_direction_matches_val = (
            (sharpe_edge > 0) == (sharpe_edge_val > 0) if abs(sharpe_edge) > 0.01 else
            (dd_edge > 0) == (dd_edge_val > 0)
        )
        p2 = p2_holdout and p2_direction_matches_val
        beats_hold = m.final_balance > m_hold.final_balance  # informational only
        print(f"\n{name}:")
        print(f"  P1 (beats funding-charged v4 on holdout final balance): "
              f"{'PASS' if p1 else 'FAIL'} (${m.final_balance:,.0f} vs ${m_v4.final_balance:,.0f})")
        print(f"  P2 (> +/-0.2 Sharpe or >=10pp DD, direction matches inner-val): "
              f"{'PASS' if p2 else 'FAIL'} "
              f"(holdout Sharpe edge {sharpe_edge:+.2f}, DD edge {dd_edge:+.1f}pp; "
              f"inner-val Sharpe edge {sharpe_edge_val:+.2f}, DD edge {dd_edge_val:+.1f}pp)")
        print(f"  P3 (falsification): reported by the branch, not re-run here "
              f"(uses only pre-2023 data) — see ledger row for outcome")
        print(f"  P4 (plateau): reported by the branch on inner-validation — "
              f"see ledger row for outcome")
        print(f"  [informational] beats buy_and_hold on holdout (standing "
              f"ROUTINE.md bar, spot vs this candidate's futures number is "
              f"not apples-to-apples leverage but reported per the rule): "
              f"{'yes' if beats_hold else 'no'} "
              f"(${m.final_balance:,.0f} vs ${m_hold.final_balance:,.0f})")
        print(f"  => P1 and P2 {'BOTH PASS' if (p1 and p2) else 'NOT BOTH PASS'}")


if __name__ == "__main__":
    main()
