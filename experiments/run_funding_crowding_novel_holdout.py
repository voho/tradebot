#!/usr/bin/env python
"""R-34 step 4: the pre-registered 2023+ holdout read for `FundingCrowdingKelly`.

Run ONCE, after the decision rule (D1-D4) was committed into
`docs/LEDGER.md` (see "R-34 pre-registration"). Two parts:

    python experiments/run_funding_crowding_novel_holdout.py context
        P1 context: funding-free, standard OOS_START="2023-01-01" holdout,
        both markets - the same protocol every other holdout row in this
        project uses. The correction is zero for all but the first ~1 of
        these ~3.6 years (funding data ends 2023-12-31), so this is read
        as "did the change hurt the strategy's standing behavior," not as
        the mechanism's test.

    python experiments/run_funding_crowding_novel_holdout.py mechanism
        P2-P4, the actual test: 2023-01-01..2023-12-31 only (the
        funding-covered sub-window), funding-free + funding-charged
        (futures) + the 0.40% Bitstamp entry tier (spot).

    python experiments/run_funding_crowding_novel_holdout.py all
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.funding_crowding_novel import FundingCrowdingKelly  # noqa: E402
from experiments.funding_signal import causal_funding_column  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
OOS_START = "2023-01-01"
SUBWINDOW = ("2023-01-01", "2023-12-31")

N_READ = 0


def _load():
    df, label = load_dataset(ROOT / "data", "spot")
    funding = load_funding(ROOT / "data")
    df = df.copy()
    df["funding"] = causal_funding_column(df.index, funding)
    return df, label, funding


DF, LABEL, REAL_FUNDING = _load()


def line(tag, m, funding_paid=0.0):
    print(f"  {tag:38s} final=${m.final_balance:>11,.0f} ({m.profit_pct:>+7.1f}%) "
          f"trades={m.num_trades:>4d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}"
          f"{f' funding_paid=${funding_paid:>8,.0f}' if funding_paid else ''}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


def context() -> None:
    """P1: funding-free, standard OOS_START holdout, both markets."""
    global N_READ
    print("P1 (context): funding-free, standard 2023-01-01 -> end-of-data holdout\n")
    for mname, market in (("spot", SPOT), ("futures", FUTURES)):
        print(f"{mname}:")
        for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                            ("kelly_regime_v4 (baseline)", get_strategy("kelly_regime_v4")),
                            ("funding_crowding_novel", FundingCrowdingKelly(funding_scale=1.0))):
            result = run_period(strat, DF, OOS_START, None, market=market,
                                start_balance=1_000.0, data_label=LABEL)
            m = compute_metrics(result)
            line(name, m)
            N_READ += 1


def _period(strategy, market, start, end, *, funding=None):
    global N_READ
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                           df=raw.df.iloc[pre:])
    N_READ += 1
    return compute_metrics(trimmed), raw.funding_paid


def mechanism() -> None:
    """P2-P4: the funding-covered sub-window only, funding-free + charged + 0.40% tier."""
    start, end = SUBWINDOW
    baseline = get_strategy("kelly_regime_v4")
    novel = FundingCrowdingKelly(funding_scale=1.0)

    print(f"\nP2/P3 (mechanism): {start} -> {end}, funding-free, both markets\n")
    for mname, market in (("spot", SPOT), ("futures", FUTURES)):
        for name, strat in (("kelly_regime_v4 (baseline)", baseline),
                            ("funding_crowding_novel", novel)):
            m, _ = _period(strat, market, start, end)
            line(f"{mname:8s} {name}", m)

    print(f"\nP2 (mechanism, real cost): {start} -> {end}, futures 5x, funding CHARGED\n")
    for name, strat in (("kelly_regime_v4 (baseline)", baseline),
                        ("funding_crowding_novel", novel)):
        m, paid = _period(strat, FUTURES, start, end, funding=REAL_FUNDING)
        line(name, m, paid)

    print(f"\nP4 (fee-tier robustness): {start} -> {end}, spot, 0.40% Bitstamp entry tier\n")
    hi_fee_spot = MarketSpec.spot(fee_rate=0.004)
    for name, strat in (("kelly_regime_v4 (baseline)", baseline),
                        ("funding_crowding_novel", novel)):
        m, _ = _period(strat, hi_fee_spot, start, end)
        line(name, m)


COMMANDS = {"context": context, "mechanism": mechanism}


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        context()
        print("\n" + "=" * 78)
        mechanism()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python {sys.argv[0]} [{'|'.join(COMMANDS)}|all]")
        return
    print(f"\n{'=' * 78}\nholdout backtests read this run: {N_READ}")


if __name__ == "__main__":
    main()
