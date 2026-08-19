#!/usr/bin/env python
"""R-33 holdout: the operator's single frozen look, after both variants reported.

Not part of either sub-agent's work — this is the pre-registered step 4
look, run once, by the operator, after the pre-registered selection rule
(docs/LEDGER.md, R-33) picked a winner on inner-validation (2022) alone.

Winner: variant A, `funding_veto`, config H=3 (funding_halflife_days),
W=60 (quantile_window_days). It dominated variant B (`carry_kelly`,
H=3d/cap=0.5) on inner-validation on BOTH log growth (-0.0632 vs -0.2546)
and max drawdown (16.9% vs 27.6%) -- no axis disagreement between the two
variants, so no tie-break was needed at that level. Within variant A
itself, H=3/W=60 (best log growth, Sharpe -0.24) and H=3/W=120 (best max
DD, 15.8% vs W=60's 16.9%) disagreed by a small margin; H=3/W=60 was
selected because it is best-or-near-best on every axis (growth, Sharpe,
and within 1.1pp of the DD optimum), the more balanced pick, and W=60 is
adjacent to W=120 and W=180 (all winning) rather than an isolated spike.

Runs, funding data permits only this window:

    python experiments/run_r33_holdout.py holdout       # step 4, P1-P3
    python experiments/run_r33_holdout.py falsification # 0.40% taker stress
    python experiments/run_r33_holdout.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

from experiments.funding_veto import FundingVeto, REAL, _period, _row, _print_table  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
HOLDOUT = ("2023-01-01", "2023-12-31")

FROZEN_H = 3
FROZEN_W = 60


def _frozen() -> FundingVeto:
    return FundingVeto(funding=REAL, funding_halflife_days=FROZEN_H,
                        quantile_window_days=FROZEN_W)


def holdout() -> list[dict]:
    """P1-P3 against docs/LEDGER.md R-33's promotion rule, 2023 only, real funding."""
    market = MarketSpec.futures(leverage=5.0)
    start, end = HOLDOUT
    rows = []
    for name, strat in (
        ("buy_and_hold", get_strategy("buy_and_hold")),
        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
        ("funding_veto(H=3,W=60)", _frozen()),
    ):
        r = _period(strat, DF, market, start, end, funding=REAL)
        rows.append(_row(name, "2023 holdout", r))
    _print_table(rows)

    v4 = next(r for r in rows if r["config"] == "kelly_regime_v4")
    fv = next(r for r in rows if r["config"] == "funding_veto(H=3,W=60)")
    bh = next(r for r in rows if r["config"] == "buy_and_hold")
    dd_gain = v4["max_dd_pct"] - fv["max_dd_pct"]
    sharpe_gain = fv["sharpe"] - v4["sharpe"]
    print(f"\nfunding_veto vs kelly_regime_v4 on the 2023 holdout:")
    print(f"  Delta max DD  = {dd_gain:+.1f}pp   (promote if >= +10pp)")
    print(f"  Delta Sharpe  = {sharpe_gain:+.2f}   (promote if beyond +/-0.2 noise floor)")
    print(f"  funding_veto vs buy_and_hold final balance: "
          f"${fv['final_balance']:,.0f} vs ${bh['final_balance']:,.0f}  "
          f"({'does not lose' if fv['final_balance'] >= bh['final_balance'] else 'LOSES'})")
    return rows


def falsification() -> list[dict]:
    """Pre-registered falsification: survive Bitstamp's 0.40% taker tier, same window."""
    BITSTAMP_TAKER = 0.004
    market = MarketSpec.futures(leverage=5.0, fee_rate=BITSTAMP_TAKER)
    start, end = HOLDOUT
    rows = []
    for name, strat in (
        ("buy_and_hold", get_strategy("buy_and_hold")),
        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
        ("funding_veto(H=3,W=60)", _frozen()),
    ):
        r = _period(strat, DF, market, start, end, funding=REAL)
        rows.append(_row(name, "2023 @ 0.40%", r))
    _print_table(rows)
    return rows


COMMANDS = {"holdout": holdout, "falsification": falsification}


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_r33_holdout.py [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    main()
