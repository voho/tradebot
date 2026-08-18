"""Step 4 of B-05: the frozen 2023 holdout, run once, by the operator.

Per docs/LEDGER.md's B-05 pre-registration: real committed funding data
(``data/btcusdt_perp_funding_8h.csv.gz``) covers 2020-01-01..2023-12-31
only, so the standard OOS_START=2023-01-01 holdout can only be evaluated
with REAL funding for its first calendar year. This script reads exactly
that slice, 2023-01-01..2023-12-31, once, for both frozen configs
proposed by the two parallel branches:

    FundingDecileGate(decile=0.80, lookback_days=180)      -- conservative
    FundingMomentumTilt(weight=0.7, lo=0.70, hi=0.95,
                         lookback_days=90)                  -- novel

against kelly_regime_v4 (funding-charged) and buy_and_hold (spot).

Usage::

    python experiments/run_b05_holdout.py holdout   # P1/P2 decision rule
    python experiments/run_b05_holdout.py feetier    # P3 falsification
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dataclasses import replace  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

from experiments.funding_decile_gate import FundingDecileGate  # noqa: E402
from experiments.funding_momentum_tilt import FundingMomentumTilt  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
assert REAL_FUNDING.index.max() < __import__("pandas").Timestamp("2024-01-01", tz="UTC"), (
    "funding file extended past 2023 - the B-05 holdout scope assumption changed"
)
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2023-12-31"

FROZEN = {
    "funding_decile_gate": lambda: FundingDecileGate(
        funding=REAL_FUNDING, decile=0.80, lookback_days=180),
    "funding_momentum_tilt": lambda: FundingMomentumTilt(
        funding=REAL_FUNDING, weight=0.7, lo=0.70, hi=0.95, lookback_days=90),
}


def _run(strategy, market, funding=None, fee_override=None, start=HOLDOUT_START,
         end=HOLDOUT_END):
    m = market
    if fee_override is not None:
        m = MarketSpec(name=market.name, fee_rate=fee_override,
                        leverage=market.leverage, pays_funding=market.pays_funding,
                        maintenance_margin_rate=market.maintenance_margin_rate,
                        allow_short=market.allow_short)
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    prefix = prefix_bars(DF, lo, strategy.warmup)
    frame = DF.iloc[lo - prefix: hi]
    raw = run_backtest(strategy, frame, m, 1_000.0, data_label=LABEL,
                        trade_start=prefix, funding=funding)
    trimmed = raw if prefix == 0 else replace(
        raw, equity=raw.equity.iloc[prefix:], df=raw.df.iloc[prefix:])
    return compute_metrics(trimmed), raw.funding_paid


def holdout() -> None:
    """P1/P2: the pre-registered decision rule, 2023 only."""
    print(f"HOLDOUT {HOLDOUT_START} .. {HOLDOUT_END} (real funding coverage ends "
          f"{REAL_FUNDING.index.max():%Y-%m-%d}), $1,000 start\n")

    rows = []
    hold_spot, _ = _run(get_strategy("buy_and_hold"), SPOT)
    rows.append(("buy_and_hold", "spot", hold_spot, 0.0))
    v4_spot, _ = _run(get_strategy("kelly_regime_v4"), SPOT)
    rows.append(("kelly_regime_v4", "spot", v4_spot, 0.0))
    v4_fut, v4_fund = _run(get_strategy("kelly_regime_v4"), FUTURES, funding=REAL_FUNDING)
    rows.append(("kelly_regime_v4", "futures_5x+funding", v4_fut, v4_fund))

    for name, make in FROZEN.items():
        s_spot, _ = _run(make(), SPOT)
        rows.append((name, "spot", s_spot, 0.0))
        s_fut, s_fund = _run(make(), FUTURES, funding=REAL_FUNDING)
        rows.append((name, "futures_5x+funding", s_fut, s_fund))

    print(f"{'strategy':24s} {'market':20s} {'final':>12s} {'profit%':>9s} "
          f"{'maxDD':>7s} {'sharpe':>7s} {'trades':>7s} {'funding$':>9s}")
    for name, market, m, fund in rows:
        print(f"{name:24s} {market:20s} ${m.final_balance:>10,.0f} "
              f"{m.profit_pct:>+8.1f}% {m.max_drawdown_pct:>6.1f}% "
              f"{m.sharpe:>7.2f} {m.num_trades:>7d} ${fund:>7,.0f}")

    print("\n--- P1: beat kelly_regime_v4 (funding-charged futures) and buy_and_hold (spot) ---")
    for name, make in FROZEN.items():
        s_spot, _ = _run(make(), SPOT)
        s_fut, s_fund = _run(make(), FUTURES, funding=REAL_FUNDING)
        p1_fut = s_fut.final_balance > v4_fut.final_balance
        p1_spot = s_spot.final_balance > hold_spot.final_balance
        print(f"{name}: futures vs v4 {'PASS' if p1_fut else 'FAIL'} "
              f"(${s_fut.final_balance:,.0f} vs ${v4_fut.final_balance:,.0f}); "
              f"spot vs hold {'PASS' if p1_spot else 'FAIL'} "
              f"(${s_spot.final_balance:,.0f} vs ${hold_spot.final_balance:,.0f})")
        print(f"  Sharpe: v4 futures {v4_fut.sharpe:.2f} -> variant {s_fut.sharpe:.2f} "
              f"(delta {s_fut.sharpe - v4_fut.sharpe:+.2f}); "
              f"maxDD: v4 {v4_fut.max_drawdown_pct:.1f}% -> variant "
              f"{s_fut.max_drawdown_pct:.1f}% (delta {s_fut.max_drawdown_pct - v4_fut.max_drawdown_pct:+.1f}pp)")
        print(f"  funding paid: v4 ${v4_fund:,.0f} -> variant ${s_fund:,.0f} "
              f"({(1 - s_fund / v4_fund) * 100 if v4_fund else float('nan'):+.1f}% change)")


def feetier() -> None:
    """P3 falsification: does the variant's edge over v4 survive 0.40% spot fee?"""
    print("P3 falsification: 0.40% Bitstamp entry taker tier, spot, 2023 holdout\n")
    v4_spot, _ = _run(get_strategy("kelly_regime_v4"), SPOT, fee_override=0.004)
    hold_spot, _ = _run(get_strategy("buy_and_hold"), SPOT, fee_override=0.004)
    print(f"buy_and_hold      spot@0.40%  ${hold_spot.final_balance:>10,.0f} "
          f"{hold_spot.profit_pct:>+8.1f}%")
    print(f"kelly_regime_v4   spot@0.40%  ${v4_spot.final_balance:>10,.0f} "
          f"{v4_spot.profit_pct:>+8.1f}%")
    for name, make in FROZEN.items():
        s_spot, _ = _run(make(), SPOT, fee_override=0.004)
        sign_at_010 = "beats v4" if s_spot.final_balance > v4_spot.final_balance else "trails v4"
        print(f"{name:24s} spot@0.40%  ${s_spot.final_balance:>10,.0f} "
              f"{s_spot.profit_pct:>+8.1f}%  ({sign_at_010})")


COMMANDS = {"holdout": holdout, "feetier": feetier}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "holdout"
    COMMANDS[cmd]()
