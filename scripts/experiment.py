#!/usr/bin/env python
"""Fast research harness for evaluating strategy variants on the real data.

Loads the committed dataset once, then runs any strategy instance (with
whatever constructor parameters you like) and prints a one-line summary.
This is the tool used to derive the parameters documented in
``docs/VALIDATION.md`` - keep experiments here rather than mutating the
registered strategies, so the comparison table stays a clean record.

Examples
--------
Reproduce the shipped strategy's risk/return frontier::

    python scripts/experiment.py frontier

Reproduce the horizon-robustness table (the ensemble-vs-members check)::

    python scripts/experiment.py horizons

Use it interactively for a new idea::

    from scripts.experiment import ev, DF, splits
    from tradebot.strategies.kelly_regime import KellyRegime
    ins, oos = splits(DF)
    ev(KellyRegime(target_vol=0.6), df=oos)      # out-of-sample only
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

# Walk-forward split used throughout docs/VALIDATION.md.
OOS_START = "2023-01-01"


def splits(df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(in-sample 2017-2022, out-of-sample 2023-present)."""
    df = DF if df is None else df
    return df.loc[:"2022-12-31"], df.loc[OOS_START:]


def ev(strategy: Strategy, df: pd.DataFrame = None, market: MarketSpec = None,
       balance: float = 1_000.0, tag: str = "") -> object:
    """Run one backtest and print a one-line summary; returns the Metrics."""
    df = DF if df is None else df
    market = SPOT if market is None else market
    t0 = time.time()
    result = run_backtest(strategy, df, market, balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"{tag or strategy.name:24s} {market.name:11s} "
          f"final=${m.final_balance:>13,.0f} ({m.profit_pct:>+9.1f}%) "
          f"trades={m.num_trades:>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} {'LIQUIDATED' if m.liquidated else ''} "
          f"[{time.time() - t0:.0f}s]")
    return m


def frontier() -> None:
    """kelly_regime risk/return frontier - shows the overbetting penalty."""
    from tradebot.strategies.kelly_regime import KellyRegime

    print("kelly_regime frontier (full period, futures 5x):")
    for vol, lev in ((0.40, 1.0), (0.55, 2.0), (0.60, 3.0), (0.80, 3.0)):
        ev(KellyRegime(target_vol=vol, max_leverage=lev), market=FUTURES,
           tag=f"vol{vol} lev{lev}x")


def horizons() -> None:
    """Single regime anchors vs the shipped three-anchor vote."""
    from tradebot.strategies.kelly_regime import KellyRegime

    print("kelly_regime horizon robustness (full period, futures 5x):")
    for h in ((30,), (50,), (100,), (200,), (30, 50, 100)):
        ev(KellyRegime(horizons=h), market=FUTURES, tag=f"anchors{h}")


def walkforward() -> None:
    """In-sample vs out-of-sample for the leading strategies."""
    from tradebot.registry import get_strategy

    ins, oos = splits()
    for name in ("buy_and_hold", "kelly_regime", "champions_council"):
        ev(get_strategy(name), df=ins, market=FUTURES, tag=f"IS  {name}")
        ev(get_strategy(name), df=oos, market=FUTURES, tag=f"OOS {name}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})\n", file=sys.stderr)
    commands = {"frontier": frontier, "horizons": horizons, "walkforward": walkforward}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in commands:
        commands[choice]()
    else:
        print(f"usage: python scripts/experiment.py [{'|'.join(commands)}]")
