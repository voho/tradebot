#!/usr/bin/env python
"""Rank every runnable strategy tried in this repo in one table.

Registered strategies come from ``reports/comparison.csv`` (the README
comparison table). Every unregistered candidate with a fixed, frozen
constructor (R-188 and R-190 through R-194; R-189's ten are registered) is
run here under that table's own protocol: full history, $1,000, 0.10% spot,
0.05% futures at 5x on the spot series ("spot (perp proxy)"), no slippage.

The full-history file is loaded explicitly. Since 2026-09-05,
``load_dataset(..., "spot")`` prefers the one-year
``btcusdt_spot_aligned_5m.csv``, so going through it would silently rank
the last year only. ``buy_and_hold`` and ``kelly_regime_v4`` are rerun and
must reproduce their committed rows before anything is written.

These runs read the 2023+ holdout as part of the full period, so they count
as holdout consultations (see docs/LEDGER.md). They are descriptive: no
verdict is changed by them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")]

import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.data import load_ohlcv_csv
from tradebot.engine import run_backtest
from tradebot.metrics import compute_metrics

OUT = ROOT / "reports/all_strategies"
FULL_HISTORY = ROOT / "data/btcusd_spot_5m.csv.gz"
MARKETS = {"spot": MarketSpec.spot(fee_rate=0.001),
           "futures_5x": MarketSpec.futures(leverage=5.0, fee_rate=0.0005)}
R188 = ("robust_kelly", "coin_betting", "level_k", "focal_levels", "mfg_crowding",
        "noise_area_breakout", "intraday_momentum", "session_drift",
        "vwap_reversion", "jump_momentum")


def candidates() -> dict[str, str]:
    """Unregistered candidate -> round label."""
    from experiments.r190_variations import CONFIGS
    from experiments.r191_strategies import PRIMARY_NAMES as R191
    from experiments.r192_strategies import PRIMARY_NAMES as R192
    from experiments.r193_strategies import PRIMARY_NAMES as R193
    from experiments.r194_strategies import PRIMARY_NAMES as R194
    out = {name: "R-188" for name in R188}
    out.update({config[0]: "R-190" for config in CONFIGS})
    for label, names in (("R-191", R191), ("R-192", R192), ("R-193", R193), ("R-194", R194)):
        out.update({name: label for name in names})
    return out


def make(name: str):
    if name in ("buy_and_hold", "kelly_regime_v4"):
        from tradebot.registry import get_strategy
        return get_strategy(name)
    if name in R188:
        from r188_shared import _cls, frozen_params
        return _cls(name)(**frozen_params()[name])
    module = {"r190": "r190_variations", "r191": "r191_strategies", "r192": "r192_strategies",
              "r193": "r193_strategies", "r194": "r194_strategies"}[name[:4]]
    return __import__(f"experiments.{module}", fromlist=["make_strategy"]).make_strategy(name)


_DF = None


def one(job):
    global _DF
    name, market = job
    if _DF is None:
        _DF = load_ohlcv_csv(FULL_HISTORY)
    m = compute_metrics(run_backtest(make(name), _DF, MARKETS[market], 1_000.0,
                                     data_label="real"))
    return dict(strategy=name, market=market, final_balance=m.final_balance,
                num_trades=m.num_trades, max_drawdown_pct=m.max_drawdown_pct,
                sharpe=m.sharpe, liquidated=bool(m.liquidated))


def run(workers: int) -> pd.DataFrame:
    names = ["buy_and_hold", "kelly_regime_v4", *candidates()]
    jobs = sorted(((n, k) for n in names for k in MARKETS),
                  key=lambda j: "transformer" not in j[0])  # slowest first
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(one, jobs))
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "candidates_full_period.csv", index=False)
    return frame


def reference_check(frame: pd.DataFrame) -> None:
    table = pd.read_csv(ROOT / "reports/comparison.csv").set_index(["strategy", "market"])
    mine = frame.set_index(["strategy", "market"])
    for key in [(n, k) for n in ("buy_and_hold", "kelly_regime_v4") for k in MARKETS]:
        a, b = mine.loc[key, "final_balance"], table.loc[key, "final_balance"]
        if abs(a - b) > max(1.0, 1e-6 * abs(b)):
            raise RuntimeError(f"{key} does not reproduce the README table: {a:,.2f} vs {b:,.2f}")


def build(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(ROOT / "reports/comparison.csv")
    rows = []
    for name, group in table.groupby("strategy"):
        g = group.set_index("market")
        rows.append(dict(strategy=name, round="registered", status="registered",
                         spot=g.loc["spot", "final_balance"], futures_5x=g.loc["futures_5x", "final_balance"],
                         trades=int(g.num_trades.max()),
                         max_dd=float(g.loc[g.final_balance.idxmax(), "max_drawdown_pct"]),
                         sharpe=float(g.sharpe.max())))
    labels = candidates()
    for name, group in frame[frame.strategy.isin(labels)].groupby("strategy"):
        g = group.set_index("market")
        rows.append(dict(strategy=name, round=labels[name], status="evaluated, NEGATIVE",
                         spot=g.loc["spot", "final_balance"], futures_5x=g.loc["futures_5x", "final_balance"],
                         trades=int(g.num_trades.max()),
                         max_dd=float(g.loc[g.final_balance.idxmax(), "max_drawdown_pct"]),
                         sharpe=float(g.sharpe.max())))
    ranked = pd.DataFrame(rows)
    ranked["best"] = ranked[["spot", "futures_5x"]].max(axis=1)
    ranked = ranked.sort_values("best", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    ranked.to_csv(OUT / "ranked.csv", index=False)
    return ranked


def money(x: float) -> str:
    return f"${x / 1000:,.1f}K" if x >= 10_000 else f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def markdown(ranked: pd.DataFrame) -> str:
    lines = ["| # | strategy | round | status | spot | futures_5x | max DD (best market) | trades | best Sharpe |",
             "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for r in ranked.itertuples():
        lines.append(f"| {r.rank} | `{r.strategy}` | {r.round} | {r.status} | {money(r.spot)} | "
                     f"{money(r.futures_5x)} | {r.max_dd:.0f}% | {r.trades:,} | {r.sharpe:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse", action="store_true", help="rebuild from saved runs")
    args = parser.parse_args()
    frame = (pd.read_csv(OUT / "candidates_full_period.csv") if args.reuse else run(args.workers))
    reference_check(frame)
    ranked = build(frame)
    (OUT / "table.md").write_text(markdown(ranked) + "\n")
    print(markdown(ranked))
