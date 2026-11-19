#!/usr/bin/env python
"""Rank every reproducible strategy tried in this repo, by profit.

One dataset and one protocol for all rows: the README comparison table's.
That means full history (``btcusd_spot_5m.csv.gz``), $1,000, 0.10% spot and
0.05% futures at 5x on the spot series ("spot (perp proxy)"), no slippage.

Rows:
- the 37 registered single-asset strategies. Each is rerun here and must
  reproduce its row in ``reports/comparison.csv``;
- 36 frozen candidates from R-188 and R-190 to R-194;
- 17 older candidates (R-62 to R-179) that build from their shipped defaults,
  run on the BTC series alone and read no other file.

Older classes that need external data, per-round calibration or a
multi-asset universe are not reconstructable here and are not ranked.

Intervals use ``scripts/inference.py``'s procedure on full-period spot: a
paired stationary bootstrap against ``buy_and_hold`` (30-day blocks, 2,000
resamples, seed 7, 95%) for log growth and max drawdown.

These full-period runs read the 2023+ holdout, so they count as holdout
consultations (docs/LEDGER.md). No verdict is changed by them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np
import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.data import load_ohlcv_csv
from tradebot.engine import run_backtest
from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                paired_bootstrap, stationary_bootstrap_indices,
                                total_log_return)
from tradebot.metrics import compute_metrics

OUT = ROOT / "reports/all_strategies"
FULL_HISTORY = ROOT / "data/btcusd_spot_5m.csv.gz"
MARKETS = {"spot": MarketSpec.spot(fee_rate=0.001),
           "futures_5x": MarketSpec.futures(leverage=5.0, fee_rate=0.0005)}
BENCHMARK = "buy_and_hold"
R188 = ("robust_kelly", "coin_betting", "level_k", "focal_levels", "mfg_crowding",
        "noise_area_breakout", "intraday_momentum", "session_drift",
        "vwap_reversion", "jump_momentum")
# Older classes reconstructable from shipped defaults: module -> round label.
OLDER = {
    "r62_conservative_voltarget_only": "R-62", "r62_novel_vote_constant_exposure": "R-62",
    "r64_novel_aim_partial_adjustment": "R-64", "r64_conservative_trade_to_boundary": "R-64",
    "r66_novel_scaled_band": "R-66", "r66_conservative_snap_to_flat": "R-66",
    "r80_conservative_hitrate_confirm": "R-80", "r128_conservative_ev_band": "R-128",
    "r129_conservative_per_expert_band": "R-129", "r129_novel_bucket_band": "R-129",
    "r131_conservative_turnover_band": "R-131", "r131_novel_turnover_throttle": "R-131",
    "r133_mechanisms": "R-133 (replicates R-131)", "r141_novel_lppls_dampener": "R-141",
    "r157_elliott_confidence_novel": "R-157", "r179_novel": "R-179",
}


def _older_classes() -> dict[str, tuple[type, str]]:
    import importlib
    import inspect
    from tradebot.strategy import Strategy
    out = {}
    for module, label in OLDER.items():
        mod = importlib.import_module(module)
        for cls in vars(mod).values():
            if (inspect.isclass(cls) and issubclass(cls, Strategy)
                    and cls.__module__ == mod.__name__ and not cls.__name__.startswith("_")):
                out[cls.name] = (cls, label)
    return out


def registered() -> list[str]:
    return sorted(pd.read_csv(ROOT / "reports/comparison.csv").strategy.unique())


def candidates() -> dict[str, str]:
    """Unregistered candidate -> round label."""
    from experiments.r190_variations import CONFIGS
    from experiments.r191_strategies import PRIMARY_NAMES as R191
    from experiments.r192_strategies import PRIMARY_NAMES as R192
    from experiments.r193_strategies import PRIMARY_NAMES as R193
    from experiments.r194_strategies import PRIMARY_NAMES as R194
    out = {name: label for name, (_, label) in _older_classes().items()}
    out.update({name: "R-188" for name in R188})
    out.update({config[0]: "R-190" for config in CONFIGS})
    for label, names in (("R-191", R191), ("R-192", R192), ("R-193", R193), ("R-194", R194)):
        out.update({name: label for name in names})
    return out


def make(name: str):
    from tradebot.registry import available_strategies, get_strategy
    if name in available_strategies():
        return get_strategy(name)
    older = _older_classes()
    if name in older:
        return older[name][0]()
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
    result = run_backtest(make(name), _DF, MARKETS[market], 1_000.0, data_label="real")
    m = compute_metrics(result)
    row = dict(strategy=name, market=market, final_balance=m.final_balance,
               num_trades=m.num_trades, max_drawdown_pct=m.max_drawdown_pct,
               sharpe=m.sharpe, liquidated=bool(m.liquidated))
    return row, daily_returns(result.equity).rename(f"{name}|{market}")


def run(workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [*registered(), *candidates()]
    jobs = sorted(((n, k) for n in names for k in MARKETS),
                  key=lambda j: "transformer" not in j[0])  # slowest first
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(one, jobs))
    frame = pd.DataFrame([row for row, _ in results])
    curves = pd.concat([curve for _, curve in results], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "runs.csv", index=False)
    curves.to_csv(OUT / "daily_returns.csv.gz")
    return frame, curves


def reference_check(frame: pd.DataFrame) -> pd.DataFrame:
    """Every registered row must reproduce the committed comparison table."""
    table = pd.read_csv(ROOT / "reports/comparison.csv").set_index(["strategy", "market"])
    mine = frame.set_index(["strategy", "market"])
    rows = []
    for key in table.index:
        a, b = mine.loc[key, "final_balance"], table.loc[key, "final_balance"]
        rows.append(dict(strategy=key[0], market=key[1], rerun=a, committed=b,
                         ok=abs(a - b) <= max(1.0, 1e-6 * abs(b))))
    check = pd.DataFrame(rows)
    check.to_csv(OUT / "reproduction_check.csv", index=False)
    return check


def intervals(curves: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Paired bootstrap vs spot buy-and-hold, exactly as scripts/inference.py."""
    window = curves.dropna(how="all")
    bench = window[f"{BENCHMARK}|spot"].fillna(0.).to_numpy()
    idx = stationary_bootstrap_indices(len(window), 30.0, 2_000, np.random.default_rng(7))
    rows = []
    for name in names:
        r = window[f"{name}|spot"].fillna(0.).to_numpy()
        growth = paired_bootstrap(r, bench, total_log_return, indices=idx, level=.95)
        dd = paired_bootstrap(r, bench, max_drawdown_from_returns, indices=idx, level=.95)
        rows.append(dict(strategy=name, d_log_growth=growth.diff.point,
                         d_log_growth_lo=growth.diff.lo, d_log_growth_hi=growth.diff.hi,
                         d_max_dd_pp=dd.diff.point, d_max_dd_lo=dd.diff.lo, d_max_dd_hi=dd.diff.hi))
    return pd.DataFrame(rows)


def build(frame: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    labels = candidates()
    rows = []
    for name, group in frame.groupby("strategy"):
        g = group.set_index("market")
        best = g.final_balance.idxmax()
        rows.append(dict(
            strategy=name, origin=labels.get(name, "registered"),
            status="evaluated, NEGATIVE" if name in labels else "registered",
            spot=g.loc["spot", "final_balance"], futures_5x=g.loc["futures_5x", "final_balance"],
            profit=g.loc[best, "final_balance"] - 1_000.0, best_market=best,
            max_dd=g.loc[best, "max_drawdown_pct"], trades=int(g.loc[best, "num_trades"])))
    ranked = pd.DataFrame(rows).merge(intervals(curves, list(frame.strategy.unique())), on="strategy")
    ranked = ranked.sort_values("profit", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    ranked.to_csv(OUT / "ranked.csv", index=False)
    return ranked


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1000:,.1f}K" if x >= 10_000 else f"{sign}${x:,.0f}" if x >= 100 else f"{sign}${x:,.2f}"


def interval(point, lo, hi, unit="") -> str:
    mark = "▲" if lo > 0 else "▼" if hi < 0 else "≈"
    return f"{mark} {point:+.2f}{unit} [{lo:+.2f}, {hi:+.2f}]"


def markdown(ranked: pd.DataFrame) -> str:
    lines = ["| # | strategy | origin | status | profit (best market) | spot | futures_5x | max DD | "
             "growth vs hold (spot) | max DD vs hold (spot) |",
             "|---:|---|---|---|---:|---:|---:|---:|---|---|"]
    for r in ranked.itertuples():
        dd = (f"{'▲' if r.d_max_dd_hi < 0 else '▼' if r.d_max_dd_lo > 0 else '≈'} "
              f"{r.d_max_dd_pp:+.1f}pp [{r.d_max_dd_lo:+.1f}, {r.d_max_dd_hi:+.1f}]")
        lines.append(f"| {r.rank} | `{r.strategy}` | {r.origin} | {r.status} | {money(r.profit)} | "
                     f"{money(r.spot)} | {money(r.futures_5x)} | {r.max_dd:.0f}% | "
                     f"{interval(r.d_log_growth, r.d_log_growth_lo, r.d_log_growth_hi)} | {dd} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse", action="store_true", help="rebuild from saved runs")
    args = parser.parse_args()
    if args.reuse:
        frame = pd.read_csv(OUT / "runs.csv")
        curves = pd.read_csv(OUT / "daily_returns.csv.gz", index_col=0)
    else:
        frame, curves = run(args.workers)
    check = reference_check(frame)
    if not check.ok.all():
        print("Registered rows that do NOT reproduce reports/comparison.csv:")
        print(check[~check.ok].to_string(index=False))
    ranked = build(frame, curves)
    (OUT / "table.md").write_text(markdown(ranked) + "\n")
    print(markdown(ranked))
