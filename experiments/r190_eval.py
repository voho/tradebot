#!/usr/bin/env python
"""R-190 fixed train/holdout battery; fresh accounts and native next-open fills.

Run ``.venv/bin/python experiments/r190_eval.py train --workers 3`` before
freezing the decision rule, then ``... holdout --workers 3``. Holdout execution
requires the operator's source/data hash manifest. Training preparation sees
only bars through 2022; later evaluation retains causal prepared history but
starts a new $1,000 account directly at each window boundary. Parent target
changes and candidate account-drift decisions use their existing code.

Each of the 14 core names receives 3 training and 52 retrospective holdout
cells; two auxiliary blend-band controls receive 3 training and 4 non-beta
holdout cells, giving 784 evaluations. The 24 seeded overlapping windows are
paired across markets and descriptive, not independent samples. Spot costs
40bp +1bp slippage; the additional 10bp BTC cell is a discount comparison.
BTC spot uses a $10 minimum order; ETH's $5 minimum and 40bp cost are generic
scenarios. Deribit prices use venue-matched aggregated 8h funding and 5bp
+1bp slippage under the native generic $5-minimum perpetual broker, rather
than exact exchange contract specifications.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd

from r190_variations import CONFIGS, PARENTS, RebalanceVariation, make_strategy
from tradebot.broker import MarketSpec
from tradebot.data import load_funding_deribit, load_ohlcv_csv
from tradebot.engine import run_backtest
from tradebot.inference import annualized_sharpe, daily_returns
from tradebot.metrics import compute_metrics
from tradebot.strategies.buy_and_hold import BuyAndHold
from tradebot.strategy import Strategy

OUT = ROOT / "reports/r190_variations"
CANDIDATES = tuple(config[0] for config in CONFIGS)
NAMES = CANDIDATES + tuple(PARENTS) + ("buy_and_hold",)
AUXILIARIES = ("r190_blend_b05", "r190_blend_b20")
WORKER_NAMES = NAMES + AUXILIARIES
FILES = {"spot": "btcusd_spot_5m.csv.gz",
         "perp": "btcusdt_deribit_perp_5m.csv.gz",
         "eth": "ethusd_coinbase_spot_5m.csv.gz"}
HOLDOUT = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-08-12 00:40", tz="UTC")
TRAIN_END = HOLDOUT - pd.Timedelta(minutes=5)


def specifications(stage):
    """Return (cell, asset, start, end, fee, slippage_bps, funded) tuples."""
    if stage == "train":
        return [
            ("inner_train", "spot", None, pd.Timestamp("2020-12-31 23:55", tz="UTC"), .004, 1., False),
            ("inner_val", "spot", pd.Timestamp("2021-01-01", tz="UTC"), TRAIN_END, .004, 1., False),
            ("funded_val", "perp", pd.Timestamp("2021-01-01", tz="UTC"), TRAIN_END, .0005, 1., True),
        ]
    if stage != "holdout":
        raise ValueError(f"Unknown stage: {stage}")
    cells = [
        ("holdout", "spot", HOLDOUT, END, .004, 1., False),
        ("discount_holdout", "spot", HOLDOUT, END, .001, 1., False),
        ("eth_holdout", "eth", HOLDOUT, END, .004, 1., False),
        ("funded_holdout", "perp", HOLDOUT, END, .0005, 1., True),
    ]
    rng = np.random.default_rng(190)
    days = (END.normalize() - HOLDOUT).days
    for k in range(24):
        length = int(rng.integers(120, 366))
        start = HOLDOUT + pd.Timedelta(days=int(rng.integers(0, days - length + 1)))
        end = start + pd.Timedelta(days=length) - pd.Timedelta(minutes=5)
        for kind, fee, funded in (("spot", .004, False), ("perp", .0005, True)):
            cells.append((f"beta_{kind}_{k:02d}", kind, start, end, fee, 1., funded))
    return cells


def validate_manifest():
    """Reject absent/changed freezes before a holdout worker prepares signals."""
    path = OUT / "manifest.json"
    if not path.exists():
        raise RuntimeError("Freeze reports/r190_variations/manifest.json before holdout")
    hashes = json.loads(path.read_text())["hashes"]
    if not hashes:
        raise RuntimeError("Frozen manifest has no source/data hashes")
    for relative, expected in hashes.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen source/data changed: {relative}")


def get_strategy(name):
    if name in AUXILIARIES:
        return RebalanceVariation(name, "blend", .05 if name.endswith("b05") else .20)
    return (PARENTS[name]() if name in PARENTS else
            BuyAndHold() if name == "buy_and_hold" else make_strategy(name))


class Prepared(Strategy):
    """Replay causal columns with original warmup and fresh parent initialization."""

    warmup = 0

    def __init__(self, strategy, first_eligible=0):
        self.strategy, self.name = strategy, strategy.name
        self.first_eligible = first_eligible
        self.decision_requests = 0

    def prepare(self, df):
        return df

    def on_bar(self, ctx):
        if ctx.i < self.first_eligible:
            return
        if self.name in PARENTS and ctx.i == self.first_eligible:
            target = float(ctx.bar["target"])
            if target > 0:
                ctx.order_notional(target)
        else:
            self.strategy.on_bar(ctx)
        self.decision_requests += len(ctx.orders)


def evaluate(strategy, prepared_frame, spec, funding=None):
    """Return (metrics dict, daily frame, native result) for one fresh cell."""
    cell, kind, start, end, fee, slip, funded = spec
    lo = 0 if start is None else int(prepared_frame.index.searchsorted(start))
    hi = int(prepared_frame.index.searchsorted(end, side="right"))
    if hi <= lo or (start is not None and prepared_frame.index[lo] != start):
        raise ValueError(f"Incomplete price coverage: {kind} {start}")
    if prepared_frame.index[hi - 1] != end:
        raise ValueError(f"Incomplete price coverage: {kind} {end}")
    sub = prepared_frame.iloc[lo:hi]
    if funded:
        expected = pd.date_range(sub.index[0].ceil("8h"), end.floor("8h"), freq="8h")
        if funding is None or not funding.loc[sub.index[0]:end].index.equals(expected):
            raise ValueError("Missing or incomplete funding coverage")
    market = (MarketSpec.futures(leverage=5., fee_rate=fee) if kind == "perp"
              else MarketSpec.spot(fee_rate=fee))
    if kind == "spot":
        market = replace(market, min_notional=10.)
    label = ("Deribit perp + Deribit 8h funding" if funded else
             "Coinbase ETH spot" if kind == "eth" else "Bitstamp BTC spot")
    replay = Prepared(strategy, first_eligible=max(0, strategy.warmup - lo))
    result = run_backtest(replay, sub, market, 1000., slippage_bps=slip,
                          data_label=label, funding=funding if funded else None)
    metric = compute_metrics(result)
    days = (sub.index[-1] - sub.index[0] + pd.Timedelta(minutes=5)).total_seconds() / 86400
    completed = sum(not trade.open_at_end for trade in result.trades)
    active = len({fill.ts.normalize() for fill in result.fills})
    calendar_days = (sub.index[-1].normalize() - sub.index[0].normalize()).days + 1
    eq = result.equity.resample("1D").last().dropna()
    ret = daily_returns(result.equity).reindex(eq.index)
    ret.iloc[0] = eq.iloc[0] / 1000. - 1.
    # Reconstruct actual marked position from native fills, including liquidation.
    delta = np.zeros(len(sub))
    for fill in result.fills:
        delta[sub.index.get_loc(fill.ts)] += fill.qty if fill.side.name == "BUY" else -fill.qty
    position = np.cumsum(delta)
    equity = result.equity.to_numpy()
    exposure = np.divide(position * sub.close.to_numpy(), equity,
                         out=np.zeros(len(sub)), where=equity > 0)
    row = metric.as_row() | {
        "cell": cell, "asset": kind, "start": str(sub.index[0]), "end": str(sub.index[-1]),
        "days": days, "fee_rate": fee, "slippage_bps": slip,
        "min_notional": market.min_notional,
        "funding_paid": result.funding_paid,
        "funding_model": "Deribit 8h aggregation" if funded else "not charged",
        "fills": len(result.fills), "fills_per_day": len(result.fills) / days,
        "completed_round_trips": completed, "round_trips_per_day": completed / days,
        "active_days_pct": 100 * active / calendar_days,
        "daily_sharpe": annualized_sharpe(ret.to_numpy()),
        "annualized_volatility": float(ret.std(ddof=1) * np.sqrt(365.25)) if len(ret) > 1 else 0.,
        "mean_abs_exposure": float(np.abs(exposure).mean()),
        "decision_requests": replay.decision_requests,
    }
    daily = pd.DataFrame({"strategy": strategy.name, "cell": cell,
                          "timestamp": eq.index.astype(str), "equity": eq.to_numpy(),
                          "return": ret.to_numpy()})
    return row, daily, result


def _write_csv(frame, path):
    temporary = path.with_name("partial_" + path.name)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_one(name, stage):
    """Evaluate one fixed name, flushing a partial ledger after every cell."""
    if stage == "holdout":
        validate_manifest()
    specs = specifications(stage)
    if name in AUXILIARIES:
        specs = [spec for spec in specs if not spec[0].startswith("beta_")]
    strategy = get_strategy(name)
    cutoff = TRAIN_END if stage == "train" else END
    prepared = {}
    for kind in dict.fromkeys(spec[1] for spec in specs):
        raw = load_ohlcv_csv(ROOT / "data" / FILES[kind]).loc[:cutoff]
        prepared[kind] = strategy.prepare(raw.copy())
    funding = load_funding_deribit(ROOT / "data")
    rows, daily = [], []
    out = OUT / name / stage
    out.mkdir(parents=True, exist_ok=True)
    for k, spec in enumerate(specs):
        row, day, result = evaluate(strategy, prepared[spec[1]], spec, funding)
        rows.append(row)
        daily.append(day)
        if spec[0] == "holdout":
            pd.DataFrame([{"timestamp": str(f.ts), "side": f.side.name,
                           "qty": f.qty, "price": f.price, "fee": f.fee,
                           "kind": f.kind, "realized_pnl": f.realized_pnl}
                          for f in result.fills], columns=["timestamp", "side", "qty", "price",
                                                          "fee", "kind", "realized_pnl"]).to_csv(
                              out / "holdout_fills.csv", index=False)
        _write_csv(pd.DataFrame(rows), out / "cells.csv")
        _write_csv(pd.concat(daily, ignore_index=True), out / "daily.csv.gz")
        print(f"{name} {stage}: {k + 1}/{len(specs)} {spec[0]} "
              f"${row['final_balance']:,.0f}, {row['fills_per_day']:.2f} fills/day", flush=True)
    return rows


def run(stage, workers=3):
    if stage == "holdout":
        validate_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(run_one, name, stage): name for name in WORKER_NAMES}
        for future in as_completed(pending):
            rows.extend(future.result())
            _write_csv(pd.DataFrame(rows).sort_values(["strategy", "cell"]), OUT / f"{stage}_cells.csv")
    _write_csv(pd.concat([pd.read_csv(OUT / name / stage / "daily.csv.gz") for name in WORKER_NAMES],
                        ignore_index=True), OUT / f"{stage}_daily.csv.gz")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "holdout"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.stage, args.workers)
