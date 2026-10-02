"""R-192 counted fresh-account evaluation of five Kelly v4 modifications.

All target strategies share R-190's explicit initial-target entry convention;
later decisions delegate to their original callback. Fees, fills, funding,
warmup and daily accounting reuse the frozen R-190 evaluator. Five primaries
and two controls receive 55 cells each; ten neighbours receive seven each.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np
import pandas as pd

from matched_hold import ConstantExposureHold
from r190_eval import evaluate as native_evaluate
from r192_strategies import PRIMARY_NAMES, ALL_NAMES, make_strategy
from tradebot.data import load_funding_deribit, load_ohlcv_csv
from tradebot.strategies.buy_and_hold import BuyAndHold
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Strategy

OUT = ROOT / "reports/r192_improvements"
CANDIDATES = tuple(PRIMARY_NAMES)
AUXILIARIES = tuple(name for name in ALL_NAMES if name not in CANDIDATES)
CONTROLS = ("buy_and_hold", "kelly_regime_v4")
NAMES = CANDIDATES + CONTROLS
WORKER_NAMES = tuple(ALL_NAMES) + CONTROLS
FILES = {"spot": "btcusd_spot_5m.csv.gz",
         "perp": "btcusdt_deribit_perp_5m.csv.gz",
         "eth": "ethusd_coinbase_spot_5m.csv.gz"}
HOLDOUT = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-08-12 00:40", tz="UTC")
TRAIN_END = HOLDOUT - pd.Timedelta(minutes=5)


def specifications(stage):
    """(cell, asset, inclusive start/end, fee, slippage_bps, funded)."""
    if stage == "train":
        return [
            ("inner_train", "spot", None, pd.Timestamp("2020-12-31 23:55", tz="UTC"), .004, 1., False),
            ("inner_val", "spot", pd.Timestamp("2021-01-01", tz="UTC"), TRAIN_END, .004, 1., False),
            ("funded_val", "perp", pd.Timestamp("2021-01-01", tz="UTC"), TRAIN_END, .0005, 1., True),
        ]
    if stage != "holdout":
        raise ValueError(f"Unknown stage: {stage}")
    specs = [
        ("holdout", "spot", HOLDOUT, END, .004, 1., False),
        ("discount_holdout", "spot", HOLDOUT, END, .001, 1., False),
        ("eth_holdout", "eth", HOLDOUT, END, .004, 1., False),
        ("funded_holdout", "perp", HOLDOUT, END, .0005, 1., True),
    ]
    rng = np.random.default_rng(192)
    days = (END.normalize() - HOLDOUT).days
    for k in range(24):
        length = int(rng.integers(120, 366))
        start = HOLDOUT + pd.Timedelta(days=int(rng.integers(0, days - length + 1)))
        end = start + pd.Timedelta(days=length) - pd.Timedelta(minutes=5)
        for kind, fee, funded in (("spot", .004, False), ("perp", .0005, True)):
            specs.append((f"beta_{kind}_{k:02d}", kind, start, end, fee, 1., funded))
    return specs


def validate_manifest():
    path = OUT / "manifest.json"
    if not path.exists():
        raise RuntimeError("Freeze reports/r192_improvements/manifest.json before holdout")
    hashes = json.loads(path.read_text())["hashes"]
    if not hashes:
        raise RuntimeError("Frozen manifest has no source/data hashes")
    for relative, expected in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source/data changed: {relative}")


def get_strategy(name, kind="spot"):
    if name == "buy_and_hold":
        if kind == "perp":
            passive = ConstantExposureHold(c=1.)
            passive.name = name
            return passive
        return BuyAndHold()
    if name == "kelly_regime_v4":
        return KellyRegimeV4()
    return make_strategy(name)


class InitialTarget(Strategy):
    """Give renamed target strategies the same fresh-account entry as v4."""

    def __init__(self, strategy, first_eligible):
        self.strategy, self.name = strategy, strategy.name
        self.warmup = strategy.warmup
        self.first_eligible = first_eligible

    def prepare(self, frame):
        return frame

    def on_bar(self, ctx):
        if ctx.i == self.first_eligible:
            target = float(ctx.bar["target"])
            if target > 0:
                ctx.order_notional(target)
        else:
            self.strategy.on_bar(ctx)


def evaluate(strategy, prepared_frame, spec, funding=None):
    """Reuse accounting, explicitly initialize every target strategy equally."""
    initial_target = strategy.name in (*ALL_NAMES, "kelly_regime_v4")
    original_name = strategy.name
    if initial_target:
        lo = 0 if spec[2] is None else int(prepared_frame.index.searchsorted(spec[2]))
        strategy = InitialTarget(strategy, max(0, strategy.warmup - lo))
    row, day, result = native_evaluate(strategy, prepared_frame, spec, funding)
    row["initial_target_convention"] = "first_globally_eligible_bar" if initial_target else "strategy_native"
    row["passive_model"] = ("constant_1x_notional_10pct_relative_band"
        if original_name == "buy_and_hold" and spec[1] == "perp" else "native")
    # Compare executed paths, not signals absorbed by the broker's deadband.
    fills = [(str(f.ts), f.side.name, round(f.qty, 10), round(f.price, 8), round(f.fee, 8))
             for f in result.fills]
    row["fill_signature"] = hashlib.sha256(json.dumps(fills).encode()).hexdigest()
    return row, day, result


def _write_csv(frame, path):
    temporary = path.with_name("partial_" + path.name)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_one(name, stage):
    if stage == "holdout":
        validate_manifest()
    if name not in WORKER_NAMES:
        raise ValueError(f"Unknown strategy: {name}")
    specs = specifications(stage)
    if name in AUXILIARIES:
        specs = [s for s in specs if not s[0].startswith("beta_")]
    out = OUT / name / stage
    if (out / "cells.csv").exists():
        raise RuntimeError(f"Refusing to overwrite evaluated cells: {out}; retain/count the prior run first")
    out.mkdir(parents=True, exist_ok=True)
    cutoff = TRAIN_END if stage == "train" else END
    prepared = {}
    for kind in dict.fromkeys(s[1] for s in specs):
        raw = load_ohlcv_csv(ROOT / "data" / FILES[kind]).loc[:cutoff]
        prepared[kind] = get_strategy(name, kind).prepare(raw.copy())
    funding = load_funding_deribit(ROOT / "data")
    if funding is not None:
        funding = funding.loc[:cutoff]
    rows, daily = [], []
    for k, spec in enumerate(specs):
        row, day, result = evaluate(get_strategy(name, spec[1]), prepared[spec[1]], spec, funding)
        rows.append(row)
        daily.append(day)
        if spec[0] in ("inner_val", "funded_val", "holdout", "funded_holdout"):
            pd.DataFrame([{"timestamp": str(f.ts), "side": f.side.name,
                "qty": f.qty, "price": f.price, "fee": f.fee, "kind": f.kind,
                "realized_pnl": f.realized_pnl} for f in result.fills],
                columns=["timestamp", "side", "qty", "price", "fee", "kind", "realized_pnl"]).to_csv(
                    out / f"{spec[0]}_fills.csv", index=False)
        _write_csv(pd.DataFrame(rows), out / "cells.csv")
        _write_csv(pd.concat(daily, ignore_index=True), out / "daily.csv.gz")
        print(f"{name} {stage}: {k + 1}/{len(specs)} {spec[0]} "
              f"${row['final_balance']:,.0f}, {row['fills_per_day']:.3f} fills/day", flush=True)
    return rows


def run(stage, workers=3):
    if stage == "holdout":
        validate_manifest()
    specifications(stage)
    OUT.mkdir(parents=True, exist_ok=True)
    existing = [OUT / name / stage / "cells.csv" for name in WORKER_NAMES
                if (OUT / name / stage / "cells.csv").exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite {len(existing)} existing result receipts")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(run_one, name, stage) for name in WORKER_NAMES]
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
