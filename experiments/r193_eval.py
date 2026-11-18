#!/usr/bin/env python
"""R-193 fresh-account evaluation, reusing R-191's cells, windows and evaluator.

Three primaries and two controls receive R-191's 3 training cells, 4 named
holdout cells and 48 seed-191 windows; six neighbours receive the 7 named
cells: 317 core evaluations. Holdout runs refuse to start without the frozen
manifest written by ``r193_report.py freeze``.
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

import pandas as pd

from experiments.r191_eval import (END, FILES, HOLDOUT, TRAIN_END,  # noqa: F401
                                   _write_csv, get_strategy as r191_strategy,
                                   specifications)
from experiments.r193_strategies import ALL_NAMES, PRIMARY_NAMES, make_strategy
from r190_eval import evaluate
from tradebot.data import load_funding_deribit, load_ohlcv_csv

OUT = ROOT / "reports/r193_attention"
CANDIDATES = tuple(PRIMARY_NAMES)
AUXILIARIES = tuple(name for name in ALL_NAMES if name not in CANDIDATES)
CONTROLS = ("buy_and_hold", "kelly_regime_v4")
NAMES = CANDIDATES + CONTROLS
WORKER_NAMES = tuple(ALL_NAMES) + CONTROLS


def validate_manifest():
    """Reject missing or changed freezes before touching holdout results."""
    path = OUT / "manifest.json"
    if not path.exists():
        raise RuntimeError("Freeze reports/r193_attention/manifest.json before holdout")
    hashes = json.loads(path.read_text())["hashes"]
    if not hashes:
        raise RuntimeError("Frozen manifest has no source/data hashes")
    for relative, expected in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source/data changed: {relative}")


def get_strategy(name, kind="spot"):
    return r191_strategy(name, kind) if name in CONTROLS else make_strategy(name)


def run_one(name, stage):
    """Run one fixed configuration, writing a receipt after every cell."""
    if stage == "holdout":
        validate_manifest()
    if name not in WORKER_NAMES:
        raise ValueError(f"Unknown strategy: {name}")
    specs = specifications(stage)
    if name in AUXILIARIES:
        specs = [spec for spec in specs if not spec[0].startswith("beta_")]
    out = OUT / name / stage
    if (out / "cells.csv").exists():
        raise RuntimeError(f"Refusing to overwrite evaluated cells: {out}; retain/count the prior run first")
    out.mkdir(parents=True, exist_ok=True)
    cutoff = TRAIN_END if stage == "train" else END
    prepared = {}
    for kind in dict.fromkeys(spec[1] for spec in specs):
        raw = load_ohlcv_csv(ROOT / "data" / FILES[kind]).loc[:cutoff]
        prepared[kind] = get_strategy(name, kind).prepare(raw.copy())
        print(f"{name} {stage}: prepared {kind}", flush=True)
    funding = load_funding_deribit(ROOT / "data")
    if funding is not None:
        funding = funding.loc[:cutoff]
    rows, daily = [], []
    for k, spec in enumerate(specs):
        row, day, result = evaluate(get_strategy(name, spec[1]), prepared[spec[1]], spec, funding)
        row["passive_model"] = ("constant_1x_notional_10pct_relative_band"
                                if name == "buy_and_hold" and spec[1] == "perp" else "native")
        rows.append(row)
        daily.append(day)
        if spec[0] == "holdout":
            pd.DataFrame([{"timestamp": str(f.ts), "side": f.side.name, "qty": f.qty,
                           "price": f.price, "fee": f.fee, "kind": f.kind,
                           "realized_pnl": f.realized_pnl} for f in result.fills],
                         columns=["timestamp", "side", "qty", "price", "fee", "kind",
                                  "realized_pnl"]).to_csv(out / "holdout_fills.csv", index=False)
        _write_csv(pd.DataFrame(rows), out / "cells.csv")
        _write_csv(pd.concat(daily, ignore_index=True), out / "daily.csv.gz")
        print(f"{name} {stage}: {k + 1}/{len(specs)} {spec[0]} "
              f"${row['final_balance']:,.0f}, {row['fills_per_day']:.3f} fills/day", flush=True)
    # Exposure path on the primary market, for the "learned always long" check.
    spot = prepared.get("spot")
    if spot is not None and name in ALL_NAMES:
        days = spot.loc[spot["r193_decision"], "target"]
        days.rename("target").to_csv(out / "spot_targets.csv.gz")
    return rows


def run(stage, workers=4):
    if stage == "holdout":
        validate_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    existing = [OUT / name / stage / "cells.csv" for name in WORKER_NAMES
                if (OUT / name / stage / "cells.csv").exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite {len(existing)} existing result receipts")
    rows = []
    # Slowest (transformer, long context) first so the pool finishes evenly.
    order = sorted(WORKER_NAMES, key=lambda n: ("transformer" not in n, n != "r193_transformer_high"))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(run_one, name, stage) for name in order]
        for future in as_completed(pending):
            rows.extend(future.result())
            _write_csv(pd.DataFrame(rows).sort_values(["strategy", "cell"]), OUT / f"{stage}_cells.csv")
    _write_csv(pd.concat([pd.read_csv(OUT / name / stage / "daily.csv.gz") for name in WORKER_NAMES],
                         ignore_index=True), OUT / f"{stage}_daily.csv.gz")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "holdout"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.stage, args.workers)
