#!/usr/bin/env python
"""R-191 actual-broker passive risk matches, including every resampled window.

Fit constant exposure to each candidate's realized daily volatility, only
for inference. Each comparison starts at c=.5, allows at most three native
simulations, and retains every attempt even when the 2% tolerance is missed.
The passively rebalanced control pays the same fees, slippage and funding.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np
import pandas as pd

from matched_hold import ConstantExposureHold
import r191_eval as evaluation


def matching_specs(stage):
    """Two validation cells, or two main holdouts and their 48 beta windows."""
    wanted = ("inner_val", "funded_val") if stage == "train" else ("holdout", "funded_holdout")
    return [spec for spec in evaluation.specifications(stage)
            if spec[0] in wanted or spec[0].startswith("beta_")]


def match_one(candidate, stage):
    if stage == "holdout":
        evaluation.validate_manifest()
    if candidate not in evaluation.CANDIDATES:
        raise ValueError(f"Unknown primary candidate: {candidate}")
    specs = matching_specs(stage)
    out = evaluation.OUT / candidate / f"{stage}_matched"
    if (out / "cells.csv").exists():
        raise RuntimeError(f"Refusing to overwrite matching attempts: {out}; retain/count the prior run first")
    references = pd.read_csv(evaluation.OUT / f"{stage}_cells.csv")
    references = references.loc[references.strategy == candidate].set_index("cell")
    if not references.index.is_unique:
        raise ValueError(f"Duplicate reference cells: {candidate} {stage}")
    cutoff = evaluation.TRAIN_END if stage == "train" else evaluation.END
    raw = {kind: evaluation.load_ohlcv_csv(evaluation.ROOT / "data" / evaluation.FILES[kind]).loc[:cutoff]
           for kind in dict.fromkeys(spec[1] for spec in specs)}
    funding = evaluation.load_funding_deribit(evaluation.ROOT / "data")
    if funding is not None:
        funding = funding.loc[:cutoff]
    out.mkdir(parents=True, exist_ok=True)
    rows, daily = [], []
    for spec in specs:
        cell, kind = spec[:2]
        target = float(references.loc[cell, "annualized_volatility"]) / np.sqrt(365.25)
        if not np.isfinite(target) or target < 0:
            raise ValueError(f"Invalid reference volatility: {candidate} {cell}")
        # This control has no learned state or warmup; only copy its measured
        # window instead of the entire multi-year dataset on every attempt.
        measured = raw[kind].loc[spec[2]:spec[3]]
        c = .5
        for attempt in range(3):
            control = ConstantExposureHold(c=c)
            control.name = f"match_{candidate}_{cell}_i{attempt}"
            row, day, _ = evaluation.evaluate(control, control.prepare(measured.copy()), spec, funding)
            achieved = float(row["annualized_volatility"]) / np.sqrt(365.25)
            error = abs(achieved - target) / target if target > 0 else float("inf")
            valid = bool(np.isfinite(error) and error <= .02)
            row.update(reference_candidate=candidate, control_c=c, match_attempt=attempt,
                       stage=stage, target_daily_volatility=target,
                       achieved_daily_volatility=achieved, relative_vol_error=error,
                       matched_valid=valid, final_selected=valid or attempt == 2)
            rows.append(row)
            daily.append(day)
            evaluation._write_csv(pd.DataFrame(rows), out / "cells.csv")
            evaluation._write_csv(pd.concat(daily, ignore_index=True), out / "daily.csv.gz")
            print(f"{candidate} {stage}: {cell} match {attempt + 1}/3 "
                  f"c={c:.6f}, relative volatility error={error:.2%}", flush=True)
            if valid:
                break
            ratio = target / achieved if achieved > 0 else float("inf")
            c = float(np.clip(c * ratio, .001, 2. if kind == "perp" else 1.))
    return rows


def run(stage, workers=3):
    if stage == "holdout":
        evaluation.validate_manifest()
    matching_specs(stage)
    evaluation.OUT.mkdir(parents=True, exist_ok=True)
    existing = [evaluation.OUT / name / f"{stage}_matched/cells.csv"
                for name in evaluation.CANDIDATES
                if (evaluation.OUT / name / f"{stage}_matched/cells.csv").exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite {len(existing)} existing matching receipts")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(match_one, candidate, stage) for candidate in evaluation.CANDIDATES]
        for future in as_completed(pending):
            rows.extend(future.result())
            evaluation._write_csv(pd.DataFrame(rows).sort_values(["reference_candidate", "cell", "match_attempt"]),
                                  evaluation.OUT / f"{stage}_matched_cells.csv")
    daily = pd.concat([pd.read_csv(evaluation.OUT / name / f"{stage}_matched/daily.csv.gz")
                       for name in evaluation.CANDIDATES], ignore_index=True)
    evaluation._write_csv(daily, evaluation.OUT / f"{stage}_matched_daily.csv.gz")
    result = pd.DataFrame(rows)
    selected = result.loc[result.final_selected]
    metadata = {"stage": stage, "attempts": len(result), "comparisons": len(selected),
                "valid_matches": int(selected.matched_valid.sum()),
                "invalid_matches": int((~selected.matched_valid).sum()),
                "attempts_by_asset": result.groupby("asset").size().to_dict(),
                "holdout_consultations": len(result) if stage == "holdout" else 0,
                "max_attempts_per_comparison": 3, "starting_c": .5,
                "relative_vol_tolerance": .02, "fitted_ex_post": True,
                "window_risk_matching": "Each candidate/window independently matched"}
    (evaluation.OUT / f"{stage}_matched_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "holdout"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.stage, args.workers)
