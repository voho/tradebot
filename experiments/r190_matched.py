#!/usr/bin/env python
"""R-190's counted, ex-post constant-exposure risk-matching controls.

Each candidate/cell starts at c=0.5 and receives at most three independent
broker simulations. The scalar is fitted to that cell's realized volatility
only for comparison; it is not a deployable forecasting rule. All attempts,
including failed matches, are retained and counted. ConstantExposureHold's
explicit-quantity orders and relative 10% rebalance band remain unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd

from matched_hold import ConstantExposureHold
import r190_eval as evaluation


def match_one(candidate, stage):
    """Return every counted attempt for one candidate's two primary cells."""
    if stage == "holdout":
        evaluation.validate_manifest()
    if candidate not in evaluation.CANDIDATES:
        raise ValueError(f"Unknown candidate: {candidate}")
    wanted = ("inner_val", "funded_val") if stage == "train" else ("holdout", "funded_holdout")
    specs = [spec for spec in evaluation.specifications(stage) if spec[0] in wanted]
    references = pd.read_csv(evaluation.OUT / f"{stage}_cells.csv")
    references = references.loc[references.strategy == candidate].set_index("cell")
    if not references.index.is_unique:
        raise ValueError(f"Duplicate reference cells: {candidate} {stage}")
    cutoff = evaluation.TRAIN_END if stage == "train" else evaluation.END
    raw = {kind: evaluation.load_ohlcv_csv(evaluation.ROOT / "data" / evaluation.FILES[kind]).loc[:cutoff]
           for kind in dict.fromkeys(spec[1] for spec in specs)}
    funding = evaluation.load_funding_deribit(evaluation.ROOT / "data").loc[:cutoff]
    out = evaluation.OUT / candidate / f"{stage}_matched"
    out.mkdir(parents=True, exist_ok=True)
    rows, daily = [], []
    for spec in specs:
        cell, kind = spec[:2]
        target = float(references.loc[cell, "annualized_volatility"]) / np.sqrt(365.25)
        if not np.isfinite(target) or target < 0:
            raise ValueError(f"Invalid reference volatility: {candidate} {cell}")
        c = .5
        for attempt in range(3):
            control = ConstantExposureHold(c=c)
            control.name = f"match_{candidate}_{cell}_i{attempt}"
            row, day, _ = evaluation.evaluate(control, control.prepare(raw[kind].copy()), spec, funding)
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
            # With zero achieved volatility the ratio's limiting value is
            # infinity: try the frozen cap, retaining the failed attempt.
            ratio = target / achieved if achieved > 0 else float("inf")
            c = float(np.clip(c * ratio, .001, 2. if kind == "perp" else 1.))
    return rows


def run(stage, workers=3):
    if stage == "holdout":
        evaluation.validate_manifest()
    evaluation.specifications(stage)  # Reject unknown stages before dispatch.
    evaluation.OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(match_one, candidate, stage) for candidate in evaluation.CANDIDATES]
        for future in as_completed(pending):
            rows.extend(future.result())
            evaluation._write_csv(pd.DataFrame(rows).sort_values(["reference_candidate", "cell", "match_attempt"]),
                                  evaluation.OUT / f"{stage}_matched_cells.csv")
    daily = pd.concat([pd.read_csv(evaluation.OUT / candidate / f"{stage}_matched/daily.csv.gz")
                       for candidate in evaluation.CANDIDATES], ignore_index=True)
    evaluation._write_csv(daily, evaluation.OUT / f"{stage}_matched_daily.csv.gz")
    result = pd.DataFrame(rows)
    selected = result.loc[result.final_selected]
    metadata = {"stage": stage, "attempts": len(result), "comparisons": len(selected),
                "valid_matches": int(selected.matched_valid.sum()),
                "invalid_matches": int((~selected.matched_valid).sum()),
                "attempts_by_asset": result.groupby("asset").size().to_dict(),
                "holdout_consultations": len(result) if stage == "holdout" else 0,
                "max_attempts_per_comparison": 3, "starting_c": .5,
                "relative_vol_tolerance": .02, "fitted_ex_post": True}
    (evaluation.OUT / f"{stage}_matched_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "holdout"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.stage, args.workers)
