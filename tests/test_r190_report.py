"""Synthetic R-190 gate partition, auxiliary plateau and consultation counts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "r190_report", Path(__file__).resolve().parents[1] / "experiments/r190_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

from r190_eval import AUXILIARIES, WORKER_NAMES, specifications


def test_decision_partition_risk_plateau_and_every_counted_attempt(monkeypatch, tmp_path):
    stages = {}
    for stage in ("train", "holdout"):
        rows = []
        for name in WORKER_NAMES:
            candidate = name.startswith("r190_")
            for spec in specifications(stage):
                if name in AUXILIARIES and spec[0].startswith("beta_"):
                    continue
                rows.append(dict(strategy=name, cell=spec[0], final_balance=1200. if candidate else 1100.,
                    daily_sharpe=1.3 if candidate else 1., annualized_volatility=.5,
                    fills=100, fills_per_day=2., liquidated=False))
        frame = pd.DataFrame(rows)
        # Arbitrary synthetic returns; DSR itself is tested in test_inference.
        daily = {(row.strategy, row.cell): pd.Series(np.tile([.01, -.005, .002], 10))
                 for row in frame.itertuples()}
        stages[stage] = (frame, daily)
    matching = {"train": (pd.DataFrame({"attempt": range(30)}), {}),
                "holdout": (pd.DataFrame({"attempt": range(40)}), {})}
    monkeypatch.setattr(report, "OUT", tmp_path)
    monkeypatch.setattr(report, "read", lambda stage, matched=False: matching[stage] if matched else stages[stage])
    monkeypatch.setattr(report, "deflated_sharpe_ratio", lambda *args, **kwargs: 1.)
    (tmp_path / "manifest.json").write_text(json.dumps({"sd_trials": .5}))
    (tmp_path / "audit.json").write_text(json.dumps({"train_evaluations": 2, "holdout_evaluations": 3}))
    boot = pd.DataFrame([dict(strategy=name, cell=cell, control=control,
        risk_valid=True, d_sharpe=.3, d_sharpe_lo=.1, d_growth_lo=.1)
        for name in report.CANDIDATES for cell in report.PRIMARY
        for control in ("parent", "matched_hold")])
    _, decisions, counts = report.decide(boot)
    assert decisions.verdict.eq("PROMOTED").all()
    assert decisions.beta_spot_valid.eq(24).all()
    assert decisions.beta_spot_wins.eq(24).all()
    assert counts["core_evaluations"] == 784
    assert counts["total_evaluations"] == 784 + 30 + 40 + 2 + 3
    assert counts["holdout_consultations"] == 736 + 40 + 3
    assert counts["cumulative_consultations_approx"] == 1503 + 736 + 40 + 3

    # Each missing conjunction rejects; a different successful gate cannot rescue it.
    risk, strict, interval, cadence = report.CANDIDATES[:4]
    boot.loc[(boot.strategy == risk) & (boot.cell == "inner_val"), "risk_valid"] = False
    boot.loc[(boot.strategy == strict) & (boot.cell == "holdout"), "d_sharpe"] = .20
    boot.loc[(boot.strategy == interval) & (boot.cell == "funded_holdout"), "d_growth_lo"] = 0.
    hold = stages["holdout"][0]
    hold.loc[(hold.strategy == cadence) & (hold.cell == "holdout"), "fills_per_day"] = 6.01
    hold.loc[(hold.strategy == "r190_blend_b05") & (hold.cell == "funded_holdout"), "final_balance"] = 1000.
    _, decisions, _ = report.decide(boot)
    indexed = decisions.set_index("strategy")
    assert not indexed.loc[risk, "point_and_risk"]
    assert not indexed.loc[strict, "point_and_risk"]
    assert not indexed.loc[interval, "intervals"]
    assert not indexed.loc[cadence, "cadence"]
    assert not indexed.loc["r190_blend_b10", "plateau"]
    rejected = [risk, strict, interval, cadence, "r190_blend_b10"]
    assert indexed.loc[rejected, "verdict"].eq("NEGATIVE").all()
    assert set(decisions.verdict) == {"PROMOTED", "NEGATIVE"}
