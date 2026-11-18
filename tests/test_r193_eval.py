"""R-193 harness inventory and freeze guards (no market data is evaluated)."""

import pytest

from experiments import r193_eval as evaluation
from experiments import r193_matched as matching


def test_core_inventory_matches_protocol():
    train, holdout = evaluation.specifications("train"), evaluation.specifications("holdout")
    full = len(evaluation.CANDIDATES) + len(evaluation.CONTROLS)
    named = sum(not spec[0].startswith("beta_") for spec in holdout)
    core = full * (len(train) + len(holdout)) + len(evaluation.AUXILIARIES) * (len(train) + named)
    assert (len(evaluation.CANDIDATES), len(evaluation.AUXILIARIES)) == (3, 6)
    assert core == 317
    assert len(evaluation.WORKER_NAMES) * len(train) == 33
    assert len(matching.matching_specs("train")) == 2 and len(matching.matching_specs("holdout")) == 50


def test_holdout_refuses_without_frozen_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="Freeze"):
        evaluation.run_one("r193_attention", "holdout")
    (tmp_path / "manifest.json").write_text('{"hashes": {"README.md": "0"}}')
    with pytest.raises(RuntimeError, match="changed"):
        evaluation.validate_manifest()


def test_controls_use_r191_constructions():
    assert type(evaluation.get_strategy("buy_and_hold", "perp")).__name__ == "ConstantExposureHold"
    assert evaluation.get_strategy("kelly_regime_v4").name == "kelly_regime_v4"
    assert evaluation.get_strategy("r193_transformer_low").config["context"] == 15
