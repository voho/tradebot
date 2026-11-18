"""R-194 harness inventory, initial-entry parity and freeze guards (no market data)."""

import pytest

from experiments import r194_eval as evaluation
from experiments import r194_report as report


def test_core_inventory_matches_protocol():
    train, holdout = evaluation.specifications("train"), evaluation.specifications("holdout")
    full = len(evaluation.CANDIDATES) + len(evaluation.CONTROLS)
    named = sum(not spec[0].startswith("beta_") for spec in holdout)
    core = full * (len(train) + len(holdout)) + len(evaluation.AUXILIARIES) * (len(train) + named)
    assert (len(evaluation.CANDIDATES), len(evaluation.AUXILIARIES)) == (3, 6)
    assert core == 317 and len(evaluation.WORKER_NAMES) * len(train) == 33
    assert report.PRIOR == 4850


def test_hybrids_get_the_parent_initial_entry_convention():
    # R-192's evaluator only initialises names it knows; R-194's must be among them.
    import inspect
    source = inspect.getsource(evaluation.evaluate)
    assert "ALL_NAMES" in source and "kelly_regime_v4" in source
    assert set(evaluation.CANDIDATES) <= set(evaluation.ALL_NAMES)


def test_holdout_refuses_without_frozen_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="Freeze"):
        evaluation.validate_manifest()
    (tmp_path / "manifest.json").write_text('{"hashes": {"README.md": "0"}}')
    with pytest.raises(RuntimeError, match="changed"):
        evaluation.validate_manifest()
