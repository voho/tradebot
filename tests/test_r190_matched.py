"""Counted solver bounds, data isolation and actual broker matching controls."""

from concurrent.futures import Future
import json

import numpy as np
import pandas as pd
import pytest

from experiments import r190_matched as matched


def setup_synthetic(monkeypatch, tmp_path, targets=(.4, 1.5)):
    evaluation = matched.evaluation
    index = pd.date_range("2022-12-31 23:30", periods=12, freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                        "close": 100., "volume": 1.}, index=index)
    specs = [("inner_val", "spot", index[1], index[5], .004, 1., False),
             ("funded_val", "perp", index[1], index[5], .0005, 1., True)]
    candidate = evaluation.CANDIDATES[0]
    pd.DataFrame([{"strategy": candidate, "cell": spec[0],
                   "annualized_volatility": vol * np.sqrt(365.25)}
                  for spec, vol in zip(specs, targets)]).to_csv(tmp_path / "train_cells.csv", index=False)
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "CANDIDATES", (candidate,))
    monkeypatch.setattr(evaluation, "specifications", lambda _: specs)
    monkeypatch.setattr(evaluation, "load_ohlcv_csv", lambda _: raw.copy())
    monkeypatch.setattr(evaluation, "load_funding_deribit", lambda _: pd.Series(.001, index=index))
    return evaluation, candidate, index


def test_solver_updates_exposure_retains_all_attempts_and_truncates_training(monkeypatch, tmp_path):
    evaluation, candidate, index = setup_synthetic(monkeypatch, tmp_path)
    seen = []

    def evaluate(control, prepared, spec, funding):
        assert prepared.index[-1] == evaluation.TRAIN_END
        assert funding.index[-1] == evaluation.TRAIN_END
        assert control.deadband == .10 and not control.static
        assert prepared.target.eq(control.c).all()
        seen.append((spec[0], control.c))
        # The deterministic synthetic response requires two attempts on spot
        # and three on perp, where the upper c=2 bound prevents a valid match.
        vol = control.c * .5
        row = {"strategy": control.name, "cell": spec[0], "asset": spec[1],
               "annualized_volatility": vol * np.sqrt(365.25)}
        day = pd.DataFrame({"strategy": [control.name], "cell": [spec[0]],
                            "timestamp": [str(index[1])], "return": [vol], "equity": [1000.]})
        return row, day, None

    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    rows = matched.match_one(candidate, "train")
    assert seen == [("inner_val", .5), ("inner_val", .8),
                    ("funded_val", .5), ("funded_val", 2.), ("funded_val", 2.)]
    assert len(rows) == 5
    frame = pd.read_csv(tmp_path / candidate / "train_matched/cells.csv")
    daily = pd.read_csv(tmp_path / candidate / "train_matched/daily.csv.gz")
    assert len(frame) == len(daily) == 5 and frame.strategy.is_unique
    final = frame.loc[frame.final_selected].set_index("cell")
    assert len(final) == 2 and final.loc["inner_val", "matched_valid"]
    assert not final.loc["funded_val", "matched_valid"]
    assert final.loc["funded_val", "relative_vol_error"] == pytest.approx(1 / 3)
    assert set(daily.strategy) == set(frame.strategy)


def test_native_broker_is_used_for_constant_control_and_zero_volatility_is_invalid(monkeypatch, tmp_path):
    evaluation, candidate, _ = setup_synthetic(monkeypatch, tmp_path, targets=(0., 0.))
    # The synthetic span contains no settlement and the inherited broker
    # applies the same fee/slippage/minimum-size settings as candidate cells.
    monkeypatch.setattr(evaluation, "load_funding_deribit", lambda _: pd.Series(dtype=float,
                        index=pd.DatetimeIndex([], tz="UTC")))
    rows = matched.match_one(candidate, "train")
    assert len(rows) == 6
    assert not any(row["matched_valid"] for row in rows)
    assert sum(row["final_selected"] for row in rows) == 2
    first = rows[0]
    assert first["fills"] == 1 and first["decision_requests"] == 1
    assert first["fee_rate"] == .004 and first["slippage_bps"] == 1.
    assert first["min_notional"] == 10. and first["final_balance"] < 1000.
    assert all(.001 <= row["control_c"] <= (2 if row["asset"] == "perp" else 1) for row in rows)


def test_holdout_checks_frozen_manifest_before_reading_any_financial_data(monkeypatch):
    def reject():
        raise RuntimeError("manifest rejected")

    monkeypatch.setattr(matched.evaluation, "validate_manifest", reject)
    monkeypatch.setattr(pd, "read_csv", lambda _: pytest.fail("Read before manifest validation"))
    with pytest.raises(RuntimeError, match="manifest rejected"):
        matched.match_one(matched.evaluation.CANDIDATES[0], "holdout")
    with pytest.raises(RuntimeError, match="manifest rejected"):
        matched.run("holdout")


def test_aggregate_metadata_counts_every_attempt_including_invalid_matches(monkeypatch, tmp_path):
    evaluation, candidate, _ = setup_synthetic(monkeypatch, tmp_path, targets=(0., 0.))
    monkeypatch.setattr(evaluation, "load_funding_deribit", lambda _: pd.Series(dtype=float,
                        index=pd.DatetimeIndex([], tz="UTC")))

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, *args):
            result = Future()
            result.set_result(fn(*args))
            return result

    monkeypatch.setattr(matched, "ProcessPoolExecutor", InlinePool)
    result = matched.run("train", workers=1)
    metadata = json.loads((tmp_path / "train_matched_meta.json").read_text())
    cells = pd.read_csv(tmp_path / "train_matched_cells.csv")
    daily = pd.read_csv(tmp_path / "train_matched_daily.csv.gz")
    assert len(result) == len(cells) == metadata["attempts"] == 6
    assert metadata["comparisons"] == metadata["invalid_matches"] == 2
    assert metadata["valid_matches"] == metadata["holdout_consultations"] == 0
    assert metadata["attempts_by_asset"] == {"perp": 3, "spot": 3}
    assert set(daily.strategy) == set(cells.strategy)
