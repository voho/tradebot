"""R-192 frozen battery, data isolation and counted actual-broker controls."""

from concurrent.futures import Future
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments import r192_eval as evaluation
from experiments import r192_matched as matching
from tradebot.strategy import Strategy


@pytest.mark.parametrize("start_index", [None, 4])
def test_renamed_unchanged_v4_has_identical_startup_fills_and_pnl(start_index):
    """A positive target held through warmup cannot favor the named parent."""
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    index = pd.date_range("2021-01-01", periods=12, freq="5min", tz="UTC")
    prepared = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
        "close": 100., "volume": 1., "target": .5}, index=index)
    prepared.loc[index[8]:, "target"] = 0.
    parent, candidate = KellyRegimeV4(), KellyRegimeV4()
    candidate.name = evaluation.CANDIDATES[0]
    parent.warmup = candidate.warmup = 3
    start = None if start_index is None else index[start_index]
    spec = ("synthetic", "spot", start, index[-1], .004, 1., False)
    a, ad, ar = evaluation.evaluate(parent, prepared, spec)
    b, bd, br = evaluation.evaluate(candidate, prepared, spec)
    assert a["fill_signature"] == b["fill_signature"]
    assert a["final_balance"] == b["final_balance"]
    assert ad["return"].equals(bd["return"])
    assert ar.equity.equals(br.equity)
    expected_first = index[4 if start_index is None else start_index + 1]
    assert [(f.ts, f.side.name) for f in ar.fills] == [(expected_first, "BUY"), (index[9], "SELL")]
    assert a["initial_target_convention"] == b["initial_target_convention"] == "first_globally_eligible_bar"


def test_fixed_battery_pairs_costs_windows_and_counts_every_configuration():
    train, holdout = evaluation.specifications("train"), evaluation.specifications("holdout")
    assert len(evaluation.CANDIDATES) == 5
    assert len(evaluation.AUXILIARIES) == 10
    assert len(evaluation.NAMES) == 7
    assert len(evaluation.WORKER_NAMES) == 17
    assert len(train) == 3 and len(holdout) == 52
    assert len(evaluation.NAMES) * 55 + len(evaluation.AUXILIARIES) * 7 == 455
    assert max(spec[3] for spec in train) < evaluation.HOLDOUT
    cells = {spec[0]: spec for spec in holdout}
    assert cells["holdout"][4:] == (.004, 1., False)
    assert cells["discount_holdout"][4:] == (.001, 1., False)
    assert cells["eth_holdout"][4:] == (.004, 1., False)
    assert cells["funded_holdout"][4:] == (.0005, 1., True)
    for k in range(24):
        spot, perp = cells[f"beta_spot_{k:02d}"], cells[f"beta_perp_{k:02d}"]
        assert spot[2:4] == perp[2:4]
        assert spot[4:] == (.004, 1., False)
        assert perp[4:] == (.0005, 1., True)
        assert evaluation.HOLDOUT <= spot[2] < spot[3] <= evaluation.END
        assert 120 <= (spot[3] - spot[2] + pd.Timedelta(minutes=5)) / pd.Timedelta(days=1) <= 365
    assert len(matching.matching_specs("train")) == 2
    assert len(matching.matching_specs("holdout")) == 50
    assert 5 * (2 + 50) == 260
    with pytest.raises(ValueError, match="Unknown stage"):
        matching.matching_specs("invalid")


def test_funded_passive_control_has_one_times_notional_and_next_open_fills():
    strategy = evaluation.get_strategy("buy_and_hold", "perp")
    assert strategy.c == 1. and strategy.deadband == .1 and not strategy.static
    assert strategy.name == "buy_and_hold"
    index = pd.date_range("2021-01-01 07:45", periods=8, freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                        "close": 100., "volume": 1.}, index=index)
    funding = pd.Series(.001, index=pd.DatetimeIndex([index[3]]))
    spec = ("funded_val", "perp", index[0], index[-1], 0., 0., True)
    row, daily, result = evaluation.evaluate(strategy, strategy.prepare(raw), spec, funding)
    assert result.fills[0].ts == index[1]
    assert result.fills[0].qty == 10.
    assert row["funding_paid"] == pytest.approx(1.)
    assert row["final_balance"] == pytest.approx(999.)
    assert row["fills"] == 1
    assert row["completed_round_trips"] == 0
    assert daily["return"].iloc[0] == pytest.approx(-.001)


def test_training_preparation_slices_data_and_freshens_each_decision_object(monkeypatch, tmp_path):
    index = pd.date_range("2022-12-31 23:30", periods=12, freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                        "close": 100., "volume": 1.}, index=index)
    seen, constructed = [], []
    name = evaluation.CANDIDATES[0]

    class CausalHold(Strategy):
        warmup = 0

        def __init__(self):
            self.name = name
            self.calls = 0
            constructed.append(self)

        def prepare(self, df):
            seen.append(df.index[-1])
            df["known"] = df.close.cumsum()
            df["target"] = 0.
            return df

        def on_bar(self, ctx):
            self.calls += 1
            if not ctx.in_market:
                ctx.order_target(1.)

    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "get_strategy", lambda *args: CausalHold())
    monkeypatch.setattr(evaluation, "load_ohlcv_csv", lambda _: raw.copy())
    monkeypatch.setattr(evaluation, "load_funding_deribit", lambda _: None)
    monkeypatch.setattr(evaluation, "specifications", lambda _: [
        ("inner_train", "spot", index[0], index[3], 0., 0., False),
        ("inner_val", "spot", index[2], index[5], 0., 0., False)])
    rows = evaluation.run_one(name, "train")
    assert seen == [evaluation.TRAIN_END]
    assert len(constructed) == 3
    assert constructed[0].calls == 0
    assert constructed[1].calls == constructed[2].calls > 0
    assert len(rows) == len(pd.read_csv(tmp_path / name / "train/cells.csv")) == 2
    assert all(row["final_balance"] == 1000. for row in rows)
    with pytest.raises(RuntimeError, match="overwrite"):
        evaluation.run_one(name, "train")


def test_holdout_guard_precedes_data_access_and_checks_frozen_hashes(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    monkeypatch.setattr(evaluation, "load_ohlcv_csv", lambda _: pytest.fail("Read before freeze validation"))
    with pytest.raises(RuntimeError, match="Freeze"):
        evaluation.run_one("buy_and_hold", "holdout")
    source = tmp_path / "source.py"
    source.write_text("frozen")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "hashes": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}))
    evaluation.validate_manifest()
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="source/data changed"):
        evaluation.run_one("buy_and_hold", "holdout")


def test_matches_retain_failed_attempts_limit_exposure_and_count_receipts(monkeypatch, tmp_path):
    ev = matching.evaluation
    name = ev.CANDIDATES[0]
    index = pd.date_range("2022-12-31 23:30", periods=12, freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                        "close": 100., "volume": 1.}, index=index)
    specs = [("inner_val", "spot", index[0], index[5], .004, 1., False),
             ("funded_val", "perp", index[0], index[5], .0005, 1., True)]
    pd.DataFrame([{"strategy": name, "cell": spec[0],
                   "annualized_volatility": target * np.sqrt(365.25)}
                  for spec, target in zip(specs, (.4, 1.5))]).to_csv(tmp_path / "train_cells.csv", index=False)
    monkeypatch.setattr(ev, "OUT", tmp_path)
    monkeypatch.setattr(ev, "CANDIDATES", (name,))
    monkeypatch.setattr(ev, "specifications", lambda _: specs)
    monkeypatch.setattr(ev, "load_ohlcv_csv", lambda _: raw.copy())
    monkeypatch.setattr(ev, "load_funding_deribit", lambda _: pd.Series(.001, index=index))
    seen = []

    def evaluate(control, prepared, spec, funding):
        assert prepared.index[-1] == funding.index[-1] == ev.TRAIN_END
        assert control.deadband == .1 and not control.static
        seen.append((spec[0], control.c))
        vol = control.c * .5
        row = {"strategy": control.name, "cell": spec[0], "asset": spec[1],
               "annualized_volatility": vol * np.sqrt(365.25)}
        daily = pd.DataFrame({"strategy": [control.name], "cell": [spec[0]],
                             "timestamp": [str(index[0])], "return": [vol], "equity": [1000.]})
        return row, daily, None

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

    monkeypatch.setattr(ev, "evaluate", evaluate)
    monkeypatch.setattr(matching, "ProcessPoolExecutor", InlinePool)
    frame = matching.run("train", workers=1)
    assert seen == [("inner_val", .5), ("inner_val", .8),
                    ("funded_val", .5), ("funded_val", 2.), ("funded_val", 2.)]
    assert len(frame) == 5
    final = frame.loc[frame.final_selected].set_index("cell")
    assert final.loc["inner_val", "matched_valid"]
    assert not final.loc["funded_val", "matched_valid"]
    meta = json.loads((tmp_path / "train_matched_meta.json").read_text())
    assert meta["attempts"] == 5 and meta["comparisons"] == 2
    assert meta["valid_matches"] == meta["invalid_matches"] == 1
    assert meta["holdout_consultations"] == 0
    assert meta["attempts_by_asset"] == {"perp": 3, "spot": 2}
    with pytest.raises(RuntimeError, match="overwrite"):
        matching.run("train", workers=1)


def test_matching_holdout_rejects_before_reference_reads(monkeypatch):
    def reject():
        raise RuntimeError("manifest rejected")

    monkeypatch.setattr(matching.evaluation, "validate_manifest", reject)
    monkeypatch.setattr(pd, "read_csv", lambda _: pytest.fail("Read before freeze validation"))
    with pytest.raises(RuntimeError, match="manifest rejected"):
        matching.match_one(matching.evaluation.CANDIDATES[0], "holdout")
    with pytest.raises(RuntimeError, match="manifest rejected"):
        matching.run("holdout")


def _report_fixture(monkeypatch, tmp_path):
    from experiments import r192_report as report
    stages, matches = {}, {}
    dates = pd.date_range("2021-01-01", periods=80, tz="UTC")
    candidate_returns = pd.Series(np.tile([.008, -.006], 40), index=dates)
    reference_returns = pd.Series(np.tile([.004, -.010], 40), index=dates)
    assert candidate_returns.std() == pytest.approx(reference_returns.std())
    for stage in ("train", "holdout"):
        rows, daily = [], {}
        for name, cell in sorted(report.expected_core(stage)):
            candidate = name in evaluation.ALL_NAMES
            rows.append(dict(strategy=name, cell=cell,
                final_balance=1200. if candidate else 1100., daily_sharpe=1.3 if candidate else 1.,
                annualized_volatility=.5, fills=10, fills_per_day=.1, liquidated=False,
                fill_signature=name))
            daily[name, cell] = candidate_returns.copy() if candidate else reference_returns.copy()
        stages[stage] = (pd.DataFrame(rows), daily)
        rows, mdaily = [], {}
        for name in evaluation.CANDIDATES:
            for spec in matching.matching_specs(stage):
                cell = spec[0]
                ref = f"match_{name}_{cell}_i0"
                rows.append(dict(strategy=ref, reference_candidate=name, cell=cell,
                    final_selected=True, matched_valid=True, control_c=.5, relative_vol_error=0.,
                    final_balance=1100.))
                mdaily[ref, cell] = reference_returns.copy()
        failed = rows[0] | dict(strategy="failed_attempt_" + stage, final_selected=False, matched_valid=False)
        rows.append(failed)
        mdaily[failed["strategy"], failed["cell"]] = reference_returns.copy()
        matches[stage] = (pd.DataFrame(rows), mdaily)
    monkeypatch.setattr(report, "OUT", tmp_path)
    monkeypatch.setattr(report, "read", lambda stage, matched=False: matches[stage] if matched else stages[stage])
    monkeypatch.setattr(report, "deflated_sharpe_ratio", lambda *args, **kwargs: 1.)
    (tmp_path / "manifest.json").write_text(json.dumps({"sd_trials": .5}))
    (tmp_path / "audit.json").write_text(json.dumps(_completed_audit()))
    boot = pd.DataFrame([dict(strategy=name, cell=cell, control=control,
        risk_valid=True, d_sharpe=.3, d_sharpe_lo=.1, d_growth_lo=.1,
        d_drawdown=-1., d_drawdown_hi=.1)
        for name in evaluation.CANDIDATES for cell in report.PRIMARY_CELLS
        for control in ("parent", "matched_hold")])
    return report, stages, matches, boot


def _completed_audit():
    return dict(subject="r192_tracking_budget", train_evaluations=2, holdout_evaluations=2,
        discrepancies=[], reproductions=[dict(strategy="r192_tracking_budget", cell=cell,
        stage=stage, status="completed") for cell, stage in (
            ("inner_val", "train"), ("funded_val", "train"),
            ("holdout", "holdout"), ("funded_holdout", "holdout"))])


@pytest.mark.parametrize("failure", ["missing", "wrong_subject", "started", "failed",
    "duplicate", "missing_cell", "extra_cell", "wrong_stage", "wrong_strategy",
    "wrong_count", "discrepancy", "missing_discrepancies"])
def test_report_fails_before_reading_results_without_complete_clean_audit(monkeypatch, tmp_path, failure):
    from experiments import r192_report as report
    monkeypatch.setattr(report, "OUT", tmp_path)
    monkeypatch.setattr(report, "read", lambda *args: pytest.fail("No results before audit validation"))
    audit = _completed_audit()
    if failure == "wrong_subject":
        audit["subject"] = "other"
    elif failure in ("started", "failed"):
        audit["reproductions"][0]["status"] = failure
    elif failure == "duplicate":
        audit["reproductions"][0] = audit["reproductions"][1].copy()
    elif failure == "missing_cell":
        audit["reproductions"].pop()
    elif failure == "extra_cell":
        audit["reproductions"].append(audit["reproductions"][0].copy())
    elif failure == "wrong_stage":
        audit["reproductions"][0]["stage"] = "holdout"
    elif failure == "wrong_strategy":
        audit["reproductions"][0]["strategy"] = "other"
    elif failure == "wrong_count":
        audit["train_evaluations"] = 3
    elif failure == "discrepancy":
        audit["discrepancies"] = [{"cell": "inner_val", "error": "mismatch"}]
    elif failure == "missing_discrepancies":
        del audit["discrepancies"]
    if failure != "missing":
        (tmp_path / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Final report requires"):
        report.decide(pd.DataFrame())


def test_daily_drawdown_includes_initial_capital_for_original_and_bootstrap_paths():
    from experiments import r192_report as report
    assert report.daily_drawdown(np.array([-.05, .02])) == pytest.approx(5.)
    assert report.daily_drawdown(np.array([-.05])) == pytest.approx(5.)
    assert report.daily_drawdown(np.array([])) == 0.
    assert report.daily_drawdown(np.array([[-.05, .02], [.05, -.02]])) == pytest.approx([5., 2.])
    assert dict(report.STATS)["drawdown"] is report.daily_drawdown


def test_report_growth_tail_partition_counts_and_strict_two_percentage_points(monkeypatch, tmp_path):
    report, stages, matches, boot = _report_fixture(monkeypatch, tmp_path)
    _, decisions, counts = report.decide(boot)
    assert decisions.verdict.eq("PROMOTED").all()
    assert decisions.route.eq("growth").all()
    assert decisions.beta_spot_valid.eq(24).all()
    assert counts["core_evaluations"] == 455
    assert counts["matching_evaluations"] == 262
    assert counts["total_evaluations"] == 455 + 262 + 4
    assert counts["holdout_consultations"] == 404 + 251 + 2
    assert counts["cumulative_consultations_approx"] == 3251 + 404 + 251 + 2
    boot["d_sharpe"] = .1
    boot["d_drawdown"] = -3.
    boot["d_drawdown_hi"] = -.1
    boot["d_growth_lo"] = 0.
    _, decisions, _ = report.decide(boot)
    assert decisions.verdict.eq("PROMOTED").all()
    assert decisions.route.eq("tail").all()
    assert decisions.beta_perp_tail_wins.eq(24).all()
    boot.loc[boot.strategy == evaluation.CANDIDATES[0], "d_drawdown"] = -2.
    _, decisions, _ = report.decide(boot)
    assert decisions.set_index("strategy").loc[evaluation.CANDIDATES[0], "verdict"] == "NEGATIVE"


def test_report_requires_binding_risk_and_all_inference_cells(monkeypatch, tmp_path):
    report, stages, matches, boot = _report_fixture(monkeypatch, tmp_path)
    inert, invalid, unstable = evaluation.CANDIDATES[:3]
    train = stages["train"][0]
    train.loc[(train.strategy == inert) & (train.cell == "inner_val"), "fill_signature"] = report.PARENT
    boot.loc[(boot.strategy == invalid) & (boot.cell == "holdout") & (boot.control == "parent"), "risk_valid"] = False
    hold = stages["holdout"][0]
    hold.loc[(hold.strategy == report.FAMILIES[unstable][1]) & (hold.cell == "holdout"), "liquidated"] = True
    _, decisions, _ = report.decide(boot)
    indexed = decisions.set_index("strategy")
    assert not indexed.loc[inert, "mechanism_binding"]
    assert not indexed.loc[invalid, "risk_valid"]
    assert not indexed.loc[unstable, "no_liquidation"]
    assert indexed.loc[[inert, invalid, unstable], "verdict"].eq("NEGATIVE").all()
    with pytest.raises(ValueError, match="Incomplete or duplicate paired"):
        report.decide(boot.iloc[:-1])


def test_training_power_compares_both_controls_and_labels_nonlinear_projection(monkeypatch, tmp_path):
    report, _, _, _ = _report_fixture(monkeypatch, tmp_path)
    result = report.power()
    assert len(result) == 5 * 2 * 2
    assert set(result.control) == {"parent", "matched_hold"}
    assert result.risk_valid.all()
    assert result.sharpe_effect_threshold.eq(.20).all()
    assert result.drawdown_effect_threshold.eq(2.).all()
    assert result.power_limitation.str.contains("nonlinear drawdown").all()
    assert np.isfinite(result.sharpe_required_days_approx).all()
    assert np.isfinite(result.drawdown_required_days_approx).all()
