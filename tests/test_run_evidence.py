"""Cached decade intervals must not annotate newly fetched one-year runs."""

from dataclasses import replace

import pandas as pd
import pytest

from conftest import make_ohlcv
from test_evidence import _row, _write
from tradebot import run


@pytest.fixture
def cached(tmp_path):
    df = make_ohlcv([100.] * (3 * 288), start="2020-01-01")
    datasets = {"spot": (df, "real"), "perp": (df, "spot (perp proxy)")}
    cfg = run.RunConfig(out_dir=tmp_path, readme=tmp_path / "README.md",
                        strategies=["buy_and_hold", "kelly_regime_v4"])
    (tmp_path / "comparison.md").write_text(
        f"Period: {run._period_label(df)}  \nData: real, spot (perp proxy)  \n")
    _write(tmp_path, [_row(name, market) | {"days": 2}
                     for market in ("spot", "futures") for name in cfg.strategies])
    pd.DataFrame([{"period": "full", "market": market, "distinguishable": False}
                  for market in ("spot", "futures")]).to_csv(
                      tmp_path / "inference/ordering.csv", index=False)
    return cfg, datasets


def test_matching_historical_run_keeps_intervals_and_ordering(cached):
    cfg, datasets = cached
    evidence, ordering = run._matching_evidence(cfg, datasets)
    assert len(evidence) == 4
    assert ordering == {"spot": (0, 1), "futures_5x": (0, 1)}
    # A partial comparison has no cached adjacent order for its own roster.
    _, ordering = run._matching_evidence(replace(cfg, strategies=["buy_and_hold"]), datasets)
    assert ordering == {}


@pytest.mark.parametrize("change", [
    {"slippage_bps": 1.}, {"spot_fee": .002}, {"futures_fee": .001},
    {"leverage": 2.}, {"max_bars": 100},
])
def test_cost_or_trim_changes_refuse_historical_evidence(cached, change):
    cfg, datasets = cached
    assert run._matching_evidence(replace(cfg, **change), datasets) == ({}, {})


def test_each_market_must_match_the_published_dates_and_source(cached):
    cfg, datasets = cached
    df, _ = datasets["perp"]
    changed = datasets | {"perp": (df.iloc[288:], "spot (perp proxy)")}
    assert run._matching_evidence(cfg, changed) == ({}, {})
    # Identical dates cannot make authentic perp prices the old spot proxy.
    assert run._matching_evidence(cfg, datasets | {"perp": (df, "real")}) == ({}, {})
    (cfg.out_dir / "comparison.md").unlink()
    assert run._matching_evidence(cfg, datasets) == ({}, {})


def test_daily_count_still_rejects_stale_bootstrap_after_report_is_overwritten(cached):
    cfg, datasets = cached
    path = cfg.out_dir / "inference/bootstrap.csv"
    old = pd.read_csv(path)
    old.loc[old.market == "futures", "days"] = 3510
    old.to_csv(path, index=False)
    assert run._matching_evidence(cfg, datasets) == ({}, {})


def test_short_resolver_run_never_reuses_old_intervals_on_repeated_report(cached, monkeypatch):
    cfg, _ = cached
    short = make_ohlcv([100.] * 20, start="2025-08-11 19:00")
    monkeypatch.setattr(run.datamod, "load_dataset", lambda *_: (short, "real"))
    monkeypatch.setattr(run, "run_chart", lambda *_: None)
    monkeypatch.setattr(run, "overlay_chart", lambda *_: None)
    cfg = replace(cfg, markets=["spot"], strategies=["buy_and_hold"])
    # First call overwrites the period metadata; the second must still refuse
    # the old bootstrap by its daily count, rather than bless that stale file.
    for _ in range(2):
        metrics, _ = run.run_matrix(cfg)
        assert len(metrics) == 1
        report = (cfg.out_dir / "comparison.md").read_text()
        assert "2025-08-11" in report
        assert "growth vs hold (spot)" not in report
        assert "Adjacent steps" not in report
