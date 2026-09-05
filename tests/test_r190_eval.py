"""R-190 preparation boundaries, fresh accounts and recorded native execution."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from tradebot.strategy import Strategy

SPEC = importlib.util.spec_from_file_location(
    "r190_eval", Path(__file__).resolve().parents[1] / "experiments/r190_eval.py")
r190 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r190)


def test_battery_counts_costs_and_paired_windows():
    train, holdout = r190.specifications("train"), r190.specifications("holdout")
    assert len(r190.CANDIDATES) == 10
    assert len(r190.NAMES) == 14
    assert len(r190.AUXILIARIES) == 2
    assert len(train) == 3 and len(holdout) == 52
    assert 14 * (len(train) + len(holdout)) + 2 * (3 + 4) == 784
    assert max(spec[3] for spec in train) < r190.HOLDOUT
    cells = {spec[0]: spec for spec in holdout}
    assert cells["holdout"][4] == .004
    assert cells["discount_holdout"][4] == .001
    for k in range(24):
        spot, perp = cells[f"beta_spot_{k:02d}"], cells[f"beta_perp_{k:02d}"]
        assert spot[2:4] == perp[2:4]
        assert spot[4:] == (.004, 1., False)
        assert perp[4:] == (.0005, 1., True)
        start, end = spot[2:4]
        assert r190.HOLDOUT <= start <= end <= r190.END
        assert 120 <= (end - start + pd.Timedelta(minutes=5)) / pd.Timedelta(days=1) <= 365


def test_fresh_cells_use_causal_preparation_next_open_and_exact_funding():
    index = pd.date_range("2023-01-01 07:30", periods=12, freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                        "close": 100., "volume": 1.}, index=index)
    observed = []

    class CausalHold(Strategy):
        name, warmup = "synthetic", 2

        def prepare(self, df):
            df["history"] = df.close.cumsum()
            return df

        def on_bar(self, ctx):
            observed.append((ctx.ts, ctx.bar["history"]))
            if not ctx.in_market:
                ctx.order_target(1.)

    strategy = CausalHold()
    prepared = strategy.prepare(raw.copy())
    spot = ("holdout", "spot", index[3], index[6], 0., 0., False)
    perp = ("funded_holdout", "perp", index[5], index[8], 0., 0., True)
    funding = pd.Series(.001, index=pd.DatetimeIndex([index[6]]))
    a, a_daily, a_result = r190.evaluate(strategy, prepared, spot)
    b, b_daily, b_result = r190.evaluate(strategy, prepared, perp, funding)
    assert [a["final_balance"], b["final_balance"]] == pytest.approx([1000., 995.])
    assert [a["funding_paid"], b["funding_paid"]] == pytest.approx([0., 5.])
    assert a_result.equity.iloc[0] == b_result.equity.iloc[0] == 1000.
    assert a_result.fills[0].ts == index[4]
    assert b_result.fills[0].ts == index[6]
    assert a["fills"] == b["fills"] == 1
    assert a["completed_round_trips"] == b["completed_round_trips"] == 0
    assert a["decision_requests"] == b["decision_requests"] == 1
    assert a["time_in_market_pct"] == b["time_in_market_pct"] == 75.
    assert a["mean_abs_exposure"] == pytest.approx(.75)
    assert b["mean_abs_exposure"] == pytest.approx(3 * 5000 / 995 / 4)
    assert a["min_notional"] == 10. and b["min_notional"] == 5.
    assert a_daily["return"].iloc[0] == 0.
    assert b_daily["return"].iloc[0] == pytest.approx(-.005)
    assert (index[3], 400.) in observed
    assert (index[5], 600.) in observed

    # Trailing observations may enter prepare but cannot affect earlier cells.
    changed = raw.copy()
    changed.loc[index[9]:, ["open", "high", "low", "close"]] *= 10
    changed_prepared = strategy.prepare(changed)
    repeated, _, _ = r190.evaluate(strategy, changed_prepared, perp, funding)
    assert repeated == b
    with pytest.raises(ValueError, match="funding coverage"):
        r190.evaluate(strategy, prepared, perp, funding.iloc[:0])
    with pytest.raises(ValueError, match="price coverage"):
        r190.evaluate(strategy, prepared.iloc[:-4], perp, funding)


def test_native_parent_initialization_keeps_global_warmup_and_later_changes():
    class Parent(Strategy):
        name, warmup = "kelly_regime_v4", 2

        def on_bar(self, ctx):
            if ctx.prev is not None and ctx.bar["target"] != ctx.prev["target"]:
                ctx.order_notional(float(ctx.bar["target"]))

    index = pd.date_range("2023-01-01", periods=8, freq="5min", tz="UTC")
    prepared = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                             "close": 100., "volume": 1., "target": .5}, index=index)
    prepared.loc[index[5]:, "target"] = 0.
    _, _, full = r190.evaluate(Parent(), prepared,
                              ("train", "spot", None, index[-1], 0., 0., False))
    _, _, later = r190.evaluate(Parent(), prepared,
                               ("holdout", "spot", index[3], index[-1], 0., 0., False))
    assert [(fill.ts, fill.side.name) for fill in full.fills] == [
        (index[3], "BUY"), (index[6], "SELL")]
    assert [(fill.ts, fill.side.name) for fill in later.fills] == [
        (index[4], "BUY"), (index[6], "SELL")]
    assert full.equity.iloc[:3].eq(1000.).all()


def test_training_preparation_cannot_see_holdout_and_flushes_cells(monkeypatch, tmp_path):
    index = pd.date_range("2022-12-31 23:30", periods=12, freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                        "close": 100., "volume": 1.}, index=index)
    seen = []

    class Hold(Strategy):
        name, warmup = "test_train", 0

        def prepare(self, df):
            seen.append(df.index[-1])
            return df

        def on_bar(self, ctx):
            if not ctx.in_market:
                ctx.order_target(1.)

    monkeypatch.setattr(r190, "OUT", tmp_path)
    monkeypatch.setattr(r190, "get_strategy", lambda _: Hold())
    monkeypatch.setattr(r190, "load_ohlcv_csv", lambda _: raw.copy())
    monkeypatch.setattr(r190, "load_funding_deribit", lambda _: None)
    monkeypatch.setattr(r190, "specifications", lambda _: [
        ("inner_val", "spot", index[1], index[5], 0., 0., False)])
    rows = r190.run_one("test_train", "train")
    assert seen == [r190.TRAIN_END]
    saved = pd.read_csv(tmp_path / "test_train/train/cells.csv")
    daily = pd.read_csv(tmp_path / "test_train/train/daily.csv.gz")
    assert len(saved) == len(rows) == len(daily) == 1
    assert pd.Timestamp(saved.end.iloc[0]) == r190.TRAIN_END
    assert saved.fills.iloc[0] == 1


def test_holdout_requires_unchanged_manifest_before_loading_data(monkeypatch, tmp_path):
    monkeypatch.setattr(r190, "ROOT", tmp_path)
    monkeypatch.setattr(r190, "OUT", tmp_path)
    monkeypatch.setattr(r190, "load_ohlcv_csv", lambda _: pytest.fail("Must validate before loading"))
    with pytest.raises(RuntimeError, match="Freeze"):
        r190.run_one("buy_and_hold", "holdout")
    source = tmp_path / "source.py"
    source.write_text("frozen")
    manifest = {"hashes": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    r190.validate_manifest()
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="source/data changed"):
        r190.run_one("buy_and_hold", "holdout")
