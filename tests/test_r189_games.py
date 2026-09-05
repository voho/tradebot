"""R-189's prepared replay must preserve causal history and reset accounts."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tradebot.broker import MarketSpec
from tradebot.engine import run_backtest
from tradebot.strategy import Strategy


SPEC = importlib.util.spec_from_file_location(
    "r189_games", Path(__file__).resolve().parents[1] / "experiments/r189_games.py")
r189 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r189)


def test_frozen_battery_pairs_dates_and_keeps_funding_in_covered_years():
    cells = r189.specifications()
    assert len(cells) == 57
    assert len(r189.CANDIDATES) == 10
    assert len(cells) * len(r189.NAMES) == 684
    assert len({c[0] for c in cells}) == len(cells)
    by_name = {c[0]: c for c in cells}
    for k in range(24):
        spot, perp = by_name[f"beta_spot_{k:02d}"], by_name[f"beta_perp_{k:02d}"]
        assert spot[2:4] == perp[2:4]
        start, end = spot[2:4]
        days = (end - start + pd.Timedelta(minutes=5)) / pd.Timedelta(days=1)
        assert 120 <= days <= 365
        assert start >= r189.HOLDOUT and end <= r189.END
    for _, kind, start, _, _, _, funded in cells:
        if funded:
            assert kind == "perp"
            assert start >= pd.Timestamp("2021-01-01", tz="UTC")


def test_prepared_cells_are_fresh_flat_funded_and_future_independent(monkeypatch, tmp_path):
    index = pd.date_range("2023-01-01 07:30", periods=12, freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                        "close": 100., "volume": 1.}, index=index)
    source = {"frame": raw}
    observed = []

    class CausalHold(Strategy):
        name = "audit_candidate"
        warmup = 1

        def prepare(self, df):
            df["history"] = df.close.cumsum()
            return df

        def on_bar(self, ctx):
            observed.append((ctx.ts, ctx.bar["history"]))
            if not ctx.in_market:
                ctx.order_target(1.)

    cells = [
        ("holdout", "spot", index[3], index[6], 0., 0., False),
        ("funded_holdout", "perp", index[5], index[8], 0., 0., True),
    ]
    monkeypatch.setattr(r189, "OUT", tmp_path)
    monkeypatch.setattr(r189, "END", index[-1])
    monkeypatch.setattr(r189, "PREFIX", 2)
    monkeypatch.setattr(r189, "FILES", {"spot": "fake", "perp": "fake"})
    monkeypatch.setattr(r189, "specifications", lambda: cells)
    monkeypatch.setattr(r189, "get_strategy", lambda _: CausalHold())
    monkeypatch.setattr(r189, "load_ohlcv_csv", lambda _: source["frame"].copy())
    funding = pd.Series(.001, index=pd.date_range(
        pd.Timestamp("2021-01-01", tz="UTC"), index[-1].floor("8h"), freq="8h"))
    monkeypatch.setattr(r189, "load_funding_deribit", lambda _: funding)

    result = r189.one_strategy(CausalHold.name)
    assert [r["final_balance"] for r in result] == pytest.approx([1000., 995.])
    assert [r["funding_paid"] for r in result] == pytest.approx([0., 5.])
    assert [r["start"] for r in result] == [str(index[3]), str(index[5])]
    assert [r["end"] for r in result] == [str(index[6]), str(index[8])]
    assert all(r["fills"] == 1 and r["completed_round_trips"] == 0 for r in result)
    assert all(r["fills_per_day"] == pytest.approx(72.) for r in result)
    assert (index[3], 400.) in observed  # Knowledge includes bars before the prefix.
    fills = pd.read_csv(tmp_path / CausalHold.name / "holdout_fills.csv")
    assert pd.Timestamp(fills.timestamp.iloc[0]) == index[4]  # Next open, never prefix.

    # prepare sees the full source, but future observations cannot alter either cell.
    source["frame"] = raw.copy()
    source["frame"].loc[index[9]:, ["open", "high", "low", "close"]] *= 10
    repeated = r189.one_strategy(CausalHold.name)
    assert repeated == result
    monkeypatch.setattr(r189, "load_funding_deribit", lambda _: funding.iloc[1:])
    with pytest.raises(AssertionError, match="funding coverage"):
        r189.one_strategy(CausalHold.name)


def test_prepared_kelly_initializes_a_fresh_account_without_changing_full_replay():
    class TargetChangesOnly(Strategy):
        name = "kelly_regime_v4"
        warmup = 0

        def on_bar(self, ctx):
            if ctx.prev is not None and ctx.bar["target"] != ctx.prev["target"]:
                ctx.order_notional(float(ctx.bar["target"]))

    index = pd.date_range("2023-01-01", periods=8, freq="5min", tz="UTC")
    prepared = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
        "close": 100., "volume": 1., "target": .5}, index=index)
    market = MarketSpec.spot(fee_rate=0.)
    initial = run_backtest(r189.Prepared(TargetChangesOnly(), initial_bar=3),
        prepared, market, 1000., trade_start=3)
    assert len(initial.fills) == 1
    assert initial.fills[0].ts == index[4]
    assert initial.fills[0].qty == pytest.approx(5.)
    assert initial.equity.iloc[:4].eq(1000.).all()
    unchanged = run_backtest(r189.Prepared(TargetChangesOnly()), prepared, market, 1000.)
    assert unchanged.fills == []
