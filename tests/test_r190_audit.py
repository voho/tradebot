"""Independent quote-cash reproduction; real-data audit runs only via __main__."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")


def prefix_csv(path, cutoff=CUTOFF):
    """Stop at the cutoff timestamp before parsing any later financial values."""
    rows = []
    with gzip.open(path, "rt") as stream:
        header = next(stream).strip().split(",")
        for line in stream:
            stamp, _, _ = line.partition(",")
            timestamp = (pd.Timestamp(int(stamp), unit="ms", tz="UTC")
                         if stamp.isdigit() else pd.Timestamp(stamp))
            if timestamp >= cutoff:
                break
            values = next(csv.reader([line]))
            rows.append([timestamp] + [float(value) for value in values[1:]])
    return pd.DataFrame(rows, columns=header).set_index("timestamp")


def independent_v4(close, horizons=(20, 40, 80)):
    """Re-derive a shipped conditional Kelly target from fixed constants."""
    votes = []
    for days in horizons:
        average = close.rolling(days * 288).mean()
        vote = pd.Series(np.nan, index=close.index)
        vote[close > 1.01 * average] = 1.
        vote[close < .99 * average] = 0.
        votes.append(vote.ffill().fillna(0.))
    fraction = sum(votes).to_numpy() / 3
    risk = np.log(close).diff().ewm(span=8 * 288, min_periods=288).std()
    risk = (risk * np.sqrt(365.25 * 288)).shift().to_numpy()
    slow = pd.Series(risk).ewm(span=180 * 288, min_periods=288).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = risk / slow
        extreme = np.minimum(.55 / risk, 2.)
        normal = np.minimum(.55 / slow, 2.)
    extreme[~np.isfinite(extreme)] = 0.
    normal[~np.isfinite(normal)] = 0.
    target, current, state = np.zeros(len(close)), 0., 0
    for i in range(len(close)):
        if np.isfinite(ratio[i]):
            if state == 0:
                state = 1 if ratio[i] > 1.7 else -1 if ratio[i] < .55 else 0
            elif state == 1 and ratio[i] < 1.2:
                state = 0
            elif state == -1 and ratio[i] > .85:
                state = 0
        desired = fraction[i] * (extreme[i] if state else normal[i])
        if abs(desired - current) > .1:
            current = desired
        target[i] = current
    return target


def quote_cash_replay(frame, target, *, candidate, leverage, fee, funding=None, band=.1):
    """Long-only equivalent book, without engine, broker or experiment helpers.

    The checked real cells must not liquidate; an independent maintenance
    check raises rather than pretending to reproduce an unimplemented path.
    """
    quote, qty, average, pending = 1000., 0., 0., None
    fees, funding_paid, requests, completed = 0., 0., 0, 0
    equity, exposure, fills = [], [], []
    funding = {} if funding is None else funding.to_dict()
    for i, (stamp, row) in enumerate(frame.iterrows()):
        opening = float(row.open)
        assert qty == 0 or quote + qty * opening > .005 * qty * opening
        if pending is not None:
            wealth = quote + qty * opening
            desired = wealth * (1 - (fee + .0001) * leverage)
            desired *= min(leverage, max(0., pending)) / opening
            change = desired - qty
            suppressed = (pending != 0 and qty != 0
                          and abs(change) * opening < .05 * wealth * leverage)
            small = change > 0 and change * opening < (10. if leverage == 1 else 5.)
            if not suppressed and not small and abs(change) >= 1e-12:
                price = opening * (1.0001 if change > 0 else .9999)
                charge = fee * abs(change) * price
                old_qty = qty
                quote -= change * price + charge
                fees += charge
                qty += change
                if abs(qty) < 1e-12:
                    qty = 0.
                if qty > old_qty:
                    average = (average * old_qty + price * change) / qty
                if qty == 0:
                    average = 0.
                    completed += old_qty > 0
                fills.append((stamp, change, price, charge))
        pending = None
        assert qty == 0 or quote + qty * row.low > .005 * qty * row.low
        if stamp in funding:
            payment = funding[stamp] * qty * row.close
            quote -= payment
            funding_paid += payment
            assert quote + qty * average >= 0  # Native funding insolvency rule.
        wealth = quote + qty * row.close
        equity.append(wealth)
        exposure.append(qty * row.close / wealth)
        if i == len(frame) - 1:
            continue
        if candidate:
            scheduled = stamp.value % pd.Timedelta(hours=4).value == 0
            if scheduled and (abs(target[i] - exposure[-1]) > band
                              or (target[i] == 0 and qty != 0)):
                pending = target[i]
        elif (i == 0 and target[i] > 0) or (i > 0 and abs(target[i] - target[i - 1]) > 1e-9):
            pending = target[i]
        requests += pending is not None
    curve = pd.Series(equity, index=frame.index)
    closes = curve.resample("1D").last()
    daily = closes / closes.shift(fill_value=1000.) - 1
    peaks = curve.cummax()
    return {
        "final_balance": float(curve.iloc[-1]), "fees_paid": float(fees),
        "funding_paid": float(funding_paid), "fills": len(fills),
        "completed_round_trips": int(completed), "decision_requests": int(requests),
        "max_drawdown_pct": float(((peaks - curve) / peaks).max() * 100),
        "mean_abs_exposure": float(np.mean(exposure)),
        "daily_sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25)),
        "annualized_volatility": float(daily.std(ddof=1) * np.sqrt(365.25)),
    }, curve, fills


def test_independent_quote_book_cost_and_next_open():
    index = pd.date_range("2021-01-01", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                          "close": 100.}, index=index)
    metrics, equity, fills = quote_cash_replay(
        frame, np.full(4, .5), candidate=True, leverage=1., fee=.004)
    quantity = 1000 * (1 - .0041) * .5 / 100
    cost = quantity * 100 * .0001 + quantity * 100.01 * .004
    assert abs(metrics["final_balance"] - (1000 - cost)) < 1e-10
    assert fills[0][0] == index[1]
    assert equity.iloc[0] == 1000.
    assert metrics["fills"] == metrics["decision_requests"] == 1


def audit_train():
    from experiments.r190_variations import make_strategy
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    output = ROOT / "reports/r190_variations"
    published = pd.read_csv(output / "train_cells.csv")
    published_daily = pd.read_csv(output / "train_daily.csv.gz")
    assert len(published) == 48
    assert not published.duplicated(["strategy", "cell"]).any()
    assert published.groupby("strategy").size().eq(3).all()
    assert pd.to_datetime(published.end, utc=True).max() < CUTOFF
    funding = prefix_csv(ROOT / "data/btcusdt_deribit_perp_funding_8h.csv.gz").funding_rate
    records, preparation = [], []
    for kind, filename, cell, leverage, fee in (
        ("spot", "btcusd_spot_5m.csv.gz", "inner_val", 1., .004),
        ("perp", "btcusdt_deribit_perp_5m.csv.gz", "funded_val", 5., .0005),
    ):
        frame = prefix_csv(ROOT / "data" / filename)
        target = independent_v4(frame.close)
        parent = KellyRegimeV4().prepare(frame.copy()).target.to_numpy()
        candidate = make_strategy("r190_v4_b10").prepare(frame.copy()).target.to_numpy()
        np.testing.assert_array_equal(target, parent)
        np.testing.assert_array_equal(target, candidate)
        # Real prefix causality across the inner-train/validation boundary.
        prefix = frame.loc[:"2020-12-31 23:55"]
        np.testing.assert_array_equal(independent_v4(prefix.close), target[:len(prefix)])
        measured = frame.loc["2021-01-01":]
        own_target = target[-len(measured):]
        assert measured.index[-1] == CUTOFF - pd.Timedelta(minutes=5)
        expected_funding = pd.date_range(measured.index[0], measured.index[-1].floor("8h"), freq="8h")
        assert funding.loc[measured.index[0]:].index.equals(expected_funding)
        preparation.append({"asset": kind, "max_prepared_timestamp": str(frame.index[-1]),
                            "parent_target_max_error": float(np.max(np.abs(target - parent)))})
        for name, variation in (("r190_v4_b10", True), ("kelly_regime_v4", False)):
            independent, curve, fills = quote_cash_replay(
                measured, own_target, candidate=variation, leverage=leverage, fee=fee,
                funding=funding if kind == "perp" else None)
            reported = published[(published.strategy == name) & (published.cell == cell)].iloc[0]
            errors = {metric: float(value - reported[metric]) for metric, value in independent.items()}
            daily = published_daily[(published_daily.strategy == name) & (published_daily.cell == cell)]
            expected = curve.resample("1D").last().to_numpy()
            daily_error = float(np.max(np.abs(expected - daily.equity.to_numpy())))
            assert daily_error < 1e-7, daily_error
            assert max(map(abs, errors.values())) < 1e-7, errors
            records.append({"strategy": name, "cell": cell, "metrics": independent,
                            "metric_errors": errors, "max_daily_equity_error": daily_error,
                            "first_fill": str(fills[0][0]) if fills else None})
            print(name, cell, independent["final_balance"], daily_error, flush=True)
    report = {
        "train_evaluations": len(records), "holdout_evaluations": 0,
        "method": "Independent parent factor reconstruction and quote-cash book; no engine/evaluator/broker replay",
        "primary_train_cells": 48, "preparation": preparation, "reproductions": records,
        "discrepancies": [],
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ("experiments/r190_variations.py", "experiments/r190_eval.py", "experiments/r190_protocol.md")},
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


def audit_holdout():
    """Four explicitly authorized post-freeze reproductions of the training lead."""
    from experiments.r190_variations import make_strategy
    from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3

    output = ROOT / "reports/r190_variations"
    report = json.loads((output / "audit.json").read_text())
    assert report["holdout_evaluations"] == 0, "Do not silently duplicate holdout audits"
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected_hash in manifest["hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash, name
    assert manifest["validation_lead"] == "r190_v3_b20"
    cutoff = pd.Timestamp("2026-08-12 00:45", tz="UTC")
    funding = prefix_csv(ROOT / "data/btcusdt_deribit_perp_funding_8h.csv.gz", cutoff).funding_rate
    for kind, filename, cell, leverage, fee in (
        ("spot", "btcusd_spot_5m.csv.gz", "holdout", 1., .004),
        ("perp", "btcusdt_deribit_perp_5m.csv.gz", "funded_holdout", 5., .0005),
    ):
        frame = prefix_csv(ROOT / "data" / filename, cutoff)
        target = independent_v4(frame.close, (30, 50, 100))
        parent = KellyRegimeV3().prepare(frame.copy()).target.to_numpy()
        candidate = make_strategy("r190_v3_b20").prepare(frame.copy()).target.to_numpy()
        np.testing.assert_array_equal(target, parent)
        np.testing.assert_array_equal(target, candidate)
        measured = frame.loc[CUTOFF:]
        own_target = target[-len(measured):]
        assert measured.index[-1] == cutoff - pd.Timedelta(minutes=5)
        expected_funding = pd.date_range(measured.index[0], measured.index[-1].floor("8h"), freq="8h")
        assert funding.loc[measured.index[0]:].index.equals(expected_funding)
        for name, variation in (("r190_v3_b20", True), ("kelly_regime_v3", False)):
            # Read already-written primary cells before starting another economic run.
            source = output / name / "holdout"
            published = pd.read_csv(source / "cells.csv")
            published_daily = pd.read_csv(source / "daily.csv.gz")
            reported = published.loc[published.cell == cell].iloc[0]
            daily = published_daily.loc[published_daily.cell == cell]
            report["holdout_evaluations"] += 1
            (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            independent, curve, fills = quote_cash_replay(
                measured, own_target, candidate=variation, leverage=leverage, fee=fee,
                funding=funding if kind == "perp" else None, band=.2)
            errors = {metric: float(value - reported[metric]) for metric, value in independent.items()}
            daily_error = float(np.max(np.abs(curve.resample("1D").last().to_numpy() - daily.equity.to_numpy())))
            assert daily_error < 1e-7, daily_error
            assert max(map(abs, errors.values())) < 1e-7, errors
            report["reproductions"].append({"strategy": name, "cell": cell, "metrics": independent,
                "metric_errors": errors, "max_daily_equity_error": daily_error,
                "first_fill": str(fills[0][0]) if fills else None})
            (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, cell, independent["final_balance"], daily_error, flush=True)
    report["holdout_manifest_hash"] = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


def audit_causality():
    """Real pre-2023 target probes; no portfolio or financial cells evaluated."""
    from experiments.r190_variations import CONFIGS, PARENTS, make_strategy

    frame = prefix_csv(ROOT / "data/btcusd_spot_5m.csv.gz").iloc[:120_000]
    cut = 95_040  # UTC midnight after every parent's full warmup.
    changed = frame.copy()
    changed.iloc[cut + 1:, :4] *= 3.
    expected = {name: cls().prepare(frame.copy()).target.to_numpy() for name, cls in PARENTS.items()}
    rows = []
    for name, parent, _ in CONFIGS:
        strategy = make_strategy(name)
        original = strategy.prepare(frame.copy())
        tampered = strategy.prepare(changed.copy())
        truncated = make_strategy(name).prepare(frame.iloc[:cut + 1].copy())
        wanted = np.mean(list(expected.values()), axis=0) if parent == "blend" else expected[parent]
        np.testing.assert_array_equal(original.target, wanted)
        for column in ("target", "r190_decision"):
            np.testing.assert_array_equal(original[column].iloc[:cut + 1], tampered[column].iloc[:cut + 1])
            np.testing.assert_array_equal(original[column].iloc[:cut + 1], truncated[column])
        assert original.r190_decision.iloc[cut] and cut > strategy.warmup
        rows.append({"strategy": name, "parent_identity": True,
                     "future_perturbation": True, "prefix_identity": True})
    output = ROOT / "reports/r190_variations/audit.json"
    report = json.loads(output.read_text())
    report["real_training_causality"] = {
        "max_timestamp": str(frame.index[-1]), "cut_timestamp": str(frame.index[cut]),
        "financial_evaluations": 0, "candidates": rows}
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": audit_train, "holdout": audit_holdout, "causality": audit_causality}[command]()
