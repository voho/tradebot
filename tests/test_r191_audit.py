"""Independent proximal-objective and quote-cash reproduction of fixed R-191 cells.

Pytest runs synthetic checks only. Real-data audits run explicitly via this
file's CLI, after primary receipts exist. The audit subject is fixed at
r191_cost_proximal; every financial replay is counted before it starts.
"""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
OUT = ROOT / "reports/r191_strategies"
SUBJECT = "r191_cost_proximal"
CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")
AUDIT_FILES = ("experiments/r191_strategies.py", "experiments/r191_eval.py",
               "experiments/r191_protocol.md", "tests/test_r191_audit.py")


def prefix_csv(path, cutoff=CUTOFF):
    """Stop before parsing financial values at or after the permitted cutoff."""
    rows = []
    with gzip.open(path, "rt") as stream:
        header = next(stream).strip().split(",")
        for line in stream:
            stamp = line.partition(",")[0]
            timestamp = (pd.Timestamp(int(stamp), unit="ms", tz="UTC")
                         if stamp.isdigit() else pd.Timestamp(stamp))
            if timestamp >= cutoff:
                break
            fields = next(csv.reader([line]))
            rows.append([timestamp] + [float(value) for value in fields[1:]])
    return pd.DataFrame(rows, columns=header).set_index("timestamp")


def independent_proximal(frame):
    """Enumerate objective extrema instead of reusing the production shrinkage rule."""
    stamps, closes = [], []
    for day, group in frame.close.groupby(frame.index.normalize(), sort=True):
        expected = pd.date_range(day, periods=288, freq="5min")
        if group.index.equals(expected):
            stamps.append(expected[-1])
            closes.append(float(group.iloc[-1]))
    close = np.asarray(closes)
    returns = np.r_[np.nan, close[1:] / close[:-1] - 1.]
    weights = np.zeros(len(close))
    prior, cost = 0., 2 * .0041
    for i in range(252, len(close)):
        history = returns[i - 62:i + 1]
        linear = 7 * float(np.mean(history))
        curvature = 28 * float(np.var(history, ddof=0))
        candidates = [prior, 0., 1.]
        if curvature > 0:
            candidates.extend((float(np.clip((linear - cost) / curvature, prior, 1.)),
                               float(np.clip((linear + cost) / curvature, 0., prior))))
        objective = [linear * w - .5 * curvature * w * w - cost * abs(w - prior)
                     for w in candidates]
        prior = candidates[int(np.argmax(objective))]
        weights[i] = prior
    daily_index = pd.DatetimeIndex(stamps)
    # Only actual completion timestamps expose the daily decision; incomplete
    # days never provide a close and never increment the 252-observation clock.
    targets = pd.Series(weights, index=daily_index).reindex(frame.index, method="ffill").fillna(0.)
    decisions = frame.index.isin(daily_index[252:])
    return targets.to_numpy(), decisions


def quote_cash_replay(frame, target, decision, *, leverage, fee, funding=None):
    """Long-only self-financing quote book; no engine, evaluator or broker calls.

    Liquidation paths are intentionally unsupported: explicit checks fail
    loudly rather than claiming reproduction of an unimplemented mechanism.
    """
    quote, qty, average, pending = 1000., 0., 0., None
    fees, funding_paid, requests, completed = 0., 0., 0, 0
    opened, lows, closes = (frame[c].to_numpy() for c in ("open", "low", "close"))
    rates = (np.zeros(len(frame)) if funding is None
             else funding.reindex(frame.index, fill_value=0.).to_numpy())
    equity, exposure, in_market = np.zeros(len(frame)), np.zeros(len(frame)), 0
    fills = []
    for i in range(len(frame)):
        opening, low, close = opened[i], lows[i], closes[i]
        assert qty == 0 or quote + qty * opening > .005 * qty * opening, "Unimplemented open liquidation"
        if pending is not None:
            wealth = quote + qty * opening
            desired = wealth * (1 - (fee + .0001) * leverage) * pending / opening
            change = desired - qty
            suppressed = pending != 0 and qty != 0 and abs(change) * opening < .05 * wealth * leverage
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
                fills.append((frame.index[i], change, price, charge))
        pending = None
        assert qty == 0 or quote + qty * low > .005 * qty * low, "Unimplemented intrabar liquidation"
        payment = rates[i] * qty * close
        quote -= payment
        funding_paid += payment
        assert quote + qty * average >= 0, "Unimplemented funding insolvency"
        wealth = quote + qty * close
        equity[i], exposure[i] = wealth, qty * close / wealth
        in_market += qty != 0
        if i != len(frame) - 1 and decision[i]:
            if abs(target[i] - exposure[i]) > .05 or (target[i] == 0 and qty != 0):
                pending = float(target[i])
                requests += 1
    curve = pd.Series(equity, index=frame.index)
    daily_close = curve.resample("1D").last()
    daily = daily_close / daily_close.shift(fill_value=1000.) - 1
    peaks = np.maximum.accumulate(equity)
    metrics = {
        "final_balance": float(equity[-1]), "fees_paid": float(fees),
        "funding_paid": float(funding_paid), "fills": len(fills),
        "completed_round_trips": int(completed), "decision_requests": int(requests),
        "max_drawdown_pct": float(np.max((peaks - equity) / peaks) * 100),
        "mean_abs_exposure": float(np.abs(exposure).mean()),
        "time_in_market_pct": float(100 * in_market / len(frame)),
        "daily_sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25)),
        "annualized_volatility": float(daily.std(ddof=1) * np.sqrt(365.25)),
    }
    return metrics, curve, fills


def _save(report):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


def _write_report(report):
    lines = ["# R-191 independent audit", "", f"Fixed subject: `{SUBJECT}`.", "",
             "Targets are reconstructed by enumerating the proximal objective's extrema,",
             "using independently identified complete days and NumPy rolling-window moments.",
             "Accounting uses a separate quote-cash book without the production engine or broker.",
             "Synthetic tests do not evaluate historical portfolios. Unsupported liquidation",
             "or funding insolvency paths fail explicitly.", "",
             f"Financial replays: **{report['train_evaluations']} training, {report['holdout_evaluations']} holdout**.", "",
             "| Cell | State | Balance | Max target error | Max daily equity error |",
             "|---|---|---|---|---|"]
    for rec in report["reproductions"]:
        lines.append(f"| {rec['cell']} | {rec['status']} | "
                     f"{rec.get('metrics', {}).get('final_balance', float('nan')):.6f} | "
                     f"{rec.get('target_max_error', float('nan')):.3g} | "
                     f"{rec.get('max_daily_equity_error', float('nan')):.3g} |")
    lines += ["", "Every financial replay is appended and counted before execution. Daily dates,",
              "costs, funding, request/fill/episode counts, realized risk and time in market",
              "must agree with the already-completed primary receipt. Zero discrepancies are",
              "claimed only for completed records; failed or started attempts remain counted.", ""]
    if report.get("post_run_source_edits"):
        lines += ["After the four replays, the audit CLI's input paths were changed to use",
                  "the committed aggregate cell/daily artifacts filtered to the fixed subject,",
                  "instead of ignored per-strategy receipts. Accounting, signal reconstruction",
                  "and thresholds did not change, and no financial replay was repeated.",
                  "The original executed audit-code hash remains on each reproduction record.", ""]
    (OUT / "audit.md").write_text("\n".join(lines))


def audit(stage):
    if stage not in ("train", "holdout"):
        raise ValueError(stage)
    manifest_hash = None
    if stage == "holdout":
        manifest_path = OUT / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest["hashes"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output = OUT / "audit.json"
    report = json.loads(output.read_text()) if output.exists() else {
        "subject": SUBJECT, "train_evaluations": 0, "holdout_evaluations": 0,
        "method": "Independent objective-extrema reconstruction and quote-cash accounting; no engine/broker replay",
        "reproductions": [], "discrepancies": [],
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in AUDIT_FILES},
    }
    assert report["subject"] == SUBJECT
    for relative, expected in report["source_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    specs = (("spot", "btcusd_spot_5m.csv.gz", "inner_val" if stage == "train" else "holdout", 1., .004),
             ("perp", "btcusdt_deribit_perp_5m.csv.gz", "funded_val" if stage == "train" else "funded_holdout", 5., .0005))
    published = pd.read_csv(OUT / f"{stage}_cells.csv")
    published = published.loc[published.strategy == SUBJECT]
    published_daily = pd.read_csv(OUT / f"{stage}_daily.csv.gz")
    published_daily = published_daily.loc[published_daily.strategy == SUBJECT]
    for _, _, cell, _, _ in specs:
        assert (published.cell == cell).sum() == 1, f"Primary cell not complete: {cell}"
        assert not any(r["cell"] == cell for r in report["reproductions"]), f"Replay already counted: {cell}"
    cutoff = CUTOFF if stage == "train" else pd.Timestamp("2026-08-12 00:45", tz="UTC")
    start = pd.Timestamp("2021-01-01", tz="UTC") if stage == "train" else CUTOFF
    funding = prefix_csv(ROOT / "data/btcusdt_deribit_perp_funding_8h.csv.gz", cutoff).funding_rate
    from experiments.r191_strategies import make_strategy
    for kind, filename, cell, leverage, fee in specs:
        frame = prefix_csv(ROOT / "data" / filename, cutoff)
        target, decision = independent_proximal(frame)
        prepared = make_strategy(SUBJECT).prepare(frame.copy())
        target_error = float(np.max(np.abs(target - prepared.target.to_numpy())))
        np.testing.assert_allclose(target, prepared.target, rtol=0, atol=1e-11)
        np.testing.assert_array_equal(decision, prepared.r191_decision)
        # A real-data prefix cut checks that later daily observations cannot
        # alter the reconstructed training/holdout boundary target history.
        prefix = frame.loc[:start - pd.Timedelta(minutes=5)]
        prefix_target, prefix_decision = independent_proximal(prefix)
        np.testing.assert_array_equal(prefix_target, target[:len(prefix)])
        np.testing.assert_array_equal(prefix_decision, decision[:len(prefix)])
        measured = frame.loc[start:]
        expected_funding = pd.date_range(start, measured.index[-1].floor("8h"), freq="8h")
        assert funding.loc[start:].index.equals(expected_funding)
        reported = published.loc[published.cell == cell].iloc[0]
        assert not reported.liquidated
        daily = published_daily.loc[published_daily.cell == cell]
        record = {"strategy": SUBJECT, "cell": cell, "stage": stage, "status": "started",
                  "target_max_error": target_error, "prefix_causality": True,
                  "max_prepared_timestamp": str(frame.index[-1]),
                  "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        report[f"{stage}_evaluations"] += 1
        report["reproductions"].append(record)
        if manifest_hash is not None:
            report["holdout_manifest_hash"] = manifest_hash
        _save(report)
        try:
            metrics, curve, fills = quote_cash_replay(measured, target[-len(measured):], decision[-len(measured):],
                leverage=leverage, fee=fee, funding=funding if kind == "perp" else None)
            errors = {metric: float(value - reported[metric]) for metric, value in metrics.items()}
            daily_curve = curve.resample("1D").last()
            assert daily_curve.index.equals(pd.DatetimeIndex(pd.to_datetime(daily.timestamp, utc=True)))
            daily_error = float(np.max(np.abs(daily_curve.to_numpy() - daily.equity.to_numpy())))
            record.update(metrics=metrics, metric_errors=errors, max_daily_equity_error=daily_error,
                          first_fill=str(fills[0][0]) if fills else None)
            assert daily_error < 1e-7, daily_error
            assert max(map(abs, errors.values())) < 1e-7, errors
            record["status"] = "completed"
        except Exception as error:
            record.update(status="failed", error=repr(error))
            report["discrepancies"].append({"cell": cell, "error": repr(error)})
            raise
        finally:
            _save(report)
            _write_report(report)
        print(SUBJECT, cell, metrics["final_balance"], daily_error, flush=True)
    return report


def test_independent_proximal_uses_only_complete_day_closes_and_matches_objective():
    from experiments.r191_strategies import make_strategy
    index = pd.date_range("2020-01-01", periods=260 * 288 + 50, freq="5min", tz="UTC")
    close = 100 * np.exp(np.arange(len(index)) * .00001 + .01 * np.sin(np.arange(len(index)) / 500))
    frame = pd.DataFrame({"close": close}, index=index)
    target, decision = independent_proximal(frame)
    native = make_strategy(SUBJECT).prepare(frame.copy())
    np.testing.assert_allclose(target, native.target, atol=1e-11, rtol=0)
    np.testing.assert_array_equal(decision, native.r191_decision)
    assert not decision[:252 * 288].any()
    assert decision.sum() == 8
    assert not decision[-50:].any()
    shortened = frame.iloc[:-30]
    cut_target, _ = independent_proximal(shortened)
    np.testing.assert_array_equal(cut_target, target[:-30])
    missing = frame.drop(index[255 * 288 + 100])
    _, missing_decision = independent_proximal(missing)
    assert missing_decision.sum() == 7


def test_independent_book_daily_next_open_cost_funding_and_residual_close():
    index = pd.date_range("2021-01-01 23:50", periods=293, freq="5min", tz="UTC")
    frame = pd.DataFrame({"open": 100., "low": 100., "close": 100.}, index=index)
    target = np.full(len(frame), .5)
    decision = np.zeros(len(frame), dtype=bool)
    decision[1] = decision[289] = True
    target[289:] = 0.
    funding = pd.Series(.001, index=pd.DatetimeIndex([index[2]]))
    metrics, curve, fills = quote_cash_replay(frame, target, decision, leverage=1., fee=.004, funding=funding)
    qty = 1000 * (1 - .0041) * .5 / 100
    expected_fees = qty * (100.01 + 99.99) * .004
    expected_funding = qty * 100 * .001
    expected_final = 1000 - qty * .02 - expected_fees - expected_funding
    assert abs(metrics["final_balance"] - expected_final) < 1e-9
    assert abs(metrics["funding_paid"] - expected_funding) < 1e-12
    assert metrics["completed_round_trips"] == 1
    assert metrics["decision_requests"] == metrics["fills"] == 2
    assert [fill[0] for fill in fills] == [index[2], index[290]]
    assert curve.iloc[:2].eq(1000).all()


if __name__ == "__main__":
    audit(sys.argv[1] if len(sys.argv) > 1 else "train")
