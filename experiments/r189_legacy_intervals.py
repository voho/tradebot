#!/usr/bin/env python
"""R-189 supplemental intervals under the historical table's holdout convention.

Run ``.venv/bin/python experiments/r189_legacy_intervals.py --workers 4``.
This separately frozen supplement runs 24 fresh $1,000 accounts from
2023-01-01 through 2026-08-12 00:40 UTC: ten candidates and two controls,
spot at 10bp and 5x spot-proxy futures at 5bp, zero slippage, no funding.
It supplies the registered-strategy evidence requirement, not promotion.

Signals retain all earlier causal Bitstamp history, as in the primary R-189
run; account evaluation uses a 100-day prefix and original control on_bar
behavior (no forced initial Kelly order). This differs from the older
incumbent's truncated learning history, so existing incumbent rows are
preserved. The buy-and-hold baseline must reproduce both official Sharpes.
Daily returns retain the historical convention of omitting the first day's
return. The unchanged scripts/inference.py helper produces paired 2,000
stationary resamples with 30-day mean blocks and seed 7. Only the twenty new
holdout intervals are merged into reports/inference/bootstrap.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from r189_games import (CANDIDATES, END, HOLDOUT, NAMES, OUT, PREFIX, ROOT,
                        Prepared)
from tradebot.broker import MarketSpec
from tradebot.data import load_ohlcv_csv
from tradebot.engine import run_backtest
from tradebot.inference import annualized_sharpe, daily_returns
from tradebot.metrics import compute_metrics
from tradebot.registry import get_strategy


OFFICIAL = ROOT / "reports/inference/bootstrap.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    paths = [Path(__file__), ROOT / "experiments/r189_games.py",
             ROOT / "scripts/inference.py", ROOT / "src/tradebot/inference.py",
             ROOT / "src/tradebot/engine.py", ROOT / "src/tradebot/broker.py",
             ROOT / "src/tradebot/strategies/intraday_games.py",
             ROOT / "src/tradebot/strategies/kelly_regime.py",
             ROOT / "src/tradebot/strategies/kelly_regime_v3.py",
             ROOT / "src/tradebot/strategies/kelly_regime_v4.py",
             ROOT / "src/tradebot/strategies/buy_and_hold.py",
             ROOT / "data/btcusd_spot_5m.csv.gz"]
    manifest = {
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol": __doc__, "evaluations": 24,
        "strategies": NAMES, "new_configurations": 0,
        "start": str(HOLDOUT), "end": str(END),
        "primary_protocol_unchanged": True,
        "hashes": {str(p.relative_to(ROOT)): _sha(p) for p in paths},
        "official_bootstrap_before_merge_sha256": _sha(OFFICIAL),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "legacy_manifest.json"
    if path.exists():
        assert json.loads(path.read_text())["hashes"] == manifest["hashes"], (
            "Supplemental source or data changed; version the protocol before rerunning")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def one_strategy(name: str):
    raw = load_ohlcv_csv(ROOT / "data/btcusd_spot_5m.csv.gz").loc[:END]
    strategy = get_strategy(name)
    frame = strategy.prepare(raw.copy())
    lo = int(frame.index.searchsorted(HOLDOUT))
    assert frame.index[lo] == HOLDOUT and frame.index[-1] == END
    prefix = min(lo, PREFIX)
    sub = frame.iloc[lo - prefix:]
    rows, curves = [], {}
    for market_name, market in (("spot", MarketSpec.spot()),
                                ("futures", MarketSpec.futures(leverage=5.0))):
        result = run_backtest(Prepared(strategy), sub, market, 1000.0,
                              slippage_bps=0.0, trade_start=prefix,
                              data_label="real" if market_name == "spot" else "spot (perp proxy)")
        result = replace(result, equity=result.equity.iloc[prefix:],
                         df=result.df.iloc[prefix:])
        metric = compute_metrics(result)
        days = (result.equity.index[-1] - result.equity.index[0]
                + pd.Timedelta(minutes=5)).total_seconds() / 86400
        rows.append(metric.as_row() | {
            "cell": "legacy_holdout", "start": str(HOLDOUT), "end": str(END),
            "days": days, "fee_rate": market.fee_rate, "slippage_bps": 0.0,
            "funding_paid": result.funding_paid, "funding_model": "not charged",
            "fills": len(result.fills), "fills_per_day": len(result.fills) / days,
            "signal_history": "all prior causal Bitstamp data",
        })
        curves[f"{name}|{market_name}"] = daily_returns(result.equity)
        print(f"{name}: legacy {market_name}, ${metric.final_balance:,.2f}, "
              f"{len(result.fills)} fills", flush=True)
    return rows, curves


def run(workers: int) -> None:
    freeze()  # Receipt precedes preparation and every supplemental evaluation.
    rows, series = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_strategy, name) for name in NAMES]
        for future in as_completed(futures):
            cells, curves = future.result()
            rows.extend(cells)
            series.update(curves)
    pd.DataFrame(rows).to_csv(OUT / "legacy_cells.csv", index=False)
    curves = pd.DataFrame(series).sort_index()
    assert len(rows) == 24 and not curves.isna().any().any()
    curves.to_csv(OUT / "legacy_daily_returns.csv.gz")

    original = pd.read_csv(OFFICIAL)
    for market in ("spot", "futures"):
        expected = original[(original.period == "holdout")
                            & (original.market == market)
                            & (original.strategy == "buy_and_hold")].iloc[0]
        actual = annualized_sharpe(curves[f"buy_and_hold|{market}"].to_numpy())
        assert len(curves) == int(expected.days)
        assert np.isclose(actual, expected.sharpe, atol=1e-10, rtol=0), (
            market, "historical benchmark mismatch", actual, expected.sharpe)
        print(f"Verified historical {market} buy-and-hold Sharpe: {actual:.12f}", flush=True)

    spec = importlib.util.spec_from_file_location("r189_legacy_inference",
                                                 ROOT / "scripts/inference.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.OUT = OUT / "legacy_inference"
    intervals = helper.bootstrap({"holdout": curves}, list(NAMES))
    intervals.to_csv(OUT / "legacy_bootstrap.csv", index=False)
    new_rows = intervals[intervals.strategy.isin(CANDIDATES)]
    assert len(new_rows) == 20
    # Preserve every historical and full-period row; replace only this supplement.
    replace_mask = ((original.period == "holdout")
                    & original.strategy.isin(CANDIDATES)
                    & original.market.isin(("spot", "futures")))
    combined = pd.concat([original.loc[~replace_mask], new_rows], ignore_index=True)
    assert not combined.duplicated(["period", "market", "strategy"]).any()
    combined.to_csv(OFFICIAL, index=False)
    print("Merged 20 new historical holdout intervals; existing rows preserved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args().workers)
