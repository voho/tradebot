#!/usr/bin/env python
"""Beta-test harness: does a candidate variant actually beat the baseline?

A variant is only interesting if it survives more than one number. This
runs every candidate through the same battery and prints them side by
side against the incumbent:

1. **Full period** — the headline, 2017-2026.
2. **Walk-forward** — in-sample 2017-2022 (contains two multi-year bears)
   vs out-of-sample 2023-2026 (a steady bull). A variant that only wins
   in-sample is curve-fitted.
3. **Monte Carlo windows** — N random windows (random start and length),
   each with a warmup prefix, evaluated only over the window. Reports the
   median, the fraction of windows the variant beats the incumbent, the
   worst window, and liquidations.

Verdict rules applied at the end: a candidate is promoted only if it
beats the incumbent on the full period AND does not degrade
out-of-sample AND wins the median Monte Carlo window. Anything else is
reported as "no better" — which is the common and honest outcome.

Usage::

    python scripts/beta_test.py                # baseline sanity check
    python scripts/beta_test.py --windows 30   # more Monte Carlo windows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402

BARS_PER_DAY = 288
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

DF, LABEL = load_dataset(ROOT / "data", "spot")


def _full(make: callable, df: pd.DataFrame, market: MarketSpec) -> dict:
    m = compute_metrics(run_backtest(make(), df, market, 1_000.0, data_label=LABEL))
    return {"final": m.final_balance, "sharpe": m.sharpe,
            "dd": m.max_drawdown_pct, "trades": m.num_trades,
            "liq": m.liquidated}


def _window(make: callable, window: pd.DataFrame, eval_start: int,
            market: MarketSpec) -> dict:
    result = run_backtest(make(), window, market, 1_000.0, data_label=LABEL)
    equity = result.equity.to_numpy(dtype=float)
    base = equity[eval_start]
    if not np.isfinite(base) or base <= 0:
        return {"return_pct": -100.0, "dd": 100.0, "liq": True}
    seg = equity[eval_start:]
    return {"return_pct": 100.0 * (seg[-1] / base - 1.0),
            "dd": max_drawdown_pct(seg), "liq": result.liquidated}


def beta_test(candidates: dict[str, callable], market: MarketSpec = FUTURES,
              windows: int = 20, min_days: int = 120, max_days: int = 730,
              seed: int = 7, incumbent: str = "kelly_regime") -> pd.DataFrame:
    ins, oos = DF.loc[:"2022-12-31"], DF.loc["2023-01-01":]

    print(f"=== full period · {market.name} · $1,000 ===", file=sys.stderr)
    rows = {}
    for name, make in candidates.items():
        rows[name] = {"full": _full(make, DF, market),
                      "is": _full(make, ins, market),
                      "oos": _full(make, oos, market)}
        f, i, o = rows[name]["full"], rows[name]["is"], rows[name]["oos"]
        print(f"{name:26s} full=${f['final']:>10,.0f} sharpe={f['sharpe']:5.2f} "
              f"DD={f['dd']:5.1f}% trades={f['trades']:4d} | "
              f"IS=${i['final']:>9,.0f} OOS=${o['final']:>8,.0f}", file=sys.stderr)

    # ---- Monte Carlo windows, identical windows for every candidate
    warmup = max(make().warmup for make in candidates.values()) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(windows):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(warmup, len(DF) - length))
        specs.append((start, length))

    mc = {name: [] for name in candidates}
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        print(f"  window {k}/{windows} {window.index[warmup]:%Y-%m-%d} "
              f"+{length // BARS_PER_DAY}d", file=sys.stderr)
        for name, make in candidates.items():
            mc[name].append(_window(make, window, warmup, market))

    out = []
    base_returns = np.array([w["return_pct"] for w in mc[incumbent]]) \
        if incumbent in mc else None
    for name in candidates:
        rets = np.array([w["return_pct"] for w in mc[name]])
        dds = np.array([w["dd"] for w in mc[name]])
        liq = np.mean([w["liq"] for w in mc[name]]) * 100.0
        row = {
            "strategy": name,
            "full $": rows[name]["full"]["final"],
            "sharpe": rows[name]["full"]["sharpe"],
            "maxDD %": rows[name]["full"]["dd"],
            "trades": rows[name]["full"]["trades"],
            "IS $": rows[name]["is"]["final"],
            "OOS $": rows[name]["oos"]["final"],
            "mc median %": np.median(rets),
            "mc worst %": rets.min(),
            "mc medDD %": np.median(dds),
            "mc liq %": liq,
        }
        if base_returns is not None:
            row["beats base %"] = float((rets > base_returns).mean() * 100.0)
        out.append(row)
    return pd.DataFrame(out)


def verdicts(table: pd.DataFrame, incumbent: str = "kelly_regime") -> None:
    if incumbent not in set(table["strategy"]):
        return
    base = table[table.strategy == incumbent].iloc[0]
    print("\n=== verdicts (promote only if it wins everywhere that matters) ===")
    for _, r in table.iterrows():
        if r["strategy"] == incumbent:
            print(f"{r['strategy']:26s} INCUMBENT")
            continue
        checks = {
            "full": r["full $"] > base["full $"],
            "oos": r["OOS $"] >= base["OOS $"] * 0.98,
            "mc": r.get("beats base %", 0) > 50.0,
            "no-liq": r["mc liq %"] <= base["mc liq %"],
        }
        verdict = "PROMOTE" if all(checks.values()) else "no better"
        failed = ",".join(k for k, ok in checks.items() if not ok)
        print(f"{r['strategy']:26s} {verdict:10s}"
              + (f"  (fails: {failed})" if failed else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--market", choices=["spot", "futures"], default="futures")
    args = ap.parse_args()

    from tradebot.registry import get_strategy

    candidates: dict[str, callable] = {"kelly_regime": lambda: get_strategy("kelly_regime")}
    # Variants under test. Add a registered name here to put it through the
    # battery; the incumbent stays first so verdicts compare against it.
    for name in ("kelly_regime_v2", "kelly_regime_v3"):
        try:
            get_strategy(name)
        except KeyError:
            continue
        candidates[name] = (lambda n=name: get_strategy(n))

    table = beta_test(candidates, market=FUTURES if args.market == "futures" else SPOT,
                      windows=args.windows)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n" + table.round(2).to_string(index=False))
    verdicts(table)


if __name__ == "__main__":
    main()
