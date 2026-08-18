#!/usr/bin/env python
"""R-33 holdout evaluation: funding gate on kelly_regime_v4 (backlog B-05).

Pre-registered in docs/LEDGER.md (section "R-33 pre-registration") BEFORE
this script read any bar dated 2023-01-01 or later. Frozen configs, the
falsification design and the decision rules all come from that commit;
nothing here is re-tuned after looking.

The holdout is 2023-01-01..2023-12-31 ONLY, not the project's usual
open-ended 2023+ split - bounded by the committed funding file's own
range (2020-01-01..2023-12-31), which is the only span either gate can
act in at all.

Usage::

    python experiments/run_funding_gate_eval.py holdout      # point estimates
    python experiments/run_funding_gate_eval.py interval     # paired bootstrap, D1
    python experiments/run_funding_gate_eval.py falsify       # 40-window Monte Carlo
    python experiments/run_funding_gate_eval.py feetier       # 0.40% Bitstamp tier
    python experiments/run_funding_gate_eval.py all
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (annualized_sharpe, daily_returns,  # noqa: E402
                                max_drawdown_from_returns, paired_bootstrap,
                                stationary_bootstrap_indices, total_log_return)
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

from experiments.funding_gate_conservative import FundingGateConservative  # noqa: E402
from experiments.funding_gate_novel import FundingGateNovel  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()
BARS_PER_DAY = 288

HOLDOUT_START = "2023-01-01"
HOLDOUT_END = "2023-12-31"          # funding data ends here - not open-ended
FUNDING_LO = "2020-01-01"
FUNDING_HI = "2023-12-31"

# Frozen at pre-registration, on inner-validation only.
VARIANTS = {
    "kelly_regime_v4": lambda: get_strategy("kelly_regime_v4"),
    "funding_gate_conservative": lambda: FundingGateConservative(
        funding=REAL, lookback_days=90, decile=0.90),
    "funding_gate_novel": lambda: FundingGateNovel(
        funding=REAL, lookback_days=60, pct_start=0.75, pct_full=0.97,
        mom_days=3.0, mom_scale=0.015, max_dampen=0.6),
}
GATED = ("funding_gate_conservative", "funding_gate_novel")


def _period(strategy, market, start=None, end=None, funding=None):
    """Fresh account over [start, end], warmed on bars before it (funding_study.py pattern)."""
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    return (raw if pre == 0 else
            replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))


def holdout() -> None:
    """Point estimates on the 2023 holdout, both markets, funding charged on futures."""
    print(f"HOLDOUT {HOLDOUT_START} .. {HOLDOUT_END}  (bounded by the committed "
          f"funding file's own range)\n")
    rows = []
    for mname, market, funding in (("spot", SPOT, None), ("futures_5x", FUTURES, REAL)):
        print(f"-- {mname} --")
        for name in ("buy_and_hold", *VARIANTS):
            strategy = VARIANTS.get(name, lambda n=name: get_strategy(n))()
            res = _period(strategy, market, HOLDOUT_START, HOLDOUT_END, funding=funding)
            m = compute_metrics(res)
            print(f"  {name:28s} final=${m.final_balance:>10,.0f} "
                  f"({m.profit_pct:>+7.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
                  f"fees=${m.fees_paid:>8,.0f} funding=${res.funding_paid:>8,.0f} "
                  f"{'LIQUIDATED' if m.liquidated else ''}")
            rows.append({"market": mname, "strategy": name,
                         "final": m.final_balance, "dd": m.max_drawdown_pct,
                         "sharpe": m.sharpe, "trades": m.num_trades,
                         "fees": m.fees_paid, "funding_paid": res.funding_paid,
                         "liquidated": m.liquidated})
    out = ROOT / "reports" / "funding_gate"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "holdout.csv", index=False)
    print(f"\nwritten: {out / 'holdout.csv'}")


def _holdout_curve(name: str, market: MarketSpec, funding) -> np.ndarray:
    strategy = VARIANTS.get(name, lambda n=name: get_strategy(n))()
    res = _period(strategy, market, HOLDOUT_START, HOLDOUT_END, funding=funding)
    return daily_returns(res.equity).to_numpy()


def interval() -> None:
    """D1: paired stationary block bootstrap on the 2023 holdout daily returns."""
    print(f"D1 - paired block bootstrap, 2023 holdout ({HOLDOUT_START}..{HOLDOUT_END})\n"
          "30-day mean block, 2,000 resamples, identical resample indices per market.\n")
    rows = []
    for mname, market, funding in (("spot", SPOT, None), ("futures_5x", FUTURES, REAL)):
        curves = {name: _holdout_curve(name, market, funding)
                  for name in ("buy_and_hold", *VARIANTS)}
        n = len(curves["kelly_regime_v4"])
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
        print(f"-- {mname} ({n} daily obs) --")
        pairs = [(g, "kelly_regime_v4") for g in GATED]
        pairs.append(("funding_gate_novel", "funding_gate_conservative"))
        pairs += [(g, "buy_and_hold") for g in GATED]
        pairs.append(("kelly_regime_v4", "buy_and_hold"))
        for a_name, b_name in pairs:
            a, b = curves[a_name], curves[b_name]
            for stat_name, stat in (("Δ log growth", total_log_return),
                                    ("Δ max drawdown (pp)", max_drawdown_from_returns),
                                    ("Δ Sharpe", annualized_sharpe)):
                r = paired_bootstrap(a, b, stat, indices=idx)
                mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
                print(f"  {a_name:26s} - {b_name:22s} {stat_name:20s} "
                      f"{mark} {r.diff.point:>+8.3f} "
                      f"[{r.diff.lo:>+8.3f}, {r.diff.hi:>+8.3f}]  P(>0)={r.p_positive:.2f}")
                rows.append({"market": mname, "a": a_name, "b": b_name,
                             "stat": stat_name, "diff": r.diff.point,
                             "lo": r.diff.lo, "hi": r.diff.hi,
                             "p_positive": r.p_positive, "significant": r.significant})
        print()
    out = ROOT / "reports" / "funding_gate"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "interval.csv", index=False)
    print(f"written: {out / 'interval.csv'}")


def falsify(trials: int = 40, min_days: int = 20, max_days: int = 180, seed: int = 19) -> None:
    """40-window Monte Carlo, windows drawn only from the funding-covered span."""
    warmup = max(get_strategy("kelly_regime_v4").warmup, FundingGateConservative().warmup) + 10
    lo_pos = int(DF.index.searchsorted(FUNDING_LO))
    hi_pos = int(DF.index.searchsorted(FUNDING_HI, side="right"))
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        lo_start = max(lo_pos, warmup)
        hi_start = hi_pos - length
        if hi_start <= lo_start:
            continue
        start = int(rng.integers(lo_start, hi_start))
        specs.append((start, length))
    print(f"falsification: {len(specs)} windows inside {FUNDING_LO}..{FUNDING_HI}, "
          f"{min_days}-{max_days}d each\n")

    rows = []
    for market_name, market in (("spot", SPOT), ("futures_5x", FUTURES)):
        for k, (start, length) in enumerate(specs, 1):
            eval_start = min(start, warmup)
            window = DF.iloc[start - eval_start: start + length]
            for gate_name in GATED:
                strategy = VARIANTS[gate_name]()
                base = VARIANTS["kelly_regime_v4"]()
                funding_arg = REAL if market.pays_funding else None
                r_gate = run_backtest(strategy, window, market, 1_000.0,
                                      trade_start=eval_start, funding=funding_arg,
                                      data_label=LABEL)
                r_base = run_backtest(base, window, market, 1_000.0,
                                      trade_start=eval_start, funding=funding_arg,
                                      data_label=LABEL)
                eq_g = r_gate.equity.to_numpy(dtype=float)[eval_start:]
                eq_b = r_base.equity.to_numpy(dtype=float)[eval_start:]
                ret_g = 100.0 * (eq_g[-1] / eq_g[0] - 1.0) if eq_g[0] > 0 else -100.0
                ret_b = 100.0 * (eq_b[-1] / eq_b[0] - 1.0) if eq_b[0] > 0 else -100.0
                dd_g, dd_b = max_drawdown_pct(eq_g), max_drawdown_pct(eq_b)
                rows.append({"market": market_name, "gate": gate_name, "trial": k,
                            "days": length // BARS_PER_DAY,
                            "ret_gate": ret_g, "ret_base": ret_b,
                            "dd_gate": dd_g, "dd_base": dd_b})
        print(f"-- {market_name} done ({len(specs)} windows x {len(GATED)} gates) --")

    res = pd.DataFrame(rows)
    out = ROOT / "reports" / "funding_gate"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "falsify.csv", index=False)
    print()
    for (market_name, gate_name), grp in res.groupby(["market", "gate"], sort=False):
        deeper = (grp["dd_gate"] > grp["dd_base"]).mean() * 100.0
        worse_return = (grp["ret_gate"] < grp["ret_base"]).mean() * 100.0
        print(f"  {market_name:11s} {gate_name:26s} deeper DD in {deeper:5.1f}% of "
              f"windows, lower return in {worse_return:5.1f}% "
              f"(median Δret {np.median(grp['ret_gate'] - grp['ret_base']):>+7.1f}pp, "
              f"median ΔDD {np.median(grp['dd_gate'] - grp['dd_base']):>+6.1f}pp)")
    print(f"\nwritten: {out / 'falsify.csv'}")


def feetier() -> None:
    """0.40% Bitstamp entry-tier survival check, spot, 2023 holdout."""
    spot40 = MarketSpec.spot(fee_rate=0.004)
    print(f"0.40% spot fee tier, holdout {HOLDOUT_START}..{HOLDOUT_END}\n")
    for name in ("buy_and_hold", *VARIANTS):
        strategy = VARIANTS.get(name, lambda n=name: get_strategy(n))()
        res = _period(strategy, spot40, HOLDOUT_START, HOLDOUT_END)
        m = compute_metrics(res)
        print(f"  {name:28s} final=${m.final_balance:>10,.0f} "
              f"({m.profit_pct:>+7.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
              f"fees=${m.fees_paid:>8,.0f}")


COMMANDS = {"holdout": holdout, "interval": interval, "falsify": falsify,
            "feetier": feetier}


def main() -> None:
    if REAL is None:
        raise SystemExit("no funding data committed")
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python {sys.argv[0]} [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    main()
