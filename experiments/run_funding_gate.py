"""Holdout evaluation for R-33 (backlog B-05): funding gate on kelly_regime_v4.

Runs the D1/D2/D3 tests pre-registered in docs/LEDGER.md ("R-33
pre-registration") BEFORE this file existed. Do not edit the decision
rules here to match what comes out -- if a rule needs to change, that goes
in the ledger as an explicit, flagged in-sample downgrade, not a silent
edit to this file.

Usage: python experiments/run_funding_gate.py [d1|d2|d3|all]
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns, max_drawdown_from_returns, paired_bootstrap,
    stationary_bootstrap_indices, total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.funding_gate_conservative import FundingDecileGate  # noqa: E402
from experiments.funding_gate_novel import FundingCrowdKelly  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

OOS_START = "2023-01-01"
FUNDING_END = "2023-12-31"

MEAN_BLOCK = 30.0
N_BOOT = 2_000
LEVEL = 0.95

FROZEN = {
    "funding_gate_conservative": lambda: FundingDecileGate(
        funding_percentile_threshold=0.90, funding_lookback_days=90),
    "funding_gate_novel": lambda: FundingCrowdKelly(
        funding_ewm_halflife_days=3.0, cost_ceiling=0.20, floor_mult=0.0),
}


def period(strategy, market, start=None, end=None, funding=None):
    """Backtest over [start, end], warmed on the bars before it. Returns (Metrics, equity)."""
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                        trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                            df=raw.df.iloc[pre:])
    return compute_metrics(trimmed), trimmed.equity, raw.funding_paid


def bootstrap_pair(eq_a: pd.Series, eq_b: pd.Series, label_a: str, label_b: str) -> None:
    ra = daily_returns(eq_a).to_numpy()
    rb = daily_returns(eq_b).to_numpy()
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    idx = stationary_bootstrap_indices(n, MEAN_BLOCK, N_BOOT, np.random.default_rng(7))
    growth = paired_bootstrap(ra, rb, total_log_return, indices=idx, level=LEVEL)
    dd = paired_bootstrap(ra, rb, max_drawdown_from_returns, indices=idx, level=LEVEL)
    star = "*" if growth.diff.lo > 0 or growth.diff.hi < 0 else " "
    dstar = "*" if dd.diff.lo > 0 or dd.diff.hi < 0 else " "
    print(f"  {label_a} vs {label_b}  (n={n} days, {MEAN_BLOCK:.0f}d block, {N_BOOT} boot)")
    print(f"    d_log_growth = {growth.diff.point:+.3f} "
          f"[{growth.diff.lo:+.3f}, {growth.diff.hi:+.3f}]{star}  P(>0)={growth.p_positive:.2f}")
    print(f"    d_max_dd_pp  = {dd.diff.point:+.2f} "
          f"[{dd.diff.lo:+.2f}, {dd.diff.hi:+.2f}]{dstar}  P(deeper)={dd.p_positive:.2f}")


def d1() -> None:
    """Primary: paired vs kelly_regime_v4, funding-covered holdout sub-window only."""
    print(f"D1 -- {OOS_START} .. {FUNDING_END}, paired vs kelly_regime_v4\n")
    for market_name, market, funding in (("spot", SPOT, None), ("futures5x", FUTURES, REAL)):
        print(f"=== {market_name} ===")
        _, eq_v4, fp_v4 = period(get_strategy("kelly_regime_v4"), market,
                                  OOS_START, FUNDING_END, funding=funding)
        _, eq_hold, fp_hold = period(get_strategy("buy_and_hold"), market,
                                      OOS_START, FUNDING_END, funding=funding)
        print(f"  kelly_regime_v4  funding_paid=${fp_v4:,.0f}")
        print(f"  buy_and_hold     funding_paid=${fp_hold:,.0f}")
        for name, factory in FROZEN.items():
            _, eq_var, fp_var = period(factory(), market, OOS_START, FUNDING_END, funding=funding)
            print(f"\n  {name}  funding_paid=${fp_var:,.0f}")
            bootstrap_pair(eq_var, eq_v4, name, "kelly_regime_v4")
        print()


def d2() -> None:
    """Secondary: the standard ROUTINE Step-4 bar over the full 2023+ holdout, vs buy_and_hold."""
    print(f"D2 -- {OOS_START} onward (full holdout), paired vs buy_and_hold, "
          "predicted uninformative by construction\n")
    for market_name, market, funding in (("spot", SPOT, None), ("futures5x", FUTURES, REAL)):
        print(f"=== {market_name} ===")
        m_v4, eq_v4, fp_v4 = period(get_strategy("kelly_regime_v4"), market,
                                     OOS_START, funding=funding)
        m_hold, eq_hold, fp_hold = period(get_strategy("buy_and_hold"), market,
                                           OOS_START, funding=funding)
        print(f"  kelly_regime_v4  final=${m_v4.final_balance:,.0f} DD={m_v4.max_drawdown_pct:.1f}% "
              f"sharpe={m_v4.sharpe:.2f} funding=${fp_v4:,.0f}")
        print(f"  buy_and_hold     final=${m_hold.final_balance:,.0f} DD={m_hold.max_drawdown_pct:.1f}% "
              f"sharpe={m_hold.sharpe:.2f} funding=${fp_hold:,.0f}"
              f"{'  LIQUIDATED' if m_hold.liquidated else ''}")
        for name, factory in FROZEN.items():
            m_var, eq_var, fp_var = period(factory(), market, OOS_START, funding=funding)
            print(f"\n  {name}  final=${m_var.final_balance:,.0f} DD={m_var.max_drawdown_pct:.1f}% "
                  f"sharpe={m_var.sharpe:.2f} funding=${fp_var:,.0f}"
                  f"{'  LIQUIDATED' if m_var.liquidated else ''}")
            bootstrap_pair(eq_var, eq_hold, name, "buy_and_hold")
        print()


def d3(trials: int = 30, min_days: int = 60, max_days: int = 180, seed: int = 42) -> None:
    """Falsification: Monte Carlo windows entirely inside the funding-covered range."""
    warmup = max(FROZEN[n]().warmup for n in FROZEN) + 10
    lo_bar = int(DF.index.searchsorted("2020-01-01"))
    hi_bar = int(DF.index.searchsorted(FUNDING_END, side="right"))
    BARS_PER_DAY = 288
    rng = np.random.default_rng(seed)

    specs = []
    for _ in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(lo_bar, hi_bar - length))
        specs.append((start, length))

    print(f"D3 -- {trials} windows, {min_days}-{max_days}d, entirely inside "
          f"2020-01-01..{FUNDING_END}, seed={seed}\n")

    for market_name, market, funding in (("spot", SPOT, None), ("futures5x", FUTURES, REAL)):
        rows = {name: [] for name in FROZEN}
        for k, (start, length) in enumerate(specs, 1):
            window = DF.iloc[start - warmup: start + length]
            eval_start = warmup
            base_result = run_backtest(get_strategy("kelly_regime_v4"), window, market,
                                        1_000.0, trade_start=eval_start, funding=funding)
            base_eq = base_result.equity.to_numpy(dtype=float)
            base0 = base_eq[eval_start]
            base_ret = 100.0 * (base_eq[-1] / base0 - 1.0) if base0 > 0 else -100.0
            for name, factory in FROZEN.items():
                res = run_backtest(factory(), window, market, 1_000.0,
                                    trade_start=eval_start, funding=funding)
                eq = res.equity.to_numpy(dtype=float)
                b0 = eq[eval_start]
                ret = 100.0 * (eq[-1] / b0 - 1.0) if b0 > 0 else -100.0
                seg = eq[eval_start:]
                peaks = np.maximum.accumulate(np.maximum(seg, 1e-9))
                dd = float(np.max((peaks - seg) / peaks) * 100.0)
                base_seg = base_eq[eval_start:]
                base_peaks = np.maximum.accumulate(np.maximum(base_seg, 1e-9))
                base_dd = float(np.max((base_peaks - base_seg) / base_peaks) * 100.0)
                rows[name].append({"d_ret": ret - base_ret, "d_dd": dd - base_dd,
                                    "deeper": dd > base_dd})
        print(f"=== {market_name} ({trials} windows) ===")
        for name in FROZEN:
            d_ret = np.array([r["d_ret"] for r in rows[name]])
            d_dd = np.array([r["d_dd"] for r in rows[name]])
            deeper = np.array([r["deeper"] for r in rows[name]])
            print(f"  {name} vs kelly_regime_v4:")
            print(f"    median d_return={np.median(d_ret):+.1f}pp  "
                  f"higher in {(d_ret > 0).mean():.0%} of windows")
            print(f"    median d_maxDD={np.median(d_dd):+.1f}pp  "
                  f"deeper in {deeper.mean():.0%} of windows")
        print()


COMMANDS = {"d1": d1, "d2": d2, "d3": d3}

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate.py [{'|'.join(COMMANDS)}|all]")
