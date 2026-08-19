#!/usr/bin/env python
"""Step 4 (ROUTINE.md) for R-33 / B-05 — frozen holdout evaluation.

Pre-registered in docs/LEDGER.md (R-33 pre-registration) BEFORE this
script was run. Reads ONLY 2023-01-01 -> 2023-12-31 as the holdout (the
funding data's own coverage ends 2023-12-31), plus a falsification
resample restricted to 2020-01-01 -> 2022-12-31 (already-read inner
data, not a fresh holdout consultation).

Usage::

    python experiments/run_b05_holdout.py holdout   # P1/P2, paired bootstrap
    python experiments/run_b05_holdout.py falsify    # P3, Monte Carlo resample
    python experiments/run_b05_holdout.py all
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.funding_gate_novel import FundingGateNovel  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import (  # noqa: E402
    annualized_sharpe,
    daily_returns,
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2023-12-31"
FALSIFY_START, FALSIFY_END = "2020-01-01", "2022-12-31"

FROZEN = dict(k=0.5, z_cap=3.0, funding_halflife_days=3.0,
              funding_zscore_window_days=180)


def _period(strategy, market, start, end, funding=None, balance=1_000.0):
    """Fresh-account backtest over [start, end], warmed on bars before it.

    Same pattern as scripts/funding_study.py's _period(): run_period()
    does not accept a `funding` kwarg, so this reproduces its manual
    trade_start/prefix + trim logic.
    """
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    assert DF.index[lo] >= pd.Timestamp(start, tz="UTC"), "start out of range"
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return trimmed


def _row(tag, result):
    m = compute_metrics(result)
    print(f"  {tag:34s} final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"fees=${m.fees_paid:>7,.0f} funding=${result.funding_paid:>7,.0f} "
          f"fills={len(result.fills):>4d}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


def holdout() -> None:
    print(f"assert: no bar before {HOLDOUT_START} is read as 'holdout data' below "
          f"except the warmup prefix, which is price history only, per R-22.\n")
    print(f"{HOLDOUT_START} .. {HOLDOUT_END}, $1,000 start\n")

    print("-- spot (funding_gate_novel is a byte-identical no-op here; one run) --")
    hold_spot = _period(get_strategy("buy_and_hold"), SPOT, HOLDOUT_START, HOLDOUT_END)
    _row("buy_and_hold  spot", hold_spot)
    v4_spot = _period(KellyRegimeV4(), SPOT, HOLDOUT_START, HOLDOUT_END)
    _row("kelly_regime_v4  spot (== funding_gate_novel)", v4_spot)

    print("\n-- futures 5x, real funding charged (the load-bearing comparison) --")
    hold_fut = _period(get_strategy("buy_and_hold"), FUTURES, HOLDOUT_START, HOLDOUT_END,
                       funding=REAL_FUNDING)
    _row("buy_and_hold  futures", hold_fut)
    v4_fut = _period(KellyRegimeV4(), FUTURES, HOLDOUT_START, HOLDOUT_END,
                     funding=REAL_FUNDING)
    _row("kelly_regime_v4  futures", v4_fut)
    gate_fut = _period(FundingGateNovel(funding=REAL_FUNDING, **FROZEN), FUTURES,
                       HOLDOUT_START, HOLDOUT_END, funding=REAL_FUNDING)
    _row("funding_gate_novel  futures (frozen)", gate_fut)

    m_v4, m_gate = compute_metrics(v4_fut), compute_metrics(gate_fut)
    p1 = m_gate.final_balance > m_v4.final_balance
    print(f"\nP1 (gate beats v4 on holdout final balance, futures, funded): "
          f"{'PASS' if p1 else 'FAIL'}  (${m_gate.final_balance:,.0f} vs "
          f"${m_v4.final_balance:,.0f})")

    r_v4 = daily_returns(v4_fut.equity).to_numpy()
    r_gate = daily_returns(gate_fut.equity).to_numpy()
    n = min(len(r_v4), len(r_gate))
    r_v4, r_gate = r_v4[-n:], r_gate[-n:]
    print(f"\npaired stationary block bootstrap, {n} daily obs, 30-day mean block, "
          f"2,000 resamples (gate - v4):")
    for name, stat in (("Sharpe", annualized_sharpe),
                       ("log growth", total_log_return),
                       ("max drawdown (pp)", max_drawdown_from_returns)):
        res = paired_bootstrap(r_gate, r_v4, stat)
        print(f"  Delta {name:20s} {res.diff}  P(gate>v4)={res.p_positive:.2f}"
              f"  {'excludes zero' if res.significant else 'contains zero'}")

    dd_v4 = max_drawdown_from_returns(r_v4)
    dd_gate = max_drawdown_from_returns(r_gate)
    sharpe_v4 = annualized_sharpe(r_v4)
    sharpe_gate = annualized_sharpe(r_gate)
    p2_sharpe = (sharpe_gate - sharpe_v4) > 0.2
    p2_dd = (dd_v4 - dd_gate) >= 10.0
    print(f"\nP2 (>+0.2 Sharpe OR >=10pp DD cut): Sharpe {sharpe_gate - sharpe_v4:+.2f} "
          f"({'PASS' if p2_sharpe else 'fail'})  "
          f"DD cut {dd_v4 - dd_gate:+.1f}pp ({'PASS' if p2_dd else 'fail'})  "
          f"-> {'PASS' if (p2_sharpe or p2_dd) else 'FAIL'}")


def falsify() -> None:
    """P3: 20-window Monte Carlo resample, restricted to 2020-2022 (no holdout read)."""
    print(f"falsification resample window: {FALSIFY_START} .. {FALSIFY_END} "
          f"(already-read inner data; not a holdout consultation)\n")
    lo = int(DF.index.searchsorted(FALSIFY_START))
    hi = int(DF.index.searchsorted(FALSIFY_END, side="right"))
    assert DF.index[hi - 1] < pd.Timestamp("2023-01-01", tz="UTC")
    span = hi - lo
    win_bars = 60 * 288
    rng = np.random.default_rng(33)
    n_windows = 20
    wins, ties, losses = 0, 0, 0
    for i in range(n_windows):
        start = lo + int(rng.integers(0, span - win_bars))
        end = start + win_bars
        warm = min(start, KellyRegimeV4().warmup)
        frame = DF.iloc[start - warm: end]
        v4 = run_backtest(KellyRegimeV4(), frame, FUTURES, 1_000.0,
                          trade_start=warm, funding=REAL_FUNDING, data_label=LABEL)
        gate = run_backtest(FundingGateNovel(funding=REAL_FUNDING, **FROZEN), frame,
                            FUTURES, 1_000.0, trade_start=warm, funding=REAL_FUNDING,
                            data_label=LABEL)
        fb_v4 = compute_metrics(v4).final_balance
        fb_gate = compute_metrics(gate).final_balance
        if fb_gate > fb_v4 * 1.001:
            wins += 1
        elif fb_gate < fb_v4 * 0.999:
            losses += 1
        else:
            ties += 1
        print(f"  window {i:2d} [{frame.index[warm]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}] "
              f"v4=${fb_v4:>9,.0f}  gate=${fb_gate:>9,.0f}  "
              f"{'gate WINS' if fb_gate > fb_v4 else 'v4 wins' if fb_v4 > fb_gate else 'tie'}")

    beat_or_tie = wins + ties
    print(f"\ngate beats-or-ties v4 in {beat_or_tie}/{n_windows} windows "
          f"({beat_or_tie / n_windows:.0%}); wins={wins} ties={ties} losses={losses}")
    print(f"P3 (beat-or-tie in >=50% of paired windows): "
          f"{'PASS' if beat_or_tie / n_windows >= 0.5 else 'FAIL'}")


COMMANDS = {"holdout": holdout, "falsify": falsify}

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_b05_holdout.py [{'|'.join(COMMANDS)}|all]")
