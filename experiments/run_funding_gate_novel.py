#!/usr/bin/env python
"""Driver for the continuous funding-z-score gate (experiments/funding_gate_novel.py).

Splits follow ROUTINE.md step 3, inner-split convention (this session used
the 2020/2021/2022 boundaries specified in the task, not the repo's default
2017-2020/2021-2022 split — chosen to sit entirely inside the real funding
data's 2020-2023 coverage):

    inner-train       2020-01-01 -> 2021-12-31   fit, sweep, iterate
    inner-validation  2022-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              NOT touched this session

Usage::

    python experiments/run_funding_gate_novel.py sweep       # the 7 configs
    python experiments/run_funding_gate_novel.py sanity      # spot no-op check
    python experiments/run_funding_gate_novel.py causality   # by-hand lookahead check
    python experiments/run_funding_gate_novel.py all
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
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")

# The task's holdout is reserved; assert at import time that nothing below
# can silently drift past it.
OOS_START = pd.Timestamp("2023-01-01", tz="UTC")

N_CONFIGS = 0  # distinct parameter configurations evaluated (not backtest count)

# The 7 pre-specified configs: a center, then one-knob-at-a-time neighbours.
CENTER = dict(k=0.5, z_cap=3.0, funding_halflife_days=3.0)
CONFIGS = [
    ("center  k=0.5 z_cap=3.0 hl=3.0d", dict(CENTER)),
    ("k=0.3               ", {**CENTER, "k": 0.3}),
    ("k=0.8               ", {**CENTER, "k": 0.8}),
    ("z_cap=2.0           ", {**CENTER, "z_cap": 2.0}),
    ("z_cap=4.0           ", {**CENTER, "z_cap": 4.0}),
    ("hl=1.5d             ", {**CENTER, "funding_halflife_days": 1.5}),
    ("hl=7.0d             ", {**CENTER, "funding_halflife_days": 7.0}),
]


def _period(strategy, market, start, end, funding=None):
    """Backtest over [start, end], warmed on the bars before it, trimmed after.

    Mirrors scripts/funding_study.py's ``_period`` — ``run_period`` in
    tradebot.window does not accept a ``funding`` kwarg, so the manual
    trade_start / trim pattern is copied here rather than reinvented.
    """
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    assert DF.index[hi - 1] < OOS_START, (
        f"refusing to read past the holdout boundary: bar {hi - 1} is "
        f"{DF.index[hi - 1]}")
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw


def ev(strategy, market, start, end, tag):
    """One backtest, one printed line. Real funding is always passed in —
    the engine ignores it on spot (market.pays_funding is False there), and
    charging it on every futures run (baselines included) is the whole
    point of this experiment."""
    m, raw = _period(strategy, market, start, end, funding=REAL_FUNDING)
    print(f"  {tag:24s} {market.name:12s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} fees=${m.fees_paid:>7,.0f} "
          f"funding=${raw.funding_paid:>7,.0f} fills={len(raw.fills):>4d}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m, raw


# ------------------------------------------------------------------- sweep


def sweep() -> None:
    global N_CONFIGS
    print(f"{len(DF):,} bars in the full dataset  {DF.index[0]:%Y-%m-%d} -> "
          f"{DF.index[-1]:%Y-%m-%d}  (data: {LABEL})")
    print(f"funding: {len(REAL_FUNDING):,} settlements "
          f"{REAL_FUNDING.index[0]:%Y-%m-%d} -> {REAL_FUNDING.index[-1]:%Y-%m-%d}\n")

    print("=== Baselines (reference, not counted as configurations) — FUTURES 5x, real funding charged ===")
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        ev(get_strategy("buy_and_hold"), FUTURES, start, end,
           tag=f"buy_and_hold [{split_name}]")
        ev(get_strategy("kelly_regime_v4"), FUTURES, start, end,
           tag=f"kelly_regime_v4 [{split_name}]")

    print("\n=== Sweep: 7 configurations x {inner-train, inner-validation} — "
          "FUTURES 5x, real funding charged ===")
    for tag, kw in CONFIGS:
        N_CONFIGS += 1
        for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
            strat = FundingGateNovel(funding=REAL_FUNDING, **kw)
            ev(strat, FUTURES, start, end, tag=f"{tag.strip()} [{split_name}]")

    print(f"\nConfigurations evaluated: {N_CONFIGS}")


# ------------------------------------------------------------------ sanity


def sanity() -> None:
    """Spot must be a no-op: target_base == target_funded, so the gate's
    strategy must produce EXACTLY the same numbers as plain kelly_regime_v4
    on spot (where on_bar always reads target_base)."""
    print("=== Spot sanity check: the gate must be a no-op on spot ===\n")
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        m_v4, raw_v4 = _period(get_strategy("kelly_regime_v4"), SPOT, start, end,
                               funding=REAL_FUNDING)
        gate = FundingGateNovel(funding=REAL_FUNDING, **CENTER)
        m_gate, raw_gate = _period(gate, SPOT, start, end, funding=REAL_FUNDING)
        print(f"  [{split_name}]")
        print(f"    kelly_regime_v4      final=${m_v4.final_balance:>11,.2f} "
              f"DD={m_v4.max_drawdown_pct:>6.2f}% sharpe={m_v4.sharpe:>6.4f} "
              f"fills={len(raw_v4.fills)}")
        print(f"    funding_gate_novel   final=${m_gate.final_balance:>11,.2f} "
              f"DD={m_gate.max_drawdown_pct:>6.2f}% sharpe={m_gate.sharpe:>6.4f} "
              f"fills={len(raw_gate.fills)}")
        eq_diff = float(np.max(np.abs(
            raw_v4.equity.to_numpy(dtype=float) - raw_gate.equity.to_numpy(dtype=float))))
        bal_diff = abs(m_v4.final_balance - m_gate.final_balance)
        print(f"    max |equity curve difference| = {eq_diff:.6e}   "
              f"final balance difference = {bal_diff:.6e}   "
              f"{'PASS - identical' if eq_diff < 1e-6 else 'FAIL - diverges'}\n")


# --------------------------------------------------------------- causality


def causality() -> None:
    """The strict on_bar peek check, run by hand for an unregistered strategy.

    Copied from experiments/run_eprocess.py's ``causality()`` — same
    two-opposite-tampers procedure, column names adapted to this strategy.
    Restricted entirely to bars dated before 2023-01-01, matching the rest
    of this session.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.loc[:"2022-12-31"].copy()
    assert df.index[-1] < OOS_START
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = FundingGateNovel(funding=REAL_FUNDING, **CENTER)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b = decisions(up), decisions(down)
    bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
    print(f"tampered from bar {cut:,} of {len(df):,} ({df.index[cut]:%Y-%m-%d}); "
          f"checked bars {bars}")
    print("FAIL - reads the future at bars " + str(bad) if bad
          else "PASS - every decision at or before the cut is unchanged")

    pa = FundingGateNovel(funding=REAL_FUNDING, **CENTER).prepare(up.copy())
    pb = FundingGateNovel(funding=REAL_FUNDING, **CENTER).prepare(down.copy())
    all_pass = not bad
    for col in ("target_base", "target_funded", "funding_ewm", "z"):
        diff = np.abs(pa[col].to_numpy(dtype=float)[:cut] -
                      pb[col].to_numpy(dtype=float)[:cut])
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        ok = worst < 1e-12
        all_pass = all_pass and ok
        print(f"  column {col:14s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    cmds = {"sweep": sweep, "sanity": sanity, "causality": causality}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in cmds.items():
            print(f"\n{'=' * 90}\n{name}\n{'=' * 90}")
            fn()
    elif choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_novel.py [{'|'.join(cmds)}|all]")
