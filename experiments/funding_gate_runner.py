#!/usr/bin/env python
"""Sweep + report driver for ``experiments/funding_gate.py`` (backlog B-05).

HOLDOUT SAFETY: every call in this file is bounded at or before
2022-12-31. ``OOS_START = 2023-01-01`` is imported only as a guard
constant and is never passed as a ``start``/``end`` anywhere below.

Usage::

    .venv/bin/python experiments/funding_gate.py        # (n/a, class only)
    .venv/bin/python experiments/funding_gate_runner.py sweep
    .venv/bin/python experiments/funding_gate_runner.py causality
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from funding_gate import KellyRegimeFundingGate  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

OOS_START = "2023-01-01"  # guard constant only -- never used as an argument below

INNER_TRAIN = (None, "2020-12-31")
INNER_VAL = ("2021-01-01", "2022-12-31")

assert INNER_TRAIN[1] < OOS_START and INNER_VAL[1] < OOS_START, (
    "a split boundary crossed into the holdout -- refusing to run")

# --------------------------------------------------------------------------
# Configurations swept: window length (days of trailing settlements) x
# (upper, lower) release percentile pair. A small hand-picked grid, NOT a
# full 3x3 cross product (that would be 9, over the 8-config cap): one
# baseline pair held fixed while window varies, one baseline window held
# fixed while the pair varies, plus one interaction point combining the
# longest window with the tightest pair. 6 configurations total.
# --------------------------------------------------------------------------
CONFIGS = [
    dict(window_days=60, upper_pct=0.90, lower_pct=0.75, tag="w60  90/75"),
    dict(window_days=90, upper_pct=0.90, lower_pct=0.75, tag="w90  90/75 (baseline)"),
    dict(window_days=180, upper_pct=0.90, lower_pct=0.75, tag="w180 90/75"),
    dict(window_days=90, upper_pct=0.95, lower_pct=0.80, tag="w90  95/80"),
    dict(window_days=90, upper_pct=0.85, lower_pct=0.70, tag="w90  85/70"),
    dict(window_days=180, upper_pct=0.95, lower_pct=0.80, tag="w180 95/80"),
]
assert len(CONFIGS) <= 8


def _period(strategy, market, start=None, end=None, funding=None):
    """Backtest over [start, end], warmed on the bars before it.

    Mirrors ``scripts/funding_study.py:_period`` -- ``run_period`` (the
    usual ``experiment.py`` harness) does not thread a ``funding`` kwarg
    through, so the manual prefix/trade_start logic is redone here to
    keep funding-charged and funding-free runs on an identical warm-up.
    """
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    assert DF.index[hi - 1] < pd.Timestamp(OOS_START, tz="UTC"), (
        "period boundary reaches the 2023+ holdout -- refusing to run")
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def _row(label, m, paid=None):
    paid_s = f"${paid:>8,.0f}" if paid is not None else f"{'':>9s}"
    print(f"  {label:34s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
          f"funding_paid={paid_s} {'LIQUIDATED' if m.liquidated else ''}")


def sweep() -> None:
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})")
    print(f"funding: {len(REAL):,} settlements  {REAL.index[0]:%Y-%m-%d} -> "
          f"{REAL.index[-1]:%Y-%m-%d}\n")

    splits = [("inner-train      2017-01-01 .. 2020-12-31", INNER_TRAIN),
              ("inner-validation 2021-01-01 .. 2022-12-31", INNER_VAL)]

    for split_label, (start, end) in splits:
        print("=" * 100)
        print(split_label)
        print("=" * 100)

        print("\n-- benchmarks --")
        bh = get_strategy("buy_and_hold")
        m, _ = _period(bh, SPOT, start, end)
        _row("buy_and_hold  SPOT", m)
        m, _ = _period(bh, FUTURES, start, end)
        _row("buy_and_hold  FUT5x (funding-free)", m)
        m, paid = _period(bh, FUTURES, start, end, funding=REAL)
        _row("buy_and_hold  FUT5x (funding-charged)", m, paid)

        v4 = get_strategy("kelly_regime_v4")
        m, _ = _period(v4, SPOT, start, end)
        _row("kelly_regime_v4  SPOT", m)
        m, _ = _period(v4, FUTURES, start, end)
        _row("kelly_regime_v4  FUT5x (funding-free)", m)
        m, paid = _period(v4, FUTURES, start, end, funding=REAL)
        _row("kelly_regime_v4  FUT5x (funding-charged)", m, paid)

        print("\n-- funding-gate configs --")
        for cfg in CONFIGS:
            tag = cfg.pop("tag")
            strat = KellyRegimeFundingGate(funding=REAL, **cfg)
            cfg["tag"] = tag

            m, _ = _period(strat, SPOT, start, end)
            _row(f"[{tag}] SPOT", m)

            m, _ = _period(strat, FUTURES, start, end)
            _row(f"[{tag}] FUT5x (funding-free)", m)

            m, paid = _period(strat, FUTURES, start, end, funding=REAL)
            _row(f"[{tag}] FUT5x (funding-charged)", m, paid)
        print()


def gate_diagnostics() -> None:
    """How often does each config actually fire, on the inner period?"""
    print("gate activity, full inner period (2017-01-01 .. 2022-12-31):\n")
    start, end = None, "2022-12-31"
    lo = 0
    hi = int(DF.index.searchsorted(end, side="right"))
    frame = DF.iloc[lo:hi]
    for cfg in CONFIGS:
        tag = cfg.pop("tag")
        strat = KellyRegimeFundingGate(funding=REAL, **cfg)
        cfg["tag"] = tag
        prepared = strat.prepare(frame.copy())
        mult = prepared["funding_gate_multiplier"].to_numpy()
        covered = ((frame.index >= REAL.index[0]) &
                   (frame.index <= REAL.index[-1]))
        gated_frac = float((mult[np.asarray(covered)] < 0.5).mean())
        print(f"  [{tag:22s}] gated (flat-forced) on "
              f"{gated_frac:5.1%} of covered bars")


def causality() -> None:
    """Two-opposite-tampers causality check (R-28's method), pre-2023 only.

    Not part of the pytest suite (unregistered strategy). Confirms every
    order queued at or before a cut bar is unchanged when only bars AT
    OR AFTER the cut are tampered with, in two opposite directions.
    """
    end = int(DF.index.searchsorted("2022-12-31", side="right"))
    df = DF.iloc[:end].iloc[-40_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(strategy, frame, bar_list, market):
        prepared = strategy.prepare(frame.copy())
        broker = PaperBroker(market=market, start_balance=10_000.0)
        out = []
        for i in bar_list:
            ctx = Context(prepared, i, broker)
            strategy.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    market = MarketSpec.futures(leverage=5.0)
    for tag_cfg in CONFIGS:
        cfg = dict(tag_cfg)
        tag = cfg.pop("tag")
        strat_a = KellyRegimeFundingGate(funding=REAL, **cfg)
        strat_b = KellyRegimeFundingGate(funding=REAL, **cfg)
        a = decisions(strat_a, up, bars, market)
        b = decisions(strat_b, down, bars, market)
        ok = all(oa == ob for oa, ob in zip(a, b))
        print(f"  [{tag:22s}] decisions at/before cut unchanged under "
              f"opposite future tampers: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for bar, oa, ob in zip(bars, a, b):
                if oa != ob:
                    print(f"      bar {bar}: {oa} vs {ob}")


COMMANDS = {"sweep": sweep, "diagnostics": gate_diagnostics, "causality": causality}


def main() -> None:
    if REAL is None:
        raise SystemExit("no funding data committed")
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        for name, fn in COMMANDS.items():
            print(f"\n{'#' * 100}\n# {name}\n{'#' * 100}\n")
            fn()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/funding_gate_runner.py [{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    main()
