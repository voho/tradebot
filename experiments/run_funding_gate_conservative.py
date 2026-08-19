#!/usr/bin/env python
"""Driver for backlog B-05 — funding as a gate on `kelly_regime_v4` (conservative).

Splits, fixed by this task (NOT the ROUTINE.md defaults — see the docstring
of ``funding_gate_conservative.py``)::

    inner-train       2020-01-01 -> 2021-12-31   fit, sweep, iterate
    inner-validation  2022-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->               NOT read in this session

Usage::

    python experiments/run_funding_gate_conservative.py sweep       # the 7 configs
    python experiments/run_funding_gate_conservative.py sanity      # spot no-op check
    python experiments/run_funding_gate_conservative.py causality   # by-hand lookahead check
    python experiments/run_funding_gate_conservative.py all         # everything
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

from experiments.funding_gate_conservative import FundingGateConservative  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")
# The 2023+ holdout is deliberately never named as a start/end argument
# anywhere in this file.

N_EVALUATED = 0  # every configuration this file evaluates, for deflated Sharpe


# --------------------------------------------------------------------- harness
#
# tradebot.window.run_period does not accept a `funding=` kwarg, so this is
# scripts/funding_study.py's `_period` pattern, copied rather than
# reinvented: warm the strategy on the bars before the window, run with
# `trade_start` so nothing fills inside the warmup prefix, then trim the
# curve back to the measured period.


def _period(strategy, market, start=None, end=None, funding=None, balance=1_000.0):
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return trimmed


def ev(strategy, start, end, market, funding=None, tag="", count=True):
    """One backtest, one line, counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = _period(strategy, market, start, end, funding=funding)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:42s} {market.name:11s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"fees=${m.fees_paid:>8,.0f} fills={len(result.fills):>5d}"
          f"{'  LIQUIDATED' if result.liquidated else ''}")
    return m, result


# ----------------------------------------------------------------------- sweep

CENTER = dict(funding_pctile_threshold=0.90, discount_factor=0.5, momentum_days=7)


def _configs():
    """Center config plus one-knob-at-a-time neighbours. 7 total."""
    out = [("center  thr=0.90 disc=0.50 mom=7d", dict(CENTER))]
    for thr in (0.85, 0.95):
        out.append((f"thr={thr:.2f}", {**CENTER, "funding_pctile_threshold": thr}))
    for disc in (0.3, 0.7):
        out.append((f"disc={disc:.1f}", {**CENTER, "discount_factor": disc}))
    for mom in (5, 10):
        out.append((f"mom={mom}d", {**CENTER, "momentum_days": mom}))
    return out


def sweep() -> None:
    print(f"funding data: {len(REAL_FUNDING):,} settlements "
          f"{REAL_FUNDING.index[0]:%Y-%m-%d} -> {REAL_FUNDING.index[-1]:%Y-%m-%d}\n")

    for (start, end), split in ((TRAIN, "INNER-TRAIN 2020-2021"),
                                 (VALID, "INNER-VALIDATION 2022")):
        print(f"\n{split} / futures 5x (funding CHARGED):")
        print("  baselines (not counted):")
        ev(get_strategy("buy_and_hold"), start, end, FUTURES, funding=REAL_FUNDING,
           tag="  buy_and_hold", count=False)
        ev(get_strategy("kelly_regime_v4"), start, end, FUTURES, funding=REAL_FUNDING,
           tag="  kelly_regime_v4", count=False)
        print("  swept configs:")
        for tag, kw in _configs():
            strat = FundingGateConservative(funding=REAL_FUNDING, **kw)
            ev(strat, start, end, FUTURES, funding=REAL_FUNDING, tag=f"  {tag}")

    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


# ---------------------------------------------------------------------- sanity


def sanity() -> None:
    """Spot must be bit-identical to plain kelly_regime_v4: the gate never fires there."""
    print("SPOT sanity check (target_base column must equal v4's target exactly):\n")
    for (start, end), split in ((TRAIN, "INNER-TRAIN 2020-2021"),
                                 (VALID, "INNER-VALIDATION 2022")):
        print(f"{split} / spot:")
        v4 = get_strategy("kelly_regime_v4")
        m_v4, res_v4 = ev(v4, start, end, SPOT, funding=None,
                          tag="  kelly_regime_v4", count=False)
        gated = FundingGateConservative(funding=REAL_FUNDING, **CENTER)
        m_gated, res_gated = ev(gated, start, end, SPOT, funding=REAL_FUNDING,
                                tag="  funding_gate_conservative (center)", count=False)

        eq_v4 = res_v4.equity.to_numpy(dtype=float)
        eq_gated = res_gated.equity.to_numpy(dtype=float)
        max_diff = float(np.max(np.abs(eq_v4 - eq_gated))) if len(eq_v4) == len(eq_gated) else float("nan")
        same_len = len(eq_v4) == len(eq_gated)
        same_fills = len(res_v4.fills) == len(res_gated.fills)
        print(f"    equity curves: same length={same_len}  max |diff|={max_diff:.6e}  "
              f"same fill count={same_fills} ({len(res_v4.fills)} vs {len(res_gated.fills)})  "
              f"{'PASS' if same_len and max_diff < 1e-6 and same_fills else 'FAIL'}\n")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict on_bar peek check, run by hand for an unregistered strategy.

    ``tests/test_causality_strict.py`` only parametrizes over *registered*
    strategies, so this experiment gets none of that protection. Same
    two-opposite-tampers procedure as ``experiments/run_eprocess.py``: bars
    after a cut are multiplied by 3 (price) / 7 (volume) in one copy and
    divided by the same factor in the other; every decision at or before
    the cut must be identical, and every prepared column must be bit-equal
    before the cut too (catches a full-series statistic, which a pure
    on_bar-order check cannot).
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    def max_col_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
        """Max |a-b| where both are defined; NaN-pattern mismatch also fails."""
        an, bn = np.isnan(a), np.isnan(b)
        if not np.array_equal(an, bn):
            return float("inf"), False
        valid = ~an
        if not valid.any():
            return 0.0, True
        worst = float(np.max(np.abs(a[valid] - b[valid])))
        return worst, worst < 1e-9

    # A window fully inside the funding series' observed coverage
    # (2020-01-01 .. 2023-12-31), so the gate is actually active on both
    # sides of the cut rather than trivially NaN throughout — otherwise
    # this would only prove causality for the no-op case.
    lo = int(DF.index.searchsorted("2020-06-01"))
    hi = int(DF.index.searchsorted("2023-06-01"))
    df = DF.iloc[lo:hi].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def strat():
        return FundingGateConservative(funding=REAL_FUNDING, **CENTER)

    def decisions(frame):
        s = strat()
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
    print(f"[price tamper] tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    result_a = ("FAIL - reads the future at bars " + str(bad)) if bad else \
        "PASS - every decision at or before the cut is unchanged"
    print(result_a)

    pa = strat().prepare(up.copy())
    pb = strat().prepare(down.copy())
    col_pass = True
    for col in ("target_base", "target_funded", "funding_ewm", "funding_pctile"):
        a_arr = pa[col].to_numpy(dtype=float)[:cut]
        b_arr = pb[col].to_numpy(dtype=float)[:cut]
        worst, ok = max_col_diff(a_arr, b_arr)
        col_pass &= ok
        print(f"  column {col:16s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if ok else 'FAIL'}")

    # Second, independent probe: tamper the FUNDING series itself after the
    # cut (rather than price), which is the part of this strategy that is
    # new relative to eprocess_regime.py's precedent. The `reindex(...,
    # method="ffill")` alignment must not let a changed future funding
    # value move any decision at or before the cut.
    fcut_ts = df.index[cut]
    fund_up = REAL_FUNDING.copy()
    fund_down = REAL_FUNDING.copy()
    later = fund_up.index >= fcut_ts
    fund_up.loc[later] = fund_up.loc[later] * 5.0 + 0.01
    fund_down.loc[later] = fund_down.loc[later] * 0.01 - 0.01

    def strat_with(funding):
        return FundingGateConservative(funding=funding, **CENTER)

    pfa = strat_with(fund_up).prepare(df.copy())
    pfb = strat_with(fund_down).prepare(df.copy())
    print(f"\n[funding tamper] tampered funding from {fcut_ts} onward "
          f"(bar {cut:,} of {len(df):,})")
    fund_col_pass = True
    for col in ("target_base", "target_funded", "funding_known", "funding_ewm", "funding_pctile"):
        a_arr = pfa[col].to_numpy(dtype=float)[:cut]
        b_arr = pfb[col].to_numpy(dtype=float)[:cut]
        worst, ok = max_col_diff(a_arr, b_arr)
        fund_col_pass &= ok
        print(f"  column {col:16s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if ok else 'FAIL'}")

    overall = (not bad) and col_pass and fund_col_pass
    print(f"\nCAUSALITY CHECK: {'PASS' if overall else 'FAIL'}")


# ------------------------------------------------------------------------ main


def all_checks() -> None:
    sweep()
    print("\n" + "=" * 78 + "\n")
    sanity()
    print("\n" + "=" * 78 + "\n")
    causality()


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "sanity": sanity, "causality": causality, "all": all_checks}
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_conservative.py [{'|'.join(cmds)}]")
