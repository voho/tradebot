#!/usr/bin/env python
"""Driver for backlog B-05 - funding as a gate on kelly_regime_v4.

Splits follow ROUTINE.md step 3, shifted to where the funding file starts
(the committed history covers 2020-01-01 .. 2023-12-31 only, per
``tradebot.data.load_funding``, so an inner-train starting in 2017 would
spend three years unable to gate on anything)::

    inner-train       2020-01-01 -> 2021-12-31   fit, sweep, iterate
    inner-validation  2022-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              NOT touched by this file

Usage::

    python experiments/run_funding_decile_gate.py sweep       # inner-train + inner-val, 12 configs
    python experiments/run_funding_decile_gate.py neighbours  # plateau check around the pick
    python experiments/run_funding_decile_gate.py causality   # by-hand lookahead probe

This file deliberately contains no ``holdout()`` / OOS command and never
reads any bar timestamped 2023-01-01 or later - the operator runs the
holdout separately, once, after this report is received.
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

from experiments.funding_decile_gate import (  # noqa: E402
    FundingDecileGate,
    causal_funding_percentile,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import prefix_bars  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")
# Deliberately not defined: an OOS/holdout constant would invite "just a
# quick look" later in this file. The operator runs step 4 separately.

N_EVALUATED = 0  # every configuration this file evaluates, for deflated Sharpe

assert FUNDING is not None, "data/btcusdt_perp_funding_8h.csv.gz is missing"
assert FUNDING.index[-1] < pd.Timestamp("2024-01-01", tz="UTC"), (
    "funding file covers more than 2020-2023 - re-check the OOS boundary "
    "before running this file, it must not read 2023-01-01 onward")


# ---------------------------------------------------------------- the harness

def run_period_funded(strategy, df, start, end, *, market, funding=None,
                       start_balance=1_000.0, data_label=""):
    """``tradebot.window.run_period``, plus a ``funding`` pass-through.

    ``run_period`` handles the warmup-prefix problem correctly (see its
    docstring) but does not accept a ``funding`` argument, and the whole
    point of this experiment is to charge funding on futures rather than
    score a funding-free proxy. This reproduces its exact prefix/trim
    logic (``prefix_bars`` is imported from the same module, not
    reimplemented) and forwards ``funding`` to ``run_backtest`` -
    everything else about how warmup is handled is unchanged.
    """
    lo = 0 if start is None else int(df.index.searchsorted(start))
    hi = len(df) if end is None else int(df.index.searchsorted(end, side="right"))
    if hi <= lo:
        raise ValueError(f"empty period: {start!r} -> {end!r}")

    prefix = prefix_bars(df, lo, strategy.warmup)
    frame = df.iloc[lo - prefix: hi]

    result = run_backtest(strategy, frame, market, start_balance,
                          data_label=data_label, trade_start=prefix, funding=funding)
    if prefix == 0:
        return result
    return replace(result, equity=result.equity.iloc[prefix:], df=result.df.iloc[prefix:])


def ev(strategy, start, end, *, market, tag="", funding=None, balance=1_000.0,
       count=True):
    """One backtest, one line, counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = run_period_funded(strategy, DF, start, end, market=market,
                               funding=funding, start_balance=balance,
                               data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:38s} {market.name:11s} "
          f"final=${m.final_balance:>10,.0f} ({m.profit_pct:>+7.1f}%) "
          f"fills={len(result.fills):>5d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} funding=${result.funding_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m, result.funding_paid


def _funding_for(market):
    """Real funding on futures (the point of the experiment); None on spot
    (spot never pays it - MarketSpec.spot().pays_funding is False - but
    passing None rather than the series keeps the call site explicit about
    which market is actually being charged)."""
    return FUNDING if market is FUTURES else None


# ------------------------------------------------------------------- configs

DECILES = (0.80, 0.85, 0.90, 0.95)
LOOKBACKS = (90, 180, 365)


def _configs():
    return [(f"d={d:.2f} lb={lb}d", dict(decile=d, lookback_days=lb))
            for d in DECILES for lb in LOOKBACKS]


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    ev(get_strategy("buy_and_hold"), start, end, market=market,
       tag="buy_and_hold", funding=_funding_for(market), count=False)
    ev(get_strategy("kelly_regime_v4"), start, end, market=market,
       tag="kelly_regime_v4", funding=_funding_for(market), count=False)


def sweep() -> None:
    """4 deciles x 3 lookbacks = 12 configs, both markets, both inner splits."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} funding_decile_gate variants:")
            for tag, kw in _configs():
                ev(FundingDecileGate(funding=FUNDING, **kw), start, end,
                   market=market, tag=tag, funding=_funding_for(market))
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def neighbours(center_decile: float = 0.90, center_lookback: int = 180) -> None:
    """Plateau, not peak: vary one knob at a time around the selection."""
    base = dict(decile=center_decile, lookback_days=center_lookback)
    d_step = 0.05
    lb_choices = LOOKBACKS
    grid = [(f"base d={center_decile:.2f} lb={center_lookback}d", {})]
    for d in (round(center_decile - d_step, 2), round(center_decile + d_step, 2)):
        if 0.0 < d < 1.0:
            grid.append((f"decile={d:.2f}", dict(decile=d)))
    idx = lb_choices.index(center_lookback) if center_lookback in lb_choices else None
    neighbour_lbs = []
    if idx is not None:
        if idx > 0:
            neighbour_lbs.append(lb_choices[idx - 1])
        if idx < len(lb_choices) - 1:
            neighbour_lbs.append(lb_choices[idx + 1])
    else:
        neighbour_lbs = [max(30, center_lookback - 90), center_lookback + 90]
    for lb in neighbour_lbs:
        grid.append((f"lookback={lb}d", dict(lookback_days=lb)))

    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nINNER-VALIDATION neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(FundingDecileGate(funding=FUNDING, **{**base, **kw}), *VALID,
               market=market, tag=tag, funding=_funding_for(market))
        print(f"INNER-TRAIN neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(FundingDecileGate(funding=FUNDING, **{**base, **kw}), *TRAIN,
               market=market, tag=tag, funding=_funding_for(market), count=False)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


# ----------------------------------------------------------------- causality

def causality() -> None:
    """Two-opposite-tampers on_bar check, plus two funding-specific checks.

    1. The standard by-hand probe from ``run_eprocess.py``: bars after a
       cut are multiplied by 3 in one copy and divided by 3 in the other;
       every ``on_bar`` decision at or before the cut must be identical.
    2. The funding percentile-rank column must be UNCHANGED by tampering
       future price bars at all (it depends only on the funding series
       and each bar's own position, never on price).
    3. Recomputing ``prepare()`` on a frame truncated to end at the cut
       must give identical ``target`` values at-and-before the cut,
       compared with the full frame - this is the check that catches a
       rolling window that has quietly become an expanding, whole-series
       computation.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    FROZEN = dict(decile=0.90, lookback_days=180)

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = FundingDecileGate(funding=FUNDING, **FROZEN)
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
    print(f"[1] on_bar decisions: tampered from bar {cut:,} of {len(df):,}; "
          f"checked bars {bars}")
    print("    FAIL - reads the future at bars " + str(bad) if bad
          else "    PASS - every decision at or before the cut is unchanged")

    # A prepared column computed over the whole series would also move at
    # rows before the cut even when no order changes.
    pa = FundingDecileGate(funding=FUNDING, **FROZEN).prepare(up.copy())
    pb = FundingDecileGate(funding=FUNDING, **FROZEN).prepare(down.copy())
    print("[2] prepared columns, price-tamper invariance before the cut:")
    all_ok_2 = True
    for col in ("target", "funding_pct_rank"):
        va = pa[col].to_numpy(dtype=float)[:cut]
        vb = pb[col].to_numpy(dtype=float)[:cut]
        # funding_pct_rank can be legitimately NaN on both sides; compare
        # like-for-like and treat "both NaN" as agreement.
        both_nan = np.isnan(va) & np.isnan(vb)
        diff = np.where(both_nan, 0.0, np.abs(va - vb))
        diff = np.nan_to_num(diff, nan=np.inf)  # one-sided NaN is a real mismatch
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        ok = worst < 1e-9
        all_ok_2 &= ok
        print(f"    column {col:20s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if ok else 'FAIL'}")
    # funding_pct_rank in particular must be BYTE-IDENTICAL across the two
    # tampers everywhere, not just before the cut - it must not depend on
    # price at all, tampered region included.
    fa = pa["funding_pct_rank"].to_numpy(dtype=float)
    fb = pb["funding_pct_rank"].to_numpy(dtype=float)
    both_nan_all = np.isnan(fa) & np.isnan(fb)
    diff_all = np.where(both_nan_all, 0.0, np.abs(fa - fb))
    diff_all = np.nan_to_num(diff_all, nan=np.inf)
    worst_all = float(np.nanmax(diff_all))
    print(f"    funding_pct_rank identical EVERYWHERE (incl. tampered region): "
          f"max |difference| = {worst_all:.3e}  "
          f"{'PASS' if worst_all < 1e-9 else 'FAIL'}  "
          f"(expected: exactly 0, since price never enters this column)")

    # (b) truncation check: prepare() on a frame that ends AT the cut must
    # match the full frame's target at-and-before the cut. This is the
    # expanding-vs-rolling check the tamper test above cannot see, because
    # tampering leaves the row COUNT unchanged - only truncation does.
    truncated = df.iloc[:cut].copy()
    pt = FundingDecileGate(funding=FUNDING, **FROZEN).prepare(truncated)
    pfull = FundingDecileGate(funding=FUNDING, **FROZEN).prepare(df.copy())
    print("[3] truncation check (rolling vs accidentally-expanding window):")
    all_ok_3 = True
    for col in ("target", "funding_pct_rank"):
        vt = pt[col].to_numpy(dtype=float)
        vf = pfull[col].to_numpy(dtype=float)[:cut]
        both_nan = np.isnan(vt) & np.isnan(vf)
        diff = np.where(both_nan, 0.0, np.abs(vt - vf))
        diff = np.nan_to_num(diff, nan=np.inf)
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        ok = worst < 1e-9
        all_ok_3 &= ok
        print(f"    column {col:20s} max |difference| truncated-vs-full = {worst:.3e}"
              f"  {'PASS' if ok else 'FAIL'}")

    overall = (not bad) and all_ok_2 and (worst_all < 1e-9) and all_ok_3
    print(f"\ncausality self-check: {'PASS' if overall else 'FAIL'}")


# ----------------------------------------------------------------- inspect

def inspect() -> None:
    """What the causal funding percentile actually looks like, before any backtest."""
    for lb in LOOKBACKS:
        pct = causal_funding_percentile(DF.index, FUNDING, lb)
        covered = pct.notna()
        first_valid = pct[covered].index[0] if covered.any() else None
        print(f"lookback={lb:>3d}d  bars with a full trailing window: "
              f"{covered.sum():>7,} / {len(DF):,}   "
              f"first valid bar: {first_valid}")
    print(f"\nfunding file: {len(FUNDING):,} settlements, "
          f"{FUNDING.index[0]:%Y-%m-%d} -> {FUNDING.index[-1]:%Y-%m-%d}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "neighbours": neighbours, "causality": causality,
            "inspect": inspect}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_decile_gate.py [{'|'.join(cmds)}]")
