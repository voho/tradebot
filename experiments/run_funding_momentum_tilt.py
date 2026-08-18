#!/usr/bin/env python
"""Driver for backlog B-05 — funding-momentum-conditioned exposure tilt.

Splits, PER THIS SESSION'S PRE-REGISTRATION (not the ROUTINE.md default —
the funding file only covers 2020-2023, so the inner split is shifted
into that window)::

    inner-train       2020-01-01 -> 2021-12-31   fit, sweep, iterate
    inner-validation  2022-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->               NOT touched by this file

This session does not run the holdout. Nothing in this file reads,
prints, or scores anything from 2023-01-01 onward — that is the
operator's step, once, after this report is read.

Usage::

    python experiments/run_funding_momentum_tilt.py sweep        # inner-train + inner-val, 24 configs
    python experiments/run_funding_momentum_tilt.py neighbours   # plateau check around FROZEN
    python experiments/run_funding_momentum_tilt.py diagnostics  # does the frac-conditioning actually bind?
    python experiments/run_funding_momentum_tilt.py causality    # by-hand lookahead probe (pre-2023 only)
"""

from __future__ import annotations

import sys
from dataclasses import replace as dc_replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.funding_momentum_tilt import FundingMomentumTilt  # noqa: E402
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
OOS_START = "2023-01-01"  # NEVER read past this in this file.

N_EVALUATED = 0  # every backtest this file runs, for bookkeeping

assert FUNDING is not None, "data/btcusdt_perp_funding_8h.csv.gz is required for this experiment"
# The funding FILE legitimately covers through 2023-12-31 (data.load_funding's
# docstring) - that is not itself a holdout violation. What matters is that
# this script never runs a backtest whose scored PERIOD extends past
# OOS_START; TRAIN and VALID below both end at 2022-12-31, so no bar this
# file evaluates ever falls in 2023+, regardless of what the funding series
# itself contains.
assert TRAIN[1] < OOS_START and VALID[1] < OOS_START


def run_period_funded(strategy, df, start, end, *, market, start_balance=1_000.0,
                       data_label="", funding=None):
    """``tradebot.window.run_period``, plus a ``funding`` pass-through.

    ``window.run_period`` does not accept ``funding`` (yet) — see
    ``experiments/run_eprocess.py``'s ``costs()`` for the same gap and the
    same fix. This reuses ``run_period``'s own warmup-prefix arithmetic
    (``prefix_bars``) verbatim rather than re-deriving it, so the
    fairness property (every strategy starts the measured period flat,
    warm, and at full balance) is identical; only funding is added on top.
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
    return dc_replace(result, equity=result.equity.iloc[prefix:], df=result.df.iloc[prefix:])


def ev(strategy, start, end, market, tag="", balance=1_000.0, count=True):
    """One backtest, one line, counted. Futures always gets funding charged."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    funding_arg = FUNDING if market.pays_funding else None
    result = run_period_funded(strategy, DF, start, end, market=market,
                               start_balance=balance, data_label=LABEL, funding=funding_arg)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:42s} {market.name:11s} "
          f"final=${m.final_balance:>10,.0f} ({m.profit_pct:>+7.1f}%) "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"trades={len(result.trades):>4d} funding=${result.funding_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m, result


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}", count=False)


# --------------------------------------------------------------------- sweep

WEIGHTS = (0.3, 0.5, 0.7, 1.0)
LO_HI = ((0.60, 0.90), (0.70, 0.95), (0.80, 0.98))
LOOKBACKS = (90, 180)


def sweep_grid() -> list[dict]:
    """The 4 x 3 x 2 = 24 distinct configurations swept this session."""
    grid = []
    for w in WEIGHTS:
        for lo, hi in LO_HI:
            for lb in LOOKBACKS:
                grid.append(dict(weight=w, lo=lo, hi=hi, lookback_days=lb))
    return grid


SWEEP_GRID = sweep_grid()
assert len(SWEEP_GRID) == 24


def _tag(cfg: dict) -> str:
    return f"w={cfg['weight']:.1f} lo={cfg['lo']:.2f} hi={cfg['hi']:.2f} lb={cfg['lookback_days']:>3d}d"


def sweep() -> list[dict]:
    rows = []
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} funding_momentum_tilt sweep:")
            for cfg in SWEEP_GRID:
                m, result = ev(FundingMomentumTilt(funding=FUNDING, **cfg), start, end,
                               market=market, tag=_tag(cfg))
                rows.append({"market": mname, "split": split, **cfg,
                             "final_balance": m.final_balance, "profit_pct": m.profit_pct,
                             "max_dd_pct": m.max_drawdown_pct, "sharpe": m.sharpe,
                             "trades": len(result.trades), "funding_paid": result.funding_paid,
                             "liquidated": m.liquidated})
    print(f"\ndistinct configurations swept: {len(SWEEP_GRID)} "
          f"(weight x (lo,hi) x lookback_days grid)")
    print(f"total backtests run in this sweep: {N_EVALUATED} "
          f"({len(SWEEP_GRID)} configs x 2 markets x 2 splits)")
    return rows


# --------------------------------------------------------------- neighbours

# Selected after inspecting the 24-config sweep (see report): the highest
# average INNER-VALIDATION Sharpe across spot+futures jointly (not spot
# alone - avoids picking a config that only works where funding is free).
# weight=1.0 scores marginally higher on spot alone but WORSE on futures
# (over-shrinks exposure, fewer trades, deeper path dependence) - see the
# report for the full table. Kept here, not hand-edited mid-run, so the
# neighbourhood check below is reproducible.
FROZEN = dict(weight=0.7, lo=0.70, hi=0.95, lookback_days=90)


def neighbours() -> None:
    """Plateau, not peak: vary one knob at a time around FROZEN."""
    grid = [("base " + _tag(FROZEN), {})]
    grid += [(f"weight={w:.1f}", dict(weight=w))
             for w in sorted({0.3, 0.5, 0.7, 1.0} - {FROZEN["weight"]})]
    grid += [(f"(lo,hi)=({lo:.2f},{hi:.2f})", dict(lo=lo, hi=hi))
             for lo, hi in LO_HI if (lo, hi) != (FROZEN["lo"], FROZEN["hi"])]
    grid += [(f"lookback_days={lb}", dict(lookback_days=lb))
             for lb in LOOKBACKS if lb != FROZEN["lookback_days"]]
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nINNER-VALIDATION neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(FundingMomentumTilt(funding=FUNDING, **{**FROZEN, **kw}), *VALID,
               market=market, tag=tag)
        print(f"INNER-TRAIN neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(FundingMomentumTilt(funding=FUNDING, **{**FROZEN, **kw}), *TRAIN,
               market=market, tag=tag, count=False)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


# ------------------------------------------------------------- diagnostics


def diagnostics() -> None:
    """Is the frac-conditioning doing anything, or is this just a funding gate?

    Uses TRAIN+VALID only (2020-01-01 -> 2022-12-31), never 2023+.
    """
    s = FundingMomentumTilt(funding=FUNDING, **FROZEN)
    window = DF.loc["2020-01-01":"2022-12-31"]
    prepared = s.prepare(window.copy())
    d = prepared[["frac", "funding_pct", "discount"]].dropna(subset=["funding_pct"])

    print(f"FROZEN config: {FROZEN}")
    print(f"bars with funding coverage in 2020-2022: {len(d):,} of {len(prepared):,}")

    fully_confirmed = d["frac"] >= 0.999
    binds = d["discount"] < 0.999
    print(f"\nfraction of covered bars with frac==1 (fully confirmed trend): "
          f"{fully_confirmed.mean():.1%}")
    print(f"discount binds (< 1) while frac==1:      {(binds & fully_confirmed).mean():.4%}  "
          f"(must be ~0 by construction)")
    print(f"discount binds (< 1) while frac<1:        "
          f"{(binds & ~fully_confirmed).mean():.1%} of ALL covered bars, "
          f"{(binds[~fully_confirmed]).mean():.1%} of the frac<1 subset")
    print(f"mean discount overall:                    {d['discount'].mean():.4f}")
    print(f"mean discount | frac==1:                  {d.loc[fully_confirmed, 'discount'].mean():.4f}")
    print(f"mean discount | frac<1:                   {d.loc[~fully_confirmed, 'discount'].mean():.4f}")
    print(f"mean discount | frac<1 & funding_pct>=hi:  "
          f"{d.loc[(~fully_confirmed) & (d['funding_pct'] >= FROZEN['hi']), 'discount'].mean():.4f}")

    # A hard, unconditional funding-only gate (frac term removed) for
    # comparison: how much MORE would it discount frac==1 bars?
    from experiments.funding_momentum_tilt import smoothstep
    hard = 1.0 - FROZEN["weight"] * smoothstep(d["funding_pct"].to_numpy(), FROZEN["lo"], FROZEN["hi"])
    print(f"\nunconditional-gate counterfactual (same weight/lo/hi, no frac term):")
    print(f"  mean discount overall:   {hard.mean():.4f}")
    print(f"  mean discount | frac==1: {hard[fully_confirmed.to_numpy()].mean():.4f}  "
          f"(the conditioning term is the entire gap vs {d.loc[fully_confirmed, 'discount'].mean():.4f} above)")


# ----------------------------------------------------------------- causality


def causality() -> None:
    """The strict on_bar peek check, run by hand for an unregistered strategy.

    Same two-opposite-tampers procedure as run_eprocess.py's causality():
    bars after a cut are multiplied by 3 in one copy, divided by 3 in the
    other, and every decision at or before the cut must be identical.
    Restricted to bars strictly before 2023-01-01 throughout, per this
    session's instruction not to touch the holdout even structurally.

    Additionally checks the two funding-specific lookahead classes this
    strategy introduces that eprocess_regime's version does not have to:
    (a) tampering future price bars must not move the funding percentile
    or discount columns at all (they depend only on the funding series,
    frac, and the bar's own position — never future price), and (b) a
    truncated frame (bars only up to a cut) must reproduce the exact same
    target/discount/funding_pct values at-and-before the cut as the full
    frame — the check that catches a rolling window quietly looking past
    its own bar.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    pre2023 = DF.loc[:"2022-12-31"]
    assert pre2023.index[-1] < pd.Timestamp(OOS_START, tz="UTC")
    df = pre2023.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = FundingMomentumTilt(funding=FUNDING, **FROZEN)
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
    print(f"[1] on_bar decision check - tampered from bar {cut:,} of {len(df):,} "
          f"(all bars < 2023-01-01); checked bars {bars}")
    print("    FAIL - reads the future at bars " + str(bad) if bad
          else "    PASS - every decision at or before the cut is unchanged")

    pa = FundingMomentumTilt(funding=FUNDING, **FROZEN).prepare(up.copy())
    pb = FundingMomentumTilt(funding=FUNDING, **FROZEN).prepare(down.copy())
    print("\n[2] prepared-column check under price tampering (before the cut):")
    all_pass_2 = True
    for col in ("target", "frac", "discount", "funding_pct"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        ok = worst < 1e-9
        all_pass_2 &= ok
        print(f"    column {col:12s} max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("    (funding_pct/discount are expected EXACT-ZERO difference: they never\n"
          "     read price at all, only the funding series, frac, and bar position -\n"
          "     tampering price should not move them by even floating-point noise.)")
    for col in ("funding_pct", "discount"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        print(f"    [check a] {col:12s} exact-zero under price tampering: "
              f"{'PASS' if worst == 0.0 else f'FAIL ({worst:.3e})'}")

    # [3] truncated-frame check: prepare() on a shorter frame (cut+50 bars)
    # must reproduce identical values at-and-before the cut.
    trunc = df.iloc[:cut + 50].copy()
    pt = FundingMomentumTilt(funding=FUNDING, **FROZEN).prepare(trunc)
    pf = FundingMomentumTilt(funding=FUNDING, **FROZEN).prepare(df.copy())
    print("\n[3] truncated-frame check (bars only up to cut+50 vs the full frame):")
    all_pass_3 = True
    for col in ("target", "frac", "discount", "funding_pct"):
        diff = np.abs(pt[col].to_numpy()[:cut] - pf[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        ok = worst < 1e-9
        all_pass_3 &= ok
        print(f"    column {col:12s} max |difference| before the cut = {worst:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\nOVERALL: {'PASS' if not bad and all_pass_2 and all_pass_3 else 'FAIL'}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    print(f"funding: {len(FUNDING):,} settlements  "
          f"{FUNDING.index[0]:%Y-%m-%d} -> {FUNDING.index[-1]:%Y-%m-%d}", file=sys.stderr)
    cmds = {"sweep": sweep, "neighbours": neighbours, "diagnostics": diagnostics,
            "causality": causality}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_momentum_tilt.py [{'|'.join(cmds)}]")
