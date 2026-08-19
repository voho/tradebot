#!/usr/bin/env python
"""Driver for backlog B-05 — funding as a gate on kelly_regime_v4.

Splits follow ROUTINE.md step 3, adapted to the funding file's coverage
(``data/btcusdt_perp_funding_8h.csv.gz`` is 2020-01-01 .. 2023-12-31 only,
so both inner splits and the holdout sit inside that window)::

    inner-train       2020-01-01 -> 2021-12-31   fit, sweep, iterate
    inner-validation  2022-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 -> 2023-12-31   step 4 only, pre-registered
                                                  (~1 year - the funding
                                                  file's coverage ends
                                                  exactly at the standard
                                                  holdout's first year)

Usage::

    python experiments/run_funding_gate.py sweep       # step 2-3, 6 variants
    python experiments/run_funding_gate.py causality    # by-hand lookahead probe
    python experiments/run_funding_gate.py holdout      # step 4, frozen config
    python experiments/run_funding_gate.py interval     # paired bootstrap
    python experiments/run_funding_gate.py falsify      # 0.40% taker tier
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

from experiments.funding_gate import FundingGatedKellyV4  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")
OOS = ("2023-01-01", "2023-12-31")

OUT = ROOT / "reports" / "funding_gate"

N_EVALUATED = 0  # distinct configurations searched in step 2-3

# The six pre-registered variants (step 2). N (lookback) and the decile
# (quantile) are justified, not guessed: 7/14 days brackets R-16's 14-day
# forward-return window; 3 days is a fast control to show the plateau (or
# its absence) below that; the top decile (0.90) is B-05's literal wording.
# gate_floor=0.5 on the last variant is the one partial-downweight config,
# to check whether "flat" (floor 0.0) is actually the better choice or
# whether a strategy that never goes fully flat does better.
VARIANTS = {
    "hard_n3":         dict(funding_lookback_days=3.0,  gate_style="hard",   gate_floor=0.0),
    "hard_n7":         dict(funding_lookback_days=7.0,  gate_style="hard",   gate_floor=0.0),
    "hard_n14":        dict(funding_lookback_days=14.0, gate_style="hard",   gate_floor=0.0),
    "smooth_n7":       dict(funding_lookback_days=7.0,  gate_style="smooth", gate_floor=0.0),
    "smooth_n14":      dict(funding_lookback_days=14.0, gate_style="smooth", gate_floor=0.0),
    "smooth_n14_f0.5": dict(funding_lookback_days=14.0, gate_style="smooth", gate_floor=0.5),
}


def _period(strategy, market, start=None, end=None, funding=None):
    """Backtest over a date range, warmed on the bars before it (funding_study.py's pattern).

    tradebot.window.run_period does not accept a funding= kwarg; this
    reproduces its trimming logic directly over tradebot.engine.run_backtest,
    which does.
    """
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw


def line(tag, m, raw):
    print(f"  {tag:22s} final=${m.final_balance:>11,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"fills={len(raw.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"funding=${raw.funding_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# ----------------------------------------------------------------------- sweep


def sweep() -> None:
    """Step 2-3. Six variants vs plain v4 and buy_and_hold, both inner splits.

    A configuration is (funding_lookback_days, gate_style, gate_floor) - six
    of them. Scoring one on a second split or a second market is another
    backtest, not another trial (the R-28/R-31 convention) - so each
    variant name is counted once, not once per split/market combination.
    """
    global N_EVALUATED
    evaluated = set()
    rows = []
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        print(f"\n{'=' * 90}\n{split_name}  {start} -> {end}\n{'=' * 90}")

        print("\n  futures 5x, funding CHARGED:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            m, raw = _period(get_strategy(name), FUTURES, start, end, funding=REAL)
            line(name, m, raw)
            rows.append(dict(split=split_name, market="futures", variant=name,
                             final=m.final_balance, dd=m.max_drawdown_pct,
                             sharpe=m.sharpe, fills=len(raw.fills),
                             fees=m.fees_paid, funding=raw.funding_paid))
        for vname, kw in VARIANTS.items():
            s = FundingGatedKellyV4(funding=REAL, **kw)
            m, raw = _period(s, FUTURES, start, end, funding=REAL)
            line(vname, m, raw)
            rows.append(dict(split=split_name, market="futures", variant=vname,
                             final=m.final_balance, dd=m.max_drawdown_pct,
                             sharpe=m.sharpe, fills=len(raw.fills),
                             fees=m.fees_paid, funding=raw.funding_paid))
            evaluated.add(vname)

        print("\n  spot 1x (no funding applies; reported as a check):")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            m, raw = _period(get_strategy(name), SPOT, start, end)
            line(name, m, raw)
            rows.append(dict(split=split_name, market="spot", variant=name,
                             final=m.final_balance, dd=m.max_drawdown_pct,
                             sharpe=m.sharpe, fills=len(raw.fills),
                             fees=m.fees_paid, funding=raw.funding_paid))
        for vname, kw in VARIANTS.items():
            s = FundingGatedKellyV4(funding=REAL, **kw)
            m, raw = _period(s, SPOT, start, end)
            line(vname, m, raw)
            rows.append(dict(split=split_name, market="spot", variant=vname,
                             final=m.final_balance, dd=m.max_drawdown_pct,
                             sharpe=m.sharpe, fills=len(raw.fills),
                             fees=m.fees_paid, funding=raw.funding_paid))
            # Counted once per distinct configuration, not per market: the
            # futures/funding-charged pass above already counted it
            # (R-28/R-31 convention - a second market is another backtest,
            # not another trial).

    N_EVALUATED = len(evaluated)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "sweep.csv", index=False)
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep.csv'}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict lookahead probe, by hand - experiments get no CI protection.

    Same two-opposite-tampers procedure as tests/test_causality_strict.py
    and R-28/R-31's run_eprocess.py / run_matched_risk.py causality()
    checks: bars after a cut are multiplied by 3 in one copy and divided
    by 3 in the other, and every decision at or before the cut must be
    identical. The column comparison (target, funding_gate_mult) is the
    part a truncation test cannot see - it catches a full-series fit
    (a mean/std/quantile taken over the whole series and applied to early
    rows), which is exactly the class of bug an expanding quantile is
    designed to avoid, so this is also the check that the expanding
    quantile was implemented correctly.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame, kw):
        s = FundingGatedKellyV4(funding=REAL, **kw)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    ok = True
    for vname, kw in VARIANTS.items():
        bad = [b for b, oa, ob in zip(bars, decisions(up, kw), decisions(down, kw))
               if oa != ob]
        pa = FundingGatedKellyV4(funding=REAL, **kw).prepare(up.copy())
        pb = FundingGatedKellyV4(funding=REAL, **kw).prepare(down.copy())
        worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut]
                                           - pb[c].to_numpy()[:cut])))
                    for c in ("target", "funding_gate_mult"))
        good = not bad and worst < 1e-12
        ok &= good
        print(f"  {vname:16s} orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- holdout


FROZEN = dict(funding_lookback_days=7.0, gate_style="hard", gate_floor=0.0)
FROZEN_NAME = "hard_n7"


def holdout() -> None:
    """Step 4. Frozen config only; decision rule is in the ledger report."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"HOLDOUT {OOS[0]} -> {OOS[1]}  (funding file's coverage ends here)")

    print("\n  futures 5x, funding CHARGED:")
    for name, s in (("buy_and_hold", get_strategy("buy_and_hold")),
                    ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                    (f"funding_gate ({FROZEN_NAME})",
                     FundingGatedKellyV4(funding=REAL, **FROZEN))):
        m, raw = _period(s, FUTURES, *OOS, funding=REAL)
        line(name, m, raw)
        rows.append(dict(market="futures", funding="charged", variant=name,
                         final=m.final_balance, dd=m.max_drawdown_pct,
                         sharpe=m.sharpe, fills=len(raw.fills),
                         fees=m.fees_paid, funding_paid=raw.funding_paid))

    print("\n  futures 5x, funding-FREE (reference, upper bound):")
    for name, s in (("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                    (f"funding_gate ({FROZEN_NAME})",
                     FundingGatedKellyV4(funding=REAL, **FROZEN))):
        m, raw = _period(s, FUTURES, *OOS)
        line(name, m, raw)
        rows.append(dict(market="futures", funding="free", variant=name,
                         final=m.final_balance, dd=m.max_drawdown_pct,
                         sharpe=m.sharpe, fills=len(raw.fills),
                         fees=m.fees_paid, funding_paid=raw.funding_paid))

    print("\n  spot 1x (no funding; check):")
    for name, s in (("buy_and_hold", get_strategy("buy_and_hold")),
                    ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                    (f"funding_gate ({FROZEN_NAME})",
                     FundingGatedKellyV4(funding=REAL, **FROZEN))):
        m, raw = _period(s, SPOT, *OOS)
        line(name, m, raw)
        rows.append(dict(market="spot", funding="n/a", variant=name,
                         final=m.final_balance, dd=m.max_drawdown_pct,
                         sharpe=m.sharpe, fills=len(raw.fills),
                         fees=m.fees_paid, funding_paid=raw.funding_paid))

    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# -------------------------------------------------------------------- interval


def interval() -> None:
    """Paired stationary block bootstrap, funding_gate vs plain v4, on the holdout.

    30-day mean block, 2,000 resamples, on daily returns - the project's
    standard settings (R-29/R-30/R-31), not reinvented here. Futures 5x
    with funding charged, since that is the decision-relevant market for
    this idea.
    """
    from tradebot.inference import (annualized_sharpe, daily_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    gated = FundingGatedKellyV4(funding=REAL, **FROZEN)
    v4 = get_strategy("kelly_regime_v4")

    rows = []
    for mname, market, funding in (("futures_funded", FUTURES, REAL),
                                   ("spot", SPOT, None)):
        res_gated = _period(gated, market, *OOS, funding=funding)[1]
        res_v4 = _period(v4, market, *OOS, funding=funding)[1]
        a = daily_returns(res_gated.equity).to_numpy()
        b = daily_returns(res_v4.equity).to_numpy()
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
        print(f"\n{mname}: paired funding_gate({FROZEN_NAME}) - kelly_regime_v4, "
              f"holdout ({n} daily observations)")
        for stat_name, stat in (("Δ log growth", total_log_return),
                                ("Δ annualized Sharpe", annualized_sharpe)):
            r = paired_bootstrap(a, b, stat, indices=idx)
            mark = "beats v4" if r.diff.lo > 0 else (
                "worse than v4" if r.diff.hi < 0 else "indistinguishable")
            print(f"  {stat_name:22s} {r.diff.point:>+8.3f} "
                  f"[{r.diff.lo:>+8.3f}, {r.diff.hi:>+8.3f}]  "
                  f"P(gate>v4)={r.p_positive:.2f}  {mark}")
            rows.append(dict(market=mname, stat=stat_name, gate=r.stat_a,
                             v4=r.stat_b, diff=r.diff.point, lo=r.diff.lo,
                             hi=r.diff.hi, p_positive=r.p_positive,
                             significant=r.significant))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intervals.csv", index=False)
    print(f"\nwritten: {OUT / 'intervals.csv'}")


# ---------------------------------------------------------------------- falsify


def falsify() -> None:
    """Pre-registered falsification: does the gate still beat v4 at the 0.40% taker tier?

    B-05 is explicitly a low-turnover idea; R-12 is the standing warning
    that higher turnover is where this project's ideas go to die. If the
    gate's extra trading (funding-driven exits/re-entries on top of the
    vote's own) costs more than it saves at Bitstamp's real entry tier,
    the idea is dead even if it looks fine at the table's usual 0.10%.
    """
    print("HOLDOUT 2023, spot, at the Bitstamp 0.40% taker tier (0.10% shown too):")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"\n  {label}")
        for name, s in (("buy_and_hold", get_strategy("buy_and_hold")),
                        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                        (f"funding_gate ({FROZEN_NAME})",
                         FundingGatedKellyV4(funding=REAL, **FROZEN))):
            m, raw = _period(s, market, *OOS)
            line(f"    {name}", m, raw)

    print("\nHOLDOUT 2023, futures 5x with funding CHARGED, at both taker tiers:")
    for lev_fee, label in ((0.0005, "0.05% (table assumption)"),
                           (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.futures(leverage=5.0, fee_rate=lev_fee)
        print(f"\n  {label}")
        for name, s in (("buy_and_hold", get_strategy("buy_and_hold")),
                        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                        (f"funding_gate ({FROZEN_NAME})",
                         FundingGatedKellyV4(funding=REAL, **FROZEN))):
            m, raw = _period(s, market, *OOS, funding=REAL)
            line(f"    {name}", m, raw)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    print(f"funding: {len(REAL):,} settlements  {REAL.index[0]:%Y-%m-%d} -> "
          f"{REAL.index[-1]:%Y-%m-%d}", file=sys.stderr)
    cmds = {"sweep": sweep, "causality": causality, "holdout": holdout,
            "interval": interval, "falsify": falsify}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate.py [{'|'.join(cmds)}]")
