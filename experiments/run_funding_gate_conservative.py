#!/usr/bin/env python
"""Driver for R-33 "conservative" — FundingDecileGate.

INNER SPLIT ONLY. This branch never reads a date on or after 2023-01-01;
that is the pre-registered holdout (docs/LEDGER.md, R-33) and is reserved
for step 4, run separately after both parallel branches report in.

    inner-train       2020-01-01 -> 2021-12-31   design (no fitting beyond
                                                   the a-priori frozen knobs)
    inner-validation  2022-01-01 -> 2022-12-31   plateau/neighbourhood check
    holdout           2023-01-01 -> 2023-12-31   NOT RUN by this file

Usage::

    python experiments/run_funding_gate_conservative.py inspect
    python experiments/run_funding_gate_conservative.py causality
    python experiments/run_funding_gate_conservative.py sweep
    python experiments/run_funding_gate_conservative.py neighbours
    python experiments/run_funding_gate_conservative.py feetier
    python experiments/run_funding_gate_conservative.py windows
    python experiments/run_funding_gate_conservative.py all
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

from experiments.funding_gate_conservative import (  # noqa: E402
    FundingDecileGate, gate_multiplier)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")  # 2020-01-01 .. 2023-12-31, settlement-indexed
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")
# The hard boundary this file must never cross. Every _period() call is
# asserted against it below, not just documented.
HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")

# Frozen before any inner-validation number was read (R-33 pre-registration).
FROZEN = dict(window_days=180, gate_in=0.90, gate_out=0.75)

N_EVALUATED = 0  # every backtest step 3 (sweep + neighbours + windows) runs


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _period(strategy, market, start, end, funding=None):
    """Copy of scripts/funding_study.py's _period(): warm on bars before the
    window, trade only inside it, funding charged as a first-class cost.

    Hard-guarded against the holdout: refuses any start/end >= 2023-01-01.
    """
    if start is not None:
        assert _as_utc(start) < HOLDOUT_START, f"refusing to touch holdout: start={start}"
    if end is not None:
        assert _as_utc(end) < HOLDOUT_START, f"refusing to touch holdout: end={end}"
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def ev(strategy, market, start, end, tag="", funding=None, count=True):
    """One backtest, one printed line, counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    m, funding_paid = _period(strategy, market, start, end, funding=funding)
    print(f"  {tag or strategy.name:38s} {market.name:12s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"trades={m.num_trades:>4d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} funding=${funding_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m, funding_paid


# --------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the gate multiplier does, before any backtest: 2020-2021 (inner-train)."""
    lo = int(DF.index.searchsorted(TRAIN[0]))
    hi = int(DF.index.searchsorted(TRAIN[1], side="right"))
    idx = DF.index[lo:hi]
    mult, pct = gate_multiplier(REAL, idx, **FROZEN)

    n_known = int(np.isfinite(pct.to_numpy()).sum())
    print(f"{TRAIN[0]} .. {TRAIN[1]}  ({len(idx):,} bars)")
    print(f"  bars with a defined percentile (post warmup): {n_known:,} "
          f"({100 * n_known / len(idx):.1f}%)")
    print(f"  bars with the gate SHUT (mult=0):              "
          f"{100 * (mult < 0.5).mean():.1f}%")
    print(f"  bars with the gate OPEN (mult=1):               "
          f"{100 * (mult >= 0.5).mean():.1f}%")

    transitions = np.diff(mult) != 0
    shut_events = int(((np.diff(mult) < 0)).sum())
    open_events = int(((np.diff(mult) > 0)).sum())
    print(f"  shut transitions (open->shut): {shut_events}")
    print(f"  reopen transitions (shut->open): {open_events}")
    print(f"  total transitions: {int(transitions.sum())}")

    finite = pct.dropna()
    if len(finite):
        print(f"  trailing funding percentile quantiles once defined: "
              + "  ".join(f"q{int(q*100)}={finite.quantile(q):.2f}"
                          for q in (0.05, 0.25, 0.5, 0.75, 0.95)))


# ------------------------------------------------------------------ causality


def causality() -> None:
    """Two orthogonal tamper checks - MUST PASS before any number is trusted.

    (1) Price tamper (the standard two-opposite-tampers check, copied from
        run_eprocess.py): the mechanism inherited from kelly_regime_v4 must
        still ignore price bars after the cut. The funding-gate logic does
        not itself read price, so this mainly re-confirms v4's own
        causality survives the subclass unchanged.
    (2) Funding tamper: the actually load-bearing check for this class.
        The committed funding series is tampered *after* a cut settlement
        in two opposite directions; every bar's gate multiplier and target
        at or before the cut must be identical. This is the check that
        would catch a lookahead in the shift/rolling-percentile logic.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    # Deliberately drawn from inside the inner split (2020-06-01..2022-12-31,
    # never touching 2023): the funding-tamper check below needs settlements
    # AFTER the cut to actually exist (the committed series runs through
    # 2023-12-31), which a window drawn from the tail of the full price
    # series (2024-2026) would not give it - that combination would tamper
    # only the last handful of already-realized 2023 settlements and pass
    # vacuously rather than because the mechanism is causal.
    lo = int(DF.index.searchsorted("2020-06-01"))
    hi = int(DF.index.searchsorted("2022-12-31", side="right"))
    df = DF.iloc[lo:hi].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    # ---- (1) price tamper ----------------------------------------------
    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = FundingDecileGate(funding=REAL, **FROZEN)
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
    print(f"(1) PRICE tamper - tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("    FAIL - reads the future price at bars " + str(bad) if bad
          else "    PASS - every decision at or before the cut is unchanged")

    pa = FundingDecileGate(funding=REAL, **FROZEN).prepare(up.copy())
    pb = FundingDecileGate(funding=REAL, **FROZEN).prepare(down.copy())
    for col in ("target",):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        print(f"    column {col:9s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-9 else 'FAIL'}")

    # ---- (2) funding tamper ---------------------------------------------
    # Cut the funding series at a settlement well inside the tampered bar
    # window (so its downstream effect, if any leaked, would show up at the
    # `bars` decision points), tamper everything from there on in two
    # opposite directions, and require every decision/target at or before
    # the cut TIMESTAMP to be identical.
    fcut_ts = df.index[cut]
    fcut_pos = int(REAL.index.searchsorted(fcut_ts))
    fcut_pos = min(max(fcut_pos, 1), len(REAL) - 1)
    f_up, f_down = REAL.copy(), REAL.copy()
    f_up.iloc[fcut_pos:] = f_up.iloc[fcut_pos:] * 3.0 + 0.01
    f_down.iloc[fcut_pos:] = f_down.iloc[fcut_pos:] / 3.0 - 0.01

    def decisions_funding(funding):
        s = FundingDecileGate(funding=funding, **FROZEN)
        prepared = s.prepare(df.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out, prepared

    fa, prep_a = decisions_funding(f_up)
    fb, prep_b = decisions_funding(f_down)
    fbad = [bar for bar, oa, ob in zip(bars, fa, fb) if oa != ob]
    print(f"\n(2) FUNDING tamper - tampered from settlement {REAL.index[fcut_pos]} "
          f"(bar {cut:,} timestamp {fcut_ts}); checked bars {bars}")
    print("    FAIL - reads future funding at bars " + str(fbad) if fbad
          else "    PASS - every decision at or before the cut is unchanged")

    diff = np.abs(prep_a["target"].to_numpy()[:cut] - prep_b["target"].to_numpy()[:cut])
    worst = float(np.nanmax(diff)) if len(diff) else 0.0
    print(f"    column target    max |difference| before the cut = {worst:.3e}"
          f"  {'PASS' if worst < 1e-9 else 'FAIL'}")


# ----------------------------------------------------------------------- sweep


def _sweep_one(start, end, split) -> None:
    print(f"\n{split}  {start} .. {end}")
    ev(get_strategy("buy_and_hold"), SPOT, start, end,
       tag="buy_and_hold (spot, no funding)", funding=None)
    ev(get_strategy("kelly_regime_v4"), FUTURES, start, end,
       tag="kelly_regime_v4 (futures, funding)", funding=REAL)
    ev(FundingDecileGate(funding=REAL, **FROZEN), FUTURES, start, end,
       tag="FundingDecileGate (frozen, futures, funding)", funding=REAL)


def sweep() -> None:
    _sweep_one(*TRAIN, "INNER-TRAIN")
    _sweep_one(*VALID, "INNER-VALIDATION")
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# ------------------------------------------------------------------ neighbours


def neighbours() -> None:
    """Plateau check, inner-validation (2022) ONLY. Center config already
    counted in sweep(); this adds the 6 neighbours (3 gate_in x fixed
    gate_out, 3 gate_out x fixed gate_in)."""
    print(f"\nINNER-VALIDATION neighbourhood  {VALID[0]} .. {VALID[1]}")
    for gi in (0.85, 0.90, 0.95):
        tag = f"gate_in={gi:.2f} (gate_out=0.75)"
        if gi == FROZEN["gate_in"]:
            print(f"  {tag:38s} == frozen center, see sweep()")
            continue
        ev(FundingDecileGate(funding=REAL, window_days=180, gate_in=gi, gate_out=0.75),
           FUTURES, *VALID, tag=tag, funding=REAL)
    for go in (0.60, 0.75, 0.85):
        tag = f"gate_out={go:.2f} (gate_in=0.90)"
        if go == FROZEN["gate_out"]:
            print(f"  {tag:38s} == frozen center, see sweep()")
            continue
        ev(FundingDecileGate(funding=REAL, window_days=180, gate_in=0.90, gate_out=go),
           FUTURES, *VALID, tag=tag, funding=REAL)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# --------------------------------------------------------------------- feetier


def feetier() -> None:
    """Inner-train + inner-validation at the 0.40% taker fee tier.

    MarketSpec.futures() takes fee_rate directly, so this is a one-line
    parametrization - no engine change needed.
    """
    futures_40bp = MarketSpec.futures(leverage=5.0, fee_rate=0.004)
    print(f"\nFee tier 0.40% (vs the default {FUTURES.fee_rate:.2%}):")
    for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
        print(f"\n{split}  {start} .. {end}  @ 0.40% taker")
        ev(get_strategy("kelly_regime_v4"), futures_40bp, start, end,
           tag="kelly_regime_v4 (0.40% fee)", funding=REAL)
        ev(FundingDecileGate(funding=REAL, **FROZEN), futures_40bp, start, end,
           tag="FundingDecileGate (0.40% fee)", funding=REAL)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# --------------------------------------------------------------------- windows


def windows(trials: int = 15, seed: int = 20260818) -> None:
    """Mini Monte Carlo: ~15 random ~6-month windows drawn from inside
    2020-01-01..2022-12-31 (never touching 2023), identical windows for
    FundingDecileGate and kelly_regime_v4 (paired).

    This resamples the FROZEN config on new data; it does not fit or select
    anything, so it is not a new trial in the deflated-Sharpe sense (it is
    counted in N_EVALUATED anyway, for a transparent tally, but should be
    read as evaluation-on-resampled-data, not search).
    """
    global N_EVALUATED
    lo_bound = int(DF.index.searchsorted("2020-01-01"))
    hi_bound = int(DF.index.searchsorted("2022-12-31", side="right"))
    contenders = [("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("FundingDecileGate", FundingDecileGate(funding=REAL, **FROZEN))]
    warmup = max(s.warmup for _, s in contenders) + 10

    rng = np.random.default_rng(seed)
    six_months = int(182 * 288)
    specs = []
    for _ in range(trials):
        start = int(rng.integers(lo_bound + warmup, hi_bound - six_months))
        length = int(rng.integers(150, 213) * 288)  # ~5-7 months
        length = min(length, hi_bound - start)
        specs.append((start, length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        # Guard: window (with warmup prefix) must never reach 2023-01-01.
        end_ts = DF.index[start + length - 1]
        assert end_ts < HOLDOUT_START, f"window {k} reaches the holdout: {end_ts}"
        window = DF.iloc[start - warmup: start + length]
        for name, strat in contenders:
            res = run_backtest(strat, window, FUTURES, 1_000.0,
                               trade_start=warmup, funding=REAL, data_label=LABEL)
            N_EVALUATED += 1
            eq = res.equity.to_numpy(dtype=float)
            base, seg = eq[warmup], eq[warmup:]
            ok = np.isfinite(base) and base > 0
            rows.append({
                "trial": k, "strategy": name,
                "start_ts": DF.index[start], "end_ts": end_ts,
                "final_balance": float(seg[-1]) if ok else float("nan"),
                "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                "funding_paid": res.funding_paid,
                "liquidated": res.liquidated,
            })
    res = pd.DataFrame(rows)

    base = res[res.strategy == "kelly_regime_v4"].set_index("trial")
    gate = res[res.strategy == "FundingDecileGate"].set_index("trial")
    d_final = gate["final_balance"] - base["final_balance"]
    d_dd = gate["max_dd_pct"] - base["max_dd_pct"]
    d_fund = gate["funding_paid"] - base["funding_paid"]

    print(f"{trials} random ~6-month windows drawn from 2020-01-01..2022-12-31 "
          f"(seed={seed}), paired (identical windows for both strategies):\n")
    print(f"  kelly_regime_v4     median final=${base.final_balance.median():>9,.0f}  "
          f"median DD={base.max_dd_pct.median():>5.1f}%  "
          f"median funding paid=${base.funding_paid.median():>7,.0f}")
    print(f"  FundingDecileGate   median final=${gate.final_balance.median():>9,.0f}  "
          f"median DD={gate.max_dd_pct.median():>5.1f}%  "
          f"median funding paid=${gate.funding_paid.median():>7,.0f}\n")
    print(f"  gate has HIGHER final balance in {100 * (d_final > 0).mean():.0f}% of windows "
          f"(median delta ${d_final.median():>+8,.0f})")
    print(f"  gate has SHALLOWER max drawdown in {100 * (d_dd < 0).mean():.0f}% of windows "
          f"(median delta {d_dd.median():>+6.1f}pp)")
    print(f"  gate pays LESS funding in {100 * (d_fund < 0).mean():.0f}% of windows "
          f"(median delta ${d_fund.median():>+8,.0f})")
    print(f"  liquidation rate: kelly_regime_v4={base.liquidated.mean():.0%}  "
          f"FundingDecileGate={gate.liquidated.mean():.0%}")
    print(f"\nconfigurations evaluated so far: {N_EVALUATED} "
          f"(re-evaluations of the FROZEN config on resampled data, not new trials)")


# ------------------------------------------------------------------------- all


def all_() -> None:
    inspect()
    print()
    causality()
    print()
    sweep()
    print()
    neighbours()
    print()
    feetier()
    print()
    windows()
    print(f"\nTOTAL configurations evaluated (sweep + neighbours + feetier + windows): "
          f"{N_EVALUATED}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    print(f"funding: {len(REAL):,} settlements  {REAL.index[0]:%Y-%m-%d} -> "
          f"{REAL.index[-1]:%Y-%m-%d}", file=sys.stderr)
    cmds = {"inspect": inspect, "causality": causality, "sweep": sweep,
            "neighbours": neighbours, "feetier": feetier, "windows": windows,
            "all": all_}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_conservative.py [{'|'.join(cmds)}]")
