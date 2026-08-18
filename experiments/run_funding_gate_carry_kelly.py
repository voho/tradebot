#!/usr/bin/env python
"""Driver for R-33's novel branch — CarryAdjustedKelly.

Redefined split, forced by funding-data availability (docs/LEDGER.md,
R-33 pre-registration) — a narrower window INSIDE the project's usual
split, not a redefinition of it::

    inner-train       2020-01-01 -> 2021-12-31   design, no fitting beyond
                                                  what is stated in the
                                                  pre-registration
    inner-validation  2022-01-01 -> 2022-12-31   plateau/neighbourhood check
    holdout           2023-01-01 -> 2023-12-31   step 4, pre-registered,
                                                  touched once — NOT this file

This driver enforces that boundary in code (see ``_guard_dates`` /
``HOLDOUT_START_IDX``): every entry point refuses any date on or after
2023-01-01, so an accidental holdout peek fails loudly instead of quietly.

Usage::

    python experiments/run_funding_gate_carry_kelly.py inspect
    python experiments/run_funding_gate_carry_kelly.py causality
    python experiments/run_funding_gate_carry_kelly.py sweep
    python experiments/run_funding_gate_carry_kelly.py neighbours
    python experiments/run_funding_gate_carry_kelly.py feetier
    python experiments/run_funding_gate_carry_kelly.py windows
    python experiments/run_funding_gate_carry_kelly.py regime_check
    python experiments/run_funding_gate_carry_kelly.py all
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

from experiments.funding_gate_carry_kelly import CarryAdjustedKelly  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")  # 2020-01-01 .. 2023-12-31, settlement-indexed
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")

FROZEN = dict(funding_halflife_days=30.0)  # fixed in the pre-registration

N_EVALUATED = 0  # every backtest counted in step 3 (sweep + neighbours)

# ---------------------------------------------------------------- holdout guard

HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")
HOLDOUT_START_IDX = int(DF.index.searchsorted(HOLDOUT_START))


def _guard_dates(*dates) -> None:
    """Hard stop: this branch's job ends at inner-validation (2022-12-31)."""
    for d in dates:
        if d is None:
            continue
        ts = pd.Timestamp(d)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts >= HOLDOUT_START:
            raise RuntimeError(
                f"refusing to run: {d} is on/after the pre-registered holdout "
                "2023-01-01 - this branch is inner-validation only")


def _guard_bar_range(lo: int, hi: int) -> None:
    if hi > HOLDOUT_START_IDX:
        raise RuntimeError(
            f"refusing to run: bar range [{lo}, {hi}) crosses the pre-registered "
            f"holdout at bar {HOLDOUT_START_IDX} (2023-01-01)")


# -------------------------------------------------------------------- backtest


def _period(strategy, market, start, end, funding=None):
    """One warm-then-trade backtest; the ``_period()`` pattern from
    ``scripts/funding_study.py``. Funding is a first-class cost, not a
    footnote: charged whenever ``funding`` is passed and the market pays it.
    """
    _guard_dates(start, end)
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    _guard_bar_range(lo, hi)
    strategy_warmup = strategy.warmup
    pre = min(lo, strategy_warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def ev(strategy, start, end, market, funding=None, tag="", count=True):
    """One backtest, one line, counted in N_EVALUATED when it's a real trial."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    m, funding_paid = _period(strategy, market, start, end, funding=funding)
    print(f"  {tag or strategy.name:30s} {market.name:11s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"trades={m.num_trades:>4d} funding=${funding_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")
    return m, funding_paid


# --------------------------------------------------------------------- inspect


def inspect() -> None:
    """Sanity check over 2020-2021: what does the funding term actually shave off?"""
    start, end = TRAIN
    _guard_dates(start, end)
    s = CarryAdjustedKelly(funding=REAL, **FROZEN)
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    _guard_bar_range(lo, hi)
    pre = min(lo, s.warmup)
    prepared = s.prepare(DF.iloc[lo - pre: hi].copy())
    window = prepared.iloc[pre:]

    ratio = (window["funding_cost"] / window["vol"].replace(0.0, np.nan)).dropna()
    binds = window["funding_cost"] > 0
    binds_meaningfully = ratio > 0.05

    print(f"CarryAdjustedKelly funding term, {start}..{end} (n={len(window):,} bars)")
    print(f"  target_vol (frozen)            {s.target_vol:.3f}")
    print(f"  funding_halflife_days (frozen) {s.funding_halflife_days:g}")
    print("\nfunding_cost / vol  (how much target_vol is shaved by):")
    print(f"  mean                {ratio.mean():.4f}")
    print(f"  median              {ratio.median():.4f}")
    print(f"  q95                 {ratio.quantile(0.95):.4f}")
    print(f"  max                 {ratio.max():.4f}")
    print(f"\nfraction of bars where funding_cost > 0 (binds at all):        "
          f"{binds.mean():.1%}")
    print(f"fraction of bars where funding_cost/vol > 0.05 (binds meaningfully): "
          f"{binds_meaningfully.mean():.1%}")
    print(f"\nmean eff_target_vol            {window['eff_target_vol'].mean():.4f}"
          f"  (vs frozen target_vol {s.target_vol:.3f})")
    print(f"bars where eff_target_vol < target_vol: "
          f"{(window['eff_target_vol'] < s.target_vol - 1e-12).mean():.1%}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """Two-opposite-tampers check, the run_eprocess.py pattern, by hand.

    ``CarryAdjustedKelly`` is not registered, so it gets none of
    ``tests/test_causality_strict.py``'s automatic protection. Bars after a
    cut are multiplied by 3 in one copy and divided by 3 in the other;
    every decision (order) at or before the cut, and every prepared column
    value before the cut, must be identical between the two copies.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    # A funding series that also has real data after the cut, so a lookahead
    # bug in the funding-side EWM/ffill has something to actually leak.
    funding = REAL if REAL is not None else None

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = CarryAdjustedKelly(funding=funding, **FROZEN)
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
    print(f"tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("FAIL - reads the future at bars " + str(bad) if bad
          else "PASS - every order decision at or before the cut is unchanged")

    # A prepared column computed over the whole series can move at rows
    # before the cut even when no order changes (e.g. an ewm/rolling stat
    # leaking across the cut) - check target AND the funding-adjustment
    # intermediates directly, which is exactly the R-31-warned bug class.
    pa = CarryAdjustedKelly(funding=funding, **FROZEN).prepare(up.copy())
    pb = CarryAdjustedKelly(funding=funding, **FROZEN).prepare(down.copy())
    all_ok = True
    for col in ("target", "eff_target_vol", "funding_cost", "vol", "frac_vote"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        ok = worst < 1e-9
        all_ok &= ok
        print(f"  column {col:15s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if ok else 'FAIL'}")
    print("\nOVERALL: " + ("PASS" if not bad and all_ok else "FAIL"))


# --------------------------------------------------------------------- sweep


def _benchmarks(start, end, tag):
    print(f"\n{tag}:")
    ev(get_strategy("buy_and_hold"), start, end, market=SPOT, tag="buy_and_hold",
       count=False)
    ev(get_strategy("kelly_regime_v4"), start, end, market=FUTURES, funding=REAL,
       tag="kelly_regime_v4", count=False)
    ev(CarryAdjustedKelly(funding=REAL, **FROZEN), start, end, market=FUTURES,
       funding=REAL, tag="CarryAdjustedKelly (frozen)")


def sweep() -> None:
    for (start, end), split in ((TRAIN, "INNER-TRAIN 2020-2021"),
                                (VALID, "INNER-VALIDATION 2022")):
        _benchmarks(start, end, split)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# ------------------------------------------------------------------ neighbours


def neighbours() -> None:
    """Plateau, not peak (P4): sweep funding_halflife_days on inner-validation only.

    The 30-day case is already counted in sweep(); this adds the two new
    ones (14d, 60d).
    """
    print("\nINNER-VALIDATION 2022 / futures 5x, funding_halflife_days sweep:")
    for hl in (14.0, 30.0, 60.0):
        s = CarryAdjustedKelly(funding=REAL, funding_halflife_days=hl)
        ev(s, *VALID, market=FUTURES, funding=REAL,
           tag=f"CarryAdjustedKelly hl={hl:g}d", count=(hl != 30.0))
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


# -------------------------------------------------------------------- feetier


def feetier() -> None:
    """Fee-tier robustness: does the ranking hold at Bitstamp's 0.40% taker tier?

    ``MarketSpec.futures(fee_rate=...)`` parametrizes the taker fee
    directly (see tradebot/broker.py), so this is straightforward.
    """
    futures_hi = MarketSpec.futures(leverage=5.0, fee_rate=0.004)
    for (start, end), split in ((TRAIN, "INNER-TRAIN 2020-2021"),
                                (VALID, "INNER-VALIDATION 2022")):
        print(f"\n{split} / futures 5x @ 0.40% taker fee:")
        ev(get_strategy("kelly_regime_v4"), start, end, market=futures_hi,
           funding=REAL, tag="kelly_regime_v4", count=False)
        ev(CarryAdjustedKelly(funding=REAL, **FROZEN), start, end, market=futures_hi,
           funding=REAL, tag="CarryAdjustedKelly (frozen)", count=False)


# --------------------------------------------------------------------- windows


def windows(trials: int = 15, seed: int = 20260818) -> None:
    """Mini Monte Carlo: ~15 random ~6-month windows inside 2020-01-01..2022-12-31.

    Identical windows for CarryAdjustedKelly and kelly_regime_v4 (paired).
    Re-evaluates the FROZEN config on resampled data - not a new trial in
    the deflated-Sharpe sense, since nothing is fit or selected here.
    """
    warmup = max(CarryAdjustedKelly(funding=REAL, **FROZEN).warmup,
                 get_strategy("kelly_regime_v4").warmup) + 10
    # The TRADED portion of every window must start at/after 2020-01-01 -
    # real funding data begins there, and before it CarryAdjustedKelly's
    # funding_cost is identically 0 (fillna(0.0) with no settlements to
    # ffill from), which would silently collapse it onto kelly_regime_v4
    # and dilute the paired comparison with trivial zero deltas.
    real_funding_start_idx = int(DF.index.searchsorted("2020-01-01"))
    lo = max(warmup, real_funding_start_idx)
    hi = HOLDOUT_START_IDX  # never touch 2023+
    length_lo, length_hi = int(150 * 288), int(210 * 288)  # ~5-7 months, centered on 6

    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(length_lo, length_hi))
        start = int(rng.integers(lo, hi - length))
        specs.append((start, length))

    contenders = [("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("CarryAdjustedKelly", CarryAdjustedKelly(funding=REAL, **FROZEN))]
    rows = []
    for k, (start, length) in enumerate(specs, 1):
        # start was drawn from [lo, hi - length), so start + length < hi ==
        # HOLDOUT_START_IDX by construction; guarded again for safety.
        _guard_bar_range(start - warmup, start + length)
        window = DF.iloc[start - warmup: start + length]
        for name, strat in contenders:
            raw = run_backtest(strat, window, FUTURES, 1_000.0,
                               trade_start=warmup, funding=REAL, data_label=LABEL)
            m = compute_metrics(raw)
            rows.append({"trial": k, "strategy": name,
                         "final_balance": m.final_balance,
                         "max_dd_pct": m.max_drawdown_pct,
                         "funding_paid": raw.funding_paid})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    res = pd.DataFrame(rows)

    a = res[res.strategy == "CarryAdjustedKelly"].set_index("trial")
    b = res[res.strategy == "kelly_regime_v4"].set_index("trial")
    d_bal = a["final_balance"] - b["final_balance"]
    d_dd = a["max_dd_pct"] - b["max_dd_pct"]
    d_fund = a["funding_paid"] - b["funding_paid"]

    print(f"\n{trials} random ~6-month windows inside 2020-01-01..2022-12-31 "
          f"(seed={seed}), paired CarryAdjustedKelly vs kelly_regime_v4:\n")
    print(f"  higher final balance:  {(d_bal > 0).mean():>5.0%}  "
          f"(median delta {d_bal.median():>+9,.0f})")
    print(f"  shallower max DD:      {(d_dd < 0).mean():>5.0%}  "
          f"(median delta {d_dd.median():>+6.1f}pp)")
    print(f"  less funding paid:     {(d_fund < 0).mean():>5.0%}  "
          f"(median delta {d_fund.median():>+9,.0f})")


# ---------------------------------------------------------------- regime_check


def regime_check() -> None:
    """KEY DIAGNOSTIC: does the funding derate bind disproportionately in up-trends?

    R-33's pre-registration names this as the single most likely failure
    mode (R-08 repeating): funding is richest exactly when the anchor vote
    is most bullish (R-10's inverse-leverage-effect states), so a
    mechanically correct carry adjustment could de-lever into precisely
    the highest-forward-Sharpe regime.
    """
    start, end = TRAIN
    _guard_dates(start, end)
    s = CarryAdjustedKelly(funding=REAL, **FROZEN)
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    _guard_bar_range(lo, hi)
    pre = min(lo, s.warmup)
    prepared = s.prepare(DF.iloc[lo - pre: hi].copy())
    window = prepared.iloc[pre:]

    binds = (window["funding_cost"] > 0).astype(float)
    binds_hard = (window["funding_cost"] / window["vol"].replace(0.0, np.nan) > 0.05)
    frac = window["frac_vote"]

    corr_binds = binds.corr(frac)
    corr_mag = window["funding_cost"].corr(frac)
    fully_bullish = frac == 1.0

    print(f"regime_check, {start}..{end} (n={len(window):,} bars)\n")
    print(f"  corr(funding_cost > 0, anchor-vote fraction)   {corr_binds:+.3f}")
    print(f"  corr(funding_cost magnitude, anchor-vote frac) {corr_mag:+.3f}")
    print(f"\n  mean anchor-vote fraction overall:              {frac.mean():.3f}")
    print(f"  mean anchor-vote fraction | funding_cost > 0:   "
          f"{frac[window['funding_cost'] > 0].mean():.3f}")
    print(f"  mean anchor-vote fraction | funding_cost == 0:  "
          f"{frac[window['funding_cost'] == 0].mean():.3f}")
    print(f"\n  fraction of bars fully bullish (frac==1.0):                    "
          f"{fully_bullish.mean():.1%}")
    print(f"  fraction of bars where derate binds meaningfully (cost/vol>0.05): "
          f"{binds_hard.mean():.1%}")
    denom = binds_hard.sum()
    coincide = (binds_hard & fully_bullish).sum() / denom if denom else float("nan")
    print(f"  of those meaningfully-binding bars, fraction ALSO fully bullish:  "
          f"{coincide:.1%}"
          f"  (vs {fully_bullish.mean():.1%} base rate - "
          f"{'ABOVE' if denom and coincide > fully_bullish.mean() else 'at/below'} "
          "base rate)")


# ------------------------------------------------------------------------ all


def all_() -> None:
    inspect()
    causality()
    sweep()
    neighbours()
    feetier()
    windows()
    regime_check()
    print(f"\n=== total configurations evaluated (sweep + neighbours): "
          f"{N_EVALUATED} ===")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    cmds = {"inspect": inspect, "causality": causality, "sweep": sweep,
            "neighbours": neighbours, "feetier": feetier, "windows": windows,
            "regime_check": regime_check, "all": all_}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_carry_kelly.py "
              f"[{'|'.join(cmds)}]")
