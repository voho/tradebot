#!/usr/bin/env python
"""Driver for backlog B-01 — e-process regime detection with Kelly sizing.

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_eprocess.py inspect     # what the e-process does
    python experiments/run_eprocess.py sweep       # inner-train + inner-val
    python experiments/run_eprocess.py neighbours  # plateau check
    python experiments/run_eprocess.py holdout     # step 4, frozen config
    python experiments/run_eprocess.py eth         # falsification test
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.eprocess_regime import BARS_PER_YEAR, EProcessRegime  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

N_EVALUATED = 0  # every configuration this file evaluates, for deflated Sharpe


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0,
       count=True):
    """One backtest, one line, counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    # fills, not `trades`: a strategy whose exposure never returns to zero
    # books one open round-trip while rebalancing (and paying) many times.
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# --------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the e-process actually computes, before any backtest."""
    s = EProcessRegime(**FROZEN)
    prepared = s.prepare(DF.copy())
    thr = np.log(1.0 / s.alpha)

    sharpe_est = prepared["lam"] * np.sqrt(BARS_PER_YEAR)
    ev_ = prepared["evidence"]
    print(f"threshold log(1/alpha) = {thr:.2f}   (alpha={s.alpha})")
    print("\nendogenous target vol = kelly_fraction x annualized Sharpe estimate:")
    print(f"  lam*sqrt(BPY) quantiles: "
          + "  ".join(f"q{int(q*100)}={sharpe_est.quantile(q):.2f}"
                      for q in (0.05, 0.25, 0.5, 0.75, 0.95)))
    print(f"  half-Kelly target vol at the median: "
          f"{0.5 * sharpe_est.median():.2f}  (repo's hand-set value: 0.55)")
    print("\nevidence process:")
    print(f"  fraction of bars at the cap (gate fully open): "
          f"{(ev_ >= thr - 1e-9).mean():.1%}")
    print(f"  fraction of bars at zero (gate shut):          "
          f"{(ev_ <= 1e-9).mean():.1%}")

    print("\nyearly mean gate (fraction of full exposure the evidence allows):")
    gate = (ev_ / thr).clip(0, 1)
    per_year = gate.groupby(prepared.index.year).mean()
    ret = DF["close"].groupby(DF.index.year).last() / \
        DF["close"].groupby(DF.index.year).first() - 1.0
    for year, g in per_year.items():
        print(f"  {year}  gate={g:.2f}   BTC year return {ret.get(year, float('nan')):>+7.1%}")

    # Pre-registered failure mode (a): is the evidence process just a
    # smoothed momentum indicator wearing a martingale costume? Compare it
    # with the incumbent's latched anchor vote on the same bars.
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
    v4 = KellyRegimeV4()
    v4_prepared = v4.prepare(DF.copy())
    votes = []
    for days in v4.horizons:
        anchor = DF["close"].rolling(int(days * 288)).mean()
        v = pd.Series(np.where(DF["close"] > anchor * (1 + v4.band), 1.0,
                               np.where(DF["close"] < anchor * (1 - v4.band),
                                        0.0, np.nan)), index=DF.index)
        votes.append(v.ffill().fillna(0.0))
    vote = sum(votes) / len(votes)
    both = pd.DataFrame({"gate": gate, "vote": vote}).dropna()
    print("\ngate vs the incumbent's latched anchor vote:")
    print(f"  correlation                       {both['gate'].corr(both['vote']):.3f}")
    print(f"  bars where both are >0            {((both.gate > 0) & (both.vote > 0)).mean():.1%}")
    print(f"  evidence open while vote shut     {((both.gate > 0) & (both.vote == 0)).mean():.1%}")
    print(f"  vote open while evidence shut     {((both.gate == 0) & (both.vote > 0)).mean():.1%}")
    print(f"  mean exposure gate {both.gate.mean():.3f} vs vote {both.vote.mean():.3f}"
          f"  -> the e-process holds {both.gate.mean() / both.vote.mean():.2f}x the incumbent")
    del v4_prepared


# ----------------------------------------------------------------------- sweep


def _variants():
    """The three variants, on a small grid. Every entry is one trial."""
    out = []
    for hl in (20.0, 60.0, 180.0):
        out.append((f"E1 gate hl={hl:g}d",
                    dict(bet_halflife_days=hl, gate=True, sizing="fixed")))
    for hl in (20.0, 60.0, 180.0):
        out.append((f"E2 kelly hl={hl:g}d",
                    dict(bet_halflife_days=hl, gate=False, sizing="kelly")))
    for hl in (20.0, 60.0, 180.0):
        out.append((f"E3 both  hl={hl:g}d",
                    dict(bet_halflife_days=hl, gate=True, sizing="kelly")))
    return out


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}",
           count=False)


def sweep() -> None:
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} variants:")
            for tag, kw in _variants():
                ev(EProcessRegime(**kw), start, end, market=market, tag=tag)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def neighbours() -> None:
    """Plateau, not peak: vary one knob at a time around the selection."""
    base = dict(bet_halflife_days=20.0, gate=True, sizing="fixed")
    grid = [("base E1 hl=20d", {})]
    grid += [(f"hl={h:g}d", dict(bet_halflife_days=h))
             for h in (10.0, 15.0, 30.0, 40.0, 90.0)]
    grid += [(f"alpha={a}", dict(alpha=a)) for a in (0.01, 0.20)]
    grid += [(f"clip={c:g}", dict(clip=c)) for c in (3.0, 8.0)]
    grid += [(f"cap_mult={m:g}", dict(evidence_cap_mult=m)) for m in (0.5, 2.0)]
    grid += [(f"deadband={d:g}", dict(deadband=d)) for d in (0.05, 0.20)]
    grid += [("decay hl=180d", dict(evidence_halflife_days=180.0))]
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nINNER-VALIDATION neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(EProcessRegime(**{**base, **kw}), *VALID, market=market, tag=tag)
        print(f"INNER-TRAIN neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(EProcessRegime(**{**base, **kw}), *TRAIN, market=market, tag=tag,
               count=False)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def causality() -> None:
    """The strict on_bar peek check, run by hand for an unregistered strategy.

    ``tests/test_causality_strict.py`` only parametrizes over *registered*
    strategies, so an experiment gets none of that protection. This is the
    same two-opposite-tampers procedure: bars after a cut are multiplied by
    3 in one copy and divided by 3 in the other, and every decision at or
    before the cut must be identical.
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

    def decisions(frame):
        s = EProcessRegime(**FROZEN)
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
          else "PASS - every decision at or before the cut is unchanged")

    # A prepared column computed over the whole series would also move at
    # rows before the cut even when no order changes; check the target and
    # evidence columns directly, which is what the skeptic actually wants.
    pa = EProcessRegime(**FROZEN).prepare(up.copy())
    pb = EProcessRegime(**FROZEN).prepare(down.copy())
    for col in ("target", "evidence", "lam"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        print(f"  column {col:9s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


# --------------------------------------------------------------------- holdout

# Frozen before the holdout was read. E1 (evidence gate, incumbent sizer) is
# the variant that isolates the e-process claim; half-life 20d was selected on
# inner-validation and coincides with the 18-28 day anchor region R-07 found
# robust, so it is not purely a validation fit. Every other knob sits at its
# a-priori default: alpha=0.05 is the conventional level, cap_mult=1.0 makes
# full exposure mean "evidence has reached the alpha threshold", clip=5 and
# deadband=0.10 are inherited unchanged from the incumbent.
FROZEN = dict(bet_halflife_days=20.0, gate=True, sizing="fixed")
# Reported too, because a parallel branch that stays silent is selection by
# the operator (ROUTINE.md, "Running directions in parallel").
ALSO = [("E2 kelly (unified sizer)", dict(bet_halflife_days=60.0, gate=False,
                                          sizing="kelly")),
        ("E3 both", dict(bet_halflife_days=60.0, gate=True, sizing="kelly"))]


def holdout() -> None:
    """Step 4. The configuration is frozen; the decision rule is in the ledger."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        for name in ("buy_and_hold", "kelly_regime", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"  {name}", count=False)
        ev(EProcessRegime(**FROZEN), *OOS, market=market,
           tag="E1 eprocess_regime (FROZEN)", count=False)
        for tag, kw in ALSO:
            ev(EProcessRegime(**kw), *OOS, market=market, tag=tag, count=False)


def eth() -> None:
    """Pre-registered falsification: does the mechanism survive on ETH?

    Same venue (Bitfinex), same window, only the asset varies - the design
    of R-17 / docs/CROSS_ASSET.md, so the incumbent's numbers there are
    the comparison.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for market in (SPOT, FUTURES):
            for name in ("buy_and_hold", "kelly_regime_v4"):
                ev(get_strategy(name), None, None, df=df, market=market,
                   tag=f"  {name}", count=False)
            ev(EProcessRegime(**FROZEN), None, None, df=df, market=market,
               tag="  eprocess_regime (frozen)", count=False)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"inspect": inspect, "sweep": sweep, "neighbours": neighbours,
            "causality": causality, "holdout": holdout, "eth": eth}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_eprocess.py [{'|'.join(cmds)}]")
