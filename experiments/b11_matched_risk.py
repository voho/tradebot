#!/usr/bin/env python
"""Driver for backlog B-11 — matched-risk frontier: e-process gate vs latched
vote at equal realized volatility.

R-28 found the e-process gate (``EProcessRegime`` E1: ``gate=True,
sizing="fixed"`` — the incumbent's own inverse-vol sizer, only the gate
differs) has the deepest drawdown reduction in the project but loses on
return, because it holds only ~0.27x the incumbent's mean exposure. That
comparison confounds gate QUALITY with RISK LEVEL — of course the strategy
holding less risk has less return; the open question is which gate
delivers more return *per unit of realized risk*.

This file raises the e-process's exposure using ``target_vol`` — the fixed
sizer's own scale parameter, ``target_vol / realized_vol``, independent of
the evidence cap / gate logic — until its realized annualized volatility on
inner-train matches ``kelly_regime_v4``'s, then compares return, Sharpe and
drawdown at that matched risk level. Every other E1 parameter is frozen at
R-28's values: ``bet_halflife_days=20, alpha=0.05, clip=5, deadband=0.10,
max_leverage=2.0, evidence_cap_mult=1.0, gate=True, sizing="fixed"``.

Explicitly NOT the way to raise exposure: ``evidence_cap_mult``. R-28's
neighbourhood sweep already measured that raising it to 2 lets stale
evidence persist and drawdown grows superlinearly (49% DD, -22% return on
inner-validation). ``target_vol`` is orthogonal to the evidence/gate logic
by construction — it only rescales the fixed inverse-vol sizer that both
strategies already share — so it isolates the risk-level question instead
of also changing gate quality.

Splits follow ROUTINE.md step 3, identical to ``experiments/run_eprocess.py``::

    inner-train       2017-01-01 -> 2020-12-31   fit / sweep target_vol
    inner-validation  2021-01-01 -> 2022-12-31   select, compare to v4
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/b11_matched_risk.py sweep       # match target_vol on inner-train
    python experiments/b11_matched_risk.py validate    # matched E1 vs v4 on inner-validation
    python experiments/b11_matched_risk.py causality   # mandatory hand-run lookahead check
    python experiments/b11_matched_risk.py holdout     # step 4, frozen config
    python experiments/b11_matched_risk.py windows     # pre-registered falsification test
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
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

N_EVALUATED = 0  # every configuration this file evaluates, for deflated Sharpe

# R-28's frozen E1 parameters, held fixed. Only target_vol is swept.
BASE = dict(bet_halflife_days=20.0, alpha=0.05, clip=5.0, deadband=0.10,
            max_leverage=2.0, evidence_cap_mult=1.0, gate=True, sizing="fixed")


def realized_vol(equity: np.ndarray) -> float:
    """Annualized volatility of per-bar equity returns (the risk actually taken)."""
    if len(equity) < 3:
        return 0.0
    prev = equity[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(equity) / prev, 0.0)
    sd = rets.std(ddof=1)
    return float(sd * np.sqrt(BARS_PER_YEAR)) if np.isfinite(sd) else 0.0


def ev(strategy, start, end, market=SPOT, tag="", balance=1_000.0, count=True):
    """One backtest, one line: final balance, realized vol, DD, Sharpe, fees. Counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = run_period(strategy, DF, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    vol = realized_vol(result.equity.to_numpy(dtype=float))
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"vol={vol * 100:>5.1f}% DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>4d} "
          f"fees=${m.fees_paid:>7,.0f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m, vol


# ------------------------------------------------------------------- step 2/3


def sweep() -> None:
    """Match target_vol to kelly_regime_v4's inner-train realized vol (spot).

    Coarse grid designed before running (step 2): 0.55 is R-28's frozen,
    uncorrected value, the rest span up toward the point where the fixed
    sizer saturates its 2x leverage cap on almost every bar, to find out
    whether the ceiling on achievable exposure (set by the gate's mean
    confidence, not by target_vol) binds before a match is reached — the
    named pre-registered failure mode.
    """
    print("INNER-TRAIN 2017-2020, spot -- matching target_vol to kelly_regime_v4:\n")
    v4_m, v4_vol = ev(get_strategy("kelly_regime_v4"), *TRAIN, tag="kelly_regime_v4 (target)",
                      count=False)
    print()
    grid = [0.55, 1.2, 2.4, 4.5, 8.0]
    results = []
    for tv in grid:
        m, vol = ev(EProcessRegime(target_vol=tv, **BASE), *TRAIN,
                    tag=f"E1 target_vol={tv:g}")
        results.append((tv, vol, m))
    print(f"\nv4 inner-train realized vol (spot): {v4_vol * 100:.1f}%")
    for tv, vol, m in results:
        print(f"  target_vol={tv:<5g} -> realized vol {vol * 100:5.1f}%  "
              f"ratio {vol / v4_vol:.2f}x")
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


def bisect_match(v4_vol: float, lo: float, hi: float, tol: float = 0.10,
                 max_iter: int = 8) -> float:
    """Bisection on target_vol until inner-train realized vol is within `tol` of v4."""
    global N_EVALUATED
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        result = run_period(EProcessRegime(target_vol=mid, **BASE), DF, *TRAIN,
                            market=SPOT, start_balance=1_000.0, data_label=LABEL)
        N_EVALUATED += 1
        vol = realized_vol(result.equity.to_numpy(dtype=float))
        print(f"  bisect target_vol={mid:6.3f} -> vol={vol * 100:5.1f}% "
              f"(target {v4_vol * 100:.1f}%, ratio {vol / v4_vol:.2f}x)")
        if abs(vol / v4_vol - 1.0) <= tol:
            return mid
        if vol < v4_vol:
            lo = mid
        else:
            hi = mid
    return mid


# --------------------------------------------------------------------- validate


def validate(target_vol: float) -> None:
    """Compare the risk-matched E1 to v4 and buy_and_hold on inner-validation."""
    print(f"\nINNER-VALIDATION 2021-2022 -- E1 target_vol={target_vol:.3f} vs v4:\n")
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"{mname}:")
        ev(get_strategy("buy_and_hold"), *VALID, market=market, tag="buy_and_hold",
           count=False)
        ev(get_strategy("kelly_regime_v4"), *VALID, market=market, tag="kelly_regime_v4",
           count=False)
        ev(EProcessRegime(target_vol=target_vol, **BASE), *VALID, market=market,
           tag=f"E1 matched (target_vol={target_vol:.3f})")
        print()
    print(f"configurations evaluated so far: {N_EVALUATED}")


# -------------------------------------------------------------------- causality


def causality(target_vol: float) -> None:
    """Mandatory hand-run lookahead check (unregistered strategy, R-28's procedure).

    Two-opposite-tampers: bars after a cut multiplied by 3 in one copy,
    divided by 3 in the other; every decision at or before the cut must be
    bit-identical between the two copies.
    """
    from tradebot.broker import PaperBroker

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
        s = EProcessRegime(target_vol=target_vol, **BASE)
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

    pa = EProcessRegime(target_vol=target_vol, **BASE).prepare(up.copy())
    pb = EProcessRegime(target_vol=target_vol, **BASE).prepare(down.copy())
    for col in ("target", "evidence", "lam"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        print(f"  column {col:9s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


# ---------------------------------------------------------------------- holdout

# Frozen after inner-train/inner-validation only; see the ledger row for the
# exact selection trace (sweep() + bisect_match() results feed this number).
FROZEN_TARGET_VOL = None  # set by __main__ / filled in report; see FROZEN below


def holdout(target_vol: float) -> None:
    """Step 4. Evaluate once on 2023+, both markets, against v4 and buy_and_hold."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        ev(get_strategy("buy_and_hold"), *OOS, market=market, tag="buy_and_hold",
           count=False)
        ev(get_strategy("kelly_regime_v4"), *OOS, market=market, tag="kelly_regime_v4",
           count=False)
        ev(EProcessRegime(target_vol=target_vol, **BASE), *OOS, market=market,
           tag=f"E1 matched-risk (target_vol={target_vol:.3f}, FROZEN)", count=False)


def windows(target_vol: float, trials: int = 40, seed: int = 42) -> None:
    """Pre-registered falsification test: 40 random Monte Carlo windows.

    Same design as scripts/stress_test.py and R-28's own windows() check:
    random start, random 90-730 day length, identical windows across
    strategies, warmup prefix that cannot trade. Paired per window.

    This is the ONE pre-registered falsification test for B-11 (chosen
    from ROUTINE.md's approved list). Kill criterion, written before
    running: the *risk-matching itself* is an empirical claim, made on one
    4-year inner-train window. If the e-process's realized volatility
    across these 40 windows is NOT within a similar band of v4's (drifting
    persistently, not just noisily, outside roughly 0.7x-1.4x of v4's
    per-window vol), the "matched risk" premise does not generalize and
    any return/drawdown comparison built on it is not trustworthy,
    regardless of which strategy looks better.
    """
    from tradebot.metrics import max_drawdown_pct

    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("E1 matched-risk", EProcessRegime(target_vol=target_vol, **BASE))]
    warmup = max(s.warmup for _, s in contenders) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(90, 731) * 288)
        specs.append((int(rng.integers(warmup, len(DF) - length)), length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for mname, market in (("spot", SPOT), ("futures", FUTURES)):
            for name, strat in contenders:
                res = run_backtest(strat, window, market, 1_000.0,
                                   trade_start=warmup, data_label=LABEL)
                eq = res.equity.to_numpy(dtype=float)
                base, seg = eq[warmup], eq[warmup:]
                ok = np.isfinite(base) and base > 0
                rows.append({"trial": k, "market": mname, "strategy": name,
                             "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                             "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                             "vol": realized_vol(seg) if ok else float("nan"),
                             "liquidated": res.liquidated})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    res = pd.DataFrame(rows)
    print()

    print(f"\n{trials} random windows (90-730 days), identical across strategies:\n")
    for mname in ("spot", "futures"):
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name in ("buy_and_hold", "kelly_regime_v4", "E1 matched-risk"):
            g = sub[sub.strategy == name].set_index("trial")
            beat = (g["return_pct"] > bench).mean()
            print(f"    {name:16s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median vol {g.vol.median() * 100:>5.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"beat hold {beat:>5.0%}  liq {g.liquidated.mean():>4.0%}")
        v4v = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["vol"]
        epv = sub[sub.strategy == "E1 matched-risk"].set_index("trial")["vol"]
        ratio = (epv / v4v).dropna()
        print(f"    vol ratio (e-process / v4) per window: median {ratio.median():.2f}x, "
              f"range {ratio.min():.2f}x-{ratio.max():.2f}x, "
              f"within 0.7x-1.4x in {((ratio >= 0.7) & (ratio <= 1.4)).mean():.0%} of windows")
        a = sub[sub.strategy == "E1 matched-risk"].set_index("trial")["return_pct"]
        b = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["return_pct"]
        d = (a - b).dropna()
        print(f"    paired return difference (matched-risk - v4): median {d.median():+.1f}pp, "
              f"beats v4 on return in {(d > 0).mean():.0%} of windows")
        ad = sub[sub.strategy == "E1 matched-risk"].set_index("trial")["max_dd_pct"]
        bd = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["max_dd_pct"]
        dd = (ad - bd).dropna()
        print(f"    paired DD difference (matched-risk - v4): median {dd.median():+.1f}pp, "
              f"deeper in {(dd > 0).mean():.0%} of windows\n")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    tv = float(args[1]) if len(args) > 1 else None
    if cmd == "sweep":
        sweep()
    elif cmd == "bisect":
        v4res = run_period(get_strategy("kelly_regime_v4"), DF, *TRAIN, market=SPOT,
                           start_balance=1_000.0, data_label=LABEL)
        v4vol = realized_vol(v4res.equity.to_numpy(dtype=float))
        lo = float(args[1]) if len(args) > 1 else 1.0
        hi = float(args[2]) if len(args) > 2 else 4.0
        star = bisect_match(v4vol, lo, hi)
        print(f"\nmatched target_vol ~= {star:.3f}")
    elif cmd == "validate" and tv is not None:
        validate(tv)
    elif cmd == "causality" and tv is not None:
        causality(tv)
    elif cmd == "holdout" and tv is not None:
        holdout(tv)
    elif cmd == "windows" and tv is not None:
        windows(tv)
    else:
        print("usage: python experiments/b11_matched_risk.py "
              "[sweep|bisect lo hi|validate TV|causality TV|holdout TV|windows TV]")
