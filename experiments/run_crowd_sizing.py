#!/usr/bin/env python
"""Driver for the crowd-sizing haircut on kelly_regime_v4 (L-12's recorded lesson).

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              NOT TOUCHED by this file

Usage::

    python experiments/run_crowd_sizing.py sweep      # step 3 grid (train)
    python experiments/run_crowd_sizing.py validate    # shortlist on inner-validation, both markets
    python experiments/run_crowd_sizing.py neighbours  # plateau check around the pick
    python experiments/run_crowd_sizing.py parity      # lam_crowd=0 reduces exactly to v4
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.crowd_sizing import CrowdSizedKellyV4  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")

OUT = ROOT / "reports" / "crowd_sizing"
BARS_PER_DAY = 288

N_EVALUATED = 0  # distinct (lam_crowd, age_scale) configurations, for deflated Sharpe

# lam_crowd: harsanyi_crowd's default (0.7) plus a spread either side, and
# 0.0 is NOT included here (that is the parity check, not a search point).
LAM_GRID = (0.3, 0.5, 0.7, 0.9)

# age_scale, expressed in days then converted to bars. harsanyi_crowd's
# default (150 bars = 12.5 hours) is included as the low extreme, but v4's
# vote is latched for weeks, not hours, so most of the grid explores much
# slower scales.
AGE_SCALE_DAYS = (0.5, 5, 10, 20, 40, 80)  # 0.5d ~= 150 bars, harsanyi's default


def age_scale_bars(days: float) -> float:
    return days * BARS_PER_DAY


def measure(strategy, start, end, *, market=SPOT, balance=1_000.0, count=False):
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = run_period(strategy, DF, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, result


def line(tag, m, result):
    print(f"  {tag:34s} final=${m.final_balance:>11,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"trades={len(result.fills):>5d} fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------------- parity


def parity() -> None:
    """lam_crowd=0.0 must reduce CrowdSizedKellyV4 EXACTLY to kelly_regime_v4."""
    print("Inner-validation, spot. The two rows must be identical (lam_crowd=0).")
    a, ra = measure(get_strategy("kelly_regime_v4"), *VALID)
    b, rb = measure(CrowdSizedKellyV4(lam_crowd=0.0), *VALID)
    line("kelly_regime_v4", a, ra)
    line("CrowdSizedKellyV4(lam_crowd=0)", b, rb)
    worst = float(np.max(np.abs(ra.equity.to_numpy() - rb.equity.to_numpy())))
    print(f"\n  max |equity difference|: {worst:.3e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")


# ----------------------------------------------------------------------- sweep


def sweep() -> None:
    """Step 3. Grid over (lam_crowd, age_scale) on inner-train, BOTH markets."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lam in LAM_GRID:
        for days in AGE_SCALE_DAYS:
            age_scale = age_scale_bars(days)
            s_spot = CrowdSizedKellyV4(lam_crowd=lam, age_scale=age_scale)
            m_spot, r_spot = measure(s_spot, *TRAIN, market=SPOT, count=True)
            s_fut = CrowdSizedKellyV4(lam_crowd=lam, age_scale=age_scale)
            m_fut, r_fut = measure(s_fut, *TRAIN, market=FUTURES, count=False)
            for mname, m, r in (("spot", m_spot, r_spot), ("futures", m_fut, r_fut)):
                rows.append({
                    "split": "inner-train", "market": mname,
                    "lam_crowd": lam, "age_scale_days": days,
                    "final": m.final_balance, "dd": m.max_drawdown_pct,
                    "sharpe": m.sharpe, "trades": len(r.fills),
                    "fees": m.fees_paid, "liquidated": m.liquidated,
                })
            print(f"lam={lam:.1f} age={days:>5g}d  "
                  f"spot final=${m_spot.final_balance:>10,.0f} DD={m_spot.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m_spot.sharpe:>5.2f}  |  "
                  f"fut final=${m_fut.final_balance:>10,.0f} DD={m_fut.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m_fut.sharpe:>5.2f}")
    pd.DataFrame(rows).to_csv(OUT / "sweep_train.csv", index=False)
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep_train.csv'}")


# -------------------------------------------------------------------- baseline


def baseline() -> None:
    """buy_and_hold and kelly_regime_v4 on both splits, both markets - for the table."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            for name in ("buy_and_hold", "kelly_regime_v4"):
                m, r = measure(get_strategy(name), start, end, market=market)
                rows.append({
                    "split": split_name, "market": mname, "strategy": name,
                    "final": m.final_balance, "dd": m.max_drawdown_pct,
                    "sharpe": m.sharpe, "trades": len(r.fills),
                    "fees": m.fees_paid, "liquidated": m.liquidated,
                })
                print(f"{split_name:16s} {mname:8s} {name:16s} "
                      f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
                      f"sharpe={m.sharpe:>5.2f} trades={len(r.fills)}")
    pd.DataFrame(rows).to_csv(OUT / "baseline.csv", index=False)
    print(f"\nwritten: {OUT / 'baseline.csv'}")


# -------------------------------------------------------------------- validate


# Shortlist selected by hand after inspecting sweep_train.csv - see report.
SHORTLIST = (
    (0.7, 20), (0.7, 40), (0.5, 40), (0.7, 80), (0.9, 40), (0.5, 20),
)


def validate() -> None:
    """Score the shortlist on inner-validation, both markets. Selection happens here."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lam, days in SHORTLIST:
        age_scale = age_scale_bars(days)
        for mname, market in MARKETS:
            s = CrowdSizedKellyV4(lam_crowd=lam, age_scale=age_scale)
            m, r = measure(s, *VALID, market=market)
            rows.append({
                "split": "inner-validation", "market": mname,
                "lam_crowd": lam, "age_scale_days": days,
                "final": m.final_balance, "dd": m.max_drawdown_pct,
                "sharpe": m.sharpe, "trades": len(r.fills),
                "fees": m.fees_paid, "liquidated": m.liquidated,
            })
            print(f"lam={lam:.1f} age={days:>3g}d  {mname:8s} "
                  f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% "
                  f"sharpe={m.sharpe:>5.2f} trades={len(r.fills)}")
    pd.DataFrame(rows).to_csv(OUT / "validate.csv", index=False)
    print(f"\nwritten: {OUT / 'validate.csv'}")


# ------------------------------------------------------------------ neighbours


def neighbours() -> None:
    """Plateau check: one-knob-at-a-time perturbations around the frozen pick."""
    OUT.mkdir(parents=True, exist_ok=True)
    lam0, days0 = 0.7, 40
    lam_neigh = (0.5, 0.6, 0.7, 0.8, 0.9)
    days_neigh = (20, 30, 40, 60, 80)
    rows = []
    for lam in lam_neigh:
        age_scale = age_scale_bars(days0)
        s = CrowdSizedKellyV4(lam_crowd=lam, age_scale=age_scale)
        m, r = measure(s, *VALID, market=SPOT)
        rows.append({"knob": "lam_crowd", "value": lam, "age_scale_days": days0,
                     "market": "spot", "final": m.final_balance,
                     "dd": m.max_drawdown_pct, "sharpe": m.sharpe})
        print(f"lam={lam:.1f} (age={days0}d fixed)  spot  "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f}")
    for days in days_neigh:
        age_scale = age_scale_bars(days)
        s = CrowdSizedKellyV4(lam_crowd=lam0, age_scale=age_scale)
        m, r = measure(s, *VALID, market=SPOT)
        rows.append({"knob": "age_scale_days", "value": days, "lam_crowd": lam0,
                     "market": "spot", "final": m.final_balance,
                     "dd": m.max_drawdown_pct, "sharpe": m.sharpe})
        print(f"age={days:>3g}d (lam={lam0} fixed)  spot  "
              f"final=${m.final_balance:>10,.0f} DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f}")
    pd.DataFrame(rows).to_csv(OUT / "neighbours.csv", index=False)
    print(f"\nwritten: {OUT / 'neighbours.csv'}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"parity": parity, "sweep": sweep, "baseline": baseline,
            "validate": validate, "neighbours": neighbours}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_crowd_sizing.py [{'|'.join(cmds)}]")
