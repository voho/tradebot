"""R-143 CONSERVATIVE branch: the backward holdout, executed.

Runs the decisive check pre-registered (and FROZEN) in
``experiments/r143_shared.py``: the matched-exposure max-drawdown gap
between ``kelly_regime_v4`` -- run with its shipped parameters, zero
refit -- and its own mean-notional-matched
:class:`experiments.matched_hold.ConstantExposureHold`, computed INSIDE
each of the six ~180-day ``PRIMARY_SUBWINDOWS`` tiling 2014-2016, on BTC
spot history no round of this project has ever touched.

Kill condition, quoted verbatim from the frozen pre-registration:

    if v4's drawdown is >= the matched hold's drawdown (i.e. the gap is
    >= 0, no advantage) in >= 50% of the sub-windows (i.e. >= 3 of 6),
    the matched-exposure drawdown-reduction property is judged a
    2017-2022 calibration artifact...

Everything else printed here is a disclosed secondary check and does not
touch that rule:

- the full 2014-01-01 -> 2016-12-31 primary window, recomputed;
- the 2013-inclusive sensitivity window (Mt. Gox-era manipulation,
  Gandal et al. 2018) -- reported, never used to override the primary;
- fully-invested ``buy_and_hold`` on every window, so the older unmatched
  comparison and R-33's matched one are both visible side by side;
- the primary window re-run at the 0.40% entry taker tier
  (``scripts/fee_study.py``'s ``BITSTAMP_TAKER``), matched inside the
  window at that fee as well;
- time-in-market and realized annualized volatility for BOTH arms in
  every window (guardrail 6).

Not registered; lives under ``experiments/`` so it is not auto-discovered
(ROUTINE.md step 5). Nothing in this file is fitted, swept or selected on
pre-2017 data.

Usage::

    python experiments/r143_conservative_backward_holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r100_shared import BARS_PER_DAY  # noqa: E402
from experiments.r143_shared import (  # noqa: E402
    PRIMARY_END,
    PRIMARY_START,
    PRIMARY_SUBWINDOWS,
    SENSITIVITY_START,
    SPOT,
    load_extended_btc_spot,
    matched_drawdown_gap,
)

# scripts/fee_study.py's BITSTAMP_TAKER: the entry (<$10k/30d) taker tier,
# this project's convention for a "real" spot fee. Restated rather than
# imported so this file has no dependency on a script's import side effects.
REAL_TAKER = 0.0040
KILL_THRESHOLD = 3  # >= 3 of 6 sub-windows with gap >= 0 kills the property


def v4():
    """kelly_regime_v4, shipped parameters, zero refit (guardrail 3)."""
    return get_strategy("kelly_regime_v4")


def arm_stats(result) -> dict:
    """Realized annualized vol + target-based time-in-market for one arm.

    Same construction as the private helper inside
    ``r143_shared.matched_drawdown_gap`` so the buy_and_hold reference
    arm is measured on exactly the same axes as the two matched arms.
    """
    m = compute_metrics(result)
    eq = result.equity.to_numpy(dtype=float)
    r = np.diff(np.log(eq[eq > 0]))
    vol = float(np.std(r) * np.sqrt(BARS_PER_DAY * 365.25)) if len(r) > 1 else float("nan")
    tim = float(np.mean(result.df["target"].abs() > 1e-9)) if "target" in result.df else float("nan")
    return {
        "final": m.final_balance, "dd": m.max_drawdown_pct, "sharpe": m.sharpe,
        "vol": vol, "tim": tim, "tim_realized": m.time_in_market_pct / 100.0,
        "trades": m.num_trades, "fees": m.fees_paid,
    }


def buy_and_hold_ref(df: pd.DataFrame, start: str, end: str,
                     market: MarketSpec = SPOT) -> dict:
    """Fully-invested buy_and_hold: the older, UNMATCHED reference arm."""
    res = run_period(get_strategy("buy_and_hold"), df, start, end,
                     market=market, start_balance=1_000.0)
    return arm_stats(res)


def row(res: dict, bh: dict) -> str:
    return (
        f"{res['start']}  {res['end']}  "
        f"{res['strat_dd']:>7.1f}% {res['hold_dd']:>7.1f}% {res['gap']:>+7.1f}pp  "
        f"${res['strat_final']:>8,.0f} ${res['hold_final']:>8,.0f}  "
        f"{res['strat_sharpe']:>6.2f} {res['hold_sharpe']:>6.2f}  "
        f"{res['strat_tim']:>6.1%} {res['hold_tim']:>6.1%}  "
        f"{res['strat_vol']:>6.1%} {res['hold_vol']:>6.1%}  "
        f"{res['mean_notional_c']:>5.3f}  "
        f"{bh['dd']:>6.1f}% ${bh['final']:>8,.0f} {bh['sharpe']:>6.2f} {bh['vol']:>6.1%}"
    )


HEADER = (
    f"{'start':10s}  {'end':10s}  {'v4 DD':>8s} {'hold DD':>8s} {'gap':>9s}  "
    f"{'v4 final':>9s} {'hold fin':>9s}  {'v4 Sh':>6s} {'hd Sh':>6s}  "
    f"{'v4 TiM':>6s} {'hd TiM':>6s}  {'v4 vol':>6s} {'hd vol':>6s}  "
    f"{'c':>5s}  {'BH DD':>7s} {'BH final':>9s} {'BH Sh':>6s} {'BH vol':>6s}"
)


def describe(res: dict, bh: dict, label: str) -> None:
    print(f"\n{label}")
    print(HEADER)
    print(row(res, bh))


def main() -> None:
    df = load_extended_btc_spot()
    print(f"extended series: {df.index.min()} -> {df.index.max()}  "
          f"({len(df):,} bars)\n")

    # ---- 1. DECISIVE CHECK: six sub-windows, matched inside each ----------
    print("=" * 172)
    print("DECISIVE CHECK -- matched-exposure drawdown gap, six ~180d sub-windows "
          "of 2014-2016, BTC spot, fee 0.10% (default)")
    print("gap = v4_DD - matchedHold_DD; NEGATIVE means v4 draws down LESS "
          "(property holds). Kill: gap >= 0 in >= 3 of 6.")
    print("=" * 172)
    print(HEADER)
    sub_results = []
    for start, end in PRIMARY_SUBWINDOWS:
        res = matched_drawdown_gap(df, v4, start, end, market=SPOT)
        bh = buy_and_hold_ref(df, start, end, market=SPOT)
        sub_results.append((res, bh))
        print(row(res, bh), flush=True)

    n_kill = sum(1 for res, _ in sub_results if res["gap"] >= 0.0)
    killed = n_kill >= KILL_THRESHOLD
    print("\n" + "-" * 172)
    print(f"gap >= 0 (no v4 advantage) in {n_kill} of {len(sub_results)} sub-windows; "
          f"kill threshold is >= {KILL_THRESHOLD}")
    print(f"VERDICT: property {'KILLED' if killed else 'SURVIVED'} "
          f"the pre-registered backward-holdout kill condition")
    gaps = np.array([res["gap"] for res, _ in sub_results])
    print(f"gap: mean {gaps.mean():+.1f}pp, median {np.median(gaps):+.1f}pp, "
          f"min {gaps.min():+.1f}pp, max {gaps.max():+.1f}pp")
    print("-" * 172)

    # ---- 2. SECONDARY: full primary window, recomputed --------------------
    full = matched_drawdown_gap(df, v4, PRIMARY_START, PRIMARY_END, market=SPOT)
    full_bh = buy_and_hold_ref(df, PRIMARY_START, PRIMARY_END, market=SPOT)
    describe(full, full_bh,
             "SECONDARY (a) -- full primary window 2014-01-01 -> 2016-12-31, "
             "exposure matched on the whole window")

    # ---- 3. SECONDARY: 2013-inclusive sensitivity window ------------------
    sens = matched_drawdown_gap(df, v4, SENSITIVITY_START, PRIMARY_END, market=SPOT)
    sens_bh = buy_and_hold_ref(df, SENSITIVITY_START, PRIMARY_END, market=SPOT)
    describe(sens, sens_bh,
             "SECONDARY (b) -- DISCLOSED SENSITIVITY 2013-01-01 -> 2016-12-31 "
             "(Mt. Gox-era manipulation; never the primary claim)")

    # ---- 4. SECONDARY: real spot fee tier ---------------------------------
    fee_market = MarketSpec.spot(fee_rate=REAL_TAKER)
    fee_full = matched_drawdown_gap(df, v4, PRIMARY_START, PRIMARY_END, market=fee_market)
    fee_bh = buy_and_hold_ref(df, PRIMARY_START, PRIMARY_END, market=fee_market)
    describe(fee_full, fee_bh,
             f"SECONDARY (c) -- primary window at the real taker tier "
             f"{REAL_TAKER:.2%} (fee_study.py BITSTAMP_TAKER)")

    print(f"\nSECONDARY (c2) -- six sub-windows at {REAL_TAKER:.2%} taker "
          f"(reported only; the decision rule above uses the default fee)")
    print(HEADER)
    fee_gaps = []
    for start, end in PRIMARY_SUBWINDOWS:
        res = matched_drawdown_gap(df, v4, start, end, market=fee_market)
        bh = buy_and_hold_ref(df, start, end, market=fee_market)
        fee_gaps.append(res["gap"])
        print(row(res, bh), flush=True)
    n_kill_fee = sum(1 for g in fee_gaps if g >= 0.0)
    print(f"at {REAL_TAKER:.2%}: gap >= 0 in {n_kill_fee} of {len(fee_gaps)} sub-windows "
          f"(disclosed, not the decision rule)")


if __name__ == "__main__":
    main()
