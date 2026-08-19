#!/usr/bin/env python
"""Driver for the drawdown-feedback experiment (parallel round, branch B).

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4/5 only, pre-registered

Usage::

    python experiments/run_drawdown_feedback.py sweep       # step 3: 6 variants
    python experiments/run_drawdown_feedback.py causality   # by-hand lookahead probe
    python experiments/run_drawdown_feedback.py holdout     # step 5, frozen config
    python experiments/run_drawdown_feedback.py interval    # step 6: paired bootstrap
    python experiments/run_drawdown_feedback.py eth         # step 7: falsification
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.drawdown_feedback import DrawdownFeedbackKelly  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
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
OOS = ("2023-01-01", None)

OUT = ROOT / "reports" / "drawdown_feedback"

# --------------------------------------------------------------------- config
#
# 6 variants of multiplier(dd) = clip(1 - (dd/dd_cap)**power, floor, 1.0).
# dd_cap chosen around and above v4's typical realized drawdowns (25-40%,
# per docs/LEDGER.md L-01/R-29) so the "barely bites in normal drawdowns"
# calibration is testable rather than assumed; power=1 is the linear
# CDaR-style case, power>1 is convex (flat in the interior, steep near the
# cap); one variant carries a nonzero floor so exposure is never fully cut.

VARIANTS = {
    "v1_linear_tight": dict(dd_cap=0.30, power=1.0, floor=0.0),
    "v2_linear_loose": dict(dd_cap=0.50, power=1.0, floor=0.0),
    "v3_convex_tight": dict(dd_cap=0.30, power=2.0, floor=0.0),
    "v4_convex_loose": dict(dd_cap=0.50, power=2.0, floor=0.0),
    "v5_convex_steep": dict(dd_cap=0.45, power=3.0, floor=0.0),
    "v6_convex_floor": dict(dd_cap=0.40, power=2.0, floor=0.15),
}

N_EVALUATED = 0  # distinct configurations searched in step 3 (for deflated Sharpe)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            count=False):
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, result


def line(tag, m, result):
    print(f"  {tag:26s} final=${m.final_balance:>11,.0f} "
          f"({m.profit_pct:>+8.1f}%) DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------- sweep


def sweep() -> None:
    """Step 3. 6 variants x 2 markets x 2 inner splits, vs v4 and buy_and_hold."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            print(f"\n{split_name} / {mname}")
            for name in ("buy_and_hold", "kelly_regime_v4"):
                m, res = measure(get_strategy(name), start, end, market=market)
                line(f"  {name}", m, res)
                rows.append({"split": split_name, "market": mname, "variant": name,
                            **m.as_row()})
            for vname, params in VARIANTS.items():
                s = DrawdownFeedbackKelly(**params)
                # count each distinct (dd_cap, power, floor) config once,
                # regardless of which market/split it is scored on - the
                # R-28/R-31 convention.
                count = (split_name == "inner-train" and mname == "spot")
                m, res = measure(s, start, end, market=market, count=count)
                line(f"  {vname}", m, res)
                rows.append({"split": split_name, "market": mname, "variant": vname,
                            **m.as_row()})
    pd.DataFrame(rows).to_csv(OUT / "sweep.csv", index=False)
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep.csv'}")


# ----------------------------------------------------------------- causality


def causality() -> None:
    """By-hand lookahead probe - experiments get no CI protection.

    Same two-opposite-tampers procedure as tests/test_causality_strict.py
    and R-28/R-31's by-hand version: bars after a cut are multiplied by 3
    in one copy and divided by 3 in the other (volume by 7), and every
    decision at or before the cut must be bit-identical. The column
    comparison (not just the queued orders) is what catches a full-series
    fit that a truncation test alone cannot see.
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

    def decisions(frame, params):
        s = DrawdownFeedbackKelly(**params)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out, prepared

    ok = True
    for vname, params in VARIANTS.items():
        oa, pa = decisions(up, params)
        ob, pb = decisions(down, params)
        bad = [b for b, x, y in zip(bars, oa, ob) if x != y]
        worst = max(
            float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
            for c in ("target", "raw_target", "dd_proxy", "dd_multiplier")
        )
        good = not bad and worst < 1e-12
        ok &= good
        print(f"  {vname:20s} orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# -------------------------------------------------------------------- holdout


FROZEN = "v4_convex_loose"  # set after step 3/4, before this is ever called


def holdout() -> None:
    """Step 5. Frozen config vs v4 and buy_and_hold, spot and futures 5x."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    params = VARIANTS[FROZEN]
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}   (frozen={FROZEN} {params})")
        for name, strat in (
            ("buy_and_hold", get_strategy("buy_and_hold")),
            ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
            (f"drawdown_feedback[{FROZEN}]", DrawdownFeedbackKelly(**params)),
        ):
            m, res = measure(strat, *OOS, market=market)
            line(f"  {name}", m, res)
            rows.append({"market": mname, "variant": name, **m.as_row()})
    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# ------------------------------------------------------------------- interval


def interval() -> None:
    """Step 6. Paired block-bootstrap: frozen variant vs plain v4, on the holdout."""
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    params = VARIANTS[FROZEN]
    rows = []
    for mname, market in MARKETS:
        a_res = run_period(DrawdownFeedbackKelly(**params), DF, *OOS, market=market,
                           start_balance=1_000.0, data_label=LABEL)
        b_res = run_period(get_strategy("kelly_regime_v4"), DF, *OOS, market=market,
                           start_balance=1_000.0, data_label=LABEL)
        a = daily_returns(a_res.equity).to_numpy()
        b = daily_returns(b_res.equity).to_numpy()
        n = min(len(a), len(b))
        a, b = a[-n:], b[-n:]
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
        print(f"\n{mname}: paired drawdown_feedback[{FROZEN}] - kelly_regime_v4 "
              f"on the 2023+ holdout ({n} daily observations)")
        for stat_name, stat in (("Δ log growth", total_log_return),
                                ("Δ max drawdown (pp)", max_drawdown_from_returns)):
            r = paired_bootstrap(a, b, stat, indices=idx)
            mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
            print(f"  {stat_name:22s} {mark} {r.diff.point:>+7.3f} "
                  f"[{r.diff.lo:>+7.3f}, {r.diff.hi:>+7.3f}]  P(>0)={r.p_positive:.2f}")
            rows.append({"market": mname, "stat": stat_name, "variant": r.stat_a,
                        "v4": r.stat_b, "diff": r.diff.point, "lo": r.diff.lo,
                        "hi": r.diff.hi, "p_positive": r.p_positive,
                        "significant": r.significant})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intervals.csv", index=False)
    print(f"\nwritten: {OUT / 'intervals.csv'}")


# ------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7. Pre-registered falsification: BTC control + ETH test, Bitfinex.

    Same venue and window R-17/R-28/R-31 used. Frozen config only, applied
    unmodified (no re-fitting on ETH) - the question is whether the SAME
    parameters cut drawdown on a second asset, not whether some parameters
    can be found that do.
    """
    params = VARIANTS[FROZEN]
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            for name, strat in (
                ("buy_and_hold", get_strategy("buy_and_hold")),
                ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                (f"drawdown_feedback[{FROZEN}]", DrawdownFeedbackKelly(**params)),
            ):
                m, res = measure(strat, None, None, df=df, market=market)
                line(f"    {name}", m, res)


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "causality": causality, "holdout": holdout,
            "interval": interval, "eth": eth}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_drawdown_feedback.py [{'|'.join(cmds)}]")
