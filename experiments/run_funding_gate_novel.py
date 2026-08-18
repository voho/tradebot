#!/usr/bin/env python
"""Driver for backlog B-05 (novel) — momentum-conditioned continuous funding fade.

Splits, narrower than the project's usual convention because the
committed funding series only covers 2020-01-01 .. 2023-12-31::

    inner-train       2020-01-01 -> 2020-12-31   sweep the hyperparameter grid
    inner-validation  2021-01-01 -> 2022-12-31   select the frozen config
    holdout           2023-01-01 ->              NOT evaluated here — reserved
                                                 for the orchestrator (step 4)

Usage::

    python experiments/run_funding_gate_novel.py sweep       # step 3: grid + table
    python experiments/run_funding_gate_novel.py causality   # by-hand lookahead probe
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

from experiments.funding_gate_novel import FundingMomentumGate  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT = MarketSpec.spot()

TRAIN = ("2020-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
# NEVER evaluated by this file — reserved for the orchestrator, step 4.
OOS_START = "2023-01-01"

OUT = ROOT / "reports" / "funding_gate_novel"

N_EVALUATED = 0  # distinct hyperparameter configurations, for the trials budget

# --------------------------------------------------------------------- grid
#
# Fixed to the project's existing conventions (180-day = kelly_regime_v3's
# anchor_span_days; 7-day matches R-16's own momentum axis):
FIXED = dict(funding_ewm_span=3, funding_lookback_days=180,
             momentum_days=7, momentum_lookback_days=180)

# The core interaction grid: 2 x 3 x 2 = 12 configurations.
GRID_CORE = [
    {"funding_threshold": ft, "momentum_threshold": mt, "fade_strength": fs}
    for ft in (0.85, 0.90)
    for mt in (0.40, 0.50, 0.60)
    for fs in (0.5, 1.0)
]
# Two spot-checks of the fixed parameters, at the grid's central point
# (funding_threshold=0.90, momentum_threshold=0.50, fade_strength=1.0).
GRID_EXTRA = [
    {"funding_threshold": 0.90, "momentum_threshold": 0.50, "fade_strength": 1.0,
     "funding_ewm_span": 6},
    {"funding_threshold": 0.90, "momentum_threshold": 0.50, "fade_strength": 1.0,
     "momentum_days": 14},
]
GRID = GRID_CORE + GRID_EXTRA  # 14 total, under the 16-config cap


def _cfg_id(cfg: dict) -> str:
    p = {**FIXED, **cfg}
    extra = ""
    if p["funding_ewm_span"] != FIXED["funding_ewm_span"]:
        extra += f" span={p['funding_ewm_span']}"
    if p["momentum_days"] != FIXED["momentum_days"]:
        extra += f" mdays={p['momentum_days']}"
    return (f"ft={p['funding_threshold']:.2f} mt={p['momentum_threshold']:.2f} "
            f"fs={p['fade_strength']:.1f}{extra}")


def build(cfg: dict) -> FundingMomentumGate:
    params = {**FIXED, **cfg}
    return FundingMomentumGate(funding=REAL, **params)


def _period(strategy, market, start, end, *, funding=None, df=None):
    """Backtest over a date range, warmed on the bars before it (funding_study.py idiom)."""
    frame = DF if df is None else df
    lo = 0 if start is None else int(frame.index.searchsorted(start))
    hi = len(frame) if end is None else int(frame.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, frame.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def _row(tag, split, strategy_factory, start, end):
    """One config's full metric set on one split: spot, futures-free, futures-charged."""
    spot_m, _ = _period(strategy_factory(), SPOT, start, end, funding=None)
    fut_free_m, _ = _period(strategy_factory(), FUTURES, start, end, funding=None)
    fut_paid_m, funding_paid = _period(strategy_factory(), FUTURES, start, end, funding=REAL)
    log_growth = float(np.log(fut_paid_m.final_balance / 1_000.0))
    return {
        "config": tag, "split": split,
        "spot_final": spot_m.final_balance, "spot_sharpe": spot_m.sharpe,
        "spot_dd": spot_m.max_drawdown_pct, "spot_trades": spot_m.num_trades,
        "fut_free_final": fut_free_m.final_balance, "fut_free_sharpe": fut_free_m.sharpe,
        "fut_free_dd": fut_free_m.max_drawdown_pct, "fut_free_trades": fut_free_m.num_trades,
        "fut_paid_final": fut_paid_m.final_balance, "fut_paid_sharpe": fut_paid_m.sharpe,
        "fut_paid_dd": fut_paid_m.max_drawdown_pct, "fut_paid_trades": fut_paid_m.num_trades,
        "funding_paid": funding_paid, "log_growth_futures_charged": log_growth,
    }


def _print_row(r):
    print(f"  {r['config']:46s} "
          f"spot=${r['spot_final']:>9,.0f} sh={r['spot_sharpe']:>5.2f} "
          f"dd={r['spot_dd']:>5.1f}% tr={r['spot_trades']:>4d}  |  "
          f"fut-free=${r['fut_free_final']:>9,.0f} sh={r['fut_free_sharpe']:>5.2f} "
          f"dd={r['fut_free_dd']:>5.1f}%  |  "
          f"fut-paid=${r['fut_paid_final']:>9,.0f} sh={r['fut_paid_sharpe']:>5.2f} "
          f"dd={r['fut_paid_dd']:>5.1f}% fund=${r['funding_paid']:>7,.0f} "
          f"logG={r['log_growth_futures_charged']:>+7.4f}")


def sweep() -> None:
    """Step 3: sweep on inner-train, select on inner-validation."""
    global N_EVALUATED
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for split, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        print(f"\n{'=' * 100}\n{split}  ({start} .. {end})\n{'=' * 100}")

        for name in ("buy_and_hold", "kelly_regime_v4"):
            r = _row(name, split, lambda name=name: get_strategy(name), start, end)
            rows.append(r)
            _print_row(r)

        print()
        for cfg in GRID:
            tag = _cfg_id(cfg)
            r = _row(tag, split, lambda cfg=cfg: build(cfg), start, end)
            rows.append(r)
            _print_row(r)
            if split == "inner-train":
                N_EVALUATED += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)

    # ---- selection: maximize (variant - v4) log-growth on inner-validation,
    # futures 5x, funding charged.
    valid = df[df.split == "inner-validation"]
    v4_log = float(valid[valid.config == "kelly_regime_v4"]["log_growth_futures_charged"].iloc[0])
    hold_log = float(valid[valid.config == "buy_and_hold"]["log_growth_futures_charged"].iloc[0])
    cand = valid[~valid.config.isin(["kelly_regime_v4", "buy_and_hold"])].copy()
    cand["edge_vs_v4"] = cand["log_growth_futures_charged"] - v4_log
    best = cand.sort_values("edge_vs_v4", ascending=False).iloc[0]

    print(f"\n{'=' * 100}\nSELECTION (inner-validation, futures 5x, funding charged)\n{'=' * 100}")
    print(f"  kelly_regime_v4   log-growth = {v4_log:+.4f}")
    print(f"  buy_and_hold      log-growth = {hold_log:+.4f}")
    print("\n  ranked by (variant - v4) log-growth on inner-validation:")
    for _, r in cand.sort_values("edge_vs_v4", ascending=False).iterrows():
        print(f"    {r['config']:46s} logG={r['log_growth_futures_charged']:>+7.4f}  "
              f"edge={r['edge_vs_v4']:>+7.4f}")
    print(f"\n  FROZEN CONFIG: {best['config']}")
    print(f"    inner-validation log-growth = {best['log_growth_futures_charged']:+.4f}  "
          f"(edge over v4 = {best['edge_vs_v4']:+.4f})")

    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep.csv'}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict lookahead probe, by hand — experiments get no CI protection.

    ``tests/test_causality_strict.py`` parametrizes over the *registry*, so
    an unregistered strategy is unprotected. Two opposite tampers of the
    frame strictly after a cut (x3 / /3 on OHLC, x7 / /7 on volume); every
    column this strategy adds must be bit-identical between the two
    tampered runs at every row at-or-before the cut. The frame is clipped
    to end at inner-validation's own end (2022-12-31) so this probe never
    touches 2023-01-01-onward data either.
    """
    hi = int(DF.index.searchsorted("2022-12-31", side="right"))
    frame = DF.iloc[max(0, hi - 400_000): hi].copy()
    print(f"causality frame: {len(frame):,} bars, {frame.index[0]:%Y-%m-%d} -> "
          f"{frame.index[-1]:%Y-%m-%d} (ends at inner-validation's own end)")

    cuts = [len(frame) - k for k in (5, 20, 100, 1_000, 20_000)]
    cols = ["funding_percentile", "momentum_percentile", "discount", "target"]
    # A representative configuration — the interaction mechanism, not the
    # tuned thresholds, is what this probe is checking.
    cfg = {"funding_threshold": 0.90, "momentum_threshold": 0.50, "fade_strength": 1.0}

    overall_pass = True
    worst_overall = 0.0
    for cut in cuts:
        up, down = frame.copy(), frame.copy()
        for c in ("open", "high", "low", "close"):
            up.iloc[cut:, up.columns.get_loc(c)] *= 3.0
            down.iloc[cut:, down.columns.get_loc(c)] /= 3.0
        up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

        pa = build(cfg).prepare(up)
        pb = build(cfg).prepare(down)
        worst = max(
            float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
            for c in cols
        )
        good = worst < 1e-9
        overall_pass &= good
        worst_overall = max(worst_overall, worst)
        print(f"  cut at bar {cut:,} of {len(frame):,}  "
              f"max |column difference| before the cut = {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    print(f"\n{'PASS' if overall_pass else 'FAIL'} — cuts tried: {cuts}; "
          f"max observed difference before any cut = {worst_overall:.3e}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    if REAL is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    print(f"funding: {len(REAL):,} settlements  {REAL.index[0]} -> {REAL.index[-1]}",
          file=sys.stderr)
    cmds = {"sweep": sweep, "causality": causality}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_novel.py [{'|'.join(cmds)}]")
