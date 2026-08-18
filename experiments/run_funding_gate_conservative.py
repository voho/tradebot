#!/usr/bin/env python
"""Driver for backlog B-05 (conservative arm) — funding decile gate on kelly_regime_v4.

Splits (funding data only exists 2020-01-01 .. 2023-12-31, so this
experiment's splits are narrower than the project's usual convention;
see experiments/funding_gate_conservative.py for the mechanism)::

    inner-train       2020-01-01 -> 2020-12-31   sweep the hyperparameter grid
    inner-validation  2021-01-01 -> 2022-12-31   select the single frozen config
    holdout           2023-01-01 ->              NOT TOUCHED by this file

Usage::

    python experiments/run_funding_gate_conservative.py sweep       # step 3
    python experiments/run_funding_gate_conservative.py causality   # by-hand lookahead probe
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

from experiments.funding_gate_conservative import FundingGateConservative  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
if REAL_FUNDING is None:
    raise SystemExit("no funding data committed; expected data/btcusdt_perp_funding_8h.csv.gz")

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2020-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS_START = "2023-01-01"  # NEVER passed to a backtest in this file

# 3 x 2 x 2 = 12 configurations, per the backlog's trials-budget cap.
GRID = []
for decile_in in (0.85, 0.90, 0.95):
    for delta in (0.05, 0.10):
        for lookback in (90, 180):
            GRID.append({
                "decile_in": decile_in,
                "decile_out": round(decile_in - delta, 2),
                "funding_lookback_days": lookback,
            })
assert len(GRID) == 12, f"grid drifted from 12 configurations: {len(GRID)}"

OUT = ROOT / "reports" / "funding_gate_conservative"

N_EVALUATED = 0  # distinct configurations swept in step 3 (trials-budget count)


# --------------------------------------------------------------------- helpers


def _period(strategy, market, start=None, end=None, *, funding=None, df=None,
            balance: float = 1_000.0):
    """One backtest over [start, end], warmed on the bars before it.

    Mirrors scripts/funding_study.py's ``_period`` — ``run_period`` (used
    for the funding-free comparisons elsewhere in the project) has no
    ``funding=`` kwarg, so a funding-charged backtest has to go through
    ``run_backtest`` directly with the same warmup-prefix handling done by
    hand. Guaranteed never to touch 2023-01-01 onward: callers only ever
    pass TRAIN or VALID below.
    """
    frame = DF if df is None else df
    lo = 0 if start is None else int(frame.index.searchsorted(start))
    hi = len(frame) if end is None else int(frame.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, frame.iloc[lo - pre: hi], market, balance,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = (raw if pre == 0 else
               replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:]))
    return compute_metrics(trimmed), raw.funding_paid


def make_variant(cfg: dict) -> FundingGateConservative:
    return FundingGateConservative(funding=REAL_FUNDING, **cfg)


def row_for(tag: str, split: str, cfg: dict | None, m, funding_paid=None) -> dict:
    r = {"tag": tag, "split": split, "final": m.final_balance,
         "sharpe": m.sharpe, "max_dd": m.max_drawdown_pct,
         "trades": m.num_trades, "funding_paid": funding_paid}
    r.update({"decile_in": None, "decile_out": None, "lookback": None} if cfg is None
              else {"decile_in": cfg["decile_in"], "decile_out": cfg["decile_out"],
                    "lookback": cfg["funding_lookback_days"]})
    return r


def log_growth(final_balance: float, start_balance: float = 1_000.0) -> float:
    return float(np.log(final_balance / start_balance))


# ------------------------------------------------------------------------ sweep


def sweep() -> None:
    global N_EVALUATED
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def measure_all(strategy, cfg, tag):
        """spot (no funding) + futures funding-free + futures funding-charged, both splits."""
        out = {}
        for split, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
            m_spot, _ = _period(strategy, SPOT, start, end, funding=None)
            rows.append(row_for(f"{tag} spot", split, cfg, m_spot, funding_paid=0.0))
            m_fut_free, _ = _period(strategy, FUTURES, start, end, funding=None)
            rows.append(row_for(f"{tag} futures(no funding)", split, cfg, m_fut_free,
                                funding_paid=0.0))
            m_fut_paid, paid = _period(strategy, FUTURES, start, end, funding=REAL_FUNDING)
            rows.append(row_for(f"{tag} futures(funding charged)", split, cfg,
                                m_fut_paid, funding_paid=paid))
            out[split] = m_fut_paid
        return out

    # Reference rows: unmodified kelly_regime_v4 and buy_and_hold.
    baseline_valid = {}
    for name in ("kelly_regime_v4", "buy_and_hold"):
        baseline_valid[name] = measure_all(get_strategy(name), None, name)

    # The 12-configuration sweep.
    results = []
    for i, cfg in enumerate(GRID):
        strategy = make_variant(cfg)
        tag = (f"gate(in={cfg['decile_in']:.2f},out={cfg['decile_out']:.2f},"
               f"lb={cfg['funding_lookback_days']}d)")
        per_split = measure_all(strategy, cfg, tag)
        N_EVALUATED += 1  # one distinct configuration, regardless of how many
                          # markets/splits it is scored on (matched_risk.py convention)
        variant_lg = log_growth(per_split["inner-validation"].final_balance)
        baseline_lg = log_growth(baseline_valid["kelly_regime_v4"]["inner-validation"].final_balance)
        results.append({**cfg, "tag": tag,
                        "valid_log_growth": variant_lg,
                        "baseline_log_growth": baseline_lg,
                        "edge": variant_lg - baseline_lg})
        print(f"[{i + 1:2d}/12] {tag:42s} "
              f"inner-valid log-growth={variant_lg:+.4f}  "
              f"vs v4={baseline_lg:+.4f}  edge={variant_lg - baseline_lg:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)
    res_df = pd.DataFrame(results).sort_values("edge", ascending=False)
    res_df.to_csv(OUT / "selection.csv", index=False)

    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'sweep.csv'}, {OUT / 'selection.csv'}")

    # --------------------------------------------------------------- full table
    print("\nFull sweep table (inner-train and inner-validation):")
    print(f"  {'tag':46s} {'split':17s} {'final':>12s} {'sharpe':>7s} "
          f"{'DD%':>6s} {'trades':>7s} {'funding$':>10s}")
    for _, r in df.iterrows():
        fp = "" if r.funding_paid is None else f"${r.funding_paid:>9,.0f}"
        print(f"  {r.tag:46s} {r.split:17s} ${r.final:>11,.0f} {r.sharpe:>7.2f} "
              f"{r.max_dd:>5.1f}% {r.trades:>7.0f} {fp:>10s}")

    # --------------------------------------------------------------- selection
    best = res_df.iloc[0]
    print(f"\nSelected on inner-validation (max variant-minus-v4 log-growth):")
    print(f"  decile_in={best.decile_in}  decile_out={best.decile_out}  "
          f"funding_lookback_days={int(best.lookback)}")
    print(f"  inner-validation log-growth: variant={best.valid_log_growth:+.4f}  "
          f"kelly_regime_v4={best.baseline_log_growth:+.4f}  "
          f"edge={best.edge:+.4f}")

    frozen = {"decile_in": float(best.decile_in), "decile_out": float(best.decile_out),
              "funding_lookback_days": int(best.lookback)}
    (OUT / "frozen.json").write_text(pd.Series(frozen).to_json(indent=2) + "\n")
    print(f"written: {OUT / 'frozen.json'}")

    # Report the frozen config's full metric set explicitly, side by side
    # with the two reference rows, on both splits.
    print("\nFrozen configuration, full detail:")
    frozen_strategy_rows = [r for r in rows if r["tag"].startswith(
        f"gate(in={frozen['decile_in']:.2f},out={frozen['decile_out']:.2f},"
        f"lb={frozen['funding_lookback_days']}d)")]
    ref_rows = [r for r in rows if r["tag"].startswith("kelly_regime_v4")
               or r["tag"].startswith("buy_and_hold")]
    for r in ref_rows + frozen_strategy_rows:
        fp = "" if r["funding_paid"] is None else f"${r['funding_paid']:>9,.0f}"
        print(f"  {r['tag']:46s} {r['split']:17s} ${r['final']:>11,.0f} "
              f"sharpe={r['sharpe']:>6.2f} DD={r['max_dd']:>5.1f}% "
              f"trades={r['trades']:>5.0f} funding={fp:>10s}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """By-hand lookahead probe — unregistered strategies get no CI protection.

    Same two-opposite-tampers procedure as experiments/run_matched_risk.py's
    ``causality()``: bars strictly after a cut are multiplied by 3 (OHLC)
    / 7 (volume) in one copy and divided by the same factors in the other.
    Every column FundingGateConservative.prepare adds must be bit-identical
    between the two tampered copies for every row at or before the cut.
    """
    df = DF.loc[:"2022-12-31"].copy()  # inner-train + inner-validation only
    offsets = (5, 20, 100, 1_000, 20_000)
    cuts = [len(df) - k for k in offsets]

    cfg = {"decile_in": 0.90, "decile_out": 0.80, "funding_lookback_days": 180}
    check_cols = ["funding_gate", "target", "funding_pct", "funding_lag"]

    from tradebot.broker import PaperBroker

    worst_overall = 0.0
    ok = True
    print(f"comparing columns {check_cols} for rows at-or-before each cut, "
          f"cuts (bars from end of df): {list(offsets)}")
    for offset, cut in zip(offsets, cuts):
        # Each cut is its OWN independent tamper: bars strictly after `cut`
        # are multiplied by 3 (OHLC) / 7 (volume) in one copy and divided
        # by the same factors in the other. Rows at-or-before `cut` must
        # then be bit-identical between the two copies.
        up, down = df.copy(), df.copy()
        for col in ("open", "high", "low", "close"):
            up.iloc[cut + 1:, up.columns.get_loc(col)] *= 3.0
            down.iloc[cut + 1:, down.columns.get_loc(col)] /= 3.0
        up.iloc[cut + 1:, up.columns.get_loc("volume")] *= 7.0
        down.iloc[cut + 1:, down.columns.get_loc("volume")] /= 7.0

        pa = make_variant(cfg).prepare(up)
        pb = make_variant(cfg).prepare(down)

        row_ok = True
        for col in check_cols:
            a = pa[col].to_numpy()[:cut + 1]
            b = pb[col].to_numpy()[:cut + 1]
            diff = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
            worst_overall = max(worst_overall, diff)
            if diff >= 1e-9:
                row_ok = False
                print(f"  offset={offset:>7,d} cut={cut:>8,d}  col={col:14s} "
                      f"max|diff|={diff:.3e}  FAIL")

        # Order-level check: decisions at (and just before) the cut must
        # also match, same tampered pair.
        def decisions(prepared_frame):
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in (cut - 2, cut - 1, cut):
                ctx = Context(prepared_frame, i, broker)
                make_variant(cfg).on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        da, db = decisions(pa), decisions(pb)
        bad = da != db
        row_ok &= not bad
        ok &= row_ok
        status = "PASS" if row_ok else "FAIL"
        print(f"  offset={offset:>7,d} cut={cut:>8,d}  columns "
              f"{'identical' if row_ok else 'DIFFER'} at-or-before cut, "
              f"orders {'match' if not bad else 'DIFFER'}  {status}")

    print(f"\nmax |column difference| at-or-before any tested cut: {worst_overall:.3e}")
    print(f"cuts tried (bars from end of df): {list(offsets)}")
    print(f"RESULT: {'PASS' if ok and worst_overall == 0.0 else ('PASS (nonzero tolerance)' if ok else 'FAIL')}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    print(f"{len(REAL_FUNDING):,} funding settlements  "
          f"{REAL_FUNDING.index[0]} -> {REAL_FUNDING.index[-1]}", file=sys.stderr)
    cmds = {"sweep": sweep, "causality": causality}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_conservative.py "
              f"[{'|'.join(cmds)}]")
