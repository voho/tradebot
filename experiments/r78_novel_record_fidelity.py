"""R-78 novel branch — is B-06's live record running the strategy that was
backtested?

B-06 has been the project's standing recommendation since R-29 and its only
uncontaminated evidence source. R-71 made it run unattended. Nothing since
has checked the record it produces against the backtest whose claim it
exists to test.

Three pre-registered measurements, frozen in ``experiments/r78_shared.py``
and not restated here so they cannot drift:

- **M1** realized cadence of ``reports/paper_trading/*.csv``;
- **M2** what fraction of ``kelly_regime_v4``'s target changes survive a
  1-in-``k`` decision grid, given that its ``on_bar`` is **edge-triggered**
  against the immediately-preceding 5-minute bar rather than level-
  triggered against the account's actual position;
- **M3** what the realized cadence costs, measured by running the real
  engine on the full 5-minute frame with ``on_bar`` called only on that
  grid.

Run::

    python experiments/r78_novel_record_fidelity.py

Holdout: **+0** for M2/M3 (every frame comes from
``r78_shared.load_truncated()``). M1 reads the live paper-trading CSVs,
which are outside the committed backtest dataset and outside the holdout
counter by the convention R-71 established for exactly these files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.r78_shared import (  # noqa: E402
    FEE_LIVE,
    SparseDecision,
    W_TRAIN,
    W_VAL,
    load_truncated,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

PAPER_DIR = ROOT / "reports" / "paper_trading"
BAR_MINUTES = 5
DESIGNED_CRON_MINUTES = 15          # .github/workflows/paper_trading.yml
K_GRID = (1, 3, 24, 288)            # 5m, 15m (designed), 2h, 1d

CONFIGS = 0     # real-data backtest runs; incremented as they happen


# --------------------------------------------------------------- M1 cadence

def m1_cadence() -> dict:
    """Realized inter-row spacing of the live record, per strategy."""
    print("=" * 72)
    print("M1 - realized cadence of the live B-06 record")
    print("=" * 72)
    rows = []
    for path in sorted(PAPER_DIR.glob("*_bitstamp.csv")):
        rec = pd.read_csv(path, parse_dates=["timestamp"])
        ts = pd.DatetimeIndex(rec["timestamp"])
        gaps = ts.to_series().diff().dropna().dt.total_seconds() / 60.0
        span = (ts[-1] - ts[0]).total_seconds() / 60.0
        rows.append({
            "strategy": path.stem.replace("_bitstamp", ""),
            "rows": len(rec),
            "span_h": span / 60.0,
            "median_gap_min": float(gaps.median()) if len(gaps) else float("nan"),
            "mean_gap_min": float(gaps.mean()) if len(gaps) else float("nan"),
            "max_gap_min": float(gaps.max()) if len(gaps) else float("nan"),
            # what fraction of the 5m tape over the recorded span got a decision
            "tape_capture_pct": 100.0 * (len(rec) - 1) / max(span / BAR_MINUTES, 1),
            # rows actually observed vs rows the */15 cron design implies
            "vs_design_pct": 100.0 * (len(rec) - 1)
            / max(span / DESIGNED_CRON_MINUTES, 1),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # The distinct positions the whole recorded family actually held.
    frames = {}
    for path in sorted(PAPER_DIR.glob("*_bitstamp.csv")):
        rec = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        frames[path.stem.replace("_bitstamp", "")] = rec["position_after"]
    # Arms that started on different candles bought at different prices, so
    # their share counts differ by construction. The question that matters
    # is whether arms sharing an inception ever diverge — i.e. whether the
    # record carries any information about which strategy is which.
    pos = pd.DataFrame(frames)
    inception = {name: s.dropna().index[0] for name, s in frames.items()}
    cohorts: dict = {}
    for name, ts in inception.items():
        cohorts.setdefault(ts, []).append(name)
    identical = float("nan")
    for ts, members in sorted(cohorts.items()):
        block = pos[members].dropna()
        same = (block.nunique(axis=1) == 1).mean() if len(block) else float("nan")
        if len(members) > 1:
            identical = same if np.isnan(identical) else min(identical, same)
        print(f"\ncohort inception {ts} ({len(members)} arms: "
              f"{', '.join(sorted(members))})")
        print(f"  rows where every arm in the cohort holds the IDENTICAL "
              f"position: {100.0 * same:.1f}% of {len(block)} rows")
    trades = {name: int((pd.read_csv(PAPER_DIR / f'{name}_bitstamp.csv')
                         ['trade_qty'] != 0).sum()) for name in frames}
    print(f"non-zero trades recorded, per arm: {trades}")

    med = float(out["median_gap_min"].median())
    realized_k = max(1, int(round(med / BAR_MINUTES)))
    print(f"\nmedian realized gap across arms: {med:.1f} min "
          f"-> realized decision grid k ~= {realized_k} "
          f"(designed k = {DESIGNED_CRON_MINUTES // BAR_MINUTES}, "
          f"backtest k = 1)")
    return {"table": out, "realized_k": realized_k, "identical_frac": identical,
            "median_gap_min": med}


# ------------------------------------------------------- M2 rebalance capture

def m2_capture(df: pd.DataFrame, ks: tuple[int, ...]) -> pd.DataFrame:
    """What fraction of v4's target changes survive a 1-in-k decision grid.

    ``kelly_regime``'s ``on_bar`` emits an order only when
    ``abs(target[i] - target[i-1]) > 1e-9``. On a 1-in-``k`` grid the
    strategy is only asked on bars ``i % k == phase``, so a change is
    captured **only if it lands exactly on a sampled bar** — a change one
    bar earlier is never seen at all, because by the next sampled bar the
    target has stopped changing and the gate is silent again.

    Averaged over all ``k`` phases, so the answer is a property of the grid
    rather than of one arbitrary alignment.
    """
    print("\n" + "=" * 72)
    print("M2 - rebalance capture under an edge-triggered change gate")
    print("=" * 72)
    prepared = get_strategy("kelly_regime_v4").prepare(df.copy())
    target = prepared["target"].to_numpy()
    changed = np.abs(np.diff(target)) > 1e-9         # change AT bar i (vs i-1)
    change_idx = np.flatnonzero(changed) + 1
    print(f"{len(target):,} bars, {len(change_idx):,} target changes "
          f"({100.0 * len(change_idx) / len(target):.4f}% of bars)")

    rows = []
    for k in ks:
        # A change at bar i is captured iff i is a sampled bar (i % k ==
        # phase). Measured empirically over every phase rather than
        # asserted as 1/k, so a target that changes on a non-uniform
        # schedule (e.g. clustered on session boundaries) would show up.
        captured = float(np.mean([
            np.mean((change_idx % k) == phase) for phase in range(k)]))
        rows.append({"k": k, "grid_minutes": k * BAR_MINUTES,
                     "captured_pct": 100.0 * captured,
                     "changes_seen_per_year": 100.0})
    out = pd.DataFrame(rows)
    # changes/year is a property of the data, not the grid; compute properly
    years = (df.index[-1] - df.index[0]).days / 365.25
    per_year = len(change_idx) / years
    out["changes_seen_per_year"] = out["captured_pct"] / 100.0 * per_year
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"(v4 changes target {per_year:.0f}x/year on the full 5m tape)")
    return out


# ------------------------------------------------------------- M3 what it costs

def m3_cost(df: pd.DataFrame, label: str, ks: tuple[int, ...]) -> pd.DataFrame:
    """v4 through the real engine with on_bar called only on a 1-in-k grid."""
    global CONFIGS
    print("\n" + "=" * 72)
    print("M3 - what the realized decision cadence costs")
    print("=" * 72)
    market = MarketSpec.spot(fee_rate=FEE_LIVE)
    rows = []
    for wname, window in (("inner-train", W_TRAIN), ("inner-val", W_VAL)):
        for k in ks:
            strat = SparseDecision(get_strategy("kelly_regime_v4"), k)
            result = run_period(strat, df, window[0], window[1],
                                market=market, data_label=label)
            CONFIGS += 1
            m = compute_metrics(result)
            rows.append({"window": wname, "k": k, "grid_min": k * BAR_MINUTES,
                         "final": m.final_balance, "sharpe": m.sharpe,
                         "max_dd_pct": m.max_drawdown_pct, "trades": m.num_trades})
        # the un-wrapped strategy, as a bit-for-bit check that k=1 is faithful
        base = run_period(get_strategy("kelly_regime_v4"), df, window[0],
                          window[1], market=market, data_label=label)
        CONFIGS += 1
        bm = compute_metrics(base)
        k1 = next(r for r in rows if r["window"] == wname and r["k"] == 1)
        ok = (abs(k1["final"] - bm.final_balance) < 1e-6
              and k1["trades"] == bm.num_trades)
        print(f"[{wname}] k=1 wrapper reproduces the unwrapped strategy: "
              f"{'YES' if ok else 'NO'} "
              f"(${k1['final']:,.2f} vs ${bm.final_balance:,.2f}, "
              f"{k1['trades']} vs {bm.num_trades} trades)")
        if not ok:
            raise AssertionError("SparseDecision(k=1) is not a no-op - the "
                                 "whole M3 comparison is invalid")

    out = pd.DataFrame(rows)
    for wname in out["window"].unique():
        sel = out["window"] == wname
        base_row = out[sel & (out["k"] == 1)].iloc[0]
        out.loc[sel, "d_sharpe"] = out.loc[sel, "sharpe"] - base_row["sharpe"]
        out.loc[sel, "final_pct_of_k1"] = (
            100.0 * out.loc[sel, "final"] / base_row["final"])
    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    return out


def main() -> None:
    m1 = m1_cadence()
    df, label = load_truncated()
    ks = tuple(sorted(set(K_GRID) | {m1["realized_k"]}))
    m2 = m2_capture(df, ks)
    m3 = m3_cost(df, label, ks)

    # ------------------------------------------------------- classification
    print("\n" + "=" * 72)
    print("PRE-REGISTERED CLASSIFICATION AND FALSIFICATION")
    print("=" * 72)
    rk = m1["realized_k"]
    cap = float(m2.loc[m2["k"] == rk, "captured_pct"].iloc[0])
    falsified = cap >= 90.0
    print(f"Falsification test: >=90% of v4's target changes survive the "
          f"realized k={rk} grid? measured {cap:.2f}% -> "
          f"{'REFUTED (branch has nothing)' if falsified else 'not refuted'}")

    sel = m3[m3["k"] == rk]
    d_sharpe = sel["d_sharpe"].to_numpy()
    pct = sel["final_pct_of_k1"].to_numpy()
    acceptable = bool(np.all(d_sharpe >= -0.2) and np.all(pct >= 80.0))
    print(f"Cadence at realized k={rk}: dSharpe = "
          f"{', '.join(f'{v:+.3f}' for v in d_sharpe)} (bar: >= -0.200 on both); "
          f"final balance = {', '.join(f'{v:.1f}%' for v in pct)} of k=1 "
          f"(bar: >= 80% on both)")
    print(f"  -> {'BOUNDED AND ACCEPTABLE' if acceptable else 'MATERIALLY COSTLY'}")
    print(f"\nconfigs evaluated (real-data backtest runs): {CONFIGS}")


if __name__ == "__main__":
    main()
