#!/usr/bin/env python
"""R-99 Kill Switch A (operator-run once, before any episode-level number):
does each of the 9 grid cells' RJ alarm z-score actually cross
`Z_THRESH=2.0` at least once across the full 2017-2022 pre-holdout history?
A cell that never fires is degenerate and disqualified from being PRIMARY,
same posture as R-96/R-97/R-98's own Kill Switch A. This is a sanity gate
on the estimator's own non-degeneracy, not a search over episode-level
performance -- no episode, lead time, or gate pass/fail number is computed
here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402
from experiments.r99_shared import (  # noqa: E402
    OOS_START, DETECTION_WINDOW_DAYS_GRID, BASELINE_WINDOW_DAYS_GRID,
    Z_THRESH, daily_rv_bv_jump, rj_signal_zscore, assert_no_holdout,
)

import pandas as pd

DATA_DIR = ROOT / "data"


def main() -> None:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}", file=sys.stderr)

    daily = daily_rv_bv_jump(df)
    rj = daily["rj"]
    n_days_with_rj = rj.notna().sum()
    print(f"Daily RV/BV computed on {len(daily)} calendar days; {n_days_with_rj} with valid RJ "
          f"(NaN days are short-bar-count days below MIN_BARS_PER_DAY).", file=sys.stderr)
    print(f"RJ summary: mean={rj.mean():.4f} median={rj.median():.4f} "
          f"p95={rj.quantile(0.95):.4f} max={rj.max():.4f}", file=sys.stderr)

    results = {}
    for det in DETECTION_WINDOW_DAYS_GRID:
        for base in BASELINE_WINDOW_DAYS_GRID:
            z = rj_signal_zscore(rj, det, base)
            max_z = float(np.nanmax(z.to_numpy())) if z.notna().any() else float("nan")
            n_fire = int((z >= Z_THRESH).sum())
            fires = n_fire > 0
            results[(det, base)] = (max_z, n_fire, fires)
            print(f"detection={det:>4}d baseline={base:>4}d  max_z={max_z:6.2f}  "
                  f"n_bars_fired={n_fire:>4}  {'FIRES' if fires else 'DEGENERATE (never fires)'}")

    # A-priori natural choice by analogy to every predecessor's own
    # BASELINE_WINDOW_DAYS=730 convention and this round's own grid centre:
    natural = (90, 730)
    print(f"\nA-priori natural grid-centre cell {natural}: "
          f"{'FIRES' if results[natural][2] else 'DEGENERATE'} "
          f"(max_z={results[natural][0]:.2f}, n_fired={results[natural][1]})")

    if results[natural][2]:
        primary = natural
        reason = "grid-centre cell fires; no substitution needed"
    else:
        # Same rule R-98 used: smallest deviation from the natural centre
        # among firing cells, chosen for non-degeneracy alone.
        firing = [(k, v) for k, v in results.items() if v[2]]
        if not firing:
            print("\nALL 9 CELLS DEGENERATE. No primary can be chosen. Round must stop here.")
            return
        def dist(cell):
            (d, b) = cell[0]
            di = DETECTION_WINDOW_DAYS_GRID.index(d)
            bi = BASELINE_WINDOW_DAYS_GRID.index(b)
            nd = DETECTION_WINDOW_DAYS_GRID.index(natural[0])
            nb = BASELINE_WINDOW_DAYS_GRID.index(natural[1])
            return abs(di - nd) + abs(bi - nb)
        firing.sort(key=dist)
        primary = firing[0][0]
        reason = "grid-centre cell degenerate; nearest firing cell chosen for non-degeneracy alone"

    print(f"\nPRIMARY = detection={primary[0]}d, baseline={primary[1]}d  ({reason})")


if __name__ == "__main__":
    main()
