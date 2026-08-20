"""R-68 operator check: are this axis's intervals too narrow for the SLOW arms?

Every bootstrap interval in R-63, R-65, R-67 and R-68 uses one block length --
`BOOT_KW = dict(mean_block=30.0, ...)` in `experiments/r63_shared.py` -- for
every arm, from R-63's 3.44 round-trips/day to R-67's 0.102/day.

The round's commissioned literature survey flagged this as a specific,
directional bias:

    Hysteresis lengthens holding periods, which lengthens the
    autocorrelation of the P&L series. A block length tuned on the raw
    arm will be too short for the banded arm and will UNDERSTATE the
    banded arm's standard error -- biasing the comparison in the
    banded arm's favour.

Politis, D. N., & White, H. (2004), "Automatic block-length selection for the
dependent bootstrap," *Econometric Reviews* 23(1), 53-70, with the correction
in Patton, A., Politis, D. N., & White, H. (2009), *Econometric Reviews*
28(4), 372-375, give the automatic selection rule. Rather than implement it
and inherit its own tuning choices, this file does the blunter and more
transportable thing: it reports the D1 interval as a FUNCTION of block
length, for a fast arm and a slow arm, and lets the reader see whether the
conclusion depends on the choice.

If the intervals are stable in block length, the concern is closed and every
published number on this axis stands as reported. If the slow arm's interval
widens materially while the fast arm's does not, then the bias the survey
names is real, it runs in the candidate's favour, and every interval this
axis has published is too narrow -- which would matter, because three rounds
have now been decided by exactly these intervals.

Prices only, W_TRAIN and W_VAL only, no holdout read (**+0**), no selection,
no verdict.

    .venv/bin/python experiments/r68_block_sensitivity.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tradebot.inference import (  # noqa: E402
    max_drawdown_from_returns,
    paired_bootstrap,
    total_log_return,
)

from experiments.r68_shared import OUT_DIR, W_TRAIN, W_VAL, load_universe, UNIVERSE_8  # noqa: E402
from experiments.r68_inference import cell_series  # noqa: E402

BLOCKS = (5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0)
ARMS = (0.000, 0.080)   # R-65's frozen winner (fast) and R-67's (slow)
N_BOOT = 2_000
SEED = 7


def autocorr_time(x: np.ndarray, max_lag: int = 60) -> float:
    """Integrated autocorrelation time of a daily return series, in days.

    tau = 1 + 2 * sum_{k=1..K} rho_k, truncated at the first non-positive
    rho (Geyer's initial-positive-sequence rule). A block bootstrap needs a
    block length comfortably longer than this.
    """
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    if n < 20:
        return float("nan")
    denom = float(np.dot(x, x))
    tau = 1.0
    for k in range(1, min(max_lag, n - 2) + 1):
        rho = float(np.dot(x[:-k], x[k:]) / denom)
        if rho <= 0:
            break
        tau += 2.0 * rho
    return tau


def main():
    frames = load_universe(UNIVERSE_8)
    rows = []
    for window, name in ((W_TRAIN, "W_TRAIN"), (W_VAL, "W_VAL")):
        print(f"\n=== {name} {window} ===")
        for delta in ARMS:
            a, b, matched, targets = cell_series(frames, window, delta)
            if a is None:
                print(f"  delta={delta}: VOIDED (risk match failed)")
                continue
            d = a - b
            tau = autocorr_time(d)
            print(f"  -- delta={delta:.3f}  (risk_matched={matched})  "
                  f"integrated autocorrelation time of the D1 difference "
                  f"series = {tau:.2f} days --")
            for mb in BLOCKS:
                kw = dict(mean_block=mb, n_boot=N_BOOT, seed=SEED)
                g = paired_bootstrap(a, b, total_log_return, **kw)
                dd = paired_bootstrap(a, b, max_drawdown_from_returns, **kw)
                width = g.diff.hi - g.diff.lo
                print(f"     block {mb:>5.0f}d   growth {g.diff.point:+.4f} "
                      f"[{g.diff.lo:+.4f}, {g.diff.hi:+.4f}]  width {width:.4f}"
                      f"   dd {dd.diff.point:+.2f}pp "
                      f"[{dd.diff.lo:+.2f}, {dd.diff.hi:+.2f}]")
                rows.append({
                    "window": name, "delta": delta, "mean_block_days": mb,
                    "autocorr_time_days": tau, "risk_matched": matched,
                    "growth_diff": g.diff.point, "growth_lo": g.diff.lo,
                    "growth_hi": g.diff.hi, "growth_width": width,
                    "dd_diff": dd.diff.point, "dd_lo": dd.diff.lo,
                    "dd_hi": dd.diff.hi,
                    "excludes_zero": bool(g.diff.lo > 0 or g.diff.hi < 0),
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "r68_block_sensitivity.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {path}")

    print("\n  Width of the D1 growth interval at the published 30-day block, "
          "versus 120 days:")
    for name in ("W_TRAIN", "W_VAL"):
        for delta in ARMS:
            sel = {r["mean_block_days"]: r for r in rows
                   if r["window"] == name and r["delta"] == delta}
            if 30.0 in sel and 120.0 in sel:
                w30, w120 = sel[30.0]["growth_width"], sel[120.0]["growth_width"]
                print(f"    {name} delta={delta:.3f}:  {w30:.4f} -> {w120:.4f}  "
                      f"({w120 / w30:.2f}x)")
    print("\n  No cell changes any verdict; nothing here is selected on.")
    print("Holdout consultations: +0 (W_TRAIN and W_VAL only).")


if __name__ == "__main__":
    main()
