"""R-68 operator measurement: is R-63's score an expected-return forecast at all?

Prices only. No strategy, no benchmark, no decision rule, no equity curve, no
holdout read. Run after the round's two branches were dispatched and before
either reported, and recorded whatever it found.

=====================================================================
WHY THIS EXISTS -- AND IT IS A CHECK ON THIS ROUND'S OWN PREMISE
=====================================================================

R-68 bounded its delta grid with a number borrowed from the optimal-band
literature (de Lataillade & Chaouki 2020, Eq. 11: the optimal tolerance
saturates at ~1.6 sigma_signal), and its novel branch was asked to DERIVE a
threshold from that literature's closed forms. The round's commissioned
survey then made the precondition explicit, and it is the same precondition
in every one of those papers:

- de Lataillade, Deremble, Potters & Bouchaud (2012), *Journal of Investment
  Strategies* 1(3), 91-115, Proposition 1: the threshold rule is optimal for
  a predictor p with **E[r_t | p_t] = A * p_t, A > 0** (relaxable to any
  continuous, odd, strictly increasing L(p)), p Markovian, and the predictor
  symmetric.
- Dai, Zhang & Zhu (2010) / Guan, Peng & Xu (2020, arXiv:2008.07082,
  Thm 3.1): the state variable is a **conditional probability of the bull
  regime** -- a sufficient statistic -- and the free boundaries are ordered
  around the frictionless indifference point of that posterior.
- The cube-root law in all its forms (Constantinides 1986; Janecek & Shreve
  2004; Muhle-Karbe, Reppen & Soner 2017) prices a band around a
  **frictionless target position**, which presumes the target is the
  optimum of an expected-return objective.

Every one of them requires the signal to be, or to be a monotone transform
of, an expected-return forecast with the right sign. **Nobody in this repo
has ever checked that for R-63's composite score.** Three rounds have now
tuned a band on it, and this round bounded a grid with a constant derived
from that theory. If the precondition fails, the closed forms are not
licensed on this signal, and a band tuned on it is an empirical device with
no theoretical warrant -- which is a different, weaker claim than the one
this round's own pre-registration made.

=====================================================================
WHAT IS MEASURED
=====================================================================

For each forward horizon h, the slope of forward log return on the score:

    A_ts(h)   pooled across assets and bars, WITH an intercept.
              The time-series reading: does a higher score predict a higher
              forward return? This is the quantity every closed form above
              calls `A`.

    A_xs(h)   the same slope after demeaning BOTH the score and the forward
              return WITHIN each bar (across the 8 assets).
              The cross-sectional reading: does a higher score RELATIVE TO
              THE OTHER ASSETS predict a higher forward return relative to
              them? This is what a top-k selector actually uses, and it is
              the version R-63's own permutation control speaks to.

Both come with 95% intervals from a stationary block bootstrap over
NON-OVERLAPPING h-blocks of bars, because overlapping forward windows make a
naive standard error meaningless -- adjacent observations share almost all
of their forward return.

=====================================================================
WHAT THIS CANNOT SHOW, STATED BEFORE THE NUMBERS
=====================================================================

**A linear slope is not the only way a signal can carry information, and a
null or negative slope does not by itself contradict R-63's permutation
control.** The strategy uses the top-ranked asset only, and only when its
score is positive. A signal whose average linear relation to forward returns
is nil can still carry information in its extreme rank -- a tail property a
pooled slope averages away. R-67's conservative arm beat 10 of 10 asset
scrambles at identical turnover on W_FULL6, and that stands.

So the claim this file can support is precise and narrow: whether the
**specific precondition the band literature requires** holds. It is not a
claim that the signal is worthless, and it must not be quoted as one.

Windows: W_TRAIN and W_VAL only. W_FULL6 is not used -- B-33 is unresolved
and already load-bearing on two rounds. **Holdout consultations: +0.**

    .venv/bin/python experiments/r68_signal_slope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import csv  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.inference import stationary_bootstrap_indices  # noqa: E402

from experiments.r68_shared import (  # noqa: E402
    BARS_PER_DAY,
    OUT_DIR,
    UNIVERSE_8,
    W_TRAIN,
    W_VAL,
    align_frames,
    cross_sectional_score,
    load_universe,
    warm_window,
)

HORIZONS_DAYS = (0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 14.0)
N_BOOT = 2_000
MEAN_BLOCK = 20.0   # in non-overlapping h-blocks
SEED = 7


def _slope(x: np.ndarray, y: np.ndarray, intercept: bool) -> float:
    """OLS slope of y on x, with or without an intercept."""
    if intercept:
        xm, ym = x.mean(), y.mean()
        den = float(np.sum((x - xm) ** 2))
        return float(np.sum((x - xm) * (y - ym)) / den) if den else float("nan")
    den = float(np.sum(x * x))
    return float(np.sum(x * y) / den) if den else float("nan")


def _boot_slope(blocks_x, blocks_y, intercept: bool):
    """Point estimate and 95% interval, resampling NON-OVERLAPPING h-blocks.

    Each block contributes one (score, forward return) pair per asset, so the
    forward windows inside a block never overlap and the block bootstrap's
    dependence assumption is the only one doing work.
    """
    n = len(blocks_x)
    x_all = np.concatenate(blocks_x)
    y_all = np.concatenate(blocks_y)
    point = _slope(x_all, y_all, intercept)
    if n < 10:
        return point, float("nan"), float("nan"), len(x_all)
    idx = stationary_bootstrap_indices(n, MEAN_BLOCK, N_BOOT,
                                       np.random.default_rng(SEED))
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sel = idx[b]
        xb = np.concatenate([blocks_x[i] for i in sel])
        yb = np.concatenate([blocks_y[i] for i in sel])
        draws[b] = _slope(xb, yb, intercept)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), len(x_all)


def measure(frames, window, name):
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(window))
    score = cross_sectional_score(warm)
    idx = score.index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    assert not (idx >= pd.Timestamp("2023-01-01", tz="UTC")).any(), \
        "holdout hygiene: W_TRAIN and W_VAL only"

    S = score.loc[idx].to_numpy(dtype=float)
    logc = np.column_stack([np.log(warm[t]["close"]).loc[idx].to_numpy(dtype=float)
                            for t in UNIVERSE_8])
    rows = []
    print(f"\n=== {name} {window}  bars={len(idx):,} ===")
    for hd in HORIZONS_DAYS:
        h = int(round(hd * BARS_PER_DAY))
        starts = np.arange(0, len(S) - h, h)   # NON-OVERLAPPING blocks

        ts_x, ts_y, xs_x, xs_y = [], [], [], []
        for i in starts:
            sx = S[i]
            sy = logc[i + h] - logc[i]
            ok = np.isfinite(sx) & np.isfinite(sy)
            if ok.sum() < 2:
                continue
            sx, sy = sx[ok], sy[ok]
            ts_x.append(sx)
            ts_y.append(sy)
            xs_x.append(sx - sx.mean())
            xs_y.append(sy - sy.mean())

        a_ts, ts_lo, ts_hi, n_ts = _boot_slope(ts_x, ts_y, intercept=True)
        a_xs, xs_lo, xs_hi, _ = _boot_slope(xs_x, xs_y, intercept=False)
        ts_sig = "SIG" if (ts_lo > 0 or ts_hi < 0) else "   "
        xs_sig = "SIG" if (xs_lo > 0 or xs_hi < 0) else "   "
        print(f"  h={hd:>5.2f}d  n_blocks={len(ts_x):>5}  n_obs={n_ts:>6}  "
              f"A_ts {a_ts:+.4e} [{ts_lo:+.3e}, {ts_hi:+.3e}] {ts_sig}   "
              f"A_xs {a_xs:+.4e} [{xs_lo:+.3e}, {xs_hi:+.3e}] {xs_sig}")
        rows.append({
            "window": name, "horizon_days": hd, "n_blocks": len(ts_x),
            "n_obs": n_ts,
            "A_ts": a_ts, "A_ts_lo": ts_lo, "A_ts_hi": ts_hi,
            "A_ts_excludes_zero": bool(ts_lo > 0 or ts_hi < 0),
            "A_xs": a_xs, "A_xs_lo": xs_lo, "A_xs_hi": xs_hi,
            "A_xs_excludes_zero": bool(xs_lo > 0 or xs_hi < 0),
        })
    return rows


def main():
    frames = load_universe(UNIVERSE_8)
    rows = []
    for window, name in ((W_TRAIN, "W_TRAIN"), (W_VAL, "W_VAL")):
        rows += measure(frames, window, name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "r68_signal_slope.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {path}")

    pos_ts = sum(1 for r in rows if r["A_ts"] > 0)
    sig_ts = sum(1 for r in rows if r["A_ts_excludes_zero"] and r["A_ts"] > 0)
    pos_xs = sum(1 for r in rows if r["A_xs"] > 0)
    sig_xs = sum(1 for r in rows if r["A_xs_excludes_zero"] and r["A_xs"] > 0)
    print(f"\n  A_ts > 0 in {pos_ts}/{len(rows)} cells, and significantly so in "
          f"{sig_ts}/{len(rows)}.")
    print(f"  A_xs > 0 in {pos_xs}/{len(rows)} cells, and significantly so in "
          f"{sig_xs}/{len(rows)}.")
    print("  The band literature's precondition is A > 0. Read the cells, not "
          "this summary.")
    print("\nHoldout consultations: +0 (W_TRAIN and W_VAL only).")


if __name__ == "__main__":
    main()
