"""R-68 operator pre-measurement (backlog **B-35**): is the forced exit
informative? The stopping premium of R-63's zero-crossing, priced.

Prices and one deterministic selection rule only. No equity curve, no
benchmark, no decision rule, no fee-tier cell, no holdout read -- so this can
neither contaminate a branch verdict nor be contaminated by one. It is the
R-68 analogue of `experiments/r63_breadth.py` (which priced the panel's
breadth before either R-63 branch reported) and `experiments/r65_decay.py`
(which priced the signal's decay before either R-65 branch reported). Both
turned an unexplained failure into a priced one; this file is run and
committed BEFORE either R-68 branch is dispatched, for the same reason.

=====================================================================
THE QUESTION
=====================================================================

R-63's frozen selection rule is `eligible = (score > 0)`: every downward
zero-crossing of the incumbent's score forces an exit to flat. R-65 measured
that channel as invariant at 0.386/day and could not touch it; R-67 broke it
30-fold with one asymmetric threshold, and its conservative arm's turnover
fell 17x while its *frictionless* edge rose. Three rounds have now improved
this signal's economics by 10-80x and every one died on the same interval.

B-34 (the next backlog item) proposes to widen the same band further. Before
another grid is run, B-35 asks the prior question, and it is a question about
prices rather than about a backtest:

    **When the incumbent's score crosses zero downward, what happens next?**

Kaminski, K. M., & Lo, A. W. (2014), "When do stop-loss rules stop losses?",
*Journal of Financial Markets* 18, 234-254, give the frame. Their stopping
premium is the difference in expected return between the stopped and the
unstopped policy, and their central result is a sign condition, not an
effect size: under a random walk the premium is **negative** (a stop merely
moves capital from a higher-yielding asset into a lower-yielding one for
nothing), and it is positive only in proportion to *negative* return
persistence following the trigger. They also report stop-losses to be of no
value at short sampling frequencies -- which, at 5-minute bars, cuts in
favour of softening.

Applied here, with a long/flat rule and a cash alternative that earns zero:

    stopping premium (flat)  =  0 - E[ r_incumbent over the next H | crossing ]
    stopping premium (rotate) =  E[ r_basket over H | crossing ] - E[ same ]

so the exit pays for itself **iff the crossing predicts a decline**. If the
crossing is noise, the premium is negative or zero and softening it -- R-67's
delta, B-34's wider delta -- is close to free, which is exactly what R-67
measured from the other direction (drawdown improved at every step, its
named failure mode F1 never arrived).

=====================================================================
THE HEADLINE STATISTIC: WHAT delta ACTUALLY COSTS, IN LOG UNITS
=====================================================================

The frequency statistics above are diagnostics. The number this file exists
to produce is the **direct price of the grace period** an asymmetric exit
threshold buys, measured on prices alone and comparable, in the same units,
against the fee it saves:

    GRACE COST(delta) = sum over events of the incumbent's cumulative log
                        return from the bar its score crosses 0 downward to
                        the bar its score first falls below -delta (or to
                        the bar it recovers above 0, whichever comes first --
                        a recovery is a grace period that ENDED WELL and its
                        return is counted with its own sign, not dropped).

    FEE SAVING(delta) = (events at delta=0 - events at delta) x 2 x fee,
                        the round trips the threshold did not take.

Both sides are log units over the same window. Their difference is the
mechanism's price, before any simulator, benchmark or bootstrap is involved,
and its SIGN is the thing three rounds of D-cells have not been able to
resolve. Note what it is not: it is not a claim about D1, which compares
whole equity paths against a volatility-matched hold. A favourable grace
cost is a necessary condition for the axis, not a sufficient one.

=====================================================================
TWO POPULATIONS, BOTH REPORTED
=====================================================================

(A) **Rule-free.** Every downward zero-crossing of every asset's score. This
    is a property of the signal and the panel, independent of any selection
    rule, and it is the larger sample.
(B) **Incumbent-only.** Crossings of the asset actually held under a plain
    k=1 "hold the highest positive scorer" rule, re-typed here in ten lines
    rather than imported from any branch file, so this measurement shares no
    code path with the arm it is meant to judge. This is the smaller and more
    relevant sample: it is the population R-67's delta actually acts on.

Both are reported for every horizon. Where they disagree, (B) is the one the
mechanism sees and (A) is the one with the error bars worth reading.

=====================================================================
CAVEATS, RECORDED BEFORE THE NUMBERS WERE READ
=====================================================================

1. **Events overlap.** A crossing at bar i and another at bar i+10 share
   almost all of a 5-day forward window. Overlapping means are unbiased but
   their naive standard errors are far too small, so every interval here is
   a stationary block bootstrap over the EVENT SEQUENCE IN TIME ORDER
   (mean block 30 events), and a NON-OVERLAPPING variant -- keep an event
   only if at least H bars have passed since the last kept event of the same
   asset -- is reported beside every overlapping one. Read the two together;
   where they disagree, the non-overlapping one is the honest sample size.
2. **The forward window runs past the end of the data** for events near the
   right edge. Those events are dropped, not truncated, which mildly biases
   the sample toward earlier regimes.
3. **This is not a backtest and its returns are not achievable.** No fees, no
   fills, no vol scale, no deadband, no position sizing. A log return here is
   what the price did, not what a portfolio would have earned.
4. **Windows.** Primary numbers are on 2020-04-01 -> 2022-12-31 (W_TRAIN +
   W_VAL, strictly before the reserved holdout). The full-panel version is
   reported as a secondary, following `r65_decay.py`'s precedent and its
   defence (prices-only), and the holdout counter is **+0**: `W_HOLD` is
   never imported or sliced here.

    .venv/bin/python experiments/r68_stopping_premium.py
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

from tradebot.inference import bootstrap_interval  # noqa: E402

from experiments.r63_shared import (  # noqa: E402
    UNIVERSE_8,
    align_frames,
    load_universe,
)
from experiments.r63_novel_xsmom_rank import (  # noqa: E402
    BARS_PER_DAY,
    cross_sectional_score,
    warm_window,
)

OUT_DIR = ROOT / "reports" / "r68_band"

FEE = 0.001  # the base taker tier every decision cell on this axis uses

# Forward horizons in days. Fixed before any number was read; spans R-63's own
# re-selection cadence to well past R-67's 6.3-day mean tenure.
HORIZONS_DAYS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 30.0)

# The exit thresholds whose grace periods are priced. delta=0.080 is R-67's
# own selected winner; the rest bracket it, including two values past the grid
# edge B-34 exists to extend. Fixed before any number was read.
DELTAS = (0.005, 0.010, 0.020, 0.040, 0.080, 0.120, 0.160)

PRE_HOLDOUT = ("2020-04-01", "2022-12-31")
FULL = ("2020-04-01", None)

BOOT = dict(mean_block=30.0, n_boot=2_000, seed=7)


# ------------------------------------------------------------------ helpers


def incumbent_path(s: np.ndarray) -> np.ndarray:
    """Plain k=1 "hold the highest positive scorer" incumbent, per bar.

    Re-typed from R-63's frozen rule as prose ("hold only positive-scoring
    assets, flat otherwise", k=1) rather than imported from any branch file,
    so this measurement shares no code path with the arms it judges. No
    buffer, no timer, no hysteresis: those are the things under test.

    Strictly causal: row i depends on row i and the previous state only.
    Returns an int array of held column indices, -1 for flat.
    """
    n, m = s.shape
    held = np.full(n, -1, dtype=np.int64)
    cur = -1
    for i in range(n):
        row = s[i]
        if cur >= 0 and not (np.isfinite(row[cur]) and row[cur] > 0.0):
            cur = -1
        if cur < 0:
            elig = np.where(np.isfinite(row) & (row > 0.0))[0]
            if elig.size:
                cur = int(elig[np.argmax(row[elig])])
        held[i] = cur
    return held


def down_crossings(s_col: np.ndarray) -> np.ndarray:
    """Bars where a score crosses from strictly positive to non-positive."""
    prev = s_col[:-1]
    cur = s_col[1:]
    hit = np.isfinite(prev) & np.isfinite(cur) & (prev > 0.0) & (cur <= 0.0)
    return np.where(hit)[0] + 1


def forward_log_return(close: np.ndarray, idx: np.ndarray, h_bars: int):
    """Log return over the next `h_bars` for each event bar; NaN past the end."""
    end = idx + h_bars
    ok = end < len(close)
    out = np.full(len(idx), np.nan)
    out[ok] = np.log(close[end[ok]] / close[idx[ok]])
    return out


def thin_non_overlapping(idx: np.ndarray, asset: np.ndarray, h_bars: int):
    """Keep an event only if >= h_bars have passed since the last kept event
    of the SAME asset. Greedy, left to right, deterministic."""
    keep = np.zeros(len(idx), dtype=bool)
    last: dict[int, int] = {}
    for j in range(len(idx)):
        a = int(asset[j])
        i = int(idx[j])
        if a not in last or i - last[a] >= h_bars:
            keep[j] = True
            last[a] = i
    return keep


def boot_mean(x: np.ndarray) -> tuple[float, float, float]:
    """Point estimate and 95% interval of a mean, stationary-block-bootstrapped
    over the event sequence in time order (blocks absorb event clustering)."""
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (float("nan"), float("nan"), float("nan"))
    # `bootstrap_interval` applies `stat` to a (n_boot, n) matrix as well as to
    # the 1-D series, so the statistic MUST reduce along the last axis only.
    iv = bootstrap_interval(x, lambda a: np.mean(a, axis=-1), **BOOT)
    return (float(iv.point), float(iv.lo), float(iv.hi))


# ------------------------------------------------------------------ measures


def collect_events(score: pd.DataFrame, incumbent_only: bool):
    """(event bar indices, asset column indices) for one population."""
    s = score.to_numpy(dtype=float)
    held = incumbent_path(s) if incumbent_only else None
    idx_list, col_list = [], []
    for c in range(s.shape[1]):
        cross = down_crossings(s[:, c])
        if incumbent_only and cross.size:
            cross = cross[held[cross - 1] == c]
        idx_list.append(cross)
        col_list.append(np.full(cross.size, c, dtype=np.int64))
    if not idx_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    idx = np.concatenate(idx_list)
    col = np.concatenate(col_list)
    order = np.argsort(idx, kind="stable")
    return idx[order], col[order]


def horizon_rows(aligned, score, window_name, pop_name, incumbent_only):
    closes = np.column_stack([aligned[t]["close"].to_numpy(dtype=float)
                              for t in score.columns])
    basket = np.log(closes).mean(axis=1)  # equal-weight basket log price
    idx, col = collect_events(score, incumbent_only)
    rows = []
    for h_days in HORIZONS_DAYS:
        h = int(round(h_days * BARS_PER_DAY))
        if not len(idx):
            continue
        end = idx + h
        ok = end < len(basket)
        fwd = np.full(len(idx), np.nan)
        fwd[ok] = np.log(closes[end[ok], col[ok]] / closes[idx[ok], col[ok]])
        bwd = np.full(len(idx), np.nan)
        bwd[ok] = basket[end[ok]] - basket[idx[ok]]

        # unconditional forward return of the same assets, all bars
        unc = []
        for c in range(closes.shape[1]):
            cl = closes[:, c]
            if len(cl) > h:
                unc.append(np.log(cl[h:] / cl[:-h]))
        unc_mean = float(np.nanmean(np.concatenate(unc))) if unc else float("nan")

        keep = thin_non_overlapping(idx, col, h)
        pt, lo, hi = boot_mean(fwd)
        npt, nlo, nhi = boot_mean(fwd[keep])
        rot_pt, rot_lo, rot_hi = boot_mean(bwd - fwd)
        finite = np.isfinite(fwd)
        rows.append({
            "window": window_name,
            "population": pop_name,
            "horizon_days": h_days,
            "n_events": int(finite.sum()),
            "n_events_nonoverlap": int((finite & keep).sum()),
            # stopping premium vs the FLAT alternative is -E[fwd]; reported as
            # E[fwd] with its interval so the sign is read once, not twice.
            "mean_fwd_log_ret": pt,
            "mean_fwd_lo": lo,
            "mean_fwd_hi": hi,
            "mean_fwd_nonoverlap": npt,
            "mean_fwd_nonoverlap_lo": nlo,
            "mean_fwd_nonoverlap_hi": nhi,
            "uncond_mean_fwd_log_ret": unc_mean,
            "stopping_premium_flat": -pt,
            "stopping_premium_flat_lo": -hi,
            "stopping_premium_flat_hi": -lo,
            "stopping_premium_rotate": rot_pt,
            "stopping_premium_rotate_lo": rot_lo,
            "stopping_premium_rotate_hi": rot_hi,
            "p_negative": float(np.nanmean(fwd[finite] < 0.0)),
            "autocorr_h": _autocorr_at(closes, h),
        })
    return rows


def _autocorr_at(closes: np.ndarray, h: int) -> float:
    """Pooled correlation between consecutive NON-OVERLAPPING h-bar log
    returns, across assets. Kaminski-Lo's persistence quantity: the stopping
    premium's sign tracks it, and it is zero under a random walk."""
    prev_all, next_all = [], []
    for c in range(closes.shape[1]):
        cl = closes[:, c]
        n_blocks = len(cl) // h
        if n_blocks < 3:
            continue
        pts = cl[: n_blocks * h : h]
        r = np.diff(np.log(pts))
        prev_all.append(r[:-1])
        next_all.append(r[1:])
    if not prev_all:
        return float("nan")
    a = np.concatenate(prev_all)
    b = np.concatenate(next_all)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def grace_rows(aligned, score, window_name):
    """The headline: what each delta's grace period costs, against what its
    avoided round trips save. Incumbent population only -- delta acts on the
    held asset and on nothing else."""
    s = score.to_numpy(dtype=float)
    closes = np.column_stack([aligned[t]["close"].to_numpy(dtype=float)
                              for t in score.columns])
    idx, col = collect_events(score, incumbent_only=True)
    n = len(s)
    rows = []
    for delta in DELTAS:
        rets, spans, ended_below, ended_recover = [], [], 0, 0
        for j in range(len(idx)):
            i0, c = int(idx[j]), int(col[j])
            i = i0
            while i + 1 < n:
                i += 1
                v = s[i, c]
                if not np.isfinite(v):
                    break
                if v <= -delta:
                    ended_below += 1
                    break
                if v > 0.0:
                    ended_recover += 1
                    break
            else:
                continue
            rets.append(float(np.log(closes[i, c] / closes[i0, c])))
            spans.append(i - i0)
        if not rets:
            continue
        rets = np.array(rets)
        pt, lo, hi = boot_mean(rets)
        n_ev = len(rets)
        # Events surviving to a delta exit are the ones the threshold does NOT
        # save a round trip on; the rest are avoided exits.
        avoided = n_ev - ended_below
        fee_saving = avoided * 2.0 * FEE
        rows.append({
            "window": window_name,
            "delta": delta,
            "n_grace_periods": n_ev,
            "ended_below_minus_delta": ended_below,
            "ended_recovered": ended_recover,
            "mean_grace_log_ret": pt,
            "grace_lo": lo,
            "grace_hi": hi,
            "total_grace_log_ret": float(rets.sum()),
            "mean_grace_bars": float(np.mean(spans)),
            "mean_grace_days": float(np.mean(spans)) / BARS_PER_DAY,
            "avoided_round_trips": avoided,
            "fee_saving_log_units": fee_saving,
            "net_log_units": float(rets.sum()) + fee_saving,
        })
    return rows


# ------------------------------------------------------------------ driver


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def run_window(frames, window, name):
    warm = align_frames({t: frames[t] for t in UNIVERSE_8}, warm_window(window))
    score_warm = cross_sectional_score(warm)
    idx = warm[UNIVERSE_8[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    aligned = {t: df.loc[idx] for t, df in warm.items()}
    score = score_warm.loc[idx]
    assert not (idx >= pd.Timestamp("2023-01-01", tz="UTC")).any() or name == "FULL", \
        "holdout hygiene: only the FULL secondary window may run past OOS_START"

    print(f"\n== {name} {window}  bars={len(idx):,} ==")
    rows = (horizon_rows(aligned, score, name, "all_assets", False)
            + horizon_rows(aligned, score, name, "incumbent", True))
    for r in rows:
        if r["horizon_days"] in (1.0, 5.0, 14.0):
            print(f"  [{r['population']:>10} H={r['horizon_days']:>5.2f}d] "
                  f"n={r['n_events']:>5} (nonovl {r['n_events_nonoverlap']:>4}) "
                  f"E[fwd] {r['mean_fwd_log_ret']:+.5f} "
                  f"[{r['mean_fwd_lo']:+.5f}, {r['mean_fwd_hi']:+.5f}]  "
                  f"uncond {r['uncond_mean_fwd_log_ret']:+.5f}  "
                  f"P(neg) {r['p_negative']:.3f}  ac {r['autocorr_h']:+.3f}")
    g = grace_rows(aligned, score, name)
    print("  -- grace periods (incumbent) --")
    for r in g:
        print(f"  [delta={r['delta']:.3f}] n={r['n_grace_periods']:>4} "
              f"span {r['mean_grace_days']:>6.2f}d  "
              f"mean {r['mean_grace_log_ret']:+.5f} "
              f"[{r['grace_lo']:+.5f}, {r['grace_hi']:+.5f}]  "
              f"total {r['total_grace_log_ret']:+.4f}  "
              f"fee saved {r['fee_saving_log_units']:+.4f}  "
              f"NET {r['net_log_units']:+.4f}")
    return rows, g


def main():
    frames = load_universe(UNIVERSE_8)
    all_rows, all_grace = [], []
    for window, name in ((PRE_HOLDOUT, "PRE_HOLDOUT"), (FULL, "FULL")):
        rows, g = run_window(frames, window, name)
        all_rows += rows
        all_grace += g
    write_csv(OUT_DIR / "r68_stopping_premium.csv", all_rows)
    write_csv(OUT_DIR / "r68_grace_cost.csv", all_grace)
    print("\nHoldout consultations: +0 (W_HOLD is never imported or sliced here).")


if __name__ == "__main__":
    main()
