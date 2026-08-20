"""R-65 operator measurement: the signal's information decay curve, and the
holding period at which it can afford itself.

Prices only. No strategy, no benchmark, no decision rule, no equity curve --
so this can neither contaminate a branch verdict nor be contaminated by one.
It is the R-65 analogue of `experiments/r63_breadth.py`, which measured the
panel's Grinold breadth before either R-63 branch reported and turned that
round from an unexplained failure into a priced one.

THE QUESTION. R-63 measured the cross-sectional trend signal at exactly one
holding period -- re-select every 5-minute bar -- and found it worth +0.480
log units gross against an 8.02 log-unit turnover bill. R-65's two branches
both slow the signal down. Before either reports, this file measures what is
geometrically available:

    IC(h)          how much of the signal survives a holding period of h
    turnover(h)    what re-selecting every h bars costs
    net(h)         the difference, per day, at 0.10%

The affordability condition is not a matter of opinion once those three
curves exist. If net(h) < 0 for every h, both branches are dead before they
report and the round's answer is arithmetic rather than empirical. If it
crosses, this says where -- and a branch landing far from the crossing point
has an implementation problem rather than a hypothesis problem.

WHAT IS MEASURED

1. **Rank IC.** At each daily sample t, the cross-sectional Spearman
   correlation between the R-63 score across the 8 assets and their forward
   log return over the next h bars. Averaged over t. This is the standard
   Grinold IC and it feeds `IR = IC*sqrt(BR)` directly, so it composes with
   R-63's measured breadth of 1.47.

2. **Top-1 spread.** The mean forward log return over h bars of the
   highest-scoring asset, minus that of the equal-weight basket, expressed
   per day. This is the quantity R-63's k=1 arm actually harvested, so it is
   the one whose sign and magnitude decide affordability.

3. **Leader persistence.** P(the top-ranked asset at t is still top-ranked
   at t+h). One minus this, over h, is the re-selection rate, and it is
   where R-63's 2.86 leader-changes per day came from.

4. **Cost at h.** Re-selecting every h bars turns over at most 2 units of
   notional per change (out of one asset, into another), so the per-day
   round-trip cost is `2 * fee * P(change | h) / (h / 288)`. Stated as an
   upper bound and labelled as one: a re-selection that keeps the incumbent
   costs nothing, which is exactly what this measures.

TWO CAVEATS ON THE OUTPUT, RECORDED BEFORE THE NUMBERS WERE READ INTO ANY
ARGUMENT.

**The h = 1-bar row is an artifact and must not be quoted.** At a 5-minute
forward horizon the top-1 spread is a per-bar quantity multiplied by 288 to
annualize it to a day, and at that horizon it is dominated by bid-ask bounce
and stale-quote effects rather than by the signal. Its implied +0.075/day is
inconsistent by two orders of magnitude with R-63's own measured gross edge
(+0.480 log units over 2,332 days = +0.000206/day) on the same signal and the
same panel, and R-63's number is the one measured through a simulator with
fills. Read the h >= 0.25-day rows; the 1-bar row is retained only so the
grid's left edge is visible.

**`leader_changes_per_day` here is not directly R-63's 2.86.** This measures
U8 (8 assets) and counts a change whenever the argmax moves, including when
the score is negative and R-63's arm would be flat. More assets and no
positivity filter both push the count up, and 5.73 against R-63's 2.86 is
the size of gap those two differences predict. The column is for the SHAPE
of the decay in h, not for a level comparison against R-63.

WINDOWS. The primary numbers are computed on 2020-04-01 -> 2022-12-31 --
W_TRAIN plus W_VAL, strictly before the reserved holdout. R-63's own breadth
measurement used the full panel period and defended it as prices-only; this
file reports that version too, as a secondary, but leads with the
pre-holdout one because leading with the conservative number costs nothing
here and removes the argument entirely.

    python3 experiments/r65_decay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r65_shared import (  # noqa: E402
    BARS_PER_DAY,
    OUT_DIR,
    UNIVERSE_8,
    align_frames,
    cross_sectional_score,
    load_universe,
)

FEE = 0.001  # the base taker tier every decision cell in this round uses

# Holding periods, in days, spanning R-63's own (re-select every bar, ~0.003
# days) to a quarter. Fixed before any number was read.
HOLD_DAYS = (1 / 288, 0.25, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90)

SAMPLE_EVERY = BARS_PER_DAY  # sample the cross-section daily; 8 assets, ~1k rows

PRE_HOLDOUT = ("2020-04-01", "2022-12-31")
FULL = ("2020-04-01", None)


def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Ranks across columns within each row. NaNs are left as NaN."""
    out = np.full(a.shape, np.nan)
    valid = np.isfinite(a)
    for i in range(a.shape[0]):
        v = valid[i]
        if v.sum() < 2:
            continue
        vals = a[i, v]
        order = np.argsort(vals, kind="stable")
        r = np.empty(len(vals))
        r[order] = np.arange(len(vals), dtype=float)
        out[i, v] = r
    return out


def _row_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation across columns, per row. On ranks this is
    Spearman. Rows with <2 jointly-finite entries return NaN."""
    m = np.isfinite(x) & np.isfinite(y)
    n = m.sum(axis=1)
    xs = np.where(m, x, np.nan)
    ys = np.where(m, y, np.nan)
    xc = xs - np.nanmean(xs, axis=1, keepdims=True)
    yc = ys - np.nanmean(ys, axis=1, keepdims=True)
    num = np.nansum(xc * yc, axis=1)
    den = np.sqrt(np.nansum(xc**2, axis=1) * np.nansum(yc**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where((den > 0) & (n >= 2), num / den, np.nan)
    return c


def measure(window) -> pd.DataFrame:
    frames = load_universe(UNIVERSE_8)
    # Warm the anchors: the 80-day anchor needs 80 days of bars before the
    # first usable score, so extend the left edge and slice after scoring.
    start, end = window
    warm = (str((pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=91)).date()), end)
    aligned = align_frames(frames, warm)

    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    logclose = np.column_stack(
        [np.log(aligned[t]["close"].to_numpy(dtype=float)) for t in assets]
    )
    idx = score.index
    keep = idx >= pd.Timestamp(start, tz="UTC")

    s_all = score.to_numpy(dtype=float)
    n = len(idx)

    rows = []
    for hd in HOLD_DAYS:
        h = max(int(round(hd * BARS_PER_DAY)), 1)
        if h >= n:
            continue

        fwd = np.full_like(logclose, np.nan)
        fwd[: n - h] = logclose[h:] - logclose[: n - h]

        ok = keep.copy()
        ok[n - h :] = False  # no forward return available
        sel = np.zeros(n, dtype=bool)
        sel[::SAMPLE_EVERY] = True
        sel &= ok
        sel &= np.isfinite(s_all).all(axis=1)
        if sel.sum() < 30:
            continue

        s = s_all[sel]
        f = fwd[sel]

        ic = _row_corr(_rank_rows(s), _rank_rows(f))
        ic_mean = float(np.nanmean(ic))
        ic_se = float(np.nanstd(ic, ddof=1) / np.sqrt(np.isfinite(ic).sum()))

        top = np.nanargmax(s, axis=1)
        top_fwd = f[np.arange(len(f)), top]
        basket_fwd = np.nanmean(f, axis=1)
        spread = top_fwd - basket_fwd
        spread_per_day = float(np.nanmean(spread) / hd)
        spread_se_per_day = float(
            np.nanstd(spread, ddof=1) / np.sqrt(np.isfinite(spread).sum()) / hd
        )

        # Leader persistence at lag h, on the same daily sample.
        top_all = np.full(n, -1, dtype=int)
        good = np.isfinite(s_all).all(axis=1)
        top_all[good] = np.nanargmax(s_all[good], axis=1)
        pi = np.arange(n)
        pj = pi + h
        pm = (pj < n) & good & keep
        pm[pj.clip(max=n - 1)] = pm[pj.clip(max=n - 1)] & good[pj.clip(max=n - 1)]
        pv = pm & np.zeros(n, dtype=bool)
        pv[::SAMPLE_EVERY] = True
        pv &= pm
        same = top_all[pi[pv]] == top_all[pj[pv]]
        p_same = float(np.mean(same)) if pv.sum() else np.nan
        p_change = 1.0 - p_same

        # Upper-bound cost: a change trades out of one asset and into
        # another, i.e. 2 units of notional, once per h bars.
        cost_per_day = 2.0 * FEE * p_change / hd

        rows.append(
            {
                "hold_days": hd,
                "hold_bars": h,
                "n_samples": int(sel.sum()),
                "ic_mean": ic_mean,
                "ic_se": ic_se,
                "ic_t": ic_mean / ic_se if ic_se > 0 else np.nan,
                "top1_spread_per_day": spread_per_day,
                "top1_spread_se_per_day": spread_se_per_day,
                "p_leader_unchanged": p_same,
                "leader_changes_per_day": p_change / hd,
                "cost_per_day_ub": cost_per_day,
                "net_per_day_ub": spread_per_day - cost_per_day,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, window in (("pre_holdout", PRE_HOLDOUT), ("full", FULL)):
        df = measure(window)
        path = OUT_DIR / f"decay_{label}.csv"
        df.to_csv(path, index=False)
        print(f"\n=== {label}  {window[0]} -> {window[1] or 'last bar'} ===")
        print(
            df[
                [
                    "hold_days",
                    "ic_mean",
                    "ic_t",
                    "top1_spread_per_day",
                    "leader_changes_per_day",
                    "cost_per_day_ub",
                    "net_per_day_ub",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )
        print(
            "  NOTE: the h=1-bar row is a microstructure artifact "
            "(see the module docstring); read h >= 0.25 days."
        )
        pos = df[(df["net_per_day_ub"] > 0) & (df["hold_days"] >= 0.25)]
        if len(pos):
            print(
                "  net > 0 at hold_days: "
                + ", ".join(f"{v:g}" for v in pos["hold_days"])
            )
        else:
            print("  net > 0 at NO holding period in the grid")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
