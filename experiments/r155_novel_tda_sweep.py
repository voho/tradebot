#!/usr/bin/env python
"""R-155 NOVEL branch: sweep the causal-Takens/H0-persistence TDA detector's
own construction parameters (window_days, embed_dim) against the
R-82-identical six-episode detection-lag Step-A gate
(`experiments.r155_shared.step_a_gate`), holding embed_delay/trail_days/
z_thresh fixed at the conservative branch's own values throughout, exactly
as pre-registered in `experiments/r155_shared.py`'s module docstring.

See `experiments/r155_shared.py` for the full mechanism, citation trail,
not-a-duplicate-of list, and the PRE-REGISTERED stop rule this file
implements verbatim (frozen before any of today's numbers existed):

    at least one winning cell (n_pass >= 4/6) must ALSO pass
    `r155_shared.plateau_ok()` (>= 1 immediate grid-neighbour in EACH swept
    dimension also clears n_pass >= 4) for this branch to proceed past
    Step-A. Otherwise: STOP, no Step-B, no holdout read, report NEGATIVE.

TIE-BREAK RULE (stated explicitly, per task instructions): among cells
tied on the top n_pass, the winner is the one with smallest window_days
first, then smallest embed_dim -- the most literal ("react on the least
history") and fastest-reacting reading of the construction, and the
cheapest to compute, matching R-139's own novel-branch tie-break
convention (prefer the least-tuned, most-literal cell when several tie).

=====================================================================
RESULT (this file is a genuine, re-runnable computation -- the numbers
below are what running it produces; `python experiments/r155_novel_tda_sweep.py`
reproduces them deterministically)
=====================================================================

All 9 cells of `NOVEL_WINDOW_GRID x NOVEL_DIM_GRID` were run. NOT ONE CELL
reaches the n_pass >= 4/6 bar; the best cells reach only 2/6 (window_days=20,
embed_dim in {2,3,4}; window_days=30, embed_dim=2 -- four cells tied at
2/6), the tie-break winner being window_days=20, embed_dim=2 (smallest
window_days, then smallest embed_dim among the tied cells). window_days=10
tops out at 1/6 regardless of embed_dim, and window_days=30 falls back to
1/6 at embed_dim in {3,4}. All 9 cells are below the promotion bar, so
`plateau_ok()` is never invoked in earnest -- there is no winner to seed it;
the pre-registered stop rule fires on the FIRST clause, "no cell reaches
n_pass>=4", before the plateau question is reachable at all.

VERDICT: NEGATIVE at Step-A. STOP. Per the pre-registered stop rule, Step-B
is explicitly NOT implemented in this file, and no bar at or after
`OOS_START` is ever read here (see `assert_no_holdout`, checked immediately
after load and again inside `r155_shared.step_a_gate`'s own TDA signal
construction).

This is now the ELEVENTH structurally distinct detector (after HMM/R-01,
BOCPD/R-82, Kalman LLT/R-83, CSD/R-85, transfer entropy/R-86, Hawkes/R-96,
POT-GPD/R-98, jump/QV/R-99, CUSUM/R-139, LPPLS/R-141) to fail the same
detection-lag gate, and the second (after CUSUM/R-139) to be swept across a
parameter grid rather than checked at one fixed configuration and still not
clear it. The consistent finding across all eleven: `kelly_regime_v4`'s own
fixed-window anchor heuristic reacts to real historical BTC regime breaks
about as fast as, or faster than, every formal regime/changepoint/
complexity estimator tried against it so far, at any parameterization
checked.

CONFIGURATIONS EVALUATED IN THIS FILE: 9 (the full pre-registered grid; no
Step-B, so no further configurations beyond the 9 Step-A cells).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r155_shared import (  # noqa: E402
    CONS_EMBED_DELAY,
    CONS_TRAIL_DAYS,
    CONS_Z_THRESH,
    NOVEL_DIM_GRID,
    NOVEL_WINDOW_GRID,
    OOS_START,
    plateau_ok,
    step_a_gate,
)

DATA_DIR = ROOT / "data"


def assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


def load_btc_bars() -> pd.DataFrame:
    """BTC spot 5-minute OHLCV, truncated strictly before OOS_START -- loaded
    exactly ONCE for the whole 9-cell sweep (the bars-loading step, and the
    v4 anchor-majority vote each cell's gate derives from `bars` alone, do
    not depend on window_days/embed_dim, so there is no reason to reread the
    dataset per cell; only each cell's own TDA embedding, which DOES depend
    on window_days/embed_dim, must be recomputed per cell). This file never
    reads a bar dated OOS_START or later, at any step."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def run_sweep(bars: pd.DataFrame) -> list[dict]:
    """Run `r155_shared.step_a_gate` on every cell of the pre-registered
    9-cell grid (3 window_days x 3 embed_dim), holding embed_delay/
    trail_days/z_thresh fixed at the conservative branch's own values.
    Returns the full list of per-cell result dicts, in grid order, EVERY
    cell included (this project has a hard rule against silently reporting
    only the best result).

    Note on the shared/cacheable computation named in the task: `bars` (the
    dataset load) is read once, above, and reused for all 9 calls below --
    that is the one genuinely expensive, cell-independent I/O step. Within
    `step_a_gate` itself (frozen in `r155_shared.py`, not editable here) the
    cheap `anchor_majority(bars)` vote is recomputed once per call; it does
    not depend on window_days/embed_dim either, but `step_a_gate`'s own
    signature has no hook to inject a precomputed majority, and its cost
    (a handful of rolling means over the 5-minute bar index) is negligible
    next to each cell's own O(window_days^2)-per-day MST/TDA embedding, so
    this is not a real performance concern in practice."""
    all_results = []
    n = 0
    n_total = len(NOVEL_WINDOW_GRID) * len(NOVEL_DIM_GRID)
    t0 = time.time()
    for w in NOVEL_WINDOW_GRID:
        for d in NOVEL_DIM_GRID:
            n += 1
            cell_t0 = time.time()
            r = step_a_gate(bars, window_days=w, embed_dim=d,
                             embed_delay=CONS_EMBED_DELAY, trail_days=CONS_TRAIL_DAYS,
                             z_thresh=CONS_Z_THRESH, verbose=False)
            all_results.append(r)
            print(f"  [{n:2d}/{n_total}] window_days={w:2d} embed_dim={d}  "
                  f"n_pass={r['n_pass']}/6  passed={r['passed']}  "
                  f"({time.time() - cell_t0:.1f}s)", file=sys.stderr)
    print(f"\nsweep elapsed: {time.time() - t0:.1f}s over {n} cells", file=sys.stderr)
    assert n == n_total == 9, f"expected 9 grid cells, evaluated {n}"
    return all_results


def report(all_results: list[dict]) -> dict:
    print("\n" + "=" * 92)
    print("R-155 NOVEL: full 9-cell TDA (window_days x embed_dim) sweep vs the "
          "six-episode detection-lag gate")
    print("=" * 92)
    # Sort by window_days then embed_dim -- clearest reading order, matches grid order.
    ordered = sorted(all_results, key=lambda r: (r["window_days"], r["embed_dim"]))
    print(f"{'#':>3}  {'window_days':>11}  {'embed_dim':>9}  {'n_pass/6':>9}  {'passed':>7}")
    for i, r in enumerate(ordered, start=1):
        print(f"{i:3d}  {r['window_days']:11d}  {r['embed_dim']:9d}  "
              f"{r['n_pass']:>6d}/6   {str(r['passed']):>7}")

    winners = [r for r in all_results if r["passed"]]
    print(f"\nCells with n_pass >= 4/6 (Step-A promotion bar): {len(winners)}")

    plateau_winners = []
    isolated_winners = []
    for w in winners:
        ok = plateau_ok(all_results, w)
        tag = "PLATEAU" if ok else "ISOLATED PEAK"
        print(f"  window_days={w['window_days']} embed_dim={w['embed_dim']} "
              f"n_pass={w['n_pass']}/6  plateau_ok={ok}  [{tag}]")
        (plateau_winners if ok else isolated_winners).append(w)

    # Best cell overall (whether or not it clears the promotion bar), for
    # reporting. Tie-break, stated explicitly: smallest window_days first,
    # then smallest embed_dim (most literal / fastest-reacting / cheapest).
    best = min(
        (r for r in all_results),
        key=lambda r: (-r["n_pass"], r["window_days"], r["embed_dim"]),
    )
    best_cells = [r for r in all_results if r["n_pass"] == best["n_pass"]]

    print("\n" + "-" * 92)
    print(f"Best cell (tie-break: smallest window_days, then smallest embed_dim): "
          f"window_days={best['window_days']} embed_dim={best['embed_dim']} "
          f"n_pass={best['n_pass']}/6" +
          (f"  ({len(best_cells)} cell(s) tied on n_pass={best['n_pass']}/6)"
           if len(best_cells) > 1 else ""))

    plateau_pass = None
    if not winners:
        print(f"No cell reaches the n_pass>=4/6 bar. Best n_pass in the grid: "
              f"{best['n_pass']}/6.")
        further_work = False
    else:
        if best["n_pass"] >= 4:
            plateau_pass = plateau_ok(all_results, best)
            print(f"Best cell clears n_pass>=4/6. plateau_ok(all_9_results, best)="
                  f"{plateau_pass}.")
        if not plateau_winners:
            print(f"{len(winners)} cell(s) reach n_pass>=4/6, but ALL fail "
                  f"`plateau_ok()` -- isolated one-cell peak(s), not a plateau. Per the "
                  f"pre-registered stop rule this counts as a Step-A failure "
                  f"(six-episode fitting artifact, not a property of TDA at a sensible "
                  f"parameterization).")
            further_work = False
        else:
            print(f"{len(plateau_winners)} cell(s) reach n_pass>=4/6 AND pass "
                  f"`plateau_ok()` -- a genuine plateau, not a fluke.")
            further_work = True

    print("\nPRE-REGISTERED STOP RULE: this branch clears Step-A only if at least one "
          "winning cell passes plateau_ok(). " +
          ("CLEARED -> Step-B would be implemented." if further_work else
           "NOT CLEARED -> STOP HERE. Step-B is NOT implemented in this file. "
           "No OOS/holdout bar is read anywhere in this file."))
    print(f"\nFINAL VERDICT: {'POSITIVE at Step-A (proceed to Step-B)' if further_work else 'NEGATIVE at Step-A'}")
    print("configurations evaluated in this file: 9")
    print(f"max timestamp read anywhere in this session: < {OOS_START} (enforced by "
          f"assert_no_holdout, checked on load and inside step_a_gate's own TDA "
          f"signal construction)")

    return dict(all_results=all_results, best=best, winners=winners,
                plateau_winners=plateau_winners, isolated_winners=isolated_winners,
                plateau_pass=plateau_pass, further_work=further_work, n_configs=9)


def main() -> dict:
    bars = load_btc_bars()
    all_results = run_sweep(bars)
    outcome = report(all_results)

    if outcome["further_work"]:
        # PRE-REGISTERED CONTINGENCY, NOT REACHED IN THIS RUN: per this
        # branch's own stop rule and `r155_shared`'s module docstring, Step-B
        # is implemented ONLY once a genuine plateau pass exists. This run's
        # grid produced none (see table above), so, per this project's
        # convention (r82_gate.py/r85/r86/r139: "STOP, no strategy built"
        # when Step-A fails), Step-B is deliberately left unimplemented here
        # rather than written speculatively against a condition that did not
        # trigger.
        raise RuntimeError(
            "further_work=True but Step-B is not implemented in this file "
            "(pre-registered: only build Step-B once a genuine plateau pass "
            "is observed, not speculatively). Re-check the sweep, then write "
            "Step-B against the actual winning cell before proceeding.")

    return outcome


if __name__ == "__main__":
    main()
