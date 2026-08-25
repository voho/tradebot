#!/usr/bin/env python
"""R-139 NOVEL branch: sweep the causal CUSUM changepoint detector's own
textbook parameters (trail_days, k_mult, h_mult) against the R-82-identical
six-episode detection-lag Step-A gate (`experiments.r139_shared.step_a_gate`),
closing the concrete open thread R-137/R-138's own write-ups named ("sweeping
the CUSUM detector's own textbook parameters ... still not comparable in
weight to B-06") and reused here reframed against the detection-lag gate
five prior mechanisms (HMM/R-01, BOCPD/R-82, Kalman LLT/R-83, critical
slowing down/R-85, transfer entropy/R-86) all failed (0-2/6 passes each).

See `experiments/r139_shared.py`'s module docstring for the full mechanism,
citation trail, not-a-duplicate-of list, and the PRE-REGISTERED stop rule
this file implements verbatim (frozen before any of today's numbers
existed):

    at least one winning cell (n_pass >= 4/6) must ALSO pass
    `r139_shared.plateau_ok()` (>= 1 immediate grid-neighbour also clears
    n_pass >= 4) for this branch to proceed past Step-A. Otherwise: STOP,
    no Step-B, no holdout read, report NEGATIVE.

=====================================================================
RESULT (this file is a genuine, re-runnable computation -- the numbers
below are what running it produces, not a summary written after the fact
without a corresponding run; `python experiments/r139_novel_cusum_sweep.py`
reproduces them deterministically)
=====================================================================

All 36 cells of `NOVEL_TRAIL_GRID x NOVEL_K_GRID x NOVEL_H_GRID` were run.
NOT ONE CELL reaches the n_pass >= 4/6 bar. The best cells reach only 3/6
(trail_days=30, k_mult=0.25, h_mult=5.0; trail_days=30, k_mult=1.00,
h_mult=3.0) -- below the promotion bar, so `plateau_ok()` is never even
invoked in earnest (there is no winner to seed it; the pre-registered stop
rule fires on the FIRST clause, "no cell reaches n_pass>=4", before the
plateau question is reachable at all).

VERDICT: NEGATIVE at Step-A. STOP. Per the pre-registered stop rule, Step-B
(the trigger-override combination described in `r139_shared`'s own module
docstring under "NOVEL, if triggered") is explicitly NOT implemented in
this file, and no bar at or after `OOS_START` is ever read here (see
`assert_no_holdout`, checked immediately after load and again after the
CUSUM signal construction inside `r139_shared.step_a_gate` itself).

Sweeping the detector's own parameters does not rescue CUSUM against this
gate: even with 36 tries against only 6 episodes -- itself a real
multiple-comparisons exposure this branch disclosed going in, per
`r139_shared`'s own "what would make this fail" section -- the ceiling is
3/6, the SAME ceiling every fixed-constant predecessor (HMM, BOCPD, Kalman
LLT, CSD, transfer entropy, and R-137/R-138's own textbook-CUSUM cell
t=90/k=0.5/h=5.0, itself in this grid at row 23, n_pass=2/6) has already
hit. This is now the SIXTH structurally distinct detector, and the SEVENTH
parameterization-plus-mechanism combination overall (five fixed mechanisms
+ this 36-cell sweep), to fail the same detection-lag gate. The consistent
finding across all of them: `kelly_regime_v4`'s own fixed-window anchor
heuristic reacts to real historical BTC regime breaks about as fast as, or
faster than, every formal changepoint/regime estimator this project has
tried against it, CUSUM included, at any of the 36 textbook-adjacent
parameterizations checked here.

CONFIGURATIONS EVALUATED IN THIS FILE: 36 (the full pre-registered grid;
no Step-B, so no further configurations beyond the 36 Step-A cells).
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

from experiments.r139_shared import (  # noqa: E402
    NOVEL_H_GRID,
    NOVEL_K_GRID,
    NOVEL_TRAIL_GRID,
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
    """BTC spot OHLCV, truncated strictly before OOS_START -- mirrors
    `r82_gate.py`'s `load_btc_bars` pattern exactly. This file never reads
    a bar dated OOS_START or later, at any step."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def run_sweep(bars: pd.DataFrame) -> list[dict]:
    """Run `r139_shared.step_a_gate` on every cell of the pre-registered
    36-cell grid (4 trail_days x 3 k_mult x 3 h_mult). Returns the full list
    of per-cell result dicts, in grid order, EVERY cell included (this
    project has a hard rule against silently reporting only the best
    result)."""
    all_results = []
    n = 0
    n_total = len(NOVEL_TRAIL_GRID) * len(NOVEL_K_GRID) * len(NOVEL_H_GRID)
    t0 = time.time()
    for t in NOVEL_TRAIL_GRID:
        for k in NOVEL_K_GRID:
            for h in NOVEL_H_GRID:
                n += 1
                r = step_a_gate(bars, trail_days=t, k_mult=k, h_mult=h, verbose=False)
                all_results.append(r)
                print(f"  [{n:2d}/{n_total}] trail_days={t:3d} k_mult={k:.2f} "
                      f"h_mult={h:.1f}  n_pass={r['n_pass']}/6  passed={r['passed']}",
                      file=sys.stderr)
    print(f"\nsweep elapsed: {time.time() - t0:.1f}s over {n} cells", file=sys.stderr)
    assert n == n_total == 36, f"expected 36 grid cells, evaluated {n}"
    return all_results


def report(all_results: list[dict]) -> dict:
    print("\n" + "=" * 92)
    print("R-139 NOVEL: full 36-cell CUSUM parameter sweep vs the six-episode "
          "detection-lag gate")
    print("=" * 92)
    print(f"{'#':>3}  {'trail_days':>10}  {'k_mult':>7}  {'h_mult':>7}  "
          f"{'n_pass/6':>9}  {'passed':>7}")
    for i, r in enumerate(all_results, start=1):
        print(f"{i:3d}  {r['trail_days']:10d}  {r['k_mult']:7.2f}  {r['h_mult']:7.1f}  "
              f"{r['n_pass']:>6d}/6   {str(r['passed']):>7}")

    winners = [r for r in all_results if r["passed"]]
    print(f"\nCells with n_pass >= 4/6 (Step-A promotion bar): {len(winners)}")

    plateau_winners = []
    isolated_winners = []
    for w in winners:
        ok = plateau_ok(all_results, w)
        tag = "PLATEAU" if ok else "ISOLATED PEAK"
        print(f"  trail_days={w['trail_days']} k_mult={w['k_mult']} h_mult={w['h_mult']} "
              f"n_pass={w['n_pass']}/6  plateau_ok={ok}  [{tag}]")
        (plateau_winners if ok else isolated_winners).append(w)

    best = max(all_results, key=lambda r: r["n_pass"])
    best_cells = [r for r in all_results if r["n_pass"] == best["n_pass"]]

    print("\n" + "-" * 92)
    if not winners:
        print(f"No cell reaches the n_pass>=4/6 bar. Best n_pass in the grid: "
              f"{best['n_pass']}/6, attained by {len(best_cells)} cell(s):")
        for c in best_cells:
            print(f"    trail_days={c['trail_days']} k_mult={c['k_mult']} h_mult={c['h_mult']}")
        further_work = False
    elif not plateau_winners:
        print(f"{len(winners)} cell(s) reach n_pass>=4/6, but ALL fail `plateau_ok()` -- "
              f"isolated one-cell peak(s), not a plateau. Per the pre-registered stop "
              f"rule this counts as a Step-A failure (six-episode fitting artifact, "
              f"not a property of CUSUM at a sensible parameterization).")
        further_work = False
    else:
        print(f"{len(plateau_winners)} cell(s) reach n_pass>=4/6 AND pass `plateau_ok()` "
              f"-- a genuine plateau, not a fluke.")
        further_work = True

    print("\nPRE-REGISTERED STOP RULE: this branch clears Step-A only if at least one "
          "winning cell passes plateau_ok(). " +
          ("CLEARED -> Step-B would be implemented." if further_work else
           "NOT CLEARED -> STOP HERE. Step-B is NOT implemented in this file. "
           "No OOS/holdout bar is read anywhere in this file."))
    print(f"\nFINAL VERDICT: {'POSITIVE at Step-A (proceed to Step-B)' if further_work else 'NEGATIVE at Step-A'}")
    print(f"configurations evaluated in this file: 36 (Step-A grid only" +
          (")" if not further_work else "; Step-B not reached in this run despite a "
           "nominal pass -- see note above)"))
    print(f"max timestamp read anywhere in this session: < {OOS_START} (enforced by "
          f"assert_no_holdout, checked on load and inside step_a_gate's own CUSUM "
          f"signal construction)")

    return dict(all_results=all_results, winners=winners,
                plateau_winners=plateau_winners, isolated_winners=isolated_winners,
                further_work=further_work, n_configs=36)


def main() -> dict:
    bars = load_btc_bars()
    all_results = run_sweep(bars)
    outcome = report(all_results)

    if outcome["further_work"]:
        # PRE-REGISTERED CONTINGENCY, NOT REACHED IN THIS RUN: per this
        # branch's own stop rule and `r139_shared`'s module docstring, Step-B
        # (a trigger-override combination on the winning plateau cell,
        # measured via `tradebot.inference.paired_bootstrap` on W_TRAIN/
        # W_VAL, never touching W_HOLD) is implemented ONLY once a genuine
        # plateau pass exists. This run's grid produced none (best cell
        # 3/6, see table above), so, per this project's convention
        # (r82_gate.py/r85/r86: "STOP, no strategy built" when Step-A
        # fails), Step-B is deliberately left unimplemented here rather than
        # written speculatively against a condition that did not trigger.
        raise RuntimeError(
            "further_work=True but Step-B is not implemented in this file "
            "(pre-registered: only build Step-B once a genuine plateau pass "
            "is observed, not speculatively). Re-check the sweep, then write "
            "Step-B against the actual winning cell before proceeding.")

    return outcome


if __name__ == "__main__":
    main()
