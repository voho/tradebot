#!/usr/bin/env python
"""R-156 NOVEL branch: sweep the causal Elliott-wave-invalidation
run-length detector's pre-registered `pct x require_fib_band` grid against
the R-82-identical six-episode detection-lag Step-A gate
(`experiments.r156_shared.step_a_gate`), exactly as pre-registered in
`experiments/r156_shared.py`'s module docstring.

See `experiments/r156_shared.py` for the full mechanism, citation trail,
not-a-duplicate-of list, and the PRE-REGISTERED stop rule this file
implements verbatim (frozen before any of today's numbers existed):

    the winning cell (highest n_pass, tie-break: prefer
    require_fib_band=True then prefer pct=0.05) only counts as a real pass
    if n_pass >= 4/6 AND `r156_shared.plateau_ok()` is True (at least one
    immediate pct-neighbour, holding require_fib_band fixed, also clears
    4/6). Otherwise: STOP, no Step-B, no holdout read, report NEGATIVE.

=====================================================================
RESULT (this file is a genuine, re-runnable computation -- the numbers
below are what running it produces; `python
experiments/r156_novel_wavecount_gate.py` reproduces them deterministically)
=====================================================================

All 6 cells of `NOVEL_PCT_GRID x NOVEL_FIB_GRID` were run at the frozen
n_draws=500. NOT ONE CELL reaches the n_pass >= 4/6 bar:

    pct=0.03 fib=True   n_pass=2/6
    pct=0.03 fib=False  n_pass=2/6
    pct=0.05 fib=True   n_pass=3/6   <- best cell overall
    pct=0.05 fib=False  n_pass=0/6
    pct=0.08 fib=True   n_pass=1/6
    pct=0.08 fib=False  n_pass=1/6

The winning cell by the pre-registered tie-break rule (highest n_pass;
prefer require_fib_band=True; then prefer pct=0.05) is pct=0.05,
require_fib_band=True at 3/6 -- one short of the promotion bar.
`plateau_ok()` is never invoked in earnest: the pre-registered stop rule
fires on the first clause ("winner n_pass < 4/6") before the plateau
question is reachable at all. The single episode this mechanism never
gets right at any grid cell is the 2020-03 COVID crash (lead is negative,
i.e. late, at every pct/fib combination) -- the sharpest, most
news-driven of the six shocks, exactly the failure mode this round's own
prior named before running anything.

VERDICT: NEGATIVE at Step-A. STOP. Per the pre-registered stop rule,
Step-B is explicitly NOT implemented in this file, and no bar at or after
`OOS_START` is ever read here (see `assert_no_holdout`, checked
immediately after load and again inside `r156_shared.step_a_gate`'s own
wave-signal construction).

This is now the ELEVENTH structurally distinct detector (after HMM/R-01,
BOCPD/R-82, Kalman LLT/R-83, CSD/R-85, transfer entropy/R-86, Hawkes/R-96,
POT-GPD/R-98, jump/QV/R-99, CUSUM/R-139, LPPLS/R-141) -- TWELFTH counting
TDA/R-155 -- to fail the same detection-lag gate. The Fibonacci-band
ablation (require_fib_band=True vs False) makes little difference at
pct=0.03 (2/6 either way) and actively hurts at pct=0.05 (3/6 -> 0/6
dropping the band) -- consistent with Batchelor & Ramyar's finding that
the specific Fibonacci ratio carries no information beyond whatever the
bare structural rule already captures, extended here from retracement
price levels to retracement-rule structure, on this project's own data.

CONFIGURATIONS EVALUATED IN THIS FILE: 6 (the full pre-registered grid; no
Step-B triggered, so no further configurations beyond the 6 Step-A cells).
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

from experiments.r156_shared import (  # noqa: E402
    NOVEL_FIB_GRID,
    NOVEL_PCT_GRID,
    OOS_START,
    assert_no_holdout,
    plateau_ok,
    step_a_gate,
)

DATA_DIR = ROOT / "data"


def load_btc_bars() -> pd.DataFrame:
    """BTC spot 5-minute OHLCV, truncated strictly before OOS_START -- loaded
    exactly ONCE for the whole 6-cell sweep. This file never reads a bar
    dated OOS_START or later, at any step."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def run_sweep(bars: pd.DataFrame) -> list[dict]:
    """Run `r156_shared.step_a_gate` on every cell of the pre-registered
    6-cell grid (3 pct x 2 require_fib_band), at the frozen n_draws=500
    default. Returns the full list of per-cell result dicts, in grid
    order, EVERY cell included (this project has a hard rule against
    silently reporting only the best result)."""
    all_results = []
    n = 0
    n_total = len(NOVEL_PCT_GRID) * len(NOVEL_FIB_GRID)
    t0 = time.time()
    for pct in NOVEL_PCT_GRID:
        for fib in NOVEL_FIB_GRID:
            n += 1
            cell_t0 = time.time()
            print(f"\n[{n}/{n_total}] pct={pct} require_fib_band={fib}", file=sys.stderr)
            r = step_a_gate(bars, pct=pct, require_fib_band=fib, verbose=True)
            all_results.append(r)
            print(f"  -> n_pass={r['n_pass']}/6  passed={r['passed']}  "
                  f"({time.time() - cell_t0:.2f}s)", file=sys.stderr)
    print(f"\nsweep elapsed: {time.time() - t0:.1f}s over {n} cells", file=sys.stderr)
    assert n == n_total == 6, f"expected 6 grid cells, evaluated {n}"
    return all_results


def report(all_results: list[dict]) -> dict:
    print("\n" + "=" * 92)
    print("R-156 NOVEL: full 6-cell wave-invalidation (pct x require_fib_band) sweep "
          "vs the six-episode detection-lag gate")
    print("=" * 92)
    ordered = sorted(all_results, key=lambda r: (r["pct"], not r["require_fib_band"]))
    print(f"{'#':>3}  {'pct':>5}  {'fib_band':>8}  {'n_pass/6':>9}  {'passed':>7}")
    for i, r in enumerate(ordered, start=1):
        print(f"{i:3d}  {r['pct']:5.2f}  {str(r['require_fib_band']):>8}  "
              f"{r['n_pass']:>6d}/6   {str(r['passed']):>7}")

    print("\nPer-episode detail, all 6 cells:")
    for r in ordered:
        print(f"\n  pct={r['pct']} require_fib_band={r['require_fib_band']} "
              f"n_pass={r['n_pass']}/6")
        for ep in r["results"]:
            if "null_median" in ep:
                print(f"    [{ep['label']}] lead={ep['lead']:+.2f}d "
                      f"null_median={ep['null_median']:+.2f}d PASS={ep['pass_b']}")
            else:
                print(f"    [{ep['label']}] no flip/detect in window -> PASS=False")

    winners = [r for r in all_results if r["passed"]]
    print(f"\nCells with n_pass >= 4/6 (Step-A promotion bar): {len(winners)}")

    # Winner selection per the pre-registered rule: highest n_pass;
    # tie-break prefer require_fib_band=True, then prefer pct=0.05
    # (the middle grid value).
    winner = max(
        all_results,
        key=lambda r: (r["n_pass"], r["require_fib_band"], -abs(r["pct"] - 0.05)),
    )
    print("\n" + "-" * 92)
    print(f"Winning cell (highest n_pass; tie-break: require_fib_band=True, "
          f"then pct=0.05): pct={winner['pct']} require_fib_band="
          f"{winner['require_fib_band']}  n_pass={winner['n_pass']}/6")

    gate_pass = False
    plateau = None
    if winner["n_pass"] >= 4:
        plateau = plateau_ok(all_results, winner)
        print(f"winner n_pass >= 4/6. plateau_ok(all_6_results, winner) = {plateau}")
        gate_pass = bool(plateau)
    else:
        print(f"winner n_pass ({winner['n_pass']}/6) < 4/6 -- gate fails outright, "
              f"plateau_ok() not invoked in earnest.")

    print("\nPRE-REGISTERED DECISION: PASS only if winner n_pass>=4/6 AND "
          "plateau_ok(). " +
          ("PASS -> Step-B contingency (see r156_shared.py docstring step 3) "
           "would be triggered." if gate_pass else
           "FAIL -> STOP HERE. Step-B is NOT implemented in this file. "
           "No OOS/holdout bar is read anywhere in this file."))
    print(f"\nFINAL VERDICT: {'POSITIVE at Step-A (proceed to Step-B proposal)' if gate_pass else 'NEGATIVE at Step-A'}")
    print("configurations evaluated in this file: 6")
    print(f"max timestamp read anywhere in this session: < {OOS_START} (enforced by "
          f"assert_no_holdout, checked on load and inside step_a_gate's own wave "
          f"signal construction)")

    return dict(all_results=all_results, winner=winner, winners=winners,
                plateau=plateau, gate_pass=gate_pass, n_configs=6)


def main() -> dict:
    bars = load_btc_bars()
    all_results = run_sweep(bars)
    outcome = report(all_results)

    if outcome["gate_pass"]:
        # PRE-REGISTERED CONTINGENCY, NOT REACHED IN THIS RUN (see below):
        # per this branch's own stop rule and `r156_shared`'s module
        # docstring step 3, if the gate passes we do NOT consume the
        # holdout here -- we only propose a Step-B pre-registration
        # (combined-signal weight, inner-validation numbers) for the
        # operator to freeze and run separately. That analysis, if
        # triggered, is written up in the final report to the operator,
        # not executed speculatively inside this file.
        raise RuntimeError(
            "gate_pass=True: per the pre-registered contingency this requires "
            "a Step-B PROPOSAL (not a holdout run) to be written up for the "
            "operator -- re-check the winning cell and write that proposal "
            "in the report rather than proceeding automatically.")

    return outcome


if __name__ == "__main__":
    main()
