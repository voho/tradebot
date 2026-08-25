#!/usr/bin/env python
"""R-139 CONSERVATIVE branch: fixed textbook-constant causal CUSUM
changepoint detector, run against the SAME six-episode Step-A
detection-lag gate that sank HMM (R-01), BOCPD (R-82), Kalman LLT (R-83),
critical slowing down (R-85) and transfer entropy (R-86) -- see
`experiments/r139_shared.py`'s module docstring for the full
not-a-duplicate-of trail, the pre-registered Step-B specification for
this branch (a `confirming_vote_frac` combination swept over
weight in {0.5, 1.0, 1.5}), and the pre-registered stop rule this file
follows.

This branch does NOT sweep CUSUM's own parameters -- that sweep is the
NOVEL branch's whole content (`NOVEL_TRAIL_GRID`/`NOVEL_K_GRID`/
`NOVEL_H_GRID` in `r139_shared.py`). This branch runs the exact textbook
constants R-137/R-138 already used verbatim (`CUSUM_TRAIL_DAYS=90,
CUSUM_K_MULT=0.5, CUSUM_H_MULT=5.0`, imported transitively from
`experiments/r138_shared.py` via `r139_shared.py`), i.e. ONE configuration.

PRE-REGISTERED STOP RULE (frozen in `r139_shared.py`'s docstring before
any of today's numbers existed): if fewer than 4/6 episodes pass Step-A,
this branch STOPS HERE -- no Step-B implementation, no holdout read,
reported as NEGATIVE at Step-A.

=====================================================================
RESULT (filled in by running this file; see printed report below for the
authoritative numbers)
=====================================================================

Independently reproducing the operator's pre-dispatch smoke test
(n_pass=2, passed=False -- 2018 bear onset and 2018 capitulation passed;
COVID, Terra/Luna and FTX did not; one episode's window/detection was
reported as possibly empty): this file's own run below is the
independent check. If the number reproduces, this branch's own verdict
follows the pre-registered stop rule -- STOP at Step-A, negative, no
Step-B, matching the base rate every one of the five prior mechanisms hit
against this identical gate (0-2/6 each). If the number does NOT
reproduce, that is a discrepancy to investigate and report plainly, not
to paper over -- see the printed report's own DISCREPANCY section.

CONFIGURATIONS EVALUATED IN THIS FILE: 1 (the fixed textbook CUSUM
constants; no sweep in this branch by pre-registration).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r139_shared import (  # noqa: E402
    CUSUM_H_MULT,
    CUSUM_K_MULT,
    CUSUM_TRAIL_DAYS,
    OOS_START,
    STRESS_EPISODES,
    step_a_gate,
)

DATA_DIR = ROOT / "data"

# Pre-registered stop-rule bar (identical to the shared gate's own bar and
# to every predecessor round's bar on this gate): >= 4/6 episodes must
# pass to proceed past Step A.
STOP_RULE_MIN_PASS = 4

# The operator's pre-dispatch smoke test, named here BEFORE this file's own
# run for an explicit, printed independent-verification comparison.
OPERATOR_SMOKE_TEST_N_PASS = 2
OPERATOR_SMOKE_TEST_PASSED = False


def assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def main() -> dict:
    print("=" * 78)
    print("R-139 CONSERVATIVE branch: fixed-constant causal CUSUM vs v4 anchor")
    print("STEP A detection-lag gate (identical machinery to R-01/R-82/R-83/R-85/R-86)")
    print("=" * 78)

    bars = load_btc_bars()
    assert_no_holdout(bars)

    print(f"\nCUSUM constants (fixed textbook, no sweep in this branch): "
          f"trail_days={CUSUM_TRAIL_DAYS}  k_mult={CUSUM_K_MULT}  h_mult={CUSUM_H_MULT}")
    print(f"episodes: {len(STRESS_EPISODES)}\n")

    gate = step_a_gate(bars, trail_days=CUSUM_TRAIL_DAYS, k_mult=CUSUM_K_MULT,
                        h_mult=CUSUM_H_MULT, verbose=True)
    assert_no_holdout(bars)  # re-assert after gate computation touched no new data

    n_pass = gate["n_pass"]
    passed = gate["passed"]

    print("\n" + "=" * 78)
    print("PER-EPISODE RESULTS")
    print("=" * 78)
    for r, (label, onset_str) in zip(gate["results"], STRESS_EPISODES):
        lead = r.get("lead", float("nan"))
        null_median = r.get("null_median", float("nan"))
        print(f"  {label:42s} onset={onset_str}  lead={lead:+.2f}d  "
              f"null_median={null_median:+.2f}d  PASS={r['pass_b']}")

    print(f"\nEpisodes passing: {n_pass}/6")
    print(f"STEP-A GATE VERDICT: "
          f"{'PASS (>= 4/6) -> would proceed to Step B' if passed else 'FAIL (< 4/6) -> STOP at Step A'}")

    print("\n" + "=" * 78)
    print("INDEPENDENT VERIFICATION OF OPERATOR'S PRE-DISPATCH SMOKE TEST")
    print("=" * 78)
    print(f"  operator reported:      n_pass={OPERATOR_SMOKE_TEST_N_PASS}  "
          f"passed={OPERATOR_SMOKE_TEST_PASSED}")
    print(f"  independently computed: n_pass={n_pass}  passed={passed}")
    if n_pass == OPERATOR_SMOKE_TEST_N_PASS and passed == OPERATOR_SMOKE_TEST_PASSED:
        print("  RESULT: REPRODUCED. Independent run matches the operator's smoke test exactly.")
    else:
        print("  RESULT: DISCREPANCY. Independent run does NOT match the operator's smoke test.")
        print("  This indicates a bug (in r139_shared.py, or in how the smoke test was described)")
        print("  and must be investigated before trusting either number.")

    print("\n" + "=" * 78)
    if n_pass < STOP_RULE_MIN_PASS:
        print("FINAL VERDICT (pre-registered stop rule, r139_shared.py docstring):")
        print(f"  n_pass={n_pass}/6 < {STOP_RULE_MIN_PASS} -> STOP AT STEP A.")
        print("  NEGATIVE. No Step-B combination logic implemented. No OOS/holdout data touched.")
        print("  This CONSERVATIVE branch's causal CUSUM detector, run at its fixed textbook")
        print("  constants, does not detect the six dated historical BTC regime transitions")
        print("  with lead time at or better than v4's own anchor-crossing reaction, at a rate")
        print("  clearing this project's pre-registered >=4/6 bar -- the same failure mode")
        print("  every one of the five prior structurally distinct detectors (HMM, BOCPD,")
        print("  Kalman LLT, critical slowing down, transfer entropy) hit against this")
        print("  identical gate.")
    else:
        print("FINAL VERDICT: n_pass >= 4 -- Step-A gate PASSES. Proceeding to Step B")
        print("  (confirming_vote_frac combination, swept over weight in {0.5, 1.0, 1.5}),")
        print("  per r139_shared.py's pre-registered specification. See below.")

    print(f"\nconfigurations evaluated in this file: 1 (fixed textbook CUSUM constants,")
    print(f"no sweep -- this branch's whole pre-registered content at Step A)")
    print(f"max timestamp read anywhere in this session: {bars.index.max()}  (< {OOS_START})")

    return dict(gate=gate, n_pass=n_pass, passed=passed)


if __name__ == "__main__":
    main()
