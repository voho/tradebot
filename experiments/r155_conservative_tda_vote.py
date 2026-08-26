#!/usr/bin/env python
"""R-155 CONSERVATIVE branch: fixed-configuration causal H0 persistent-
homology (Takens-embedded MST total persistence) topological-instability
detector, run against the SAME six-episode Step-A detection-lag gate that
sank HMM (R-01), BOCPD (R-82), Kalman LLT (R-83), critical slowing down
(R-85), transfer entropy (R-86), Hawkes (R-96), POT/GPD (R-98),
bipower-variation jump/QV (R-99), CUSUM (R-139) and LPPLS (R-141) -- see
`experiments/r155_shared.py`'s module docstring for the full
not-a-duplicate-of trail, the TDA construction (causal Takens embedding ->
MST-equivalent H0 total persistence -> trailing z-score alarm -> run
length), and the pre-registered stop rule this file follows.

This branch does NOT sweep the TDA construction's own parameters -- that
sweep is the NOVEL branch's whole content (`NOVEL_WINDOW_GRID` x
`NOVEL_DIM_GRID` in `r155_shared.py`). This branch runs the ONE
pre-registered conservative configuration: `window_days=20` (v4's fastest
anchor horizon), `embed_dim=3`, `embed_delay=1`, `trail_days=90`,
`z_thresh=2.0` -- all fixed a priori per `r155_shared.py`'s docstring, not
tuned on this data.

PRE-REGISTERED STOP RULE (frozen in `r155_shared.py`'s docstring before any
of this file's numbers existed): if fewer than 4/6 episodes pass Step-A,
this branch STOPS HERE -- no Step-B implementation (no vote combination),
no holdout read, reported as NEGATIVE at Step-A. The bar is not relaxed
after seeing the numbers.

CONFIGURATIONS EVALUATED IN THIS FILE: 1 (the fixed conservative TDA
configuration; no sweep in this branch by pre-registration).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r155_shared import (  # noqa: E402
    CONS_EMBED_DELAY,
    CONS_EMBED_DIM,
    CONS_TRAIL_DAYS,
    CONS_WINDOW_DAYS,
    CONS_Z_THRESH,
    OOS_START,
    STRESS_EPISODES,
    step_a_gate,
)

DATA_DIR = ROOT / "data"

# Pre-registered stop-rule bar (identical to the shared gate's own bar and
# to every predecessor round's bar on this gate): >= 4/6 episodes must
# pass to proceed past Step A.
STOP_RULE_MIN_PASS = 4


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
    print("R-155 CONSERVATIVE branch: fixed-config causal TDA (H0 persistence) vs v4 anchor")
    print("STEP A detection-lag gate (identical machinery to R-01/R-82/R-83/R-85/R-86/R-139)")
    print("=" * 78)

    bars = load_btc_bars()
    assert_no_holdout(bars)

    print(f"\nTDA construction (fixed, no sweep in this branch): "
          f"window_days={CONS_WINDOW_DAYS}  embed_dim={CONS_EMBED_DIM}  "
          f"embed_delay={CONS_EMBED_DELAY}  trail_days={CONS_TRAIL_DAYS}  "
          f"z_thresh={CONS_Z_THRESH}")
    print(f"episodes: {len(STRESS_EPISODES)}\n")

    gate = step_a_gate(bars, window_days=CONS_WINDOW_DAYS, embed_dim=CONS_EMBED_DIM,
                        embed_delay=CONS_EMBED_DELAY, trail_days=CONS_TRAIL_DAYS,
                        z_thresh=CONS_Z_THRESH, verbose=True)
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
    if n_pass < STOP_RULE_MIN_PASS:
        print("FINAL VERDICT (pre-registered stop rule, r155_shared.py docstring):")
        print(f"  n_pass={n_pass}/6 < {STOP_RULE_MIN_PASS} -> STOP AT STEP A.")
        print("  NEGATIVE. No Step-B vote-combination logic implemented. No OOS/holdout data touched.")
        print("  This CONSERVATIVE branch's causal H0-persistence (TDA) detector, run at its fixed")
        print("  pre-registered configuration, does not detect the six dated historical BTC regime")
        print("  transitions with lead time at or better than v4's own anchor-crossing reaction, at a")
        print("  rate clearing this project's pre-registered >=4/6 bar -- the same failure mode every")
        print("  one of the ten prior structurally distinct detectors (HMM, BOCPD, Kalman LLT,")
        print("  critical slowing down, transfer entropy, Hawkes, POT/GPD, jump/QV, CUSUM, LPPLS) hit")
        print("  against this identical gate.")
    else:
        print("FINAL VERDICT: n_pass >= 4 -- Step-A gate PASSES. Proceeding to Step B")
        print("  is NOT implemented in this file -- per r155_shared.py's own stated prior this is")
        print("  considered unlikely; a genuine pass here would require building Step-B separately,")
        print("  following the number rather than the prior.")

    print(f"\nconfigurations evaluated in this file: 1 (fixed conservative TDA configuration,")
    print(f"no sweep -- this branch's whole pre-registered content at Step A)")
    print(f"max timestamp read anywhere in this session: {bars.index.max()}  (< {OOS_START})")

    return dict(gate=gate, n_pass=n_pass, passed=passed)


if __name__ == "__main__":
    main()
