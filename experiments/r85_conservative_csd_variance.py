#!/usr/bin/env python
"""R-85 CONSERVATIVE branch: Step-A detection-lag gate for critical slowing
down (CSD), single-indicator (rolling-variance trend z-score only), run
BEFORE any strategy/confirming-vote code -- identical "operator measurement"
convention as R-82/R-83's own gate files, and the same pre-registered
Step-A gate methodology, for direct comparability.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM. See `r85_shared.py`'s module docstring for the full citation
   trail (Scheffer et al. 2009; Wen et al. 2020; Dakos et al. 2012; the
   skeptical arXiv:2607.27070 framing) and not-a-duplicate-of list. One
   sentence: a system approaching a critical transition recovers more
   slowly from small perturbations, which shows up as a RISING variance of
   its own return fluctuations before the transition -- this branch tests
   whether that rise, converted to a causal trend z-score
   (`csd_var_z`), crosses a fixed threshold with LESS lag than v4's own
   fixed 20/40/80-day anchor-crossing heuristic, on the same six dated
   historical BTC regime transitions R-82/R-83 used.

2. DETECTION-LAG DEFINITION. For each episode, within a +/-60-day search
   window around its onset (`r85_shared.episode_window`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`r85_shared.nearest_transition`,
     `direction="down"` -- identical rule R-82/R-83 used).
   - CSD's reaction: the nearest bar where `csd_var_z` crosses UP through
     `Z_THRESH=2.0` (`r85_shared.nearest_csd_alarm`), closest to the onset.
   - LEAD = (v4_flip_time - detect_time) in days. Positive = CSD alarmed
     before v4's own gate reacted.

3. NULL. `r85_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) `csd_var_z` series (block_days=5, n_draws=500,
   seed=8501, fixed before running) and recomputes "nearest CSD alarm to
   the real, unshifted v4 flip time" against each shifted copy -- adapted
   from R-82's `null_leads` for a z-score-threshold-crossing detector
   instead of a run-length-threshold one; logic is otherwise identical.

4. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82/R-83): an episode counts as a PASS if
   BOTH (a) LEAD >= 0, AND (b) the true LEAD is >= the null distribution's
   median. PROCEED TO STEP B (build the confirming-vote strategy) only if
   >= 4 of the 6 episodes PASS. If fewer than 4 pass: STOP, report this
   file's result as the whole branch's product, write it up as NEGATIVE,
   do not build any strategy/confirming-vote code. The bar is not relaxed,
   narrowed, or otherwise adjusted after seeing the numbers.

5. WHAT WOULD MAKE THIS GATE FAIL, named now (and named in `r85_shared.py`
   as this round's own pre-registered EXPECTATION, not a hoped-for
   result): CSD detects the slow, endogenous 2018 build-ups early (rising
   variance has time to accumulate) but lags the sudden, exogenous shocks
   (COVID crash, 2021 top, Terra/Luna, FTX) because variance only rises
   *after* the shock's own volatility burst has already happened -- the
   same failure pattern BOCPD (R-82, 2/6) and Kalman LLT (R-83, 1/6) hit,
   and the one arXiv:2607.27070 itself reports (CSD silent in 2 of 7
   sudden liquidation cascades).

CONFIGURATIONS EVALUATED IN THIS FILE: 0 (a fixed, non-swept measurement
gate, using `r85_shared.Z_THRESH=2.0` throughout -- no threshold search
here; that search, if the gate passes, belongs to Step B and is
pre-registered separately there).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r85_shared import (  # noqa: E402
    OOS_START,
    STRESS_EPISODES,
    Z_THRESH,
    anchor_majority,
    block_bootstrap_shifts,
    csd_daily_causal_signals,
    episode_window,
    nearest_csd_alarm,
    nearest_transition,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 8501


def assert_no_holdout(df: pd.DataFrame) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def null_leads(csd_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r82_gate.py's `null_leads`: same circular block-shift
    construction, but the detector is a threshold-crossing z-score
    (`high = vals >= z_thresh`, up-cross) rather than a run-length
    threshold (`short = vals <= k_short`, down-cross)."""
    local = csd_z.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        high = shifted >= z_thresh
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = high[1:] & ~high[:-1]
        cross[0] = bool(high[0]) if n_bars else False
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        detect_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - detect_time).total_seconds() / 86400.0
    return leads


def gate() -> dict:
    print("=" * 78)
    print("R-85 CONSERVATIVE: CSD (variance) vs v4 anchor -- STEP A detection-lag gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    csd = csd_daily_causal_signals(bars)
    assert_no_holdout(csd)
    var_z = csd["csd_var_z"]

    print(f"\nz_thresh={Z_THRESH}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, onset=onset_str, pass_b=False, lead=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_csd_alarm(var_z, window, onset, Z_THRESH)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no CSD alarm'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str,
                                 flip=flip_time, detect=detect_time,
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(var_z, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    CSD(var) nearest alarm (z>={Z_THRESH}): {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             detect=detect_time, lead=lead, null_median=null_median,
                             pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 4

    print("\n" + "=" * 78)
    for r in results:
        print(f"  {r['label']:42s} lead={r.get('lead', float('nan')):+.2f}d  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/6")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B (build confirming-vote strategy)' if passed else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session: "
          f"{max(bars.index.max(), csd.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed, bars=bars, csd=csd)


def causality_probe() -> bool:
    """Independent re-verification of r85_shared.py's own causal truncation
    claim for `csd_daily_causal_signals`, per this project's rule that
    every round re-runs the probe itself rather than trusting a prior
    claim. Probes at a point well inside the 2018 episode window so the
    check exercises real, non-degenerate signal values."""
    bars = load_btc_bars()
    check_at = bars.index.get_indexer([pd.Timestamp("2018-06-01", tz="UTC")], method="nearest")[0]

    def build(df: pd.DataFrame) -> np.ndarray:
        return csd_daily_causal_signals(df)["csd_var_z"].to_numpy()

    ok = truncation_causality_probe(build, bars, check_at, shorter_by=40_000)
    print(f"\ncausal truncation probe (csd_var_z, check_at index {check_at} "
          f"~ {bars.index[check_at]}): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    causality_probe()
    gate()
