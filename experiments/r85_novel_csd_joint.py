#!/usr/bin/env python
"""R-85 NOVEL branch: critical slowing down (CSD), joint two-indicator
(variance AND lag-1 autocorrelation) confirmation, as an earlier-warning
mechanism for `kelly_regime_v4`'s detection lag. Step-A detection-lag gate,
run BEFORE any strategy/confirming-vote code -- identical methodology to
R-82 (BOCPD, 2/6 FAIL) and R-83 (causal Kalman LLT, 1/6 FAIL), for direct
comparability. See `experiments/r85_shared.py`'s module docstring for the
full citation trail (Scheffer et al. 2009; Wen et al. 2020; Kuehn 2011;
Dakos et al. 2012; arXiv:2607.27070) and not-a-duplicate-of list.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM. One sentence: critical slowing down -- a system approaching
   a critical transition recovers more slowly from small perturbations,
   which shows up as RISING variance and RISING lag-1 autocorrelation of
   its own return fluctuations BEFORE the transition -- should give
   earlier warning of the six dated historical BTC regime breaks in
   `r85_shared.STRESS_EPISODES` than v4's own fixed 20/40/80-day anchor
   heuristic, IF both indicators agreeing (Dakos et al. 2012's joint-
   confirmation recommendation) is a genuine event-locked property rather
   than generic autocorrelation/persistence structure in two slow-moving
   statistics.

2. DETECTION-LAG DEFINITION. For each episode, within a +/-60-day search
   window around its onset (`r85_shared.episode_window`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`r85_shared.nearest_transition`,
     `direction="down"`, identical rule R-82/R-83 used).
   - This branch's reaction: the nearest bar where BOTH `csd_var_z` AND
     `csd_autocorr_z` are simultaneously >= `Z_THRESH_JOINT` (1.5),
     closest to the onset (`r85_shared.nearest_joint_csd_alarm`).
   - LEAD = (v4_flip_time - detect_time) in days. Positive = the joint CSD
     alarm fired before v4's own gate reacted.

3. NULL. `r85_shared.block_bootstrap_shifts` draws ONE circular block-shift
   per replicate and applies that SAME shift to BOTH `csd_var_z` and
   `csd_autocorr_z` together within the local episode window (not two
   independent shifts -- a joint indicator's null must preserve whatever
   real co-movement exists between the two series), then recomputes
   "nearest joint alarm to the real, unshifted v4 flip time" against each
   shifted pair. block_days=5, n_draws=500, seed=8502 -- fixed before
   running, not changed after seeing any result.

4. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed): an episode counts as a PASS if BOTH (a) LEAD >= 0 (the joint
   CSD alarm fired at or before v4's own nearest reaction), AND (b) the
   true LEAD is >= the null distribution's median. Identical weak-bar
   logic to R-82/R-83: this gate only asks whether the joint CSD alarm is
   not worse than chance at matching v4's own reaction speed, not that it
   beats v4 outright. PROCEED TO STEP B (build the confirming-vote SIZE-
   axis strategy) only if >= 4 of the 6 episodes PASS. If fewer than 4
   pass: STOP, report this file's result as the whole branch's product, do
   not build any strategy code. The bar is not relaxed after seeing the
   numbers.

5. WHAT WOULD MAKE THIS GATE FAIL, named now (this is the round's own
   pre-registered expectation, per `r85_shared.py`'s docstring and
   arXiv:2607.27070's finding that CSD is silent in sudden, news-driven
   shocks): the joint CSD alarm detects the two slow-building 2018
   episodes early but lags on the four sudden 2020-2022 shocks (COVID
   crash, 2021 top, Terra/Luna, FTX) -- the same failure mode BOCPD (2/6)
   and Kalman LLT (1/6) hit. A second, distinct failure mode: the
   AND-gate's low joint base rate (~0.40% of bars, disclosed in
   `r85_shared.py`) means it may simply not fire inside many +/-60-day
   windows at all ("no joint alarm found" -> automatic FAIL for that
   episode by construction).

CONFIGURATIONS EVALUATED IN THIS FILE: 0 (a fixed, non-swept measurement
gate using the single pre-fixed `Z_THRESH_JOINT=1.5` throughout -- no
threshold search here; a Step-B sweep, if the gate passes, is
pre-registered separately below and run only if reached).
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
    Z_THRESH_JOINT,
    anchor_majority,
    block_bootstrap_shifts,
    csd_daily_causal_signals,
    episode_window,
    nearest_joint_csd_alarm,
    nearest_transition,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 8502


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


def null_leads(var_z: pd.Series, ac_z: pd.Series, window: pd.DatetimeIndex,
                onset: pd.Timestamp, flip_time: pd.Timestamp,
                z_thresh: float = Z_THRESH_JOINT, n_draws: int = N_DRAWS,
                block_days: int = BLOCK_DAYS, seed: int = NULL_SEED) -> np.ndarray:
    """Same circular block-shift applied jointly to both local indicator
    series per draw (not two independent shifts), preserving whatever
    real co-movement exists between them, then recomputes the nearest
    joint-alarm-to-real-flip-time lead against each shifted pair."""
    local_a = var_z.reindex(window).to_numpy()
    local_b = ac_z.reindex(window).to_numpy()
    n_bars = len(local_a)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted_a = local_a[shift]
        shifted_b = local_b[shift]
        high = (shifted_a >= z_thresh) & (shifted_b >= z_thresh)
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


def run_truncation_probes(bars: pd.DataFrame) -> None:
    """Independent re-verification (not just trust of r85_shared.py's own
    claim) that the CSD signals are strictly causal: row t's value must
    not change if bars after t are dropped."""
    check_at = len(bars) - 40_000
    assert check_at > 400_000, "not enough bars for a meaningful truncation probe"

    def build_var_z(df: pd.DataFrame) -> np.ndarray:
        return csd_daily_causal_signals(df)["csd_var_z"].to_numpy()

    def build_ac_z(df: pd.DataFrame) -> np.ndarray:
        return csd_daily_causal_signals(df)["csd_autocorr_z"].to_numpy()

    ok_var = truncation_causality_probe(build_var_z, bars, check_at, shorter_by=20_000)
    ok_ac = truncation_causality_probe(build_ac_z, bars, check_at, shorter_by=20_000)
    print(f"\nCausal truncation probe (independent re-check), row {check_at}:")
    print(f"    csd_var_z      causal: {ok_var}")
    print(f"    csd_autocorr_z causal: {ok_ac}")
    assert ok_var and ok_ac, "CSD signal(s) FAILED the independent causal truncation probe"


def gate() -> dict:
    print("=" * 78)
    print("R-85 NOVEL BRANCH: joint CSD (variance AND autocorr) vs v4 anchor -- "
          "STEP A detection-lag gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    csd = csd_daily_causal_signals(bars)
    assert_no_holdout(csd)
    var_z = csd["csd_var_z"]
    ac_z = csd["csd_autocorr_z"]

    run_truncation_probes(bars)

    print(f"\nz_thresh_joint={Z_THRESH_JOINT}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_joint_csd_alarm(var_z, ac_z, window, onset, Z_THRESH_JOINT)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no joint CSD alarm'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=flip_time,
                                 detect=detect_time, pass_b=False, lead=float("nan"),
                                 null_median=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(var_z, ac_z, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    joint CSD nearest alarm (var_z>={Z_THRESH_JOINT} AND autocorr_z>={Z_THRESH_JOINT}): {detect_time}")
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
    print(f"GATE VERDICT: {'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session (so far): "
          f"{max(bars.index.max(), csd.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed, bars_max_ts=bars.index.max())


if __name__ == "__main__":
    gate()
