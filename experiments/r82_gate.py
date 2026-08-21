#!/usr/bin/env python
"""R-82 operator measurement: the Step-A detection-lag gate, run BEFORE
either branch is dispatched -- same "operator measurement" convention as
R-68/R-78's own pre-measurement sub-sections in docs/LEDGER.md, applied
here because both branches would otherwise duplicate the identical
question (does BOCPD detect known regime breaks with shorter lag than
v4's own anchor heuristic?) using the identical shared engine, and this
project's parallelism rule says duplicate measurement work should not be
paid for twice.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM. See `r82_shared.py`'s module docstring for the full
   citation trail (Adams & MacKay 2007; Bianchi/Prata/Vecchio 2025) and
   not-a-duplicate-of list. One sentence: a formal Bayesian changepoint
   estimator, run on the SAME price series v4 already reads, should
   detect the six dated historical BTC regime transitions in
   `r82_shared.STRESS_EPISODES` with AT LEAST as short a lag as v4's own
   fixed 20/40/80-day anchor-crossing heuristic, because it is explicitly
   built to estimate "how long has the current regime run" rather than
   asserting a fixed window a priori.

2. DETECTION-LAG DEFINITION. For each episode, within a +/-60-day search
   window around its onset (`r82_shared.episode_window`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`r82_shared.nearest_transition`,
     `direction="down"` -- the same primary rule R-81 used and disclosed
     a correction for; "down" because de-risking in reaction to stress is
     the mechanism actually being measured).
   - BOCPD's reaction: the nearest bar where the MAP run length crosses
     down to <= `K_SHORT_DAYS` (5), to the onset
     (`r82_shared.nearest_bocpd_detection`).
   - LEAD = (v4_flip_time - bocpd_detect_time) in days. Positive = BOCPD
     detected the break before v4's own gate reacted.

3. NULL. `r82_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) MAP-run-length-derived detection series
   (block_days=5, N=500 draws, seed=82, fixed before running) and
   recomputes "nearest BOCPD detection to the real, unshifted v4 flip
   time" against each shifted copy -- the same construction R-81 used for
   its own level/positioning signal (not R-79's placebo-OFFSET device,
   which is built for a different confound: an arbitrary CYCLICAL
   partition of a trending series, not applicable to a detection-lag
   comparison against a fixed real reaction time).

4. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed): an episode counts as a PASS if BOTH (a) LEAD >= 0 (BOCPD
   detected at or before v4's own nearest reaction -- ">=", not ">",
   because unlike the INFO-axis rounds this is not a claim about an
   independent, EARLIER-arriving information channel; it is a claim that
   a formal estimator of the SAME information is at least as fast as an
   ad hoc heuristic, so simultaneous detection is a legitimate pass, not
   a wash), AND (b) the true LEAD is >= the null distribution's median
   (a materially weaker bar than R-81's "beats the 90th percentile",
   deliberately: R-81 was arguing an INDEPENDENT signal has genuine
   predictive lead-time, a strong claim: this gate is only arguing BOCPD
   is not worse than chance at matching v4's own reaction speed, a weak
   claim appropriate to what Step B actually needs -- Step B does not
   need BOCPD to beat v4 outright here, only to be a plausible, non-
   degenerate candidate worth spending a Step-B sweep on). PROCEED TO
   STEP B (dispatch both branches) only if >= 4 of the 6 episodes PASS.
   If fewer than 4 pass: STOP, report this file's result as the whole
   round's product, write it up as NEGATIVE, do not dispatch branches.
   The bar is not relaxed after seeing the numbers.

5. WHAT WOULD MAKE THIS GATE FAIL, named now: BOCPD's detections cluster
   AFTER v4's own reaction (the same failure mode every INFO-axis signal
   hit against price, now against price's OWN heuristic instead), or the
   detections are not distinguishable from an arbitrary time-shift of the
   same run-length series (i.e. genericautocorrelation/persistence
   structure in a slow-moving statistic rather than a real event-locked
   property).

CONFIGURATIONS EVALUATED IN THIS FILE: 0 (a fixed, non-swept measurement
gate, using `r82_shared.DEFAULT_HAZARD_LAMBDA` throughout -- no
hazard-rate search here; that search, if the gate passes, belongs to
Step B and is pre-registered separately in each branch's own file).
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

from experiments.r82_shared import (  # noqa: E402
    K_SHORT_DAYS,
    OOS_START,
    STRESS_EPISODES,
    anchor_majority,
    block_bootstrap_shifts,
    bocpd_daily_causal_signals,
    episode_window,
    nearest_bocpd_detection,
    nearest_transition,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 82


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


def null_leads(map_rl: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, k_short: int = K_SHORT_DAYS,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    local = map_rl.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        short = shifted <= k_short
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = short[1:] & ~short[:-1]
        cross[0] = bool(short[0])
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
    print("R-82 OPERATOR MEASUREMENT: BOCPD vs v4 anchor -- STEP A detection-lag gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    bocpd = bocpd_daily_causal_signals(bars)
    assert_no_holdout(bocpd)
    map_rl = bocpd["bocpd_map_run_length"]

    print(f"\nk_short={K_SHORT_DAYS}d  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_bocpd_detection(map_rl, window, onset, K_SHORT_DAYS)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no BOCPD detection'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(map_rl, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    BOCPD nearest detection (run_length<={K_SHORT_DAYS}d): {detect_time}")
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
    print(f"GATE VERDICT: {'PASS -> proceed to Step B (dispatch both branches)' if passed else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file: 0 (fixed measurement gate)")
    print(f"max timestamp read anywhere in this session: "
          f"{max(bars.index.max(), bocpd.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed)


if __name__ == "__main__":
    gate()
