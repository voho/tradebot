#!/usr/bin/env python
"""R-86 CONSERVATIVE branch: Step-A detection-lag gate for transfer entropy
TE_{volume_change -> return}, single-indicator, run BEFORE any strategy/
confirming-vote code -- identical "operator measurement" convention as
R-82/R-83/R-85's own gate files, and the same pre-registered Step-A gate
methodology, for direct comparability.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM. See `r86_shared.py`'s module docstring for the full citation
   trail (Schreiber 2000; Garcia-Medina & Hernandez C. 2020) and
   not-a-duplicate-of list. One sentence: does daily log-volume-change
   carry directed information about next-day log-returns beyond what
   returns' own history already explains -- i.e. is
   `TE_{volume_change -> return}` (`r86_shared.transfer_entropy`, applied
   causally via `rolling_transfer_entropy`), converted to a causal trend
   z-score (`te_z`, `r86_shared.trend_zscore`), elevated and RISING ahead
   of a regime break, with LESS lag than v4's own fixed 20/40/80-day
   anchor-crossing heuristic, on the same six dated historical BTC regime
   transitions R-82/R-83/R-85 used. This branch reads no data beyond the
   already-committed BTC OHLCV close/volume series `kelly_regime_v4`
   itself already uses -- no new external data, no new coverage-gap risk.

2. DETECTION-LAG DEFINITION. For each episode, within a +/-60-day search
   window around its onset (`r86_shared.episode_window`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`r86_shared.nearest_transition`,
     `direction="down"` -- identical rule R-82/R-83/R-85 used).
   - TE's reaction: the nearest bar where `te_z` crosses UP through
     `Z_THRESH=2.0` (`r86_shared.nearest_te_alarm`), closest to the onset.
   - LEAD = (v4_flip_time - detect_time) in days. Positive = TE alarmed
     before v4's own gate reacted.

3. NULL. `r86_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) `te_z` series (block_days=5, n_draws=500,
   seed=8601, fixed before running and never altered afterward) and
   recomputes "nearest TE alarm to the real, unshifted v4 flip time"
   against each shifted copy -- adapted from R-85's `null_leads` (itself
   adapted from R-82's) for the identical threshold-crossing detector
   construction; logic is otherwise identical.

4. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82/R-83/R-85): an episode counts as a
   PASS if BOTH (a) LEAD >= 0, AND (b) the true LEAD is >= the null
   distribution's median. PROCEED TO STEP B (build the confirming-vote
   strategy) only if >= 4 of the 6 episodes PASS. If fewer than 4 pass:
   STOP, report this file's result as the whole branch's product, write
   it up as NEGATIVE, do not build any strategy/confirming-vote code, do
   not touch any data on or after 2023-01-01. The bar is not relaxed,
   narrowed, or otherwise adjusted after seeing the numbers.

5. WHAT WOULD MAKE THIS GATE FAIL, named now (and named in `r86_shared.py`
   as this round's own pre-registered EXPECTATION, not a hoped-for
   result): TE_{volume->return}, like variance (CSD, R-85), a Bayesian
   run-length posterior (BOCPD, R-82) and a linear state-space filter
   (Kalman LLT, R-83) before it, is itself a statistic OF price/volume
   fluctuations, so it can only rise once those fluctuations have already
   become unusual -- which is exactly the moment v4's own fixed-window
   anchor is also starting to react. If TE also lags every sudden
   2020-2022 shock and only leads (if anything) the slow 2018 build-up,
   or fails to lead anything at all (as R-84's conservative volume branch
   did), that is a fifth independent mechanism converging on the same
   conclusion the ledger's standing diagnosis is already leaning toward:
   this six-episode gate is unwinnable by any estimator computed from
   this project's own committed price/volume history, and the finding is
   about the gate/dataset rather than about any one technique.

CONFIGURATIONS EVALUATED IN THIS FILE: 0 (a fixed, non-swept measurement
gate, using `r86_shared.Z_THRESH=2.0` throughout -- no threshold search
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

from experiments.r86_shared import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    DETECTION_WINDOW_DAYS,
    OOS_START,
    STRESS_EPISODES,
    TE_SUB_WINDOW_DAYS,
    Z_THRESH,
    align_daily_causal,
    anchor_majority,
    block_bootstrap_shifts,
    daily_log_returns,
    daily_log_volume_change,
    episode_window,
    nearest_te_alarm,
    nearest_transition,
    rolling_transfer_entropy,
    trend_zscore,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 8601


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


def build_te_z_aligned(bars: pd.DataFrame) -> pd.DataFrame:
    """Full signal-building pipeline (steps 1-3 of the task spec):
    daily log-volume-change (x) -> daily log-return (y), inner-joined on
    date, rolling TE_{x->y}, causal trend z-score, causally aligned onto
    `bars`' 5-minute index. Returns a 1-column DataFrame (`te_z`) indexed
    like `bars`."""
    x = daily_log_volume_change(bars)
    y = daily_log_returns(bars)
    idx = x.index.intersection(y.index)
    x = x.reindex(idx)
    y = y.reindex(idx)
    te = rolling_transfer_entropy(x, y, sub_window_days=TE_SUB_WINDOW_DAYS)
    z = trend_zscore(te, detection_window_days=DETECTION_WINDOW_DAYS,
                      baseline_window_days=BASELINE_WINDOW_DAYS)
    aligned = align_daily_causal(pd.DataFrame({"te_z": z}), bars)
    return aligned


def null_leads(te_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r85's `null_leads` (itself adapted from r82_gate.py's):
    same circular block-shift construction, applied here to the TE trend
    z-score threshold-crossing detector."""
    local = te_z.reindex(window).to_numpy()
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
    print("R-86 CONSERVATIVE: TE(volume->return) vs v4 anchor -- STEP A detection-lag gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    signal = build_te_z_aligned(bars)
    assert_no_holdout(signal)
    te_z = signal["te_z"]

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
        detect_time = nearest_te_alarm(te_z, window, onset, Z_THRESH)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no TE alarm'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str,
                                 flip=flip_time, detect=detect_time,
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(te_z, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    TE(volume->return) nearest alarm (z>={Z_THRESH}): {detect_time}")
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
          f"{max(bars.index.max(), signal.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed, bars=bars, signal=signal)


def causality_probe() -> bool:
    """Independent re-verification of the causal truncation claim for this
    file's own signal-building pipeline (`build_te_z_aligned`), per this
    project's rule that every round re-runs the probe itself rather than
    trusting a prior claim. Probes at a point well inside the 2018 episode
    window so the check exercises real, non-degenerate signal values."""
    bars = load_btc_bars()
    check_at = bars.index.get_indexer([pd.Timestamp("2018-06-01", tz="UTC")], method="nearest")[0]

    def build(df: pd.DataFrame) -> np.ndarray:
        return build_te_z_aligned(df)["te_z"].to_numpy()

    ok = truncation_causality_probe(build, bars, check_at, shorter_by=40_000)
    print(f"\ncausal truncation probe (te_z, check_at index {check_at} "
          f"~ {bars.index[check_at]}): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    causality_probe()
    gate()
