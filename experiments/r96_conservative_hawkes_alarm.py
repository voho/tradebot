#!/usr/bin/env python
"""R-96 CONSERVATIVE branch: Step-A detection-lag gate for a self-exciting
Hawkes point process (jump-clustering intensity), single-indicator, run
BEFORE any strategy/confirming-vote code -- identical "operator measurement"
convention as R-82/R-83/R-85/R-86's own gate files, and the same
pre-registered Step-A gate methodology, for direct comparability.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM. See `r96_shared.py`'s module docstring for the full citation
   trail (Hawkes 1971; Bacry, Mastromatteo & Muzy 2015; Barndorff-Nielsen &
   Shephard 2004/2006; 2025 arXiv/quant-fin surveys) and not-a-duplicate-of
   list. One sentence: does the Hawkes conditional intensity of price-jump
   clustering, converted to a causal z-score (`hawkes_intensity_zscore`),
   cross its alarm threshold with LESS lag than `kelly_regime_v4`'s own
   fixed 20/40/80-day anchor-crossing heuristic, on the same six dated
   historical BTC regime transitions R-82/R-83/R-85/R-86 used
   (`r96_shared.STRESS_EPISODES`)?

2. PRIMARY CONFIG (fixed a priori, the only one used for the gate decision
   -- no sweep in Step-A, matching R-82/R-85's own convention of "0
   configurations evaluated, a fixed measurement gate"): `n=0.5` (the
   middle of `N_GRID`), `halflife_days=7` (the middle of
   `HALFLIFE_DAYS_GRID`) -- the a-priori median cell of each grid, chosen
   before any real number was computed. Event flag via
   `intraday_relative_jump(bars)`; `lam = hawkes_intensity_daily(event_flag,
   n=0.5, halflife_days=7)`; `z = hawkes_intensity_zscore(lam)`; then `z`
   (daily-indexed) is causally aligned onto `bars`' 5-minute index via
   `align_daily_causal(z, bars)` before the episode gate runs.

3. DETECTION-LAG DEFINITION. For each episode, within a +/-60-day search
   window around its onset (`r96_shared.episode_window`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`r96_shared.nearest_transition`,
     `direction="down"` -- identical rule R-82/83/85/86 used).
   - Hawkes's reaction: the nearest bar where the aligned z-score crosses
     UP through `Z_THRESH=2.0` (`r96_shared.nearest_hawkes_alarm`), closest
     to the onset.
   - LEAD = (v4_flip_time - detect_time) in days. Positive = Hawkes
     alarmed before v4's own gate reacted.

4. NULL. `r96_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) aligned Hawkes z-score series (block_days=7 --
   this signal's own halflife, a daily-cadence-derived signal, one week
   block --, n_draws=500, seed=9601, this round's own seed, distinct from
   R-85's 8501/R-88's 8801, fixed before running) and recomputes "nearest
   Hawkes alarm to the real, unshifted v4 flip time" against each shifted
   copy -- adapted from R-85's `null_leads` line-for-line, just plugged
   with the Hawkes z-score/seed/block_days.

5. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82/83/85/86): an episode counts as a PASS
   if BOTH (a) LEAD >= 0, AND (b) the true LEAD is >= the null
   distribution's median. PROCEED TO STEP B (build the confirming-vote
   strategy) only if >= 4 of the 6 episodes PASS. If fewer than 4 pass:
   STOP, report this file's result as the whole branch's product, write it
   up as NEGATIVE, do not build any strategy/confirming-vote code. The bar
   is not relaxed, narrowed, or otherwise adjusted after seeing the
   numbers.

6. WHAT WOULD MAKE THIS GATE FAIL, named now (and named in `r96_shared.py`
   as this round's own pre-registered EXPECTATION, not a hoped-for
   result): the same pattern that has now beaten six consecutive
   mechanisms built on six different theoretical bases -- an estimator
   computed FROM price (here: jump event timing) can only rise once a jump
   has already happened, exactly the moment v4's own fixed-window anchor
   is also starting to react. Hawkes intensity likely also lags sudden
   shocks and at best leads only the slow 2018 build-up.

7. ROBUSTNESS GRID (informational only, computed AFTER and clearly labeled
   as NOT used for the Step-B gate decision -- avoids picking-best-of-9
   selection bias): after reporting the primary-config gate verdict, the
   same episode-detection-lag procedure (LEAD only, no null/no
   PASS-per-episode) is run for the other 8 `(n, halflife_days)` cells in
   `N_GRID x HALFLIFE_DAYS_GRID` minus the primary, so the primary
   decision's neighbourhood (plateau vs. peak) is visible.

CONFIGURATIONS EVALUATED IN THIS FILE: 0 for the gate decision (a fixed,
non-swept measurement gate, using `r96_shared.Z_THRESH=2.0` and the single
primary (n=0.5, halflife_days=7) cell throughout -- no threshold/grid
search feeds the gate) + 8 informational robustness cells (reported
separately, explicitly not used for the Step-B decision).
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

from experiments.r96_shared import (  # noqa: E402
    HALFLIFE_DAYS_GRID,
    N_GRID,
    OOS_START,
    STRESS_EPISODES,
    Z_THRESH,
    align_daily_causal,
    anchor_majority,
    assert_no_holdout,
    block_bootstrap_shifts,
    episode_window,
    hawkes_intensity_daily,
    hawkes_intensity_zscore,
    intraday_relative_jump,
    nearest_hawkes_alarm,
    nearest_transition,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 7
NULL_SEED = 9601

PRIMARY_N = 0.5
PRIMARY_HALFLIFE_DAYS = 7


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def hawkes_aligned_zscore(bars: pd.DataFrame, n: float, halflife_days: float) -> pd.Series:
    """Compose the full causal pipeline: daily jump-event flag -> Hawkes
    intensity -> z-score -> causal 5-minute alignment onto `bars`."""
    event_flag = intraday_relative_jump(bars)
    lam = hawkes_intensity_daily(event_flag, n=n, halflife_days=halflife_days)
    z = hawkes_intensity_zscore(lam)
    return align_daily_causal(z, bars)


def null_leads(hawkes_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r85's `null_leads` line-for-line: same circular
    block-shift construction, plugged with the aligned Hawkes z-score and
    this round's own seed/block_days."""
    local = hawkes_z.reindex(window).to_numpy()
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
    print("R-96 CONSERVATIVE: Hawkes intensity vs v4 anchor -- STEP A detection-lag gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    hawkes_z = hawkes_aligned_zscore(bars, PRIMARY_N, PRIMARY_HALFLIFE_DAYS)
    assert_no_holdout(hawkes_z.dropna())

    print(f"\nPRIMARY CONFIG: n={PRIMARY_N}  halflife_days={PRIMARY_HALFLIFE_DAYS}  "
          f"z_thresh={Z_THRESH}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, onset=onset_str, pass_b=False, lead=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_hawkes_alarm(hawkes_z, window, onset, Z_THRESH)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no Hawkes alarm'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str,
                                 flip=flip_time, detect=detect_time,
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(hawkes_z, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    Hawkes nearest alarm (z>={Z_THRESH}): {detect_time}")
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
    print(f"\nconfigurations evaluated for the gate decision: 0 (fixed measurement gate, "
          f"primary config n={PRIMARY_N}, halflife_days={PRIMARY_HALFLIFE_DAYS})")

    max_ts = max(bars.index.max(), hawkes_z.dropna().index.max())
    print(f"max timestamp read anywhere in this session: {max_ts}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed, bars=bars,
                majority=majority, hawkes_z=hawkes_z, max_ts=max_ts)


def robustness_grid(bars: pd.DataFrame, majority: pd.Series) -> None:
    """INFORMATIONAL ONLY -- computed AFTER the primary-config gate
    decision above, and explicitly NOT used for the Step-B gate decision.
    Reports LEAD (days) per episode for the 8 (n, halflife_days) cells in
    N_GRID x HALFLIFE_DAYS_GRID other than the primary (0.5, 7), so the
    primary cell's neighbourhood (plateau vs. peak) is visible. No
    null/PASS bookkeeping -- LEAD or 'no alarm found' only."""
    print("\n" + "=" * 78)
    print("ROBUSTNESS GRID (informational only -- NOT used for the Step-B gate decision)")
    print(f"8 cells of N_GRID x HALFLIFE_DAYS_GRID minus primary "
          f"(n={PRIMARY_N}, halflife_days={PRIMARY_HALFLIFE_DAYS})")
    print("=" * 78)

    cells = [(n, hl) for n in N_GRID for hl in HALFLIFE_DAYS_GRID
             if not (n == PRIMARY_N and hl == PRIMARY_HALFLIFE_DAYS)]
    assert len(cells) == 8, f"expected 8 non-primary cells, got {len(cells)}"

    max_ts_seen = bars.index.max()
    for n, hl in cells:
        z_cell = hawkes_aligned_zscore(bars, n, hl)
        assert_no_holdout(z_cell.dropna())
        max_ts_seen = max(max_ts_seen, z_cell.dropna().index.max())
        print(f"\n  n={n}  halflife_days={hl}")
        for label, onset_str in STRESS_EPISODES:
            onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
            if len(window) == 0:
                print(f"    [{label}] window outside data coverage")
                continue
            flip_time = nearest_transition(majority, window, onset, direction="down")
            detect_time = nearest_hawkes_alarm(z_cell, window, onset, Z_THRESH)
            if flip_time is None or detect_time is None:
                print(f"    [{label}] no alarm found")
                continue
            lead = (flip_time - detect_time).total_seconds() / 86400.0
            print(f"    [{label}] LEAD = {lead:+.2f}d")

    print(f"\nrobustness cells evaluated (informational, not gating): {len(cells)}")
    print(f"max timestamp read anywhere in robustness grid: {max_ts_seen}  (< {OOS_START})")


def causality_probe() -> bool:
    """Independent re-verification of the composed
    intraday_relative_jump + hawkes_intensity_daily +
    hawkes_intensity_zscore + align_daily_causal pipeline's own causal
    truncation claim, per this project's rule that every round re-runs the
    probe itself rather than trusting a prior claim. Probes at a point
    well inside the 2018 episode window so the check exercises real,
    non-degenerate signal values."""
    bars = load_btc_bars()
    check_at = bars.index.get_indexer([pd.Timestamp("2018-06-01", tz="UTC")], method="nearest")[0]

    def build(df: pd.DataFrame) -> np.ndarray:
        return hawkes_aligned_zscore(df, PRIMARY_N, PRIMARY_HALFLIFE_DAYS).to_numpy()

    ok = truncation_causality_probe(build, bars, check_at, shorter_by=40_000)
    print(f"\ncausal truncation probe (hawkes z-score, n={PRIMARY_N}, "
          f"halflife_days={PRIMARY_HALFLIFE_DAYS}, check_at index {check_at} "
          f"~ {bars.index[check_at]}): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    causality_probe()
    gate_result = gate()
    robustness_grid(gate_result["bars"], gate_result["majority"])
