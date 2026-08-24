#!/usr/bin/env python
"""R-117 CONSERVATIVE branch: Step-A detection-lag gate for a Donchian-
channel BREAKOUT ensemble, at `kelly_regime_v4`'s own three horizons (20,
40, 80 calendar days), run BEFORE any strategy/confirming-vote code --
identical "operator measurement" convention as R-82/R-83/R-85/R-86/R-96's
own gate files, and the same pre-registered Step-A gate methodology, for
direct comparability. See `r117_shared.py`'s module docstring for the full
citation trail (Donchian/Turtle rules; Zarattini, Pagani & Barbon 2025) and
the not-a-duplicate-of list against R-01/R-82/R-83/R-85/R-86/R-96/R-98/
R-84/R-60 and R-105.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM, one sentence: does a causal Donchian-channel breakout
   ensemble (`r117_shared.donchian_ensemble_frac`, mean of three latched
   0/1 range-breakout votes, one per lookback in `PRIMARY_LOOKBACKS`),
   at v4's own three horizons (20, 40, 80 calendar days), detect the six
   known historical BTC regime transitions with LESS lag than
   `kelly_regime_v4`'s own 3-anchor mean-crossing vote's nearest downward
   transition -- the identical detection-lag question R-82/R-83/R-85/
   R-86/R-96/R-98/R-84/R-60/R-01 asked of their own mechanisms, all nine
   of which failed it. This branch reads no data beyond the already-
   committed BTC OHLCV `high`/`low`/`close` series `kelly_regime_v4`
   itself already uses -- no new external data, no new coverage-gap risk.

2. PRIMARY PRE-REGISTERED CONFIGURATION: lookbacks=(20, 40, 80) -- v4's
   OWN exact horizons, chosen as the single primary cell for a clean
   minimal-substitution comparison (detector family changes from mean-
   crossing to range-breakout; horizons and hysteresis-latch shape stay
   identical). Two additional grid cells, (10, 20, 40) [faster] and
   (40, 60, 80) [slower-ish, capped at 80 days to respect
   `TargetStrategy`'s own 80-day warmup used elsewhere in this round],
   are computed too, but ONLY as plateau/robustness context reported
   alongside the primary result -- never used to override the primary
   cell's own pass/fail verdict. Mirrors R-96's own primary-cell-plus-
   grid convention.

3. DETECTION-LAG DEFINITION. IDENTICAL construction to
   `r96_conservative_hawkes_alarm.py`'s `gate()`/`null_leads()` -- for
   each episode, within a +/-60-day search window around its onset
   (`r117_shared.episode_window`, `WINDOW_DAYS=60`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `v4_vote_frac(df)` to the onset (`r117_shared.nearest_transition`,
     `direction="down"` -- identical rule R-82 through R-96 used).
   - Donchian ensemble's reaction: the nearest DOWNWARD transition of
     `donchian_ensemble_frac(df, lookbacks)` to the onset, via the SAME
     generic `nearest_transition(..., direction="down")` -- the ensemble
     fraction takes discrete values in {0, 1/3, 2/3, 1} for a 3-member
     ensemble, exactly like v4's own vote, so `nearest_transition` applies
     unchanged; no threshold-crossing helper is needed (unlike R-96's
     continuous z-score, which needed one).
   - LEAD = (v4_flip_time - donchian_detect_time) in days. Positive means
     the Donchian ensemble detected the transition BEFORE v4's own gate
     reacted.

4. NULL. `r117_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) Donchian ensemble fraction series (block_days=5,
   n_draws=500, seed=11701 -- fixed now, disclosed, never altered after
   seeing results) and recomputes "nearest downward transition to the
   real, unshifted v4 flip time" against each shifted copy, using the SAME
   `nearest_transition(direction="down")` logic -- adapted from R-96's
   `null_leads` (itself adapted from R-86's/R-85's/R-82's) for a discrete
   vote series instead of a continuous z-score.

5. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82 through R-96): an episode counts as a
   PASS if BOTH (a) LEAD >= 0, AND (b) the true LEAD is >= the null
   distribution's median. Using the PRIMARY CELL ONLY (lookbacks=
   (20, 40, 80)), PROCEED TO STEP B (build the confirming-vote strategy)
   only if >= 4 of the 6 episodes PASS. If fewer than 4 pass on the
   primary cell: STOP, report this file's result as the whole
   conservative branch's product, write it up as NEGATIVE, do not build
   any strategy/confirming-vote code, do not touch any data on or after
   2023-01-01. The bar is not relaxed, narrowed, or otherwise adjusted
   after seeing the numbers, and the other 2 grid cells never override
   this verdict.

6. WHAT WOULD MAKE THIS GATE FAIL, named now (copied/adapted from
   `r117_shared.py`'s own "WHAT WOULD MAKE EACH BRANCH FAIL" section, and
   named there as this branch's own pre-registered EXPECTATION, not a
   hoped-for result): a breakout detector requires price to have already
   cleared the recent extreme -- if anything, this is a STRUCTURALLY MORE
   lagging construction than v4's own "1% past a rolling mean" band, since
   a fresh N-day high/low is by construction a rarer, later event than a
   1%-past-the-mean crossing. The single most likely outcome, named before
   any bar was read, is that this becomes the TENTH mechanism to fail the
   identical Step-A gate the same way its nine predecessors did -- lagging
   every sudden 2020-2022 shock and, at best, tying or narrowly leading
   only the slow 2018 build-up. A clean NEGATIVE closing regime-timing
   mechanism #10 is the fully expected, fully successful outcome of this
   branch; report honestly whichever way it actually comes out.

CONFIGURATIONS EVALUATED IN THIS FILE: up to 18 (6 STRESS_EPISODES x 3
grid cells -- PRIMARY (20,40,80) plus 2 robustness cells (10,20,40) and
(40,60,80)). Of these, exactly ONE grid cell -- the primary cell named in
point 2 above -- is the pre-registered decision cell whose >=4/6 pass
count determines whether this branch proceeds to Step B. The other 2 grid
cells are non-decision robustness/plateau context only. The causal
truncation probe (point 6 of the task spec, run once at the primary
lookback set) is a separate bug check, not counted among the 18.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r117_shared import (  # noqa: E402
    BARS_PER_DAY,
    OOS_START,
    STRESS_EPISODES,
    assert_no_holdout,
    block_bootstrap_shifts,
    donchian_ensemble_frac,
    episode_window,
    hr,
    load_btc,
    nearest_transition,
    v4_vote_frac,
)

WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 11701

PRIMARY_LOOKBACKS = (20, 40, 80)
GRID = [
    (20, 40, 80),   # PRIMARY -- v4's own exact horizons
    (10, 20, 40),   # faster
    (40, 60, 80),   # slower-ish, capped at 80d per TargetStrategy's own warmup
]


def null_leads(donch_frac: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r96_conservative_hawkes_alarm's `null_leads`: same
    circular block-shift construction, applied here to the discrete
    Donchian ensemble fraction series via the SAME generic
    `nearest_transition(direction="down")` logic (no threshold-crossing
    helper needed -- the ensemble fraction is already discrete)."""
    local = donch_frac.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = pd.Series(local[shift], index=window)
        detect_time = nearest_transition(shifted, window, onset, direction="down")
        if detect_time is None:
            continue
        leads[k] = (flip_time - detect_time).total_seconds() / 86400.0
    return leads


def gate(lookbacks: tuple[int, ...], bars: pd.DataFrame, v4_frac: pd.Series,
         is_primary: bool) -> dict:
    tag = " [PRIMARY]" if is_primary else ""
    tag_str = "_".join(str(l) for l in lookbacks)
    print("=" * 78)
    print(f"R-117 CONSERVATIVE: Donchian ensemble{lookbacks} vs v4 anchor vote -- "
          f"STEP A detection-lag gate{tag}")
    print("=" * 78)

    donch_frac = donchian_ensemble_frac(bars, lookbacks)
    assert_no_holdout(donch_frac)

    print(f"\nlookbacks={lookbacks}d  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"), pass_b=False))
            continue

        flip_time = nearest_transition(v4_frac, window, onset, direction="down")
        detect_time = nearest_transition(donch_frac, window, onset, direction="down")

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no v4 vote transition' if flip_time is None else 'no Donchian ensemble transition'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=flip_time, detect=detect_time,
                                 lead=float("nan"), null_median=float("nan"), pass_b=False))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(donch_frac, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 vote nearest downward flip: {flip_time}")
        print(f"    Donchian{lookbacks} nearest downward transition: {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time, detect=detect_time,
                             lead=lead, null_median=null_median, pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 4

    print("\n" + "-" * 78)
    for r in results:
        print(f"  {r['label']:42s} lead={r.get('lead', float('nan')):+.2f}d  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/6  (lookbacks={lookbacks}){tag}")
    print(f"max timestamp read anywhere for this cell: "
          f"{max(bars.index.max(), donch_frac.index.max())}  (< {OOS_START})")

    return dict(lookbacks=lookbacks, tag=tag_str, is_primary=is_primary,
                results=results, n_pass=n_pass, passed=passed, signal=donch_frac)


def truncation_causality_probe(bars: pd.DataFrame, lookbacks: tuple[int, ...],
                                check_at: int, shorter_by: int = 40_000) -> bool:
    """Standard truncation probe (mirrors r96_shared's own
    `truncation_causality_probe` / r82_shared's identical pattern): does
    `donchian_ensemble_frac(...)[check_at]` change if bars after it are
    dropped? Returns True if causal (identical both ways)."""
    full = donchian_ensemble_frac(bars, lookbacks).to_numpy()
    short = donchian_ensemble_frac(bars.iloc[: check_at + shorter_by].copy(), lookbacks).to_numpy()
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))


def causality_probe(bars: pd.DataFrame) -> bool:
    """Independent re-verification of the causal truncation claim for
    `donchian_ensemble_frac` at the PRIMARY lookback set, run before any
    headline result is trusted, per this project's rule that every round
    re-runs the probe itself rather than trusting a prior claim. Probes at
    a point well inside the 2018 episode window so the check exercises
    real, non-degenerate signal values."""
    check_at = bars.index.get_indexer(
        [pd.Timestamp("2018-06-01", tz="UTC")], method="nearest")[0]
    ok = truncation_causality_probe(bars, PRIMARY_LOOKBACKS, check_at, shorter_by=40_000)
    print(f"\ncausal truncation probe (donchian_ensemble_frac, lookbacks={PRIMARY_LOOKBACKS}, "
          f"check_at index {check_at} ~ {bars.index[check_at]}): {'PASS' if ok else 'FAIL'}")
    return ok


def run_full_grid() -> dict:
    bars = load_btc()
    assert_no_holdout(bars)
    print(f"BTC (spot): {len(bars):,} bars  {bars.index[0]} -> {bars.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)

    v4_frac = v4_vote_frac(bars)
    assert_no_holdout(v4_frac)

    cells = []
    primary_cell = None
    max_ts_seen = bars.index.max()
    n_configs = 0
    for lookbacks in GRID:
        is_primary = (lookbacks == PRIMARY_LOOKBACKS)
        cell = gate(lookbacks, bars, v4_frac, is_primary)
        cells.append(cell)
        n_configs += len(cell["results"])
        if cell["is_primary"]:
            primary_cell = cell
        max_ts_seen = max(max_ts_seen, cell["signal"].index.max())

    assert primary_cell is not None, "primary lookback set (20,40,80) missing from grid"

    hr("R-117 CONSERVATIVE: FULL 3-CELL GRID SUMMARY (Donchian ensemble vs v4 anchor vote)")
    print(f"{'lookbacks':>14} {'n_pass/6':>10}  {'primary?':>9}")
    for cell in cells:
        marker = "<-- PRIMARY" if cell["is_primary"] else ""
        print(f"{str(cell['lookbacks']):>14} {cell['n_pass']:>8}/6  {marker}")

    hr("R-117 CONSERVATIVE: VERDICT")
    print(f"PRIMARY CELL (lookbacks={PRIMARY_LOOKBACKS}): {primary_cell['n_pass']}/6 episodes pass")
    decision_pass = primary_cell["passed"]
    print(f"DECISION RULE (>=4/6 on primary cell only): "
          f"{'PASS' if decision_pass else 'FAIL'}")
    if decision_pass:
        print("BRANCH VERDICT: PROMOTE-CANDIDATE for Step B. Per dispatch scope, Step-B "
              "strategy/backtest code is NOT built in this file -- STOPPING here anyway "
              "and reporting this as a live follow-on for a future round.")
    else:
        print("BRANCH VERDICT: NEGATIVE. Donchian breakout, substituted into the identical "
              "Step-A detection-lag gate, is the tenth mechanism to fail it -- consistent "
              "with this round's own pre-registered expectation (r117_shared.py). No "
              "Step-B strategy/backtest code was built; no bar on or after "
              f"{OOS_START} was read.")

    print(f"\nconfigurations evaluated: {n_configs} "
          f"(6 episodes x {len(GRID)} grid cells; 1 primary decision cell + "
          f"{len(GRID) - 1} non-decision robustness cells)")
    print(f"max timestamp read anywhere in this session: {max_ts_seen}  (< {OOS_START})")
    assert max_ts_seen < pd.Timestamp(OOS_START, tz="UTC"), "holdout bar read"

    return dict(cells=cells, primary=primary_cell, n_configs=n_configs,
                max_ts_seen=max_ts_seen, decision_pass=decision_pass)


if __name__ == "__main__":
    bars_for_probe = load_btc()
    assert_no_holdout(bars_for_probe)
    hr("R-117 CONSERVATIVE: CAUSAL TRUNCATION PROBE (run before any headline result is trusted)")
    probe_ok = causality_probe(bars_for_probe)
    if not probe_ok:
        print("\n*** CAUSALITY PROBE FAILED -- STOPPING. Results below would NOT be "
              "trustworthy; a lookahead bug must be investigated, not reported around. ***",
              file=sys.stderr)
        sys.exit(1)

    grid_result = run_full_grid()
