#!/usr/bin/env python
"""R-96 CONSERVATIVE branch: Step-A detection-lag gate for a self-exciting
Hawkes point-process intensity fit to BTC's own intraday jump-event
history, single-indicator, run BEFORE any strategy/confirming-vote code --
identical "operator measurement" convention as R-82/R-83/R-85/R-86's own
gate files, and the same pre-registered Step-A gate methodology, for
direct comparability.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM. See `r96_shared.py`'s module docstring for the full citation
   trail (Hawkes 1971; Bacry, Mastromatteo & Muzy 2015) and
   not-a-duplicate-of list. One sentence: does the causal daily
   Hawkes-intensity z-score (`r96_shared.hawkes_intensity_zscore` applied
   to `r96_shared.hawkes_intensity_daily` applied to
   `r96_shared.intraday_relative_jump`, then `r96_shared.align_daily_causal`
   onto the 5-minute bars), crossing UP through `Z_THRESH=2.0`
   (`r96_shared.nearest_hawkes_alarm`), alarm BEFORE `anchor_majority`'s own
   nearest downward transition (`r96_shared.nearest_transition(...,
   direction="down")`), on the same six dated historical BTC regime
   transitions R-82/R-83/R-85/R-86 used -- the identical detection-lag
   question those four rounds asked of their own mechanisms. This branch
   reads no data beyond the already-committed BTC OHLCV close series
   `kelly_regime_v4` itself already uses -- no new external data, no new
   coverage-gap risk.

2. PRIMARY PRE-REGISTERED CONFIGURATION: n=0.5, halflife_days=7 -- the
   CENTER of `N_GRID=(0.3,0.5,0.7)` and `HALFLIFE_DAYS_GRID=(3,7,14)`,
   chosen as the single primary cell precisely so the decision is not
   "pick whichever of the 9 grid cells scores best" (that would be an
   undisclosed multiple-comparisons search across an a-priori grid, which
   this project's routine explicitly treats as p-hacking). The other 8
   grid cells are computed too, but ONLY as a plateau/robustness context
   reported alongside the primary result -- never used to override the
   primary cell's own pass/fail verdict. This mirrors this project's own
   established convention of reporting a primary cell plus a robustness
   grid (e.g. how B-21's ledger entry separates its primary config from
   "the grid's actual best cell").

3. DETECTION-LAG DEFINITION. IDENTICAL construction to
   `r86_conservative_te_volume_return.py`'s `gate()`/`null_leads()` -- for
   each episode, within a +/-60-day search window around its onset
   (`r96_shared.episode_window`, `WINDOW_DAYS=60`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`r96_shared.nearest_transition`,
     `direction="down"` -- identical rule R-82/R-83/R-85/R-86 used).
   - Hawkes's reaction: the nearest bar where the Hawkes intensity z-score
     crosses UP through `Z_THRESH=2.0` (`r96_shared.nearest_hawkes_alarm`),
     closest to the onset.
   - LEAD = (v4_flip_time - detect_time) in days. Positive = the Hawkes
     alarm fired before v4's own gate reacted.

4. NULL. `r96_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) Hawkes z-score series (block_days=5, n_draws=500,
   seed=9601 -- fixed now, disclosed, never altered after seeing results;
   note this is a DIFFERENT seed than R-86 used (8601), chosen arbitrarily
   and fixed before running, not selected after seeing any number) and
   recomputes "nearest alarm to the real, unshifted v4 flip time" against
   each shifted copy -- adapted from R-86's `null_leads` (itself adapted
   from R-85's, itself from R-82's) for the identical threshold-crossing
   detector construction; logic is otherwise identical.

5. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82/R-83/R-85/R-86): an episode counts as a
   PASS if BOTH (a) LEAD >= 0, AND (b) the true LEAD is >= the null
   distribution's median. Using the PRIMARY CONFIG ONLY (n=0.5,
   halflife_days=7), PROCEED TO STEP B (build the confirming-vote
   strategy) only if >= 4 of the 6 episodes PASS. If fewer than 4 pass on
   the primary config: STOP, report this file's result as the whole
   conservative branch's product, write it up as NEGATIVE, do not build
   any strategy/confirming-vote code, do not touch any data on or after
   2023-01-01. The bar is not relaxed, narrowed, or otherwise adjusted
   after seeing the numbers, and the other 8 grid cells never override
   this verdict.

6. WHAT WOULD MAKE THIS GATE FAIL, named now (copied/adapted from
   `r96_shared.py`'s own "WHAT WOULD MAKE THIS FAIL" section, and named in
   that file as this round's own pre-registered EXPECTATION, not a
   hoped-for result): the same pattern that has now beaten six consecutive
   mechanisms built on six different theoretical bases -- an estimator
   computed FROM price (here: jump event timing) can only rise once a jump
   has already happened, which is exactly the moment v4's own fixed-window
   anchor is also starting to react. If Hawkes intensity also lags every
   sudden 2020-2022 shock and only (at best) leads the slow 2018 build-up,
   that is the seventh independent mechanism converging on the same
   conclusion the ledger's standing diagnosis already leans toward: this
   six-episode gate is unwinnable by any estimator computed from this
   project's own committed price history, whatever field it is drawn
   from, and the finding is about the gate/dataset rather than about any
   one technique.

CONFIGURATIONS EVALUATED IN THIS FILE: 9 (the full `N_GRID x
HALFLIFE_DAYS_GRID` a-priori grid -- the gate is computed and printed for
all 9 cells so the primary cell's result can be read in its plateau
context). Of these 9, exactly ONE -- the primary cell (n=0.5,
halflife_days=7) named in point 2 above -- is the pre-registered decision
cell whose pass/fail verdict determines whether this branch proceeds to
Step B. The other 8 are non-decision robustness/plateau context only, in
the same spirit as R-86_conservative's own "CONFIGURATIONS EVALUATED IN
THIS FILE: 0" line, adapted here because this mechanism (unlike TE, which
had a single fixed sub-window) is defined over an a-priori 3x3 grid that
must be disclosed and reported in full rather than silently reduced to
one number.
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
BLOCK_DAYS = 5
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


def build_hawkes_z_aligned(bars: pd.DataFrame, n: float, halflife_days: float,
                            jump_flag: pd.Series | None = None) -> pd.DataFrame:
    """Full signal-building pipeline: intraday relative-jump event flag ->
    causal daily Hawkes intensity -> causal z-score -> causally aligned
    onto `bars`' 5-minute index. Returns a 1-column DataFrame (`hawkes_z`)
    indexed like `bars`. `jump_flag` may be supplied precomputed (it does
    not depend on `n`/`halflife_days`) to avoid recomputing it once per
    grid cell."""
    if jump_flag is None:
        jump_flag = intraday_relative_jump(bars)
    lam = hawkes_intensity_daily(jump_flag, n, halflife_days)
    z = hawkes_intensity_zscore(lam)
    aligned = align_daily_causal(pd.DataFrame({"hawkes_z": z}), bars)
    return aligned


def null_leads(hawkes_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r86_conservative's `null_leads` (itself adapted from
    r85's, itself from r82_gate.py's): same circular block-shift
    construction, applied here to the Hawkes intensity z-score
    threshold-crossing detector."""
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


def gate(n: float, halflife_days: float, bars: pd.DataFrame, majority: pd.Series,
          jump_flag: pd.Series) -> dict:
    is_primary = (n == PRIMARY_N and halflife_days == PRIMARY_HALFLIFE_DAYS)
    tag = " [PRIMARY]" if is_primary else ""
    print("=" * 78)
    print(f"R-96 CONSERVATIVE: Hawkes(n={n}, halflife={halflife_days}d) vs v4 anchor "
          f"-- STEP A detection-lag gate{tag}")
    print("=" * 78)

    signal = build_hawkes_z_aligned(bars, n, halflife_days, jump_flag=jump_flag)
    assert_no_holdout(signal)
    hawkes_z = signal["hawkes_z"]

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
        print(f"    Hawkes(n={n},hl={halflife_days}d) nearest alarm (z>={Z_THRESH}): {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             detect=detect_time, lead=lead, null_median=null_median,
                             pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 4

    print("\n" + "-" * 78)
    for r in results:
        print(f"  {r['label']:42s} lead={r.get('lead', float('nan')):+.2f}d  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/6  (n={n}, halflife_days={halflife_days}){tag}")
    print(f"max timestamp read anywhere for this cell: "
          f"{max(bars.index.max(), signal.index.max())}  (< {OOS_START})")

    return dict(n=n, halflife_days=halflife_days, is_primary=is_primary,
                results=results, n_pass=n_pass, passed=passed, signal=signal)


def causality_probe() -> bool:
    """Independent re-verification of the causal truncation claim for this
    file's own signal-building pipeline (`build_hawkes_z_aligned`), run at
    the PRIMARY config, per this project's rule that every round re-runs
    the probe itself rather than trusting a prior claim. Probes at a point
    well inside the 2018 episode window so the check exercises real,
    non-degenerate signal values."""
    bars = load_btc_bars()
    check_at = bars.index.get_indexer([pd.Timestamp("2018-06-01", tz="UTC")], method="nearest")[0]

    def build(df: pd.DataFrame) -> np.ndarray:
        return build_hawkes_z_aligned(df, PRIMARY_N, PRIMARY_HALFLIFE_DAYS)["hawkes_z"].to_numpy()

    ok = truncation_causality_probe(build, bars, check_at, shorter_by=40_000)
    print(f"\ncausal truncation probe (hawkes_z, n={PRIMARY_N}, halflife={PRIMARY_HALFLIFE_DAYS}d, "
          f"check_at index {check_at} ~ {bars.index[check_at]}): {'PASS' if ok else 'FAIL'}")
    return ok


def run_full_grid() -> dict:
    bars = load_btc_bars()
    majority = anchor_majority(bars)
    jump_flag = intraday_relative_jump(bars)
    assert_no_holdout(jump_flag)

    cells = []
    primary_cell = None
    max_ts_seen = bars.index.max()
    for halflife_days in HALFLIFE_DAYS_GRID:
        for n in N_GRID:
            cell = gate(n, halflife_days, bars, majority, jump_flag)
            cells.append(cell)
            if cell["is_primary"]:
                primary_cell = cell
            max_ts_seen = max(max_ts_seen, cell["signal"].index.max())

    assert primary_cell is not None, "primary config (n=0.5, halflife=7) missing from grid"

    print("\n" + "=" * 78)
    print("R-96 CONSERVATIVE: FULL 3x3 GRID SUMMARY (Hawkes(n, halflife) vs v4 anchor)")
    print("=" * 78)
    print(f"{'n':>5} {'halflife_days':>14} {'n_pass/6':>10}  {'primary?':>9}")
    for cell in cells:
        marker = "<-- PRIMARY" if cell["is_primary"] else ""
        print(f"{cell['n']:>5} {cell['halflife_days']:>14} {cell['n_pass']:>8}/6  {marker}")

    print(f"\nPRIMARY CONFIG (n={PRIMARY_N}, halflife_days={PRIMARY_HALFLIFE_DAYS}): "
          f"{primary_cell['n_pass']}/6 episodes pass")
    print(f"GATE VERDICT (primary config only, per pre-registered stop rule): "
          f"{'PASS -> proceed to Step B (build confirming-vote strategy)' if primary_cell['passed'] else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file: 9 (full N_GRID x HALFLIFE_DAYS_GRID; "
          f"1 primary decision cell + 8 non-decision robustness cells -- see module docstring)")
    print(f"max timestamp read anywhere in this session: {max_ts_seen}  (< {OOS_START})")

    return dict(cells=cells, primary=primary_cell)


if __name__ == "__main__":
    probe_ok = causality_probe()
    grid_result = run_full_grid()
    if not probe_ok:
        print("\n*** CAUSALITY PROBE FAILED -- results above are NOT trustworthy. ***",
              file=sys.stderr)
