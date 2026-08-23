#!/usr/bin/env python
"""R-96 CONSERVATIVE branch: self-exciting Hawkes point process intensity
z-score (`hawkes_intensity_zscore`) as a single-indicator REGIME-TIMING
ALARM against the same six dated historical BTC regime transitions R-82
(BOCPD), R-83 (Kalman LLT), R-84 (vote-latch modulation), R-85 (critical
slowing down / variance) and R-86 (transfer entropy) were tested against --
Step-A measurement gate ONLY, run BEFORE any strategy/confirming-vote code,
identical "operator measurement" convention as those files' own gates.

=====================================================================
PRE-REGISTRATION (frozen before any real-market Hawkes number in this file
was computed -- docs/ROUTINE.md steps 1-2/4). If anything below is later
contradicted by what actually happened, that is stated in the results
section, not edited back into this banner.
=====================================================================

1. MECHANISM. See `r96_shared.py`'s module docstring for the full citation
   trail (Hawkes 1971; Bacry, Mastromatteo & Muzy 2015; Barndorff-Nielsen &
   Shephard 2004/2006 for the event-time filter) and not-a-duplicate-of
   argument against R-01/R-82/R-83/R-84/R-85/R-86/R-88/R-77/R-93/the
   thirteen INFO-axis rounds/R-62. One sentence: a self-exciting point
   process fit to BTC's own daily jump-event timing (Huang-Tauchen /
   Andersen-Bollerslev-Diebold relative-jump statistic computed from this
   project's own 5-minute bars) gives a conditional intensity `lambda(t)`
   that should rise sharply once a jump cluster starts, and this branch
   tests whether that rise, converted to a causal z-score against its own
   trailing baseline (`hawkes_intensity_zscore`), crosses a fixed threshold
   with LESS lag than v4's own fixed 20/40/80-day anchor-crossing
   heuristic, on the identical six dated historical BTC regime transitions
   R-82/83/84/85/86 used. This branch reads no data beyond the
   already-committed BTC OHLCV close series `kelly_regime_v4` itself
   already consumes.

2. STEP A -- THE MANDATORY MEASUREMENT GATE, run BEFORE any strategy code,
   swept over `r96_shared.py`'s pre-registered 3x3 grid
   (`N_GRID=(0.3,0.5,0.7)` branching ratio x `HALFLIFE_DAYS_GRID=(3,7,14)`
   decay half-life days -- 9 candidate Hawkes parameterizations, all fixed
   a priori in `r96_shared.py` before any real-market number existed).

   PRIMARY CELL (chosen now, mirroring R-85/R-86's own primary-cell
   convention of picking the middle of a small a-priori grid): `n=0.5`,
   `halflife_days=7` -- the middle branching ratio and middle decay horizon
   of the grid. The other 8 cells are reported as a robustness/sensitivity
   sweep, not searched over for the best result; the pre-registered stop
   rule below applies to the primary cell only.

   ALARM DEFINITION: `hawkes_intensity_zscore` crosses UP through
   `r96_shared.Z_THRESH=2.0` (`r96_shared.nearest_hawkes_alarm`), closest
   to the episode onset within a +/-60-day search window
   (`r96_shared.episode_window`, `WINDOW_DAYS=60`, identical window R-82/
   83/85/86 used).

   ANCHOR-GATE "FLIP" DEFINITION: the nearest DOWNWARD transition of
   `anchor_majority` to the onset (`r96_shared.nearest_transition`,
   `direction="down"`), identical rule R-82/83/85/86 used.

   LEAD = (v4_flip_time - hawkes_alarm_time) in days. Positive = the
   Hawkes alarm crossed before v4's own gate reacted.

   NULL: `r96_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) Hawkes z-score array (`block_days=5`,
   `n_draws=500`, `seed=9601`, fixed before running and never altered
   afterward -- following the `85*100+1`/`86*100+1` seed-naming convention
   R-85/R-86 used) and recomputes "nearest alarm crossing to the real,
   unshifted v4 flip time" against each shifted copy. The SAME 500 shift
   draws (same seed, same per-episode window length) are reused across all
   9 grid cells for a given episode, so the 9 cells are compared against
   directly comparable null draws, not independently reseeded ones.

   PRE-REGISTERED STOP RULE, decision-rule ambiguity DISCLOSED HERE, before
   running, not after seeing a number (see deviation note in the results
   section for how this was resolved): `r96_shared.py`'s own docstring
   states the six-episode gate scaffolding is "IDENTICAL...copied verbatim"
   from R-82/83/84/85/86, whose own gate FILES (r85_conservative_csd_
   variance.py, r86_conservative_te_volume_return.py) use "lead >= 0 AND
   lead >= the null distribution's MEDIAN" as the per-episode pass rule.
   R-88's conservative gate (a related but not-identical precedent -- a
   3-episode table, not this round's 6-episode one) instead uses "lead > 0
   AND lead > the null's 90th PERCENTILE". Both conventions are computed
   and reported per episode per cell in this file; the PRIMARY, pre-
   registered pass/fail table uses the MEDIAN convention, because
   `r96_shared.py` explicitly ties this round's episode-gate scaffolding to
   R-82/83/84/85/86 (the 6-episode, median-rule family) rather than to R-88
   (the 3-episode, p90-rule family) for direct numeric comparability across
   all six-plus-one rounds on the identical table. The stricter p90
   convention is reported alongside as a sensitivity check, never as a
   second chance to pass the gate.

   PROCEED TO STEP B (build the confirming-vote strategy) only if >= 4 of
   the 6 episodes PASS (median convention) on the PRIMARY cell (n=0.5,
   halflife=7). If fewer than 4 pass: STOP, report this file's result as
   the whole branch's product, write it up as NEGATIVE, do not build any
   strategy/confirming-vote code, do not touch any data on or after
   2023-01-01. The bar is not relaxed, narrowed, or otherwise adjusted
   after seeing the numbers, and is not swapped to whichever of the 9
   cells happens to score highest.

3. WHAT WOULD MAKE STEP A FAIL, named now (and named in `r96_shared.py` as
   this round's own pre-registered EXPECTATION, not a hoped-for result):
   the same pattern that has now beaten six consecutive mechanisms built on
   six different theoretical bases -- an estimator computed FROM price
   (here: jump event timing) can only rise once a jump has already
   happened, which is exactly the moment v4's own fixed-window anchor is
   also starting to react. If Hawkes intensity also lags every sudden
   2020-2022 shock and only (at best) leads the slow 2018 build-up, that is
   the seventh independent mechanism converging on the same conclusion the
   ledger's standing diagnosis already leans toward.

4. CONFIGS EVALUATED IN STEP A: 0 in the strategy-evaluation sense (a
   fixed, non-swept MEASUREMENT gate over 9 pre-registered Hawkes
   parameterizations -- no strategy backtest, no `ev()` call, no
   optimization; this project's standing accounting convention for this
   exact construction, R-82/83/85/86's own Step-A studies). The 9-cell
   grid is a robustness report, not a search: the stop rule is evaluated
   on the primary cell alone, so no "best of 9" selection occurs. Step B's
   count, if reached, is itemized in that section's own pre-registration
   below.

5. HOLDOUT DISCIPLINE. This file asserts, at every load/build point, that
   no bar with timestamp >= OOS_START (2023-01-01) is ever read
   (`r96_shared.assert_no_holdout`).

6. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step
   A's numbers exist; only executed if Step A's stop rule passes on the
   primary cell).

   CONFIRMING-VOTE CONSTRUCTION, reusing `r96_shared.confirming_vote_frac`
   exactly as R-88's `TakerFlowConfirmKelly` does: `meta_vote[i] = 1` on any
   bar where the primary cell's Hawkes z-score `>= Z_THRESH` ("alarm
   confirmed"), otherwise `meta_vote` carries forward its last confirmed
   value (a latch); before the first confirmation, `meta_vote` tracks the
   fastest (20-day) anchor's own then-current vote each bar (R-84/R-88's
   convention). `frac = confirming_vote_frac(anchor_sum, meta_vote,
   weight)`. `weight=0` must recover `kelly_regime_v4` bit-for-bit
   (identity check, run first).

   SWEEP GRID (fixed a priori): `weight` in {0.5, 1.0, 2.0, 4.0} x the
   alarm's own trailing windows, i.e. `(n, halflife_days)` in
   {(0.3,7), (0.5,7), (0.7,7), (0.5,3), (0.5,14)} (the primary cell plus
   its four immediate grid neighbours -- a plateau check, not a full 3x3
   re-sweep, matching R-84/R-88's "bracket the primary 2x on each side"
   convention) -- 20 configurations, evaluated on inner-train (end=
   2020-12-31) and inner-validation (2021-01-01 to 2022-12-31), spot and
   futures. Identity check: 1 configuration. Total Step B configurations,
   if reached: 21, plus the ETH falsification run.

   FALSIFICATION TEST: run the frozen primary config (weight=1.0, n=0.5,
   halflife=7) on ETH over its full pre-holdout coverage, and check that
   the same qualitative sign/direction of edge over `kelly_regime_v4`
   replicates.

   MANDATORY: the causal-truncation probe
   (`r96_shared.truncation_causality_probe`) is run on the frozen primary
   candidate's constructed target series regardless of Step A's verdict.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r96_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    HALFLIFE_DAYS_GRID,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    N_GRID,
    OOS_START,
    STRESS_EPISODES,
    V4_BAND,
    V4_HORIZONS,
    Z_THRESH,
    align_daily_causal,
    anchor_majority,
    anchor_votes,
    assert_no_holdout,
    block_bootstrap_shifts,
    confirming_vote_frac,
    episode_window,
    hawkes_intensity_daily,
    hawkes_intensity_zscore,
    intraday_relative_jump,
    nearest_hawkes_alarm,
    nearest_transition,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

# ---- Step A constants, fixed a priori (see banner item 2) -----------------
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 9601
MIN_EPISODES_PASS = 4          # of 6, matching R-82/83/85/86's bar
PRIMARY_N = 0.5
PRIMARY_HALFLIFE = 7


# --------------------------------------------------------------------- data

def load_btc_bars() -> pd.DataFrame:
    """BTC spot, truncated strictly before OOS_START at load time."""
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


# ------------------------------------------------------------------ Hawkes z

def build_hawkes_z(df: pd.DataFrame, n: float, halflife_days: float) -> pd.Series:
    """Full pipeline: intraday relative-jump event flags -> causal Hawkes
    intensity -> causal z-score -> aligned onto `df`'s 5-minute index. All
    three stages are `r96_shared.py` utilities, reused verbatim."""
    flag = intraday_relative_jump(df)
    assert_no_holdout(flag)
    lam = hawkes_intensity_daily(flag, n, halflife_days)
    z = hawkes_intensity_zscore(lam)
    aligned = align_daily_causal(z, df)
    assert_no_holdout(aligned)
    return aligned


# -------------------------------------------------------------- null (batch)

def null_leads_batch(z_local: np.ndarray, shift_matrix: np.ndarray,
                      window: pd.DatetimeIndex, onset: pd.Timestamp,
                      flip_time: pd.Timestamp, z_thresh: float = Z_THRESH) -> np.ndarray:
    """Vectorized form of the r85/r86/r88 `null_leads` pattern: circularly
    block-shift the LOCAL z-score array and recompute "nearest alarm
    crossing to the real, unshifted onset" against the fixed, real
    `flip_time`, for every draw at once. Mathematically identical per-draw
    logic to `nearest_hawkes_alarm` (up-cross through `z_thresh`, first bar
    counts as a cross too) -- vectorized only because this file evaluates
    9 grid cells x 6 episodes x 500 draws, where a plain Python loop over
    every draw (r85/r86/r88's convention, fine for their single-indicator
    gates) becomes materially slower.
    """
    shifted = z_local[shift_matrix]                       # (n_draws, n_bars)
    high = shifted >= z_thresh
    cross = np.zeros_like(high)
    cross[:, 1:] = high[:, 1:] & ~high[:, :-1]
    cross[:, 0] = high[:, 0]
    any_cross = cross.any(axis=1)

    deltas_days = np.abs((window - onset).total_seconds().to_numpy()) / 86400.0
    masked = np.where(cross, deltas_days[None, :], np.inf)
    best_idx = np.argmin(masked, axis=1)

    window_arr = window.to_numpy()
    detect_times = window_arr[best_idx]
    leads = (flip_time.to_numpy() - detect_times) / np.timedelta64(1, "D")
    leads = np.where(any_cross, leads.astype(float), np.nan)
    return leads


# --------------------------------------------------------------------- gate

def gate() -> dict:
    print("=" * 78)
    print("R-96 CONSERVATIVE: Hawkes intensity z-score vs v4 anchor -- STEP A gate")
    print("=" * 78)

    bars = load_btc_bars()
    majority = anchor_majority(bars)
    assert_no_holdout(majority)

    print(f"\nz_thresh={Z_THRESH}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}")
    print(f"grid: n in {N_GRID}  halflife_days in {HALFLIFE_DAYS_GRID}  "
          f"(primary cell: n={PRIMARY_N}, halflife={PRIMARY_HALFLIFE})\n")

    # Precompute, once per episode, the onset/window/flip and the shared
    # block-bootstrap shift matrix -- reused identically across all 9 grid
    # cells, so cells are compared against directly comparable null draws.
    episode_info = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            episode_info.append(dict(label=label, onset_str=onset_str, onset=onset,
                                      window=window, flip_time=None, shift_matrix=None))
            continue
        flip_time = nearest_transition(majority, window, onset, direction="down")
        n_bars = len(window)
        shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=BLOCK_DAYS,
                                         n_draws=N_DRAWS, seed=NULL_SEED)
        shift_matrix = np.array(shifts)
        episode_info.append(dict(label=label, onset_str=onset_str, onset=onset,
                                  window=window, flip_time=flip_time,
                                  shift_matrix=shift_matrix))

    all_cells: dict[tuple[float, float], list[dict]] = {}
    z_cache: dict[tuple[float, float], pd.Series] = {}

    for n in N_GRID:
        for halflife in HALFLIFE_DAYS_GRID:
            z = build_hawkes_z(bars, n, halflife)
            z_cache[(n, halflife)] = z
            cell_results = []
            for info in episode_info:
                label, onset_str = info["label"], info["onset_str"]
                onset, window, flip_time = info["onset"], info["window"], info["flip_time"]
                if len(window) == 0:
                    cell_results.append(dict(label=label, onset=onset_str, lead=float("nan"),
                                              pass_median=False, pass_p90=False,
                                              reason="no bars in window"))
                    continue
                detect_time = nearest_hawkes_alarm(z, window, onset, Z_THRESH)
                if flip_time is None or detect_time is None:
                    cell_results.append(dict(
                        label=label, onset=onset_str, lead=float("nan"),
                        pass_median=False, pass_p90=False,
                        reason=("no anchor-gate transition" if flip_time is None
                                else "no hawkes alarm")))
                    continue

                lead = (flip_time - detect_time).total_seconds() / 86400.0
                z_local = z.reindex(window).to_numpy()
                leads_null = null_leads_batch(z_local, info["shift_matrix"], window,
                                               onset, flip_time, Z_THRESH)
                valid = leads_null[~np.isnan(leads_null)]
                null_median = float(np.median(valid)) if len(valid) else float("nan")
                null_p90 = float(np.percentile(valid, 90)) if len(valid) else float("nan")
                pass_lead0 = lead >= 0
                pass_median = pass_lead0 and (not np.isnan(null_median)) and (lead >= null_median)
                pass_p90 = (lead > 0) and (not np.isnan(null_p90)) and (lead > null_p90)

                cell_results.append(dict(
                    label=label, onset=onset_str, flip=flip_time, detect=detect_time,
                    lead=lead, null_median=null_median, null_p90=null_p90,
                    n_valid_null=int(len(valid)), pass_median=pass_median, pass_p90=pass_p90,
                    reason=None))
            all_cells[(n, halflife)] = cell_results

    # ---- primary-cell detailed report ----
    primary_key = (PRIMARY_N, PRIMARY_HALFLIFE)
    primary_results = all_cells[primary_key]
    print("=" * 78)
    print(f"PRIMARY CELL DETAIL: n={PRIMARY_N}  halflife_days={PRIMARY_HALFLIFE}")
    print("=" * 78)
    for r in primary_results:
        if r.get("reason"):
            print(f"[{r['label']}] onset={r['onset']}: {r['reason']}. FAIL by construction.")
            continue
        print(f"[{r['label']}] onset={r['onset']}")
        print(f"    v4 anchor nearest downward flip: {r['flip']}")
        print(f"    Hawkes z nearest alarm (z>={Z_THRESH}): {r['detect']}")
        print(f"    LEAD = {r['lead']:+.2f}d   null median={r['null_median']:+.2f}d  "
              f"null p90={r['null_p90']:+.2f}d  (valid draws {r['n_valid_null']}/{N_DRAWS})")
        print(f"    PASS (median rule, PRIMARY): {r['pass_median']}   "
              f"PASS (p90 rule, sensitivity): {r['pass_p90']}")

    primary_n_pass_median = sum(1 for r in primary_results if r["pass_median"])
    primary_n_pass_p90 = sum(1 for r in primary_results if r["pass_p90"])
    passed = primary_n_pass_median >= MIN_EPISODES_PASS

    print("\n" + "=" * 78)
    print(f"PRIMARY CELL episodes passing (median rule): {primary_n_pass_median}/6")
    print(f"PRIMARY CELL episodes passing (p90 rule, sensitivity only): {primary_n_pass_p90}/6")
    print(f"GATE VERDICT (primary cell, median rule, pre-registered): "
          f"{'PASS -> proceed to Step B' if passed else 'FAIL -> STOP, no strategy built'}")

    # ---- full 9-cell grid table ----
    print("\n" + "=" * 78)
    print("FULL 3x3 GRID -- pass counts per cell (robustness/sensitivity, not a search)")
    print("=" * 78)
    header = "n\\halflife   " + "  ".join(f"{hl:>10d}d" for hl in HALFLIFE_DAYS_GRID)
    print(header)
    for n in N_GRID:
        row = [f"n={n:<3}      "]
        for halflife in HALFLIFE_DAYS_GRID:
            res = all_cells[(n, halflife)]
            n_med = sum(1 for r in res if r["pass_median"])
            n_p90 = sum(1 for r in res if r["pass_p90"])
            mark = " *PRIMARY*" if (n, halflife) == primary_key else ""
            row.append(f"{n_med}/6 med, {n_p90}/6 p90{mark}")
        print(row[0] + "   " + "   |   ".join(row[1:]))

    print("\nPer-episode pass/fail (median rule), all 9 cells:")
    col_w = 14
    print(f"{'episode':42s}" + "".join(f"n={n},hl={hl:>2d}".rjust(col_w)
                                        for n in N_GRID for hl in HALFLIFE_DAYS_GRID))
    for i, info in enumerate(episode_info):
        row_str = f"{info['label']:42s}"
        for n in N_GRID:
            for halflife in HALFLIFE_DAYS_GRID:
                r = all_cells[(n, halflife)][i]
                mark = "PASS" if r["pass_median"] else "fail"
                lead_str = f"{r['lead']:+.1f}d" if np.isfinite(r.get("lead", np.nan)) else "n/a"
                row_str += f"{(mark + ' ' + lead_str):>{col_w}s}"
        print(row_str)

    print(f"\nconfigurations evaluated in this file's Step A: 0 (fixed measurement gate "
          f"over a pre-registered 9-cell robustness grid; the stop rule is evaluated on "
          f"the primary cell only -- no search over the 9 cells occurs)")
    max_ts = max(bars.index.max(), *(z.index.max() for z in z_cache.values()))
    print(f"max timestamp read anywhere in this session so far: {max_ts}  (< {OOS_START})")

    return dict(all_cells=all_cells, primary_results=primary_results,
                primary_n_pass_median=primary_n_pass_median,
                primary_n_pass_p90=primary_n_pass_p90, passed=passed,
                bars=bars, z_cache=z_cache)


# --------------------------------------------------------------- causality probe

def causality_probe(bars: pd.DataFrame) -> list[bool]:
    """Mandatory causal-truncation probe (task item 4 / banner item 6),
    on the primary cell's constructed Hawkes z-score, at 3 truncation
    points spanning the 2018 / 2020 / 2022 episodes."""
    print("\n" + "=" * 78)
    print("CAUSAL-TRUNCATION PROBE on the primary-cell Hawkes z-score construction")
    print(f"(n={PRIMARY_N}, halflife_days={PRIMARY_HALFLIFE})")
    print("=" * 78)

    def build(d: pd.DataFrame) -> np.ndarray:
        return build_hawkes_z(d, PRIMARY_N, PRIMARY_HALFLIFE).to_numpy()

    probe_points = ["2018-06-01", "2020-06-01", "2022-06-01"]
    results = []
    for ts_str in probe_points:
        check_at = bars.index.get_indexer(
            [pd.Timestamp(ts_str, tz="UTC")], method="nearest")[0]
        ok = truncation_causality_probe(build, bars, check_at, shorter_by=40_000)
        print(f"[causality] check_at={check_at} (~{ts_str}): {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


# ==========================================================================
# STEP B -- built only if the gate above passes on the primary cell.
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, n: float, halflife_days: float,
                       z_thresh: float = Z_THRESH,
                       horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """Signed-alarm confirming vote (R-88's `compute_meta_vote` pattern,
    keyed on the Hawkes z-score alarm instead of taker-flow direction):
    `meta_vote[i] = 1` on any bar where the Hawkes z-score `>= z_thresh`
    ("alarm confirmed"); otherwise carries forward its last confirmed
    value. Before the first confirmation, tracks the fastest (20-day)
    anchor's own then-current vote each bar."""
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    z = build_hawkes_z(df, n, halflife_days).to_numpy()
    alarm = z >= z_thresh

    n_bars = len(df)
    meta = np.empty(n_bars)
    last = fast_vote[0]
    confirmed_ever = False
    for i in range(n_bars):
        if alarm[i]:
            last = 1.0
            confirmed_ever = True
        elif not confirmed_ever:
            last = fast_vote[i]
        meta[i] = last
    return meta


class HawkesAlarmKelly:
    """kelly_regime_v4 + a Hawkes-alarm-confirmed vote (R-96 conservative,
    unregistered). Structurally v3/v4's own prepare(), with the plain
    3-anchor average `frac = anchor_sum/3` replaced by
    `confirming_vote_frac(anchor_sum, meta_vote, weight)`. `weight=0` must
    recover v4 bit-for-bit. Not `@register`ed -- stays in experiments/ per
    docs/ROUTINE.md, identical structure to R-88's `TakerFlowConfirmKelly`.
    """

    name = "r96_conservative_hawkes_alarm_kelly"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, weight: float = 1.0, n: float = PRIMARY_N,
                 halflife_days: float = PRIMARY_HALFLIFE, z_thresh: float = Z_THRESH,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.weight = weight
        self.n = n
        self.halflife_days = halflife_days
        self.z_thresh = z_thresh
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = anchor_votes(df, horizons=self.horizons, band=self.band)
        anchor_sum = sum(v.to_numpy() for v in votes)

        meta_vote = compute_meta_vote(df, self.n, self.halflife_days, self.z_thresh,
                                       horizons=self.horizons, band=self.band)
        frac = confirming_vote_frac(anchor_sum, meta_vote, self.weight)

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                    min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n_bars = len(df)
        target = np.zeros(n_bars)
        pos = 0.0
        state = 0
        for i in range(n_bars):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


def run_identity_check(df_full: pd.DataFrame) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    from tradebot.registry import get_strategy

    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = HawkesAlarmKelly(weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe_step_b(df_full: pd.DataFrame) -> list[bool]:
    df = df_full.loc[:INNER_TRAIN_END].copy()

    def build(d: pd.DataFrame) -> np.ndarray:
        return HawkesAlarmKelly(weight=1.0).prepare(d.copy())["target"].to_numpy()

    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(build, df, check_at)
        print(f"[causality, target series] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, weight: float, n: float, halflife_days: float, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = HawkesAlarmKelly(weight=weight, n=n, halflife_days=halflife_days)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES) -> dict:
    """weight x the alarm's own trailing window `(n, halflife_days)` grid:
    the primary cell plus its four immediate grid neighbours (a plateau
    check, not a full 3x3 re-sweep), per banner item 6."""
    window_grid = [(PRIMARY_N, PRIMARY_HALFLIFE),
                    (0.3, PRIMARY_HALFLIFE), (0.7, PRIMARY_HALFLIFE),
                    (PRIMARY_N, 3), (PRIMARY_N, 14)]
    results = {}
    for weight in (0.5, 1.0, 2.0, 4.0):
        for (n, halflife_days) in window_grid:
            tag = f"w{weight} n{n} hl{halflife_days}"
            results[(weight, n, halflife_days)] = eval_config(
                ev, SPOT, FUTURES, weight, n, halflife_days, tag)
    return results


def run_eth_falsification(ev, weight: float = 1.0) -> dict:
    """ETH falsification (banner item 6): frozen primary config on ETH,
    over ETH's full pre-holdout coverage, strictly inside training."""
    from tradebot.broker import MarketSpec
    from tradebot.data import load_ohlcv_csv
    from tradebot.registry import get_strategy

    spot = MarketSpec.spot()
    eth_bars = load_ohlcv_csv(DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz")
    eth_bars = eth_bars.loc[eth_bars.index < pd.Timestamp(OOS_START, tz=eth_bars.index.tz)].copy()
    assert_no_holdout(eth_bars)
    print(f"ETH bars: {len(eth_bars):,}  {eth_bars.index[0]} -> {eth_bars.index[-1]}")

    v4 = get_strategy("kelly_regime_v4")
    cand = HawkesAlarmKelly(weight=weight)
    out = {}
    m_v4 = ev(v4, df=eth_bars, market=spot, end=INNER_VAL_END, tag="ETH falsification: v4")
    m_cand = ev(cand, df=eth_bars, market=spot, end=INNER_VAL_END, tag="ETH falsification: candidate")
    delta = m_cand.sharpe - m_v4.sharpe
    print(f"  ETH (full -> {INNER_VAL_END}): v4 sharpe={m_v4.sharpe:.2f}  "
          f"candidate sharpe={m_cand.sharpe:.2f}  delta={delta:+.2f}")
    out["ETH"] = dict(v4=m_v4.sharpe, cand=m_cand.sharpe, delta=delta)
    return out


def run_step_b() -> None:
    from scripts.experiment import DF, FUTURES, SPOT, ev
    from tradebot.registry import get_strategy

    print("\n" + "=" * 78)
    print("STEP B (gate passed on primary cell): sweep + mandatory checks")
    print("=" * 78)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(DF)
    n_configs += 1

    print("\n=== causality probe (target series) ===")
    run_causality_probe_step_b(DF)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=INNER_TRAIN_END)),
                                ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)

    print("\n=== sweep ===")
    sweep_results = run_sweep(ev, SPOT, FUTURES)
    n_configs += len(sweep_results)

    print("\n=== ETH falsification ===")
    run_eth_falsification(ev)

    print(f"\nTotal Step B configurations evaluated: {n_configs}")


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    gate_result = gate()
    causality_probe(gate_result["bars"])
    if gate_result["passed"]:
        run_step_b()
    else:
        print("\nSTEP A FAILED the pre-registered stop rule (primary cell, median "
              "convention, >= 4/6). Per this file's own pre-registration, no strategy "
              "is built and no Step-B code runs. This gate result is this branch's "
              "whole product.")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmds = {"gate": gate, "main": main}
    choice = sys.argv[1] if len(sys.argv) > 1 else "main"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r96_conservative_hawkes_alarm.py [{'|'.join(cmds)}]")
