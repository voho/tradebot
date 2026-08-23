#!/usr/bin/env python
"""R-99 CONSERVATIVE branch: Step-A detection-lag gate for a causal rolling
Generalized Hurst Exponent (GHE) scaling-law estimator fit to BTC's own
daily log-price series, single-indicator, run BEFORE any strategy/
confirming-vote code -- identical "operator measurement" convention as
R-82/R-83/R-84/R-85/R-86/R-96/R-98's own gate files, and the same
pre-registered Step-A gate methodology, for direct comparability.

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed, 2026-08-23). If anything below is later contradicted by what
actually happened, that is stated in the results section, not edited back
into this banner.
=====================================================================

1. MECHANISM (one sentence). See `r99_shared.py`'s module docstring for the
   full citation trail (Hurst 1951; Mandelbrot & Van Ness 1968; Barabasi &
   Vicsek 1991; Di Matteo 2007; Bariviera 2017 arXiv:1709.08090; Takaishi
   2018 / arXiv:1804.05916 -- the last two confirming this exact rolling
   Hurst-exponent machinery is applied to BTC specifically) and the full
   not-a-duplicate-of case against the eight prior regime-timing mechanisms
   (R-01 HMM, R-82 BOCPD, R-83 Kalman LLT, R-84 vote-latch modulation, R-85
   CSD, R-86 transfer entropy, R-96 Hawkes, R-98 GPD/POT) and the fourteen
   INFO-axis rounds. One sentence: does the causal rolling-GHE(q=1)
   scaling-exponent z-score (`r99_shared.ghe_signal_zscore` applied to
   `r99_shared.rolling_ghe_signal(r99_shared.daily_log_prices(bars), ...)`,
   then `r99_shared.align_daily_causal` onto the 5-minute bars) crossing UP
   through `Z_THRESH=2.0` (`r99_shared.nearest_alarm`), alarm BEFORE
   `anchor_majority`'s own nearest downward transition (this file's local
   `nearest_transition(..., direction="down")` -- see note below), on the
   identical six dated historical BTC regime transitions R-82/R-83/R-84/
   R-85/R-86/R-96/R-98 used. This branch reads no data beyond the already-
   committed BTC OHLCV close series `kelly_regime_v4` itself already uses --
   no new external data, no new coverage-gap risk.

   NOTE on `nearest_transition`: this helper is NOT present in
   `experiments/r99_shared.py` (the frozen shared module for this round
   only ships `nearest_alarm`, the GHE-side detector). Per the task
   instructions, it is copied here LOCALLY, byte-for-byte identical to
   `r98_shared.py`'s (and r82-r96_shared.py's) own `nearest_transition`,
   rather than editing the frozen shared module. See the copy below.

2. PRIMARY PRE-REGISTERED CONFIGURATION: `PRIMARY_FIT_WINDOW_DAYS=180` --
   chosen and FROZEN by `r99_shared.py` (operator-authored, not this file)
   BEFORE this file ever ran, as the grid's centre cell (3/6/12-month
   trailing window), matching R-85/86/96/98's own BASELINE_WINDOW_DAYS
   convention as closely as GHE's own grid allows -- see `r99_shared.py`'s
   docstring for the full account, including the Kill Switch A
   non-degeneracy check that may override it (run by the operator before
   any episode-level number; if it does, the override and reason are
   recorded in the results section below, not silently substituted here).
   This file does NOT re-pick the primary cell; it is read verbatim from
   `r99_shared.PRIMARY_FIT_WINDOW_DAYS`. The full `FIT_WINDOW_DAYS_GRID`
   (3 cells: 90/180/365 days) is computed and reported for plateau/
   robustness context, exactly mirroring R-96/R-98's own "primary cell
   decides, the other cells are context only" convention -- never used to
   override the primary cell's own pass/fail verdict, precisely so the
   decision is not "pick whichever grid cell scores best" (an undisclosed
   multiple-comparisons search this project's routine treats as p-hacking).
   Unlike R-98's GPD branch (a 3x3 = 9-cell grid: threshold quantile x fit
   window), GHE has a single swept axis here -- `LAG_GRID_DAYS` is a fixed
   structural constant in `r99_shared.py` (not swept), so the grid is 1x3.

3. DETECTION-LAG DEFINITION. IDENTICAL construction to
   `r98_conservative_gpd_alarm.py`'s `gate()`/`null_leads()` (itself
   identical to R-82 through R-96's own chain) -- for each episode, within
   a +/-60-day search window around its onset (`r99_shared.episode_window`,
   `WINDOW_DAYS=60`):
   - v4's own reaction: the nearest DOWNWARD transition of `anchor_majority`
     to the onset (local `nearest_transition`, `direction="down"`).
   - GHE's reaction: the nearest bar where the GHE scaling-exponent z-score
     crosses UP through `Z_THRESH=2.0` (`r99_shared.nearest_alarm`), closest
     to the onset.
   - LEAD = (v4_flip_time - detect_time) in days. Positive = the GHE alarm
     fired before v4's own gate reacted.

4. NULL. `r99_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) GHE z-score series (block_days=5, n_draws=500,
   seed=9902 -- this round's own conservative-branch seed, fixed now,
   disclosed, never altered after seeing results) and recomputes "nearest
   alarm to the real, unshifted v4 flip time" against each shifted copy --
   adapted from R-98's `null_leads` (itself adapted from R-82 through
   R-96's own chain); logic is otherwise identical.

5. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82/R-83/R-84/R-85/R-86/R-96/R-98): an
   episode counts as a PASS if BOTH (a) LEAD >= 0, AND (b) the true LEAD is
   >= the null distribution's median. Using the PRIMARY CONFIG ONLY
   (fit_window_days=180), PROCEED TO STEP B (build the confirming-vote
   strategy) only if >= 4 of the 6 episodes PASS. If fewer than 4 pass on
   the primary config: STOP, report this file's result as the whole
   conservative branch's product, write it up as NEGATIVE, do not build any
   strategy/confirming-vote code, do not touch any data on or after
   2023-01-01. The bar is not relaxed, narrowed, or otherwise adjusted after
   seeing the numbers, and the other 2 grid cells never override this
   verdict.

6. WHAT WOULD MAKE THIS GATE FAIL, named now (copied/adapted from
   `r99_shared.py`'s own "WHAT WOULD MAKE THIS FAIL" section, and named
   there as this round's own pre-registered EXPECTATION, not a hoped-for
   result): THE SAME PATTERN THAT HAS BEATEN EIGHT CONSECUTIVE PRIOR
   MECHANISMS BUILT ON EIGHT DIFFERENT THEORETICAL BASES IS EXPECTED TO
   REPEAT -- an estimator computed FROM price (here: the scaling exponent
   of recent log-price increments) can only shift once a move has already
   happened, which is exactly the moment v4's own fixed-window anchor is
   also starting to react. If GHE also lags every sudden 2020-2022 shock
   and only (at best) leads the slow 2018 build-up, that is the ninth
   independent mechanism converging on the same conclusion the ledger's
   standing diagnosis already leans toward: this six-episode gate is
   unwinnable by any estimator computed from this project's own committed
   price history, whatever field it is drawn from. This is stated as the
   modal expectation BEFORE any real-data number exists in this file,
   precisely so a disappointing result cannot later be read as a surprise
   that justifies loosening the stop rule. Two distinguishable failure
   modes are named up front so the results section can say which one
   occurred, rather than reporting a bare "failed": (i) the alarm NEVER
   crosses `Z_THRESH` inside a given episode's +/-60-day window at all
   (no detection event exists to compare), vs (ii) the alarm DOES cross
   inside the window, but later than v4's own reaction (a genuine but
   losing race).

7. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step A's
   numbers exist; only executed if Step A's primary cell clears >=4/6).

   CONFIRMING-VOTE CONSTRUCTION: `confirming_vote_frac` (imported unchanged
   from `r99_shared.py`) requires a DISCRETE {0,1} `meta_vote` (R-80's
   lesson, reused verbatim by R-88/R-94/R-95/R-98). GHE carries no
   direction of its own (same property GPD tail shape had, R-98), so the
   vote's direction on any "confirmed" (alarm-firing) bar comes from v4's
   own FASTEST (20-day) anchor vote, identical construction to R-94/R-95/
   R-98's `compute_meta_vote`: `meta_vote[i] = fast_anchor_vote[i]` on any
   bar where `ghe_z[i] >= Z_THRESH` ("confirmed"); otherwise
   `meta_vote[i] = meta_vote[i-1]` (carry forward, R-53/54/55's
   hysteresis-latch pattern). Before the first confirmed bar, defaults to
   the fast anchor's own then-current value (no dilution while
   unconfirmed, R-84's convention).

       frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
            = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity check,
   run first, before any swept configuration).

   WEIGHT SWEEP GRID (fixed a priori, disclosed now): weight in
   {0.5, 1.0, 2.0, 4.0} -- 4 values, matching R-88/R-94/R-95/R-98's own
   count for a like-for-like comparison, at the Step-A primary cell's own
   `fit_window_days`.

   MANDATORY CHECKS: (i) identity check (weight=0 vs. kelly_regime_v4,
   bit-for-bit); (ii) causal-truncation probe on the Step-A GHE z-score
   construction itself, at 2-3 different `check_at` values; (iii)
   inner-train / inner-validation evaluation vs. `kelly_regime_v4` and
   `buy_and_hold` on both spot and BTC futures 5x via `scripts.experiment.
   ev`.

   PRE-REGISTERED PROMOTION-CANDIDATE BAR (only reached if everything above
   clears, per `docs/ROUTINE.md` step 4's standard promotion bar, applied
   here to the INNER split only -- never the true 2023+ holdout): beats
   `kelly_regime_v4`/`buy_and_hold` on inner-validation beyond the +/-0.2
   Sharpe noise floor, OR a clear drawdown/tail improvement at matched
   exposure; AND the weight-grid neighbourhood is a plateau, not a peak
   (report all 4 swept weights, not just the winner). If ANY fail: report
   NEGATIVE. Nothing in this file is authorized to read the holdout under
   any outcome.

8. CONFIGS EVALUATED IN STEP A: 0 for the backtest-trials ledger (3 grid
   cells x 6 episodes = 18 gate diagnostics, no `ev()` calls -- identical
   accounting convention to every prior Step-A gate in this project, R-73
   through R-98). Step B's count, if reached: itemized in the results
   section below (identity check + causality probe checkpoints + baselines
   x 2 splits x 2 markets + weight sweep x 2 splits x 2 markets).
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
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r99_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    FIT_WINDOW_DAYS_GRID,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    PRIMARY_FIT_WINDOW_DAYS,
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
    daily_log_prices,
    episode_window,
    ghe_signal_zscore,
    nearest_alarm,
    rolling_ghe_signal,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 9902

WEIGHT_GRID = (0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------- local
#   `nearest_transition` is not exported by r99_shared.py (only the GHE-side
#   `nearest_alarm` is). Copied here byte-for-byte from r98_shared.py (itself
#   identical to r82-r96_shared.py's own helper) per the task instructions,
#   rather than editing the frozen shared module.

def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    vals = series.reindex(window).to_numpy()
    changed = np.zeros(len(vals), dtype=bool)
    if direction == "down":
        changed[1:] = vals[1:] < vals[:-1]
    elif direction == "any":
        changed[1:] = vals[1:] != vals[:-1]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    idx = np.where(changed)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


# --------------------------------------------------------------------- data

def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def build_ghe_z_aligned(bars: pd.DataFrame, fit_window_days: int,
                         log_prices: pd.Series | None = None) -> pd.DataFrame:
    """Full signal-building pipeline: daily log close -> causal rolling
    GHE(q=1) fit -> causal z-score of the GHE series -> causally aligned
    onto `bars`' 5-minute index. Returns a 1-column DataFrame (`ghe_z`)
    indexed like `bars`. `log_prices` may be supplied precomputed (it does
    not depend on `fit_window_days`) to avoid recomputing it once per grid
    cell."""
    if log_prices is None:
        log_prices = daily_log_prices(bars)
    ghe = rolling_ghe_signal(log_prices, fit_window_days)
    z = ghe_signal_zscore(ghe)
    aligned = align_daily_causal(pd.DataFrame({"ghe_z": z}), bars)
    return aligned


def null_leads(ghe_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r98_conservative's `null_leads` (itself adapted from
    r82 through r96's own chain): same circular block-shift construction,
    applied here to the GHE scaling-exponent z-score threshold-crossing
    detector."""
    local = ghe_z.reindex(window).to_numpy()
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


# ----------------------------------------------------------------- step A

def gate(fit_window_days: int, bars: pd.DataFrame, majority: pd.Series,
         log_prices: pd.Series) -> dict:
    is_primary = fit_window_days == PRIMARY_FIT_WINDOW_DAYS
    tag = " [PRIMARY]" if is_primary else ""
    print("=" * 78)
    print(f"R-99 CONSERVATIVE: GHE(fit_window={fit_window_days}d) vs v4 anchor "
          f"-- STEP A detection-lag gate{tag}")
    print("=" * 78)

    signal = build_ghe_z_aligned(bars, fit_window_days, log_prices=log_prices)
    assert_no_holdout(signal)
    ghe_z = signal["ghe_z"]

    print(f"\nz_thresh={Z_THRESH}  search window=+/-{WINDOW_DAYS}d  "
          f"null: {N_DRAWS} draws, block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, onset=onset_str, pass_b=False, lead=float("nan"),
                                 failure_mode="no coverage"))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_alarm(ghe_z, window, onset, Z_THRESH)

        if flip_time is None or detect_time is None:
            mode = "no anchor-gate transition" if flip_time is None else "alarm never crosses inside window"
            print(f"[{label}] onset={onset_str}: {mode} found in +/-{WINDOW_DAYS}d window. "
                  f"FAIL by construction.")
            results.append(dict(label=label, onset=onset_str,
                                 flip=flip_time, detect=detect_time,
                                 pass_b=False, lead=float("nan"), null_median=float("nan"),
                                 failure_mode=mode))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(ghe_z, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)
        failure_mode = None if pass_b else ("alarm crosses but lags v4's own reaction" if lead < 0
                                             else "alarm crosses, leads v4, but not vs. null")

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    GHE(fw={fit_window_days}d) nearest alarm (z>={Z_THRESH}): {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             detect=detect_time, lead=lead, null_median=null_median,
                             pass_b=pass_b, failure_mode=failure_mode))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 4

    print("\n" + "-" * 78)
    for r in results:
        print(f"  {r['label']:42s} lead={r.get('lead', float('nan')):+.2f}d  PASS={r['pass_b']}"
              f"  {r.get('failure_mode') or ''}")
    print(f"\nEpisodes passing: {n_pass}/6  (fit_window_days={fit_window_days}){tag}")
    print(f"max timestamp read anywhere for this cell: "
          f"{max(bars.index.max(), signal.index.max())}  (< {OOS_START})")

    return dict(fit_window_days=fit_window_days, is_primary=is_primary,
                results=results, n_pass=n_pass, passed=passed, signal=signal)


def causality_probe() -> bool:
    """Independent re-verification of the causal truncation claim for this
    file's own signal-building pipeline (`build_ghe_z_aligned`), run at the
    PRIMARY config, at 2-3 different `check_at` points spread across the
    pre-holdout history so the check exercises real, non-degenerate signal
    values in multiple episodes (this project's rule that every round
    re-runs the probe itself rather than trusting a prior claim)."""
    bars = load_btc_bars()
    check_points = ("2018-06-01", "2020-06-01", "2022-06-01")

    def build(df: pd.DataFrame) -> np.ndarray:
        return build_ghe_z_aligned(df, PRIMARY_FIT_WINDOW_DAYS)["ghe_z"].to_numpy()

    all_ok = True
    for ts_str in check_points:
        check_at = bars.index.get_indexer(
            [pd.Timestamp(ts_str, tz="UTC")], method="nearest")[0]
        ok = truncation_causality_probe(build, bars, check_at, shorter_by=40_000)
        print(f"causal truncation probe (ghe_z, fit_window={PRIMARY_FIT_WINDOW_DAYS}d, "
              f"check_at index {check_at} ~ {bars.index[check_at]}): "
              f"{'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def run_full_grid() -> dict:
    bars = load_btc_bars()
    majority = anchor_majority(bars)
    log_prices = daily_log_prices(bars)
    assert_no_holdout(log_prices)

    cells = []
    primary_cell = None
    max_ts_seen = bars.index.max()
    for fit_window_days in FIT_WINDOW_DAYS_GRID:
        cell = gate(fit_window_days, bars, majority, log_prices)
        cells.append(cell)
        if cell["is_primary"]:
            primary_cell = cell
        max_ts_seen = max(max_ts_seen, cell["signal"].index.max())

    assert primary_cell is not None, (
        f"primary config (fit_window={PRIMARY_FIT_WINDOW_DAYS}) missing from grid")

    print("\n" + "=" * 78)
    print("R-99 CONSERVATIVE: FULL 1x3 GRID SUMMARY (GHE(fit_window) vs v4 anchor)")
    print("=" * 78)
    print(f"{'fit_window_days':>16} {'n_pass/6':>10}  {'primary?':>9}")
    for cell in cells:
        marker = "<-- PRIMARY" if cell["is_primary"] else ""
        print(f"{cell['fit_window_days']:>16} {cell['n_pass']:>8}/6  {marker}")

    print(f"\nPRIMARY CONFIG (fit_window_days={PRIMARY_FIT_WINDOW_DAYS}): "
          f"{primary_cell['n_pass']}/6 episodes pass")
    print(f"GATE VERDICT (primary config only, per pre-registered stop rule): "
          f"{'PASS -> proceed to Step B (build confirming-vote strategy)' if primary_cell['passed'] else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file's Step A toward the backtest "
          f"trials ledger: 0 (fixed measurement gate; 3 grid cells x 6 episodes = "
          f"18 episode-lead measurements, no ev() calls -- see module docstring)")
    print(f"max timestamp read anywhere in this session so far: {max_ts_seen}  (< {OOS_START})")

    return dict(cells=cells, primary=primary_cell)


# ==========================================================================
# STEP B -- built only if the primary cell above passes (banner item 7).
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, ghe_z: pd.Series,
                       horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """GHE-alarm-confirmed latch on the FASTEST anchor's own 0/1 vote
    (banner item 7): `meta_vote[i] = fast_anchor_vote[i]` on any bar where
    `ghe_z[i] >= Z_THRESH` ("confirmed"); otherwise `meta_vote[i] =
    meta_vote[i-1]` (carries forward). Before the first confirmed bar,
    defaults to the fast anchor's own then-current value. Causal: `ghe_z`
    and each anchor vote are both causal; the latch update at i depends
    only on values at <= i."""
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    z = ghe_z.reindex(df.index).to_numpy()
    confirmed = z >= Z_THRESH

    n = len(df)
    meta = np.empty(n)
    meta[0] = fast_vote[0]
    for i in range(1, n):
        meta[i] = fast_vote[i] if confirmed[i] else meta[i - 1]
    return meta


class GheConfirmKelly(Strategy):
    """kelly_regime_v4 + a GHE-scaling-alarm-confirmed fast-anchor vote
    (R-99 conservative, unregistered). Structurally v3/v4's own prepare(),
    with the plain 3-anchor average `frac = anchor_sum/3` replaced by
    `confirming_vote_frac(anchor_sum, meta_vote, weight)`. `weight=0` must
    recover v4 bit-for-bit. Modelled directly on
    `r98_conservative_gpd_alarm.py`'s `GpdConfirmKelly`. Not `@register`ed
    -- stays in experiments/ per docs/ROUTINE.md."""

    name = "r99_conservative_ghe_confirm"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, ghe_z: pd.Series, weight: float = 1.0,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.ghe_z = ghe_z
        self.weight = weight
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

        z = self.ghe_z.reindex(df.index)
        meta_vote = compute_meta_vote(df, z, horizons=self.horizons, band=self.band)
        frac = confirming_vote_frac(anchor_sum, meta_vote, self.weight)

        # Identical conditional-volatility-targeting scale to kelly_regime_v3/_v4.
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

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
        for i in range(n):
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

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


def build_target_primary(df: pd.DataFrame, ghe_z: pd.Series) -> np.ndarray:
    """Target-construction function for a Step-B causal truncation spot
    check, frozen at the pre-registered primary candidate (weight=1.0)."""
    return GheConfirmKelly(ghe_z=ghe_z, weight=1.0).prepare(df.copy())["target"].to_numpy()


def run_identity_check(df_full: pd.DataFrame, ghe_z: pd.Series) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = GheConfirmKelly(ghe_z=ghe_z, weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def eval_config(ev, SPOT, FUTURES, ghe_z: pd.Series, weight: float, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = GheConfirmKelly(ghe_z=ghe_z, weight=weight)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES, ghe_z: pd.Series) -> dict:
    results = {}
    for weight in WEIGHT_GRID:
        tag = f"w{weight}"
        results[weight] = eval_config(ev, SPOT, FUTURES, ghe_z, weight, tag)
    return results


def run_step_b(gate_result: dict) -> None:
    from scripts.experiment import DF, FUTURES, SPOT, OOS_START as EXP_OOS_START, ev

    assert EXP_OOS_START == OOS_START, (
        f"scripts.experiment.OOS_START ({EXP_OOS_START}) != r99_shared.OOS_START "
        f"({OOS_START}) -- refusing to proceed with a mismatched holdout boundary")

    print("\n" + "=" * 78)
    print(f"STEP B (primary cell passed): weight sweep + mandatory checks -- "
          f"primary cell fit_window_days={PRIMARY_FIT_WINDOW_DAYS}")
    print("=" * 78)

    df_trunc = DF.loc[DF.index < pd.Timestamp(OOS_START, tz=DF.index.tz)]
    log_prices = daily_log_prices(df_trunc)
    ghe_z_signal = build_ghe_z_aligned(df_trunc, PRIMARY_FIT_WINDOW_DAYS,
                                        log_prices=log_prices)["ghe_z"]
    assert_no_holdout(ghe_z_signal)
    # Reindex onto the full DF's index (train+val only will be used by ev()
    # via start/end, but prepare() is called on whatever frame ev() passes
    # it, so the Series must cover the full pre-OOS range DF itself does).
    ghe_z_full = ghe_z_signal.reindex(DF.index)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(DF, ghe_z_full)
    n_configs += 1

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=INNER_TRAIN_END)),
                                ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)
                n_configs += 1

    print(f"\n=== weight sweep: {WEIGHT_GRID} ===")
    run_sweep(ev, SPOT, FUTURES, ghe_z_full)
    n_configs += len(WEIGHT_GRID) * 4  # 2 splits x 2 markets per weight

    print(f"\nTotal Step B configurations evaluated: {n_configs}")


# ------------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    probe_ok = causality_probe()
    grid_result = run_full_grid()
    if not probe_ok:
        print("\n*** CAUSALITY PROBE FAILED -- results above are NOT trustworthy. ***",
              file=sys.stderr)
        return
    if grid_result["primary"]["passed"]:
        run_step_b(grid_result)
    else:
        print("\nSTEP A FAILED the pre-registered stop rule (primary cell < 4/6). "
              "Per this file's own pre-registration, no strategy is built and no "
              "Step-B code runs. This gate result is this branch's whole product.")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmds = {"gate": run_full_grid, "main": main}
    choice = sys.argv[1] if len(sys.argv) > 1 else "main"
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/r99_conservative_ghe_alarm.py [{'|'.join(cmds)}]")
