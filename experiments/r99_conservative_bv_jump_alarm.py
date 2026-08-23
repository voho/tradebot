#!/usr/bin/env python
"""R-99 CONSERVATIVE branch: Step-A detection-lag gate for a causal rolling
Barndorff-Nielsen & Shephard bipower-variation jump/continuous decomposition
of BTC's own realized quadratic variation, computed from this project's
NATIVE 5-minute bars, single-indicator, run BEFORE any strategy/confirming-
vote code -- identical "operator measurement" convention as R-82/R-83/R-84/
R-85/R-86/R-96/R-98's own gate files, and the same pre-registered Step-A
gate methodology, for direct comparability.

=====================================================================
PRE-REGISTRATION (frozen before any lead/lag number in this file was
computed, 2026-08-23). If anything below is later contradicted by what
actually happened, that is stated in the results section, not edited back
into this banner.
=====================================================================

1. MECHANISM (one sentence). See `r99_shared.py`'s module docstring for the
   full citation trail (Barndorff-Nielsen & Shephard 2004/2006; Huang &
   Tauchen 2005; Andersen, Bollerslev & Diebold 2007; Shen, Urquhart & Wang
   2020 -- the last confirming this exact jump-detection machinery is
   applied to BTC specifically) and the full not-a-duplicate-of case against
   the eight prior regime-timing mechanisms (R-01 HMM, R-82 BOCPD, R-83
   Kalman LLT, R-84 vote-latch modulation, R-85 CSD, R-86 transfer entropy,
   R-96 Hawkes, R-98 POT/GPD) and the fourteen INFO-axis rounds. One
   sentence: does the causal rolling relative-jump-measure z-score
   (`r99_shared.rj_signal_zscore` applied to `r99_shared.daily_rv_bv_jump
   (...)['rj']`, then `r99_shared.align_daily_causal` onto the 5-minute
   bars) crossing UP through `Z_THRESH=2.0` (`r99_shared.nearest_alarm`),
   alarm BEFORE `anchor_majority`'s own nearest downward transition
   (`r99_shared.nearest_transition(..., direction="down")`), on the
   identical six dated historical BTC regime transitions R-82/R-83/R-84/
   R-85/R-86/R-96/R-98 used. This branch reads no data beyond the already-
   committed BTC OHLCV close series `kelly_regime_v4` itself already uses at
   its own native 5-minute cadence -- no new external data, no new
   coverage-gap risk. Unlike all eight predecessors, this IS the first
   construction on this axis that structurally requires native 5-minute
   bars to exist as a statistic at all (bipower variation is undefined --
   indistinguishable from realized variance -- on daily-resampled closes;
   see `r99_shared.py`'s docstring, "the one genuine methodological first").

2. PRIMARY PRE-REGISTERED CONFIGURATION: `PRIMARY_DETECTION_WINDOW_DAYS=90,
   PRIMARY_BASELINE_WINDOW_DAYS=730` -- chosen and FROZEN by `r99_shared.py`
   (operator-authored, not this file) BEFORE this file ever ran, via a
   disclosed non-degeneracy check (Kill Switch A, `r99_killswitch_a.py`):
   the a-priori grid-centre cell (detection=90d, baseline=730d) fires
   cleanly (max_z=3.45, 19 bars >= 2.0 across six years) -- only the
   slowest corner (180d/1095d) is degenerate -- so no substitution was
   needed, unlike R-98 where the natural centre cell was itself the
   degenerate one. This file does NOT re-pick the primary cell; it is read
   verbatim from `r99_shared.PRIMARY_DETECTION_WINDOW_DAYS` /
   `r99_shared.PRIMARY_BASELINE_WINDOW_DAYS`. The full
   `DETECTION_WINDOW_DAYS_GRID x BASELINE_WINDOW_DAYS_GRID` (9 cells) is
   computed and reported for plateau/robustness context, exactly mirroring
   R-96's/R-98's own "primary cell decides, the other 8 are context only"
   convention -- never used to override the primary cell's own pass/fail
   verdict, precisely so the decision is not "pick whichever of the 9 grid
   cells scores best" (an undisclosed multiple-comparisons search this
   project's routine treats as p-hacking).

3. DETECTION-LAG DEFINITION. IDENTICAL construction to
   `r98_conservative_gpd_alarm.py`'s `gate()`/`null_leads()` -- for each
   episode, within a +/-60-day search window around its onset
   (`r99_shared.episode_window`, `WINDOW_DAYS=60`):
   - v4's own reaction: the nearest DOWNWARD transition of `anchor_majority`
     to the onset (`r99_shared.nearest_transition`, `direction="down"` --
     identical rule R-82/R-83/R-84/R-85/R-86/R-96/R-98 all used).
   - RJ's reaction: the nearest bar where the causal RJ z-score
     (`r99_shared.rj_signal_zscore` on `daily_rv_bv_jump(...)['rj']`,
     aligned via `align_daily_causal`) crosses UP through `Z_THRESH=2.0`
     (`r99_shared.nearest_alarm`), closest to the onset.
   - LEAD = (v4_flip_time - detect_time) in days. Positive = the RJ alarm
     fired before v4's own gate reacted.

4. NULL. `r99_shared.block_bootstrap_shifts` circularly block-shifts the
   LOCAL (episode-window) RJ z-score series (block_days=5, n_draws=500,
   seed=9901 -- fixed now, disclosed, never altered after seeing results; a
   FRESH seed not used by any prior round's own gate file (81, 81081, 82,
   83, 84, 8501, 8502, 8601, 8602, 88, 8801, 93, 95, 9601, 9801), chosen by
   this round's own ID (99) x100 + 01, matching R-85/R-86/R-88/R-98's own
   "round-id-derived" seed convention, fixed before running, not selected
   after seeing any number) and recomputes "nearest alarm to the real,
   unshifted v4 flip time" against each shifted copy -- adapted from R-98's
   `null_leads` (itself adapted from R-82 through R-96's own chain); logic
   is otherwise identical.

5. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82/R-83/R-84/R-85/R-86/R-96/R-98): an
   episode counts as a PASS if BOTH (a) LEAD >= 0, AND (b) the true LEAD is
   >= the null distribution's median. Using the PRIMARY CONFIG ONLY
   (detection_window_days=90, baseline_window_days=730), PROCEED TO STEP B
   (build the confirming-vote strategy) only if >= 4 of the 6 episodes PASS.
   If fewer than 4 pass on the primary config: STOP, report this file's
   result as the whole conservative branch's product, write it up as
   NEGATIVE, do not build any strategy/confirming-vote code, do not touch
   any data on or after 2023-01-01. The bar is not relaxed, narrowed, or
   otherwise adjusted after seeing the numbers, and the other 8 grid cells
   never override this verdict.

6. WHAT WOULD MAKE THIS GATE FAIL, named now (copied/adapted from
   `r99_shared.py`'s own "WHAT WOULD MAKE THIS FAIL" section, and named
   there as this round's own pre-registered EXPECTATION, not a hoped-for
   result): THE SAME PATTERN THAT HAS BEATEN EIGHT CONSECUTIVE PRIOR
   MECHANISMS BUILT ON EIGHT DIFFERENT THEORETICAL BASES IS EXPECTED TO
   REPEAT -- a statistic computed FROM price (here: how much of today's
   realized variance is attributable to discontinuous jumps, relative to a
   smooth trailing baseline) can only shift once a large discontinuous move
   has already printed, which is exactly the moment v4's own fixed-window
   anchor is also starting to react, or later (a jump, by definition,
   resolves within a single day or a handful of bars -- it has no
   structural reason to lead a slower multi-week regime transition by
   weeks, the way a genuinely anticipatory signal would have to). If
   bipower-variation jump activity also lags every sudden 2020-2022 shock
   and only (at best) fires near the one slow 2018 build-up, that is the
   ninth independent mechanism converging on the same conclusion the
   ledger's standing diagnosis already leans toward: this six-episode gate
   is unwinnable by any estimator computed from this project's own
   committed price history, whatever field it is drawn from. This is stated
   as the modal expectation BEFORE any real-data number exists in this
   file, precisely so a disappointing result cannot later be read as a
   surprise that justifies loosening the stop rule.

7. STEP B -- CONTINGENT PRE-REGISTRATION (design frozen now, BEFORE Step A's
   numbers exist; only executed if Step A's primary cell clears >=4/6).

   CONFIRMING-VOTE CONSTRUCTION: `confirming_vote_frac` (imported unchanged
   from `r99_shared.py`) requires a DISCRETE {0,1} `meta_vote` (R-80's
   lesson, reused verbatim by R-88/R-94/R-95/R-98). The relative jump
   measure carries no direction of its own (same property GPD tail shape
   and FGI extremity had), so the vote's direction on any "confirmed"
   (alarm-firing) bar comes from v4's own FASTEST (20-day) anchor vote,
   identical construction to R-94/R-95/R-98's `compute_meta_vote`:
   `meta_vote[i] = fast_anchor_vote[i]` on any bar where `rj_z[i] >=
   Z_THRESH` ("confirmed"); otherwise `meta_vote[i] = meta_vote[i-1]`
   (carry forward, R-53/54/55's hysteresis-latch pattern). Before the first
   confirmed bar, defaults to the fast anchor's own then-current value (no
   dilution while unconfirmed, R-84's convention).

       frac = confirming_vote_frac(anchor_sum, meta_vote, weight)
            = (anchor_sum + weight * meta_vote) / (3 + weight)

   `weight=0` must recover `kelly_regime_v4` bit-for-bit (identity check,
   run first, before any swept configuration).

   WEIGHT SWEEP GRID (fixed a priori, disclosed now): weight in
   {0.5, 1.0, 2.0, 4.0} -- 4 values, matching R-88/R-94/R-95/R-98's own
   count for a like-for-like comparison, at the Step-A primary cell's own
   (detection_window_days, baseline_window_days).

   MANDATORY CHECKS: (i) identity check (weight=0 vs. kelly_regime_v4,
   bit-for-bit); (ii) causal-truncation probe on the Step-B target
   construction itself; (iii) inner-train / inner-validation evaluation vs.
   `kelly_regime_v4` and `buy_and_hold` on both spot and futures via
   `scripts.experiment.ev`; (iv) ETH falsification on
   `ethusd_coinbase_spot_5m.csv.gz`, same frozen construction, restricted to
   pre-2020 dates per `docs/ROUTINE.md` step 2's falsification convention
   (disclosed: the RJ jump/continuous decomposition is computed on the
   traded asset's OWN 5-minute price path, so `rj_z` here is refit on ETH's
   own returns using the identical primary (detection_window_days,
   baseline_window_days) -- this tests whether the CONSTRUCTION generalizes
   across the asset being traded, not a cross-asset transfer of a
   BTC-fitted estimator, same disclosure R-94/R-95/R-98 made for their own
   asset-specific inputs).

   PRE-REGISTERED PROMOTION-CANDIDATE BAR (only reached if everything above
   clears, per `docs/ROUTINE.md` step 4's standard promotion bar): beats
   `kelly_regime_v4`/`buy_and_hold` beyond the +/-0.2 Sharpe noise floor, OR
   a clear drawdown/tail improvement at matched exposure; AND passes ETH
   falsification (same sign); AND the weight-grid neighbourhood is a
   plateau, not a peak (report all 4 swept weights, not just the winner).
   If ANY fail: report NEGATIVE -- this is an inner-split read only, never
   the 2023+ holdout; nothing in this file is authorized to read the
   holdout under any outcome.

8. CONFIGS EVALUATED IN STEP A: 0 for the backtest-trials ledger (9 grid
   cells x 6 episodes = 54 gate diagnostics, no `ev()` calls -- identical
   accounting convention to every prior Step-A gate in this project, R-73
   through R-98). Step B's count, if reached: itemized in the results
   section below (identity check + causality probe checkpoints + weight
   sweep x 2 splits x 2 markets + ETH falsification).
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

from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from experiments.r99_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BASELINE_WINDOW_DAYS_GRID,
    DETECTION_WINDOW_DAYS_GRID,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    PRIMARY_BASELINE_WINDOW_DAYS,
    PRIMARY_DETECTION_WINDOW_DAYS,
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
    daily_rv_bv_jump,
    episode_window,
    nearest_alarm,
    nearest_transition,
    rj_signal_zscore,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 9901

WEIGHT_GRID = (0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------- data

def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def build_rj_z_aligned(bars: pd.DataFrame, detection_window_days: int, baseline_window_days: int,
                        daily_jump: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full signal-building pipeline: native 5-minute log returns grouped by
    UTC day -> causal RV/BV/jump/RJ (`daily_rv_bv_jump`) -> causal z-score of
    RJ (`rj_signal_zscore`) -> causally aligned onto `bars`' 5-minute index
    (`align_daily_causal`). Returns a 1-column DataFrame (`rj_z`) indexed
    like `bars`. `daily_jump` (the `daily_rv_bv_jump(bars)` frame) may be
    supplied precomputed -- it does not depend on
    `detection_window_days`/`baseline_window_days` -- to avoid recomputing
    RV/BV once per grid cell."""
    if daily_jump is None:
        daily_jump = daily_rv_bv_jump(bars)
    z = rj_signal_zscore(daily_jump["rj"], detection_window_days, baseline_window_days)
    aligned = align_daily_causal(pd.DataFrame({"rj_z": z}), bars)
    return aligned


def null_leads(rj_z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, z_thresh: float = Z_THRESH,
                n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r98_conservative's `null_leads` (itself adapted from
    r82 through r96's own chain): same circular block-shift construction,
    applied here to the RJ z-score threshold-crossing detector."""
    local = rj_z.reindex(window).to_numpy()
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

def gate(detection_window_days: int, baseline_window_days: int, bars: pd.DataFrame,
          majority: pd.Series, daily_jump: pd.DataFrame) -> dict:
    is_primary = (detection_window_days == PRIMARY_DETECTION_WINDOW_DAYS and
                  baseline_window_days == PRIMARY_BASELINE_WINDOW_DAYS)
    tag = " [PRIMARY]" if is_primary else ""
    print("=" * 78)
    print(f"R-99 CONSERVATIVE: RJ(detection={detection_window_days}d, "
          f"baseline={baseline_window_days}d) vs v4 anchor -- STEP A detection-lag gate{tag}")
    print("=" * 78)

    signal = build_rj_z_aligned(bars, detection_window_days, baseline_window_days, daily_jump=daily_jump)
    assert_no_holdout(signal)
    rj_z = signal["rj_z"]

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
        detect_time = nearest_alarm(rj_z, window, onset, Z_THRESH)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no RJ alarm'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str,
                                 flip=flip_time, detect=detect_time,
                                 pass_b=False, lead=float("nan"), null_median=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(rj_z, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        null_p90 = float(np.quantile(valid, 0.90)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    RJ(detection={detection_window_days}d,baseline={baseline_window_days}d) "
              f"nearest alarm (z>={Z_THRESH}): {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d  "
              f"null p90={null_p90:+.2f}d  (valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             detect=detect_time, lead=lead, null_median=null_median,
                             null_p90=null_p90, pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 4

    print("\n" + "-" * 78)
    for r in results:
        print(f"  {r['label']:42s} lead={r.get('lead', float('nan')):+.2f}d  "
              f"null_median={r.get('null_median', float('nan')):+.2f}d  "
              f"null_p90={r.get('null_p90', float('nan')):+.2f}d  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/6  "
          f"(detection={detection_window_days}d, baseline={baseline_window_days}d){tag}")
    print(f"max timestamp read anywhere for this cell: "
          f"{max(bars.index.max(), signal.index.max())}  (< {OOS_START})")

    return dict(detection_window_days=detection_window_days, baseline_window_days=baseline_window_days,
                is_primary=is_primary, results=results, n_pass=n_pass, passed=passed,
                signal=signal)


def causality_probe() -> bool:
    """Independent re-verification of the causal truncation claim for this
    file's own signal-building pipeline (`build_rj_z_aligned`), run at the
    PRIMARY config, per this project's rule that every round re-runs the
    probe itself rather than trusting a prior claim. Probes at a point well
    inside the 2018 episode window so the check exercises real,
    non-degenerate signal values. Uses `r99_shared.truncation_causality_probe`
    directly (default shorter_by=20,000 bars dropped)."""
    bars = load_btc_bars()
    check_at = bars.index.get_indexer([pd.Timestamp("2018-06-01", tz="UTC")], method="nearest")[0]

    def build(df: pd.DataFrame) -> np.ndarray:
        return build_rj_z_aligned(df, PRIMARY_DETECTION_WINDOW_DAYS,
                                   PRIMARY_BASELINE_WINDOW_DAYS)["rj_z"].to_numpy()

    ok = truncation_causality_probe(build, bars, check_at)
    print(f"\ncausal truncation probe (rj_z, detection={PRIMARY_DETECTION_WINDOW_DAYS}d, "
          f"baseline={PRIMARY_BASELINE_WINDOW_DAYS}d, check_at index {check_at} ~ "
          f"{bars.index[check_at]}): {'PASS' if ok else 'FAIL'}")
    return ok


def run_full_grid() -> dict:
    bars = load_btc_bars()
    majority = anchor_majority(bars)
    daily_jump = daily_rv_bv_jump(bars)
    assert_no_holdout(daily_jump)

    cells = []
    primary_cell = None
    max_ts_seen = bars.index.max()
    for detection_window_days in DETECTION_WINDOW_DAYS_GRID:
        for baseline_window_days in BASELINE_WINDOW_DAYS_GRID:
            cell = gate(detection_window_days, baseline_window_days, bars, majority, daily_jump)
            cells.append(cell)
            if cell["is_primary"]:
                primary_cell = cell
            max_ts_seen = max(max_ts_seen, cell["signal"].index.max())

    assert primary_cell is not None, (
        f"primary config (detection={PRIMARY_DETECTION_WINDOW_DAYS}, "
        f"baseline={PRIMARY_BASELINE_WINDOW_DAYS}) missing from grid")

    print("\n" + "=" * 78)
    print("R-99 CONSERVATIVE: FULL 3x3 GRID SUMMARY (RJ(detection, baseline) vs v4 anchor)")
    print("=" * 78)
    print(f"{'detection_days':>15} {'baseline_days':>15} {'n_pass/6':>10}  {'primary?':>9}")
    for cell in cells:
        marker = "<-- PRIMARY" if cell["is_primary"] else ""
        print(f"{cell['detection_window_days']:>15} {cell['baseline_window_days']:>15} "
              f"{cell['n_pass']:>8}/6  {marker}")

    print(f"\nPRIMARY CONFIG (detection={PRIMARY_DETECTION_WINDOW_DAYS}d, "
          f"baseline={PRIMARY_BASELINE_WINDOW_DAYS}d): "
          f"{primary_cell['n_pass']}/6 episodes pass")
    print(f"GATE VERDICT (primary config only, per pre-registered stop rule): "
          f"{'PASS -> proceed to Step B (build confirming-vote strategy)' if primary_cell['passed'] else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated in this file's Step A toward the backtest "
          f"trials ledger: 0 (fixed measurement gate; 9 grid cells x 6 episodes = "
          f"54 episode-lead measurements, no ev() calls -- see module docstring)")
    print(f"max timestamp read anywhere in this session so far: {max_ts_seen}  (< {OOS_START})")

    return dict(cells=cells, primary=primary_cell)


# ==========================================================================
# STEP B -- built only if the primary cell above passes (banner item 7).
# ==========================================================================

def compute_meta_vote(df: pd.DataFrame, rj_z: pd.Series,
                       horizons: tuple[int, ...] = V4_HORIZONS,
                       band: float = V4_BAND) -> np.ndarray:
    """RJ-alarm-confirmed latch on the FASTEST anchor's own 0/1 vote
    (banner item 7): `meta_vote[i] = fast_anchor_vote[i]` on any bar where
    `rj_z[i] >= Z_THRESH` ("confirmed"); otherwise `meta_vote[i] =
    meta_vote[i-1]` (carries forward). Before the first confirmed bar,
    defaults to the fast anchor's own then-current value. Causal: `rj_z`
    and each anchor vote are both causal; the latch update at i depends
    only on values at <= i."""
    fast_vote = anchor_votes(df, horizons=horizons, band=band)[0].to_numpy()
    z = rj_z.reindex(df.index).to_numpy()
    confirmed = z >= Z_THRESH

    n = len(df)
    meta = np.empty(n)
    meta[0] = fast_vote[0]
    for i in range(1, n):
        meta[i] = fast_vote[i] if confirmed[i] else meta[i - 1]
    return meta


class BvJumpConfirmKelly(Strategy):
    """kelly_regime_v4 + a bipower-variation-jump-alarm-confirmed fast-anchor
    vote (R-99 conservative, unregistered). Structurally v3/v4's own
    prepare(), with the plain 3-anchor average `frac = anchor_sum/3`
    replaced by `confirming_vote_frac(anchor_sum, meta_vote, weight)`.
    `weight=0` must recover v4 bit-for-bit. Modelled directly on
    `r98_conservative_gpd_alarm.py`'s `GpdConfirmKelly`. Not `@register`ed
    -- stays in experiments/ per docs/ROUTINE.md."""

    name = "r99_conservative_bv_jump_confirm"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4

    def __init__(self, rj_z: pd.Series, weight: float = 1.0,
                 horizons: tuple[int, ...] = V4_HORIZONS, band: float = V4_BAND,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85) -> None:
        self.rj_z = rj_z
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

        z = self.rj_z.reindex(df.index)
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


def build_target_primary(df: pd.DataFrame, rj_z: pd.Series) -> np.ndarray:
    """Target-construction function for the Step-B causal truncation probe,
    frozen at the pre-registered primary candidate (weight=1.0)."""
    return BvJumpConfirmKelly(rj_z=rj_z, weight=1.0).prepare(df.copy())["target"].to_numpy()


def run_identity_check(df_full: pd.DataFrame, rj_z: pd.Series) -> float:
    """weight=0 must recover kelly_regime_v4 bit-for-bit. Returns max|diff|."""
    df = df_full.loc[:INNER_TRAIN_END].copy()
    v4_target = get_strategy("kelly_regime_v4").prepare(df.copy())["target"].to_numpy()
    cand_target = BvJumpConfirmKelly(rj_z=rj_z, weight=0.0).prepare(df.copy())["target"].to_numpy()
    max_diff = float(np.max(np.abs(v4_target - cand_target)))
    print(f"[identity] weight=0 vs kelly_regime_v4, {len(df):,} bars: "
          f"max|diff| = {max_diff:.3e}")
    return max_diff


def run_causality_probe_step_b(df_full: pd.DataFrame, rj_z: pd.Series) -> list[bool]:
    df = df_full.loc[:INNER_TRAIN_END].copy()
    results = []
    for check_at in (150_000, 250_000, 350_000):
        ok = truncation_causality_probe(
            lambda d: build_target_primary(d, rj_z), df, check_at)
        print(f"[causality] check_at={check_at}: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def eval_config(ev, SPOT, FUTURES, rj_z: pd.Series, weight: float, tag: str) -> dict:
    out = {}
    for split_name, kw in (
        ("train", dict(end=INNER_TRAIN_END)),
        ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END)),
    ):
        for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
            strat = BvJumpConfirmKelly(rj_z=rj_z, weight=weight)
            m = ev(strat, market=mkt, tag=f"{tag} {split_name} {mkt_name}", **kw)
            out[(split_name, mkt_name)] = m
    return out


def run_sweep(ev, SPOT, FUTURES, rj_z: pd.Series) -> dict:
    results = {}
    for weight in WEIGHT_GRID:
        tag = f"w{weight}"
        results[weight] = eval_config(ev, SPOT, FUTURES, rj_z, weight, tag)
    return results


def run_eth_falsification(ev, rj_z_eth_construction_fn, weight: float) -> dict:
    """ETH falsification: the SAME frozen construction (BTC-fit RJ jump
    alarm gating v4's fast anchor) applied UNMODIFIED to ETH price action,
    restricted to pre-2020 dates per docs/ROUTINE.md step 2's falsification
    convention. Disclosed explicitly (banner item 7): the RJ jump/continuous
    decomposition is fit on the traded asset's OWN native 5-minute return
    history, so `rj_z` here is refit on ETH's own returns using the
    identical primary (detection_window_days, baseline_window_days) -- this
    tests whether the CONSTRUCTION generalizes across the asset being
    traded, not a cross-asset transfer of the BTC-fitted estimator (same
    disclosure R-94/R-95/R-98 made for their own asset-specific inputs)."""
    from tradebot.broker import MarketSpec

    spot = MarketSpec.spot()
    path = DATA_DIR / "ethusd_coinbase_spot_5m.csv.gz"
    if not path.exists():
        print(f"  ETH falsification: {path.name} not found in data/ -- SKIPPED, "
              f"reported as not run, not as a pass.")
        return {"skipped": True}

    eth_df = load_ohlcv_csv(path)
    pre2020_cutoff = pd.Timestamp("2020-01-01", tz=eth_df.index.tz)
    eth_df = eth_df.loc[eth_df.index < pre2020_cutoff].copy()
    assert_no_holdout(eth_df)
    if len(eth_df) == 0:
        print("  ETH falsification: no pre-2020 bars available -- SKIPPED.")
        return {"skipped": True}

    eth_rj_z = rj_z_eth_construction_fn(eth_df)
    assert_no_holdout(eth_rj_z)

    cand = BvJumpConfirmKelly(rj_z=eth_rj_z, weight=weight)
    v4 = get_strategy("kelly_regime_v4")
    m_v4 = ev(v4, df=eth_df, market=spot, tag="ETH (coinbase, pre-2020): v4")
    m_cand = ev(cand, df=eth_df, market=spot, tag="ETH (coinbase, pre-2020): candidate")
    delta = m_cand.sharpe - m_v4.sharpe
    print(f"  ETH (coinbase, pre-2020): v4 sharpe={m_v4.sharpe:.2f}  "
          f"candidate sharpe={m_cand.sharpe:.2f}  delta={delta:+.2f}")
    return dict(v4=m_v4.sharpe, cand=m_cand.sharpe, delta=delta,
                bars=len(eth_df), start=str(eth_df.index[0]), end=str(eth_df.index[-1]))


def run_step_b(gate_result: dict) -> None:
    from scripts.experiment import DF, FUTURES, SPOT, OOS_START as EXP_OOS_START, ev

    assert EXP_OOS_START == OOS_START, (
        f"scripts.experiment.OOS_START ({EXP_OOS_START}) != r99_shared.OOS_START "
        f"({OOS_START}) -- refusing to proceed with a mismatched holdout boundary")

    print("\n" + "=" * 78)
    print(f"STEP B (primary cell passed): weight sweep + mandatory checks -- "
          f"primary cell detection={PRIMARY_DETECTION_WINDOW_DAYS}d, "
          f"baseline={PRIMARY_BASELINE_WINDOW_DAYS}d")
    print("=" * 78)

    df_trunc = DF.loc[DF.index < pd.Timestamp(OOS_START, tz=DF.index.tz)]
    daily_jump = daily_rv_bv_jump(df_trunc)
    rj_z_signal = build_rj_z_aligned(df_trunc, PRIMARY_DETECTION_WINDOW_DAYS,
                                      PRIMARY_BASELINE_WINDOW_DAYS, daily_jump=daily_jump)["rj_z"]
    assert_no_holdout(rj_z_signal)
    # Reindex onto the full DF's index (train+val only will be used by ev()
    # via start/end, but prepare() is called on whatever frame ev() passes
    # it, so the Series must cover the full pre-OOS range DF itself does).
    rj_z_full = rj_z_signal.reindex(DF.index)

    n_configs = 0
    print("\n=== identity check ===")
    run_identity_check(DF, rj_z_full)
    n_configs += 1

    print("\n=== causality probe (Step B target construction) ===")
    run_causality_probe_step_b(DF, rj_z_full)

    print("\n=== baselines (kelly_regime_v4, buy_and_hold) ===")
    for name in ("kelly_regime_v4", "buy_and_hold"):
        for split_name, kw in (("train", dict(end=INNER_TRAIN_END)),
                                ("val", dict(start=INNER_VAL_START, end=INNER_VAL_END))):
            for mkt_name, mkt in (("spot", SPOT), ("futures", FUTURES)):
                ev(get_strategy(name), market=mkt, tag=f"{name} {split_name} {mkt_name}", **kw)
                n_configs += 1

    print(f"\n=== weight sweep: {WEIGHT_GRID} ===")
    run_sweep(ev, SPOT, FUTURES, rj_z_full)
    n_configs += len(WEIGHT_GRID) * 4  # 2 splits x 2 markets per weight

    print("\n=== ETH falsification (pre-2020) ===")

    def eth_rj_z_fn(eth_df: pd.DataFrame) -> pd.Series:
        eth_daily_jump = daily_rv_bv_jump(eth_df)
        return build_rj_z_aligned(eth_df, PRIMARY_DETECTION_WINDOW_DAYS,
                                   PRIMARY_BASELINE_WINDOW_DAYS, daily_jump=eth_daily_jump)["rj_z"]

    run_eth_falsification(ev, eth_rj_z_fn, weight=1.0)
    n_configs += 2  # v4 + candidate on ETH

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
        print(f"usage: python experiments/r99_conservative_bv_jump_alarm.py [{'|'.join(cmds)}]")
