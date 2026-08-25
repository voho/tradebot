#!/usr/bin/env python
"""R-124 CONSERVATIVE branch: ``FracDiffVoteKellyV4`` -- a v4-SHAPED
directional vote (three latched anchors, averaged, then multiplied by v4's
own untouched ``scale`` and v4's own 10% deadband) built on Fixed-Window
Fractionally-Differenced (FFD) `log(close)` instead of raw `close`. Full
citation trail, literature grounding (Lopez de Prado 2018 Ch.5; Hosking
1981), the axis this attacks (SIZE -- a 27th+ construction in that family,
but a structurally new INPUT REPRESENTATION substituted into the vote's own
slot, `scale` left completely untouched per R-62), and the exhaustive
non-duplication argument against every related prior round all live in
``experiments/r124_shared.py``'s own module docstring (read in full before
this file was written); none of that is re-derived here beyond the
one-paragraph summary in item 1 below. This file does not edit, and never
reads a bar at or after ``OOS_START`` from, that module or any other file
under ``experiments/`` or ``src/``.

=====================================================================
PRE-REGISTRATION (frozen before any real-data bind_frac, r_sq, d_sharpe,
or bootstrap number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): replace v4's three raw-`close` mean-crossing
   anchors with three rolling z-scores of the SAME frozen FFD series
   (`r124_shared.causal_ffd_log_close`, `d=r124_shared.FFD_D=0.85`, window
   ~50 bars ~4.2 hours) at v4's own horizons (20/40/80 days), latched
   bullish/bearish exactly like v4's own `_latched_anchor_vote` (hysteresis
   via `np.where(...).ffill().fillna(0.0)`), averaged into `frac`, then fed
   through v4's UNCHANGED `v4_scale` and UNCHANGED `apply_deadband`. Only
   the vote's INPUT SERIES and the (necessarily different, since FFD is
   stationary, not a price level) threshold statistic change -- the
   3-anchor / hysteresis-latch / average / scale / deadband ARCHITECTURE is
   byte-for-byte v4's own.

   IMPORTANT NAMED TENSION (from `r124_shared.py`'s own docstring, restated
   here because it is the single most important fact for interpreting
   whatever this file finds): `FFD_D=0.85` was selected purely for
   stationarity (minimum `d` that clears the causal ADF-lite test on
   inner-train), which happens to leave a SHORT-memory, fast-reacting
   series (window ~50 bars ~4.2 hours, correlation with the raw log-price
   level ~0.83) relative to v4's own SLOW 20/40/80-DAY anchors. A rolling
   z-score of a ~4.2-hour-memory series, even smoothed over a 20-80 DAY
   window, is measuring something structurally close to "how far has the
   recent few hours' worth of local, short-memory price innovation
   deviated from its own multi-week average," not "is price above or below
   its own multi-week trend" the way v4's raw-close band asks. This is
   named now as the primary reason a NEGATIVE result would not be
   surprising: the input series' own memory (hours) and the vote's smoothing
   window (weeks-months) are mismatched by roughly two orders of magnitude,
   which is exactly the sort of construction R-124's own pre-registration
   warns is "SMOOTHER than raw returns but NOISIER... than a slow 20/40/80-day
   rolling-mean anchor."

2. CONSTRUCTION (exact):

       ffd[t]                = r124_shared.causal_ffd_log_close(df)[t]
       roll_mean_h[t]        = ffd.rolling(h*288).mean()[t]      # h in {20,40,80} days
       roll_std_h[t]         = ffd.rolling(h*288).std()[t]
       z_h[t]                = (ffd[t]-roll_mean_h[t]) / roll_std_h[t]   (NaN if std<=0)
       vote_h[t]             = 1.0 if z_h[t] >  Z_BAND
                              = 0.0 if z_h[t] < -Z_BAND
                              = hold-previous (ffill, then fillna(0.0))   otherwise
       frac[t]                = mean(vote_20[t], vote_40[t], vote_80[t])
       raw[t]                 = frac[t] * r124_shared.v4_scale(df)[t]     # scale UNTOUCHED
       target[t]              = r124_shared.apply_deadband(raw)[t]        # deadband UNTOUCHED

   DEFAULT/PRIMARY CONFIG: `Z_BAND` selected by the Step-0 non-degeneracy
   rule below (grid centre 0.10 preferred, confirmed or overridden only by
   that rule, never by a performance number).

3. WHY A Z-SCORE, NOT V4'S OWN PERCENTAGE BAND: `close` is a price LEVEL
   (v4's 1% band asks "is price 1% above/below its own rolling mean level");
   the FFD series is, by construction, STATIONARY (mean/variance do not
   drift with the price level), so a fixed percentage-of-level band is not
   even well-defined on it (a 1% move in a mean-zero stationary series has
   no natural anchor). A rolling z-score is the direct, standard
   stationary-series analogue: it asks the same qualitative question ("is
   the current value unusually far from its own recent local behaviour, in
   the SAME units the series itself has been fluctuating in") using a
   scale-free statistic instead of a percentage-of-level one. This is the
   ONLY architectural change forced by the input substitution; everything
   else (hysteresis-latch shape, 3-horizon averaging, `scale`, deadband) is
   copied unchanged from v4.

4. STEP-0 NON-DEGENERACY GRID AND SELECTION RULE (identical shape and
   selection-rule convention to R-102 through R-105's own Step-0 gates,
   using this round's own pre-registered `Z_GRID`/`SELECTION_ORDER`
   defined below rather than reusing `r124_shared.STEP0_FLOOR_GRID`, which
   is a discount-floor grid inherited from a different construction that
   does not apply here):

   Grid: `Z_BAND in Z_GRID = (0.05, 0.10, 0.20)` (3 cells, fixed a priori).
   For each cell, on BTC's inner-train window (2017-01-01 -> 2020-12-31)
   only:
     - `bind_frac` = fraction of inner-train bars where
       `|frac_candidate[t] - v4_vote_frac(df)[t]| > 0.01`. Kill switch A:
       must be `> r124_shared.BIND_FRAC_THRESH` (0.01) -- i.e. the
       candidate vote must actually disagree with v4's own vote on a
       non-trivial fraction of bars, not be a near-identical relabelling.
     - `r_sq` = `r124_shared.r_squared(candidate_raw_desired, v4_raw_desired(df))`,
       both over the FULL BTC frame, masked to inner-train before the R^2
       is taken. Kill switch B: must be `< r124_shared.R2_THRESH` (0.98) --
       i.e. the candidate's raw desired-exposure path must not be a
       near-exact rescale of v4's own.
   A cell QUALIFIES iff both kill switches pass.

   SELECTION RULE (non-degeneracy ONLY -- no performance number is
   inspected before this rule is applied): the primary cell is the first
   `Z_BAND` in `SELECTION_ORDER = (0.10, 0.05, 0.20)` (grid centre first,
   then the fixed fall-through order given in the task spec) that
   qualifies. If NONE of the three cells qualify, this file STOPS at
   Step-0 -- a legitimate, informative NEGATIVE result per
   `docs/ROUTINE.md`, not a bug to route around. No B1-B5 code runs in
   that case, and no bar on/after `OOS_START` (2023-01-01) is ever touched
   either way.

5. CAUSAL TRUNCATION PROBE, run on the final composed `build_target`
   closure (bound to the selected primary `Z_BAND`) BEFORE trusting any
   headline (B1-B5) number, via `r124_shared.causal_truncation_probe_series`
   on real BTC data -- this project has had lookahead bugs before (one
   produced a $3.7e23 balance), so this check is never skipped regardless
   of how the Step-0 numbers look.

6. PROMOTION BAR (docs/ROUTINE.md's own bar, operationalized via
   `r124_shared.py`'s centralized B1/B2/B4/B5 machinery -- reused verbatim,
   not re-implemented, so this file's gates are byte-for-byte the same code
   every recent SIZE/ERR-axis round's gates run through):
     B1 (gating): `r124_shared.b1_from_inner_val` on the primary
        `compare()` call's own `inner_val` rows, both markets -- `d_sharpe
        > +0.2` OR the paired bootstrap interval's lower bound excludes
        zero on the positive side. Must pass on BOTH markets.
     B2 (diagnostic ONLY, never itself gates promotion):
        `r124_shared.b2_diagnostic` -- drawdown improvement counted only
        where risk-matched (R-33's own standing rule).
     B3 (plateau, gating): re-run B1's inner-val comparison
        (`r124_shared.inner_val_rows`) at the two NON-primary `Z_GRID`
        values, both markets (4 rows; the primary Z_BAND's 2 rows are
        reused directly from the primary `compare()` call's own `inner_val`
        rows rather than recomputed). PASS requires a majority (ties count
        as pass, per R-105's own tie-inclusive convention) of these 4
        neighbor rows sharing the SAME-MARKET primary cell's `d_sharpe`
        sign. If Step-0 finds only ONE of the three `Z_GRID` cells
        qualifying (i.e. the two neighbor cells are themselves degenerate
        relabellings or near-non-events per the Step-0 kill switches), this
        is disclosed explicitly and B3's majority result is reported as a
        single-qualifying-point context check, not a genuine 3-point
        plateau.
     B4 (ETH falsification, gating, FULL PASS REQUIRED): `compare()` with
        `include_eth=True` (the same call used for B1) already produces the
        `eth_replication` slice rows; `r124_shared.b4_eth_falsification`
        against those vs. the primary `inner_val` rows. Same sign as BTC
        inner-val, BOTH markets required for a full pass.
     B5 (fee-tier robustness, gating): `r124_shared.b5_fee_tier` at the
        0.40% real taker tier, primary Z_BAND, both markets -- no sign
        reversal vs. the standard-fee inner_val result on either market.
   PROMOTE-candidate only if Step-0 selects a primary AND the causal probe
   passes AND B1 AND B3 AND B4 (full) AND B5 all hold (B2 is
   diagnostic-only). Default: NEGATIVE. This file never reads or reports
   any bar at or after `OOS_START` (2023-01-01) regardless of outcome --
   the real holdout is a separate, later step gated on ALL of the above
   passing, out of scope for this file.

7. WHAT WOULD MAKE THIS FAIL: named already, in full, in `r124_shared.py`'s
   own "WHAT WOULD MAKE EACH BRANCH FAIL" section (a fractionally-
   differenced input is smoother than raw returns but noisier/faster than
   v4's own slow anchors; nothing about the FFD construction adds trend
   information the anchors do not already extract) and restated above in
   item 1's "IMPORTANT NAMED TENSION" (the ~4.2-hour FFD memory vs. the
   20-80-DAY z-score smoothing window is a two-order-of-magnitude mismatch
   that gives this construction no obvious reason to carry more usable
   directional information than v4's own raw-close band, even though it is
   mechanically guaranteed to disagree with v4 on SOME bars by the Step-0
   kill switches). A clean NEGATIVE on B1 (inner-validation) or B4 (ETH
   falsification) is the fully expected, fully successful outcome; reported
   honestly, whichever way it actually comes out, in the results below.

CONFIGURATIONS EVALUATED IN THIS FILE (IF Step-0 selects a primary): 3
(Step-0 Z_BAND grid) + 6 (primary config's full `compare()`: inner_train x2
markets + inner_val x2 markets + eth_replication x2 markets) + 4 (B3's two
non-primary Z_BAND grid cells x 2 markets; the primary's own 2 inner_val
rows are reused directly from the primary `compare()` call) + 2 (B5's 0.40%
fee tier, 2 markets) = 15 total. IF Step-0 finds no qualifying cell, this
file stops after the 3 Step-0 cells and reports that outcome directly (no
B1-B5 code runs, no ETH data or bar on/after OOS_START ever touched).

----------------------------------------------------------------------
Run: python experiments/r124_conservative_fracdiff_vote.py
(from the repo root, with the project venv active)
----------------------------------------------------------------------
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

from experiments.r124_shared import (  # noqa: E402
    BARS_PER_DAY,
    BIND_FRAC_THRESH,
    FFD_D,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    OOS_START,
    R2_THRESH,
    SPOT,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    causal_ffd_log_close,
    causal_truncation_probe_series,
    compare,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    print_plateau_table,
    print_rows,
    r_squared,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS

# ---------------------------------------------------------- pre-registered
Z_GRID = (0.05, 0.10, 0.20)
SELECTION_ORDER = (0.10, 0.05, 0.20)   # grid centre first, then the fixed fall-through order


# ================================================================== (1)
# The mechanism itself: v4's own 3-anchor / hysteresis-latch / average /
# scale / deadband architecture, fed a rolling z-score of the frozen FFD
# series instead of a percentage-of-level band on raw `close`.
# ==================================================================

def _latched_fracdiff_vote(ffd: pd.Series, days: int, z_band: float) -> pd.Series:
    """One anchor's own latched 0/1 vote on the FFD series -- identical
    hysteresis-latch shape to r102_shared._latched_anchor_vote, using a
    rolling z-score threshold instead of a percentage-of-level band (the
    FFD series is stationary, so a percentage-of-level band is not
    well-defined on it)."""
    window = int(days * BARS_PER_DAY)
    roll_mean = ffd.rolling(window).mean()
    roll_std = ffd.rolling(window).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_arr = roll_mean.to_numpy()
        std_arr = roll_std.to_numpy()
        ffd_arr = ffd.to_numpy()
        z = np.where(std_arr > 0, (ffd_arr - mean_arr) / std_arr, np.nan)
    v = pd.Series(
        np.where(z > z_band, 1.0, np.where(z < -z_band, 0.0, np.nan)),
        index=ffd.index,
    )
    return v.ffill().fillna(0.0)


def fracdiff_vote_frac(df: pd.DataFrame, z_band: float) -> pd.Series:
    """The candidate's own directional vote: mean of three latched FFD
    z-score votes at v4's own horizons (20/40/80 days)."""
    ffd = causal_ffd_log_close(df, d=FFD_D)
    votes = [_latched_fracdiff_vote(ffd, days, z_band) for days in V4_HORIZONS]
    frac = sum(votes) / len(votes)
    return frac


def fracdiff_raw_desired(df: pd.DataFrame, z_band: float) -> np.ndarray:
    """frac (FFD z-score vote) * v4's own UNCHANGED scale, before deadband."""
    return fracdiff_vote_frac(df, z_band).to_numpy() * v4_scale(df)


def build_target(df: pd.DataFrame, z_band: float) -> np.ndarray:
    return apply_deadband(fracdiff_raw_desired(df, z_band))


def make_build_target(z_band: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, z_band=z_band)
    _build.__name__ = f"fracdiff_vote_z{z_band:g}"
    return _build


# ================================================================== (2)
# Pre-flight self-test: frac in [0,1], and the vote reduces to v4's own
# vote-shape guarantees (bounded average of three 0/1-valued series).
# ==================================================================

def self_test_bounds(btc: pd.DataFrame) -> bool:
    ok = True
    for z in Z_GRID:
        frac = fracdiff_vote_frac(btc, z)
        in_range = bool(frac.between(0.0, 1.0).all())
        print(f"  Z_BAND={z:.2f}: frac in [0,1] everywhere? {in_range}")
        ok = ok and in_range
    return ok


# ================================================================== (3)
# Step-0 non-degeneracy grid.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], int]:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    v4_frac = v4_vote_frac(btc).to_numpy()
    v4_raw = v4_raw_desired(btc)

    rows = []
    for z in Z_GRID:
        frac_c = fracdiff_vote_frac(btc, z).to_numpy()
        raw_c = frac_c * v4_scale(btc)
        bind_frac = float(np.mean(np.abs(frac_c[mask] - v4_frac[mask]) > 0.01))
        r_sq = r_squared(raw_c[mask], v4_raw[mask])
        qualifies = (bind_frac > BIND_FRAC_THRESH) and (r_sq < R2_THRESH)
        rows.append(dict(z_band=z, bind_frac=bind_frac, r_sq=r_sq, qualifies=qualifies))
    return rows, n_bars


def select_primary(rows: list[dict]) -> dict | None:
    by_z = {r["z_band"]: r for r in rows}
    for z in SELECTION_ORDER:
        r = by_z.get(z)
        if r is not None and r["qualifies"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, {n_bars:,} bars)")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r_sq < {R2_THRESH}")
    hdr = f"{'Z_BAND':>7s} {'bind_frac':>10s} {'r_sq':>8s} {'qualifies':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = " <- grid centre" if abs(r["z_band"] - 0.10) < 1e-9 else ""
        print(f"{r['z_band']:7.2f} {r['bind_frac']:10.4f} {r['r_sq']:8.4f} "
              f"{'YES' if r['qualifies'] else 'no':>10s}{tag}")


# ================================================================== (4)
# Causal truncation probe.
# ==================================================================

def run_causal_probe(df: pd.DataFrame, build_fn) -> bool:
    print(f"\ncausal_truncation_probe_series({build_fn.__name__}, btc):")
    try:
        causal_truncation_probe_series(build_fn, df)
        print("  PASS")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


# ================================================================== (5)
# Promotion bar: B1 (gating), B2 (diagnostic), B3 (Z_GRID plateau, gating),
# B4 (ETH falsification, gating), B5 (fee tier, gating).
# ==================================================================

def run_promotion_bar(primary_z: float, step0_rows: list[dict], btc: pd.DataFrame,
                      eth: pd.DataFrame) -> dict:
    label = f"fracdiff_vote_z{primary_z:g}"
    build_primary = make_build_target(primary_z)

    hr(f"PROMOTION BAR -- PRIMARY CONFIG Z_BAND={primary_z:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                   markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    # ---- B1
    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)

    # ---- B2 (diagnostic only)
    b2_ok, b2_cells = b2_diagnostic(inner_val_primary)

    # ---- B3: the two NON-primary Z_GRID cells, both markets; primary reused.
    n_qualifying = sum(1 for r in step0_rows if r["qualifies"])
    plateau_rows: dict[float, list[dict]] = {
        primary_z: [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                         d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                         vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                         boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                         boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                    for r in inner_val_primary]
    }
    neighbor_zs = [z for z in Z_GRID if abs(z - primary_z) > 1e-12]
    for z in neighbor_zs:
        bf = make_build_target(z)
        blabel = f"fracdiff_vote_z{z:g}"
        plateau_rows[z] = inner_val_rows(bf, blabel, btc)

    primary_sign_by_market = {r["market"]: np.sign(r["d_sharpe"]) for r in plateau_rows[primary_z]}
    neighbor_rows = [r for z in neighbor_zs for r in plateau_rows[z]]
    matches = [np.sign(r["d_sharpe"]) == primary_sign_by_market.get(r["market"], np.nan)
              for r in neighbor_rows]
    n_match = sum(matches)
    n_total = len(matches)
    b3_pass = (n_match >= n_total / 2.0) if n_total else False
    b3_single_point = n_qualifying <= 1

    # ---- B4: ETH falsification
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)

    # ---- B5: fee-tier robustness
    hr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary config, BTC inner-validation")
    b5_pass, b5_cells = b5_fee_tier(build_primary, label, btc, inner_val_primary)

    all_pass = b1_pass and b3_pass and b4_full and b5_pass

    return dict(
        label=label, z_band=primary_z,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells,
        b3_plateau_rows=plateau_rows, b3_pass=b3_pass, b3_single_point=b3_single_point,
        b3_n_match=n_match, b3_n_total=n_total,
        b4_cells=b4_cells, b4_partial_pass=b4_partial, b4_full_pass=b4_full,
        b5_cells=b5_cells, b5_pass=b5_pass,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 2 * len(neighbor_zs) + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-124 CONSERVATIVE: FracDiffVoteKellyV4 -- v4's own 3-anchor/latch/"
       "scale/deadband architecture, fed a rolling z-score of the frozen "
       "Fixed-Window Fractionally-Differenced log(close) instead of raw close")
    print(f"FFD_D (frozen, selected causally on inner-train only, r124_shared.py) = {FFD_D}")
    print("mechanism: replace v4's 1%-of-level anchor band with a rolling z-score band on the")
    print("SAME stationary FFD series at v4's own three horizons (20/40/80d); everything else")
    print("(hysteresis latch, 3-anchor average, v4_scale, apply_deadband) is v4's own, unchanged.")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    hr("PRE-FLIGHT SELF-TEST (frac bounds, all Z_GRID cells)")
    bounds_ok = self_test_bounds(btc)
    print(f"  -> bounds self-test: {bounds_ok}")
    if not bounds_ok:
        print("\nSELF-TEST FAILURE -- stopping before any Step-0 number is trusted.")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="ABORTED (self-test failure)", max_ts=max(max_ts_seen))

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY KILL SWITCH (run BEFORE any Sharpe/compare() number)")
    step0_rows, n_bars = step0_grid(btc)
    print_step0_table(step0_rows, n_bars)

    primary = select_primary(step0_rows)

    if primary is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell has both bind_frac > 1% and r_sq < 0.98 on inner-train: the")
        print("FFD z-score vote is either a near-total relabelling of v4's own vote (fails to")
        print("bind) or a near-exact rescale of v4's own raw desired-exposure path on the")
        print("pre-registered Z_BAND grid. Per this file's own pre-registration, this Step-0")
        print("table is the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0.")
        print("No causal probe, B1-B5 code, or ETH load runs.")
        n_configs = len(Z_GRID)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 grid only)")
        max_ts = max(max_ts_seen)
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                   n_configs=n_configs, max_ts=max_ts, verdict="NEGATIVE (Step-0 kill switch)")

    is_center = abs(primary["z_band"] - 0.10) < 1e-9
    print(f"\nPRIMARY CONFIG SELECTED (non-degeneracy rule only): Z_BAND={primary['z_band']:g}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r_sq={primary['r_sq']:.4f})")
    print(f"  selection: {'grid-centre cell (0.10) qualified' if is_center else 'grid-centre cell did NOT qualify; next cell in SELECTION_ORDER=(0.10,0.05,0.20) chosen'}")
    n_qualifying = sum(1 for r in step0_rows if r["qualifies"])
    print(f"  {n_qualifying}/{len(step0_rows)} Z_GRID cells qualify Step-0's kill switches "
          f"(relevant for B3's plateau-vs-single-point framing below)")

    build_primary = make_build_target(primary["z_band"])

    # ==================================================== CAUSAL PROBE
    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    probe_ok = run_causal_probe(btc, build_primary)
    print(f"\nCAUSAL PROBE PASS: {probe_ok}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    bar = run_promotion_bar(primary["z_band"], step0_rows, btc, eth)

    hr("B1 -- inner-validation Sharpe leg, both markets "
       "(dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  voided={c['voided']}")

    hr(f"B3 -- plateau: Z_GRID={Z_GRID} at primary selection, inner-validation, both markets")
    print_plateau_table(bar["b3_plateau_rows"])
    if bar["b3_single_point"]:
        print(f"\nNOTE: only {n_qualifying}/{len(Z_GRID)} Z_GRID cell(s) passed Step-0's own "
              "kill switches -- the neighbor cell(s) below are reported for context, but this "
              "is NOT a genuine 3-point plateau; treat B3 as a single-qualifying-point result.")
    print(f"\nB3 (neighbor rows sharing primary's same-market d_sharpe sign, ties pass): "
          f"{bar['b3_n_match']}/{bar['b3_n_total']} -> {bar['b3_pass']}")

    hr("B4 -- ETH falsification (pre-registered)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  ETH d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  same_sign_as_btc={c['same_sign_as_btc']}")
    print(f"B4 FULL PASS (both markets): {bar['b4_full_pass']}")
    print(f"B4 PARTIAL PASS (at least one market): {bar['b4_partial_pass']}")

    hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  @0.40% d_sharpe={c['d_sharpe']:+.4f}  "
              f"@0.40% boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"@0.10% boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hr("VERDICT")
    print(f"causal probe = {probe_ok}   B1 = {bar['b1_pass']}   B2 = diagnostic-only   "
          f"B3 = {bar['b3_pass']} (single_point={bar['b3_single_point']})   "
          f"B4(full) = {bar['b4_full_pass']}   B4(partial) = {bar['b4_partial_pass']}   "
          f"B5 = {bar['b5_pass']}")
    all_gates_pass = probe_ok and bar["b1_pass"] and bar["b3_pass"] and bar["b4_full_pass"] and bar["b5_pass"]
    verdict = "PROMOTE-candidate" if all_gates_pass else "NEGATIVE"
    print(f"\nALL GATING CLAUSES PASS (causal AND B1 AND B3 AND B4-full AND B5): {all_gates_pass}")
    print(f"VERDICT: {verdict}")
    if not all_gates_pass:
        failed = [name for name, ok in (
            ("causal probe", probe_ok), ("B1", bar["b1_pass"]), ("B3", bar["b3_pass"]),
            ("B4 (full)", bar["b4_full_pass"]), ("B5", bar["b5_pass"]),
        ) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    n_configs = len(Z_GRID) + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"({len(Z_GRID)} Step-0 grid + 6 primary compare() + "
          f"{bar['n_configs_promotion_bar'] - 6 - 2} B3 neighbor-grid rows "
          f"[primary's own 2 inner_val rows reused from compare()] + 2 B5 fee-tier)")
    max_ts = max(max_ts_seen)
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary, passed_step0=True,
               probe_ok=probe_ok, promotion_bar=bar, verdict=verdict, n_configs=n_configs,
               max_ts=max_ts)


if __name__ == "__main__":
    main()
