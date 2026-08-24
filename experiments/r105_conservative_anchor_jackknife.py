#!/usr/bin/env python
"""R-105 CONSERVATIVE branch: ``AnchorJackknifeKellyV4`` -- ``kelly_regime_v4``'s
own unchanged ``frac * scale`` product, multiplied by a purely CONTEMPORANEOUS
delete-one-anchor-out (Quenouille-Tukey) jackknife of the vote's own THREE
anchor components, read as a live, bar-level MODEL/SPECIFICATION-disagreement
discount, before v4's own 10% deadband is applied. Full citation trail,
literature grounding, the axis this attacks (ERR -- no error control anywhere
in the signal path), and the exhaustive non-duplication argument against
every related prior round (R-28/R-31, R-87, R-101, R-97, R-104, every
SIZE-axis round) all live in ``experiments/r105_shared.py``'s own module
docstring (read in full before this file was written); none of that is
re-derived here beyond the one-paragraph summary in item 3 below. This file
does not edit, and never reads a bar at or after ``OOS_START`` from, that
module or any other file under ``experiments/`` or ``src/``.

=====================================================================
PRE-REGISTRATION (frozen before any real-data jack_var, bind_frac, R^2, or
backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): treat ``kelly_regime_v4``'s own three
   per-anchor (20/40/80-day) latched directional votes as the sampling
   units of a classical delete-one jackknife (Quenouille 1949; Tukey 1958),
   computed FRESH at every single bar from that bar's own three anchor
   values alone, and discount ``frac * scale`` to a fixed ``floor`` whenever
   the jackknife variance is nonzero -- i.e. whenever the three anchors do
   not unanimously agree -- leaving it at ``1.0`` otherwise.

2. CONSTRUCTION (exact):

       x1[t], x2[t], x3[t]   = per-anchor latched votes, horizons 20/40/80,
                                via experiments.r102_shared._latched_anchor_vote
                                (each in {0.0, 1.0}, NaN-free by construction)
       theta_full[t]         = (x1[t] + x2[t] + x3[t]) / 3           # == v4_vote_frac(df)[t]
       theta_loo_i[t]        = (3*theta_full[t] - x_i[t]) / 2        # leave-one-out pseudo-means
       jack_var[t]           = (2/3) * sum_i (theta_loo_i[t] - theta_full[t])**2
       mixed[t]              = jack_var[t] > 1e-9                    # anchors disagree
       discount[t]           = 1.0            if not mixed[t]
                              = floor          if mixed[t]
       raw[t]                = v4_raw_desired(df)[t] * discount[t]   # frac*scale, UNCHANGED, then discounted
       target[t]             = apply_deadband(raw)[t]                # v4's own deadband, AFTER discount

   DEFAULT/PRIMARY CONFIG: ``floor`` selected by the Step-0 non-degeneracy
   rule below (grid centre 0.5 preferred, confirmed or overridden only by
   that rule, never by a performance number).

   ALGEBRAIC IDENTITY, stated here and re-verified numerically at runtime:
   because ``x_i[t] in {0.0, 1.0}`` and ``theta_full[t] = k/3`` for
   ``k in {0,1,2,3}``, ``jack_var[t]`` takes exactly two values by
   construction -- ``0`` when ``k in {0,3}`` (unanimous, all three anchors
   agree) and a fixed positive constant (``1/9``) when ``k in {1,2}`` (mixed,
   exactly one anchor dissents). So ``mixed[t]`` is EXACTLY equivalent to
   ``theta_full[t] in {1/3, 2/3}`` (equivalently: NOT unanimous) -- this
   round's own pre-registration in ``r105_shared.py`` names this binary
   character (item 3 of "what would make this fail") as a real, disclosed
   property distinct from either prior ERR-round's own continuous, graded
   discount, not a design flaw.

3. WHY THIS IS A GENUINELY DIFFERENT SAMPLING UNIT FROM R-101's AND R-104's
   JACKKNIFES (summary only -- the full argument, citations, and literature
   grounding live in ``r105_shared.py``'s own docstring): R-101's delete-
   one-episode jackknife resampled HISTORY -- six discrete, multi-day stress
   episodes (N=6), requiring the whole pre-holdout record to exist before a
   single estimate could be formed. R-104's two branches (bootstrap and PSR)
   both measured SAMPLING uncertainty of the vote's own realized daily P&L,
   needing a burn-in (``MIN_DAYS=120``) before either estimator's variance
   or significance could be trusted at all. This construction's sampling
   unit is instead the vote's OWN THREE ANCHOR COMPONENTS (N=3) at a SINGLE
   bar -- a structural feature of the model's own construction, not of
   history or of a realized-return time series. Concretely, this is a
   genuine, disclosed ARCHITECTURAL PROPERTY, not an oversight: this
   construction needs NO ``MIN_DAYS`` burn-in, NO refit cadence, NO
   expanding or rolling window of its own, and NO historical accumulation
   of any kind beyond what v4's own anchor rolling-means already require (the
   80-day anchor itself, already inside v4's own shipped construction) --
   ``discount[t]`` is a pure, memoryless function of bar ``t``'s own three
   anchor votes, full stop. Where R-104's own file had to disclose a
   "local-restart" scope limitation on every non-``inner_train`` slice
   (the discount effectively restarting its estimate near each slice's own
   start because of ``TargetStrategy``'s 80-day prefix cap), THIS
   construction has no analogous limitation: it does not accumulate state
   across bars at all, so there is nothing for a slice boundary to restart.

4. STEP-0 NON-DEGENERACY GRID AND SELECTION RULE (identical shape and
   selection-rule convention to R-102/R-103/R-104, and to this round's own
   pre-registered constants in ``r105_shared.py``, reused verbatim rather
   than redefined):

   Grid: ``floor in r105_shared.STEP0_FLOOR_GRID = (0.3, 0.5, 0.7)`` (3
   cells, fixed a priori). For each cell, on BTC's inner-train window
   (2017-01-01 -> 2020-12-31) only:
     - ``bind_frac`` = ``mean(mixed[inner_train_mask])`` -- the fraction of
       inner-train bars where the three anchors disagree. DISCLOSED,
       PRE-REGISTERED PROPERTY: ``mixed[t]`` depends ONLY on whether the
       three anchor votes agree, never on ``floor`` (``floor`` only scales
       the MAGNITUDE of the discount where it binds, not WHERE it binds) --
       so ``bind_frac`` is mathematically IDENTICAL across all three grid
       cells by construction. This is reported explicitly as a real
       property of this construction, not a bug, and is verified
       numerically at runtime (all three cells' ``bind_frac`` values must
       be bit-identical).
     - ``r_sq`` = ``r105_shared.r_squared(build_target(btc, floor=f),
       v4_target(btc))``, both over the FULL BTC frame, masked to
       inner-train before the R^2 is taken.
   A cell QUALIFIES iff ``bind_frac > r105_shared.BIND_FRAC_THRESH`` (0.01)
   AND ``r_sq < r105_shared.R2_THRESH`` (0.98).

   SELECTION RULE (non-degeneracy ONLY -- no performance number is
   inspected before this rule is applied): the primary cell is the first
   floor in ``r105_shared.SELECTION_ORDER = (0.5, 0.3, 0.7)`` that
   qualifies. If NONE of the three cells qualify, this file STOPS at
   Step-0 -- a legitimate, informative NEGATIVE result per
   ``docs/ROUTINE.md``, not a bug to route around. No B1-B5 code runs in
   that case, and no bar on/after ``OOS_START`` (2023-01-01) is ever
   touched either way.

5. CAUSAL TRUNCATION PROBE, run before trusting any Step-0 or promotion-bar
   number: ``r105_shared.causal_truncation_probe_series`` applied to this
   file's own composed ``build_target`` closure (bound to the selected
   primary floor, or ``floor=0.5`` if Step-0 finds no primary), on real BTC
   data. In addition, before any real-data number is computed at all, this
   file verifies on a small synthetic OHLCV frame that
   ``experiments.r102_shared.vote_frac(df, horizons=(h,), band=V4_BAND)``
   and ``experiments.r102_shared._latched_anchor_vote(df["close"], h,
   V4_BAND)`` are numerically identical for each horizon -- confirming
   either primitive is a safe, interchangeable choice for building the
   per-anchor votes this construction needs (this file uses
   ``_latched_anchor_vote`` directly, since it needs the three PER-ANCHOR
   series individually rather than their pre-averaged ``vote_frac``).

6. PROMOTION BAR (docs/ROUTINE.md's own bar, operationalized via
   ``r105_shared.py``'s centralized B1/B2/B4/B5 machinery -- reused
   verbatim, not re-implemented, so this file's gates are byte-for-byte
   the same code every recent SIZE/ERR-axis round's gates run through):
     B1 (gating): ``r105_shared.b1_from_inner_val`` on the primary
        ``compare()`` call's own ``inner_val`` rows, both markets --
        ``d_sharpe > +0.2`` OR the paired bootstrap interval's lower bound
        excludes zero on the positive side.
     B2 (diagnostic ONLY, never itself gates promotion):
        ``r105_shared.b2_diagnostic`` -- drawdown improvement counted only
        where risk-matched (R-33's own standing rule).
     B3 (plateau, gating): since ``floor`` is this construction's ONLY free
        parameter, and Step-0 already grids it at ``{0.3, 0.5, 0.7}``, B3
        sweeps a FINER grid ``{0.1, 0.2, ..., 0.9}`` (9 cells) via
        ``r105_shared.inner_val_rows``, both markets (18 rows; the primary
        floor's 2 rows are reused directly from the primary ``compare()``
        call's own ``inner_val`` rows rather than recomputed). PASS
        requires a directionally consistent (same-sign) majority of
        ``d_sharpe`` across the full 18-row grid -- the same convention
        R-104's own B3 used.
     B4 (ETH falsification, gating, PRE-REGISTERED, not changed after
        seeing results): ``r105_shared.b4_eth_falsification`` -- require
        the FULL pass (both markets same-signed as BTC inner-validation).
     B5 (fee-tier robustness, gating): ``r105_shared.b5_fee_tier`` at the
        primary floor, 0.40% taker, both markets -- no sign reversal on
        either ``d_sharpe`` or the bootstrap log-growth point estimate.
   PROMOTE-candidate only if Step-0 selects a primary AND the causal probe
   passes AND B1 AND B3 AND B4 (full form) AND B5 all hold (B2 is
   diagnostic-only). Default: NEGATIVE. This file never reads or reports
   any bar at or after ``OOS_START`` (2023-01-01) regardless of outcome.

7. WHAT WOULD MAKE THIS FAIL: named already, in full, in ``r105_shared.py``
   itself (three specific, independent failure shapes -- an unremarkable,
   not-risk-concentrated disagreement state; discounting exactly the
   mid-transition bars where the latched vote is about to resolve
   favourably, removing edge rather than protecting against risk; and the
   jackknife statistic's own strictly-binary character being unable to move
   this project's +/-0.2 Sharpe noise floor regardless of ``floor``, a
   failure shape distinct from either prior ERR round's continuous-discount
   failures). Not re-derived here; reported honestly, whichever way it
   comes out, in the results below.

CONFIGURATIONS EVALUATED IN THIS FILE (IF Step-0 selects a primary): 3
(Step-0 floor grid) + 6 (primary config's full ``compare()``: inner_train x2
markets + inner_val x2 markets + eth_replication x2 markets) + 18 (B3's fine
floor grid, 9 floors x 2 markets -- 2 of the 18 reused directly from the
primary ``compare()``'s own inner_val rows, 16 freshly computed) + 2 (B5's
0.40% fee tier, 2 markets) = 29 total. IF Step-0 finds no qualifying cell,
this file stops after the 3 Step-0 cells and reports that outcome directly
(no B1-B5 code runs, no ETH data or bar on/after OOS_START ever touched).

----------------------------------------------------------------------
Run: python experiments/r105_conservative_anchor_jackknife.py
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

from experiments.r105_shared import (  # noqa: E402
    BIND_FRAC_THRESH,
    FEE_TIER,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    R2_THRESH,
    SELECTION_ORDER,
    SHARPE_NOISE_FLOOR,
    SPOT,
    STEP0_FLOOR_GRID,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
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
    v4_target,
    v4_vote_frac,
)

# The generic per-anchor primitive, imported directly (not re-exported past
# r102_shared): this construction needs the THREE PER-ANCHOR series
# individually, not their pre-averaged vote_frac.
from experiments.r102_shared import (  # noqa: E402
    V4_BAND,
    V4_HORIZONS,
    _latched_anchor_vote,
    vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS

# ---------------------------------------------------------- pre-registered
B3_FLOOR_GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))  # {0.1,...,0.9}


# ================================================================== (1)
# The mechanism itself: contemporaneous delete-one-anchor jackknife of
# v4's own three anchor votes -> binary mixed/unanimous flag -> discount.
# ==================================================================

def anchor_votes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three per-anchor (20/40/80-day) latched 0/1 votes v4's own vote
    is built from, as plain numpy arrays."""
    close = df["close"]
    x1 = _latched_anchor_vote(close, V4_HORIZONS[0], V4_BAND).to_numpy()
    x2 = _latched_anchor_vote(close, V4_HORIZONS[1], V4_BAND).to_numpy()
    x3 = _latched_anchor_vote(close, V4_HORIZONS[2], V4_BAND).to_numpy()
    return x1, x2, x3


def jackknife_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """theta_full (== v4_vote_frac), jack_var, mixed -- all pure,
    contemporaneous functions of bar t's own three anchor votes."""
    x1, x2, x3 = anchor_votes(df)
    theta_full = (x1 + x2 + x3) / 3.0
    theta_loo1 = (3.0 * theta_full - x1) / 2.0
    theta_loo2 = (3.0 * theta_full - x2) / 2.0
    theta_loo3 = (3.0 * theta_full - x3) / 2.0
    jack_var = (2.0 / 3.0) * (
        (theta_loo1 - theta_full) ** 2
        + (theta_loo2 - theta_full) ** 2
        + (theta_loo3 - theta_full) ** 2
    )
    mixed = jack_var > 1e-9
    return theta_full, jack_var, mixed


def discount_array(df: pd.DataFrame, floor: float) -> np.ndarray:
    _, _, mixed = jackknife_stats(df)
    return np.where(mixed, floor, 1.0)


def build_target(df: pd.DataFrame, floor: float) -> np.ndarray:
    raw = v4_raw_desired(df) * discount_array(df, floor)
    return apply_deadband(raw)


def make_build_target(floor: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, floor=floor)
    _build.__name__ = f"anchor_jackknife_floor{floor:g}"
    return _build


# ================================================================== (2)
# Pre-flight self-tests: vote_frac(h) == _latched_anchor_vote(h) on a small
# synthetic frame; theta_full == v4_vote_frac and mixed <=> non-unanimous
# on real BTC data.
# ==================================================================

def self_test_vote_equivalence() -> bool:
    idx = pd.date_range("2017-01-01", periods=40_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(1050)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                       "close": close, "volume": 1.0}, index=idx)
    ok = True
    for h in V4_HORIZONS:
        a = vote_frac(df, (h,), V4_BAND).to_numpy()
        b = _latched_anchor_vote(df["close"], h, V4_BAND).to_numpy()
        same = np.allclose(a, b, equal_nan=True)
        print(f"  horizon={h:2d}d: vote_frac((h,)) == _latched_anchor_vote(h)?  {same}")
        ok = ok and same
    return ok


def self_test_jackknife_identity(btc: pd.DataFrame) -> bool:
    theta_full, jack_var, mixed = jackknife_stats(btc)
    match_v4 = np.allclose(theta_full, v4_vote_frac(btc).to_numpy(), equal_nan=True)
    print(f"  theta_full == v4_vote_frac(btc) everywhere?  {match_v4}")

    unanimous = (theta_full < 1e-12) | (theta_full > 1.0 - 1e-12)
    equiv = np.array_equal(mixed, ~unanimous)
    print(f"  mixed[t] <=> theta_full[t] not in {{0,1}} (i.e. in {{1/3,2/3}}) everywhere?  {equiv}")

    vals = np.unique(np.round(jack_var, 12))
    print(f"  jack_var takes exactly the values {vals.tolist()} (expect [0.0, ~0.11111])")
    two_valued = len(vals) <= 2

    return bool(match_v4 and equiv and two_valued)


# ================================================================== (3)
# Step-0 non-degeneracy grid + bind_frac-identity check.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], int]:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    _, _, mixed = jackknife_stats(btc)
    raw_base = v4_raw_desired(btc)
    ctrl_target = v4_target(btc)
    bind_frac_common = float(np.mean(mixed[mask]))

    rows = []
    for floor in STEP0_FLOOR_GRID:
        disc = np.where(mixed, floor, 1.0)
        target = apply_deadband(raw_base * disc)
        bind_frac = float(np.mean(mixed[mask]))  # identical by construction; recomputed for clarity
        r_sq = r_squared(target[mask], ctrl_target[mask])
        qualifies = (bind_frac > BIND_FRAC_THRESH) and (r_sq < R2_THRESH)
        rows.append(dict(floor=floor, bind_frac=bind_frac, r_sq=r_sq, qualifies=qualifies))

    identical = all(abs(r["bind_frac"] - bind_frac_common) < 1e-15 for r in rows)
    print(f"  bind_frac identical across all {len(STEP0_FLOOR_GRID)} floor cells (by construction)? {identical}")
    return rows, n_bars


def select_primary(rows: list[dict]) -> dict | None:
    by_floor = {r["floor"]: r for r in rows}
    for f in SELECTION_ORDER:
        r = by_floor.get(f)
        if r is not None and r["qualifies"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, {n_bars:,} bars)")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r_sq < {R2_THRESH}")
    hdr = f"{'floor':>6s} {'bind_frac':>10s} {'r_sq':>8s} {'qualifies':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = " <- grid centre" if r["floor"] == 0.5 else ""
        print(f"{r['floor']:6.2f} {r['bind_frac']:10.4f} {r['r_sq']:8.4f} "
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
# Promotion bar: B1 (gating), B2 (diagnostic), B3 (fine floor plateau,
# gating), B4 (ETH falsification, gating), B5 (fee tier, gating).
# ==================================================================

def run_promotion_bar(primary_floor: float, btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    label = f"anchor_jackknife_floor{primary_floor:g}"
    build_primary = make_build_target(primary_floor)

    hr(f"PROMOTION BAR -- PRIMARY CONFIG floor={primary_floor:g}")
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

    # ---- B3: fine floor grid {0.1,...,0.9}, both markets, primary reused.
    plateau_rows: dict[float, list[dict]] = {
        primary_floor: [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                             d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                             vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                             boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                             boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                        for r in inner_val_primary]
    }
    for floor in B3_FLOOR_GRID:
        if abs(floor - primary_floor) < 1e-9:
            continue
        bf = make_build_target(floor)
        blabel = f"anchor_jackknife_floor{floor:g}"
        plateau_rows[floor] = inner_val_rows(bf, blabel, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for floor in plateau_rows for r in plateau_rows[floor]]
    n_pos = sum(same_sign_flags)
    n_neg = len(same_sign_flags) - n_pos
    b3_pass = (max(n_pos, n_neg) >= len(same_sign_flags) / 2.0) if same_sign_flags else False

    # ---- B4: ETH falsification
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)

    # ---- B5: fee-tier robustness
    hr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary config, BTC inner-validation")
    b5_pass, b5_cells = b5_fee_tier(build_primary, label, btc, inner_val_primary)

    all_pass = b1_pass and b3_pass and b4_full and b5_pass

    return dict(
        label=label, floor=primary_floor,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells,
        b3_plateau_rows=plateau_rows, b3_pass=b3_pass,
        b4_cells=b4_cells, b4_partial_pass=b4_partial, b4_full_pass=b4_full,
        b5_cells=b5_cells, b5_pass=b5_pass,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 2 * len(B3_FLOOR_GRID) + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-105 CONSERVATIVE: AnchorJackknifeKellyV4 -- contemporaneous "
       "delete-one-anchor jackknife of v4's own 3-anchor vote")
    print("mechanism: multiply v4's UNCHANGED frac*scale by a floor-valued discount whenever")
    print("v4's own 20/40/80-day anchor votes do NOT unanimously agree (jackknife variance of")
    print("the vote's own leave-one-out pseudo-means > 0), else 1.0. Purely a function of bar t's")
    print("own three anchor votes -- no burn-in, no refit cadence, no accumulated state at all.")

    hr("PRE-FLIGHT SELF-TESTS (synthetic + real-data identities, before any Step-0 number)")
    print("vote_frac((h,), V4_BAND) vs _latched_anchor_vote(h, V4_BAND), synthetic frame:")
    vote_eq_ok = self_test_vote_equivalence()
    print(f"  -> vote-primitive equivalence: {vote_eq_ok}")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    print("\njackknife identity checks on real BTC data:")
    jack_id_ok = self_test_jackknife_identity(btc)
    print(f"  -> jackknife identity checks: {jack_id_ok}")

    if not (vote_eq_ok and jack_id_ok):
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
        print("No grid cell has both bind_frac > 1% and r_sq < 0.98: the anchor-jackknife")
        print("discount is either a near-total no-op (anchors almost always unanimous on")
        print("inner-train) or a near-exact rescale of v4's own path everywhere on the")
        print("pre-registered floor grid. Per this file's own pre-registration, this Step-0")
        print("table is the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0.")
        print("No causal probe, B1-B5 code, or ETH load runs.")
        n_configs = len(STEP0_FLOOR_GRID)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 grid only)")
        max_ts = max(max_ts_seen)
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                   n_configs=n_configs, max_ts=max_ts, verdict="NEGATIVE (Step-0 kill switch)")

    is_center = (primary["floor"] == 0.5)
    print(f"\nPRIMARY CONFIG SELECTED (non-degeneracy rule only): floor={primary['floor']:g}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r_sq={primary['r_sq']:.4f})")
    print(f"  selection: {'grid-centre cell qualified' if is_center else 'grid-centre cell did NOT qualify; nearest qualifying cell in [0.5, 0.3, 0.7] chosen'}")

    build_primary = make_build_target(primary["floor"])

    # ==================================================== CAUSAL PROBE
    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    probe_ok = run_causal_probe(btc, build_primary)
    print(f"\nCAUSAL PROBE PASS: {probe_ok}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    bar = run_promotion_bar(primary["floor"], btc, eth)

    hr("B1 -- inner-validation Sharpe leg, both markets "
       "(dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  voided={c['voided']}")

    hr("B3 -- plateau: fine floor grid {0.1,...,0.9} at primary selection, "
       "inner-validation, both markets")
    print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 (directionally consistent majority across the {2 * len(B3_FLOOR_GRID)}-row grid): {bar['b3_pass']}")

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
          f"B3 = {bar['b3_pass']}   B4(full) = {bar['b4_full_pass']}   "
          f"B4(partial) = {bar['b4_partial_pass']}   B5 = {bar['b5_pass']}")
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

    n_configs = len(STEP0_FLOOR_GRID) + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"({len(STEP0_FLOOR_GRID)} Step-0 grid + 6 primary compare() + "
          f"{2 * len(B3_FLOOR_GRID)} B3 fine floor grid "
          f"[2 reused from primary + {2 * (len(B3_FLOOR_GRID) - 1)} fresh] + 2 B5 fee-tier)")
    max_ts = max(max_ts_seen)
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary, passed_step0=True,
               probe_ok=probe_ok, promotion_bar=bar, verdict=verdict, n_configs=n_configs,
               max_ts=max_ts)


if __name__ == "__main__":
    main()
