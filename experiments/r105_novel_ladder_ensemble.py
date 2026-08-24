#!/usr/bin/env python
"""R-105 NOVEL branch: ``AltLadderEnsembleKellyV4`` -- ``kelly_regime_v4``'s own
directional VOTE, read against a pre-registered ENSEMBLE of alternative
anchor-ladder specifications of that same vote, used as a live, causal,
bar-level specification-DISAGREEMENT discount on ``frac * scale``.

EXACT CONSTRUCTION. Ensemble membership is a fixed, a-priori geometric
doubling-ladder family (the same structure ``kelly_regime_v4``'s own
docstring cites: MACD 12/26, HAR daily/weekly/monthly) -- 5 base horizons
``B in (10, 15, 20, 25, 30)`` days, each expanding to a ladder
``(B, 2B, 4B)``: ``(10,20,40)``, ``(15,30,60)``, ``(20,40,80)`` (the shipped
primary ladder), ``(25,50,100)``, ``(30,60,120)``. NOT swept or fitted. For
each member ladder ``m``, ``frac_m = r105_shared.vote_frac(df, horizons=m,
band=V4_BAND)`` (the existing generic anchor-vote primitive, reused
verbatim). At every bar ``t``:

  1. ``disagreement[t] = mean over the 4 NON-PRIMARY members of
     |frac_m[t] - frac_primary[t]|`` in [0, 1].
  2. A DERIVED reference scale, ``disp_ref[t] = the 90th percentile of
     disagreement[0..t-1]`` -- an EXPANDING quantile using only bars
     strictly before ``t`` (``disagreement.shift(1).expanding(min_periods=
     BURN_IN_BARS).quantile(0.90)``), with a ``MIN_DAYS=120``
     calendar-day burn-in during which ``discount[t] = 1.0`` identically
     (R-104's own convention, for direct cross-round comparability).
  3. ``discount[t] = clip(1 - disagreement[t] / max(disp_ref[t], 1e-9),
     floor, 1.0)`` past the burn-in, else 1.0.
  4. ``build_target(df, floor) = apply_deadband(v4_raw_desired(df) *
     discount_array(df, floor))`` -- ``vote`` and ``scale`` themselves are
     completely UNTOUCHED. NONE of the 4 non-primary ladders' own signals
     are ever traded, blended, or selected between -- they exist ONLY to
     compute the disagreement statistic that discounts the SHIPPED
     20/40/80 vote's own exposure.

Literature grounding (Baltas & Kosowski 2013's time-scale-diversification
finding; Raftery et al. 2005's ensemble-spread-as-uncertainty principle),
the full citation trail, and the exhaustive non-duplication argument
against every related prior round (R-28/R-31, R-87, R-101, R-97, R-104,
every SIZE-axis round, and this round's own CONSERVATIVE sibling) all live
in ``experiments/r105_shared.py``'s own module docstring (read in full
before this file was written); not re-derived here. This file does not
edit, and never reads a bar at or after ``r105_shared.OOS_START`` from,
that module or any other file under ``experiments/`` or ``src/``.

=====================================================================
PRE-REGISTRATION (frozen before any real-data discount, bind_frac, R^2, or
backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

STEP-0 GRID: ``floor in r105_shared.STEP0_FLOOR_GRID = (0.3, 0.5, 0.7)`` at
the fixed ``MIN_DAYS=120`` burn-in, on BTC's inner-train window only. A
cell qualifies iff ``bind_frac > r105_shared.BIND_FRAC_THRESH`` AND
``r_sq < r105_shared.R2_THRESH``. Primary selection:
``r105_shared.SELECTION_ORDER = (0.5, 0.3, 0.7)``, first qualifying floor
in that order; STOP (NEGATIVE, no B1-B5) if none qualify.

DISCLOSED, PRE-REGISTERED PROPERTY OF THIS CONSTRUCTION (not a bug):
because ``discount[t] < 1.0`` exactly whenever ``disagreement[t] > 0`` AND
``disp_ref[t] > 1e-9`` -- a condition that does not depend on ``floor``'s
magnitude, only on whether the members disagree at all -- ``bind_frac`` is
expected to come out IDENTICAL across all three floor cells. ``floor``
only controls how DEEP the clip goes once binding, never WHETHER it
binds. This is verified numerically below, not assumed.

PROMOTION BAR: same shape as every SIZE/ERR-axis round since R-89, via
``r105_shared.compare()`` unchanged.
  B1 (gating): ``r105_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets.
  B2 (diagnostic ONLY, never gates): ``r105_shared.b2_diagnostic``.
  B3 (plateau, gating): sweep ``r105_shared.B3_MIN_DAYS_GRID = (60, 120,
     250)`` -- the burn-in parameter -- at the selected primary floor,
     inner_val, both markets. The ``MIN_DAYS=120`` cell is reused directly
     from the primary ``compare()``'s own inner_val rows rather than
     recomputed. PASS requires a directionally consistent (same-sign
     majority) region across the 6 cells, not an isolated spike.
  B4 (ETH falsification, gating, pre-registered): ``r105_shared.
     b4_eth_falsification`` -- require FULL pass (both markets same-signed
     as BTC inner_val).
  B5 (fee-tier robustness, gating): ``r105_shared.b5_fee_tier`` at 0.40%
     taker, primary cell, BTC inner_val, both markets -- no sign reversal.
PROMOTE-candidate only if the causal-truncation probe passes AND B1 AND B3
AND B4 (full) AND B5 all hold (B2 is diagnostic-only). Default: NEGATIVE.
No threshold or decision rule is changed after seeing a number.

CAUSAL SAFETY: ``r105_shared.causal_truncation_probe_series`` on the
composed ``build_target`` (primary floor), PLUS an explicit, additional
check that ``disp_ref[t]`` is algebraically reproducible from
``disagreement[:t]`` alone and is unaffected by perturbing
``disagreement[t:]`` -- because this branch's normalization is a
NONLINEAR expanding-quantile statistic (not a simple linear recursion),
this extra check is warranted beyond the generic truncation probe alone.

SCOPE LIMITATION, DISCLOSED, NOT PATCHED AROUND (identical in shape to
R-103's RLS branch and both of R-104's branches -- a documented trap this
file avoids repeating): ``TargetStrategy.warmup`` (imported unchanged from
``r105_shared``) defaults to 80 calendar days
(``80 * BARS_PER_DAY + 10`` bars), and ``tradebot.window.run_period`` hands
``build_target`` only ``frame = df.iloc[lo - prefix : hi]`` -- a prefix
capped at that 80-day ``warmup``, not this file's own ``MIN_DAYS=120``
burn-in. ``inner_train`` and ``eth_replication`` both start at their own
frame's TRUE beginning (``lo=0``, so ``prefix=0`` regardless of
``warmup``), so they are unaffected. ``inner_val`` (2021-01-01, mid-frame)
is NOT: it receives only ~80 prior days, less than the 120-day burn-in, so
the expanding disagreement/disp_ref computation effectively RESTARTS
close to ``inner_val``'s own start rather than continuing continuously
from 2017 -- ``discount[t] = 1.0`` (parity with v4) for roughly the first
40 days of every ``inner_val``-based B1/B3/B5 cell, and ``disp_ref``
afterward is built from a locally-restarted (not continuous) history. The
same failure mode named in R-103/R-104's own module docstrings
(inflating ``warmup`` globally silences ``on_bar`` entirely on any frame
shorter than the sentinel -- 0 trades, flat equity) rules out "fixing"
this by raising ``warmup``, so it is disclosed here instead, exactly as
R-104's own two branches did. Net effect: this UNDERSTATES rather than
overstates any genuine effect on ``inner_val``-based cells, and it is most
severe for B3's ``burn_in_bars=250d`` cell (over half of ``inner_val``'s
own measured window sits inside that longer burn-in given the same
capped 80-day prefix).

WHAT WOULD MAKE THIS FAIL: named in full in ``r105_shared.py``'s own
module docstring (three independent failure shapes: disagreement may be a
common, unremarkable state rather than concentrated around genuine regime
transitions, reproducing the R-87/R-104 "real but inert" pattern by a
third estimator; discounting mid-transition disagreement could remove
part of the vote's edge rather than protect against risk, as R-59/R-60
found when touching the vote's timing; or, since ``bind_frac`` cannot
distinguish floors by construction, the entire STEP0_FLOOR_GRID sweep may
carry no non-degeneracy information beyond its first cell). Reported
honestly below, whichever way it comes out.

CONFIGURATIONS EVALUATED IN THIS FILE (IF Step-0 selects a primary): 3
(Step-0 floor grid) + 6 (primary config's full ``compare()``: inner_train
x2 markets + inner_val x2 markets + eth_replication x2 markets) + 6 (B3's
burn-in-days grid, 3 configs x 2 markets -- 2 of the 6 reused directly
from the primary ``compare()``'s own inner_val rows, 4 freshly computed)
+ 2 (B5's 0.40% fee tier, 2 markets) = 17 total. IF Step-0 finds no
qualifying cell, this file stops after the 3 Step-0 cells.

USAGE
-----
    python experiments/r105_novel_ladder_ensemble.py
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
    B3_MIN_DAYS_GRID,
    BARS_PER_DAY,
    BIND_FRAC_THRESH,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    R2_THRESH,
    SELECTION_ORDER,
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
from experiments.r102_shared import V4_BAND, vote_frac  # noqa: E402

# ---------------------------------------------------------- pre-registered
MIN_DAYS = 120                      # burn-in (calendar days), R-104's own convention
BURN_IN_BARS = MIN_DAYS * BARS_PER_DAY
FLOOR_DEFAULT = 0.5                 # grid centre, confirmed/overridden only by Step-0

BASES = (10, 15, 20, 25, 30)
PRIMARY_BASE = 20
NON_PRIMARY_BASES = tuple(b for b in BASES if b != PRIMARY_BASE)  # (10, 15, 25, 30)
LADDERS = {b: (b, 2 * b, 4 * b) for b in BASES}
assert LADDERS[PRIMARY_BASE] == (20, 40, 80), LADDERS[PRIMARY_BASE]


# ================================================================== (1)
# The mechanism itself: 5-member anchor-ladder ensemble, disagreement
# against the primary (20,40,80) member, expanding-quantile normalization,
# clip-discount on v4's own UNCHANGED frac*scale.
# ==================================================================

def compute_member_fracs(df: pd.DataFrame) -> dict[int, pd.Series]:
    """One ``vote_frac`` call per ensemble member (5 total, including the
    primary ladder itself, computed identically to how ``v4_vote_frac``
    would compute it)."""
    return {b: vote_frac(df, horizons=LADDERS[b], band=V4_BAND) for b in BASES}


def compute_disagreement(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[int, pd.Series]]:
    """``(disagreement, frac_primary, members)``. ``disagreement[t]`` is the
    mean absolute distance of the 4 non-primary members from the primary
    member's vote at bar t -- in [0, 1] since every ``frac_*`` is itself in
    [0, 1]. None of the 4 non-primary members' own signal is ever used for
    anything but this statistic."""
    members = compute_member_fracs(df)
    frac_primary = members[PRIMARY_BASE]
    diffs = [(members[b] - frac_primary).abs() for b in NON_PRIMARY_BASES]
    disagreement = sum(diffs) / len(NON_PRIMARY_BASES)
    return disagreement, frac_primary, members


def compute_disp_ref(disagreement: pd.Series, burn_in_bars: int = BURN_IN_BARS) -> pd.Series:
    """Expanding 90th percentile of disagreement[0..t-1], strictly causal:
    ``.shift(1)`` moves bar t's own disagreement value out of the window
    entirely, and ``.expanding(min_periods=burn_in_bars)`` requires
    `burn_in_bars` PRIOR bars to exist before producing a non-NaN value --
    the first non-NaN value appears exactly at position `burn_in_bars`,
    built only from disagreement[0 .. burn_in_bars-1]."""
    return disagreement.shift(1).expanding(min_periods=burn_in_bars).quantile(0.90)


def discount_from(disagreement: pd.Series, disp_ref: pd.Series, floor: float,
                  burn_in_bars: int = BURN_IN_BARS) -> np.ndarray:
    n = len(disagreement)
    idx = np.arange(n)
    dis = disagreement.to_numpy()
    ref = disp_ref.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 - dis / np.maximum(ref, 1e-9)
    raw = np.clip(raw, floor, 1.0)
    discount = np.where(idx < burn_in_bars, 1.0, raw)
    # Anywhere disp_ref is still NaN past the nominal burn-in (should not
    # happen given the construction above, guarded regardless): fall back
    # to no discount rather than propagate a NaN into v4_raw_desired.
    discount = np.where(np.isfinite(discount), discount, 1.0)
    return discount


def discount_array(df: pd.DataFrame, floor: float, burn_in_bars: int = BURN_IN_BARS) -> np.ndarray:
    disagreement, _frac_primary, _members = compute_disagreement(df)
    disp_ref = compute_disp_ref(disagreement, burn_in_bars)
    return discount_from(disagreement, disp_ref, floor, burn_in_bars)


def build_target(df: pd.DataFrame, floor: float = FLOOR_DEFAULT,
                 burn_in_bars: int = BURN_IN_BARS) -> np.ndarray:
    """``build_target(df, floor) = apply_deadband(v4_raw_desired(df) *
    discount_array(df, floor))`` -- the identical SLOT convention every
    recent SIZE/ERR-axis round uses. ``vote`` and ``scale`` are completely
    untouched; only the frac*scale PRODUCT is discounted, before v4's own
    deadband is applied (v4's composition order, unchanged)."""
    discount = discount_array(df, floor, burn_in_bars)
    raw = v4_raw_desired(df) * discount
    return apply_deadband(raw)


def make_build_target(floor: float, burn_in_bars: int = BURN_IN_BARS):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, floor=floor, burn_in_bars=burn_in_bars)
    _build.__name__ = f"alt_ladder_ens_floor{floor:g}_burn{burn_in_bars // BARS_PER_DAY}d"
    return _build


# ================================================================== (2)
# Step-0 non-degeneracy grid + explicit bind_frac-identical check.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> dict:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    disagreement, frac_primary, members = compute_disagreement(btc)

    # Sanity check: the (20,40,80) member IS the primary vote, numerically.
    v4_direct = v4_vote_frac(btc).to_numpy()
    primary_matches = bool(np.allclose(frac_primary.to_numpy(), v4_direct,
                                       equal_nan=True, atol=1e-12))

    disp_ref = compute_disp_ref(disagreement, BURN_IN_BARS)
    raw_base = v4_raw_desired(btc)
    ctrl_target = v4_target(btc)

    rows = []
    for floor in STEP0_FLOOR_GRID:
        discount = discount_from(disagreement, disp_ref, floor, BURN_IN_BARS)
        target = apply_deadband(raw_base * discount)
        bind_frac = float(np.mean(discount[mask] < 1.0 - 1e-9))
        r_sq = r_squared(target[mask], ctrl_target[mask])
        qualifies = (bind_frac > BIND_FRAC_THRESH) and (r_sq < R2_THRESH)
        rows.append(dict(floor=floor, bind_frac=bind_frac, r_sq=r_sq, qualifies=qualifies))

    bind_fracs = [r["bind_frac"] for r in rows]
    bind_frac_identical = bool(np.allclose(bind_fracs, bind_fracs[0], atol=1e-12))

    return dict(rows=rows, n_bars=n_bars, primary_matches=primary_matches,
               bind_frac_identical=bind_frac_identical,
               disagreement=disagreement, disp_ref=disp_ref)


def select_primary(rows: list[dict]) -> dict | None:
    by_floor = {r["floor"]: r for r in rows}
    for f in SELECTION_ORDER:
        r = by_floor.get(f)
        if r is not None and r["qualifies"]:
            return r
    return None


def print_step0_table(step0: dict) -> None:
    rows, n_bars = step0["rows"], step0["n_bars"]
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars, MIN_DAYS={MIN_DAYS} burn-in)")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r_sq < {R2_THRESH}")
    hdr = f"{'floor':>6s} {'bind_frac':>10s} {'r_sq':>8s} {'qualifies':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = "  <- grid centre" if r["floor"] == 0.5 else ""
        print(f"{r['floor']:6.2f} {r['bind_frac']:10.4f} {r['r_sq']:8.4f} "
              f"{'YES' if r['qualifies'] else 'no':>10s}{tag}")
    print(f"\nsanity check -- (20,40,80) member IS v4_vote_frac numerically: "
          f"{step0['primary_matches']}")
    print(f"disclosed property -- bind_frac identical across all 3 floors "
          f"(discount<1 iff disagreement>0 and disp_ref>0, independent of floor's "
          f"magnitude): {step0['bind_frac_identical']}  "
          f"(values: {[f'{b:.4f}' for b in [r['bind_frac'] for r in rows]]})")


# ================================================================== (3)
# Explicit disp_ref causality check (beyond the generic truncation probe):
# an EXPANDING QUANTILE is a nonlinear function of its window, so this
# verifies directly, algebraically, that disp_ref[t] depends ONLY on
# disagreement[0..t-1] -- never on disagreement[t] or any later bar.
# ==================================================================

def explicit_disp_ref_causality_check(btc: pd.DataFrame, n_samples: int = 6) -> dict:
    disagreement, _frac_primary, _members = compute_disagreement(btc)
    disp_ref = compute_disp_ref(disagreement, BURN_IN_BARS)
    dis_vals = disagreement.to_numpy()
    n = len(dis_vals)

    rng = np.random.default_rng(105)
    candidates = np.arange(BURN_IN_BARS, n)
    sample_t = sorted(rng.choice(candidates, size=min(n_samples, len(candidates)),
                                 replace=False).tolist())

    results = []
    for t in sample_t:
        manual = float(np.quantile(dis_vals[:t], 0.90))
        actual = float(disp_ref.iloc[t])
        matches_manual = bool(np.isclose(manual, actual, atol=1e-9, rtol=1e-9))

        # Perturb disagreement[t:] (everything at or after bar t) and check
        # disp_ref[t] is completely unaffected.
        dis_pert = dis_vals.copy()
        dis_pert[t:] = dis_pert[t:] + 7.0
        disp_ref_pert = (pd.Series(dis_pert, index=disagreement.index)
                         .shift(1).expanding(min_periods=BURN_IN_BARS).quantile(0.90))
        unaffected = bool(np.isclose(float(disp_ref_pert.iloc[t]), actual, atol=1e-9, rtol=1e-9))

        results.append(dict(t=int(t), manual_quantile=manual, actual_disp_ref=actual,
                           matches_manual_recompute=matches_manual,
                           unaffected_by_future_perturbation=unaffected))

    all_ok = all(r["matches_manual_recompute"] and r["unaffected_by_future_perturbation"]
                for r in results)
    return dict(results=results, all_ok=all_ok)


def print_explicit_causality_check(check: dict) -> None:
    hr("EXPLICIT disp_ref CAUSALITY CHECK (nonlinear expanding-quantile, sampled bars)")
    hdr = f"{'t':>10s} {'manual_q90':>12s} {'actual':>12s} {'recompute_ok':>13s} {'future_pert_ok':>15s}"
    print(hdr)
    print("-" * len(hdr))
    for r in check["results"]:
        print(f"{r['t']:10d} {r['manual_quantile']:12.6f} {r['actual_disp_ref']:12.6f} "
              f"{str(r['matches_manual_recompute']):>13s} {str(r['unaffected_by_future_perturbation']):>15s}")
    print(f"\nALL SAMPLED BARS PASS (disp_ref[t] reproducible from disagreement[:t] alone, "
          f"AND unaffected by perturbing disagreement[t:]): {check['all_ok']}")


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# burn-in-days sweep), B4 (gating falsification), B5 (gating fee tier).
# ==================================================================

def run_b3(primary_floor: float, inner_val_primary_rows: list[dict],
          btc: pd.DataFrame) -> tuple[dict[int, list[dict]], bool]:
    plateau_rows: dict[int, list[dict]] = {}
    for md in B3_MIN_DAYS_GRID:
        if md == MIN_DAYS:
            plateau_rows[md] = [dict(label=f"alt_ladder_ens_floor{primary_floor:g}_burn{md}d",
                                     market=r["market"], d_sharpe=r["d_sharpe"], d_dd=r["d_dd"],
                                     exposure_ratio=r["exposure_ratio"], vol_ratio=r["vol_ratio"],
                                     risk_matched=r["risk_matched"],
                                     boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                     boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                                for r in inner_val_primary_rows]
        else:
            bf = make_build_target(primary_floor, burn_in_bars=md * BARS_PER_DAY)
            label = f"alt_ladder_ens_floor{primary_floor:g}_burn{md}d"
            plateau_rows[md] = inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


def run_promotion_bar(primary_floor: float, btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    build_primary = make_build_target(primary_floor, BURN_IN_BARS)
    label = f"alt_ladder_ens_floor{primary_floor:g}_burn{MIN_DAYS}d"

    hr(f"PROMOTION BAR -- PRIMARY CELL floor={primary_floor:g}, MIN_DAYS={MIN_DAYS}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                  markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3(primary_floor, inner_val_primary, btc)
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)
    b5_pass, b5_cells = b5_fee_tier(build_primary, label, btc, inner_val_primary)

    all_pass = b1_pass and b3_pass and b4_full and b5_pass

    return dict(
        label=label, floor=primary_floor, compare_rows=rows,
        inner_val_primary=inner_val_primary, eth_primary=eth_primary,
        b1_pass=b1_pass, b1_cells=b1_cells,
        b2_pass=b2_pass, b2_cells=b2_cells,
        b3_pass=b3_pass, b3_rows=b3_rows,
        b4_partial=b4_partial, b4_full=b4_full, b4_cells=b4_cells,
        b5_pass=b5_pass, b5_cells=b5_cells,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 6 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-105 NOVEL: AltLadderEnsembleKellyV4 -- alternative anchor-ladder "
       "ENSEMBLE-DISAGREEMENT discount on v4's own frac*scale")
    print("mechanism: 5 pre-registered alternative anchor-ladder specifications of v4's own")
    print("directional vote (geometric doubling-ladder family, bases 10/15/20/25/30 days),")
    print("read as a live spec-disagreement statistic against the SHIPPED (20,40,80) member;")
    print("expanding-quantile-normalized, [floor,1.0]-clipped, multiplied onto v4's UNCHANGED")
    print("frac*scale before v4's own deadband. None of the 4 non-primary ladders' signals")
    print("are ever traded -- they exist only to compute the disagreement statistic.")
    print(f"\nensemble members: {LADDERS}")
    print(f"primary (traded) member: base={PRIMARY_BASE} -> ladder={LADDERS[PRIMARY_BASE]}")
    print(f"non-primary (disagreement-only) members: bases={NON_PRIMARY_BASES}")
    print(f"MIN_DAYS={MIN_DAYS} calendar days = {BURN_IN_BARS:,} bars burn-in "
          f"(BARS_PER_DAY={BARS_PER_DAY})")
    print(f"TargetStrategy.warmup LEFT AT SHARED DEFAULT ({TargetStrategy.warmup:,} bars, "
          f"{TargetStrategy.warmup / BARS_PER_DAY:g} calendar days) -- NOT inflated (R-103/R-104's "
          f"documented trap: a global sentinel silences on_bar on any shorter frame). Disclosed "
          f"consequence: inner_val's prefix (~80d) is shorter than this file's own MIN_DAYS=120 "
          f"burn-in, so inner_val-based cells see a locally-RESTARTED disagreement/disp_ref history, "
          f"not the continuous-since-2017 one inner_train/eth_replication see -- see pre-registration's "
          f"own SCOPE LIMITATION paragraph.")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY KILL SWITCH (run BEFORE any Sharpe/compare() number)")
    step0 = step0_grid(btc)
    print_step0_table(step0)

    causality_check = explicit_disp_ref_causality_check(btc)
    print_explicit_causality_check(causality_check)

    primary = select_primary(step0["rows"])

    if primary is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell has both bind_frac > 1% and r_sq < 0.98: the alt-ladder-ensemble")
        print("disagreement discount is either a near-total no-op or a near-exact rescale of")
        print("v4's own path everywhere on the pre-registered floor grid -- the R-87/R-104-shaped")
        print("'inert' failure mode this round's own pre-registration named as a live possibility,")
        print("by a third (specification-disagreement) estimator. Per this file's own")
        print("pre-registration, this Step-0 table (plus the causality check above) is the")
        print("branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0. No promotion-bar")
        print("code runs, and no ETH data or bar on/after OOS_START is ever touched.")

        hr("VERDICT")
        print("Step-0 (3-cell floor grid, bind_frac>1% AND r_sq<0.98): FAIL (no cell qualifies)")
        print(f"explicit disp_ref causality check: {causality_check['all_ok']}")
        print("B1: NOT COMPUTED (Step-0 kill switch)")
        print("B2: NOT COMPUTED (Step-0 kill switch)")
        print("B3: NOT COMPUTED (Step-0 kill switch)")
        print("B4: NOT COMPUTED (Step-0 kill switch)")
        print("B5: NOT COMPUTED (Step-0 kill switch)")
        print("VERDICT: NEGATIVE (Step-0 kill switch)")

        n_configs = len(STEP0_FLOOR_GRID)
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated (total): {n_configs} (3 Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0=step0, primary=None, passed_step0=False,
                   causality_check=causality_check, n_configs=n_configs, max_ts=max_ts,
                   verdict="NEGATIVE (Step-0 kill switch)")

    is_center = (primary["floor"] == 0.5)
    print(f"\nPRIMARY CELL SELECTED (non-degeneracy rule only): floor={primary['floor']:g}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r_sq={primary['r_sq']:.4f})")
    print(f"  selection: {'grid-centre cell qualified' if is_center else 'grid-centre cell did NOT qualify; nearest qualifying cell in [0.5, 0.3, 0.7] chosen'}")

    build_primary = make_build_target(primary["floor"], BURN_IN_BARS)

    # ==================================================== CAUSAL PROBE
    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    print(f"causal_truncation_probe_series({build_primary.__name__}, btc):")
    try:
        probe_ok = causal_truncation_probe_series(build_primary, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    causal_safety_ok = bool(probe_ok) and causality_check["all_ok"]
    print(f"\nCAUSAL SAFETY (truncation probe AND explicit disp_ref check) PASS: {causal_safety_ok}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    bar = run_promotion_bar(primary["floor"], btc, eth)

    hr("B1 -- inner-validation, both markets (dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hr("B3 -- plateau: burn-in-days sweep {60, 120, 250} at primary floor, inner-validation, both markets")
    print_plateau_table(bar["b3_rows"])
    print(f"\nB3 (directionally consistent majority across the 6-cell grid): {bar['b3_pass']}")

    hr("B4 -- ETH falsification (pre-registered)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PARTIAL PASS (>=1 market): {bar['b4_partial']}   B4 FULL PASS (both markets): {bar['b4_full']}")

    hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier d_sharpe={c['d_sharpe']:+.4f}  "
              f"standard-fee d_sharpe={c['base_d_sharpe']:+.4f}  "
              f"fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hr("VERDICT")
    print(f"causal safety (probe AND explicit disp_ref check): {causal_safety_ok}")
    print(f"B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4_full={bar['b4_full']}  B5={bar['b5_pass']}")
    all_applicable_pass = (causal_safety_ok and bar["b1_pass"] and bar["b3_pass"] and
                          bar["b4_full"] and bar["b5_pass"])
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")
    if not causal_safety_ok:
        print("NOTE: verdict driven (at least in part) by a causal-safety check failure -- "
              "a lookahead is a bug report first, per docs/ROUTINE.md's own precedence.")

    n_configs = 3 + bar["n_configs_promotion_bar"]
    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(3 Step-0 grid + 6 primary-cell compare() + 6 B3 plateau [3 burn-in values x 2 markets, "
          f"2 reused from primary] + 2 B5 fee-tier)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0=step0, causality_check=causality_check,
               primary=primary, passed_step0=True, probe_ok=probe_ok,
               causal_safety_ok=causal_safety_ok, promotion_bar=bar, verdict=verdict,
               n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
