#!/usr/bin/env python
"""R-122 NOVEL branch: ``SyntheticRegimeswitchKnnNoveltyBrake`` -- R-109's own
nonparametric k-nearest-neighbour distributional-novelty statistic, but
scored against a POOLED, EXTERNALLY-CALIBRATED SYNTHETIC reference panel
(R-119's 3-state bull/chop/bear regime-switching jump-diffusion generator,
whose bear-state severity/duration is calibrated from an external BTC crash
catalogue, never fit to this project's own price data) instead of BTC's own
trailing 730-day real-price reference window -- the richer of this round's
two pairings, deliberately crossing R-119's structurally different generator
with R-109's own nonparametric novel-branch algorithm rather than reusing
either construction unchanged.

The complete pre-registration for this round -- direction, literature
citations, non-duplication argument against R-109/R-112/R-115/R-121 and
against R-118/R-119, and the four named failure risks -- lives in
``experiments/r122_shared.py``'s own module docstring, written by the
operator before either branch was dispatched, and is NOT re-derived here:
read that file in full first. This file imports ONLY from
``experiments.r122_shared`` (itself re-exporting r109_shared's unchanged
control machinery, feature builders, discount architecture, Step-0 gate and
B1-B5 promotion bar), never edits it, never coordinates with or reads the
conservative branch's file, and never reads a bar at or after
``r122_shared.OOS_START`` (2023-01-01).

MECHANISM, in one sentence: build the pooled synthetic reference panel ONCE
from 20 seeded draws of R-119's regime-switching generator, score each real
day's (log_vol, anchor_disp, kurtosis) feature vector by its mean Euclidean
distance (in the synthetic panel's own standardized space) to its k=10
nearest neighbours in that FIXED reference set, normalize to a causal [0,1]
percentile-rank novelty state, and multiplicatively discount v4's unchanged
``frac * scale`` product exactly as every prior round in this family does.

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this):

  1. ``ref_panel = r122_shared.build_synthetic_reference_panel(
     r122_shared.regimeswitch_path_generator, n_draws=r122_shared.
     N_SYNTH_DRAWS, seed_base=0)`` -- built ONCE at the top of ``main()``,
     zero real price data, R-119's frozen novel generator reused unchanged.
  2. ``daily = r122_shared.build_daily_features(df, r122_shared.
     REFERENCE_FEATURE_BUILDERS)`` -- the SAME 3-feature panel (log_vol,
     anchor_disp, kurtosis) r122_shared restricts both branches to (volume-
     free, since R-119's generators emit constant volume -- see
     r122_shared's own docstring for why the richer 5-feature panel is out
     of scope this round, a disclosed scope limit, not an oversight).
  3. ``dist = r122_shared.synthetic_knn_distance(daily, ref_panel, k=10)``
     -- day t's mean Euclidean distance (per-feature standardized against
     the FIXED synthetic panel's own mean/std) to its k nearest neighbours
     in that fixed set. This is the nonparametric, density-based sibling of
     the conservative branch's single fitted Gaussian (Mahalanobis)
     construction against the same panel -- matching R-109's own novel
     branch's algorithm-CLASS choice (Ramaswamy, Rastogi & Shim 2000;
     Breunig, Kriegel, Ng & Sander 2000's LOF family: no single-Gaussian/
     elliptical assumption about the reference distribution's shape), the
     one axis this round varies being the reference's DATA SOURCE (pooled
     synthetic vs. BTC's own real trailing history), not the algorithm.
  4. ``state = r122_shared.causal_rolling_percentile_rank(dist)`` -- a
     causal rolling percentile rank onto [0, 1] against the state's own
     trailing 730-day history (unchanged from every prior round in this
     family; the reference PANEL is fixed and synthetic, but the
     percentile-rank NORMALIZATION of the resulting distance series still
     walks forward over real calendar time, exactly as R-109's did).
  5. ``target = r122_shared.apply_discount(df, state, thresh, max_discount)
     = r122_shared.v4_target(df) * (1 - discount)`` -- v4's own vote and
     scale are completely UNTOUCHED; only the ``frac * scale`` PRODUCT is
     discounted, the same slot-in architecture every round in this family
     uses.

``k=10`` -- PRE-REGISTERED, REUSED (NOT SWEPT), before any real-data number
was read: this matches ``rolling_knn_distance``'s own historical default (R-
109's novel branch verified this programmatically against the live function
signature; ``synthetic_knn_distance`` here takes ``k`` as an explicit
keyword rather than a signature default, so the match is disclosed here by
direct equality assertion against the value R-109 itself verified, not
re-derived independently). Reasoning, unchanged from R-109: the kNN outlier-
scoring literature (Ramaswamy et al. 2000; Breunig et al. 2000's LOF)
typically reports ``k`` in the 10-50 range for moderate-size reference sets,
10 is the conservative (small) end of that range, and no diagnostics were
run to search for a "better" ``k`` before touching inner-validation data --
a search here would itself be a hidden trial, against this project's
standing no-hidden-search discipline. No sweep of ``k`` is performed in this
file; the Step-0 grid below (``thresh``/``max_discount`` only) is the entire
configuration search this branch performs before inner-validation is read.

CAUSAL SAFETY FIRST: ``r122_shared.causal_truncation_probe_series`` applied
to this file's own ``build_target`` (the FULL composed pipeline: features ->
fixed-reference kNN distance -> percentile-rank state -> discount ->
``v4_target * (1 - discount)``) at the pre-registered primary
(``PRIMARY_THRESH``, ``PRIMARY_MAXD``), run on BTC's full non-holdout frame,
BEFORE the Step-0 grid is scored and well before any inner-validation/ETH
performance number is computed. Because the reference panel is FIXED and
synthetic (never built from any slice of the real ``df`` being scored),
this branch's causal exposure is in some sense simpler than R-109's own
walk-forward refit: a real day's score depends only on that day's own
(already 1-bar-shifted) feature vector and the time-invariant synthetic
panel, never on any other real day at all (verified directly, at the
reference-distance level, by ``r122_shared.py``'s own self-test truncation/
permutation checks). The remaining causal exposure is entirely in
``causal_rolling_percentile_rank``'s own trailing-window walk-forward
normalization of the real-day distance SERIES, unchanged from every prior
round and already covered by ``r109_shared.py``'s self-test. If the probe
fails here, that is a real bug to find and report -- per this project's own
bug-fix-before-freezing discipline (R-121's ledger precedent: a bug found by
this exact probe, before any inner-val number is read, does not corrupt the
round if fixed before any decision-bearing number exists) -- not something
to work around.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read):
sweep ``r122_shared.STEP0_THRESH_GRID x r122_shared.STEP0_MAXD_GRID`` (3 x 2
= 6 cells) on BTC's ``INNER_TRAIN_START..INNER_TRAIN_END`` via
``r122_shared.step0_gate(btc.loc[INNER_TRAIN_START:INNER_TRAIN_END], state,
thresh, max_discount)``, where ``state = compute_full_state(btc)`` is
computed over BTC's FULL non-holdout frame (mirrors R-109's own convention:
causally identical, for every bar inside inner-train, to computing state on
the inner-train slice alone). A cell QUALIFIES iff ``step0_gate(...)
['passed']`` is True (bind_frac > 1%, R^2 vs v4's own target < 0.98, R^2 vs
v4's own realized-vol input < 0.90, state CoV >= 5%). PRIMARY cell: the
first ``(thresh, max_discount)`` in ``r122_shared.SELECTION_ORDER`` that
qualifies. If NONE qualify: STOP, report Step-0 FAIL as a complete NEGATIVE
result -- no inner-validation or ETH bar is read past that point. Read with
r122_shared's own named risk #1 in mind: the synthetic panel's own feature-
space distribution may simply not overlap real BTC/ETH's at all, producing
a degenerate, saturated statistic Step-0's kill switches were built to catch
but not specifically designed for.

PROMOTION BAR (only if Step-0 passes; identical shape to every SIZE/ERR-axis
round since R-89, via ``r122_shared``'s re-exported gate functions):
  B1 (gating): ``r122_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets (spot, futures_5x) -- dSharpe > +0.2 OR
     bootstrap excludes zero favourably.
  B2 (diagnostic ONLY, never gates): ``r122_shared.b2_diagnostic``.
  B3 (plateau, gating): the FULL 6-cell Step-0 ``(thresh, max_discount)``
     grid's own inner-validation numbers (both markets each, primary cell's
     2 rows reused directly from its own ``compare()`` rather than
     recomputed) -- PASS requires a directionally consistent (same-sign)
     majority across the resulting 12 cells. Since ``k`` is NOT swept (see
     above), this 6-cell grid is the entirety of this branch's plateau
     evidence.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test): ``r122_shared.b4_eth_falsification`` on
     ``r122_shared.compare(..., eth=r122_shared.load_eth())`` -- does the
     SAME-SIGN effect replicate on ETH? Require FULL pass (both markets
     same-signed as BTC inner_val). This is the one test that distinguishes
     r122_shared's own named risk #4 (a synthetic reference calibrated from
     BTC-specific crash-catalogue literature may still be BTC-flavoured by
     a different route than a real-data reference, rather than genuinely
     asset-neutral) from a genuine improvement.
  B5 (cost robustness, gating): ``r122_shared.b5_fee_tier`` at the 0.40%
     taker tier, primary cell, BTC inner_val, both markets -- no sign
     reversal.

PRE-REGISTERED DECISION RULE, stated verbatim and NOT altered after seeing
any number: PROMOTE-candidate only if the causal-truncation probe AND B1
(both markets) AND B3 (plateau majority) AND B4 (full) AND B5 all pass.
Anything else is NEGATIVE. B2 never gates. Default: NEGATIVE.

THIS BRANCH'S OWN SPECIFIC EXPOSURE TO EACH OF r122_shared.py's 4 NAMED
FAILURE RISKS:
  (1) Feature-space non-overlap / degenerate saturation: the regime-
      switching generator's bear state is calibrated to be considerably
      HARSHER than BTC's own realized history (R-119's own docstring: ~2.3x
      steeper drift than R-118's fitted bear regime, comparable vol) -- if
      that harsher bear state pushes the synthetic panel's own log_vol/
      kurtosis region systematically away from real BTC/ETH's, every real
      day could register as maximally "novel" (or, in the opposite
      direction, the panel's much longer pooled history, ~27,000 rows
      across bull/chop/bear vs. BTC's own 730-day window, could instead
      make the reference OVER-dispersed, collapsing real-day novelty scores
      toward the low end everywhere) -- Step-0's bind_frac/state_cv kill
      switches are the read on this, disclosed as genuine rather than
      nominal per r122_shared's own risk #1.
  (2) Inert-by-construction: if Step-0 passes, the same "v4 already reacts
      before novelty registers" pattern R-109/R-112/R-115/R-121 all hit
      (Step-0 passes, B1 does not) is exactly as available here as there --
      swapping the reference's data source does nothing to fix a timing
      problem in v4's own vote, per r122_shared's own risk #2.
  (3) Generator-purpose mismatch: R-119's regime-switching generator was
      built and validated for producing realistic PRICE PATHS for a Kelly-
      sizer CVaR parameter sweep, not for reproducing the fine-grained SHAPE
      of BTC's own realized trend-anchor-dispersion or kurtosis
      distributions -- there is no guarantee it is calibrated finely enough
      for THIS use, per r122_shared's own risk #3; Step-0's diagnostics are
      read with this in mind.
  (4) BTC-flavoured-by-a-different-route: the generator's own calibration
      numbers (bear drawdown/duration, jump sizes) were themselves sourced
      from BTC-specific crash catalogues and jump studies (R-119's own
      citations), so a synthetic reference built from them is not
      obviously more asset-neutral than a real-BTC-history reference would
      be -- B4 (ETH falsification) is the one test that distinguishes these
      two readings, per r122_shared's own risk #4.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets, 2 of
the 12 reused directly from the primary ``compare()``'s own inner_val rows,
10 freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total. IF
Step-0 finds no qualifying cell, this file stops after the 6 Step-0 cells
(6 total). No ``k`` sweep is performed (see above), so it adds 0
configurations to either count.

USAGE
-----
    python experiments/r122_novel_regimeswitch_knn.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r122_shared  # noqa: E402

# ---------------------------------------------------------- pre-registered
K = 10   # rolling_knn_distance's own historical default (verified by R-109's
         # novel branch against the live function signature there); reused
         # here, not swept -- see banner reasoning above.

assert K == 10, "K does not match the pre-registered, R-109-verified default -- text is stale"


# ================================================================== (1)
# The mechanism itself: 3-feature panel -> FIXED-synthetic-reference kNN
# distance -> percentile-rank state -> discount on v4's own UNCHANGED
# frac*scale.
# ==================================================================

def compute_full_state(df: pd.DataFrame, ref_panel: pd.DataFrame, k: int = K) -> pd.Series:
    """features -> synthetic-reference kNN distance -> causal percentile-rank
    state, over whatever frame `df` is (the caller decides how much history
    it contains -- this function itself makes no reference to any fixed
    calendar date). `ref_panel` is the FIXED pooled synthetic reference,
    built once by the caller and passed in unchanged."""
    daily = r122_shared.build_daily_features(df, r122_shared.REFERENCE_FEATURE_BUILDERS)
    dist = r122_shared.synthetic_knn_distance(daily, ref_panel, k=k)
    return r122_shared.causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, ref_panel: pd.DataFrame, thresh: float = r122_shared.PRIMARY_THRESH,
                  max_discount: float = r122_shared.PRIMARY_MAXD, k: int = K) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount), where
    discount is driven by the synthetic-reference kNN novelty state built
    from `df` alone plus the FIXED `ref_panel`. Self-contained modulo
    `ref_panel` (a pure function of `df` given the fixed panel), so it is
    directly usable as a `TargetStrategy` candidate on any window
    (inner_train, inner_val, eth_replication, or a truncated probe frame)."""
    state = compute_full_state(df, ref_panel, k=k)
    return r122_shared.apply_discount(df, state, thresh, max_discount)


def make_build_target(ref_panel: pd.DataFrame, thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, ref_panel, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"synth_regimeswitch_knn_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r122_shared.STEP0_THRESH_GRID x r122_shared.STEP0_MAXD_GRID,
# scored via r122_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame (mirrors R-109's own step0_grid
# convention).
# ==================================================================

def step0_grid(btc: pd.DataFrame, ref_panel: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    state = compute_full_state(btc, ref_panel)
    df_inner_train = btc.loc[r122_shared.INNER_TRAIN_START:r122_shared.INNER_TRAIN_END]
    rows = []
    for thresh in r122_shared.STEP0_THRESH_GRID:
        for maxd in r122_shared.STEP0_MAXD_GRID:
            gate = r122_shared.step0_gate(df_inner_train, state, thresh, maxd)
            rows.append(dict(thresh=thresh, max_discount=maxd, **gate))
    return rows, state


def select_primary(rows: list[dict]) -> dict | None:
    by_key = {(r["thresh"], r["max_discount"]): r for r in rows}
    for key in r122_shared.SELECTION_ORDER:
        r = by_key.get(key)
        if r is not None and r["passed"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {r122_shared.INNER_TRAIN_START} -> "
          f"{r122_shared.INNER_TRAIN_END}, {n_bars:,} bars, state built from k={K} kNN distance "
          f"against the FIXED pooled synthetic regime-switching reference panel)")
    print(f"QUALIFY = bind_frac > {r122_shared.BIND_FRAC_THRESH:.0%} AND r2_vs_v4 < "
          f"{r122_shared.R2_VS_V4_THRESH} AND r2_vs_vol < {r122_shared.R2_VS_VOL_THRESH} AND "
          f"state_cv >= {r122_shared.CV_KILL_THRESH:.0%}")
    hdr = (f"{'thresh':>7s} {'max_d':>6s} {'bind_frac':>10s} {'r2_vs_v4':>9s} "
           f"{'r2_vs_vol':>9s} {'state_cv':>9s} {'passed':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = "  <- SELECTION_ORDER[0]" if (r["thresh"], r["max_discount"]) == r122_shared.SELECTION_ORDER[0] else ""
        print(f"{r['thresh']:7.2f} {r['max_discount']:6.2f} {r['bind_frac']:10.4f} "
              f"{r['r2_vs_v4']:9.4f} {r['r2_vs_vol']:9.4f} {r['state_cv']:9.4f} "
              f"{'YES' if r['passed'] else 'no':>7s}{tag}")


# ================================================================== (3)
# B3 plateau: the full 6-cell (thresh, max_discount) grid's own
# inner-validation numbers (both markets), primary cell's 2 rows reused
# directly from its own compare().
# ==================================================================

def run_b3_full_grid(step0_rows: list[dict], primary_key: tuple[float, float],
                      inner_val_primary: list[dict], btc: pd.DataFrame,
                      ref_panel: pd.DataFrame) -> tuple[dict, bool]:
    plateau_rows: dict[tuple[float, float], list[dict]] = {}
    for r in step0_rows:
        key = (r["thresh"], r["max_discount"])
        if key == primary_key:
            plateau_rows[key] = [dict(market=c["market"], d_sharpe=c["d_sharpe"], d_dd=c["d_dd"],
                                       exposure_ratio=c["exposure_ratio"], vol_ratio=c["vol_ratio"],
                                       risk_matched=c["risk_matched"],
                                       boot_d_loggrowth=c["boot_d_loggrowth"], boot_lo=c["boot_lo"],
                                       boot_hi=c["boot_hi"], excludes_zero=c["excludes_zero"])
                                  for c in inner_val_primary]
        else:
            bf = make_build_target(ref_panel, *key)
            label = f"synth_regimeswitch_knn_t{key[0]:g}_m{key[1]:g}"
            plateau_rows[key] = r122_shared.inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# above), B4 (gating falsification), B5 (gating fee tier).
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                       btc: pd.DataFrame, eth: pd.DataFrame, ref_panel: pd.DataFrame) -> dict:
    thresh, maxd = primary_key
    build_primary = make_build_target(ref_panel, thresh, maxd)
    label = f"synth_regimeswitch_knn_t{thresh:g}_m{maxd:g}"

    r122_shared.hr(f"PROMOTION BAR -- PRIMARY CELL thresh={thresh:g}, max_discount={maxd:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = r122_shared.compare(build_primary, label=label, btc=btc, eth=eth,
                                markets=(r122_shared.SPOT, r122_shared.FUTURES), include_eth=True)
    r122_shared.print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = r122_shared.b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = r122_shared.b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3_full_grid(step0_rows, primary_key, inner_val_primary, btc, ref_panel)
    b4_partial, b4_full, b4_cells = r122_shared.b4_eth_falsification(eth_primary, inner_val_primary)
    b5_pass, b5_cells = r122_shared.b5_fee_tier(build_primary, label, btc, inner_val_primary)

    all_pass = b1_pass and b3_pass and b4_full and b5_pass

    return dict(
        label=label, thresh=thresh, max_discount=maxd, compare_rows=rows,
        inner_val_primary=inner_val_primary, eth_primary=eth_primary,
        b1_pass=b1_pass, b1_cells=b1_cells,
        b2_pass=b2_pass, b2_cells=b2_cells,
        b3_pass=b3_pass, b3_rows=b3_rows,
        b4_partial=b4_partial, b4_full=b4_full, b4_cells=b4_cells,
        b5_pass=b5_pass, b5_cells=b5_cells,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 12 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    r122_shared.hr("R-122 NOVEL: SyntheticRegimeswitchKnnNoveltyBrake -- k-nearest-neighbour "
                    "DISTRIBUTIONAL-NOVELTY discount scored against a POOLED, EXTERNALLY-"
                    "CALIBRATED SYNTHETIC (regime-switching) reference panel")
    print("mechanism: build a pooled synthetic reference panel ONCE (20 draws of R-119's 3-state")
    print("bull/chop/bear regime-switching jump-diffusion generator, bear severity/duration from an")
    print("external BTC crash catalogue, zero real price data) -> 3-feature OHLCV-only daily panel")
    print("(log_vol, anchor_disp, kurtosis) on real BTC/ETH -> mean Euclidean distance (per-feature")
    print("standardized against the FIXED synthetic panel) to its k nearest neighbours in that fixed")
    print("reference SET (nonparametric/density-based, R-109's own novel-branch algorithm, unchanged)")
    print("-> causal rolling percentile-rank state in [0,1] -> linear discount on v4's UNCHANGED")
    print("frac*scale product. Full grounding in r122_shared.py's own module docstring.")
    print(f"\nk={K}  (rolling_knn_distance's own historical default, reused not swept -- see banner)")
    print(f"N_SYNTH_DRAWS={r122_shared.N_SYNTH_DRAWS}")
    print(f"STEP0_THRESH_GRID={r122_shared.STEP0_THRESH_GRID}  STEP0_MAXD_GRID={r122_shared.STEP0_MAXD_GRID}  "
          f"({len(r122_shared.STEP0_THRESH_GRID) * len(r122_shared.STEP0_MAXD_GRID)} cells)")
    print(f"SELECTION_ORDER={r122_shared.SELECTION_ORDER}")

    # ============================================== (1) Build the pooled
    # synthetic reference panel ONCE, before anything else.
    r122_shared.hr("BUILDING POOLED SYNTHETIC REFERENCE PANEL "
                    "(R-119 regime-switching generator, zero real price data)")
    t_panel0 = time.time()
    ref_panel = r122_shared.build_synthetic_reference_panel(
        r122_shared.regimeswitch_path_generator, n_draws=r122_shared.N_SYNTH_DRAWS, seed_base=0)
    t_panel = time.time() - t_panel0
    print(f"ref_panel: {len(ref_panel):,} rows x {ref_panel.shape[1]} features "
          f"({r122_shared.N_SYNTH_DRAWS} draws, seeds 0..{r122_shared.N_SYNTH_DRAWS - 1}), "
          f"built in {t_panel:.1f}s")
    print(f"ref_panel feature summary:\n{ref_panel.describe().to_string()}")

    btc = r122_shared.load_btc()
    max_ts_seen.append(btc.index.max())
    r122_shared.assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {r122_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ================================================== CAUSAL SAFETY FIRST
    r122_shared.hr("CAUSAL TRUNCATION PROBE (composed build_target at the pre-registered primary "
                    "(thresh, max_discount), real BTC data, run BEFORE Step-0 or any inner-val/ETH number)")
    probe_fn = make_build_target(ref_panel, r122_shared.PRIMARY_THRESH, r122_shared.PRIMARY_MAXD)
    print(f"causal_truncation_probe_series({probe_fn.__name__}, btc):")
    try:
        probe_ok = r122_shared.causal_truncation_probe_series(probe_fn, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL SAFETY (truncation probe) PASS: {probe_ok}")

    # ============================================================= STEP 0
    r122_shared.hr("STEP 0 -- NON-DEGENERACY / NON-DUPLICATION KILL SWITCH "
                    "(run BEFORE any inner-validation Sharpe/PnL number)")
    step0_rows, state = step0_grid(btc, ref_panel)
    n_bars_inner_train = int(np.sum(
        (btc.index >= pd.Timestamp(r122_shared.INNER_TRAIN_START, tz="UTC")) &
        (btc.index <= pd.Timestamp(r122_shared.INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train)

    primary_row = select_primary(step0_rows)

    results_json: dict = dict(
        branch="synthetic_regimeswitch_knn",
        ref_panel_rows=int(len(ref_panel)),
        ref_panel_cols=int(ref_panel.shape[1]),
        n_synth_draws=r122_shared.N_SYNTH_DRAWS,
        k=K,
        probe_ok=bool(probe_ok),
        step0_rows=step0_rows,
    )

    if primary_row is None:
        r122_shared.hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the synthetic-")
        print("reference kNN novelty discount is either a near-total no-op, a near-exact rescale of")
        print("v4's own path, a relabelled volatility rescale, or degenerate everywhere on the")
        print("pre-registered grid (r122_shared's own named risk #1 or #3).")
        print("Per this file's own pre-registration, this Step-0 table (plus the causal-safety")
        print("probe above) is the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0.")
        print("No promotion-bar code runs, and no inner-validation Sharpe/PnL number or ETH bar")
        print("is ever read.")

        r122_shared.hr("VERDICT")
        print("Step-0 (6-cell thresh x max_discount grid): FAIL (no cell qualifies)")
        print(f"causal truncation probe: {probe_ok}")
        print("B1: NOT COMPUTED (Step-0 kill switch)")
        print("B2: NOT COMPUTED (Step-0 kill switch)")
        print("B3: NOT COMPUTED (Step-0 kill switch)")
        print("B4: NOT COMPUTED (Step-0 kill switch)")
        print("B5: NOT COMPUTED (Step-0 kill switch)")
        print("VERDICT: NEGATIVE (Step-0 kill switch)")

        n_configs = len(step0_rows)
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated (total): {n_configs} (6 Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {r122_shared.OOS_START}: {max_ts < pd.Timestamp(r122_shared.OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")

        results_json.update(
            primary_cell=None, passed_step0=False,
            n_configs_evaluated=n_configs,
            max_timestamp=str(max_ts),
            max_timestamp_before_oos=bool(max_ts < pd.Timestamp(r122_shared.OOS_START, tz="UTC")),
            verdict="NEGATIVE (Step-0 kill switch)",
        )
        out_path = ROOT / "experiments" / "r122_novel_results.json"
        with open(out_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)
        print(f"Saved results -> {out_path}")

        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                    probe_ok=probe_ok, n_configs=n_configs, max_ts=max_ts,
                    verdict="NEGATIVE (Step-0 kill switch)")

    primary_key = (primary_row["thresh"], primary_row["max_discount"])
    is_selection0 = (primary_key == r122_shared.SELECTION_ORDER[0])
    print(f"\nPRIMARY CELL SELECTED (Step-0 non-degeneracy rule only): "
          f"thresh={primary_key[0]:g}, max_discount={primary_key[1]:g}  "
          f"(bind_frac={primary_row['bind_frac']:.4f}, r2_vs_v4={primary_row['r2_vs_v4']:.4f}, "
          f"r2_vs_vol={primary_row['r2_vs_vol']:.4f}, state_cv={primary_row['state_cv']:.4f})")
    print(f"  selection: {'SELECTION_ORDER[0] qualified' if is_selection0 else 'SELECTION_ORDER[0] did NOT qualify; next qualifying cell in SELECTION_ORDER chosen'}")

    eth = r122_shared.load_eth()
    max_ts_seen.append(eth.index.max())
    r122_shared.assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {r122_shared.OOS_START})")

    bar = run_promotion_bar(primary_key, step0_rows, btc, eth, ref_panel)

    r122_shared.hr("B1 -- inner-validation, both markets (dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    r122_shared.hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    r122_shared.hr("B3 -- plateau: FULL 6-cell (thresh, max_discount) Step-0 grid, inner-validation, both markets")
    r122_shared.print_plateau_table(bar["b3_rows"])
    print(f"\nB3 (directionally consistent majority across the 12-cell grid): {bar['b3_pass']}")

    r122_shared.hr("B4 -- ETH falsification (pre-registered)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PARTIAL PASS (>=1 market): {bar['b4_partial']}   B4 FULL PASS (both markets): {bar['b4_full']}")

    r122_shared.hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier d_sharpe={c['d_sharpe']:+.4f}  "
              f"standard-fee d_sharpe={c['base_d_sharpe']:+.4f}  "
              f"fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    r122_shared.hr("VERDICT")
    print(f"causal safety (truncation probe): {probe_ok}")
    print(f"B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4_full={bar['b4_full']}  B5={bar['b5_pass']}")
    all_applicable_pass = (probe_ok and bar["b1_pass"] and bar["b3_pass"] and
                            bar["b4_full"] and bar["b5_pass"])
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")
    if not probe_ok:
        print("NOTE: verdict driven (at least in part) by a causal-safety check failure -- "
              "a lookahead is a bug report first, per docs/ROUTINE.md's own precedence.")

    n_configs = len(step0_rows) + bar["n_configs_promotion_bar"]
    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(6 Step-0 grid + 6 primary-cell compare() + 12 B3 plateau "
          f"[6 (thresh,max_discount) cells x 2 markets, 2 reused from primary] + "
          f"2 B5 fee-tier; k not swept, adds 0)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {r122_shared.OOS_START}: {max_ts < pd.Timestamp(r122_shared.OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    def _clean(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    def _clean_rows(rows):
        return [{k2: _clean(v2) for k2, v2 in row.items()} for row in rows]

    results_json.update(
        primary_cell=dict(thresh=primary_key[0], max_discount=primary_key[1]),
        primary_row={k2: _clean(v2) for k2, v2 in primary_row.items()},
        passed_step0=True,
        compare_rows=_clean_rows(bar["compare_rows"]),
        b1_pass=bool(bar["b1_pass"]), b1_cells=_clean_rows(bar["b1_cells"]),
        b2_pass=bool(bar["b2_pass"]), b2_cells=_clean_rows(bar["b2_cells"]),
        b3_pass=bool(bar["b3_pass"]),
        b3_rows={f"{k2[0]}_{k2[1]}": _clean_rows(v2) for k2, v2 in bar["b3_rows"].items()},
        b4_partial=bool(bar["b4_partial"]), b4_full=bool(bar["b4_full"]),
        b4_cells=_clean_rows(bar["b4_cells"]),
        b5_pass=bool(bar["b5_pass"]), b5_cells=_clean_rows(bar["b5_cells"]),
        n_configs_evaluated=n_configs,
        max_timestamp=str(max_ts),
        max_timestamp_before_oos=bool(max_ts < pd.Timestamp(r122_shared.OOS_START, tz="UTC")),
        all_applicable_pass=bool(all_applicable_pass),
        verdict=verdict,
    )
    out_path = ROOT / "experiments" / "r122_novel_results.json"
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"Saved results -> {out_path}")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, state=state,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts,
                ref_panel=ref_panel)


if __name__ == "__main__":
    main()
