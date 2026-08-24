#!/usr/bin/env python
"""R-122 CONSERVATIVE branch: ``SyntheticGbmMahalanobisNoveltyBrakeKellyV4`` --
``kelly_regime_v4``'s own unchanged ``frac * scale`` product, multiplicatively
discounted by the literal, single-fitted-Gaussian Mahalanobis distance of
TODAY's real 3-feature market-state vector from a FIXED, pooled, purely
SYNTHETIC reference distribution built from 20 seeded draws of R-119's plain
GBM-diffusion + externally-calibrated compound-Poisson-jump path generator
(zero real price data anywhere in the reference), pairing R-119's simplest
generator with R-109's simplest novelty algorithm -- deliberately the
simplest reading of this round's direction.

Everything about WHY this round exists, the literature grounding (Rabanser/
Gunnemann/Lipton 2019; Abbas/Azmat/Horesh/Yurochkin 2025; Wiese/Knobloch/
Korn/Kretschmer 2020; the R-119-frozen crash-catalogue calibration sources),
the exhaustive non-duplication argument against R-109/R-112/R-115/R-118/
R-119/R-121/R-113, and this round's four NAMED failure risks all live in
``experiments/r122_shared.py``'s own module docstring, written by the
operator BEFORE either branch was dispatched -- read it in full first; it is
NOT re-derived here. This file imports ONLY from ``experiments.r122_shared``
(itself re-exporting r109_shared's unchanged control machinery), never edits
that file, never coordinates with or reads the novel branch's file
(``experiments/r122_novel_regimeswitch_knn.py``, a disjoint file owned by a
different session), and never reads a bar at or after
``r122_shared.OOS_START`` (2023-01-01).

EXACT CONSTRUCTION (the entire mechanism, per the operator's dispatch
instructions -- no other logic, gate, or heuristic added on top):

  0. ``ref_panel = r122_shared.build_synthetic_reference_panel(
     r122_shared.gbm_jump_path_generator, n_draws=r122_shared.N_SYNTH_DRAWS,
     seed_base=0)`` -- built ONCE, at the top of ``main()``. 20 seeded draws
     of R-119's CONSERVATIVE generator (plain GBM diffusion + an externally-
     calibrated compound-Poisson jump overlay -- no regime-switching
     structure at all), pooled into one fixed, time-invariant reference
     panel. Zero real price bars anywhere in this panel.
  1. ``daily = r122_shared.build_daily_features(df,
     r122_shared.REFERENCE_FEATURE_BUILDERS)`` -- the SAME 3-feature panel
     (``log_vol``, ``anchor_disp``, ``kurtosis``) R-109's conservative branch
     used, computed on REAL BTC/ETH data.
  2. ``dist = r122_shared.synthetic_mahalanobis_distance(daily, ref_panel)``
     -- the FIXED-reference Mahalanobis distance: one mean/inverse-covariance
     fit ONCE from the whole pooled synthetic panel (never refit, never
     rolling -- unlike R-109's own rolling 730-day REAL-data reference), the
     parametric, single-fitted-Gaussian sibling of R-109's own algorithm
     choice, matching R-109's conservative branch's algorithm class exactly
     while swapping only the reference distribution's data SOURCE (real ->
     synthetic).
  3. ``state = r122_shared.causal_rolling_percentile_rank(dist)`` -- causal
     rolling percentile rank onto [0, 1] against the STATE's own trailing
     history (the distance-to-state normalization step is unchanged from
     every prior round in this family; only the distance itself is scored
     against a fixed synthetic reference rather than a rolling real one).
  4. ``target = r122_shared.apply_discount(df, state, thresh, max_discount)
     = v4_target(df) * (1 - discount)``, discount ramping linearly from 0 at
     ``state <= thresh`` to ``max_discount`` at ``state == 1``.
     ``kelly_regime_v4``'s own vote and scale are completely UNTOUCHED; only
     the ``frac * scale`` PRODUCT is discounted -- the identical slot-in
     architecture every ERR-axis round since R-87 uses.

THIS BRANCH'S OWN SPECIFIC EXPOSURE TO ``r122_shared.py``'s FOUR NAMED
FAILURE RISKS (restated here as required, not re-derived):
  (1) Feature-space non-overlap / saturation: this branch is the FIRST
      place either of R-119's two generators is scored against Mahalanobis
      distance at all (R-119 never built a novelty statistic; R-109 never
      touched synthetic data) -- if the GBM+jump generator's realized
      vol/anchor-dispersion/kurtosis land systematically outside real BTC's
      own feature range, EVERY real day will register maximally "novel," a
      degenerate, saturated statistic Step-0's ``R2_VS_V4_THRESH``/
      ``state_cv`` checks are designed, but not specifically built, to catch
      -- the Step-0 table below is read with this risk explicitly in mind.
      This branch's exposure is arguably HIGHER than the novel branch's kNN
      distance, which only needs local density agreement rather than a
      single global elliptical fit to hold.
  (2) v4's reactive, latched vote already pricing in "unusual" conditions
      before ANY novelty statistic (real-referenced or synthetic-referenced)
      can register them: this risk is completely orthogonal to the
      reference's data source, so this branch is exactly as exposed to the
      "Step-0 passes, B1 does not" inert pattern as R-109/R-112/R-115/R-121
      all were -- a sixth reproduction of that exact shape, on a fifth
      reference-distribution variant, would be uninformative about the
      swap this round tests but fully consistent with the pattern.
  (3) Generator-purpose mismatch: R-119's plain GBM+jump generator was
      designed and validated for producing realistic PRICE PATHS for a
      Kelly-sizer CVaR parameter sweep, not for reproducing the fine-grained
      shape of BTC's own trend-anchor-dispersion or kurtosis distributions.
      Being the CONSERVATIVE (simplest) generator -- no regime-switching,
      no bear-market state -- it is the more STRUCTURALLY IMPOVERISHED of
      R-119's two generators, so if this risk bites anywhere in this round,
      it is expected to bite HARDER here than in the novel branch's
      regime-switching generator, a disclosed asymmetry between the two
      branches' exposure to this identical risk, not a defect specific to
      this file.
  (4) BTC-flavoured-by-calibration-literature, not by raw data: this
      branch's reference panel touches zero real BTC or ETH bars, but its
      GBM+jump parameters (80%/yr vol, jump sizes) were themselves
      calibrated from BTC-specific crash catalogues (R-119's own citations)
      -- B4 (ETH falsification) is exactly the pre-registered test that
      distinguishes "truly asset-neutral" from "BTC-flavoured by a
      different route," unchanged from every prior round in this family.

CONFIGURATIONS EVALUATED (formula, per the operator's dispatch): 6 (Step-0
grid, 3 thresh x 2 max_discount) if Step-0 finds no qualifying cell (STOP);
else 6 (Step-0) + 6 (primary cell's full ``compare()``: inner_train x2
markets + inner_val x2 markets + eth_replication x2 markets) + 12 (B3's full
6-cell grid x 2 markets, 2 of the 12 reused directly from the primary
``compare()``'s own inner_val rows, 10 freshly computed) + 2 (B5's 0.40% fee
tier, 2 markets) = 26 total. The synthetic reference panel's 20 draws are
built ONCE, before Step-0, and are not themselves a swept configuration.

PRE-REGISTERED DECISION RULE (stated verbatim, unaltered after seeing any
number): PROMOTE-candidate only if the causal-truncation probe AND B1 (both
markets) AND B3 (plateau majority) AND B4 (full) AND B5 all pass. Anything
else is NEGATIVE. B2 is diagnostic only and never gates.

USAGE
-----
    python experiments/r122_conservative_gbm_mahalanobis.py
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

RESULTS_PATH = ROOT / "experiments" / "r122_conservative_results.json"


# ================================================================== (1)
# The mechanism itself: ref_panel (built once in main()) -> features ->
# fixed-reference Mahalanobis distance -> causal percentile-rank state ->
# discount on v4's own UNCHANGED frac*scale.
# ==================================================================

def compute_full_state(df: pd.DataFrame, ref_panel: pd.DataFrame) -> pd.Series:
    """features -> fixed-reference Mahalanobis distance -> causal
    percentile-rank state, over whatever frame `df` is (the caller decides
    how much history it contains). `ref_panel` is the pooled, purely
    synthetic reference built once by the caller -- never rebuilt here."""
    daily = r122_shared.build_daily_features(df, r122_shared.REFERENCE_FEATURE_BUILDERS)
    dist = r122_shared.synthetic_mahalanobis_distance(daily, ref_panel)
    return r122_shared.causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, ref_panel: pd.DataFrame, thresh: float,
                  max_discount: float) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount), where
    discount is driven by the fixed-synthetic-reference Mahalanobis novelty
    state built from `df` and `ref_panel` alone."""
    state = compute_full_state(df, ref_panel)
    return r122_shared.apply_discount(df, state, thresh, max_discount)


def make_build_target(ref_panel: pd.DataFrame, thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, ref_panel, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"synthetic_gbm_mahalanobis_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r122_shared.STEP0_THRESH_GRID x r122_shared.STEP0_MAXD_GRID,
# scored via r122_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame (mirrors R-109's own convention -- see
# r122_shared.py's docstring/r109_novel's banner for why this is causally
# identical to computing state on the inner-train slice alone).
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


def print_step0_table(rows: list[dict], n_bars: int, n_ref_rows: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {r122_shared.INNER_TRAIN_START} -> "
          f"{r122_shared.INNER_TRAIN_END}, {n_bars:,} bars, state built against a FIXED pooled "
          f"synthetic reference panel of {n_ref_rows:,} rows, {r122_shared.N_SYNTH_DRAWS} draws)")
    print(f"QUALIFY = bind_frac > {r122_shared.BIND_FRAC_THRESH:.0%} AND "
          f"r2_vs_v4 < {r122_shared.R2_VS_V4_THRESH} AND "
          f"r2_vs_vol < {r122_shared.R2_VS_VOL_THRESH} AND "
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
                      inner_val_primary: list[dict], ref_panel: pd.DataFrame,
                      btc: pd.DataFrame) -> tuple[dict, bool]:
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
            label = f"synthetic_gbm_mahalanobis_t{key[0]:g}_m{key[1]:g}"
            plateau_rows[key] = r122_shared.inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# above), B4 (gating falsification), B5 (gating fee tier).
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                       ref_panel: pd.DataFrame, btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    thresh, maxd = primary_key
    build_primary = make_build_target(ref_panel, thresh, maxd)
    label = f"synthetic_gbm_mahalanobis_t{thresh:g}_m{maxd:g}"

    r122_shared.hr(f"PROMOTION BAR -- PRIMARY CELL thresh={thresh:g}, max_discount={maxd:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = r122_shared.compare(build_primary, label=label, btc=btc, eth=eth,
                                markets=(r122_shared.SPOT, r122_shared.FUTURES), include_eth=True)
    r122_shared.print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = r122_shared.b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = r122_shared.b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3_full_grid(step0_rows, primary_key, inner_val_primary, ref_panel, btc)
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


# --------------------------------------------------------------------- json helpers

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    r122_shared.hr("R-122 CONSERVATIVE: SyntheticGbmMahalanobisNoveltyBrakeKellyV4 -- "
                   "pooled, externally-calibrated SYNTHETIC-reference Mahalanobis "
                   "novelty discount on v4's own frac*scale")
    print("mechanism: build a pooled reference panel from 20 seeded draws of R-119's plain")
    print("GBM-diffusion + externally-calibrated compound-Poisson-jump generator (zero real")
    print("price data), fit ONE fixed mean/covariance from it, score each real day's 3-feature")
    print("market-state vector's Mahalanobis distance from that FIXED synthetic reference,")
    print("causal-percentile-rank it to a [0,1] novelty state, and linearly discount v4's")
    print("UNCHANGED frac*scale product. Full grounding in r122_shared.py's own module docstring.")

    hr = r122_shared.hr
    hr("BUILD POOLED SYNTHETIC REFERENCE PANEL (once, before any real-data number is read)")
    ref_panel = r122_shared.build_synthetic_reference_panel(
        r122_shared.gbm_jump_path_generator, n_draws=r122_shared.N_SYNTH_DRAWS, seed_base=0)
    print(f"ref_panel: {len(ref_panel):,} rows x {ref_panel.shape[1]} cols "
          f"({r122_shared.N_SYNTH_DRAWS} draws, seed_base=0, generator="
          f"r122_shared.gbm_jump_path_generator [R-119 conservative GBM+jump])")
    print(f"ref_panel columns: {list(ref_panel.columns)}")
    print(f"ref_panel feature means: {ref_panel.mean().to_dict()}")
    print(f"ref_panel feature stds:  {ref_panel.std().to_dict()}")

    btc = r122_shared.load_btc()
    max_ts_seen.append(btc.index.max())
    r122_shared.assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {r122_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ================================================== CAUSAL SAFETY FIRST
    hr("CAUSAL TRUNCATION PROBE (composed build_target at the pre-registered primary "
       "(thresh, max_discount), real BTC data, run BEFORE Step-0 or any inner-val/ETH number)")
    probe_fn = make_build_target(ref_panel, r122_shared.PRIMARY_THRESH, r122_shared.PRIMARY_MAXD)
    print(f"causal_truncation_probe_series({probe_fn.__name__}, btc):")
    bug_note = None
    try:
        r122_shared.causal_truncation_probe_series(probe_fn, btc)
        probe_ok = True
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL SAFETY (truncation probe) PASS: {probe_ok}")

    if not probe_ok:
        hr("CAUSAL SAFETY FAILURE -- STOPPING HERE")
        print("A lookahead bug is a bug report first, per docs/ROUTINE.md's own precedence.")
        print("No Step-0, B1-B5, or ETH data is read past this point.")
        max_ts = max(max_ts_seen)
        result = dict(branch="synthetic_gbm_mahalanobis", ref_panel_rows=len(ref_panel),
                      ref_panel_cols=list(ref_panel.columns), probe_ok=False,
                      bug_note=bug_note, verdict="ABORTED (causal safety failure)",
                      n_configs=0, max_ts=str(max_ts))
        RESULTS_PATH.write_text(json.dumps(_jsonable(result), indent=2))
        print(f"\n[{time.time() - t0:.0f}s]")
        return result

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY / NON-DUPLICATION KILL SWITCH "
       "(run BEFORE any inner-validation Sharpe/PnL number)")
    step0_rows, state = step0_grid(btc, ref_panel)
    n_bars_inner_train = int(np.sum(
        (btc.index >= pd.Timestamp(r122_shared.INNER_TRAIN_START, tz="UTC")) &
        (btc.index <= pd.Timestamp(r122_shared.INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train, len(ref_panel))

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the")
        print("fixed-synthetic-reference Mahalanobis novelty discount is either a near-total")
        print("no-op, a near-exact rescale of v4's own path, a relabelled volatility rescale,")
        print("or degenerate everywhere on the pre-registered grid.")
        print("Per this file's own pre-registration, this Step-0 table (plus the causal-safety")
        print("probe above) is the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0.")
        print("No promotion-bar code runs, and no inner-validation Sharpe/PnL number or ETH bar")
        print("is ever read.")

        hr("VERDICT")
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
        ok_ts = max_ts < pd.Timestamp(r122_shared.OOS_START, tz="UTC")
        print(f"\nconfigurations evaluated (total): {n_configs} (6 Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {r122_shared.OOS_START}: {ok_ts})")
        assert ok_ts, "max timestamp read is not before OOS_START -- holdout violation"
        print(f"\n[{time.time() - t0:.0f}s]")

        result = dict(
            branch="synthetic_gbm_mahalanobis", ref_panel_rows=len(ref_panel),
            ref_panel_cols=list(ref_panel.columns), probe_ok=probe_ok,
            step0_rows=step0_rows, primary=None, passed_step0=False,
            n_configs=n_configs, max_ts=str(max_ts),
            max_ts_before_oos_start=bool(ok_ts),
            verdict="NEGATIVE (Step-0 kill switch)",
        )
        RESULTS_PATH.write_text(json.dumps(_jsonable(result), indent=2))
        return result

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

    bar = run_promotion_bar(primary_key, step0_rows, ref_panel, btc, eth)

    hr("B1 -- inner-validation, both markets (dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hr("B3 -- plateau: FULL 6-cell (thresh, max_discount) Step-0 grid, inner-validation, both markets")
    r122_shared.print_plateau_table(bar["b3_rows"])
    print(f"\nB3 (directionally consistent majority across the 12-cell grid): {bar['b3_pass']}")

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
    ok_ts = max_ts < pd.Timestamp(r122_shared.OOS_START, tz="UTC")
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(6 Step-0 grid + 6 primary-cell compare() + 12 B3 plateau "
          f"[6 (thresh,max_discount) cells x 2 markets, 2 reused from primary] + "
          f"2 B5 fee-tier)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {r122_shared.OOS_START}: {ok_ts})")
    assert ok_ts, "max timestamp read is not before OOS_START -- holdout violation"

    print(f"\n[{time.time() - t0:.0f}s]")

    result = dict(
        branch="synthetic_gbm_mahalanobis", ref_panel_rows=len(ref_panel),
        ref_panel_cols=list(ref_panel.columns), probe_ok=probe_ok,
        step0_rows=step0_rows, primary=primary_row, passed_step0=True,
        promotion_bar=dict(
            label=bar["label"], thresh=bar["thresh"], max_discount=bar["max_discount"],
            compare_rows=bar["compare_rows"],
            b1_pass=bar["b1_pass"], b1_cells=bar["b1_cells"],
            b2_pass=bar["b2_pass"], b2_cells=bar["b2_cells"],
            b3_pass=bar["b3_pass"],
            b3_rows={f"{k[0]:g}_{k[1]:g}": v for k, v in bar["b3_rows"].items()},
            b4_partial=bar["b4_partial"], b4_full=bar["b4_full"], b4_cells=bar["b4_cells"],
            b5_pass=bar["b5_pass"], b5_cells=bar["b5_cells"],
            all_pass=bar["all_pass"],
        ),
        verdict=verdict, n_configs=n_configs, max_ts=str(max_ts),
        max_ts_before_oos_start=bool(ok_ts),
    )
    RESULTS_PATH.write_text(json.dumps(_jsonable(result), indent=2))

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, state=state,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
