#!/usr/bin/env python
"""R-121 NOVEL branch: ``SigChi2TailProbKellyV4`` -- an ANALYTIC (closed-form)
chi-squared tail-probability discount on ``kelly_regime_v4``'s own
``frac * scale`` product, built entirely from ``experiments/r121_shared.py``'s
(operator-authored, READ-ONLY) shared infrastructure.

MECHANISM, in one sentence: build the 3-dimensional (log-price, log-realized-
vol, log-volume) depth-2 truncated PATH-SIGNATURE feature panel
(``r121_shared.build_sig3_features``, 6 columns: 3 total-increment terms plus
3 signed Levy-area cross terms), compute each day's SQUARED Mahalanobis
distance to the mean/covariance of its own trailing 730-day reference window
(``r121_shared.rolling_mahalanobis_distance(...) ** 2``, reusing R-109's own
causally-verified primitive unmodified), map that quadratic form through the
EXACT closed-form chi-squared(df=6) CDF (``r121_shared.chi2_cdf_even_df``,
an even-degrees-of-freedom Erlang/gamma(m,2) survival-function identity, no
scipy) to get an ANALYTIC ``[0,1]`` tail probability
(``r121_shared.rolling_signature_tailprob``), and multiplicatively discount
v4's unchanged exposure whenever today's tail probability is extreme -- the
SAME ``apply_discount``/threshold-ramp architecture R-87/R-104/R-105/R-106/
R-109/R-112/R-115 all use.

WHAT IS ACTUALLY NEW HERE, stated precisely (both halves matter, see
``r121_shared.py``'s own module docstring for the full non-duplication
argument against R-109/R-112/R-115, not re-derived here):
  (a) FEATURE REPRESENTATION: a path signature (order-sensitive: sees HOW
      price/vol/volume co-moved through the window, not just their
      marginal levels) in place of R-109's five point-in-time scalar
      moments (log_vol, anchor_disp, kurtosis, volume_z, skew).
  (b) SCORING MECHANISM -- the change specific to THIS branch, versus this
      round's own conservative sibling which keeps R-109's empirical
      percentile-rank scoring on top of the new features: this branch
      replaces the empirical rank ENTIRELY with a single closed-form
      analytic CDF, calibrated ONCE per reference window from the fitted
      Gaussian covariance, never re-ranked against the state's own
      historical trajectory. ``compute_full_state`` below returns
      ``r121_shared.rolling_signature_tailprob(...)`` DIRECTLY -- no
      ``causal_rolling_percentile_rank`` is applied on top of it. Motivated
      by Gasteratos, Jacquier, Lemercier, Lyons & Salvi (2025), "Novelty
      detection on path space" (arXiv:2512.03243, existence verified by the
      operator via WebSearch before this round was designed -- see
      ``r121_shared.py``'s own docstring), whose actual contribution is a
      formal tail-bound on the false-positive rate of a signature-based
      novelty test, in contrast to an ad hoc empirical percentile. This
      branch is an explicitly disclosed SIMPLIFICATION of that paper's own
      (non-Gaussian, more general) construction: a closed-form chi-squared
      tail probability under an approximate-Gaussian-reference assumption
      of the signature-feature distribution -- not a reproduction of the
      paper's full shuffle-product/CVaR/one-class-SVM machinery, which this
      environment cannot build (no scipy, confirmed; see
      ``r121_shared.py``'s dependency note). This is why the novel branch's
      path is 3-dimensional (6 signature features, an EVEN count): the
      feature count is chosen so the chi-squared CDF has an exact closed
      form rather than requiring a numerical incomplete-gamma
      approximation.

Full literature grounding, complete non-duplication argument against every
prior ERR-axis round (R-28/retracted, R-87, R-104, R-105, R-106, R-109,
R-112, R-115) and against this round's own conservative sibling, the hand-
verified Levy-area construction, and the dependency note on why df must be
even all live in ``experiments/r121_shared.py``'s own module docstring,
written by the operator before either branch was dispatched, and are NOT
re-derived here -- read that file in full first. This file imports ONLY
from ``experiments.r121_shared`` (itself chaining r109_shared ->
r106_shared -> ... -> r102_shared's unchanged control machinery, plus
r105_shared's B1-B5 promotion-bar helpers) and never edits it, never
coordinates with or reads the conservative branch's file, and never reads a
bar at or after ``r121_shared.OOS_START`` (2023-01-01).

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this):
  1. ``daily = r121_shared.build_sig3_features(df)`` -- the 6-column
     depth-2 signature panel over the (log-price, log-vol, log-volume)
     path, strictly causal (each day's row reads only bars strictly before
     that day's own first bar).
  2. ``state = r121_shared.rolling_signature_tailprob(daily)`` -- day t's
     ``chi2_cdf_even_df(rolling_mahalanobis_distance(daily)[t] ** 2,
     df=6)``: an ANALYTIC ``[0,1]`` tail probability against the trailing
     730-day reference window's own fitted covariance. Returned DIRECTLY,
     with NO empirical percentile-rank re-scoring on top -- the one
     substantive mechanism difference from R-109's own novel branch and
     from this round's own conservative sibling.
  3. ``target = r121_shared.apply_discount(df, state, thresh,
     max_discount) = r121_shared.v4_target(df) * (1 - discount)``, where
     ``discount`` ramps linearly from 0 at ``state <= thresh`` to
     ``max_discount`` at ``state == 1``. ``v4``'s own vote and scale are
     completely UNTOUCHED; only the ``frac * scale`` PRODUCT is discounted.

NUISANCE PARAMETERS -- there are none new to sweep. ``rolling_signature_
tailprob``'s own ``window``/``min_periods``/``ridge_eps`` defaults
(``BASELINE_WINDOW_DAYS``=730, ``MIN_REF_DAYS``=180, ``RIDGE_EPS``) are used
unmodified -- identical values to R-109's own reference-window architecture,
per this round's own explicit design (hold the window architecture fixed,
vary only the feature map and, on this branch, the scoring mechanism). No
window/ridge sweep is performed in this file.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read,
identical shape to R-109's own): sweep ``r121_shared.STEP0_THRESH_GRID x
r121_shared.STEP0_MAXD_GRID`` (3 x 2 = 6 cells) on BTC's
``INNER_TRAIN_START..INNER_TRAIN_END`` via ``r121_shared.step0_gate(df.loc[
INNER_TRAIN_START:INNER_TRAIN_END], state, thresh, max_discount)``, where
``state`` is computed over BTC's FULL non-holdout frame (mirrors R-109's own
step0_grid convention -- causality guarantees this is identical, for every
bar inside inner-train, to computing the statistic on the inner-train slice
alone). A cell QUALIFIES iff ``step0_gate(...)['passed']`` is True
(bind_frac > 1%, R^2 vs v4's own target < 0.98, R^2 vs v4's own realized-vol
input < 0.90, state CoV >= 5%). The operator has already confirmed on real
BTC inner-train data that the primary cell (thresh=0.90, max_discount=1.0)
PASSES (bind_frac=0.0876, r2_vs_v4=0.914, r2_vs_vol=-3.065, state_cv=1.177)
-- not re-derived here, only reproduced by running the identical grid.
PRIMARY cell: the first ``(thresh, max_discount)`` in
``r121_shared.SELECTION_ORDER`` that qualifies. If NONE qualify: STOP,
report Step-0 FAIL as a complete NEGATIVE result -- no inner-validation or
ETH bar is read past that point.

PROMOTION BAR (only if Step-0 passes; identical shape to R-109's own, via
``r121_shared``'s re-exported gate functions):
  B1 (gating): ``r121_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets (spot, futures_5x) -- dSharpe > +0.2 OR
     bootstrap excludes zero favourably.
  B2 (diagnostic ONLY, never gates): ``r121_shared.b2_diagnostic`` --
     drawdown improvement, counted only where risk-matched.
  B3 (plateau, gating): the FULL 6-cell Step-0 ``(thresh, max_discount)``
     grid's own inner-validation B1-style numbers (both markets each,
     primary cell's 2 rows reused directly from its own ``compare()``
     rather than recomputed) -- PASS requires a directionally consistent
     (same-sign) majority across the resulting 12 cells.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test per docs/ROUTINE.md step 2): ``r121_shared.
     b4_eth_falsification`` on ``r121_shared.compare(..., eth=
     r121_shared.load_eth())`` -- does the SAME-SIGN effect replicate on
     ETH? Require FULL pass (both markets same-signed as BTC inner_val).
     Named failure mode (from ``r121_shared.py``'s own docstring point 4):
     the reference distribution is fit predominantly on BTC's single
     2017-2020 supercycle and may not generalise to ETH at all -- exactly
     R-109/R-112/R-115's own repeated failure mode, now tested against a
     materially different novelty statistic per R-115's own closing line.
  B5 (cost robustness, gating): ``r121_shared.b5_fee_tier`` at the 0.40%
     taker tier, primary cell, BTC inner_val, both markets -- no sign
     reversal.
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau) AND B4 (full) AND B5 all hold (B2 is diagnostic-only).
Default: NEGATIVE. No threshold or decision rule is changed after seeing any
number -- anything contradicted by what actually happened is stated in the
results section below, never edited back into this banner.

CAUSAL SAFETY: ``r121_shared.causal_truncation_probe_series`` applied to
this file's own ``build_target`` (the FULL composed pipeline: sig3 features
-> squared Mahalanobis distance -> analytic chi-squared CDF state ->
discount -> ``v4_target * (1 - discount)``), run on BTC's full non-holdout
frame, BEFORE the Step-0 grid is scored and well before any inner-
validation/ETH performance number is computed. If it does not pass exactly,
that is a real bug to find and report, not something to work around.

WHAT WOULD MAKE THIS FAIL, named now (``r121_shared.py``'s own five
pre-registered concerns, restated here as this branch's specific exposure
to each -- see that module's docstring for the full statement of each):
(1) the sig3 panel's tail probability collapsing into a relabelled
realized-vol rescale (guarded by Step-0's own ``R2_VS_VOL_THRESH`` kill
switch); (2) v4's own reactive, latched vote already pricing in
order-sensitive novelty as fast as a signature can detect it, reproducing
the "real but inert" (Step-0 passes, B1 does not) pattern; (3) the Levy-area
term being highly correlated with R-109's own point-in-time features in
practice despite being mathematically distinct in construction (diagnostic
only in ``r121_shared.py``, not re-checked here); (4) BTC-specific reference
calibration failing to generalise to ETH (B4, this round's pre-registered
falsification test); (5) THIS BRANCH'S OWN SPECIFIC RISK: the closed-form
chi-squared tail probability assumes the sig3-feature reference distribution
is approximately multivariate Gaussian -- if the true reference distribution
is heavy-tailed or multimodal, the analytic p-value may be systematically
mis-calibrated (e.g. an empirical distribution of the state that is far from
uniform on days when it "should" be, under the chi-squared assumption). This
is checked below as a diagnostic (the state's own empirical decile
occupancy on BTC inner-train) -- not a formal gate, since even a
miscalibrated-but-informative statistic can still pass the actual B1-B5 bar,
but flagged explicitly if it looks badly off.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets, 2 of
the 12 reused directly from the primary ``compare()``'s own inner_val rows,
10 freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total. IF
Step-0 finds no qualifying cell, this file stops after the 6 Step-0 cells
(6 total).

USAGE
-----
    python experiments/r121_novel_signature_tailbound.py
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

from experiments.r121_shared import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    MIN_REF_DAYS,
    OOS_START,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    SELECTION_ORDER,
    SPOT,
    STEP0_MAXD_GRID,
    STEP0_THRESH_GRID,
    apply_discount,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    build_sig3_features,
    causal_truncation_probe_series,
    compare,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    print_plateau_table,
    print_rows,
    rolling_signature_tailprob,
    step0_gate,
)

# ---------------------------------------------------------- pre-registered
# r121_shared does not itself re-export PRIMARY_THRESH/PRIMARY_MAXD as named
# module-level constants, so they are derived here from SELECTION_ORDER[0]
# (r109_shared's own pre-registered primary cell, unchanged by this round) --
# tied to the shared module rather than hand-copied, and asserted against the
# documented value so a stale SELECTION_ORDER would be caught immediately.
PRIMARY_THRESH, PRIMARY_MAXD = SELECTION_ORDER[0]
assert (PRIMARY_THRESH, PRIMARY_MAXD) == (0.90, 1.0), (
    f"SELECTION_ORDER[0] changed from the documented (0.90, 1.0) to "
    f"{SELECTION_ORDER[0]} -- pre-registration text is stale")


# ================================================================== (1)
# The mechanism itself: sig3 signature panel -> squared Mahalanobis distance
# -> ANALYTIC chi-squared(df=6) CDF state (no percentile-rank re-scoring) ->
# discount on v4's own UNCHANGED frac*scale.
# ==================================================================

def compute_full_state(df: pd.DataFrame) -> pd.Series:
    """sig3 features -> analytic chi-squared tail-probability state, over
    whatever frame `df` is (the caller decides how much history it
    contains -- this function itself makes no reference to any fixed
    calendar date). Returned DIRECTLY from `rolling_signature_tailprob` --
    NO `causal_rolling_percentile_rank` is applied on top: this state is
    already an analytic [0,1] tail probability, not a raw distance needing
    empirical ranking. This is the one substantive mechanism difference
    from both R-109's own novel (kNN) branch and from this round's own
    conservative sibling."""
    daily = build_sig3_features(df)
    return rolling_signature_tailprob(daily)


def build_target(df: pd.DataFrame, thresh: float = PRIMARY_THRESH,
                  max_discount: float = PRIMARY_MAXD) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount),
    where discount is driven by the analytic chi-squared tail-probability
    state built from `df` alone. Self-contained (a pure function of `df`),
    so it is directly usable as a `TargetStrategy` candidate on any window
    (inner_train, inner_val, eth_replication, or a truncated probe frame)."""
    state = compute_full_state(df)
    return apply_discount(df, state, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"sig_chi2_tailprob_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r121_shared.STEP0_THRESH_GRID x r121_shared.STEP0_MAXD_GRID,
# scored via r121_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame (see banner above for why this is causally
# identical to computing state on the inner-train slice alone, for every bar
# inside it).
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    state = compute_full_state(btc)
    df_inner_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    rows = []
    for thresh in STEP0_THRESH_GRID:
        for maxd in STEP0_MAXD_GRID:
            gate = step0_gate(df_inner_train, state, thresh, maxd)
            rows.append(dict(thresh=thresh, max_discount=maxd, **gate))
    return rows, state


def select_primary(rows: list[dict]) -> dict | None:
    by_key = {(r["thresh"], r["max_discount"]): r for r in rows}
    for key in SELECTION_ORDER:
        r = by_key.get(key)
        if r is not None and r["passed"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars, state = analytic chi2(df=6) tail-prob of squared Mahalanobis "
          f"distance in sig3 feature space, BASELINE_WINDOW_DAYS={BASELINE_WINDOW_DAYS}, "
          f"MIN_REF_DAYS={MIN_REF_DAYS})")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r2_vs_v4 < {R2_VS_V4_THRESH} "
          f"AND r2_vs_vol < {R2_VS_VOL_THRESH} AND state_cv >= {CV_KILL_THRESH:.0%}")
    hdr = (f"{'thresh':>7s} {'max_d':>6s} {'bind_frac':>10s} {'r2_vs_v4':>9s} "
           f"{'r2_vs_vol':>9s} {'state_cv':>9s} {'passed':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = "  <- SELECTION_ORDER[0]" if (r["thresh"], r["max_discount"]) == SELECTION_ORDER[0] else ""
        print(f"{r['thresh']:7.2f} {r['max_discount']:6.2f} {r['bind_frac']:10.4f} "
              f"{r['r2_vs_v4']:9.4f} {r['r2_vs_vol']:9.4f} {r['state_cv']:9.4f} "
              f"{'YES' if r['passed'] else 'no':>7s}{tag}")


def print_state_calibration_diagnostic(state: pd.Series, btc: pd.DataFrame) -> None:
    """Diagnostic only (item (5) in the module docstring's failure-mode
    list): if the chi-squared/Gaussian-reference approximation were
    well-calibrated, the state's own empirical distribution on BTC
    inner-train would be close to UNIFORM on [0,1] (that is what a
    genuine CDF of the true data-generating distribution would produce
    when evaluated on iid draws from that same distribution). Reported as
    decile occupancy -- not a gate, never used to reject or accept
    anything below."""
    s = state.loc[INNER_TRAIN_START:INNER_TRAIN_END].dropna()
    hr("DIAGNOSTIC (not a gate): state calibration -- empirical decile occupancy on BTC inner-train")
    print("If the chi-squared/Gaussian-reference approximation were well-calibrated, each decile")
    print("bucket below would hold ~10% of observations. Large deviations indicate the analytic")
    print("tail probability is systematically mis-calibrated versus the true reference distribution")
    print("(a disclosed, pre-registered risk of this branch's simplification -- see module docstring).")
    if len(s) == 0:
        print("  (no non-NaN state observations on inner-train -- cannot compute)")
        return
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(s.to_numpy(), bins=edges)
    frac = counts / counts.sum()
    for i in range(10):
        bar = "#" * int(round(frac[i] * 50))
        print(f"  [{edges[i]:.1f},{edges[i+1]:.1f}) {frac[i]:6.2%}  {bar}")
    print(f"  n={len(s):,}  min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}  "
          f"(uniform reference mean would be 0.5)")


# ================================================================== (3)
# B3 plateau: the full 6-cell (thresh, max_discount) grid's own
# inner-validation numbers (both markets), primary cell's 2 rows reused
# directly from its own compare().
# ==================================================================

def run_b3_full_grid(step0_rows: list[dict], primary_key: tuple[float, float],
                      inner_val_primary: list[dict], btc: pd.DataFrame) -> tuple[dict, bool]:
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
            bf = make_build_target(*key)
            label = f"sig_chi2_tailprob_t{key[0]:g}_m{key[1]:g}"
            plateau_rows[key] = inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# above), B4 (gating falsification), B5 (gating fee tier).
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                       btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    thresh, maxd = primary_key
    build_primary = make_build_target(thresh, maxd)
    label = f"sig_chi2_tailprob_t{thresh:g}_m{maxd:g}"

    hr(f"PROMOTION BAR -- PRIMARY CELL thresh={thresh:g}, max_discount={maxd:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                    markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3_full_grid(step0_rows, primary_key, inner_val_primary, btc)
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)
    b5_pass, b5_cells = b5_fee_tier(build_primary, label, btc, inner_val_primary)

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

    hr("R-121 NOVEL: SigChi2TailProbKellyV4 -- analytic chi-squared "
       "tail-probability discount on v4's own frac*scale")
    print("mechanism: 3D (log-price, log-vol, log-volume) depth-2 truncated path-signature")
    print("panel (6 features: 3 total-increment + 3 signed Levy-area cross terms) -> squared")
    print("Mahalanobis distance to the trailing 730-day reference window's own fitted")
    print("covariance -> EXACT closed-form chi-squared(df=6) CDF (no scipy) as an ANALYTIC [0,1]")
    print("tail probability -> linear discount on v4's UNCHANGED frac*scale product. NO empirical")
    print("percentile-rank re-scoring on top -- the state above IS the final novelty state. A")
    print("disclosed simplification of Gasteratos, Jacquier, Lemercier, Lyons & Salvi (2025)'s")
    print("signature-based novelty tail-bound framing under a Gaussian-reference approximation.")
    print("Full grounding in r121_shared.py's own module docstring.")
    print(f"\nSTEP0_THRESH_GRID={STEP0_THRESH_GRID}  STEP0_MAXD_GRID={STEP0_MAXD_GRID}  "
          f"({len(STEP0_THRESH_GRID) * len(STEP0_MAXD_GRID)} cells)")
    print(f"SELECTION_ORDER={SELECTION_ORDER}")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ================================================== CAUSAL SAFETY FIRST
    hr("CAUSAL TRUNCATION PROBE (composed build_target at the pre-registered primary "
       "(thresh, max_discount), real BTC data, run BEFORE Step-0 or any inner-val/ETH number)")
    probe_fn = make_build_target(PRIMARY_THRESH, PRIMARY_MAXD)
    print(f"causal_truncation_probe_series({probe_fn.__name__}, btc):")
    try:
        probe_ok = causal_truncation_probe_series(probe_fn, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL SAFETY (truncation probe) PASS: {probe_ok}")

    if not probe_ok:
        hr("VERDICT")
        print("CAUSAL SAFETY PROBE FAILED -- this is a real bug, not a result. Stopping here per")
        print("the pre-registered rule: a lookahead is a bug report first, per docs/ROUTINE.md's")
        print("own precedence. No Step-0 grid, inner-validation number, or ETH bar is read.")
        print(f"causal truncation probe: {probe_ok}")
        print("Step-0: NOT COMPUTED (causal-safety stop)")
        print("B1: NOT COMPUTED (causal-safety stop)")
        print("B2: NOT COMPUTED (causal-safety stop)")
        print("B3: NOT COMPUTED (causal-safety stop)")
        print("B4: NOT COMPUTED (causal-safety stop)")
        print("B5: NOT COMPUTED (causal-safety stop)")
        print("VERDICT: NEGATIVE (causal-safety probe failure)")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated (total): 0")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, probe_ok=probe_ok, passed_step0=False, n_configs=0,
                    max_ts=max_ts, verdict="NEGATIVE (causal-safety probe failure)")

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY / NON-DUPLICATION KILL SWITCH "
       "(run BEFORE any inner-validation Sharpe/PnL number)")
    step0_rows, state = step0_grid(btc)
    n_bars_inner_train = int(np.sum((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                     (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train)
    print_state_calibration_diagnostic(state, btc)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the analytic")
        print("chi-squared tail-probability discount is either a near-total no-op, a near-exact")
        print("rescale of v4's own path, a relabelled volatility rescale, or degenerate everywhere")
        print("on the pre-registered grid.")
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
        print(f"\nconfigurations evaluated (total): {n_configs} (6 Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                    probe_ok=probe_ok, n_configs=n_configs, max_ts=max_ts,
                    verdict="NEGATIVE (Step-0 kill switch)")

    primary_key = (primary_row["thresh"], primary_row["max_discount"])
    is_selection0 = (primary_key == SELECTION_ORDER[0])
    print(f"\nPRIMARY CELL SELECTED (Step-0 non-degeneracy rule only): "
          f"thresh={primary_key[0]:g}, max_discount={primary_key[1]:g}  "
          f"(bind_frac={primary_row['bind_frac']:.4f}, r2_vs_v4={primary_row['r2_vs_v4']:.4f}, "
          f"r2_vs_vol={primary_row['r2_vs_vol']:.4f}, state_cv={primary_row['state_cv']:.4f})")
    print(f"  selection: {'SELECTION_ORDER[0] qualified' if is_selection0 else 'SELECTION_ORDER[0] did NOT qualify; next qualifying cell in SELECTION_ORDER chosen'}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    bar = run_promotion_bar(primary_key, step0_rows, btc, eth)

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
    print_plateau_table(bar["b3_rows"])
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
    print(f"Step-0: PASS  primary cell thresh={primary_key[0]:g}, max_discount={primary_key[1]:g}  "
          f"(bind_frac={primary_row['bind_frac']:.4f}, r2_vs_v4={primary_row['r2_vs_v4']:.4f}, "
          f"r2_vs_vol={primary_row['r2_vs_vol']:.4f}, state_cv={primary_row['state_cv']:.4f})")
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
          f"2 B5 fee-tier)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, state=state,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
