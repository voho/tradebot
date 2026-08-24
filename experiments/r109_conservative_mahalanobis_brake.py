#!/usr/bin/env python
"""R-109 CONSERVATIVE branch: ``MahalanobisNoveltyBrakeKellyV4`` -- ``kelly_regime_v4``'s
own unchanged ``frac * scale`` product (``v4_target``, deadband already applied),
multiplied by a bounded, monotonic discount driven by the literal, textbook
Mahalanobis (1936; De Maesschalck, Jouan-Rimbaud & Massart 2000) distance of
TODAY's 3-feature market-state vector from the mean/covariance of its OWN
trailing reference distribution -- a DISTRIBUTIONAL-NOVELTY / dataset-shift
uncertainty proxy (Rabanser, Gunnemann & Lipton 2019, "Failing Loudly"),
independent of any model's output, disagreement, or historical significance.
Full citation trail, literature grounding, the axis this attacks (ERR -- no
error control anywhere in the signal path), and the exhaustive non-duplication
argument against every related prior round (R-28/retracted, R-87, R-104,
R-105, R-106, every SIZE-axis round, every regime-timing round) all live in
``experiments/r109_shared.py``'s own module docstring (read in full before
this file was written); not re-derived here beyond the one-paragraph summaries
in item 3 below. This file NEVER edits ``r109_shared.py`` (or any other file
under ``experiments/``/``src/``), and never reads a bar at or after
``OOS_START`` (2023-01-01) from any data source, regardless of outcome.

=====================================================================
PRE-REGISTRATION (frozen before any real-data bind_frac, R^2, state_cv, or
backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in the
results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): build a 3-feature daily market-state panel
   (log realized volatility, anchor-ladder dispersion, return kurtosis --
   all pure, causal, OHLCV-derived, ``r109_shared.FEATURE_BUILDERS``),
   compute each day's Mahalanobis distance from the mean/covariance of its
   own strictly-prior 730-day (min 180-day) reference window, normalize that
   distance to a [0,1] novelty STATE via a causal rolling percentile rank
   against its own trailing history, and discount ``kelly_regime_v4``'s own
   unchanged, already-deadbanded final target multiplicatively whenever
   TODAY's state looks unlike its own recent past -- regardless of whether
   any model disagrees with any other model (R-106) or whether the vote's
   historical edge is statistically significant (R-104).

2. CONSTRUCTION (exact, entirely ``r109_shared`` primitives, no new logic
   added on top -- this is what makes this the CONSERVATIVE, literal-
   textbook-statistic branch, contrasted with the NOVEL branch's
   nonparametric k-nearest-neighbour distance, built independently and not
   read or coordinated with here):

       daily[t]        = r109_shared.build_daily_features(df)[t]        # 3 cols:
                          # log_vol, anchor_disp, kurtosis (default FEATURE_BUILDERS)
       dist[t]         = r109_shared.rolling_mahalanobis_distance(daily)[t]
                          # covariance-weighted distance of daily[t] from
                          # mean/cov of daily[t-730 : t-1] (strictly prior days,
                          # min 180 prior days required, else NaN)
       state[t]        = r109_shared.causal_rolling_percentile_rank(dist,
                              window=BASELINE_WINDOW_DAYS,
                              min_periods=MIN_REF_DAYS)[t]                # in [0,1]
       discount_frac[t] = r109_shared.discount_series_for(df, state,
                              thresh, max_discount)[t]                    # in [0,1],
                          # 0 below `thresh`, ramps linearly to `max_discount`
                          # as state -> 1 (r109_shared.novelty_discount)
       target[t]        = v4_target(df)[t] * (1.0 - discount_frac[t])
                          # == r109_shared.apply_discount(df, state, thresh,
                          #    max_discount)[t] -- v4's OWN final,
                          #    already-deadbanded target, multiplicatively
                          #    discounted. Nothing else touches `frac` or
                          #    `scale`.

   DEFAULT/PRIMARY CONFIG: (`thresh`, `max_discount`) selected by the
   pre-registered Step-0 grid + `SELECTION_ORDER` below (grid entry
   (0.90, 1.0), `r109_shared.PRIMARY_THRESH`/`PRIMARY_MAXD`, preferred first,
   confirmed or overridden only by that rule, never by a performance number).

3. WHY THIS IS NOT A DUPLICATE (summary only -- the full argument, with
   citations, lives in ``r109_shared.py``'s own docstring, read in full
   before this file was written): every prior ERR-axis round (R-28, R-87,
   R-104, R-105, R-106) measured uncertainty as a property of the VOTE's own
   edge, confidence, significance, or cross-model spread. This construction
   never reads ``kelly_regime_v4``'s returns, exposure, or P&L to build its
   statistic (unlike R-104's sampling-significance branches), never
   constructs an alternative anchor ladder or jackknifes the vote's own
   three components (unlike R-105), and imports none of R-106's four
   detector modules or their cross-sectional dispersion (unlike R-106) --
   its distance statistic is a property of ONE feature vector (today's
   market state) against ONE reference distribution (that same feature's
   own trailing history), computable with no second model, vote, or P&L
   series to compare against at all. It is also not a duplicate of any
   regime-timing round (R-01, R-82, R-83, R-85, R-86, R-96, R-98, R-99):
   this file computes no detection lag and races against no dated
   stress-episode calendar; it produces a continuous distance/novelty
   statistic at every bar, gated at Step-0 on its own dispersion and
   degeneracy, never on a hit-rate against ``STRESS_EPISODES``.

4. STEP-0 GRID AND SELECTION RULE (pre-registered, run BEFORE any
   inner-validation Sharpe/PnL number is read): sweep
   ``r109_shared.STEP0_THRESH_GRID = (0.80, 0.90, 0.95)`` x
   ``r109_shared.STEP0_MAXD_GRID = (0.5, 1.0)`` -- 6 cells, fixed a priori --
   on BTC's inner-train window (``INNER_TRAIN_START`` -> ``INNER_TRAIN_END``)
   via ``r109_shared.step0_gate(df_inner_train, state_full_history, thresh,
   max_discount)``, where ``state_full_history`` is this file's own
   ``novelty_state(btc)`` computed ONCE on the FULL non-holdout BTC frame
   (2017 -> pre-2023 -- giving the 730-day reference window genuine history
   to draw on well before inner-train even starts) and ``df_inner_train =
   btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]`` is the caller-restricted
   frame ``step0_gate`` itself further restricts its bind_frac/R^2 checks to
   (its own ``state_cv`` check reads the passed-in `state` series' own raw
   dispersion, unrestricted by `df` -- a documented property of
   ``r109_shared.step0_gate`` used exactly as instructed, not a bug in this
   file). Each cell reports ``bind_frac``, ``r2_vs_v4``, ``r2_vs_vol``,
   ``state_cv``, ``passed`` (= ``bind_frac > 1%`` AND ``r2_vs_v4 < 0.98``
   AND ``r2_vs_vol < 0.90`` AND ``state_cv >= 5%``, all four
   ``r109_shared``-fixed thresholds).

   SELECTION RULE (non-degeneracy ONLY -- no performance number is inspected
   before this rule is applied): the primary cell is the first
   ``(thresh, max_discount)`` in ``r109_shared.SELECTION_ORDER =
   ((0.90,1.0), (0.95,1.0), (0.80,0.5), (0.90,0.5), (0.95,0.5), (0.80,1.0))``
   that qualifies. If NONE of the six cells qualify, this file STOPS at
   Step-0 -- a legitimate, informative NEGATIVE result per
   ``docs/ROUTINE.md``, not a bug to route around. No causal probe, B1-B5
   code, or ETH load runs in that case, and no bar on/after ``OOS_START``
   (2023-01-01) is ever touched either way.

5. DISCLOSED, DERIVED (NOT FIT) WARMUP OVERRIDE, same class of correction
   R-106's conservative branch made under its own name (its "detector
   warmup" section 6): ``TargetStrategy``'s framework default warmup
   (v4's own 80-day anchor requirement, ``80*BARS_PER_DAY+10``) is shorter
   than this construction's OWN genuine reference-window requirement
   (``max(V4_HORIZONS)=80`` days for the anchor-dispersion feature itself to
   first be valid, PLUS ``r109_shared.MIN_REF_DAYS=180`` more days before the
   Mahalanobis distance is defined at all -- 260 days minimum). Calling
   ``r109_shared.compare()``/``inner_val_rows()``/``b5_fee_tier()`` VERBATIM
   would silently instantiate the candidate with the 80-day default,
   truncating the state to its NaN fallback (``discount_series_for``'s own
   ``.fillna(0.0)`` -> no brake, identical to v4) for a large fraction of
   every non-inner-train slice, for a data-availability reason rather than a
   genuine one (BTC has 4 full years of pre-2021 history available; the
   80-day default just never asks for more of it). This file therefore
   defines ``compare_107``/``inner_val_rows_107``/``b5_fee_tier_107`` --
   BYTE-IDENTICAL to ``r102_shared.compare()``/``r105_shared.inner_val_rows()``/
   ``r105_shared.b5_fee_tier()`` except the CANDIDATE ``TargetStrategy`` is
   instantiated with ``warmup=CANDIDATE_WARMUP_BARS`` (300 days -- a
   round-number margin above the 260-day derived minimum, verified against
   the observed first-valid-state timestamp on real BTC data below) instead
   of the 80-day class default. The CONTROL (``kelly_regime_v4``) strategy's
   warmup is left completely UNCHANGED in every call, so every
   ``d_sharpe``/paired-bootstrap comparison in this file remains
   apples-to-apples between candidate and control (only the candidate sees
   more pre-slice history to warm its own indicators; both start trading at
   the identical bar). The dict-only, no-strategy-construction gate helpers
   (``b1_from_inner_val``, ``b2_diagnostic``, ``b4_eth_falsification``) are
   reused VERBATIM from ``r109_shared`` (re-exported from ``r105_shared`` --
   they only ever read already-computed row dicts, so the warmup change
   flows through correctly without touching their code).

6. CAUSAL TRUNCATION PROBE, run before trusting any Step-0 or promotion-bar
   number: ``r109_shared.causal_truncation_probe_series`` applied to this
   file's own composed ``build_target`` closure (bound to the selected
   primary ``(thresh, max_discount)``, or ``(0.90, 1.0)`` if Step-0 finds no
   primary) -- the FULL pipeline (features -> distance -> state -> discount
   -> ``v4_target * (1-discount)``), on real BTC data. In addition, a wiring
   identity check: ``build_target(df, thresh=T, max_discount=0.0)`` (a
   ``max_discount`` OUTSIDE the pre-registered grid, used only as a
   wiring check) must equal ``v4_target(df)`` EXACTLY for any `T`, since
   ``max_discount=0`` forces ``discount_frac`` to be identically zero
   regardless of `state` -- confirming the discount map and ``v4_target``
   are wired together correctly with no accidental double-application or
   scale error.

7. PROMOTION BAR (docs/ROUTINE.md's own bar, operationalized via
   ``r109_shared``'s centralized B1/B2/B4 machinery plus this file's own
   warmup-aware B3/B5, reused from R-105/R-106's own convention):
     B1 (gating): ``r109_shared.b1_from_inner_val`` on the primary
        ``compare_107()`` call's own ``inner_val`` rows, both markets --
        ``d_sharpe > +0.2`` OR the paired bootstrap interval's lower bound
        excludes zero on the positive side.
     B2 (diagnostic ONLY, never itself gates promotion):
        ``r109_shared.b2_diagnostic`` -- drawdown improvement, counted only
        where risk-matched (R-33's own standing rule).
     B3 (plateau, gating): the FULL pre-registered 6-cell Step-0 grid's own
        inner-validation B1 numbers (both markets, 12 rows total -- the
        primary cell's 2 rows reused directly from the primary
        ``compare_107()`` call, the other 10 freshly computed via
        ``inner_val_rows_107``), reported via ``print_plateau_table``. PASS
        requires the primary cell's IMMEDIATE GRID NEIGHBOURS (same
        ``max_discount``, adjacent ``thresh`` in the 3-point sorted grid;
        same ``thresh``, the other ``max_discount`` in the 2-point grid) to
        show the SAME SIGN of ``d_sharpe`` as the primary cell, per market
        -- "a plateau, not an isolated spike," exactly per
        ``docs/ROUTINE.md``'s own promotion-bar language, without requiring
        the neighbours to also clear +0.2.
     B4 (ETH falsification, gating, PRE-REGISTERED AS THIS ROUND'S ONE
        FALSIFICATION TEST per ``docs/ROUTINE.md`` step 2, chosen now,
        before any inner-validation number is read): does the SAME-SIGN
        effect on BTC inner-validation REPLICATE on ETH?
        ``r109_shared.b4_eth_falsification`` -- require the FULL pass (both
        markets same-signed as BTC inner-validation). This is the single
        pre-registered falsification test for this branch; nothing else in
        this file is held out as an alternative.
     B5 (fee-tier robustness, gating): ``b5_fee_tier_107`` at the primary
        cell, 0.40% taker, both markets -- no sign reversal on either
        ``d_sharpe`` or the bootstrap log-growth point estimate.
   PROMOTE-candidate only if Step-0 selects a primary AND the causal probe
   passes AND B1 AND B3 AND B4 (full form) AND B5 all hold (B2 is
   diagnostic-only). Default: NEGATIVE. If ALL of these pass, this file
   reports a HOLDOUT CANDIDATE and STOPS -- it does not itself read, print,
   or hold in memory any bar at or after ``OOS_START`` (2023-01-01); the
   centralized holdout-consultation decision is the operator's, made after
   both R-109 branches report, per ``docs/ROUTINE.md``'s "Running directions
   in parallel" section.

8. WHAT WOULD MAKE THIS FAIL: named already, in full, in ``r109_shared.py``
   itself (four specific, independent failure risks -- collapsing into a
   relabelled volatility rescale, guarded by the R2_VS_VOL_THRESH kill
   switch; v4's own reactive latched vote already pricing in "unusual"
   conditions by the time they register as distributionally novel,
   reproducing the R-87/R-104/R-105/R-106 "real but inert" pattern; a
   rolling 730-day reference drifting with a slow multi-year regime so the
   brake only ever fires at the speed of a genuine break rather than
   persisting through it; and a reference distribution dominated by BTC's
   single 2017-2020 supercycle failing to generalise to ETH, exactly what B4
   is designed to catch). Not re-derived here; reported honestly, whichever
   way it comes out, in the results below.

CONFIGURATIONS EVALUATED IN THIS FILE (IF Step-0 selects a primary): 6
(Step-0 grid, thresh x max_discount) + 6 (primary config's full
``compare_107()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets -- 2 of
the 12 reused directly from the primary ``compare_107()``'s own inner_val
rows, 10 freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total.
IF Step-0 finds no qualifying cell, this file stops after the 6 Step-0 cells
and reports that outcome directly (no causal probe, B1-B5 code, or ETH data
ever touched).

----------------------------------------------------------------------
Run: python experiments/r109_conservative_mahalanobis_brake.py
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

from experiments.r109_shared import (  # noqa: E402
    BARS_PER_DAY,
    BASELINE_WINDOW_DAYS,
    ETH_SLICE_NAME,
    FEE_TIER,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_REF_DAYS,
    OOS_START,
    PRIMARY_MAXD,
    PRIMARY_THRESH,
    SELECTION_ORDER,
    SHARPE_NOISE_FLOOR,
    SLICES,
    SPOT,
    STEP0_MAXD_GRID,
    STEP0_THRESH_GRID,
    TargetStrategy,
    V4_HORIZONS,
    apply_discount,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    build_daily_features,
    causal_rolling_percentile_rank,
    causal_truncation_probe_series,
    fee_at,
    hr,
    load_btc,
    load_eth,
    paired_diff,
    print_plateau_table,
    print_rows,
    rolling_mahalanobis_distance,
    run_slice,
    step0_gate,
    v4_target,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert STEP0_THRESH_GRID == (0.80, 0.90, 0.95), STEP0_THRESH_GRID
assert STEP0_MAXD_GRID == (0.5, 1.0), STEP0_MAXD_GRID
assert SELECTION_ORDER[0] == (PRIMARY_THRESH, PRIMARY_MAXD), SELECTION_ORDER

# ---------------------------------------------------------- pre-registered
# Derived (not fit): max anchor horizon (80d, for the anchor_disp feature's
# own first-valid point) + MIN_REF_DAYS (180d, for the Mahalanobis distance's
# own min reference) + a 40-day round-number margin = 300 days.
CANDIDATE_WARMUP_DAYS = max(V4_HORIZONS) + MIN_REF_DAYS + 40   # = 300
CANDIDATE_WARMUP_BARS = int(CANDIDATE_WARMUP_DAYS * BARS_PER_DAY) + 10


# ================================================================== (1)
# The mechanism itself: build_daily_features -> rolling_mahalanobis_distance
# -> causal_rolling_percentile_rank -> apply_discount. Every step is an
# r109_shared primitive; nothing new is added on top.
# ==================================================================

def novelty_state(df: pd.DataFrame) -> pd.Series:
    """The [0,1] novelty STATE for every day covered by `df`: the causal
    rolling-percentile-rank normalization of the rolling Mahalanobis
    distance of the 3-feature daily panel from its own strictly-prior
    reference window."""
    daily = build_daily_features(df)
    dist = rolling_mahalanobis_distance(daily)
    return causal_rolling_percentile_rank(dist, window=BASELINE_WINDOW_DAYS,
                                          min_periods=MIN_REF_DAYS)


def build_target(df: pd.DataFrame, thresh: float, max_discount: float) -> np.ndarray:
    """v4's own unchanged, already-deadbanded target, multiplicatively
    discounted by this bar's own novelty state -- computed FRESH from
    whatever frame `df` actually is (the full pipeline is a pure function of
    `df` alone, as required for use as a `TargetStrategy` closure)."""
    state = novelty_state(df)
    return apply_discount(df, state, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"mahalanobis_brake_t{thresh:g}_m{max_discount:g}"
    return _build


def cell_label(thresh: float, max_discount: float) -> str:
    return f"mahalanobis_brake_t{thresh:g}_m{max_discount:g}"


# ================================================================== (2)
# Pre-flight self-tests.
# ==================================================================

def self_test_max_discount_zero_identity(df: pd.DataFrame) -> bool:
    """max_discount=0.0 (OUTSIDE the pre-registered grid, wiring check
    only): discount_frac === 0.0 regardless of state -> build_target must
    equal v4_target exactly, for any thresh."""
    ok = True
    for thresh in (0.5, 0.90, 0.999):
        a = build_target(df, thresh=thresh, max_discount=0.0)
        b = v4_target(df)
        same = np.allclose(a, b, equal_nan=True)
        print(f"  build_target(thresh={thresh:g}, max_discount=0.0) == v4_target exactly? {same}")
        ok = ok and same
    return ok


def self_test_first_valid_state(btc: pd.DataFrame, state_full: pd.Series) -> int:
    fv = state_full.first_valid_index()
    days = (fv - btc.index[0]).days if fv is not None else -1
    print(f"  first valid novelty state: {fv}  ({days} days after frame start)  "
          f"CANDIDATE_WARMUP_BARS covers {CANDIDATE_WARMUP_BARS / BARS_PER_DAY:.0f} days")
    return days


# ================================================================== (3)
# Step-0: 6-cell (thresh x max_discount) grid, via r109_shared.step0_gate.
# ==================================================================

def step0_grid(btc: pd.DataFrame, state_full: pd.Series) -> list[dict]:
    df_inner_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    rows = []
    for thresh in STEP0_THRESH_GRID:
        for max_discount in STEP0_MAXD_GRID:
            gate = step0_gate(df_inner_train, state_full, thresh, max_discount)
            rows.append(dict(thresh=thresh, max_discount=max_discount, **gate))
    return rows


def select_primary(rows: list[dict]) -> dict | None:
    by_key = {(r["thresh"], r["max_discount"]): r for r in rows}
    for key in SELECTION_ORDER:
        r = by_key.get(key)
        if r is not None and r["passed"]:
            return r
    return None


def print_step0_table(rows: list[dict]) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"state built on the FULL non-holdout BTC history)")
    print("QUALIFY = bind_frac>1% AND r2_vs_v4<0.98 AND r2_vs_vol<0.90 AND state_cv>=5%")
    hdr = (f"{'thresh':>7s} {'max_d':>6s} {'bind_frac':>10s} {'r2_vs_v4':>9s} "
          f"{'r2_vs_vol':>10s} {'state_cv':>9s} {'passed':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = " <- PRIMARY (pre-registered default)" if (r["thresh"], r["max_discount"]) == SELECTION_ORDER[0] else ""
        print(f"{r['thresh']:7.2f} {r['max_discount']:6.2f} {r['bind_frac']:10.4f} "
              f"{r['r2_vs_v4']:9.4f} {r['r2_vs_vol']:10.4f} {r['state_cv']:9.4f} "
              f"{'YES' if r['passed'] else 'no':>7s}{tag}")


# ================================================================== (4)
# compare_107 / inner_val_rows_107 -- byte-identical to r102_shared.compare()
# / r105_shared.inner_val_rows() except the CANDIDATE TargetStrategy uses
# CANDIDATE_WARMUP_BARS instead of the 80-day framework default. See
# docstring item 5.
# ==================================================================

def compare_107(candidate_build, *, label: str, btc: pd.DataFrame, eth: pd.DataFrame,
                markets: tuple = (SPOT, FUTURES), include_eth: bool = True,
                seed: int = 0) -> list[dict]:
    assert_no_holdout(btc, "compare_107(): btc")
    if include_eth:
        assert_no_holdout(eth, "compare_107(): eth")

    cand = TargetStrategy(candidate_build, name=f"r109_{label}", warmup=CANDIDATE_WARMUP_BARS)
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in SLICES.items()]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(cand, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                        if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                        if b.realized_vol else float("nan"))
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
                "d_log_growth": a.log_growth - b.log_growth,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
                "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                                if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
                "boot_d_loggrowth": pr.diff.point,
                "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def inner_val_rows_107(build_fn, label: str, btc: pd.DataFrame,
                       markets: tuple = (SPOT, FUTURES)) -> list[dict]:
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r109_{label}", warmup=CANDIDATE_WARMUP_BARS)
    rows = []
    for market in markets:
        a = run_slice(cand, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        b = run_slice(ctrl, btc, INNER_VAL_START, INNER_VAL_END, "inner_val", market)
        pr = paired_diff(a.daily, b.daily)
        exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                    if b.mean_abs_exposure else float("nan"))
        vol_ratio = (a.realized_vol / b.realized_vol
                    if b.realized_vol else float("nan"))
        risk_matched = (bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                       if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False)
        rows.append(dict(
            label=label, market=market.name,
            d_sharpe=a.sharpe - b.sharpe, d_dd=a.max_drawdown_pct - b.max_drawdown_pct,
            exposure_ratio=exp_ratio, vol_ratio=vol_ratio, risk_matched=risk_matched,
            boot_d_loggrowth=pr.diff.point, boot_lo=pr.diff.lo, boot_hi=pr.diff.hi,
            excludes_zero=bool(pr.diff.lo > 0 or pr.diff.hi < 0),
        ))
    return rows


def b5_fee_tier_107(build_primary, label: str, btc: pd.DataFrame,
                    inner_val_primary: list[dict]) -> tuple[bool, list[dict]]:
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_rows_107(build_primary, label, btc, markets=fee_markets)
    cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_primary if c["market"] == r["market"]), None)
        d_sharpe_no_reversal = (base is not None and
                               not (np.sign(r["d_sharpe"]) != np.sign(base["d_sharpe"])
                                    and r["d_sharpe"] != 0 and base["d_sharpe"] != 0))
        dlog_no_reversal = (base is not None and
                          not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"])
                               and r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                          base_d_sharpe=base["d_sharpe"] if base else float("nan"),
                          boot_d_loggrowth=r["boot_d_loggrowth"],
                          base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                          d_sharpe_no_reversal=d_sharpe_no_reversal,
                          dlog_no_reversal=dlog_no_reversal,
                          no_reversal=d_sharpe_no_reversal and dlog_no_reversal))
    return all(c["no_reversal"] for c in cells), cells


# ================================================================== (5)
# B3: plateau over the FULL 6-cell (thresh x max_discount) Step-0 grid.
# ==================================================================

def b3_plateau(primary_key: tuple[float, float],
               plateau_rows: dict[tuple[float, float], list[dict]]) -> tuple[bool, list[dict]]:
    t0, m0 = primary_key
    sorted_thresh = sorted(STEP0_THRESH_GRID)
    sorted_maxd = sorted(STEP0_MAXD_GRID)
    ti = sorted_thresh.index(t0)
    mi = sorted_maxd.index(m0)
    neighbour_keys = []
    for i in (ti - 1, ti + 1):
        if 0 <= i < len(sorted_thresh) and sorted_thresh[i] != t0:
            neighbour_keys.append((sorted_thresh[i], m0))
    for j in (mi - 1, mi + 1):
        if 0 <= j < len(sorted_maxd) and sorted_maxd[j] != m0:
            neighbour_keys.append((t0, sorted_maxd[j]))

    primary_signs = {r["market"]: np.sign(r["d_sharpe"]) for r in plateau_rows[primary_key]}
    detail = []
    ok = True
    for nb in neighbour_keys:
        for r in plateau_rows[nb]:
            prim_sign = primary_signs.get(r["market"], 0.0)
            same = bool(np.sign(r["d_sharpe"]) == prim_sign)
            detail.append(dict(neighbour_key=nb, market=r["market"], d_sharpe=r["d_sharpe"],
                               primary_d_sharpe=next(pr["d_sharpe"] for pr in plateau_rows[primary_key]
                                                     if pr["market"] == r["market"]),
                               same_sign_as_primary=same))
            ok = ok and same
    return ok, detail


# ================================================================== (6)
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


# ================================================================== (7)
# Promotion bar: B1 (gating), B2 (diagnostic), B3 (full-grid plateau,
# gating), B4 (ETH falsification, gating), B5 (fee tier, gating).
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                      btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    thresh, max_discount = primary_key
    label = cell_label(thresh, max_discount)
    build_primary = make_build_target(thresh, max_discount)

    hr(f"PROMOTION BAR -- PRIMARY CONFIG thresh={thresh:g} max_discount={max_discount:g}")
    print("compare_107() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare_107(build_primary, label=label, btc=btc, eth=eth,
                       markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    # ---- B1
    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)

    # ---- B2 (diagnostic only)
    b2_ok, b2_cells = b2_diagnostic(inner_val_primary)

    # ---- B3: full 6-cell Step-0 grid, both markets, primary reused.
    plateau_rows: dict[tuple[float, float], list[dict]] = {
        primary_key: [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                           d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                           vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                           boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                           boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                      for r in inner_val_primary]
    }
    for row in step0_rows:
        key = (row["thresh"], row["max_discount"])
        if key == primary_key:
            continue
        bf = make_build_target(*key)
        blabel = cell_label(*key)
        plateau_rows[key] = inner_val_rows_107(bf, blabel, btc)

    b3_pass, b3_detail = b3_plateau(primary_key, plateau_rows)

    # ---- B4: ETH falsification (this round's ONE pre-registered
    # falsification test).
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)

    # ---- B5: fee-tier robustness
    hr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary config, BTC inner-validation")
    b5_pass, b5_cells = b5_fee_tier_107(build_primary, label, btc, inner_val_primary)

    n_b3_rows = sum(len(v) for v in plateau_rows.values())

    return dict(
        label=label, thresh=thresh, max_discount=max_discount,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells,
        b3_plateau_rows=plateau_rows, b3_detail=b3_detail, b3_pass=b3_pass,
        b4_cells=b4_cells, b4_partial_pass=b4_partial, b4_full_pass=b4_full,
        b5_cells=b5_cells, b5_pass=b5_pass,
        n_configs_promotion_bar=6 + n_b3_rows + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-109 CONSERVATIVE: MahalanobisNoveltyBrakeKellyV4 -- literal "
       "Mahalanobis-distance dataset-shift brake on v4's own final target")
    print("mechanism: multiply v4's UNCHANGED, already-deadbanded final target by (1 - discount),")
    print("where discount ramps from 0 to max_discount as TODAY's 3-feature market-state vector's")
    print("Mahalanobis distance from its own trailing 730-day (min 180-day) reference distribution")
    print("rises past `thresh` (as a causal rolling percentile rank in [0,1]). Pure textbook")
    print("De Maesschalck/Jouan-Rimbaud/Massart (2000) statistic; no other logic added.")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    hr("PRE-FLIGHT SELF-TESTS (before any Step-0 number is trusted)")
    print("max_discount=0.0 wiring identity (outside the pre-registered grid):")
    identity_ok = self_test_max_discount_zero_identity(btc)
    print(f"  -> max_discount=0.0 identity: {identity_ok}")

    print("\nbuilding full-history novelty state on the FULL non-holdout BTC frame "
          "(used ONLY for the Step-0 gate, per this file's own pre-registration):")
    state_full = novelty_state(btc)
    fv_days = self_test_first_valid_state(btc, state_full)

    if not identity_ok:
        print("\nSELF-TEST FAILURE -- stopping before any Step-0 number is trusted.")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(verdict="ABORTED (self-test failure)", max_ts=max(max_ts_seen))

    # ============================================================= STEP 0
    hr("STEP 0 -- 6-CELL (thresh x max_discount) GATE (run BEFORE any Sharpe/compare() number)")
    step0_rows = step0_grid(btc, state_full)
    print_step0_table(step0_rows)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell passes all four Step-0 clauses (bind_frac>1%, r2_vs_v4<0.98,")
        print("r2_vs_vol<0.90, state_cv>=5%) on the pre-registered 6-cell grid. Per this file's")
        print("own pre-registration, this Step-0 table is the branch's ENTIRE product, reported")
        print("NEGATIVE / stopped-at-Step-0. No causal probe, B1-B5 code, or ETH load runs.")
        n_configs = len(step0_rows)
        print(f"\nconfigurations evaluated: {n_configs} (Step-0 grid only)")
        max_ts = max(max_ts_seen)
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                   n_configs=n_configs, max_ts=max_ts, verdict="NEGATIVE (Step-0 gate)")

    primary_key = (primary_row["thresh"], primary_row["max_discount"])
    is_default = (primary_key == SELECTION_ORDER[0])
    print(f"\nPRIMARY CONFIG SELECTED (non-degeneracy rule only): thresh={primary_key[0]:g} "
          f"max_discount={primary_key[1]:g}  (bind_frac={primary_row['bind_frac']:.4f}, "
          f"r2_vs_v4={primary_row['r2_vs_v4']:.4f}, r2_vs_vol={primary_row['r2_vs_vol']:.4f}, "
          f"state_cv={primary_row['state_cv']:.4f})")
    print(f"  selection: {'pre-registered default cell qualified' if is_default else 'default cell did NOT qualify; next cell in SELECTION_ORDER chosen'}")

    build_primary = make_build_target(*primary_key)

    # ==================================================== CAUSAL PROBE
    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data)")
    probe_ok = run_causal_probe(btc, build_primary)
    print(f"\nCAUSAL PROBE PASS: {probe_ok}")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    bar = run_promotion_bar(primary_key, step0_rows, btc, eth)

    hr("B1 -- inner-validation Sharpe leg, both markets "
       "(dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  voided={c['voided']}")

    hr("B3 -- plateau: full 6-cell Step-0 grid at primary selection, "
       "inner-validation, both markets")
    print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 immediate-neighbour same-sign detail:")
    for d in bar["b3_detail"]:
        print(f"  neighbour={d['neighbour_key']!s:>16s} {d['market']:>9s}  "
              f"d_sharpe={d['d_sharpe']:+.4f}  primary_d_sharpe={d['primary_d_sharpe']:+.4f}  "
              f"same_sign={d['same_sign_as_primary']}")
    print(f"\nB3 (primary's immediate grid neighbours all same-signed): {bar['b3_pass']}")

    hr("B4 -- ETH falsification (PRE-REGISTERED as this round's one falsification test)")
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
    verdict = "HOLDOUT CANDIDATE (all gates pass; operator makes the centralized holdout call)" if all_gates_pass else "NEGATIVE"
    print(f"\nALL GATING CLAUSES PASS (causal AND B1 AND B3 AND B4-full AND B5): {all_gates_pass}")
    print(f"VERDICT: {verdict}")
    if not all_gates_pass:
        failed = [name for name, ok in (
            ("causal probe", probe_ok), ("B1", bar["b1_pass"]), ("B3", bar["b3_pass"]),
            ("B4 (full)", bar["b4_full_pass"]), ("B5", bar["b5_pass"]),
        ) if not ok]
        print(f"Reason(s): {', '.join(failed)}")

    n_configs = len(step0_rows) + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"({len(step0_rows)} Step-0 grid + 6 primary compare_107() + "
          f"{sum(len(v) for v in bar['b3_plateau_rows'].values())} B3 full-grid rows "
          f"[2 reused from primary + rest fresh] + 2 B5 fee-tier)")
    max_ts = max(max_ts_seen)
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
    print("NO bar at or after 2023-01-01 was ever read by this file, regardless of outcome.")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary_row, passed_step0=True,
               probe_ok=probe_ok, promotion_bar=bar, verdict=verdict, n_configs=n_configs,
               max_ts=max_ts)


if __name__ == "__main__":
    main()
