#!/usr/bin/env python
"""R-112 CONSERVATIVE branch: ``ReturnspaceKnnNoveltyBrakeKellyV4`` -- R-109
novel branch's own k-nearest-neighbour DISTRIBUTIONAL-NOVELTY discount on
``kelly_regime_v4``'s ``frac * scale`` product, with EXACTLY ONE change: the
``anchor_disp`` feature (R-109's mean pairwise %% dispersion among SMA(20/40/
80) PRICE-LEVEL anchors) is replaced by
``experiments.r112_shared.feature_anchor_dispersion_returns`` (the same
20/40/80-horizon dispersion computed from rolling MEAN LOG RETURNS instead of
SMA price levels). Every other feature (``log_vol``, ``kurtosis``,
``volume_z``, ``skew``), the reference construction (single-asset, strictly-
prior, trailing 730-day window), the distance metric (kNN, ``k=10``,
``refit_every=30``), the Step-0 grid, and the full B1-B5 promotion bar are
IDENTICAL to R-109's novel branch, unchanged.

This file implements the CONSERVATIVE half of R-109's own named, disclosed
follow-on (`docs/LEDGER.md`, R-109 Verdict section): "a reference
distribution NOT dominated by one price cycle (e.g. detrended/return-space
features only, or an explicit multi-asset reference pool) might close the
novel branch's B4 gap without needing a new mechanism." The full literature
grounding, non-duplication argument against R-109 and every prior ERR-axis
round, and the four named failure risks all live in
``experiments/r112_shared.py``'s own module docstring, written by the
operator before either R-112 branch was dispatched, and are NOT re-derived
here -- read that file in full first. This file imports ONLY from
``experiments.r112_shared`` (itself chaining ``r109_shared`` ->
``r105_shared`` -> ``r102_shared``'s unchanged control machinery, plus
``r63_shared`` for the NOVEL branch's own pooling, unused here) and never
edits it, never coordinates with or reads the novel branch's file (which
attacks the reference-SET half of the same follow-on sentence via
CORAL-pooling across ``r63_shared.UNIVERSE_6``), and never reads a bar at or
after ``r112_shared.OOS_START`` (2023-01-01).

MECHANISM, in one sentence: identical to R-109 novel -- build a 5-feature
OHLCV-only daily market-state panel, score each day's mean Euclidean distance
(per-feature standardized) to its ``k`` nearest neighbours in its own
trailing 730-day reference SET, normalize to a causal [0,1] percentile-rank
"novelty state", and multiplicatively discount v4's unchanged exposure
whenever today's state looks unlike its own recent history -- except the
``anchor_disp`` feature is now built entirely from RETURNS (rolling mean log
return over 20/40/80 days), never from a price level or SMA of one, so the
panel as a whole no longer carries any feature keyed on absolute price
trajectory through BTC's own 2017-2020 supercycle.

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this; diff against R-109 novel is confined to
step 1's feature dict):

  1. ``daily = r112_shared.build_daily_features(df,
     r112_shared.RETURNSPACE_FEATURE_BUILDERS)`` -- the 5-feature panel
     (``log_vol``, ``anchor_disp`` [now return-space], ``kurtosis``,
     ``volume_z``, ``skew``), all pure OHLCV-derived, causal (each column
     already shifted by 1 bar inside its own builder), zero new data
     channels. THE ONLY CHANGE from R-109 novel: ``RETURNSPACE_FEATURE_
     BUILDERS`` in place of ``NOVEL_FEATURE_BUILDERS`` -- a dict identical
     in every key except ``anchor_disp``, which now calls
     ``r112_shared.feature_anchor_dispersion_returns`` instead of
     ``r109_shared.feature_anchor_dispersion``.
  2. ``dist = r112_shared.rolling_knn_distance(daily, k=K, refit_every=
     REFIT_EVERY)`` -- UNCHANGED from R-109 novel: day t's mean Euclidean
     distance (per-feature z-scored against the SAME reference window used
     to fit) to its ``k`` nearest neighbours among the strictly-prior 730
     days of THIS SAME INSTRUMENT's own history, refit every
     ``REFIT_EVERY`` days (walk-forward, strictly causal).
  3. ``state = r112_shared.causal_rolling_percentile_rank(dist)`` --
     UNCHANGED: a causal rolling percentile rank onto [0, 1] against the
     state's own trailing 730-day history.
  4. ``target = r112_shared.apply_discount(df, state, thresh,
     max_discount) = r112_shared.v4_target(df) * (1 - discount)`` --
     UNCHANGED architecture: ``v4``'s own vote and scale are completely
     untouched; only the ``frac * scale`` PRODUCT is discounted.

``k`` AND ``refit_every`` -- held at R-109's own pre-registered defaults,
``k=10`` and ``refit_every=30`` (verified programmatically below against the
live function signature, not hand-copied), per this round's own explicit
instruction not to sweep them: R-109's reasoning for not searching them
(a search would itself be a hidden trial; the kNN literature's own typical
range brackets 10 already) is unchanged by swapping one feature's
definition, so it is not re-litigated here. No sweep of ``k``/
``refit_every`` is performed in this file -- the Step-0 grid below
(``thresh``/``max_discount`` only) is the entire configuration search this
branch performs before inner-validation is read.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read,
IDENTICAL to R-109 novel): sweep ``r112_shared.STEP0_THRESH_GRID x
r112_shared.STEP0_MAXD_GRID`` (3 x 2 = 6 cells) on BTC's
``INNER_TRAIN_START..INNER_TRAIN_END`` via ``r112_shared.step0_gate(df.loc
[INNER_TRAIN_START:INNER_TRAIN_END], state, thresh, max_discount)``, where
``state`` is computed over BTC's FULL non-holdout frame (causality
guarantees this is identical, for every bar inside inner-train, to computing
the statistic on the inner-train slice alone). A cell QUALIFIES iff
``step0_gate(...)['passed']`` is True (bind_frac > 1%, R^2 vs v4's own
target < 0.98, R^2 vs v4's own realized-vol input < 0.90, state CoV >= 5%).
PRIMARY cell: the first ``(thresh, max_discount)`` in
``r112_shared.SELECTION_ORDER`` that qualifies. If NONE qualify: STOP,
report Step-0 FAIL as a complete NEGATIVE result -- no inner-validation or
ETH bar is read past that point.

PROMOTION BAR (only if Step-0 passes; identical shape and thresholds to
R-109, via ``r112_shared``'s re-exported gate functions):
  B1 (gating): ``r112_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets (spot, futures_5x) -- dSharpe > +0.2 OR
     bootstrap excludes zero favourably.
  B2 (diagnostic ONLY, never gates): ``r112_shared.b2_diagnostic`` --
     drawdown improvement, counted only where risk-matched.
  B3 (plateau, gating): the FULL 6-cell Step-0 ``(thresh, max_discount)``
     grid's own inner-validation B1-style numbers (both markets each,
     primary cell's 2 rows reused directly from its own ``compare()``
     rather than recomputed) -- PASS requires a directionally consistent
     (same-sign) majority across the resulting 12 cells.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test per docs/ROUTINE.md step 2): ``r112_shared.
     b4_eth_falsification`` on ``r112_shared.compare(..., eth=
     r112_shared.load_eth())`` -- does the SAME-SIGN effect replicate on
     ETH? Require FULL pass (both markets same-signed as BTC inner_val).
     This is the exact clause R-109 novel FAILED (spot inverted sign); this
     branch's entire reason to exist is testing whether removing the one
     price-level-anchored feature closes that gap.
  B5 (cost robustness, gating): ``r112_shared.b5_fee_tier`` at the 0.40%
     taker tier, primary cell, BTC inner_val, both markets -- no sign
     reversal.
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau) AND B4 (full) AND B5 all hold (B2 is diagnostic-only).
Default: NEGATIVE. No threshold or decision rule is changed after seeing
any number -- anything contradicted by what actually happened is stated in
the results section below, never edited back into this banner.

CAUSAL SAFETY: ``r112_shared.causal_truncation_probe_series`` applied to
this file's own ``build_target`` (the FULL composed pipeline: features ->
kNN distance -> percentile-rank state -> discount -> ``v4_target * (1 -
discount)``), run on BTC's full non-holdout frame, BEFORE the Step-0 grid
is scored and well before any inner-validation/ETH performance number is
computed. If it does not pass exactly, that is a real bug in this file's
own new feature builder to find and report, not something to work around.

WHAT WOULD MAKE THIS FAIL, named now (``r112_shared.py``'s own pre-registered
concerns (1) and (2), restated as this branch's specific exposure to each):
(1) the B4 generalization gap may not be a feature-choice artifact at all,
but a deeper property of ``kelly_regime_v4``'s own vote interacting with
ETH's shorter history -- if so, this branch does not close B4 either, which
is itself informative (it would show R-109's own most likely cause was not
the true one). (2) the return-space ``anchor_disp`` may simply carry less
information than the price-level one it replaces, reproducing a Step-0-
passes/B1-fails "real but inert" pattern on BTC itself, before B4 is even
reached.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets, 2 of
the 12 reused directly from the primary ``compare()``'s own inner_val rows,
10 freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total. IF
Step-0 finds no qualifying cell, this file stops after the 6 Step-0 cells
(6 total). No ``k``/``refit_every`` sweep is performed, so it adds 0
configurations to either count.

USAGE
-----
    python experiments/r112_conservative_returnspace_knn.py
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r112_shared import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    MIN_REF_DAYS,
    OOS_START,
    PRIMARY_MAXD,
    PRIMARY_THRESH,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    RETURNSPACE_FEATURE_BUILDERS,
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
    build_daily_features,
    causal_rolling_percentile_rank,
    causal_truncation_probe_series,
    compare,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    print_plateau_table,
    print_rows,
    rolling_knn_distance,
)

# ---------------------------------------------------------- pre-registered
K = 10             # rolling_knn_distance's own default -- verified below;
                    # NOT swept, per this round's own instruction (matches
                    # R-109 novel's own pre-registered reasoning)
REFIT_EVERY = 30   # rolling_knn_distance's own default -- verified below

_sig = inspect.signature(rolling_knn_distance).parameters
assert _sig["k"].default == K, ("K does not match rolling_knn_distance's "
                                 f"own default ({_sig['k'].default}) -- pre-registration text is stale")
assert _sig["refit_every"].default == REFIT_EVERY, (
    "REFIT_EVERY does not match rolling_knn_distance's own default "
    f"({_sig['refit_every'].default}) -- pre-registration text is stale")


# ================================================================== (1)
# The mechanism itself: 5-feature panel (return-space anchor_disp) -> kNN
# distance -> percentile-rank state -> discount on v4's own UNCHANGED
# frac*scale. Identical to R-109 novel except `RETURNSPACE_FEATURE_BUILDERS`
# in place of `NOVEL_FEATURE_BUILDERS`.
# ==================================================================

def compute_full_state(df: pd.DataFrame, k: int = K, refit_every: int = REFIT_EVERY) -> pd.Series:
    """features -> kNN distance -> causal percentile-rank state, over
    whatever frame `df` is (the caller decides how much history it
    contains -- this function itself makes no reference to any fixed
    calendar date). THE ONLY CHANGE from R-109 novel's own function of the
    same name: `RETURNSPACE_FEATURE_BUILDERS` instead of
    `NOVEL_FEATURE_BUILDERS`."""
    daily = build_daily_features(df, RETURNSPACE_FEATURE_BUILDERS)
    dist = rolling_knn_distance(daily, k=k, refit_every=refit_every)
    return causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, thresh: float = PRIMARY_THRESH,
                  max_discount: float = PRIMARY_MAXD, k: int = K,
                  refit_every: int = REFIT_EVERY) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount),
    where discount is driven by the return-space-feature kNN novelty state
    built from `df` alone. Self-contained (a pure function of `df`), so it
    is directly usable as a `TargetStrategy` candidate on any window
    (inner_train, inner_val, eth_replication, or a truncated probe frame)."""
    state = compute_full_state(df, k=k, refit_every=refit_every)
    return apply_discount(df, state, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"returnspace_knn_novelty_brake_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r112_shared.STEP0_THRESH_GRID x r112_shared.STEP0_MAXD_GRID,
# scored via r112_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame. Identical structure to R-109 novel.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    from experiments.r112_shared import step0_gate  # local import, matches r109_shared's own idiom
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
          f"{n_bars:,} bars, state built from k={K}, refit_every={REFIT_EVERY}d, "
          f"BASELINE_WINDOW_DAYS={BASELINE_WINDOW_DAYS}, MIN_REF_DAYS={MIN_REF_DAYS}, "
          f"feature panel=RETURNSPACE_FEATURE_BUILDERS)")
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


# ================================================================== (3)
# B3 plateau: the full 6-cell (thresh, max_discount) grid's own
# inner-validation numbers (both markets), primary cell's 2 rows reused
# directly from its own compare(). Identical to R-109 novel.
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
            label = f"returnspace_knn_novelty_brake_t{key[0]:g}_m{key[1]:g}"
            plateau_rows[key] = inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# above), B4 (gating falsification), B5 (gating fee tier). Identical to
# R-109 novel's own promotion-bar shape.
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                       btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    thresh, maxd = primary_key
    build_primary = make_build_target(thresh, maxd)
    label = f"returnspace_knn_novelty_brake_t{thresh:g}_m{maxd:g}"

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

    hr("R-112 CONSERVATIVE: ReturnspaceKnnNoveltyBrakeKellyV4 -- R-109 novel's "
       "kNN novelty brake with anchor_disp replaced by a return-space analogue")
    print("mechanism: 5-feature OHLCV-only daily market-state panel (log_vol, anchor_disp")
    print("[RETURN-SPACE, this round's only change vs R-109 novel], kurtosis, volume_z, skew) ->")
    print("mean Euclidean distance (per-feature standardized) to the k nearest neighbours in a")
    print("trailing 730-day SINGLE-ASSET reference SET, refit every refit_every days (walk-forward,")
    print("strictly causal) -> causal rolling percentile-rank state in [0,1] -> linear discount on")
    print("v4's UNCHANGED frac*scale product. Full grounding in r112_shared.py's own module docstring.")
    print(f"\nk={K}, refit_every={REFIT_EVERY}d  (both = rolling_knn_distance's own defaults, "
          f"verified programmatically above; no sweep performed, per this round's own instruction)")
    print(f"STEP0_THRESH_GRID={STEP0_THRESH_GRID}  STEP0_MAXD_GRID={STEP0_MAXD_GRID}  "
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

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY / NON-DUPLICATION KILL SWITCH "
       "(run BEFORE any inner-validation Sharpe/PnL number)")
    step0_rows, state = step0_grid(btc)
    n_bars_inner_train = int(np.sum((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                     (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the return-space")
        print("kNN novelty discount is either a near-total no-op, a near-exact rescale of v4's own")
        print("path, a relabelled volatility rescale, or degenerate everywhere on the pre-registered")
        print("grid. Per this file's own pre-registration, this Step-0 table (plus the causal-safety")
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
          f"2 B5 fee-tier; k/refit_every not swept, adds 0)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, state=state,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
