#!/usr/bin/env python
"""R-109 NOVEL branch: ``KnnNoveltyBrakeKellyV4`` -- a nonparametric,
k-nearest-neighbour DISTRIBUTIONAL-NOVELTY discount on ``kelly_regime_v4``'s
own ``frac * scale`` product, the density-based sibling of the conservative
branch's single fitted Gaussian (Mahalanobis) construction, built entirely
from ``experiments/r109_shared.py``'s (operator-authored, READ-ONLY) shared
infrastructure.

MECHANISM, in one sentence: build a richer 5-feature OHLCV-only daily market-
state panel (realized vol, anchor-ladder dispersion, return kurtosis, volume
z-score, return skew), score each day's mean Euclidean distance (in per-
feature standardized space) to its ``k`` nearest neighbours in its own
trailing 730-day reference SET (Ramaswamy, Rastogi & Shim 2000; Breunig,
Kriegel, Ng & Sander 2000's "LOF" family -- nonparametric, density-based,
makes no single-Gaussian/elliptical assumption about the reference
distribution's shape, unlike the conservative branch's Mahalanobis distance),
normalize that distance to a causal [0,1] percentile-rank "novelty state",
and multiplicatively discount v4's unchanged exposure whenever today's state
looks unlike its own recent history. The full literature grounding (Rabanser,
Gunnemann & Lipton 2019 "Failing Loudly" for the general dataset-shift
framing; De Maesschalck, Jouan-Rimbaud & Massart 2000 for the conservative
branch's Mahalanobis alternative; Ramaswamy et al. 2000 / Breunig et al. 2000
for this branch's kNN family; Liu, Ting & Zhou 2008 "Isolation Forest" as the
literature's closely-related tree-ensemble alternative, cited as this
branch's own motivation for choosing a genuinely different ALGORITHM CLASS
-- nonparametric/density-based rather than a single fitted Gaussian, not
merely a different parameter grid on the same statistic) and the complete
non-duplication argument against every prior ERR-axis round (R-28/retracted,
R-87, R-104, R-105, R-106) and against the conservative branch's Mahalanobis
construction all live in ``experiments/r109_shared.py``'s own module
docstring, written by the operator before either branch was dispatched, and
are NOT re-derived here -- read that file in full first. This file imports
ONLY from ``experiments.r109_shared`` (itself chaining r106_shared ->
r105_shared -> ... -> r102_shared's unchanged control machinery) and never
edits it, never coordinates with or reads the conservative branch's file, and
never reads a bar at or after ``r109_shared.OOS_START`` (2023-01-01).

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this):

  1. ``daily = r109_shared.build_daily_features(df,
     r109_shared.NOVEL_FEATURE_BUILDERS)`` -- the 5-feature panel
     (``log_vol``, ``anchor_disp``, ``kurtosis``, ``volume_z``, ``skew``),
     all pure OHLCV-derived, causal (each column already shifted by 1 bar
     inside its own builder), zero new data channels.
  2. ``dist = r109_shared.rolling_knn_distance(daily, k=K, refit_every=
     REFIT_EVERY)`` -- day t's mean Euclidean distance (per-feature
     z-scored against the SAME reference window used to fit) to its ``k``
     nearest neighbours among the strictly-prior 730 days, refit every
     ``REFIT_EVERY`` days (walk-forward, still strictly causal by
     construction -- verified below and already covered by
     ``r109_shared.py``'s own self-test).
  3. ``state = r109_shared.causal_rolling_percentile_rank(dist)`` -- a
     causal rolling percentile rank onto [0, 1] against the state's own
     trailing 730-day history.
  4. ``target = r109_shared.apply_discount(df, state, thresh,
     max_discount) = r109_shared.v4_target(df) * (1 - discount)``, where
     ``discount`` ramps linearly from 0 at ``state <= thresh`` to
     ``max_discount`` at ``state == 1``. ``v4``'s own vote and scale are
     completely UNTOUCHED; only the ``frac * scale`` PRODUCT is discounted,
     exactly the same slot-in architecture R-87/R-104/R-105/R-106 all use.

``k`` AND ``refit_every`` -- PRE-REGISTERED NUISANCE-PARAMETER CHOICE, before
any real-data number was read: kept at ``rolling_knn_distance``'s own
defaults, ``k=10`` and ``refit_every=30`` (verified programmatically below
against the live function signature, not hand-copied). Reasoning: the kNN
outlier-scoring literature (Ramaswamy et al. 2000; Breunig et al. 2000's LOF)
typically reports ``k`` in the 10-50 range for moderate-size reference sets,
and 10 is the conservative (small) end of that range appropriate for a
730-day reference window on a single-instrument daily panel -- no
diagnostics were run to search for a "better" ``k`` before touching
inner-validation data, per this project's standing discipline (a search here
would itself be a hidden trial). ``refit_every=30`` (a monthly walk-forward
cadence) matches every prior regime/turbulence-detector round's own refit or
recompute convention in this ledger (BOCPD/Kalman/CSD/Hawkes all effectively
recompute at a similar or finer cadence) and is explicitly named in
``rolling_knn_distance``'s own docstring as "the more realistic deployable
construction." No sweep of ``k``/``refit_every`` is performed in this
file -- this project's convention is 1-5 variants per branch, not an
open-ended search, and no diagnostics-based reason to deviate from the
shared module's own defaults was found. This choice contributes exactly
ONE ``(k, refit_every)`` configuration to the round; the Step-0 grid below
(swept over ``thresh``/``max_discount`` only) is the entire configuration
search this branch performs before inner-validation is read.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read):
sweep ``r109_shared.STEP0_THRESH_GRID x r109_shared.STEP0_MAXD_GRID`` (3 x 2
= 6 cells) on BTC's ``INNER_TRAIN_START..INNER_TRAIN_END`` via
``r109_shared.step0_gate(df.loc[INNER_TRAIN_START:INNER_TRAIN_END], state,
thresh, max_discount)``, where ``state`` is computed over BTC's FULL
non-holdout frame (mirrors R-105's own step0_grid convention: the statistic
is computed once over the whole available series, then only the inner-train
MASK is scored -- causality guarantees this is identical, for every bar
inside inner-train, to computing the statistic on the inner-train slice
alone, since inner-train already starts at the earliest available date and
no bar's state value can depend on any later bar). A cell QUALIFIES iff
``step0_gate(...)['passed']`` is True (bind_frac > 1%, R^2 vs v4's own
target < 0.98, R^2 vs v4's own realized-vol input < 0.90, state CoV >=
5%). PRIMARY cell: the first ``(thresh, max_discount)`` in
``r109_shared.SELECTION_ORDER`` that qualifies. If NONE qualify: STOP,
report Step-0 FAIL as a complete NEGATIVE result -- no inner-validation or
ETH bar is read past that point.

PROMOTION BAR (only if Step-0 passes; identical shape to every SIZE/ERR-axis
round since R-89, via ``r109_shared``'s re-exported gate functions):
  B1 (gating): ``r109_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets (spot, futures_5x) -- dSharpe > +0.2 OR
     bootstrap excludes zero favourably.
  B2 (diagnostic ONLY, never gates): ``r109_shared.b2_diagnostic`` --
     drawdown improvement, counted only where risk-matched.
  B3 (plateau, gating): the FULL 6-cell Step-0 ``(thresh, max_discount)``
     grid's own inner-validation B1-style numbers (both markets each,
     primary cell's 2 rows reused directly from its own ``compare()``
     rather than recomputed) -- PASS requires a directionally consistent
     (same-sign) majority across the resulting 12 cells, not an isolated
     spike. Since ``k``/``refit_every`` are NOT swept (see above), this
     6-cell ``(thresh, max_discount)`` grid is the entirety of this
     branch's plateau evidence.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test per docs/ROUTINE.md step 2): ``r109_shared.
     b4_eth_falsification`` on ``r109_shared.compare(..., eth=
     r109_shared.load_eth())`` -- does the SAME-SIGN effect replicate on
     ETH? Require FULL pass (both markets same-signed as BTC inner_val).
     Named failure mode (from ``r109_shared.py``'s own docstring point 4):
     the reference distribution is fit predominantly on BTC's single
     2017-2020 supercycle and may not generalise to ETH's shorter,
     differently-shaped history at all.
  B5 (cost robustness, gating): ``r109_shared.b5_fee_tier`` at the 0.40%
     taker tier, primary cell, BTC inner_val, both markets -- no sign
     reversal.
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau) AND B4 (full) AND B5 all hold (B2 is diagnostic-only).
Default: NEGATIVE. No threshold or decision rule is changed after seeing
any number -- anything contradicted by what actually happened is stated in
the results section below, never edited back into this banner.

CAUSAL SAFETY: ``r109_shared.causal_truncation_probe_series`` applied to
this file's own ``build_target`` (the FULL composed pipeline: features ->
kNN distance -> percentile-rank state -> discount -> ``v4_target * (1 -
discount)``), run on BTC's full non-holdout frame, BEFORE the Step-0 grid
is scored and well before any inner-validation/ETH performance number is
computed. ``rolling_knn_distance`` refits only every ``REFIT_EVERY`` days
for speed/realism, but a refit at day t only ever uses days strictly before
t (verified directly in ``r109_shared.py``'s own self-test with an explicit
truncation comparison) -- the probe is expected to pass exactly; if it does
not, that is a real bug to find and report, not something to work around.

WHAT WOULD MAKE THIS FAIL, named now (all four are ``r109_shared.py``'s own
pre-registered concerns, restated here as this branch's specific exposure to
each): (1) the 5-feature panel's kNN distance collapsing into a relabelled
realized-vol rescale (guarded by Step-0's own ``R2_VS_VOL_THRESH`` kill
switch); (2) v4's own reactive, latched vote already pricing in "unusual"
conditions by the time they register as distributionally novel, reproducing
the R-87/R-104/R-105/R-106 "real but inert" pattern (Step-0 passes, B1 does
not) by a sixth, structurally different estimator and a SECOND algorithm
class (nonparametric, after the conservative branch's parametric one) within
the SAME round; (3) the 730-day rolling reference window itself drifting
with a slow multi-year volatility regime, so novelty only ever registers at
the speed of a genuine break rather than persisting through the stress
period the strategy needs protecting through; (4) BTC-specific reference
calibration failing to generalise to ETH (B4).

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets, 2 of
the 12 reused directly from the primary ``compare()``'s own inner_val rows,
10 freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total. IF
Step-0 finds no qualifying cell, this file stops after the 6 Step-0 cells
(6 total). No ``k``/``refit_every`` sweep is performed (see above), so it
adds 0 configurations to either count.

USAGE
-----
    python experiments/r109_novel_knn_novelty_brake.py
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

from experiments.r109_shared import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    MIN_REF_DAYS,
    NOVEL_FEATURE_BUILDERS,
    OOS_START,
    PRIMARY_MAXD,
    PRIMARY_THRESH,
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
K = 10             # rolling_knn_distance's own default -- verified below
REFIT_EVERY = 30   # rolling_knn_distance's own default -- verified below

_sig = inspect.signature(rolling_knn_distance).parameters
assert _sig["k"].default == K, ("K does not match rolling_knn_distance's "
                                 f"own default ({_sig['k'].default}) -- pre-registration text is stale")
assert _sig["refit_every"].default == REFIT_EVERY, (
    "REFIT_EVERY does not match rolling_knn_distance's own default "
    f"({_sig['refit_every'].default}) -- pre-registration text is stale")


# ================================================================== (1)
# The mechanism itself: 5-feature panel -> kNN distance -> percentile-rank
# state -> discount on v4's own UNCHANGED frac*scale.
# ==================================================================

def compute_full_state(df: pd.DataFrame, k: int = K, refit_every: int = REFIT_EVERY) -> pd.Series:
    """features -> kNN distance -> causal percentile-rank state, over
    whatever frame `df` is (the caller decides how much history it
    contains -- this function itself makes no reference to any fixed
    calendar date)."""
    daily = build_daily_features(df, NOVEL_FEATURE_BUILDERS)
    dist = rolling_knn_distance(daily, k=k, refit_every=refit_every)
    return causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, thresh: float = PRIMARY_THRESH,
                  max_discount: float = PRIMARY_MAXD, k: int = K,
                  refit_every: int = REFIT_EVERY) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount),
    where discount is driven by the kNN novelty state built from `df`
    alone. Self-contained (a pure function of `df`), so it is directly
    usable as a `TargetStrategy` candidate on any window (inner_train,
    inner_val, eth_replication, or a truncated probe frame)."""
    state = compute_full_state(df, k=k, refit_every=refit_every)
    return apply_discount(df, state, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"knn_novelty_brake_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r109_shared.STEP0_THRESH_GRID x r109_shared.STEP0_MAXD_GRID,
# scored via r109_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame (mirrors R-105's own step0_grid
# convention -- see banner above for why this is causally identical to
# computing state on the inner-train slice alone, for every bar inside it).
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    from experiments.r109_shared import step0_gate  # local import, matches r109_shared's own idiom
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
          f"BASELINE_WINDOW_DAYS={BASELINE_WINDOW_DAYS}, MIN_REF_DAYS={MIN_REF_DAYS})")
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
            label = f"knn_novelty_brake_t{key[0]:g}_m{key[1]:g}"
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
    label = f"knn_novelty_brake_t{thresh:g}_m{maxd:g}"

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

    hr("R-109 NOVEL: KnnNoveltyBrakeKellyV4 -- k-nearest-neighbour "
       "DISTRIBUTIONAL-NOVELTY discount on v4's own frac*scale")
    print("mechanism: 5-feature OHLCV-only daily market-state panel (log_vol, anchor_disp,")
    print("kurtosis, volume_z, skew) -> mean Euclidean distance (per-feature standardized) to")
    print("the k nearest neighbours in a trailing 730-day reference SET, refit every")
    print("refit_every days (walk-forward, strictly causal) -> causal rolling percentile-rank")
    print("state in [0,1] -> linear discount on v4's UNCHANGED frac*scale product. Nonparametric")
    print("/ density-based (Ramaswamy et al. 2000 / Breunig et al. 2000's kNN-distance family),")
    print("the algorithm-class sibling of the conservative branch's single fitted Gaussian")
    print("(Mahalanobis) construction. Full grounding in r109_shared.py's own module docstring.")
    print(f"\nk={K}, refit_every={REFIT_EVERY}d  (both = rolling_knn_distance's own defaults, "
          f"verified programmatically above; no sweep performed -- see banner reasoning)")
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
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the kNN novelty")
        print("discount is either a near-total no-op, a near-exact rescale of v4's own path, a")
        print("relabelled volatility rescale, or degenerate everywhere on the pre-registered grid.")
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
