#!/usr/bin/env python
"""R-112 NOVEL branch: ``PooledRefKnnNoveltyBrakeKellyV4`` -- R-109 novel
branch's exact k-nearest-neighbour DISTRIBUTIONAL-NOVELTY discount on
``kelly_regime_v4``'s own ``frac * scale`` product, with EXACTLY ONE change
from R-109's own winning construction: the reference SET the kNN distance is
measured against is no longer the target instrument's own trailing 730 days
alone, but that window POOLED with the contemporaneous trailing windows of
R-63's own six-instrument panel (``UNIVERSE_6``: BCH, LTC, ETC, DASH, LINK,
XTZ), each standardized against its own local mean/std before pooling
(CORAL-style domain alignment, Sun & Saenko 2016). This file's entire
non-duplication argument, literature grounding, and the round's own named
failure modes live in ``experiments/r112_shared.py``'s module docstring --
read that file in full first; nothing is re-derived here.

MECHANISM, in one sentence: identical to R-109 novel
(``experiments/r109_novel_knn_novelty_brake.py``) -- same 5-feature OHLCV-
only daily market-state panel (``r109_shared.NOVEL_FEATURE_BUILDERS``:
log_vol, anchor_disp, kurtosis, volume_z, skew), same kNN distance shape
(k=10, refit_every=30), same causal percentile-rank state, same linear
discount on v4's unchanged ``frac * scale`` -- except the distance is now
``r112_shared.rolling_knn_distance_pooled`` instead of ``r109_shared.
rolling_knn_distance``: the reference set at every refit is the target's own
trailing 730 days UNION the six ``UNIVERSE_6`` instruments' own contemporaneous
trailing 730 days, each z-scored against its own local mean/std before being
concatenated into one pooled reference. Neither BTC nor ETH is ever a pool
member (asserted inside ``r112_shared.load_pool_daily_panels``): BTC is the
primary target and ETH is reserved solely for the pre-registered B4
falsification test, exactly as in R-109.

WHY THIS IS THE FOLLOW-ON R-109 NAMED: R-109's own closing verdict (docs/
LEDGER.md, R-109 section) diagnosed its novel branch's B4 failure (ETH spot
sign inversion, futures partial replication) as "the novelty statistic's own
reference distribution, built predominantly from BTC's 2017-2020 supercycle,
does not transfer to ETH's shorter, differently-shaped history" and named,
verbatim, "an explicit multi-asset reference pool" as one of two disclosed
follow-ons that "might close the novel branch's B4 gap without needing a new
mechanism." This file is exactly that follow-on, and nothing else -- the
5-feature panel, the kNN metric shape, the discount architecture, the Step-0
grid, and the full B1-B5 promotion bar are all held byte-identical to R-109's
own novel branch; only ``rolling_knn_distance`` -> ``rolling_knn_distance_
pooled`` changes.

``pool_dailies`` (the six ``UNIVERSE_6`` instruments' own daily feature
panels, built via ``r112_shared.load_pool_daily_panels()``) is built EXACTLY
ONCE at the top of ``main()`` and threaded through every Step-0/B3/B5 cell as
a fixed argument -- never rebuilt per config, and never a function of the
target ``df`` argument being probed/sliced. This is what keeps the causal
truncation probe meaningful as a check on the TARGET argument alone: the pool
is a closed-over constant, exactly the pattern ``r112_shared.py``'s own
self-test already exercises for ``rolling_knn_distance_pooled``.

``k`` AND ``refit_every`` -- held at R-109's own pre-registered defaults,
``k=10`` and ``refit_every=30`` (verified programmatically below against the
live ``rolling_knn_distance_pooled`` signature). No sweep is performed; the
only Step-0 degrees of freedom are ``(thresh, max_discount)``, identical grid
to R-109 (``r109_shared.STEP0_THRESH_GRID x STEP0_MAXD_GRID``, 3 x 2 = 6
cells, same ``SELECTION_ORDER``).

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read):
identical to R-109 novel's own rule, restated for the pooled statistic: sweep
``STEP0_THRESH_GRID x STEP0_MAXD_GRID`` (6 cells) on BTC's ``INNER_TRAIN_
START..INNER_TRAIN_END`` via ``step0_gate(df.loc[INNER_TRAIN_START:INNER_
TRAIN_END], state, thresh, max_discount)``, where ``state`` is computed over
BTC's FULL non-holdout frame with the pooled kNN distance (causality
guarantees this is identical, for every bar inside inner-train, to computing
the statistic on the inner-train slice alone). A cell QUALIFIES iff
``step0_gate(...)['passed']`` is True. PRIMARY cell: the first ``(thresh,
max_discount)`` in ``SELECTION_ORDER`` that qualifies. If NONE qualify: STOP,
report Step-0 FAIL as a complete NEGATIVE result.

PROMOTION BAR (only if Step-0 passes; identical shape and gate code to
R-109/R-105):
  B1 (gating): ``b1_from_inner_val`` on the primary cell's inner_val rows,
     both markets (spot, futures_5x) -- dSharpe > +0.2 OR bootstrap excludes
     zero favourably.
  B2 (diagnostic ONLY, never gates): ``b2_diagnostic`` -- drawdown
     improvement, counted only where risk-matched.
  B3 (plateau, gating): the FULL 6-cell Step-0 ``(thresh, max_discount)``
     grid's own inner-validation numbers (both markets each, primary cell's
     2 rows reused directly from its own ``compare()``) -- PASS requires a
     directionally consistent (same-sign) majority across the resulting 12
     cells.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test): ``b4_eth_falsification`` on ``compare(..., eth=
     load_eth())`` -- does the SAME-SIGN effect replicate on ETH? Require
     FULL pass (both markets same-signed as BTC inner_val). This is the
     exact clause R-109 novel failed (ETH spot sign inversion) and the one
     this round's ENTIRE construction change targets.
  B5 (cost robustness, gating): ``b5_fee_tier`` at the 0.40% taker tier,
     primary cell, BTC inner_val, both markets -- no sign reversal.
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau) AND B4 (full) AND B5 all hold (B2 is diagnostic-only).
Default: NEGATIVE. No threshold or decision rule is changed after seeing any
number.

CAUSAL SAFETY: ``causal_truncation_probe_series`` applied to this file's own
``build_target`` (features -> pooled kNN distance -> percentile-rank state ->
discount -> ``v4_target * (1 - discount)``), with ``pool_dailies`` built once
and closed over BEFORE the probe runs, so the probe is a fair causality check
on the target argument alone (see ``r112_shared.py``'s own module docstring
and self-test for exactly this pattern). Run on BTC's full non-holdout frame,
BEFORE the Step-0 grid is scored and well before any inner-validation/ETH
performance number is computed. If the probe fails, that is a bug in this
file to find and fix, not something to route around.

WHAT WOULD MAKE THIS FAIL, named now (all are ``r112_shared.py``'s own
pre-registered concerns, restated as this branch's specific exposure): (1)
the B4 gap may not be a reference-set artifact at all but a deeper property
of ``kelly_regime_v4``'s own vote interacting with ETH's shorter history
(R-57's general finding) -- if so, pooling does not close B4 either. (2) The
six-instrument pool may dilute the statistic's discriminative power rather
than its style-specificity -- raising the effective reference variance
enough that BTC's own genuinely novel days no longer register as distant (a
Step-0 ``bind_frac``/``state_cv`` failure). (3) Even if B4 closes, the
"true novel days" identified may differ enough from R-109's own single-asset
statistic that any B1 gain is not attributable to the same mechanism R-109
validated on BTC -- an interpretation risk, not a kill switch; the promotion
bar governs regardless.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets, 2 of the
12 reused directly from the primary ``compare()``'s own inner_val rows, 10
freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total. IF Step-0
finds no qualifying cell, this file stops after the 6 Step-0 cells (6
total). No ``k``/``refit_every`` sweep is performed, so it adds 0
configurations to either count.

USAGE
-----
    python experiments/r112_novel_pooled_reference_knn.py
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
    UNIVERSE_6,
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
    load_pool_daily_panels,
    print_plateau_table,
    print_rows,
    rolling_knn_distance_pooled,
)

# ---------------------------------------------------------- pre-registered
K = 10             # rolling_knn_distance's own default, held fixed -- verified below
REFIT_EVERY = 30   # rolling_knn_distance's own default, held fixed -- verified below

_sig = inspect.signature(rolling_knn_distance_pooled).parameters
assert _sig["k"].default == K, ("K does not match rolling_knn_distance_pooled's "
                                 f"own default ({_sig['k'].default}) -- pre-registration text is stale")
assert _sig["refit_every"].default == REFIT_EVERY, (
    "REFIT_EVERY does not match rolling_knn_distance_pooled's own default "
    f"({_sig['refit_every'].default}) -- pre-registration text is stale")


# ================================================================== (1)
# The mechanism itself: 5-feature panel -> POOLED kNN distance (target +
# six UNIVERSE_6 instruments, each CORAL-standardized) -> percentile-rank
# state -> discount on v4's own UNCHANGED frac*scale. `pool_dailies` is
# always a fixed argument/closure, never rebuilt here and never a function
# of `df`.
# ==================================================================

def compute_full_state(df: pd.DataFrame, pool_dailies: dict[str, pd.DataFrame],
                        k: int = K, refit_every: int = REFIT_EVERY) -> pd.Series:
    """features -> pooled kNN distance -> causal percentile-rank state, over
    whatever frame `df` is (the caller decides how much history it
    contains). `pool_dailies` is a fixed reference pool, independent of
    `df` -- this is what keeps the function causal in `df` alone."""
    daily = build_daily_features(df, NOVEL_FEATURE_BUILDERS)
    dist = rolling_knn_distance_pooled(daily, pool_dailies, k=k, refit_every=refit_every)
    return causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, pool_dailies: dict[str, pd.DataFrame],
                  thresh: float = PRIMARY_THRESH, max_discount: float = PRIMARY_MAXD,
                  k: int = K, refit_every: int = REFIT_EVERY) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount), where
    discount is driven by the pooled-kNN novelty state built from `df` and
    the fixed `pool_dailies` closure. Directly usable as a `TargetStrategy`
    candidate on any window (inner_train, inner_val, eth_replication, or a
    truncated probe frame) once `pool_dailies` is bound."""
    state = compute_full_state(df, pool_dailies, k=k, refit_every=refit_every)
    return apply_discount(df, state, thresh, max_discount)


def make_build_target(pool_dailies: dict[str, pd.DataFrame], thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, pool_dailies, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"knn_pooled_novelty_brake_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: STEP0_THRESH_GRID x STEP0_MAXD_GRID, scored via step0_gate on
# BTC inner-train, state computed over the FULL non-holdout BTC frame with
# the pooled kNN distance and the fixed `pool_dailies` closure.
# ==================================================================

def step0_grid(btc: pd.DataFrame, pool_dailies: dict[str, pd.DataFrame]) -> tuple[list[dict], pd.Series]:
    from experiments.r112_shared import step0_gate  # local import, matches r109_shared's own idiom
    state = compute_full_state(btc, pool_dailies)
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
          f"POOLED reference: target + {len(UNIVERSE_6)} UNIVERSE_6 instruments, CORAL-standardized)")
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
                      inner_val_primary: list[dict], btc: pd.DataFrame,
                      pool_dailies: dict[str, pd.DataFrame]) -> tuple[dict, bool]:
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
            bf = make_build_target(pool_dailies, *key)
            label = f"knn_pooled_novelty_brake_t{key[0]:g}_m{key[1]:g}"
            plateau_rows[key] = inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# above), B4 (gating falsification), B5 (gating fee tier).
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                       btc: pd.DataFrame, eth: pd.DataFrame,
                       pool_dailies: dict[str, pd.DataFrame]) -> dict:
    thresh, maxd = primary_key
    build_primary = make_build_target(pool_dailies, thresh, maxd)
    label = f"knn_pooled_novelty_brake_t{thresh:g}_m{maxd:g}"

    hr(f"PROMOTION BAR -- PRIMARY CELL thresh={thresh:g}, max_discount={maxd:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                    markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3_full_grid(step0_rows, primary_key, inner_val_primary, btc, pool_dailies)
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

    hr("R-112 NOVEL: PooledRefKnnNoveltyBrakeKellyV4 -- R-109 novel's exact "
       "kNN novelty brake, reference set POOLED across UNIVERSE_6 (CORAL-standardized)")
    print("mechanism: identical to r109_novel_knn_novelty_brake.py's 5-feature OHLCV-only daily")
    print("market-state panel (log_vol, anchor_disp, kurtosis, volume_z, skew) -> mean Euclidean")
    print("distance (per-feature standardized) to the k nearest neighbours -- except the reference")
    print("SET is now the target's own trailing 730 days UNION the six UNIVERSE_6 instruments' own")
    print("contemporaneous trailing 730 days, each CORAL-standardized (Sun & Saenko 2016) against")
    print("its own local mean/std before pooling -> causal rolling percentile-rank state in [0,1]")
    print("-> linear discount on v4's UNCHANGED frac*scale product. Full grounding, non-duplication")
    print("argument, and named failure modes in r112_shared.py's own module docstring.")
    print(f"\nk={K}, refit_every={REFIT_EVERY}d  (both = rolling_knn_distance_pooled's own defaults, "
          f"verified programmatically above; no sweep performed -- see banner reasoning)")
    print(f"STEP0_THRESH_GRID={STEP0_THRESH_GRID}  STEP0_MAXD_GRID={STEP0_MAXD_GRID}  "
          f"({len(STEP0_THRESH_GRID) * len(STEP0_MAXD_GRID)} cells)")
    print(f"SELECTION_ORDER={SELECTION_ORDER}")
    print(f"POOL: UNIVERSE_6={UNIVERSE_6}")

    hr("BUILDING SIX-INSTRUMENT POOL DAILY FEATURE PANELS (built ONCE, reused as a fixed "
       "closure for every Step-0/B3/B5 cell below -- never rebuilt per config)")
    pool_dailies = load_pool_daily_panels()
    for name, panel in pool_dailies.items():
        assert_no_holdout(panel, f"main(): pool instrument {name}")
        max_ts_seen.append(panel.index.max())
        print(f"  {name:>6s}: {len(panel):,} daily rows, {panel.index[0].date()} -> "
              f"{panel.index[-1].date()}  (< {OOS_START})")
    assert "BTC" not in pool_dailies and "ETH" not in pool_dailies

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ================================================== CAUSAL SAFETY FIRST
    hr("CAUSAL TRUNCATION PROBE (composed build_target at the pre-registered primary "
       "(thresh, max_discount), real BTC data, fixed pool_dailies closure, run BEFORE "
       "Step-0 or any inner-val/ETH number)")
    probe_fn = make_build_target(pool_dailies, PRIMARY_THRESH, PRIMARY_MAXD)
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
    step0_rows, state = step0_grid(btc, pool_dailies)
    n_bars_inner_train = int(np.sum((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                     (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the pooled-")
        print("reference kNN novelty discount is either a near-total no-op, a near-exact rescale")
        print("of v4's own path, a relabelled volatility rescale, or degenerate everywhere on the")
        print("pre-registered grid. Per this file's own pre-registration, this Step-0 table (plus")
        print("the causal-safety probe above) is the branch's ENTIRE product, reported NEGATIVE /")
        print("stopped-at-Step-0. No promotion-bar code runs, and no inner-validation Sharpe/PnL")
        print("number or ETH bar is ever read.")

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
        print(f"max timestamp read anywhere in this branch (BTC, ETH, all six pool instruments): "
              f"{max_ts}  (< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
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

    bar = run_promotion_bar(primary_key, step0_rows, btc, eth, pool_dailies)

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
    print(f"max timestamp read anywhere in this branch (BTC, ETH, all six pool instruments): "
          f"{max_ts}  (< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, pool_dailies=pool_dailies, step0_rows=step0_rows, state=state,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
