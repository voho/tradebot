#!/usr/bin/env python
"""R-114 NOVEL branch: ``StratifiedHazardKellyV4`` -- a covariate-STRATIFIED
life-table hazard discount on ``kelly_regime_v4``'s own ``frac * scale``
product, the Diebold, Lee & Weinbach (1994) "time-varying transition
probability" sibling of the conservative branch's duration-ONLY life table,
built entirely from ``experiments/r114_shared.py``'s (operator-authored,
READ-ONLY) shared infrastructure.

MECHANISM, in one sentence: estimate a 2D (duration bucket x realized-vol
tertile) life-table hazard of ``kelly_regime_v4``'s own vote-regime ending
soon, walk-forward and causally, each cell empirical-Bayes shrunk toward the
duration-only marginal table, convert today's hazard into a causal rolling
percentile-rank state in [0,1], and multiplicatively discount v4's unchanged
exposure whenever that state is high. No new statistic is invented in this
file -- every function used below (``regime_state_daily``,
``regime_duration_daily``, ``covariate_tertile_daily``,
``rolling_stratified_hazard``, ``hazard_to_state``, ``align_daily_to_bars``,
``apply_discount``) is already implemented, causally verified by its own
self-test, and READ-ONLY in ``experiments/r114_shared.py``. This file's job
is composition, not invention. The complete literature grounding (Diebold &
Rudebusch 1990's duration-only life table; Maheu & McCurdy 2000's duration-
dependent regime-switching model, motivating the Fibonacci-like duration
grid; Diebold, Lee & Weinbach 1994's covariate-dependent transition-
probability extension, this branch's own direct motivation; Cutler & Ederer
1958's actuarial life-table estimator itself) and the complete non-
duplication argument against every prior ERR-axis round (R-28/retracted,
R-87, R-104, R-105, R-106) and against R-109/R-112/R-113's distributional-
novelty constructions all live in ``r114_shared.py``'s own module docstring,
written by the operator before either branch was dispatched, and are NOT
re-derived here -- read that file in full first. This file imports ONLY from
``experiments.r114_shared`` (itself chaining r109_shared -> r106_shared ->
r105_shared -> ... -> r102_shared's unchanged control machinery) and
``experiments.r105_shared``'s re-exported gate functions, never edits any
shared/frozen file, never imports from or waits on the conservative branch's
own file (``experiments/r114_conservative_lifetable_hazard.py``, which may
not exist yet or may be mid-write by a concurrent process), and never reads
a bar at or after ``r114_shared.OOS_START`` (2023-01-01).

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this):

  1. ``state = r114_shared.regime_state_daily(df)`` -- v4's own vote
     (``v4_vote_frac``, unmodified) binarized (>=0.5 bullish-leaning
     majority) into a daily bull/bear regime series.
  2. ``duration = r114_shared.regime_duration_daily(state)`` -- days since
     the regime last flipped (causal running count).
  3. ``cov = r114_shared.covariate_tertile_daily(df)`` -- a causal [0,1,2]
     tertile of v4's own realized-vol input (log-scaled), ranked against its
     own trailing 730-day history.
  4. ``hazard = r114_shared.rolling_stratified_hazard(state, duration,
     cov)`` -- the 2D (duration bucket x vol tertile) life-table hazard,
     each cell empirical-Bayes shrunk toward the marginal (duration-only)
     life table with prior strength ``r114_shared.MIN_SPELLS_CELL_PRIOR``,
     refit walk-forward every ``r114_shared.REFIT_EVERY_DAYS`` days using
     only spells completed strictly before the refit day (already verified
     causal by ``r114_shared.py``'s own self-test).
  5. ``state01 = r114_shared.hazard_to_state(hazard)`` -- a causal rolling
     percentile-rank of the raw hazard into [0, 1].
  6. ``bar_state = r114_shared.align_daily_to_bars(state01,
     df).fillna(0.0)`` -- forward-filled onto the 5-minute bar index.
  7. ``target = r114_shared.apply_discount(df, bar_state, thresh,
     max_discount) = r114_shared.v4_target(df) * (1 - discount)``, where
     ``discount`` ramps linearly from 0 at ``bar_state <= thresh`` to
     ``max_discount`` at ``bar_state == 1``. ``v4``'s own vote and scale are
     completely UNTOUCHED; only the ``frac * scale`` PRODUCT is discounted,
     exactly the same slot-in architecture every ERR-axis round since R-87
     uses.

PRE-REGISTERED NUISANCE-PARAMETER CHOICES -- none are chosen or swept in
this file. ``REFIT_EVERY_DAYS``, ``LAPLACE``, ``MIN_SPELLS``,
``MIN_SPELLS_CELL_PRIOR``, ``N_VOL_TERTILES``, and ``DURATION_BUCKET_EDGES``
are all frozen at their own defaults inside ``r114_shared.py``, fixed by the
operator before either branch was dispatched, and this file never touches
them or introduces a new nuisance parameter of its own. The ONLY grid swept
anywhere in this file is ``r114_shared.STEP0_THRESH_GRID x
r114_shared.STEP0_MAXD_GRID`` (3 x 2 = 6 cells), the identical Step-0 grid
every ERR-axis round since R-104 sweeps, with ``r114_shared.SELECTION_ORDER``
for primary-cell selection -- same convention as
``r109_novel_knn_novelty_brake.py``'s own ``step0_grid``/``select_primary``.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read):
sweep the 6-cell grid on BTC's ``INNER_TRAIN_START..INNER_TRAIN_END`` via
``r114_shared.step0_gate(df.loc[INNER_TRAIN_START:INNER_TRAIN_END], state,
thresh, max_discount)``, where ``state`` (here: ``bar_state``, step 6 above)
is computed over BTC's FULL non-holdout frame first, then only the
inner-train slice is scored (causally identical to computing it on
inner-train alone, per r109_shared's own established convention; do not
recompute state separately per slice). A cell QUALIFIES iff
``step0_gate(...)['passed']`` is True (bind_frac > 1%, R^2 vs v4's own
target < 0.98, R^2 vs v4's own realized-vol input < 0.90, state CoV >= 5%).
PRIMARY cell: the first ``(thresh, max_discount)`` in
``r114_shared.SELECTION_ORDER`` that qualifies. If NONE qualify: STOP,
report Step-0 FAIL as a complete NEGATIVE result -- no inner-validation or
ETH bar is read past that point.

Also printed once, near the top of the Step-0 output (this branch's own
disclosed small-sample diagnostic, not a gate): the number of completed
vote-flip spells available by the end of inner-train
(``r114_shared.count_inner_train_spells``), and the Pearson correlation
between this branch's own ``state01`` and the conservative branch's own
state, the latter reproduced read-only by calling
``r114_shared.rolling_lifetable_hazard`` + ``hazard_to_state`` directly on
the SAME ``state``/``duration`` this branch already computed (never
importing the conservative branch's own file) -- a disclosed non-
duplication check between this round's two closest methodological
relatives, flagged prominently if it exceeds 0.98 (cf. R-111's novel branch,
which scored rho=0.971 against its own comparison point and was flagged as
a near-duplicate concern).

PROMOTION BAR (only if Step-0 passes; identical shape to every SIZE/ERR-axis
round since R-89, via ``r114_shared``'s re-exported gate functions):
  B1 (gating): ``b1_from_inner_val`` on the primary cell's inner_val rows,
     both markets (spot, futures_5x) -- dSharpe > +0.2 OR bootstrap excludes
     zero favourably.
  B2 (diagnostic ONLY, never gates): ``b2_diagnostic`` -- drawdown
     improvement, counted only where risk-matched.
  B3 (plateau, gating): the FULL 6-cell Step-0 ``(thresh, max_discount)``
     grid's own inner-validation numbers (both markets each, primary cell's
     2 rows reused directly from its own ``compare()`` rather than
     recomputed) -- PASS requires a directionally consistent (same-sign)
     majority across the resulting 12 cells, not an isolated spike.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test per docs/ROUTINE.md step 2): ``b4_eth_falsification``
     on ``compare(..., eth=load_eth())`` -- does the SAME-SIGN effect
     replicate on ETH? Require FULL pass (both markets same-signed as BTC
     inner_val). Named failure mode (``r114_shared.py``'s own docstring
     point 4): the life table is fit predominantly on BTC's single
     2017-2020 supercycle and may not generalise to ETH's shorter,
     differently-shaped history at all.
  B5 (cost robustness, gating): ``b5_fee_tier`` at the 0.40% taker tier,
     primary cell, BTC inner_val, both markets -- no sign reversal.
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau) AND B4 (full) AND B5 all hold (B2 is diagnostic-only).
Default: NEGATIVE. No threshold or decision rule is changed after seeing any
number -- anything contradicted by what actually happened is stated in the
results section below, never edited back into this banner.

CAUSAL SAFETY: ``r114_shared.causal_truncation_probe_series`` applied to
this file's own ``build_target`` (the FULL composed pipeline: state ->
duration -> covariate tertile -> stratified hazard -> percentile-rank state
-> bar alignment -> discount -> ``v4_target * (1 - discount)``), run on
BTC's full non-holdout frame at the pre-registered primary cell
(``thresh=r114_shared.SELECTION_ORDER[0][0]``,
``max_discount=r114_shared.SELECTION_ORDER[0][1]``), BEFORE the Step-0 grid
is scored and well before any inner-validation/ETH performance number is
computed. The underlying hazard functions are already verified causal by
``r114_shared.py``'s own self-test, but the COMPOSED pipeline -- including
``align_daily_to_bars`` and ``apply_discount`` -- needs its own check. The
probe is expected to PASS; if it does not, that is a real bug in THIS FILE
to find and report, not something to route around.

WHAT WOULD MAKE THIS FAIL, named now (``r114_shared.py``'s own "what would
make this fail" points 1-5, restated here as this branch's specific
exposure to each): (1) thin completed-spell sample size, worse here than in
the conservative branch because the 3-way vol-tertile stratification
splits an already-thin inner-train spell count into even sparser cells --
watched for directly by disclosing the completed-spell count above and by
``MIN_SPELLS`` (Step-0 fails by construction, not by tuning, if inner-train
produces fewer than 15 completed spells at all). (2) Duration dependence in
a fitted regime-switching MODEL (Maheu & McCurdy's own object) is not
guaranteed to transfer to a HEURISTIC latched vote (this project's own
object) -- the two need not share the same hazard shape. (3) Even if the
hazard is real and estimable, v4's vote may already price in "old" regimes
by the time they are old enough to register -- reproducing the
R-87/R-104/R-105/R-106/R-109 "real but inert" pattern (Step-0 passes, B1
does not). (4) BTC-specific calibration failing to generalise to ETH's
shorter, differently-shaped history (B4). (5) THIS BRANCH'S OWN SPECIFIC
RISK: the empirical-Bayes shrinkage toward the marginal (duration-only)
table could be strong enough that this branch collapses to a near-exact
copy of the conservative branch's own result -- measured directly by the
Pearson correlation check described above, disclosed honestly whatever its
value, never used to alter this file's construction -- or conversely too
weak, letting sparse cells drive noisy, spurious-looking hazard spikes.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets, 2 of
the 12 reused directly from the primary ``compare()``'s own inner_val rows,
10 freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total. IF
Step-0 finds no qualifying cell, this file stops after the 6 Step-0 cells
(6 total). No hazard nuisance parameter is swept (see above), so it adds 0
configurations to either count.

USAGE
-----
    python experiments/r114_novel_stratified_hazard.py
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

from experiments.r114_shared import (  # noqa: E402
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    DURATION_BUCKET_EDGES,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    LAPLACE,
    MIN_REF_DAYS,
    MIN_SPELLS,
    MIN_SPELLS_CELL_PRIOR,
    N_DURATION_BUCKETS,
    N_VOL_TERTILES,
    OOS_START,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    REFIT_EVERY_DAYS,
    SELECTION_ORDER,
    SPOT,
    STEP0_MAXD_GRID,
    STEP0_THRESH_GRID,
    align_daily_to_bars,
    apply_discount,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    causal_truncation_probe_series,
    compare,
    count_inner_train_spells,
    covariate_tertile_daily,
    hazard_to_state,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    print_plateau_table,
    print_rows,
    regime_duration_daily,
    regime_state_daily,
    rolling_lifetable_hazard,
    rolling_stratified_hazard,
    step0_gate,
)

PRIMARY_THRESH, PRIMARY_MAXD = SELECTION_ORDER[0]


# ================================================================== (1)
# The mechanism itself: regime state -> duration -> covariate tertile ->
# stratified hazard -> percentile-rank state -> bar alignment -> discount on
# v4's own UNCHANGED frac*scale. Composition only -- every function called
# below is r114_shared's own, unmodified.
# ==================================================================

def compute_daily_pieces(df: pd.DataFrame):
    """The five daily-frequency mechanism steps (1-5 in the banner above),
    returned individually so both `compute_full_state` and the Step-0
    diagnostics (completed-spell count, non-duplication correlation check)
    can reuse them without duplicating logic."""
    state = regime_state_daily(df)
    duration = regime_duration_daily(state)
    cov = covariate_tertile_daily(df)
    hazard = rolling_stratified_hazard(state, duration, cov)
    state01 = hazard_to_state(hazard)
    return state, duration, cov, hazard, state01


def compute_full_state(df: pd.DataFrame) -> pd.Series:
    """Steps 1-6: daily mechanism -> forward-filled bar-level state in
    [0, 1], over whatever frame `df` is (the caller decides how much
    history it contains -- this function makes no reference to any fixed
    calendar date)."""
    _, _, _, _, state01 = compute_daily_pieces(df)
    return align_daily_to_bars(state01, df).fillna(0.0)


def build_target(df: pd.DataFrame, thresh: float = PRIMARY_THRESH,
                  max_discount: float = PRIMARY_MAXD) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount),
    where discount is driven by the stratified-hazard state built from `df`
    alone. Self-contained (a pure function of `df`), so it is directly
    usable as a `TargetStrategy` candidate on any window (inner_train,
    inner_val, eth_replication, or a truncated probe frame)."""
    bar_state = compute_full_state(df)
    return apply_discount(df, bar_state, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"stratified_hazard_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r114_shared.STEP0_THRESH_GRID x r114_shared.STEP0_MAXD_GRID,
# scored via r114_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame (see banner above for why this is causally
# identical to computing state on the inner-train slice alone).
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
          f"{n_bars:,} bars, state built from REFIT_EVERY_DAYS={REFIT_EVERY_DAYS}d, "
          f"LAPLACE={LAPLACE}, MIN_SPELLS={MIN_SPELLS}, "
          f"MIN_SPELLS_CELL_PRIOR={MIN_SPELLS_CELL_PRIOR}, N_VOL_TERTILES={N_VOL_TERTILES}, "
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
            label = f"stratified_hazard_t{key[0]:g}_m{key[1]:g}"
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
    label = f"stratified_hazard_t{thresh:g}_m{maxd:g}"

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

    hr("R-114 NOVEL: StratifiedHazardKellyV4 -- covariate-stratified life-table hazard "
       "discount on v4's own frac*scale")
    print("mechanism: v4's own vote regime state -> duration since last flip -> a 2D")
    print("(duration bucket x realized-vol tertile) life-table hazard (Diebold, Lee & Weinbach")
    print("1994's time-varying-transition-probability extension of Diebold & Rudebusch 1990's")
    print("duration-only life table, Cutler & Ederer 1958's actuarial estimator), each cell")
    print("empirical-Bayes shrunk toward the marginal (duration-only) life table, refit")
    print("walk-forward every REFIT_EVERY_DAYS days using only completed spells -> causal")
    print("rolling percentile-rank state in [0,1] -> forward-filled onto bars -> linear discount")
    print("on v4's UNCHANGED frac*scale product. Full grounding in r114_shared.py's own module")
    print("docstring.")
    print(f"\nPre-registered, frozen constants (not swept, defined in r114_shared.py): "
          f"REFIT_EVERY_DAYS={REFIT_EVERY_DAYS}, LAPLACE={LAPLACE}, MIN_SPELLS={MIN_SPELLS}, "
          f"MIN_SPELLS_CELL_PRIOR={MIN_SPELLS_CELL_PRIOR}, N_VOL_TERTILES={N_VOL_TERTILES},")
    print(f"DURATION_BUCKET_EDGES={DURATION_BUCKET_EDGES} ({N_DURATION_BUCKETS} buckets)")
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

    # Daily mechanism pieces, computed once over BTC's FULL non-holdout
    # frame, reused for the disclosed small-sample and non-duplication
    # diagnostics below.
    state, duration, cov, hazard, state01 = compute_daily_pieces(btc)

    n_days_inner_train = int(np.sum(state.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_spells_inner_train = count_inner_train_spells(state, duration, n_days_inner_train)
    print(f"\nCompleted inner-train spells available (spells fully ended strictly before day "
          f"{n_days_inner_train}, i.e. through {INNER_TRAIN_END}): {n_spells_inner_train}")
    print(f"DISCLOSED SMALL-SAMPLE CAVEAT: this branch stratifies that spell count across "
          f"{N_VOL_TERTILES} vol tertiles x {N_DURATION_BUCKETS} duration buckets -- sparser per")
    print(f"cell than the conservative branch's marginal (duration-only) table by construction; "
          f"heavy empirical-Bayes shrinkage (prior_n={MIN_SPELLS_CELL_PRIOR} pseudo-spells) toward")
    print("that marginal table is built into rolling_stratified_hazard specifically to guard "
          "against a noisy, non-shrunk cell estimate looking spuriously informative.")

    hr("NON-DUPLICATION CHECK vs conservative branch's own state (read-only use of "
       "r114_shared.rolling_lifetable_hazard on this branch's own state/duration; the "
       "conservative branch's own file is never imported or waited on)")
    marginal_hazard = rolling_lifetable_hazard(state, duration)
    conservative_state01 = hazard_to_state(marginal_hazard)
    common_idx = state01.dropna().index.intersection(conservative_state01.dropna().index)
    a_common = state01.reindex(common_idx)
    b_common = conservative_state01.reindex(common_idx)
    corr = float(a_common.corr(b_common)) if len(common_idx) > 1 else float("nan")
    print(f"n common non-null daily observations: {len(common_idx)}")
    print(f"Pearson correlation, this branch's state01 vs conservative branch's own "
          f"(duration-only) state01: {corr:.4f}")
    if np.isfinite(corr) and corr > 0.98:
        print("*** NEAR-DUPLICATE RISK: correlation exceeds 0.98 (cf. R-111's novel branch, "
              "which scored rho=0.971 against its own comparison point and was flagged as a "
              "near-duplicate concern in this ledger) -- disclosed as measured, construction "
              "unchanged. ***")
    else:
        print("(<= 0.98: not flagged as a near-duplicate by this round's own disclosed "
              "threshold; disclosed as measured either way.)")

    step0_rows, _ = step0_grid(btc)
    n_bars_inner_train = int(np.sum((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                     (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the stratified")
        print("hazard discount is either a near-total no-op, a near-exact rescale of v4's own")
        print("path, a relabelled volatility rescale, or degenerate everywhere on the")
        print("pre-registered grid. Per this file's own pre-registration, this Step-0 table")
        print("(plus the causal-safety probe, spell count, and non-duplication check above) is")
        print("the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0. No")
        print("promotion-bar code runs, and no inner-validation Sharpe/PnL number or ETH bar is")
        print("ever read.")

        hr("VERDICT")
        print("Step-0 (6-cell thresh x max_discount grid): FAIL (no cell qualifies)")
        print(f"causal truncation probe: {probe_ok}")
        print(f"completed inner-train spells: {n_spells_inner_train}")
        print(f"correlation vs conservative branch's own state: {corr:.4f}")
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
                    probe_ok=probe_ok, n_spells_inner_train=n_spells_inner_train,
                    corr_vs_conservative=corr, n_configs=n_configs, max_ts=max_ts,
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
    print(f"completed inner-train spells: {n_spells_inner_train}")
    print(f"correlation vs conservative branch's own state: {corr:.4f}")
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
          f"2 B5 fee-tier; no hazard nuisance parameter swept, adds 0)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, state=state, duration=duration,
                cov=cov, hazard=hazard, state01=state01, corr_vs_conservative=corr,
                n_spells_inner_train=n_spells_inner_train,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
