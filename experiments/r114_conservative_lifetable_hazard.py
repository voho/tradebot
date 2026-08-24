#!/usr/bin/env python
"""R-114 CONSERVATIVE branch: ``LifetableHazardKellyV4`` -- a marginal
(duration-only) actuarial life-table hazard discount on ``kelly_regime_v4``'s
own ``frac * scale`` product, built entirely from
``experiments/r114_shared.py``'s (operator-authored, READ-ONLY) shared
infrastructure. This file assembles a pipeline; it invents no new statistic
-- every function it calls (``regime_state_daily``, ``regime_duration_daily``,
``rolling_lifetable_hazard``, ``hazard_to_state``) is defined, and its own
causality already verified by self-test, in ``r114_shared.py``.

MECHANISM, in one sentence: is ``kelly_regime_v4``'s own 3-anchor vote,
binarized into a daily bull/bear regime and latched until it flips,
CURRENTLY OLD relative to the empirical (Cutler & Ederer 1958 life-table)
distribution of how long regimes of that vote have lasted historically
before flipping -- i.e. is today's implied hazard of an imminent regime
change elevated relative to its own recent history -- and discount exposure
multiplicatively when it is. Full literature grounding (Diebold & Rudebusch
1990's nonparametric life-table duration-dependence test on business-cycle
phases; Maheu & McCurdy 2000's duration-dependent Markov-switching bull/bear
model; Cutler & Ederer 1958's own actuarial hazard estimator) and the
complete non-duplication argument against every prior ERR-axis round
(R-28/retracted, R-87, R-104, R-105, R-106, R-109, R-112, R-113) live in
``r114_shared.py``'s own module docstring, written by the operator before
either R-114 branch was dispatched, and are NOT re-derived here -- read that
file in full first. This file imports ONLY from ``experiments.r114_shared``
(itself chaining r109_shared -> r106_shared -> r105_shared -> ... ->
r102_shared's unchanged control machinery, plus r105_shared's B1-B5 gate
functions), never edits any shared file, never coordinates with or reads the
NOVEL branch's file, and never reads a bar at or after
``r114_shared.OOS_START`` (2023-01-01).

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this):

  1. ``state = r114_shared.regime_state_daily(df)`` -- v4's own
     ``v4_vote_frac`` binarized (>=0.5 bullish-leaning majority) and resampled
     to one observation per calendar day.
  2. ``duration = r114_shared.regime_duration_daily(state)`` -- day t's count
     of consecutive days the binarized state has held its current value
     (causal running forward count).
  3. ``hazard = r114_shared.rolling_lifetable_hazard(state, duration)`` --
     day t's marginal (duration-only) Cutler & Ederer (1958) actuarial life-
     table hazard for its own current duration bucket, refit walk-forward
     every ``REFIT_EVERY_DAYS`` days using only spells completed strictly
     before the refit day (``end_pos < t``) -- already verified causal by
     ``r114_shared.py``'s own self-test (an explicit truncation-vs-full-series
     comparison), trusted here rather than re-derived.
  4. ``state01 = r114_shared.hazard_to_state(hazard)`` -- a causal rolling
     percentile-rank of the raw hazard probability into [0, 1] against its
     own trailing 730-day history (is TODAY's hazard high relative to its
     OWN recent history, not relative to its raw magnitude -- matches every
     prior ERR-axis round's own convention).
  5. ``target = r114_shared.apply_discount(df, state01, thresh,
     max_discount) = r114_shared.v4_target(df) * (1 - discount)``, where
     ``discount`` ramps linearly from 0 at ``state01 <= thresh`` to
     ``max_discount`` at ``state01 == 1``. Internally, ``apply_discount``
     (via ``discount_series_for``) performs the daily-to-bar alignment
     (``align_daily_to_bars`` + ``fillna(0.0)``) itself -- this file passes
     it the DAILY ``state01`` series directly, exactly the same calling
     convention R-109's own ``build_target`` uses for its (also daily) kNN-
     novelty state, so the alignment is performed exactly once, not
     duplicated. ``v4``'s own vote and scale are completely UNTOUCHED; only
     the ``frac * scale`` PRODUCT is discounted.

PRE-REGISTERED NUISANCE PARAMETERS -- none introduced by this file. Every
constant the mechanism depends on (``REFIT_EVERY_DAYS=30``, ``LAPLACE=1.0``,
``MIN_SPELLS=15``, ``DURATION_BUCKET_EDGES`` -- the Fibonacci-like widening
grid) is frozen inside ``r114_shared.py`` at its own pre-registered default,
verified programmatically below against the live module (not hand-copied),
and is NOT swept anywhere in this file. The ONLY grid swept is
``r114_shared.STEP0_THRESH_GRID x r114_shared.STEP0_MAXD_GRID`` (3 x 2 = 6
cells, identical grid to every prior round since R-104), with
``r114_shared.SELECTION_ORDER`` for primary-cell selection.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read):
sweep the 6-cell grid on BTC's ``INNER_TRAIN_START..INNER_TRAIN_END`` via
``r114_shared.step0_gate(df.loc[INNER_TRAIN_START:INNER_TRAIN_END], state01,
thresh, max_discount)``, where ``state01`` is computed over BTC's FULL non-
holdout frame first (mirrors R-109's own convention: causality guarantees
this is identical, for every bar inside inner-train, to computing the
statistic on the inner-train slice alone). A cell QUALIFIES iff
``step0_gate(...)['passed']`` is True (bind_frac > 1%, R^2 vs v4's own target
< 0.98, R^2 vs v4's own realized-vol input < 0.90, state CoV >= 5%). PRIMARY
cell: the first ``(thresh, max_discount)`` in ``r114_shared.SELECTION_ORDER``
that qualifies. If NONE qualify: STOP, report Step-0 FAIL as a complete
NEGATIVE result -- no inner-validation or ETH bar is read past that point.

Also printed once, near the top of the Step-0 section, per this round's own
pre-registration: ``r114_shared.count_inner_train_spells(state, duration,
<inner-train length in days>)`` -- the number of completed vote-flip spells
available by the end of inner-train. This is a DISCLOSED small-sample-size
diagnostic, not a second gate (``MIN_SPELLS`` inside
``rolling_lifetable_hazard`` already enforces the kill switch structurally,
by producing NaN/no-discount hazard until enough spells exist) -- this
project's own standing concern about thin effective sample sizes (n~=3 at
the program level) applies here too, in the specific form of thin completed-
spell counts over a single 2017-2020 BTC supercycle, and is named explicitly
rather than left implicit.

PROMOTION BAR (only if Step-0 passes; identical shape to every SIZE/ERR-axis
round since R-89, via ``r114_shared``'s re-exported gate functions):
  B1 (gating): ``r114_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets (spot, futures_5x) -- dSharpe > +0.2 OR
     bootstrap excludes zero favourably.
  B2 (diagnostic ONLY, never gates): ``r114_shared.b2_diagnostic`` --
     drawdown improvement, counted only where risk-matched.
  B3 (plateau, gating): the FULL 6-cell Step-0 ``(thresh, max_discount)``
     grid's own inner-validation numbers (both markets each, primary cell's
     2 rows reused directly from its own ``compare()`` rather than
     recomputed) -- PASS requires a directionally consistent (same-sign)
     majority across the resulting 12 cells. Since no other nuisance
     parameter is swept, this 6-cell grid is the entirety of this branch's
     plateau evidence.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test per docs/ROUTINE.md step 2): ``r114_shared.
     b4_eth_falsification`` on ``r114_shared.compare(..., eth=
     r114_shared.load_eth())`` -- does the SAME-SIGN effect replicate on
     ETH? Requires FULL pass (both markets same-signed as BTC inner_val).
     Named failure mode (``r114_shared.py``'s own docstring point 4): the
     life table is fit predominantly on BTC's single 2017-2020 supercycle
     and may not generalise to ETH's shorter, differently-shaped history.
  B5 (cost robustness, gating): ``r114_shared.b5_fee_tier`` at the 0.40%
     taker tier, primary cell, BTC inner_val, both markets -- no sign
     reversal.
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau) AND B4 (full) AND B5 all hold (B2 is diagnostic-only).
Default: NEGATIVE. No threshold or decision rule is changed after seeing any
number -- anything contradicted by what actually happened is stated in the
results section below, never edited back into this banner.

CAUSAL SAFETY: ``r114_shared.causal_truncation_probe_series`` applied to
this file's own ``build_target`` (the FULL composed pipeline: vote ->
binarized state -> duration -> life-table hazard -> percentile-rank state ->
discount -> ``v4_target * (1 - discount)``), run on BTC's full non-holdout
frame, at the pre-registered primary cell
(``thresh=r114_shared.SELECTION_ORDER[0][0]``,
``max_discount=r114_shared.SELECTION_ORDER[0][1]``), BEFORE the Step-0 grid
is scored and well before any inner-validation/ETH performance number is
computed. The underlying hazard functions are already verified causal by
``r114_shared.py``'s own self-test, but the COMPOSED pipeline -- including
the daily-to-bar alignment and discount application performed inside
``apply_discount`` -- needs its own end-to-end check. Expected to PASS; if
it does not, that is a bug in THIS file to find and fix, not something to
route around (per docs/ROUTINE.md step 4's own precedence: a lookahead is a
bug report first).

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists
(``r114_shared.py``'s own "what would make this fail" list, points 1-4,
which are specific to this marginal/duration-only branch):
  (1) ``kelly_regime_v4``'s own vote, latched with a 1% hysteresis band on
      three slow (20/40/80-day) anchors, may simply not flip often enough
      over the inner-train window (2017-2020) to populate a life table with
      enough completed spells for the hazard estimate to be anything but
      noise -- this round's own disclosed version of the project's standing
      n~=3 concern, applied to regime-FLIP count. Watched for directly by
      ``MIN_SPELLS`` (if inner-train produces fewer completed spells than
      ``MIN_SPELLS``, the hazard is structurally NaN/no-op and Step-0 fails
      by construction, not by tuning) and disclosed quantitatively via
      ``count_inner_train_spells`` above.
  (2) Duration dependence in a fitted regime-switching MODEL (Maheu &
      McCurdy 2000's own object, fit with a likelihood) is not guaranteed to
      transfer to duration dependence in a HEURISTIC latched vote (this
      project's own object) -- the two need not share the same hazard
      shape, and this round's whole premise could simply be false on this
      data.
  (3) Even if the hazard is real and estimable, ``kelly_regime_v4``'s vote
      may already price in "old" regimes by the time they are old enough to
      register -- reproducing the R-87/R-104/R-105/R-106 "real but inert"
      pattern (Step-0 passes, B1 does not) by a seventh, structurally
      different estimator.
  (4) The life table is fit predominantly on BTC's single 2017-2020
      supercycle (one or two long bull/bear spells dominating the
      completed-spell sample) and may not generalise to ETH's shorter,
      differently-shaped history at all -- exactly what the pre-registered
      B4 falsification test above is designed to catch.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid, 3 thresh x 2 max_discount) + 6 (primary cell's full
``compare()``: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 12 (B3's full 6-cell grid x 2 markets, 2 of
the 12 reused directly from the primary ``compare()``'s own inner_val rows,
10 freshly computed) + 2 (B5's 0.40% fee tier, 2 markets) = 26 total. IF
Step-0 finds no qualifying cell, this file stops after the 6 Step-0 cells
(6 total). No other nuisance-parameter sweep is performed, so it adds 0
configurations to either count.

USAGE
-----
    python experiments/r114_conservative_lifetable_hazard.py
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
    MIN_SPELLS,
    OOS_START,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    REFIT_EVERY_DAYS,
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
    causal_truncation_probe_series,
    compare,
    count_inner_train_spells,
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
    step0_gate,
)

# ---------------------------------------------------------- pre-registered
# The primary cell's own (thresh, max_discount) -- first entry of
# SELECTION_ORDER, per this round's own pre-registration text.
PRIMARY_THRESH, PRIMARY_MAXD = SELECTION_ORDER[0]


# ================================================================== (1)
# The mechanism itself: binarized vote -> duration -> marginal life-table
# hazard -> causal percentile-rank state -> discount on v4's own UNCHANGED
# frac*scale. No new statistic invented here -- pure composition of
# r114_shared's own pre-verified, causal building blocks.
# ==================================================================

def compute_full_state(df: pd.DataFrame) -> pd.Series:
    """vote -> binarized daily regime state -> duration-since-flip ->
    walk-forward marginal life-table hazard -> causal percentile-rank state
    in [0, 1], over whatever frame `df` is (the caller decides how much
    history it contains -- this function makes no reference to any fixed
    calendar date)."""
    state = regime_state_daily(df)
    duration = regime_duration_daily(state)
    hazard = rolling_lifetable_hazard(state, duration)
    return hazard_to_state(hazard)


def build_target(df: pd.DataFrame, thresh: float = PRIMARY_THRESH,
                  max_discount: float = PRIMARY_MAXD) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount),
    where discount is driven by the life-table hazard state built from `df`
    alone. Self-contained (a pure function of `df`), so it is directly
    usable as a `TargetStrategy` candidate on any window (inner_train,
    inner_val, eth_replication, or a truncated probe frame)."""
    state01 = compute_full_state(df)
    return apply_discount(df, state01, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"lifetable_hazard_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r114_shared.STEP0_THRESH_GRID x r114_shared.STEP0_MAXD_GRID,
# scored via r114_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame (mirrors R-109's own step0_grid convention
# -- see banner above for why this is causally identical to computing state
# on the inner-train slice alone, for every bar inside it). Also returns the
# intermediate `state`/`duration` daily series so main() can report the
# disclosed completed-spell diagnostic without recomputing anything.
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], pd.Series, pd.Series, pd.Series]:
    state = regime_state_daily(btc)
    duration = regime_duration_daily(state)
    hazard = rolling_lifetable_hazard(state, duration)
    state01 = hazard_to_state(hazard)

    df_inner_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    rows = []
    for thresh in STEP0_THRESH_GRID:
        for maxd in STEP0_MAXD_GRID:
            gate = step0_gate(df_inner_train, state01, thresh, maxd)
            rows.append(dict(thresh=thresh, max_discount=maxd, **gate))
    return rows, state, duration, state01


def select_primary(rows: list[dict]) -> dict | None:
    by_key = {(r["thresh"], r["max_discount"]): r for r in rows}
    for key in SELECTION_ORDER:
        r = by_key.get(key)
        if r is not None and r["passed"]:
            return r
    return None


def print_step0_table(rows: list[dict], n_bars: int, n_spells: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars, state built from marginal life-table hazard: "
          f"REFIT_EVERY_DAYS={REFIT_EVERY_DAYS}, LAPLACE={LAPLACE}, MIN_SPELLS={MIN_SPELLS}, "
          f"{len(DURATION_BUCKET_EDGES) - 1} duration buckets)")
    print(f"DISCLOSED SMALL-SAMPLE DIAGNOSTIC: completed vote-flip spells available by the end "
          f"of inner-train = {n_spells}  (MIN_SPELLS={MIN_SPELLS} kill switch; this project's own "
          f"standing thin-effective-sample-size concern, here applied to regime-flip count)")
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
            label = f"lifetable_hazard_t{key[0]:g}_m{key[1]:g}"
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
    label = f"lifetable_hazard_t{thresh:g}_m{maxd:g}"

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

    hr("R-114 CONSERVATIVE: LifetableHazardKellyV4 -- marginal (duration-only) "
       "actuarial life-table hazard discount on v4's own frac*scale")
    print("mechanism: v4's own 3-anchor vote -> binarized daily bull/bear regime -> days")
    print("since last flip (duration) -> walk-forward Cutler & Ederer (1958) marginal life-table")
    print("hazard for the current duration bucket (refit every REFIT_EVERY_DAYS days, only spells")
    print("completed strictly before the refit day) -> causal rolling percentile-rank state in")
    print("[0,1] -> linear discount on v4's UNCHANGED frac*scale product. Duration-dependence")
    print("axis (Diebold & Rudebusch 1990 / Maheu & McCurdy 2000), the first ERR-axis construction")
    print("keyed on the regime's own AGE rather than sampling significance, specification/model")
    print("disagreement, or distributional novelty. Full grounding in r114_shared.py's own module")
    print("docstring.")
    print(f"\nREFIT_EVERY_DAYS={REFIT_EVERY_DAYS}  LAPLACE={LAPLACE}  MIN_SPELLS={MIN_SPELLS}  "
          f"DURATION_BUCKET_EDGES={DURATION_BUCKET_EDGES}  (all frozen in r114_shared.py, "
          f"none swept by this file)")
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
    step0_rows, state, duration, state01 = step0_grid(btc)
    n_bars_inner_train = int(np.sum((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                     (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))))

    inner_train_end_pos = int(np.searchsorted(
        state.index.values, pd.Timestamp(INNER_TRAIN_END, tz="UTC").to_datetime64(), side="right"))
    n_spells = count_inner_train_spells(state, duration, inner_train_end_pos)

    print_step0_table(step0_rows, n_bars_inner_train, n_spells)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the life-table")
        print("hazard discount is either a near-total no-op, a near-exact rescale of v4's own path,")
        print("a relabelled volatility rescale, or degenerate everywhere on the pre-registered grid.")
        print("Per this file's own pre-registration, this Step-0 table (plus the causal-safety")
        print("probe above) is the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0.")
        print("No promotion-bar code runs, and no inner-validation Sharpe/PnL number or ETH bar")
        print("is ever read.")

        hr("VERDICT")
        print("Step-0 (6-cell thresh x max_discount grid): FAIL (no cell qualifies)")
        print(f"completed inner-train spells: {n_spells}  (MIN_SPELLS={MIN_SPELLS})")
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
                    probe_ok=probe_ok, n_spells=n_spells, n_configs=n_configs, max_ts=max_ts,
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
    print(f"completed inner-train spells: {n_spells}  (MIN_SPELLS={MIN_SPELLS})")
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
          f"2 B5 fee-tier; no other nuisance-parameter sweep, adds 0)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, state=state, duration=duration,
                state01=state01, primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                n_spells=n_spells, promotion_bar=bar, verdict=verdict, n_configs=n_configs,
                max_ts=max_ts)


if __name__ == "__main__":
    main()
