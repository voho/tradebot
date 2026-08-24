#!/usr/bin/env python
"""R-121 CONSERVATIVE branch: ``SigKnnNoveltyBrakeKellyV4`` -- R-109's own
kNN distributional-novelty exposure discount on ``kelly_regime_v4``'s
``frac * scale`` product, UNCHANGED architecture end to end, with its ONE
input swapped: instead of R-109's 5 (or, on the conservative side, 3)
point-in-time hand-engineered scalar features, the novelty statistic here is
computed from a depth-2 truncated PATH SIGNATURE of the trailing
(log-price, log-realized-vol) path -- an order-sensitive summary of HOW price
and volatility co-moved within a window, not merely their marginal levels at
one instant.

MECHANISM, in one sentence: ``experiments/r121_shared.py``'s
``build_sig2_features`` (3 daily columns -- ``s1_log_price``, ``s1_log_vol``,
``levy_log_price_log_vol``, the depth-2 signature of the 2D
(log-price, log-vol) path over a trailing 1-day window, strictly causal) ->
``rolling_knn_distance`` (R-109's own nonparametric k-nearest-neighbour
distance primitive, UNCHANGED, k=10, refit_every=30, its own defaults) ->
``causal_rolling_percentile_rank`` (R-109's own causal [0,1] state
normalization, UNCHANGED) -> ``apply_discount`` (R-109's own linear discount
curve bolted onto v4's unchanged ``frac * scale``, UNCHANGED). Every piece of
this pipeline downstream of the feature panel -- the kNN distance function,
the percentile-rank state construction, the discount curve, the Step-0
kill-switch thresholds, the B1-B5 promotion bar -- is byte-identical to
R-109's own conservative/novel branches; the ONLY thing that differs from
R-109's own novel (kNN) branch is which feature panel feeds
``rolling_knn_distance``. This isolates a single question, named verbatim by
R-115's own closing line: is a materially different novelty STATISTIC (not a
repair of the REFERENCE POOL, which R-112/R-115 already tried and which both
still failed B4) the missing piece that R-109's kNN construction needed to
survive ETH falsification?

The full non-duplication argument against R-109/R-112/R-113/R-115 and every
other ERR-axis round, the literature grounding for path signatures as a
novelty statistic, the hand-verified Levy-area construction, the
scipy-unavailability note, and this round's five named failure risks all live
in ``experiments/r121_shared.py``'s own module docstring, written by the
operator before either branch was dispatched -- NOT re-derived here; read
that file in full first. This file imports ONLY from ``experiments.
r121_shared`` (itself chaining r109_shared -> r106_shared -> ... ->
r102_shared's unchanged control machinery, plus r105_shared's promotion-bar
gates) and, for the two pre-registered primary-cell constants R-121's own
shared module does not itself re-export (``PRIMARY_THRESH``,
``PRIMARY_MAXD`` -- present in ``r109_shared`` at the values verified below),
directly from ``experiments.r109_shared`` -- read-only in both cases. This
file never edits ``r121_shared.py``, never coordinates with or reads the
NOVEL branch's file, and never reads a bar at or after
``r121_shared.OOS_START`` (2023-01-01).

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this):

  1. ``daily = r121_shared.build_sig2_features(df)`` -- the 3-column
     depth-2 signature panel of the (log-price, log-realized-vol) path,
     already verified by the operator (module docstring + self-test) to be
     strictly causal.
  2. ``dist = r121_shared.rolling_knn_distance(daily, k=K, refit_every=
     REFIT_EVERY)`` -- R-109's own kNN distance primitive, UNCHANGED.
  3. ``state = r121_shared.causal_rolling_percentile_rank(dist)`` -- R-109's
     own causal percentile-rank state construction, UNCHANGED.
  4. ``target = r121_shared.apply_discount(df, state, thresh,
     max_discount) = r121_shared.v4_target(df) * (1 - discount)`` -- R-109's
     own discount curve bolted onto v4's unchanged ``frac * scale`` product,
     UNCHANGED.

``k`` AND ``refit_every`` -- PRE-REGISTERED, kept at
``rolling_knn_distance``'s own defaults (``k=10``, ``refit_every=30``,
verified programmatically below against the live function signature), NOT
swept, for the SAME reason R-109's own novel branch gave: no diagnostics were
run to search for a "better" k/refit_every before touching inner-validation
data (a search there would itself be a hidden trial), and this project's
1-5-variants-per-branch convention is already spent on the
(thresh, max_discount) grid below. This choice contributes exactly ONE
(k, refit_every) configuration to the round.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read,
identical in shape to R-109's own): sweep ``r121_shared.STEP0_THRESH_GRID x
r121_shared.STEP0_MAXD_GRID`` (3 x 2 = 6 cells) on BTC's
``INNER_TRAIN_START..INNER_TRAIN_END`` via ``r121_shared.step0_gate(df.loc[
INNER_TRAIN_START:INNER_TRAIN_END], state, thresh, max_discount)``, where
``state`` is computed over BTC's FULL non-holdout frame (mirrors R-109's own
step0_grid convention -- causality guarantees this is identical, for every
bar inside inner-train, to computing the statistic on the inner-train slice
alone). A cell QUALIFIES iff ``step0_gate(...)['passed']`` is True. PRIMARY
cell: the first ``(thresh, max_discount)`` in ``r121_shared.SELECTION_ORDER``
that qualifies. The operator has already confirmed on real BTC inner-train
data that the pre-registered primary cell (thresh=0.90, max_discount=1.0)
PASSES Step-0 (bind_frac=0.0623, r2_vs_v4=0.967, r2_vs_vol=-3.271,
state_cv=0.652) -- this file re-derives that number from scratch rather than
hard-coding it, but does not treat a different outcome as license to change
the pre-registered rule.

PROMOTION BAR (only if Step-0 passes; identical shape to R-109's own, via
``r121_shared``'s re-exported gate functions):
  B1 (gating): ``b1_from_inner_val`` on the primary cell's inner_val rows,
     both markets (spot, futures_5x).
  B2 (diagnostic ONLY, never gates): ``b2_diagnostic``.
  B3 (plateau, gating): the FULL 6-cell Step-0 grid's own inner-validation
     numbers (both markets each, primary cell's 2 rows reused directly from
     its own ``compare()``) -- PASS requires a directionally consistent
     majority across the 12 resulting cells.
  B4 (ETH falsification, gating, PRE-REGISTERED as this branch's ONE
     falsification test): ``b4_eth_falsification`` on ``compare(...,
     eth=load_eth())`` -- FULL pass required (both markets same-signed as
     BTC inner_val). This is the exact test R-109's kNN construction failed
     on ETH spot (sign inverted); it is this round's entire reason for
     existing.
  B5 (cost robustness, gating): ``b5_fee_tier`` at the 0.40% taker tier,
     primary cell, BTC inner_val, both markets.
PROMOTE-candidate only if the causal-truncation probe AND B1 (both markets)
AND B3 (plateau) AND B4 (full) AND B5 all hold (B2 is diagnostic-only).
Default: NEGATIVE.

CAUSAL SAFETY: ``causal_truncation_probe_series`` applied to this file's own
composed ``build_target`` (features -> kNN distance -> percentile-rank state
-> discount -> ``v4_target * (1 - discount)``), run on BTC's full
non-holdout frame, BEFORE the Step-0 grid is scored and before any
inner-validation/ETH performance number is computed. If it does not pass,
that is a real bug to find and report, not something to work around.

DIAGNOSTIC (not a gate): ``levy_vs_r109_features_corr`` -- the Pearson
correlation between this round's sig2 kNN distance series and R-109's own
``anchor_disp``/``kurtosis`` conservative-branch features (via
``experiments.r109_shared.FEATURE_BUILDERS`` and ``build_daily_features``,
computed the identical way R-109 itself computed them). Answers
``r121_shared.py``'s own named failure risk #3: is the path-signature
statistic actually new information, or a renamed copy of R-109's own
features in different notation? Reported, never gated on.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 6
(Step-0 grid) + 6 (primary cell's full ``compare()``: inner_train x2 markets
+ inner_val x2 markets + eth_replication x2 markets) + 12 (B3's full 6-cell
grid x 2 markets, 2 of the 12 reused directly from the primary
``compare()``'s own inner_val rows) + 2 (B5's 0.40% fee tier, 2 markets) =
26 total -- matching R-109's own count exactly, since the grid shape and
promotion bar are unchanged. IF Step-0 finds no qualifying cell, this file
stops after the 6 Step-0 cells (6 total). No k/refit_every sweep is
performed, so it adds 0 configurations to either count.

USAGE
-----
    python experiments/r121_conservative_signature_knn.py
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

import experiments.r121_shared as r121_shared  # noqa: E402
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
    build_sig2_features,
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
    step0_gate,
)

# r121_shared re-exports everything r109_shared exports EXCEPT the two
# pre-registered primary-cell constants (PRIMARY_THRESH, PRIMARY_MAXD) and
# the original point-in-time feature-builder machinery (FEATURE_BUILDERS,
# build_daily_features) needed only for this file's own diagnostic
# correlation check below -- both pulled directly, read-only, from
# experiments.r109_shared itself (which r121_shared already imports from).
import experiments.r109_shared as r109_shared  # noqa: E402
from experiments.r109_shared import PRIMARY_MAXD, PRIMARY_THRESH  # noqa: E402

assert PRIMARY_THRESH == 0.90 and PRIMARY_MAXD == 1.0, (
    "pre-registration text is stale: PRIMARY_THRESH/PRIMARY_MAXD no longer "
    f"match r109_shared's own values ({PRIMARY_THRESH}, {PRIMARY_MAXD})")

# ---------------------------------------------------------- pre-registered
K = 10             # rolling_knn_distance's own default -- verified below
REFIT_EVERY = 30   # rolling_knn_distance's own default -- verified below

_sig = inspect.signature(rolling_knn_distance).parameters
assert _sig["k"].default == K, ("K does not match rolling_knn_distance's "
                                 f"own default ({_sig['k'].default}) -- pre-registration text is stale")
assert _sig["refit_every"].default == REFIT_EVERY, (
    "REFIT_EVERY does not match rolling_knn_distance's own default "
    f"({_sig['refit_every'].default}) -- pre-registration text is stale")

MODEL_LABEL = "sig_conservative_knn"


# ================================================================== (1)
# The mechanism itself: 2D (log-price, log-vol) path -> depth-2 signature
# panel -> kNN distance (R-109's own primitive, UNCHANGED) -> percentile-rank
# state -> discount on v4's own UNCHANGED frac*scale.
# ==================================================================

def compute_full_state(df: pd.DataFrame, k: int = K, refit_every: int = REFIT_EVERY) -> pd.Series:
    """sig2 path-signature features -> kNN distance -> causal percentile-rank
    state, over whatever frame `df` is (the caller decides how much history
    it contains -- this function itself makes no reference to any fixed
    calendar date)."""
    daily = build_sig2_features(df)
    dist = rolling_knn_distance(daily, k=k, refit_every=refit_every)
    return causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, thresh: float = PRIMARY_THRESH,
                  max_discount: float = PRIMARY_MAXD, k: int = K,
                  refit_every: int = REFIT_EVERY) -> np.ndarray:
    """The ENTIRE mechanism, composed: v4_target(df) * (1 - discount), where
    discount is driven by the sig2 kNN novelty state built from `df` alone.
    Self-contained (a pure function of `df`), so it is directly usable as a
    TargetStrategy candidate on any window (inner_train, inner_val,
    eth_replication, or a truncated probe frame)."""
    state = compute_full_state(df, k=k, refit_every=refit_every)
    return apply_discount(df, state, thresh, max_discount)


def make_build_target(thresh: float, max_discount: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, thresh=thresh, max_discount=max_discount)
    _build.__name__ = f"{MODEL_LABEL}_t{thresh:g}_m{max_discount:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r121_shared.STEP0_THRESH_GRID x r121_shared.STEP0_MAXD_GRID,
# scored via r121_shared.step0_gate on BTC inner-train, state computed over
# the FULL non-holdout BTC frame (mirrors R-109's own step0_grid convention).
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
          f"{n_bars:,} bars, state built from sig2 path-signature features, k={K}, "
          f"refit_every={REFIT_EVERY}d, BASELINE_WINDOW_DAYS={BASELINE_WINDOW_DAYS}, "
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
            label = f"{MODEL_LABEL}_t{key[0]:g}_m{key[1]:g}"
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
    label = f"{MODEL_LABEL}_t{thresh:g}_m{maxd:g}"

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


# ================================================================== (5)
# Diagnostic (not a gate): is the sig2 kNN distance series actually new
# information, or a renamed copy of R-109's own anchor_disp/kurtosis
# conservative-branch features? r121_shared.py's own named failure risk #3.
# ==================================================================

def diagnostic_levy_vs_r109_corr(btc: pd.DataFrame, k: int = K,
                                  refit_every: int = REFIT_EVERY) -> dict:
    daily_sig2 = build_sig2_features(btc)
    dist = rolling_knn_distance(daily_sig2, k=k, refit_every=refit_every)

    r109_builders = {name: r109_shared.FEATURE_BUILDERS[name] for name in ("anchor_disp", "kurtosis")}
    daily_r109 = r109_shared.build_daily_features(btc, r109_builders)

    out = {}
    for name in ("anchor_disp", "kurtosis"):
        joined = pd.concat([dist.rename("dist"), daily_r109[name]], axis=1, sort=False).dropna()
        if len(joined) >= 3 and joined["dist"].std() > 0 and joined[name].std() > 0:
            out[name] = float(np.corrcoef(joined["dist"], joined[name])[0, 1])
        else:
            out[name] = float("nan")
        out[f"{name}_n"] = int(len(joined))
    return out


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-121 CONSERVATIVE: SigKnnNoveltyBrakeKellyV4 -- path-SIGNATURE "
       "feature panel feeding R-109's own UNCHANGED kNN novelty-discount pipeline")
    print("mechanism: depth-2 truncated signature of the trailing 1-day (log-price, log-vol)")
    print("path (s1_log_price, s1_log_vol, levy_log_price_log_vol) -> R-109's own kNN mean")
    print("distance to k nearest neighbours in a trailing 730-day reference SET, refit every")
    print("refit_every days (walk-forward, strictly causal) -> R-109's own causal rolling")
    print("percentile-rank state in [0,1] -> R-109's own linear discount on v4's UNCHANGED")
    print("frac*scale product. Every stage downstream of the feature panel is byte-identical")
    print("to R-109's own novel (kNN) branch; only the feature MAP differs. Full grounding in")
    print("r121_shared.py's own module docstring.")
    print(f"\nk={K}, refit_every={REFIT_EVERY}d  (both = rolling_knn_distance's own defaults, "
          f"verified programmatically above; no sweep performed -- see banner reasoning)")
    print(f"STEP0_THRESH_GRID={STEP0_THRESH_GRID}  STEP0_MAXD_GRID={STEP0_MAXD_GRID}  "
          f"({len(STEP0_THRESH_GRID) * len(STEP0_MAXD_GRID)} cells)")
    print(f"SELECTION_ORDER={SELECTION_ORDER}")
    print(f"PRIMARY_THRESH={PRIMARY_THRESH}, PRIMARY_MAXD={PRIMARY_MAXD}  "
          f"(pre-registered; imported read-only from experiments.r109_shared)")

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
        print("CAUSAL SAFETY PROBE FAILED -- this is a real bug, not a research result.")
        print("Stopping here per docs/ROUTINE.md's own precedence: a lookahead is a bug report")
        print("first. No Step-0, promotion-bar, or diagnostic code runs.")
        print(f"causal truncation probe: {probe_ok}")
        print("Step-0: NOT RUN (causal probe failed)")
        print("B1/B2/B3/B4/B5: NOT COMPUTED (causal probe failed)")
        print("VERDICT: NEGATIVE (causal safety probe failure -- bug, not a result)")
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated (total): 0 (stopped at causal safety probe)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, probe_ok=False, passed_step0=False, n_configs=0,
                    max_ts=max_ts, verdict="NEGATIVE (causal safety probe failure)")

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY / NON-DUPLICATION KILL SWITCH "
       "(run BEFORE any inner-validation Sharpe/PnL number)")
    step0_rows, state = step0_grid(btc)
    n_bars_inner_train = int(np.sum((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                                     (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))))
    print_step0_table(step0_rows, n_bars_inner_train)

    primary_row = select_primary(step0_rows)

    # -------------------------------------------------------- diagnostic
    hr("DIAGNOSTIC (not a gate) -- sig2 kNN distance series vs R-109's own "
       "anchor_disp/kurtosis conservative-branch features (r121_shared.py's own named "
       "failure risk #3: is this a materially new statistic, or a renamed copy?)")
    corr = diagnostic_levy_vs_r109_corr(btc)
    print(f"  corr(sig2_knn_dist, anchor_disp) = {corr['anchor_disp']:+.4f}  (n={corr['anchor_disp_n']:,})")
    print(f"  corr(sig2_knn_dist, kurtosis)    = {corr['kurtosis']:+.4f}  (n={corr['kurtosis_n']:,})")
    print("  levy_vs_r109_features_corr = "
          f"{{'anchor_disp': {corr['anchor_disp']:+.4f}, 'kurtosis': {corr['kurtosis']:+.4f}}}")

    if primary_row is None:
        hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 6 (thresh, max_discount) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the sig2 kNN")
        print("novelty discount is either a near-total no-op, a near-exact rescale of v4's own")
        print("path, a relabelled volatility rescale, or degenerate everywhere on the")
        print("pre-registered grid. Per this file's own pre-registration, this Step-0 table (plus")
        print("the causal-safety probe and diagnostic above) is the branch's ENTIRE product,")
        print("reported NEGATIVE / stopped-at-Step-0. No promotion-bar code runs, and no")
        print("inner-validation Sharpe/PnL number or ETH bar is ever read.")

        hr("VERDICT")
        print("Step-0 (6-cell thresh x max_discount grid): FAIL (no cell qualifies)")
        print(f"causal truncation probe: {probe_ok}")
        print(f"levy_vs_r109_features_corr: {corr}")
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
                    probe_ok=probe_ok, levy_vs_r109_features_corr=corr,
                    n_configs=n_configs, max_ts=max_ts,
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

    hr("B4 -- ETH falsification (pre-registered, this branch's ONE falsification test)")
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
    print(f"Step-0 primary cell: thresh={primary_key[0]:g}, max_discount={primary_key[1]:g}  "
          f"(bind_frac={primary_row['bind_frac']:.4f}, r2_vs_v4={primary_row['r2_vs_v4']:.4f}, "
          f"r2_vs_vol={primary_row['r2_vs_vol']:.4f}, state_cv={primary_row['state_cv']:.4f})")
    print(f"levy_vs_r109_features_corr: {corr}")
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
                levy_vs_r109_features_corr=corr,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
