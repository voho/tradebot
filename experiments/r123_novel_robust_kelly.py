#!/usr/bin/env python
"""R-123 NOVEL branch: ``RobustKellyKnnNoveltyBrake`` -- R-109's own
nonparametric k-nearest-neighbour distributional-novelty statistic, but
entering the vote/scale computation DIRECTLY as a Sun & Boyd (2018)-motivated
joint multiplier on the pre-deadband ``frac * scale`` product, instead of
sitting as a multiplicative discount on top of ``kelly_regime_v4``'s already-
deadbanded final target the way R-109/R-112/R-115/R-121/R-122 all did.

The complete pre-registration for this round -- direction, literature
citations (Sun & Boyd 2018, arXiv:1812.10371; the Pinsker-bound first-order
derivation of the ``1 - c*sqrt(epsilon)`` multiplier), non-duplication
argument against every prior ERR-axis round including the five prior
discount-on-final-target attempts, and the four named failure risks -- lives
in ``experiments/r123_shared.py``'s own module docstring, written by the
operator before either branch was dispatched, and is NOT re-derived here:
read that file in full first. This file imports ONLY from
``experiments.r123_shared`` (itself re-exporting r109_shared's unchanged
control machinery, feature panel, and B1-B5 promotion-bar functions), never
edits it, never coordinates with or reads the conservative branch's file, and
never reads a bar at or after ``r123_shared.OOS_START`` (2023-01-01).

MECHANISM, in one sentence: build R-109's own 5-feature OHLCV-only daily
market-state panel, score each day's mean Euclidean distance (kNN, k=10) to
its k nearest neighbours in its own trailing 730-day reference SET, normalize
to a causal [0,1] percentile-rank novelty state exactly as R-109 did, but
instead of discounting v4's finished ``v4_target`` output, multiply the
PRE-deadband product ``frac * scale`` jointly by ``robust_multiplier(state) =
1 - c*sqrt(eps_max*state)`` (concave in state, no threshold -- binds at every
state > 0) and only THEN apply the deadband.

EXACT CONSTRUCTION (the entire mechanism -- no other logic, gate, or
heuristic is added on top of this):

  1. ``daily = r123_shared.build_daily_features(df,
     r123_shared.NOVEL_FEATURE_BUILDERS)`` -- the 5-feature panel (log_vol,
     anchor_disp, kurtosis, volume_z, skew), unchanged from R-109's own
     novel branch.
  2. ``dist = r123_shared.rolling_knn_distance(daily, k=K)`` -- R-109's own
     kNN distance statistic, k=10 (verified programmatically below against
     the live function's own default, not hand-copied), refit_every=30
     (the function's own default, not swept -- see R-109's own reasoning,
     reused verbatim: no diagnostics-based search for a "better" k/
     refit_every before touching inner-validation data).
  3. ``state = r123_shared.causal_rolling_percentile_rank(dist)`` -- a
     causal rolling percentile rank onto [0, 1] against the statistic's own
     trailing 730-day history, unchanged from R-109.
  4. ``target = r123_shared.novel_target(df, state, eps_max, c)`` -- THE
     architectural change this round makes: ``frac * scale`` (v4's own
     UNCHANGED vote and volatility-target, reproduced byte-for-byte) is
     jointly multiplied by ``robust_multiplier(state, eps_max, c) = 1 -
     c*sqrt(eps_max*state)`` (clipped to [0,1]) BEFORE the deadband is
     applied, rather than a discount applied to v4's own already-deadbanded
     final path.

``k``/``refit_every`` -- PRE-REGISTERED, REUSED FROM R-109 (NOT SWEPT): the
same reasoning as every prior round in this family -- the kNN literature's
own 10-50 range, 10 at its conservative end, refit_every=30 a monthly
walk-forward cadence matching this ledger's other regime detectors. No
sweep is performed here; the Step-0 grid below (``eps_max``/``c`` only, 2x2
= 4 cells, a smaller grid than the conservative branch's 3x2=6 since this
branch has 2 free parameters x 2 grid points each) is the entirety of this
branch's configuration search before inner-validation is read.

CAUSAL SAFETY FIRST: ``r123_shared.causal_truncation_probe_series`` is run
twice -- once at import time, in this file's own ``_self_test()`` (bottom of
file, synthetic data, mirroring ``r123_shared.py``'s own module-level
self-test convention), and again in ``main()`` on real BTC data at the
pre-registered primary cell, before the Step-0 grid is scored and well
before any inner-validation/ETH performance number is computed.

STEP-0 RULE (frozen before any inner-validation Sharpe/PnL number is read):
sweep ``r123_shared.NOVEL_EPS_GRID x r123_shared.NOVEL_C_GRID`` (2 x 2 = 4
cells) on BTC's ``INNER_TRAIN_START..INNER_TRAIN_END`` via
``r123_shared.step0_gate_generic(btc.loc[INNER_TRAIN_START:INNER_TRAIN_END],
novel_target(...), novel_discount_fraction(...), state)``, where ``state =
compute_full_state(btc)`` is computed over BTC's FULL non-holdout frame
(mirrors R-109's own convention: causally identical, for every bar inside
inner-train, to computing state on the inner-train slice alone). A cell
QUALIFIES iff ``step0_gate_generic(...)['passed']`` is True (bind_frac > 1%,
R^2 vs v4's own target < 0.98, R^2 vs v4's own realized-vol input < 0.90,
state CoV >= 5%). PRIMARY cell: the first ``(eps_max, c)`` in
``r123_shared.NOVEL_SELECTION_ORDER`` that qualifies -- ``NOVEL_PRIMARY =
(0.5, 1.414)`` is ``NOVEL_SELECTION_ORDER[0]``. If NONE qualify: STOP, report
Step-0 FAIL as a complete NEGATIVE result -- no inner-validation or ETH bar
is read past that point.

PROMOTION BAR (only if Step-0 passes; identical shape to every SIZE/ERR-axis
round since R-89, via ``r123_shared``'s re-exported gate functions):
  B1 (gating): ``r123_shared.b1_from_inner_val`` on the primary cell's
     inner_val rows, both markets (spot, futures_5x).
  B2 (diagnostic ONLY, never gates): ``r123_shared.b2_diagnostic``.
  B3 (plateau, gating): the FULL 4-cell Step-0 ``(eps_max, c)`` grid's own
     inner-validation numbers (both markets each, primary cell's 2 rows
     reused directly from its own ``compare()``) -- PASS requires a
     directionally consistent (same-sign) majority across the resulting
     8 cells. Since k/refit_every are NOT swept, this 4-cell grid is the
     entirety of this branch's plateau evidence.
  B4 (ETH falsification, gating, PRE-REGISTERED as this round's ONE
     falsification test): ``r123_shared.b4_eth_falsification`` -- require
     FULL pass (both markets same-signed as BTC inner_val).
  B5 (cost robustness, gating): ``r123_shared.b5_fee_tier`` at the 0.40%
     taker tier, primary cell, BTC inner_val, both markets.

PRE-REGISTERED DECISION RULE, stated verbatim and NOT altered after seeing
any number: PROMOTE-candidate only if the causal-truncation probe AND B1
(both markets) AND B3 (plateau majority) AND B4 (full) AND B5 all pass.
Anything else is NEGATIVE. B2 never gates. Default: NEGATIVE.

CONFIGURATIONS EVALUATED IN THIS FILE (if Step-0 selects a primary): 4
(Step-0 grid, 2 eps_max x 2 c) + 6 (primary cell's full ``compare()``:
inner_train x2 markets + inner_val x2 markets + eth_replication x2 markets)
+ 8 (B3's full 4-cell grid x 2 markets, 2 of the 8 reused directly from the
primary ``compare()``'s own inner_val rows, 6 freshly computed) + 2 (B5's
0.40% fee tier, 2 markets) = 20 total. IF Step-0 finds no qualifying cell,
this file stops after the 4 Step-0 cells (4 total). No k/refit_every sweep
is performed, so it adds 0 configurations to either count.

USAGE
-----
    python experiments/r123_novel_robust_kelly.py
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

from experiments import r123_shared  # noqa: E402

# ---------------------------------------------------------- pre-registered
K = 10   # rolling_knn_distance's own default -- verified below against the
         # live function signature, not hand-copied. Not swept (see banner).

_sig = inspect.signature(r123_shared.rolling_knn_distance).parameters
assert _sig["k"].default == K, ("K does not match rolling_knn_distance's "
                                 f"own default ({_sig['k'].default}) -- pre-registration text is stale")


# ================================================================== (1)
# The mechanism itself: 5-feature panel -> kNN distance -> percentile-rank
# state -> joint sqrt-shaped robust multiplier on v4's own UNCHANGED
# frac*scale, applied BEFORE the deadband.
# ==================================================================

def compute_full_state(df: pd.DataFrame, k: int = K) -> pd.Series:
    """features -> kNN distance -> causal percentile-rank state, over
    whatever frame `df` is (the caller decides how much history it
    contains -- this function itself makes no reference to any fixed
    calendar date). Identical construction to R-109's own novel branch."""
    daily = r123_shared.build_daily_features(df, r123_shared.NOVEL_FEATURE_BUILDERS)
    dist = r123_shared.rolling_knn_distance(daily, k=k)
    return r123_shared.causal_rolling_percentile_rank(dist)


def build_target(df: pd.DataFrame, eps_max: float = r123_shared.NOVEL_PRIMARY[0],
                  c: float = r123_shared.NOVEL_PRIMARY[1], k: int = K) -> np.ndarray:
    """The ENTIRE mechanism, composed: state is recomputed from `df` alone
    (still strictly causal -- `compute_full_state` never looks beyond the
    frame it is given, and `rolling_knn_distance`/`causal_rolling_percentile_
    rank` are themselves walk-forward), then `r123_shared.novel_target`
    applies the joint sqrt-shaped multiplier to `frac*scale` and deadbands.
    Self-contained (a pure function of `df`), so it is directly usable as a
    `TargetStrategy` candidate on any window (inner_train, inner_val,
    eth_replication, or a truncated probe frame) -- matching R-109's own
    novel branch's per-frame recomputation idiom."""
    state = compute_full_state(df, k=k)
    return r123_shared.novel_target(df, state, eps_max, c)


def make_build_target(eps_max: float, c: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_target(df, eps_max=eps_max, c=c)
    _build.__name__ = f"robust_kelly_knn_eps{eps_max:g}_c{c:g}"
    return _build


# ================================================================== (2)
# Step-0 grid: r123_shared.NOVEL_EPS_GRID x r123_shared.NOVEL_C_GRID (4
# cells), scored via r123_shared.step0_gate_generic on BTC inner-train,
# state computed over the FULL non-holdout BTC frame (mirrors R-109's own
# step0_grid convention: causally identical, for every bar inside
# inner-train, to computing state on the inner-train slice alone).
# ==================================================================

def step0_grid(btc: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    state = compute_full_state(btc)
    df_inner_train = btc.loc[r123_shared.INNER_TRAIN_START:r123_shared.INNER_TRAIN_END]
    rows = []
    for eps_max in r123_shared.NOVEL_EPS_GRID:
        for c in r123_shared.NOVEL_C_GRID:
            candidate_path = r123_shared.novel_target(df_inner_train, state, eps_max, c)
            discount_fraction = r123_shared.novel_discount_fraction(df_inner_train, state, eps_max, c)
            gate = r123_shared.step0_gate_generic(df_inner_train, candidate_path, discount_fraction, state)
            rows.append(dict(eps_max=eps_max, c=c, **gate))
    return rows, state


def select_primary(rows: list[dict]) -> dict | None:
    by_key = {(r["eps_max"], r["c"]): r for r in rows}
    for key in r123_shared.NOVEL_SELECTION_ORDER:
        r = by_key.get(key)
        if r is not None and r["passed"]:
            return r
    return None


# ================================================================== (3)
# B3 plateau: the full 4-cell (eps_max, c) grid's own inner-validation
# numbers (both markets), primary cell's 2 rows reused directly from its
# own compare().
# ==================================================================

def run_b3_full_grid(step0_rows: list[dict], primary_key: tuple[float, float],
                      inner_val_primary: list[dict], btc: pd.DataFrame) -> tuple[dict, bool]:
    plateau_rows: dict[tuple[float, float], list[dict]] = {}
    for r in step0_rows:
        key = (r["eps_max"], r["c"])
        if key == primary_key:
            plateau_rows[key] = [dict(market=c["market"], d_sharpe=c["d_sharpe"], d_dd=c["d_dd"],
                                       exposure_ratio=c["exposure_ratio"], vol_ratio=c["vol_ratio"],
                                       risk_matched=c["risk_matched"],
                                       boot_d_loggrowth=c["boot_d_loggrowth"], boot_lo=c["boot_lo"],
                                       boot_hi=c["boot_hi"], excludes_zero=c["excludes_zero"])
                                  for c in inner_val_primary]
        else:
            bf = make_build_target(*key)
            label = f"robust_kelly_knn_eps{key[0]:g}_c{key[1]:g}"
            plateau_rows[key] = r123_shared.inner_val_rows(bf, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


# ================================================================== (4)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# above), B4 (gating falsification), B5 (gating fee tier).
# ==================================================================

def run_promotion_bar(primary_key: tuple[float, float], step0_rows: list[dict],
                       btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    eps_max, c = primary_key
    build_primary = make_build_target(eps_max, c)
    label = f"robust_kelly_knn_eps{eps_max:g}_c{c:g}"

    r123_shared.hr(f"PROMOTION BAR -- PRIMARY CELL eps_max={eps_max:g}, c={c:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = r123_shared.compare(build_primary, label=label, btc=btc, eth=eth,
                                markets=(r123_shared.SPOT, r123_shared.FUTURES), include_eth=True)
    r123_shared.print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = r123_shared.b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = r123_shared.b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3_full_grid(step0_rows, primary_key, inner_val_primary, btc)
    b4_partial, b4_full, b4_cells = r123_shared.b4_eth_falsification(eth_primary, inner_val_primary)
    b5_pass, b5_cells = r123_shared.b5_fee_tier(build_primary, label, btc, inner_val_primary)

    all_pass = b1_pass and b3_pass and b4_full and b5_pass

    return dict(
        label=label, eps_max=eps_max, c=c, compare_rows=rows,
        inner_val_primary=inner_val_primary, eth_primary=eth_primary,
        b1_pass=b1_pass, b1_cells=b1_cells,
        b2_pass=b2_pass, b2_cells=b2_cells,
        b3_pass=b3_pass, b3_rows=b3_rows,
        b4_partial=b4_partial, b4_full=b4_full, b4_cells=b4_cells,
        b5_pass=b5_pass, b5_cells=b5_cells,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 8 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    r123_shared.hr("R-123 NOVEL: RobustKellyKnnNoveltyBrake -- k-nearest-neighbour "
                    "DISTRIBUTIONAL-NOVELTY statistic entering the vote/scale product DIRECTLY "
                    "via a Sun & Boyd (2018)-motivated joint sqrt-shaped multiplier")
    print("mechanism: R-109's own 5-feature OHLCV-only daily panel (log_vol, anchor_disp,")
    print("kurtosis, volume_z, skew) -> kNN distance (k=10) to trailing 730-day reference SET ->")
    print("causal percentile-rank state in [0,1] -> multiplier = 1 - c*sqrt(eps_max*state), applied")
    print("JOINTLY to v4's UNCHANGED frac*scale product BEFORE the deadband (not a discount on the")
    print("finished, already-deadbanded v4_target the way R-109/R-112/R-115/R-121/R-122 all did).")
    print("Full grounding in r123_shared.py's own module docstring.")
    print(f"\nk={K}  (rolling_knn_distance's own default, verified programmatically above; "
          f"refit_every left at its own default (30d); neither swept -- see banner reasoning)")
    print(f"NOVEL_EPS_GRID={r123_shared.NOVEL_EPS_GRID}  NOVEL_C_GRID={r123_shared.NOVEL_C_GRID}  "
          f"({len(r123_shared.NOVEL_EPS_GRID) * len(r123_shared.NOVEL_C_GRID)} cells)")
    print(f"NOVEL_SELECTION_ORDER={r123_shared.NOVEL_SELECTION_ORDER}  "
          f"NOVEL_PRIMARY={r123_shared.NOVEL_PRIMARY}")

    btc = r123_shared.load_btc()
    max_ts_seen.append(btc.index.max())
    r123_shared.assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {r123_shared.OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    # ================================================== CAUSAL SAFETY FIRST
    r123_shared.hr("CAUSAL TRUNCATION PROBE (real BTC data, composed build_target at "
                    "NOVEL_PRIMARY, run BEFORE Step-0 or any inner-val/ETH number)")
    probe_fn = make_build_target(*r123_shared.NOVEL_PRIMARY)
    print(f"causal_truncation_probe_series({probe_fn.__name__}, btc):")
    try:
        probe_ok = r123_shared.causal_truncation_probe_series(probe_fn, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\nCAUSAL SAFETY (truncation probe) PASS: {probe_ok}")

    # ============================================================= STEP 0
    r123_shared.hr("STEP 0 -- NON-DEGENERACY / NON-DUPLICATION KILL SWITCH "
                    "(run BEFORE any inner-validation Sharpe/PnL number)")
    step0_rows, state = step0_grid(btc)
    for r in step0_rows:
        label = f"eps_max={r['eps_max']:g}, c={r['c']:g}"
        r123_shared.print_step0_report(label, r)

    primary_row = select_primary(step0_rows)

    if primary_row is None:
        r123_shared.hr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("None of the 4 (eps_max, c) cells has bind_frac>1% AND r2_vs_v4<0.98 AND")
        print("r2_vs_vol<0.90 AND state_cv>=5% simultaneously on BTC inner-train: the joint")
        print("sqrt-shaped robust multiplier is either a near-total no-op, a near-exact rescale of")
        print("v4's own path, a relabelled volatility rescale, or degenerate everywhere on the")
        print("pre-registered grid.")
        print("Per this file's own pre-registration, this Step-0 table (plus the causal-safety")
        print("probe above) is the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0.")
        print("No promotion-bar code runs, and no inner-validation Sharpe/PnL number or ETH bar")
        print("is ever read.")

        r123_shared.hr("VERDICT")
        print("Step-0 (4-cell eps_max x c grid): FAIL (no cell qualifies)")
        print(f"causal truncation probe: {probe_ok}")
        print("B1: NOT COMPUTED (Step-0 kill switch)")
        print("B2: NOT COMPUTED (Step-0 kill switch)")
        print("B3: NOT COMPUTED (Step-0 kill switch)")
        print("B4: NOT COMPUTED (Step-0 kill switch)")
        print("B5: NOT COMPUTED (Step-0 kill switch)")
        print("VERDICT: NEGATIVE (Step-0 kill switch)")

        n_configs = len(step0_rows)
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated (total): {n_configs} (4 Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {r123_shared.OOS_START}: {max_ts < pd.Timestamp(r123_shared.OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                    probe_ok=probe_ok, n_configs=n_configs, max_ts=max_ts,
                    verdict="NEGATIVE (Step-0 kill switch)")

    primary_key = (primary_row["eps_max"], primary_row["c"])
    is_selection0 = (primary_key == r123_shared.NOVEL_SELECTION_ORDER[0])
    print(f"\nPRIMARY CELL SELECTED (Step-0 non-degeneracy rule only): "
          f"eps_max={primary_key[0]:g}, c={primary_key[1]:g}  "
          f"(bind_frac={primary_row['bind_frac']:.4f}, r2_vs_v4={primary_row['r2_vs_v4']:.4f}, "
          f"r2_vs_vol={primary_row['r2_vs_vol']:.4f}, state_cv={primary_row['state_cv']:.4f})")
    print(f"  selection: {'NOVEL_SELECTION_ORDER[0] (== NOVEL_PRIMARY) qualified' if is_selection0 else 'NOVEL_SELECTION_ORDER[0] did NOT qualify; next qualifying cell chosen'}")

    eth = r123_shared.load_eth()
    max_ts_seen.append(eth.index.max())
    r123_shared.assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {r123_shared.OOS_START})")

    bar = run_promotion_bar(primary_key, step0_rows, btc, eth)

    r123_shared.hr("B1 -- inner-validation, both markets (dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    r123_shared.hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    r123_shared.hr("B3 -- plateau: FULL 4-cell (eps_max, c) Step-0 grid, inner-validation, both markets")
    r123_shared.print_plateau_table(bar["b3_rows"])
    print(f"\nB3 (directionally consistent majority across the 8-cell grid): {bar['b3_pass']}")

    r123_shared.hr("B4 -- ETH falsification (pre-registered)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PARTIAL PASS (>=1 market): {bar['b4_partial']}   B4 FULL PASS (both markets): {bar['b4_full']}")

    r123_shared.hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier d_sharpe={c['d_sharpe']:+.4f}  "
              f"standard-fee d_sharpe={c['base_d_sharpe']:+.4f}  "
              f"fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    r123_shared.hr("TRADE COUNT (turnover check vs. v4's own unmodified path, per slice/market)")
    for r in bar["compare_rows"]:
        print(f"  {r['slice']:>16s} {r['market']:>9s}  cand_trades={r['cand_trades']:>5d}  "
              f"ctrl_trades(v4)={r['ctrl_trades']:>5d}")

    r123_shared.hr("VERDICT")
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
          f"(4 Step-0 grid + 6 primary-cell compare() + 8 B3 plateau "
          f"[4 (eps_max,c) cells x 2 markets, 2 reused from primary] + "
          f"2 B5 fee-tier; k/refit_every not swept, adds 0)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {r123_shared.OOS_START}: {max_ts < pd.Timestamp(r123_shared.OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, state=state,
                primary=primary_row, passed_step0=True, probe_ok=probe_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    """Mirrors r123_shared.py's own module-level self-test convention: the
    causal-truncation probe on this file's FULL composed candidate-build
    function (features -> kNN distance -> percentile-rank state ->
    novel_target, recomputed per-frame), on synthetic data, run at import
    time -- BEFORE any real-data number from main() is trusted."""
    idx = pd.date_range("2017-01-01", periods=250_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(123)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
                       index=idx)

    probe_fn = make_build_target(*r123_shared.NOVEL_PRIMARY)
    assert r123_shared.causal_truncation_probe_series(probe_fn, df), \
        "causal truncation probe failed on synthetic data for this file's own build_target " \
        "at NOVEL_PRIMARY -- a lookahead bug, must be fixed before any real-data number is trusted"

    path = probe_fn(df)
    assert np.isfinite(path).sum() > 1000, "build_target produced almost no finite output on synthetic data"

    v4_path = r123_shared.v4_target(df)
    m = np.isfinite(path) & np.isfinite(v4_path)
    assert np.all(np.abs(path[m]) <= np.abs(v4_path[m]) + 1e-9), \
        "joint-multiplier target exceeds v4's own magnitude somewhere on synthetic data"

    # A second cell (max grid corner) sanity-checks make_build_target's own
    # parameter plumbing independent of the primary cell.
    probe_fn2 = make_build_target(r123_shared.NOVEL_EPS_GRID[0], r123_shared.NOVEL_C_GRID[0])
    assert r123_shared.causal_truncation_probe_series(probe_fn2, df)


_self_test()


if __name__ == "__main__":
    main()
