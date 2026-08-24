#!/usr/bin/env python
"""R-103 NOVEL branch: ``CausalRLSKellyV4`` -- a multiplicative discount on
``kelly_regime_v4``'s UNCHANGED ``frac * scale`` product, keyed on a
CONTINUOUSLY, ONLINE-fit forecast of forward realized volatility from the
current relative signed jump (RSJ), estimated by recursive least squares
(RLS) with an exponential forgetting factor -- the direct, pre-registered
follow-on this round's own dispatch names, replacing R-102 novel's
a-priori-gridded discount coefficient with a causally-fit one.

Full citation trail, literature grounding, and the "not a duplicate of"
argument against R-102 novel/R-101/R-87/every other SIZE-axis round/the
parallel conservative branch of this same round are all already established
in ``r103_shared.py``'s own module docstring (read in full before this file
was written) and are not re-derived here. This file adapts only the
candidate construction and the promotion-bar driver from
``r102_novel_signed_jump_discount.py``, whose Step-0/causal-probe/
promotion-bar/``main()`` shape it mirrors deliberately.

=====================================================================
PRE-REGISTRATION (frozen before any real-data RSJ, forecast, discount, or
backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence): if BTC's forward realized volatility is
   predictable from its current relative signed jump (Patton & Sheppard
   2015's asymmetric-persistence claim, already this round's shared
   grounding), then a CONTINUOUSLY re-estimated causal regression of that
   relationship -- updated by a small step at every bar whose forward label
   has just resolved, via online RLS with an exponential forgetting factor,
   rather than jumping between periodic point-in-time fits -- should track
   genuine regime drift in the RSJ/vol relationship more smoothly than
   either R-102 novel's fixed a-priori coefficient or the conservative
   branch's periodic batch refit, without touching v4's validated
   vote/scale architecture at all.

2. CONSTRUCTION (exact, applied on top of v4's own UNCHANGED
   ``frac * scale``, never modifying either factor; matches this round's
   dispatch verbatim):

       raw[t]      = v4_raw_desired(df)[t]                      # frac*scale, UNCHANGED
       rsj[t]      = relative_signed_jump(df)[t]                 # r103_shared (re-exported from r102_shared), causal
       vol[t]      = v4_symmetric_vol(df)[t]                      # r103_shared (re-exported), causal
       forecast, theta0, theta1 = causal_online_rls_forecast(
           rsj, vol, horizon_bars=1440, forgetting=FORGETTING, min_updates=2000)
       z[t]        = causal_ewm_zscore(forecast, span=25920)[t]   # ~90-day EWM standardization
       disc[t]     = danger_discount(z, k, floor)
       desired[t]  = raw[t] * disc[t]
       target[t]   = apply_deadband(desired)                       # v4's own deadband, AFTER discount

   ``horizon_bars=1440`` (5 calendar days, matched to the conservative
   branch for direct comparability) and ``min_updates=2000`` (roughly a
   week of continuously-resolving labels before trusting theta) are FIXED,
   not swept -- both are named, structural choices about the forecasting
   regression's own definition, not free parameters this branch's Step-0
   grid is meant to search.

   FORGETTING-FACTOR DERIVATION (computed, not guessed): RLS updates at
   essentially every bar once warmed (the label resolves continuously, not
   periodically -- unlike the conservative branch's periodic refit), so a
   per-bar forgetting factor must correspond to a sensible MEMORY HORIZON
   in real time, not an arbitrary small decimal. Pre-registered from a
   target half-life of 2 years (comparable in spirit to the conservative
   branch's expanding-from-2017 window, but with smooth decay instead of a
   hard cutoff):

       half_life_bars = 2 * 365.25 * 288 = 210384.0
       FORGETTING     = 0.5 ** (1 / half_life_bars)
                       = 0.999996705329118551   (18 significant digits, round-trips)

   This is a deliberate, argued-for SCOPE decision, named here rather than
   swept: this project's own standing convention (see ``kelly_regime_ev``'s
   derived no-trade band) strongly prefers a derived constant over a tuned
   one, and sweeping estimator hyperparameters on top of the discount grid
   would multiply trials without testing this round's actual question (is
   a causally-fit, continuously-updating weight better than an a-priori
   one?). ``Z_SPAN = 25920`` (90 days at ``BARS_PER_DAY = 288``) is fixed
   by the same logic, given directly by this round's dispatch as the
   forecast-standardization window.

   NaN forecast/z (early-warmup bars, before ``min_updates`` RLS updates
   have occurred or before the causal EWM z-score has its own
   ``min_periods`` history) is treated as "no information" -> discount =
   1.0 (``danger_discount``'s own convention, r103_shared), never as 0 and
   never as an error. Unlike R-102 novel, the discount does NOT hard-code
   "only fire when RSJ < 0" -- it fires whenever the fitted forecast
   predicts an ABOVE-typical forward-vol increase (``z > 0``), letting the
   causal fit's own sign decide, per ``r103_shared.danger_discount``'s own
   construction.

   SCOPE DECISION -- RLS state continuity across evaluation frames: this
   file reuses ``r102_shared``'s ``TargetStrategy``/``run_slice``/
   ``compare()`` machinery UNCHANGED (per this round's own instruction to
   adapt only the candidate construction), which hands a candidate's
   ``build_target(df)`` only ``strategy.warmup`` bars (v4's own 80-day
   default) of prefix before each evaluated period, not the full history
   since the dataset's true start. R-101's own module docstring records
   why this is not simply patched by inflating ``warmup``: the SAME
   attribute also gates ``i >= strategy.warmup`` inside
   ``tradebot.engine.run_backtest``, so a large sentinel value that buys a
   long prefix for one period silences ``on_bar`` entirely for any shorter
   one (R-101's own "0 trades, $1,000.00 unchanged" failure). For
   ``inner_train`` this is immaterial: it starts at the dataset's true
   beginning, so there is no meaningful prefix to give it either way -- the
   Step-0 table below and ``inner_train``'s own promotion-bar reading are
   therefore built from IDENTICAL RLS state. For ``inner_val``,
   ``eth_replication`` and the fee-tier/plateau cells (all of which reuse
   the ``inner_val``/``eth`` frame), the RLS state effectively RESTARTS
   from zero at that frame's own beginning (80 days before the period, per
   v4's shipped warmup) rather than carrying continuous state forward from
   2017 -- with ``min_updates=2000`` and ``horizon_bars=1440`` reached
   after roughly 3,440 bars (~12 days), well inside the 80-day prefix, so
   every scored bar in those slices still has a warmed, non-degenerate
   forecast; it is simply a forecast fit on a *local* ~80-day window, not
   the full multi-year history the "2-year half-life since inception"
   framing implies. Only the Step-0 table's own direct, top-level
   computation (over the true full pre-holdout series, exactly as
   pre-registered below) and the theta1 diagnostic drawn from it reflect
   genuine continuous-since-2017 RLS state. This is named here, before any
   number is read, as a deliberate scope decision inherited from R-101's
   own precedent, not discovered after the fact.

3. STEP-0 NON-DEGENERACY GRID AND SELECTION RULE, PRE-REGISTERED BEFORE
   ANY GRID NUMBER WAS COMPUTED (identical grid SHAPE to R-102 novel and
   IDENTICAL grid to the conservative branch, so both branches' results
   are directly comparable cell-for-cell; ``k`` re-scaled from R-102
   novel's because ``z`` is already standardized):

   Grid: ``k in {0.15, 0.30, 0.60}`` x ``floor in {0.3, 0.5, 0.7}`` (9
   cells, fixed a priori). For each cell, on the BTC INNER-TRAIN slice
   only (2017-01-01 -> 2020-12-31), compute ``forecast[t]``, ``z[t]`` and
   the resulting ``discount[t]``/``target[t]`` over the FULL pre-holdout
   BTC series (so the RLS has its natural, continuous warmup from the
   dataset's true start), then restrict the two reported statistics to
   bars whose timestamp falls inside the inner-train window:
     - ``bind_frac``  = fraction of those bars with ``discount < 0.95``
       (the mechanism is actually binding, not a no-op);
     - ``r2``         = ``r_squared(target, v4_target(df))`` restricted to
       the same bars (is the candidate's exposure path a near-exact
       rescale of v4's own, i.e. degenerate).
   A cell QUALIFIES iff ``bind_frac > 0.01`` AND ``r2 < 0.98``.

   SELECTION RULE (non-degeneracy ONLY, no performance number is
   inspected before this rule is applied): the PRIMARY cell is the
   GRID-CENTER cell, ``k=0.30, floor=0.5``, IF it qualifies. If the center
   cell does NOT qualify, the PRIMARY cell is instead the qualifying cell
   closest to the center in grid-index distance
   (``sqrt((log2(k) - log2(0.30))^2 + (floor - 0.5)^2)``), ties broken by
   smallest ``k`` then smallest ``floor`` -- copied verbatim from R-102
   novel's ``select_primary``. If NO cell qualifies at all, this file
   STOPS at Step-0: the branch's entire product is this Step-0 table,
   written up NEGATIVE / stopped-at-Step-0, and no promotion-bar code
   below is run.

4. CAUSAL TRUNCATION PROBES, run before trusting any Step-0 or
   promotion-bar number: ``r102_shared.causal_truncation_probe_series``
   applied to (a) this file's own composed ``build_target`` closure for
   the PRIMARY cell and (b) ``causal_online_rls_forecast``'s own forecast
   output, wrapped as a ``df -> np.ndarray`` closure. ``r103_shared.py``'s
   self-test already checked the RLS estimator itself on SYNTHETIC data
   with an explicit no-lookahead perturbation check; this file re-runs an
   EQUIVALENT check on REAL BTC data as independent verification, not a
   re-derivation -- perturbing a late region of the real BTC series and
   confirming zero effect on any theta value computed strictly before that
   region could enter the estimator either as a regressor or as a label
   endpoint. This is the single most safety-critical thing in this branch,
   since RLS state persists across the whole series: it is stated plainly
   below if it fails.

5. PROMOTION BAR (docs/ROUTINE.md's own bar, operationalized for this
   round exactly as R-102 novel's was -- frozen BEFORE any B1-B5 number
   below was computed): B1 (paired block-bootstrap, inner-validation, both
   markets: PASS if a positive point estimate excludes zero OR
   ``d_sharpe > +0.2``), B2 (risk-matched drawdown, diagnostic-only, never
   itself gates promotion), B3 (plateau across the other 8 grid cells'
   inner-validation numbers, directionally consistent, not an isolated
   spike), B4 (ETH same-sign falsification via ``load_eth()``), B5 (0.40%
   fee-tier survival, no sign reversal on the primary cell). PROMOTE-
   candidate only if B1, B3, B4, B5 ALL pass (B2 is diagnostic-only);
   otherwise NEGATIVE. This file evaluates only up to the inner-validation
   / ETH-replication boundary -- OOS_START (2023-01-01) is never read.

6. WHAT WOULD MAKE THIS FAIL, named now, before any real-data number
   exists: (a) a forgetting factor derived from a 2-year half-life could
   still be TOO FAST for BTC's actual regime-change frequency (theta
   whipsaws on noise, a real and nameable failure this file's own theta1
   diagnostic below is built to detect) or TOO SLOW (never meaningfully
   adapts within the ~6-year pre-holdout series, behaving like a static
   fit and gaining nothing over the conservative branch's periodic refit);
   (b) continuous per-bar updating on heavily overlapping (autocorrelated)
   forward-vol labels -- every bar's label overlaps its 1,440-bar-horizon
   neighbours almost completely -- could make the RLS estimator
   numerically unstable (a real risk with a P-matrix update at every bar)
   or, once compared on the SAME grid as the conservative branch,
   effectively indistinguishable from a periodic refit; (c) as with R-102
   novel, RSJ-forecasts-future-vol could simply not hold on BTC's own
   5-minute-bar-derived statistics (Patton & Sheppard's own result is from
   decades of DAILY equity data), in which case causal fitting inherits
   the same failure mode R-102 novel's a-priori grid hit, just estimated
   rather than guessed.

CONFIGURATIONS EVALUATED IN THIS FILE: 9 (Step-0 grid, sharing one cached
RLS/forecast/z computation over the full pre-holdout BTC series) + 6
(primary cell's full ``compare()``: 2 BTC slices x 2 markets + ETH x 2
markets) + 16 (the other 8 grid cells' inner-validation numbers, both
markets, for the B3 plateau check) + 2 (primary cell at the 0.40% fee
tier, both markets) = 33 total, IF the Step-0 gate does not stop the
branch early -- identical count formula to R-102 novel, since the grid
shape and promotion-bar structure are unchanged.

USAGE
-----
    python experiments/r103_novel_causal_rls.py
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

from experiments.r103_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    causal_ewm_zscore,
    causal_online_rls_forecast,
    causal_truncation_probe_series,
    compare,
    danger_discount,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    relative_signed_jump,
    run_slice,
    v4_raw_desired,
    v4_symmetric_vol,
    v4_target,
)

# ---------------------------------------------------------- pre-registered
HORIZON_BARS = 1440
MIN_UPDATES = 2000
HALF_LIFE_BARS = 2 * 365.25 * 288
FORGETTING = 0.5 ** (1.0 / HALF_LIFE_BARS)
Z_SPAN = 25920

GRID_K = (0.15, 0.30, 0.60)
GRID_FLOOR = (0.3, 0.5, 0.7)
CENTER_K, CENTER_FLOOR = 0.30, 0.5
BIND_FRAC_THRESH = 0.01
R2_THRESH = 0.98
FEE_TIER = 0.004
SHARPE_NOISE_FLOOR = 0.2


def hdr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The mechanism itself: a pure function of df. The expensive RLS/forecast/
# z computation is cached by frame signature (length + first/last
# timestamp) so that the many evaluations below that reuse an IDENTICAL
# frame (same market, same slice -- SPOT/FUTURES/fee-tier all see the same
# price series) trigger the underlying per-bar Python loop only ONCE per
# distinct frame, never once per grid cell or per market. This is a pure
# performance optimization: build_target(df) remains a deterministic,
# side-effect-free function of df, and the causal-truncation probes below
# (which call it on frames of DIFFERENT length/content) are unaffected.
# ==================================================================

_FORECAST_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _cache_key(df: pd.DataFrame) -> tuple:
    return (len(df), df.index[0], df.index[-1] if len(df) else None)


def _forecast_and_z(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (forecast, theta0, theta1, z), cached by frame signature."""
    key = _cache_key(df)
    cached = _FORECAST_CACHE.get(key)
    if cached is not None:
        return cached
    rsj = relative_signed_jump(df)
    vol = v4_symmetric_vol(df)
    forecast, theta0, theta1 = causal_online_rls_forecast(
        rsj, vol, horizon_bars=HORIZON_BARS, forgetting=FORGETTING, min_updates=MIN_UPDATES)
    z = causal_ewm_zscore(forecast, span=Z_SPAN)
    result = (forecast, theta0, theta1, z)
    _FORECAST_CACHE[key] = result
    return result


def make_build_target(k: float, floor: float):
    """A pure ``df -> np.ndarray`` closure: v4's own unchanged
    ``frac * scale``, discounted by the causally-fit RLS forecast's
    z-scored danger discount, then v4's own deadband applied AFTER the
    discount."""

    def build(df: pd.DataFrame) -> np.ndarray:
        raw = v4_raw_desired(df)
        _, _, _, z = _forecast_and_z(df)
        disc = danger_discount(z, k, floor)
        desired = raw * disc
        return apply_deadband(desired)

    build.__name__ = f"causal_rls_k{k:g}_floor{floor:g}"
    return build


def build_rls_forecast_only(df: pd.DataFrame) -> np.ndarray:
    """``df -> forecast``, bypassing the discount/target machinery
    entirely -- used for the causal-truncation probe on the RLS
    estimator's OWN output, per this round's dispatch."""
    forecast, _, _, _ = _forecast_and_z(df)
    return forecast


build_rls_forecast_only.__name__ = "causal_online_rls_forecast[forecast]"


# ================================================================== (2)
# Step-0 non-degeneracy grid.
# ==================================================================

def step0_grid(df: pd.DataFrame) -> tuple[list[dict], int, np.ndarray, np.ndarray]:
    """9-cell grid, computed on the FULL pre-holdout price history (so the
    RLS has its natural, continuous warmup from the dataset's true start),
    with the two reported statistics restricted to bars inside the
    inner-train window. Also returns (theta1, inner_train_mask) for the
    theta1 diagnostic, drawn from this SAME cached computation."""
    mask = np.asarray((df.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                       (df.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    raw = v4_raw_desired(df)
    _, _, theta1, z = _forecast_and_z(df)
    ctrl_target = v4_target(df)

    rows = []
    for k in GRID_K:
        for floor in GRID_FLOOR:
            disc = danger_discount(z, k, floor)
            desired = raw * disc
            target = apply_deadband(desired)
            bind_frac = float(np.mean(disc[mask] < 0.95))
            r2 = r_squared(target[mask], ctrl_target[mask])
            qualifies = (bind_frac > BIND_FRAC_THRESH) and (r2 < R2_THRESH)
            rows.append(dict(k=k, floor=floor, bind_frac=bind_frac, r2=r2,
                              qualifies=qualifies))
    return rows, n_bars, theta1, mask


def select_primary(rows: list[dict]) -> dict | None:
    """Pre-registered selection: grid-center cell if it qualifies, else
    the qualifying cell closest to the center in grid-index distance,
    ties broken by smallest k then smallest floor. None if nothing
    qualifies. Copied verbatim (formula) from R-102 novel's
    ``select_primary``."""
    qualifying = [r for r in rows if r["qualifies"]]
    if not qualifying:
        return None
    for r in rows:
        if r["k"] == CENTER_K and r["floor"] == CENTER_FLOOR and r["qualifies"]:
            return r

    def dist(r: dict) -> tuple:
        d = ((np.log2(r["k"]) - np.log2(CENTER_K)) ** 2 +
             (r["floor"] - CENTER_FLOOR) ** 2) ** 0.5
        return (d, r["k"], r["floor"])

    return sorted(qualifying, key=dist)[0]


def print_step0_table(rows: list[dict], n_bars: int) -> None:
    print(f"\nSTEP-0 GRID (inner-train slice, {INNER_TRAIN_START} -> {INNER_TRAIN_END}, "
          f"{n_bars:,} bars)")
    print(f"QUALIFY = bind_frac > {BIND_FRAC_THRESH:.0%} AND r2 < {R2_THRESH}")
    hdr_line = f"{'k':>6s} {'floor':>6s} {'bind_frac':>10s} {'r2':>8s} {'qualifies':>10s}"
    print(hdr_line)
    print("-" * len(hdr_line))
    for r in rows:
        print(f"{r['k']:6.2f} {r['floor']:6.2f} {r['bind_frac']:10.4f} {r['r2']:8.4f} "
              f"{'YES' if r['qualifies'] else 'no':>10s}")


def print_theta1_diagnostic(theta1: np.ndarray, df: pd.DataFrame,
                            inner_train_mask: np.ndarray) -> None:
    """Diagnostic only (not a gate): sign/stability of the fitted RSJ
    regression slope, over inner-train and inner-validation, drawn from
    the SAME continuous-since-2017 RLS state computed for the Step-0
    table."""
    inner_val_mask = np.asarray((df.index >= pd.Timestamp(INNER_VAL_START, tz="UTC")) &
                                (df.index <= pd.Timestamp(INNER_VAL_END, tz="UTC")))
    hdr("THETA1 (RSJ regression slope) DIAGNOSTIC -- informative, not a gate")
    for name, mask in (("inner-train", inner_train_mask), ("inner-validation", inner_val_mask)):
        t1 = theta1[mask]
        t1 = t1[np.isfinite(t1)]
        if len(t1) == 0:
            print(f"  {name:16s}: no warmed theta1 values in this window")
            continue
        frac_pos = float(np.mean(t1 > 0))
        sign_changes = int(np.sum(np.diff(np.sign(t1)) != 0))
        print(f"  {name:16s}: n={len(t1):>7,d}  mean={np.mean(t1):+.5f}  "
              f"std={np.std(t1):.5f}  min={np.min(t1):+.5f}  max={np.max(t1):+.5f}  "
              f"frac_positive={frac_pos:.3f}  sign_changes={sign_changes}")
    finite = theta1[np.isfinite(theta1)]
    if len(finite) > 1:
        step = np.abs(np.diff(finite))
        print(f"\n  per-update |theta1[t]-theta1[t-1]| over the full warmed path: "
              f"mean={np.mean(step):.3e}  median={np.median(step):.3e}  max={np.max(step):.3e}")
        print("  (with FORGETTING = %.15f, each update is a small, bounded step by "
              "construction -- a continuous, smoothly-drifting path is expected here, "
              "structurally UNLIKE the conservative branch's periodic refit, which would "
              "instead show a piecewise-constant step function jumping at each refit "
              "boundary. Whether that translates into a materially different or better "
              "forecast is a separate, empirical question the promotion bar below "
              "addresses; this diagnostic only describes the SHAPE of theta1's path, "
              "which the RLS construction guarantees to be smooth by design.)" % FORGETTING)


# ================================================================== (3)
# Causal truncation probes + independent real-data no-lookahead check.
# ==================================================================

def run_causal_probes(df: pd.DataFrame, build_primary) -> tuple[bool, bool]:
    print("\ncausal_truncation_probe_series(build_target[primary], df):")
    try:
        ok_build = causal_truncation_probe_series(build_primary, df)
        print("  PASS")
    except AssertionError as e:
        ok_build = False
        print(f"  FAIL: {e}")

    print("\ncausal_truncation_probe_series(causal_online_rls_forecast[forecast], df):")
    try:
        ok_forecast = causal_truncation_probe_series(build_rls_forecast_only, df)
        print("  PASS")
    except AssertionError as e:
        ok_forecast = False
        print(f"  FAIL: {e}")

    return ok_build, ok_forecast


def real_data_no_lookahead_check(df: pd.DataFrame) -> bool:
    """Independent verification (not a re-derivation) of r103_shared's own
    synthetic-data no-lookahead check, re-run here on REAL BTC data: a
    single late region of the price series is perturbed, and every theta
    value computed strictly before that region could enter the RLS
    estimator (either as a regressor x[s] or as a label endpoint
    log_vol[t]/log_vol[s]) must match the unperturbed run EXACTLY. This is
    the single most safety-critical check in this branch, since RLS state
    persists across the whole series -- any lookahead leak would silently
    corrupt every theta value computed afterwards."""
    _, _, theta1_orig, _ = _forecast_and_z(df)  # reuses the cached, unperturbed computation

    perturb_from = (3 * len(df)) // 4
    df2 = df.copy()
    cols = ["open", "high", "low", "close"]
    idx = df2.columns.get_indexer(cols)
    tail = df2.iloc[perturb_from:].copy()
    tail.iloc[:, idx] = tail.iloc[:, idx] * 5.0 + 1.0
    df2.iloc[perturb_from:] = tail

    rsj2 = relative_signed_jump(df2)
    vol2 = v4_symmetric_vol(df2)
    _, _, theta1_pert = causal_online_rls_forecast(
        rsj2, vol2, horizon_bars=HORIZON_BARS, forgetting=FORGETTING, min_updates=MIN_UPDATES)

    safe_upto = perturb_from - HORIZON_BARS - 1
    m = np.isfinite(theta1_orig[:safe_upto]) & np.isfinite(theta1_pert[:safe_upto])
    n_checked = int(m.sum())
    ok = n_checked > 50 and np.allclose(
        theta1_orig[:safe_upto][m], theta1_pert[:safe_upto][m], atol=1e-10, rtol=0.0)
    print(f"\nreal-data no-lookahead perturbation check (BTC, perturb_from bar {perturb_from:,} "
          f"of {len(df):,}, safe_upto={safe_upto:,}):")
    print(f"  theta1 values compared (pre-perturbation-effect region): {n_checked:,}")
    print(f"  RESULT: {'PASS -- theta1 unaffected before the perturbed region enters as a regressor or label' if ok else 'FAIL -- RLS estimator LEAKS FUTURE INFORMATION into a pre-perturbation theta value'}")
    return ok


# ================================================================== (4)
# Promotion bar: B1 (bootstrap), B2 (risk-matched dd), B3 (plateau),
# B4 (ETH), B5 (fee tier). Copied nearly verbatim from R-102 novel's
# run_promotion_bar/inner_val_rows/print_plateau_table -- only the
# candidate construction (make_build_target) and grid constants differ.
# ==================================================================

def inner_val_rows(build_fn, label: str, btc: pd.DataFrame,
                    markets=(SPOT, FUTURES)) -> list[dict]:
    ctrl = TargetStrategy(v4_target, name="kelly_regime_v4")
    cand = TargetStrategy(build_fn, name=f"r103_{label}")
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


def print_plateau_table(all_rows: dict[tuple, list[dict]]) -> None:
    hdr_line = (f"{'k':>6s} {'floor':>6s} {'market':>9s} {'dSh':>7s} {'dDD':>7s} "
                f"{'expR':>5s} {'volR':>5s} {'RM':>3s} {'dlogG':>7s} "
                f"{'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr_line)
    print("-" * len(hdr_line))
    for (k, floor), rows in all_rows.items():
        for r in rows:
            print(f"{k:6.2f} {floor:6.2f} {r['market']:>9s} {r['d_sharpe']:+7.2f} "
                  f"{r['d_dd']:+7.1f} {r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
                  f"{'Y' if r['risk_matched'] else 'n':>3s} {r['boot_d_loggrowth']:+7.3f} "
                  f"{r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
                  f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def run_promotion_bar(primary: dict, btc: pd.DataFrame, eth: pd.DataFrame,
                       step0_rows: list[dict]) -> dict:
    k, floor = primary["k"], primary["floor"]
    build_primary = make_build_target(k, floor)
    label = f"causal_rls_k{k:g}_floor{floor:g}"

    hdr(f"PROMOTION BAR -- PRIMARY CELL k={k:g}, floor={floor:g}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=label, btc=btc, eth=eth,
                   markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_rows_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_rows_primary = [r for r in rows if r["slice"] == "eth_replication"]

    # B1: positive point estimate excluding zero, OR d_sharpe > +0.2, on
    # inner-validation, BOTH markets.
    b1_cells = []
    for r in inner_val_rows_primary:
        passes = (r["excludes_zero"] and r["boot_d_loggrowth"] > 0) or (r["d_sharpe"] > SHARPE_NOISE_FLOOR)
        b1_cells.append(dict(market=r["market"], passes=passes,
                              boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                              d_sharpe=r["d_sharpe"]))
    b1_pass = all(c["passes"] for c in b1_cells)

    # B2: drawdown improvement counts only where risk_matched; diagnostic-only.
    b2_cells = []
    for r in inner_val_rows_primary:
        b2_cells.append(dict(market=r["market"], risk_matched=r["risk_matched"],
                              d_dd=r["d_dd"], voided=not r["risk_matched"]))
    b2_pass = True  # never itself blocks promotion; only voids the dd number if unmatched

    # B3: plateau -- the other 8 grid cells' inner-validation numbers, both markets.
    other_cells = [(r["k"], r["floor"]) for r in step0_rows
                   if not (r["k"] == k and r["floor"] == floor)]
    plateau_rows = {}
    for ok, ofloor in other_cells:
        bf = make_build_target(ok, ofloor)
        olabel = f"causal_rls_k{ok:g}_floor{ofloor:g}"
        plateau_rows[(ok, ofloor)] = inner_val_rows(bf, olabel, btc)
    plateau_rows[(k, floor)] = [dict(label=label, market=r["market"], d_sharpe=r["d_sharpe"],
                                      d_dd=r["d_dd"], exposure_ratio=r["exposure_ratio"],
                                      vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
                                      boot_d_loggrowth=r["boot_d_loggrowth"], boot_lo=r["boot_lo"],
                                      boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
                                 for r in inner_val_rows_primary]
    same_sign_as_primary = [r["d_sharpe"] > 0 for cell, rr in plateau_rows.items() for r in rr]
    b3_pass_directionally_consistent = (sum(same_sign_as_primary) >= len(same_sign_as_primary) / 2.0)

    # B4: ETH falsification -- same sign as BTC inner-val, both markets.
    b4_cells = []
    for r in eth_rows_primary:
        btc_match = next((c for c in inner_val_rows_primary if c["market"] == r["market"]), None)
        same_sign = (btc_match is not None and
                     np.sign(r["d_sharpe"]) == np.sign(btc_match["d_sharpe"]) and
                     r["d_sharpe"] != 0)
        b4_cells.append(dict(market=r["market"], d_sharpe=r["d_sharpe"],
                              excludes_zero=r["excludes_zero"],
                              boot_lo=r["boot_lo"], boot_hi=r["boot_hi"],
                              same_sign_as_btc=same_sign))
    b4_pass = all(c["same_sign_as_btc"] for c in b4_cells)

    # B5: fee tier -- 0.40% taker, primary cell, inner-val, both markets.
    hdr("B5 -- FEE-TIER SURVIVAL (0.40% taker), primary cell, inner-validation")
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    fee_rows = inner_val_rows(build_primary, label, btc, markets=fee_markets)
    for r in fee_rows:
        print(f"  {r['market']:>9s}  d_sharpe={r['d_sharpe']:+.3f}  "
              f"boot[{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}]  excl0={r['excludes_zero']}")
    b5_cells = []
    for r in fee_rows:
        base = next((c for c in inner_val_rows_primary if c["market"] == r["market"]), None)
        no_reversal = (base is not None and
                       not (np.sign(r["boot_d_loggrowth"]) != np.sign(base["boot_d_loggrowth"])
                            and r["boot_d_loggrowth"] != 0 and base["boot_d_loggrowth"] != 0))
        b5_cells.append(dict(market=r["market"], boot_d_loggrowth=r["boot_d_loggrowth"],
                              base_boot_d_loggrowth=base["boot_d_loggrowth"] if base else float("nan"),
                              no_reversal=no_reversal))
    b5_pass = all(c["no_reversal"] for c in b5_cells)

    all_pass = b1_pass and b2_pass and b3_pass_directionally_consistent and b4_pass and b5_pass

    return dict(
        label=label, k=k, floor=floor,
        compare_rows=rows,
        b1_cells=b1_cells, b1_pass=b1_pass,
        b2_cells=b2_cells, b2_pass=b2_pass,
        b3_plateau_rows=plateau_rows, b3_pass=b3_pass_directionally_consistent,
        b4_cells=b4_cells, b4_pass=b4_pass,
        b5_cells=b5_cells, b5_pass=b5_pass,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 16 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()

    hdr("R-103 NOVEL: CausalRLSKellyV4 -- forgetting-factor derivation")
    print(f"half_life_bars = 2 * 365.25 * 288 = {HALF_LIFE_BARS}")
    print(f"FORGETTING     = 0.5 ** (1 / half_life_bars) = {FORGETTING!r}")
    print(f"HORIZON_BARS = {HORIZON_BARS}  MIN_UPDATES = {MIN_UPDATES}  Z_SPAN = {Z_SPAN}")

    hdr("R-103 NOVEL: CausalRLSKellyV4 -- Step-0 non-degeneracy grid")
    print("mechanism: continuous online RLS forecast of forward realized vol from RSJ,")
    print("z-scored and converted to a discount that fires when the fitted forecast")
    print("predicts an above-typical forward vol increase (z>0), not hard-coded to")
    print("RSJ<0 -- the fitted regression's own sign decides.")

    btc = load_btc()
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    assert_no_holdout(btc, "main(): btc")

    step0_rows, n_bars, theta1_full, inner_train_mask = step0_grid(btc)
    print_step0_table(step0_rows, n_bars)
    print_theta1_diagnostic(theta1_full, btc, inner_train_mask)

    primary = select_primary(step0_rows)

    if primary is None:
        hdr("STEP-0 GATE: NO CELL QUALIFIES -- STOPPING HERE")
        print("No grid cell has both bind_frac > 1% and r2 < 0.98: the causally-fit RLS")
        print("discount is either a near-total no-op or a near-exact rescale of v4's own")
        print("path everywhere on the pre-registered grid. Per this file's own")
        print("pre-registration, this Step-0 table is the branch's ENTIRE product,")
        print("written up NEGATIVE / stopped-at-Step-0. No promotion-bar code is run,")
        print("and no data on/after 2023-01-01 is touched.")
        print(f"\nconfigurations evaluated: 9 (Step-0 grid only)")
        print(f"max timestamp read anywhere in this branch: {btc.index.max()}  (< {OOS_START})")
        print(f"\n[{time.time()-t0:.0f}s]")
        return dict(btc=btc, step0_rows=step0_rows, primary=None, passed_step0=False,
                    n_configs=9)

    print(f"\nPRIMARY CELL SELECTED (non-degeneracy rule only): "
          f"k={primary['k']:g}, floor={primary['floor']:g}  "
          f"(bind_frac={primary['bind_frac']:.4f}, r2={primary['r2']:.4f})")
    is_center = (primary["k"] == CENTER_K and primary["floor"] == CENTER_FLOOR)
    print(f"  selection: {'grid-center cell qualified' if is_center else 'grid-center cell did NOT qualify; nearest qualifying cell chosen'}")

    build_primary = make_build_target(primary["k"], primary["floor"])

    hdr("CAUSAL-TRUNCATION PROBES")
    probe_build_ok, probe_forecast_ok = run_causal_probes(btc, build_primary)
    probes_pass = probe_build_ok and probe_forecast_ok
    print(f"\nBOTH PROBES PASS: {probes_pass}")

    real_lookahead_ok = real_data_no_lookahead_check(btc)

    eth = load_eth()
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")
    assert_no_holdout(eth, "main(): eth")

    bar = run_promotion_bar(primary, btc, eth, step0_rows)

    hdr("B1 -- inner-validation paired block-bootstrap (log-growth), both markets")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  boot[{c['boot_lo']:+.3f},{c['boot_hi']:+.3f}]  "
              f"d_sharpe={c['d_sharpe']:+.3f}  PASS={c['passes']}")
    print(f"B1 PASS (all markets): {bar['b1_pass']}")

    hdr("B2 -- risk-matched drawdown check (VOID unless risk_matched)")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.1f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hdr("B3 -- plateau, not peak: all 9 grid cells' inner-validation numbers")
    print_plateau_table(bar["b3_plateau_rows"])
    print(f"\nB3 (directionally consistent region, not an isolated spike): {bar['b3_pass']}")

    hdr("B4 -- ETH falsification (same-sign replication)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.3f}  "
              f"boot[{c['boot_lo']:+.3f},{c['boot_hi']:+.3f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PASS: {bar['b4_pass']}")

    hdr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hdr("VERDICT")
    print(f"causal probes pass: {probes_pass}   real-data no-lookahead check: {real_lookahead_ok}")
    print(f"B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4={bar['b4_pass']}  B5={bar['b5_pass']}")
    all_applicable_pass = (probes_pass and real_lookahead_ok and
                            bar["b1_pass"] and bar["b3_pass"] and bar["b4_pass"] and bar["b5_pass"])
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")
    if not (probes_pass and real_lookahead_ok):
        print("NOTE: verdict driven (at least in part) by a causal-safety check failure, "
              "not by B1-B5 alone -- the decision rule as pre-registered treats causal "
              "safety as a prerequisite to trusting ANY B1-B5 number, consistent with "
              "docs/ROUTINE.md's own precedence (a lookahead is a bug report first).")

    n_configs = 9 + bar["n_configs_promotion_bar"]
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(9 Step-0 grid + 6 primary-cell compare() + 16 plateau (8 cells x 2 markets) "
          f"+ 2 fee-tier)")
    print(f"max timestamp read anywhere in this branch: "
          f"{max(btc.index.max(), eth.index.max())}  (< {OOS_START})")

    print(f"\n[{time.time()-t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0_rows=step0_rows, primary=primary,
                passed_step0=True, probe_build_ok=probe_build_ok,
                probe_forecast_ok=probe_forecast_ok, real_lookahead_ok=real_lookahead_ok,
                promotion_bar=bar, verdict=verdict, n_configs=n_configs)


if __name__ == "__main__":
    main()
