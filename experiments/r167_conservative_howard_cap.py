#!/usr/bin/env python
"""R-167 CONSERVATIVE branch: peeled (union-bound-over-refits) Hoeffding UCB
(Howard, Ramdas, McAuliffe & Sekhon 2021 "stitching/peeling" construction)
calibration of a multiplicative cap `lambda` on `kelly_regime_v4`'s own
SCALE output (`frac*scale`, pre-deadband), refit at GEOMETRICALLY (doubling)
spaced calendar offsets over an EXPANDING (never fixed-trailing) causal
window. Direction, citations, non-duplication argument, disclosed
simplifications, the five named failure modes, and the pre-registered
PROMOTE-CANDIDATE decision rule all live in `experiments/r167_shared.py`'s
module docstring (read there first -- this file does not repeat that
reasoning and does not edit that module, which is frozen/read-only).

THE MECHANISM, exactly (this is the CONSERVATIVE half of R-167's two
engines; the sibling NOVEL branch instead uses a fixed-fraction betting
confidence sequence -- someone else's file, not touched here):

At refit j (j = 0, 1, 2, ...), a calendar offset of
`FIRST_REFIT_DAYS * 2**j` days after the start of whatever causal history is
available, recompute `lambda` as the largest value in `r167_shared.LAMBDA_GRID`
whose peeled-Hoeffding UCB on the tail-loss indicator
`1{|exposure_prev*lambda*ret| > tau}` is <= `alpha`
(`r167_shared.peeled_hoeffding_calibrate`), evaluated at THIS refit's own
confidence budget `delta_j = 6*delta / (pi^2*(j+1)^2)` (so the total budget
spent across all refits, present and future, never exceeds `delta` --
`r167_shared.refit_delta`), over an EXPANDING window: ALL causal history
strictly before the refit point, never a fixed trailing window. Between
refits, lambda is held fixed at the value from the most recent refit; before
the first refit, lambda is left UNSET, and `r167_shared.broadcast_daily_lambda`'s
own `.ffill().fillna(1.0)` supplies the honest "no cap yet" default -- never
backfilled. The resulting daily lambda series is fed through
`r161_shared.build_capped_target` (imported unmodified via `r167_shared`),
which multiplies it onto v4's own unmodified `frac*scale` and then applies
v4's own unmodified 10% deadband -- the only difference from v4 anywhere in
this file.

This is a fundamentally different schedule shape from R-161's own periodic
batch RCPS refit (fixed calendar cadence, fixed trailing calibration
window): here the window only ever GROWS, and refits get geometrically
SPARSER over time -- both required by the peeled-Hoeffding union bound (see
`r167_shared.expanding_window_lambda_geometric`'s own docstring for why a
fixed calendar cadence was tried first, at design time, and rejected).

CONFIGS (6 total): the primary sweep (`TAU_GRID x ALPHA_GRID`, 4 configs)
uses `r167_shared`'s own `MIN_DAYS_FIRST_REFIT=20` and calls
`r167_shared.expanding_window_lambda_geometric` directly, unmodified. Two
robustness configs at (TAU_PRIMARY, ALPHA_PRIMARY) vary only the first-refit
offset (10 days / 40 days) -- since `expanding_window_lambda_geometric`
hard-codes the shared module's `MIN_DAYS_FIRST_REFIT` constant with no
override parameter, and `r167_shared.py` is frozen, this file's own
`expanding_window_lambda_geometric_local` below reimplements ONLY the
day-offset loop (identical doubling logic, `first_refit_days` parameterized
instead of a module constant) -- the calibration math itself
(`peeled_hoeffding_calibrate`, `howard_ucb_at_refit`, `refit_delta`) is
still called unmodified from `r167_shared` in every config.

DISCLOSED WORKAROUND (identical reasoning to R-161's own, read before
trusting the inner-validation numbers): `compare()`'s own `TargetStrategy`
always warms up for a fixed ~80 days, with no override exposed through
`compare()`'s public signature -- far short of what an expanding-window
calibrator could use if it saw more history. `_full_history_for` below
(copied/adapted from `r161_conservative_rcps_cap.py`) identifies which
already-loaded, already-pre-holdout-truncated full asset frame (BTC or ETH)
a `df` handed in by `compare()`/`TargetStrategy` is a slice of (matched by
one real, unperturbed bar's own close price) and calibrates from that
asset's full available history up to `df`'s own last bar, never beyond it.
This changes only how much of `df`'s own already-legitimate, already-causal
history the calibration step can see -- never what `df` itself contains,
never v4's own control path, and never a bar at or after `OOS_START`. Falls
back to `df` alone (still fully causal, just possibly warmup-starved)
whenever detection fails; the causal truncation probes below (STEP 2)
confirm this never breaks causality either way.

DECISION-RULE OPERATIONALIZATION (r167_shared's prose, made mechanical,
identical convention to R-161's own file): clause (a) -- the tail-loss
exceedance comparison -- and its ETH-replication counterpart in clause (c)
depend ONLY on price (via `calibration_frame`, which never reads a
`MarketSpec`), so they are IDENTICAL for SPOT and FUTURES_5x on the same
underlying asset by construction; reported honestly below rather than
silently computed once and duplicated. Clause (b) (the Sharpe/bootstrap/
drawdown OR-clause) and its ETH counterpart DO differ by market and are
taken directly from `compare()`'s own per-(slice,market) rows. Clause (c)
("the SAME direction of improvement on (a)+(b) reproduces... not inverted")
is read literally: BOTH (a) and (b) must hold again on the eth_replication
slice, on both markets, for the config to pass (c). Clause (c), for THIS
branch, IS the pre-registered falsification test (ETH sign-replication) --
no separate step is needed for it.

======================================================================
HEADLINE RESULT, stated before the detail: filled in after running against
real data -- see the printed VERDICT block and this session's final report
for the actual numbers.
======================================================================

Run: `. .venv/bin/activate && python experiments/r167_conservative_howard_cap.py`
(from the repo root).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r167_shared import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_PRIMARY,
    CONST_CAP_R2_THRESH,
    DELTA_TOTAL,
    ETH_SLICE_NAME,
    FUTURES,
    GATE_MIN_BINDING_FRACTION,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    LAMBDA_GRID,
    MIN_DAYS_FIRST_REFIT,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TAU_GRID,
    TAU_PRIMARY,
    assert_no_holdout,
    binding_fraction,
    build_capped_target,
    calibration_frame,
    causal_truncation_probe_series,
    compare,
    constant_cap_r2,
    days_to_first_nonzero_lambda,
    exceedance_rate,
    expanding_window_lambda_geometric,
    load_btc,
    load_eth,
    peeled_hoeffding_calibrate,
    print_rows,
    rcps_calibrate,      # R-161's own engine -- head-to-head comparison only.
    synthetic_known_tail_frame,
)

# ================================================================== (1)
# Pre-registered sweep grid -- 6 configs total, exactly as specified:
#   4 = TAU_GRID x ALPHA_GRID at FIRST_REFIT_DAYS=MIN_DAYS_FIRST_REFIT (=20)
#       [PRIMARY grid, uses r167_shared.expanding_window_lambda_geometric
#        directly, unmodified]
#   1 = robustness: faster first refit (FIRST_REFIT_DAYS=10)
#   1 = robustness: slower first refit (FIRST_REFIT_DAYS=40)
# ==================================================================
FIRST_REFIT_DAYS_FAST = 10
FIRST_REFIT_DAYS_SLOW = 40

CONFIGS: list[dict] = []
for _tau, _alpha in itertools.product(TAU_GRID, ALPHA_GRID):
    CONFIGS.append(dict(tau=_tau, alpha=_alpha, first_refit_days=MIN_DAYS_FIRST_REFIT,
                        use_shared=True, kind="primary_grid"))
CONFIGS.append(dict(tau=TAU_PRIMARY, alpha=ALPHA_PRIMARY, first_refit_days=FIRST_REFIT_DAYS_FAST,
                    use_shared=False, kind="robust_fast_first_refit"))
CONFIGS.append(dict(tau=TAU_PRIMARY, alpha=ALPHA_PRIMARY, first_refit_days=FIRST_REFIT_DAYS_SLOW,
                    use_shared=False, kind="robust_slow_first_refit"))
assert len(CONFIGS) == 6, len(CONFIGS)


def config_label(cfg: dict) -> str:
    return f"howard_t{cfg['tau']}_a{cfg['alpha']}_f{cfg['first_refit_days']}"


# ================================================================== (2)
# Local geometric-doubling scheduler, parameterized by first_refit_days --
# used ONLY for the two robustness configs. Identical doubling logic to
# r167_shared.expanding_window_lambda_geometric; the only change is that
# MIN_DAYS_FIRST_REFIT (a module constant there, with no override parameter)
# becomes a function argument here. All calibration math
# (peeled_hoeffding_calibrate / howard_ucb_at_refit / refit_delta) is still
# called UNMODIFIED from r167_shared in every case, including here.
# ==================================================================

def expanding_window_lambda_geometric_local(cal: pd.DataFrame, calibrate_fn,
                                             first_refit_days: int) -> pd.Series:
    idx = cal.index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    start, end = idx[0], idx[-1]
    values: dict[pd.Timestamp, float] = {}
    j = 0
    while True:
        offset_days = first_refit_days * (2 ** j)
        refit_day = start + pd.Timedelta(days=offset_days)
        if refit_day > end:
            break
        next_offset_days = first_refit_days * (2 ** (j + 1))
        next_refit = start + pd.Timedelta(days=next_offset_days)
        window = cal.loc[cal.index < refit_day]
        if len(window) > 0:
            lam = calibrate_fn(window["exposure_prev"].to_numpy(),
                                window["ret"].to_numpy(), j)
            applicable = idx[(idx >= refit_day) & (idx < next_refit)]
            for d in applicable:
                values[d] = lam
        j += 1
    return pd.Series(values, dtype=float).sort_index()


def make_calibrate_fn(tau: float, alpha: float):
    """Closure matching the (exposure_prev, ret, refit_index) -> lambda
    signature both geometric schedulers call, wrapping
    r167_shared.peeled_hoeffding_calibrate unmodified."""
    def _fn(exposure_prev: np.ndarray, ret: np.ndarray, refit_index: int) -> float:
        return peeled_hoeffding_calibrate(exposure_prev, ret, alpha=alpha, tau=tau,
                                          refit_index=refit_index,
                                          lambda_grid=LAMBDA_GRID, delta=DELTA_TOTAL)
    return _fn


def sparse_lambda_for(cal: pd.DataFrame, cfg: dict) -> pd.Series:
    """Dispatch to the shared (unmodified) scheduler for primary-grid
    configs, or the local first-refit-day-parameterized scheduler for the
    two robustness configs."""
    calibrate_fn = make_calibrate_fn(cfg["tau"], cfg["alpha"])
    if cfg["use_shared"]:
        return expanding_window_lambda_geometric(cal, calibrate_fn)
    return expanding_window_lambda_geometric_local(cal, calibrate_fn, cfg["first_refit_days"])


# ================================================================== (3)
# Candidate: v4's own unmodified frac*scale, capped by the geometric
# peeled-Hoeffding lambda path, then v4's own unmodified 10% deadband. See
# the module docstring's DISCLOSED WORKAROUND for why _full_history_for
# exists (copied/adapted from r161_conservative_rcps_cap.py).
# ==================================================================

_BTC_FULL: pd.DataFrame | None = None
_ETH_FULL: pd.DataFrame | None = None
_BUILD_CACHE: dict[tuple, np.ndarray] = {}


def _ensure_full_history() -> None:
    global _BTC_FULL, _ETH_FULL
    if _BTC_FULL is None:
        _BTC_FULL = load_btc()
    if _ETH_FULL is None:
        _ETH_FULL = load_eth()


def _full_history_for(df: pd.DataFrame) -> pd.DataFrame:
    """Identify which of the two already-loaded, already-pre-holdout-
    truncated full asset frames `df` is a slice of (matched by one real,
    unperturbed bar's own close price), and return that FULL frame so
    calibration can reach further back than whatever prefix compare()'s
    fixed-warmup TargetStrategy happened to hand in. Falls back to `df`
    itself if no match is found -- always still causal, see module
    docstring."""
    _ensure_full_history()
    if len(df) == 0:
        return df
    mid = len(df) // 2
    ts = df.index[mid]
    sample_close = float(df["close"].iloc[mid])
    for full in (_BTC_FULL, _ETH_FULL):
        if ts in full.index:
            fc = float(full.loc[ts, "close"])
            if abs(fc - sample_close) <= 1e-6 * max(1.0, abs(fc)):
                return full
    return df


def make_candidate_build(cfg: dict):
    """Candidate `build_target(df) -> np.ndarray`, exactly matching
    compare()'s expected candidate_build signature."""
    tau, alpha = cfg["tau"], cfg["alpha"]
    first_refit_days, use_shared = cfg["first_refit_days"], cfg["use_shared"]

    def build(df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros(0)
        key = (tau, alpha, first_refit_days, use_shared,
              df.index[0].value, df.index[-1].value, len(df))
        cached = _BUILD_CACHE.get(key)
        if cached is not None:
            return cached
        full = _full_history_for(df)
        cutoff = df.index[-1]
        full_upto = full.loc[:cutoff]
        cal = calibration_frame(full_upto)
        sparse = sparse_lambda_for(cal, cfg)
        target = build_capped_target(df, sparse)
        _BUILD_CACHE[key] = target
        return target

    build.__name__ = config_label(cfg)
    return build


# ================================================================== (4)
# Direct (non-compare()-mediated) lambda/exceedance helpers, used for the
# A1/A2 kill switches and the pre-registered clause-(a) tail-loss
# exceedance-rate comparison -- NOT for backtesting.
# ==================================================================

def lambda_over_window(price_df: pd.DataFrame, cfg: dict,
                       window_start=None, window_end=None):
    """Full daily lambda path (ffill+fillna(1.0), i.e. exactly what
    broadcast_daily_lambda would hand the live strategy) over price_df's own
    calibration frame, restricted to [window_start, window_end]."""
    cal_full = calibration_frame(price_df)
    sparse_lambda = sparse_lambda_for(cal_full, cfg)
    full_days = sparse_lambda.reindex(cal_full.index).ffill().fillna(1.0)
    mask = pd.Series(True, index=cal_full.index)
    tz = cal_full.index.tz
    if window_start is not None:
        mask &= cal_full.index >= pd.Timestamp(window_start, tz=tz)
    if window_end is not None:
        mask &= cal_full.index <= pd.Timestamp(window_end, tz=tz)
    return cal_full.loc[mask], full_days.loc[mask]


def exceedance_clause_a(price_df: pd.DataFrame, cfg: dict,
                        window_start=None, window_end=None) -> tuple[float, float, bool]:
    """Pre-registered clause (a): candidate's realized tail-loss exceedance
    rate at tau, using the ACTUAL per-day calibrated lambda, vs v4's own
    uncapped (lambda=1) rate, over the given window."""
    cal_w, lam_w = lambda_over_window(price_df, cfg, window_start, window_end)
    exp_w = cal_w["exposure_prev"].to_numpy()
    ret_w = cal_w["ret"].to_numpy()
    lam_arr = lam_w.to_numpy()
    if len(exp_w) == 0:
        return float("nan"), float("nan"), False
    cand = float(np.mean(np.abs(exp_w * lam_arr * ret_w) > cfg["tau"]))
    ctrl = exceedance_rate(exp_w, ret_w, 1.0, cfg["tau"])
    return cand, ctrl, bool(cand < ctrl)


def killswitch_diagnostics(price_df: pd.DataFrame, cfg: dict,
                           window_start=None, window_end=None) -> tuple[float, float]:
    """A1 (binding_fraction) / A2 (constant_cap_r2), on the FULL
    (ffill+fillna(1.0)) daily lambda path over the given window -- exactly
    what gets fed to the live strategy, warmup included."""
    _cal_w, lam_w = lambda_over_window(price_df, cfg, window_start, window_end)
    return binding_fraction(lam_w, thresh=0.98), constant_cap_r2(lam_w)


def b_ok(c: dict) -> bool:
    """Clause (b)'s OR: bootstrap CI excludes zero positively, OR
    d_sharpe clears the noise floor, OR a real risk-matched drawdown win."""
    return bool(c["d_sharpe"] >= SHARPE_NOISE_FLOOR
               or (c["excludes_zero"] and c["boot_d_loggrowth"] > 0)
               or (c["risk_matched"] and c["d_dd"] < 0))


# --------------------------------------------------------------- reporting

def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def cell(rows: list[dict], label: str, slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["label"] == label and r["slice"] == slice_name and r["market"] == market:
            return r
    return None


def main() -> None:
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-167 CONSERVATIVE -- peeled-Hoeffding-capped kelly_regime_v4: "
       "anytime-valid (Howard, Ramdas, McAuliffe\n& Sekhon 2021) union-bound "
       "calibration of a multiplicative cap on v4's own frac*scale, on a "
       "GEOMETRICALLY-\nspaced, EXPANDING (never fixed-trailing) causal "
       "window. See r167_shared.py's module docstring for\ndirection/"
       "citations/non-duplication/decision rule; this file implements only "
       "the peeled-Hoeffding /\ngeometric-refit algorithm.")
    print(f"\nCONFIGS ({len(CONFIGS)} total):")
    for cfg in CONFIGS:
        print(f"    {config_label(cfg):35s} kind={cfg['kind']}  "
              f"use_shared_scheduler={cfg['use_shared']}")

    # ========================================================== STEP 0
    hr("STEP 0 -- geometric peeled-Hoeffding schedule unit checks "
       "(tiny synthetic calibration frame)")
    idx0 = pd.date_range("2020-01-01", periods=500, freq="1D", tz="UTC")
    rng0 = np.random.default_rng(0)
    cal0 = pd.DataFrame({"exposure_prev": rng0.uniform(0.5, 1.5, len(idx0)),
                        "ret": rng0.normal(0, 0.02, len(idx0))}, index=idx0)
    calibrate_fn0 = make_calibrate_fn(tau=0.05, alpha=0.05)
    lam0_shared = expanding_window_lambda_geometric(cal0, calibrate_fn0)
    lam0_local = expanding_window_lambda_geometric_local(cal0, calibrate_fn0,
                                                          MIN_DAYS_FIRST_REFIT)
    same_as_shared = bool(len(lam0_shared) == len(lam0_local)
                          and np.allclose(lam0_shared.to_numpy(), lam0_local.to_numpy())
                          and (lam0_shared.index == lam0_local.index).all())
    check_no_early = bool((lam0_shared.index >= idx0[0]
                          + pd.Timedelta(days=MIN_DAYS_FIRST_REFIT)).all())
    check_bounds = bool(len(lam0_shared) == 0
                        or ((lam0_shared >= 0.0) & (lam0_shared <= 1.0)).all())
    check_nonempty = len(lam0_shared) > 0
    print(f"    No lambda entry before day {MIN_DAYS_FIRST_REFIT} (warmup honesty): "
          f"{check_no_early}")
    print(f"    All lambda values in [0,1]: {check_bounds}")
    print(f"    Non-empty (schedule actually produces refits): {check_nonempty} "
          f"(n_days_with_lambda={len(lam0_shared)})")
    print(f"    Local reimplementation (at first_refit_days={MIN_DAYS_FIRST_REFIT}) "
          f"matches shared r167_shared.expanding_window_lambda_geometric exactly: "
          f"{same_as_shared}")
    step0_ok = check_no_early and check_bounds and check_nonempty and same_as_shared
    print(f"    STEP 0: {'PASS' if step0_ok else 'FAIL'}")
    if not step0_ok:
        raise AssertionError("geometric peeled-Hoeffding schedule unit checks failed "
                             "-- stopping.")

    # ========================================================== STEP 1
    hr("STEP 1a -- calibration self-test on synthetic_known_tail_frame "
       f"(n=400,000, true_tail_prob=0.05, seed=161)\nat PRIMARY "
       f"(tau={TAU_PRIMARY}, alpha={ALPHA_PRIMARY}), single-shot peeled UCB "
       "at refit_index=0 (loosest/first-refit budget) -- failure mode (2), "
       "miscalibration under this project's serial correlation")
    synth = synthetic_known_tail_frame(n=400_000, true_tail_prob=0.05, seed=161)
    cal_synth = calibration_frame(synth)
    exp_synth = cal_synth["exposure_prev"].to_numpy()
    ret_synth = cal_synth["ret"].to_numpy()
    lam_synth = peeled_hoeffding_calibrate(exp_synth, ret_synth, alpha=ALPHA_PRIMARY,
                                           tau=TAU_PRIMARY, refit_index=0,
                                           lambda_grid=LAMBDA_GRID, delta=DELTA_TOTAL)
    achieved_synth = exceedance_rate(exp_synth, ret_synth, lam_synth, TAU_PRIMARY)
    print(f"    Synthetic calibration days: {len(cal_synth)}")
    print(f"    Single (refit_index=0) calibrated lambda: {lam_synth:.4f}")
    print(f"    Realized (achieved) exceedance rate at that lambda: {achieved_synth:.4f}")
    print(f"    Nominal alpha (target UCB): {ALPHA_PRIMARY:.4f}")
    print(f"    KNOWN ground-truth true_tail_prob: 0.0500")
    print(f"    |achieved - true_tail_prob| = {abs(achieved_synth - 0.05):.4f}")
    print(f"    |achieved - alpha|          = {abs(achieved_synth - ALPHA_PRIMARY):.4f}")
    calib_self_test_ok = achieved_synth <= ALPHA_PRIMARY + 0.02
    print(f"    Achieved rate at/below alpha+2pp margin: "
          f"{'PASS' if calib_self_test_ok else 'FAIL -- see writeup, not a hard stop'}")

    hr("STEP 1b -- days_to_first_nonzero_lambda, PRIMARY config, real BTC "
       "inner-train data\n(failure mode (5): head-to-head vs R-161's own "
       "PRIMARY engine, imported from r161_shared unmodified)")
    btc_full = load_btc()
    btc_train_for_step1 = btc_full.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    cal_btc_train = calibration_frame(btc_train_for_step1)
    primary_cfg = CONFIGS[0]
    assert primary_cfg["tau"] == TAU_PRIMARY and primary_cfg["alpha"] == ALPHA_PRIMARY
    lam_path_primary = sparse_lambda_for(cal_btc_train, primary_cfg)
    d2f_howard = days_to_first_nonzero_lambda(lam_path_primary, cal_btc_train.index[0])
    print(f"    BTC inner-train calibration days available: {len(cal_btc_train)} "
          f"({cal_btc_train.index[0]} -> {cal_btc_train.index[-1]})")
    print(f"    R-167 CONSERVATIVE (peeled-Hoeffding, geometric refit) "
          f"days_to_first_nonzero_lambda: {d2f_howard}")
    print("    (This may legitimately be inf at PRIMARY alpha=0.05 -- a design-time "
          "check already found\n     this per r167_shared.py's own docstring; "
          "reported honestly either way, not a hard stop.)")

    exp_bt = cal_btc_train["exposure_prev"].to_numpy()
    ret_bt = cal_btc_train["ret"].to_numpy()
    idx_bt = cal_btc_train.index
    d2f_r161 = float("inf")
    for n in range(1, len(idx_bt) + 1):
        lam_n = rcps_calibrate(exp_bt[:n], ret_bt[:n], alpha=ALPHA_PRIMARY,
                               tau=TAU_PRIMARY, lambda_grid=LAMBDA_GRID, delta=DELTA_TOTAL)
        if lam_n > 0.0:
            d2f_r161 = float((idx_bt[n - 1] - idx_bt[0]).days)
            break
    print(f"    R-161's own PRIMARY engine (rcps_calibrate, imported unmodified, "
          f"same real BTC data,\n     same DELTA_TOTAL, growing sample -- fixed-"
          f"window analogue) days_to_first_nonzero_lambda: {d2f_r161}")
    if d2f_howard == float("inf") and d2f_r161 == float("inf"):
        fm5_verdict = "TIE (both inf on this window) -- not a clean win either way"
    elif d2f_howard < d2f_r161:
        fm5_verdict = "R-167 CONSERVATIVE binds STRICTLY EARLIER -- failure mode (5) avoided"
    elif d2f_howard == d2f_r161:
        fm5_verdict = "TIE (identical days-to-first) -- failure mode (5) not avoided"
    else:
        fm5_verdict = "R-167 CONSERVATIVE binds LATER (or not at all) -- failure mode (5) TRIPPED"
    print(f"    Failure mode (5) check: {fm5_verdict}")

    # ========================================================== STEP 2
    hr("STEP 2 -- causal truncation probes")
    eth = load_eth()
    max_ts_seen.append(btc_full.index.max())
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(btc_full, "BTC full")
    assert_no_holdout(eth, "ETH full")
    _ensure_full_history()
    btc_train = btc_full.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(btc_train, "BTC inner-train")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc_full):,} bars, "
          f"{btc_full.index[0]} -> {btc_full.index[-1]}")
    print(f"ETH (truncated < {OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    print(f"BTC inner-train slice: {len(btc_train):,} bars, "
          f"{btc_train.index[0]} -> {btc_train.index[-1]}")

    print("\n  (2a) REQUIRED probe -- shared self-test pattern: "
          "build_capped_target with a FIXED, already-computed\n       lambda "
          "path (0.7 for every calibrated day), on real BTC inner-train data.")

    def _fixed_lambda_probe(d: pd.DataFrame) -> np.ndarray:
        idx_d = calibration_frame(d).index
        lam_d = pd.Series(0.7, index=idx_d)
        return build_capped_target(d, lam_d)

    causal_ok = True
    try:
        ok_fixed = causal_truncation_probe_series(_fixed_lambda_probe, btc_train)
        print(f"       causal_truncation_probe_series(fixed-lambda build_capped_target, "
              f"btc_train): {'PASS' if ok_fixed else 'FAIL'}")
    except AssertionError as e:
        ok_fixed = False
        print(f"       FAIL ({e})")
    causal_ok = causal_ok and ok_fixed

    print("\n  (2b) REQUIRED probe -- the FULL per-config geometric peeled-"
          "Hoeffding candidate builder\n       (PRIMARY config, including the "
          "_full_history_for asset-detection workaround), on the same real "
          "data.")
    primary_build = make_candidate_build(primary_cfg)
    try:
        ok_full = causal_truncation_probe_series(primary_build, btc_train)
        print(f"       causal_truncation_probe_series(primary geometric "
              f"peeled-Hoeffding build, btc_train): {'PASS' if ok_full else 'FAIL'}")
    except AssertionError as e:
        ok_full = False
        print(f"       FAIL ({e})")
    causal_ok = causal_ok and ok_full

    print(f"\n    Causality (both probes): {'PASS' if causal_ok else 'FAIL'}")
    if not causal_ok:
        hr("VERDICT")
        print("    VERDICT: NEGATIVE (causal truncation probe FAILED -- stopping "
              "before any promotion-bar evaluation).")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    # ========================================================== STEP 3
    hr("STEP 3 -- A1/A2 kill switches & diagnostics, per config, on BTC "
       "inner_train+inner_val COMBINED\n(binding_fraction on the FULL "
       "ffill+fillna(1.0) daily lambda path -- warmup days count against "
       "binding, honestly)")
    btc_combined = btc_full.loc[INNER_TRAIN_START:INNER_VAL_END]
    print(f"\nCombined window: {btc_combined.index[0]} -> {btc_combined.index[-1]} "
          f"({len(btc_combined):,} bars)")
    print("\nNOTE: binding_fraction/constant_cap_r2 depend only on price via "
          "calibration_frame (never on\nMarketSpec/leverage/fees), so they "
          "are IDENTICAL for SPOT and FUTURES_5x on the same asset --\n"
          "reported once per config below, applying to BOTH markets.")
    print(f"\n    {'config':35s} {'binding_frac':>13s} {'>=thresh?':>10s} "
          f"{'const_cap_R2':>13s} {'near-degenerate?':>17s}")
    killswitch_results: dict[str, tuple[float, float]] = {}
    for cfg in CONFIGS:
        label = config_label(cfg)
        bf, r2 = killswitch_diagnostics(btc_combined, cfg, INNER_TRAIN_START, INNER_VAL_END)
        killswitch_results[label] = (bf, r2)
        a1_pass = bf >= GATE_MIN_BINDING_FRACTION
        degenerate = r2 >= CONST_CAP_R2_THRESH
        print(f"    {label:35s} {bf:>13.4f} {'YES' if a1_pass else 'no':>10s} "
              f"{r2:>13.4f} {'YES (mode 5)' if degenerate else 'no':>17s}")

    any_binding = any(bf >= GATE_MIN_BINDING_FRACTION for bf, _ in killswitch_results.values())
    print(f"\n    A1 (>=1 config with binding_fraction >= "
          f"{GATE_MIN_BINDING_FRACTION}, on at least one market -- both "
          f"markets\n        are identical here since calibration is "
          f"price-only): {'PASS -- at least one config binds' if any_binding else 'FAIL (TRIPPED) -- every config is inert'}")

    # ========================================================== STEP 4
    hr("STEP 4 -- main sweep: full compare() per config "
       "(inner_train / inner_val / eth_replication x SPOT/FUTURES_5x)")
    all_rows: dict[str, list[dict]] = {}
    config_cells = 0
    for cfg in CONFIGS:
        label = config_label(cfg)
        build = make_candidate_build(cfg)
        rows = compare(build, label=label, btc=btc_full, eth=eth)
        config_cells += len(rows)
        all_rows[label] = rows
        print(f"\n  -- {label} ({cfg['kind']}) --")
        print_rows(rows)

    # ========================================================== STEP 5
    hr("STEP 5 -- clause (a): tail-loss exceedance-rate comparison "
       "(candidate vs v4-uncapped), per config")
    print("\nNOTE: exceedance rates depend only on price (via calibration_frame), "
          "so, again, identical for\nSPOT and FUTURES_5x on the same asset.")
    print(f"\n    {'config':35s} {'val: cand':>10s} {'val: ctrl':>10s} "
          f"{'val a?':>7s}   {'eth: cand':>10s} {'eth: ctrl':>10s} {'eth a?':>7s}")
    clause_a_results: dict[str, dict] = {}
    for cfg in CONFIGS:
        label = config_label(cfg)
        cand_v, ctrl_v, a_val = exceedance_clause_a(btc_full, cfg,
                                                     INNER_VAL_START, INNER_VAL_END)
        cand_e, ctrl_e, a_eth = exceedance_clause_a(eth, cfg, None, None)
        clause_a_results[label] = dict(cand_v=cand_v, ctrl_v=ctrl_v, a_val=a_val,
                                       cand_e=cand_e, ctrl_e=ctrl_e, a_eth=a_eth)
        print(f"    {label:35s} {cand_v:>10.4f} {ctrl_v:>10.4f} "
              f"{'YES' if a_val else 'no':>7s}   {cand_e:>10.4f} {ctrl_e:>10.4f} "
              f"{'YES' if a_eth else 'no':>7s}")

    # ========================================================== STEP 6
    hr("STEP 6 -- PROMOTE-CANDIDATE decision rule (r167_shared.py's own "
       "pre-registration, clauses a/b/c)")
    print("\nOperationalization (see module docstring): clause (c) requires "
          "BOTH (a) and (b) to hold again on\nthe eth_replication slice, on "
          "both markets -- not merely a same-sign check on one metric. For "
          "this\nbranch, clause (c) IS the pre-registered falsification test "
          "(ETH sign-replication).")
    decision_table = []
    any_promotes = False
    for cfg in CONFIGS:
        label = config_label(cfg)
        rows = all_rows[label]
        ca = clause_a_results[label]

        val_s = cell(rows, label, "inner_val", SPOT.name)
        val_f = cell(rows, label, "inner_val", FUTURES.name)
        eth_s = cell(rows, label, ETH_SLICE_NAME, SPOT.name)
        eth_f = cell(rows, label, ETH_SLICE_NAME, FUTURES.name)

        b_val_s, b_val_f = b_ok(val_s), b_ok(val_f)
        ab_both_markets = ca["a_val"] and b_val_s and b_val_f

        b_eth_s, b_eth_f = b_ok(eth_s), b_ok(eth_f)
        c_both_markets = ca["a_eth"] and b_eth_s and b_eth_f

        promote = ab_both_markets and c_both_markets
        any_promotes = any_promotes or promote

        decision_table.append(dict(label=label, cfg=cfg, promote=promote,
                                   ab_both_markets=ab_both_markets,
                                   c_both_markets=c_both_markets))

        print(f"\n    {label} ({cfg['kind']}):")
        print(f"      (a) inner_val exceedance strictly lower: {ca['a_val']}  "
              f"(cand={ca['cand_v']:.4f} vs ctrl={ca['ctrl_v']:.4f})")
        print(f"      (b) spot     inner_val  dSharpe={val_s['d_sharpe']:+.3f}  "
              f"boot=[{val_s['boot_lo']:+.4f},{val_s['boot_hi']:+.4f}]  "
              f"dDD={val_s['d_dd']:+.2f}  risk_matched={val_s['risk_matched']}  "
              f"b_ok={b_val_s}")
        print(f"      (b) futures  inner_val  dSharpe={val_f['d_sharpe']:+.3f}  "
              f"boot=[{val_f['boot_lo']:+.4f},{val_f['boot_hi']:+.4f}]  "
              f"dDD={val_f['d_dd']:+.2f}  risk_matched={val_f['risk_matched']}  "
              f"b_ok={b_val_f}")
        print(f"      (a)+(b) both markets on inner_val: {ab_both_markets}")
        print(f"      (a) eth_replication exceedance strictly lower: {ca['a_eth']}  "
              f"(cand={ca['cand_e']:.4f} vs ctrl={ca['ctrl_e']:.4f})")
        print(f"      (b) spot     eth_repl   dSharpe={eth_s['d_sharpe']:+.3f}  "
              f"boot point={eth_s['boot_d_loggrowth']:+.4f}  b_ok={b_eth_s}")
        print(f"      (b) futures  eth_repl   dSharpe={eth_f['d_sharpe']:+.3f}  "
              f"boot point={eth_f['boot_d_loggrowth']:+.4f}  b_ok={b_eth_f}")
        print(f"      (c) [falsification test] (a)+(b) reproduce on ETH, "
              f"both markets: {c_both_markets}")
        print(f"      PROMOTE-CANDIDATE ((a)+(b) on inner_val AND (c) on ETH, "
              f"both markets): {'PASS' if promote else 'fail'}")

    # ========================================================== STEP 7
    hr("STEP 7 -- configuration count")
    print(f"    Distinct (tau, alpha, first_refit_days) configs swept: "
          f"{len(CONFIGS)}")
    print(f"    Each run through full compare() (3 slices x 2 markets = 6 cells): "
          f"{config_cells} total cells")
    print("    Plus (not counted toward the 6 trials above): geometric-schedule "
          "unit checks (1 synthetic run),\n    the calibration self-test (1 "
          "synthetic run at PRIMARY tau/alpha), the days-to-first-nonzero-lambda\n"
          "    head-to-head vs R-161's own engine (1 real-data comparison, "
          "STEP 1b), 2 causal truncation probes,\n    and the A1/A2 kill-switch/"
          "diagnostic sweep (6 configs on BTC inner_train+inner_val combined) "
          "-- none of\n    these contribute a real-data Sharpe/growth number.")

    # ========================================================== VERDICT
    hr("VERDICT")
    print(f"    Any config satisfying r167_shared.py's PROMOTE-CANDIDATE "
          f"decision rule: {'YES' if any_promotes else 'NO'}")
    if any_promotes:
        winners = [d["label"] for d in decision_table if d["promote"]]
        print(f"    Config(s) clearing the bar: {winners}")
        print("    Per the pre-registered gate, this branch would move to the "
              "holdout ONLY after the operator freezes the specific config "
              "and logs it -- NOT done automatically by this script.")
    else:
        print("    VERDICT: NEGATIVE. No config clears (a)+(b) on both markets "
              "on inner-validation AND (c) on both markets on the ETH "
              "replication test.")
    print(f"\n    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")

    hr("HOLDOUT")
    holdout_msg = ("NOT YET -- gate cleared, awaiting operator go-ahead per the routine"
                   if any_promotes else "NO")
    print(f"    Holdout consulted: {holdout_msg}")
    print("    This script never reads a bar at or after OOS_START (2023-01-01); "
          "`load_btc`/`load_eth`\n    truncate before it and `compare`/`run_slice` "
          "assert against it on every call.")


if __name__ == "__main__":
    main()
