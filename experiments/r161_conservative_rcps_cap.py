#!/usr/bin/env python
"""R-161 CONSERVATIVE branch: periodic, batch Risk-Controlling Prediction
Sets (RCPS, Bates et al. 2021) calibration of a multiplicative cap `lambda`
on `kelly_regime_v4`'s own SCALE output (`frac*scale`, pre-deadband).
Direction, citations, non-duplication argument, kill switches, and the
pre-registered decision rule all live in `experiments/r161_shared.py`'s
module docstring (read there first -- this file does not repeat that
reasoning and does not edit that module, which is frozen/read-only).

THE MECHANISM, exactly (periodic batch refit -- the CONSERVATIVE half of the
Bates et al. (2021) / Angelopoulos et al. (2024) pair; the sibling NOVEL
branch instead tracks lambda with a continuous online update):

Every `REFIT_DAYS` CALENDAR days, recompute `lambda` as the LARGEST value in
`r161_shared.LAMBDA_GRID` whose Hoeffding UCB on the tail-loss indicator
`1{|exposure_prev*lambda*ret| > tau}` is <= `alpha`
(`r161_shared.rcps_calibrate`), using only the trailing `CALIB_DAYS` days of
`r161_shared.calibration_frame(df)` STRICTLY BEFORE the refit point. Between
refits, lambda is held fixed at the value from the most recent refit; before
the first refit (fewer than `CALIB_DAYS` days of history exist yet), lambda
is left UNSET, and `r161_shared.broadcast_daily_lambda`'s own
`.ffill().fillna(1.0)` supplies the honest "no cap yet" default -- never
backfilled. The resulting daily lambda series is fed through
`r161_shared.build_capped_target`, which multiplies it onto v4's own
unmodified `frac*scale` and then applies v4's own unmodified 10% deadband --
the only difference from v4 anywhere in this file.

DISCLOSED WORKAROUND (read before trusting the inner-validation numbers):
`compare()`'s own `TargetStrategy` (r102_shared, re-exported through
r147_shared/r161_shared) always warms up for a fixed ~80 days
(`80*BARS_PER_DAY+10`), with no override exposed through `compare()`'s
public signature -- sized for v4's own longest (80-day) anchor, far short of
this branch's own CALIB_DAYS (365/730). Two ways to fix this were rejected:
(1) mutating `TargetStrategy`'s shared class-level `warmup` attribute before
calling `compare()` would also change how much history v4's own EWM-based
`scale` warms up on, non-negligibly shifting the CONTROL's own numbers
(`V4_ANCHOR_SPAN_DAYS=180` is a slow EWM, not yet converged at 80 days) --
an unacceptable confound. (2) leaving it alone would silently starve every
inner-validation cell of a fair look at this mechanism: with only an 80-day
prefix before 2021-01-01, a CALIB_DAYS=365 config would not complete its
first calibration until ~9 months into the 2-year inner-validation window,
an artifact of `compare()`'s own plumbing, not of the mechanism, and exactly
the kind of "looks right and is not" warmup handicap `window.py`'s own
docstring warns about. Instead, `_full_history_for` below identifies which
already-loaded, already-truncated-before-`OOS_START` asset (BTC or ETH) a
`df` handed in by `compare()`/`TargetStrategy` is a slice of (by matching one
real, unperturbed bar's own close price -- never a bar the candidate is not
already causally entitled to see, since both `_BTC_FULL`/`_ETH_FULL` are the
SAME pre-holdout-truncated frames `compare()` itself loads) and calibrates
from that asset's full available history UP TO `df`'s own last bar, never
beyond it. This changes only how much of `df`'s own already-legitimate,
already-causal history the calibration step can see -- never what `df`
itself contains, never v4's own control path, and never a bar at or after
`OOS_START`. Falls back to `df` alone (still fully causal, just possibly
warmup-starved) whenever detection fails, which the causal truncation
probes below confirm never breaks causality either way (STEP 2).

DECISION-RULE OPERATIONALIZATION (r161_shared's prose, made mechanical):
Clause (a) -- the tail-loss exceedance comparison -- and its ETH-replication
counterpart in clause (c) depend ONLY on price (via `calibration_frame`,
which never reads a `MarketSpec`), so they are IDENTICAL for SPOT and
FUTURES_5x on the same underlying asset by construction; this is reported
honestly below rather than silently computed once and duplicated. Clause
(b) (the Sharpe/bootstrap/drawdown OR-clause) and its ETH counterpart DO
differ by market (fees, leverage) and are taken directly from `compare()`'s
own per-(slice,market) rows. Clause (c) ("the SAME direction of improvement
on (a)+(b) reproduces... not inverted") is read literally: BOTH (a) and (b)
must hold again on the eth_replication slice, on both markets, for the
config to pass (c) -- not merely a same-sign check on one metric.

======================================================================
HEADLINE RESULT, stated before the detail: filled in by main() below after
running against real data -- see the printed VERDICT block and this
session's final report for the actual numbers.
======================================================================

Run: `. .venv/bin/activate && python experiments/r161_conservative_rcps_cap.py`
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

from experiments.r161_shared import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_PRIMARY,
    CONST_CAP_R2_THRESH,
    ETH_SLICE_NAME,
    FUTURES,
    GATE_MIN_BINDING_FRACTION,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    LAMBDA_GRID,
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
    exceedance_rate,
    load_btc,
    load_eth,
    print_rows,
    rcps_calibrate,
    synthetic_known_tail_frame,
)

# ================================================================== (1)
# Pre-registered sweep grid -- 6 configs total, exactly as specified:
#   4 = TAU_GRID x ALPHA_GRID at (CALIB_DAYS=365, REFIT_DAYS=90) [PRIMARY grid]
#   1 = robustness: longer calibration window (CALIB_DAYS=730)
#   1 = robustness: slower refit cadence (REFIT_DAYS=180)
# ==================================================================
CALIB_DAYS_PRIMARY = 365
REFIT_DAYS_PRIMARY = 90

CONFIGS: list[dict] = []
for _tau, _alpha in itertools.product(TAU_GRID, ALPHA_GRID):
    CONFIGS.append(dict(tau=_tau, alpha=_alpha, calib_days=CALIB_DAYS_PRIMARY,
                        refit_days=REFIT_DAYS_PRIMARY, kind="primary_grid"))
CONFIGS.append(dict(tau=TAU_PRIMARY, alpha=ALPHA_PRIMARY, calib_days=730,
                    refit_days=REFIT_DAYS_PRIMARY, kind="robust_long_calib"))
CONFIGS.append(dict(tau=TAU_PRIMARY, alpha=ALPHA_PRIMARY, calib_days=CALIB_DAYS_PRIMARY,
                    refit_days=180, kind="robust_slow_refit"))
assert len(CONFIGS) == 6, len(CONFIGS)


def config_label(cfg: dict) -> str:
    return f"rcps_t{cfg['tau']}_a{cfg['alpha']}_c{cfg['calib_days']}_r{cfg['refit_days']}"


# ================================================================== (2)
# The periodic RCPS refit schedule itself: a causal daily lambda series.
# ==================================================================

def periodic_rcps_lambda(cal: pd.DataFrame, tau: float, alpha: float,
                          calib_days: int, refit_days: int,
                          lambda_grid: np.ndarray = LAMBDA_GRID) -> pd.Series:
    """Periodic-batch RCPS calibration (Bates et al. 2021), this branch's own
    algorithm (vs. the sibling NOVEL branch's continuous online tracking):
    recompute lambda every `refit_days` CALENDAR days, from the trailing
    `calib_days` days of `cal` (r161_shared.calibration_frame's own daily
    (exposure_prev, ret) rows) STRICTLY BEFORE the refit point. Returns a
    SPARSE series: an entry only for days from the first completed
    calibration window onward (days before that have NO entry here --
    `r161_shared.broadcast_daily_lambda`'s own `.ffill().fillna(1.0)` is what
    supplies the pre-registered warmup default of 1.0, never done here).

    Causal by construction: the value used for any day in
    `[refit_day, next_refit_day)` is computed only from `cal` rows dated in
    `[refit_day - calib_days, refit_day)` -- strictly before `refit_day`,
    and never re-examined once computed. Whether `cal` contains MORE rows
    beyond `next_refit_day` (e.g. because the caller handed in more history
    than is strictly needed) never changes any value already assigned,
    which is exactly what the causal truncation probes in STEP 2 check.
    """
    idx = cal.index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    start, end = idx[0], idx[-1]
    first_refit = start + pd.Timedelta(days=calib_days)
    values: dict[pd.Timestamp, float] = {}
    refit_day = first_refit
    while refit_day <= end:
        window_start = refit_day - pd.Timedelta(days=calib_days)
        window = cal.loc[(cal.index >= window_start) & (cal.index < refit_day)]
        next_refit = refit_day + pd.Timedelta(days=refit_days)
        if len(window) > 0:
            lam = rcps_calibrate(window["exposure_prev"].to_numpy(),
                                 window["ret"].to_numpy(), alpha, tau, lambda_grid)
            applicable = idx[(idx >= refit_day) & (idx < next_refit)]
            for d in applicable:
                values[d] = lam
        refit_day = next_refit
    return pd.Series(values, dtype=float).sort_index()


# ================================================================== (3)
# Candidate: v4's own unmodified frac*scale, capped by the periodic RCPS
# lambda path, then v4's own unmodified 10% deadband. See the module
# docstring's DISCLOSED WORKAROUND for why `_full_history_for` exists.
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
    calibration can reach further back than whatever prefix `compare()`'s
    fixed-warmup `TargetStrategy` happened to hand in. Falls back to `df`
    itself if no match is found (e.g. a synthetic frame, or a perturbed
    probe tail) -- always still causal, see the module docstring."""
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


def make_candidate_build(tau: float, alpha: float, calib_days: int, refit_days: int):
    """Candidate `build_target(df) -> np.ndarray`, exactly matching
    `compare()`'s expected `candidate_build` signature."""

    def build(df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros(0)
        key = (tau, alpha, calib_days, refit_days,
              df.index[0].value, df.index[-1].value, len(df))
        cached = _BUILD_CACHE.get(key)
        if cached is not None:
            return cached
        full = _full_history_for(df)
        cutoff = df.index[-1]
        full_upto = full.loc[:cutoff]
        cal = calibration_frame(full_upto)
        daily_lambda = periodic_rcps_lambda(cal, tau, alpha, calib_days, refit_days)
        target = build_capped_target(df, daily_lambda)
        _BUILD_CACHE[key] = target
        return target

    build.__name__ = f"rcps_t{tau}_a{alpha}_c{calib_days}_r{refit_days}"
    return build


# ================================================================== (4)
# Direct (non-compare()-mediated) lambda/exceedance helpers, used for the
# A1/A2 kill switches and the pre-registered clause-(a) tail-loss
# exceedance-rate comparison -- NOT for backtesting.
# ==================================================================

def lambda_over_window(price_df: pd.DataFrame, tau: float, alpha: float,
                       calib_days: int, refit_days: int,
                       window_start=None, window_end=None):
    """Full daily lambda path (ffill+fillna(1.0), i.e. exactly what
    `broadcast_daily_lambda` would hand the live strategy) over `price_df`'s
    own calibration frame, restricted to [window_start, window_end]
    (inclusive; None means unbounded on that side)."""
    cal_full = calibration_frame(price_df)
    sparse_lambda = periodic_rcps_lambda(cal_full, tau, alpha, calib_days, refit_days)
    full_days = sparse_lambda.reindex(cal_full.index).ffill().fillna(1.0)
    mask = pd.Series(True, index=cal_full.index)
    tz = cal_full.index.tz
    if window_start is not None:
        mask &= cal_full.index >= pd.Timestamp(window_start, tz=tz)
    if window_end is not None:
        mask &= cal_full.index <= pd.Timestamp(window_end, tz=tz)
    return cal_full.loc[mask], full_days.loc[mask]


def exceedance_clause_a(price_df: pd.DataFrame, tau: float, alpha: float,
                        calib_days: int, refit_days: int,
                        window_start=None, window_end=None) -> tuple[float, float, bool]:
    """Pre-registered clause (a): candidate's realized tail-loss exceedance
    rate at `tau`, using the ACTUAL per-day calibrated lambda (not a single
    scalar), vs v4's own uncapped (lambda=1) rate, over the given window.
    Returns (candidate_rate, control_rate, candidate_strictly_lower)."""
    cal_w, lam_w = lambda_over_window(price_df, tau, alpha, calib_days, refit_days,
                                      window_start, window_end)
    exp_w = cal_w["exposure_prev"].to_numpy()
    ret_w = cal_w["ret"].to_numpy()
    lam_arr = lam_w.to_numpy()
    if len(exp_w) == 0:
        return float("nan"), float("nan"), False
    cand = float(np.mean(np.abs(exp_w * lam_arr * ret_w) > tau))
    ctrl = exceedance_rate(exp_w, ret_w, 1.0, tau)
    return cand, ctrl, bool(cand < ctrl)


def killswitch_diagnostics(price_df: pd.DataFrame, tau: float, alpha: float,
                           calib_days: int, refit_days: int,
                           window_start=None, window_end=None) -> tuple[float, float]:
    """A1 (binding_fraction) / A2 (constant_cap_r2), on the FULL
    (ffill+fillna(1.0)) daily lambda path over the given window -- i.e. on
    exactly what gets fed to the live strategy, warmup included."""
    _cal_w, lam_w = lambda_over_window(price_df, tau, alpha, calib_days, refit_days,
                                       window_start, window_end)
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

    hr("R-161 CONSERVATIVE -- RCPS-capped kelly_regime_v4: periodic batch "
       "Risk-Controlling Prediction Sets\n(Bates et al. 2021) calibration of "
       "a multiplicative cap on v4's own frac*scale. See r161_shared.py's\n"
       "module docstring for direction/citations/non-duplication/decision "
       "rule; this file implements only the\nperiodic-refit algorithm.")
    print(f"\nCONFIGS ({len(CONFIGS)} total):")
    for cfg in CONFIGS:
        print(f"    {config_label(cfg):35s} kind={cfg['kind']}")

    # ========================================================== STEP 0
    hr("STEP 0 -- periodic RCPS schedule unit checks (tiny synthetic calibration frame)")
    idx0 = pd.date_range("2020-01-01", periods=500, freq="1D", tz="UTC")
    rng0 = np.random.default_rng(0)
    cal0 = pd.DataFrame({"exposure_prev": rng0.uniform(0.5, 1.5, len(idx0)),
                        "ret": rng0.normal(0, 0.02, len(idx0))}, index=idx0)
    lam0 = periodic_rcps_lambda(cal0, tau=0.05, alpha=0.05, calib_days=100, refit_days=30)
    check_no_early = bool((lam0.index >= idx0[0] + pd.Timedelta(days=100)).all())
    check_bounds = bool(len(lam0) == 0 or ((lam0 >= 0.0) & (lam0 <= 1.0)).all())
    check_nonempty = len(lam0) > 0
    print(f"    No lambda entry before day 100 (warmup honesty): {check_no_early}")
    print(f"    All lambda values in [0,1]: {check_bounds}")
    print(f"    Non-empty (schedule actually produces refits): {check_nonempty} "
          f"(n_days_with_lambda={len(lam0)})")
    step0_ok = check_no_early and check_bounds and check_nonempty
    print(f"    STEP 0: {'PASS' if step0_ok else 'FAIL'}")
    if not step0_ok:
        raise AssertionError("periodic_rcps_lambda unit checks failed -- stopping.")

    # ========================================================== STEP 1
    hr("STEP 1 -- calibration self-test on synthetic_known_tail_frame "
       f"(n=400,000, true_tail_prob=0.05, seed=161)\nat PRIMARY "
       f"(tau={TAU_PRIMARY}, alpha={ALPHA_PRIMARY}) -- failure mode (2), "
       "miscalibration under this project's serial correlation")
    synth = synthetic_known_tail_frame(n=400_000, true_tail_prob=0.05, seed=161)
    cal_synth = calibration_frame(synth)
    exp_synth = cal_synth["exposure_prev"].to_numpy()
    ret_synth = cal_synth["ret"].to_numpy()
    lam_synth = rcps_calibrate(exp_synth, ret_synth, alpha=ALPHA_PRIMARY, tau=TAU_PRIMARY,
                               lambda_grid=LAMBDA_GRID)
    achieved_synth = exceedance_rate(exp_synth, ret_synth, lam_synth, TAU_PRIMARY)
    print(f"    Synthetic calibration days: {len(cal_synth)}")
    print(f"    Single global calibrated lambda: {lam_synth:.4f}")
    print(f"    Realized (achieved) exceedance rate at that lambda: {achieved_synth:.4f}")
    print(f"    Nominal alpha (target UCB): {ALPHA_PRIMARY:.4f}")
    print(f"    KNOWN ground-truth true_tail_prob: 0.0500")
    print(f"    |achieved - true_tail_prob| = {abs(achieved_synth - 0.05):.4f}")
    print(f"    |achieved - alpha|          = {abs(achieved_synth - ALPHA_PRIMARY):.4f}")
    calib_self_test_ok = achieved_synth <= ALPHA_PRIMARY + 0.02
    print(f"    Achieved rate at/below alpha+2pp margin: "
          f"{'PASS' if calib_self_test_ok else 'FAIL -- see writeup, not a hard stop'}")

    # ========================================================== STEP 2
    hr("STEP 2 -- causal truncation probes")
    btc = load_btc()
    eth = load_eth()
    max_ts_seen.append(btc.index.max())
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(btc, "BTC full")
    assert_no_holdout(eth, "ETH full")
    _ensure_full_history()
    btc_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(btc_train, "BTC inner-train")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH (truncated < {OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    print(f"BTC inner-train slice: {len(btc_train):,} bars, "
          f"{btc_train.index[0]} -> {btc_train.index[-1]}")

    print("\n  (2a) REQUIRED probe -- r161_shared's own self-test pattern: "
          "build_capped_target with a FIXED,\n       already-computed lambda "
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

    print("\n  (2b) BONUS probe (beyond the minimum) -- the FULL per-config "
          "periodic-refit candidate builder\n       (PRIMARY config, including "
          "the _full_history_for asset-detection workaround), on the same "
          "real data.")
    primary_build = make_candidate_build(TAU_PRIMARY, ALPHA_PRIMARY,
                                         CALIB_DAYS_PRIMARY, REFIT_DAYS_PRIMARY)
    try:
        ok_full = causal_truncation_probe_series(primary_build, btc_train)
        print(f"       causal_truncation_probe_series(primary periodic-RCPS "
              f"build, btc_train): {'PASS' if ok_full else 'FAIL'}")
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
    btc_combined = btc.loc[INNER_TRAIN_START:INNER_VAL_END]
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
        bf, r2 = killswitch_diagnostics(btc_combined, cfg["tau"], cfg["alpha"],
                                        cfg["calib_days"], cfg["refit_days"],
                                        INNER_TRAIN_START, INNER_VAL_END)
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
        build = make_candidate_build(cfg["tau"], cfg["alpha"], cfg["calib_days"], cfg["refit_days"])
        rows = compare(build, label=label, btc=btc, eth=eth)
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
        cand_v, ctrl_v, a_val = exceedance_clause_a(
            btc, cfg["tau"], cfg["alpha"], cfg["calib_days"], cfg["refit_days"],
            INNER_VAL_START, INNER_VAL_END)
        cand_e, ctrl_e, a_eth = exceedance_clause_a(
            eth, cfg["tau"], cfg["alpha"], cfg["calib_days"], cfg["refit_days"],
            None, None)
        clause_a_results[label] = dict(cand_v=cand_v, ctrl_v=ctrl_v, a_val=a_val,
                                       cand_e=cand_e, ctrl_e=ctrl_e, a_eth=a_eth)
        print(f"    {label:35s} {cand_v:>10.4f} {ctrl_v:>10.4f} "
              f"{'YES' if a_val else 'no':>7s}   {cand_e:>10.4f} {ctrl_e:>10.4f} "
              f"{'YES' if a_eth else 'no':>7s}")

    # ========================================================== STEP 6
    hr("STEP 6 -- PROMOTE-CANDIDATE decision rule (r161_shared.py's own "
       "pre-registration, clauses a/b/c)")
    print("\nOperationalization (see module docstring): clause (c) requires "
          "BOTH (a) and (b) to hold again on\nthe eth_replication slice, on "
          "both markets -- not merely a same-sign check on one metric.")
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
        print(f"      (c) (a)+(b) reproduce on ETH, both markets: {c_both_markets}")
        print(f"      PROMOTE-CANDIDATE ((a)+(b) on inner_val AND (c) on ETH, "
              f"both markets): {'PASS' if promote else 'fail'}")

    # ========================================================== STEP 7
    hr("STEP 7 -- configuration count")
    print(f"    Distinct (tau, alpha, calib_days, refit_days) configs swept: "
          f"{len(CONFIGS)}")
    print(f"    Each run through full compare() (3 slices x 2 markets = 6 cells): "
          f"{config_cells} total cells")
    print("    Plus: periodic-schedule unit checks (1 synthetic run), the "
          "calibration self-test\n    (1 synthetic run at PRIMARY tau/alpha), "
          "2 causal truncation probes, and the A1/A2\n    kill-switch/"
          "diagnostic sweep (6 configs on BTC inner_train+inner_val combined) "
          "-- none of these\n    contribute a real-data Sharpe/growth number, "
          "so they are not counted toward the 6 trials above.")

    # ========================================================== VERDICT
    hr("VERDICT")
    print(f"    Any config satisfying r161_shared.py's PROMOTE-CANDIDATE "
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
