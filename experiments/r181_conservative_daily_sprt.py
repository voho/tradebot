#!/usr/bin/env python
"""R-181 CONSERVATIVE branch: gate `kelly_regime_v4`'s exposure-INCREASING
re-targets behind a classical Wald (1945) SPRT whose evidence unit is the
CAUSAL, ONE-DAY-LAGGED daily aggregate log return (`r181_shared`'s
`causal_daily_log_return_broadcast`), on the hypothesis (Zhang, Mykland &
Ait-Sahalia 2005; Ait-Sahalia, Mykland & Zhang 2005) that R-174's own raw
5-minute-bar SPRT was fighting market-microstructure noise it did not need
to, and that resolving the same test at daily cadence would reach a verdict
inside an episode's short (median 2.42-day) lifetime where R-174's per-bar
version could not (its own implied sample size was 1e4-1e5 BARS, two to
three orders of magnitude longer than an episode survives).

Full design, citations, non-duplication argument and the frozen, numeric
decision rule (including BOTH reachability prechecks below, run in this
exact order, each a hard STOP condition) live in `experiments/r181_direction.md`
("Frozen decision rules" -> "Conservative branch -- daily-lagged SPRT"),
written and committed by the operator BEFORE this branch was dispatched.
`experiments/r181_shared.py` (read-only, NOT edited here) provides
`run_asymmetric_gate` (R-174's own neutral gate state machine, reused
unchanged), `causal_daily_log_return_broadcast` /
`causal_daily_log_sigma_broadcast` (this round's new, self-tested daily
evidence-unit primitives), `MU1_DAILY`/`TAU_DAILY` (derived from BTC
inner-train), and `median_increase_episode_gap_days` (measures, rather than
assumes, v4's own re-target cadence).

======================================================================
HEADLINE RESULT, stated before the detail: this branch STOPS at
Reachability precheck B (Step 2 of the pre-registration) -- the daily-lagged
evidence unit's own Wald average-sample-number (ASN), computed honestly
below from real inner-train sigma and the pre-registered MU1_DAILY/TAU_DAILY,
exceeds v4's own measured 2.42-day median re-target cadence by roughly two
to three orders of magnitude at EVERY alpha in the grid -- the identical
failure shape R-174 found at 5-minute resolution, just with a smaller
exponent. Precheck A (kappa, the noise-inflation premise) PASSES easily
(kappa ~= 1.30 > 1.10), so the daily aggregate genuinely does carry less
realized-variance noise than the raw 5-minute sum -- but not remotely
enough less to close a three-orders-of-magnitude gap. Per the
pre-registration's own explicit instruction, the SPRT step_fn/episode-state
pair, the kill switches, and the compare() promotion-bar sweep are
NOT implemented -- "STOP HERE ... no SPRT code is written" (precheck A's own
wording) is treated as applying with equal force to precheck B's own,
separately-named stop condition. See main()'s printed output and this
session's final report for the actual numbers.
======================================================================

Run: `uv run python experiments/r181_conservative_daily_sprt.py` (repo root).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r181_shared import (  # noqa: E402
    ALPHA_GRID,
    BARS_PER_DAY,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    KAPPA_MIN,
    MU1_DAILY,
    OOS_START,
    TAU_DAILY,
    assert_no_holdout,
    causal_daily_log_sigma_broadcast,
    load_btc,
    median_increase_episode_gap_days,
)

MU0_DAILY = 0.0  # H0: no drift, fixed by the pre-registration's own framing
# beta (type-II error budget) is not pre-registered by r181_direction.md
# ("it does not [specify one] -- use your best-justified choice and state
# it"). Chosen: BETA = 0.10 (power = 0.90), a standard textbook default
# (e.g. Wald 1945 sec. 3.9's own worked examples use 0.05-0.10) and, on this
# project's own precedent, the exact value R-174's primary alpha (0.10) used
# for BOTH alpha and beta symmetrically. Fixed at 0.10 across the WHOLE alpha
# grid here (not re-tied to each alpha) because the pre-registration asks
# for one beta choice, disclosed once, not a second free parameter swept
# alongside alpha.
BETA_DEFAULT = 0.10


# ================================================================== (1)
# Reachability precheck A: kappa, the noise-inflation ratio.
#
#   kappa = mean_over_days[ sum_{i in day} r_i^2 ]  (5-minute bar returns)
#           -------------------------------------------------------------
#                  var[ daily close-to-close log return ]
#
# If the 5-minute process were a pure, noise-free diffusion with no serial
# correlation, the sum of squared within-day increments would be an unbiased
# (in expectation, via the quadratic-variation / Ito-isometry identity for
# mean-zero, uncorrelated increments) estimator of that same day's
# close-to-close log-return VARIANCE, so kappa ~= 1. Market-microstructure
# noise (bid/ask bounce, discreteness) inflates the numerator (every one of
# ~288 bar returns per day carries an independent noise contribution) far
# more than it inflates the denominator (only the day's own two endpoint
# noise terms enter one difference) -- Ait-Sahalia, Mykland & Zhang (2005)'s
# own load-bearing claim, and the premise precheck A tests directly on real
# data before any SPRT code is written.
# ==================================================================

def compute_kappa(df: pd.DataFrame) -> dict:
    """kappa and its two ingredients, computed on the given frame (inner-train
    BTC, per the pre-registration). Uses ALL calendar days present, including
    any partial day at either edge (negligible effect over ~1,460 inner-train
    days; the bars-per-day count is reported below for transparency rather
    than silently filtered)."""
    close = df["close"]
    bar_r = np.log(close).diff()
    day = close.index.floor("1D")

    daily_sq_sum = (bar_r ** 2).groupby(day).sum()
    numerator = float(daily_sq_sum.mean())

    daily_close = close.resample("1D").last().dropna()
    daily_log_r = np.diff(np.log(daily_close.to_numpy()))
    denominator = float(np.var(daily_log_r, ddof=1))  # sample variance

    bars_per_day_counts = bar_r.groupby(day).count()

    return dict(
        numerator=numerator, denominator=denominator,
        kappa=numerator / denominator,
        n_days_numerator=int(len(daily_sq_sum)),
        n_days_denominator=int(len(daily_log_r)),
        bars_per_day_min=int(bars_per_day_counts.min()),
        bars_per_day_median=float(bars_per_day_counts.median()),
        bars_per_day_max=int(bars_per_day_counts.max()),
    )


# ================================================================== (2)
# Reachability precheck B: the Wald (1945) average-sample-number (ASN), in
# DAYS, of the daily-lagged SPRT under H1 (drift = TAU_DAILY), compared
# against v4's own measured median increase-episode gap.
#
# Wald boundaries (Wald 1945 sec. 3.3-3.5; restated identically in Siegmund
# 1985 ch. 2, and in this project's own R-174 branch file, independently
# re-derived here rather than imported since r174_conservative_wald_sprt.py
# is a sibling round's file, not shared infrastructure):
#     a = ln((1 - beta) / alpha)     -- upper boundary (accept H1)
#     b = ln(beta / (1 - alpha))     -- lower boundary (accept H0), b < 0 < a
#
# For a Gaussian mean test (H0: mu=0, H1: mu=tau, known, shared sigma), the
# per-observation log-likelihood-ratio increment is
#     Z = tau * (x - tau/2) / sigma^2
# (linear in x), so its expectation under H1 (E[x] = tau) is EXACTLY
#     E_1[Z] = tau * (tau - tau/2) / sigma^2 = tau^2 / (2 * sigma^2)
# -- the KL divergence between N(0,sigma^2) and N(tau,sigma^2), the
# "information per observation" Wald's own theory runs on.
#
# Wald's fundamental identity of sequential analysis (E[S_N] = E[N]*E[Z] for
# an iid-increment random walk stopped at a boundary) gives the classical
# ASN-under-H1 approximation (ignoring boundary overshoot, the standard,
# conservative-in-the-relevant-direction simplification -- overshoot would
# only ADD to the true ASN, strengthening a "too slow" verdict, never
# reversing it):
#     E_1[N] ~= [ (1-beta)*a + beta*b ] / E_1[Z]
# with N measured in EVIDENCE UNITS -- here, DAYS, since the evidence unit is
# the once-per-calendar-day-lagged daily aggregate return, not a raw bar.
# ==================================================================

def _wald_boundaries(alpha: float, beta: float) -> tuple[float, float]:
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError(f"alpha, beta must be in (0,1): got {alpha}, {beta}")
    a = math.log((1.0 - beta) / alpha)
    b = math.log(beta / (1.0 - alpha))
    return a, b


def asn_under_h1_days(alpha: float, beta: float, tau_daily: float, sigma_daily: float) -> float:
    """Wald ASN under H1, in DAYS (the evidence unit here), per the algebra
    in the section-2 docstring above."""
    a, b = _wald_boundaries(alpha, beta)
    info_per_day = (tau_daily ** 2) / (2.0 * sigma_daily ** 2)
    return ((1.0 - beta) * a + beta * b) / info_per_day


def daily_sigma_representative(df: pd.DataFrame, span_days: int = 30) -> tuple[float, int]:
    """Representative daily sigma for the ASN formula's known-variance
    Gaussian model. Sigma varies over time (an EWM estimate), so a single
    SUMMARY is needed; the MEDIAN over all finite (post-warmup) values is
    used -- robust to the small number of early-warmup/high-vol-regime
    outliers a MEAN would be pulled around by, and the natural "typical
    day" reading for a reachability precheck. Returns (median, n_finite)."""
    sigma = causal_daily_log_sigma_broadcast(df, span_days=span_days)
    finite = sigma[np.isfinite(sigma)]
    return float(np.median(finite)), int(len(finite))


def compute_asn_table(alphas: tuple[float, ...], beta: float, tau_daily: float,
                      sigma_daily: float) -> list[dict]:
    rows = []
    for alpha in alphas:
        a, b = _wald_boundaries(alpha, beta)
        asn = asn_under_h1_days(alpha, beta, tau_daily, sigma_daily)
        rows.append(dict(alpha=alpha, beta=beta, A=a, B=b, asn_days=asn))
    return rows


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    # ---- kappa formula: mechanically does what it claims ----------------
    # Pure diffusion, NO microstructure noise: sum of squared within-day
    # 5-min log-returns should be an unbiased-in-expectation estimator of
    # that day's own close-to-close return variance, so kappa should land
    # close to 1 (finite-sample noise on ~1,000 days keeps this a loose
    # band, not an exact equality).
    n_days = 1000
    idx = pd.date_range("2020-01-01", periods=BARS_PER_DAY * n_days, freq="5min", tz="UTC")
    rng = np.random.default_rng(181)
    true_sigma_bar = 0.0006
    true_log = np.cumsum(rng.normal(0.0, true_sigma_bar, len(idx)))
    clean_close = np.exp(true_log) * 10_000
    df_clean = pd.DataFrame({"open": clean_close, "high": clean_close * 1.0001,
                             "low": clean_close * 0.9999, "close": clean_close,
                             "volume": 1.0}, index=idx)
    k_clean = compute_kappa(df_clean)
    assert 0.7 < k_clean["kappa"] < 1.5, (
        f"no-noise synthetic kappa should be close to 1.0, got {k_clean['kappa']:.4f}")

    # Same true process, WITH added iid microstructure noise on every observed
    # price: kappa must read MEASURABLY higher than the no-noise case (the
    # exact mechanism precheck A tests for on real data).
    noise = rng.normal(0.0, true_sigma_bar * 2.0, len(idx))
    noisy_close = np.exp(true_log + noise) * 10_000
    df_noisy = pd.DataFrame({"open": noisy_close, "high": noisy_close * 1.0002,
                             "low": noisy_close * 0.9998, "close": noisy_close,
                             "volume": 1.0}, index=idx)
    k_noisy = compute_kappa(df_noisy)
    assert k_noisy["kappa"] > 2.0 * k_clean["kappa"], (
        f"microstructure noise should measurably inflate kappa: "
        f"clean={k_clean['kappa']:.4f} noisy={k_noisy['kappa']:.4f}")

    # ---- Wald boundary algebra -------------------------------------------
    a10, b10 = _wald_boundaries(0.10, 0.10)
    assert abs(a10 - math.log(9.0)) < 1e-12, a10
    assert abs(b10 - math.log(1.0 / 9.0)) < 1e-12, b10
    assert abs(a10 + b10) < 1e-12  # alpha==beta => a == -b, Wald 1945

    # ---- E_1[Z] closed form matches the linear increment's own value at
    # r = tau (exact, since the increment is LINEAR in r and E_1[r] = tau).
    tau_toy, sigma_toy = 0.002, 0.03
    info_direct = (tau_toy ** 2) / (2.0 * sigma_toy ** 2)
    llr_increment_at_mean = tau_toy * (tau_toy - tau_toy / 2.0) / (sigma_toy ** 2)
    assert abs(info_direct - llr_increment_at_mean) < 1e-15

    # ---- ASN formula: positive, and monotonically LARGER for a SMALLER
    # alpha at fixed beta (tighter type-I budget needs more evidence).
    asn_a10 = asn_under_h1_days(0.10, BETA_DEFAULT, tau_toy, sigma_toy)
    asn_a05 = asn_under_h1_days(0.05, BETA_DEFAULT, tau_toy, sigma_toy)
    asn_a20 = asn_under_h1_days(0.20, BETA_DEFAULT, tau_toy, sigma_toy)
    assert asn_a10 > 0 and asn_a05 > 0 and asn_a20 > 0
    assert asn_a05 > asn_a10 > asn_a20, (asn_a05, asn_a10, asn_a20)

    # ---- daily_sigma_representative: runs, returns a plausible positive
    # value on a slice of real BTC inner-train data.
    btc = load_btc()
    sample = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    sigma_med, n_finite = daily_sigma_representative(sample)
    assert np.isfinite(sigma_med) and 0.0 < sigma_med < 1.0, sigma_med
    assert n_finite > 1000, n_finite

    print(f"r181_conservative_daily_sprt self-test OK. "
          f"kappa(clean synthetic)={k_clean['kappa']:.4f} "
          f"kappa(noisy synthetic)={k_noisy['kappa']:.4f} "
          f"ASN sanity(alpha=.10,.05,.20)={asn_a10:.2f},{asn_a05:.2f},{asn_a20:.2f}")


_self_test()


# ================================================================== main
def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def main() -> None:
    hr("R-181 CONSERVATIVE -- daily-lagged Wald SPRT gate on kelly_regime_v4's "
       "exposure-increasing re-targets.\nH0: daily drift=0, H1: daily "
       "drift=TAU_DAILY, evidence unit = causal 1-day-lagged daily aggregate "
       "log return.")
    print("Self-test: PASS (module failed to import otherwise; see printed "
          "line above main()'s own output).")
    print(f"MU1_DAILY = TAU_DAILY (inner-train BTC mean daily log-return) = "
          f"{MU1_DAILY:.6e}")
    print(f"ALPHA_GRID = {ALPHA_GRID}   BETA (fixed, disclosed choice) = "
          f"{BETA_DEFAULT}  (power = {1 - BETA_DEFAULT:.2f})")

    btc = load_btc()
    max_ts_seen = btc.index.max()
    assert_no_holdout(btc, "BTC full")
    btc_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(btc_train, "BTC inner-train")

    # ========================================================== STEP 1
    hr(f"STEP 1 -- Reachability precheck A: kappa (noise-inflation ratio), "
       f"BTC inner-train ({INNER_TRAIN_START}..{INNER_TRAIN_END})")
    k = compute_kappa(btc_train)
    print(f"    numerator   mean_over_days[ sum(5min log-return^2) ]      = "
          f"{k['numerator']:.6e}   ({k['n_days_numerator']} days; bars/day "
          f"min={k['bars_per_day_min']} median={k['bars_per_day_median']:.0f} "
          f"max={k['bars_per_day_max']})")
    print(f"    denominator var[ daily close-to-close log-return ] (ddof=1) = "
          f"{k['denominator']:.6e}   ({k['n_days_denominator']} day-over-day diffs)")
    print(f"    kappa = numerator / denominator = {k['kappa']:.4f}")
    print(f"    KAPPA_MIN (r181_shared) = {KAPPA_MIN}")
    kappa_pass = k["kappa"] > KAPPA_MIN
    print(f"    Precheck A: {'PASS (kappa > KAPPA_MIN)' if kappa_pass else 'FAIL (kappa <= KAPPA_MIN)'}")

    if not kappa_pass:
        hr("VERDICT")
        print(f"    STOP at Reachability precheck A. kappa={k['kappa']:.4f} <= "
              f"KAPPA_MIN={KAPPA_MIN}: the noise-inflation premise is false on "
              "this data -- the 5-minute realized-variance sum is NOT "
              "measurably inflated relative to the daily close-to-close "
              "variance, so a daily-aggregate evidence unit has no noise "
              "advantage to exploit. Per r181_direction.md's own instruction, "
              "NO SPRT code is written.")
        print("    VERDICT: NEGATIVE by construction.")
        print(f"\n    Max timestamp read anywhere in this run: {max_ts_seen}   "
              f"(OOS_START = {OOS_START}; strictly earlier: "
              f"{max_ts_seen < pd.Timestamp(OOS_START, tz='UTC')})")
        return

    # ========================================================== STEP 2
    hr("STEP 2 -- Reachability precheck B: Wald ASN (days, under H1) vs the "
       "measured median increase-episode gap")
    sigma_daily, n_finite_sigma = daily_sigma_representative(btc_train)
    print(f"    Representative daily sigma (MEDIAN of causal_daily_log_sigma_"
          f"broadcast, span_days=30, over {n_finite_sigma} finite bar-level "
          f"values on BTC inner-train) = {sigma_daily:.6e}")
    print(f"    TAU_DAILY (H1 drift) = {TAU_DAILY:.6e}   BETA = {BETA_DEFAULT}")

    asn_rows = compute_asn_table(ALPHA_GRID, BETA_DEFAULT, TAU_DAILY, sigma_daily)
    gap_days = median_increase_episode_gap_days(btc_train)
    print(f"\n    Measured median increase-episode gap (BTC inner-train, "
          f"median_increase_episode_gap_days) = {gap_days:.3f} days")
    print(f"\n    {'alpha':>6s} {'A (upper)':>10s} {'B (lower)':>10s} "
          f"{'ASN_H1 (days)':>14s} {'ASN <= gap?':>12s}")
    all_exceed = True
    for row in asn_rows:
        clears = row["asn_days"] <= gap_days
        all_exceed = all_exceed and (not clears)
        print(f"    {row['alpha']:>6.2f} {row['A']:>10.4f} {row['B']:>10.4f} "
              f"{row['asn_days']:>14.2f} {'YES' if clears else 'no':>12s}")
    best_row = min(asn_rows, key=lambda r: r["asn_days"])
    ratio_min = best_row["asn_days"] / gap_days
    print(f"\n    Smallest (ASN / measured gap) ratio across all three alpha "
          f"= {ratio_min:.1f}x over budget (best case: alpha={best_row['alpha']})")

    hr("VERDICT")
    print(f"    Precheck A: PASS (kappa={k['kappa']:.4f} > {KAPPA_MIN})")
    if all_exceed:
        print("    Precheck B: FAIL -- ASN in days exceeds the measured "
              f"{gap_days:.2f}-day median episode gap for ALL THREE alpha in "
              f"ALPHA_GRID (by {ratio_min:.0f}x even in the best case, "
              f"alpha={best_row['alpha']}). Per r181_direction.md step 2's "
              "own instruction: STOP HERE, "
              "the SPRT engine (new_episode_state/step_fn), kill switches A1/"
              "A2, and the compare() promotion-bar sweep are NOT implemented "
              "-- the daily-lagged fix shortens resolution time (precheck A "
              "confirms real, if modest, noise reduction) but nowhere near "
              "enough to resolve inside a ~2.4-day episode; the same failure "
              "shape R-174 found at 5-minute cadence, reproduced with a "
              "smaller (but still enormous) exponent at daily cadence.")
        print("\n    VERDICT: NEGATIVE by construction.")
        print("    Configs evaluated (compare() runs): 0")
        print("    Diagnostic-only evaluations (not counted as trials/configs, "
              "no real-data Sharpe/growth number comes from them): 1 kappa "
              f"computation, {len(ALPHA_GRID)} ASN evaluations (one per alpha "
              "in ALPHA_GRID), 1 median-episode-gap measurement.")
    else:
        # Not reached on the committed dataset as of this run (all three
        # alphas exceeded the gap) -- if the data or shared constants ever
        # change and at least one alpha clears, this branch's own next step
        # (building make_sprt/step_fn and wiring it into run_asymmetric_gate,
        # per r181_direction.md step 3) is NOT implemented in this file and
        # must be added deliberately, not silently assumed to already exist.
        print("    Precheck B: at least one alpha clears (ASN <= measured gap). "
              "This file, as written, stops here anyway: building the actual "
              "SPRT engine, kill switches, and promotion-bar sweep is Step 3 "
              "of r181_direction.md and is NOT implemented in this run.")
        print("\n    VERDICT: INCOMPLETE -- rerun after implementing Step 3 "
              "(not expected on the committed dataset; see the ASN table above).")

    print(f"\n    Max timestamp read anywhere in this run: {max_ts_seen}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max_ts_seen < pd.Timestamp(OOS_START, tz='UTC')})")

    hr("HOLDOUT")
    print("    Holdout consulted: NO. This script never reads a bar at or "
          "after OOS_START (2023-01-01); `load_btc` truncates before it and "
          "`assert_no_holdout` guards every load point above.")


if __name__ == "__main__":
    main()
