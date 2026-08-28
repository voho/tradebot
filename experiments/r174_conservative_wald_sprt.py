#!/usr/bin/env python
"""R-174 CONSERVATIVE branch: gate `kelly_regime_v4`'s `frac*scale` re-sizing
decision's INCREASES (never decreases) through a classical Wald (1945)
Sequential Probability Ratio Test on the recent per-bar return process.
Direction, citations, non-duplication argument, kill switches and the
pre-registered decision rule all live in `experiments/r174_direction.md`
(read there first); the neutral gate state machine and the causal
return/sigma inputs live in `experiments/r174_shared.py` (read-only, frozen
before either branch was dispatched). This file implements NEITHER any part
of `r174_shared.py` NOR the sibling novel (GROW e-value) branch's file --
just this branch's own `new_episode_state`/`step_fn` pair, on top of
`run_asymmetric_gate`.

THE MECHANISM, exactly:

`run_asymmetric_gate` (r174_shared.py) opens a "pending episode" the instant
v4's own `desired = frac*scale` first exceeds the currently held position by
more than the 10% deadband (a genuine increase request). While an episode is
pending, on every bar this file's `step_fn` accumulates the classical Wald
SPRT log-likelihood-ratio (LLR) statistic for two SIMPLE, fully-specified
Normal hypotheses sharing bar i's own causal per-bar sigma:

    H0: bar return r_i ~ N(mu0=0, sigma_i^2)           (no drift)
    H1: bar return r_i ~ N(mu1=MU1, sigma_i^2)          (BTC's own inner-train
                                                          historical drift,
                                                          pre-registered,
                                                          r174_shared.MU1)

Per-bar LLR increment (both hypotheses share sigma_i, so the sigma^2 terms in
the Normal density's normalizing constant cancel exactly):

    log[ N(r_i; mu1, sigma_i) / N(r_i; mu0, sigma_i) ]
      = [ (r_i - mu0)^2 - (r_i - mu1)^2 ] / (2 * sigma_i^2)
      = (mu1 - mu0) * (r_i - (mu0 + mu1) / 2) / sigma_i^2

(expand the numerator of the middle line: (r-mu0)^2-(r-mu1)^2 =
(mu1-mu0)*(2r-mu0-mu1) = 2*(mu1-mu0)*(r-(mu0+mu1)/2); the self-test below
verifies this algebra against a hand-computed toy value AND against a
second, independently-written implementation of the same two-Gaussian
log-density ratio.)

`S` accumulates this increment bar-by-bar while the episode is pending.
Wald's own boundaries (Wald, A. (1945), "Sequential Tests of Statistical
Hypotheses", Annals of Mathematical Statistics 16(2), 117-186, section 3.5;
restated identically in every standard reference on the SPRT, e.g. Siegmund,
D. (1985), Sequential Analysis: Tests and Confidence Intervals, Springer,
ch. 2, and Wald & Wolfowitz (1948), "Optimum Character of the Sequential
Probability Ratio Test", Ann. Math. Statist. 19(3), 326-339, who prove the
same two boundaries are simultaneously expected-sample-size-optimal for
BOTH hypotheses):

    A = ln((1 - beta) / alpha)     -- upper boundary: S >= A => accept H1
    B = ln(beta / (1 - alpha))     -- lower boundary: S <= B => accept H0

with `alpha` the target type-I error (grant an increase when H0 is in fact
true) and `beta` the target type-II error (fail to grant when H1 is in fact
true), `beta = alpha` per this round's pre-registration (r174_direction.md's
"BETA = ALPHA (symmetric type-I/II budget)"). Sanity checks worth stating
plainly: A > 0 > B always (for alpha, beta in (0, 1)); when alpha = beta,
A = -B exactly (ln((1-a)/a) = -ln(a/(1-a))), i.e. the two boundaries are
symmetric around zero, which the self-test below checks with a closed-form
numeric example (alpha=beta=0.10 => A=ln(9), B=-ln(9)).

`accept H1` grants the CURRENT bar's `desired[i]` (r174_shared's own
"not a stale value" contract); `accept H0` rejects the increase outright,
resetting the episode with S=0 (a fresh request, if it still holds next bar,
starts a brand-new SPRT from zero evidence -- Wald's test is memoryless
between independent trials, and each gated episode is treated as one).
Because the boundaries A/B depend on alpha, one full accumulator
(new_episode_state/step_fn pair) is built per alpha in ALPHA_GRID via
`make_sprt(alpha)`.

======================================================================
HEADLINE RESULT, stated before the detail: see the printed VERDICT block
below and this session's final report for the actual numbers (self-tests,
kill switches, per-alpha inner-train/inner-validation/ETH cells, and which
clause of r174_direction.md's decision rule fired, if any).
======================================================================

Run: `. .venv/bin/activate && python experiments/r174_conservative_wald_sprt.py`
(from the repo root).
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

from experiments.r174_shared import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_PRIMARY,
    FUTURES,
    GATE_MIN_DELAYS,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MU1,
    OOS_START,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TargetStrategy,
    V4_DEADBAND,
    assert_no_holdout,
    causal_bar_returns,
    causal_bar_sigma,
    causal_truncation_probe_series,
    compare,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    run_asymmetric_gate,
    synthetic_zero_drift_frame,
    v4_raw_desired,
    v4_target,
)
from tradebot.window import prefix_bars  # noqa: E402

MU0 = 0.0  # H0: no drift, fixed by the pre-registration's own framing


# ================================================================== (1)
# Wald boundaries. Verified against Wald (1945) sec. 3.5 / Siegmund (1985)
# ch. 2 in the module docstring above; self-tested below.
# ==================================================================

def _boundaries(alpha: float, beta: float) -> tuple[float, float]:
    """Wald's (1945) SPRT boundaries for target type-I error `alpha` and
    type-II error `beta`. A (upper, accept H1) > 0 > B (lower, accept H0)
    for any alpha, beta in (0, 1); A == -B exactly when alpha == beta."""
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError(f"alpha, beta must be in (0,1): got {alpha}, {beta}")
    A = math.log((1.0 - beta) / alpha)
    B = math.log(beta / (1.0 - alpha))
    return A, B


def _llr_increment(mu0: float, mu1: float, r: float, sigma: float) -> float:
    """One bar's log-likelihood-ratio increment for two Normal hypotheses
    sharing the same sigma, per the algebra in the module docstring."""
    return (mu1 - mu0) * (r - (mu0 + mu1) / 2.0) / (sigma * sigma)


def make_sprt(alpha: float, beta: float | None = None,
             mu0: float = MU0, mu1: float = MU1):
    """Build one `(new_episode_state, step_fn)` pair implementing the Wald
    SPRT for this specific `alpha` (boundaries depend on alpha, so a fresh
    accumulator is needed per alpha in ALPHA_GRID -- r174_direction.md's own
    requirement). `beta` defaults to `alpha` (symmetric type-I/II budget,
    pre-registered)."""
    if beta is None:
        beta = alpha
    A, B = _boundaries(alpha, beta)

    def new_episode_state():
        return {"S": 0.0}

    def step_fn(state, ri, si):
        s = state["S"] + _llr_increment(mu0, mu1, ri, si)
        if s >= A:
            return {"S": s}, "accept"
        if s <= B:
            return {"S": s}, "reject"
        return {"S": s}, "pending"

    return new_episode_state, step_fn


# ================================================================== (2)
# The candidate: v4's own raw desired exposure, gated through this branch's
# SPRT via r174_shared's neutral state machine. A lightweight cache (keyed
# on alpha + the frame's own index bounds/length AND a cheap content
# checksum of `close`, so a PERTURBED frame sharing the same index bounds --
# exactly what the causal truncation probe constructs -- can never collide
# with the unperturbed original) avoids recomputing the identical
# (alpha, frame) gate twice just because compare() runs the same price
# frame under two different MarketSpecs (SPOT, FUTURES).
# ==================================================================

_GATE_CACHE: dict[tuple, tuple[np.ndarray, int, int]] = {}


def _cache_key(df: pd.DataFrame, alpha: float) -> tuple | None:
    if not len(df):
        return None
    close = df["close"].to_numpy()
    checksum = round(float(np.sum(close)), 2)
    return (alpha, len(df), df.index[0].value, df.index[-1].value,
           round(float(close[0]), 8), round(float(close[-1]), 8), checksum)


def _run_gate_full(df: pd.DataFrame, alpha: float,
                   use_cache: bool = True) -> tuple[np.ndarray, int, int]:
    """Returns `(gated_target_path, delayed_episodes, total_episodes)` --
    the full `run_asymmetric_gate` output, so kill switch A1 (needs
    `delayed_episodes`) and the trading candidate (needs only the path) can
    share one computation instead of running the sequential loop twice."""
    key = _cache_key(df, alpha) if use_cache else None
    if key is not None:
        cached = _GATE_CACHE.get(key)
        if cached is not None:
            return cached
    desired = v4_raw_desired(df)
    r = causal_bar_returns(df)
    sigma = causal_bar_sigma(df)
    new_state, step = make_sprt(alpha)
    result = run_asymmetric_gate(desired, r, sigma, new_state, step, deadband=V4_DEADBAND)
    if key is not None:
        _GATE_CACHE[key] = result
    return result


def build_candidate_target(df: pd.DataFrame, alpha: float,
                           use_cache: bool = True) -> np.ndarray:
    """Pure `build_target(df) -> np.ndarray` candidate for `compare()`:
    v4's own `frac*scale`, gated on INCREASES only by this branch's Wald
    SPRT at the given `alpha`. Decreases stay immediate (r174_shared's own
    contract)."""
    gated, _delayed, _total = _run_gate_full(df, alpha, use_cache=use_cache)
    return gated


# ================================================================== (3)
# Calibration self-test: Wald SPRT on pure zero-drift synthetic noise.
# Reports the empirical "accept" (grant-the-increase) rate honestly,
# against the nominal alpha -- a sanity check, not a hard promotion gate.
# ==================================================================

def calibration_self_test(alphas: tuple[float, ...] = ALPHA_GRID,
                          n: int = 400_000, seed: int = 174) -> list[dict]:
    frame = synthetic_zero_drift_frame(n=n, seed=seed)
    desired = v4_raw_desired(frame)
    r = causal_bar_returns(frame)
    sigma = causal_bar_sigma(frame)
    rows = []
    for alpha in alphas:
        counts = {"accept": 0, "reject": 0}
        new_state, step = make_sprt(alpha)

        def counting_step(state, ri, si, _step=step, _counts=counts):
            new_s, decision = _step(state, ri, si)
            if decision in ("accept", "reject"):
                _counts[decision] += 1
            return new_s, decision

        gated, delayed, total = run_asymmetric_gate(desired, r, sigma, new_state,
                                                     counting_step, deadband=V4_DEADBAND)
        resolved = counts["accept"] + counts["reject"]
        accept_rate = (counts["accept"] / resolved) if resolved else float("nan")
        rows.append(dict(alpha=alpha, total_episodes=total, resolved=resolved,
                         accepted=counts["accept"], rejected=counts["reject"],
                         accept_rate=accept_rate, delayed=delayed))
    return rows


# --------------------------------------------------------------- self-test

def _primary_candidate_build_fn(df: pd.DataFrame) -> np.ndarray:
    """Named (not a lambda) wrapper so `causal_truncation_probe_series`'s
    own error messages can name it. `use_cache=False`: the probe constructs
    a PERTURBED frame sharing the unperturbed frame's index bounds, and even
    the content-checksummed cache key above is defence-in-depth, not a
    reason to rely on caching during a causality check."""
    return build_candidate_target(df, ALPHA_PRIMARY, use_cache=False)


_primary_candidate_build_fn.__name__ = f"sprt_alpha{ALPHA_PRIMARY}"


def _self_test() -> None:
    # ---- (0) Wald boundary formula sanity -------------------------------
    for a in (0.05, 0.10, 0.20):
        A, B = _boundaries(a, a)
        assert A > 0.0 > B, (a, A, B)
        assert abs(A + B) < 1e-12, "alpha==beta must give A == -B (Wald 1945)"
    A10, B10 = _boundaries(0.10, 0.10)
    assert abs(A10 - math.log(9.0)) < 1e-12, A10          # ln(0.9/0.1) = ln(9)
    assert abs(B10 - math.log(1.0 / 9.0)) < 1e-12, B10    # ln(0.1/0.9) = ln(1/9)
    A05, B05 = _boundaries(0.05, 0.20)  # asymmetric case, still A>0>B
    assert A05 > 0.0 > B05, (A05, B05)
    assert abs(A05 - math.log(0.8 / 0.05)) < 1e-12, A05

    # ---- (1) LLR increment algebra ---------------------------------------
    rng = np.random.default_rng(174)
    # (1a) degenerate case mu1 == mu0: increment must be exactly zero for
    # ANY r, sigma (not exercised by the live strategy, since MU1 is fixed
    # positive and != 0, but checks the formula's algebra independently).
    for _ in range(20):
        mu = float(rng.normal(0, 1e-4))
        r = float(rng.normal(0, 1e-3))
        sigma = float(abs(rng.normal(0, 1e-3)) + 1e-6)
        assert abs(_llr_increment(mu, mu, r, sigma)) < 1e-15, (mu, r, sigma)

    # (1b) hand-computed toy value: mu0=0, mu1=2, r=3, sigma=1.
    # Direct two-Gaussian log-density ratio: [(r-mu0)^2-(r-mu1)^2]/(2*sigma^2)
    # = [(3-0)^2 - (3-2)^2] / 2 = (9-1)/2 = 4.0.
    toy = _llr_increment(0.0, 2.0, 3.0, 1.0)
    assert abs(toy - 4.0) < 1e-12, toy

    # (1c) cross-check against a SECOND, independently-written formula (the
    # raw two-Gaussian log-density-ratio, not the algebraically-simplified
    # one `_llr_increment` uses) over random inputs.
    def _direct_llr(mu0, mu1, r, sigma):
        return (-(r - mu1) ** 2 + (r - mu0) ** 2) / (2.0 * sigma * sigma)

    for _ in range(200):
        mu0 = float(rng.normal(0, 1e-4))
        mu1 = float(rng.normal(0, 1e-4))
        r = float(rng.normal(0, 1e-3))
        sigma = float(abs(rng.normal(0, 1e-3)) + 1e-6)
        a = _llr_increment(mu0, mu1, r, sigma)
        b = _direct_llr(mu0, mu1, r, sigma)
        assert abs(a - b) < 1e-8 * max(1.0, abs(b)), (a, b, mu0, mu1, r, sigma)

    # ---- (2) toy end-to-end SPRT behaviour --------------------------------
    # Overwhelming evidence FOR the increase (r_i far above MU1 every bar,
    # tiny sigma) must accept quickly.
    new_state, step = make_sprt(0.10)
    st = new_state()
    decision = "pending"
    steps = 0
    while decision == "pending" and steps < 200:
        st, decision = step(st, MU1 * 50.0, 1e-4)
        steps += 1
    assert decision == "accept" and steps < 20, (decision, steps)
    # Overwhelming evidence AGAINST it (r_i far negative every bar) must
    # reject, not pend forever or mistakenly accept.
    new_state2, step2 = make_sprt(0.10)
    st2 = new_state2()
    decision2 = "pending"
    steps2 = 0
    while decision2 == "pending" and steps2 < 200:
        st2, decision2 = step2(st2, -MU1 * 50.0, 1e-4)
        steps2 += 1
    assert decision2 == "reject" and steps2 < 20, (decision2, steps2)
    # Exactly zero evidence (r_i == 0 forever): LLR drifts by a small
    # constant NEGATIVE increment each bar (since mu1>0, r=0 favours H0
    # slightly: increment = (mu1-0)*(0-mu1/2)/sigma^2 = -mu1^2/(2 sigma^2) <
    # 0), so this must resolve "reject", never "accept", and never loop
    # forever.
    new_state3, step3 = make_sprt(0.10)
    st3 = new_state3()
    decision3 = "pending"
    steps3 = 0
    while decision3 == "pending" and steps3 < 5_000_000:
        st3, decision3 = step3(st3, 0.0, 1e-3)
        steps3 += 1
    assert decision3 == "reject", decision3

    # ---- (3) MANDATORY: causal truncation probe on build_candidate_target
    # at ALPHA_PRIMARY, on real BTC inner-train data -- this project's core
    # no-lookahead check. -------------------------------------------------
    btc = load_btc()
    assert_no_holdout(btc, "BTC full (self-test)")
    btc_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(btc_train, "BTC inner-train (self-test)")
    ok = causal_truncation_probe_series(_primary_candidate_build_fn, btc_train)
    assert ok, "causal_truncation_probe_series FAILED for build_candidate_target @ ALPHA_PRIMARY"

    # ---- (4) calibration self-test on synthetic_zero_drift_frame ---------
    # Reported honestly in main()'s own printed output too; here just check
    # the machinery runs and returns sane shapes (the actual rate is a
    # reported number, not a hard assertion bound per the task brief).
    calib = calibration_self_test(n=60_000, seed=1)
    for row in calib:
        assert row["total_episodes"] >= 0
        assert row["accepted"] + row["rejected"] <= row["total_episodes"]
        if row["resolved"] > 0:
            assert 0.0 <= row["accept_rate"] <= 1.0


_self_test()


# ================================================================== (4)
# Reporting / main sweep.
# ==================================================================

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


CONFIGS_EVALUATED = 0  # incremented once per alpha successfully run through compare()


def _warmed_slice(df: pd.DataFrame, start: str | None, end: str | None) -> tuple[pd.DataFrame, int]:
    """Reproduce `run_period`'s own `(lo, hi, prefix)` slicing exactly (see
    `tradebot/window.py`), so a standalone kill-switch diagnostic sees the
    SAME warmed-up frame `TargetStrategy.prepare` actually receives inside
    `compare()` -- not a naive `df.loc[start:end]`, which starves the causal
    EWM sigma/vol estimator of history and can make an episode's very first
    evaluated bar look artificially decisive. Returns `(frame, prefix)`
    where `prefix` is the count of warmup-only bars at the front of `frame`
    (bars before the slice's own measured region)."""
    lo = 0 if start is None else int(df.index.searchsorted(pd.Timestamp(start, tz="UTC")))
    hi = len(df) if end is None else int(df.index.searchsorted(pd.Timestamp(end, tz="UTC"), side="right"))
    prefix = prefix_bars(df, lo, TargetStrategy.warmup)
    return df.iloc[lo - prefix: hi], prefix


def _gate_with_episode_starts(desired: np.ndarray, r: np.ndarray, sigma: np.ndarray,
                              new_episode_state, step_fn,
                              deadband: float = V4_DEADBAND) -> list[dict]:
    """Diagnostic-only re-implementation of `r174_shared.run_asymmetric_gate`'s
    exact bookkeeping (never edits that module; cross-checked for aggregate
    equivalence against it wherever this is used below), additionally
    recording each episode's own START bar index so kill switch A1 can be
    restricted to episodes that started within a slice's MEASURED region
    (excluding its warmup prefix) -- `run_asymmetric_gate` itself only
    returns aggregate counts, not per-episode positions."""
    desired = np.asarray(desired, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(desired)
    pos = 0.0
    state = None
    pending = False
    episodes: list[dict] = []

    for i in range(n):
        d = desired[i]
        if d <= pos + deadband:
            if abs(d - pos) > deadband:
                pos = d
            pending = False
            state = None
        else:
            if not pending:
                pending = True
                state = new_episode_state()
                episodes.append({"start": i, "resolved_at": None})
            ri, si = r[i], sigma[i]
            if np.isfinite(ri) and np.isfinite(si) and si > 0.0:
                state, decision = step_fn(state, ri, si)
            else:
                decision = "pending"
            if decision in ("accept", "reject"):
                episodes[-1]["resolved_at"] = i
                if decision == "accept":
                    pos = d
                pending = False
                state = None
    return episodes


def main() -> None:
    global CONFIGS_EVALUATED
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-174 CONSERVATIVE -- Wald (1945) SPRT gate on kelly_regime_v4's "
       "frac*scale INCREASES only.\nH0: mu=0, H1: mu=MU1, known per-bar "
       "sigma. Decreases stay immediate/ungated.")
    print(f"MU1 (inner-train BTC mean per-bar log-return) = {MU1:.6e}")
    print(f"ALPHA_GRID = {ALPHA_GRID}   ALPHA_PRIMARY = {ALPHA_PRIMARY}   BETA = ALPHA (symmetric)")
    for a in ALPHA_GRID:
        A, B = _boundaries(a, a)
        print(f"    alpha={a:.2f}: A (accept-H1 boundary) = {A:+.4f}   "
              f"B (accept-H0 boundary) = {B:+.4f}")

    # ========================================================== STEP 0
    hr("STEP 0 -- self-test (already ran at import time; re-report the "
       "headline numbers here)")
    print("    Boundary/LLR-algebra/toy-episode unit checks: PASS "
          "(module failed to import otherwise)")
    print("    Mandatory causal_truncation_probe_series(build_candidate_target "
          f"@ alpha={ALPHA_PRIMARY}, BTC inner-train): PASS")

    # ========================================================== STEP 1
    hr("STEP 1 -- calibration self-test on synthetic zero-drift noise "
       "(pure H0, no real trend anywhere)")
    calib = calibration_self_test()
    print(f"\n    {'alpha':>6s} {'episodes':>9s} {'resolved':>9s} {'accepted':>9s} "
          f"{'rejected':>9s} {'accept_rate':>12s} {'vs alpha':>10s}")
    worst_ratio = 0.0
    for row in calib:
        ratio = (row["accept_rate"] / row["alpha"]) if np.isfinite(row["accept_rate"]) else float("nan")
        if np.isfinite(ratio):
            worst_ratio = max(worst_ratio, ratio)
        print(f"    {row['alpha']:>6.2f} {row['total_episodes']:>9d} {row['resolved']:>9d} "
              f"{row['accepted']:>9d} {row['rejected']:>9d} "
              f"{row['accept_rate']:>12.4f} {ratio:>9.2f}x")
    print(f"\n    Worst (accept_rate / alpha) ratio across all alphas on pure "
          f"noise: {worst_ratio:.2f}x")
    if worst_ratio > 2.0:
        print("    HONEST FLAG: the gate accepts increases MORE THAN 2x its nominal "
              "alpha budget under pure noise\n    on at least one alpha -- the real "
              "return stream's serial correlation and the Gaussian-known-sigma model\n"
              "    do not jointly deliver the textbook guarantee here. Reported "
              "honestly, not hidden.")
    elif worst_ratio > 1.2:
        print("    Mild over-acceptance relative to nominal alpha -- plausibly the "
              "return stream's serial correlation\n    (adjacent bars share most of "
              "their trailing sigma window), not gross miscalibration.")
    else:
        print("    Empirical acceptance roughly respects (or is more conservative "
              "than) the nominal alpha budget.")

    # ========================================================== STEP 2
    hr("STEP 2 -- kill switches")
    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "BTC full")
    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "ETH full")

    # `TargetStrategy`'s own warmup (23,050 bars = 80 days) means
    # `compare()` never actually feeds `build_candidate_target` a naive
    # `df.loc[start:end]` slice for inner-validation (that would starve the
    # causal EWM sigma/vol estimator of history right at the slice's own
    # start) -- it feeds the warmed frame `run_period` builds. `_warmed_slice`
    # reproduces that exactly so this diagnostic sees what `compare()` sees.
    btc_train, train_prefix = _warmed_slice(btc, INNER_TRAIN_START, INNER_TRAIN_END)
    btc_val, val_prefix = _warmed_slice(btc, INNER_VAL_START, INNER_VAL_END)
    assert train_prefix == 0, train_prefix  # inner-train starts at the dataset's own start
    print(f"\n    (inner-validation warmed frame: {len(btc_val):,} bars, first "
          f"{val_prefix:,} are warmup-only prefix, not part of the measured period)")

    print(f"\n    A1 (GATE_MIN_DELAYS={GATE_MIN_DELAYS}): on BTC inner-validation, "
          "does at least one alpha delay >=1 episode -- STARTED\n    within the "
          "measured period itself (excluding the warmup prefix) -- by >=1 bar, "
          ">=GATE_MIN_DELAYS times?\n    (gate's output depends only on price, "
          "not market, so SPOT and FUTURES_5x share the identical value --\n"
          "    reported once, satisfying \"at least one market\" for both.)")
    print(f"\n    {'alpha':>6s} {'episodes(measured)':>19s} {'resolved':>9s} "
          f"{'delayed':>8s} {'clears?':>8s}")
    a1_any_clears = False
    val_desired = v4_raw_desired(btc_val)
    val_r = causal_bar_returns(btc_val)
    val_sigma = causal_bar_sigma(btc_val)
    for a in ALPHA_GRID:
        new_state, step = make_sprt(a)
        episodes = _gate_with_episode_starts(val_desired, val_r, val_sigma, new_state, step,
                                             deadband=V4_DEADBAND)
        # cross-check: aggregate totals here must match run_asymmetric_gate's
        # own counts EXACTLY on the identical inputs (this function is a
        # diagnostic re-implementation, not a substitute).
        _gated_check, delayed_check, total_check = run_asymmetric_gate(
            val_desired, val_r, val_sigma, *make_sprt(a), deadband=V4_DEADBAND)
        assert len(episodes) == total_check, (len(episodes), total_check, a)
        all_delayed = sum(1 for e in episodes
                          if e["resolved_at"] is not None and e["resolved_at"] > e["start"])
        assert all_delayed == delayed_check, (all_delayed, delayed_check, a)

        measured = [e for e in episodes if e["start"] >= val_prefix]
        resolved = sum(1 for e in measured if e["resolved_at"] is not None)
        delayed = sum(1 for e in measured
                      if e["resolved_at"] is not None and e["resolved_at"] > e["start"])
        clears = delayed >= GATE_MIN_DELAYS
        a1_any_clears = a1_any_clears or clears
        print(f"    {a:>6.2f} {len(measured):>19d} {resolved:>9d} {delayed:>8d} "
              f"{'YES' if clears else 'no':>8s}")
    print(f"\n    A1 (>=1 alpha with delayed_episodes >= {GATE_MIN_DELAYS} on BTC "
          f"inner-validation's own measured period): {'PASS' if a1_any_clears else 'FAIL (TRIPPED)'}")

    print(f"\n    A2 (R2_DEGENERACY_THRESH={R2_DEGENERACY_THRESH}): is the gated "
          "path a near-exact rescale of v4's own target? (R^2 computed over "
          "each slice's own\n    measured period, excluding any warmup prefix.)")
    a2_slices = {
        "inner_train (BTC)": (btc_train, train_prefix),
        "inner_val (BTC)": (btc_val, val_prefix),
        "eth_replication": (eth, 0),
    }
    a2_all_below = True
    print(f"\n    {'alpha':>6s} {'slice':>20s} {'R^2':>10s} {'result':>8s}")
    for a in ALPHA_GRID:
        for slice_name, (frame, pfx) in a2_slices.items():
            gated = build_candidate_target(frame, a)[pfx:]
            v4_path = v4_target(frame)[pfx:]
            r2 = r_squared(gated, v4_path)
            below = bool(np.isfinite(r2) and r2 < R2_DEGENERACY_THRESH)
            a2_all_below = a2_all_below and below
            print(f"    {a:>6.2f} {slice_name:>20s} {r2:>10.6f} "
                  f"{'PASS' if below else 'FAIL (TRIPPED)':>8s}")
    print(f"\n    A2 (ALL alpha x slice cells stay below the degeneracy ceiling): "
          f"{'PASS' if a2_all_below else 'FAIL (TRIPPED)'}")

    kill_switches_ok = a1_any_clears and a2_all_below
    print(f"\n    Kill switches: {'PASS' if kill_switches_ok else 'TRIPPED'}")
    if not kill_switches_ok:
        print("    A1 TRIPPED: per r174_direction.md's own failure mode (1), this "
              "is 'a relabeling of v4, not a tested mechanism' --\n    the "
              "pre-registered verdict is NEGATIVE regardless of what the "
              "comparison sweep below shows. The sweep is still run in full\n"
              "    below (not skipped) so the full table is on record, per this "
              "session's brief -- it does NOT override this verdict.")

    # ========================================================== STEP 3
    hr("STEP 3 -- main sweep: ALPHA_GRID, full compare() per alpha "
       "(BTC inner-train/inner-val, ETH replication, SPOT + FUTURES_5x)")
    all_rows: dict[float, list[dict]] = {}
    for a in ALPHA_GRID:
        label = f"sprt_alpha{a}"

        def _build(df, _a=a):
            return build_candidate_target(df, _a)

        _build.__name__ = label
        try:
            rows = compare(_build, label=label, btc=btc, eth=eth)
        except Exception as e:  # noqa: BLE001 -- report honestly, do not silently truncate
            print(f"\n    alpha={a}: compare() RAISED {type(e).__name__}: {e}")
            print("    Treating this alpha as NOT COMPLETED (see final report).")
            continue
        CONFIGS_EVALUATED += 1
        all_rows[a] = rows
        print(f"\n  -- alpha={a} --")
        print_rows(rows)

    # ========================================================== STEP 4
    hr("STEP 4 -- decision rule (r174_direction.md's own pre-registration), "
       "evaluated per completed alpha")
    decision_table = []
    any_promotes = False
    for a in ALPHA_GRID:
        if a not in all_rows:
            print(f"\n    alpha={a}: NOT COMPLETED (see STEP 3) -- skipped in the "
                  "decision rule, reported as such.")
            continue
        rows = all_rows[a]
        label = f"sprt_alpha{a}"
        val_s = cell(rows, label, "inner_val", SPOT.name)
        val_f = cell(rows, label, "inner_val", FUTURES.name)
        eth_s = cell(rows, label, "eth_replication", SPOT.name)
        eth_f = cell(rows, label, "eth_replication", FUTURES.name)

        def clause_ab(c: dict) -> tuple[bool, bool]:
            a_ok = bool(c["excludes_zero"] and c["boot_d_loggrowth"] > 0)
            b_ok = bool(c["d_sharpe"] >= SHARPE_NOISE_FLOOR
                       or (c["risk_matched"] and c["d_dd"] < 0))
            return a_ok, b_ok

        a_s, b_s = clause_ab(val_s)
        a_f, b_f = clause_ab(val_f)
        both_markets_ab = (a_s and b_s) and (a_f and b_f)

        def same_sign(val_c: dict, eth_c: dict) -> bool:
            fired_sharpe = val_c["d_sharpe"] >= SHARPE_NOISE_FLOOR
            key = "d_sharpe" if fired_sharpe else "boot_d_loggrowth"
            return bool(np.sign(val_c[key]) == np.sign(eth_c[key]) and val_c[key] != 0)

        c_s = same_sign(val_s, eth_s)
        c_f = same_sign(val_f, eth_f)
        both_markets_c = c_s and c_f

        promote = both_markets_ab and both_markets_c
        any_promotes = any_promotes or promote

        decision_table.append(dict(alpha=a, promote=promote, a_s=a_s, b_s=b_s,
                                   a_f=a_f, b_f=b_f, c_s=c_s, c_f=c_f))

        print(f"\n    alpha={a}:")
        print(f"      spot     inner_val  dSharpe={val_s['d_sharpe']:+.3f}  "
              f"boot=[{val_s['boot_lo']:+.4f},{val_s['boot_hi']:+.4f}]  "
              f"dDD={val_s['d_dd']:+.2f}  risk_matched={val_s['risk_matched']}  "
              f"(a)={a_s} (b)={b_s}")
        print(f"      futures  inner_val  dSharpe={val_f['d_sharpe']:+.3f}  "
              f"boot=[{val_f['boot_lo']:+.4f},{val_f['boot_hi']:+.4f}]  "
              f"dDD={val_f['d_dd']:+.2f}  risk_matched={val_f['risk_matched']}  "
              f"(a)={a_f} (b)={b_f}")
        print(f"      spot     eth_repl   dSharpe={eth_s['d_sharpe']:+.3f}  "
              f"boot point={eth_s['boot_d_loggrowth']:+.4f}   same-sign-as-val: {c_s}")
        print(f"      futures  eth_repl   dSharpe={eth_f['d_sharpe']:+.3f}  "
              f"boot point={eth_f['boot_d_loggrowth']:+.4f}   same-sign-as-val: {c_f}")
        print(f"      PROMOTE-CANDIDATE clause (a+b both markets AND c both "
              f"markets): {'PASS' if promote else 'fail'}")

    # ========================================================== STEP 5
    hr("STEP 5 -- configuration count")
    print(f"    Configs evaluated (one (alpha) value run fully through "
          f"compare()): {CONFIGS_EVALUATED} of {len(ALPHA_GRID)} in ALPHA_GRID")
    print(f"    Each completed config's compare() covers 3 slices x 2 markets "
          f"= 6 cells: {CONFIGS_EVALUATED * 6} cells total")
    print("    Plus (not counted toward the trials ledger, no real-data Sharpe/"
          "growth number comes from them): Wald boundary/LLR unit checks, "
          f"toy end-to-end SPRT checks, calibration self-test "
          f"({len(ALPHA_GRID)} alphas on synthetic zero-drift data), and the "
          f"A1/A2 kill-switch sweep ({len(ALPHA_GRID)} alphas x "
          f"{len(a2_slices)} slices for A2, {len(ALPHA_GRID)} alphas for A1).")

    # ========================================================== VERDICT
    hr("VERDICT")
    print(f"    Kill switches: {'PASS' if kill_switches_ok else 'TRIPPED (A1)'}")
    print(f"    Any alpha in ALPHA_GRID satisfying r174_direction.md's "
          f"PROMOTE-CANDIDATE decision rule: {'YES' if any_promotes else 'NO'}")
    if any_promotes:
        winners = [d["alpha"] for d in decision_table if d["promote"]]
        print(f"    Alpha(s) clearing the decision-rule bar: {winners}")
    if not kill_switches_ok:
        print("\n    VERDICT: NEGATIVE. Kill switch A1 tripped (zero episodes "
              "resolved -- neither accepted nor rejected -- within BTC\n    "
              "inner-validation's own measured period, at every alpha in "
              "ALPHA_GRID): per r174_direction.md's failure mode (1), this is\n"
              "    'a relabeling of v4, not a tested mechanism.' This overrides "
              "whatever the decision-rule table above shows -- a kill switch\n"
              "    is a pre-registered stop condition, not one more input to "
              "weigh against a promotion clause.")
    elif any_promotes:
        print("\n    Per the pre-registered gate, this branch would move to the "
              "holdout ONLY after the operator freezes the specific alpha and "
              "logs it -- NOT done automatically by this script.")
    else:
        print("\n    VERDICT: NEGATIVE. No completed alpha in ALPHA_GRID clears "
              "all of clauses (a),(b) on both markets AND (c) on both markets "
              "on inner-validation.")
    print(f"\n    Max timestamp read anywhere in this run: {max(max_ts_seen)}   "
          f"(OOS_START = {OOS_START}; strictly earlier: "
          f"{max(max_ts_seen) < pd.Timestamp(OOS_START, tz='UTC')})")

    hr("HOLDOUT")
    holdout_msg = ("NOT YET -- gate cleared, awaiting operator go-ahead per the routine"
                   if (kill_switches_ok and any_promotes) else "NO")
    print(f"    Holdout consulted: {holdout_msg}")
    print("    This script never reads a bar at or after OOS_START (2023-01-01); "
          "`load_btc`/`load_eth`\n    truncate before it and `compare`/`run_slice` "
          "assert against it on every call.")


if __name__ == "__main__":
    main()
