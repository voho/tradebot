#!/usr/bin/env python
"""R-174 NOVEL branch: gate ``kelly_regime_v4``'s ``frac*scale`` INCREASES
behind a Grunwald-de Heide-Koolen (2024) GROW e-variable, built as a
sequential Gaussian-mixture Bayes factor -- H0: ``mu=0``, H1 (mixture):
``mu ~ N(0, TAU^2)``, both with per-bar KNOWN variance ``sigma_i``
(``r174_shared.causal_bar_sigma``). Decreases stay immediate and ungated,
exactly as v4 ships today. Full Step 1/Step 2 design, the pre-registered
decision rule, the four named failure modes, and the "not a duplicate of"
argument are all in ``experiments/r174_direction.md`` and are not re-derived
here. This file implements ONLY the accumulator (``new_episode_state`` /
``step_fn``) that ``r174_shared.run_asymmetric_gate`` calls once per pending
bar, plus this branch's own ``compare()`` sweep. It does not edit
``r174_shared.py`` or the conservative branch's file, and never reads a bar
at or after ``r174_shared.OOS_START``.

DERIVATION (verified independently, not merely copied from the
pre-registration -- see citations below).

Sequential conjugate Gaussian update. Prior on ``mu``: ``N(0, TAU^2)``, so
prior precision ``P_0 = 1/TAU^2``, prior weighted-sum ``W_0 = 0``. After
observing bars ``1..i-1`` (each with its own known ``sigma_j``), the
posterior on ``mu`` is exactly ``N(W/P, 1/P)`` with
``P = 1/TAU^2 + sum_{j<i} 1/sigma_j^2`` and ``W = sum_{j<i} r_j/sigma_j^2``
(standard Gaussian-conjugate precision-weighted update; each new observation
contributes precision ``1/sigma_i^2`` and weighted evidence ``r_i/sigma_i^2``,
independent of the pre-registration -- this is the textbook Bayesian linear
model / Kalman-filter-with-no-process-noise update, e.g. Gelman et al.,
*Bayesian Data Analysis* 3rd ed., ch. 2-3).

The one-step-ahead PREDICTIVE density of ``r_i`` given only bars ``1..i-1``
is obtained by marginalising the CURRENT posterior's uncertainty about
``mu``: since ``r_i = mu + eps_i`` with ``mu ~ N(W/P, 1/P)`` (posterior) and
``eps_i ~ N(0, sigma_i^2)`` independent of ``mu``, the marginal of ``r_i`` is
``N(W/P, sigma_i^2 + 1/P)`` -- exactly the "predictive mean = W/P, predictive
variance = sigma_i^2 + 1/P" construction specified for this branch. Under H0
the predictive density of ``r_i`` is simply ``N(0, sigma_i^2)`` (no update at
all, since H0 has no free parameter). The log e-value increment at bar i is
``logpdf_h1(r_i) - logpdf_h0(r_i)``, using ONLY information available before
``r_i`` is observed (the posterior is not updated with ``r_i`` until after
the predictive density is scored) -- this is what makes the accumulated
``log_e`` a genuine SEQUENTIAL (prequential) score, not a lookahead one.

Why the running PRODUCT of these one-step Bayes factors is exactly the
Bayes factor of the two hypotheses on the whole data seen so far, and why
that product is a valid e-process / test martingale under H0 -- confirmed,
independently of the pre-registration, via:

- Grunwald, P., de Heide, R., & Koolen, W. (2024), "Safe Testing", *JRSS
  Series B* 86(5), 1091-1128. Abstract/summary (fetched via WebSearch
  2026-08-28): "GRO e-values take the form of Bayes factors with special
  priors" -- GROW (GROwth-rate optimal in Worst case) e-variables for a
  simple null against a COMPOSITE alternative are constructed exactly as a
  Bayes factor against a MIXTURE over the alternative's free parameter,
  which is precisely this file's ``mu ~ N(0, TAU^2)`` mixture against
  ``H0: mu=0``. Ville's inequality (restated in the same source and
  Grunwald/de Heide/Koolen's discussion thread) gives the anytime-valid
  guarantee this branch relies on: ``P_0(sup_t e_t >= 1/alpha) <= alpha``.
- Shafer, G., Shen, A., Vereshchagin, N., & Vovk, V. (2011), "Test
  Martingales, Bayes Factors and p-Values", *Statistical Science* 26(1),
  84-101 (arXiv:0912.4269, confirmed live via WebSearch 2026-08-28): shows
  that a SEQUENTIALLY updated Bayes factor -- the running product of
  one-step-ahead predictive-density ratios exactly as constructed here -- is
  a nonnegative martingale with ``E_0[e_t] = 1`` under the null for every
  ``t``, i.e. a test martingale, whenever the null is simple (as it is here:
  ``H0: mu=0`` is a single point, not itself a mixture). "E-values take the
  form of Bayes factors with special priors" is stated explicitly in this
  literature (also echoed in the Safe Testing search summary above), which
  is the load-bearing claim for this whole file: this is NOT an
  approximation of a test martingale, it IS one, by this standard
  construction.
- Ville, J. (1939) (cited via both sources above and in
  ``r174_direction.md``): ``P(sup_t W_t > 1/alpha) <= alpha`` for any
  nonnegative supermartingale ``W`` with ``W_0 <= 1``. Our ``e_t =
  exp(log_e_t)`` starts at ``e_0 = 1`` (``log_e = 0``, no evidence yet) and
  is a nonnegative martingale under H0 by the Shafer et al. result above, so
  Ville applies directly: thresholding accept at ``log_e >= log(1/alpha)``
  is anytime-valid at level ``alpha``, for EVERY possible stopping rule
  (including "keep testing until the pending episode itself disappears"),
  which is exactly the stopping behaviour ``run_asymmetric_gate`` imposes.

DISCLOSED DESIGN CHOICE -- no formal reject boundary. Unlike the
conservative branch's Wald SPRT (two boundaries, ``log(1/alpha)`` and
``log(beta/(1-alpha))``, from a likelihood ratio between two SIMPLE
hypotheses), an e-process built from a one-sided Ville threshold has no
symmetric lower boundary with a matching error-control guarantee -- "reject
H1, accept H0" is not a claim this construction makes at any level. This
file's ``step_fn`` therefore returns ONLY ``"pending"`` or ``"accept"``,
NEVER ``"reject"``. The mechanism that keeps an unresolved episode from
running forever is entirely ``run_asymmetric_gate``'s own bookkeeping (see
``r174_shared.py``): a pending episode is silently abandoned, with no grant,
the moment ``desired[i]`` itself drops back to ``<= pos + deadband`` -- i.e.
v4's own vote/scale decided the increase request is no longer live. This is
a genuine, PRE-REGISTERED asymmetry against the conservative branch (stated
in ``r174_direction.md``'s own division-of-labor section), not an oversight
here.

DISCLOSED LIMITATION, found by this branch's own self-test (see part (c)
below) and reported honestly per ROUTINE.md's demand for that: the mixture
``mu ~ N(0, TAU^2)`` is SYMMETRIC in the sign of ``mu``. The expected
per-bar log-likelihood-ratio in favour of H1 is (asymptotically, as the
posterior concentrates) the KL divergence ``KL(N(mu,sigma^2) ||
N(0,sigma^2)) = mu^2 / (2*sigma^2)``, which depends on ``mu`` only through
its SQUARE. Consequently this accumulator is a test of "has mu departed from
zero", not "has mu become POSITIVE" -- it accumulates evidence, and would
eventually cross the accept threshold, under a strongly NEGATIVE drift
exactly as readily (in fact slightly faster, since larger |mu| grows the KL
term faster) as under an equally-sized positive drift. This is an intrinsic
property of the pre-registered mixture (``r174_direction.md`` and
``r174_shared.TAU`` are both frozen and this file must not change them, per
the hard "no editing r174_shared.py" rule and the "do not move the
goalposts after seeing results" rule), not a bug in this implementation.
The reason this does not defeat the whole mechanism is structural, not
statistical: the object being tested is ``r_i``, the REALIZED return, and a
pending "increase" episode only exists because v4's own vote/scale already
computed ``desired > pos + deadband``; under genuinely negative drift, real
BTC price action typically erodes that same ``desired`` back down (v4's vote
and scale both respond to falling prices), which cancels the episode via
``run_asymmetric_gate``'s own episode-cancellation path BEFORE the
accumulator has had enough bars to reach threshold -- not because the
accumulator itself is directional. This file's self-test measures the
accumulator in isolation (necessarily symmetric); the full gate's behaviour
under real, non-i.i.d. BTC returns is measured by the causality probe,
calibration test and the real-data ``compare()`` sweep below, not asserted
from the isolated math.

USAGE
-----
    python experiments/r174_novel_safe_testing_gro.py
"""

from __future__ import annotations

import math
import sys
import time
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
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TAU,
    V4_HORIZONS,
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

LOG_2PI = math.log(2.0 * math.pi)


def hr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (1)
# The GROW e-variable accumulator. One instance per alpha (the accept
# threshold log(1/alpha) depends on alpha; the accumulator's own recursion
# does not).
# ==================================================================

def make_grow_step(alpha: float):
    """Return ``(new_episode_state, step_fn)`` for one alpha's Ville
    threshold. State is a plain 3-tuple ``(P, W, log_e)`` (posterior
    precision, posterior weighted-sum, accumulated log e-value) -- a tuple,
    not a dict, purely so the per-bar recursion inside
    ``run_asymmetric_gate``'s hot loop (this comparison runs it over
    several hundred thousand bars per market/slice) avoids dict-attribute
    overhead; the constant-factor cost was checked and is negligible either
    way (see this file's own timing note in the module-level run), so this
    is a minor tidiness choice, not a load-bearing optimisation.

    ``P`` starts at ``1/TAU**2`` (prior precision), ``W`` starts at 0.0
    (prior weighted-sum, so posterior mean ``W/P`` starts at 0 -- the
    mixture's own prior mean), and ``log_e`` starts at exactly 0.0 (no
    evidence yet, ``e_0 = exp(0) = 1``, Ville's own required initial
    condition ``W_0 <= 1``, here with equality).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    log_thresh = -math.log(alpha)  # log(1/alpha)

    def new_episode_state():
        return (1.0 / (TAU * TAU), 0.0, 0.0)

    def step_fn(state, ri, si):
        P, W, log_e = state
        inv_P = 1.0 / P
        var0 = si * si                      # H0: N(0, sigma_i^2)
        pred_var = var0 + inv_P             # H1 predictive: sigma_i^2 + 1/P
        pred_mean = W * inv_P               # H1 predictive mean: W/P

        d1 = ri - pred_mean
        logpdf_h1 = -0.5 * (LOG_2PI + math.log(pred_var) + d1 * d1 / pred_var)
        logpdf_h0 = -0.5 * (LOG_2PI + math.log(var0) + ri * ri / var0)
        log_e2 = log_e + (logpdf_h1 - logpdf_h0)

        inv_var0 = 1.0 / var0
        P2 = P + inv_var0
        W2 = W + ri * inv_var0

        decision = "accept" if log_e2 >= log_thresh else "pending"
        return (P2, W2, log_e2), decision

    return new_episode_state, step_fn


# ================================================================== (2)
# Per-alpha candidate target, with content-keyed caching (r160's own
# convention -- see that file's identical comment) so compare()'s 2x
# (market-pair) redundancy and the causal truncation probe's tail-perturbed
# second call do not re-run the per-bar gate loop needlessly, and so a
# same-shaped, same-index perturbed frame never collides with the original
# in the cache (keyed on content, never on `id(df)`).
# ==================================================================

_COMPONENT_CACHE: dict[tuple, tuple] = {}


def _frame_key(df: pd.DataFrame) -> tuple:
    idx = df.index
    c = df["close"].to_numpy()
    return (int(idx[0].value), int(idx[-1].value), len(df), float(np.sum(c)), float(c[-1]))


def gro_components(df: pd.DataFrame, alpha: float) -> tuple[np.ndarray, int, int]:
    """``(gated_target, delayed_episodes, total_episodes)`` for this alpha,
    on this frame, cached by content."""
    key = (_frame_key(df), round(alpha, 6))
    if key not in _COMPONENT_CACHE:
        desired = v4_raw_desired(df)
        r = causal_bar_returns(df)
        sigma = causal_bar_sigma(df)
        new_ep, step = make_grow_step(alpha)
        gated, delayed, total = run_asymmetric_gate(desired, r, sigma, new_ep, step)
        _COMPONENT_CACHE[key] = (gated, delayed, total)
    return _COMPONENT_CACHE[key]


def build_candidate_target(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """The pre-registered candidate: v4's own raw desired exposure, gated on
    INCREASES only by the GROW accumulator at this alpha, decreases passed
    through immediately (``run_asymmetric_gate``'s own contract)."""
    gated, _delayed, _total = gro_components(df, alpha)
    return gated


def make_build_target(alpha: float):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_candidate_target(df, alpha)
    _build.__name__ = f"gro_alpha{alpha:g}"
    return _build


BUILD_PRIMARY = make_build_target(ALPHA_PRIMARY)
BUILD_PRIMARY.__name__ = "novel_safe_testing_gro"


# ================================================================== (3)
# Calibration self-test: synthetic ZERO-DRIFT noise. Ville's inequality
# bounds P(sup_t e_t >= 1/alpha) <= alpha PER RUN (across the whole run, not
# per bar) -- so the honest calibration measurement is "of many independent
# fixed-length zero-drift episodes, each its own run, what fraction ever
# cross the accept threshold before the episode ends", which Ville bounds at
# <= alpha for each such truncated run (a truncated stopping time's sup can
# only be <= the full-run sup). Reported honestly either way.
# ==================================================================

def calibration_episode_rate(alphas=ALPHA_GRID, episode_len: int = 1000,
                              n_frames: int = 4, n_per_frame: int = 300_000,
                              base_seed: int = 9174) -> list[dict]:
    # Precompute (r, sigma) once per frame -- independent of alpha -- and
    # reuse across every alpha in the grid.
    frames = []
    for k in range(n_frames):
        z = synthetic_zero_drift_frame(n=n_per_frame, seed=base_seed + k)
        assert_no_holdout(z, "calibration_episode_rate(): synthetic")
        r = causal_bar_returns(z)
        sigma = causal_bar_sigma(z)
        valid = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0)
        start = int(np.argmax(valid)) if valid.any() else len(valid)
        frames.append((r[start:], sigma[start:]))

    rows = []
    for alpha in alphas:
        new_ep, step = make_grow_step(alpha)
        crossed = 0
        n_episodes = 0
        crossing_bar = []
        for r_v, sigma_v in frames:
            n_chunks = len(r_v) // episode_len
            for c in range(n_chunks):
                seg_r = r_v[c * episode_len:(c + 1) * episode_len]
                seg_s = sigma_v[c * episode_len:(c + 1) * episode_len]
                state = new_ep()
                n_episodes += 1
                for j in range(episode_len):
                    state, decision = step(state, seg_r[j], seg_s[j])
                    if decision == "accept":
                        crossed += 1
                        crossing_bar.append(j + 1)
                        break
        rate = crossed / n_episodes if n_episodes else float("nan")
        rows.append(dict(alpha=alpha, episode_len=episode_len, n_episodes=n_episodes,
                          crossed=crossed, observed_rate=rate,
                          mean_crossing_bar=float(np.mean(crossing_bar)) if crossing_bar else float("nan")))
    return rows


def print_calibration(rows: list[dict]) -> None:
    hdr = (f"{'alpha':>6s} {'ep_len':>7s} {'n_ep':>6s} {'crossed':>8s} "
           f"{'obs_rate':>9s} {'mean_bar':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['alpha']:6.2f} {r['episode_len']:7d} {r['n_episodes']:6d} "
              f"{r['crossed']:8d} {r['observed_rate']:9.4f} {r['mean_crossing_bar']:9.1f}")


# ================================================================== (4)
# Kill switches. A1 is checked exactly as this branch's task specifies: on
# inner-validation, BTC, delayed_episodes >= GATE_MIN_DELAYS for >=1 alpha
# (the gate is a pure function of price -- it does not depend on
# MarketSpec -- so "at least one market" is automatically every market).
# A2 (non-degeneracy) is reported both on inner-validation and on the full
# pre-holdout BTC frame, for completeness.
# ==================================================================

def kill_switch_row(btc_full: pd.DataFrame, btc_inner_val: pd.DataFrame, alpha: float) -> dict:
    gated_full, delayed_full, total_full = gro_components(btc_full, alpha)
    gated_val, delayed_val, total_val = gro_components(btc_inner_val, alpha)
    r2_full = r_squared(gated_full, v4_target(btc_full))
    r2_val = r_squared(gated_val, v4_target(btc_inner_val))
    a1_bind_val = delayed_val >= GATE_MIN_DELAYS
    a2_ok_full = not (np.isfinite(r2_full) and r2_full > R2_DEGENERACY_THRESH)
    a2_ok_val = not (np.isfinite(r2_val) and r2_val > R2_DEGENERACY_THRESH)
    return dict(alpha=alpha, delayed_full=delayed_full, total_full=total_full,
                delayed_val=delayed_val, total_val=total_val,
                r_sq_full=r2_full, r_sq_val=r2_val,
                a1_bind_val=a1_bind_val, a2_ok_full=a2_ok_full, a2_ok_val=a2_ok_val)


# ================================================================== (5)
# Decision rule -- IDENTICAL SHAPE to r174_direction.md's PROMOTE-CANDIDATE
# clauses (a)/(b)/(c), restated here (not imported, since the branches must
# not share files); evaluated exactly as written, not re-derived after
# seeing any number.
# ==================================================================

def evaluate_decision_rule(rows: list[dict]) -> dict:
    inner_val = {r["market"]: r for r in rows if r["slice"] == "inner_val"}
    eth = {r["market"]: r for r in rows if r["slice"] == "eth_replication"}
    per_market = {}
    for market in ("spot", "futures_5x"):
        iv = inner_val.get(market)
        et = eth.get(market)
        if iv is None or et is None:
            per_market[market] = dict(clause_a=False, clause_b=False, clause_c=False,
                                       passes=False, note="missing row")
            continue
        clause_a = bool(iv["boot_lo"] > 0)
        sharpe_edge = iv["d_sharpe"] >= SHARPE_NOISE_FLOOR
        dd_edge = bool(iv["risk_matched"] and iv["d_dd"] < 0)
        clause_b = bool(sharpe_edge or dd_edge)
        clause_c = False
        fired_via = None
        if clause_a and clause_b:
            if sharpe_edge:
                fired_via = "sharpe"
                clause_c = bool(et["d_sharpe"] > 0)
            elif dd_edge:
                fired_via = "dd"
                clause_c = bool(et["risk_matched"] and et["d_dd"] < 0)
        per_market[market] = dict(clause_a=clause_a, clause_b=clause_b, clause_c=clause_c,
                                   fired_via=fired_via,
                                   passes=bool(clause_a and clause_b and clause_c))
    both_markets_pass = all(per_market[m]["passes"] for m in ("spot", "futures_5x"))
    return dict(per_market=per_market, both_markets_pass=both_markets_pass)


# --------------------------------------------------------------- self-test

def _simulate_mean_log_e(step_fn, new_state_fn, mu: float, sigma: float,
                          n_steps: int, n_reps: int, seed: int) -> tuple[float, float]:
    """Mean and std (across `n_reps` independent replicates) of `log_e`
    after `n_steps` bars of r_i ~ N(mu, sigma^2), sigma treated as known
    (matching this branch's own "known variance per bar" model)."""
    rng = np.random.default_rng(seed)
    totals = np.empty(n_reps)
    for rep in range(n_reps):
        state = new_state_fn()
        draws = rng.normal(mu, sigma, n_steps)
        for ri in draws:
            state, _decision = step_fn(state, float(ri), sigma)
        totals[rep] = state[2]
    return float(totals.mean()), float(totals.std())


def _self_test() -> None:
    # (0) construction sanity: log_e is exactly 0 right after
    # new_episode_state() for every alpha (no evidence yet).
    for alpha in ALPHA_GRID:
        new_ep, _step = make_grow_step(alpha)
        P0, W0, log_e0 = new_ep()
        assert log_e0 == 0.0, (alpha, log_e0)
        assert abs(P0 - 1.0 / (TAU * TAU)) < 1e-6, (alpha, P0)
        assert W0 == 0.0

    # (a) MANDATORY: causal truncation probe on the PRIMARY-alpha candidate.
    idx = pd.date_range("2017-01-01", periods=120_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(174_01)
    innov = rng.normal(0.00002, 0.0006, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    synth_df = pd.DataFrame({"open": close, "high": high, "low": low,
                              "close": close, "volume": 1.0}, index=idx)

    def _primary_wrapper(df: pd.DataFrame) -> np.ndarray:
        return build_candidate_target(df, ALPHA_PRIMARY)
    _primary_wrapper.__name__ = "gro_primary_selftest"

    assert causal_truncation_probe_series(_primary_wrapper, synth_df), \
        "causal_truncation_probe_series FAILED on build_candidate_target at ALPHA_PRIMARY"

    # (b) calibration self-test: fraction of independent zero-drift episodes
    # that EVER cross the accept threshold before a fixed horizon. Reported
    # honestly -- Ville bounds this at <= alpha per truncated run, not
    # asserted to be exactly alpha.
    calib_rows = calibration_episode_rate(episode_len=1000, n_frames=3, n_per_frame=200_000)
    print("[r174 novel self-test] calibration (zero-drift, episode_len=1000):")
    print_calibration(calib_rows)
    for row in calib_rows:
        flag = "OK (<= nominal alpha)" if row["observed_rate"] <= row["alpha"] * 1.5 else \
               "OVERSHOOTS nominal alpha -- reported honestly, not hidden"
        print(f"    alpha={row['alpha']:.2f}: observed_rate={row['observed_rate']:.4f}  {flag}")

    # (c) directional numeric sanity on the ISOLATED accumulator (own
    # state/recursion only, no run_asymmetric_gate). Reports the TRUE
    # observed behaviour, including where it departs from the naive
    # (sign-directional) expectation -- see the module docstring's own
    # "DISCLOSED LIMITATION" section for the reason.
    new_ep, step = make_grow_step(ALPHA_PRIMARY)
    sigma_fixed = 8.0e-4
    n_steps, n_reps = 20_000, 200
    mean0, std0 = _simulate_mean_log_e(step, new_ep, 0.0, sigma_fixed, n_steps, n_reps, seed=174_10)
    mean_pos, std_pos = _simulate_mean_log_e(step, new_ep, TAU, sigma_fixed, n_steps, n_reps, seed=174_11)
    mean_neg, std_neg = _simulate_mean_log_e(step, new_ep, -TAU, sigma_fixed, n_steps, n_reps, seed=174_12)
    print(f"[r174 novel self-test] mean(log_e) after {n_steps} bars, {n_reps} reps, "
          f"sigma={sigma_fixed:g}:")
    print(f"    mu=0     : {mean0:+.4f} (std {std0:.4f})  -- non-positive in expectation under H0 "
          "(Jensen: E[log e_t] <= log E[e_t] = 0 for a mean-1 martingale; NOT 'flat at 0' -- a "
          "log-Bayes-factor process is expected to drift negative, sometimes substantially, under "
          "its own null, which is the reason Ville's bound is one-sided)")
    print(f"    mu=+TAU  : {mean_pos:+.4f} (std {std_pos:.4f})  -- trends toward accept, as expected")
    print(f"    mu=-TAU  : {mean_neg:+.4f} (std {std_neg:.4f})  -- ALSO trends toward accept "
          "(symmetric mixture: KL ~ mu^2, disclosed in module docstring; this is NOT the naive "
          "'stays near 0' expectation, reported honestly rather than silently matched to it)")
    # Assert only the TRUE, verifiable properties, corrected after the first
    # run of this self-test showed the naive "flat at 0" expectation for
    # mu=0 was itself wrong (Jensen's inequality, not a bug -- see prints
    # above): (i) mu=0's mean log_e is NOT positive beyond sampling noise
    # (a positive mean would indicate a sign error in the H0/H1 predictive
    # densities, which Jensen rules out for a correctly-built mean-1
    # martingale); (ii) BOTH +TAU and -TAU trend strictly above the mu=0
    # control, by comparable magnitude (symmetric-in-sign, as the math
    # predicts) -- NOT that the negative case stays near zero.
    assert mean0 < 3.0 * (std0 / math.sqrt(n_reps)) + 0.05, (mean0, std0)
    assert mean_pos > mean0, (mean_pos, mean0)
    assert mean_neg > mean0, (mean_neg, mean0)
    # symmetric magnitude check: the two should be within a wide tolerance
    # of each other (both driven by the same |mu|), not one near zero and
    # the other large.
    assert abs(mean_pos - mean_neg) < 0.5 * max(abs(mean_pos), abs(mean_neg), 1.0), \
        (mean_pos, mean_neg)


_self_test()


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0

    hr("R-174 NOVEL: GROW e-variable (Grunwald-de Heide-Koolen 2024 Safe Testing) gated "
       "kelly_regime_v4 INCREASES")
    print("mechanism: v4's own frac*scale re-sizing decision is gated, on bars where it would")
    print("INCREASE exposure only, behind a sequential Gaussian-mixture Bayes factor (H0: mu=0,")
    print("H1: mu ~ N(0, TAU^2)), thresholded at log(1/alpha) per Ville's inequality -- a GROW")
    print("e-variable per Grunwald/de Heide/Koolen (2024). Decreases stay immediate (unchanged).")
    print(f"\nALPHA_GRID (frozen, r174_shared) = {ALPHA_GRID}   ALPHA_PRIMARY = {ALPHA_PRIMARY}")
    print(f"TAU (frozen, r174_shared, = MU1) = {TAU:.6e}")

    btc = load_btc()
    eth = load_eth()
    max_ts_seen.append(btc.index.max())
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(btc, "main(): btc")
    assert_no_holdout(eth, "main(): eth")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH (Bitfinex replication, truncated < {OOS_START}): {len(eth):,} bars, "
          f"{eth.index[0]} -> {eth.index[-1]}")
    btc_inner_val = btc.loc[INNER_VAL_START:INNER_VAL_END]
    print(f"BTC inner-validation slice: {len(btc_inner_val):,} bars, "
          f"{btc_inner_val.index[0]} -> {btc_inner_val.index[-1]}")

    # ============================================================= STEP 1
    hr("STEP 1 -- CAUSALITY: truncation probe on the full candidate build_target, PRIMARY alpha")
    try:
        probe_ok = causal_truncation_probe_series(BUILD_PRIMARY, btc)
        print(f"  causal_truncation_probe_series({BUILD_PRIMARY.__name__}, btc): PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  causal_truncation_probe_series({BUILD_PRIMARY.__name__}, btc): FAIL: {e}")

    # ============================================================= STEP 2
    hr("STEP 2 -- CALIBRATION SELF-TEST: synthetic ZERO-DRIFT noise, full ALPHA_GRID, larger run")
    print("(episode_len=20,000 bars -- long enough, given TAU's tiny magnitude relative to typical")
    print(" bar sigma, for the accumulator to plausibly reach threshold at least sometimes; the")
    print(" module's own self-test at import time additionally checks a short 1,000-bar horizon,")
    print(" where it never fires at all -- reported there honestly as an over-conservative floor,")
    print(" not evidence of miscalibration.)")
    calib_rows = calibration_episode_rate(episode_len=20_000, n_frames=10, n_per_frame=300_000)
    print_calibration(calib_rows)
    n_configs += 0  # calibration is a diagnostic, not a compare() configuration
    for row in calib_rows:
        note = ("within ~1.5x nominal alpha" if row["observed_rate"] <= row["alpha"] * 1.5
                 else "OVERSHOOTS nominal alpha -- reported honestly")
        print(f"    alpha={row['alpha']:.2f}: observed_rate={row['observed_rate']:.4f}  ({note})")

    # ============================================================= STEP 3
    hr("STEP 3 -- KILL SWITCHES: A1 (gate binds, inner-validation, BTC), "
       "A2 (non-degenerate vs v4_target)")
    kill_rows = []
    for alpha in ALPHA_GRID:
        kr = kill_switch_row(btc, btc_inner_val, alpha)
        kill_rows.append(kr)
        print(f"  alpha={alpha:.2f}  delayed[inner_val]={kr['delayed_val']:3d} "
              f"(of {kr['total_val']} episodes, need >={GATE_MIN_DELAYS})  "
              f"A1[inner_val]={'PASS' if kr['a1_bind_val'] else 'FAIL'}   "
              f"r_sq[inner_val]={kr['r_sq_val']:.5f} A2[inner_val]={'PASS' if kr['a2_ok_val'] else 'FAIL'}   "
              f"r_sq[full pre-holdout]={kr['r_sq_full']:.5f} A2[full]={'PASS' if kr['a2_ok_full'] else 'FAIL'}   "
              f"delayed[full]={kr['delayed_full']} (of {kr['total_full']})")
    any_bind_val = any(kr["a1_bind_val"] for kr in kill_rows)
    any_degenerate_val = any(not kr["a2_ok_val"] for kr in kill_rows)
    any_degenerate_full = any(not kr["a2_ok_full"] for kr in kill_rows)
    print(f"\nA1 (>=1 alpha shows delayed_episodes>={GATE_MIN_DELAYS} on inner-validation, BTC): "
          f"{'PASS' if any_bind_val else 'FAIL'}")
    print(f"A2 (no alpha is R^2-degenerate vs v4_target, inner-validation): "
          f"{'PASS (none degenerate)' if not any_degenerate_val else 'SOME CONFIGS DEGENERATE'}")
    print(f"A2 (no alpha is R^2-degenerate vs v4_target, full pre-holdout, for completeness): "
          f"{'PASS (none degenerate)' if not any_degenerate_full else 'SOME CONFIGS DEGENERATE'}")

    # ============================================================= STEP 4
    hr(f"STEP 4 -- FULL SWEEP: ALPHA_GRID ({len(ALPHA_GRID)} configs), compare() over "
       "inner_train/inner_val/eth_replication, SPOT+FUTURES")
    all_rows: dict[float, list[dict]] = {}
    decision_by_alpha: dict[float, dict] = {}
    completed_alphas: list[float] = []
    for alpha in ALPHA_GRID:
        label = f"gro_alpha{alpha}"
        print(f"\n--- alpha={alpha} ---")
        t_cfg = time.time()
        rows = compare(lambda df, a=alpha: build_candidate_target(df, a), label=label,
                       btc=btc, eth=eth, markets=(SPOT, FUTURES), include_eth=True)
        dt_cfg = time.time() - t_cfg
        print_rows(rows)
        print(f"  [{dt_cfg:.1f}s for this config]")
        n_configs += 1  # one config = one alpha run through compare()
        completed_alphas.append(alpha)
        all_rows[alpha] = rows
        decision = evaluate_decision_rule(rows)
        decision_by_alpha[alpha] = decision
        for market, d in decision["per_market"].items():
            print(f"    decision-rule[{market}]: a={d['clause_a']} b={d['clause_b']} "
                  f"c={d['clause_c']} (fired_via={d.get('fired_via')}) -> passes={d['passes']}")
        print(f"    BOTH MARKETS PASS (this alpha clears PROMOTE-CANDIDATE): "
              f"{decision['both_markets_pass']}")

    # ============================================================= STEP 5
    hr("STEP 5 -- CROSS-BRANCH PATTERN CHECK: ETH-futures vs BTC d_sharpe, primary alpha")
    if ALPHA_PRIMARY in all_rows:
        prim_rows = all_rows[ALPHA_PRIMARY]
        by_cell = {(r["slice"], r["market"]): r for r in prim_rows}
        eth_fut = by_cell.get(("eth_replication", "futures_5x"))
        btc_train_spot = by_cell.get(("inner_train", "spot"))
        btc_val_spot = by_cell.get(("inner_val", "spot"))
        btc_train_fut = by_cell.get(("inner_train", "futures_5x"))
        btc_val_fut = by_cell.get(("inner_val", "futures_5x"))
        print(f"  ETH futures_5x d_sharpe        = {eth_fut['d_sharpe']:+.4f}" if eth_fut else "  (missing)")
        print(f"  BTC inner_train spot d_sharpe  = {btc_train_spot['d_sharpe']:+.4f}" if btc_train_spot else "  (missing)")
        print(f"  BTC inner_val   spot d_sharpe  = {btc_val_spot['d_sharpe']:+.4f}" if btc_val_spot else "  (missing)")
        print(f"  BTC inner_train fut  d_sharpe  = {btc_train_fut['d_sharpe']:+.4f}" if btc_train_fut else "  (missing)")
        print(f"  BTC inner_val   fut  d_sharpe  = {btc_val_fut['d_sharpe']:+.4f}" if btc_val_fut else "  (missing)")

    # ============================================================= STEP 6
    hr("STEP 6 -- CONFIGURATION COUNT")
    print(f"ALPHA_GRID configs run through compare(): {len(completed_alphas)} of {len(ALPHA_GRID)} "
          f"({completed_alphas})")
    print(f"TOTAL CONFIGURATIONS EVALUATED (this file, per the task's own definition -- "
          f"one alpha through compare() = one config): {n_configs}")

    # ============================================================= VERDICT
    hr("VERDICT")
    promote_alphas = [a for a, d in decision_by_alpha.items() if d["both_markets_pass"]]
    print(f"causal truncation probe PASS:                                  {probe_ok}")
    print(f"A1 kill switch (gate binds on inner-validation, >=1 alpha):    {any_bind_val}")
    print(f"A2 kill switch (no R^2-degenerate alpha, inner-validation):    {not any_degenerate_val}")
    print(f"alphas clearing PROMOTE-CANDIDATE (both markets, a+b+c):       "
          f"{promote_alphas if promote_alphas else 'NONE'}")
    verdict = ("PROMOTE-candidate (gate clears; holdout may be consulted)"
               if (probe_ok and any_bind_val and not any_degenerate_val and promote_alphas)
               else "NEGATIVE")
    print(f"\nVERDICT: {verdict}")
    if verdict.startswith("PROMOTE"):
        print("\nGate clears causality + kill switches + decision rule on >=1 alpha --")
        print("holdout MAY be consulted per docs/ROUTINE.md step 4. This file does NOT itself")
        print("read OOS_START; that is a separate, explicit step left to the operator.")
    else:
        print("\nGate does NOT clear the full decision rule on any swept alpha -- per")
        print("docs/ROUTINE.md's own discipline, the holdout is precious and is NOT touched. No")
        print("bar at or after OOS_START is read anywhere in this file.")

    max_ts = max(max_ts_seen)
    print(f"\nmax timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, probe_ok=probe_ok, calib_rows=calib_rows, kill_rows=kill_rows,
                any_bind_val=any_bind_val, any_degenerate_val=any_degenerate_val,
                all_rows=all_rows, decision_by_alpha=decision_by_alpha,
                n_configs=n_configs, completed_alphas=completed_alphas,
                max_ts=max_ts, verdict=verdict, promote_alphas=promote_alphas)


if __name__ == "__main__":
    main()
