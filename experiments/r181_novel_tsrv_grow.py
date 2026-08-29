#!/usr/bin/env python
"""R-181 NOVEL branch: gate ``kelly_regime_v4``'s ``frac*scale`` INCREASES
behind a Grunwald-de Heide-Koolen (2024) GROW/safe-testing mixture e-value,
computed at 5-minute cadence but fed the TWO-SCALE, noise-bias-corrected
realized-variance (TSRV, Zhang-Mykland-Ait-Sahalia 2005) sigma
(``r181_shared.causal_bar_sigma_tsrv``) instead of the naive per-bar sigma
R-174's own novel branch used, and a MIXTURE alternative over a plausible
RANGE of drift magnitudes that CONCENTRATES (shrinks toward the low end of
that range) as a pending episode ages -- the "state-dependent alternative"
distinct from R-174's fixed-TAU point alternative. Decreases stay immediate
and ungated, exactly as v4 ships today and exactly as R-174's gate already
established. Full Step 1/Step 2 design, the pre-registered decision rule,
the four named failure modes, and the "not a duplicate of" argument are all
in ``experiments/r181_direction.md`` and are not re-derived here. This file
implements ONLY this branch's own non-degeneracy precheck, accumulator
(``new_episode_state``/``step_fn``) that ``r181_shared.run_asymmetric_gate``
calls once per pending bar, and its own ``compare()`` sweep. It does not
edit ``r181_shared.py`` or the conservative branch's file, and never reads a
bar at or after ``r181_shared.OOS_START``.

MIXTURE CONSTRUCTION -- the one genuinely novel piece of math in this file
----------------------------------------------------------------------------
R-174's novel branch used a SINGLE fixed Gaussian prior on the drift,
``mu ~ N(0, TAU^2)``, integrated in closed form via the standard conjugate
Bayes recursion. That mixture (a) never changes width over an episode's
life (only the POSTERIOR narrows, as a side effect of more data, not the
prior itself) and (b) is symmetric in the sign of ``mu`` (R-174's own
disclosed limitation: it accumulates evidence for "mu has departed from
zero" in either direction, not specifically "mu is positive"), which is an
odd fit for a gate that only ever fires on exposure-INCREASE requests.

This branch instead builds a DISCRETIZED mixture of ``J=6`` SIMPLE
point-mass alternatives, ``H1_j: r_i ~ N(mu_j(n), sigma_i^2)``, against the
same simple null ``H0: r_i ~ N(0, sigma_i^2)``, where:

- ``mu_j(n) = TAU_PER_BAR * WIDTH_SCALE * m_j * shrink(n)``, ``m_j`` drawn
  from a geometric grid ``M_GRID = (1, 2, 4, 8, 16, 32)`` (multiples of
  ``TAU_PER_BAR``, the inner-train mean per-bar log-return -- the natural
  per-bar drift scale already derived, once, in ``r181_shared.py``) --
  i.e. "plausible drift values from about the long-run per-bar mean up to
  32x that", a range wide enough to cover both a genuinely modest signal
  and a fast, large intraday move, chosen because a short-lived deadband
  episode can move far faster than the multi-year per-bar average.
- Fixed, DATA-INDEPENDENT prior weights ``pi_j proportional to 1/m_j``
  (chosen before touching any evidence -- more prior mass on the more
  "typical"/modest magnitudes, a standard decaying-tail belief, not tuned
  post hoc), so this is a genuine MIXTURE (a weighted sum of e-processes),
  not a single fixed-point alternative.
- ONLY POSITIVE candidate values (``m_j > 0``) -- unlike R-174's symmetric
  ``N(0, TAU^2)``, this mixture tests specifically for POSITIVE drift, the
  hypothesis that actually justifies the exposure increase the gate is
  deciding on. Verified directional (not merely asserted) in this file's
  own self-test part (c) below: strongly positive drift trends the
  accumulator toward accept; strongly negative drift does not.
- ``shrink(n) = SHRINK_FLOOR + (1 - SHRINK_FLOOR) * exp(-n / N_HALF_BARS)``,
  ``n`` = bars already observed in the CURRENT pending episode (0 on the
  bar that opens it -- widest, most uncertain setting; the state-dependent
  piece the direction doc calls for). ``shrink`` decays from 1.0 toward a
  strictly positive floor ``SHRINK_FLOOR=0.25`` (never collapsing the
  mixture to a point mass at zero, which would be a degenerate H1) with a
  half-life ``N_HALF_BARS=12`` (~1 hour at 5-minute cadence). JUSTIFICATION
  for shrinking DOWNWARD with age: a genuinely large, fast drift tends to
  either resolve the episode (grant the increase) or get cancelled by
  v4's own vote/scale before many bars pass; an episode still pending
  after many bars is progressively weaker evidence that a LARGE drift is
  what is actually in play, so the RANGE of magnitudes this branch is
  willing to entertain narrows toward its low end as the episode ages --
  exactly mirroring, in spirit, an alpha-spending-like sequential design,
  but expressed here as a shrinking composite alternative rather than a
  shrinking rejection boundary.

WHY THIS IS STILL A VALID TEST MARTINGALE UNDER H0 (mu=0), even though
``mu_j(n)`` changes bar to bar within an episode (unlike R-174's fixed-TAU
recursion, which corresponds to ONE coherent joint Bayesian model for the
whole episode): the general result (Shafer, Shen, Vereshchagin & Vovk,
2011, *Statistical Science* 26(1), 84-101, arXiv:0912.4269 -- already cited,
independently confirmed, and reused from R-174) requires only that at each
step ``i``, ``f_1,i`` is SOME valid probability density in ``r_i`` (any
Gaussian, in particular, integrates to 1) whose parameters (here, ``mu_j(n)``
and ``sigma_i``) are ``F_{i-1}``-measurable -- known strictly BEFORE ``r_i``
is observed. ``n`` is a deterministic count of bars already elapsed in the
episode (never a function of any not-yet-observed return) and ``sigma_i``
is R-181's own TSRV-corrected, already-causal per-bar sigma
(``r181_shared.causal_bar_sigma_tsrv``, self-tested and READ-ONLY here) --
so ``mu_j(n)`` is F_{i-1}-measurable regardless of how it was chosen. This
does NOT require the sequence ``{f_1,i}`` to trace back to one single fixed
joint prior the way R-174's conjugate recursion does; ANY predictable
one-step forecaster gives ``E_{H0}[f_1,i(r_i)/f_0,i(r_i) | F_{i-1}] =
integral f_1,i(r) dr = 1`` exactly, so the running product
``e_t = prod_i f_1,i(r_i)/f_0,i(r_i)`` is still a nonnegative martingale
with ``e_0=1`` under H0, and each of the J point-mass components is such a
product. A FIXED (data-independent) convex combination of martingales is
itself a martingale (textbook), so the overall mixture
``E_t = sum_j pi_j * e_{j,t}`` is too, and Ville's inequality (Ville, 1939,
also reused from R-174) applies to it directly: thresholding accept at
``log(E_t) >= log(1/alpha)`` is anytime-valid at level ``alpha`` for every
stopping rule, including ``run_asymmetric_gate``'s own "abandon a pending
episode the moment v4's own request drops back below the deadband".

Grunwald, de Heide & Koolen (2024), "Safe Testing", *JRSS-B* 86(5),
1091-1128, describe GRO/GROW e-variables for a composite alternative
exactly as Bayes-factor mixtures over the alternative's free parameter
(reused from R-174's own citation); this file discretizes that mixture
over drift VALUES (point masses) rather than integrating a single
continuous Gaussian in closed form, which the direction doc's own language
explicitly allows ("a discretized or analytically-integrated Gaussian
mixture prior").

DISCLOSED DESIGN CHOICE -- no formal reject boundary, exactly R-174's own
disclosed choice and for the identical reason: a one-sided Ville threshold
has no symmetric lower boundary with a matching error-control guarantee.
``step_fn`` below returns ONLY "pending" or "accept", NEVER "reject";
``run_asymmetric_gate``'s own bookkeeping (abandon a pending episode with
no grant the moment ``desired[i]`` drops back to ``<= pos + deadband``) is
what keeps an unresolved episode from running forever, exactly as in R-174.

DISCLOSED LIMITATION -- the weights ``pi_j proportional to 1/m_j`` and the
shrink schedule (``SHRINK_FLOOR``, ``N_HALF_BARS``) are this branch's own
design choices, not derived from data or optimized post hoc (chosen before
any real-data run of this file); ``WIDTH_SCALE`` is the swept free
parameter this branch's own promotion-bar plateau check is evaluated over,
exactly matching the direction doc's "sweeping your own free parameter
(mixture width...)" instruction.

USAGE
-----
    uv run python experiments/r181_novel_tsrv_grow.py
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

from experiments.r181_shared import (  # noqa: E402
    ALPHA_PRIMARY,
    BARS_PER_DAY,
    FUTURES,
    GATE_MIN_DELAYS,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TAU_PER_BAR,
    TSRV_MIN_REDUCTION,
    assert_no_holdout,
    causal_bar_returns,
    causal_bar_sigma_tsrv,
    causal_truncation_probe_series,
    compare,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    run_asymmetric_gate,
    two_scale_realized_variance,
    two_scale_realized_variance_naive,
    v4_raw_desired,
    v4_target,
)

LOG_2PI = math.log(2.0 * math.pi)


def hr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ================================================================== (0)
# This branch's own frozen constants (chosen before any real-data run of
# this file, per Step 1's own discipline). WINDOW_BARS_* feed the
# non-degeneracy precheck below; the precheck's own outcome determines
# which one is actually used for every later step (fixed, not re-swept).
# ==================================================================
WINDOW_BARS_PRIMARY = BARS_PER_DAY
WINDOW_BARS_RETRY = BARS_PER_DAY // 2

M_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)   # multiples of TAU_PER_BAR
_pi_raw = tuple(1.0 / m for m in M_GRID)
_pi_sum = sum(_pi_raw)
PI_WEIGHTS = tuple(w / _pi_sum for w in _pi_raw)
LOG_PI = tuple(math.log(w) for w in PI_WEIGHTS)
assert abs(sum(PI_WEIGHTS) - 1.0) < 1e-12, PI_WEIGHTS

SHRINK_FLOOR = 0.25       # the mixture never collapses fully to a point at 0
N_HALF_BARS = 12          # ~1 hour at 5-min cadence; age-shrink half-life

WIDTH_SCALE_GRID = (0.5, 1.0, 2.0)   # this branch's own swept free parameter


def shrink(n: int) -> float:
    return SHRINK_FLOOR + (1.0 - SHRINK_FLOOR) * math.exp(-n / N_HALF_BARS)


def _logsumexp(xs) -> float:
    m = max(xs)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ================================================================== (1)
# The GROW mixture e-variable accumulator. One instance per
# (alpha, width_scale). State = (n, tuple of J per-component running
# log-e values). log_e_j starts at 0.0 (e_j,0 = 1, no evidence yet).
# ==================================================================

def make_grow_mixture_step(alpha: float, width_scale: float):
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    log_thresh = -math.log(alpha)
    mu_base = tuple(TAU_PER_BAR * width_scale * m for m in M_GRID)
    J = len(M_GRID)

    def new_episode_state():
        return (0, tuple(0.0 for _ in range(J)))

    def step_fn(state, ri, si):
        n, logs = state
        var0 = si * si
        sh = shrink(n)
        new_logs = []
        log_terms = []
        for j in range(J):
            mu = mu_base[j] * sh
            # logpdf_N(ri; mu, var0) - logpdf_N(ri; 0, var0), algebraically
            # simplified (the H0 term cancels exactly since both share the
            # same variance var0): (ri*mu - 0.5*mu*mu) / var0.
            inc = (ri * mu - 0.5 * mu * mu) / var0
            lej = logs[j] + inc
            new_logs.append(lej)
            log_terms.append(LOG_PI[j] + lej)
        log_e_mix = _logsumexp(log_terms)
        decision = "accept" if log_e_mix >= log_thresh else "pending"
        return (n + 1, tuple(new_logs)), decision

    return new_episode_state, step_fn


# ================================================================== (2)
# Content-keyed caches -- TSRV sigma is expensive (O(n * window_bars)) and
# does NOT depend on width_scale/alpha, so it is cached separately from
# the gate output (which does), avoiding redundant recomputation across
# compare()'s 2x market-pair redundancy AND across the width_scale sweep.
# ==================================================================

_SIGMA_CACHE: dict[tuple, np.ndarray] = {}
_COMPONENT_CACHE: dict[tuple, tuple] = {}


def _frame_key(df: pd.DataFrame) -> tuple:
    idx = df.index
    c = df["close"].to_numpy()
    return (int(idx[0].value), int(idx[-1].value), len(df), float(np.sum(c)), float(c[-1]))


def _cached_sigma_tsrv(df: pd.DataFrame, window_bars: int) -> np.ndarray:
    key = (_frame_key(df), int(window_bars))
    if key not in _SIGMA_CACHE:
        _SIGMA_CACHE[key] = causal_bar_sigma_tsrv(df, window_bars=window_bars)
    return _SIGMA_CACHE[key]


def gro_components(df: pd.DataFrame, width_scale: float, window_bars: int,
                   alpha: float = ALPHA_PRIMARY) -> tuple[np.ndarray, int, int]:
    key = (_frame_key(df), round(width_scale, 6), int(window_bars), round(alpha, 6))
    if key not in _COMPONENT_CACHE:
        desired = v4_raw_desired(df)
        r = causal_bar_returns(df)
        sigma = _cached_sigma_tsrv(df, window_bars)
        new_ep, step = make_grow_mixture_step(alpha, width_scale)
        gated, delayed, total = run_asymmetric_gate(desired, r, sigma, new_ep, step)
        _COMPONENT_CACHE[key] = (gated, delayed, total)
    return _COMPONENT_CACHE[key]


def build_candidate_target(df: pd.DataFrame, width_scale: float, window_bars: int) -> np.ndarray:
    gated, _delayed, _total = gro_components(df, width_scale, window_bars)
    return gated


def make_build_target(width_scale: float, window_bars: int):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return build_candidate_target(df, width_scale, window_bars)
    _build.__name__ = f"tsrv_grow_ws{width_scale:g}_w{window_bars}"
    return _build


# ================================================================== (3)
# Step 1's own non-degeneracy precheck: does TSRV read materially (>=10%)
# below the naive same-window RV on the MAJORITY of finite windows, on
# BTC inner-train? If not at window_bars=BARS_PER_DAY, one disclosed retry
# at BARS_PER_DAY // 2 is allowed before concluding NEGATIVE.
# ==================================================================

def tsrv_reduction_fraction(df: pd.DataFrame, window_bars: int) -> tuple[float, int]:
    """Fraction of finite (tsrv, naive) window pairs where TSRV reads
    ``>= TSRV_MIN_REDUCTION`` below naive RV, and the count of such finite
    windows (the denominator)."""
    tsrv = two_scale_realized_variance(df, window_bars=window_bars)
    naive = two_scale_realized_variance_naive(df, window_bars=window_bars)
    m = np.isfinite(tsrv) & np.isfinite(naive) & (naive > 0)
    n_finite = int(m.sum())
    if n_finite == 0:
        return float("nan"), 0
    reduced = tsrv[m] <= naive[m] * (1.0 - TSRV_MIN_REDUCTION)
    return float(np.mean(reduced)), n_finite


# ================================================================== (4)
# Kill switches A1/A2, identical numeric bars to the conservative branch's,
# applied per tested WIDTH_SCALE value at the precheck-fixed window_bars.
# ==================================================================

def kill_switch_row(btc_inner_val: pd.DataFrame, width_scale: float, window_bars: int) -> dict:
    gated_val, delayed_val, total_val = gro_components(btc_inner_val, width_scale, window_bars)
    r2_val = r_squared(gated_val, v4_target(btc_inner_val))
    a1_pass = delayed_val >= GATE_MIN_DELAYS
    a2_pass = not (np.isfinite(r2_val) and r2_val > R2_DEGENERACY_THRESH)
    return dict(width_scale=width_scale, delayed_val=delayed_val, total_val=total_val,
                r_sq_val=r2_val, a1_pass=a1_pass, a2_pass=a2_pass,
                kill_pass=bool(a1_pass and a2_pass))


# ================================================================== (5)
# Promotion bar, restated (not imported) exactly as r181_direction.md's
# novel-branch clause (3) / conservative clause (5) reads: >=1 tested value
# clears dSharpe>=+0.2 risk-matched on BOTH spot and futures_5x inner-val,
# OR a >=5pp risk-matched drawdown cut with paired 95% CI excluding zero;
# ETH-replication must not sign-flip; plateau across >=2 of the tested
# values (not a spike at one).
# ==================================================================

def evaluate_promotion(rows_by_value: dict[float, list[dict]]) -> dict:
    per_value = {}
    for wv, rows in rows_by_value.items():
        inner_val = {r["market"]: r for r in rows if r["slice"] == "inner_val"}
        eth = {r["market"]: r for r in rows if r["slice"] == "eth_replication"}
        market_detail = {}
        for market in ("spot", "futures_5x"):
            iv = inner_val.get(market)
            et = eth.get(market)
            if iv is None or et is None:
                market_detail[market] = dict(passes=False, note="missing row")
                continue
            sharpe_edge = bool(iv["risk_matched"] and iv["d_sharpe"] >= SHARPE_NOISE_FLOOR)
            dd_edge = bool(iv["risk_matched"] and iv["d_dd"] <= -5.0 and iv["excludes_zero"])
            edge = sharpe_edge or dd_edge
            fired_via = "sharpe" if sharpe_edge else ("dd" if dd_edge else None)
            sign_ok = False
            if fired_via == "sharpe":
                sign_ok = bool(et["d_sharpe"] > 0)
            elif fired_via == "dd":
                sign_ok = bool(et["risk_matched"] and et["d_dd"] < 0)
            market_detail[market] = dict(sharpe_edge=sharpe_edge, dd_edge=dd_edge, edge=edge,
                                         fired_via=fired_via, sign_ok=sign_ok,
                                         passes=bool(edge and sign_ok))
        both = all(market_detail[m].get("passes", False) for m in ("spot", "futures_5x"))
        per_value[wv] = dict(market=market_detail, both_markets_pass=both)
    passing_values = [wv for wv, d in per_value.items() if d["both_markets_pass"]]
    return dict(per_value=per_value, passing_values=passing_values,
                plateau=len(passing_values) >= 2)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    # (a) construction sanity: log_e starts at exactly 0 for every
    # component, for every width_scale, and shrink(0) == 1.0 exactly.
    assert abs(shrink(0) - 1.0) < 1e-12, shrink(0)
    assert shrink(10_000) > SHRINK_FLOOR - 1e-9 and shrink(10_000) < SHRINK_FLOOR + 1e-6
    for ws in WIDTH_SCALE_GRID:
        new_ep, _step = make_grow_mixture_step(ALPHA_PRIMARY, ws)
        n0, logs0 = new_ep()
        assert n0 == 0 and all(le == 0.0 for le in logs0), (ws, n0, logs0)

    # shrink is monotone non-increasing in n and stays strictly positive.
    ns = list(range(0, 200, 5))
    shr = [shrink(n) for n in ns]
    assert all(shr[i] >= shr[i + 1] - 1e-12 for i in range(len(shr) - 1)), shr
    assert all(s > 0.0 for s in shr), shr

    # (b) MANDATORY: causal truncation probe on the candidate build, at the
    # primary width_scale, using the PRIMARY window_bars, on synthetic data
    # long enough for TSRV windows to be finite through most of the frame.
    idx = pd.date_range("2017-01-01", periods=BARS_PER_DAY * 40, freq="5min", tz="UTC")
    rng = np.random.default_rng(181_01)
    innov = rng.normal(0.00002, 0.0006, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    synth_df = pd.DataFrame({"open": close, "high": high, "low": low,
                             "close": close, "volume": 1.0}, index=idx)

    _wrapper = make_build_target(1.0, WINDOW_BARS_PRIMARY)
    assert causal_truncation_probe_series(_wrapper, synth_df), \
        "causal_truncation_probe_series FAILED on build_candidate_target"

    # (c) directional numeric sanity on the ISOLATED accumulator: strongly
    # POSITIVE drift must trend the mixture toward accept; strongly
    # NEGATIVE drift of the SAME magnitude must NOT (unlike R-174's
    # symmetric N(0,TAU^2) mixture) -- this branch's mixture is one-sided
    # by construction (all m_j > 0), verified here rather than merely
    # asserted from the math.
    new_ep, step = make_grow_mixture_step(ALPHA_PRIMARY, 1.0)
    sigma_fixed = 8.0e-4
    n_steps, n_reps = 4_000, 150

    def _mean_log_e(mu: float, seed: int) -> float:
        rng_l = np.random.default_rng(seed)
        totals = np.empty(n_reps)
        for rep in range(n_reps):
            state = new_ep()
            draws = rng_l.normal(mu, sigma_fixed, n_steps)
            for ri in draws:
                state, _decision = step(state, float(ri), sigma_fixed)
            n_, logs_ = state
            log_terms = [LOG_PI[j] + logs_[j] for j in range(len(M_GRID))]
            totals[rep] = _logsumexp(log_terms)
        return float(totals.mean())

    mean0 = _mean_log_e(0.0, seed=181_10)
    mean_pos = _mean_log_e(TAU_PER_BAR * 4.0, seed=181_11)
    mean_neg = _mean_log_e(-TAU_PER_BAR * 4.0, seed=181_12)
    print(f"[r181 novel self-test] mean(log_E_mixture) after {n_steps} bars, {n_reps} reps: "
          f"mu=0: {mean0:+.4f}  mu=+4*TAU: {mean_pos:+.4f}  mu=-4*TAU: {mean_neg:+.4f}")
    assert mean_pos > mean0, (mean_pos, mean0)
    assert mean_neg <= mean0 + 1.0, (mean_neg, mean0)  # negative drift must NOT trend toward accept
    assert mean_pos > mean_neg + 1.0, (mean_pos, mean_neg)  # one-sided: clearly asymmetric

    # (d) tsrv_reduction_fraction runs and returns a plausible value on a
    # slice of real BTC inner-train data (does not assert the >50% bar
    # itself here -- that is Step 1's own precheck in main(), measured on
    # real data and reported honestly either way).
    btc = load_btc()
    sample = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    frac, n_finite = tsrv_reduction_fraction(sample.iloc[:BARS_PER_DAY * 60], WINDOW_BARS_PRIMARY)
    assert n_finite > 100, n_finite
    assert 0.0 <= frac <= 1.0, frac

    print("r181_novel_tsrv_grow self-test OK.")


_self_test()


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []
    n_configs = 0  # project convention: one (value) through compare() = one config

    hr("R-181 NOVEL: TSRV-corrected GROW/safe-testing MIXTURE e-value gating "
       "kelly_regime_v4 INCREASES")
    print("mechanism: v4's own frac*scale re-sizing decision is gated, on bars where it would")
    print("INCREASE exposure only, behind a sequential mixture-of-point-mass Bayes factor")
    print("(H0: mu=0; H1: a fixed-weight mixture over mu_j(n) = TAU_PER_BAR*WIDTH_SCALE*m_j*shrink(n),")
    print("m_j in M_GRID, shrink(n) narrowing the entertained range as the pending episode ages),")
    print("thresholded at log(1/alpha) per Ville's inequality -- fed R-181's own TSRV-corrected")
    print("per-bar sigma instead of raw per-bar sigma. Decreases stay immediate (unchanged).")
    print(f"\nALPHA_PRIMARY (frozen, r181_shared/r174_shared) = {ALPHA_PRIMARY}")
    print(f"TAU_PER_BAR (frozen, r181_shared, = inner-train mean per-bar log-return) = "
          f"{TAU_PER_BAR:.6e}")
    print(f"M_GRID = {M_GRID}   PI_WEIGHTS = {tuple(round(w, 4) for w in PI_WEIGHTS)}")
    print(f"SHRINK_FLOOR = {SHRINK_FLOOR}   N_HALF_BARS = {N_HALF_BARS}")
    print(f"WIDTH_SCALE_GRID (this branch's own swept free parameter) = {WIDTH_SCALE_GRID}")

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
    btc_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    btc_inner_val = btc.loc[INNER_VAL_START:INNER_VAL_END]
    print(f"BTC inner-train slice: {len(btc_train):,} bars, "
          f"{btc_train.index[0]} -> {btc_train.index[-1]}")
    print(f"BTC inner-validation slice: {len(btc_inner_val):,} bars, "
          f"{btc_inner_val.index[0]} -> {btc_inner_val.index[-1]}")

    # ============================================================= STEP 1
    hr("STEP 1 -- NON-DEGENERACY PRECHECK: does TSRV read >=10% below naive RV on the "
       "MAJORITY (>50%) of finite windows, BTC inner-train?")
    t_pre = time.time()
    frac_primary, n_finite_primary = tsrv_reduction_fraction(btc_train, WINDOW_BARS_PRIMARY)
    print(f"  window_bars={WINDOW_BARS_PRIMARY} (BARS_PER_DAY): "
          f"fraction={frac_primary:.4f}  (n_finite_windows={n_finite_primary})  "
          f"[{time.time() - t_pre:.1f}s]")
    precheck_window = WINDOW_BARS_PRIMARY
    precheck_frac = frac_primary
    precheck_n_finite = n_finite_primary
    retried = False
    precheck_pass = frac_primary > 0.5
    if not precheck_pass:
        retried = True
        print(f"  fraction {frac_primary:.4f} is NOT > 0.5 at window_bars={WINDOW_BARS_PRIMARY} -- "
              f"DISCLOSED RETRY per r181_direction.md's own allowance, at "
              f"window_bars={WINDOW_BARS_RETRY} (BARS_PER_DAY // 2)")
        t_pre2 = time.time()
        frac_retry, n_finite_retry = tsrv_reduction_fraction(btc_train, WINDOW_BARS_RETRY)
        print(f"  window_bars={WINDOW_BARS_RETRY}: fraction={frac_retry:.4f}  "
              f"(n_finite_windows={n_finite_retry})  [{time.time() - t_pre2:.1f}s]")
        precheck_window = WINDOW_BARS_RETRY
        precheck_frac = frac_retry
        precheck_n_finite = n_finite_retry
        precheck_pass = frac_retry > 0.5

    print(f"\nPRECHECK RESULT: {'PASS' if precheck_pass else 'FAIL'} "
          f"(fraction={precheck_frac:.4f} at window_bars={precheck_window}, "
          f"retried={retried})")

    if not precheck_pass:
        hr("VERDICT")
        print("Non-degeneracy precheck FAILS at BOTH window_bars tried "
              f"(BARS_PER_DAY={WINDOW_BARS_PRIMARY}: fraction={frac_primary:.4f}; "
              f"BARS_PER_DAY//2={WINDOW_BARS_RETRY}: fraction={precheck_frac:.4f}), "
              "neither exceeding 0.5.")
        print("No measurable noise-correction headroom on this data at either window length --")
        print("per r181_direction.md's frozen novel-branch rule, STOPPING HERE. The GROW/mixture")
        print("engine is fully implemented above (and self-tested) but is NOT run on real")
        print("compare() data past this point -- no kill switches, no promotion-bar comparison.")
        verdict = "NEGATIVE (stopped at non-degeneracy precheck)"
        print(f"\nVERDICT: {verdict}")
        max_ts = max(max_ts_seen)
        print(f"\nmax timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(precheck_pass=False, frac_primary=frac_primary, frac_retry=precheck_frac,
                    retried=retried, verdict=verdict, n_configs=0, max_ts=max_ts)

    WINDOW_BARS = precheck_window  # fixed for every later step, per the precheck's own outcome

    # ============================================================= STEP 2
    hr(f"STEP 2 -- KILL SWITCHES per tested WIDTH_SCALE value (window_bars={WINDOW_BARS} fixed "
       "from the precheck above)")
    kill_rows = []
    for ws in WIDTH_SCALE_GRID:
        t_ks = time.time()
        kr = kill_switch_row(btc_inner_val, ws, WINDOW_BARS)
        kill_rows.append(kr)
        print(f"  width_scale={ws:.2f}  delayed[inner_val]={kr['delayed_val']:3d} "
              f"(of {kr['total_val']} episodes, need >={GATE_MIN_DELAYS})  "
              f"A1={'PASS' if kr['a1_pass'] else 'FAIL'}   "
              f"r_sq[inner_val]={kr['r_sq_val']:.5f}  A2={'PASS' if kr['a2_pass'] else 'FAIL'}   "
              f"KILL_PASS={'PASS' if kr['kill_pass'] else 'FAIL'}  [{time.time() - t_ks:.1f}s]")
    any_kill_pass = any(kr["kill_pass"] for kr in kill_rows)
    print(f"\nAny width_scale clears BOTH A1 and A2 on BTC inner-validation: "
          f"{'YES' if any_kill_pass else 'NO'}")

    if not any_kill_pass:
        hr("VERDICT")
        print("Both kill switches (A1: gate delays >=3 episodes; A2: gated series is not an")
        print(f"R^2-degenerate copy of v4_target, R^2 < {R2_DEGENERACY_THRESH}) FAIL for EVERY")
        print("tested width_scale value. Per r181_direction.md's frozen novel-branch rule,")
        print("STOPPING HERE -- no compare() sweep is run.")
        verdict = "NEGATIVE (stopped at kill switches, all width_scale values fail)"
        print(f"\nVERDICT: {verdict}")
        max_ts = max(max_ts_seen)
        print(f"\nmax timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(precheck_pass=True, precheck_frac=precheck_frac, precheck_window=WINDOW_BARS,
                    retried=retried, kill_rows=kill_rows, any_kill_pass=False,
                    verdict=verdict, n_configs=0, max_ts=max_ts)

    passing_ws = [kr["width_scale"] for kr in kill_rows if kr["kill_pass"]]

    # ============================================================= STEP 3
    hr(f"STEP 3 -- FULL SWEEP: compare() for each width_scale clearing kill switches "
       f"({len(passing_ws)} of {len(WIDTH_SCALE_GRID)}: {passing_ws})")
    all_rows: dict[float, list[dict]] = {}
    for ws in passing_ws:
        label = f"tsrv_grow_ws{ws:g}"
        print(f"\n--- width_scale={ws} (window_bars={WINDOW_BARS}) ---")
        t_cfg = time.time()
        rows = compare(lambda df, w=ws: build_candidate_target(df, w, WINDOW_BARS),
                       label=label, btc=btc, eth=eth, markets=(SPOT, FUTURES), include_eth=True)
        dt_cfg = time.time() - t_cfg
        print_rows(rows)
        print(f"  [{dt_cfg:.1f}s for this config]")
        n_configs += 1
        all_rows[ws] = rows

    n_backtest_cells = len(passing_ws) * 2 * 3  # width_values(passed) x markets x slices

    # ============================================================= STEP 4
    hr("STEP 4 -- PROMOTION BAR")
    promo = evaluate_promotion(all_rows)
    for ws, d in promo["per_value"].items():
        for market, md in d["market"].items():
            print(f"  width_scale={ws}  market={market:10s}  {md}")
        print(f"  width_scale={ws}  BOTH MARKETS PASS: {d['both_markets_pass']}")
    print(f"\nValues clearing the promotion bar (sharpe/dd edge + ETH no-sign-flip, "
          f"BOTH markets): {promo['passing_values'] if promo['passing_values'] else 'NONE'}")
    print(f"Plateau (>= 2 of the {len(WIDTH_SCALE_GRID)} tested width_scale values pass): "
          f"{promo['plateau']}")

    # ============================================================= STEP 5
    hr("STEP 5 -- CONFIGURATION COUNT")
    print(f"width_scale values tested (precheck+kill-switch stage): {len(WIDTH_SCALE_GRID)} "
          f"({WIDTH_SCALE_GRID})")
    print(f"width_scale values clearing kill switches (proceeded to compare()): "
          f"{len(passing_ws)} ({passing_ws})")
    print(f"configs run through compare() (project convention, 1 value = 1 config): {n_configs}")
    print(f"backtest cells (width_scale x market x slice, this file's own granular count): "
          f"{n_backtest_cells}  ({len(passing_ws)} values x 2 markets x 3 slices)")

    # ============================================================= VERDICT
    hr("VERDICT")
    verdict = ("PROMOTE-CANDIDATE (kill switches clear, plateau across >=2 tested values; "
               "holdout may be consulted)"
               if promo["plateau"] else "NEGATIVE")
    print(f"non-degeneracy precheck:                                   PASS "
          f"(fraction={precheck_frac:.4f} at window_bars={WINDOW_BARS}, retried={retried})")
    print(f"kill switches clear for >=1 width_scale:                   {any_kill_pass} "
          f"({passing_ws})")
    print(f"width_scale values clearing the promotion bar (both mkts): "
          f"{promo['passing_values'] if promo['passing_values'] else 'NONE'}")
    print(f"plateau (>=2 of {len(WIDTH_SCALE_GRID)} tested values):    {promo['plateau']}")
    print(f"\nVERDICT: {verdict}")
    if verdict.startswith("PROMOTE"):
        print("\nGate clears the precheck + kill switches + promotion bar on a plateau of >=2")
        print("width_scale values -- holdout MAY be consulted per docs/ROUTINE.md step 4. This")
        print("file does NOT itself read OOS_START; that is a separate, explicit step left to")
        print("the operator.")
    else:
        print("\nGate does NOT clear the full promotion bar on a plateau of >=2 tested values --")
        print("per docs/ROUTINE.md's own discipline, the holdout is precious and is NOT touched.")
        print("No bar at or after OOS_START is read anywhere in this file.")

    max_ts = max(max_ts_seen)
    print(f"\nmax timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(precheck_pass=True, precheck_frac=precheck_frac, precheck_window=WINDOW_BARS,
                retried=retried, kill_rows=kill_rows, any_kill_pass=any_kill_pass,
                passing_ws=passing_ws, all_rows=all_rows, promo=promo,
                n_configs=n_configs, n_backtest_cells=n_backtest_cells,
                verdict=verdict, max_ts=max_ts)


if __name__ == "__main__":
    main()
