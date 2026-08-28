"""Shared, read-only pre-registration for the R-171 round (08-28).

DIRECTION, one sentence: replace `kelly_regime_v4`'s conditional
volatility-target `scale` factor (`min(target_vol/vol, max_leverage)`
under the v3 hysteresis vol-state machine) with a scalar `b_t` learned
online, causally, bar-by-bar, by the Online Newton Step (ONS) algorithm
applied to the already-decided vote's own realized bet-payoff -- i.e. the
single-asset (leverage-vs-cash) special case of Agarwal, A., Hazan, E.,
Kale, S. & Schapire, R. E. (2006), "Algorithms for Portfolio Management
Based on the Newton Method," ICML 2006, 9-16.

Full Step 1/Step 2 design (constraint attacked [N~=3, secondarily SIZE],
non-duplication against L-11/R-149/R-28/R-31/R-152/R-153/R-93's
Grossman-Zhou closure/R-164/R-166/the CPPI pair/Busseti-Ryu-Boyd/
CRRA-Merton/"per-vote-state Kelly fraction"/the walk-forward-refit and
minimax-reselection closures/R-147, simulability, named failure modes,
noise-floor arithmetic, the pre-registered promotion rule) is in
`docs_scratch_direction.md` at the repo root, written by the research
sub-agent that proposed this round and reviewed by the operator before
this file was frozen. Read that file for the full argument; this module
is the FROZEN, executable form of ONLY its shared primitives -- written
by the operator BEFORE either branch is dispatched, and neither branch
may edit it or each other's file (R-89-through-R-170's own convention).
This module implements NEITHER branch's strategy variant (conservative:
one shared ONS accumulator; novel: three vol-state-conditioned "sleeping
expert" accumulators) -- that is each branch's own job, on top of the
`ons_scale` primitive below.

============================================================================
UNIT CHOICE, disclosed up front (design doc's own flagged risk: "exactly
the kind of silent unit mismatch this codebase's ledger has been burned by
before"): `r_t` -- the vote's own realized bet-payoff each bar, the
quantity ONS's gradient/Hessian are taken of -- is `frac_t * SIMPLE
(arithmetic) return_t`, NOT log return. The paper's (and Hazan's textbook
ONS chapter's) loss is `-log(1 + b*r)` with `r` a SIMPLE price-relative-1
return: wealth compounds as `prod_t (1 + b*r_t)`, which is only the
correct multiplicative wealth update when `r_t` is a simple return.
Using a log return here would silently double-log-transform the wealth
recursion. Every `ret` array passed to `ons_scale` below must be a SIMPLE
return; `asset_simple_return` is provided to compute it causally and
correctly (bar t's own `close_t/close_{t-1} - 1`, NOT `.shift(1)` --
`ret_t` plays the role of a REALIZED OUTCOME observed at the close of bar
t, exactly like `r = np.log(close).diff()` inside `v4_symmetric_vol`
before ITS OWN `.shift(1)` is applied to turn it into a pre-bar-t decision
input; ONS's own causality comes from using `ret_s, s < t` to produce
`b_t`, not from shifting `ret` itself -- see the docstring of `ons_scale`).

============================================================================
PAPER-VS-TEXTBOOK PROVENANCE OF eps/beta -- READ THIS BEFORE TRUSTING A
NUMBER BELOW. The Agarwal-Hazan-Kale-Schapire (2006) paper's own Theorem 1
was located and read in full (icml.pdf, cis.upenn.edu/~mkearns/finread/
icml.pdf, Sections 2-3): its ONS is parameterized by a MARKET VARIABILITY
PARAMETER `alpha_mkt` (a lower bound on every scaled price-relative
`r_t(j) > 0`, "no stock crashes to zero"), over an n-STOCK PROBABILITY
SIMPLEX with price-relative vectors normalized so `max_j r_t(j) = 1`, and
sets `beta = alpha_mkt / (8*sqrt(n))` (eta=0, delta=1 for the pure
logarithmic-regret guarantee; the paper's OWN experimental section in
fact uses different, hand-set values, beta=1, delta=1/8, "for the purpose
of experimentation" -- i.e. even the paper's authors did not use their
own worst-case-derived beta in practice). None of this maps cleanly onto
this round's problem: a SINGLE scalar `b in [0, max_leverage]` (leverage
of one already-decided vote vs. cash, not an n>=2-stock simplex), with
SIGNED simple returns (can be negative -- no "no-junk-bond" floor, no
max-normalized-to-1 rescaling). Per the design doc's own explicit
fallback instruction, this module therefore uses the general
Online-Newton-Step-for-exp-concave-losses form from Hazan, E.,
"Introduction to Online Convex Optimization" (2nd ed., 2023 revision,
arXiv:1909.05207 -- also the monograph version, Foundations and Trends
in Machine Learning), Chapter 4 ("Second-Order Methods"), Algorithm 12 &
Theorem 4.5, LOCATED AND READ IN FULL (not guessed): for a decision set
of diameter `D`, a per-round loss `f_t` with `alpha`-exp-concavity
constant `alpha` and gradient bound `G`, Algorithm 12 run with
    gamma = (1/2) * min(1/(4*G*D), alpha)      [the update's own step size]
    eps   = 1 / (gamma**2 * D**2)              [A_0 = eps, i.e. a_0 in 1-D]
guarantees `Regret_T <= (1/alpha + G*D) * n * log T`. This is the SAME
functional form the design doc itself specified as an acceptable
fallback (the design doc's own "gamma"/"beta" names map to this module's
`alpha`/`beta = gamma` respectively -- the design doc's draft formula
`beta = (1/2)*min(1/(4GD), gamma)` double-applies the leading 1/2 and is
superseded here by Theorem 4.5's exact statement, a tightening made
BEFORE freezing per ROUTINE.md's fix-before-freeze allowance, not a
loosening after a result was seen).

`alpha` itself is NOT assumed -- it is computed exactly for this specific
scalar loss. For `f_t(b) = -log(1 + b*r_t)`:
    f_t'(b)  = -r_t / (1 + b*r_t)
    f_t''(b) =  r_t^2 / (1 + b*r_t)^2  =  [f_t'(b)]^2
i.e. `f_t''(b) = 1 * [f_t'(b)]^2` EXACTLY, for every b where `1+b*r_t>0`
-- so by Lemma 4.2's defining inequality (`nabla^2 f >= alpha * nabla f
nabla f^T`), `f_t` is exactly 1-exp-concave: `alpha = 1`, an exact
elementary-calculus fact for this one-dimensional log-loss, not a
plugged-in guess.

`G` -- the gradient bound -- IS data-derived, per the design doc's own
instruction, from BTC bars strictly before `INNER_VAL_START` (2021-01-01,
i.e. inner-train only; inner-validation and the holdout are never read
for this), post-warmup (matching `kelly_regime_v4.warmup =
80*BARS_PER_DAY+10`, the same warmup `TargetStrategy` defaults to).
`G = max_t sup_{b in [0,D]} |f_t'(b)|`; since `|r/(1+br)|` is monotone in
`b` for fixed-sign `r` (increasing `b` shrinks it for `r>0`, grows it for
`r<0`, up to the bankruptcy boundary), the sup over `b` is attained at an
endpoint, `b=0` or `b=D`, so `G = max(max_t|r_t|, max_t |r_t|/(1+D*r_t))`
over the safe (non-bankrupt-at-D) bars. Reproducible one-off derivation
(run once, before either branch was dispatched; NOT re-run at import --
this module never calls `load_btc()` at module scope, matching every
prior frozen module's self-test-on-synthetic-data convention):

    df = load_btc()
    train = df[df.index < pd.Timestamp(INNER_VAL_START, tz="UTC")]
    train = train.iloc[80 * BARS_PER_DAY + 10:]        # post-warmup
    frac = v4_vote_frac(train).to_numpy()
    r = frac * asset_simple_return(train)
    r = r[np.isfinite(r)]                              # n = 397,717 bars
    D = V4_MAX_LEVERAGE                                 # 2.0
    g0 = np.abs(r)                                      # |grad| at b=0
    denom_D = 1 + D * r
    gD = np.abs(r[denom_D > 1e-6] / denom_D[denom_D > 1e-6])   # |grad| at b=D
    G = max(g0.max(), gD.max())                          # = 0.09843034663...
    # 0 bars were within 1e-6 of bankruptcy at b=D on this window.

Final numeric values (BTC, inner-train, frozen, used identically -- not
re-derived -- for ETH by the conservative branch, matching this arm's
own "parameter-light, not swept" design):
    G_BTC_INNER_TRAIN = 0.09843034663178550   (see derivation above)
    D  = V4_MAX_LEVERAGE = 2.0
    4*G*D = 0.7874427730542837  =>  1/(4*G*D) = 1.2699335548... > 1
    alpha = 1                        (exact, see derivation above)
    ONS_BETA_BTC = (1/2)*min(1/(4*G*D), alpha) = (1/2)*min(1.2699..., 1) = 0.5
    ONS_EPS_BTC  = 1 / (ONS_BETA_BTC**2 * D**2) = 1 / (0.25*4) = 1.0
`b_0` (design doc: "a neutral, non-fitted value... do not tune it") is
set to the domain midpoint, `max_leverage/2 = 1.0` -- both of the design
doc's suggested choices ("max_leverage/2, or 1.0") coincide numerically
at `V4_MAX_LEVERAGE=2.0`, so there is no free choice being disguised
here.

============================================================================
SHARED-TOOLKIT CHAIN -- ONE DISCLOSED GAP, fixed before freezing (per
ROUTINE.md's fix-before-freeze allowance): `V4_MAX_LEVERAGE` and
`V4_TARGET_VOL` are defined in `r102_shared.py` but the re-export chain
r102 -> r103 -> r104 -> r105 -> r147 -> r161 carries `V4_VOL_SPAN` past
r102 (via r103) but NOT `V4_MAX_LEVERAGE`/`V4_TARGET_VOL` -- verified by
`grep -n "V4_MAX_LEVERAGE\|V4_TARGET_VOL" experiments/r10[2-5]_shared.py
experiments/r147_shared.py experiments/r161_shared.py`. `V4_MAX_LEVERAGE`
is imported directly from `experiments.r102_shared` (the ultimate,
canonical source every other module in the chain already depends on
transitively) rather than duplicated as a bare literal.

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r102_shared import V4_MAX_LEVERAGE, V4_TARGET_VOL  # noqa: E402,F401
from experiments.r161_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    V4_BAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND
assert abs(V4_MAX_LEVERAGE - 2.0) < 1e-12, V4_MAX_LEVERAGE

V4_WARMUP_BARS = 80 * BARS_PER_DAY + 10   # kelly_regime_v4.warmup, TargetStrategy's default

# ------------------------------------------------------------------------
# Pre-registered ONS constants -- FIXED before either branch was
# dispatched. Provenance for every number is in the module docstring
# above; nothing here is a swept or fitted value.
# ------------------------------------------------------------------------
G_BTC_INNER_TRAIN = 0.09843034663178550   # max|grad| observed, BTC bars < INNER_VAL_START, post-warmup
ONS_ALPHA = 1.0                           # exact exp-concavity constant of -log(1+b*r) (see docstring)
ONS_BETA_BTC = 0.5                        # (1/2)*min(1/(4*G*D), ONS_ALPHA), Hazan OCO Thm 4.5
ONS_EPS_BTC = 1.0                         # 1/(ONS_BETA_BTC**2 * V4_MAX_LEVERAGE**2), Thm 4.5
ONS_B0 = V4_MAX_LEVERAGE / 2.0            # domain midpoint, neutral/non-fitted (design doc)

# Kill-switch / promotion-bar constants -- see docs_scratch_direction.md
# section cited in each comment. None of these is fitted to a result.
SHARPE_NOISE_FLOOR = 0.2                  # design doc S4 / R-20's own promotion bar
R2_KILL_THRESH = 0.95                     # design doc S2(4)(c): "R^2 > 0.95" (failure mode KS-B)
CORNER_LOCKIN_THRESH = 0.80               # design doc S2(4)(a): ">80% of inner-train bars pinned"
SHARPE_DELTA_PROMOTE = 0.2                # design doc S4 item 1
DD_REDUCTION_PROMOTE_PP = 5.0             # design doc S4 item 1: ">= 5 percentage points"
EXPOSURE_MATCH_BAND = (0.9, 1.1)          # design doc S4 item 1: exposure_ratio & vol_ratio both in this band


# ==========================================================================
# (1) Causal simple-return primitive. Bar t's OWN realized return (an
#     outcome observed at the close of bar t), NOT a decision input, so it
#     is deliberately NOT .shift(1)'d -- see the module docstring's unit
#     note. ons_scale's own loop is what enforces causality of b_t.
# ==========================================================================


def asset_simple_return(df: pd.DataFrame) -> np.ndarray:
    """Bar t's own simple (arithmetic) return, close_t/close_{t-1} - 1.

    NaN at bar 0. This is the SIMPLE-return convention ons_scale requires
    (see module docstring) -- do not substitute log(close).diff() here.
    """
    close = df["close"].to_numpy(dtype=float)
    out = np.full(len(close), np.nan)
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


# ==========================================================================
# (2) Online Newton Step, single-scalar leverage-vs-cash case (Agarwal,
#     Hazan, Kale & Schapire 2006 / Hazan OCO Thm 4.5 -- see docstring).
# ==========================================================================


def ons_scale(frac: np.ndarray, ret: np.ndarray, max_leverage: float,
               eps: float, beta: float, b0: float | None = None) -> np.ndarray:
    """Per-bar Online Newton Step leverage `b_t`, causal by construction.

    `frac` is the already-decided vote (v4_vote_frac or equivalent);
    `ret` is the asset's own SIMPLE return each bar (asset_simple_return
    -- NOT log return, see module docstring). `r_t = frac_t * ret_t` is
    the vote's own realized bet-payoff.

    Loop invariant: `out[t]` is `b_t`, the leverage USED to size bar t's
    exposure, computed from `a`/`b` state that only ever incorporated
    `grad_s` for `s < t` (i.e. `r_s` for bars strictly before t). `r_t`
    itself -- observed only at the close of bar t -- is used AFTER
    `out[t]` is recorded, to advance the state for `out[t+1]`. This is
    exactly the same causal shape as `apply_deadband`'s own loop (record
    `target[i]` from state carried in, then update state using bar i's
    own value for use at i+1) and is checked by `_self_test`'s
    causal-truncation probe below.

    Denominator guard: `1 + b_t*r_t` would be <= 0 only if the bet at
    that leverage had gone bankrupt this bar (loss >= 100%). The
    denominator is clipped to a small positive floor (1e-6) rather than
    left to blow up or flip sign -- this caps the resulting gradient's
    MAGNITUDE at a large finite value while preserving its (correct,
    de-leveraging) SIGN, rather than reversing it. On the BTC inner-train
    window used to derive G above, zero bars triggered this guard at
    `b=max_leverage`; it is retained for robustness on other slices
    (ETH, inner-validation) this module does not inspect.
    """
    frac = np.asarray(frac, dtype=float)
    ret = np.asarray(ret, dtype=float)
    n = len(frac)
    assert len(ret) == n, (len(frac), len(ret))
    b = float(V4_MAX_LEVERAGE / 2.0 if b0 is None else b0)
    a = float(eps)
    out = np.empty(n, dtype=float)
    for t in range(n):
        out[t] = b
        r_t = frac[t] * ret[t]
        if not np.isfinite(r_t):
            continue
        denom = 1.0 + b * r_t
        denom = denom if denom > 1e-6 else 1e-6
        grad = -r_t / denom
        a += grad * grad
        b = b - (1.0 / beta) * grad / a
        b = min(max(b, 0.0), max_leverage)
    return out


# ==========================================================================
# (3) Kill-switch helpers (design doc S2(4) named failure modes, S4
#     decision rule). Pure functions over arrays a caller already has --
#     no strategy or backtest is run from this module.
# ==========================================================================


def corner_lockin_fraction(b: np.ndarray, max_leverage: float, tol: float = 1e-6) -> float:
    """Fraction of bars where `b` sits within `tol` of 0 or max_leverage.

    Failure mode (a) in the design doc: > CORNER_LOCKIN_THRESH (0.80) is
    a named design failure, not a cue to re-tune eps after looking.
    """
    b = np.asarray(b, dtype=float)
    at_zero = np.abs(b - 0.0) <= tol
    at_max = np.abs(b - max_leverage) <= tol
    return float(np.mean(at_zero | at_max)) if len(b) else float("nan")


def exposure_artifact_r2(b_ons: np.ndarray, b_incumbent: np.ndarray) -> float:
    """R^2 of the ONS-learned scale path against v4's own incumbent scale.

    Failure mode (c) / KS-B in the design doc: R2_KILL_THRESH == 0.95,
    quoting design doc S2(4)(c) ("R^2 > 0.95"). Reuses `r_squared`
    (experiments.r102_shared, re-exported through the r161_shared chain)
    verbatim rather than reimplementing it.
    """
    return r_squared(np.asarray(b_ons, dtype=float), np.asarray(b_incumbent, dtype=float))


# ==========================================================================
# (4) Causal-truncation self-test (standard convention this repo's
#     frozen modules all use -- see r102/r147/r161_shared's own
#     `_self_test`, and tests/test_causality_strict.py for the pattern
#     this generalizes). Synthetic data only, matching every prior round's
#     `_self_test` -- no real data is read at import time.
# ==========================================================================


def _ons_build(df: pd.DataFrame) -> np.ndarray:
    """Wraps ons_scale as a pure df -> array builder for the shared
    causal_truncation_probe_series harness (v4's own vote + this
    module's frozen BTC eps/beta -- exercising the wiring, not re-fitting
    eps/beta per call)."""
    frac = v4_vote_frac(df).to_numpy()
    ret = asset_simple_return(df)
    return ons_scale(frac, ret, V4_MAX_LEVERAGE, ONS_EPS_BTC, ONS_BETA_BTC, b0=ONS_B0)


def _self_test() -> None:
    """Fast checks on synthetic data. Mirrors r102/r147/r161_shared's convention."""
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(171)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": 1.0}, index=idx)

    # (1) asset_simple_return: NaN at bar 0, finite elsewhere, matches
    # a direct pandas pct_change computation.
    ret = asset_simple_return(df)
    assert not np.isfinite(ret[0])
    assert np.all(np.isfinite(ret[1:]))
    assert np.allclose(ret[1:], df["close"].pct_change().to_numpy()[1:], rtol=1e-12)

    # (2) ons_scale: domain-respecting, deterministic, starts at b0.
    frac = v4_vote_frac(df).to_numpy()
    b = ons_scale(frac, ret, V4_MAX_LEVERAGE, ONS_EPS_BTC, ONS_BETA_BTC, b0=ONS_B0)
    assert len(b) == len(df)
    assert b[0] == ONS_B0
    assert np.all(b >= 0.0 - 1e-12) and np.all(b <= V4_MAX_LEVERAGE + 1e-12)
    b_again = ons_scale(frac, ret, V4_MAX_LEVERAGE, ONS_EPS_BTC, ONS_BETA_BTC, b0=ONS_B0)
    assert np.array_equal(b, b_again), "ons_scale is not deterministic"

    # (3) Causal-truncation probe on the full ons_scale pipeline -- the
    # standard test this codebase uses everywhere to catch lookahead
    # bugs (tests/test_causality_strict.py's pattern, generalized).
    assert causal_truncation_probe_series(_ons_build, df)

    # (4) Kill-switch helpers: sane bounds on synthetic data.
    assert 0.0 <= corner_lockin_fraction(b, V4_MAX_LEVERAGE) <= 1.0
    assert corner_lockin_fraction(np.array([0.0, 0.0, V4_MAX_LEVERAGE]), V4_MAX_LEVERAGE) == 1.0
    assert corner_lockin_fraction(np.array([1.0, 1.0, 1.0]), V4_MAX_LEVERAGE) == 0.0
    r2_self = exposure_artifact_r2(b, b)
    assert abs(r2_self - 1.0) < 1e-9
    r2_noise = exposure_artifact_r2(b, rng.normal(0, 1, len(b)))
    assert r2_noise < 0.5

    # (5) alpha=1 exact-exp-concavity identity, checked numerically:
    # f''(b) == f'(b)**2 for -log(1+b*r) at a grid of (b, r) points.
    bs = np.array([0.0, 0.3, 1.0, 1.7, 2.0])
    rs = np.array([-0.05, -0.01, 0.01, 0.05])   # r=0 excluded: f'=f''=0, degenerate but trivially true
    h = 1e-3
    for r_val in rs:
        for b_val in bs:
            def f(bb: float) -> float:
                return -np.log(1.0 + bb * r_val)
            fp = (f(b_val + h) - f(b_val - h)) / (2 * h)
            fpp = (f(b_val + h) - 2 * f(b_val) + f(b_val - h)) / (h * h)
            assert abs(fpp - fp ** 2) < 1e-4 * max(1.0, fp ** 2), (b_val, r_val, fpp, fp ** 2)


_self_test()
