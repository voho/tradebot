"""Shared, read-only pre-registration for the R-172 round (08-28).

DIRECTION, one sentence: correct `kelly_regime_v4`'s vote for the
post-selection bias of implicitly trusting whichever of its **8** discrete
3-anchor agreement patterns (`(v20,v40,v80) in {0,1}^3`) happens to be
realized, using False Coverage-statement Rate (FCR) control (Benjamini &
Yekutieli 2005, JASA 100(469), 71-81) to build a simultaneously-valid
one-sided confidence bound for the currently-active pattern's own
historical forward-return edge.

Full Step 1/Step 2 design (constraint attacked [ERR], non-duplication
against R-31/R-87/R-104/R-105/R-106/R-109-R-123/R-114/R-116/R-147/R-160/
R-161/R-167/R-171, simulability, named failure modes, noise-floor
arithmetic, the pre-registered promotion rule) is in
`experiments/r172_direction.md`, written by the operator (based on a
research sub-agent's proposal, independently verified) BEFORE either
branch was dispatched. This module implements NEITHER branch's strategy
variant (conservative: binary FCR gate on `frac`; novel: FCR-width-driven
per-bar `vote_gamma`) -- that is each branch's own job, on top of the
`fcr_lower_bounds` primitive below. Neither branch may edit this file or
each other's file (R-89-through-R-171's own convention).

============================================================================
WHY A NORMAL APPROXIMATION, NOT A STUDENT-t OR A DISTRIBUTION-FREE BOUND --
disclosed up front, matching this project's own convention of disclosing
simplifications (r102_shared.py's undemeaned-vs-EWM-std semivariance note;
r161_shared.py's Hoeffding-not-Bentkus note, "outside this project's
dependency set"). This project ships numpy/pandas/matplotlib only (see
pyproject.toml) -- no scipy, so no library Student-t or Beta/Bentkus
quantile function is available. A literal Hoeffding bound (as r161_shared
uses for the SCALE tail-loss cap) needs an a-priori almost-sure bound on
the random variable; a pattern's own forward H-day SIMPLE return has no
natural a-priori bound the way a [0,1] loss indicator does, and inventing
one after inspecting the data would be exactly the kind of post-hoc
threshold-picking ROUTINE.md warns against. The normal approximation
(Acklam's rational inverse-normal-CDF, self-tested below against known
reference quantiles) is therefore used, GATED by a minimum bucket size
(`MIN_N=30`, a standard CLT-reliability heuristic, fixed before any data
was inspected) below which no bound is reported at all -- see
`fcr_lower_bounds`'s docstring for the causal "no evidence yet" default
this implies.

============================================================================
FCR FORMULA PROVENANCE -- Benjamini & Yekutieli (2005)'s own procedure
(confirmed by web search against the paper's abstract/summary, independent
of the research sub-agent's report that first proposed this direction):
for `R` parameters SELECTED out of `m` CONSIDERED, use per-parameter
confidence level `1 - q*R/m` (equivalently, per-parameter significance
`alpha = q*R/m`) rather than the marginal `1-q`. This round selects
EXACTLY ONE pattern per day (the day's own realized `p(d)`), so `R=1`
throughout; `m=8` is the FULL candidate space (all possible 3-anchor
patterns), fixed regardless of how many patterns have actually been
observed by day `d` -- using the observed-so-far count instead would let
the correction weaken simply because rare patterns had not yet appeared,
the wrong direction for a conservative bound (Berk, Brown, Buja, Zhang &
Zhao 2013, "Valid Post-Selection Inference," Annals of Statistics 41(2),
802-837, DOI 10.1214/12-AOS1077 -- the "valid under any selection rule"
argument for using the full candidate count).

    Q_FCR = 0.10          # matches r161_shared.py's own HB_DELTA=0.10
                           # precedent (1-delta=90% confidence); NOT fitted
                           # to this round's data.
    M_PATTERNS = 8         # fixed candidate space: 2**3 anchors.
    ALPHA_CORRECTED = Q_FCR / M_PATTERNS = 0.0125
    Z_ONE_SIDED = norm_ppf(1 - ALPHA_CORRECTED)        = norm_ppf(0.9875)
    Z_TWO_SIDED = norm_ppf(1 - ALPHA_CORRECTED / 2.0)  = norm_ppf(0.99375)

Both z-values are DERIVED from Q_FCR/M_PATTERNS by the module at import
time (not hand-copied), and `_self_test` checks `norm_ppf` against four
independent known reference quantiles
(Phi^-1(0.95)=1.6448536, Phi^-1(0.975)=1.9599640,
Phi^-1(0.995)=2.5758293, Phi^-1(0.9875)=2.2414027) before anything else in
this module runs, so a coefficient transcription error in `norm_ppf`
cannot silently produce a wrong bound.

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).
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

from experiments.r102_shared import (  # noqa: E402,F401
    _latched_anchor_vote,
    V4_BAND,
    V4_HORIZONS,
    V4_MAX_LEVERAGE,
    V4_TARGET_VOL,
)
from experiments.r161_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    ETH_SLICE_NAME,
    FEE_TIER,
    FUTURES,
    GATE_MIN_BINDING_FRACTION,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    SLICES,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    compare,
    daily_close,
    daily_last_of,
    daily_log_return,
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

# ------------------------------------------------------------------------
# Pre-registered FCR constants -- FIXED before either branch was
# dispatched. Provenance for every number is in the module docstring
# above; nothing here is a swept or fitted value.
# ------------------------------------------------------------------------
Q_FCR = 0.10
M_PATTERNS = 8
ALPHA_CORRECTED = Q_FCR / M_PATTERNS
HORIZON_DAYS = 5             # forward-return label window, a priori
MIN_N = 30                   # CLT-reliability floor before any bound is reported
WIDTH_CAP_RATIO = 2.0        # novel branch: clip(width/width_ref, 0, this)

# Kill-switch / promotion-bar constants -- see r172_direction.md's section
# cited in each comment. None of these is fitted to a result. (No
# corner-lock-in constant here, unlike r171_shared.py's ONS module: that
# failure mode is specific to an unbounded learned scalar converging to a
# domain boundary, which does not apply to this round's bounded gate/
# exponent construction.)
R2_KILL_THRESH = 0.95                     # KS-B, direction doc S6
EXPOSURE_MATCH_BAND = (0.9, 1.1)          # direction doc S7 item 1
SHARPE_DELTA_PROMOTE = 0.2                # direction doc S7 item 1
DD_REDUCTION_PROMOTE_PP = 5.0             # direction doc S7 item 1


# ==========================================================================
# (1) Acklam's rational approximation to the inverse standard-normal CDF.
#     No scipy in this project's dependency set (see module docstring).
#     Self-tested against known reference quantiles below.
# ==========================================================================

_ACKLAM_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACKLAM_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
_ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's algorithm, ~1e-9 relative error).

    `p` must be strictly inside (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"norm_ppf: p={p} not in (0,1)")
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    p_high = 1.0 - _P_LOW
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


Z_ONE_SIDED = norm_ppf(1.0 - ALPHA_CORRECTED)          # ~2.2414
Z_TWO_SIDED = norm_ppf(1.0 - ALPHA_CORRECTED / 2.0)    # ~2.4977


# ==========================================================================
# (2) The 8-pattern classification. Reproduces v4's own three anchor votes
#     EXACTLY via r102_shared's own `_latched_anchor_vote` (not
#     reimplemented) and packs them into one integer per bar: bit i =
#     horizons[i]'s vote (i=0 -> 20d, i=1 -> 40d, i=2 -> 80d). frac(t) ==
#     popcount(pattern(t)) / 3 by construction (checked in _self_test).
# ==========================================================================


def anchor_pattern(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                    band: float = V4_BAND) -> np.ndarray:
    """Per-bar integer in [0, 2**len(horizons)); bit i = horizons[i]'s
    latched 0/1 vote. Causal by construction (each vote is `.ffill()`'d
    from a rolling mean, identical to v4's own `frac` computation)."""
    close = df["close"]
    pattern = np.zeros(len(df), dtype=np.int64)
    for i, days in enumerate(horizons):
        v = _latched_anchor_vote(close, days, band).to_numpy().astype(np.int64)
        pattern |= (v << i)
    return pattern


# ==========================================================================
# (3) Causal, embargoed, FCR-corrected per-pattern confidence bound.
#     Operates at DAILY resolution (matching r161_shared.py's own
#     daily_close / daily_log_return / daily_last_of convention) to keep
#     each pattern's historical sample close to independent observations
#     rather than ~288x-oversampled 5-minute bars.
# ==========================================================================


def daily_pattern(df: pd.DataFrame) -> pd.Series:
    """Each day's own decision-time pattern (last bar of the day)."""
    pattern = anchor_pattern(df)
    return daily_last_of(pattern.astype(float), df.index).round().astype(int)


def daily_forward_simple_return(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Day d's forward `horizon_days`-day simple return, close[d+H]/close[d]-1,
    indexed by day d (the day whose pattern this label scores).

    NOT causal as a per-day feature -- it looks `horizon_days` into the
    future by construction. It exists ONLY to build the historical
    resolved-label pool inside `fcr_lower_bounds`, which enforces
    causality via an explicit embargo (a label for day d' is usable at
    day d only if d' + horizon_days <= d - 1). Never used as a live
    decision input for the day whose pattern it belongs to.
    """
    close = daily_close(df)
    return close.shift(-horizon_days) / close - 1.0


def fcr_lower_bounds(df: pd.DataFrame, horizon_days: int = HORIZON_DAYS,
                      q: float = Q_FCR, m_patterns: int = M_PATTERNS,
                      min_n: int = MIN_N):
    """Causal, day-by-day FCR-corrected bound on the ACTIVE pattern's own
    historical mean forward return.

    Returns (index, lcb, ucb, n_used) -- all length len(index) (the
    intersection of the daily pattern and forward-return series). `lcb`
    is the one-sided FCR-corrected lower confidence bound at level
    `1 - q/m_patterns` on the currently-active pattern's mean forward
    return; `ucb` is the corresponding two-sided upper bound at the same
    corrected alpha (used only by the novel branch, for interval width).
    Both are NaN wherever `n_used < min_n` ("no evidence yet" -- see
    module docstring for why this defaults to v4's own unmodified
    behavior in both branches, rather than to a punitive default).

    CAUSALITY: day d's bound uses ONLY resolved labels from days
    d' <= d - 1 - horizon_days (i.e. d''s own `horizon_days`-forward
    window has fully closed strictly before day d begins). This is
    checked end-to-end by `causal_truncation_probe_series` in
    `_self_test` below, via `_fcr_lcb_build` (a `df -> np.ndarray`
    wrapper broadcasting `lcb` to bar frequency).
    """
    pattern_s = daily_pattern(df)
    fwd_s = daily_forward_simple_return(df, horizon_days)
    idx = pattern_s.index.intersection(fwd_s.index)
    pattern = pattern_s.reindex(idx).to_numpy()
    fwd = fwd_s.reindex(idx).to_numpy()
    n = len(idx)

    alpha = q / m_patterns
    z_one = norm_ppf(1.0 - alpha)
    z_two = norm_ppf(1.0 - alpha / 2.0)

    lcb = np.full(n, np.nan)
    ucb = np.full(n, np.nan)
    n_used = np.zeros(n, dtype=np.int64)

    buckets: list[list[float]] = [[] for _ in range(m_patterns)]
    released = -1  # index of the last day whose label has been folded into buckets

    for t in range(n):
        target_released = t - 1 - horizon_days
        while released < target_released:
            released += 1
            y = fwd[released]
            if np.isfinite(y):
                buckets[int(pattern[released])].append(float(y))

        p = int(pattern[t])
        sample = buckets[p]
        m = len(sample)
        n_used[t] = m
        if m < min_n:
            continue
        arr = np.asarray(sample, dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
        se = sd / math.sqrt(m)
        if se <= 0.0:
            lcb[t] = mean
            ucb[t] = mean
            continue
        lcb[t] = mean - z_one * se
        ucb[t] = mean + z_two * se

    return idx, lcb, ucb, n_used


def broadcast_daily(daily_values: np.ndarray, daily_index: pd.DatetimeIndex,
                     bar_index: pd.DatetimeIndex, fill_value: float) -> np.ndarray:
    """Forward-fill a daily series (already causal -- built only from data
    through the prior day) to bar frequency, matching `r161_shared.py`'s
    own `broadcast_daily_lambda` convention. Days before any value exists
    default to `fill_value` (the "no evidence yet" state)."""
    s = pd.Series(np.asarray(daily_values, dtype=float), index=daily_index)
    days = pd.DatetimeIndex(bar_index).floor("D")
    return s.reindex(days).ffill().fillna(fill_value).to_numpy()


# ==========================================================================
# (4) Kill-switch helpers (direction doc S6). Pure functions over arrays a
#     caller already has -- no strategy or backtest is run from this module.
# ==========================================================================


def binding_fraction(differs: np.ndarray) -> float:
    """KS-A: fraction of bars where the candidate's factor differs from
    v4's own baseline. `differs` is a boolean array the caller builds
    (conservative: gated_frac != frac; novel: gamma_t != 1.0)."""
    differs = np.asarray(differs, dtype=bool)
    return float(np.mean(differs)) if len(differs) else float("nan")


def relabeling_r2(candidate_target: np.ndarray, v4_raw_target: np.ndarray) -> float:
    """KS-B: R^2 of the candidate's final target path against v4's own
    raw (pre-deadband) frac*scale path. Reuses `r_squared` verbatim."""
    return r_squared(np.asarray(candidate_target, dtype=float),
                      np.asarray(v4_raw_target, dtype=float))


# ==========================================================================
# (5) Self-test on synthetic data. Mirrors r102/r147/r161_shared's
#     convention: fast checks, no real data read at import time.
# ==========================================================================


def _fcr_lcb_build(df: pd.DataFrame) -> np.ndarray:
    """Wraps fcr_lower_bounds as a pure df -> array builder (broadcast to
    bar frequency) for the causal_truncation_probe_series harness."""
    idx, lcb, _ucb, _n = fcr_lower_bounds(df)
    return broadcast_daily(lcb, idx, df.index, fill_value=np.nan)


def _self_test() -> None:
    """Fast checks on synthetic data. Mirrors r102/r147/r161_shared's convention."""
    # (0) norm_ppf against four independent known reference quantiles --
    # checked FIRST, before anything downstream trusts it.
    ref = {0.95: 1.6448536269514722, 0.975: 1.959963984540054,
           0.995: 2.5758293035489004, 0.9875: 2.241402727604947}
    for p, expect in ref.items():
        got = norm_ppf(p)
        assert abs(got - expect) < 1e-6, (p, got, expect)
    assert abs(Z_ONE_SIDED - norm_ppf(0.9875)) < 1e-12
    assert abs(Z_TWO_SIDED - norm_ppf(0.99375)) < 1e-12
    assert Z_TWO_SIDED > Z_ONE_SIDED > 1.6449   # tighter than a bare 95% one-sided z

    idx = pd.date_range("2017-01-01", periods=200_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(172)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                        "close": close, "volume": 1.0}, index=idx)

    # (1) anchor_pattern: bounded in [0, 8), and popcount/3 matches v4's
    # own vote_frac exactly (both built from the identical
    # _latched_anchor_vote calls).
    pattern = anchor_pattern(df)
    assert pattern.min() >= 0 and pattern.max() < 8
    popcount = np.array([bin(int(p)).count("1") for p in pattern], dtype=float)
    frac_from_pattern = popcount / 3.0
    frac_direct = v4_vote_frac(df).to_numpy()
    assert np.allclose(frac_from_pattern, frac_direct, atol=1e-12), \
        "anchor_pattern's popcount/3 must equal v4_vote_frac exactly"

    # (2) fcr_lower_bounds: shapes align, lcb <= ucb wherever both finite,
    # NaN wherever n_used < MIN_N, and n_used is non-decreasing along any
    # fixed pattern's own occurrences (buckets only ever grow).
    fidx, lcb, ucb, n_used = fcr_lower_bounds(df)
    assert len(lcb) == len(ucb) == len(n_used) == len(fidx)
    both_finite = np.isfinite(lcb) & np.isfinite(ucb)
    assert np.all(lcb[both_finite] <= ucb[both_finite] + 1e-9)
    assert np.all(np.isnan(lcb[n_used < MIN_N]))
    assert np.all(np.isfinite(lcb[n_used >= MIN_N]) | (n_used[n_used >= MIN_N] == 0))

    # (3) Causal-truncation probe on the full fcr_lower_bounds pipeline --
    # the standard convention this codebase uses everywhere to catch
    # lookahead bugs (tests/test_causality_strict.py's pattern, generalized
    # to a builder with an internal embargo rather than a plain .shift(1)).
    assert causal_truncation_probe_series(_fcr_lcb_build, df,
                                           cuts=(0.5, 0.7, 0.9))

    # (4) broadcast_daily: forward-fills correctly, defaults to fill_value
    # before the first daily value.
    daily_idx = fidx[:5]
    daily_vals = np.array([np.nan, 1.0, np.nan, 2.0, np.nan])
    bar_idx = pd.date_range(daily_idx[0], periods=BARS_PER_DAY * 6, freq="5min", tz="UTC")
    bc = broadcast_daily(daily_vals, daily_idx, bar_idx, fill_value=-99.0)
    assert bc[0] == -99.0          # before first non-NaN value
    assert np.isclose(bc[BARS_PER_DAY], 1.0)      # day 1 ffilled forward
    assert np.isclose(bc[3 * BARS_PER_DAY], 2.0)  # day 3's own value

    # (5) kill-switch helpers: sane bounds on synthetic data.
    differs = np.array([True, False, False, True, True])
    assert abs(binding_fraction(differs) - 0.6) < 1e-12
    r2_self = relabeling_r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert abs(r2_self - 1.0) < 1e-9
    r2_noise = relabeling_r2(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
    assert r2_noise < 0.5


_self_test()
