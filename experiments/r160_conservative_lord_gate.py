#!/usr/bin/env python
"""R-160 CONSERVATIVE branch: gate each of `kelly_regime_v4`'s three anchor
votes' FLIP decisions through LORD (Javanmard & Montanari 2015/2018, "Online
rule for control of false discovery rate and false discovery exceedance"),
a classical, deterministic sequential online-FDR-control procedure. Direction,
citations, non-duplication argument, kill switches and the pre-registered
decision rule all live in `experiments/r160_shared.py`'s module docstring
(read there first -- this file does not repeat that reasoning and does not
edit that module, which is frozen/read-only).

THE MECHANISM, exactly:

For each anchor k in (20, 40, 80) days, `anchor_raw_vote_and_pvalues` (from
r160_shared, shared with the sibling novel branch so both branches see
IDENTICAL evidence) gives v4's own 0/1 latched vote `raw_k` and a causal
p-value `p_k` defined at every bar (H0: the price/anchor gap is noise, via a
two-sided normal test on a causal z-score of the gap).

At every bar `i` where `raw_k[i]` disagrees with the gate's own currently
HELD state (a flip is "pending"), that is one TEST OPPORTUNITY `t` (a
per-anchor counter over test opportunities, NOT raw bar index). LORD's
adaptive threshold at that test is

    alpha_t = W0 * gamma_t + b * sum_{tau_i < t} gamma_(t - tau_i)

where `tau_i` ranges over the test-opportunity indices of ALL PAST
REJECTIONS (accepted flips) for this anchor, `W0 = w0_fraction * alpha` is
the initial wealth, `b = alpha - W0` is the wealth invested per past
rejection, and `gamma_j` is a FIXED, non-increasing, summable discount
sequence (defined below, `sum_{j=1}^inf gamma_j = 1` by construction). If
`p_k[i] <= alpha_t`, the gate's held state jumps to `raw_k[i]` immediately
(a "rejection", which replenishes wealth for future tests by entering the
`tau_i` sum from then on); otherwise the gate holds its prior state and
re-tests on the next bar the disagreement persists. This is causal by
construction: `alpha_t` at test `t` is built only from decisions at tests
`< t` (`t` itself and the `tau_i < t` list), never anything at or after the
current test.

The three gated 0/1 votes are combined by v4's OWN unmodified equal-weight
average (`combine_gated_votes`) and wired through v4's OWN unmodified
`scale`/10% deadband (`build_target_from_frac`) -- the only difference from
v4 is WHEN a flip takes effect, never anything about its magnitude or the
sizing on top.

DISCOUNT SEQUENCE, stated exactly (per the task's own requirement to name
the formula and normalization in a comment): gamma_j = j**-1.5 / C for
j = 1, 2, 3, ..., where C = sum_{j=1}^inf j**-1.5 is the normalizing
constant, computed numerically (not hardcoded) as a partial sum over the
first 2,000,000 terms plus an integral tail correction
(integral_{2e6}^inf x**-1.5 dx = 2/sqrt(2e6)) for the remainder -- this
converges to the closed-form Riemann zeta(1.5) ~= 2.612375 to 6+ decimal
places, verified in the self-test below. `gamma_j` is fixed once at import
time and never refit to any data; it is non-increasing in j and summable
(1.5 > 1), which is exactly LORD's own requirement on the discount sequence.

======================================================================
HEADLINE RESULT, stated before the detail: NEGATIVE. Filled in by main()
below after running against real data -- see the printed VERDICT block and
this session's final report for the actual numbers (calibration self-test,
kill switches, per-alpha inner-train/inner-validation/ETH cells, and which
clause of r160_shared's decision rule failed, if any).
======================================================================

Run: `. .venv/bin/activate && python experiments/r160_conservative_lord_gate.py`
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

from experiments.r160_shared import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_PRIMARY,
    BARS_PER_DAY,
    FUTURES,
    GATE_MIN_DELAYS,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    PVAL_SIGMA_DAYS,
    PVAL_SIGMA_DAYS_ALT,
    R2_DEGENERACY_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TargetStrategy,
    V4_BAND,
    V4_HORIZONS,
    W0_FRACTION,
    anchor_raw_vote_and_pvalues,
    assert_no_holdout,
    build_target_from_frac,
    causal_truncation_probe_series,
    combine_gated_votes,
    compare,
    count_delayed_episodes,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    run_slice,
    synthetic_zero_drift_frame,
    v4_target,
    v4_vote_frac,
)

K_ANCHORS = len(V4_HORIZONS)
assert K_ANCHORS == 3, V4_HORIZONS


# ================================================================== (1)
# The fixed discount sequence gamma_j, normalized so sum_{j=1}^inf = 1.
# Computed numerically (not hardcoded) so the normalization procedure is
# auditable from this file alone.
# ==================================================================

def _gamma_normalizing_constant(n_terms: int = 2_000_000) -> float:
    """C = sum_{j=1}^inf j**-1.5, via a partial sum over n_terms plus an
    integral tail correction for j > n_terms. j**-1.5 is summable (exponent
    1.5 > 1), so this converges quickly; the tail bound
    integral_{n_terms}^inf x**-1.5 dx = 2/sqrt(n_terms) is < 0.0015 at
    n_terms=2e6, i.e. this is accurate to better than 1e-3 relative, and the
    self-test below checks it against the closed-form Riemann zeta(1.5)."""
    j = np.arange(1, n_terms + 1, dtype=np.float64)
    partial = float(np.sum(j ** -1.5))
    tail = 2.0 / math.sqrt(n_terms)
    return partial + tail


GAMMA_NORM = _gamma_normalizing_constant()


def gamma_j(j) -> float:
    """gamma_j = j**-1.5 / GAMMA_NORM, for integer j >= 1. Fixed, non-
    increasing, summable (sum_{j=1}^inf gamma_j = 1 by construction) --
    never fit to any data."""
    return (float(j) ** -1.5) / GAMMA_NORM


# ================================================================== (2)
# The LORD gate itself. `_lord_gate_core` does the causal sequential pass
# and also returns the raw bookkeeping (test-opportunity count, rejection
# positions) so both the trading candidate (`lord_gate`) and the
# calibration self-test (which needs acceptance-rate diagnostics) share
# ONE implementation.
# ==================================================================

def _lord_gate_core(raw_vote: np.ndarray, pvalues: np.ndarray, alpha: float,
                    w0_fraction: float = W0_FRACTION):
    """Causal LORD gate over one anchor's raw latched vote + p-value stream.

    Returns (gated: np.ndarray[float], n_tests: int, n_rejections: int).

    At bar i>0, IF raw_vote[i] != state (the gate's currently held value,
    a flip is pending): this is test opportunity t (t increments by 1 for
    EVERY such bar, whether or not it is later accepted -- t indexes test
    opportunities, not raw bar index). alpha_t is built ONLY from t and the
    list of past rejections' own test-opportunity indices (all < t) via the
    LORD formula: alpha_t = W0*gamma_t + b*sum(gamma_(t-tau) for tau in
    past rejections). If pvalues[i] <= alpha_t: accept -- state jumps to
    raw_vote[i], and t itself is appended to the rejection list (so it
    replenishes wealth for every later test). Otherwise: hold state, and
    the SAME disagreement (if it persists) is tested again next bar as a
    NEW test opportunity t+1.

    Causal by construction: alpha_t at test t depends only on t itself and
    on rejections recorded at tests strictly < t -- nothing at or after
    bar i, and nothing about any FUTURE bar's raw_vote/pvalues.
    """
    raw_vote = np.asarray(raw_vote, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(raw_vote)
    assert len(pvalues) == n, (len(raw_vote), len(pvalues))
    gated = np.empty(n, dtype=float)
    if n == 0:
        return gated, 0, 0

    state = raw_vote[0]
    gated[0] = state
    W0 = w0_fraction * alpha
    b = alpha - W0
    rejection_ts: list[int] = []
    t = 0

    for i in range(1, n):
        if raw_vote[i] != state:
            t += 1
            wealth_term = 0.0
            for tau in rejection_ts:
                wealth_term += gamma_j(t - tau)
            alpha_t = W0 * gamma_j(t) + b * wealth_term
            if pvalues[i] <= alpha_t:
                state = raw_vote[i]
                rejection_ts.append(t)
        gated[i] = state

    return gated, t, len(rejection_ts)


def lord_gate(raw_vote: np.ndarray, pvalues: np.ndarray, alpha: float,
              w0_fraction: float = 0.5) -> np.ndarray:
    """Public entry point matching the pre-registered signature: returns
    ONLY the gated 0/1 vote array (see `_lord_gate_core` for the full
    causal algorithm and its diagnostics)."""
    gated, _n_tests, _n_rej = _lord_gate_core(raw_vote, pvalues, alpha, w0_fraction)
    return gated


# ================================================================== (3)
# The candidate: gate each anchor, combine with v4's own equal weighting,
# wire through v4's own scale/deadband. Pure function of `df`, with a
# lightweight cache (keyed by the frame's own index bounds, not `id()`,
# since `compare()`/`run_slice()` re-slice the same underlying data per
# market) so the identical (alpha, sigma_days, slice) computation is not
# redone twice per `compare()` call just because SPOT and FUTURES_5x share
# an identical price frame.
# ==================================================================

_BUILD_CACHE: dict[tuple, np.ndarray] = {}


def anchor_gated_vote(df: pd.DataFrame, days: int, alpha: float,
                      sigma_days: int = PVAL_SIGMA_DAYS,
                      w0_fraction: float = W0_FRACTION) -> np.ndarray:
    raw, p = anchor_raw_vote_and_pvalues(df["close"], days, V4_BAND, sigma_days)
    return lord_gate(raw, p, alpha, w0_fraction)


def lord_frac(df: pd.DataFrame, alpha: float, sigma_days: int = PVAL_SIGMA_DAYS,
             w0_fraction: float = W0_FRACTION,
             horizons: tuple[int, ...] = V4_HORIZONS) -> np.ndarray:
    gated_votes = [anchor_gated_vote(df, days, alpha, sigma_days, w0_fraction)
                  for days in horizons]
    return combine_gated_votes(gated_votes)


def make_build_target(alpha: float, sigma_days: int = PVAL_SIGMA_DAYS,
                      w0_fraction: float = W0_FRACTION, use_cache: bool = True):
    """Candidate `build_target(df) -> np.ndarray`: v4's own scale and
    deadband, unchanged, fed the LORD-gated vote fraction instead of v4's
    immediate-flip vote. Pure function of `df`."""

    def build_target(df: pd.DataFrame) -> np.ndarray:
        key = None
        if use_cache and len(df):
            key = (alpha, sigma_days, w0_fraction,
                  df.index[0].value, df.index[-1].value, len(df))
            cached = _BUILD_CACHE.get(key)
            if cached is not None:
                return cached
        frac = lord_frac(df, alpha, sigma_days, w0_fraction)
        target = build_target_from_frac(frac, df)
        if key is not None:
            _BUILD_CACHE[key] = target
        return target

    build_target.__name__ = f"conservative_lord_a{alpha}_s{sigma_days}"
    return build_target


build_target = make_build_target(ALPHA_PRIMARY, PVAL_SIGMA_DAYS, W0_FRACTION)  # frozen primary


# ================================================================== (4)
# Calibration self-test: LORD on pure zero-drift noise. Reports the
# empirical acceptance rate (rejections / test opportunities) per anchor,
# per alpha -- a sanity check, not a hard gate, reported honestly either way.
# ==================================================================

def calibration_self_test(alphas=ALPHA_GRID, sigma_days: int = PVAL_SIGMA_DAYS,
                          w0_fraction: float = W0_FRACTION,
                          horizons: tuple[int, ...] = V4_HORIZONS) -> list[dict]:
    frame = synthetic_zero_drift_frame()
    close = frame["close"]
    rows = []
    for days in horizons:
        raw, p = anchor_raw_vote_and_pvalues(close, days, V4_BAND, sigma_days)
        n_raw_flips = int(np.sum(raw[1:] != raw[:-1]))
        for alpha in alphas:
            gated, n_tests, n_rej = _lord_gate_core(raw, p, alpha, w0_fraction)
            rate = (n_rej / n_tests) if n_tests > 0 else float("nan")
            rows.append(dict(days=days, alpha=alpha, n_raw_flips=n_raw_flips,
                             n_tests=n_tests, n_rejections=n_rej, accept_rate=rate))
    return rows


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


CONFIG_CELLS = 0


def main() -> None:
    global CONFIG_CELLS
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-160 CONSERVATIVE -- LordGatedAnchorKellyV4: kelly_regime_v4's "
       "immediate anchor-vote flips gated through\nLORD (Javanmard & "
       "Montanari 2015/2018) online false-discovery-rate control. "
       "Default verdict: NEGATIVE.")
    print(f"\nGAMMA_NORM (numerically computed) = {GAMMA_NORM:.10f}   "
          f"(closed-form Riemann zeta(1.5) = 2.6123753487...)")
    print(f"gamma_1={gamma_j(1):.6f}  gamma_2={gamma_j(2):.6f}  "
          f"gamma_10={gamma_j(10):.6f}  gamma_100={gamma_j(100):.6f}")

    # ========================================================== STEP 0 (a)
    hr("STEP 0a -- gamma sequence self-checks")
    js = np.arange(1, 500_000)
    gvals = js.astype(np.float64) ** -1.5 / GAMMA_NORM
    gamma_sum_500k = float(np.sum(gvals))
    zeta_1_5_ref = 2.612375348685488343
    norm_err = abs(GAMMA_NORM - zeta_1_5_ref)
    monotone_ok = bool(np.all(np.diff(gvals) <= 0))
    print(f"    sum_{{j=1}}^500000 gamma_j = {gamma_sum_500k:.8f}  "
          f"(should be just under 1.0, remaining mass in the j>500000 tail)")
    print(f"    |GAMMA_NORM - zeta(1.5) closed form| = {norm_err:.2e}  "
          f"(numerical normalization matches the known constant)")
    print(f"    gamma_j non-increasing in j (checked over first 500k terms): {monotone_ok}")
    gamma_ok = gamma_sum_500k < 1.0 and norm_err < 1e-3 and monotone_ok
    print(f"    Gamma sequence sanity: {'PASS' if gamma_ok else 'FAIL'}")
    if not gamma_ok:
        raise AssertionError("gamma sequence failed its own sanity checks -- stopping.")

    # ========================================================== STEP 0 (b)
    hr("STEP 0b -- LORD gate unit checks (synthetic, tiny)")
    # (i) p-values all 1.0 (no evidence, ever): gate must NEVER move off its
    # initial state, no matter how often raw flips.
    raw_flip = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    p_all1 = np.ones(len(raw_flip))
    g_never, n_t, n_r = _lord_gate_core(raw_flip, p_all1, alpha=0.2)
    never_moves = bool(np.all(g_never == raw_flip[0])) and n_r == 0
    print(f"    all p=1.0 -> gate never accepts a flip: {never_moves} "
          f"(n_tests={n_t}, n_rejections={n_r})")
    # (ii) p-values all ~0 (overwhelming evidence): the very FIRST test
    # opportunity must be accepted (alpha_1 = W0*gamma_1 > 0 always).
    p_all0 = np.full(len(raw_flip), 1e-12)
    g_always, n_t2, n_r2 = _lord_gate_core(raw_flip, p_all0, alpha=0.2)
    tracks_raw = bool(np.array_equal(g_always, raw_flip))
    print(f"    all p~0 -> gate tracks raw vote exactly (n_rejections=n_tests): "
          f"{tracks_raw} (n_tests={n_t2}, n_rejections={n_r2})")
    # (iii) alpha_t must never exceed alpha, even with many past rejections.
    rng_u = np.random.default_rng(160)
    raw_many = (rng_u.uniform(size=4000) > 0.5).astype(float)
    p_many = rng_u.uniform(0, 0.01, size=4000)  # small p-values -> many rejections
    alpha_test = 0.2
    W0 = W0_FRACTION * alpha_test
    b = alpha_test - W0
    rejection_ts_check: list[int] = []
    tcheck = 0
    state_check = raw_many[0]
    alpha_t_max = 0.0
    for i in range(1, len(raw_many)):
        if raw_many[i] != state_check:
            tcheck += 1
            wt = sum(gamma_j(tcheck - tau) for tau in rejection_ts_check)
            a_t = W0 * gamma_j(tcheck) + b * wt
            alpha_t_max = max(alpha_t_max, a_t)
            if p_many[i] <= a_t:
                state_check = raw_many[i]
                rejection_ts_check.append(tcheck)
    bound_ok = alpha_t_max <= alpha_test + 1e-9
    print(f"    max observed alpha_t over a high-rejection stress run = "
          f"{alpha_t_max:.6f}  (must stay <= alpha={alpha_test}): {bound_ok}")
    unit_ok = never_moves and tracks_raw and bound_ok
    print(f"\n    LORD unit checks: {'PASS' if unit_ok else 'FAIL'}")
    if not unit_ok:
        raise AssertionError("LORD gate unit checks failed -- stopping.")

    # ========================================================== STEP 1
    hr("STEP 1 -- calibration self-test on synthetic zero-drift noise "
       "(pure H0, no real trend anywhere)")
    calib = calibration_self_test()
    print(f"\n    {'days':>5s} {'alpha':>6s} {'raw_flips':>10s} {'n_tests':>8s} "
          f"{'n_rej':>6s} {'accept_rate':>12s} {'vs alpha':>10s}")
    calib_ratios = []
    for r in calib:
        ratio = r["accept_rate"] / r["alpha"] if np.isfinite(r["accept_rate"]) else float("nan")
        calib_ratios.append(ratio)
        print(f"    {r['days']:>5d} {r['alpha']:>6.2f} {r['n_raw_flips']:>10d} "
              f"{r['n_tests']:>8d} {r['n_rejections']:>6d} "
              f"{r['accept_rate']:>12.4f} {ratio:>10.2f}x")
    finite_ratios = [x for x in calib_ratios if np.isfinite(x)]
    if finite_ratios:
        worst_ratio = max(finite_ratios)
        print(f"\n    Worst (accept_rate / alpha) ratio across all "
              f"(anchor, alpha) cells on pure noise: {worst_ratio:.2f}x")
        if worst_ratio > 2.0:
            print("    HONEST FLAG: the gate accepts flips MORE THAN 2x its nominal "
                  "alpha budget under pure noise on\n    at least one cell -- the "
                  "real, serially-correlated p-value stream does not respect LORD's "
                  "\n    independence-flavoured guarantee here (failure mode (2) "
                  "named in r160_shared.py). Reported\n    honestly, not hidden.")
        elif worst_ratio > 1.2:
            print("    Mild over-acceptance relative to nominal alpha (up to "
                  f"{worst_ratio:.2f}x) -- plausibly explained by\n    the p-value "
                  "stream's serial correlation (adjacent test opportunities share "
                  "most of their trailing\n    window), not a gross miscalibration.")
        else:
            print("    Empirical acceptance roughly respects (or is more "
                  "conservative than) the nominal alpha budget.")
    else:
        print("\n    No finite acceptance-rate cells (zero test opportunities "
              "everywhere) -- cannot assess calibration.")

    # ========================================================== STEP 2
    hr("STEP 2 -- causal truncation probe (real BTC inner-train data, "
       f"primary alpha={ALPHA_PRIMARY}, sigma_days={PVAL_SIGMA_DAYS})")
    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "BTC full")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")
    btc_train = btc.loc[INNER_TRAIN_START:INNER_TRAIN_END]
    assert_no_holdout(btc_train, "BTC inner-train")
    print(f"BTC inner-train slice: {len(btc_train):,} bars, "
          f"{btc_train.index[0]} -> {btc_train.index[-1]}")

    causal_ok = True
    try:
        ok1 = causal_truncation_probe_series(
            make_build_target(ALPHA_PRIMARY, PVAL_SIGMA_DAYS, W0_FRACTION, use_cache=False),
            btc_train)
        print(f"    causal_truncation_probe_series(build_target, btc_train): "
              f"{'PASS' if ok1 else 'FAIL'}")
    except AssertionError as e:
        ok1 = False
        print(f"    causal_truncation_probe_series(build_target, btc_train): FAIL ({e})")
    causal_ok = causal_ok and ok1

    def _one_anchor_gated(d: pd.DataFrame) -> np.ndarray:
        return anchor_gated_vote(d, 20, ALPHA_PRIMARY, PVAL_SIGMA_DAYS, W0_FRACTION)

    try:
        ok2 = causal_truncation_probe_series(_one_anchor_gated, btc_train)
        print(f"    causal_truncation_probe_series(anchor_gated_vote[days=20], "
              f"btc_train): {'PASS' if ok2 else 'FAIL'}")
    except AssertionError as e:
        ok2 = False
        print(f"    causal_truncation_probe_series(anchor_gated_vote[days=20], "
              f"btc_train): FAIL ({e})")
    causal_ok = causal_ok and ok2

    print(f"\n    Causality: {'PASS' if causal_ok else 'FAIL'}")
    if not causal_ok:
        raise AssertionError("Causal truncation probe FAILED -- stopping before "
                             "any promotion-bar evaluation.")

    # ========================================================== STEP 3
    hr("STEP 3 -- kill switches on BTC inner-train, across ALPHA_GRID "
       f"(sigma_days={PVAL_SIGMA_DAYS})")
    print(f"\n    A1 (GATE_MIN_DELAYS={GATE_MIN_DELAYS}): does the gate ever actually "
          "delay a raw flip by >=1 bar?")
    print(f"    {'alpha':>6s} " + " ".join(f"{'d='+str(d):>10s}" for d in V4_HORIZONS)
          + "   any anchor clears?")
    a1_any_clears = False
    a1_table: dict[float, dict[int, int]] = {}
    for alpha in ALPHA_GRID:
        row = {}
        for days in V4_HORIZONS:
            raw, p = anchor_raw_vote_and_pvalues(btc_train["close"], days, V4_BAND,
                                                 PVAL_SIGMA_DAYS)
            gated = lord_gate(raw, p, alpha, W0_FRACTION)
            n_delayed = count_delayed_episodes(raw, gated)
            row[days] = n_delayed
        a1_table[alpha] = row
        clears = any(v >= GATE_MIN_DELAYS for v in row.values())
        a1_any_clears = a1_any_clears or clears
        print(f"    {alpha:>6.2f} " + " ".join(f"{row[d]:>10d}" for d in V4_HORIZONS)
              + f"   {'YES' if clears else 'no'}")
    print(f"\n    A1 (>=1 alpha with >=1 anchor delaying >={GATE_MIN_DELAYS} "
          f"episodes): {'PASS' if a1_any_clears else 'FAIL (TRIPPED)'}")

    print(f"\n    A2 (R2_DEGENERACY_THRESH={R2_DEGENERACY_THRESH}): is the candidate's "
          "frac path a near-exact rescale of v4's own vote?")
    v4_frac_train = v4_vote_frac(btc_train).to_numpy()
    a2_all_below = True
    for alpha in ALPHA_GRID:
        frac_a = lord_frac(btc_train, alpha, PVAL_SIGMA_DAYS, W0_FRACTION)
        r2 = r_squared(frac_a, v4_frac_train)
        below = r2 < R2_DEGENERACY_THRESH
        a2_all_below = a2_all_below and below
        print(f"    alpha={alpha:>6.2f}  R^2(candidate frac, v4_vote_frac) = {r2:.6f}  "
              f"-> {'PASS' if below else 'FAIL (TRIPPED)'}")
    print(f"\n    A2 (ALL alphas stay below the degeneracy ceiling): "
          f"{'PASS' if a2_all_below else 'FAIL (TRIPPED)'}")

    kill_switches_ok = a1_any_clears and a2_all_below
    print(f"\n    Kill switches: {'PASS (proceeding)' if kill_switches_ok else 'TRIPPED -- STOP, NEGATIVE'}")
    if not kill_switches_ok:
        hr("VERDICT")
        print("    VERDICT: NEGATIVE (stopped at kill switches -- per the "
              "pre-registered rule)")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    # ========================================================== STEP 4
    hr("STEP 4 -- main sweep: ALPHA_GRID x {PVAL_SIGMA_DAYS, PVAL_SIGMA_DAYS_ALT}, "
       "full compare() per cell\n(BTC inner-train / inner-val, ETH replication, "
       "SPOT + FUTURES_5x)")
    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "ETH full")

    sigma_grid = (PVAL_SIGMA_DAYS, PVAL_SIGMA_DAYS_ALT)
    all_rows: dict[tuple[float, int], list[dict]] = {}
    for sigma_days in sigma_grid:
        for alpha in ALPHA_GRID:
            label = f"conservative_lord_a{alpha}_s{sigma_days}"
            bt = make_build_target(alpha, sigma_days, W0_FRACTION)
            rows = compare(bt, label=label, btc=btc, eth=eth)
            CONFIG_CELLS += len(rows)
            all_rows[(alpha, sigma_days)] = rows
            print(f"\n  -- alpha={alpha}, sigma_days={sigma_days} --")
            print_rows(rows)

    # ========================================================== STEP 5
    hr("STEP 5 -- decision rule (r160_shared.py's own pre-registration), "
       f"evaluated per alpha at the PRIMARY sigma_days={PVAL_SIGMA_DAYS}")
    decision_table = []
    any_promotes = False
    for alpha in ALPHA_GRID:
        rows = all_rows[(alpha, PVAL_SIGMA_DAYS)]
        label = f"conservative_lord_a{alpha}_s{PVAL_SIGMA_DAYS}"
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

        decision_table.append(dict(
            alpha=alpha, val_s=val_s, val_f=val_f, eth_s=eth_s, eth_f=eth_f,
            a_s=a_s, b_s=b_s, a_f=a_f, b_f=b_f, c_s=c_s, c_f=c_f,
            promote=promote,
        ))

        print(f"\n    alpha={alpha}:")
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

    # ========================================================== STEP 6
    hr("STEP 6 -- sigma_days robustness (PVAL_SIGMA_DAYS vs PVAL_SIGMA_DAYS_ALT), "
       "inner_val cells")
    print(f"\n    {'alpha':>6s} {'sigma':>6s} {'spot dSharpe':>13s} {'fut dSharpe':>12s}")
    for alpha in ALPHA_GRID:
        for sigma_days in sigma_grid:
            rows = all_rows[(alpha, sigma_days)]
            label = f"conservative_lord_a{alpha}_s{sigma_days}"
            vs = cell(rows, label, "inner_val", SPOT.name)
            vf = cell(rows, label, "inner_val", FUTURES.name)
            print(f"    {alpha:>6.2f} {sigma_days:>6d} {vs['d_sharpe']:>+13.3f} "
                  f"{vf['d_sharpe']:>+12.3f}")

    # ========================================================== STEP 7
    hr("STEP 7 -- configuration count")
    print(f"    Distinct (alpha, sigma_days) build_target configs swept: "
          f"{len(ALPHA_GRID)} x {len(sigma_grid)} = {len(ALPHA_GRID) * len(sigma_grid)}")
    print(f"    Each run through full compare() (3 slices x 2 markets = 6 cells): "
          f"{CONFIG_CELLS} total cells")
    print(f"    Plus: gamma unit checks, calibration self-test "
          f"({len(V4_HORIZONS)} anchors x {len(ALPHA_GRID)} alphas = "
          f"{len(V4_HORIZONS) * len(ALPHA_GRID)} cells on synthetic data, not counted "
          f"toward the trials ledger since no real-data Sharpe/growth number "
          f"comes from them), and the A1/A2 kill-switch sweep "
          f"({len(ALPHA_GRID)} alphas x {len(V4_HORIZONS)} anchors on BTC inner-train).")

    # ========================================================== VERDICT
    hr("VERDICT")
    print(f"    Any alpha in ALPHA_GRID satisfying r160_shared.py's "
          f"PROMOTE-CANDIDATE decision rule: {'YES' if any_promotes else 'NO'}")
    if any_promotes:
        winners = [d["alpha"] for d in decision_table if d["promote"]]
        print(f"    Alpha(s) clearing the bar: {winners}")
        print("    Per the pre-registered gate, this branch would move to the "
              "holdout ONLY after the operator freezes the specific alpha and "
              "logs it -- NOT done automatically by this script.")
    else:
        print("    VERDICT: NEGATIVE. No alpha in ALPHA_GRID clears all of "
              "clauses (a),(b) on both markets AND (c) on both markets on "
              "inner-validation.")
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
