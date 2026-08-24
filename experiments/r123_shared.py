"""Shared, read-only utilities and pre-registration for the R-123 round (08-24).

DIRECTION, in one sentence: take the ERR-axis distributional-novelty statistic
five prior rounds (R-109, R-112, R-115, R-121, R-122) have all applied as a
MULTIPLICATIVE DISCOUNT bolted onto `kelly_regime_v4`'s already-deadbanded
final target -- `v4_target(df) * (1 - discount)` -- and move it INTO the
vote/scale computation itself, the one architectural variant R-122's own
backlog re-ranking names as the last untried axis of variation before this
ERR sub-axis is closed for good.

**Why this and not a sixth reference-pool/feature-map/data-source repair.**
Docs/LEDGER.md's own "Re-ranked 08-24 after R-122" backlog entry: "This
closes the distributional-novelty ERR sub-axis's third and last named axis
of variation: reference-pool construction (R-112, R-115), feature map
(R-121), and reference data source -- real vs. synthetic (R-122) -- five
attempts across three independent axes, 0 of 5 closing the ETH gap. A
future session preferring this specific discount-on-`frac*scale`
architecture for a novelty brake needs it to enter the vote or scale
directly rather than sit as a multiplicative discount on top of them;
otherwise this ERR sub-axis is closed." This round takes up that named,
concrete next step verbatim, holding the feature panel and distance
statistic fixed at R-109's own original (real rolling 730-day BTC history)
convention -- the one thing every prior round already varied -- so
architecture is the single new variable under test.

**Literature grounding, fetched and read via WebSearch this round:**

- Baker, R. D., & McHale, I. G. (2013), "Optimal Betting Under Parameter
  Uncertainty: Improving the Kelly Criterion", *Decision Analysis* 10(3),
  189-199. The CONSERVATIVE branch's exact mechanism: the Kelly criterion
  as usually applied treats the estimated edge as certain; Baker & McHale
  show that under genuine parameter uncertainty the growth-optimal policy
  SHRINKS the edge estimate itself (not the resulting stake, after the
  stake has already been computed and sized) toward "no edge" in proportion
  to how uncertain that estimate is. Applied here: `frac` (the 3-anchor
  vote, kelly_regime_v4's own stand-in for "the market currently has a
  positive-drift edge") is the quantity that should shrink toward its
  neutral/no-edge value (0, flat -- see `kelly_regime_v4`'s own docstring:
  "when the crowd turns distributor, the Kelly fraction of a negative-drift
  bet is zero") under high estimation uncertainty, BEFORE it is multiplied
  by `scale` and BEFORE the deadband -- not the final sized+deadbanded
  position, discounted after the fact. This is a different economic claim
  from R-109's ("the market state looks unlike anything the vote has been
  calibrated against, so reduce exposure defensively") even though both use
  the identical novelty statistic as their uncertainty proxy: Baker-McHale's
  claim is specifically about the RELIABILITY OF THE EDGE ESTIMATE, and it
  says shrink the estimate, not the bet.
- Sun, Q., & Boyd, S. (2018), "Distributional Robust Kelly Gambling:
  Optimal Strategy under Uncertainty in the Long-Run", arXiv:1812.10371.
  The NOVEL branch's mechanism: rather than a single fixed reference
  distribution, the bettor optimizes worst-case log-growth over a
  divergence ball of radius `epsilon` around the empirical/estimated return
  distribution, and the robust-optimal fraction is a function of that whole
  ball, not a point estimate corrected after the fact. Applied here: the
  novelty state sets the ball's radius (`epsilon = EPS_MAX * state`) and a
  single JOINT multiplier -- derived below from a first-order approximation,
  not the paper's own convex program, which is intractable to re-solve at
  every bar inside this project's causal per-bar simulation loop -- is
  applied to the PRE-deadband product `frac * scale` jointly, rather than
  being decomposed into a vote-side or scale-side correction the way the
  conservative branch's shrink is. This is architecturally distinct from
  the conservative branch in two ways: (a) it does not decompose into "shrink
  frac, leave scale alone" -- it treats frac*scale as one joint quantity to
  correct, motivated by the fact that Sun & Boyd's ambiguity ball is over
  the RETURN DISTRIBUTION as a whole, which drives both the vote's implied
  edge and the volatility target's implied risk simultaneously, not two
  independently-estimated quantities; (b) its functional form in `state` is
  a DIFFERENT SHAPE (concave, `sqrt`, derived below) from the conservative
  branch's and R-109's own piecewise-LINEAR ramp, so the two branches also
  differ in how aggressively they discount near the threshold vs. at the
  extreme.

  **The approximation, stated explicitly (this is a disclosed
  simplification, not a re-derivation of Sun & Boyd's own convex program):**
  Pinsker's inequality (a standard information-theoretic bound; see e.g.
  Csiszar 1967 or any information theory text) bounds the total-variation
  distance between two distributions by their KL divergence:
  `TV(P,Q) <= sqrt(KL(P||Q) / 2)`. Treating `epsilon` as a KL-divergence
  budget, the worst-case adversarial perturbation within that budget can
  move probability mass by at most `sqrt(epsilon/2)` in total variation. A
  first-order Taylor expansion of Kelly log-growth under a small
  probability-mass perturbation of that size gives a correction to the
  optimal fraction that scales linearly in the TV bound, i.e. as
  `sqrt(epsilon)` -- concave near `epsilon=0`, unlike the conservative
  branch's linear-in-`state` ramp. `NOVEL_C` is the proportionality constant
  from that first-order expansion, swept on a pre-registered grid rather
  than derived exactly (the exact constant depends on the local curvature
  of the log-growth function at the current `frac*scale`, which this round
  does not attempt to estimate online); this is disclosed explicitly so a
  future reader does not mistake it for an exact re-implementation of Sun &
  Boyd's own LP.

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path). This is the sixth attempt at the distributional-novelty
ERR sub-axis specifically (R-109, R-112, R-115, R-121, R-122, this round),
and the eleventh ERR-axis attempt project-wide across all four notions of
uncertainty this project's framework has proposed (sampling significance:
R-87, R-104 x2; specification/model disagreement: R-105 x2, R-106 x2;
distributional novelty: R-109, R-112, R-115, R-121, R-122; temporal
duration dependence: R-114 x2).

**Not a duplicate of:**
- R-109 / R-112 / R-115 / R-121 / R-122 (all five): every one applies
  `v4_target(df) * (1 - discount)` -- a multiplicative discount on the
  FINAL, post-deadband target -- verified unchanged across all five by each
  round's own Step-0 `R2_VS_V4_THRESH` kill switch (candidate path must
  differ from a pure v4 rescale, but the FORM of that difference, in every
  prior round, is "the same path, uniformly shrunk"). This round's two
  branches instead modify `frac` (conservative) or the pre-deadband
  `frac*scale` product jointly (novel) BEFORE the deadband is applied --
  which means the deadband itself now sees a different input and can latch
  or release differently than it would under any prior round's construction,
  a genuine behavioral difference the discount-after-deadband architecture
  cannot produce by construction (post-deadband discounting can never change
  WHEN a rebalance triggers, only how large the resulting position is once
  one does).
- Every SIZE-axis round (R-34...R-122's own SIZE-axis lineage, 27+
  attempts): all retune `scale`'s magnitude directly or supply a different
  volatility/state input to it, with no novelty/uncertainty statistic
  involved at all. This round's `scale` computation
  (`v4_symmetric_vol` -> `conditional_target_scale`) is reproduced
  byte-for-byte unchanged; only what multiplies against it changes.
- Every other ERR-axis round (R-87, R-104, R-105, R-106, R-114): none of
  them used a distributional-novelty statistic, and none of them altered
  where in the pipeline (vote vs. scale vs. final target) their own
  uncertainty proxy entered -- this round's novelty is specifically the
  APPLICATION POINT, not the uncertainty statistic (which is reused
  verbatim from R-109) or a new notion of uncertainty.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither branch may edit it. Nothing here reads
a bar at or after OOS_START (2023-01-01); every function that walks a data
frame is either called through `assert_no_holdout`-guarded slices (`compare()`,
`run_slice()`, inherited unmodified through r102_shared -> ... -> r109_shared)
or is explicitly restricted to non-holdout ranges by the caller.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) If B4 (the ETH falsification test) fails again on both branches, that is
the SIXTH consecutive attempt on this ERR sub-axis to fail it, after varying
reference-pool construction, feature map, reference data source, AND NOW
architecture -- the last axis of variation the backlog names. A sixth
failure closes this sub-axis in full generality (not merely "this
architecture doesn't work either") and strengthens the calibration-window
explanation (BTC's 2017-2020-supercycle-dominated reference, named as the
live suspect in R-115's own re-ranking) over any single mechanism
explanation, since architecture was the one variable this round's design
holds as the sole difference from five prior failures.
(2) Baker-McHale's own result is derived for a discrete win/loss bet with a
single estimated probability; `kelly_regime_v4`'s vote is a continuous,
already-latched three-way average, and mapping "shrink toward no-edge" onto
"multiply the vote fraction toward zero" is itself a modeling choice this
round makes explicit rather than a direct application of their closed form
-- if that mapping is wrong, the conservative branch could fail not because
shrinkage-Kelly is the wrong idea but because THIS particular
operationalization of it onto a latched multi-anchor vote is.
(3) The novel branch's `sqrt`-shaped correction is a first-order
approximation with an unfit constant (`NOVEL_C`, swept on a small grid
rather than derived); if the true worst-case correction is smaller (weaker
than linear in the TV bound) or much larger (dominated by higher-order
terms this round's Taylor expansion drops), the swept grid may miss the
region where the mechanism would have worked, which is a genuine risk of
this specific approximation rather than of distributionally-robust Kelly as
an idea.
(4) Modifying `frac` or the pre-deadband product changes deadband
triggering, which could increase turnover relative to every prior round's
post-deadband-only discount -- a cost-side failure mode none of R-109
through R-122 could produce by construction, checked here via each
branch's own reported trade count against v4's unmodified 143 (full period)
baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from r109_shared (itself chaining r106_shared ->
# r105_shared -> r102_shared): identical control machinery and identical
# novelty features/distance statistics, so every number this round produces
# is directly comparable to R-109...R-122's.
from experiments.r109_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BASELINE_WINDOW_DAYS,
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    ETH_SLICE_NAME,
    FEATURE_BUILDERS,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_REF_DAYS,
    NOVEL_FEATURE_BUILDERS,
    OOS_START,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    SLICES,
    SPOT,
    TargetStrategy,
    align_daily_to_bars,
    apply_deadband,
    assert_no_holdout,
    build_daily_features,
    causal_rolling_percentile_rank,
    compare,
    fee_at,
    load_btc,
    load_eth,
    novelty_discount,
    paired_diff,
    print_rows,
    r_squared,
    rolling_knn_distance,
    rolling_mahalanobis_distance,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

from experiments.r105_shared import (  # noqa: E402,F401
    FEE_TIER,
    SHARPE_NOISE_FLOOR,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    inner_val_rows,
    print_plateau_table,
)

from experiments.r102_shared import (  # noqa: E402,F401
    V4_HORIZONS,
    causal_truncation_probe_series,
    v4_symmetric_vol,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# Reference construction (features, window, min-periods) is UNCHANGED from
# R-109 on purpose: this round's only new variable is the application point.
# ------------------------------------------------------------------------

# Conservative branch: Mahalanobis distance + frac-shrink, R-109's own
# conservative convention (thresh/max_discount grid identical to R-109).
CONS_THRESH_GRID = (0.80, 0.90, 0.95)
CONS_MAXD_GRID = (0.5, 1.0)
CONS_PRIMARY = (0.90, 1.0)
CONS_SELECTION_ORDER = ((0.90, 1.0), (0.95, 1.0), (0.80, 0.5), (0.90, 0.5), (0.95, 0.5), (0.80, 1.0))

# Novel branch: kNN distance (k=10) + joint sqrt-shaped robust multiplier,
# R-109's own novel convention for the distance statistic. EPS_MAX*C^2 spans
# the same {0.5, 1.0} max-discount-at-state=1 range as the conservative grid
# for comparability, via mult(state=1) = 1 - C*sqrt(EPS_MAX).
NOVEL_EPS_GRID = (0.25, 0.5)
NOVEL_C_GRID = (1.0, 1.414)
NOVEL_PRIMARY = (0.5, 1.414)   # mult(state=1) = 1 - 1.414*sqrt(0.5) = 1 - 1.0 = 0.0 (full discount)
NOVEL_SELECTION_ORDER = ((0.5, 1.414), (0.5, 1.0), (0.25, 1.414), (0.25, 1.0))


# ------------------------------------------------------------------------
# (1) Conservative application point: shrink `frac`, not the final target.
# ------------------------------------------------------------------------

def frac_shrink_fraction(df: pd.DataFrame, state: pd.Series, thresh: float,
                          max_discount: float) -> np.ndarray:
    """The [0,1] shrink FRACTION applied to `frac`, identical formula to
    R-109's `novelty_discount` (piecewise-linear ramp from `thresh` to 1),
    aligned onto bars. Reused as-is: this round's difference is WHERE this
    number is applied, not what it is."""
    aligned = align_daily_to_bars(state, df).fillna(0.0)
    return novelty_discount(aligned, thresh, max_discount).to_numpy()


def conservative_target(df: pd.DataFrame, state: pd.Series, thresh: float,
                         max_discount: float) -> np.ndarray:
    """frac shrunk toward 0 (flat) by the novelty state, THEN multiplied by
    scale, THEN deadbanded -- Baker & McHale's shrink-the-edge-estimate
    applied to v4's own vote, in place of R-109's shrink-the-final-position."""
    shrink = frac_shrink_fraction(df, state, thresh, max_discount)
    frac = v4_vote_frac(df).to_numpy()
    scale = v4_scale(df)
    desired = frac * (1.0 - shrink) * scale
    return apply_deadband(desired)


# ------------------------------------------------------------------------
# (2) Novel application point: joint sqrt-shaped multiplier on frac*scale,
# pre-deadband, distinct functional form from the conservative branch.
# ------------------------------------------------------------------------

def robust_multiplier(state_arr: np.ndarray, eps_max: float, c: float) -> np.ndarray:
    """Sun & Boyd (2018)-motivated worst-case correction (see module
    docstring for the Pinsker-bound derivation): `epsilon = eps_max*state`,
    `multiplier = 1 - c*sqrt(epsilon)`, clipped to [0, 1]. Concave in
    `state` near 0 -- a different shape from the conservative branch's
    piecewise-linear ramp, with no threshold (binds at every state > 0,
    unlike the conservative branch's `thresh`-gated ramp)."""
    eps = eps_max * np.clip(state_arr, 0.0, 1.0)
    mult = 1.0 - c * np.sqrt(eps)
    return np.clip(mult, 0.0, 1.0)


def novel_discount_fraction(df: pd.DataFrame, state: pd.Series, eps_max: float,
                             c: float) -> np.ndarray:
    """The [0,1] amount removed from frac*scale (`1 - multiplier`), for the
    Step-0 R2-vs-vol kill switch."""
    aligned = align_daily_to_bars(state, df).fillna(0.0).to_numpy()
    return 1.0 - robust_multiplier(aligned, eps_max, c)


def novel_target(df: pd.DataFrame, state: pd.Series, eps_max: float, c: float) -> np.ndarray:
    """frac*scale jointly multiplied (pre-deadband) by the robust
    correction -- the round's "novel" claim: the correction is not
    decomposed into a vote-side or scale-side term, and its shape in
    `state` differs from the conservative branch's."""
    aligned = align_daily_to_bars(state, df).fillna(0.0).to_numpy()
    mult = robust_multiplier(aligned, eps_max, c)
    frac = v4_vote_frac(df).to_numpy()
    scale = v4_scale(df)
    desired = frac * scale * mult
    return apply_deadband(desired)


# ------------------------------------------------------------------------
# (3) Generalized Step-0 gate: takes precomputed arrays so both branches'
# differing architectures (frac-shrink vs. joint-multiplier) share one
# kill-switch implementation rather than each re-deriving it.
# ------------------------------------------------------------------------

def step0_gate_generic(df: pd.DataFrame, candidate_path: np.ndarray,
                        discount_fraction: np.ndarray, state: pd.Series,
                        bind_frac_thresh: float = BIND_FRAC_THRESH,
                        r2_v4_thresh: float = R2_VS_V4_THRESH,
                        r2_vol_thresh: float = R2_VS_VOL_THRESH,
                        cv_kill_thresh: float = CV_KILL_THRESH) -> dict:
    """Same four kill switches as R-109's `step0_gate` (bind_frac, not-a-v4-
    rescale, not-a-vol-rescale, non-degenerate state), generalized to accept
    an arbitrary candidate path and discount-fraction array so it applies to
    both this round's architectures without re-deriving the formula that
    produced them."""
    disc_arr = np.asarray(discount_fraction, dtype=float)
    finite = np.isfinite(disc_arr)
    bind_frac = float(np.mean(disc_arr[finite] > 1e-9)) if finite.any() else 0.0
    bind_ok = bind_frac > bind_frac_thresh

    cand = np.asarray(candidate_path, dtype=float)
    v4_path = v4_target(df)
    n = min(len(cand), len(v4_path))
    both = np.isfinite(cand[:n]) & np.isfinite(v4_path[:n])
    r2_v4 = r_squared(cand[:n][both], v4_path[:n][both]) if both.any() else 1.0
    not_v4_rescale = r2_v4 < r2_v4_thresh

    vol_path = np.asarray(v4_symmetric_vol(df), dtype=float)
    m = min(len(disc_arr), len(vol_path))
    both_vol = np.isfinite(disc_arr[:m]) & np.isfinite(vol_path[:m])
    r2_vol = r_squared(disc_arr[:m][both_vol], vol_path[:m][both_vol]) if both_vol.any() else 1.0
    not_vol_rescale = r2_vol < r2_vol_thresh

    s = state.to_numpy(dtype=float) if hasattr(state, "to_numpy") else np.asarray(state, dtype=float)
    s_finite = s[np.isfinite(s)]
    cv = float(s_finite.std() / s_finite.mean()) if len(s_finite) and s_finite.mean() else float("nan")
    non_degenerate = np.isfinite(cv) and cv >= cv_kill_thresh

    passed = bind_ok and not_v4_rescale and not_vol_rescale and non_degenerate
    return dict(bind_frac=bind_frac, bind_ok=bind_ok, r2_vs_v4=r2_v4,
                not_v4_rescale=not_v4_rescale, r2_vs_vol=r2_vol,
                not_vol_rescale=not_vol_rescale, state_cv=cv,
                non_degenerate=non_degenerate, passed=passed)


def print_step0_report(label: str, gate: dict) -> None:
    print(f"\n--- Step-0 gate: {label} ---")
    print(f"bind_frac={gate['bind_frac']:.4f} (kill <= {BIND_FRAC_THRESH}) -> "
          f"{'ok' if gate['bind_ok'] else 'KILL'}")
    print(f"R^2 vs v4_target={gate['r2_vs_v4']:.4f} (kill >= {R2_VS_V4_THRESH}) -> "
          f"{'ok' if gate['not_v4_rescale'] else 'KILL (near-exact v4 rescale)'}")
    print(f"R^2 vs v4 realized vol={gate['r2_vs_vol']:.4f} (kill >= {R2_VS_VOL_THRESH}) -> "
          f"{'ok' if gate['not_vol_rescale'] else 'KILL (relabelled vol rescale)'}")
    print(f"state CoV={gate['state_cv']:.4f} (kill < {CV_KILL_THRESH}) -> "
          f"{'ok' if gate['non_degenerate'] else 'KILL (degenerate)'}")
    verdict = "PASS" if gate["passed"] else "FAIL"
    print(f"STEP-0 GATE VERDICT ({label}): {verdict}")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=250_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(123)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
                       index=idx)

    daily = build_daily_features(df)
    dist_maha = rolling_mahalanobis_distance(daily)
    state_maha = causal_rolling_percentile_rank(dist_maha, window=BASELINE_WINDOW_DAYS,
                                                 min_periods=MIN_REF_DAYS)

    # (1) Conservative: causal-truncation probe on the whole target builder.
    def _cons_builder(frame):
        d = build_daily_features(frame)
        dist = rolling_mahalanobis_distance(d)
        st = causal_rolling_percentile_rank(dist, window=BASELINE_WINDOW_DAYS,
                                             min_periods=MIN_REF_DAYS)
        return conservative_target(frame, st, *CONS_PRIMARY)

    assert causal_truncation_probe_series(_cons_builder, df)

    cons_path = _cons_builder(df)
    assert np.isfinite(cons_path).sum() > 1000
    # Shrinking frac toward 0 must never PRODUCE a larger magnitude than v4's own path.
    v4_path = v4_target(df)
    m = np.isfinite(cons_path) & np.isfinite(v4_path)
    assert np.all(np.abs(cons_path[m]) <= np.abs(v4_path[m]) + 1e-9), \
        "frac-shrink target exceeds v4's own magnitude somewhere -- shrink math is wrong"

    # (2) Novel: robust_multiplier sanity -- monotone non-increasing in state,
    # bounded in [0,1], multiplier(0) == 1 (no correction at zero novelty).
    st_grid = np.linspace(0, 1, 101)
    mult = robust_multiplier(st_grid, eps_max=NOVEL_PRIMARY[0], c=NOVEL_PRIMARY[1])
    assert np.isclose(mult[0], 1.0)
    assert (np.diff(mult) <= 1e-12).all(), "robust_multiplier is not monotone non-increasing"
    assert (mult >= 0.0).all() and (mult <= 1.0).all()
    # At the primary cell, mult(state=1) should be (approximately) 0 by construction.
    assert mult[-1] < 0.05, f"NOVEL_PRIMARY does not reach near-full discount at state=1: {mult[-1]}"

    def _novel_builder(frame):
        d = build_daily_features(frame, NOVEL_FEATURE_BUILDERS)
        dist = rolling_knn_distance(d, k=10)
        st = causal_rolling_percentile_rank(dist, window=BASELINE_WINDOW_DAYS,
                                             min_periods=MIN_REF_DAYS)
        return novel_target(frame, st, *NOVEL_PRIMARY)

    assert causal_truncation_probe_series(_novel_builder, df)
    novel_path = _novel_builder(df)
    assert np.isfinite(novel_path).sum() > 1000
    m2 = np.isfinite(novel_path) & np.isfinite(v4_path)
    assert np.all(np.abs(novel_path[m2]) <= np.abs(v4_path[m2]) + 1e-9), \
        "joint-multiplier target exceeds v4's own magnitude somewhere"

    # (3) step0_gate_generic smoke test: a discount fraction IDENTICAL to
    # v4's own vol path must fail the not_vol_rescale kill switch (r_squared
    # of a series against itself is exactly 1.0, >= R2_VS_VOL_THRESH), and a
    # discount fraction independent of vol (white noise) must pass it.
    vol_path = np.asarray(v4_symmetric_vol(df), dtype=float)
    fake_gate = step0_gate_generic(df, v4_path, vol_path, state_maha)
    assert fake_gate["not_vol_rescale"] is False, "step0_gate_generic failed to catch a vol rescale"
    noise_discount = rng.uniform(0, 1, len(vol_path))
    real_gate = step0_gate_generic(df, cons_path, noise_discount, state_maha)
    assert real_gate["not_vol_rescale"] is True, "step0_gate_generic flagged independent noise as a vol rescale"


_self_test()
