"""Shared, read-only utilities and pre-registration for the R-147 round (08-26).

DIRECTION, in one sentence: `kelly_regime_v4`'s directional vote combines its
three anchors (20/40/80-day) by a fixed, unweighted 1/3-each average with no
notion of which anchor is currently more trustworthy -- replace that with
combination weights derived from a formal treatment of each anchor's own
estimated reliability: classical (known-variance) James-Stein shrinkage
across the 3 shipped anchors (conservative), or a sequential Bayesian
posterior over a 5-member alternative-ladder ensemble (novel).

**Which constraint this attacks: ERR.** No error control exists anywhere in
the vote's OWN COMBINATION step: `frac = (vote_20d + vote_40d + vote_80d) / 3`
(`kelly_regime_v3.py:64` / `kelly_regime.py`'s identical base construction) is
an arithmetic average with no estimate of which anchor is more reliable and
no shrinkage of noisy per-anchor estimates toward anything. This is not a
SIZE-axis proposal (it does not touch `scale`, the vol target, the deadband,
or propose a new trend statistic) -- both branches hold each anchor's OWN
point estimate (a plain rolling-mean-crossing vote, v4's own construction,
unmodified) fixed, and change only the WEIGHTS used to combine them. It is
also explicitly not an N~3 claim: the "3" James-Stein classically addresses
(simultaneously estimating >=3 means) is the 3 ANCHORS, unrelated to this
project's chronic ~3-regime-event effective sample size.

**Literature, fetched via WebSearch before either branch was dispatched:**

- James, W., & Stein, C. (1961), "Estimation with Quadratic Loss", *Proc. 4th
  Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1,
  361-379. For k>=3 simultaneously estimated means, shrinking each toward a
  common (grand) mean strictly dominates the individual (equal-trust/MLE)
  estimator in total quadratic risk. A purely statistical result: no cost
  model, no instrument count, not finance-specific.
- Efron, B., & Morris, C. (1975), "Data Analysis Using Stein's Estimator and
  its Generalizations", *JASA* 70(350), 311-319. The empirical-Bayes
  operationalization, `c = (k-3)*sigma_hat^2 / S`, requires the common
  variance to be ESTIMATED from the same k groups and is only non-degenerate
  for k>=4 -- a real wrinkle at exactly k=3 anchors, flagged honestly and
  resolved below (not glossed over): this round uses the classical
  KNOWN-VARIANCE James-Stein form instead, `c = (k-2)*sigma2/S` with `sigma2`
  a PLUG-IN (not jointly estimated) binomial variance -- valid at k=3 since
  k-2=1>0, and the standard textbook resolution of "the shrinkage constant is
  undefined at k=3" (see e.g. Efron & Morris 1977, "Stein's Paradox in
  Statistics", *Scientific American* 236(5), for the k>=3 known-variance
  case vs. k>=4 estimated-variance case distinction).
- Hoeting, J.A., Madigan, D., Raftery, A.E., Volinsky, C.T. (1999), "Bayesian
  Model Averaging: A Tutorial", *Statistical Science* 14(4), 382-417. The
  standard reference for combining several candidate models by (an
  approximation to) their posterior probability rather than by picking one
  or averaging unweighted -- the formal basis for the novel branch's
  ladder-ensemble posterior weighting. General statistics, no finance claim.
- Benhamou, Ohana, Etienne, Guez, Setrouk & Jacquot (2025), "Re-evaluating
  Short- and Long-Term Trend Factors in CTA Replication: A Bayesian
  Graphical Approach", arXiv:2507.15876. On real CTA-replication data
  (commodity futures, transaction costs included), a Bayesian graphical
  model that dynamically weights short-/long-term trend factors outperforms
  naive equal-weight and simple-ensemble baselines -- the closest recent
  precedent for "reweight the anchor horizons formally rather than average
  them", directly analogous to this project's own 20/40/80-day ladder.
  Read with the diversification caveat that sank R-05 attached: their result
  is on a diversified multi-instrument panel, not one instrument at 10bps,
  which is why this round is pre-registered modestly rather than as an
  expected repeat.
- Valeyre, S. (2025), "Breaking the Trend: How to Avoid Cherry-Picked
  Signals", arXiv:2504.10914 (already this project's B-42/R-92 citation).
  The standing counter-prediction: a single simple EMA is close to optimal
  for a trend system, and combining multiple indicators mainly risks
  cherry-picking rather than adding real information. Named here, before any
  code, as the reason both branches below are built to provably DEGENERATE
  to v4's own equal-weight vote when the data give no genuine evidence of
  unequal anchor/ladder reliability -- the mechanism cannot manufacture an
  edge out of noise by construction; see the Step-0 kill switches below.

**Not a duplicate of:**
- R-40 (`kelly_regime_v8_ladder_bag`): an UNWEIGHTED average across a bagged
  ladder set, for noise reduction. This round's weights are DERIVED from a
  formal reliability estimate of each component, never a flat average of
  more components.
- R-105 (`r105_conservative_anchor_jackknife.py` / `r105_novel_ladder_ensemble.py`):
  the closest methodological relative -- both use a delete-one-anchor
  jackknife (conservative) and the SAME 5-member alternative-ladder family
  reused verbatim below (novel, `BASES`/`LADDERS`). R-105 uses inter-model
  DISAGREEMENT as a multiplicative DISCOUNT applied on top of the shipped
  vote/scale (the shipped 20/40/80 vote is traded unmodified at every bar,
  per R-105's own Step-0 kill switch). This round does not discount
  anything: it changes the COMBINATION WEIGHTS that produce `frac` itself,
  and the shipped 3-of-3/5-of-5 equal-weight vote is not what is traded --
  a reliability-WEIGHTED vote is. Orthogonal use of the same ensemble.
- R-114 (`r114_conservative_lifetable_hazard.py` / `r114_novel_stratified_hazard.py`):
  genuinely uses empirical-Bayes shrinkage, but of a covariate-stratified
  HAZARD TABLE (probability the current regime spell ends soon, conditional
  on its age and volatility state), shrunk toward its own marginal table,
  feeding a discount multiplier on `frac*scale`. It never touches how the 3
  anchors are combined into `frac`.
- R-146 (robust anchor STATISTIC: median, jump-exclusion): changes each
  anchor's OWN point estimate of trend (a rolling median or jump-excluding
  mean, in place of v4's plain rolling mean), holding the equal 1/3
  combination fixed. This round holds each anchor's own point estimate (v4's
  plain rolling-mean-crossing vote, unmodified) fixed and changes only the
  combination weights -- the complementary half of the same architecture,
  deliberately untouched by R-146's own filing.
- `kelly_regime_v2` (NOT promoted): applies a fixed power-law transform
  `frac**gamma` to the vote's own scalar OUTPUT, after combination. Both
  branches below keep `vote_gamma=1.0` throughout, to avoid confounding
  with this already-closed, already-rejected mechanism.
- R-62 (vote-alone vs. scale-alone factor isolation): runs the SHIPPED,
  unmodified vote; does not touch its combination weights at all.
- Every SIZE-axis round (R-34...R-146, 26+ attempts): all retune `scale`'s
  magnitude/input or the vote's post-hoc transform/timing/statistic; none
  changes the WEIGHTS used to combine the 3 (or 5-ladder) anchor votes.
- `hedge_experts`' Hedge/multiplicative-weights combinator (R-129/R-130,
  closed): a regret-minimizing GRADIENT-based online-learning weight update
  over a 10-expert panel, shown under ablation to reduce to a smooth trading
  rule with no real learning signal. Neither branch below runs a gradient
  update of any kind: the conservative branch is a closed-form (non-
  iterative) shrinkage estimator recomputed independently at each bar from a
  trailing window; the novel branch is exact Beta-Bernoulli conjugate
  Bayesian updating (a cumulative posterior, not a regret-minimizing
  weight-update rule) -- a different mathematical object, and it reweights
  `kelly_regime_v4`'s own vote/ladder family, not `hedge_experts`' panel.
- Ledger-wide grep (this round, before any code) confirms zero prior hits
  for "James-Stein", "empirical Bayes" (outside R-114's unrelated
  hazard-table use), or "Bayesian model averaging" anywhere in the ledger.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it, the r89-r146 convention.
Nothing here reads a bar at or after OOS_START (2023-01-01); every function
that walks a data frame is either called through `assert_no_holdout`-guarded
slices (`compare()`, `run_slice()`, inherited unmodified from the
r102_shared -> r104_shared -> r105_shared chain) or restricted explicitly to
inner-train/inner-validation.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) The shrinkage/posterior weights never move materially away from
equal-weight (degenerate -- a relabeled v4; the Step-0 kill switches below
are built to catch exactly this before any Sharpe number is read).
(2) They move away from equal-weight, but the movement is uncorrelated with
(or opposed to) which anchor/ladder was actually more reliable GOING
FORWARD -- a real but harmful/inert reweighting, the pattern that has now
closed 6 of 7 prior ERR-axis constructions (R-87 x2, R-104, R-105 x2, R-114
x2). (3) Any improvement on BTC inverts sign on ETH (the specific failure
mode of R-109/R-112/R-114-novel) -- the natural risk here since reliability
is estimated from each instrument's own trailing history (BTC's, when run
on BTC; ETH's own, when run on ETH, since both branches are pure functions
of whatever `df` is passed to `compare()`, the r102_shared convention).
(4) The improvement, if any, sits inside the +/-0.2 Sharpe noise floor
(R-20) -- the modal outcome, given the base rate on this axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from r105_shared (itself chaining r104_shared,
# r103_shared, r102_shared): identical control machinery, so every number
# this round produces is directly comparable to R-102...R-146's own.
from experiments.r105_shared import (  # noqa: E402,F401
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
    causal_truncation_probe_vote,
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
    vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

# ------------------------------------------------------------------------
# Pre-registered constants shared by BOTH branches' Step-0 gate and B1-B6
# promotion bar -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------
FEE_TIER = 0.0040                 # 0.40% taker, B3 (the fee-robustness check)
SHARPE_NOISE_FLOOR = 0.2          # ROUTINE.md's own promotion bar (R-20)
R2_DEGENERACY_THRESH = 0.98       # A2 kill switch: candidate frac vs v4_vote_frac
BIND_FRAC_THRESH = 0.01           # A1 kill switch: weights must differ from
                                  # equal-weight on >1% of inner-train bars

# Conservative branch: pre-registered rolling-window (pseudo-spell-count)
# grid for the James-Stein plug-in variance, M in "number of trailing
# completed spells". PRIMARY is the first-listed value; the rest are a
# plateau/robustness check, per ROUTINE.md's "report the neighbours, not
# just the winner". Not fitted -- chosen a priori as a log-ish spread
# bracketing a "several months to a few years of spells" range.
JS_M_GRID = (8, 4, 16, 32)
JS_M_PRIMARY = JS_M_GRID[0]

# Novel branch: pre-registered 5-member alternative-ladder family, reused
# VERBATIM from R-105's own novel branch (same a-priori bases, not refit).
LADDER_BASES: tuple[int, ...] = (10, 15, 20, 25, 30)
LADDERS: dict[int, tuple[int, int, int]] = {b: (b, 2 * b, 4 * b) for b in LADDER_BASES}
assert LADDERS[20] == (20, 40, 80), LADDERS[20]

# Novel branch: pre-registered weakly-informative Beta prior grid for the
# Beta-Bernoulli posterior. PRIMARY is Beta(2,2) (weak, symmetric, prior
# mean 0.5 -- deliberately uninformative about which ladder is better before
# any data is seen). The rest are a robustness/plateau check.
BETA_PRIOR_GRID = ((2.0, 2.0), (1.0, 1.0), (5.0, 5.0))
BETA_PRIOR_PRIMARY = BETA_PRIOR_GRID[0]


# ================================================================== (1)
# Generic, causal spell/hit-rate primitives. Shared by BOTH branches so
# each anchor's (conservative) or each ladder's (novel) own reliability
# is measured by IDENTICAL machinery -- the r89-r146 convention.
# ==================================================================

def latched_state(frac: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Binarize a (possibly fractional, e.g. 0/.33/.67/1) vote fraction into
    a strict long(1.0)/flat(0.0) majority state. A single already-binary
    0/1 anchor vote passes through unchanged (threshold at 0.5 leaves 0->0,
    1->1). Ties are impossible for an odd number of equal-weight anchors."""
    return (frac > threshold).astype(float)


def spell_hit_series(state: pd.Series, close: pd.Series) -> pd.Series:
    """Causal spell/hit labelling for an arbitrary binary state series.

    A "spell" is a maximal run of a constant `state` value. At the bar a
    spell ENDS (state flips relative to the previous bar), this returns
    hit=1.0 if that just-ended spell's own realized price change (from the
    spell's first bar's close to its last bar's close) was directionally
    consistent with the state it held (price rose while state==1, did not
    rise while state==0), else 0.0. NaN everywhere else (mid-spell, and at
    bar 0).

    Causal by construction: the label at bar i uses only
    `close[seg_start .. i-1]`, i.e. bars strictly before i, and is assigned
    AT bar i (the first bar of the new spell) -- available for use starting
    that same bar onward, since it depends on nothing at or after bar i.
    """
    st = state.to_numpy()
    c = np.log(close.to_numpy())
    n = len(st)
    hit = np.full(n, np.nan)
    if n == 0:
        return pd.Series(hit, index=state.index)
    seg_start = 0
    for i in range(1, n):
        if st[i] != st[i - 1]:
            ended_cum = c[i - 1] - c[seg_start]
            seg_state = st[i - 1]
            h = 1.0 if (ended_cum > 0.0 if seg_state == 1.0 else ended_cum <= 0.0) else 0.0
            hit[i] = h
            seg_start = i
    return pd.Series(hit, index=state.index)


def rolling_reliability(hit: pd.Series, m: int) -> pd.Series:
    """Causal rolling mean hit-rate over the last `m` completed spells.

    Works on the sparse sub-series of actual hit EVENTS (bars where a spell
    just ended) so that `m` means "m spells", not "m bars" -- then forward-
    fills each event's value onto every bar until the next event. Causal:
    the rolling mean at an event bar uses only events at or before that bar
    (all in the past); ffill only propagates a value forward in time.
    Returns NaN before the first completed spell (warmup)."""
    events = hit.dropna()
    if len(events) == 0:
        return pd.Series(np.nan, index=hit.index)
    roll = events.rolling(m, min_periods=1).mean()
    return roll.reindex(hit.index).ffill()


def beta_bernoulli_posterior_mean(hit: pd.Series, a0: float, b0: float) -> pd.Series:
    """Causal, fully sequential (expanding, not rolling) Beta-Bernoulli
    posterior mean hit-rate: (a0 + cumulative_hits) / (a0 + b0 + n_spells).
    Same event-sparse-then-ffill construction as `rolling_reliability`, so
    it is causal for the identical reason. Returns NaN before the first
    completed spell (warmup)."""
    events = hit.dropna()
    if len(events) == 0:
        return pd.Series(np.nan, index=hit.index)
    cum_hits = events.cumsum()
    cum_n = pd.Series(np.arange(1, len(events) + 1), index=events.index, dtype=float)
    post_mean = (a0 + cum_hits) / (a0 + b0 + cum_n)
    return post_mean.reindex(hit.index).ffill()


def normalize_weights(values: np.ndarray, fallback_equal: bool = True) -> np.ndarray:
    """Row-normalize a (n_bars, k) array of non-negative reliability scores
    into weights summing to 1 along axis=1. Falls back to equal weights
    (1/k) wherever any column is NaN or the row sum is non-positive -- the
    warmup-safe default that reproduces v4's own equal-weight vote exactly
    whenever reliability is undefined or uninformative."""
    values = np.asarray(values, dtype=float)
    n, k = values.shape
    row_sum = np.nansum(values, axis=1)
    any_nan = np.any(~np.isfinite(values), axis=1)
    bad = any_nan | ~(row_sum > 0)
    out = np.divide(values, row_sum[:, None], out=np.full_like(values, 1.0 / k), where=~bad[:, None])
    if fallback_equal:
        out[bad] = 1.0 / k
    return out


# ================================================================== (2)
# Step-0 kill switches, applied identically to both branches' final
# candidate `frac` path before any Sharpe/holdout number is read.
# ==================================================================

def bind_frac(weights: np.ndarray) -> float:
    """Fraction of bars where the weight vector differs non-trivially
    (any component off by >1e-6) from equal-weight. A1 kill switch: must
    exceed BIND_FRAC_THRESH, else the branch never leaves equal-weight and
    is a relabeling of v4 by construction, not a tested mechanism."""
    k = weights.shape[1]
    equal = 1.0 / k
    differs = np.any(np.abs(weights - equal) > 1e-6, axis=1)
    return float(np.mean(differs))


def build_target_from_frac(frac: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Wire an arbitrary (already-computed) directional vote fraction path
    through v4's OWN unmodified scale/deadband machinery -- the only
    difference between a candidate and v4 is `frac`'s own construction."""
    scale = v4_scale(df)
    desired = np.asarray(frac, dtype=float) * scale
    return apply_deadband(desired)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Fast checks on synthetic data, mirroring r89-r146_shared.py's own
    `_self_test()` convention: run at import time, fail loudly and early."""
    idx = pd.date_range("2017-01-01", periods=150_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(147)
    regime_len = 20_000
    n_regimes = len(idx) // regime_len + 1
    drift_signs = np.resize([1.0, -1.0], n_regimes)
    drift_per_bar = np.repeat(drift_signs, regime_len)[: len(idx)] * 0.00004
    innov = rng.normal(0, 0.0006, len(idx))
    close = 10_000 * np.exp(np.cumsum(innov + drift_per_bar))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) latched_state / spell_hit_series on a trivial synthetic ladder.
    frac = vote_frac(df, V4_HORIZONS, V4_BAND)
    state = latched_state(frac)
    assert set(np.unique(state.to_numpy())) <= {0.0, 1.0}
    hit = spell_hit_series(state, df["close"])
    events = hit.dropna()
    assert len(events) > 5, "expected several spell transitions on synthetic trending data"
    assert events.between(0.0, 1.0).all()

    # (2) causal truncation probe on spell_hit_series and both reliability
    # estimators, via the shared generic series-truncation probe.
    def _hit_builder(d: pd.DataFrame) -> np.ndarray:
        return spell_hit_series(latched_state(vote_frac(d, V4_HORIZONS, V4_BAND)),
                                d["close"]).to_numpy()

    def _reliab_builder(d: pd.DataFrame) -> np.ndarray:
        h = spell_hit_series(latched_state(vote_frac(d, V4_HORIZONS, V4_BAND)), d["close"])
        return rolling_reliability(h, JS_M_PRIMARY).to_numpy()

    def _post_builder(d: pd.DataFrame) -> np.ndarray:
        h = spell_hit_series(latched_state(vote_frac(d, V4_HORIZONS, V4_BAND)), d["close"])
        return beta_bernoulli_posterior_mean(h, *BETA_PRIOR_PRIMARY).to_numpy()

    assert causal_truncation_probe_series(_hit_builder, df)
    assert causal_truncation_probe_series(_reliab_builder, df)
    assert causal_truncation_probe_series(_post_builder, df)

    # (3) rolling_reliability / beta_bernoulli_posterior_mean stay in [0, 1]
    # where defined, and equal a sensible base rate on this synthetic
    # trending series (roughly > 0.5, since the series has real drift).
    rel = rolling_reliability(hit, JS_M_PRIMARY)
    fin = rel.dropna()
    assert len(fin) > 3 and fin.between(0.0, 1.0).all()
    post = beta_bernoulli_posterior_mean(hit, *BETA_PRIOR_PRIMARY)
    fin2 = post.dropna()
    assert len(fin2) > 3 and fin2.between(0.0, 1.0).all()

    # (4) normalize_weights: equal input -> equal output; NaN row -> equal
    # fallback; a dominant column -> majority (not necessarily all) weight
    # on that column.
    w = normalize_weights(np.array([[0.5, 0.5, 0.5], [0.9, 0.1, 0.1], [np.nan, 0.4, 0.4]]))
    assert np.allclose(w[0], 1.0 / 3.0)
    assert np.allclose(w[2], 1.0 / 3.0)  # NaN row -> equal fallback
    assert w[1, 0] > w[1, 1] == w[1, 2]

    # (5) bind_frac: all-equal weights -> 0.0; a perturbed row -> > 0.
    k = 3
    eq = np.full((100, k), 1.0 / k)
    assert bind_frac(eq) == 0.0
    pert = eq.copy()
    pert[50] = [0.5, 0.3, 0.2]
    assert abs(bind_frac(pert) - 0.01) < 1e-9

    # (6) build_target_from_frac reproduces v4_target exactly when fed
    # v4's own unmodified vote fraction.
    assert np.allclose(build_target_from_frac(v4_vote_frac(df).to_numpy(), df), v4_target(df))

    # (7) LADDERS sanity.
    assert LADDERS[20] == V4_HORIZONS
    for b in LADDER_BASES:
        assert LADDERS[b] == (b, 2 * b, 4 * b)


_self_test()
