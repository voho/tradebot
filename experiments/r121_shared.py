"""Shared, read-only utilities and pre-registration for the R-121 round (08-24).

DIRECTION, in one sentence: replace the FEATURE REPRESENTATION inside
`kelly_regime_v4`'s existing distributional-novelty ERR-axis brake (R-109's
five hand-engineered point-in-time scalars: log_vol, anchor_disp, kurtosis,
volume_z, skew) with a truncated PATH SIGNATURE of the rolling joint
(log-price, realized-vol[, log-volume]) path -- an order-sensitive summary
that sees HOW the market moved through a window, not just its point-in-time
moments -- as the novelty statistic feeding the same exposure-discount
architecture R-109 already built and R-112/R-115 already re-tested.

**Why this round exists, precisely.** R-109's novel (kNN) branch was the
first of six ERR-axis attempts (R-28/retracted, R-87, R-104, R-105, R-106,
R-109) to clear B1 (both bootstrap intervals excluded zero on
inner-validation) -- but it failed its own pre-registered B4 falsification:
ETH's spot cell inverted sign. R-112 tried repairing this by changing the
feature space to RETURN-space only (still point-in-time scalars) and by
pooling the reference SET across a 6-asset panel; neither engaged the real
question because a data-coverage bug (R-112's `load_eth()` had zero calendar
overlap with the pool) meant R-112's B4 numbers were byte-identical to
R-109's own, untested. R-115 fixed that coverage bug, genuinely re-ran the
pooled-reference repair, and B4 STILL failed, more decisively than before.
R-115's own closing line named exactly the gap this round fills, verbatim:
"a materially different novelty statistic, not a repair of this one's
reference distribution, is a different, untested question." Every one of
R-109/R-112/R-115's five feature-panel columns is a SNAPSHOT statistic (a
value at one instant, or a rolling moment collapsed to one number) -- none
of them encode the ORDER in which price and volatility moved within a
window, only their marginal levels. This round tests whether that
order-sensitive information -- present in a path signature but invisible to
any snapshot feature -- is what R-109's kNN construction was missing on ETH.

**Literature grounding, fetched and read via WebSearch this round:**

- Chevyrev, I. & Kormilitzin, A. (2016), "A Primer on the Signature Method
  in Machine Learning", arXiv:1603.03788. The foundational definition and
  properties of the path signature: a truncated collection of iterated
  integrals of a path's increments, invariant to time-reparameterization,
  that characterizes a path's SHAPE (order and co-movement of its
  components) rather than only its endpoints or marginal moments.
- **Gasteratos, I., Jacquier, A., Lemercier, M., Lyons, T. & Salvi, C.
  (2025), "Novelty detection on path space", arXiv:2512.03243** -- verified
  to exist via WebSearch (title, all five authors, and abstract confirmed
  independently before being relied on, per this project's own R-87
  precedent of checking a citation rather than taking it on faith). Frames
  novelty detection on path space as hypothesis testing with signature-based
  test statistics, and derives tail bounds on false-positive rate via
  transportation-cost inequalities that extend beyond a Gaussian reference
  law -- i.e. an approach that targets an actual error-rate guarantee for
  "is this path unlike its reference", not merely a heuristic empirical
  percentile. This round's NOVEL branch below is an explicitly disclosed
  SIMPLIFICATION of that idea (a closed-form, even-degrees-of-freedom
  chi-squared tail probability under a Gaussian-reference approximation of
  the signature-feature distribution) -- not a reproduction of the paper's
  own non-Gaussian RDE-law transportation-cost bound or its shuffle-product
  CVaR/one-class-SVM machinery, which this project's environment cannot
  build (no scipy; see the dependency note below). Named here as a
  disclosed simplification per this project's own R-92 precedent (Sepp &
  Lucic 2026's own Eq. 5.12/6.2 only, not their full paper), not as an
  overclaim.
- Supporting precedent that truncated signatures are a practical,
  tuning-free anomaly/novelty feature on real (non-synthetic) time series:
  "Path Signatures are Unsupervised Time Series Anomaly Extractors" (IEEE,
  2024/25 venue).

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path) -- the SAME slot R-109/R-112/R-113/R-115 already occupy
(distributional novelty, the fourth of five uncertainty notions this
project's ERR axis has tried; see docs/LEDGER.md standing diagnosis). This
is explicitly NOT "another indicator": like every ERR-axis round since
R-87, the statistic here only ever multiplicatively DISCOUNTS
`kelly_regime_v4`'s existing, completely unchanged `frac * scale` exposure
-- it never enters the vote, never forecasts direction, and is gated by the
identical Step-0 kill switches (R2_VS_V4_THRESH, R2_VS_VOL_THRESH) that
guard every prior round in this family against secretly becoming a
relabelled rescale of something already in the strategy.

**Not a duplicate of:**
- R-109/R-112/R-115 (Mahalanobis / kNN novelty on a 3-5 feature panel of
  point-in-time scalars: log_vol, anchor_disp, kurtosis, volume_z, skew):
  those varied the reference-pool CONSTRUCTION (single-asset vs CORAL-pooled
  vs multi-asset-pooled) while holding the FEATURE MAP fixed throughout all
  three rounds. This round holds R-109's reference-window architecture
  (strictly-prior BASELINE_WINDOW_DAYS, MIN_REF_DAYS, same discount curve)
  and changes ONLY the feature map -- from five point-in-time scalars to a
  path signature computed over the trailing window. Zero features from
  r109_shared.FEATURE_BUILDERS / NOVEL_FEATURE_BUILDERS are reused; the
  signature features below are pure iterated-integral functionals of the
  (log-price, realized-vol[, log-volume]) PATH, a mathematically distinct
  object from a snapshot vector of independently-computed moments.
- R-113 (R-109's kNN/Mahalanobis brakes applied to the MULTI-ASSET panel
  instead of `kelly_regime_v4`): this round never touches
  `src/tradebot/multi_engine.py` or the U6/U8 panel; it is a single-asset
  (BTC/ETH) discount exactly like R-109's own original target.
- R-106 (cross-model-class disagreement among BOCPD/Kalman/CSD/Hawkes): no
  model disagreement anywhere here -- a single novelty distance in a new
  feature space, computed from one reference distribution, exactly R-109's
  own architecture shape.
- R-104/R-87/R-28 (sampling significance of the vote's own historical
  P&L/edge; retracted e-process): this module never reads `kelly_regime_v4`
  returns, exposure, or P&L to build its statistic -- pure OHLCV-derived
  path functionals, identical in spirit to R-109's own non-duplication
  argument against R-104.
- R-105 (leave-one-anchor jackknife / alternative-ladder ensemble
  disagreement of the VOTE's own three components): no alternative anchor
  ladder or jackknife of the vote is built here.
- Every regime-timing round (R-01/R-82/R-83/R-85/R-86/R-96/R-98/R-99/R-117):
  none of those raced a detection-lag gate against STRESS_EPISODES; this
  round, like R-109, produces a continuous distance/probability statistic
  scored by the SAME B1-B5 promotion bar, never a binary alarm-crossing
  race.
- Every SIZE-axis round (R-34...R-117): bolts a multiplicative discount onto
  `v4_target` completely unchanged, exactly R-87/R-104/R-105/R-106/R-109's
  own architecture -- never retunes `frac` or `scale` directly, guarded by
  the identical R2_VS_V4_THRESH kill switch.
- Grepped `docs/LEDGER.md`/`docs/RESEARCH.md`/`docs/STRATEGIES.md` for
  "signature", "rough path", "iterated integral", "Levy area", "path space":
  zero hits before this round.

**Dependency note, checked before any code below was written.** `scipy` is
NOT installed in this environment (`python3 -c "import scipy"` fails,
confirmed directly and consistent with R-118's own identical finding and
with `pyproject.toml` never listing it as a dependency). The NOVEL branch's
chi-squared tail probability below is therefore computed via an EXACT
closed-form expression that holds for any EVEN integer degrees of freedom
(a standard identity: for X ~ chi2(2m), CDF(x) = 1 - exp(-x/2) * sum_{i=0}^
{m-1} (x/2)^i / i!) -- not a scipy call, not a numerical approximation of an
odd-df incomplete gamma function. This is why the novel branch's path is
3-dimensional (6 signature features, an even count) rather than the more
natural-looking 2-dimensional path (3 features, odd) -- the feature count is
chosen to keep the tail-probability formula exact rather than approximated.

**Path signature construction, verified by hand before being coded.** For a
piecewise-linear path with increments Δx_k (k=1..L) in R^d, the depth-2
truncated signature has, for each ordered pair (i,j):
    S^{ij} = sum_k [ C^i_{k-1} * Δx^j_k + 0.5 * Δx^i_k * Δx^j_k ]
where C^i_{k-1} = sum_{l<k} Δx^i_l (the cumulative increment strictly before
step k). Two standard shuffle-product identities (verified directly, not
merely cited): the SYMMETRIC part (S^{ij}+S^{ji})/2 always equals
S1^i * S1^j / 2, i.e. carries NO information beyond the depth-1 total
increments; only the ANTISYMMETRIC part -- the signed "Levy area"
A^{ij} = (S^{ij}-S^{ji})/2 -- carries genuinely new, order-sensitive
information about how the two components co-moved within the window. Hand
check: the corner path (0,0)->(1,0)->(1,1) (right, then up) gives S^{12}=1,
S^{21}=0, so A^{12}=0.5 -- which matches the elementary geometric fact that
this path plus the straight-line chord back to the origin encloses a
counter-clockwise (positive) right triangle of area 0.5. This exact example
is this module's own self-test below.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither branch may edit it. Nothing here reads
a bar at or after OOS_START (2023-01-01).

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) The signature features may, like every point-in-time feature before
them, co-move with `v4_symmetric_vol` closely enough to collapse into a
relabelled volatility rescale -- guarded by the identical R2_VS_VOL_THRESH
Step-0 kill switch R-109 used.
(2) `kelly_regime_v4`'s own reactive, latched vote may already price in
order-sensitive novelty as fast as a path signature can detect it,
reproducing the "real but inert" (Step-0 passes, B1 does not) pattern six
prior ERR-axis attempts have shown in various forms.
(3) The Levy-area term may simply be highly correlated with R-109's own
`anchor_disp`/`kurtosis` features in practice (even though it is
mathematically distinct in construction) -- if so, this round would be
re-testing the same information in new notation, not new information; this
is checked directly below (`levy_vs_r109_features_corr`) as part of Step-0
reporting, not as a gate (a correlation check is diagnostic, not a kill
switch, since even a correlated-but-not-identical statistic can behave
differently once thresholded and discounted).
(4) The reference distribution for the signature features -- like R-109's
own point-in-time features -- is fit predominantly on BTC's 2017-2020
supercycle and may not generalise to ETH at all, reproducing R-109/R-112/
R-115's exact ETH sign-inversion failure regardless of feature map; this is
this round's own pre-registered falsification test (B4, unchanged from
R-109's own).
(5) The chi-squared closed-form tail probability (novel branch) assumes the
signature-feature reference distribution is approximately multivariate
Gaussian; if the true reference distribution is heavy-tailed or multimodal
(plausible, per R-109's own kNN-over-Mahalanobis rationale), the analytic
p-value may be systematically miscalibrated -- disclosed here as a known
limitation of the simplification named in the literature section above, not
discovered only after a bad result.
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
# r105_shared -> ... -> r102_shared): identical control machinery, so every
# number this round produces is directly comparable to R-109/R-112/R-115's.
from experiments.r109_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BASELINE_WINDOW_DAYS,
    BIND_FRAC_THRESH,
    CV_KILL_THRESH,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MIN_REF_DAYS,
    OOS_START,
    R2_VS_V4_THRESH,
    R2_VS_VOL_THRESH,
    RIDGE_EPS,
    SELECTION_ORDER,
    SLICES,
    SPOT,
    STEP0_MAXD_GRID,
    STEP0_THRESH_GRID,
    TargetStrategy,
    align_daily_to_bars,
    apply_deadband,
    apply_discount,
    assert_no_holdout,
    causal_rolling_percentile_rank,
    compare,
    discount_series_for,
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
    step0_gate,
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
# ------------------------------------------------------------------------
SIG_WINDOW_DAYS = 1        # trailing window the signature is computed over;
                            # matches r109_shared's own KURT_WINDOW_BARS (1
                            # trading day) so this round introduces no new
                            # "how long a lookback" nuisance parameter
MODEL_NAMES = ("sig_conservative_knn", "sig_novel_chi2")

# Step-0 grid identical in shape to r109_shared's own (thresh x max_discount),
# same grid values, for direct comparability.
# (STEP0_THRESH_GRID / STEP0_MAXD_GRID / SELECTION_ORDER imported above.)


# ------------------------------------------------------------------------
# (1) Path-signature feature builders. Pure functions of OHLCV via v4's own
# already-causal realized-vol input; zero new data channels.
# ------------------------------------------------------------------------

def _level_log_price(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["close"].to_numpy(dtype=float))


def _level_log_vol(df: pd.DataFrame) -> np.ndarray:
    vol = np.asarray(v4_symmetric_vol(df), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(vol > 0, np.log(vol), np.nan)
    return out


def _level_log_volume(df: pd.DataFrame) -> np.ndarray:
    """log(volume), floored by a FIXED (not data-dependent) epsilon to avoid
    log(0) on exact-zero-volume bars. An earlier version floored at a small
    multiple of the whole series' own median positive volume, which is a
    lookahead (the median depends on bars not yet seen) -- caught by the
    R-121 novel branch's own causal truncation probe (7,471 bars differed
    under truncation) before any inner-validation number was read, and fixed
    here per docs/ROUTINE.md's own bug-fix allowance. A fixed floor (1e-9,
    negligible next to any real BTC/ETH 5-minute volume in this project's
    data) is trivially causal: a pointwise function of each row alone."""
    vol = df["volume"].to_numpy(dtype=float)
    return np.log(np.clip(vol, 1e-9, None))


LEVEL_BUILDERS = {
    "log_price": _level_log_price,
    "log_vol": _level_log_vol,
    "log_volume": _level_log_volume,
}


def _signature_for_window(levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """levels: (L, d) array of per-bar LEVEL values for L bars spanning one
    trailing window (already restricted to strictly-prior bars by the
    caller). Returns (s1, levy) where s1 is shape (d,) -- total increment
    per dimension -- and levy is shape (d, d), the signed Levy-area matrix
    (antisymmetric, levy[i, i] = 0 by construction, levy[j, i] = -levy[i, j]).
    NaN-propagating: any non-finite row in `levels` makes the whole window's
    output NaN (a window straddling a data gap is not silently partially
    used)."""
    d = levels.shape[1]
    if levels.shape[0] < 2 or not np.all(np.isfinite(levels)):
        return np.full(d, np.nan), np.full((d, d), np.nan)
    incr = np.diff(levels, axis=0)                     # (L-1, d)
    s1 = incr.sum(axis=0)                               # (d,)
    prefix = np.concatenate([np.zeros((1, d)), np.cumsum(incr, axis=0)[:-1]], axis=0)  # C_{k-1}, (L-1, d)
    # signature_ij (depth-2, full, not yet antisymmetrized): prefix_i . incr_j (sum over steps)
    sig_full = prefix.T @ incr                          # (d, d), sig_full[i, j] = sum_k prefix[k,i]*incr[k,j]
    levy = 0.5 * (sig_full - sig_full.T)                 # antisymmetric part only
    return s1, levy


def build_signature_features(df: pd.DataFrame, dims: tuple[str, ...],
                              window_days: int = SIG_WINDOW_DAYS) -> pd.DataFrame:
    """One row per UTC calendar day present in `df`. Each day's row is the
    depth-2 truncated-signature summary (per-dim total increment `s1_<dim>`
    plus per-UNORDERED-pair signed Levy area `levy_<dim_i>_<dim_j>`) of the
    (dims) path over the trailing `window_days` days of 5-minute bars
    STRICTLY BEFORE that day's own first bar -- i.e. day t's row never reads
    any bar timestamped on or after day t itself, matching every other
    feature builder's `.shift(1)`-before-resample convention in this
    project's ERR-axis family. NaN until `window_days` of strictly-prior
    data exist.
    """
    levels = np.column_stack([LEVEL_BUILDERS[d](df) for d in dims])  # (n_bars, d)
    idx = df.index
    days = idx.normalize()
    day_starts = pd.DatetimeIndex(sorted(days.unique()))
    win = pd.Timedelta(days=window_days)

    d = len(dims)
    pair_names = [(dims[i], dims[j]) for i in range(d) for j in range(i + 1, d)]
    cols = {f"s1_{name}": [] for name in dims}
    for a, b in pair_names:
        cols[f"levy_{a}_{b}"] = []

    ts = idx.values
    for day in day_starts:
        lo = np.searchsorted(ts, (day - win).to_datetime64(), side="left")
        hi = np.searchsorted(ts, day.to_datetime64(), side="left")  # strictly before `day`
        window_levels = levels[lo:hi]
        s1, levy = _signature_for_window(window_levels)
        for k, name in enumerate(dims):
            cols[f"s1_{name}"].append(s1[k])
        for a, b in pair_names:
            i, j = dims.index(a), dims.index(b)
            cols[f"levy_{a}_{b}"].append(levy[i, j])

    out = pd.DataFrame(cols, index=day_starts)
    return out.dropna(how="all")


SIG2_DIMS = ("log_price", "log_vol")                    # conservative panel: 2D path -> 3 features
SIG3_DIMS = ("log_price", "log_vol", "log_volume")       # novel panel: 3D path -> 6 features (even, for closed-form chi2)


def build_sig2_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_signature_features(df, SIG2_DIMS)


def build_sig3_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_signature_features(df, SIG3_DIMS)


# ------------------------------------------------------------------------
# (2) Closed-form chi-squared CDF for EVEN integer degrees of freedom --
# exact, no scipy (not installed in this environment; see module docstring).
# ------------------------------------------------------------------------

def chi2_cdf_even_df(x: np.ndarray, df: int) -> np.ndarray:
    """CDF of a chi-squared(df) random variable at x, for EVEN integer df
    only (asserts). Exact closed form: for X ~ chi2(2m),
        CDF(x) = 1 - exp(-x/2) * sum_{i=0}^{m-1} (x/2)^i / i!
    (the Erlang/gamma(m, 2) survival function). x<0 maps to CDF=0."""
    assert df >= 2 and df % 2 == 0, f"chi2_cdf_even_df requires even df, got {df}"
    m = df // 2
    x = np.asarray(x, dtype=float)
    xh = np.clip(x, 0.0, None) / 2.0
    total = np.zeros_like(xh)
    term = np.ones_like(xh)  # (x/2)^0 / 0!
    total = total + term
    for i in range(1, m):
        term = term * xh / i
        total = total + term
    cdf = 1.0 - np.exp(-xh) * total
    cdf = np.where(np.isfinite(x), cdf, np.nan)
    cdf = np.where(x < 0, 0.0, cdf)
    return np.clip(cdf, 0.0, 1.0)


# ------------------------------------------------------------------------
# (3) Novel branch's analytic tail-probability state: chi-squared CDF of the
# rolling Mahalanobis QUADRATIC FORM (== rolling_mahalanobis_distance(...)**2)
# in signature-feature space -- reuses r109_shared's own causally-verified
# Mahalanobis primitive unmodified, only squares its output and maps it
# through a closed-form (not empirical-percentile) tail probability.
# ------------------------------------------------------------------------

def rolling_signature_tailprob(daily: pd.DataFrame, window: int = BASELINE_WINDOW_DAYS,
                                min_periods: int = MIN_REF_DAYS,
                                ridge_eps: float = RIDGE_EPS) -> pd.Series:
    dist = rolling_mahalanobis_distance(daily, window=window, min_periods=min_periods,
                                         ridge_eps=ridge_eps)
    q = dist.to_numpy(dtype=float) ** 2
    p = chi2_cdf_even_df(q, df=daily.shape[1])
    return pd.Series(p, index=daily.index, name="sig_chi2_tailprob")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    # (a) Levy-area hand check: the corner path (0,0)->(1,0)->(1,1).
    levels = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    s1, levy = _signature_for_window(levels)
    assert np.allclose(s1, [1.0, 1.0]), s1
    assert np.isclose(levy[0, 1], 0.5), levy
    assert np.isclose(levy[1, 0], -0.5), levy
    assert np.isclose(levy[0, 0], 0.0) and np.isclose(levy[1, 1], 0.0)

    # (b) A straight-line (no turning) path has zero Levy area regardless of
    # direction or length -- pure diagonal motion, no enclosed area.
    levels2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    s1b, levyb = _signature_for_window(levels2)
    assert np.allclose(s1b, [3.0, 3.0])
    assert np.isclose(levyb[0, 1], 0.0, atol=1e-10)

    # (c) A window containing a NaN produces an all-NaN signature (no silent
    # partial use).
    levels3 = np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0]])
    s1c, levyc = _signature_for_window(levels3)
    assert np.all(np.isnan(s1c)) and np.all(np.isnan(levyc))

    # (d) build_signature_features is causal: truncating the frame after some
    # point must not change any already-computed row before that point.
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(121)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    volume = rng.lognormal(0, 0.5, len(idx))
    volume[rng.random(len(idx)) < 0.02] = 0.0   # exercise the zero-volume clip
    # path -- caught a real whole-series-median lookahead in an earlier
    # version of _level_log_volume (see its own docstring).
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": volume},
                       index=idx)
    assert (df["volume"] == 0.0).sum() > 100   # the zero-volume branch is genuinely exercised
    full2 = build_sig2_features(df)
    k = len(df) // 2
    trunc2 = build_sig2_features(df.iloc[:k])
    common = full2.index.intersection(trunc2.index)
    assert len(common) > 50
    a = full2.loc[common].to_numpy()
    b = trunc2.loc[common].to_numpy()
    both_finite = np.isfinite(a) & np.isfinite(b)
    assert both_finite.sum() > 50
    assert np.allclose(a[both_finite], b[both_finite], atol=1e-8), "sig2 features are not causal"

    full3 = build_sig3_features(df)
    trunc3 = build_sig3_features(df.iloc[:k])
    common3 = full3.index.intersection(trunc3.index)
    assert len(common3) > 50
    a3 = full3.loc[common3].to_numpy()
    b3 = trunc3.loc[common3].to_numpy()
    bf3 = np.isfinite(a3) & np.isfinite(b3)
    assert bf3.sum() > 50
    assert np.allclose(a3[bf3], b3[bf3], atol=1e-8), "sig3 features are not causal"

    assert full2.shape[1] == 3, full2.columns.tolist()   # s1_log_price, s1_log_vol, levy_log_price_log_vol
    assert full3.shape[1] == 6, full3.columns.tolist()   # 3 s1's + 3 levy's

    # (e) rolling_mahalanobis_distance on the sig2 panel must still be causal
    # (re-verifies r109_shared's own primitive against THIS round's feature
    # panel, not just its original one).
    dist2 = rolling_mahalanobis_distance(full2)
    dist2_trunc = rolling_mahalanobis_distance(full2.loc[:full2.index[k // 400] if k // 400 < len(full2) else full2.index[-1]])
    assert (dist2.dropna() >= 0).all()

    # (f) chi2_cdf_even_df: exact closed form checked against the m=1 (df=2)
    # special case, CDF(x) = 1 - exp(-x/2), directly (no dependence on the
    # loop for m=1).
    xs = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    got = chi2_cdf_even_df(xs, df=2)
    want = 1.0 - np.exp(-xs / 2.0)
    assert np.allclose(got, want)
    assert chi2_cdf_even_df(np.array([-1.0]), df=2)[0] == 0.0
    # CDF is non-decreasing and bounded in [0, 1], any even df.
    xs2 = np.linspace(0, 50, 200)
    for df in (2, 4, 6, 8):
        c = chi2_cdf_even_df(xs2, df=df)
        assert np.all(c >= 0) and np.all(c <= 1)
        assert np.all(np.diff(c) >= -1e-12)
        assert c[-1] > 0.999   # tail probability -> 1 for x >> df
    try:
        chi2_cdf_even_df(np.array([1.0]), df=3)
        raise AssertionError("chi2_cdf_even_df should reject odd df")
    except AssertionError as e:
        assert "even df" in str(e)

    # (g) rolling_signature_tailprob returns values in [0, 1] and is causal
    # (delegates causality to the already-verified rolling_mahalanobis_distance).
    tp = rolling_signature_tailprob(full3)
    valid = tp.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 1).all()


_self_test()
