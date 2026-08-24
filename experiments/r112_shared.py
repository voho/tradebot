"""Shared, read-only utilities and pre-registration for the R-112 round (08-24).

DIRECTION, in one sentence: close R-109's own named, disclosed follow-on --
give the kNN distributional-novelty brake (R-109 novel branch: the FIRST of
six ERR-axis attempts to clear B1 on both markets) a reference distribution
that is not dominated by one instrument's single price cycle, by changing
EXACTLY ONE thing about R-109's own winning construction per branch, and
re-running the identical B1-B5 promotion bar including the identical
pre-registered B4 (ETH) falsification test R-109 itself failed.

R-109's own closing paragraph (`docs/LEDGER.md`, R-109 Verdict section) named
this explicitly: "a future session with a genuine reason to retry this
specific axis has one clearly named, disclosed follow-on -- a reference
distribution NOT dominated by one price cycle (e.g. detrended/return-space
features only, or an explicit multi-asset reference pool) might close the
novel branch's B4 gap without needing a new mechanism." This round takes both
halves of that sentence literally, as the two branches:

- CONSERVATIVE: keep R-109 novel branch's exact construction (5-feature
  panel, single-asset trailing-730-day reference, kNN distance, k=10,
  refit_every=30, identical Step-0/B1-B5 gates) and change ONLY the one
  feature that is not already scale-free in return space --
  `anchor_disp`, R-109's mean pairwise %% distance among SMA(20/40/80) PRICE
  levels -- for a return-space analogue built from rolling MEAN LOG RETURNS
  over the same three horizons instead of SMA price levels. Every other
  feature (log_vol, kurtosis, volume_z, skew) is already a property of
  returns/volume, not of the price level or its trajectory through one
  cycle, and is carried over unchanged.
- NOVEL: keep R-109 novel branch's exact 5-feature panel UNCHANGED and
  change ONLY the reference SET -- instead of a single instrument's own
  trailing 730 days, pool that instrument's own trailing window WITH the
  contemporaneous trailing windows of R-63's own six-instrument panel (BCH,
  LTC, ETC, DASH, LINK, XTZ -- `experiments/r63_shared.UNIVERSE_6`), each
  standardized against ITS OWN local mean/std before pooling (Sun & Saenko
  2016's CORAL: align each domain's own first and second moments before
  treating them as one distribution -- the standard domain-alignment
  prescription for combining several distinct-but-related distributions
  without one dominating the pooled one by its absolute scale). Neither
  BTC nor ETH is ever a pool member: BTC is the primary target instrument
  and ETH is reserved, exactly as in R-109, for the B4 falsification test
  alone -- pooling either into its own or the other's reference would leak
  exactly the generalization test this round exists to run honestly.

**Literature grounding, fetched and read via WebSearch this round:**

- Rabanser, S., Gunnemann, S., & Lipton, Z. (2019), "Failing Loudly: An
  Empirical Study of Methods for Detecting Dataset Shift", *NeurIPS 2019*.
  Carried over from R-109 unchanged -- the general dataset-shift framing
  this whole ERR-axis sub-line rests on.
- Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019),
  "Invariant Risk Minimization", *arXiv:1907.02893*. The general
  domain-generalization argument this round's NOVEL branch operationalizes
  directly: a statistic calibrated against a single environment (here, one
  instrument's own price history) risks learning that environment's
  idiosyncratic, non-transferable structure rather than the invariant
  property that actually generalizes; training/calibrating against MULTIPLE
  environments jointly is the standard prescription for forcing a statistic
  to discard what is specific to any one of them.
- Sun, B., & Saenko, K. (2016), "Deep CORAL: Correlation Alignment for Deep
  Domain Adaptation", *ECCV Workshops*. The specific mechanical recipe the
  NOVEL branch's pooling uses: align each domain (here, each instrument's
  own trailing reference window) to zero mean / unit variance BEFORE
  combining domains into one pooled reference, so no single domain's
  absolute scale (e.g., an instrument that is structurally noisier or
  calmer than BTC) dominates the pooled distance metric.
- The 2023-2025 novelty-detection-under-domain-shift literature surveyed
  this round (e.g. arXiv:2309.12301 "Environment-biased Feature Ranking for
  Novelty Detection Robustness"; arXiv:2504.21247 "Subject Information
  Extraction for Novelty Detection with Domain Shifts") converges on the
  same diagnosis R-109 reached independently from its own data: a novelty
  detector calibrated on one environment's dominant characteristics learns
  to conflate that environment's "style" (here: BTC's specific 2017-2020
  price trajectory and the absolute SMA levels it produced) with genuine,
  transferable "core" novelty, and degrades under even mild shift to a
  related-but-distinct environment (here: ETH) -- precisely R-109's own
  named B4 failure mode, restated in that literature's own vocabulary. Both
  of this round's branches are direct, literature-grounded attacks on
  "style" contamination: the CONSERVATIVE branch removes a style-carrying
  feature (price-level-based SMA dispersion); the NOVEL branch dilutes
  style-specific calibration by domain-aligned pooling across environments.

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path) -- the same constraint as R-109, R-106, R-105, R-104, R-87,
R-28. This is explicitly NOT a seventh independent ERR-axis mechanism: it is
a targeted repair of R-109's own sixth mechanism's one named failure mode,
which is why the round is scoped as two single-line changes rather than a
fresh construction.

**Not a duplicate of:**
- R-109 (both branches): R-109's conservative branch used Mahalanobis
  distance (a different algorithm class) and its novel branch used a
  single-asset reference; this round touches neither the choice of distance
  METRIC (both R-112 branches use kNN, R-109's own winning metric, held
  fixed) nor introduces any new feature TYPE -- CONSERVATIVE replaces one
  feature's definition (price-level SMA dispersion -> return-space mean-
  return dispersion) while NOVEL replaces the reference SET's provenance
  (one instrument -> six-instrument CORAL-pooled), never both in the same
  branch, and never the distance statistic itself.
- R-105/R-106 (specification/cross-model disagreement): neither branch
  computes disagreement between models or alternative anchor-ladder
  specifications of `kelly_regime_v4`'s own vote; both remain, like R-109,
  a single continuous distance-to-reference statistic.
- R-63/R-65/R-67/R-68/R-72/R-107/R-110/R-111 (the multi-asset CROSS-
  SECTIONAL SCORE/allocator family, `src/tradebot/multi_engine.py`): the
  NOVEL branch's use of `r63_shared.UNIVERSE_6`/`load_universe` is
  restricted to READING those six instruments' own OHLCV to build a
  reference SET for a single-instrument (BTC or ETH) discount brake bolted
  onto `kelly_regime_v4`'s existing `frac * scale` product -- it builds no
  cross-sectional score, calls no ranking/allocation function, imports no
  portfolio-construction code, and trades no six-instrument panel; the six
  instruments are read-only training data for a novelty reference, exactly
  as R-63's own panel is a read-only ranking input there, but the output
  here is a discount on ONE instrument's OWN exposure, never a portfolio.
- Every SIZE-axis round (R-34...R-103): both branches bolt a multiplicative
  discount onto `v4_target` completely unchanged, guarded by the identical
  Step-0 `R2_VS_V4_THRESH`/`R2_VS_VOL_THRESH` kill switches R-87 onward all
  use.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither branch may edit it. Nothing here reads
a bar at or after `OOS_START` (2023-01-01); `load_pool_daily_panels` below
explicitly truncates every one of the six pool instruments below `OOS_START`
before any feature is built from them.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) The B4 generalization gap may not be a feature- or reference-set
artifact at all, but a deeper property of `kelly_regime_v4`'s own vote
interacting with ETH's shorter history (R-57's own general finding, present
across many single-asset constructions in this ledger) -- if so, NEITHER
branch closes B4, and that is itself the informative result: it would
demonstrate the gap is not the specific thing R-109 named as its own most
likely cause.
(2) The CONSERVATIVE branch's return-space feature may simply carry less
information than the price-level one it replaces (trend-anchor agreement in
absolute price terms may be doing real work the return-space analogue
cannot reproduce), reproducing the "real but inert" Step-0-passes/B1-fails
pattern on BTC itself, before B4 is even reached.
(3) The NOVEL branch's cross-asset pool may dilute the statistic's
discriminative power rather than its style-specificity -- six differently-
behaved altcoins pooled together may simply raise the effective reference
variance enough that BTC's own genuinely novel days no longer register as
distant (a Step-0 `bind_frac`/`state_cv` failure), the generic
"broadening a reference until it explains everything" failure mode.
(4) Even if both branches individually improve on R-109's B4 result, the
underlying "true novel days" may differ enough across instruments that a
pooled or detrended statistic answers a materially different question from
R-109's own single-asset one, in which case the resulting B1 gain (if any)
may not be attributable to the same mechanism R-109 already validated on
BTC's own history -- flagged here as an interpretation risk, not a
kill switch: the promotion bar (B1/B3/B4/B5) governs regardless.
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
# ... -> r102_shared): identical control machinery, so every number this
# round produces is directly comparable to R-105...R-109's.
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
    KURT_WINDOW_BARS,
    MIN_REF_DAYS,
    MODEL_NAMES,
    NOVEL_FEATURE_BUILDERS,
    OOS_START,
    PRIMARY_MAXD,
    PRIMARY_THRESH,
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
    build_daily_features,
    causal_rolling_percentile_rank,
    compare,
    discount_series_for,
    fee_at,
    feature_anchor_dispersion,
    feature_kurtosis,
    feature_log_vol,
    feature_skew,
    feature_volume_z,
    hr,
    load_btc,
    load_eth,
    novelty_discount,
    paired_diff,
    print_rows,
    print_step0_report,
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
)

from experiments.r63_shared import (  # noqa: E402,F401
    UNIVERSE_6,
    load_universe,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS

# ------------------------------------------------------------------------
# (1) CONSERVATIVE branch's own feature: return-space analogue of
# `feature_anchor_dispersion` (mean pairwise dispersion of PRICE-level
# SMA(20/40/80) anchors, normalized by close) -- here built from mean
# LOG-RETURN over the same three horizons instead of SMA price levels, so
# the feature never touches a price level or anything derived from one
# instrument's own price trajectory through a cycle, only from returns.
# ------------------------------------------------------------------------

def feature_anchor_dispersion_returns(df: pd.DataFrame, horizons=V4_HORIZONS) -> np.ndarray:
    """Mean absolute pairwise difference among rolling MEAN LOG RETURNS over
    the same 20/40/80-day horizons `feature_anchor_dispersion` uses for its
    SMA price anchors -- the return-space analogue: "how far apart do
    short/med/long trend estimates currently sit," expressed entirely in
    returns rather than in price levels, shifted by 1 bar. Uses the SAME
    anchor horizons the shipped vote uses; introduces no new nuisance
    parameter, exactly like the feature it replaces."""
    r = np.log(df["close"]).diff()
    means = [r.rolling(h * BARS_PER_DAY, min_periods=h * BARS_PER_DAY).mean()
             for h in horizons]
    n = len(means)
    pair_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_diffs.append((means[i] - means[j]).abs())
    disp = pd.concat(pair_diffs, axis=1).mean(axis=1)
    return disp.shift(1).to_numpy()


RETURNSPACE_FEATURE_BUILDERS = dict(NOVEL_FEATURE_BUILDERS)
RETURNSPACE_FEATURE_BUILDERS["anchor_disp"] = feature_anchor_dispersion_returns


# ------------------------------------------------------------------------
# (2) NOVEL branch's own reference construction: CORAL-style (Sun & Saenko
# 2016) domain-aligned pooling of a target instrument's own trailing
# reference window WITH the contemporaneous trailing windows of the six
# UNIVERSE_6 instruments, each standardized against its OWN local mean/std
# before being pooled -- causal by construction (every window used is
# strictly prior to the day being scored, for every instrument).
# ------------------------------------------------------------------------

def load_pool_daily_panels(builders: dict | None = None,
                            tickers=UNIVERSE_6) -> dict[str, pd.DataFrame]:
    """Daily NOVEL_FEATURE_BUILDERS-column feature panels for the six
    UNIVERSE_6 instruments (BCH, LTC, ETC, DASH, LINK, XTZ), each truncated
    strictly below OOS_START before any feature is built -- neither BTC nor
    ETH is ever a member of `tickers` (`UNIVERSE_6` excludes both by
    construction, asserted below)."""
    assert "BTC" not in tickers and "ETH" not in tickers, (
        "pool must never include the target/falsification instruments")
    builders = builders or NOVEL_FEATURE_BUILDERS
    frames = load_universe(tickers)
    panels = {}
    cutoff = pd.Timestamp(OOS_START, tz="UTC")
    for name, df in frames.items():
        df = df.loc[df.index < cutoff]
        assert_no_holdout(df, f"load_pool_daily_panels(): {name}")
        panels[name] = build_daily_features(df, builders)
    return panels


def _zscore(window: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mu = window.mean(axis=0)
    sd = window.std(axis=0)
    sd = sd.where(sd > 0, 1.0)
    return mu, sd


def rolling_knn_distance_pooled(target_daily: pd.DataFrame,
                                 pool_dailies: dict[str, pd.DataFrame],
                                 window: int = BASELINE_WINDOW_DAYS,
                                 min_periods: int = MIN_REF_DAYS, k: int = 10,
                                 refit_every: int = 30) -> pd.Series:
    """Day t's mean Euclidean distance to its `k` nearest neighbours in a
    POOLED reference set: the target instrument's own strictly-prior
    `window`-day window, UNION each `pool_dailies` instrument's
    contemporaneous strictly-prior `window`-day window -- each piece
    standardized against its OWN local mean/std (CORAL, Sun & Saenko 2016)
    before concatenation, so no one instrument's absolute scale dominates
    the pooled distance metric. The query point itself is standardized
    against the TARGET's own local mean/std (the frame it will be compared
    within). Refit only every `refit_every` days (walk-forward, still
    strictly causal -- a refit at day t uses only days < t from every
    source, target and pool alike), identical cadence convention to
    `rolling_knn_distance`. `pool_dailies` is a fixed closure over
    instruments the target itself is never truncated against, so this
    function is causal in the target argument alone -- verified by the
    causal-truncation probe in this module's self-test and reused by both
    branches' own truncation probes on `df` (the target frame) only."""
    cols = list(target_daily.columns)
    idx = target_daily.index
    out = pd.Series(np.nan, index=idx, dtype=float, name="knn_pooled")
    last_refit = None
    mu_t = sd_t = None
    ref_z = None
    win = pd.Timedelta(days=window)
    for t in idx:
        if last_refit is None or (t - last_refit).days >= refit_every:
            lo = t - win
            own_win = target_daily.loc[(target_daily.index < t) & (target_daily.index >= lo), cols].dropna()
            if len(own_win) >= min_periods:
                mu_t, sd_t = _zscore(own_win)
                pieces = [(own_win - mu_t) / sd_t]
                for pdaily in pool_dailies.values():
                    pwin = pdaily.loc[(pdaily.index < t) & (pdaily.index >= lo), cols].dropna()
                    if len(pwin) >= min_periods:
                        mu_p, sd_p = _zscore(pwin)
                        pieces.append((pwin - mu_p) / sd_p)
                ref_z = pd.concat(pieces, axis=0).to_numpy(dtype=float)
                last_refit = t
        if ref_z is None:
            continue
        x = target_daily.loc[t, cols]
        if x.isna().any():
            continue
        xz = ((x - mu_t) / sd_t).to_numpy(dtype=float)
        dists = np.sqrt(((ref_z - xz) ** 2).sum(axis=1))
        kk = min(k, len(dists))
        nearest = np.partition(dists, kk - 1)[:kk]
        out.loc[t] = float(nearest.mean())
    return out


def hr2(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=250_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(112)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                        "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
                       index=idx)

    # (1) return-space anchor-dispersion feature: causal.
    assert causal_truncation_probe_series(feature_anchor_dispersion_returns, df)
    daily_rs = build_daily_features(df, RETURNSPACE_FEATURE_BUILDERS)
    assert set(daily_rs.columns) == set(NOVEL_FEATURE_BUILDERS.keys())
    assert len(daily_rs) > 100
    # Genuinely different from the price-level feature it replaces.
    daily_px = build_daily_features(df, NOVEL_FEATURE_BUILDERS)
    common = daily_rs.index.intersection(daily_px.index)
    a = daily_rs.loc[common, "anchor_disp"].dropna()
    b = daily_px.loc[common, "anchor_disp"].dropna()
    both = a.index.intersection(b.index)
    assert len(both) > 50
    assert not np.allclose(a.loc[both].to_numpy(), b.loc[both].to_numpy(), equal_nan=True)

    # (2) pooled kNN reference: causal in the TARGET argument, with a fixed
    # synthetic pool closure (independent random walks -- structurally
    # unrelated to the target series, standing in for six alt instruments).
    def make_pool(n_assets=2, seed=0):
        pool = {}
        r = np.random.default_rng(seed)
        for i in range(n_assets):
            innov_i = r.normal(0, 0.0006, len(idx))
            drift_i = np.cumsum(np.full(len(idx), 0.00001 * (i + 1)))
            close_i = 8_000 * np.exp(np.cumsum(innov_i) + drift_i)
            df_i = pd.DataFrame({"open": close_i, "high": close_i * 1.0005,
                                  "low": close_i * 0.9995, "close": close_i,
                                  "volume": r.lognormal(0, 0.5, len(idx))}, index=idx)
            pool[f"ALT{i}"] = build_daily_features(df_i, NOVEL_FEATURE_BUILDERS)
        return pool

    pool_dailies = make_pool()

    def probe_fn(frame: pd.DataFrame) -> np.ndarray:
        daily = build_daily_features(frame, NOVEL_FEATURE_BUILDERS)
        dist = rolling_knn_distance_pooled(daily, pool_dailies, refit_every=15)
        aligned = dist.reindex(dist.index.union(frame.index)).sort_index().ffill().reindex(frame.index)
        return aligned.to_numpy()

    assert causal_truncation_probe_series(probe_fn, df)

    daily_t = build_daily_features(df, NOVEL_FEATURE_BUILDERS)
    dist_pooled = rolling_knn_distance_pooled(daily_t, pool_dailies, refit_every=15)
    assert (dist_pooled.dropna() >= 0).all()
    # Pooled distance must differ from the single-asset distance on the
    # same target (the pool genuinely changes the statistic, not a no-op).
    dist_single = rolling_knn_distance(daily_t, refit_every=15)
    c2 = dist_pooled.dropna().index.intersection(dist_single.dropna().index)
    assert len(c2) > 50
    assert not np.allclose(dist_pooled.loc[c2].to_numpy(), dist_single.loc[c2].to_numpy())

    # Explicit truncation check on the pooled statistic itself (belt and
    # braces beyond the composed-probe check above): values up to k must be
    # identical whether computed on the full target frame or one truncated
    # at k+1, with the SAME fixed pool closure both times.
    k = len(daily_t) // 2
    daily_t_trunc = daily_t.iloc[:k + 1]
    dist_trunc = rolling_knn_distance_pooled(daily_t_trunc, pool_dailies, refit_every=15)
    common3 = dist_pooled.index[:k]
    a3, b3 = dist_pooled.reindex(common3).to_numpy(), dist_trunc.reindex(common3).to_numpy()
    ok3 = np.isfinite(a3) & np.isfinite(b3)
    assert ok3.sum() > 10
    assert np.allclose(a3[ok3], b3[ok3], atol=1e-8), "pooled kNN distance is not causal"

    # UNIVERSE_6 never includes BTC or ETH.
    assert "BTC" not in UNIVERSE_6 and "ETH" not in UNIVERSE_6


_self_test()
