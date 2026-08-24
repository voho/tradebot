"""Shared, read-only utilities and pre-registration for the R-122 round (08-24).

DIRECTION, in one sentence: does replacing R-109's distributional-novelty
reference distribution -- BTC's OWN trailing 730-day rolling window of daily
market-state features -- with a POOLED, EXTERNALLY-CALIBRATED SYNTHETIC
reference distribution built from zero real price data, escape the ETH
falsification gap (B4) that has now survived four repair attempts on that
reference-pool construction unchanged (R-109, R-112, R-115) and one attempt
at the feature map instead (R-121)?

**Direct precedent, and the reason this round exists.** R-121 (08-24, same
day) tried a materially different NOVELTY STATISTIC (order-sensitive
path-signature features) feeding R-109's own unchanged reference-pool
architecture, and reproduced the identical BTC/ETH spot sign-inversion
failure shape R-109/R-112/R-115 all hit. R-121's own closing line, quoted
directly from `docs/LEDGER.md`: *"a future session preferring this specific
construction needs a reference distribution NOT dominated by BTC's own
single supercycle at all (e.g. a purely-synthetic or externally-calibrated
reference, in the spirit of R-118/R-119's N approx 3 line, applied here
instead), which is a materially different, untested question from either
axis tried so far."* This round is exactly that question -- the fifth
attempt on this construction, and the first to vary the REFERENCE
DISTRIBUTION'S DATA SOURCE (synthetic vs. real) rather than its algorithm,
feature map, or pooling scope.

**Which constraint this attacks: ERR** (no error control anywhere in the
signal path) -- the same distributional-novelty sub-axis as R-109/R-112/
R-115/R-121 (the fourth notion of uncertainty this ledger's ERR-axis rounds
have tried, after sampling significance R-28/R-87/R-104 and specification
disagreement R-105/R-106). Like every round in this family, the statistic
only ever discounts `kelly_regime_v4`'s existing, completely unchanged
`frac * scale`; it never enters the vote, and is guarded by the identical
R2_VS_V4_THRESH / R2_VS_VOL_THRESH Step-0 kill switches R-109 built,
imported unchanged below.

**Literature grounding, fetched and verified via WebSearch this round
(2026-08-24):**

- Rabanser, S., Gunnemann, S., & Lipton, Z. (2019), "Failing Loudly: An
  Empirical Study of Methods for Detecting Dataset Shift", *NeurIPS 2019*.
  Reused unchanged from R-109 -- the general dataset-shift framing this
  whole sub-axis rests on.
- Abbas, M., Azmat, M., Horesh, R., & Yurochkin, M. (2025), "Out-of-
  Distribution Detection using Synthetic Data Generation", arXiv:2502.03323
  (title, all four authors, abstract and venue independently confirmed via
  WebFetch on the arXiv abstract page this round). Establishes, in a
  DIFFERENT domain (LLM-generated synthetic proxies for text
  classification -- toxicity/sentiment/reward-model/misalignment detection,
  nine dataset pairs), that constructing reference/proxy material
  SYNTHETICALLY, without sourcing it from the deployment distribution's own
  real history, is an active, published design pattern for OOD/novelty
  detection. Disclosed precisely: their method synthesizes the
  OUT-of-distribution side (via an LLM) to train a discriminator against
  real in-distribution data; this round's construction is the mirror image
  -- it synthesizes the REFERENCE (in-distribution / "normal") side (via a
  calibrated stochastic-process simulator) and scores REAL data against it
  -- and the domain (financial time series) is entirely different from
  theirs (text). Cited here only for the general principle that a
  synthetically-constructed reference is a legitimate substitute for an
  empirically-sourced one in this literature, not as a domain match or a
  method this round reproduces.
- Wiese, M., Knobloch, R., Korn, R., & Kretschmer, P. (2020), "Quant GANs:
  Deep Generation of Financial Time Series", *Quantitative Finance* 20(9),
  1419-1440 (title, all four authors, journal, volume and page range
  independently confirmed via WebSearch this round). Establishes that
  purely-synthetic, generated financial price paths are already a standard
  quant-finance tool for stress-testing and risk-model validation when real
  historical data is scarce or non-representative -- directly on-domain
  (financial time series), unlike the Abbas et al. citation above.
  Disclosed precisely: Wiese et al.'s generator is a GAN trained to match
  the stylized facts of REAL historical returns; this round's generator is
  the much simpler, non-learned, externally-calibrated stochastic-process
  simulator R-119 already built and froze (GBM+jump / regime-switching Monte
  Carlo, parameters read off published literature, never fit to any price
  series at all, real or synthetic) -- cited for the general precedent that
  synthetic price-path generation for out-of-sample stress/reference
  purposes is an established, not a novel, idea in this literature, not as
  a reproduction of the GAN method itself.
- Scaillet, Treccani & Trevisan (2020); MDPI *Mathematics* 9(20) 2567
  (2021); the four-source BTC crash catalogue (CCN, NYDIG, Live Volatile,
  CNBC 2026-02-12) -- all three reused UNCHANGED from `r119_shared.py`,
  frozen there this same day and re-verified present and unedited in that
  file before this round imports them. No new external calibration number
  is introduced by this round; every synthetic price path below is built
  from parameters this project already froze and used once (R-119).

**Not a duplicate of:**

- R-109 / R-112 / R-115 / R-121 (four attempts, distributional-novelty
  family): every one of those rounds' reference distribution is built EXCLUSIVELY
  from BTC's own real trailing history (a rolling 730-day window of real
  daily features, walk-forward refit) -- R-112 varied the feature space
  (return-space), R-115 varied the pooling (BTC+ETH Coinbase-native), R-121
  varied the feature MAP (path-signatures). None of the four ever built a
  reference distribution containing zero real price bars. This round's
  reference panel is built ENTIRELY from synthetic OHLCV paths generated by
  `r119_conservative_gbm_jump.path_generator` / `r119_novel_
  regimeswitch_external.path_generator` -- verified below (`assert
  "load_btc" not in ...` style checks are impractical for a generator
  function, so verification is instead: the panel-building functions in
  this file take a `path_generator` callable and a `seed` range as their
  ONLY external state, never a `df` of real prices, and the real BTC/ETH
  frames loaded by `main()` in each branch are used ONLY to (a) compute the
  real day's own feature vector to be SCORED against the fixed reference,
  never to build the reference itself, and (b) run the unchanged
  promotion-bar `compare()`/B1-B5 machinery, exactly as every prior round in
  this family does).
- R-118 / R-119 (same day, N approx 3 axis): both use synthetic Monte Carlo
  paths from the SAME two generator families (plain GBM+jump vs. externally-
  calibrated regime-switching) -- but for a structurally different PURPOSE:
  selecting `kelly_regime_v4`'s own free parameters (ladder base,
  target_vol, max_leverage) by a robust CVaR criterion scored across many
  synthetic draws. This round never touches `r118_shared.GRID`,
  `select_config`, `robust_score`, `evaluate_candidate`, or any
  parameter-selection machinery at all -- `kelly_regime_v4`'s own shipped
  parameters are used completely unmodified throughout (imported from
  `experiments.r102_shared` via the standard `v4_target`/`v4_vote_frac`/
  `v4_scale` chain, exactly as R-109/R-112/R-115/R-121 do). The only things
  imported from R-119's modules are the two `path_generator(seed) -> df`
  functions themselves, reused as pure synthetic-OHLCV SOURCES for an
  entirely different downstream use (an ERR-axis novelty reference, not an
  N approx 3 selection criterion).
- R-113 (kNN/Mahalanobis applied to the multi-asset panel): single-instrument
  only here, not touched.

**Is it simulable here?** Yes. The synthetic reference panels are built from
`path_generator` calls the project already froze and validated in R-119
(deterministic, seeded, self-tested there); the real-data side reuses
`r109_shared`'s unchanged feature builders, discount architecture, Step-0
gate and B1-B5 promotion bar. No new data file, no new external API call
inside either branch's own execution.

**What would make each branch fail, named now, before any real-data number
was read:**

1. The pooled synthetic reference panel's own FEATURE-SPACE distribution
   (of `log_vol`, `anchor_disp`, `kurtosis` -- computed on synthetic OHLCV by
   the SAME builders used on real data) may simply not overlap with real
   BTC/ETH's own feature-space region at all -- e.g. if the externally-
   calibrated generators' realized volatility, trend-anchor dispersion or
   kurtosis land systematically outside the real data's own range, EVERY
   real day would register as maximally "novel" all the time (a degenerate,
   saturated statistic), which the reused R2_VS_V4_THRESH / state_cv Step-0
   kill switches are designed to catch, but were not designed FOR
   specifically (they were built for R-109's real-reference construction) --
   disclosed as a genuine, not merely nominal, risk this round's Step-0
   table must be read with in mind.
2. Even if Step-0 passes (the statistic has genuine, non-degenerate
   dispersion and is not a rescale of `v4_target`/realized vol), the
   underlying problem R-109/R-112/R-115/R-121 all diagnosed --
   `kelly_regime_v4`'s own reactive, latched vote already pricing in
   "unusual" conditions by the time they register as distributionally novel
   by ANY statistic -- has nothing to do with where the reference
   distribution comes from, and would reproduce the identical "Step-0
   passes, B1 does not" inert pattern for a fifth time.
3. The externally-calibrated generators (R-119's own conservative GBM+jump,
   round-number 80%/yr vol with literature jump overlay; and novel 3-state
   regime-switching, literature-sourced bear severity/duration) were
   designed and validated for producing REALISTIC PRICE PATHS for a Kelly-
   sizer parameter-selection sweep, not for reproducing the fine-grained
   SHAPE of BTC's own realized trend-anchor-dispersion or kurtosis
   distributions -- there is no guarantee a generator good enough for one
   purpose is calibrated finely enough for the other. This is exactly the
   generator-purpose mismatch risk failure mode 1 names concretely, and
   Step-0's diagnostics are read with it in mind rather than papered over.
4. If Step-0 and B1 both pass, the reference's real geographic/temporal
   origin (BTC only, never ETH) is moot -- since the reference is fully
   synthetic and touches neither BTC nor ETH price data, this construction
   has no reason a priori to be MORE BTC-specific than a real-data reference
   would be, but it equally has no reason to be LESS so: the generators'
   own calibration numbers (jump sizes, bear severity) were themselves
   sourced from BTC-specific crash catalogues and BTC-specific jump studies
   (R-119's own citations), so a synthetic reference built from them is not
   asset-neutral either -- it is BTC-flavoured by a different route (its
   calibration literature) rather than BTC-flavoured by its raw data. B4
   (ETH falsification) is exactly the test that distinguishes these two
   readings, which is why it remains this round's one pre-registered
   falsification test, unchanged.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither branch may edit it. Nothing here reads
a bar at or after `OOS_START` (2023-01-01); the synthetic reference-panel
builder below never reads ANY real price file, and every function that
walks a real data frame is guarded by the same `assert_no_holdout` machinery
R-109/.../R-121 already use, imported unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported unchanged from r109_shared (itself chaining r106_shared ->
# ... -> r102_shared): identical control machinery, feature builders,
# discount architecture and Step-0 gate, so every number this round produces
# is directly comparable to R-109/R-112/R-115/R-121's.
from experiments.r109_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
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
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    build_daily_features,
    causal_rolling_percentile_rank,
    causal_truncation_probe_series,
    compare,
    discount_series_for,
    fee_at,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    novelty_discount,
    paired_diff,
    print_plateau_table,
    print_rows,
    print_step0_report,
    r_squared,
    run_slice,
    step0_gate,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)

# The two frozen synthetic path generators from R-119 -- imported as pure
# `seed -> OHLCV DataFrame` sources, reused unchanged. Neither module's
# import touches OOS_START+ data (r119_novel's one internally-sourced
# scalar, ORDINARY_VOL_ANNUAL, reads only `load_inner_train_btc()`, already
# holdout-safe and disclosed in R-119's own docstring).
from experiments.r119_conservative_gbm_jump import (  # noqa: E402
    path_generator as gbm_jump_path_generator,
)
from experiments.r119_novel_regimeswitch_external import (  # noqa: E402
    path_generator as regimeswitch_path_generator,
)

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# ------------------------------------------------------------------------

# Number of synthetic draws pooled into the reference panel. Chosen once, as
# a round number, before any panel was built or inspected: each draw
# contributes roughly 1461 - (warmup for the slowest anchor, 80 days) ~=
# 1380 valid daily rows (the panel builder below drops NaN warmup rows), so
# 20 draws give a pooled reference of roughly 27,000 rows -- about 37x
# R-109's own 730-day real-reference window, and (unlike that rolling
# window) fixed and time-invariant rather than walk-forward, since nothing
# about a purely synthetic reference depends on calendar time at all.
N_SYNTH_DRAWS = 20
MIN_REF_ROWS = 500  # defensive floor; panel construction asserts above this

# Both branches use r109_shared's original 3-feature panel (log_vol,
# anchor_disp, kurtosis) rather than the 5-feature NOVEL_FEATURE_BUILDERS
# R-109's own novel branch used. Disclosed reason, not an oversight: two of
# the five features in that richer panel (`volume_z`, `skew` is fine, but
# `volume_z` specifically) depend on realistic bar-level VOLUME, and both
# R-119 synthetic generators emit a flat, constant `volume=1.0` column
# (disclosed in their own module docstrings -- volume plays no role in
# `kelly_regime_v4`'s own signal, so R-119 never needed to simulate it).
# Computing `feature_volume_z` on a constant-volume synthetic series is
# degenerate by construction (zero rolling std -> undefined z-score), so
# reusing the 5-feature panel here would require inventing a NEW synthetic
# volume-generation model -- a free modelling choice with no literature
# anchor from this round's own citations, and a confound this round is not
# designed to test. Both branches below are therefore restricted to the
# 3-feature, volume-free panel; this is a disclosed SCOPE LIMIT of this
# round, not a finding.
REFERENCE_FEATURE_BUILDERS = FEATURE_BUILDERS

MODEL_NAMES = ("synthetic_gbm_mahalanobis", "synthetic_regimeswitch_knn")


# ------------------------------------------------------------------------
# (1) Pooled synthetic reference panel -- built ONCE from `n_draws` seeded
# synthetic OHLCV paths, ZERO real price data. Fixed and time-invariant:
# no rolling window, no walk-forward refit, because nothing about a
# synthetic reference depends on real calendar time.
# ------------------------------------------------------------------------

def build_synthetic_reference_panel(path_generator, n_draws: int = N_SYNTH_DRAWS,
                                     seed_base: int = 0,
                                     builders: dict[str, callable] | None = None) -> pd.DataFrame:
    """Draw `n_draws` synthetic OHLCV paths from `path_generator(seed)` for
    `seed` in `[seed_base, seed_base + n_draws)`, compute the SAME daily
    feature panel (`build_daily_features`, reused unchanged from
    `r109_shared`) on each, and concatenate into one pooled reference
    DataFrame. Reads no real data whatsoever -- `path_generator` is one of
    the two frozen R-119 functions above, which touch only `np.random` and
    (for the regime-switching generator) the one already-disclosed,
    non-holdout `ORDINARY_VOL_ANNUAL` scalar computed once at R-119's own
    import time."""
    builders = builders or REFERENCE_FEATURE_BUILDERS
    panels = []
    for i in range(n_draws):
        seed = seed_base + i
        synth_df = path_generator(seed)
        daily = build_daily_features(synth_df, builders)
        panels.append(daily.dropna(how="any"))
    pooled = pd.concat(panels, axis=0, ignore_index=True)
    assert len(pooled) >= MIN_REF_ROWS, (
        f"pooled synthetic reference panel has only {len(pooled)} rows "
        f"(< {MIN_REF_ROWS}) -- warmup ate almost everything, a real bug")
    assert np.all(np.isfinite(pooled.to_numpy())), "non-finite value in pooled reference panel"
    return pooled


# ------------------------------------------------------------------------
# (2) Reference-distance statistics against the FIXED synthetic panel.
# Both are causal by construction: a real day t's score depends only on
# day t's own (already 1-bar-shifted) feature vector and the fixed
# synthetic panel -- never on any other real day, past or future.
# ------------------------------------------------------------------------

def fit_mahalanobis_reference(ref_panel: pd.DataFrame, ridge_eps: float = RIDGE_EPS):
    """Fit ONE mean/inverse-covariance from the whole pooled synthetic
    panel. Returns (mu, inv_cov)."""
    arr = ref_panel.to_numpy(dtype=float)
    mu = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False)
    k = cov.shape[0]
    cov = cov + ridge_eps * np.eye(k) * (np.trace(cov) / k if np.trace(cov) > 0 else 1.0)
    inv = np.linalg.inv(cov)
    return mu, inv


def synthetic_mahalanobis_distance(daily_real: pd.DataFrame, ref_panel: pd.DataFrame,
                                    ridge_eps: float = RIDGE_EPS) -> pd.Series:
    """Day t's Mahalanobis distance of `daily_real.loc[t]` from the FIXED
    mean/covariance of the pooled synthetic reference panel -- computed once
    for the whole panel, applied identically to every real day (no rolling
    window, no refit: the reference never changes)."""
    mu, inv = fit_mahalanobis_reference(ref_panel, ridge_eps)
    arr = daily_real.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    finite_rows = np.all(np.isfinite(arr), axis=1)
    d = arr[finite_rows] - mu
    dist = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, inv, d), 0.0, None))
    out[finite_rows] = dist
    return pd.Series(out, index=daily_real.index, name="synthetic_mahalanobis")


def fit_knn_reference(ref_panel: pd.DataFrame):
    """Per-feature standardize the pooled synthetic panel against its OWN
    mean/std. Returns (mu, sd, standardized_reference_array)."""
    arr = ref_panel.to_numpy(dtype=float)
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd, (arr - mu) / sd


def synthetic_knn_distance(daily_real: pd.DataFrame, ref_panel: pd.DataFrame,
                            k: int = 10) -> pd.Series:
    """Day t's mean Euclidean distance (in the synthetic panel's own
    standardized space) to its `k` nearest neighbours in the FIXED pooled
    synthetic reference SET -- one standardization, computed once, applied
    to every real day."""
    mu, sd, ref_z = fit_knn_reference(ref_panel)
    arr = daily_real.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    kk = min(k, len(ref_z))
    for t in range(n):
        x = arr[t]
        if not np.all(np.isfinite(x)):
            continue
        xz = (x - mu) / sd
        dists = np.sqrt(((ref_z - xz) ** 2).sum(axis=1))
        nearest = np.partition(dists, kk - 1)[:kk]
        out[t] = float(nearest.mean())
    return pd.Series(out, index=daily_real.index, name="synthetic_knn")


def hr2(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    # --- cheap toy generator for self-test speed (NOT R-119's real ones,
    # which are exercised directly by each branch's own main()) ---
    def _toy_generator(seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = 60_000  # ~208 days at 288 bars/day -- enough for warmup + margin
        idx = pd.date_range("2017-01-01", periods=n, freq="5min", tz="UTC")
        innov = rng.normal(0, 0.0006, n)
        close = 10_000 * np.exp(np.cumsum(innov))
        return pd.DataFrame(
            {"open": close, "high": close * 1.0005, "low": close * 0.9995,
             "close": close, "volume": rng.lognormal(0, 0.5, n)},
            index=idx,
        )

    panel = build_synthetic_reference_panel(_toy_generator, n_draws=3,
                                             builders=REFERENCE_FEATURE_BUILDERS)
    assert panel.shape[1] == 3
    assert len(panel) >= MIN_REF_ROWS

    # A real daily panel to score, built the SAME way, on a DIFFERENT
    # synthetic run (stands in for "real data" in this self-test).
    real_like = build_daily_features(_toy_generator(999), REFERENCE_FEATURE_BUILDERS)
    real_like = real_like.dropna(how="any").iloc[:50]

    # --- Mahalanobis: non-negative, finite, deterministic ---
    dist_m = synthetic_mahalanobis_distance(real_like, panel)
    assert np.all(np.isfinite(dist_m.to_numpy()))
    assert (dist_m.to_numpy() >= 0).all()
    dist_m2 = synthetic_mahalanobis_distance(real_like, panel)
    assert np.allclose(dist_m.to_numpy(), dist_m2.to_numpy()), "Mahalanobis not deterministic"

    # --- kNN: non-negative, finite, deterministic ---
    dist_k = synthetic_knn_distance(real_like, panel, k=5)
    assert np.all(np.isfinite(dist_k.to_numpy()))
    assert (dist_k.to_numpy() >= 0).all()
    dist_k2 = synthetic_knn_distance(real_like, panel, k=5)
    assert np.allclose(dist_k.to_numpy(), dist_k2.to_numpy()), "kNN not deterministic"

    # --- KEY INVARIANT this round's construction introduces: a real day's
    # score must not depend on any OTHER real day at all (stronger than
    # R-109's own rolling-window causality -- here the reference is fully
    # fixed, so truncating or reordering the REAL panel must not move any
    # remaining day's score by even a bit). Verify by scoring a truncated
    # and a row-permuted version of `real_like` and checking the common
    # rows agree exactly. ---
    trunc = real_like.iloc[:20]
    dist_m_trunc = synthetic_mahalanobis_distance(trunc, panel)
    assert np.allclose(dist_m.iloc[:20].to_numpy(), dist_m_trunc.to_numpy()), \
        "Mahalanobis score of a real day changed when other real days were removed"
    dist_k_trunc = synthetic_knn_distance(trunc, panel, k=5)
    assert np.allclose(dist_k.iloc[:20].to_numpy(), dist_k_trunc.to_numpy()), \
        "kNN score of a real day changed when other real days were removed"

    perm = real_like.sample(frac=1.0, random_state=7)
    dist_m_perm = synthetic_mahalanobis_distance(perm, panel).reindex(real_like.index)
    assert np.allclose(dist_m.to_numpy(), dist_m_perm.to_numpy()), \
        "Mahalanobis score depends on real-panel row order"

    # --- degenerate reference panel (constant columns) must not explode ---
    const_panel = pd.DataFrame({"a": np.full(400, 1.0), "b": np.full(400, 2.0),
                                 "c": np.full(400, 3.0)})
    dist_const = synthetic_mahalanobis_distance(real_like, const_panel)
    assert np.all(np.isfinite(dist_const.to_numpy()))

    # --- both R-119 path generators importable and callable at a small
    # scale is checked directly by each branch's own main(); here we only
    # verify the two names resolve to callables so an import-time typo
    # fails loudly in THIS shared module rather than silently in a branch ---
    assert callable(gbm_jump_path_generator)
    assert callable(regimeswitch_path_generator)


_self_test()
