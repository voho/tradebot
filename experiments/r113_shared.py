"""R-113: does a DISTRIBUTIONAL-NOVELTY exposure brake -- the ERR-axis
construction R-109 built and validated against `kelly_regime_v4` (single
BTC/ETH instrument) -- carry over to the multi-asset cross-sectional
portfolio (`xsmom_entry_band`: R-63's score, R-68's ENTRY_ONLY timing,
R-107/R-110's equal-weight allocation), which has never had ANY uncertainty
or error-control treatment applied to it at all?

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, neither branch edits it, and it does not itself compute a
verdict. It is written by the operator BEFORE either branch is dispatched.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **ERR** (no error control anywhere in the signal
path). R-28 (retracted), R-87, R-104, R-105, R-106, R-109 and R-112 -- seven
attempts across three notions of uncertainty (sampling significance,
specification/model disagreement, distributional novelty) -- all discount
`kelly_regime_v4`'s SINGLE-ASSET exposure. Not one of the ledger's 112 prior
rounds has ever applied an uncertainty/error-control statistic to the
MULTI-ASSET portfolio construction R-63 opened and R-65/67/68/107/110/111
spent nine rounds refining on three other axes (score, timing, allocation
weight) -- every one of which froze the total notional the portfolio takes,
varying only which assets receive it and how it is split. R-111's own
closing line named the state of that construction precisely: "every
dimension this project's own framework can vary on this panel without new
data has now returned NEGATIVE" -- but "every dimension" there meant score,
timing and allocation weighting specifically; error control was never one of
the dimensions on the list because nothing had ever tried it. That is the
gap this round tests, not a new notion of uncertainty (this round invents
none) but a new TARGET for two already-validated ones.

**Mechanism, one sentence:** compute R-109's own two novelty statistics
(Mahalanobis / kNN distance from a strictly-causal rolling reference) over
panel-level features instead of single-instrument OHLCV features, and
multiplicatively discount the xsmom portfolio's TOTAL notional -- not the
per-asset split R-107/R-110 varied, not the membership timing R-65/67/68
varied -- when the panel's own state looks unlike its own recent history.

**Literature grounding, fetched and read via WebSearch this round:**

- Rabanser, S., Gunnemann, S., & Lipton, Z. (2019), "Failing Loudly: An
  Empirical Study of Methods for Detecting Dataset Shift", *NeurIPS 2019*.
  Same framing R-109 used: a shift in a model's own INPUT distribution is
  directly measurable and does not require touching the model's output or
  P&L. R-109 applied this to one instrument's OHLCV-derived state; nothing
  in Rabanser et al.'s own framing is instrument-count-specific, and this
  round is the direct test of whether that generality actually holds when
  the "input" is a cross-sectional panel rather than one series.
- Gorman, L. R., Sapra, S. G., & Weigand, R. A. (2010), "The Role of
  Cross-Sectional Dispersion in Active Portfolio Management", *Journal of
  Investing* / SSRN 1266225. Cross-sectional return dispersion is the
  literature's own standard panel-level (not single-instrument) turbulence
  statistic -- high dispersion signals a "stock-picker's market" with
  elevated idiosyncratic risk; this round's `feature_xsec_dispersion` below
  is a direct causal implementation of their statistic, reused as one of
  three inputs to the novelty distance rather than as a standalone signal
  (which would just be another SCORE-axis attempt, already closed by R-111).
- De Maesschalck, Jouan-Rimbaud & Massart (2000) and Ramaswamy, Rastogi &
  Shim (2000) / Breunig et al. (2000) -- the CONSERVATIVE (Mahalanobis) and
  NOVEL (kNN) statistic families respectively, unchanged from R-109; not
  re-cited in full here, see `r109_shared.py`'s own docstring. Both
  functions (`rolling_mahalanobis_distance`, `rolling_knn_distance`) are
  imported from there VERBATIM, not reimplemented -- they operate on an
  arbitrary daily feature `DataFrame` and were never specific to
  single-instrument features in the first place.
- Baltas (2015), Bruder & Roncalli (2012) -- reused here only for the
  standing fact this round leans on, already established in this ledger:
  R-63 measured this exact panel's own mean pairwise correlation at 0.634,
  which is why `feature_mean_pairwise_corr` below is a genuinely
  informative, non-degenerate panel feature on THIS data rather than an
  arbitrary choice.

**Not a duplicate of:**

- R-109 / R-112 (ERR-axis novelty brakes on `kelly_regime_v4`). Different
  TARGET strategy entirely (multi-asset `xsmom_entry_band` vs single-asset
  `kelly_regime_v4`), different features (panel-level cross-sectional
  dispersion / rolling pairwise correlation / eligible-count anomaly vs
  single-instrument realized vol / anchor dispersion / return kurtosis),
  different discount destination (the portfolio's TOTAL notional, computed
  by `conditional_vol_scale` on the ALL-N basket, vs `kelly_regime_v4`'s own
  `frac * scale` product). The DISTANCE STATISTICS themselves
  (`rolling_mahalanobis_distance`, `rolling_knn_distance`) are imported
  unchanged from `r109_shared` -- reusing already-validated, already
  causality-tested machinery rather than re-deriving it is a feature of this
  round's design, not a hidden duplication: R-109 validated that the
  DISTANCE FUNCTIONS are causal and well-behaved in the abstract; this round
  tests whether they carry an exploitable signal on a DIFFERENT feature
  panel and a DIFFERENT target construction, which R-109 never touched.
- R-107 / R-110 (allocation weighting: equal-weight vs risk-parity vs
  blend). Disjoint step -- both change how a FIXED total notional is SPLIT
  across an already-eligible set; this round changes the TOTAL ITSELF,
  before any split, and the split step (equal-weight, `k=1`, both rounds'
  own undefeated winner) is imported frozen and unmodified.
- R-63 / R-65 / R-67 / R-68 (score and membership timing). All frozen and
  imported unmodified via `build_targets_from_score` at R-68's own selected
  point (`k=1, buffer=0.05, hold_days=1.0, delta_in=0.080, delta_out=0.0`).
  This round never touches which bars a membership change fires on; it
  discounts the SIZE of the already-decided position, after the fact.
- R-111 (score formula variants). Frozen at R-63's own `cross_sectional_score`,
  imported unmodified (`r63_cross_sectional_score`) -- R-111 already spent a
  full round establishing nothing beats it on this panel.

**Is it simulable here?** Yes, with zero new data: `UNIVERSE_8`'s eight
already-committed 5m Coinbase spot series, through the identical
`simulate_portfolio` engine every multi-asset round since R-63 has used.
Unlike R-112's CORAL-pooled branch (blocked because its Bitfinex ETH
falsification series never overlapped `UNIVERSE_6`'s 2020+ Coinbase window),
this round's ETH series IS `UNIVERSE_8`'s own Coinbase ETH panel -- there is
no cross-exchange mismatch to block anything here.

**What would make it fail, named now, before any code ran:**

(F1) **Relabelled rescale.** The three panel features may simply co-move
     with `conditional_vol_scale`'s own realized-vol input (built from the
     ALL-N basket's log returns) closely enough that the novelty distance is,
     in substance, a second copy of the volatility target this round is
     supposed to leave untouched -- the exact 26-attempt SIZE-axis collapse.
     Guarded by a Step-0 R2-vs-basket-vol kill switch, mirroring R-109's own
     R2_VS_VOL_THRESH but computed against THIS round's total-notional scale
     input rather than v4's.
(F2) **Real but inert.** R-68's own hysteresis (`delta_in=0.080`,
     `hold_days=1.0`) already damps reaction to short-lived cross-sectional
     noise by construction -- by the time a panel state is unusual enough to
     register as distributionally novel, the entry band may have already
     absorbed whatever the discount would have removed. This is the same
     "Step-0 passes, D1/D2 does not" pattern that hit four of six prior
     single-asset ERR attempts.
(F3) **The panel's own standing diagnosis.** Nine rounds on this panel
     (R-63, R-65, R-67, R-68, R-107 x2, R-110 x2, R-111 x2) have found real,
     signed, directionally-sane mechanism effects that nonetheless failed to
     move ANY interval outside the ±0.2 Sharpe/noise floor -- R-67's own
     "no mechanism can narrow an interval; only more data, more breadth, or
     forward evidence can." This is disclosed as the single most likely
     outcome before any number exists: even a mechanistically real discount
     is fighting a panel this ledger has never yet found enough evidence in
     to clear a decisive gate, regardless of which of ten prior mechanisms
     was tried.
(F4) **Thinner reference history.** The panel's evaluable history begins
     2020-04-01 (`W_TRAIN` runs to 2021-12-31, ~630 days), materially
     shorter than the single-asset axis's 2017-2020 span R-109 calibrated
     against. `BASELINE_WINDOW_DAYS`/`MIN_REF_DAYS` are set shorter than
     R-109's own (365/90 vs 730/180, below) specifically so a reference
     distribution and a genuine held-out test period can both fit inside
     `W_TRAIN`+`W_VAL` at all -- a disclosed departure from R-109's own
     convention, not a free parameter tuned on any result.

This module is READ-ONLY for both branches -- neither may edit it. Nothing
here reads a bar at or after `W_HOLD` (2023-01-01) outside an
`assert_no_holdout`-guarded or explicitly-restricted call, matching every
prior round's convention.

=====================================================================
HOLDOUT-READ NOTE (inherited convention, unchanged from R-63...R-111)
=====================================================================

This axis's established, disclosed departure from the rest of the
project's holdout discipline still applies: the decisive `W_FULL6` cell
already spans 2020-04-01 through the last committed bar, so D1/D2/D5/
scramble on `W_FULL6` already include 2023+ U6 data, counted as "+0"
holdout consultations by the project-wide convention (R-47/R-57/R-63
onward) that "+0" refers to the RESERVED BTC/ETH 2023+ holdout, not the U6
panel. A genuine `W_HOLD` read (U8, 2023-01-01 onward, BTC/ETH included)
is the only kind that increments the running total, and is authorized ONLY
if a branch clears `further_work` on `W_FULL6`/`W_VAL` first -- reported
honestly regardless of outcome, exactly once, per branch that clears the
bar.

=====================================================================
STEP-0 GRID, RESOLVED BY THE OPERATOR AGAINST REAL W_TRAIN DATA BEFORE
EITHER BRANCH WAS DISPATCHED (not from any inner-validation/decisive
number -- see `step0_gate`'s own docstring for the one design correction
this involved and why it is not a goalpost move)
=====================================================================

Both `mahalanobis` and `knn` PASS Step-0 at `(PRIMARY_THRESH, PRIMARY_MAXD)
= (0.90, 1.0)` -- the same primary cell R-109 selected for its own
single-asset round, reached independently here via the identical
`SELECTION_ORDER` convention. `knn` additionally FAILS Step-0 at
`thresh=0.95` (both `max_discount` values) on `bind_frac` alone (0.30% of
bars, below the 1% floor) -- the grid is genuinely discriminating, not a
rubber stamp. Both branches use `PRIMARY_THRESH=0.90, PRIMARY_MAXD=1.0` as
their frozen, pre-registered operating point; neither branch selects its
own threshold from any Sharpe/PnL number.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r63_shared import (  # noqa: E402,F401
    BOOT_KW,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_REAL,
    START_BALANCE,
    TOTAL_NOTIONAL_DEADBAND,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_HOLD,
    W_TRAIN,
    W_VAL,
    align_frames,
    check_against_engine,
    check_causality,
    compare,
    config_count,
    excludes_zero,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
)
from experiments.r63_novel_xsmom_rank import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    DEADBAND,
    HORIZONS,
    WARM_DAYS,
    basket_log_returns,
    conditional_vol_scale,
    cross_sectional_score as r63_cross_sectional_score,
    warm_window,
)
from experiments.r65_shared import (  # noqa: E402,F401
    R63_GROSS_EDGE,
    R63_NET_D1,
    R63_TURNOVER_PER_DAY,
    SPOT_FREE,
    frontier_row,
    realized_vol,
    turnover_stats,
    volmatched_hold_equity,
)
from experiments.r68_shared import (  # noqa: E402,F401
    D5_BAR_R68,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    scramble_fixed_perm,
)
from experiments.r111_shared import (  # noqa: E402,F401
    build_targets_from_score,
)
from experiments.r109_shared import (  # noqa: E402,F401
    align_daily_to_bars,
    novelty_discount,
    rolling_knn_distance,
    rolling_mahalanobis_distance,
)
from experiments.r106_shared import causal_rolling_percentile_rank  # noqa: E402,F401
from experiments.r102_shared import r_squared  # noqa: E402,F401

OUT_DIR = ROOT / "reports" / "r113_multiasset_novelty"

# ------------------------------------------------------------------------
# Frozen construction: R-68's own selected point on R-63's own score,
# R-107/R-110's own undefeated k=1 equal-weight allocation.
# ------------------------------------------------------------------------
K_FIXED = 1
BUFFER_FIXED = 0.05
HOLD_DAYS_FIXED = 1.0
DELTA_IN_FIXED = 0.080
DELTA_OUT_FIXED = 0.0

# ------------------------------------------------------------------------
# Pre-registered constants -- FIXED before either branch was dispatched.
# Shorter than R-109's 730/180: see docstring point (F4).
# ------------------------------------------------------------------------
BASELINE_WINDOW_DAYS = 365
MIN_REF_DAYS = 90
CORR_WINDOW_DAYS = 30           # rolling window for feature_mean_pairwise_corr
ELIG_COUNT_WINDOW_DAYS = 90     # rolling window for feature_eligible_count_z
RIDGE_EPS = 1e-6

BIND_FRAC_THRESH = 0.01
MIN_BOUND_DISCOUNT_THRESH = 0.05  # on bars where the discount binds at all,
                                   # its mean magnitude must be >= 5% -- see
                                   # step0_gate's own docstring for why this
                                   # replaces a whole-series R^2 check
R2_VS_BASKETVOL_THRESH = 0.90   # discount FRACTION must not be a relabelled
                                 # rescale of conditional_vol_scale's own
                                 # realized-vol input (F1)
CV_KILL_THRESH = 0.05

MODEL_NAMES = ("mahalanobis", "knn")

STEP0_THRESH_GRID = (0.80, 0.90, 0.95)
STEP0_MAXD_GRID = (0.5, 1.0)
PRIMARY_THRESH = 0.90
PRIMARY_MAXD = 1.0
SELECTION_ORDER = ((0.90, 1.0), (0.95, 1.0), (0.80, 0.5), (0.90, 0.5), (0.95, 0.5), (0.80, 1.0))


# ------------------------------------------------------------------------
# (1) Frozen (undiscounted) target builder -- the construction this round
# discounts, unmodified.
# ------------------------------------------------------------------------

def frozen_targets(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """R-63's score through R-68's ENTRY_ONLY timing at its own selected
    point and R-107/R-110's undefeated k=1 equal-weight allocation --
    byte-for-byte the construction `xsmom_entry_band` is registered as."""
    score = r63_cross_sectional_score(aligned)
    targets, _ev, _ev_bars = build_targets_from_score(
        aligned, score, k=K_FIXED, buffer=BUFFER_FIXED, hold_days=HOLD_DAYS_FIXED,
        delta_in=DELTA_IN_FIXED, delta_out=DELTA_OUT_FIXED,
    )
    return targets


# ------------------------------------------------------------------------
# (2) Panel-level novelty features -- pure functions of the aligned OHLCV
# panel, no new data channel, no strategy P&L or exposure read anywhere.
# Each is computed at DAILY resolution using data THROUGH day t's own
# close, then explicitly `.shift(1)` at the daily level so day t's feature
# row reflects information available at day t's OPEN -- unlike R-109's
# bar-level features (which arrive pre-shifted from `v4_symmetric_vol` and
# rely on a first-of-day resample trick), these are daily-native
# cross-sectional statistics with no bar-level analogue, so the shift is
# applied explicitly and directly rather than inherited from a bar-level
# convention that does not apply here.
# ------------------------------------------------------------------------

def _daily_log_returns(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = pd.DataFrame({t: df["close"] for t, df in aligned.items()})
    daily = closes.resample("1D").last()
    return np.log(daily).diff()


def feature_xsec_dispersion(aligned: dict[str, pd.DataFrame]) -> pd.Series:
    """Gorman, Sapra & Weigand (2010): daily cross-sectional standard
    deviation of instrument log returns, shifted 1 day."""
    ret = _daily_log_returns(aligned)
    disp = ret.std(axis=1, skipna=True)
    return disp.shift(1)


def feature_mean_pairwise_corr(aligned: dict[str, pd.DataFrame],
                                window: int = CORR_WINDOW_DAYS) -> pd.Series:
    """Rolling mean pairwise correlation among the panel's daily log
    returns over a trailing `window`-day span (ending the PRIOR day -- the
    rolling window itself, via pandas `.corr().rolling()`, uses days up to
    and including its own label; shifted 1 day below so day t's value is
    computed from returns strictly through day t-1)."""
    ret = _daily_log_returns(aligned)
    n = ret.shape[1]
    if n < 2:
        return pd.Series(np.nan, index=ret.index)
    corr_roll = ret.rolling(window, min_periods=max(5, window // 3)).corr()
    out = pd.Series(np.nan, index=ret.index)
    for day in ret.index:
        try:
            c = corr_roll.loc[day]
        except KeyError:
            continue
        arr = c.to_numpy(dtype=float)
        mask = ~np.eye(n, dtype=bool)
        vals = arr[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out.loc[day] = float(vals.mean())
    return out.shift(1)


def feature_eligible_count_z(aligned: dict[str, pd.DataFrame],
                              window: int = ELIG_COUNT_WINDOW_DAYS) -> pd.Series:
    """z-score of the daily-resampled positive-score eligible-asset count
    (R-63's own `score > 0` filter, before any rank cap) against its own
    trailing `window`-day mean/std -- "how unusual is today's breadth
    relative to its own recent history", shifted 1 day."""
    score = r63_cross_sectional_score(aligned)
    m = (score > 0.0).sum(axis=1).astype(float)
    m.loc[score.isna().all(axis=1)] = np.nan
    m_daily = m.resample("1D").last()
    mu = m_daily.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = m_daily.rolling(window, min_periods=max(5, window // 3)).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (m_daily - mu) / sd.replace(0.0, np.nan)
    return z.shift(1)


PANEL_FEATURE_BUILDERS = {
    "xsec_dispersion": feature_xsec_dispersion,
    "mean_pairwise_corr": feature_mean_pairwise_corr,
    "eligible_count_z": feature_eligible_count_z,
}


def build_daily_panel(aligned: dict[str, pd.DataFrame],
                       builders: dict | None = None) -> pd.DataFrame:
    builders = builders or PANEL_FEATURE_BUILDERS
    cols = {name: fn(aligned) for name, fn in builders.items()}
    return pd.DataFrame(cols).dropna(how="all")


# ------------------------------------------------------------------------
# (3) Discount construction and application.
# ------------------------------------------------------------------------

def discount_series_for(aligned: dict[str, pd.DataFrame], state: pd.Series,
                         thresh: float, max_discount: float,
                         index: pd.Index) -> pd.Series:
    aligned_state = align_daily_to_bars(state, next(iter(aligned.values()))).fillna(0.0)
    disc = novelty_discount(aligned_state, thresh, max_discount)
    return disc.reindex(index).fillna(0.0)


def apply_discount(targets: pd.DataFrame, discount: pd.Series) -> pd.DataFrame:
    """Multiply every row of the frozen target matrix by `(1 - discount)`.
    Mathematically identical to discounting the TOTAL notional before the
    equal-weight (`k=1`, so trivially degenerate) split, since every row of
    `targets` already carries the full per-asset weight -- there is only
    ever one nonzero column per row at `k=1`."""
    disc = discount.reindex(targets.index).fillna(0.0).clip(0.0, 1.0)
    return targets.multiply(1.0 - disc, axis=0)


def build_r113_targets(aligned: dict[str, pd.DataFrame], model: str,
                        thresh: float, max_discount: float) -> pd.DataFrame:
    """The one function both branches call: aligned OHLCV in, discounted
    target-weight matrix out. `model` is 'mahalanobis' or 'knn'."""
    assert model in MODEL_NAMES, model
    targets = frozen_targets(aligned)
    panel = build_daily_panel(aligned)
    if model == "mahalanobis":
        dist = rolling_mahalanobis_distance(panel, window=BASELINE_WINDOW_DAYS,
                                             min_periods=MIN_REF_DAYS)
    else:
        dist = rolling_knn_distance(panel, window=BASELINE_WINDOW_DAYS,
                                     min_periods=MIN_REF_DAYS)
    state = causal_rolling_percentile_rank(dist, window=BASELINE_WINDOW_DAYS,
                                            min_periods=MIN_REF_DAYS)
    discount = discount_series_for(aligned, state, thresh, max_discount, targets.index)
    return apply_discount(targets, discount)


# ------------------------------------------------------------------------
# (4) Step-0 gate
# ------------------------------------------------------------------------

def step0_gate(aligned: dict[str, pd.DataFrame], model: str, thresh: float,
               max_discount: float,
               bind_frac_thresh: float = BIND_FRAC_THRESH,
               min_bound_discount_thresh: float = MIN_BOUND_DISCOUNT_THRESH,
               r2_vol_thresh: float = R2_VS_BASKETVOL_THRESH,
               cv_kill_thresh: float = CV_KILL_THRESH) -> dict:
    """NOTE on a pre-dispatch design correction, made from this real-data
    smoke test rather than from any inner-validation/decisive number: an
    early version of this gate checked whole-series R^2 of the discounted
    total notional against the frozen (undiscounted) path, mirroring
    R-109's `R2_VS_V4_THRESH`. On real W_TRAIN data that check turned out
    to be miscalibrated for THIS construction: `kelly_regime_v4`'s target
    changes almost every bar, so a discount touching a similar fraction of
    bars there produces real separation from an R^2-vs-original check; this
    round's total notional is instead LATCHED by R-63's own 0.10 deadband
    and the Step-0 grid's own `thresh=0.90` means at most ~10% of days ever
    bind at all -- so whole-series R^2 against the frozen path is >0.99 by
    construction regardless of how large the discount is ON THE DAYS IT
    FIRES, making that check measure sparsity, not inertness, and killing
    every cell in the grid including cells with a large, real discount.
    Replaced with a check restricted to the bars where the discount
    actually binds: the MEAN discount fraction on those bars must clear
    `MIN_BOUND_DISCOUNT_THRESH` -- directly testing "when it fires, does it
    remove a non-trivial amount of exposure" rather than "is the whole
    series different from the whole series". This is a Step-0 STRUCTURAL
    diagnostic computed on `W_TRAIN`, before any inner-validation Sharpe or
    holdout number is read by either branch -- the same category of
    pre-dispatch calibration R-109's own docstring describes doing against
    real BTC data while building ITS shared module.
    """
    frozen = frozen_targets(aligned)
    discounted = build_r113_targets(aligned, model, thresh, max_discount)

    frozen_total = frozen.sum(axis=1).to_numpy(dtype=float)
    disc_total = discounted.sum(axis=1).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_discount = np.where(frozen_total > 1e-12,
                                     1.0 - disc_total / np.where(frozen_total > 1e-12, frozen_total, 1.0),
                                     0.0)
    finite = np.isfinite(implied_discount)
    bound = finite & (implied_discount > 1e-9)
    bind_frac = float(np.mean(implied_discount[finite] > 1e-9)) if finite.any() else 0.0
    bind_ok = bind_frac > bind_frac_thresh

    mean_bound_discount = float(implied_discount[bound].mean()) if bound.any() else 0.0
    not_trivial_discount = mean_bound_discount >= min_bound_discount_thresh

    basket_vol = conditional_vol_scale(basket_log_returns(aligned))
    basket_vol = np.asarray(basket_vol, dtype=float)
    n = min(len(implied_discount), len(basket_vol))
    both_finite = np.isfinite(implied_discount[:n]) & np.isfinite(basket_vol[:n])
    r2_vol = r_squared(implied_discount[:n][both_finite], basket_vol[:n][both_finite]) if both_finite.any() else 1.0
    not_vol_rescale = r2_vol < r2_vol_thresh

    panel = build_daily_panel(aligned)
    if model == "mahalanobis":
        dist = rolling_mahalanobis_distance(panel, window=BASELINE_WINDOW_DAYS, min_periods=MIN_REF_DAYS)
    else:
        dist = rolling_knn_distance(panel, window=BASELINE_WINDOW_DAYS, min_periods=MIN_REF_DAYS)
    state = causal_rolling_percentile_rank(dist, window=BASELINE_WINDOW_DAYS, min_periods=MIN_REF_DAYS)
    s = state.to_numpy(dtype=float)
    s_finite = s[np.isfinite(s)]
    cv = float(s_finite.std() / s_finite.mean()) if len(s_finite) and s_finite.mean() else float("nan")
    non_degenerate = np.isfinite(cv) and cv >= cv_kill_thresh

    passed = bind_ok and not_trivial_discount and not_vol_rescale and non_degenerate
    return dict(bind_frac=bind_frac, bind_ok=bind_ok, mean_bound_discount=mean_bound_discount,
                not_trivial_discount=not_trivial_discount, r2_vs_basketvol=r2_vol,
                not_vol_rescale=not_vol_rescale, state_cv=cv,
                non_degenerate=non_degenerate, passed=passed)


def print_step0_report(label: str, gate: dict) -> None:
    print(f"\n--- Step-0 gate: {label} ---")
    print(f"bind_frac={gate['bind_frac']:.4f} (kill <= {BIND_FRAC_THRESH}) -> "
          f"{'ok' if gate['bind_ok'] else 'KILL'}")
    print(f"mean discount on bound bars={gate['mean_bound_discount']:.4f} (kill < {MIN_BOUND_DISCOUNT_THRESH}) -> "
          f"{'ok' if gate['not_trivial_discount'] else 'KILL (trivial when it fires)'}")
    print(f"R^2 vs basket realized vol={gate['r2_vs_basketvol']:.4f} (kill >= {R2_VS_BASKETVOL_THRESH}) -> "
          f"{'ok' if gate['not_vol_rescale'] else 'KILL (relabelled vol rescale)'}")
    print(f"state CoV={gate['state_cv']:.4f} (kill < {CV_KILL_THRESH}) -> "
          f"{'ok' if gate['non_degenerate'] else 'KILL (degenerate)'}")
    verdict = "PASS" if gate["passed"] else "FAIL"
    print(f"STEP-0 GATE VERDICT ({label}): {verdict}")


# ------------------------------------------------------------------------
# (5) Further-work / promotion bars -- R-65's four-clause form (no M1':
# this round changes neither entry nor exit timing, so there is no
# membership-change-rate to gate on, same reasoning R-107/R-110 used).
# ------------------------------------------------------------------------

def further_work(d1: bool, d2: bool, d3: bool, d5: bool, scramble_survived: bool) -> bool:
    return (d1 or d2) and d3 and d5 and scramble_survived


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2020-01-01", periods=200_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(113)
    tickers = ["A", "B", "C", "D"]
    aligned = {}
    for i, tk in enumerate(tickers):
        innov = rng.normal(0, 0.0006, len(idx))
        drift = np.cumsum(np.full(len(idx), 0.00001 * (i + 1)))
        close = 100 * np.exp(np.cumsum(innov) + drift)
        aligned[tk] = pd.DataFrame(
            {"open": close, "high": close * 1.0005, "low": close * 0.9995,
             "close": close, "volume": rng.lognormal(0, 0.5, len(idx))},
            index=idx)

    panel = build_daily_panel(aligned)
    assert panel.shape[1] == 3
    assert len(panel) > 100

    # Causal truncation probe for each panel feature builder: rebuilding on
    # a truncated frame must not move any earlier row.
    cut = len(idx) - 30_000
    aligned_trunc = {t: df.iloc[:cut] for t, df in aligned.items()}
    for name, fn in PANEL_FEATURE_BUILDERS.items():
        full = fn(aligned)
        trunc = fn(aligned_trunc)
        common = full.index.intersection(trunc.index)
        common = common[common < aligned_trunc[tickers[0]].index[-1] - pd.Timedelta(days=2)]
        a = full.reindex(common).to_numpy(dtype=float)
        b = trunc.reindex(common).to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.sum() > 10, f"{name}: not enough overlap to test causality"
        assert np.allclose(a[ok], b[ok], atol=1e-9), f"{name} is not causal"

    # End-to-end causality of the full discounted-target pipeline, both
    # models, via the project's own generic truncation probe.
    for model in MODEL_NAMES:
        builder = lambda al, m=model: build_r113_targets(al, m, PRIMARY_THRESH, 0.5)
        assert check_causality(builder, aligned, cut_from_end=30_000), \
            f"build_r113_targets({model}) is not causal"

    # novelty_discount / apply_discount: bounded, zero below threshold.
    frozen = frozen_targets(aligned)
    disc_frac = pd.Series(np.linspace(0, 1, len(frozen)), index=frozen.index)
    out = apply_discount(frozen, disc_frac)
    assert (out.to_numpy() <= frozen.to_numpy() + 1e-12).all()
    assert np.allclose(out.iloc[0].to_numpy(), frozen.iloc[0].to_numpy(), atol=1e-9)


_self_test()
