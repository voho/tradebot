"""Shared, read-only utilities and pre-registration for the R-146 round (08-25).

DIRECTION, in one sentence: `kelly_regime_v4`'s three regime-vote anchors are
plain rolling means (SMA) of `close`; this round replaces the anchor
STATISTIC itself with a robust/order-statistic construction, holding the
1%-band / latching-hysteresis / 3-way-averaging vote architecture and the
untouched conditional-volatility `scale` completely fixed, and asks whether
a less outlier-sensitive anchor changes v4's own risk/return profile.

**Which constraint this attacks: SIZE.** R-62 (four independent
confirmations: the original factor-isolation test, 21 point-estimate scale
retunes R-34->R-60, R-87's conformal-dispersion construction, and R-136's
vol-estimator generalization) established that `frac` (the 3-anchor vote)
carries `kelly_regime_v4`'s entire matched-exposure signature while `scale`
(the conditional volatility target) carries none of it in isolation. Every
SIZE-axis round that has touched the vote to date (R-06/R-07 span sweeps,
R-40 ladder bagging, R-45 walk-forward re-estimation, R-89/R-92 closed-form
span derivation, R-59/R-60 per-asset recalibration and cadence rescaling)
varies WHICH spans feed the anchor, HOW MANY anchors vote, or the vote's
RESPONSE CURVE (kelly_regime_v2's convex gamma) -- never the anchor's own
central-tendency STATISTIC. This round is the first to ask whether the
plain arithmetic mean (SMA) is even the right building block for the
anchor, independent of its span.

**Not a duplicate of:**
- R-06, R-07 (anchor SPAN sweeps), R-40 (ladder BAGGING across spans),
  R-45 (walk-forward span RE-ESTIMATION), R-89/R-92 (Sepp & Lucic 2026
  closed-form derivation of the optimal SPAN from a fitted generative
  model): all five vary which window LENGTH(S) feed a plain rolling mean.
  None replaces the mean itself with a different statistic; this round
  holds the shipped 20/40/80-day ladder fixed and only changes what
  "average" means inside each window.
- R-89's own conservative branch (band width / hysteresis asymmetry sweep,
  `docs/LEDGER.md` R-89 section): a different axis of the SAME vote
  architecture (the +/-1% latch band, not the anchor's own statistic) --
  closed, 1% already sits at/beside that sweep's own optimum. Untouched
  here; both branches keep `V4_BAND=0.01` exactly.
- R-59, R-60 (per-asset `target_vol` recalibration; OU half-life vote
  RESCALING and CUSUM-cadence vote TIMING, aimed at the multi-asset
  panel's drawdown property, gated on D1-D5): retime or rescale the
  EXISTING SMA-based vote; neither replaces the anchor statistic, and
  neither is scored against this project's standard single-asset B1-B5
  promotion bar (see R-146's own bar below).
- R-01 (HMM), R-82 (BOCPD), R-83 (causal Kalman local-linear-trend
  filter), R-85 (critical slowing down), R-86 (transfer entropy), R-96
  (Hawkes self-exciting process), R-98 (POT/GPD tail shape), R-99
  (Barndorff-Nielsen-Shephard bipower-variation jump/continuous split),
  R-139 (CUSUM), R-141 (LPPLS): nine structurally distinct WHOLESALE
  regime-timing estimators, each replacing the vote/anchor mechanism
  entirely and scored against a six-episode DETECTION-LAG gate (does it
  detect a known historical regime break faster than v4's own
  20/40/80-day anchor-crossing heuristic?) -- a different question from
  this round's (does a more ROBUST anchor, same architecture otherwise,
  change v4's own risk/return profile on the standard B1-B5 bar?). None
  of the nine holds the band/hysteresis/3-way-vote architecture fixed;
  each replaces it outright. R-60's own novel branch used a CUSUM-based
  vote-cadence rescaling, evaluated on the panel's D1-D5 gate, not this
  bar; its own D2 control failed outright (`docs/LEDGER.md` R-60).
- R-99, R-102, R-103 (bipower-variation jump/continuous split; realized
  downside/upside semivariance; a-priori-gridded and causally-fit
  signed-jump-asymmetry discounts): all four operate on `scale` (v4's
  conditional volatility target), explicitly disclaimed as not touching
  the vote/anchor (R-99's own module docstring; R-102's/R-103's citation
  chain is Barndorff-Nielsen/Kinnebrock/Shephard 2010 and Patton &
  Sheppard 2015, both about REALIZED VARIANCE, not about the anchor's own
  location statistic). This round's object -- a rolling location
  (central-tendency) estimator of PRICE LEVEL -- is mathematically
  unrelated to a decomposition of realized QUADRATIC VARIATION; the
  4-for-4 closure of that object (R-99/R-102/R-103, `docs/LEDGER.md`
  section D re-ranking after R-103) does not bear on this round's
  question. This round's jump-robust construction (novel branch) does
  reuse "is this bar a jump" machinery in the sense of Lee & Mykland
  (2008) rather than R-99's own Barndorff-Nielsen-Shephard bipower
  statistic (see novel branch citation below) -- a different jump test,
  applied to a different end (filtering the anchor's own price input,
  not sizing v4's volatility target or gating a regime alarm).
- The 19+ INFO-axis rounds: every one introduces a NEW EXTERNAL data
  channel (volume, funding, DVOL, MVRV, stablecoin supply, macro, on-chain
  hash rate/active addresses, Wikipedia pageviews, Fear & Greed, Deribit
  term structure level/momentum/slope). Neither branch here reads
  anything beyond the already-committed BTC/ETH OHLCV files' own `close`
  column.

**Literature grounding, fetched and read via WebSearch before either branch
was dispatched:**
- Huber, P. J. (1964), "Robust Estimation of a Location Parameter,"
  *Annals of Mathematical Statistics* 35(1), 73-101, and Hampel, F. R.,
  Ronchetti, E. M., Rousseeuw, P. J., & Stahel, W. A. (1986), *Robust
  Statistics: The Approach Based on Influence Functions*, Wiley --
  establish the median as the location M-estimator with the maximal
  (50%) breakdown point against the arithmetic mean's 0% breakdown point:
  a single arbitrarily large outlier can move a mean's estimate without
  bound, but cannot move a median's past its adjacent order statistic.
  This is the conservative branch's entire citation and mechanism: no
  claim about BTC specifically, just the textbook robustness property of
  the statistic being substituted.
- Levine, A., & Pedersen, L. H. (2016), "Which Trend Is Your Friend?"
  *Financial Analysts Journal* 72(3), 51-66 -- proves that
  moving-average crossovers, time-series momentum, the Hodrick-Prescott
  filter, and the Kalman filter are, formally, equivalent representations
  of ONE underlying linear filter of past returns, differing only in
  their (all still LINEAR) return-weighting scheme. Cited here as the
  reason this round does NOT test an EMA-anchor or HP-filter-anchor
  variant: per Levine-Pedersen's own equivalence proof, any such swap is
  a reparameterization of the same linear-filter class the SMA anchor
  already belongs to (and R-89's own single-anchor decomposition found
  anchor RANKING unstable across inner-train/inner-validation, an N~3
  warning against selecting among same-class linear variants). The
  median (an order statistic, non-linear in its inputs) and the
  jump-trimmed mean (also non-linear: which bars are excluded depends on
  the data itself, in exactly the way Huber's M-estimation formalizes)
  both sit OUTSIDE that equivalence class, which is the actual gap this
  round targets.
- Lee, S. S., & Mykland, P. A. (2008), "Jumps in Financial Markets: A New
  Nonparametric Test and Jump Dynamics," *Review of Financial Studies*
  21(6), 2535-2563 -- a per-INCREMENT (not per-window) nonparametric jump
  test: flag return r_i as a jump when |r_i| divided by a LOCAL,
  jump-robust volatility estimate exceeds a threshold, calibrated by
  extreme-value theory. This is the novel branch's citation and
  mechanism: a causal, per-bar jump flag (local vol estimated from a
  robust, MAD-based trailing dispersion measure so the jump test's own
  denominator is not itself inflated by the jump it is testing for),
  used to EXCLUDE (mask to NaN, not cap-and-reintegrate) the flagged
  bar's REAL close from the rolling window average that feeds the
  anchor -- structurally different from R-99's Barndorff-Nielsen &
  Shephard (2004, 2006) BIPOWER VARIATION, which is an aggregate
  (whole-window) jump/continuous VARIANCE decomposition, not a per-bar
  location-input filter. An earlier draft of this branch instead
  WINSORIZED (capped) the flagged return and reconstructed a synthetic
  price path via cumulative sum -- this module's own self-test caught,
  before either branch was dispatched, that capping only sharp downward
  legs while ordinary gradual recoveries pass through unflagged
  introduces a one-sided compounding drift (BTC 2017-2020: a synthetic
  $6.1M endpoint against the real ~$29K). Exclusion (masking) cannot
  drift this way: every unflagged bar keeps its exact real price level,
  so the anchor differs from v4's own SMA only in which bars are
  averaged, never in what value an included bar contributes.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
Levine & Pedersen's own equivalence result, if the median or the
jump-trimmed mean turn out in practice to correlate near-perfectly with
the plain SMA on this specific 20/40/80-day ladder (BTC's 5-minute return
distribution may simply not have enough single-bar outlier mass at these
long horizons for a robust location estimator to diverge materially from
the mean it replaces) -- the Step-0 kill switch below (R^2 > 0.98 against
v4's own unmodified vote) is exactly the check for this, and if it trips,
this round would be the 27th confirmation of "SIZE-axis constructions
collapse to v4's own path," this time on the vote side of the product for
the first time. If it does NOT trip (a genuinely different vote path), the
mechanism could still fail either direction Baur & Dimpfl (2018, already
this project's own v3/v4 citation) already flagged for `scale`: BTC's
documented INVERSE leverage effect means large POSITIVE moves, not only
crashes, drive its volatility, so a robust anchor that discounts large
moves of EITHER sign could just as easily suppress the vote's reaction to
a genuine bullish breakout as protect it from a liquidation-cascade
whipsaw -- named now as the substantive economic risk, not merely a
methodological one.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it, so both are measured by
identical machinery, the r89-r145 convention. Nothing here reads a bar at
or after OOS_START (2023-01-01); `compare()` asserts this explicitly for
every slice it runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.inference import (  # noqa: E402
    daily_returns as inference_daily_returns,
    paired_bootstrap,
    total_log_return,
)
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# ---------------------------------------------------------------- splits
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# kelly_regime_v4's own shipped constants (do not change: the control must
# be v4, not a re-parameterisation of it). Verified against
# src/tradebot/strategies/{kelly_regime.py,kelly_regime_v3.py,kelly_regime_v4.py}.
V4_HORIZONS: tuple[int, ...] = (20, 40, 80)
V4_BAND = 0.01
V4_TARGET_VOL = 0.55
V4_MAX_LEVERAGE = 2.0
V4_VOL_SPAN = 8 * BARS_PER_DAY
V4_DEADBAND = 0.10
V4_ANCHOR_SPAN_DAYS = 180
V4_HIGH_IN, V4_HIGH_OUT = 1.70, 1.20
V4_LOW_IN, V4_LOW_OUT = 0.55, 0.85


# ------------------------------------------------------------------ data

def assert_no_holdout(df: pd.DataFrame, label: str = "") -> None:
    """Fail loudly if any bar at or after the holdout boundary is present."""
    if len(df) and df.index[-1] >= pd.Timestamp(OOS_START, tz="UTC"):
        raise AssertionError(
            f"{label}: frame reaches {df.index[-1]}, at/after OOS_START={OOS_START}")


def _truncate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df[df.index < pd.Timestamp(OOS_START, tz="UTC")]
    assert_no_holdout(out, label)
    return out


def load_btc() -> pd.DataFrame:
    """The committed BTC spot series, truncated before the holdout."""
    df, _label = load_dataset(ROOT / "data", "spot")
    return _truncate(df, "BTC")


def load_eth() -> pd.DataFrame:
    """Bitfinex ETH (this project's standing cross-asset replication series)."""
    return _truncate(load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz"), "ETH")


# ================================================================== (1)
# kelly_regime_v4's own construction, reproduced EXACTLY (copied from
# r102_shared.py so control numbers stay directly comparable across rounds).
# ==================================================================

def _latched_anchor_vote_from_anchor(close: pd.Series, anchor: pd.Series,
                                     band: float = V4_BAND) -> pd.Series:
    """One anchor's latched 0/1 vote, given an arbitrary (already-computed)
    anchor series instead of always recomputing a plain rolling mean --
    the factoring point this round needs: v4's own vote calls this with a
    plain SMA anchor; the candidates call it with a robust one instead."""
    v = pd.Series(
        np.where(close > anchor * (1.0 + band), 1.0,
                 np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
        index=close.index,
    )
    return v.ffill().fillna(0.0)


def _latched_anchor_vote(close: pd.Series, days: int, band: float = V4_BAND) -> pd.Series:
    """v4's own anchor: a plain rolling mean (SMA)."""
    anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
    return _latched_anchor_vote_from_anchor(close, anchor, band)


def vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
             band: float = V4_BAND, vote_gamma: float = 1.0) -> pd.Series:
    """kelly_regime_v4's own directional vote, as a standalone causal function of OHLCV."""
    close = df["close"]
    votes = [_latched_anchor_vote(close, days, band) for days in horizons]
    frac = sum(votes) / len(votes)
    if vote_gamma != 1.0:
        frac = frac ** vote_gamma
    return frac


def v4_vote_frac(df: pd.DataFrame) -> pd.Series:
    """kelly_regime_v4's own shipped vote (horizons=20,40,80, band=1%), for the control."""
    return vote_frac(df, V4_HORIZONS, V4_BAND)


def v4_symmetric_vol(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    """v4's own TOTAL (symmetric) EWM volatility input, exactly as shipped."""
    r = np.log(df["close"]).diff()
    return (r.ewm(span=span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def conditional_target_scale(vol: np.ndarray, anchor_span_days: int = V4_ANCHOR_SPAN_DAYS,
                              high_in: float = V4_HIGH_IN, high_out: float = V4_HIGH_OUT,
                              low_in: float = V4_LOW_IN, low_out: float = V4_LOW_OUT,
                              target_vol: float = V4_TARGET_VOL,
                              max_leverage: float = V4_MAX_LEVERAGE) -> np.ndarray:
    """v3/v4's extremes-only conditional volatility-target STATE MACHINE
    (reproduced exactly; UNTOUCHED by this round -- both branches feed it
    v4's own `v4_symmetric_vol`, only the vote's anchor statistic changes)."""
    vol = np.asarray(vol, dtype=float)
    slow = (pd.Series(vol).ewm(span=anchor_span_days * BARS_PER_DAY,
                               min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(target_vol / vol, max_leverage)
        steady = np.minimum(target_vol / slow, max_leverage)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(vol)
    out = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > high_in else (-1 if x < low_in else 0)
            elif state == 1 and x < high_out:
                state = 0
            elif state == -1 and x > low_out:
                state = 0
        out[i] = full[i] if state != 0 else steady[i]
    return out


def v4_scale(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v3/v4's conditional volatility-target scale factor,
    reproduced exactly. UNTOUCHED by both branches of this round."""
    return conditional_target_scale(v4_symmetric_vol(df))


def apply_deadband(desired: np.ndarray, deadband: float = V4_DEADBAND) -> np.ndarray:
    """v4's own 10% re-target deadband, applied to a desired-exposure path."""
    target = np.zeros(len(desired))
    pos = 0.0
    for i, d in enumerate(desired):
        if abs(d - pos) > deadband:
            pos = float(d)
        target[i] = pos
    return target


def v4_raw_desired(df: pd.DataFrame) -> np.ndarray:
    """v4's desired exposure BEFORE its own 10% deadband: frac * scale."""
    return v4_vote_frac(df).to_numpy() * v4_scale(df)


def v4_target(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v4's complete, final target path (post-deadband) -- the control."""
    return apply_deadband(v4_raw_desired(df))


# ================================================================== (2)
# Robust anchor constructions. Both replace ONLY the per-anchor statistic
# inside `_latched_anchor_vote`; band, hysteresis-via-ffill, 3-way
# averaging, gamma, scale and deadband are all untouched and identical to
# v4's own.
# ==================================================================

def median_anchor_vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                            band: float = V4_BAND) -> pd.Series:
    """CONSERVATIVE candidate: replace each SMA anchor with a rolling
    MEDIAN of the same window length (Huber 1964 / Hampel et al. 1986:
    the location M-estimator with maximal, 50%, breakdown point). Nothing
    else about the vote changes."""
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).median()
        votes.append(_latched_anchor_vote_from_anchor(close, anchor, band))
    return sum(votes) / len(votes)


def _causal_jump_flag(close: pd.Series, mad_span: int = BARS_PER_DAY,
                      threshold: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Lee & Mykland (2008)-style causal per-bar jump flag: bar i is a jump
    when |r_i| exceeds `threshold` times a LOCAL, jump-robust dispersion
    estimate. The local estimate is a trailing, causal MAD of returns
    STRICTLY BEFORE bar i (shifted by 1, so bar i's own possibly-extreme
    return can never inflate its own test denominator), scaled by
    1.4826 for consistency with the standard deviation under normality.
    Returns (jump_flag: bool array, r: the raw log-return array)."""
    r = np.log(close).diff()
    abs_r = r.abs()
    med_abs = abs_r.rolling(mad_span, min_periods=mad_span // 4).median()
    mad = (abs_r - med_abs).abs().rolling(mad_span, min_periods=mad_span // 4).median()
    local_vol = (1.4826 * mad).shift(1)
    r_np = r.to_numpy()
    local_vol_np = local_vol.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(local_vol_np > 0, np.abs(r_np) / local_vol_np, 0.0)
    flag = np.isfinite(z) & (z > threshold)
    return flag, r_np, local_vol_np


def jump_masked_close(df: pd.DataFrame, mad_span: int = BARS_PER_DAY,
                      threshold: float = 4.0) -> pd.Series:
    """NOVEL candidate's cleaned input series: bars flagged as jumps (Lee &
    Mykland 2008-style, see `_causal_jump_flag`) are masked to NaN; every
    other bar's REAL close is left untouched. Feeding this into a rolling
    MEAN (`.rolling(...).mean()` skips NaN within the window, provided
    `min_periods` non-NaN observations remain) computes an anchor that
    excludes a small number of single-bar dislocations (flash crashes,
    liquidation cascades, thin-book prints) from the trend average,
    without discarding or reweighting any non-flagged bar and, critically,
    WITHOUT reconstructing a synthetic price path (an earlier version of
    this construction reconstructed price via cumsum of winsorized
    log-returns and was caught, before either branch was dispatched, by
    this module's own self-test: capping only the DOWNWARD leg of sharp
    moves while ordinary gradual recoveries pass through unflagged
    introduces a one-sided drift that compounds over years -- BTC
    2017-2020 diverged to a synthetic $6.1M endpoint against the real
    ~$29K. Masking preserves every unflagged bar's real price level
    exactly, so no such drift is possible by construction)."""
    close = df["close"]
    flag, _r, _local_vol = _causal_jump_flag(close, mad_span, threshold)
    return close.where(~flag, other=np.nan)


def jump_robust_anchor_vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                                 band: float = V4_BAND, mad_span: int = BARS_PER_DAY,
                                 threshold: float = 4.0) -> pd.Series:
    """NOVEL candidate: each anchor is a rolling MEAN of `close` with
    jump-flagged bars excluded (masked to NaN, which pandas' rolling mean
    skips), rather than a plain SMA over every bar. The vote is computed
    against the REAL close (as v4's own vote is); only the anchor's own
    smoothing input excludes a handful of flagged bars per window."""
    close = df["close"]
    masked = jump_masked_close(df, mad_span, threshold)
    votes = []
    for days in horizons:
        window = int(days * BARS_PER_DAY)
        anchor = masked.rolling(window, min_periods=max(1, window // 4)).mean()
        votes.append(_latched_anchor_vote_from_anchor(close, anchor, band))
    return sum(votes) / len(votes)


# ------------------------------------------------------- causal truncation

def causal_truncation_probe_series(build_fn, df: pd.DataFrame,
                                   cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    """Standard truncation probe: truncate the input frame to [:k] and
    recompute; the shared prefix must match the full-series computation
    exactly, AND perturbing the tail must not move any value inside the
    shared prefix."""
    full = np.asarray(build_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        if k < BARS_PER_DAY * 2:
            continue
        part = np.asarray(build_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-10, rtol=1e-9):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-10, rtol=1e-9)))
            raise AssertionError(f"{build_fn.__name__} causality FAIL at cut={cut}: {bad} bars differ")
        perturbed = df.copy()
        tail = perturbed.iloc[k:].copy()
        for col in ("open", "high", "low", "close"):
            if col in tail.columns:
                tail[col] = tail[col] * 3.7 + 1.0
        perturbed.iloc[k:] = tail
        pert = np.asarray(build_fn(perturbed), dtype=float)
        pm = np.isfinite(a) & np.isfinite(pert[:k])
        if not np.allclose(a[pm], pert[:k][pm], atol=1e-10, rtol=1e-9):
            raise AssertionError(f"{build_fn.__name__} peeks at bar>=k, cut={cut}")
    return True


def causal_truncation_probe_vote(build_fn, df: pd.DataFrame,
                                 cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    full = np.asarray(build_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        part = np.asarray(build_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-12, rtol=0.0):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-12, rtol=0.0)))
            raise AssertionError(f"{build_fn.__name__} vote causality FAIL at cut={cut}: {bad} bars differ")
    return True


# ================================================================== (3)
# compare(): run any pure `build_target(df) -> np.ndarray` candidate over
# inner-train, inner-validation and the ETH replication slice, vs
# kelly_regime_v4, never touching OOS_START. Structurally identical to
# r102_shared.py's compare().
# ==================================================================

SLICES: dict[str, tuple[str | None, str | None]] = {
    "inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
    "inner_val": (INNER_VAL_START, INNER_VAL_END),
}
ETH_SLICE_NAME = "eth_replication"

for _name, (_s, _e) in SLICES.items():
    if _e is not None:
        assert pd.Timestamp(_e) < pd.Timestamp(OOS_START), (
            f"SLICES[{_name!r}] end={_e} is not before OOS_START={OOS_START}")


@dataclass
class SliceResult:
    name: str
    market: str
    final_balance: float
    sharpe: float
    max_drawdown_pct: float
    num_trades: int
    log_growth: float
    daily: np.ndarray
    mean_abs_exposure: float
    realized_vol: float


def daily_simple_returns(equity: pd.Series) -> np.ndarray:
    return inference_daily_returns(equity).to_numpy()


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy."""

    name = "r146_control"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r146_control",
                warmup: int | None = None) -> None:
        self._build = build_target
        self.name = name
        if warmup is not None:
            self.warmup = warmup

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target"] = np.asarray(self._build(df), dtype=float)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)


def run_slice(strategy: Strategy, df: pd.DataFrame, start: str | None, end: str | None,
             slice_name: str, market: MarketSpec = SPOT,
             balance: float = 1_000.0) -> SliceResult:
    if end is not None:
        assert pd.Timestamp(end) < pd.Timestamp(OOS_START), (
            f"run_slice({slice_name!r}): end={end} is not before OOS_START={OOS_START}")
    assert_no_holdout(df, slice_name)

    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
    assert_no_holdout(res.equity.to_frame(), f"{slice_name} result")
    m = compute_metrics(res)
    d = daily_simple_returns(res.equity)
    exposure = res.df["target"].to_numpy() if "target" in res.df.columns else np.array([np.nan])
    return SliceResult(
        name=slice_name, market=market.name, final_balance=m.final_balance,
        sharpe=m.sharpe, max_drawdown_pct=m.max_drawdown_pct,
        num_trades=m.num_trades, log_growth=float(total_log_return(d)), daily=d,
        mean_abs_exposure=float(np.nanmean(np.abs(exposure))),
        realized_vol=float(np.nanstd(d) * np.sqrt(365.25)) if len(d) > 1 else float("nan"),
    )


def paired_diff(candidate: np.ndarray, control: np.ndarray, *,
                mean_block: float = 30.0, n_boot: int = 2_000, seed: int = 0):
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                            np.asarray(control[-n:], dtype=float),
                            total_log_return, mean_block=mean_block,
                            n_boot=n_boot, seed=seed)


def compare(candidate_build, *, label: str, btc: pd.DataFrame | None = None,
           eth: pd.DataFrame | None = None, control_build=None,
           markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
           include_eth: bool = True, seed: int = 0) -> list[dict]:
    if control_build is None:
        control_build = v4_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r146_{label}")
    ctrl = TargetStrategy(control_build, name="kelly_regime_v4")

    rows = []
    jobs = [(name, start, end, btc) for name, (start, end) in SLICES.items()]
    if include_eth:
        jobs.append((ETH_SLICE_NAME, None, None, eth))

    for slice_name, start, end, df in jobs:
        for market in markets:
            a = run_slice(cand, df, start, end, slice_name, market)
            b = run_slice(ctrl, df, start, end, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            exp_ratio = (a.mean_abs_exposure / b.mean_abs_exposure
                        if b.mean_abs_exposure else float("nan"))
            vol_ratio = (a.realized_vol / b.realized_vol
                        if b.realized_vol else float("nan"))
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_log_growth": a.log_growth, "ctrl_log_growth": b.log_growth,
                "d_log_growth": a.log_growth - b.log_growth,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "exposure_ratio": exp_ratio, "vol_ratio": vol_ratio,
                "risk_matched": bool(0.9 <= exp_ratio <= 1.1 and 0.9 <= vol_ratio <= 1.1)
                                if np.isfinite(exp_ratio) and np.isfinite(vol_ratio) else False,
                "boot_d_loggrowth": pr.diff.point,
                "boot_lo": pr.diff.lo, "boot_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def print_rows(rows: list[dict]) -> None:
    hdr = (f"{'label':26s} {'slice':16s} {'market':11s} {'cand$':>10s} {'ctrl$':>10s} "
          f"{'dSh':>6s} {'dDD':>7s} {'expR':>5s} {'volR':>5s} {'RM':>3s} "
          f"{'dlogG':>7s} {'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:26]:26s} {r['slice']:16s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['ctrl_final']:10,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['boot_d_loggrowth']:+7.3f} {r['boot_lo']:+8.3f},{r['boot_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


def fee_at(market: MarketSpec, fee_rate: float) -> MarketSpec:
    return MarketSpec(name=market.name, leverage=market.leverage, fee_rate=fee_rate,
                      allow_short=market.allow_short,
                      maintenance_margin_rate=market.maintenance_margin_rate,
                      min_notional=market.min_notional, pays_funding=market.pays_funding)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or np.std(b) == 0:
        return float("nan")
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ============================================================ pre-registration
#
# PROMOTION BAR (identical shape to R-89/R-93/R-97/R-99/R-101/R-102/R-103):
#
#  A2 (Step-0 non-inertness kill switch): R^2 of the candidate's own raw
#     vote_frac path against v4's unmodified `v4_vote_frac`, computed on
#     inner-train. If R^2 > 0.98, STOP -- the candidate is a disguised
#     reparameterization of v4's own vote, not a genuinely different one,
#     and no Sharpe number is read past this point.
#  B1: bootstrap paired difference in total log-growth, inner-validation,
#     BOTH markets: ΔSharpe > +0.2 OR the 95% bootstrap interval excludes
#     zero.
#  B2 (diagnostic, not gating): risk-matched drawdown -- report
#     exposure_ratio / vol_ratio; both in [0.9, 1.1] is a genuine
#     risk-matched comparison, per this project's own standing R-33 rule.
#  B3: plateau -- report the SAME sign/direction of B1's result across a
#     pre-registered parameter grid (conservative: no free parameter beyond
#     the window length already fixed at v4's own 20/40/80-day ladder, so
#     B3 here is read across the mad_span/threshold grid's ANALOGUE for
#     the novel branch, and across a robustness check on the median
#     itself -- see each branch's own script); a single winning cell with
#     no support around it does not clear this bar.
#  B4: ETH same-sign falsification -- the candidate's ΔSharpe (or
#     bootstrap direction) on the ETH replication slice must agree in SIGN
#     with the BTC inner-validation result on at least one market.
#  B5: 0.40% taker-fee-tier re-run (fee_at(SPOT, 0.004)) -- the edge, if
#     any, must not require the 0.10% fee tier to exist.
#
# Promote only if A2 does not trip AND B1 passes on >=1 market AND B4
# passes AND B5's edge (if B1 passed) survives in sign. Anything else is
# NEGATIVE. This is the SAME bar both branches must clear; neither may
# weaken it after seeing a number.
# ============================================================


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(146)
    innov = rng.normal(0, 0.0006, len(idx))
    # Inject a few sharp one-bar jumps so the jump-flag machinery has
    # something non-degenerate to find on synthetic data.
    jump_idx = rng.choice(len(idx), size=15, replace=False)
    innov[jump_idx] += rng.choice([-1, 1], size=15) * rng.uniform(0.01, 0.03, size=15)
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) v4 control self-consistency (copied convention from r102).
    raw = v4_raw_desired(df)
    assert np.allclose(v4_target(df), apply_deadband(raw))
    assert np.array_equal(v4_vote_frac(df).to_numpy(), vote_frac(df, V4_HORIZONS).to_numpy())
    assert vote_frac(df).between(0.0, 1.0).all()
    assert np.allclose(v4_scale(df), conditional_target_scale(v4_symmetric_vol(df)))

    # (2) median-anchor vote is well-formed and genuinely different from v4's.
    med = median_anchor_vote_frac(df)
    assert med.between(0.0, 1.0).all()
    assert not np.allclose(med.to_numpy(), v4_vote_frac(df).to_numpy())

    # (3) jump-flag machinery: fires on injected jumps more than on ordinary
    # bars (sanity, not a promotion-relevant statistic), and RS(jump) sums
    # to a small, non-degenerate fraction of bars.
    flag, r, local_vol = _causal_jump_flag(df["close"])
    assert flag.sum() > 0, "no jumps ever flagged on synthetic data with injected jumps"
    assert flag.mean() < 0.05, "jump flag fires implausibly often -- threshold too loose"

    # (4) jump-masked close: flagged bars become NaN, every other bar is
    # exactly the real close (no reconstruction, so no drift is possible).
    masked = jump_masked_close(df)
    unflagged = ~flag
    assert np.allclose(masked.to_numpy()[unflagged], df["close"].to_numpy()[unflagged])
    assert masked.isna().sum() == flag.sum()

    # (5) jump-robust-anchor vote is well-formed and differs from both v4's
    # and the median candidate's.
    jr = jump_robust_anchor_vote_frac(df)
    assert jr.between(0.0, 1.0).all()
    assert not np.allclose(jr.to_numpy(), v4_vote_frac(df).to_numpy())

    # (6) causal truncation probes -- the properties that matter for this
    # round: no candidate may peek at future bars.
    assert causal_truncation_probe_vote(lambda d: v4_vote_frac(d).to_numpy(), df)
    assert causal_truncation_probe_vote(lambda d: median_anchor_vote_frac(d).to_numpy(), df)
    assert causal_truncation_probe_vote(lambda d: jump_robust_anchor_vote_frac(d).to_numpy(), df)
    assert causal_truncation_probe_series(lambda d: jump_masked_close(d).to_numpy(), df)
    assert causal_truncation_probe_series(v4_scale, df)

    # (7) r_squared sanity.
    assert abs(r_squared(med.to_numpy(), med.to_numpy()) - 1.0) < 1e-9
    assert r_squared(med.to_numpy(), rng.normal(0, 1, len(med))) < 0.5


_self_test()
