"""Shared, read-only utilities and pre-registration for the R-102 round (08-23).

DIRECTION, in one sentence: decompose `kelly_regime_v4`'s realized-volatility
input by the SIGN of the underlying 5-minute returns -- realized DOWNSIDE and
UPSIDE semivariance, computed from this project's own native 5-minute bars --
and test two ways of using that decomposition on the SIZE axis: replace the
conditional-vol-target's `scale` input with the downside component alone
(conservative), or leave `scale` untouched and add an independent
signed-asymmetry discount on top of `frac * scale` (novel).

**Literature grounding, fetched and read via WebSearch before either branch
was dispatched:**

- Barndorff-Nielsen, O. E., Kinnebrock, S., & Shephard, N. (2010),
  "Measuring Downside Risk -- Realised Semivariance", in T. Bollerslev,
  J. Russell & M. Watson (eds.), *Volatility and Time Series Econometrics:
  Essays in Honor of Robert F. Engle*, Oxford University Press (Nuffield
  College Economics WP 2008-W02; SSRN 1262194). Confirmed live via
  WebSearch this round: "The paper introduces a new measure of the
  variation of asset prices based on high frequency data, called realized
  semivariance (RS)... Realized Variance can be decomposed to realized
  downside semivariance and realized upside semivariance." Formally, for a
  day (or any fixed window) with M observed high-frequency log returns
  r_1..r_M: RV = sum(r_i^2), RS- = sum(r_i^2 * 1{r_i<0}), RS+ = sum(r_i^2 *
  1{r_i>0}), and RV = RS- + RS+ EXACTLY (the two indicator sets partition
  every non-zero return). This is a genuinely different quadratic-variation
  decomposition than R-99's bipower-variation JUMP/CONTINUOUS split (which
  partitions by magnitude structure across adjacent returns, not by sign of
  a single return) -- orthogonal machinery, same native-5-minute data
  requirement.
- Ang, A., Chen, J., & Xing, Y. (2006), "Downside Risk", *Review of
  Financial Studies* 19(4), 1191-1239 (NBER WP 11824) -- establishes that
  downside risk (covariation with the market conditional on the market
  falling) is priced SEPARATELY from total risk/beta: portfolios sorted on
  downside risk earn a return premium total-volatility sorts do not fully
  capture. Cited here only as the economic reason to expect the downside
  component specifically, not total variance, to carry the crash-relevant
  information -- not as evidence on BTC itself, which this round measures
  directly.
- Patton, A. J., & Sheppard, K. (2015), "Good Volatility, Bad Volatility:
  Signed Jumps and the Persistence of Volatility", *Review of Economics and
  Statistics* 97(3), 683-697 (earlier circulated as "Signed Jumps and the
  Persistence of Volatility", Oxford Dept. of Economics WP). Defines the
  relative signed jump measure RSJ_t = (RS+_t - RS-_t) / RV_t in [-1, 1]
  and shows, across equity index and individual-stock data, that NEGATIVE
  RSJ (variation dominated by the downside component) predicts SIGNIFICANTLY
  HIGHER future realized volatility than positive RSJ of the same
  magnitude -- an asymmetric persistence result, distinct from (and a
  refinement of) the older "leverage effect" (negative returns raise future
  vol) since RSJ is a SHAPE statistic of the current period's own
  variation, not merely the sign of the current period's return. This is
  the novel branch's citation and its named mechanism.
- Baur, D. G., & Dimpfl, T. (2018), "Asymmetric volatility in
  cryptocurrencies", *Economics Letters* 173, 148-151 -- already this
  project's own citation for `kelly_regime_v3`/`v4`'s conditional-vol-target
  architecture (BTC's documented INVERSE leverage effect: positive shocks
  raise volatility MORE than negative ones, the opposite of equities).
  Cited here as the reason this round's conservative branch is a genuine
  open question rather than a foregone conclusion: if BTC's inverse
  leverage effect means UPSIDE variation is actually the larger and more
  volatility-informative component on this asset, a downside-only target
  could be actively worse than v4's existing total-variance target, not
  merely a relabelling of it -- named now, before any real-data number, as
  this round's own falsification risk (see WHAT WOULD MAKE THIS FAIL below).

**Which constraint this attacks: SIZE.** `kelly_regime_v4` sizes
`desired = frac * scale`, and R-62 (four independent confirmations, most
recently R-87) established that `frac` (the 3-anchor vote) carries the
strategy's entire matched-exposure signature while `scale` (the conditional
volatility target) carries none of it in isolation -- but every SIZE-axis
round to date that touched `scale` (R-38, R-45, R-46, R-59, R-60, R-93,
R-97) retuned its MAGNITUDE or supplied an entirely different state variable
(market volatility level, the strategy's own realized drawdown, a
distributional-robustness ambiguity radius), never its SIGN STRUCTURE. This
round is the first to ask whether `scale`'s total-variance input is even the
right building block, independent of how it is subsequently thresholded --
by splitting the SAME realized quadratic variation v4 already consumes into
its downside and upside halves, a decomposition that is well-defined (and,
per Barndorff-Nielsen/Kinnebrock/Shephard, only estimable this way) BECAUSE
this project's data is native 5-minute bars, not daily closes.

**Not a duplicate of:**
- R-93 (Grossman-Zhou), R-97 (Wasserstein-DRO): both replace `scale` with a
  function of an exogenous market-level statistic (Grossman-Zhou) or a
  robustness bound (Wasserstein-DRO) computed at DAILY-OR-COARSER
  resolution from close-to-close returns or regime-cycle counts. Neither
  reads a single intraday (sub-daily) bar; both are computable from a
  daily-resampled price series alone. This round's entire object (RS-/RS+)
  is UNDEFINED on daily-resampled data -- a day's single close-to-close
  return has one sign, so "downside semivariance" of a daily series
  degenerates to either the whole squared return or zero, discarding
  exactly the intraday sign-mixing information the decomposition exists to
  capture. Structurally distinct data requirement, not merely a different
  formula on the same inputs.
- R-62 (factor isolation: vote alone vs. scale alone, scale forced to a
  frozen constant or v4's own unmodified target): does not replace scale's
  underlying statistic at all.
- R-87 (Adaptive Conformal Inference on the vote's dispersion / the Kelly
  scale's dispersion estimator): an online COVERAGE-calibration wrapper
  around an existing point estimate, not a different point estimate of
  realized variation itself.
- R-101 (delete-one-episode jackknife confidence multiplier): a
  resampling-based estimate of PARAMETER uncertainty over six discrete
  historical episodes; this round reads no episode calendar and computes a
  continuous statistic from every bar, not six point estimates.
- R-99 (bipower-variation jump/continuous decomposition): the closest
  methodological relative -- both are native-5-minute quadratic-variation
  decompositions -- but R-99 (a) partitions by MAGNITUDE STRUCTURE across
  adjacent return pairs (a jump test), (b) is a regime-timing ALARM fed
  additively into the vote/gate, exactly like R-82/83/84/85/86/96/98, and
  (c) explicitly "does not touch `scale` at all" (its own module docstring).
  This round partitions by the SIGN of a single return (a completely
  different mathematical object: RS-/RS+ sum to RV exactly, with no jump
  test or asymptotic null involved), and its conservative branch's whole
  point is to replace `scale`'s input -- the one thing R-99 named as
  outside its own scope.
- The fifteen INFO-axis rounds: every one introduces a NEW EXTERNAL data
  channel. Neither branch here reads anything beyond the already-committed
  BTC OHLCV file's own `close` column at its own native cadence.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it, so both are measured by
identical machinery, exactly the r89-r101 convention. Nothing here reads a
bar at or after OOS_START (2023-01-01); `compare()` asserts this explicitly
for every slice it runs.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
Baur & Dimpfl's own inverse-leverage finding, taken at face value, says
POSITIVE shocks raise BTC's volatility MORE than negative ones -- which
would mean the UPSIDE component, not the downside one, carries more of the
"vol level that predicts danger" information this project's whole
`kelly_regime` family conditions on, and a downside-only target could
systematically UNDER-react to exactly the upside-vol-driven breakouts v3/v4
already de-lever into. If downside and upside EWM variance are, in
practice, highly correlated on this asset (BTC's realized vol clusters
regardless of return sign, since large moves beget large moves of either
sign -- the ordinary volatility-clustering stylized fact, orthogonal to
which SIGN dominates), the conservative branch's own kill switch (A2 below)
will show the substitution is a near-identical rescale of v4's existing
`scale`, reproducing the now-familiar "22+ SIZE-axis attempts collapse to
v4's own path" finding in a new guise. For the novel branch specifically:
Patton & Sheppard's asymmetric-persistence result is from equity index and
single-stock daily data; if BTC's own realized-variance process does not
share it (a real possibility this project has already seen once, at B-42,
when a different equity/futures-literature closed form failed to transfer
because BTC's own autocorrelation-vs-drift balance did not match the
theory's assumed regime), the RSJ overlay will either be inert (a
near-constant discount of ~1.0 whenever it matters) or will discount
during ordinary bidirectional volatility clustering that is not actually
predictive of anything, adding turnover without adding information.
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
# vote_frac / v4_scale / apply_deadband / v4_target: kelly_regime_v4's own
# construction, reproduced EXACTLY. Copied verbatim from r93_shared.py so
# both rounds' control numbers stay directly comparable.
# ==================================================================

def _latched_anchor_vote(close: pd.Series, days: int, band: float = V4_BAND) -> pd.Series:
    """One anchor's own latched 0/1 vote, exactly as v4 computes each of its three."""
    anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
    v = pd.Series(
        np.where(close > anchor * (1.0 + band), 1.0,
                 np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
        index=close.index,
    )
    return v.ffill().fillna(0.0)


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
    """v3/v4's extremes-only conditional volatility-target STATE MACHINE, factored
    out as a pure function of an arbitrary causal volatility series `vol` (v4
    itself calls this with `v4_symmetric_vol`; the conservative branch calls it
    with a downside-only series instead -- same architecture, different input).

    Hold a constant notional (`target_vol / slow_ewm(vol)`) while `vol / slow`
    sits inside [low_in, high_in]; switch to full inverse-vol sizing
    (`target_vol / vol`) once it breaks out, latching until it retraces inside
    [low_out, high_out]. Identical state machine to `KellyRegimeV3.prepare`.
    """
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
    """kelly_regime_v3/v4's conditional volatility-target scale factor, reproduced
    exactly (as `conditional_target_scale` fed v4's own symmetric vol input)."""
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
# Realized downside/upside semivariance from NATIVE 5-minute log returns.
# Barndorff-Nielsen, Kinnebrock & Shephard (2010): RV = RS- + RS+ EXACTLY,
# where RS-/RS+ sum only the squared returns of one sign. Implemented here as
# a continuous EWM analogue (same smoothing shape as v4's own `.std()`-based
# `vol`) rather than a daily-resampled sum, so the SAME time granularity and
# span parameter can be compared apples-to-apples against v4's own
# `v4_symmetric_vol` -- the only thing that differs is which returns are
# squared before smoothing.
#
# DISCLOSED SIMPLIFICATION: v4's own `vol` uses `.ewm(...).std()`, the
# exponentially-weighted STANDARD DEVIATION (centred on the EWM mean of `r`,
# per pandas' definition), not the raw second moment `sqrt(ewm_mean(r**2))`
# the realized-semivariance literature's RV is literally defined as. On
# 5-minute BTC log returns the EWM mean of `r` is negligible relative to its
# own dispersion (typical bar-level mean magnitude is several orders smaller
# than the bar-level standard deviation), so the two constructions are
# numerically close but not identical; `raw_ewm_total_vol` below is the exact
# undemeaned analogue of `v4_symmetric_vol`, used ONLY for the A2 kill-switch
# identity check `raw_down**2 + raw_up**2 == raw_total**2` (exact, by
# construction, since r_neg*r_pos=0 elementwise for every bar), never as a
# strategy input in place of v4's own shipped `v4_symmetric_vol`.
# ==================================================================

def _signed_ewm_var(r: pd.Series, span: int, sign: str) -> pd.Series:
    """EWM mean of r^2, restricted to one sign of r (the other sign's bars
    contribute exactly zero to the sum -- the discrete analogue of RS-/RS+).
    `sign` is 'down' (r < 0) or 'up' (r > 0). Causal: `.ewm()` is a strictly
    backward-looking recursion by construction."""
    if sign == "down":
        r_signed = r.clip(upper=0.0)
    elif sign == "up":
        r_signed = r.clip(lower=0.0)
    else:
        raise ValueError(f"sign must be 'down' or 'up', got {sign!r}")
    return (r_signed ** 2).ewm(span=span, min_periods=BARS_PER_DAY).mean()


def downside_ewm_vol(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    """Causal, annualized realized DOWNSIDE semivariance-based volatility:
    sqrt(EWM[r^2 * 1{r<0}] * BARS_PER_YEAR), shifted by 1 bar (identical
    shift convention to `v4_symmetric_vol`, so bar i's value uses only
    returns up to and including bar i, and is available at bar i+1's open)."""
    r = np.log(df["close"]).diff()
    var_down = _signed_ewm_var(r, span, "down")
    return (np.sqrt(var_down) * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def upside_ewm_vol(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    """Causal, annualized realized UPSIDE semivariance-based volatility;
    same construction as `downside_ewm_vol`, restricted to r > 0."""
    r = np.log(df["close"]).diff()
    var_up = _signed_ewm_var(r, span, "up")
    return (np.sqrt(var_up) * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def raw_ewm_total_vol(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    """Undemeaned analogue of `v4_symmetric_vol` (sqrt(EWM[r^2]) rather than
    `.ewm().std()`), used only to verify RS- + RS+ == RV exactly; never a
    strategy input."""
    r = np.log(df["close"]).diff()
    var_total = (r ** 2).ewm(span=span, min_periods=BARS_PER_DAY).mean()
    return (np.sqrt(var_total) * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def relative_signed_jump(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    """Patton & Sheppard (2015)'s RSJ_t = (RS+_t - RS-_t) / RV_t, continuous
    EWM analogue: bounded in [-1, 1] (NaN where RV is exactly 0, which the
    later `np.nan_to_num` calls in both branches must handle explicitly
    rather than silently). Negative RSJ = variation dominated by the
    downside component ("bad volatility" in Patton & Sheppard's own
    terminology); positive RSJ = dominated by the upside component."""
    r = np.log(df["close"]).diff()
    var_down = _signed_ewm_var(r, span, "down").shift(1)
    var_up = _signed_ewm_var(r, span, "up").shift(1)
    var_total = var_down + var_up
    with np.errstate(divide="ignore", invalid="ignore"):
        rsj = np.where(var_total.to_numpy() > 0,
                       (var_up.to_numpy() - var_down.to_numpy()) / var_total.to_numpy(),
                       np.nan)
    return rsj


# ------------------------------------------------------- causal truncation

def causal_truncation_probe_series(build_fn, df: pd.DataFrame,
                                   cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    """Standard truncation probe for any `df -> np.ndarray` builder in this
    module: truncate the input frame to [:k] and recompute; the shared
    prefix must match the full-series computation exactly, AND perturbing
    the tail must not move any value inside the shared prefix."""
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
        # Perturb the tail; the shared prefix must be completely unaffected.
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


def causal_truncation_probe_vote(df: pd.DataFrame,
                                 cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    """Truncate the input frame to [:k] and recompute vote_frac; the shared
    prefix must match the full-series computation exactly."""
    full = vote_frac(df).to_numpy()
    for cut in cuts:
        k = int(len(df) * cut)
        part = vote_frac(df.iloc[:k]).to_numpy()
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-12, rtol=0.0):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-12, rtol=0.0)))
            raise AssertionError(f"vote_frac causality FAIL at cut={cut}: {bad} bars differ")
    return True


# ================================================================== (3)
# compare(): run any pure `build_target(df) -> np.ndarray` candidate over
# inner-train, inner-validation and the ETH replication slice, vs
# kelly_regime_v4, never touching OOS_START. Structurally identical to
# r93_shared.py's compare(), simplified because every candidate on THIS
# round is a pure function of price (no live-equity dependency, unlike
# R-93's Grossman-Zhou), so both branches use TargetStrategy directly.
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
    """Daily SIMPLE returns of a bar-frequency equity curve."""
    return inference_daily_returns(equity).to_numpy()


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy."""

    name = "r102_control"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r102_control",
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
    """One backtest over an explicit [start, end] window, with a warm prefix."""
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
    """Paired stationary-block-bootstrap difference in total log growth."""
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                            np.asarray(control[-n:], dtype=float),
                            total_log_return, mean_block=mean_block,
                            n_boot=n_boot, seed=seed)


def compare(candidate_build, *, label: str, btc: pd.DataFrame | None = None,
           eth: pd.DataFrame | None = None, control_build=None,
           markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
           include_eth: bool = True, seed: int = 0) -> list[dict]:
    """Candidate (a pure `build_target(df) -> np.ndarray`) vs kelly_regime_v4
    on inner-train, inner-validation, and the ETH replication slice, on every
    market. Never reads a bar at or after OOS_START."""
    if control_build is None:
        control_build = v4_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r102_{label}")
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
    """One fixed-width line per cell, so branches' output is diffable."""
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
    """Same market spec, at a different taker fee (cost-robustness checks)."""
    return MarketSpec(name=market.name, leverage=market.leverage, fee_rate=fee_rate,
                      allow_short=market.allow_short,
                      maintenance_margin_rate=market.maintenance_margin_rate,
                      min_notional=market.min_notional, pays_funding=market.pays_funding)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Simple R^2 of `a` against `b` (used for the A2 non-inertness kill
    switch: is the candidate's scale/exposure path a near-exact rescale of
    v4's own?), over the finite overlap of both series."""
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


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Fast checks on synthetic data. Mirrors r89-r99_shared.py's convention."""
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(102)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) vote_frac / v4_target self-consistency.
    raw = v4_raw_desired(df)
    assert np.allclose(v4_target(df), apply_deadband(raw)), \
        "v4_target != apply_deadband(v4_raw_desired)"
    assert np.array_equal(v4_vote_frac(df).to_numpy(), vote_frac(df, V4_HORIZONS).to_numpy())
    assert vote_frac(df).between(0.0, 1.0).all()

    # (2) RS- + RS+ == RV EXACTLY (the defining identity of the decomposition),
    # checked via the undemeaned analogue so the identity is exact, not
    # merely close, and confirming the ".std()" vs raw-second-moment
    # disclosed simplification is small: v4_symmetric_vol vs raw_ewm_total_vol.
    down = downside_ewm_vol(df)
    up = upside_ewm_vol(df)
    total_raw = raw_ewm_total_vol(df)
    m = np.isfinite(down) & np.isfinite(up) & np.isfinite(total_raw)
    assert m.sum() > 1000
    lhs = down[m] ** 2 + up[m] ** 2
    rhs = total_raw[m] ** 2
    assert np.allclose(lhs, rhs, rtol=1e-9, atol=1e-12), \
        "RS- + RS+ != RV identity failed on synthetic data"
    total_demeaned = v4_symmetric_vol(df)
    md = np.isfinite(total_raw) & np.isfinite(total_demeaned)
    rel_diff = np.abs(total_raw[md] - total_demeaned[md]) / np.maximum(total_demeaned[md], 1e-9)
    assert np.median(rel_diff) < 0.05, ".std() vs raw second-moment diverges more than disclosed"

    # (3) RSJ bounded in [-1, 1] where defined.
    rsj = relative_signed_jump(df)
    finite = rsj[np.isfinite(rsj)]
    assert len(finite) > 1000
    assert np.all(finite >= -1.0 - 1e-9) and np.all(finite <= 1.0 + 1e-9)

    # (4) causal truncation probes.
    assert causal_truncation_probe_vote(df)
    assert causal_truncation_probe_series(downside_ewm_vol, df)
    assert causal_truncation_probe_series(upside_ewm_vol, df)
    assert causal_truncation_probe_series(relative_signed_jump, df)
    assert causal_truncation_probe_series(v4_scale, df)

    # (5) r_squared sanity: identical series -> 1.0; unrelated -> not 1.0.
    assert abs(r_squared(down, down) - 1.0) < 1e-9
    assert r_squared(down, rng.normal(0, 1, len(down))) < 0.5

    # conditional_target_scale smoke test against v4_scale.
    assert np.allclose(v4_scale(df), conditional_target_scale(v4_symmetric_vol(df)))


_self_test()
