"""Shared, read-only utilities and pre-registration for the R-104 round (08-24).

DIRECTION, in one sentence: wrap `kelly_regime_v4`'s raw exposure (`frac *
scale`, before its own 10% deadband) in a **Conformal Risk Control**
discount -- a formal, distribution-free bound on the probability that a
single bar's realized loss under the current exposure exceeds a
pre-registered threshold -- calibrated once on a held-out split of
inner-train (conservative) or updated causally, bar by bar, by an online
feedback controller (novel).

**Literature grounding, fetched and read via WebSearch before either branch
was dispatched:**

- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024),
  "Conformal Risk Control", *ICLR 2024* (spotlight; arXiv:2208.02814;
  openreview.net/pdf?id=33XGfHLtZg). Generalizes split conformal
  prediction from controlling MISCOVERAGE specifically to controlling the
  expectation of any bounded, LAMBDA-MONOTONE loss function: choose the
  smallest lambda such that `(1/(n+1)) * (sum_i L_i(lambda) + B) <= alpha`
  on a calibration set, and the same finite-sample guarantee that split
  conformal has for coverage now holds for `E[L(lambda)] <= alpha` on a
  fresh exchangeable point, up to an O(1/n) correction. Their own worked
  examples (false negative rate, graph distance, token F1) all use this
  identical "shrink/grow a decision by lambda until an empirical average
  loss clears a target" recipe -- this round's use of it (shrink exposure
  by `d` until empirical loss-exceedance-rate clears a target `alpha`) is
  the direct application of Algorithm 1 to a single scalar bounded 0/1
  loss, not an extension of their method.
- Feldman, S., Bates, S., & Romano, Y. (2023), "Achieving Risk Control in
  Online Learning Settings", *TMLR* (arXiv:2205.09095), and the online
  variant sketched in Angelopoulos et al. (2024) Appendix D: replace the
  batch calibration with a scalar feedback controller,
  `lambda_{t+1} = lambda_t + eta * (L_t(lambda_t) - alpha)`, which achieves
  the SAME long-run risk-control target under arbitrary (non-exchangeable,
  even adversarial) sequences, with no distributional assumption at all --
  the same control-theoretic step Gibbs & Candes (2021) use for coverage in
  Adaptive Conformal Inference, applied here to a REALIZED LOSS rather than
  a miscoverage indicator. This is the novel branch's citation and
  mechanism.

**Which constraint this attacks: ERR**, primarily -- "no error control
anywhere in the signal path" is one of this project's four standing
constraints, and only two prior rounds have attacked it directly: R-28's
e-process (retired by R-31: the whole effect was an exposure-level
artifact, not a real gate) and R-87's Adaptive Conformal Inference (wrapped
around the VOTE's own confidence calibration and, separately, the Kelly
SCALE's dispersion estimator -- both COVERAGE-style constructions, i.e.
"is my confidence interval calibrated", NEGATIVE on the inner-validation
noise floor and ETH sign-replication). This round is the third ERR-axis
attempt and the first to control a REALIZED LOSS functional directly
(`P(bar loss > tau) <= alpha`) rather than calibrating a confidence
interval's coverage -- a different formal object (Angelopoulos et al. 2024
Section 1 state explicitly that risk control strictly generalizes coverage
control; miscoverage is the special case `L = 1{y not in C}`, which is what
R-87's ACI targeted). It also touches SIZE, since the mechanism's OUTPUT is
an exposure discount -- but unlike every one of the 26 prior SIZE-axis
attempts, the state variable driving it is neither an exogenous market
statistic (volatility level, drawdown, a decomposition of v4's own realized
variance) nor a resampling/robustness bound over the six historical stress
episodes; it is a formal, provable control on the strategy's OWN realized
outcome distribution, updated from every bar rather than six sparse events.

**Not a duplicate of:**
- R-28 (e-process): a sequential LIKELIHOOD-RATIO test for whether the
  trend signal is currently profitable at all (a hypothesis test on the
  EDGE), retired because its whole effect was an exposure-level artifact
  (R-31). This round does not test whether the signal is informative; it
  bounds a LOSS probability of whatever exposure the signal already
  produces, unconditionally on whether the signal is "working".
- R-87 (Adaptive Conformal Inference on vote confidence / Kelly-scale
  dispersion): controls MISCOVERAGE of a confidence interval (`P(vote
  confidence's 90% interval fails to cover) <= 10%`, or equivalently for
  the scale's dispersion estimate) -- the finite-sample guarantee target is
  "my uncertainty band is calibrated". This round's guarantee target is "my
  bar-level realized loss stays below a fixed dollar-equivalent threshold
  at a fixed rate" -- a directly economic, not epistemic, quantity, and the
  mathematical object generalizes R-87's (per Angelopoulos et al. 2024's
  own framing: coverage is risk control's special case, not the reverse).
- R-97 (Wasserstein-DRO keyed on regime-cycle count) / R-101 (delete-one
  jackknife over six stress episodes): both derive a confidence/ambiguity
  multiplier from the SAME six discrete historical episodes. This round
  reads no episode calendar at all -- every one of ~1M+ bars supplies one
  loss observation, continuously, both for the static calibration split
  and for the online controller's every-bar update.
- R-99/R-102 (bipower-variation jump split; downside/upside semivariance;
  RSJ discount, R-102/R-103): all replace or discount `scale`'s INPUT
  STATISTIC (a different point estimate, or a different weighting, of
  realized variance itself). This round never modifies `scale`'s
  volatility input; it multiplies the FINISHED `frac * scale` product by a
  separately-calibrated discount, exactly as R-102's novel overlay
  architecture did -- but the discount here comes from a formal
  loss-control guarantee on realized P&L, not from a signed-jump-asymmetry
  regression.
- The fifteen INFO-axis rounds: no new external data channel. Both
  branches read only the already-committed OHLCV `close` column, at the
  identical cadence v4 itself uses.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both. Nothing here reads a bar at or after OOS_START
(2023-01-01); `compare()` asserts this explicitly for every slice it runs.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists:
(1) **The familiar SIZE-axis collapse.** If the calibrated/online discount
`d_t` sits at a near-constant value once the calibration/burn-in period
ends, this is the 27th SIZE-axis construction to collapse to "a
near-constant rescale of v4's own exposure" (kill switch A2: R^2 of the
candidate's raw exposure path against v4's own must be < 0.98, matching
every prior round's convention). This is a live risk: v4's own bar-level
loss distribution may simply be stationary enough on inner-train that
`tau` calibrated from it never gets seriously threatened later, in which
case CRC has nothing to react to. (2) **The guarantee not binding.** If the
static calibration's chosen `d` is 0 (the unmodified exposure already
clears the target risk rate with room to spare), the conservative branch is
inert by construction -- a valid, informative negative, not a bug. (3) **A0
measurement gate.** If the online controller's empirical exceedance rate on
inner-validation does not track `alpha` (i.e. the controller is not doing
what conformal risk control claims), no Sharpe/drawdown number is
meaningful and the round stops there, exactly as R-79/R-84/R-88/R-100's
Step-A gates stopped those rounds before any strategy-level number was
read.
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

# ------------------------------------------------------- CRC pre-registration
# Target risk rate and calibration/burn-in split, fixed BEFORE either branch
# reads a real-data loss number.
CRC_ALPHA = 0.01              # target: <=1% of bars exceed the loss threshold
CRC_TAU_QUANTILE = 0.99       # tau = v4's own 99th pct single-bar loss on the calibration split
CRC_CAL_FRAC = 0.5            # first half of inner-train = calibration; second half = check
CRC_D_GRID = tuple(round(x, 3) for x in np.arange(0.0, 0.951, 0.05))  # static branch's search grid
CRC_ONLINE_ETA = 0.02         # online controller's step size (Feldman-Bates-Romano-style)
CRC_D_MAX = 0.95              # discount is capped short of fully-flat (avoid a degenerate d=1 solution)


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
# construction, reproduced EXACTLY. Same convention as r89-r103_shared.py.
# ==================================================================

def _latched_anchor_vote(close: pd.Series, days: int, band: float = V4_BAND) -> pd.Series:
    anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
    v = pd.Series(
        np.where(close > anchor * (1.0 + band), 1.0,
                 np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
        index=close.index,
    )
    return v.ffill().fillna(0.0)


def vote_frac(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
             band: float = V4_BAND, vote_gamma: float = 1.0) -> pd.Series:
    close = df["close"]
    votes = [_latched_anchor_vote(close, days, band) for days in horizons]
    frac = sum(votes) / len(votes)
    if vote_gamma != 1.0:
        frac = frac ** vote_gamma
    return frac


def v4_vote_frac(df: pd.DataFrame) -> pd.Series:
    return vote_frac(df, V4_HORIZONS, V4_BAND)


def v4_symmetric_vol(df: pd.DataFrame, span: int = V4_VOL_SPAN) -> np.ndarray:
    r = np.log(df["close"]).diff()
    return (r.ewm(span=span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def conditional_target_scale(vol: np.ndarray, anchor_span_days: int = V4_ANCHOR_SPAN_DAYS,
                              high_in: float = V4_HIGH_IN, high_out: float = V4_HIGH_OUT,
                              low_in: float = V4_LOW_IN, low_out: float = V4_LOW_OUT,
                              target_vol: float = V4_TARGET_VOL,
                              max_leverage: float = V4_MAX_LEVERAGE) -> np.ndarray:
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
    return conditional_target_scale(v4_symmetric_vol(df))


def apply_deadband(desired: np.ndarray, deadband: float = V4_DEADBAND) -> np.ndarray:
    target = np.zeros(len(desired))
    pos = 0.0
    for i, d in enumerate(desired):
        if abs(d - pos) > deadband:
            pos = float(d)
        target[i] = pos
    return target


def v4_raw_desired(df: pd.DataFrame) -> np.ndarray:
    """v4's desired exposure BEFORE its own 10% deadband: frac * scale. Always >= 0
    (frac in [0,1], scale >= 0) -- v4 never shorts, so every loss-relevant bar
    has exposure >= 0 and a loss occurs exactly when the forward return is
    negative."""
    return v4_vote_frac(df).to_numpy() * v4_scale(df)


def v4_target(df: pd.DataFrame) -> np.ndarray:
    return apply_deadband(v4_raw_desired(df))


# ================================================================== (2)
# Conformal Risk Control machinery.
#
# Bar-level loss: for exposure e_i decided using information available at
# bar i (frac[i]*scale[i], both causal), the realized outcome is r[i+1] =
# log(close[i+1]/close[i]) (the SAME .diff()/.shift(1) convention v4_scale
# itself uses to stay causal). loss[i+1] = max(0, -e[i] * r[i+1]) -- exactly
# 0 whenever e[i] <= 0 (v4 is never short) or the bar was flat/up.
# ==================================================================

def bar_forward_loss(exposure: np.ndarray, close: pd.Series) -> np.ndarray:
    """loss[i] = max(0, -exposure[i-1] * r[i]), r[i] = log(close[i]/close[i-1]).
    loss[0] = 0 (no prior exposure). Causal: loss[i] uses only exposure[i-1]
    (itself a causal function of bars <= i-1) and the realized bar-i return,
    which is exactly the information available once bar i closes -- the same
    timing every other strategy's fill and every prior round's forward-loss
    probe (R-96, R-98, R-99) uses."""
    exposure = np.asarray(exposure, dtype=float)
    r = np.log(close).diff().to_numpy()
    loss = np.zeros(len(exposure))
    loss[1:] = np.maximum(0.0, -exposure[:-1] * r[1:])
    return np.nan_to_num(loss, nan=0.0)


def calibrate_tau(df: pd.DataFrame, cal_end: str, q: float = CRC_TAU_QUANTILE) -> float:
    """tau = the q-th percentile of v4's OWN unmodified bar loss, measured
    only on bars strictly before `cal_end` -- fixed once, before either
    branch's discount is computed, exactly as this row's Step-1 filing
    requires ("name the threshold now, before any code that reacts to it")."""
    e = v4_raw_desired(df)
    loss = bar_forward_loss(e, df["close"])
    mask = df.index < pd.Timestamp(cal_end, tz="UTC")
    cal_loss = loss[mask.to_numpy()]
    assert len(cal_loss) > BARS_PER_DAY * 30, "calibration window too short"
    return float(np.quantile(cal_loss, q))


def crc_static_lambda(loss_fn, grid=CRC_D_GRID, alpha: float = CRC_ALPHA) -> float:
    """Angelopoulos et al. (2024) Algorithm 1, literally: smallest d in
    `grid` (loss is non-increasing in d by construction: bigger discount ->
    smaller exposure -> smaller-or-equal |loss| every bar) such that the
    finite-sample-corrected empirical risk on the calibration set clears
    alpha. `loss_fn(d)` returns the calibration-set 0/1 exceedance array for
    discount `d`. B=1 (the loss is a 0/1 indicator, bounded)."""
    for d in grid:
        L = loss_fn(d)
        n = len(L)
        risk_hat = (float(np.sum(L)) + 1.0) / (n + 1.0)
        if risk_hat <= alpha:
            return float(d)
    return float(grid[-1])


def crc_online_lambda_path(exceed_indicator_at_d0: np.ndarray, exposure_d0: np.ndarray,
                           close: pd.Series, alpha: float = CRC_ALPHA,
                           eta: float = CRC_ONLINE_ETA, d_max: float = CRC_D_MAX,
                           d0: float = 0.0) -> np.ndarray:
    """Feldman-Bates-Romano (2023) / Angelopoulos et al. (2024, App. D)
    online risk-control controller, applied causally bar by bar over the
    WHOLE series (inner-train then inner-validation in one continuous pass,
    standard online-learning convention -- no distributional assumption, no
    calibration/check split needed, decisions strictly use only bars <
    current index):

        d_{t+1} = clip(d_t + eta * (L_t(d_t) - alpha), 0, d_max)

    where L_t(d_t) is the REALIZED (not d0-baseline) loss indicator under
    the discount actually in force at t. This function recomputes the loss
    at the CURRENT d_t each step (not the passed-in d0 baseline), because
    the loss is only informative about the discount actually applied."""
    e0 = np.asarray(exposure_d0, dtype=float)
    n = len(e0)
    d_path = np.zeros(n)
    d = float(d0)
    tau = _ONLINE_TAU[0]
    r = np.log(close).diff().to_numpy()
    r = np.nan_to_num(r, nan=0.0)
    for i in range(n):
        d_path[i] = d
        if i >= 1:
            loss_i = max(0.0, -e0[i - 1] * (1.0 - d_path[i - 1]) * r[i])
            L = 1.0 if loss_i > tau else 0.0
            d = float(np.clip(d + eta * (L - alpha), 0.0, d_max))
    return d_path


# module-level mutable cell so crc_online_lambda_path can read a tau set by
# the caller without changing its own signature (kept private; branches call
# `set_online_tau` before `crc_online_lambda_path`).
_ONLINE_TAU = [0.0]


def set_online_tau(tau: float) -> None:
    _ONLINE_TAU[0] = float(tau)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of `a` against `b` -- the A2 non-inertness kill switch every
    SIZE-axis round since R-89 has used: is the candidate's exposure path a
    near-exact rescale of v4's own?"""
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


# ------------------------------------------------------- causal truncation

def causal_truncation_probe_series(build_fn, df: pd.DataFrame,
                                   cuts: tuple[float, ...] = (0.35, 0.55, 0.80)) -> bool:
    full = np.asarray(build_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        if k < BARS_PER_DAY * 2:
            continue
        part = np.asarray(build_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-9, rtol=1e-8):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-9, rtol=1e-8)))
            raise AssertionError(f"{build_fn.__name__} causality FAIL at cut={cut}: {bad} bars differ")
        perturbed = df.copy()
        tail = perturbed.iloc[k:].copy()
        for col in ("open", "high", "low", "close"):
            if col in tail.columns:
                tail[col] = tail[col] * 3.7 + 1.0
        perturbed.iloc[k:] = tail
        pert = np.asarray(build_fn(perturbed), dtype=float)
        pm = np.isfinite(a) & np.isfinite(pert[:k])
        if not np.allclose(a[pm], pert[:k][pm], atol=1e-9, rtol=1e-8):
            raise AssertionError(f"{build_fn.__name__} peeks at bar>=k, cut={cut}")
    return True


# ================================================================== (3)
# compare(): run any pure `build_target(df) -> np.ndarray` candidate over
# inner-train, inner-validation and the ETH replication slice, vs
# kelly_regime_v4, never touching OOS_START. Same convention as r89-r103.
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

    name = "r104_control"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r104_control",
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

    cand = TargetStrategy(candidate_build, name=f"r104_{label}")
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


def exceedance_rate(exposure: np.ndarray, close: pd.Series, tau: float) -> float:
    """Empirical P(bar loss > tau) under `exposure` -- the quantity CRC
    claims to control at `<= CRC_ALPHA` (+ O(1/n)). Used as the A0
    measurement gate."""
    loss = bar_forward_loss(exposure, close)
    return float(np.mean(loss > tau))


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(104)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    # (1) v4 reproduction self-consistency.
    raw = v4_raw_desired(df)
    assert np.allclose(v4_target(df), apply_deadband(raw))
    assert vote_frac(df).between(0.0, 1.0).all()
    assert np.all(raw >= -1e-12), "v4 raw exposure must never be negative"

    # (2) bar_forward_loss: zero exposure -> zero loss; loss only on down bars
    # while long; loss[0] == 0.
    e_flat = np.zeros(len(df))
    assert np.allclose(bar_forward_loss(e_flat, df["close"]), 0.0)
    e_full = np.ones(len(df))
    loss_full = bar_forward_loss(e_full, df["close"])
    r = np.log(df["close"]).diff().to_numpy()
    assert loss_full[0] == 0.0
    up_mask = np.isfinite(r) & (r > 0)
    assert np.allclose(loss_full[1:][up_mask[1:]], 0.0)
    down_mask = np.isfinite(r) & (r < 0)
    assert np.allclose(loss_full[1:][down_mask[1:]], -r[1:][down_mask[1:]])

    # (3) crc_static_lambda: monotone synthetic loss_fn -> returns smallest d
    # clearing alpha, and is non-increasing in the target risk hardness.
    def fake_loss_fn(d, base_rate=0.05):
        n = 10_000
        k = int(round(base_rate * (1 - d) * n))
        arr = np.zeros(n)
        arr[:k] = 1.0
        return arr
    d_easy = crc_static_lambda(lambda d: fake_loss_fn(d, base_rate=0.05), alpha=0.06)
    assert d_easy == 0.0, "already clears alpha at d=0, should not discount"
    d_hard = crc_static_lambda(lambda d: fake_loss_fn(d, base_rate=0.05), alpha=0.01)
    assert d_hard > 0.0, "must discount to clear a stricter alpha"

    # (4) crc_online_lambda_path: causal (perturbing the tail must not move
    # the path's shared prefix) and stays within [0, CRC_D_MAX].
    set_online_tau(np.quantile(loss_full[loss_full > 0], 0.99))
    path = crc_online_lambda_path(None, e_full, df["close"])
    assert np.all(path >= 0.0) and np.all(path <= CRC_D_MAX)
    k = 40_000
    path_part = crc_online_lambda_path(None, e_full[:k], df["close"].iloc[:k])
    assert np.allclose(path[:k], path_part, atol=1e-12), "online path is not causal"

    # (5) r_squared sanity.
    assert abs(r_squared(raw, raw) - 1.0) < 1e-9
    assert r_squared(raw, rng.normal(0, 1, len(raw))) < 0.5

    # (6) causal truncation probes on the pieces branches will reuse directly.
    assert causal_truncation_probe_series(v4_raw_desired, df)
    assert causal_truncation_probe_series(v4_scale, df)


_self_test()
