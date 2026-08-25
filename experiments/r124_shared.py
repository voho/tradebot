"""Shared, read-only utilities and pre-registration for the R-124 round (08-25).

DIRECTION, in one sentence: this project's every prior construction has fed
`kelly_regime_v4`'s vote/detectors either the raw price LEVEL (non-stationary,
full memory -- v4's own rolling-mean anchors) or plain log-RETURNS (d=1,
stationary, but memory of the price level destroyed on every difference) --
FIXED-WINDOW FRACTIONAL DIFFERENTIATION (Lopez de Prado 2018, *Advances in
Financial Machine Learning*, Wiley, Ch. 5; the operator (1-L)^d generalized
to fractional d by Hosking 1981, *Biometrika* 68(1), 165-176) is a genuinely
different point on that same axis: a fractional order 0<d<1, chosen as the
MINIMUM d that a causal, in-sample-only stationarity test accepts, keeps the
series stationary while preserving materially more of its own long memory
than d=1 discards. This round asks whether that richer-but-still-stationary
input closes either (a) a SIZE-axis vote construction fed by it instead of
raw close [conservative], or (b) an eleventh regime-timing detection
mechanism built on its own threshold crossings [novel] -- judged by the
identical B1-B5 promotion bar / Step-A detection-lag gate every prior round
in each family used.

**Literature grounding, fetched via WebSearch this round:**

- Lopez de Prado, M. (2018), *Advances in Financial Machine Learning*,
  Wiley, Chapter 5 ("Fractionally Differentiated Features"). The direct
  motivating citation and the source of the Fixed-Window FFD construction
  used below (binomial Grunwald-Letnikov weights, truncated once |w_k|
  falls below a fixed threshold, applied as a rolling dot product -- causal
  by construction). His own stated finding, confirmed by this round's own
  WebSearch: raw prices (d=0) are non-stationary; returns (d=1) are
  stationary but "memory-less"; a fractional d strictly between 0 and 1
  typically exists that is stationary while retaining significant
  correlation with the original series -- motivating, not load-bearing:
  his own examples are equity/futures daily series and an unrelated ML
  pipeline (a classifier), not this project's 5-minute BTC bars, cost
  model or vote architecture.
- Hosking, J. R. M. (1981), "Fractional Differencing," *Biometrika* 68(1),
  165-176. The original derivation of the fractional-differencing operator
  and its binomial-series weight expansion this round's `ffd_weights`
  implements directly (ARFIMA long-memory literature).
- Chen, Y., Liu, Y., et al. (2025), "Unlocking Multifractal and Long-Memory
  Dynamics in Cryptocurrency Markets," *Fractal and Fractional* 10(6):379 --
  confirmed live via WebSearch: applies a Grunwald-Letnikov fractional
  memory operator to BTC/ETH/BNB daily data (2018-2025) inside an
  LSTM-N-BEATS forecasting model. Cited only as evidence that fractional
  memory operators are an active, currently-published line of crypto
  research -- their downstream model (a deep forecaster on daily bars) is
  unrelated to this project's vote/detector architecture and 5-minute
  cadence, so nothing about their reported performance transfers here;
  both branches below re-derive `d` and re-measure everything from scratch
  on this project's own data, promotion bar and cost model.

**Which constraint each branch attacks:**

- Conservative: SIZE (a 27th+ construction in that family, but -- like
  R-105's anchor-ladder ensemble and R-117's Donchian ensemble --
  substituting a structurally new INPUT REPRESENTATION into the vote's own
  slot, not a new detector family and not a retuned parameter of the
  shipped mean-anchor family. `scale` is left completely untouched, per
  R-62's finding that it carries none of v4's matched-exposure signature).
- Novel: regime-timing (an ELEVENTH structurally distinct formal estimator
  of "has a known historical regime break just happened", judged by the
  identical Step-A detection-lag gate R-82 through R-117 used -- reused
  verbatim, not re-derived, for direct comparability).

**Not a duplicate of:**

- R-105 (anchor-ladder ensemble) / R-117 conservative-and-novel (Donchian
  breakout ensemble): both vary the DETECTOR (mean-crossing lookback, or
  detector family itself: mean-crossing vs. range-breakout) while feeding
  it the SAME raw-close input series v4 always has. This round holds the
  detector's FUNCTIONAL FORM fixed (a latched band/threshold-crossing vote,
  structurally identical to v4's own) and instead varies the INPUT SERIES
  itself -- a third, orthogonal axis of variation neither prior round
  touched (confirmed by grep: no prior "fractional diff", "frac_diff",
  "fracdiff", "grunwald", or "hosking" mention anywhere in this file).
- R-46/R-61 (Hurst exponent gating): a rolling estimate of the SAME raw
  price series' self-similarity/persistence exponent, used to GATE trading
  on/off (trade only while H<0.5). Fractional differentiation does not
  estimate a persistence exponent for gating -- it TRANSFORMS the input
  series itself into a new, stationary-but-memory-preserving series that a
  vote or detector is then built on top of; the two constructions share no
  code and answer different questions (Hurst: "is the series persistent
  right now?" vs. FFD: "what is the minimum amount of differencing that
  makes this series usable without destroying its memory?").
- R-01(HMM)/R-82(BOCPD)/R-83(Kalman LLT)/R-85(CSD)/R-86(transfer entropy)/
  R-96(Hawkes)/R-98(POT/GPD)/R-84(vote-latch/volume)/R-60(CUSUM)/R-117
  (Donchian): ten regime-timing mechanisms, all state-space, point-process,
  information-theoretic, extreme-value or price-range estimators computed
  on the RAW price series (level or simple returns). None applies a
  fractional-order differencing PREPROCESSING step before detection; this
  round's novel branch is a threshold-crossing detector on a genuinely
  different (fractionally-differenced) input series, not a twelfth variant
  of any of the ten mechanism FAMILIES already tried.
- R-04/R-80 (meta-labeling + triple-barrier, Lopez de Prado 2018's OTHER
  chapter): a secondary classifier deciding whether to act on the primary
  vote's signal, trained on triple-barrier labels. Unrelated machinery --
  no classifier, no labeling, no barriers anywhere in this round; this is
  purely a deterministic, parameter-free (once `d` is fixed) input
  transform.

Confirmed by grep of docs/LEDGER.md before this file was written: no prior
round mentions "fractional diff", "fracdiff", "frac_diff", "grunwald", or
"hosking" in any form.

**Is it simulable here?** Yes, zero new data. Fractional differentiation of
`log(close)` needs only the close series every registered strategy already
reads. Fully causal by construction: `causal_ffd` is a finite-impulse-
response filter over strictly-PAST bars (weight `w_k` multiplies `x[t-k]`
for `k=0..window-1`; bar `t`'s own future is never referenced), verified by
this file's own truncation self-test below.

**What would make each branch fail, named now, before any code ran:**

- Conservative (SIZE-axis substitution): a fractionally-differenced input
  is, by construction, SMOOTHER than raw returns but NOISIER (more
  mean-reverting, faster-crossing) than a slow 20/40/80-day rolling-mean
  anchor -- v4's own anchors are already a heavy low-pass filter, and nothing
  about the FFD construction adds trend information the anchors do not
  already extract; the most likely outcome, named before any bar was read,
  is the "real but inert" (or real-and-worse, per R-117's own precedent for
  a first-time-tried input/detector swap) pattern this project has seen
  repeatedly. A clean NEGATIVE on B1 (inner-validation) or B4 (ETH
  falsification) is the fully expected, fully successful outcome.
- Novel (regime-timing gate): a threshold crossing on a stationarized series
  reacts to LOCAL deviations, which is a fundamentally noisier, more
  frequent event than v4's own slow 1%-past-a-180-bar-mean crossings -- the
  same "structurally more reactive but not necessarily more USEFUL" tension
  R-117's Donchian novel branch hit. The named, pre-registered expectation
  is an eleventh Step-A gate failure, most likely via FALSE-POSITIVE
  over-triggering (many small crossings unrelated to the six historical
  stress episodes) rather than the pure-lag failure mode of the ten mean-
  based/range-based predecessors -- a qualitatively different failure
  signature worth distinguishing from "just another slow detector," even
  though the pre-registered STOP rule (below) treats both failure shapes
  identically as NEGATIVE.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it. The `d` selection below
touches ONLY `INNER_TRAIN_START..INNER_TRAIN_END` (2017-01-01 to
2020-12-31); nothing in this file reads a bar at or after OOS_START
(2023-01-01), enforced by `assert_no_holdout` inherited unmodified from the
r102..r117 chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from the r102..r117 chain: identical control
# machinery (compare(), the B1-B5 promotion bar, TargetStrategy, causal
# probes, fee tiers, data loaders) so every number this round produces is
# directly comparable to R-101 through R-117's own.
from experiments.r117_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BIND_FRAC_THRESH,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    R2_THRESH,
    SHARPE_NOISE_FLOOR,
    SLICES,
    SPOT,
    STEP0_FLOOR_GRID,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    causal_truncation_probe_series,
    compare,
    fee_at,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    paired_diff,
    print_plateau_table,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)
from experiments.r102_shared import V4_BAND, V4_HORIZONS, vote_frac  # noqa: E402,F401
from experiments.r82_shared import (  # noqa: E402,F401
    STRESS_EPISODES,
    anchor_majority,
    block_bootstrap_shifts,
    episode_window,
    nearest_transition,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

# ------------------------------------------------------------------------
# (1) Fixed-Window Fractional Differentiation -- causal by construction.
# ------------------------------------------------------------------------

FFD_THRESH = 1e-4                       # weight-truncation threshold (Lopez de Prado's own convention)
FFD_MAX_WINDOW = 60 * BARS_PER_DAY      # hard cap: 60 days of 5-minute bars
D_GRID = tuple(round(0.05 + 0.05 * i, 2) for i in range(19))   # 0.05 .. 0.95, step 0.05
ADF_LAGS = 5                            # augmented lags in the causal ADF-lite regression
ADF_CRIT_5PCT = -2.86                   # fixed asymptotic 5% critical value (constant, no trend);
                                         # a disclosed simplification -- no MacKinnon small-sample
                                         # correction is applied. Used only to SELECT d, not to
                                         # assert a formal hypothesis-test claim.


def ffd_weights(d: float, thresh: float = FFD_THRESH, max_window: int = FFD_MAX_WINDOW) -> np.ndarray:
    """Grunwald-Letnikov binomial weights for fractional order `d`, truncated
    once |w_k| < thresh (or at max_window, whichever comes first).

    w_0 = 1; w_k = -w_{k-1} * (d - k + 1) / k. w[0] is the weight on the
    CURRENT bar; w[k] is the weight on the bar `k` steps in the past.
    """
    w = [1.0]
    k = 1
    while k < max_window:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return np.array(w, dtype=float)


def causal_ffd(x: np.ndarray, d: float, thresh: float = FFD_THRESH,
               max_window: int = FFD_MAX_WINDOW) -> np.ndarray:
    """Fixed-window fractionally-differenced series, causal by construction.

    out[t] = sum_{k=0}^{W-1} w_k * x[t-k], computed via FFT convolution for
    O(n log n) speed (W can run to thousands of bars). out[t] is NaN for
    t < W-1 (insufficient history), exactly analogous to v4's own rolling-
    mean anchors returning NaN during warmup.

    Causal by construction: out[t] is a fixed linear combination of
    x[t], x[t-1], ..., x[t-W+1] only -- x[t+1:] never enters the sum for
    any t. Verified empirically by `_self_test`'s truncation probe below.
    """
    x = np.asarray(x, dtype=float)
    w = ffd_weights(d, thresh, max_window)
    n, m = len(x), len(w)
    if n == 0:
        return np.array([])
    x_filled = np.where(np.isfinite(x), x, 0.0)
    size = 1
    total = n + m - 1
    while size < total:
        size *= 2
    xf = np.fft.rfft(x_filled, size)
    wf = np.fft.rfft(w, size)
    full = np.fft.irfft(xf * wf, size)[:n]
    out = full.copy()
    out[: m - 1] = np.nan
    out[~np.isfinite(x)] = np.nan
    return out


def _adf_lite_tstat(y: np.ndarray, lags: int = ADF_LAGS) -> float:
    """A minimal, disclosed-as-simplified Augmented Dickey-Fuller regression
    (no statsmodels dependency in this project): regress dy_t on
    [1, y_{t-1}, dy_{t-1}, ..., dy_{t-lags}] by OLS; return the t-statistic
    on the y_{t-1} coefficient. More negative = stronger rejection of a
    unit root (non-stationarity). Used only as a SELECTION criterion across
    the `D_GRID` sweep below, against a fixed asymptotic critical value --
    not offered as a formal per-cell hypothesis-test p-value.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    dy = np.diff(y)
    n = len(dy) - lags
    if n < 50:
        return float("nan")
    cols = [np.ones(n), y[lags:-1]]
    for lag in range(1, lags + 1):
        cols.append(dy[lags - lag: len(dy) - lag])
    X = np.column_stack(cols)
    target = dy[lags:]
    beta, _res, _rank, _sv = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = n - X.shape[1]
    if dof <= 0:
        return float("nan")
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(sigma2 * xtx_inv[1, 1])
    if se <= 0 or not np.isfinite(se):
        return float("nan")
    return float(beta[1] / se)


def select_d_causal(log_close_inner_train: np.ndarray, grid: tuple[float, ...] = D_GRID,
                    crit: float = ADF_CRIT_5PCT) -> tuple[float, list[dict]]:
    """Minimum d in `grid` whose FFD series (resampled to one obs/day, the
    literature's own convention for an ADF check) rejects a unit root at
    `crit` -- Lopez de Prado's own recommended selection rule ("pick the
    minimum d on the plot that crosses the critical-value line"). Computed
    ONCE, on inner-train data ONLY, then frozen for both branches.

    Returns (chosen_d, full_grid_diagnostics) -- the diagnostics are printed
    by the operator before dispatch so the choice is auditable, not just
    asserted.
    """
    rows = []
    chosen = None
    n = len(log_close_inner_train)
    daily_idx = np.arange(BARS_PER_DAY - 1, n, BARS_PER_DAY)  # one obs/day, causal (last bar of day)
    for d in grid:
        series = causal_ffd(log_close_inner_train, d)
        daily = series[daily_idx]
        t = _adf_lite_tstat(daily)
        corr = float(np.corrcoef(
            np.nan_to_num(series, nan=0.0)[np.isfinite(series) & np.isfinite(log_close_inner_train)],
            log_close_inner_train[np.isfinite(series) & np.isfinite(log_close_inner_train)],
        )[0, 1]) if np.isfinite(series).sum() > 100 else float("nan")
        stationary = np.isfinite(t) and t < crit
        rows.append(dict(d=d, adf_t=t, stationary=stationary, corr_with_level=corr,
                         window=len(ffd_weights(d))))
        if stationary and chosen is None:
            chosen = d
    if chosen is None:
        chosen = 1.0  # fall back to plain returns if nothing on the grid passes
    return chosen, rows


# ------------------------------------------------------------------------
# (2) Selection, run once here (inner-train only), frozen for both branches.
# ------------------------------------------------------------------------

_btc_inner_train = load_btc()
_btc_inner_train = _btc_inner_train[
    (_btc_inner_train.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC"))
    & (_btc_inner_train.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))
]
assert_no_holdout(_btc_inner_train, "r124_shared: d-selection frame")
_log_close_inner_train = np.log(_btc_inner_train["close"].to_numpy())

FFD_D, FFD_D_GRID_DIAGNOSTICS = select_d_causal(_log_close_inner_train)


def causal_ffd_log_close(df: pd.DataFrame, d: float = FFD_D) -> pd.Series:
    """The one input transform both branches build on: FFD at the frozen
    `d`, applied to `log(close)` over whatever frame is passed (each
    branch is responsible for calling this only on pre-OOS frames, exactly
    like every other primitive in this file)."""
    return pd.Series(causal_ffd(np.log(df["close"].to_numpy()), d), index=df.index)


# --------------------------------------------------------------------- self-test

def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(124)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                       "close": close, "volume": 1.0}, index=idx)

    # (a) d=0 recovers the input exactly (once warmed up: window=1).
    w0 = ffd_weights(0.0)
    assert len(w0) == 1 and np.isclose(w0[0], 1.0)
    out0 = causal_ffd(close, 0.0)
    assert np.allclose(out0, close)

    # (b) d=1 recovers a first difference (weights [1, -1]).
    w1 = ffd_weights(1.0)
    assert np.allclose(w1[:2], [1.0, -1.0])

    # (c) causality: truncating the future must not change any past output.
    log_close = np.log(close)
    check_at = 45_000
    for d in (0.2, 0.4, 0.6):
        full = causal_ffd(log_close, d)
        short = causal_ffd(log_close[: check_at + 5_000], d)
        a, b = full[check_at], short[check_at]
        assert np.isclose(a, b, equal_nan=True), f"causal_ffd not causal at d={d}"

    # (d) weight decay: larger d truncates to a SHORTER window (faster decay).
    assert len(ffd_weights(0.1)) > len(ffd_weights(0.9))

    # (e) ADF-lite sanity: a random walk should NOT reject; white noise SHOULD.
    rw = np.cumsum(rng.normal(0, 1, 5000))
    wn = rng.normal(0, 1, 5000)
    t_rw = _adf_lite_tstat(rw)
    t_wn = _adf_lite_tstat(wn)
    assert t_wn < t_rw, "white noise should reject a unit root more strongly than a random walk"
    assert t_wn < ADF_CRIT_5PCT, "white noise should clear the 5% ADF-lite critical value"

    # (f) causal_ffd_log_close wraps causal_ffd identically.
    s = causal_ffd_log_close(df, d=0.3)
    assert np.allclose(s.to_numpy(), causal_ffd(np.log(close), 0.3), equal_nan=True)

    # (g) causal_truncation_probe_series (project's own generic probe) also
    # accepts this transform, so future rounds' automated checks work on it.
    assert causal_truncation_probe_series(lambda d_: causal_ffd_log_close(d_, d=FFD_D).to_numpy(), df)


_self_test()

if __name__ == "__main__":
    hr("R-124 shared: causal d-selection (inner-train only, 2017-2020)")
    print(f"{'d':>5s} {'adf_t':>8s} {'stationary':>11s} {'corr_w_level':>13s} {'window':>7s}")
    for row in FFD_D_GRID_DIAGNOSTICS:
        print(f"{row['d']:5.2f} {row['adf_t']:8.2f} {str(row['stationary']):>11s} "
              f"{row['corr_with_level']:13.3f} {row['window']:7d}")
    print(f"\nFrozen FFD_D = {FFD_D}")
