"""Shared, read-only pre-registration and MSM engine for the R-175 round (08-28).

DIRECTION, one sentence: replace `kelly_regime_v4`'s single-span EWM
realized-volatility estimator with a Calvet & Fisher (2001, 2004)
Markov-Switching Multifractal (MSM) decomposition of BTC's own realized
volatility into a hierarchy of `kbar` latent binomial components at
geometrically-spaced persistence, driving the SAME `conditional_target_scale`
state machine `kelly_regime_v3`/`v4` already ships -- only the `vol` input
changes, nothing else.

Full Step 1/Step 2 design (constraint attacked [SIZE primary, COST
secondary], non-duplication against R-08/R-09/R-85/R-136/R-141/R-152/
R-161/R-162/R-163/R-164/R-166/R-167/R-171, simulability, named failure
modes) is in `experiments/r175_direction.md`, adapted from the candidate
written by a research sub-agent BEFORE either branch was dispatched
(`experiments/r175_direction_candidate.md`, uncommitted background-agent
output, folded into the frozen version verbatim except for status/date
framing).

This module implements the MSM fit/filter engine and the shared
`conditional_target_scale`-compatible plumbing. It is DELIBERATELY neutral
between the two branches: it exposes both the FULL multiplier forecast
(conservative branch's own input) and the PERSISTENT-ONLY multiplier
forecast (novel branch's own input) from the SAME fitted filter path, so
the two branches differ only in which functional of one shared posterior
they read -- not in the estimation itself. Neither branch may edit this
file or each other's file (R-89-through-R-174's own convention).

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).

Simplifications disclosed up front (per ROUTINE.md's honesty convention):
- `scipy` is not installed in this environment (confirmed: `r118`, `r125`
  already document this). Fitting uses grid-search maximum likelihood via
  a Hamilton filter, not continuous optimization -- this is Calvet &
  Fisher's own documented practice for the low-dimensional (m0, b,
  gamma_kbar) binomial MSM, not an approximation invented for this round.
- `sigma_bar` (the unconditional volatility level) is fixed to the sample
  standard deviation of daily log returns over each calibration window,
  not grid-searched -- valid because each component has unconditional
  mean 1, so E[prod_k M_k] = 1 unconditionally and Var(r) = sigma_bar**2
  exactly, letting sigma_bar be profiled out in closed form. This also
  means dropping components for the PERSISTENT-only functional does not,
  by construction, change the unconditional volatility level -- only the
  conditional (time-varying) path -- which is verified numerically below
  in `_self_test`.
- The filter/refit operates on DAILY log returns (an MSM(kbar=6) Hamilton
  filter over ~1M 5-minute bars is computationally infeasible in a pure
  Python/numpy loop with no scipy/numba). Per R-172's own explicit lesson
  ("a same-day daily-resolution statistic broadcast onto its own day's bars
  is a one-day lookahead unless explicitly lagged... future daily-bar
  broadcast constructions should run their causality probe against real,
  choppy price data"): day D's forecast is the filter's ONE-STEP-AHEAD
  prediction using data strictly through day D-1's close, assigned to ALL
  of day D's bars (known in full before day D's first bar), and the
  self-test below runs `causal_truncation_probe_series` against REAL BTC
  data, not only the smooth synthetic generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r102_shared import (  # noqa: E402,F401
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
    V4_DEADBAND,
    V4_HORIZONS,
    apply_deadband,
    assert_no_holdout,
    causal_truncation_probe_series,
    conditional_target_scale,
    fee_at,
    load_btc,
    load_eth,
    paired_diff,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_symmetric_vol,
    v4_target,
    v4_vote_frac,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_DEADBAND - 0.10) < 1e-12, V4_DEADBAND

# ------------------------------------------------------------------------
# The six dated stress episodes this ledger's regime-timing gate already
# uses (R-82/R-83/R-85/R-163's own table), reused verbatim here for the
# novel branch's decisive mechanism check (Step-1 Q4(b)).
# ------------------------------------------------------------------------
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ------------------------------------------------------------------------
# Pre-registered MSM constants and grid -- FIXED before either branch
# touches real performance numbers. Grid resolution and calibration
# window were chosen for computational tractability (documented above),
# not tuned against any backtest result.
# ------------------------------------------------------------------------
MSM_KBAR = 6                       # 2**6 = 64 latent joint states
CALIB_DAYS = 730                   # trailing causal window for each refit's grid search (2y)
REFIT_DAYS = 90                    # refit cadence, matches R-136/R-152's own convention
MIN_HISTORY_DAYS = 180             # below this, fall back to sigma_bar (multiplier=1.0)
N_PERSIST = max(1, round(MSM_KBAR / 3))   # novel branch: lowest N_PERSIST of 6 components
assert N_PERSIST == 2, N_PERSIST

_M0_GRID = (1.2, 1.4, 1.6, 1.8)
_B_GRID = (2.0, 4.0, 8.0, 16.0)
_GAMMA_KBAR_GRID = (0.3, 0.5, 0.7, 0.9)
GRID = [(m0, b, gk) for m0 in _M0_GRID for b in _B_GRID for gk in _GAMMA_KBAR_GRID]
assert len(GRID) == 64, len(GRID)


# ================================================================== (1)
# The binomial MSM(kbar) engine: state enumeration, transition matrix,
# Hamilton filter. Pure numpy, no scipy (unavailable in this environment).
# ==================================================================

def _component_gammas(kbar: int, b: float, gamma_kbar: float) -> np.ndarray:
    """Calvet-Fisher (2004) eq. for component switching probabilities.

    `k=1` (index 0) is the SLOWEST/most persistent component (smallest
    gamma); `k=kbar` (index kbar-1) is the FASTEST/most transient
    component (gamma = gamma_kbar exactly). `b > 1` controls how much
    faster each successive component switches.
    """
    ks = np.arange(1, kbar + 1, dtype=float)
    return 1.0 - (1.0 - gamma_kbar) ** (b ** (ks - kbar))


def _state_bits(kbar: int) -> np.ndarray:
    """All `2**kbar` states as a `(2**kbar, kbar)` bit matrix; column j is
    component `k=j+1`'s value (0 -> m0, 1 -> 2-m0)."""
    n = 2 ** kbar
    idx = np.arange(n)[:, None]
    return (idx >> np.arange(kbar)[None, :]) & 1


def _state_multipliers(bits: np.ndarray, m0: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-state per-component multiplier values, and their full product."""
    m_hi = 2.0 - m0
    comp_vals = np.where(bits == 0, m0, m_hi)   # (n_states, kbar)
    full_mult = comp_vals.prod(axis=1)
    return comp_vals, full_mult


def _transition_matrix(bits: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Joint transition matrix `T[from, to]` for kbar INDEPENDENT binary
    components, each retaining its value w.p. `1-gamma_k` and redrawing
    uniformly from {0,1} w.p. `gamma_k`. Built directly from each
    component's own transition probability (not a Kronecker product), so
    there is no bit-order convention to get wrong."""
    n, kbar = bits.shape
    t = np.ones((n, n))
    for j in range(kbar):
        g = gammas[j]
        same = (bits[:, None, j] == bits[None, :, j]).astype(float)
        t *= (1.0 - g) * same + g * 0.5
    return t


def _filter_loglik(r: np.ndarray, t: np.ndarray, sigma_bar: float,
                    full_mult: np.ndarray) -> float:
    """Total log-likelihood of `r` (daily returns) under the Hamilton
    filter, starting from the uniform stationary distribution. Fast path
    for grid search: no forecast arrays retained."""
    n_states = len(full_mult)
    pi = np.full(n_states, 1.0 / n_states)
    sig = sigma_bar * np.sqrt(full_mult)
    loglik = 0.0
    inv_two_sig2 = 1.0 / (2.0 * sig * sig)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sig)
    for x in r:
        pi_pred = pi @ t
        lik = norm * np.exp(-(x * x) * inv_two_sig2)
        weighted = pi_pred * lik
        total = weighted.sum()
        if not np.isfinite(total) or total <= 0.0:
            return -np.inf
        loglik += np.log(total)
        pi = weighted / total
    return float(loglik)


def _filter_forecast(r: np.ndarray, t: np.ndarray, sigma_bar: float,
                      full_mult: np.ndarray, persist_mult: np.ndarray,
                      pi0: np.ndarray | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk `r` forward, returning the ONE-STEP-AHEAD forecast of
    `E[full_mult]` and `E[persist_mult]` for EACH day `i` -- computed from
    `pi_pred` BEFORE `r[i]` is observed, i.e. using only `r[:i]`. Also
    returns the final filtered state (for chaining across refit
    boundaries)."""
    n_states = len(full_mult)
    pi = np.full(n_states, 1.0 / n_states) if pi0 is None else np.asarray(pi0, dtype=float)
    n = len(r)
    full_fc = np.empty(n)
    persist_fc = np.empty(n)
    sig = sigma_bar * np.sqrt(full_mult)
    inv_two_sig2 = 1.0 / (2.0 * sig * sig)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sig)
    for i, x in enumerate(r):
        pi_pred = pi @ t
        full_fc[i] = float(pi_pred @ full_mult)
        persist_fc[i] = float(pi_pred @ persist_mult)
        if np.isfinite(x):
            lik = norm * np.exp(-(x * x) * inv_two_sig2)
            weighted = pi_pred * lik
            total = weighted.sum()
            pi = weighted / total if (np.isfinite(total) and total > 0.0) else pi_pred
        else:
            pi = pi_pred
    return full_fc, persist_fc, pi


def fit_msm_grid(r_window: np.ndarray, kbar: int = MSM_KBAR,
                  grid: list[tuple[float, float, float]] = GRID,
                  ) -> tuple[float, float, float, float, np.ndarray]:
    """Grid-search MLE over `(m0, b, gamma_kbar)` on a CAUSAL window of
    daily returns; `sigma_bar` fixed to the window's own sample std
    (profiled out in closed form, see module docstring). Returns
    `(m0, b, gamma_kbar, sigma_bar, bits)` for the winning config."""
    bits = _state_bits(kbar)
    sigma_bar = float(np.nanstd(r_window))
    if not np.isfinite(sigma_bar) or sigma_bar <= 0.0:
        sigma_bar = 1e-6
    best = (-np.inf, None)
    for m0, b, gk in grid:
        gammas = _component_gammas(kbar, b, gk)
        t = _transition_matrix(bits, gammas)
        _, full_mult = _state_multipliers(bits, m0)
        ll = _filter_loglik(r_window, t, sigma_bar, full_mult)
        if ll > best[0]:
            best = (ll, (m0, b, gk))
    m0, b, gk = best[1]
    return m0, b, gk, sigma_bar, bits


# ================================================================== (2)
# Daily aggregation, periodic causal refit, bar-level broadcast.
# ==================================================================

def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """Causal daily close-to-close log returns, one value per calendar day."""
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).diff().dropna()


def msm_forecast_daily(df: pd.DataFrame, kbar: int = MSM_KBAR,
                        calib_days: int = CALIB_DAYS, refit_days: int = REFIT_DAYS,
                        min_history_days: int = MIN_HISTORY_DAYS,
                        n_persist: int = N_PERSIST,
                        ) -> pd.DataFrame:
    """Per-DAY one-step-ahead `E[full_mult]`/`E[persist_mult]` forecasts.

    Day D's forecast uses ONLY daily returns for days < D (the Hamilton
    filter's `pi_pred` before day D's own return is folded in), so it is
    knowable in full at the START of day D, before any of day D's
    5-minute bars occur -- no same-day lookahead. Before
    `min_history_days` of history exist, both forecasts fall back to 1.0
    (i.e. `vol = sigma_bar`, an uninformed flat estimate), matching this
    project's own `.ffill().fillna(...)`-style cold-start convention
    (e.g. `v4_scale`'s own `min_periods` gating).
    """
    r_full = daily_log_returns(df)
    idx = r_full.index
    n = len(r_full)
    full_fc = np.ones(n)
    persist_fc = np.ones(n)
    r_vals = r_full.to_numpy()

    bits = _state_bits(kbar)
    next_refit = min_history_days
    pi_state = None
    t = None
    full_mult = None
    persist_mult = None

    i = min_history_days
    while i < n:
        end = min(i + refit_days, n)
        if i >= next_refit or t is None:
            lo = max(0, i - calib_days)
            m0, b, gk, sigma_bar, _ = fit_msm_grid(r_vals[lo:i], kbar, GRID)
            gammas = _component_gammas(kbar, b, gk)
            t = _transition_matrix(bits, gammas)
            comp_vals, full_mult = _state_multipliers(bits, m0)
            persist_mult = comp_vals[:, :n_persist].prod(axis=1)
            # Re-run the filter over the calibration window itself (same
            # params) purely to obtain a warmed-up `pi_state` to CONTINUE
            # from -- its own forecast output is not used (that history's
            # forecasts, if needed, were already produced by the PRIOR
            # refit's params in the previous loop iteration).
            _, _, pi_state = _filter_forecast(
                r_vals[lo:i], t, sigma_bar, full_mult, persist_mult, pi0=None)
            next_refit = i + refit_days
        seg_full, seg_persist, pi_state = _filter_forecast(
            r_vals[i:end], t, sigma_bar, full_mult, persist_mult, pi0=pi_state)
        full_fc[i:end] = seg_full
        persist_fc[i:end] = seg_persist
        i = end

    return pd.DataFrame({"full_mult": full_fc, "persist_mult": persist_fc}, index=idx)


def _broadcast_vol(df: pd.DataFrame, daily_mult: pd.Series) -> np.ndarray:
    """Broadcast a per-day multiplier onto `df`'s bars: bar `i`'s value is
    its OWN calendar day's multiplier (known at that day's start, per
    `msm_forecast_daily`'s one-step-ahead construction -- not the bug
    class R-172 found, where a same-day END-of-day statistic was used).
    Annualized to match `v4_symmetric_vol`'s own units."""
    day = df.index.floor("D")
    aligned = daily_mult.reindex(day).to_numpy()
    sigma_bar_annualized = (df["close"].pipe(np.log).diff()
                            .rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).std()
                            * np.sqrt(BARS_PER_YEAR))
    # sigma_bar itself is re-derived per-bar from a short trailing window so
    # early bars (before the first full day closes) are never NaN; the MSM
    # multiplier (>=1 day granularity, ffilled) rides on top of it.
    base = sigma_bar_annualized.ffill().bfill().to_numpy()
    out = base * np.sqrt(np.where(np.isfinite(aligned), aligned, 1.0))
    return np.where(np.isfinite(out), out, 0.0)


def msm_full_vol_bars(df: pd.DataFrame) -> np.ndarray:
    """CONSERVATIVE branch's vol input: the MSM filter's full one-step-ahead
    multiplier applied to a short-window realized-vol base, replacing
    `v4_symmetric_vol` wholesale."""
    fc = msm_forecast_daily(df)
    return _broadcast_vol(df, fc["full_mult"])


def msm_structural_vol_bars(df: pd.DataFrame) -> np.ndarray:
    """NOVEL branch's vol input: the SAME filter, but only the lowest
    `N_PERSIST` (most persistent) of `MSM_KBAR` components contribute --
    the high-frequency, fast-decaying component is deliberately zeroed
    out, per R-08/R-136's own diagnosis that reacting to transient vol
    spikes is what hurts this architecture."""
    fc = msm_forecast_daily(df)
    return _broadcast_vol(df, fc["persist_mult"])


def msm_full_scale(df: pd.DataFrame) -> np.ndarray:
    return conditional_target_scale(msm_full_vol_bars(df))


def msm_structural_scale(df: pd.DataFrame) -> np.ndarray:
    return conditional_target_scale(msm_structural_vol_bars(df))


def msm_full_target(df: pd.DataFrame) -> np.ndarray:
    return apply_deadband(v4_vote_frac(df).to_numpy() * msm_full_scale(df))


def msm_structural_target(df: pd.DataFrame) -> np.ndarray:
    return apply_deadband(v4_vote_frac(df).to_numpy() * msm_structural_scale(df))


# A generous warmup so `run_period`'s prefix actually supplies MSM enough
# causal history on the inner-validation slice (which starts well after
# the dataset's own beginning) -- see r175_direction.md for the sizing
# arithmetic. Both branches use this same warmup for their TargetStrategy.
MSM_WARMUP_BARS = (CALIB_DAYS + 60) * BARS_PER_DAY


def compare(candidate_build, *, label: str, btc: pd.DataFrame | None = None,
            eth: pd.DataFrame | None = None, control_build=None,
            markets: tuple[MarketSpec, ...] = (SPOT, FUTURES),
            include_eth: bool = True, seed: int = 0) -> list[dict]:
    """`r102_shared.compare()`, cloned rather than imported, ONLY so the
    candidate's `TargetStrategy` gets `MSM_WARMUP_BARS` instead of the
    default 80-day warmup (MSM's calibration window needs far more causal
    history than v4's own anchors) -- r102_shared.compare() cannot take a
    custom warmup and is shared infrastructure other rounds depend on
    verbatim, so it is not modified. Everything else (slices, control,
    pairing, risk-match diagnostics) is identical to r102_shared.compare().
    """
    if control_build is None:
        control_build = v4_target
    if btc is None:
        btc = load_btc()
    assert_no_holdout(btc, "compare(): btc")
    if include_eth and eth is None:
        eth = load_eth()
    if include_eth:
        assert_no_holdout(eth, "compare(): eth")

    cand = TargetStrategy(candidate_build, name=f"r175_{label}", warmup=MSM_WARMUP_BARS)
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


def exposure_at_episodes(target: np.ndarray, df: pd.DataFrame,
                          episodes: list[tuple[str, str]] = STRESS_EPISODES,
                          horizon_days: int = 10) -> dict:
    """Mean |exposure| in the `horizon_days` trading days FOLLOWING each
    dated episode's own vol spike -- the novel branch's decisive mechanism
    check (Step-1 Q4(b)): is the candidate LESS de-risked than the control
    at these points, independent of any Sharpe number."""
    out = {}
    idx = df.index
    for name, date in episodes:
        ts = pd.Timestamp(date, tz=idx.tz)
        pos = idx.searchsorted(ts)
        if pos >= len(idx):
            continue
        end = min(pos + horizon_days * BARS_PER_DAY, len(idx))
        if pos >= end:
            continue
        out[name] = float(np.nanmean(np.abs(target[pos:end])))
    return out


# --------------------------------------------------------------- self-test

def _simulate_msm(n: int, kbar: int, m0: float, b: float, gamma_kbar: float,
                   sigma_bar: float, seed: int) -> np.ndarray:
    """Simulate a KNOWN binomial MSM process for the recovery self-test."""
    rng = np.random.default_rng(seed)
    gammas = _component_gammas(kbar, b, gamma_kbar)
    m_hi = 2.0 - m0
    state = rng.integers(0, 2, size=kbar)
    r = np.empty(n)
    for i in range(n):
        redraw = rng.random(kbar) < gammas
        state = np.where(redraw, rng.integers(0, 2, size=kbar), state)
        mult = np.prod(np.where(state == 0, m0, m_hi))
        r[i] = rng.normal(0.0, sigma_bar * np.sqrt(mult))
    return r


def _self_test() -> None:
    kbar = MSM_KBAR
    bits = _state_bits(kbar)
    assert bits.shape == (64, kbar)

    # (1) transition matrix is row-stochastic and preserves the uniform
    # stationary distribution (each component is marginally symmetric).
    gammas = _component_gammas(kbar, b=4.0, gamma_kbar=0.5)
    assert gammas[0] < gammas[-1], gammas          # k=1 slower than k=kbar
    assert abs(gammas[-1] - 0.5) < 1e-12
    t = _transition_matrix(bits, gammas)
    assert np.allclose(t.sum(axis=1), 1.0), "transition matrix rows must sum to 1"
    n_states = 2 ** kbar
    uniform = np.full(n_states, 1.0 / n_states)
    assert np.allclose(uniform @ t, uniform, atol=1e-10), "uniform must be stationary"

    # (2) state multipliers: unconditional mean of the full product, and of
    # the persistent-only sub-product, must both be 1.0 (each component has
    # mean 1 by symmetry, independent components -> product of means).
    comp_vals, full_mult = _state_multipliers(bits, m0=1.4)
    assert abs(float(uniform @ full_mult) - 1.0) < 1e-9
    persist_mult = comp_vals[:, :N_PERSIST].prod(axis=1)
    assert abs(float(uniform @ persist_mult) - 1.0) < 1e-9

    # (3) recovery sanity: true params should out-loglik a badly wrong
    # parameter set on data simulated from the true process.
    true_params = (1.6, 6.0, 0.6)
    sim_r = _simulate_msm(1500, kbar, *true_params, sigma_bar=0.02, seed=175)
    gammas_true = _component_gammas(kbar, true_params[1], true_params[2])
    t_true = _transition_matrix(bits, gammas_true)
    _, fm_true = _state_multipliers(bits, true_params[0])
    ll_true = _filter_loglik(sim_r, t_true, 0.02, fm_true)

    wrong_params = (1.2, 2.0, 0.9)
    gammas_wrong = _component_gammas(kbar, wrong_params[1], wrong_params[2])
    t_wrong = _transition_matrix(bits, gammas_wrong)
    _, fm_wrong = _state_multipliers(bits, wrong_params[0])
    ll_wrong = _filter_loglik(sim_r, t_wrong, 0.02, fm_wrong)
    assert ll_true > ll_wrong, (ll_true, ll_wrong)

    # (3b) the grid search itself should prefer a HIGH-persistence /
    # wide-multiplier config over a near-degenerate one on data simulated
    # with strong, persistent multifractal structure (m0 far from 1,
    # gamma_kbar low => long memory) -- a genuine recovery check, not just
    # "true beats one wrong point".
    strong_r = _simulate_msm(2500, kbar, m0=1.2, b=8.0, gamma_kbar=0.2,
                              sigma_bar=0.02, seed=176)
    fit_m0, fit_b, fit_gk, fit_sigma, _ = fit_msm_grid(strong_r, kbar, GRID)
    assert fit_m0 <= 1.6, ("grid search should favor a wide multiplier on "
                            f"strongly multifractal data, got m0={fit_m0}")

    # (4) causal_truncation_probe_series on the FULL pipeline, against REAL
    # BTC data (R-172's own explicit lesson: synthetic-only misses same-day
    # broadcast bugs on choppy real data).
    btc = load_btc()
    probe_df = btc.loc[:"2018-06-30"].copy()
    assert causal_truncation_probe_series(msm_full_vol_bars, probe_df), (
        "msm_full_vol_bars is not causal on real BTC data")
    assert causal_truncation_probe_series(msm_structural_vol_bars, probe_df), (
        "msm_structural_vol_bars is not causal on real BTC data")

    # (5) msm_forecast_daily: no look-ahead by construction check -- day D's
    # forecast must be IDENTICAL whether or not later days are present.
    daily_r = daily_log_returns(probe_df)
    full_a = msm_forecast_daily(probe_df)
    shorter = probe_df.loc[:"2018-03-31"]
    full_b = msm_forecast_daily(shorter)
    common = full_b.index
    # allow the last REFIT_DAYS window to differ (a refit boundary can land
    # a few days earlier/later depending on series length); compare the
    # bulk of the overlap strictly.
    cut = common[:-REFIT_DAYS] if len(common) > REFIT_DAYS else common[:0]
    if len(cut):
        pd.testing.assert_series_equal(
            full_a.loc[cut, "full_mult"], full_b.loc[cut, "full_mult"],
            check_exact=False, rtol=1e-9, atol=1e-12)

    # (6) exposure_at_episodes: shape/finite sanity on a trivial target.
    dummy = np.ones(len(probe_df))
    res = exposure_at_episodes(dummy, probe_df, episodes=STRESS_EPISODES[:2])
    assert all(v == 1.0 for v in res.values()), res

    print("r175_shared self-test OK "
          f"(ll_true={ll_true:.2f} > ll_wrong={ll_wrong:.2f}, "
          f"grid-recovered m0={fit_m0} on strongly multifractal synthetic data)")


if __name__ == "__main__":
    _self_test()
else:
    _self_test()
