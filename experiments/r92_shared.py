"""Shared, read-only utilities and pre-registration for the R-92 round (08-21).

DIRECTION, in one sentence: attack `kelly_regime_v4`'s anchor **SPAN** --
20/40/80 days, chosen a priori as a doubling ladder (Muller et al. 1997;
Corsi 2009's HAR) and never fitted -- by DERIVING it from a fitted generative
model of BTC's own autocorrelation structure, via **Sepp & Lucic (2026)**,
"The Science and Practice of Trend-Following Systems," arXiv:2607.19497
(also SSRN 3167787; companion code at github.com/ArturSepp/TrendFollowingSystems),
instead of searching it empirically the way every prior anchor-span round did.
This is backlog item **B-42**, filed by R-89, left OPEN by R-90 and R-91 (both
of which worked B-41 and B-40, the sibling follow-ons from the same literature
pass) and named again in every re-ranking since as one of two remaining OPEN
items alongside B-32 (pure multi-asset registration infrastructure, no
strategy-improvement angle).

**Which constraint it attacks: SIZE.** The anchor span feeds `frac`, the vote
factor R-62/R-87 established (four independent ways) carries v4's entire
signature -- this is the same axis R-89 (latch geometry, response-function
shape) and R-91 (state-dependence) worked, now on its third free dimension:
the horizon itself. Explicitly not a new INFO signal (no new data channel:
same OHLCV close series) and not a regime-timing DETECTION mechanism (no
six-episode gate here -- this derives a single structural parameter of the
existing vote, it does not try to detect the date of a historical transition
faster than v4's own anchors do).

**The theory, precisely.** Sepp & Lucic derive a closed-form annualized Sharpe
ratio for a European (EWMA-filtered) trend-following system as a function of
the filter's smoothing parameter alone, given two moments of the instrument's
own volatility-normalized returns z_t = r_t / sigma_t: its autocorrelation
function rho(m) and its drift (mean) mu. Their Eq. (2.5) defines the EWMA
smoothing parameter from a span S (in the same time unit as the return
series): nu = 1 - 2/(S+1). Their Eq. (5.3) defines the autocorrelation
generating function Psi_nu = [sum_{m=1..inf} nu^m * rho(m)]. Their Eq. (5.8)
defines two loadings A_nu = ((1-nu)/nu) * Psi_nu and B_nu = ((1-nu)/(1+nu)) *
(1 + 2*Psi_nu). Their Eq. (5.12) gives the (frictionless, kurtosis-loading
dropped -- see the disclosed simplification below) annualized Sharpe:

    SR(nu) = [sqrt(a)*A_nu + mu_an^2/sqrt(a)] /
             sqrt(B_nu + A_nu^2 + (mu_an^2/a)*(1 + B_nu + 2*A_nu))

where mu_an = sqrt(a)*mu is the annualized drift and a is the annualization
factor (a=365.25 here: BTC trades every day, no session gaps). Their Section
6.2 gives the AR(1) special case this round uses: for an AR(1) process with
lag-1 autocorrelation phi, Psi_nu = nu*phi / (1 - nu*phi) in closed form --
no numerical summation needed, just one fitted scalar phi.

**Disclosed simplification (a named, upfront limitation, not a finding).**
The paper's full Eq. (5.12) includes a kurtosis loading term
kappa * K_nu, where kappa is the excess kurtosis of the filter's innovations
and K_nu is a second infinite sum over the filter's impulse-response
coefficients. That term's exact form depends on machinery (a martingale
difference decomposition of the EWMA output) this round does not reproduce.
Both branches below use the KURTOSIS-FREE form of Eq. (5.12) quoted above.
Since BTC daily returns carry material positive excess kurtosis, this is a
real approximation, disclosed here before either branch runs, and named as
part of this round's own falsification risk: if the derived span performs
badly, "the kurtosis term was dropped" is one of the two most likely reasons
why (the other being the architecture mismatch below).

**Second disclosed simplification: EWMA vs. v4's own SMA-with-latch anchors.**
Sepp & Lucic's theory is for a linear EWMA filter. `kelly_regime_v4`'s anchors
are simple moving averages with a 1% hysteresis band and latching (a
different, nonlinear filter -- closer to their "American" binary-system
classification, which R-89/R-90/R-91 already treat this paper's citations as
applying to directionally, not literally). This round bridges the two by
matching each filter's MEAN LAG: an EWMA of span S has mean lag (S-1)/2
(days); a trailing SMA of N days has mean lag (N-1)/2 (days) -- identical
when S=N. So "span" in Sepp & Lucic's sense is treated as equal to v4's own
day-denominated `horizons` parameter directly, with no unit conversion. This
is a first-moment match, not an exact filter-shape match, and is the second
named source of risk if the derived span fails.

**Data convention.** The AR(1) fit and the SR(nu) evaluation both run on
DAILY close-to-close log returns, volatility-normalized by a trailing
20-day realized-vol estimate (z_t = r_t / sigma_t, matching the paper's own
z_t convention and its a=260-trading-day annualization replaced here by
a=365.25 for a 24/7 market), computed CAUSALLY and restricted to
inner-train only (2017-01-01 .. 2020-12-31) for both branches' frozen fit.
Nothing here reads inner-validation or the holdout to choose phi or mu.

**Not a duplicate of.** R-06/R-07/R-40/R-45 (empirical anchor-span SWEEPS --
grid search over backtested Sharpe, the exact "search it" this item exists
to replace with "derive it"). R-89 (latch geometry / response-function shape
-- both leave the vote's three anchor SPANS at the shipped 20/40/80 and vary
what the vote does with them). R-90 (path-dependent trailing-stop exit --
does not touch the vote or its anchors at all). R-91 (state-dependence of
the horizon via a GHM turning-point scaler multiplied ON TOP of the
unchanged 20/40/80 vote -- this round instead re-derives the 20/40/80 span
values themselves; R-91's own shared module names this exact distinction).
R-62 (factor-isolation: vote alone vs. scale alone at FIXED existing spans --
this round does not isolate factors, it changes one factor's own inputs).

**Falsification test, named now, before any code.** Sepp & Lucic's own
headline empirical claim (verified by web search before either branch was
dispatched) is that an instrument's sample autocorrelation function and
drift alone predict its backtested trend-following Sharpe with pooled
correlation 0.99 across 84 futures contracts -- i.e., the closed-form SR(nu)
curve, evaluated at each candidate span, should be a genuinely informative
(not flat, not degenerate) function of span on BTC's own data. If the
CAUSAL, inner-train-only AR(1) fit returns phi <= 0 (no exploitable positive
autocorrelation at any horizon this project's data can supply, which would
mean the paper's own precondition for trend-following profitability --
positive long-run autocorrelation -- does not hold on this instrument, the
same style of upfront kill switch R-89's Step-0 and R-91's A0 used), OR if
SR(nu) has no INTERIOR optimum over the feasible span grid -- monotonically
decreasing (the closed form wants the shortest possible span: noise-chasing,
not a ladder) or monotonically increasing all the way to the grid boundary
(the closed form wants the longest possible span: it degenerates toward
"never resize", i.e. buy-and-hold, exactly what happens whenever the
instrument's constant drift term dominates the AR(1) timing term at every
span in the feasible range) -- the branch is disqualified by pre-registration
and reported NEGATIVE regardless of any downstream backtest number, exactly
the R-89/R-91 convention for a Step-0 / A0 kill switch.

Two branches, disjoint files, both measured by this module:

- **conservative** (`r92_conservative_ar1_static_span.py`) -- fit ONE static
  AR(1) phi (and drift mu) once, on inner-train only; solve for the SR(nu)-
  maximizing span on a fine grid; freeze a doubling ladder
  (0.5x, 1x, 2x the derived center span) exactly replacing v4's (20, 40, 80)
  with (derived/2, derived, derived*2) rounded to the nearest integer day;
  no re-fitting, no lookahead, a single frozen fit exactly as v4's own
  20/40/80 is a single frozen choice (but derived analytically rather than
  swept).
- **novel** (`r92_novel_rolling_ar1_span.py`) -- the SAME closed-form
  machinery, but the fit is CAUSAL and re-estimated periodically (annually,
  on an EXPANDING window that only ever grows forward in time, at each
  January 1 boundary starting from the first date with >= 2 years of daily
  history) rather than frozen once, so the derived ladder can itself track
  slow changes in BTC's own autocorrelation structure over the backtest --
  a genuinely different construction from every prior anchor-span round,
  all of which used a single fixed ladder for the whole series.

**Pre-registered decision rule, identical structure for both branches
(frozen before any number is read), the R-89/R-90/R-91 convention.**

*Step A -- mechanism gate, per configuration, before any performance number
is read:*
- **A1 identity.** Not applicable in the R-89/R-91 sense (there is no
  "scaler == 1" identity point here, since this changes the anchors
  themselves) -- replaced by **A1' reproducibility**: re-running the AR(1)
  fit and the SR(nu) grid twice from a clean import must return the exact
  same derived span (determinism check).
- **A2 non-inertness.** R^2 of the candidate's exposure path against v4's own
  must be < 0.98 on inner-train (else the derived span rounds back to
  effectively 20/40/80 and the configuration is inert by construction --
  reported, not scored as a genuine test of the theory).
- **A3 causality.** `causal_truncation_probe` passes at two cut depths for
  the branch's own build function, AND (novel branch only) the annual
  re-estimation checkpoints are verified to use only data strictly before
  each checkpoint date.
- **A0 (this round's kill switch, checked BEFORE Step A).** The
  falsification test above: causal inner-train phi > 0, and SR(nu) is not
  monotonically decreasing over the whole feasible grid. If either fails,
  the branch is disqualified by pre-registration and reported NEGATIVE
  regardless of any downstream number.

*Step B -- selection, on inner-train and inner-validation only; the holdout
is not read by either branch.* The finalist is the frozen (conservative) or
checkpointed (novel) derived-span configuration itself -- there is no sweep
to select among on Step B, by construction (that is the point of "derive,
don't search"), so Step B here is purely the measurement of the ONE derived
configuration against v4 on both slices and both markets.

*Promotion bar -- default REJECT. All must hold:*
- **B1.** The paired block-bootstrap difference vs v4 excludes zero in at
  least one of the four (slice x market) cells, and its point estimate is
  positive in all four.
- **B2.** Either delta-Sharpe > +0.2 (the R-20 noise floor) on
  inner-validation on both markets, OR a max-drawdown improvement on both
  markets where `risk_matched` (exposure ratio and vol ratio vs v4 both in
  [0.9, 1.1]) is true for both -- an unmatched drawdown improvement is not
  evidence, per the standing R-28/R-32/R-33 rule.
- **B3.** The neighbourhood is a plateau, not a peak: the SR(nu) closed form
  evaluated one grid step either side of the derived optimum must move in
  the same direction the closed form predicts (a check on the THEORY's own
  internal consistency, not a re-sweep of the strategy).
- **B4 falsification, ETH replication.** The frozen/checkpointed finalist
  must show the same SIGN of improvement over v4 on Bitfinex ETH pre-2023
  (inner-train only; ETH coverage starts 2019-03-14), on both markets.
  Failing it is NEGATIVE.
- **B5 cost robustness.** The improvement must not reverse sign at a 0.40%
  taker fee (Bitstamp's real entry tier).

Named counter-prediction (what would make this fail, written before any
code): if the A0 kill switch fires -- BTC's own daily z_t carries no
positive AR(1) autocorrelation on inner-train, or the SR(nu) curve has no
interior optimum -- Sepp & Lucic's own precondition for a profitable
European trend system does not hold here, and BOTH branches should be
expected to fail independent of their internal designs, the same structural
outcome R-89's novel branch hit with Schmidhuber's cubic term and R-91 hit
with GHM's turning-point ranking.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both branches: neither may edit it, so both are
measured by identical machinery. Nothing here reads a bar at or after
OOS_START.
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
ANNUALIZATION_DAYS = 365.25  # crypto: 24/7, no session gaps

# ---------------------------------------------------------------- splits
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

# kelly_regime_v4's own shipped constants (do not change: the control must
# be v4, not a re-parameterisation of it).
V4_HORIZONS: tuple[int, ...] = (20, 40, 80)
V4_BAND = 0.01
V4_TARGET_VOL = 0.55
V4_MAX_LEVERAGE = 2.0
V4_VOL_SPAN = 8 * BARS_PER_DAY
V4_DEADBAND = 0.10
V4_ANCHOR_SPAN_DAYS = 180
V4_HIGH_IN, V4_HIGH_OUT = 1.70, 1.20
V4_LOW_IN, V4_LOW_OUT = 0.55, 0.85

# Feasible span grid for the closed-form optimum search (days). 5..200 covers
# everything from faster than v4's own fastest anchor to beyond its slowest.
SPAN_GRID_DAYS = np.arange(5, 201, 1)

# Vol-normalization window for z_t (days), matching the paper's own use of a
# short realized-vol estimate, not v4's 8-day EWMA vol_span (kept distinct on
# purpose: this is a MEASUREMENT convention for fitting phi/mu, unrelated to
# v4's own trading-time vol target).
Z_VOL_WINDOW_DAYS = 20


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
    """Bitfinex ETH (the series R-17/R-47/R-89/R-90/R-91 use for cross-asset replication)."""
    return _truncate(load_ohlcv_csv(ROOT / "data" / "ethusd_bitfinex_5m.csv.gz"), "ETH")


# ------------------------------------------------------- v4's own factors
# (reproduced exactly, same convention as r89_shared.py/r90_shared.py/r91_shared.py)

def _latched_anchor_vote(close: pd.Series, days: int, band: float = V4_BAND) -> np.ndarray:
    """One anchor's own latched 0/1 vote, reproduced exactly as v4 computes each of its three."""
    anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
    v = pd.Series(
        np.where(close > anchor * (1.0 + band), 1.0,
                 np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
        index=close.index,
    )
    return v.ffill().fillna(0.0).to_numpy()


def vote_frac(df: pd.DataFrame, horizons: tuple[int, ...], band: float = V4_BAND) -> np.ndarray:
    """The latched anchor vote for an ARBITRARY horizon tuple (v4's own construction, generalised)."""
    close = df["close"]
    votes = [_latched_anchor_vote(close, days, band) for days in horizons]
    return sum(votes) / len(votes)


def v4_vote_frac(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v4's own shipped vote (horizons=20,40,80), for the control."""
    return vote_frac(df, V4_HORIZONS, V4_BAND)


def v4_scale(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v3/v4's conditional volatility-target scale factor, reproduced exactly."""
    r = np.log(df["close"]).diff()
    vol = (r.ewm(span=V4_VOL_SPAN, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
    slow = (pd.Series(vol).ewm(span=V4_ANCHOR_SPAN_DAYS * BARS_PER_DAY,
                               min_periods=BARS_PER_DAY).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(slow > 0, vol / slow, np.nan)
        full = np.minimum(V4_TARGET_VOL / vol, V4_MAX_LEVERAGE)
        steady = np.minimum(V4_TARGET_VOL / slow, V4_MAX_LEVERAGE)
    full = np.where(np.isfinite(full), full, 0.0)
    steady = np.where(np.isfinite(steady), steady, 0.0)

    n = len(df)
    out = np.zeros(n)
    state = 0
    for i in range(n):
        x = ratio[i]
        if np.isfinite(x):
            if state == 0:
                state = 1 if x > V4_HIGH_IN else (-1 if x < V4_LOW_IN else 0)
            elif state == 1 and x < V4_HIGH_OUT:
                state = 0
            elif state == -1 and x > V4_LOW_OUT:
                state = 0
        out[i] = full[i] if state != 0 else steady[i]
    return out


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
    return v4_vote_frac(df) * v4_scale(df)


def v4_target(df: pd.DataFrame) -> np.ndarray:
    """kelly_regime_v4's complete, final target path (post-deadband)."""
    return apply_deadband(v4_raw_desired(df))


def span_ladder_target(df: pd.DataFrame, horizons: tuple[int, ...]) -> np.ndarray:
    """v4's own architecture (vote x scale, then deadband) with an ARBITRARY horizon ladder."""
    desired = vote_frac(df, horizons, V4_BAND) * v4_scale(df)
    return apply_deadband(desired)


# --------------------------------------------------- Sepp & Lucic (2026)
# arXiv:2607.19497, Sections 2, 5.1-5.2, 6.2. See module docstring for the
# disclosed kurtosis-free and EWMA-vs-SMA simplifications.

def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily close-to-close log returns from a 5m OHLCV frame, causal (last bar of each day)."""
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).diff().dropna()


def vol_normalized_returns(daily_r: pd.Series, window_days: int = Z_VOL_WINDOW_DAYS) -> pd.Series:
    """z_t = r_t / sigma_t, sigma_t a CAUSAL trailing realized-vol estimate (shifted by one day
    so z_t never uses its own day's return to normalize itself)."""
    sigma = daily_r.rolling(window_days).std().shift(1)
    z = daily_r / sigma
    return z.replace([np.inf, -np.inf], np.nan).dropna()


def fit_ar1(z: pd.Series) -> tuple[float, float]:
    """Lag-1 autocorrelation phi and mean mu of a (already vol-normalized) daily return series.

    phi is the plain Pearson correlation of z_t against z_{t-1} -- the AR(1)
    coefficient estimator Sepp & Lucic's Section 6.2 example uses.
    """
    z = z.to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 30:
        return float("nan"), float("nan")
    mu = float(np.mean(z))
    x0, x1 = z[:-1], z[1:]
    x0c, x1c = x0 - x0.mean(), x1 - x1.mean()
    denom = np.sqrt(np.sum(x0c ** 2) * np.sum(x1c ** 2))
    phi = float(np.sum(x0c * x1c) / denom) if denom > 0 else float("nan")
    return phi, mu


def nu_of_span(span_days: np.ndarray | float) -> np.ndarray | float:
    """Eq. (2.5): EWMA smoothing parameter from a span (same units as the return series)."""
    return 1.0 - 2.0 / (span_days + 1.0)


def sharpe_ar1(nu: np.ndarray, phi: float, mu: float,
              a: float = ANNUALIZATION_DAYS) -> np.ndarray:
    """Eq. (5.12), AR(1) closed form (Section 6.2), kurtosis loading dropped (disclosed above).

    psi_nu = nu*phi / (1 - nu*phi)                              (Sec. 6.2 AR(1) special case)
    A_nu   = (1-nu)/nu * psi_nu                                  (Eq. 5.8)
    B_nu   = (1-nu)/(1+nu) * (1 + 2*psi_nu)                      (Eq. 5.8)
    mu_an  = sqrt(a) * mu
    SR(nu) = [sqrt(a)*A_nu + mu_an^2/sqrt(a)] /
             sqrt(B_nu + A_nu^2 + (mu_an^2/a)*(1 + B_nu + 2*A_nu))
    """
    nu = np.asarray(nu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_nu = nu * phi / (1.0 - nu * phi)
        a_nu = (1.0 - nu) / nu * psi_nu
        b_nu = (1.0 - nu) / (1.0 + nu) * (1.0 + 2.0 * psi_nu)
    mu_an = np.sqrt(a) * mu
    numer = np.sqrt(a) * a_nu + mu_an ** 2 / np.sqrt(a)
    denom_sq = b_nu + a_nu ** 2 + (mu_an ** 2 / a) * (1.0 + b_nu + 2.0 * a_nu)
    with np.errstate(invalid="ignore"):
        sr = np.where(denom_sq > 0, numer / np.sqrt(np.maximum(denom_sq, 1e-18)), np.nan)
    return sr


def derive_optimal_span(phi: float, mu: float, grid: np.ndarray = SPAN_GRID_DAYS
                        ) -> tuple[int, np.ndarray, np.ndarray]:
    """Grid-evaluate the closed form (continuous in span, not a backtest sweep) and return
    the SR(nu)-maximizing span (days), the full SR curve, and the grid it was evaluated on.
    """
    nu = nu_of_span(grid.astype(float))
    sr = sharpe_ar1(nu, phi, mu)
    if not np.any(np.isfinite(sr)):
        return -1, sr, grid
    best_idx = int(np.nanargmax(sr))
    return int(grid[best_idx]), sr, grid


def kill_switch_a0(phi: float, sr_curve: np.ndarray, grid: np.ndarray = SPAN_GRID_DAYS,
                   edge_tolerance: int = 3) -> tuple[bool, str]:
    """This round's A0 pre-registered gate. Returns (passes, reason).

    The disqualifying case, checked directly rather than via a global-
    monotonicity proxy (an earlier draft of this gate used
    ``all(diff >= 0)`` / ``all(diff <= 0)``, which a curve with a small
    interior wiggle near one edge can narrowly evade while still having its
    TRUE maximum sitting on the boundary -- caught on BTC's own inner-train
    fit before either branch was dispatched, and fixed here, disclosed as a
    pre-registration bug fix per ROUTINE.md's allowance, not a goalpost move
    made after any performance number was read): the closed-form optimum
    landing within ``edge_tolerance`` grid steps of either edge of the
    feasible span grid means there is no INTERIOR optimum to derive a finite
    three-anchor ladder from -- either the formula wants the shortest
    possible span (noise-chasing) or, far more likely given BTC's own
    strong drift, the longest one (degenerating toward "never resize", i.e.
    buy-and-hold, exactly what Sepp & Lucic's own SR(nu) does as nu->1
    whenever the constant drift term dominates the AR(1) timing term: A_nu
    and B_nu both vanish and SR(nu) asymptotes to the instrument's raw
    annualized Sharpe mu_an, independent of any span choice at all).
    """
    if not np.isfinite(phi) or phi <= 0:
        return False, f"phi={phi:.4f} <= 0: no positive AR(1) autocorrelation on inner-train"
    finite_mask = np.isfinite(sr_curve)
    if int(np.sum(finite_mask)) < 3:
        return False, "SR(nu) curve degenerate (fewer than 3 finite points)"
    best_idx = int(np.nanargmax(np.where(finite_mask, sr_curve, -np.inf)))
    n = len(grid)
    if best_idx < edge_tolerance or best_idx >= n - edge_tolerance:
        return False, (f"SR(nu) optimum at span={grid[best_idx]} sits on the grid boundary "
                        f"[{grid[0]}, {grid[-1]}] (index {best_idx}/{n - 1}): no interior "
                        f"optimum, closed form is degenerate over the feasible ladder range")
    return True, "ok"


# ------------------------------------------------------------- evaluation
# (identical machinery to r89_shared.py/r90_shared.py/r91_shared.py, so the
# four rounds' output tables are directly comparable)

SLICES: dict[str, tuple[str | None, str | None]] = {
    "inner_train": (INNER_TRAIN_START, INNER_TRAIN_END),
    "inner_val": (INNER_VAL_START, INNER_VAL_END),
}


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


def run_slice(strategy: Strategy, df: pd.DataFrame, slice_name: str,
              market: MarketSpec = SPOT, balance: float = 1_000.0) -> SliceResult:
    """One backtest over a named slice, with a warm (non-trading) prefix."""
    start, end = SLICES[slice_name]
    res = run_period(strategy, df, start, end, market=market, start_balance=balance)
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


def daily_simple_returns(equity: pd.Series) -> np.ndarray:
    """Daily SIMPLE returns of a bar-frequency equity curve."""
    return inference_daily_returns(equity).to_numpy()


class TargetStrategy(Strategy):
    """Wrap a pure ``build_target(df) -> np.ndarray`` as a runnable strategy."""

    name = "r92_target"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, build_target, name: str = "r92_target",
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


def compare(build_candidate, df: pd.DataFrame, *, label: str,
            control_build=None, markets=(SPOT, FUTURES),
            slice_names=("inner_train", "inner_val"), warmup: int | None = None,
            seed: int = 0) -> list[dict]:
    """Candidate vs control on every (slice, market) cell, one table."""
    if control_build is None:
        control_build = v4_target

    cand_path = np.asarray(build_candidate(df), dtype=float)
    ctrl_path = np.asarray(control_build(df), dtype=float)
    rsq = r_squared(cand_path, ctrl_path)

    cand = TargetStrategy(build_candidate, name=label, warmup=warmup)
    ctrl = TargetStrategy(control_build, name="kelly_regime_v4")

    rows = []
    for slice_name in slice_names:
        for market in markets:
            a = run_slice(cand, df, slice_name, market)
            b = run_slice(ctrl, df, slice_name, market)
            pr = paired_diff(a.daily, b.daily, seed=seed)
            rows.append({
                "label": label, "slice": slice_name, "market": market.name,
                "r2_vs_control": rsq,
                "cand_final": a.final_balance, "ctrl_final": b.final_balance,
                "cand_sharpe": a.sharpe, "ctrl_sharpe": b.sharpe,
                "d_sharpe": a.sharpe - b.sharpe,
                "cand_dd": a.max_drawdown_pct, "ctrl_dd": b.max_drawdown_pct,
                "d_dd": a.max_drawdown_pct - b.max_drawdown_pct,
                "cand_trades": a.num_trades, "ctrl_trades": b.num_trades,
                "exposure_ratio": (a.mean_abs_exposure / b.mean_abs_exposure
                                   if b.mean_abs_exposure else float("nan")),
                "vol_ratio": (a.realized_vol / b.realized_vol
                              if b.realized_vol else float("nan")),
                "risk_matched": bool(
                    0.9 <= (a.mean_abs_exposure / b.mean_abs_exposure if b.mean_abs_exposure else np.nan) <= 1.1
                    and 0.9 <= (a.realized_vol / b.realized_vol if b.realized_vol else np.nan) <= 1.1),
                "d_loggrowth": pr.diff.point,
                "d_lo": pr.diff.lo, "d_hi": pr.diff.hi,
                "excludes_zero": bool(pr.diff.lo > 0 or pr.diff.hi < 0),
            })
    return rows


def print_rows(rows: list[dict]) -> None:
    """One fixed-width line per cell, so branches' output is diffable."""
    hdr = (f"{'label':26s} {'slice':11s} {'market':11s} {'cand$':>10s} {'ctrl$':>10s} "
           f"{'dSh':>6s} {'dDD':>7s} {'expR':>5s} {'volR':>5s} {'RM':>3s} "
           f"{'dlogG':>7s} {'[lo':>8s},{'hi]':>8s} {'excl0':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:26]:26s} {r['slice']:11s} {r['market']:11s} "
              f"{r['cand_final']:10,.0f} {r['ctrl_final']:10,.0f} "
              f"{r['d_sharpe']:+6.2f} {r['d_dd']:+7.1f} "
              f"{r['exposure_ratio']:5.2f} {r['vol_ratio']:5.2f} "
              f"{'Y' if r['risk_matched'] else 'n':>3s} "
              f"{r['d_loggrowth']:+7.3f} {r['d_lo']:+8.3f},{r['d_hi']:+8.3f} "
              f"{'YES' if r['excludes_zero'] else 'no':>5s}")


# --------------------------------------------------------------- inference

def paired_diff(candidate: np.ndarray, control: np.ndarray, *,
                mean_block: float = 30.0, n_boot: int = 2_000, seed: int = 0):
    """Paired stationary-block-bootstrap difference in total log growth."""
    n = min(len(candidate), len(control))
    return paired_bootstrap(np.asarray(candidate[-n:], dtype=float),
                            np.asarray(control[-n:], dtype=float),
                            total_log_return, mean_block=mean_block,
                            n_boot=n_boot, seed=seed)


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of ``a`` against ``b`` -- the standing "is it merely v4 again?" check."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.std(b) == 0 or np.std(a) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def causal_truncation_probe(build_target_fn, df: pd.DataFrame,
                            cuts: tuple[float, ...] = (0.55, 0.80)) -> bool:
    """Rebuild the target on truncated frames; the shared prefix must match."""
    full = np.asarray(build_target_fn(df), dtype=float)
    for cut in cuts:
        k = int(len(df) * cut)
        part = np.asarray(build_target_fn(df.iloc[:k]), dtype=float)
        a, b = full[:k], part
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], atol=1e-12, rtol=0.0):
            bad = int(np.sum(~np.isclose(a[m], b[m], atol=1e-12, rtol=0.0)))
            raise AssertionError(f"causality FAIL at cut={cut}: {bad} bars differ")
    return True


def fee_at(market: MarketSpec, fee_rate: float) -> MarketSpec:
    """Same market spec, at a different taker fee (for the B5 cost-robustness check)."""
    return MarketSpec(name=market.name, leverage=market.leverage, fee_rate=fee_rate,
                      allow_short=market.allow_short,
                      maintenance_margin_rate=market.maintenance_margin_rate,
                      min_notional=market.min_notional, pays_funding=market.pays_funding)


# --------------------------------------------------------------- self-test

def _self_test() -> None:
    """Assert the identity points both branches depend on. Run on import."""
    idx = pd.date_range("2017-01-01", periods=120_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(92)
    # A mildly trending, positively-autocorrelated synthetic series so the
    # AR(1) fit and the derive_optimal_span plumbing have something non-
    # degenerate to chew on in the self-test.
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    high = close * (1.0 + np.abs(rng.normal(0, 0.0004, len(idx))))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0004, len(idx))))
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": 1.0}, index=idx)

    raw = v4_raw_desired(df)
    assert np.allclose(v4_target(df), apply_deadband(raw)), "v4_target != apply_deadband(v4_raw_desired)"
    assert np.array_equal(v4_vote_frac(df), vote_frac(df, V4_HORIZONS)), "vote_frac generalisation disagrees with v4"

    dr = daily_log_returns(df)
    z = vol_normalized_returns(dr)
    assert len(z) > 30
    phi, mu = fit_ar1(z)
    assert np.isfinite(phi) and -1.0 <= phi <= 1.0
    assert np.isfinite(mu)

    span, sr_curve, grid = derive_optimal_span(phi, mu)
    assert span == -1 or (grid[0] <= span <= grid[-1])

    nu = nu_of_span(np.array([20.0, 40.0, 80.0]))
    assert nu.shape == (3,)
    sr = sharpe_ar1(nu, phi=0.05, mu=0.001)
    assert sr.shape == (3,)

    assert causal_truncation_probe(v4_target, df)
    tiny_grid = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    ok, reason = kill_switch_a0(0.05, np.array([0.1, 0.3, 0.5, 0.6, 0.55, 0.4, 0.2, 0.1]),
                                grid=tiny_grid, edge_tolerance=1)
    assert ok  # interior peak at index 3/7
    ok2, _ = kill_switch_a0(-0.02, np.array([0.1, 0.3, 0.5, 0.6, 0.55, 0.4, 0.2, 0.1]),
                            grid=tiny_grid, edge_tolerance=1)
    assert not ok2  # phi <= 0
    ok3, _ = kill_switch_a0(0.05, np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]),
                            grid=tiny_grid, edge_tolerance=1)
    assert not ok3  # peak at the left edge
    ok4, _ = kill_switch_a0(0.05, np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]),
                            grid=tiny_grid, edge_tolerance=1)
    assert not ok4  # peak at the right edge (BTC's own inner-train case, see pilot check)


_self_test()
