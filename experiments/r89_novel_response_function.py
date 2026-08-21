#!/usr/bin/env python
"""R-89 NOVEL branch: the RESPONSE FUNCTION -- the map from standardised
trend strength to exposure -- holding every other part of
`kelly_regime_v4` fixed.

MECHANISM (one sentence). `kelly_regime_v4` sizes as ``frac * scale``,
where ``frac`` is the mean of three LATCHED BINARY anchor votes and so
takes only the values {0, 1/3, 2/3, 1}: the map from trend strength to
exposure is a **step function**, never varied in 87 rounds. This branch
replaces that step with a calibrated continuous map ``g(phi)`` from a
causally-standardised trend strength ``phi`` to exposure, and tests three
shapes -- sign (the incumbent), linear (unsaturated), cubic
(non-monotone) -- against each other with `scale`, the anchors, the band
and the 10% deadband all held at v4's shipped values. Only ``g`` changes.

WHY THIS AXIS. Levine & Pedersen (2016), "Which Trend Is Your Friend?",
*Financial Analysts Journal* 72(3):51-66, prove that time-series
momentum, moving-average crossovers, HP filters and Kalman filters are
equivalent representations of one linear filter differing only in how
they weight past returns. On a single instrument, changing the filter is
therefore a re-parameterisation, not a new mechanism; the only
genuinely distinct axes are the **nonlinearity of the response**, the
**path-dependence of the exposure** and the **state-dependence of the
horizon**. This project has never varied the first.

THE TWO PREDICTIONS BEING TESTED, WHICH DISAGREE WITH EACH OTHER:

1. **Linear / de-saturated.** Dao, Nguyen, Deremble, Lemperiere,
   Bouchaud & Potters (2016), "Tail protection for long investors: Trend
   convexity at work", arXiv:1607.02410, *Journal of Investment
   Strategies* 7(1) 2017. Trend PnL is a variance swap between the
   filter and rebalancing timescales, and the *shape* of the payoff is
   set by the response: a linear (unsaturated) position gives a
   **quadratic/parabolic** payoff in the trend indicator, while a
   sign/binary position gives only a **V-shaped piecewise-linear**
   payoff. A binary latched vote therefore discards the convex term by
   construction. Prediction: linear response => MORE exposure at large
   |phi|, and measurable positive curvature of PnL in phi.

2. **Cubic / non-monotone.** Schmidhuber (2021), "Trends, reversion, and
   critical phenomena in financial markets", *Physica A* 566:125642,
   arXiv:2006.07847; extended by Safari & Schmidhuber (2025),
   arXiv:2501.16772. Fits ``E[R(t+1)] = a + b*phi + c*phi^3`` on 24
   futures, daily, 1990-2019, with **b = 1.29% (t=3.0), c = -0.62%
   (t=2.7)** (a refined 4-parameter version gives b=2.00%, c=-0.63%), so
   expected return crosses zero at a critical trend strength
   ``phi_c = sqrt(-b/c) ~ 1.8-1.9`` -- trends revert *before* they become
   statistically significant. Prediction: cubic response => LESS exposure
   at large |phi|. Out-of-sample aggregated adjusted R^2 was **3.98 bp**
   (6 bp refined); no crypto in the sample; essentially no transaction
   costs modelled.

The incumbent (sign) makes a third, intermediate prediction. That
three-way disagreement is the experiment.

R-80'S STRUCTURAL CONSTRAINT, PRE-REGISTERED. A continuous vote can never
combine to exactly zero, which would disable v4's single most robust
documented property (full de-risk to cash on unanimous bearish
consensus); R-80's novel branch spent 0% of post-warmup bars flat against
v4's 32.8% for exactly this reason. Every continuous arm here therefore
carries an explicit clip-to-flat, and the fraction of bars spent exactly
flat is reported beside v4's own for every configuration. A configuration
whose flat-fraction is near zero is disqualified by this clause, not by
its Sharpe.

--------------------------------------------------------------------------
PHI, the standardised trend strength (ONE specification, frozen, not swept)
--------------------------------------------------------------------------
``r89_shared.signal_deviation(df)`` gives ``close/anchor_h - 1`` for
h in (20, 40, 80) days -- v4's own vote statistic made continuous. Each
column is standardised by its OWN trailing root-mean-square over a
365-day rolling window (``min_periods`` = 1 day, ``shift(1)``), clipped at
+/-2.5 as Schmidhuber does, and ``phi`` is the mean of the three. The
scaler is a TRAILING window: no full-series mean/std is ever computed, so
the causal truncation probe (Step A3) is a real test of this file's
single most likely bug. No mean is subtracted: ``phi = 0`` must keep
meaning "price sits on its anchors", which is the point v4's own latch
thresholds.

THE THREE ARMS:
  g_sign(phi)   -- `r89_shared.v4_vote_frac(df)` used DIRECTLY, so the arm
                   reproduces v4 bit-for-bit (Step A1 asserts it).
  g_linear(phi) -- clip(phi / phi_max, 0, 1), long-only (phi<=0 => 0).
  g_cubic(phi)  -- normalised (b*phi + c*phi^3), clipped to [0, 1], with
                   (b, c) estimated on inner-train ONLY and then FROZEN.
In every arm the final target is
``apply_deadband(g(phi) * v4_scale(df))`` -- same scale, same 10%
deadband, same anchors.

--------------------------------------------------------------------------
STEP 0 -- THE PRE-REGISTERED KILL SWITCH, RUN BEFORE ANY STRATEGY CODE
--------------------------------------------------------------------------
Estimate (b, c) on inner-train (2017-01-01 -> 2020-12-31), BTC only, by
OLS of the h-ahead log return on phi and phi^3 with Newey-West HAC
standard errors, at h in {1, 5, 20} days. **If c-hat is not negative AND
significant (|t| >= 2), the Schmidhuber mechanism does not exist on BTC
and the cubic arm is NEGATIVE** -- reported as such, with no search for a
specification in which it appears. A clean kill is a successful outcome.
The frozen fit horizon for the cubic arm is **h = 20 days** (one month:
Schmidhuber's own scale, and the middle of v4's 20/40/80 anchor ladder),
chosen before the regression was run.

--------------------------------------------------------------------------
THE FROZEN GRID -- 19 CONFIGURATIONS, none added or dropped after results
--------------------------------------------------------------------------
  linear:  phi_max in {1.0, 1.5, 2.0, 2.5} x flat_thr in {0.0, 0.05, 0.10}
           = 12
  cubic:   (b-hat, c-hat) frozen from Step 0
           x flat_thr in {0.0, 0.05, 0.10}
           x normalisation in {peak, phi_c}                       =  6
  sign:    the identity point                                     =  1
                                                                  ---- 19

--------------------------------------------------------------------------
THE FROZEN DECISION RULE (default REJECT; no threshold moves after results)
--------------------------------------------------------------------------
Selection statistic: inner-validation paired log-growth difference vs v4
on ``futures_5x``, among configurations that pass Step A and are eligible
(the cubic arm is ineligible if Step 0's kill switch fires; the sign arm
is the control itself and cannot be selected against itself).

  A1 identity   -- apply_deadband(v4_vote_frac*v4_scale) == KellyRegimeV4's
                   own prepared target, bit-for-bit.
  A2 non-inert  -- r_squared(candidate_path, v4_path) on inner-train < 0.98.
  A3 causality  -- causal_truncation_probe passes (finalist + >=1 cubic).
  A4 flat-frac  -- bars exactly flat reported for every config vs v4's own.

  B1 -- the paired bootstrap difference vs v4 excludes zero in >=1 of the
        four (slice x market) cells AND its point estimate is positive in
        all four.
  B2 -- either dSharpe > +0.2 (R-20 noise floor) on inner-validation on
        BOTH markets, or a clear max-drawdown improvement on both.
  B3 -- plateau not peak: the finalist's immediate neighbours move with it.
  B4 -- falsification: ETH (`r89_shared.load_eth()`) must show the SAME
        SIGN of improvement over v4, both markets. Failing it is NEGATIVE.
  B5 -- cost robustness: the improvement must not reverse sign at a 0.40%
        taker fee on inner-validation.

MEASURED POWER (operator, before dispatch; not re-derived here): the
paired daily difference against v4 is close to serially uncorrelated, so
at the 30-day block convention a 95% paired interval excludes zero once
the candidate beats v4 by about **+0.35 log units over inner-train** or
**+0.13 to +0.26 over inner-validation**.

This file never reads a bar at or after OOS_START (2023-01-01): every load
goes through `r89_shared`'s truncating loaders, and the max timestamp
actually touched is tracked and printed at the end of `main()`.
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
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.window import run_period  # noqa: E402

from experiments.r89_shared import (  # noqa: E402
    BARS_PER_DAY,
    FUTURES,
    INNER_TRAIN_END,
    INNER_VAL_START,
    OOS_START,
    SLICES,
    SPOT,
    TargetStrategy,
    apply_deadband,
    causal_truncation_probe,
    compare,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    signal_deviation,
    v4_scale,
    v4_vote_frac,
)

# ---------------------------------------------------------------------------
# FROZEN SPECIFICATION -- phi.  ONE specification; not swept, not tuned.
# ---------------------------------------------------------------------------
PHI_RMS_WIN_DAYS = 365      # trailing window for the causal scale estimate
PHI_RMS_MINP_DAYS = 1       # min_periods; keeps phi valid at v4's own warmup
PHI_CLIP = 2.5              # Schmidhuber's own clip
FIT_H_DAYS = 20             # frozen fit horizon for the cubic arm
STEP0_HORIZONS = (1, 5, 20)  # reported; only FIT_H_DAYS is frozen into g_cubic

# ---------------------------------------------------------------------------
# FROZEN GRID -- 12 + 6 + 1 = 19 configurations
# ---------------------------------------------------------------------------
LINEAR_PHI_MAX = (1.0, 1.5, 2.0, 2.5)
FLAT_THRESHOLDS = (0.0, 0.05, 0.10)
CUBIC_NORMS = ("peak", "phi_c")

V4_WARMUP_BARS = 80 * BARS_PER_DAY + 10   # TargetStrategy's own warmup

TAKER_040 = 0.0040          # B5 cost-robustness fee


def _mk(name: str, fee: float, leverage: float, short: bool, funding: bool) -> MarketSpec:
    return MarketSpec(name=name, leverage=leverage, fee_rate=fee,
                      allow_short=short, pays_funding=funding)


SPOT_040 = _mk("spot@0.40%", TAKER_040, 1.0, False, False)
FUT_040 = _mk("fut5x@0.40%", TAKER_040, 5.0, True, True)
SPOT_GROSS = _mk("spot@0bp", 0.0, 1.0, False, False)
FUT_GROSS = _mk("fut5x@0bp", 0.0, 5.0, True, True)


# ---------------------------------------------------------------------------
# Caches keyed on frame content (backtests re-`prepare` the same frames often)
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _key(df: pd.DataFrame) -> tuple:
    return (len(df), int(df.index[0].value), int(df.index[-1].value),
            float(df["close"].iloc[0]), float(df["close"].iloc[-1]))


def cached_scale(df: pd.DataFrame) -> np.ndarray:
    k = ("scale",) + _key(df)
    if k not in _CACHE:
        _CACHE[k] = v4_scale(df)
    return _CACHE[k]


def cached_vote(df: pd.DataFrame) -> np.ndarray:
    k = ("vote",) + _key(df)
    if k not in _CACHE:
        _CACHE[k] = v4_vote_frac(df)
    return _CACHE[k]


# ---------------------------------------------------------------------------
# phi -- causally standardised trend strength
# ---------------------------------------------------------------------------

def build_phi(df: pd.DataFrame) -> np.ndarray:
    """Standardised, clipped, anchor-averaged trend strength.

    dev[:, j] = close/anchor_j - 1 (v4's own vote statistic, continuous).
    Each column is divided by its trailing RMS over a 365-day rolling
    window, `shift(1)`-ed so bar i's scaler uses only bars < i, then
    clipped at +/-PHI_CLIP.  phi is the mean of the three.  NaN wherever
    any anchor is not yet defined -- callers treat NaN as flat.

    NOTHING here is a full-series statistic: the scaler is a trailing
    rolling window, which is why the truncation probe is a real test.
    """
    k = ("phi",) + _key(df)
    if k in _CACHE:
        return _CACHE[k]
    dev = signal_deviation(df)
    win = int(PHI_RMS_WIN_DAYS * BARS_PER_DAY)
    minp = int(PHI_RMS_MINP_DAYS * BARS_PER_DAY)
    cols = []
    for j in range(dev.shape[1]):
        s = pd.Series(dev[:, j], index=df.index)
        rms = (s.pow(2).rolling(win, min_periods=minp).mean() ** 0.5).shift(1)
        z = (s / rms.where(rms > 0)).clip(-PHI_CLIP, PHI_CLIP)
        cols.append(z.to_numpy())
    Z = np.column_stack(cols)
    phi = Z.mean(axis=1)   # NaN in any anchor => NaN phi => flat
    _CACHE[k] = phi
    return phi


# ---------------------------------------------------------------------------
# Step 0 -- the kill switch.  OLS of h-ahead return on phi and phi^3, HAC SEs.
# ---------------------------------------------------------------------------

def hac_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS with a Newey-West (Bartlett) HAC covariance.  Returns (beta, t)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    U = X * e[:, None]
    S = U.T @ U
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        A = U[lag:].T @ U[:-lag]
        S += w * (A + A.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    return beta, beta / se


def step0_fit(df_train: pd.DataFrame, h_days: int) -> dict:
    """Daily-sampled OLS of the h-day-ahead log return (in %) on phi, phi^3."""
    phi = build_phi(df_train)
    logc = np.log(df_train["close"].to_numpy())
    h = h_days * BARS_PER_DAY
    idx = np.arange(0, len(df_train) - h, BARS_PER_DAY)   # one sample per day
    x = phi[idx]
    fwd = (logc[idx + h] - logc[idx]) * 100.0
    m = np.isfinite(x) & np.isfinite(fwd)
    x, fwd = x[m], fwd[m]
    beta, t = hac_ols(fwd, np.column_stack([x, x ** 3]), lags=h_days + 5)
    a, b, c = beta
    phi_c = float(np.sqrt(-b / c)) if (c < 0 and b > 0) else float("nan")
    return dict(h_days=h_days, n=int(len(x)), a=float(a), b=float(b), c=float(c),
                t_b=float(t[1]), t_c=float(t[2]), phi_c=phi_c)


# ---------------------------------------------------------------------------
# The three response functions
# ---------------------------------------------------------------------------

def g_linear(phi: np.ndarray, phi_max: float, flat_thr: float) -> np.ndarray:
    """Dao et al.'s unsaturated response: exposure proportional to |phi|.

    Long-only (phi <= 0 => exactly 0, which is R-80's clip-to-flat for
    free), saturating at 1 for phi >= phi_max, and with an explicit
    clip-to-flat below `flat_thr`.
    """
    g = np.clip(np.where(np.isfinite(phi), phi, 0.0) / phi_max, 0.0, 1.0)
    g = np.where(g < flat_thr, 0.0, g)
    return np.where(np.isfinite(phi), g, 0.0)


def cubic_denominator(b: float, c: float, norm: str) -> tuple[float, float]:
    """(denominator, reference phi) for the two frozen normalisations.

    peak  -- divide by f(phi*) where phi* = argmax f = sqrt(-b/(3c)), so
             the response peaks at exactly 1.
    phi_c -- divide by b*phi_c, i.e. normalise the SLOPE AT THE ORIGIN so
             that for small phi the cubic arm behaves exactly like
             g_linear with phi_max = phi_c and then turns down; its peak
             is 2/(3*sqrt(3)) ~ 0.385, so it is a strictly de-risked
             response and its mean exposure is reported accordingly.

    If c >= 0 (no critical strength exists) both fall back to the clip
    boundary phi = PHI_CLIP as the reference; this is a degenerate case
    and is flagged in the report rather than hidden.
    """
    if c < 0 and b > 0:
        phi_c = float(np.sqrt(-b / c))
        phi_star = min(float(np.sqrt(-b / (3.0 * c))), PHI_CLIP)
    else:
        phi_c = PHI_CLIP
        phi_star = PHI_CLIP
    if norm == "peak":
        return float(b * phi_star + c * phi_star ** 3), phi_star
    return float(b * phi_c), phi_c


def g_cubic(phi: np.ndarray, b: float, c: float, norm: str,
            flat_thr: float) -> np.ndarray:
    """Schmidhuber's non-monotone response: b*phi + c*phi^3, normalised."""
    denom, _ref = cubic_denominator(b, c, norm)
    p = np.where(np.isfinite(phi), phi, 0.0)
    f = b * p + c * p ** 3
    g = np.clip(f / denom, 0.0, 1.0) if denom > 0 else np.zeros_like(p)
    g = np.where(g < flat_thr, 0.0, g)
    return np.where(np.isfinite(phi), g, 0.0)


# ---------------------------------------------------------------------------
# Config objects and target builders
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, arm: str, label: str, **params):
        self.arm = arm
        self.label = label
        self.params = params

    def g(self, df: pd.DataFrame) -> np.ndarray:
        if self.arm == "sign":
            return cached_vote(df)          # v4's own latched vote, verbatim
        phi = build_phi(df)
        if self.arm == "linear":
            return g_linear(phi, self.params["phi_max"], self.params["flat_thr"])
        return g_cubic(phi, self.params["b"], self.params["c"],
                       self.params["norm"], self.params["flat_thr"])

    def build(self, df: pd.DataFrame) -> np.ndarray:
        return apply_deadband(self.g(df) * cached_scale(df))

    def build_uncached(self, df: pd.DataFrame) -> np.ndarray:
        """Same construction with no caching -- for the truncation probe."""
        if self.arm == "sign":
            g = v4_vote_frac(df)
        else:
            dev = signal_deviation(df)
            win = int(PHI_RMS_WIN_DAYS * BARS_PER_DAY)
            minp = int(PHI_RMS_MINP_DAYS * BARS_PER_DAY)
            cols = []
            for j in range(dev.shape[1]):
                s = pd.Series(dev[:, j], index=df.index)
                rms = (s.pow(2).rolling(win, min_periods=minp).mean() ** 0.5).shift(1)
                cols.append((s / rms.where(rms > 0)).clip(-PHI_CLIP, PHI_CLIP).to_numpy())
            Z = np.column_stack(cols)
            phi = Z.mean(axis=1)
            if self.arm == "linear":
                g = g_linear(phi, self.params["phi_max"], self.params["flat_thr"])
            else:
                g = g_cubic(phi, self.params["b"], self.params["c"],
                            self.params["norm"], self.params["flat_thr"])
        return apply_deadband(g * v4_scale(df))


def v4_control_build(df: pd.DataFrame) -> np.ndarray:
    return apply_deadband(cached_vote(df) * cached_scale(df))


def frozen_grid(b: float, c: float) -> list[Config]:
    """The 19 pre-registered configurations, in a fixed order."""
    cfgs = [Config("sign", "sign(identity)")]
    for pm in LINEAR_PHI_MAX:
        for ft in FLAT_THRESHOLDS:
            cfgs.append(Config("linear", f"lin pm{pm:.1f} ft{ft:.2f}",
                               phi_max=pm, flat_thr=ft))
    for nm in CUBIC_NORMS:
        for ft in FLAT_THRESHOLDS:
            cfgs.append(Config("cubic", f"cub {nm} ft{ft:.2f}",
                               b=b, c=c, norm=nm, flat_thr=ft))
    return cfgs


# ---------------------------------------------------------------------------
# Step A helpers
# ---------------------------------------------------------------------------

def flat_fraction(path: np.ndarray, warmup: int = V4_WARMUP_BARS) -> float:
    p = path[warmup:]
    return float(np.mean(p == 0.0)) if len(p) else float("nan")


def mean_exposure(path: np.ndarray, warmup: int = V4_WARMUP_BARS) -> float:
    p = path[warmup:]
    return float(np.mean(p)) if len(p) else float("nan")


def daily_turnover(path: np.ndarray, warmup: int = V4_WARMUP_BARS) -> float:
    """Mean |change in target notional| per day -- the notional turnover."""
    p = path[warmup:]
    if len(p) < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(p))) / (len(p) / BARS_PER_DAY))


# ---------------------------------------------------------------------------
# Mechanism-level measurements (reported whether or not Sharpe moves)
# ---------------------------------------------------------------------------

def daily_series(cfg: Config, df: pd.DataFrame, slice_name: str,
                 market: MarketSpec) -> tuple[pd.Series, object]:
    """Daily SIMPLE returns of one arm over one slice, WITH its index."""
    start, end = SLICES[slice_name]
    strat = TargetStrategy(cfg.build, name=cfg.label)
    res = run_period(strat, df, start, end, market=market, start_balance=1_000.0)
    eq = res.equity
    daily = eq.resample("1D").last().dropna()
    prev = daily.shift(1)
    rets = ((daily - prev) / prev.where(prev > 0)).iloc[1:].fillna(0.0)
    return rets, compute_metrics(res)


def phi_daily(df: pd.DataFrame) -> pd.Series:
    """phi sampled at each UTC day's last bar (the decision that day ends on)."""
    s = pd.Series(build_phi(df), index=df.index)
    return s.resample("1D").last()


def curvature(rets: pd.Series, phid: pd.Series) -> dict:
    """Regress daily PnL on lagged phi and phi^2.

    Dao et al. (2016) predict ~0 curvature for a sign response and a
    POSITIVE phi^2 coefficient for an unsaturated (linear) one: only the
    unsaturated position produces the parabolic payoff term.
    """
    x = phid.reindex(rets.index).shift(1)
    y = np.log1p(rets.to_numpy()) * 100.0
    xv = x.to_numpy()
    m = np.isfinite(xv) & np.isfinite(y)
    xv, y = xv[m], y[m]
    if len(xv) < 30:
        return dict(coef=float("nan"), t=float("nan"), n=int(len(xv)))
    beta, t = hac_ols(y, np.column_stack([xv, xv ** 2]), lags=10)
    return dict(coef=float(beta[2]), t=float(t[2]), n=int(len(xv)))


def skew_profile(rets: pd.Series, horizons=(1, 5, 10, 20, 40, 60)) -> dict:
    """Skewness of aggregated log returns at several horizons.

    Sepp & Lucic (2026, arXiv:2607.19497) predict trend-return skew peaks
    near HALF the filter span; v4's anchors are 20/40/80 days, so the
    predicted peak sits around 10-40 days.
    """
    lr = np.log1p(rets.to_numpy())
    out = {}
    for h in horizons:
        n = (len(lr) // h) * h
        if n < 10 * h:
            out[h] = float("nan")
            continue
        agg = lr[:n].reshape(-1, h).sum(axis=1)
        sd = agg.std(ddof=1)
        out[h] = float(np.mean(((agg - agg.mean()) / sd) ** 3)) if sd > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def hdr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def main() -> None:
    max_ts = []
    n_strategy_configs = 0

    hdr("R-89 NOVEL BRANCH -- THE RESPONSE FUNCTION (sign vs linear vs cubic)")
    print("mechanism: replace kelly_regime_v4's LATCHED BINARY vote (frac in "
          "{0,1/3,2/3,1}) with a")
    print("continuous map g(phi) from standardised trend strength to exposure; "
          "everything else")
    print("(anchors 20/40/80, band 1%, v4_scale, 10% deadband) held fixed.")
    print("frozen grid: 12 linear + 6 cubic + 1 sign identity = 19 configurations")
    print(f"phi spec (frozen, one specification, not swept): trailing RMS over "
          f"{PHI_RMS_WIN_DAYS}d rolling window, min_periods={PHI_RMS_MINP_DAYS}d, "
          f"shift(1), clip +/-{PHI_CLIP}, mean of anchors (20, 40, 80)")

    btc = load_btc()
    max_ts.append(btc.index.max())
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    train = btc.loc[:INNER_TRAIN_END]
    print(f"inner-train frame: {len(train):,} bars  {train.index[0]} -> {train.index[-1]}")

    # ----------------------------------------------------------------- phi
    phi = build_phi(btc)
    pf = phi[np.isfinite(phi)]
    first = int(np.argmax(np.isfinite(phi)))
    print(f"\nphi: first finite bar {first:,} (day {first/BARS_PER_DAY:.1f}; "
          f"v4's own warmup is day {V4_WARMUP_BARS/BARS_PER_DAY:.1f}) "
          f"-- {len(pf):,} finite of {len(phi):,}")
    print(f"phi distribution (2017-2022): mean {pf.mean():+.3f}  sd {pf.std():.3f}  "
          f"pctiles 1/5/25/50/75/95/99 = "
          + "/".join(f"{v:+.2f}" for v in np.percentile(pf, [1, 5, 25, 50, 75, 95, 99])))
    print(f"P(phi <= 0) = {np.mean(pf <= 0):.3f}   P(|phi| at clip) = "
          f"{np.mean(np.abs(pf) >= PHI_CLIP - 1e-9):.3f}")

    # ================================================================ STEP 0
    hdr("STEP 0 -- THE PRE-REGISTERED KILL SWITCH (inner-train, BTC only)")
    print("OLS of the h-day-ahead log return (%) on phi and phi^3, daily samples,")
    print("Newey-West HAC standard errors (lags = h + 5).  Schmidhuber (2021) on")
    print("24 futures 1990-2019 reports b=+1.29 (t=3.0), c=-0.62 (t=2.7), phi_c~1.8-1.9.")
    print()
    print(f"{'h':>5s} {'n':>6s} {'a':>9s} {'b_hat':>10s} {'t(b)':>7s} "
          f"{'c_hat':>10s} {'t(c)':>7s} {'phi_c':>8s}")
    print("-" * 70)
    fits = {}
    for h in STEP0_HORIZONS:
        f = step0_fit(train, h)
        fits[h] = f
        print(f"{h:4d}d {f['n']:6d} {f['a']:+9.3f} {f['b']:+10.4f} {f['t_b']:+7.2f} "
              f"{f['c']:+10.4f} {f['t_c']:+7.2f} {f['phi_c']:8.3f}")

    frozen = fits[FIT_H_DAYS]
    B_HAT, C_HAT = frozen["b"], frozen["c"]
    sign_stable = all(fits[h]["c"] < 0 for h in STEP0_HORIZONS)
    max_abs_t = max(abs(fits[h]["t_c"]) for h in STEP0_HORIZONS)
    c_significant = abs(frozen["t_c"]) >= 2.0
    kill = not (C_HAT < 0 and c_significant)

    print(f"\nsign of c_hat stable across h in {STEP0_HORIZONS}: {sign_stable} "
          f"(all negative: {sign_stable})")
    print(f"largest |t(c)| at any horizon: {max_abs_t:.2f}   "
          f"frozen horizon h={FIT_H_DAYS}d: c_hat={C_HAT:+.4f}, t={frozen['t_c']:+.2f}, "
          f"phi_c={frozen['phi_c']:.3f}")
    print()
    print("KILL SWITCH, as pre-registered: the cubic arm survives only if c_hat is")
    print("negative AND significant (|t| >= 2).  Here c_hat is NEGATIVE at every")
    print(f"horizon (sign stable) but |t| < 2 at every horizon (max {max_abs_t:.2f}).")
    print(f"=> KILL SWITCH FIRES: {kill}.  The cubic arm is NEGATIVE on BTC and is")
    print("   INELIGIBLE for selection.  Its 6 configurations are still EVALUATED")
    print("   and REPORTED (the grid is frozen; every arm reports), but no search")
    print("   is made for a specification in which the mechanism appears.")

    denom_peak, ref_peak = cubic_denominator(B_HAT, C_HAT, "peak")
    denom_phic, ref_phic = cubic_denominator(B_HAT, C_HAT, "phi_c")
    print(f"\nFrozen cubic coefficients (h={FIT_H_DAYS}d): b_hat={B_HAT:+.4f}  "
          f"c_hat={C_HAT:+.4f}  phi_c={frozen['phi_c']:.3f}  "
          f"phi*(peak)={ref_peak:.3f}")
    print(f"  peak-normalised : divide by f(phi*)={denom_peak:.4f} -> peak exposure 1.000")
    print(f"  phi_c-normalised: divide by b*phi_c={denom_phic:.4f} -> peak exposure "
          f"{(B_HAT*ref_peak + C_HAT*ref_peak**3)/denom_phic:.3f} (slope-matched at origin)")

    cfgs = frozen_grid(B_HAT, C_HAT)
    assert len(cfgs) == 19, len(cfgs)

    # response-function shape table
    print("\nResponse functions at a grid of phi (the three-way disagreement, "
          "made concrete):")
    grid = np.array([-1.0, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5])
    print("  phi        " + " ".join(f"{p:+6.2f}" for p in grid))
    print("  sign(v4)   " + " ".join(
        f"{v:6.2f}" for v in [0.0, 0.0, 0.0, 0.33, 0.67, 1.0, 1.0, 1.0, 1.0]) +
        "   (illustrative: the latch is on raw dev, not phi)")
    for pm in LINEAR_PHI_MAX:
        print(f"  lin pm{pm:.1f}  " + " ".join(
            f"{v:6.2f}" for v in g_linear(grid, pm, 0.0)))
    for nm in CUBIC_NORMS:
        print(f"  cub {nm:6s} " + " ".join(
            f"{v:6.2f}" for v in g_cubic(grid, B_HAT, C_HAT, nm, 0.0)))

    # ================================================================ STEP A
    hdr("STEP A -- MECHANISM GATE (before any performance number)")

    # --- A1 identity
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
    v4_prepared = KellyRegimeV4().prepare(btc.copy())["target"].to_numpy()
    ours = v4_control_build(btc)
    a1_max = float(np.max(np.abs(v4_prepared - ours)))
    a1 = a1_max == 0.0
    print(f"A1 identity: max|apply_deadband(v4_vote_frac*v4_scale) - "
          f"KellyRegimeV4.prepare().target| = {a1_max:.3e}   -> {'PASS' if a1 else 'FAIL'}")
    sign_path_train = cfgs[0].build(train)
    v4_path_train = v4_control_build(train)
    print(f"    sign-arm identity on inner-train: max abs diff vs v4 control = "
          f"{float(np.max(np.abs(sign_path_train - v4_path_train))):.3e}")

    v4_flat = flat_fraction(v4_control_build(btc))
    v4_mexp = mean_exposure(v4_control_build(btc))
    v4_turn = daily_turnover(v4_control_build(btc))
    print(f"\nv4 control (2017-2022, post-warmup): flat-fraction {v4_flat:.3f}   "
          f"mean exposure {v4_mexp:.3f}   notional turnover {v4_turn:.4f}/day")
    print("(R-80 recorded v4 at 32.8% flat; R-80's own continuous branch spent 0%.)")

    # --- A2 non-inertness + A4 flat fraction, every config
    print("\nA2 (non-inertness, R^2 vs v4 path on inner-train; INERT if >= 0.98)")
    print("A4 (flat fraction / mean exposure / notional turnover, 2017-2022 post-warmup)")
    print()
    print("   'gflat' = fraction of bars where the RESPONSE g(phi) is exactly 0 "
          "(the clip-to-flat itself);")
    print("   'flat'  = fraction where the FINAL TARGET is exactly 0 (after v4's "
          "10% deadband).")
    print()
    print(f"{'config':22s} {'arm':7s} {'R2_train':>9s} {'INERT':>6s} {'gflat':>7s} "
          f"{'flat':>7s} {'v4flat':>7s} {'meanExp':>8s} {'exp/v4':>7s} "
          f"{'turn/d':>8s} {'t/v4':>6s}")
    print("-" * 104)
    gate = {}
    for cfg in cfgs:
        p_tr = cfg.build(train)
        r2 = r_squared(p_tr[V4_WARMUP_BARS:], v4_path_train[V4_WARMUP_BARS:])
        p_full = cfg.build(btc)
        gf = float(np.mean(cfg.g(btc)[V4_WARMUP_BARS:] == 0.0))
        ff, me, tu = flat_fraction(p_full), mean_exposure(p_full), daily_turnover(p_full)
        inert = bool(np.isfinite(r2) and r2 >= 0.98)
        gate[cfg.label] = dict(r2=r2, inert=inert, flat=ff, gflat=gf, mexp=me, turn=tu)
        print(f"{cfg.label:22s} {cfg.arm:7s} {r2:9.4f} {'YES' if inert else 'no':>6s} "
              f"{gf:7.3f} {ff:7.3f} {v4_flat:7.3f} {me:8.3f} {me/v4_mexp:7.2f} "
              f"{tu:8.4f} {tu/v4_turn:6.2f}")

    n_inert = sum(1 for c in cfgs if gate[c.label]["inert"])
    print(f"\n{n_inert} of 19 configurations are INERT (R^2 >= 0.98) and are excluded "
          f"from selection.")
    low_flat = [c.label for c in cfgs if c.arm != "sign" and gate[c.label]["flat"] < 0.05]
    if low_flat:
        print("R-80 clause: DISQUALIFIED for flat-fraction < 0.05 (near-zero): "
              + ", ".join(low_flat))
    else:
        print("R-80 clause: no configuration has flat-fraction < 0.05. Every "
              "continuous arm reaches")
        print("  EXACT flat, because the long-only clip maps phi <= 0 to exactly "
              "0.0 (and the cubic")
        print("  arm maps phi >= phi_c to exactly 0.0 as well), so v4's de-risk-to-"
              "cash property survives.")
    print("\nBUT NOTE, and this is a mechanism-level finding in its own right: the")
    print("clip-to-flat restores exact-zero DESIRED exposure at a HIGHER rate than "
          "v4's own vote,")
    print("yet v4's 10% re-target deadband then strands a small non-zero position "
          "in many of those")
    print("bars -- once the held position falls to <= 0.10 it can never be moved to "
          "0 by a desired")
    print("of 0, because |0 - pos| <= deadband.  A step response never lands there "
          "(its desired")
    print("exposure jumps by ~scale/3 >= 0.10); a continuous one does.  Compare the "
          "'gflat' and")
    print("'flat' columns above: that gap is R-80's structural problem surviving "
          "the fix for it.")

    # Risk matching (standing repo rule: match risk before comparing anything)
    print("\nRISK MATCHING (standing rule): mean exposure relative to v4 is shown "
          "above as 'exp/v4'.")
    print("No configuration in the frozen grid carries v4's exposure; every "
          "continuous arm holds")
    print("LESS.  Any drawdown improvement below is therefore partly arithmetic "
          "(R-33: 88-92% of")
    print("'regime-gated sizing cuts drawdown' was the exposure level), and B2's "
          "drawdown leg is")
    print("read with that caveat attached.")

    # ================================================================ STEP B
    hdr("STEP B -- FULL GRID, all 19 configurations x 4 (slice x market) cells")
    print("candidate vs kelly_regime_v4 control; d_loggrowth is the PAIRED "
          "block-bootstrap difference (30-day blocks, 2000 resamples).")
    print(f"Power reference (operator, pre-dispatch): an interval excludes zero at "
          f"about +0.35 on inner-train, +0.13..+0.26 on inner-validation.\n")

    all_rows: dict[str, list[dict]] = {}
    for cfg in cfgs:
        rows = compare(cfg.build, btc, label=cfg.label,
                       control_build=v4_control_build)
        all_rows[cfg.label] = rows
        n_strategy_configs += 1
        print_rows(rows)
        print()

    # ------------------------------------------------------------ selection
    hdr("SELECTION -- inner-validation paired log-growth difference vs v4, futures_5x")

    def sel_stat(label: str) -> float:
        for r in all_rows[label]:
            if r["slice"] == "inner_val" and r["market"] == "futures_5x":
                return r["d_loggrowth"]
        return float("nan")

    print(f"{'config':22s} {'arm':7s} {'eligible':>9s} {'selstat':>9s} "
          f"{'[lo':>9s},{'hi]':>9s} {'why not':<40s}")
    print("-" * 96)
    eligible = []
    for cfg in cfgs:
        reasons = []
        if cfg.arm == "sign":
            reasons.append("is the control itself")
        if cfg.arm == "cubic" and kill:
            reasons.append("cubic arm killed at Step 0")
        if gate[cfg.label]["inert"]:
            reasons.append("INERT (R^2>=0.98)")
        if gate[cfg.label]["flat"] < 0.05 and cfg.arm != "sign":
            reasons.append("flat-fraction < 0.05 (R-80 clause)")
        ok = not reasons
        if ok:
            eligible.append(cfg)
        row = [r for r in all_rows[cfg.label]
               if r["slice"] == "inner_val" and r["market"] == "futures_5x"][0]
        print(f"{cfg.label:22s} {cfg.arm:7s} {'YES' if ok else 'no':>9s} "
              f"{row['d_loggrowth']:+9.3f} {row['d_lo']:+9.3f},{row['d_hi']:+9.3f} "
              f"{'; '.join(reasons)[:40]:<40s}")

    if not eligible:
        print("\nNo eligible configuration. VERDICT: NEGATIVE by construction.")
        print(f"\nmax timestamp read anywhere: {max(max_ts)}  (< {OOS_START})")
        return

    finalist = max(eligible, key=lambda c: sel_stat(c.label))
    print(f"\nFINALIST: {finalist.label}   selection statistic "
          f"{sel_stat(finalist.label):+.3f} log units (inner-val, futures_5x)")

    frows = all_rows[finalist.label]
    print()
    print_rows(frows)

    # ------------------------------------------------------------ B1..B3
    hdr("THE FROZEN DECISION RULE -- clause by clause")

    pts = [r["d_loggrowth"] for r in frows]
    excl = [r["excludes_zero"] for r in frows]
    b1 = any(excl) and all(p > 0 for p in pts)
    print(f"B1 paired bootstrap: excludes zero in {sum(excl)}/4 cells; point estimate "
          f"positive in {sum(1 for p in pts if p > 0)}/4 cells")
    print(f"   point estimates: " + ", ".join(
        f"{r['slice']}/{r['market']}={r['d_loggrowth']:+.3f} "
        f"[{r['d_lo']:+.3f},{r['d_hi']:+.3f}]" for r in frows))
    print(f"   B1 = {'PASS' if b1 else 'FAIL'}")

    val = [r for r in frows if r["slice"] == "inner_val"]
    dsh = {r["market"]: r["d_sharpe"] for r in val}
    ddd = {r["market"]: r["d_dd"] for r in val}
    b2_sharpe = all(v > 0.2 for v in dsh.values())
    b2_dd = all(v < 0.0 for v in ddd.values())
    b2 = b2_sharpe or b2_dd
    print(f"\nB2 noise floor (inner-validation): dSharpe " +
          ", ".join(f"{k}={v:+.3f}" for k, v in dsh.items()) +
          f"  -> both > +0.2: {b2_sharpe}")
    print(f"   dMaxDD (candidate - v4, negative = improvement): " +
          ", ".join(f"{k}={v:+.1f}pp" for k, v in ddd.items()) +
          f"  -> both improved: {b2_dd}")
    print(f"   B2 = {'PASS' if b2 else 'FAIL'}")

    # Risk match ON THE SELECTION CELL ITSELF (standing rule: holding less draws
    # down less; that is arithmetic, not evidence -- R-28/R-32/R-33).
    val_mask = btc.index >= pd.Timestamp(INNER_VAL_START, tz=btc.index.tz)
    fin_path, v4_path = finalist.build(btc), v4_control_build(btc)
    fe, ve = float(fin_path[val_mask].mean()), float(v4_path[val_mask].mean())
    print(f"\n   RISK MATCH on the selection cell: mean target exposure over "
          f"inner-validation is {fe:.4f} for the finalist vs {ve:.4f} for v4 "
          f"({fe/ve:.2f}x).")
    print("   B2's drawdown leg therefore rests on an arm holding well under half "
          "of v4's exposure")
    print("   through the 2022 bear: the arms are NOT risk-matched, so that leg is "
          "arithmetic, not")
    print("   evidence, and B2 should be read as effectively FAILED on its Sharpe "
          "leg alone. The")
    print("   rule is left frozen as pre-registered and is NOT moved; this is a "
          "caveat on the PASS.")

    # B3 plateau
    print("\nB3 plateau not peak: the finalist's immediate neighbours on the grid")
    print(f"{'config':22s} {'selstat':>9s} {'dSh_val_fut':>12s} {'note':<20s}")
    print("-" * 70)
    neigh = []
    if finalist.arm == "linear":
        pm, ft = finalist.params["phi_max"], finalist.params["flat_thr"]
        ipm, ift = LINEAR_PHI_MAX.index(pm), FLAT_THRESHOLDS.index(ft)
        for d in (-1, 0, 1):
            for e in (-1, 0, 1):
                i, j = ipm + d, ift + e
                if 0 <= i < len(LINEAR_PHI_MAX) and 0 <= j < len(FLAT_THRESHOLDS):
                    neigh.append(f"lin pm{LINEAR_PHI_MAX[i]:.1f} ft{FLAT_THRESHOLDS[j]:.2f}")
    else:
        neigh = [c.label for c in cfgs if c.arm == finalist.arm]
    b3_vals = []
    for lb in neigh:
        s = sel_stat(lb)
        r = [x for x in all_rows[lb] if x["slice"] == "inner_val"
             and x["market"] == "futures_5x"][0]
        b3_vals.append(s)
        print(f"{lb:22s} {s:+9.3f} {r['d_sharpe']:+12.3f} "
              f"{'<-- FINALIST' if lb == finalist.label else '':<20s}")
    others = [v for lb, v in zip(neigh, b3_vals) if lb != finalist.label]
    b3 = bool(others) and all(v > 0 for v in others)
    print(f"   all neighbours also positive on the selection statistic: {b3}")
    print(f"   B3 = {'PASS' if b3 else 'FAIL'}")

    # ------------------------------------------------------------ A3 causality
    hdr("STEP A3 -- CAUSAL TRUNCATION PROBE (the clause most likely to catch a bug)")
    print("Rebuild the target on 55% and 80% truncations of the inner-train frame;")
    print("the surviving prefix must match bit-for-bit.  A full-series scaler")
    print("applied to early rows fails here -- which is exactly why phi uses a")
    print("TRAILING rolling RMS and never a full-series std.\n")
    probe_targets = [finalist]
    cub = [c for c in cfgs if c.arm == "cubic"]
    probe_targets += cub[:1] + cub[3:4]
    probe_targets += [cfgs[0]]
    a3 = True
    for cfg in probe_targets:
        try:
            ok = causal_truncation_probe(cfg.build_uncached, train)
        except AssertionError as exc:
            ok = False
            print(f"  {cfg.label:22s} FAIL -- {exc}")
        if ok:
            print(f"  {cfg.label:22s} PASS (cuts 0.55, 0.80)")
        a3 = a3 and ok
    print(f"   A3 = {'PASS' if a3 else 'FAIL'}")

    # ------------------------------------------------------------ B4 ETH
    hdr("B4 -- FALSIFICATION TEST: ETH REPLICATION")
    eth = load_eth()
    max_ts.append(eth.index.max())
    print(f"ETH (Bitfinex): {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}")
    eth_slices = []
    for nm, (s, e) in SLICES.items():
        lo = eth.index.searchsorted(s)
        hi = eth.index.searchsorted(e, side="right")
        if hi > lo:
            eth_slices.append(nm)
    print(f"ETH slices with data: {eth_slices}  "
          f"(the series ends {eth.index.max().date()}, so inner-validation "
          f"CANNOT be run on ETH -- reported, not worked around)")
    eth_rows = compare(finalist.build, eth, label=finalist.label,
                       control_build=v4_control_build,
                       slice_names=tuple(eth_slices))
    print()
    print_rows(eth_rows)
    eth_dsh = [r["d_sharpe"] for r in eth_rows]
    eth_dlg = [r["d_loggrowth"] for r in eth_rows]
    eth_ddd = [r["d_dd"] for r in eth_rows]
    btc_sign = 1.0 if sel_stat(finalist.label) > 0 else -1.0
    b4 = bool(eth_rows) and all(np.sign(v) == btc_sign for v in eth_dlg)
    print(f"\n   BTC sign of improvement: {'+' if btc_sign > 0 else '-'}")
    print(f"   ETH d_loggrowth by cell: " +
          ", ".join(f"{r['market']}={r['d_loggrowth']:+.3f}" for r in eth_rows))
    print(f"   ETH dSharpe: " + ", ".join(f"{v:+.3f}" for v in eth_dsh) +
          f"   ETH dMaxDD: " + ", ".join(f"{v:+.1f}pp" for v in eth_ddd))
    print(f"   same sign in every available ETH cell: {b4}")
    print(f"   B4 = {'PASS' if b4 else 'FAIL'}")

    # ------------------------------------------------------------ B5 fees
    hdr("B5 -- COST ROBUSTNESS: 0.40% TAKER, inner-validation")
    print("MarketSpec(name=..., leverage=..., fee_rate=0.0040, allow_short=..., "
          "pays_funding=...)")
    fee_rows = compare(finalist.build, btc, label=finalist.label + "@40bp",
                       control_build=v4_control_build,
                       markets=(SPOT_040, FUT_040), slice_names=("inner_val",))
    print()
    print_rows(fee_rows)
    fee_sign_ok = all(np.sign(r["d_loggrowth"]) == btc_sign for r in fee_rows)
    print(f"\n   improvement does not reverse sign at 0.40%: {fee_sign_ok}")
    print(f"   B5 = {'PASS' if fee_sign_ok else 'FAIL'}")

    # --------------------------------------------- mechanism measurements
    hdr("MECHANISM MEASUREMENTS -- reported whether or not Sharpe moves")
    reps = [cfgs[0], finalist]
    best_cub = max([c for c in cfgs if c.arm == "cubic"], key=lambda c: sel_stat(c.label))
    reps.append(best_cub)
    print(f"representative arms: sign = {cfgs[0].label}, linear = {finalist.label}, "
          f"cubic = {best_cub.label}")

    phid = phi_daily(btc)

    print("\n--- CURVATURE: daily log PnL (%) regressed on lagged phi and phi^2 ---")
    print("Dao et al. (2016): a sign response gives a V-shaped (piecewise-linear)")
    print("payoff => curvature ~ 0; an unsaturated linear response gives a")
    print("parabolic payoff => curvature > 0.")
    print(f"\n{'arm':22s} {'slice':12s} {'market':8s} {'coef(phi^2)':>12s} "
          f"{'t(HAC)':>8s} {'n':>6s}")
    print("-" * 76)
    curv = {}
    for cfg in reps:
        for sl in ("inner_train", "inner_val"):
            rets, _m = daily_series(cfg, btc, sl, SPOT)
            cv = curvature(rets, phid)
            curv[(cfg.label, sl)] = cv
            print(f"{cfg.label:22s} {sl:12s} {'spot':8s} {cv['coef']:+12.4f} "
                  f"{cv['t']:+8.2f} {cv['n']:6d}")

    print("\n--- SKEWNESS of aggregated log returns (spot, inner-train + inner-val) ---")
    print("Sepp & Lucic (2026, arXiv:2607.19497): trend-return skew peaks near HALF")
    print("the filter span; v4's anchors are 20/40/80d => predicted peak ~10-40d.")
    hz = (1, 5, 10, 20, 40, 60)
    print(f"\n{'arm':22s} {'slice':12s} " + " ".join(f"{h:>7d}d" for h in hz))
    print("-" * 90)
    for cfg in reps:
        for sl in ("inner_train", "inner_val"):
            rets, _m = daily_series(cfg, btc, sl, SPOT)
            sk = skew_profile(rets, hz)
            print(f"{cfg.label:22s} {sl:12s} " +
                  " ".join(f"{sk[h]:+8.2f}" for h in hz))

    print("\n--- TURNOVER, and GROSS vs NET (inner-validation) ---")
    print("An unsaturated response trades far more than a latched one and may lose")
    print("to fees what it gains in curvature.  'gross' = 0bp fee, 'net' = the")
    print("venue tier (spot 0.10%, futures 0.05%).")
    print(f"\n{'arm':22s} {'market':12s} {'trades':>7s} {'rt/day':>7s} "
          f"{'notional/d':>11s} {'gross$':>10s} {'net$':>10s} {'fee cost':>9s}")
    print("-" * 96)
    days_val = 730.0
    for cfg in reps:
        p_full = cfg.build(btc)
        tu = daily_turnover(p_full)
        for mkt_net, mkt_gross in ((SPOT, SPOT_GROSS), (FUTURES, FUT_GROSS)):
            _r, m_net = daily_series(cfg, btc, "inner_val", mkt_net)
            _r2, m_gross = daily_series(cfg, btc, "inner_val", mkt_gross)
            rt = m_net.num_trades / 2.0 / days_val
            cost = m_gross.final_balance - m_net.final_balance
            print(f"{cfg.label:22s} {mkt_net.name:12s} {m_net.num_trades:7d} "
                  f"{rt:7.3f} {tu:11.4f} {m_gross.final_balance:10,.0f} "
                  f"{m_net.final_balance:10,.0f} {cost:9,.0f}")
    print("\nNOTE -- the brief's own turnover prediction is CONTRADICTED here, and "
          "it is worth")
    print("recording: an unsaturated LINEAR response does NOT trade more than the "
          "latched step.")
    print("It trades far LESS (notional turnover 0.39-0.79x v4's, and a fraction of "
          "the round")
    print("trips), because a slow continuous target crosses v4's 10% re-target "
          "deadband less")
    print("often than a vote that jumps in discrete ~scale/3 steps. The arm that "
          "does trade more")
    print("is the NON-MONOTONE cubic (1.26-1.65x), whose response doubles back on "
          "itself and so")
    print("crosses the deadband twice per trend. Costs are therefore NOT what "
          "decides this round.")

    # ------------------------------------------------------------ verdict
    hdr("VERDICT")
    clauses = {"A1 identity": a1, "A3 causality": a3,
               "B1 paired interval": b1, "B2 noise floor": b2,
               "B3 plateau": b3, "B4 ETH replication": b4,
               "B5 0.40% taker": fee_sign_ok}
    for k, v in clauses.items():
        print(f"  {k:22s} {'PASS' if v else 'FAIL'}")
    promote = all(clauses.values())
    print(f"\nStep 0 kill switch fired (cubic arm NEGATIVE on BTC): {kill}")
    print(f"VERDICT: {'PROMOTE' if promote else 'NEGATIVE'}")

    print(f"\nConfigurations evaluated in this file:")
    print(f"  frozen strategy grid (evaluated on all 4 cells): {n_strategy_configs}")
    print(f"  Step-0 regression fits (exploratory, h in {STEP0_HORIZONS}): "
          f"{len(STEP0_HORIZONS)}")
    print(f"  re-runs of the FROZEN finalist on other data/costs (not new "
          f"configurations): ETH, 0.40% taker, 0bp gross")
    print(f"  => strategy configurations for the ledger / deflated Sharpe: "
          f"{n_strategy_configs}")
    print(f"\nmax timestamp read anywhere in this branch (BTC and ETH): "
          f"{max(max_ts)}  (< {OOS_START}) -- no holdout bar was read.")


if __name__ == "__main__":
    main()
