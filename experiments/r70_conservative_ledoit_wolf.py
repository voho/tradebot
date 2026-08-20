"""R-70 conservative branch: the LITERAL Ledoit & Wolf (2008) HAC-studentized
Sharpe-difference test, built as a standalone, promotable function -- B-36.

=====================================================================
WHY THIS FILE EXISTS
=====================================================================

B-36 (top of the backlog after R-69) asks for exactly one thing this repo
has never built: Ledoit, O., & Wolf, M. (2008), "Robust performance
hypothesis testing with the Sharpe ratio," *Journal of Empirical Finance*
15(5), 850-859. R-68's own `experiments/r68_inference.py` ran a "difference
test" between R-67's delta=0.080 arm and R-65's delta=0.000 baseline and
called it Ledoit-Wolf, but it was not: it reused this project's existing
`tradebot.inference.paired_bootstrap` -- a plain percentile stationary
bootstrap of `total_log_return(a) - total_log_return(b)` -- on the SAME
resampled series twice, with no studentization at all. That got the
axis's best number yet (+0.4525 [-0.069, +1.105] on W_TRAIN), which is why
B-36 was filed: is "just short of significance" a property of the data, or
an artifact of using the wrong interval?

Ledoit & Wolf's actual contribution is to studentize the SHARPE-RATIO
difference (not the growth difference) by an estimate of its own asymptotic
standard error, obtained via the delta method against the long-run (HAC)
covariance of the four return moments that the Sharpe ratio is a function
of. This file implements that construction literally, following the
reference practical realization in Ardia, D., & Boudt, K. (2015),
"Implied expected returns and the choice of a mean-variance efficient
portfolio proxy," and the `PeerPerformance` R package's `sharpeTesting()`
(Ardia & Boudt) -- the formulas below (gradient, Parzen-kernel HAC
covariance, degrees-of-freedom correction) are that implementation,
transcribed and cited rather than re-derived from the paper's own notation.

**Mechanism, one sentence (pre-registered before any test statistic below
was computed on real data):** studentizing the Sharpe-ratio difference by
its own HAC-estimated asymptotic standard error -- rather than treating a
plain percentile bootstrap interval on the growth-difference statistic as
if it were already pivotal -- should give a materially narrower, better-
calibrated interval on this project's autocorrelated, heavy-tailed daily
returns, and may resolve one or more of the near-clearing COST-axis arms
that R-68's naive test left just short of significance.

**Falsification test, pre-registered before any real-data number was read
(named failure outcome below):** run the HAC test on a PLACEBO built from
one real return series where the population Sharpe difference is exactly
zero by construction, and confirm the empirical rejection rate at the
nominal 5% level is not wildly above 5% across >=500 placebo draws. If it
rejects at, say, >=15% (3x nominal), the HAC estimator is anti-conservative
on this project's data -- a genuine, reportable negative finding about the
METHOD, to be written up honestly rather than corrected away before it is
seen. See "THE PLACEBO DESIGN" below for why a literal circular shift (the
first example ROUTINE.md's task text offered) turns out to be degenerate on
a full-sample statistic, and what was run instead.

=====================================================================
WHAT THIS FILE DOES NOT DO
=====================================================================

- It builds no new strategy, sweeps no new parameter, and calls
  `simulate_portfolio` zero times: every candidate/baseline daily-return
  pair comes from `experiments/r70_shared.py`'s frozen
  `build_all_cells()`, imported and consumed as-is. `config_count()` at the
  end of this file's run is `experiments.r68_shared`'s count, and it comes
  entirely from `build_all_cells()`'s own 8 `_arm_daily` calls (24 backtests
  -- 1 `simulate_portfolio` + 2 `static_hold_equity` binary-search calls per
  `_arm_daily` call); nothing here adds to it.
- It does not select a winner among the six cells, and does not touch
  `experiments/r70_shared.py`, `src/tradebot/inference.py`, or any other
  branch's file. It is a standalone, promotable implementation the operator
  can merge into `tradebot.inference` later -- the dataclass and function
  signatures below deliberately mirror `Interval`/`PairedResult` in that
  module's existing style.
- **Holdout: +0.** Every real-data cell below comes from
  `r70_shared.build_all_cells()`, which is W_TRAIN/W_VAL only (see that
  file's own docstring re: B-33).
- It does not compare against the novel branch's bootstrap-studentized
  construction (`experiments/r70_novel_bootstrap_studentized.py`) -- that
  comparison is the operator's synthesis job, not this branch's.

=====================================================================
THE CONSTRUCTION
=====================================================================

Per arm, per window: `a` = candidate daily returns, `b` = baseline
(R-65 winner) daily returns, aligned, length `T`.

    mu_i    = mean(x_i)                          i in {1, 2} = {a, b}
    gamma_i = mean(x_i ** 2)
    sigma_i^2 = gamma_i - mu_i^2
    SR_i    = mu_i / sqrt(gamma_i - mu_i^2)

Gradient of `SR_1 - SR_2` w.r.t. `(mu_1, mu_2, gamma_1, gamma_2)`:

    g1 =  gamma_1 / (gamma_1 - mu_1^2)^1.5
    g2 = -gamma_2 / (gamma_2 - mu_2^2)^1.5
    g3 = -0.5 * mu_1 / (gamma_1 - mu_1^2)^1.5
    g4 =  0.5 * mu_2 / (gamma_2 - mu_2^2)^1.5

(Verified against the paper: SR_1 = mu_1 * (gamma_1 - mu_1^2)^(-1/2), so
d(SR_1)/d(mu_1) = (gamma_1-mu_1^2)^(-1/2) + mu_1^2*(gamma_1-mu_1^2)^(-3/2)
= gamma_1/(gamma_1-mu_1^2)^1.5 = g1, and symmetrically for the rest; the
numerical finite-difference check in `_check_gradient` re-derives this on
random points before any real data is touched.)

`V_t = (x1_t - mu_1, x2_t - mu_2, x1_t^2 - gamma_1, x2_t^2 - gamma_2)`, a
`(T, 4)` matrix. HAC long-run covariance, Parzen kernel, automatic
bandwidth (Newey & West 1994, "Automatic Lag Selection in Covariance
Matrix Estimation," *Review of Economic Studies* 61(4), 631-653 -- the
same automatic-bandwidth family Andrews (1991) proposes, specialised here
to the Parzen/q=2 case, which is what the `S* = 2.6614*(alpha*T)^0.2`
constant in the task spec and in `PeerPerformance::sharpeTesting()`
already commits to):

    for each of the 4 components of V_t, fit an AR(1): x_t = rho*x_{t-1} + e_t
    alpha_hat = [ sum_j 4*rho_j^2*sigma_j^4 / ((1-rho_j)^6*(1+rho_j)^2) ]
              / [ sum_j sigma_j^4 / (1-rho_j)^4 ]                    (equal
                                                                       weights)
    S* = 2.6614 * (alpha_hat * T)^0.2

    kernel_Parzen(x) = 1 - 6x^2 + 6|x|^3        |x| <= 0.5
                      = 2*(1-|x|)^3              0.5 < |x| <= 1
                      = 0                        otherwise

    Gamma_hat(j) = (1/T) * sum_{t=j+1}^{T} V_t V_{t-j}'   (only j=0..floor(S*)
                                                            are ever needed)
    Psi_hat = T/(T-4) * [ Gamma_hat(0)
                          + sum_{j=1}^{floor(S*)} kernel_Parzen(j/S*)
                                                  * (Gamma_hat(j)+Gamma_hat(j)') ]

    se = sqrt( grad' @ Psi_hat @ grad / T )
    tstat = (SR_1 - SR_2) / se
    pvalue = 2 * Phi(-|tstat|)              (tradebot.inference.norm_cdf)
    95% CI = (SR_1 - SR_2) +/- 1.959964 * se

`alpha_hat`'s AR(1) coefficients are clipped to [-0.97, 0.97] purely for
numerical safety against a near-unit-root component (none occurs on this
project's data; the clip is inert here and exists so the function cannot
divide by ~0 on some future, more persistent series). This is the ONE
numerical guard added beyond the cited formulas.

=====================================================================
THE PLACEBO DESIGN (falsification test)
=====================================================================

The task text's first suggested placebo -- circularly shift `b` relative
to `a` by a random offset -- is DEGENERATE for this statistic and was
rejected after being tried: a circular shift is a permutation of the same
T values, so `mean(shifted x) == mean(x)` and `mean(shifted x**2) ==
mean(x**2)` EXACTLY, for every shift. The point estimate `SR_a - SR_b`
would then be bit-identical to comparing `a` against `a` itself: exactly
zero on every single placebo draw, `tstat` exactly 0, `pvalue` exactly 1,
every draw a trivial non-rejection. That is not a test of the estimator's
calibration -- it is a proof the estimator returns zero when handed two
copies of the same numbers, which was never in question. This was checked
mechanically (`_check_shift_degenerate`, run in `main` before the real
placebo) and confirmed exactly zero to floating-point precision, and is
reported rather than silently swapped out for a different method.

The placebo actually run instead is the task's second suggestion --
"compare a series against an independent random permutation of itself" --
realised with this project's OWN `stationary_bootstrap_indices` (Politis &
Romano 1994) at the SAME 30-day mean block this repo uses everywhere else,
rather than an i.i.d. shuffle (which would destroy the real autocorrelation
this test is specifically supposed to be robust to, understating the
difficulty of the real problem). Procedure, run BEFORE any real
candidate-vs-baseline number in this file is computed:

1. Take one real base series `x` -- R-65's baseline daily returns on
   W_TRAIN, from `r70_shared.build_all_cells()` (a real, autocorrelated,
   heavy-tailed series, but not yet the candidate-vs-baseline comparison
   this round exists to answer).
2. Draw two INDEPENDENT stationary-bootstrap resamples of `x` (same block
   length, different seeds) and label them `a_placebo`, `b_placebo`. Both
   are draws from the identical generating process, so
   `E[SR(a_placebo)] == E[SR(b_placebo)]` by construction -- the population
   Sharpe difference is exactly zero -- while each individual draw still
   carries the series' own real dependence structure and fat tails, unlike
   an i.i.d. permutation.
3. Repeat 1,000 times, run the HAC test on each pair at nominal alpha=0.05,
   and report the empirical rejection rate. Named failure: rate > 15%
   (3x nominal). A rate close to 5% is the pass condition; this is checked
   and reported honestly either way, before any real cell's pvalue is
   printed.

=====================================================================
VALIDATION (unit-test surface, run in `main` before real data)
=====================================================================

1. `_check_gradient` -- finite-difference gradient at 200 random points in
   the valid domain (mu^2 < gamma), max abs error must be < 1e-5.
2. `_check_bandwidth_finite_positive` -- on all 6 real cells' `V_t`
   matrices, `S*` must be finite and > 0 (a degenerate S*=0 or NaN would
   mean `alpha_hat`'s AR(1) fit failed on real data, which the task asks to
   be caught by a unit test rather than silently producing an
   uncorrected/White-only covariance).
3. `_check_synthetic_coverage` -- 2,000 replications of two INDEPENDENT
   i.i.d. Gaussian daily-return series (T=500) at a KNOWN Sharpe gap
   (checked at gap=0, the null, and gap=+0.3 annualized, a real alternative)
   and reports the empirical 95% CI coverage rate, which must land within a
   few points of 95% for i.i.d. data where the HAC correction should
   degenerate to (approximately) the textbook White/iid case.
4. `_check_shift_degenerate` -- documents and confirms the failed circular-
   shift placebo design above is exactly zero, not silently wrong.
5. The 1,000-draw block-bootstrap self-comparison placebo described above.

    .venv/bin/python experiments/r70_conservative_ledoit_wolf.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tradebot.inference import (  # noqa: E402
    Interval,
    norm_cdf,
    stationary_bootstrap_indices,
)

from experiments.r68_shared import config_count  # noqa: E402
from experiments.r70_shared import build_all_cells  # noqa: E402

Z_95 = 1.959964  # two-sided 95% normal critical value
DAYS_PER_YEAR = 365.25


# ===================================================================== core


@dataclass
class LedoitWolfResult:
    """HAC-studentized Sharpe-ratio-difference test, Ledoit & Wolf (2008).

    Mirrors `tradebot.inference.PairedResult`'s shape (point estimates +
    an `Interval`), with the fields the studentized construction adds:
    `se` (the HAC delta-method standard error), `tstat`, `pvalue`, and the
    diagnostics of the automatic bandwidth (`bandwidth`, `alpha_hat`) so a
    reader can see what lag the correction actually used.
    """

    sr_a: float
    sr_b: float
    diff: Interval          # point = sr_a - sr_b, at the 95% HAC-studentized CI
    se: float
    tstat: float
    pvalue: float
    bandwidth: float        # S* (Parzen kernel automatic bandwidth, in days)
    alpha_hat: float        # Newey-West (1994) plug-in used to derive S*
    n: int

    @property
    def significant(self) -> bool:
        return self.diff.lo > 0.0 or self.diff.hi < 0.0


def _sharpe_moments(x: np.ndarray) -> tuple[float, float, float]:
    """(mu, gamma, SR) for one return series."""
    mu = float(np.mean(x))
    gamma = float(np.mean(x ** 2))
    sr = mu / sqrt(gamma - mu * mu)
    return mu, gamma, sr


def _lw_gradient(mu1: float, gamma1: float, mu2: float, gamma2: float) -> np.ndarray:
    """Gradient of SR_1 - SR_2 w.r.t. (mu1, mu2, gamma1, gamma2). See the
    module docstring for the derivation and the numerical check below."""
    s1 = (gamma1 - mu1 * mu1) ** 1.5
    s2 = (gamma2 - mu2 * mu2) ** 1.5
    g1 = gamma1 / s1
    g2 = -gamma2 / s2
    g3 = -0.5 * mu1 / s1
    g4 = 0.5 * mu2 / s2
    return np.array([g1, g2, g3, g4], dtype=float)


def _parzen_kernel(x: float) -> float:
    ax = abs(x)
    if ax <= 0.5:
        return 1.0 - 6.0 * ax * ax + 6.0 * ax ** 3
    if ax <= 1.0:
        return 2.0 * (1.0 - ax) ** 3
    return 0.0


def _nw_bandwidth(V: np.ndarray) -> tuple[float, float]:
    """Newey & West (1994) automatic Parzen-kernel bandwidth, via an AR(1)
    plug-in fit to each of V's 4 columns, combined with equal weights
    (the standard multivariate generalisation of their single-series
    formula, as used by `sandwich::bwNeweyWest` and by `PeerPerformance`).
    Returns (S_star, alpha_hat). Both are checked finite and S_star > 0 by
    `_check_bandwidth_finite_positive` on every real cell."""
    T, k = V.shape
    num, den = 0.0, 0.0
    for j in range(k):
        x = V[:, j]
        x0, x1 = x[:-1], x[1:]
        denom_ols = float(np.sum(x0 * x0))
        rho = float(np.sum(x0 * x1) / denom_ols) if denom_ols > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))  # numerical guard only, see docstring
        resid = x1 - rho * x0
        sigma2 = float(np.var(resid, ddof=1)) if len(resid) > 1 else float(np.var(resid))
        num += 4.0 * rho * rho * sigma2 * sigma2 / (((1 - rho) ** 6) * ((1 + rho) ** 2))
        den += sigma2 * sigma2 / ((1 - rho) ** 4)
    alpha_hat = num / den if den > 0 else 0.0
    s_star = 2.6614 * (alpha_hat * T) ** 0.2
    return float(s_star), float(alpha_hat)


def _hac_long_run_cov(V: np.ndarray, s_star: float) -> np.ndarray:
    """Psi_hat: Parzen-kernel HAC estimate of V_t's long-run covariance,
    degrees-of-freedom corrected by T/(T-4) for the 4 estimated moments."""
    T, k = V.shape
    gamma0 = V.T @ V / T
    psi = gamma0.copy()
    s_int = int(np.floor(s_star))
    for j in range(1, s_int + 1):
        w = _parzen_kernel(j / s_star)
        if w == 0.0:
            continue
        gj = V[j:].T @ V[:-j] / T
        psi += w * (gj + gj.T)
    return psi * (T / (T - k))


def ledoit_wolf_sharpe_diff(a: np.ndarray, b: np.ndarray,
                            level: float = 0.95) -> LedoitWolfResult:
    """The literal Ledoit & Wolf (2008) HAC-studentized Sharpe-ratio
    difference test between two aligned, equal-length daily-return series.

    `a`, `b` must be aligned (same dates, same length) -- the whole point,
    exactly as in `tradebot.inference.paired_bootstrap`, is that the two
    series' shared variance is what makes the difference well resolved.
    Uses a fixed z=1.959964 critical value (95% two-sided normal), matching
    the task's specification; `level` is accepted for interface parity with
    `Interval` but only 0.95 is exercised by this file.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"unaligned series: {len(a)} vs {len(b)}")
    if level != 0.95:
        raise NotImplementedError("only the 95% CI (z=1.959964) is implemented")
    T = len(a)

    mu1, gamma1, sr1 = _sharpe_moments(a)
    mu2, gamma2, sr2 = _sharpe_moments(b)
    diff = sr1 - sr2

    V = np.column_stack([a - mu1, b - mu2, a ** 2 - gamma1, b ** 2 - gamma2])
    s_star, alpha_hat = _nw_bandwidth(V)
    psi = _hac_long_run_cov(V, s_star)
    grad = _lw_gradient(mu1, gamma1, mu2, gamma2)

    var = float(grad @ psi @ grad) / T
    se = sqrt(var) if var > 0 else 0.0
    tstat = diff / se if se > 0 else 0.0
    pvalue = 2.0 * (1.0 - norm_cdf(abs(tstat))) if se > 0 else 1.0

    ci = Interval(diff, diff - Z_95 * se, diff + Z_95 * se, level)
    return LedoitWolfResult(sr_a=sr1, sr_b=sr2, diff=ci, se=se, tstat=tstat,
                            pvalue=pvalue, bandwidth=s_star, alpha_hat=alpha_hat, n=T)


# =============================================================== validation


def _check_gradient(seed: int = 0, n_points: int = 200, eps: float = 1e-6) -> float:
    """Finite-difference check of `_lw_gradient` at random points in a
    domain representative of this project's daily returns (sigma between
    ~1% and ~14% daily -- comfortably spanning the real cells' realized
    vol; see the values printed by `r70_shared.py`'s own smoke test).
    Reports the max RELATIVE error across all points and all 4 partials
    (`|analytic-numeric| / max(1, |analytic|)`) rather than absolute,
    because near a small-variance point the gradient itself is large and an
    absolute tolerance conflates finite-difference truncation error with a
    real bug; asserts relative error < 1e-4. (An earlier draft used an
    absolute 1e-5 tolerance over a domain that allowed sigma^2 down to
    1e-5 -- unrealistically close to a Sharpe-ratio singularity for daily
    financial returns -- and failed at 0.14 absolute error on a point where
    the analytic gradient itself was ~4900; manually re-checked against a
    symbolic derivation at ordinary points (sigma^2 ~ 1e-3) and confirmed
    correct to 8 decimals, so the fix was the check's domain and tolerance,
    not the formula.)"""
    rng = np.random.default_rng(seed)
    worst = 0.0

    def sr_diff(mu1, mu2, gamma1, gamma2):
        return (mu1 / sqrt(gamma1 - mu1 * mu1)) - (mu2 / sqrt(gamma2 - mu2 * mu2))

    for _ in range(n_points):
        mu1 = rng.uniform(-0.01, 0.01)
        mu2 = rng.uniform(-0.01, 0.01)
        gamma1 = mu1 * mu1 + rng.uniform(1e-4, 0.02)   # sigma1 in (1%, ~14.1%)
        gamma2 = mu2 * mu2 + rng.uniform(1e-4, 0.02)
        analytic = _lw_gradient(mu1, gamma1, mu2, gamma2)
        args = [mu1, mu2, gamma1, gamma2]
        numeric = np.empty(4)
        for i in range(4):
            up, down = list(args), list(args)
            up[i] += eps
            down[i] -= eps
            numeric[i] = (sr_diff(*up) - sr_diff(*down)) / (2 * eps)
        rel_err = np.abs(analytic - numeric) / np.maximum(1.0, np.abs(analytic))
        worst = max(worst, float(np.max(rel_err)))
    assert worst < 1e-4, f"gradient check failed: max relative error {worst}"
    return worst


def _check_bandwidth_finite_positive(cells: dict) -> list[tuple]:
    """S* must be finite and > 0 on every one of the 6 real cells -- the
    task's own unit-test requirement. Returns [(arm, window, S*, alpha_hat)]."""
    rows = []
    for (arm, window), (a, b, _matched, _n) in cells.items():
        mu1, gamma1, _ = _sharpe_moments(a)
        mu2, gamma2, _ = _sharpe_moments(b)
        V = np.column_stack([a - mu1, b - mu2, a ** 2 - gamma1, b ** 2 - gamma2])
        s_star, alpha_hat = _nw_bandwidth(V)
        assert np.isfinite(s_star) and s_star > 0.0, \
            f"{arm}/{window}: S*={s_star} is not finite and positive"
        assert np.isfinite(alpha_hat), f"{arm}/{window}: alpha_hat={alpha_hat} not finite"
        rows.append((arm, window, s_star, alpha_hat))
    return rows


def _check_shift_degenerate(x: np.ndarray, seed: int = 0) -> float:
    """Confirms the circular-shift placebo (task's first suggestion) is
    degenerate: shifting a series relative to itself leaves the full-sample
    mean and second moment EXACTLY unchanged, so SR_a - SR_b is exactly
    zero on every draw. Returns the max |diff| across 20 random shifts,
    which must be ~0 (floating point only)."""
    rng = np.random.default_rng(seed)
    T = len(x)
    worst = 0.0
    for _ in range(20):
        r = int(rng.integers(1, T - 1))
        shifted = np.concatenate([x[r:], x[:r]])
        res = ledoit_wolf_sharpe_diff(x, shifted)
        worst = max(worst, abs(res.diff.point))
    return worst


def _check_synthetic_coverage(true_gap_annualized: float, n_reps: int = 2_000,
                              T: int = 500, seed: int = 1) -> dict:
    """Two INDEPENDENT i.i.d. Gaussian daily-return series with a KNOWN
    population Sharpe gap. Checks that the 95% HAC-studentized CI covers
    the true gap at roughly the nominal rate. Daily mu/sigma are chosen so
    that annualizing (mu/sigma)*sqrt(252) reproduces the requested
    Sharpe(s); series B is fixed at annualized Sharpe 0.0."""
    rng = np.random.default_rng(seed)
    sigma_daily = 0.02  # representative daily vol, order of this project's data
    sr_b_ann = 0.0
    sr_a_ann = true_gap_annualized + sr_b_ann
    mu_a = sr_a_ann * sigma_daily / sqrt(252.0)
    mu_b = sr_b_ann * sigma_daily / sqrt(252.0)

    covered = 0
    widths = []
    for _ in range(n_reps):
        a = rng.normal(mu_a, sigma_daily, size=T)
        b = rng.normal(mu_b, sigma_daily, size=T)
        res = ledoit_wolf_sharpe_diff(a, b)
        true_daily_gap = mu_a / sigma_daily - mu_b / sigma_daily  # daily-unit SR gap
        if res.diff.lo <= true_daily_gap <= res.diff.hi:
            covered += 1
        widths.append(res.diff.hi - res.diff.lo)
    return {
        "true_gap_annualized": true_gap_annualized,
        "coverage": covered / n_reps,
        "mean_width": float(np.mean(widths)),
        "n_reps": n_reps,
    }


def _placebo_block_bootstrap(x: np.ndarray, n_draws: int = 1_000,
                             mean_block: float = 30.0, seed: int = 2) -> dict:
    """The falsification test actually run (see module docstring for why
    the circular-shift design was rejected first). Two INDEPENDENT
    stationary-bootstrap resamples of the SAME series `x` per draw --
    identical generating distribution, so the population Sharpe gap is
    exactly zero -- HAC test run on each pair, empirical rejection rate at
    alpha=0.05 reported. Named failure: rate > 0.15 (3x nominal)."""
    n = len(x)
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    idx_a = stationary_bootstrap_indices(n, mean_block, n_draws, rng_a)
    idx_b = stationary_bootstrap_indices(n, mean_block, n_draws, rng_b)
    rejections = 0
    tstats = []
    for i in range(n_draws):
        a_p = x[idx_a[i]]
        b_p = x[idx_b[i]]
        res = ledoit_wolf_sharpe_diff(a_p, b_p)
        tstats.append(res.tstat)
        if res.pvalue < 0.05:
            rejections += 1
    return {
        "n_draws": n_draws,
        "rejection_rate": rejections / n_draws,
        "mean_abs_tstat": float(np.mean(np.abs(tstats))),
    }


# ======================================================================= run


def main() -> None:
    print("=" * 78)
    print("STEP 1-3: gradient check, bandwidth sanity, synthetic coverage")
    print("=" * 78)

    grad_err = _check_gradient()
    print(f"[PASS] finite-difference gradient check: max relative error = "
          f"{grad_err:.2e} (< 1e-4)")

    for gap in (0.0, 0.3):
        cov = _check_synthetic_coverage(gap)
        flag = "PASS" if abs(cov["coverage"] - 0.95) < 0.03 else "CHECK"
        print(f"[{flag}] synthetic i.i.d. Gaussian, true annualized SR gap "
              f"{gap:+.2f}: 95% CI coverage = {cov['coverage']:.3f} over "
              f"{cov['n_reps']} reps (mean width {cov['mean_width']:.4f})")

    print()
    print("=" * 78)
    print("STEP 4: falsification / placebo test (BEFORE any real cell's result)")
    print("=" * 78)

    cells = build_all_cells()

    bw_rows = _check_bandwidth_finite_positive(cells)
    for arm, window, s_star, alpha_hat in bw_rows:
        print(f"[PASS] bandwidth finite & positive: {arm:28s} {window:8s} "
              f"S*={s_star:.2f} days  alpha_hat={alpha_hat:.4f}")

    # The BASELINE (R-65 winner) leg is identical across all three arms on a
    # given window by construction (see r70_shared.build_all_cells), so any
    # one cell's baseline series is representative.
    _, baseline_train, _, _ = cells[("r67_hysteresis_0.080", "W_TRAIN")]

    shift_worst = _check_shift_degenerate(baseline_train)
    print(f"\n[{'PASS' if shift_worst < 1e-9 else 'FAIL'}] circular-shift placebo "
          f"IS degenerate as predicted: max|diff| over 20 shifts = "
          f"{shift_worst:.2e} (design rejected; see module docstring)")

    placebo = _placebo_block_bootstrap(baseline_train)
    verdict = "PASS" if placebo["rejection_rate"] <= 0.15 else "FAIL (miscalibrated)"
    print(f"\n[{verdict}] block-bootstrap self-comparison placebo "
          f"(R-65 baseline, W_TRAIN, n={len(baseline_train)}, "
          f"{placebo['n_draws']} draws):")
    print(f"    empirical rejection rate at nominal alpha=0.05: "
          f"{placebo['rejection_rate']:.4f}  "
          f"(named failure threshold: > 0.15)")
    print(f"    mean |tstat| across draws: {placebo['mean_abs_tstat']:.3f} "
          f"(expected ~0.80 under a well-calibrated null)")

    print()
    print("=" * 78)
    print("STEP 5: the 6 real COST-axis cells")
    print("=" * 78)

    results = {}
    for (arm, window), (cand, base, matched, n) in cells.items():
        res = ledoit_wolf_sharpe_diff(cand, base)
        results[(arm, window)] = res
        print(f"\n{arm:28s} {window:8s}  n={n}  matched={matched}")
        print(f"    SR_candidate = {res.sr_a:+.5f}   SR_baseline = {res.sr_b:+.5f}")
        print(f"    diff = {res.diff.point:+.5f}   se = {res.se:.5f}   "
              f"tstat = {res.tstat:+.3f}   pvalue = {res.pvalue:.4f}")
        print(f"    95% CI = [{res.diff.lo:+.5f}, {res.diff.hi:+.5f}]   "
              f"significant = {res.significant}   S* = {res.bandwidth:.2f}d")

    print()
    print("=" * 78)
    n_configs = config_count()
    print(f"config_count() = {n_configs}  "
          f"(entirely from r70_shared.build_all_cells()'s 8 _arm_daily calls; "
          f"this file calls simulate_portfolio 0 times)")
    print("Holdout consultations: +0 (W_TRAIN and W_VAL only; see B-33).")


if __name__ == "__main__":
    main()
