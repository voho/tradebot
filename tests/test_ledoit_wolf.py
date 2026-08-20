"""Tests for the two Ledoit & Wolf (2008)-style Sharpe-difference tests
(R-70, ``docs/LEDGER.md``, closing backlog item B-36).

Two independent standard-error estimators for the same studentized
statistic: :func:`ledoit_wolf_sharpe_diff` (analytic Parzen-kernel HAC) and
:func:`bootstrap_studentized_sharpe_diff` (nonparametric stationary
bootstrap, this module's own established convention). They are expected to
agree closely on the *point estimate* (same statistic, same data) and
*approximately* on significance — R-70 found real disagreement is possible
on individual near-zero cells, which is informative rather than a bug, but
both must still be individually well-calibrated. That calibration is what
these tests check, on synthetic data with a known answer, the same way
``test_paired_bootstrap_*`` already does for the plain percentile test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tradebot.inference import (
    bootstrap_studentized_sharpe_diff, ledoit_wolf_sharpe_diff,
    stationary_bootstrap_indices, _lw_gradient, _sharpe_moments,
)


# ------------------------------------------------------- ledoit_wolf_sharpe_diff

def test_lw_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    eps = 1e-6

    def sr_diff(mu1, mu2, gamma1, gamma2):
        return (mu1 / np.sqrt(gamma1 - mu1 * mu1)
               - mu2 / np.sqrt(gamma2 - mu2 * mu2))

    worst = 0.0
    for _ in range(200):
        mu1 = rng.uniform(-0.01, 0.01)
        mu2 = rng.uniform(-0.01, 0.01)
        gamma1 = mu1 * mu1 + rng.uniform(1e-4, 0.02)   # sigma in (1%, ~14%)
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
    assert worst < 1e-4


def test_sharpe_moments_matches_hand_calculation():
    x = np.array([0.01, -0.02, 0.015, 0.0, 0.03])
    mu, gamma, sr = _sharpe_moments(x)
    assert mu == pytest.approx(x.mean())
    assert gamma == pytest.approx((x ** 2).mean())
    assert sr == pytest.approx(x.mean() / x.std(ddof=0))


def test_lw_finds_no_difference_between_a_series_and_itself():
    rng = np.random.default_rng(2)
    rets = rng.normal(0.001, 0.01, size=400)
    res = ledoit_wolf_sharpe_diff(rets, rets.copy())
    assert res.diff.point == pytest.approx(0.0, abs=1e-12)
    assert res.tstat == 0.0
    assert not res.significant


def test_lw_is_antisymmetric():
    rng = np.random.default_rng(5)
    a = rng.normal(0.002, 0.01, size=300)
    b = rng.normal(0.0, 0.012, size=300)
    ab = ledoit_wolf_sharpe_diff(a, b)
    ba = ledoit_wolf_sharpe_diff(b, a)
    assert ab.diff.point == pytest.approx(-ba.diff.point)
    assert ab.se == pytest.approx(ba.se)


def test_lw_detects_a_large_real_difference():
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 0.01, size=800)
    good = noise + 0.004                    # same path, much better drift
    res = ledoit_wolf_sharpe_diff(good, noise)
    assert res.significant
    assert res.pvalue < 0.01


def test_lw_rejects_unaligned_series():
    with pytest.raises(ValueError):
        ledoit_wolf_sharpe_diff(np.zeros(10), np.zeros(11))


def test_lw_bandwidth_finite_and_positive_on_autocorrelated_data():
    rng = np.random.default_rng(9)
    n = 500
    e = rng.normal(0.0, 0.01, size=n)
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = 0.2 * a[i - 1] + e[i]        # AR(1), matches this project's daily data
    b = rng.normal(0.0, 0.01, size=n)
    res = ledoit_wolf_sharpe_diff(a, b)
    assert np.isfinite(res.bandwidth) and res.bandwidth > 0.0
    assert np.isfinite(res.alpha_hat)


def test_lw_synthetic_coverage_near_nominal():
    """95% CI should cover a known true Sharpe gap close to 95% of the time
    on i.i.d. Gaussian data, where the HAC correction should reduce to
    (approximately) the textbook i.i.d. case. Matches R-70's own validation
    (coverage 0.951-0.952 over 2,000 reps); this test uses fewer reps to
    stay fast, with a looser tolerance band."""
    rng = np.random.default_rng(1)
    n, sigma = 300, 0.02
    true_gap = 0.03  # daily-unit Sharpe gap
    covered = 0
    reps = 300
    for i in range(reps):
        a = rng.normal(true_gap * sigma, sigma, size=n)
        b = rng.normal(0.0, sigma, size=n)
        res = ledoit_wolf_sharpe_diff(a, b)
        covered += res.diff.lo <= true_gap <= res.diff.hi
    coverage = covered / reps
    assert 0.85 <= coverage <= 1.0, f"coverage {coverage} far from nominal 0.95"


# ------------------------------------------------ bootstrap_studentized_sharpe_diff

def test_boot_studentized_finds_no_difference_between_a_series_and_itself():
    rng = np.random.default_rng(2)
    rets = rng.normal(0.001, 0.01, size=400)
    res = bootstrap_studentized_sharpe_diff(rets, rets.copy(), n_boot=300,
                                            mean_block=10.0)
    assert res.diff == pytest.approx(0.0, abs=1e-12)
    assert res.se_boot == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(res.tstat)  # se_boot == 0: guarded, not a divide-by-zero
    assert not res.significant_normal
    assert not res.significant_studentized


def test_boot_studentized_detects_a_large_real_difference():
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 0.01, size=800)
    good = noise + 0.004                    # same path, much better drift
    res = bootstrap_studentized_sharpe_diff(good, noise, n_boot=400,
                                            mean_block=10.0)
    assert res.significant_normal and res.significant_studentized
    assert res.tstat > 3.0


def test_boot_studentized_rejects_unaligned_series():
    with pytest.raises(ValueError):
        bootstrap_studentized_sharpe_diff(np.zeros(10), np.zeros(11))


def test_boot_studentized_uses_paired_resamples():
    """The same resample index must be applied to both series -- otherwise
    the shared-variance cancellation the whole construction depends on is
    lost. Supplying explicit indices and checking against a hand-rolled
    paired computation catches a regression to independent resampling."""
    rng = np.random.default_rng(3)
    n = 200
    a = rng.normal(0.001, 0.01, size=n)
    b = rng.normal(0.0005, 0.011, size=n)
    idx = stationary_bootstrap_indices(n, 15.0, 50, np.random.default_rng(0))
    res = bootstrap_studentized_sharpe_diff(a, b, indices=idx)

    from tradebot.inference import annualized_sharpe
    boot_diff = (annualized_sharpe(a[idx], periods_per_year=1.0)
                - annualized_sharpe(b[idx], periods_per_year=1.0))
    assert res.se_boot == pytest.approx(float(np.std(boot_diff, ddof=1)))


def test_boot_studentized_supplied_indices_must_match_the_series():
    idx = stationary_bootstrap_indices(10, 3.0, 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        bootstrap_studentized_sharpe_diff(np.zeros(20), np.zeros(20), indices=idx)


def test_boot_studentized_reproducible_with_fixed_seed():
    rng = np.random.default_rng(6)
    a = rng.normal(0.001, 0.01, size=250)
    b = rng.normal(0.0, 0.01, size=250)
    r1 = bootstrap_studentized_sharpe_diff(a, b, n_boot=200, seed=42)
    r2 = bootstrap_studentized_sharpe_diff(a, b, n_boot=200, seed=42)
    assert r1.se_boot == r2.se_boot
    assert r1.normal_ci.lo == r2.normal_ci.lo


# ----------------------------------------------------- cross-method agreement

def test_both_methods_agree_closely_on_the_point_estimate():
    """Both functions compute essentially the same Sharpe-difference
    statistic; only their standard-error estimators differ. The point
    estimates are not bit-identical -- ``ledoit_wolf_sharpe_diff`` uses the
    population std (``ddof=0``, required by its delta-method construction:
    mu and gamma=E[X**2] are the only two smooth sample-mean statistics
    available to differentiate), ``bootstrap_studentized_sharpe_diff`` uses
    this module's own ``annualized_sharpe`` (``ddof=1``, sample std) -- but
    the gap is the standard small-sample bias correction,
    ``sqrt(T/(T-1))``, a fraction of a percent at this project's sample
    sizes, not a disagreement about what is being measured."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0015, 0.012, size=500)
    b = rng.normal(0.0005, 0.011, size=500)
    hac = ledoit_wolf_sharpe_diff(a, b)
    boot = bootstrap_studentized_sharpe_diff(a, b, n_boot=500)
    assert hac.diff.point == pytest.approx(boot.diff, rel=0.01)
