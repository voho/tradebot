"""Tests for the trials-aware inference tools.

These are the checks that make the intervals in ``docs/VALIDATION.md``
believable: a statistic that cannot detect a difference it *should* detect
is as dangerous as one that invents differences that are not there, and
both failure modes are cheap to test on synthetic data with a known answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.inference import (
    DAYS_PER_YEAR, annualized_sharpe, bootstrap_interval, cpcv_splits,
    daily_returns, deflated_sharpe_ratio, expected_max_sharpe, fold_mask,
    group_bounds, max_drawdown_from_returns, min_track_record_length, moments,
    norm_cdf, norm_ppf, paired_bootstrap, probabilistic_sharpe_ratio,
    purged_train_mask,
    stationary_bootstrap_indices, total_log_return,
)


# ------------------------------------------------------------- basic stats

def test_daily_returns_from_bar_equity():
    idx = pd.date_range("2024-01-01", periods=3 * 288, freq="5min", tz="UTC")
    equity = pd.Series(np.linspace(1_000.0, 1_030.0, len(idx)), index=idx)
    rets = daily_returns(equity)
    assert len(rets) == 2
    assert (rets > 0).all()


def test_daily_returns_survive_a_wipeout():
    idx = pd.date_range("2024-01-01", periods=3 * 288, freq="5min", tz="UTC")
    equity = pd.Series([1_000.0] * 288 + [0.0] * (2 * 288), index=idx)
    rets = daily_returns(equity)
    assert rets.iloc[0] == pytest.approx(-1.0)
    assert rets.iloc[1] == 0.0  # zero equity: no return, not NaN


def test_sharpe_matches_hand_calculation():
    rets = np.array([0.01, -0.005, 0.02, 0.0, 0.007])
    expected = rets.mean() / rets.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)
    assert annualized_sharpe(rets) == pytest.approx(expected)


def test_stats_are_axis_aware():
    stack = np.array([[0.01, 0.02, -0.01], [0.0, 0.0, 0.0]])
    sharpes = annualized_sharpe(stack)
    assert sharpes.shape == (2,)
    assert sharpes[1] == 0.0  # zero variance -> zero, not a divide-by-zero
    assert annualized_sharpe(stack[0]) == pytest.approx(sharpes[0])
    assert max_drawdown_from_returns(stack).shape == (2,)
    assert total_log_return(stack).shape == (2,)


def test_max_drawdown_from_returns():
    rets = np.array([0.5, -0.5, 0.0])  # 1.0 -> 1.5 -> 0.75
    assert max_drawdown_from_returns(rets) == pytest.approx(50.0)


def test_total_log_return_handles_a_total_loss():
    assert np.isfinite(total_log_return(np.array([-1.0, 0.0, 0.0])))


# --------------------------------------------------------------- bootstrap

def test_stationary_bootstrap_shape_and_range():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(100, 10.0, 50, rng)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_stationary_bootstrap_builds_contiguous_blocks():
    """With a long mean block the resample should mostly walk forward."""
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 50.0, 20, rng)
    steps = np.diff(idx, axis=1) % 500
    assert (steps == 1).mean() > 0.9


def test_stationary_bootstrap_is_seeded():
    a = stationary_bootstrap_indices(50, 5.0, 10, np.random.default_rng(3))
    b = stationary_bootstrap_indices(50, 5.0, 10, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.001, 0.01, size=500)
    ci = bootstrap_interval(rets, annualized_sharpe, n_boot=300, mean_block=10.0)
    assert ci.lo < ci.point < ci.hi


def test_paired_bootstrap_finds_no_difference_between_a_series_and_itself():
    rng = np.random.default_rng(2)
    rets = rng.normal(0.001, 0.01, size=400)
    res = paired_bootstrap(rets, rets.copy(), annualized_sharpe, n_boot=300,
                           mean_block=10.0)
    assert res.diff.point == pytest.approx(0.0, abs=1e-12)
    assert not res.significant
    assert res.p_positive == 0.0  # every resample gives exactly zero


def test_paired_bootstrap_detects_a_large_real_difference():
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 0.01, size=800)
    good = noise + 0.004                    # same path, much better drift
    res = paired_bootstrap(good, noise, annualized_sharpe, n_boot=400,
                           mean_block=10.0)
    assert res.significant and res.p_positive > 0.99


def test_paired_bootstrap_rejects_unaligned_series():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(10), np.zeros(11), annualized_sharpe)


def test_supplied_indices_must_match_the_series():
    idx = stationary_bootstrap_indices(10, 3.0, 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        bootstrap_interval(np.zeros(11), annualized_sharpe, indices=idx)


# --------------------------------------------------------- deflated Sharpe

def test_norm_ppf_inverts_norm_cdf():
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_moments_of_a_normal_sample():
    rng = np.random.default_rng(5)
    skew, kurt = moments(rng.normal(size=200_000))
    assert abs(skew) < 0.05
    assert kurt == pytest.approx(3.0, abs=0.1)


def test_expected_max_sharpe_grows_with_trials():
    values = [expected_max_sharpe(n, 0.5) for n in (2, 10, 100, 1_000)]
    assert values == sorted(values)
    assert expected_max_sharpe(1, 0.5) == 0.0


def test_probabilistic_sharpe_rises_with_track_record():
    args = dict(skew=0.0, kurtosis=3.0)
    short = probabilistic_sharpe_ratio(1.0, 100, **args)
    long = probabilistic_sharpe_ratio(1.0, 3_000, **args)
    assert 0.5 < short < long < 1.0


def test_negative_skew_and_fat_tails_lower_the_probabilistic_sharpe():
    normal = probabilistic_sharpe_ratio(1.0, 1_000, skew=0.0, kurtosis=3.0)
    ugly = probabilistic_sharpe_ratio(1.0, 1_000, skew=-1.5, kurtosis=12.0)
    assert ugly < normal


def test_deflated_sharpe_falls_as_the_search_widens():
    args = dict(n_obs=3_000, skew=0.0, kurtosis=3.0, sd_trials=0.4)
    values = [deflated_sharpe_ratio(1.2, n_trials=n, **args)
              for n in (2, 10, 100, 1_000)]
    assert values == sorted(values, reverse=True)


def test_deflated_sharpe_refuses_the_best_of_many_noise_trials():
    """The headline guarantee: searching harder must not manufacture proof."""
    rng = np.random.default_rng(6)
    n_days, n_trials = 3_500, 200
    noise = rng.normal(0.0, 0.02, size=(n_trials, n_days))
    sharpes = annualized_sharpe(noise)
    best = int(np.argmax(sharpes))
    skew, kurt = moments(noise[best])
    dsr = deflated_sharpe_ratio(float(sharpes[best]), n_days, skew, kurt,
                                n_trials, float(sharpes.std(ddof=1)))
    assert probabilistic_sharpe_ratio(float(sharpes[best]), n_days, skew, kurt) > 0.5
    assert dsr < 0.95


def test_min_track_record_length_is_infinite_below_the_benchmark():
    assert min_track_record_length(0.3, 0.0, 3.0, benchmark=0.9) == float("inf")
    finite = min_track_record_length(1.5, 0.0, 3.0, benchmark=0.5)
    assert np.isfinite(finite) and finite > 1


# --------------------------------------------------------------- purged CV

def test_group_bounds_partition_the_series():
    bounds = group_bounds(1_000, 7)
    assert bounds[0][0] == 0 and bounds[-1][1] == 1_000
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))


def test_group_bounds_rejects_impossible_splits():
    with pytest.raises(ValueError):
        group_bounds(10, 1)
    with pytest.raises(ValueError):
        group_bounds(10, 11)


def test_cpcv_splits_enumerate_every_combination():
    splits = cpcv_splits(6, 2)
    assert len(splits) == 15
    assert len(set(splits)) == 15
    with pytest.raises(ValueError):
        cpcv_splits(4, 4)


def test_purge_and_embargo_remove_the_neighbourhood_of_the_test_set():
    n, bounds = 1_000, group_bounds(1_000, 10)
    train = purged_train_mask(n, bounds, (5,), purge=20, embargo=30)
    test = fold_mask(n, bounds, (5,))
    assert not (train & test).any()
    # group 5 is [500, 600): 20 before and 30 after must also be gone
    assert not train[480:630].any()
    assert train[479] and train[630]


def test_purged_train_mask_is_strictly_smaller_than_the_complement():
    n, bounds = 500, group_bounds(500, 5)
    for groups in cpcv_splits(5, 2):
        train = purged_train_mask(n, bounds, groups, purge=10, embargo=10)
        assert train.sum() < (~fold_mask(n, bounds, groups)).sum()
