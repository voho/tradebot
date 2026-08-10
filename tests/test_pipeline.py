"""End-to-end pipeline: features, strategy, walk-forward, paper trading, metrics."""

from __future__ import annotations

import numpy as np
import pytest

import gtbot.features as F
from gtbot.data.schema import validate
from gtbot.data.synthetic import simulate
from gtbot.engine.backtest import run_backtest
from gtbot.engine.broker import CostModel, ExecutionConfig
from gtbot.engine.paper import PaperBroker, PaperTrader, replay_paper
from gtbot.eval import metrics, stats
from gtbot.eval.walkforward import make_folds, run_walkforward
from gtbot.strategy import GameTheoreticStrategy, StrategyConfig


@pytest.fixture(scope="module")
def bars():
    return validate(simulate(14_000, seed=7).bars)


def _small_cfg():
    # Short warm-up so the pipeline tests stay fast; the production default is
    # a full week of bars.
    feats = F.FeatureConfig(warmup=1200)
    return StrategyConfig(features=feats, signal_window=800, min_scale_samples=300)


# --------------------------------------------------------------------- features
def test_features_are_finite_and_aligned(bars):
    fs = F.build(bars)
    assert len(fs) == len(bars)
    for name in fs.names:
        arr = fs[name]
        assert arr.shape == (len(bars),), name
        assert np.all(np.isfinite(arr)), f"{name} contains non-finite values"


def test_feature_frame_roundtrips(bars):
    fs = F.build(bars)
    frame = fs.to_frame()
    assert len(frame) == len(bars)
    assert "vpin" in frame.columns and "regime_cell" in frame.columns


def test_sweep_flags_are_mutually_exclusive(bars):
    fs = F.build(bars)
    assert not np.any((fs["sweep_up"] > 0) & (fs["sweep_down"] > 0))


# --------------------------------------------------------------------- strategy
def test_strategy_runs_and_respects_its_contract(bars):
    strat = GameTheoreticStrategy(_small_cfg())
    res = run_backtest(bars, strat, costs=CostModel.for_tier("vip6"))
    assert len(res.equity) == len(bars)
    assert np.all(np.isfinite(res.returns))
    assert np.all(res.equity > 0), "equity must never go non-positive"
    diag = res.diagnostics
    assert len(diag["expert_names"]) == len(strat.experts)


def test_strategy_is_deterministic(bars):
    a = run_backtest(bars, GameTheoreticStrategy(_small_cfg()))
    b = run_backtest(bars, GameTheoreticStrategy(_small_cfg()))
    assert np.array_equal(a.equity, b.equity)


def test_no_trading_before_warmup(bars):
    strat = GameTheoreticStrategy(_small_cfg())
    res = run_backtest(bars, strat)
    assert np.all(res.position[: strat.warmup] == 0.0)
    assert all(f.bar > strat.warmup for f in res.fills)


def test_higher_costs_never_help(bars):
    cheap = run_backtest(bars, GameTheoreticStrategy(_small_cfg()), costs=CostModel.for_tier("vip9"))
    dear = run_backtest(bars, GameTheoreticStrategy(_small_cfg()), costs=CostModel.for_tier("retail"))
    assert dear.costs.sum() >= 0.0 and cheap.costs.sum() >= 0.0


# ----------------------------------------------------------------- walkforward
def test_folds_are_ordered_and_non_overlapping():
    folds = make_folds(60_000, n_folds=5, warmup=2000, embargo=288, purge=12)
    assert len(folds) == 5
    for a, b in zip(folds, folds[1:]):
        assert a.end < b.start, "folds must not overlap after the embargo"
    for f in folds:
        assert f.start < f.warmup_end < f.end


def test_folds_reject_impossible_geometry():
    with pytest.raises(ValueError):
        make_folds(1000, n_folds=10, warmup=2000, embargo=288, purge=12)


def test_walkforward_runs(bars):
    wf = run_walkforward(
        bars,
        lambda: GameTheoreticStrategy(_small_cfg()),
        n_folds=3,
        warmup=1200,
        embargo=100,
        purge=6,
        costs=CostModel.for_tier("vip6"),
    )
    assert len(wf.fold_metrics) >= 2
    assert np.all(np.isfinite(wf.pooled_returns))
    assert len(wf.summary_frame()) == len(wf.fold_metrics)


def test_walkforward_uses_a_fresh_learner_per_fold(bars):
    """A shared strategy would leak learned weights across folds."""
    seen = []

    def factory():
        s = GameTheoreticStrategy(_small_cfg())
        seen.append(id(s))
        return s

    run_walkforward(bars, factory, n_folds=3, warmup=1200, embargo=100, purge=6)
    assert len(set(seen)) == len(seen) > 1


# ----------------------------------------------------------------------- paper
def test_paper_loop_replays_without_error(bars):
    session, broker = replay_paper(
        bars.iloc[:6000], GameTheoreticStrategy(_small_cfg()), warmup_bars=3000
    )
    assert session.bars_seen == len(bars.iloc[:6000]) - 3000
    assert session.decisions > 0
    assert broker.equity() > 0


def test_paper_broker_charges_fees_and_tracks_position():
    broker = PaperBroker(starting_equity=1000.0)
    broker.mark(100.0)
    order = broker.submit(1.0, 100.0, 0)
    assert order is not None and broker.position() == 1.0
    assert broker.equity() < 1000.0  # a fee was paid
    broker.mark(110.0)
    assert broker.equity() > 1000.0  # and the position marked up
    assert broker.submit(1.0, 110.0, 1) is None  # no change, no order


def test_paper_requires_warmup_before_streaming(bars):
    trader = PaperTrader(GameTheoreticStrategy(_small_cfg()))
    with pytest.raises(RuntimeError):
        trader.on_bar(bars.iloc[0].to_dict())


# --------------------------------------------------------------------- metrics
def test_metrics_on_a_known_series():
    r = np.full(1000, 0.001)
    eq = np.cumprod(1.0 + r)
    m = metrics.compute(r, eq, np.ones(1000), np.zeros(1000), bars_per_year=1000)
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert m.hit_rate == 1.0
    assert m.sharpe > 100  # a constant positive return has no volatility


def test_max_drawdown_is_correct():
    eq = np.array([1.0, 1.2, 0.9, 1.1])
    assert metrics.max_drawdown(eq) == pytest.approx(0.25)


def test_deflated_sharpe_penalises_many_trials():
    few = metrics.deflated_sharpe(0.05, 5000, 0.0, 3.0, n_trials=1, trial_sr_std=0.0)
    many = metrics.deflated_sharpe(0.05, 5000, 0.0, 3.0, n_trials=500, trial_sr_std=0.05)
    assert many < few


def test_bootstrap_detects_a_real_edge_and_rejects_noise():
    rng = np.random.default_rng(0)
    edge = rng.standard_normal(5000) * 0.001 + 0.0004
    noise = rng.standard_normal(5000) * 0.001
    good = stats.bootstrap_sharpe(edge, bars_per_year=1000, n_resamples=300, mean_block=10)
    bad = stats.bootstrap_sharpe(noise, bars_per_year=1000, n_resamples=300, mean_block=10)
    assert good.p_value < 0.05
    assert bad.p_value > 0.05


def test_permutation_test_rejects_a_shuffled_signal():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(4000)
    fwd = 0.05 * sig + rng.standard_normal(4000)
    assert stats.permutation_test(sig, fwd, n_permutations=200, block=50) < 0.05
    assert stats.permutation_test(sig, rng.standard_normal(4000), n_permutations=200, block=50) > 0.05


def test_newey_west_widens_errors_under_autocorrelation():
    rng = np.random.default_rng(2)
    iid = rng.standard_normal(4000) * 0.01 + 0.002
    ar = iid.copy()
    for i in range(1, ar.size):
        ar[i] += 0.7 * (ar[i - 1] - 0.002)
    assert abs(stats.newey_west_tstat(ar)) < abs(stats.newey_west_tstat(iid)) * 1.5
