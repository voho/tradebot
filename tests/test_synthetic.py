"""The simulator must reproduce BTCUSD 5-minute stylised facts.

These are the properties the strategy's edge is allowed to depend on.  If the
generator drifts away from them — too much volatility, implausibly strong mean
reversion — then any backtest run on it is measuring the generator, not the
strategy.  Tolerances are wide because these are cross-seed statistics, but
they are tight enough to catch a miscalibration.

Reference values for BTCUSDT 5m: ~40-70% annualised volatility, excess kurtosis
well above the Gaussian 3, slowly decaying volatility autocorrelation, and a
small negative first-order return autocorrelation (roughly -0.01 to -0.06).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gtbot.data.schema import BTCUSD_5M, SchemaError, validate
from gtbot.data.synthetic import block_bootstrap, make_random_walk, simulate

BPY = BTCUSD_5M.bars_per_year


@pytest.fixture(scope="module")
def paths():
    return [simulate(20_000, seed=s) for s in range(4)]


def _logret(bars):
    return np.diff(np.log(bars["close"].to_numpy()))


def test_output_satisfies_the_canonical_schema(paths):
    for p in paths:
        validate(p.bars)  # raises on any violation


def test_annualised_volatility_is_plausible(paths):
    vols = [_logret(p.bars).std() * math.sqrt(BPY) for p in paths]
    assert 0.35 < float(np.mean(vols)) < 0.80, vols


def test_returns_have_fat_tails(paths):
    for p in paths:
        r = _logret(p.bars)
        kurt = float(((r - r.mean()) ** 4).mean() / r.var() ** 2)
        assert kurt > 5.0, f"kurtosis {kurt:.2f} is too Gaussian for 5m crypto"


def test_first_order_autocorrelation_is_small_and_negative(paths):
    acfs = [float(np.corrcoef(_logret(p.bars)[:-1], _logret(p.bars)[1:])[0, 1]) for p in paths]
    mean = float(np.mean(acfs))
    assert -0.10 < mean < 0.0, f"acf(1)={mean:.4f}: implausible for an efficient market"


def test_volatility_clusters_and_decays_slowly(paths):
    for p in paths:
        a = np.abs(_logret(p.bars))
        near = float(np.corrcoef(a[:-1], a[1:])[0, 1])
        far = float(np.corrcoef(a[:-288], a[288:])[0, 1])
        assert near > 0.10, "no volatility clustering"
        assert far > 0.0, "volatility memory dies too fast"
        assert far < near, "long-lag dependence should be weaker than short-lag"


def test_intraday_seasonality_is_present(paths):
    """Volume should vary systematically across the UTC day."""
    p = paths[0]
    vol = p.bars["volume"].to_numpy()
    hour = (p.bars["ts"].to_numpy() // 3_600_000) % 24
    means = np.array([vol[hour == h].mean() for h in range(24)])
    assert means.max() / means.min() > 1.3


def test_order_flow_is_unbiased_and_uncorrelated_with_volume(paths):
    """A spurious volume/direction link would fake every order-flow feature."""
    for p in paths:
        share = p.bars["taker_buy_base"].to_numpy() / p.bars["volume"].to_numpy()
        assert abs(float(share.mean()) - 0.5) < 0.02
        ofi = 2.0 * share - 1.0
        assert abs(float(np.corrcoef(p.bars["volume"].to_numpy(), ofi)[0, 1])) < 0.05


def test_truth_is_not_reachable_from_the_bars(paths):
    """Latent state must be returned separately so no strategy can consume it."""
    for p in paths:
        assert set(p.truth.columns) & set(p.bars.columns) == {"ts"}


def test_simulation_is_reproducible():
    a = simulate(2_000, seed=11).bars
    b = simulate(2_000, seed=11).bars
    assert np.array_equal(a["close"].to_numpy(), b["close"].to_numpy())
    c = simulate(2_000, seed=12).bars
    assert not np.array_equal(a["close"].to_numpy(), c["close"].to_numpy())


def test_random_walk_control_has_no_reversion():
    bars = make_random_walk(40_000, seed=1)
    validate(bars)
    r = np.diff(np.log(bars["close"].to_numpy()))
    assert abs(float(np.corrcoef(r[:-1], r[1:])[0, 1])) < 0.03


def test_block_bootstrap_preserves_the_marginal_distribution():
    original = simulate(20_000, seed=2).bars
    surrogate = block_bootstrap(original, seed=2)
    validate(surrogate)
    a = np.diff(np.log(original["close"].to_numpy()))
    b = np.diff(np.log(surrogate["close"].to_numpy()))
    assert b.std() == pytest.approx(a.std(), rel=0.25)


def test_validate_rejects_broken_frames():
    bars = simulate(500, seed=0).bars
    broken = bars.copy()
    broken.loc[10, "high"] = broken.loc[10, "low"] - 1.0
    with pytest.raises(SchemaError):
        validate(broken)

    gapped = bars.drop(index=[20]).reset_index(drop=True)
    with pytest.raises(SchemaError):
        validate(gapped, strict=True)
    validate(gapped, strict=False)  # tolerated when explicitly allowed
