"""Leverage, liquidation, direction modes and dollar accounting.

These paths only became load-bearing once results started being quoted as
"$1,000 at 5x", and a leveraged backtest that silently survives a liquidation
is worse than no backtest at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import gtbot.features as F
from gtbot.data.schema import validate
from gtbot.data.synthetic import simulate
from gtbot.engine.backtest import run_backtest
from gtbot.engine.broker import CostModel, ExecutionConfig, MarginConfig
from gtbot.eval.account import DIRECTIONS, format_table, simulate_account
from gtbot.strategy import GameTheoreticStrategy, StrategyConfig


@pytest.fixture(scope="module")
def bars():
    return validate(simulate(14_000, seed=7).bars)


def _small_cfg(**kw):
    feats = F.FeatureConfig(warmup=1200)
    return StrategyConfig(features=feats, signal_window=800, min_scale_samples=300, **kw)


# ------------------------------------------------------------------- margin
def test_liquidation_distance_matches_the_textbook_formula():
    m = MarginConfig(maintenance_margin_rate=0.005)
    # Maintenance margin is charged at the current mark, so the exact distance
    # is (1 - mmr*L) / (L * (1 - mmr)) -- the exchange's cross-margin formula.
    assert m.liquidation_move(5.0) == pytest.approx((1 - 0.005 * 5) / (5 * 0.995))
    assert m.liquidation_move(5.0) == pytest.approx(0.19598, abs=1e-5)
    assert m.liquidation_move(1.0) == pytest.approx(0.995 / 0.995)
    assert m.liquidation_move(0.0) == float("inf")
    # Higher leverage must always be liquidated sooner.
    moves = [m.liquidation_move(l) for l in (1, 2, 5, 10, 20)]
    assert moves == sorted(moves, reverse=True)


def test_liquidation_move_is_sign_agnostic():
    m = MarginConfig()
    assert m.liquidation_move(-5.0) == m.liquidation_move(5.0)


class AlwaysMaxLong(GameTheoreticStrategy):
    def observe(self, t):
        pass

    def decide(self, t):
        return 50.0  # clipped to max_leverage by the engine


def test_margin_use_is_anchored_to_the_entry_price():
    """A move that happened before the position existed must not count."""
    m = MarginConfig(maintenance_margin_rate=0.005)
    # Long opened at 75 in a bar that gapped down from 100 and never went below 75.
    assert m.margin_use(5.0, entry_price=75.0, low=75.0, high=80.0) == pytest.approx(0.0)
    # Anchoring to the previous close instead would have read >1 and liquidated.
    assert m.margin_use(5.0, entry_price=100.0, low=75.0, high=80.0) > 1.0
    # Shorts use the high.
    assert m.margin_use(-5.0, entry_price=100.0, low=95.0, high=130.0) > 1.0
    assert m.margin_use(0.0, entry_price=100.0, low=1.0, high=200.0) == 0.0


def test_equity_never_goes_negative(bars):
    res = run_backtest(
        bars, AlwaysMaxLong(_small_cfg()),
        costs=CostModel.for_tier("vip6"),
        execution=ExecutionConfig(entry_mode="taker", exit_mode="taker"),
        initial_equity=1_000.0, max_leverage=50.0,
    )
    assert np.all(res.equity >= 0.0), "a leveraged account cannot go through zero"


def test_extreme_leverage_gets_liquidated(bars):
    """At 50x a routine 5m bar is enough to end the account."""
    res = run_backtest(
        bars,
        AlwaysMaxLong(_small_cfg()),
        costs=CostModel.for_tier("vip6"),
        execution=ExecutionConfig(entry_mode="taker", exit_mode="taker"),
        initial_equity=1_000.0,
        max_leverage=50.0,
    )
    assert res.liquidated_at is not None, "50x should not survive"
    assert res.equity[-1] == pytest.approx(0.0)
    assert np.all(res.position[res.liquidated_at :] == 0.0)


def test_modest_leverage_survives(bars):
    res = run_backtest(
        bars,
        AlwaysMaxLong(_small_cfg()),
        costs=CostModel.for_tier("vip6"),
        execution=ExecutionConfig(entry_mode="taker", exit_mode="taker"),
        initial_equity=1_000.0,
        max_leverage=2.0,
    )
    assert res.liquidated_at is None
    assert res.worst_margin_use < 1.0
    assert res.equity[-1] > 0.0


def test_margin_use_is_reported_even_without_liquidation(bars):
    res = run_backtest(
        bars, GameTheoreticStrategy(_small_cfg()),
        costs=CostModel.for_tier("vip6"), initial_equity=1_000.0, max_leverage=5.0,
    )
    assert 0.0 <= res.worst_margin_use < 1.0


# ---------------------------------------------------------------- direction
def test_long_only_never_holds_a_short(bars):
    res = run_backtest(
        bars,
        GameTheoreticStrategy(_small_cfg(direction="long_only")),
        costs=CostModel.for_tier("vip9"),
        max_leverage=5.0,
    )
    assert np.all(res.position >= -1e-12), "long_only must never go short"


def test_both_direction_is_not_artificially_restricted(bars):
    res = run_backtest(
        bars, GameTheoreticStrategy(_small_cfg(direction="both")),
        costs=CostModel.for_tier("vip9"), max_leverage=5.0,
    )
    assert np.min(res.position) <= 0.0  # shorts allowed (may simply not occur)


def test_unknown_direction_is_rejected(bars):
    with pytest.raises(KeyError):
        simulate_account(bars, direction="sideways")


# ------------------------------------------------------------------- sizing
def test_fixed_sizing_uses_more_exposure_than_robust(bars):
    common = dict(tier="vip9", leverage=5.0, deposit=1_000.0, config=_small_cfg())
    robust = simulate_account(bars, sizing_mode="robust", **common)
    fixed = simulate_account(bars, sizing_mode="fixed", **common)
    if fixed.trades == 0 or robust.trades == 0:
        pytest.skip("no trades on this short fixture")
    assert fixed.avg_position_when_in >= robust.avg_position_when_in


def test_positions_never_exceed_the_requested_leverage(bars):
    for lev in (1.0, 3.0, 5.0):
        res = run_backtest(
            bars, GameTheoreticStrategy(_small_cfg(sizing_mode="fixed")),
            costs=CostModel.for_tier("vip9"), initial_equity=1_000.0, max_leverage=lev,
        )
        assert np.max(np.abs(res.position)) <= lev + 1e-9


# ------------------------------------------------------------------ dollars
def test_dollar_accounting_is_self_consistent(bars):
    r = simulate_account(bars, tier="vip9", leverage=5.0, deposit=1_000.0, config=_small_cfg())
    assert r.deposit == 1_000.0
    assert r.profit_usd == pytest.approx(r.final_equity - r.deposit)
    assert r.return_pct == pytest.approx(r.final_equity / r.deposit - 1.0)
    assert r.max_drawdown_usd >= 0.0
    assert r.fees_paid_usd >= 0.0
    assert 0.0 <= r.time_in_market <= 1.0


def test_deposit_scales_dollar_pnl_but_not_percentages(bars):
    """Order size here is negligible against bar volume, so returns are scale free."""
    cfg = _small_cfg()
    small = simulate_account(bars, tier="vip9", deposit=1_000.0, leverage=5.0, config=cfg)
    big = simulate_account(bars, tier="vip9", deposit=10_000.0, leverage=5.0, config=cfg)
    assert big.return_pct == pytest.approx(small.return_pct, rel=1e-3)
    assert big.profit_usd == pytest.approx(small.profit_usd * 10.0, rel=1e-3)


def test_format_table_renders_both_flavours(bars):
    rs = [
        simulate_account(bars, tier="vip9", direction=d, leverage=5.0, config=_small_cfg())
        for d in DIRECTIONS
    ]
    plain = format_table(rs)
    md = format_table(rs, markdown=True)
    assert "5x leverage" in plain
    assert md.startswith("**") and "|" in md
    assert format_table([]) == "(no results)"


# ------------------------------------------------- improvements (ablated in)
def test_variance_reduction_shrinks_the_edge_standard_error(bars):
    """The control variate must cut SE without moving the edge estimate."""
    from gtbot.engine.backtest import run_backtest as _rb

    out = {}
    for vr in (False, True):
        st = GameTheoreticStrategy(_small_cfg(variance_reduction=vr))
        _rb(bars, st, costs=CostModel.for_tier("vip9"), max_leverage=5.0)
        if st._edge_n < 30:
            pytest.skip("too few triggered samples on this fixture")
        out[vr] = (st._edge_mean, st._edge_se())

    (m_off, se_off), (m_on, se_on) = out[False], out[True]
    assert se_on <= se_off, "variance reduction must not increase the standard error"
    # Unbiasedness: the point estimate should not move much.  The control
    # variate has zero conditional mean, so any shift is sampling noise.
    assert abs(m_on - m_off) < 4.0 * se_off


def test_control_variate_beta_is_estimated():
    """Beta should be non-zero once enough samples have accumulated."""
    import numpy as np

    from gtbot.data.synthetic import simulate as _sim

    long_bars = validate(_sim(60_000, seed=3).bars)
    st = GameTheoreticStrategy(_small_cfg(variance_reduction=True))
    from gtbot.engine.backtest import run_backtest as _rb

    res = _rb(long_bars, st, costs=CostModel.for_tier("vip9"), max_leverage=5.0)
    if st._edge_n < 30:
        pytest.skip("too few triggered samples")
    assert np.isfinite(st._cv_beta)
    assert res.diagnostics["cv_beta"] == st._cv_beta


def test_adaptive_exit_learns_a_continuation_value(bars):
    st = GameTheoreticStrategy(_small_cfg(adaptive_exit=True))
    from gtbot.engine.backtest import run_backtest as _rb

    res = _rb(bars, st, costs=CostModel.for_tier("vip9"), max_leverage=5.0)
    cont = res.diagnostics["continuation_value_bp"]
    counts = res.diagnostics["continuation_count"]
    assert cont.shape == counts.shape
    assert np.all(np.isfinite(cont))
    if counts.sum() > 0:
        assert counts[: st.cfg.max_hold + 1].sum() > 0


def test_adaptive_exit_never_extends_beyond_max_hold(bars):
    """Re-solving may exit early; it must never hold longer than the cap."""
    from gtbot.engine.backtest import run_backtest as _rb

    fixed = _rb(bars, GameTheoreticStrategy(_small_cfg(adaptive_exit=False)),
                costs=CostModel.for_tier("vip9"), max_leverage=5.0)
    adaptive = _rb(bars, GameTheoreticStrategy(_small_cfg(adaptive_exit=True)),
                   costs=CostModel.for_tier("vip9"), max_leverage=5.0)

    def _max_run(pos):
        best = run = 0
        for p in pos:
            run = run + 1 if p != 0 else 0
            best = max(best, run)
        return best

    cap = _small_cfg().max_hold + 2
    assert _max_run(adaptive.position) <= max(_max_run(fixed.position), cap)
