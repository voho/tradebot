"""Tests for tradebot.multiasset: does the composition step itself look
ahead, and does it combine curves the way its docstring claims.

Uses small synthetic fixtures (make_ohlcv), not the real BTC/ETH data
this module was designed against — the real-data cross-check against
experiments/kelly_regime_dual_fixed.py's own numbers lives in
experiments/b17_multiasset_adapter.py (R-49) and is not re-run in CI.
"""

import numpy as np
import pytest

from conftest import make_ohlcv
from tradebot.broker import MarketSpec
from tradebot.multiasset import (MultiAssetSpec, combine_equity_curves,
                                 run_multi_backtest)
from tradebot.strategy import Context, Strategy

MARKET = MarketSpec.spot()


class BuyAndHoldToy(Strategy):
    name = "_test_toy_buy_and_hold"
    warmup = 0

    def on_bar(self, ctx: Context) -> None:
        if not ctx.in_market:
            ctx.order_target(1.0)


class PeekerToy(Strategy):
    """Deliberately looks one bar ahead - must break the causality check
    below, the way test_causality_strict.py's own Peeker does. Records
    every decision so the test can compare ORDERS at a fixed bar rather
    than downstream equity, which would conflate "the peek changed the
    decision" with "the tampered bar's own price differs" (fills lag a
    decision by one bar, but the tampered region's price data does not)."""

    name = "_test_toy_peeker"
    warmup = 5

    def __init__(self):
        self.decisions: dict[int, float] = {}

    def prepare(self, df):
        self._df = df
        return df

    def on_bar(self, ctx: Context) -> None:
        nxt = float(self._df["close"].iloc[ctx.i + 1])
        target = 1.0 if nxt > float(self._df["close"].iloc[ctx.i]) else -1.0
        self.decisions[ctx.i] = target
        ctx.order_target(target)


def _two_leg_specs(strategy_cls, seed_a=1, seed_b=2, n=400, offset_b=False):
    """Two legs of ``n`` synthetic 5m bars each. ``offset_b=True`` starts
    leg B two months after leg A (used only by the "flat before a late
    leg starts" test below) — every other test keeps both legs on the
    SAME start date, because a truncation-based causality check compares
    a full run against a shorter one, and if a leg's OWN last bar differs
    between those two runs its mark-to-market equity legitimately differs
    too (more bars of price movement seen = a different final value) —
    that is real, not a lookahead bug, but forward-filling it across a
    calendar gap into a comparison point can look like one. Same start
    date for both legs sidesteps that trap entirely."""
    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)
    df_a = make_ohlcv(100.0 * np.cumprod(1.0 + rng_a.normal(0, 0.01, n)))
    b_start = "2025-03-01" if offset_b else "2025-01-01"
    df_b = make_ohlcv(50.0 * np.cumprod(1.0 + rng_b.normal(0, 0.01, n)), start=b_start)
    return [
        MultiAssetSpec("A", strategy_cls(), df_a, MARKET),
        MultiAssetSpec("B", strategy_cls(), df_b, MARKET),
    ]


def test_weights_must_sum_to_one():
    specs = _two_leg_specs(BuyAndHoldToy)
    with pytest.raises(ValueError, match="sum to 1.0"):
        run_multi_backtest(specs, [0.5, 0.6], 1_000.0)


def test_spec_weight_count_mismatch_rejected():
    specs = _two_leg_specs(BuyAndHoldToy)
    with pytest.raises(ValueError, match="specs but"):
        run_multi_backtest(specs, [1.0], 1_000.0)


def test_portfolio_start_balance_is_the_sum_of_leg_balances():
    specs = _two_leg_specs(BuyAndHoldToy)
    result = run_multi_backtest(specs, [0.5, 0.5], 1_000.0)
    assert result.portfolio.equity.iloc[0] == pytest.approx(1_000.0, abs=0.5)
    assert result.leg_results[0].start_balance == pytest.approx(500.0)
    assert result.leg_results[1].start_balance == pytest.approx(500.0)


def test_combine_equity_curves_flat_before_a_late_leg_starts():
    """A leg starting later contributes its flat start_balance until its
    own data begins, never a peek at its first real value (docstring
    claim in tradebot.multiasset.combine_equity_curves)."""
    specs = _two_leg_specs(BuyAndHoldToy, offset_b=True)
    result = run_multi_backtest(specs, [0.5, 0.5], 1_000.0)
    idx = result.portfolio.equity.index
    b_start = result.leg_results[1].equity.index[0]
    before_b = idx[idx < b_start]
    assert len(before_b) > 0
    a_only = result.leg_results[0].equity.reindex(before_b)
    combined = result.portfolio.equity.reindex(before_b)
    # portfolio == leg A's own equity + leg B's flat, untouched starting balance
    assert np.allclose(combined.to_numpy(), a_only.to_numpy() + 500.0)


def test_composition_does_not_introduce_lookahead():
    """Mirrors test_causality_strict.py's truncation pattern one level up:
    the summed PORTFOLIO curve, not any one leg's causality (already
    covered elsewhere), must be identical before a cut whether or not
    data after the cut exists."""
    specs_full = _two_leg_specs(BuyAndHoldToy)
    full = run_multi_backtest(specs_full, [0.5, 0.5], 1_000.0)

    specs_trunc = _two_leg_specs(BuyAndHoldToy)
    for spec in specs_trunc:
        spec.df = spec.df.iloc[:250].copy()
    trunc = run_multi_backtest(specs_trunc, [0.5, 0.5], 1_000.0)

    idx = trunc.portfolio.equity.index
    diff = (full.portfolio.equity.reindex(idx) - trunc.portfolio.equity).abs()
    assert float(diff.max()) < 1e-9


def test_the_causality_check_catches_a_real_on_bar_peek():
    """Guard the guard: comparing downstream EQUITY across a tamper is not
    enough here — a tampered bar's own price differs between the up/down
    runs regardless of any peek, and a fill lags its decision by one bar,
    so an equity diff at/after the cut would prove nothing. Instead this
    mirrors test_causality_strict.py's own ``_decisions`` approach:
    compare the ORDER queued at a bar strictly before the tamper, which a
    peek changes and nothing else can."""
    cut = 250

    def tampered_specs(direction: float):
        specs = _two_leg_specs(PeekerToy)
        for spec in specs:
            df = spec.df.copy()
            for col in ("open", "high", "low", "close"):
                df.iloc[cut:, df.columns.get_loc(col)] *= direction
            spec.df = df
        return specs

    up_specs = tampered_specs(3.0)
    down_specs = tampered_specs(1 / 3.0)
    run_multi_backtest(up_specs, [0.5, 0.5], 1_000.0)
    run_multi_backtest(down_specs, [0.5, 0.5], 1_000.0)

    decision_bar = cut - 1  # peeks at `cut`, the first tampered row
    up_decisions = [s.strategy.decisions[decision_bar] for s in up_specs]
    down_decisions = [s.strategy.decisions[decision_bar] for s in down_specs]
    assert up_decisions != down_decisions, "the peek detector no longer detects a peek"


def test_combine_equity_curves_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one leg"):
        combine_equity_curves([])


def test_available_multi_asset_strategies_is_empty_today():
    """No registered strategy declares `instruments` yet (R-49: the
    convention exists, nothing uses it) - this pins that fact so it is a
    visible, deliberate change the day a strategy does."""
    from tradebot.multiasset import available_multi_asset_strategies
    assert available_multi_asset_strategies() == {}
