"""Multi-asset strategy registration (backlog **B-32**).

Two kinds of check. The first is generic to the registration path itself:
the registry is separate from the single-asset one, discovery works, and a
registered strategy's ``build_targets`` is causal (the truncation probe
pattern ``experiments/r63_shared.py::check_causality`` established, applied
here to a registered strategy instead of an experiment function). The
second is specific to this round's registered candidate,
``xsmom_entry_band``: it must reproduce R-68's own already-published numbers
(``docs/LEDGER.md`` `### R-68`, `reports/r68_band/conservative_cells.csv`)
through the NEW infrastructure, to machine precision -- the primary
correctness check on this module, more important than anything else here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.broker import MarketSpec
from tradebot.multi_engine import UNIVERSE_6, align_frames, load_universe
from tradebot.multi_strategy import (
    MultiAssetStrategy,
    available_multi_asset_strategies,
    get_multi_asset_strategy,
    register_multi_asset,
    run_multi_asset_backtest,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# R-68's own published numbers for the frozen ENTRY_ONLY d=0.080 cell,
# W_FULL6/U6, spot @0.10% (SPOT_BASE) -- docs/LEDGER.md `### R-68` and
# reports/r68_band/conservative_cells.csv (the `config_kind=="decision",
# bench=="VOLMATCH_HOLD"` row). Reproducing these exactly through the new
# registration infrastructure is this module's primary correctness check.
R68_FINAL_BALANCE = 6533.706771810798
R68_N_BARS = 671271
R68_N_DAYS = 2332
R68_CAND_DD_PCT = 75.03027378053862  # daily-return-based max drawdown
R68_MATCHED_HOLD_GROWTH_DIFF = 0.9272591378159367  # vs MATCHED_HOLD, prices


# ------------------------------------------------------------------ registry


def test_registry_is_separate_from_the_single_asset_one():
    from tradebot.registry import available_strategies

    single = set(available_strategies())
    multi = set(available_multi_asset_strategies())
    # The two registries share no names -- they are genuinely separate
    # dicts, not two views onto one namespace.
    assert single.isdisjoint(multi)


def test_xsmom_entry_band_is_registered():
    strategies = available_multi_asset_strategies()
    assert "xsmom_entry_band" in strategies
    strat = get_multi_asset_strategy("xsmom_entry_band")
    assert isinstance(strat, MultiAssetStrategy)
    assert strat.instruments == UNIVERSE_6
    assert strat.describe()  # non-empty one-line docstring


def test_register_multi_asset_requires_name_and_instruments():
    with pytest.raises(ValueError):
        @register_multi_asset
        class _NoInstruments(MultiAssetStrategy):
            name = "_test_no_instruments"
            instruments = ()

            def build_targets(self, aligned):
                raise NotImplementedError

    with pytest.raises(ValueError):
        @register_multi_asset
        class _NoName(MultiAssetStrategy):
            instruments = ("BTC",)

            def build_targets(self, aligned):
                raise NotImplementedError


# ---------------------------------------------------------------- causality


@pytest.fixture(scope="module")
def real_aligned_u6():
    """A real, aligned U6 slice -- long enough to warm the 80-day anchor and
    still leave bars on both sides of the truncation cut."""
    frames = load_universe(UNIVERSE_6, DATA_DIR)
    # ~150 days: comfortably longer than the 91-day warmup this strategy
    # needs, short enough to keep the test fast.
    return align_frames(frames, ("2021-01-01", "2021-05-31"))


def test_xsmom_entry_band_build_targets_is_causal(real_aligned_u6):
    """Truncation probe: rebuilding on data cut short must not change any
    row strictly before the cut. Mirrors
    `experiments/r63_shared.py::check_causality`, applied to the registered
    strategy's own `build_targets` rather than an experiment function."""
    strat = get_multi_asset_strategy("xsmom_entry_band")
    idx = next(iter(real_aligned_u6.values())).index
    cut = len(idx) - 5_000
    assert cut > 1, "fixture window too short for the truncation probe"

    full = strat.build_targets(real_aligned_u6)
    truncated_aligned = {t: df.iloc[:cut] for t, df in real_aligned_u6.items()}
    trunc = strat.build_targets(truncated_aligned)

    m = min(cut, len(trunc))
    a = np.nan_to_num(full.iloc[:m].to_numpy(dtype=float), nan=0.0)
    b = np.nan_to_num(trunc.iloc[:m].to_numpy(dtype=float), nan=0.0)
    assert np.allclose(a, b, atol=1e-12, rtol=0.0)


def test_build_targets_output_shape_matches_input(real_aligned_u6):
    strat = get_multi_asset_strategy("xsmom_entry_band")
    targets = strat.build_targets(real_aligned_u6)
    idx = next(iter(real_aligned_u6.values())).index
    assert targets.index.equals(idx)
    assert list(targets.columns) == list(strat.instruments)
    # Long-only, unlevered: every weight in [0, 1] and each row sums <= 1.
    w = targets.to_numpy(dtype=float)
    assert np.all(w >= -1e-12)
    assert np.all(w.sum(axis=1) <= 1.0 + 1e-9)


# ------------------------------------------------------- run_multi_asset_backtest


def test_run_multi_asset_backtest_pads_and_slices_to_the_window():
    """A short window still gets a warm first bar, because the engine pads
    the alignment left by `warmup_days` before calling `build_targets`."""
    strat = get_multi_asset_strategy("xsmom_entry_band")
    market = MarketSpec.spot()
    eq = run_multi_asset_backtest(strat, DATA_DIR, market, start_balance=1_000.0,
                                  window=("2021-06-01", "2021-06-30"))
    assert eq.index[0] >= pd.Timestamp("2021-06-01", tz="UTC")
    assert eq.index[-1] < pd.Timestamp("2021-07-01", tz="UTC")
    assert (eq > 0).all()


# ------------------------------------------------------- R-68 reproduction


def test_xsmom_entry_band_reproduces_r68s_published_numbers():
    """The primary correctness check on this whole module (B-32): running
    R-68's frozen ENTRY_ONLY d=0.080 configuration through the NEW
    registration infrastructure must reproduce R-68's own already-published
    W_FULL6/U6 numbers to machine precision, not merely approximately."""
    strat = get_multi_asset_strategy("xsmom_entry_band")
    market = MarketSpec.spot()  # 0.10% taker, R-68's SPOT_BASE
    eq = run_multi_asset_backtest(strat, DATA_DIR, market, start_balance=1_000.0,
                                  window=("2020-04-01", None))

    assert len(eq) == R68_N_BARS
    assert eq.iloc[-1] == pytest.approx(R68_FINAL_BALANCE, rel=1e-12)

    from tradebot.inference import daily_returns, max_drawdown_from_returns

    dr = daily_returns(eq).to_numpy(dtype=float)
    assert len(dr) == R68_N_DAYS
    assert max_drawdown_from_returns(dr) == pytest.approx(R68_CAND_DD_PCT, rel=1e-9)
