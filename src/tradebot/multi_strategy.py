"""Multi-asset strategy interface and registry (backlog **B-32**).

``tradebot.strategy.Strategy`` sees one instrument at a time: ``on_bar``'s
``Context`` exposes exactly one price series and one position. Five research
rounds (R-63, R-65, R-67, R-68) built a causal, bar-by-bar cross-sectional
allocator across an 8-instrument panel entirely inside ``experiments/``,
using a bespoke simulation loop, because that family cannot be expressed as
a ``Strategy`` and no registration path existed for it -- exactly the gap
``tradebot.multiasset``'s own module docstring names ("this design cannot
express a strategy that needs a shared risk or leverage budget decided
*during* the run"). This module is that registration path.

Mirrors ``tradebot.strategy``/``tradebot.registry``'s shape deliberately, one
level up: a strategy sees the WHOLE aligned panel on every decision instead
of one instrument, and returns a full bar x asset target-weight matrix
instead of one order at a time.

Registered separately from the single-asset ``_REGISTRY`` in
``tradebot.registry`` -- multi-asset strategies are not instrument-scoped
and do not fit that registry's per-instrument assumptions, and this module
never imports or mutates it.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.multi_engine import (
    START_BALANCE,
    align_frames,
    load_universe,
    simulate_portfolio,
)


class MultiAssetStrategy(ABC):
    """Base class for bar-by-bar cross-asset allocators.

    Subclass, set ``name`` (unique within the multi-asset registry) and
    ``instruments`` (the tickers this strategy needs, drawn from
    ``tradebot.multi_engine.UNIVERSE_8``), optionally ``warmup_days`` (how
    much calendar to align *before* the evaluation window so a lookback
    indicator is warm on the first evaluated bar), and implement
    :meth:`build_targets`.
    """

    name: str = "base"
    instruments: tuple[str, ...] = ()
    warmup_days: int = 0

    @abstractmethod
    def build_targets(self, aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Bar x asset desired-weight matrix, decided at each bar's CLOSE.

        ``aligned`` is ``{ticker: OHLCV frame}``, every frame sharing the
        identical bar index (see :func:`tradebot.multi_engine.align_frames`).
        The returned frame MUST share that same index and have one column
        per ticker in ``aligned``.

        MUST be causal: row ``i`` may depend only on rows ``<= i`` of every
        frame in ``aligned`` (rolling / ewm / a forward loop with state are
        all fine; a whole-series mean, std or quantile is not). This is the
        exact contract the ``experiments/r63_shared.py``-family
        ``build_targets``-shaped functions already satisfy -- wrapping one
        of those is a thin adapter, not new logic.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """One-line description used in reports, mirroring
        ``tradebot.strategy.Strategy.describe``."""
        return (self.__doc__ or "").strip().splitlines()[0] if self.__doc__ else ""


_MULTI_REGISTRY: dict[str, type[MultiAssetStrategy]] = {}
_DISCOVERED = False


def register_multi_asset(cls: type[MultiAssetStrategy]) -> type[MultiAssetStrategy]:
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(f"{cls.__name__} must set a unique 'name' class attribute")
    if not getattr(cls, "instruments", None):
        raise ValueError(f"{cls.__name__} must set a non-empty 'instruments' tuple")
    if name in _MULTI_REGISTRY and _MULTI_REGISTRY[name] is not cls:
        raise ValueError(f"duplicate multi-asset strategy name: {name!r}")
    _MULTI_REGISTRY[name] = cls
    return cls


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    import tradebot.multi_strategies as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
    _DISCOVERED = True


def available_multi_asset_strategies() -> dict[str, type[MultiAssetStrategy]]:
    """Name -> class for every registered multi-asset strategy."""
    _discover()
    return dict(_MULTI_REGISTRY)


#: The window every registered multi-asset strategy is evaluated on by
#: default: R-63/R-65/R-67/R-68's own W_FULL6 (2020-04-01 -> the last
#: committed bar). One shared constant so ``run.py`` and
#: ``scripts/inference.py`` cannot drift from each other on what "the
#: multi-asset run" means.
DEFAULT_WINDOW: tuple[str, str | None] = ("2020-04-01", None)

PORTFOLIO_MARKET = "portfolio"


def get_multi_asset_strategy(name: str) -> MultiAssetStrategy:
    """Instantiate a registered multi-asset strategy by name."""
    _discover()
    try:
        return _MULTI_REGISTRY[name]()
    except KeyError:
        known = ", ".join(sorted(_MULTI_REGISTRY)) or "(none)"
        raise KeyError(f"unknown multi-asset strategy {name!r}; available: {known}") from None


def run_multi_asset_backtest(
    strategy: MultiAssetStrategy,
    data_dir: str | Path,
    market: MarketSpec,
    start_balance: float = START_BALANCE,
    window: tuple[str, str | None] = DEFAULT_WINDOW,
) -> pd.Series:
    """Load, align, decide and simulate one multi-asset strategy end to end.

    Pads the alignment window by ``strategy.warmup_days`` on the left so a
    lookback indicator inside :meth:`build_targets` is warm at the first bar
    of ``window``, then slices the padded targets/prices back down to
    ``window`` before handing them to
    :func:`tradebot.multi_engine.simulate_portfolio` -- the same
    warm-then-slice pattern ``experiments/r63_novel_xsmom_rank.py``'s
    ``warm_window``/``build_cell`` and its R-65/R-67/R-68 descendants use,
    so a strategy imported from that lineage needs no adaptation here.

    The padding/slicing is an ENGINE concern, not a :meth:`build_targets`
    one: a strategy's ``build_targets`` is a pure function of whatever
    aligned panel it is handed, causal by construction, and does not need to
    know whether that panel was padded.
    """
    frames = load_universe(strategy.instruments, data_dir)

    start, end = window
    if strategy.warmup_days > 0:
        warm_start = str(
            (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=strategy.warmup_days)).date()
        )
    else:
        warm_start = start
    aligned_padded = align_frames(frames, (warm_start, end))

    targets_padded = strategy.build_targets(aligned_padded)
    if list(targets_padded.columns) != list(strategy.instruments):
        raise ValueError(
            f"{strategy.name}.build_targets returned columns "
            f"{list(targets_padded.columns)}, expected {list(strategy.instruments)}"
        )
    if not targets_padded.index.equals(aligned_padded[strategy.instruments[0]].index):
        raise ValueError(f"{strategy.name}.build_targets changed the bar index")

    idx = aligned_padded[strategy.instruments[0]].index
    idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        idx = idx[idx < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]

    aligned = {t: df.loc[idx] for t, df in aligned_padded.items()}
    targets = targets_padded.loc[idx]

    return simulate_portfolio(targets, aligned, market, start_balance)
