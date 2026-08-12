"""Strategy registry: adding a strategy = one decorated class in one file.

Any module dropped into ``tradebot/strategies/`` is imported automatically;
classes decorated with ``@register`` become available to the runner and CLI.
"""

from __future__ import annotations

import importlib
import pkgutil

from tradebot.strategy import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}
_DISCOVERED = False


def register(cls: type[Strategy]) -> type[Strategy]:
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(f"{cls.__name__} must set a unique 'name' class attribute")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"duplicate strategy name: {name!r}")
    _REGISTRY[name] = cls
    return cls


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    import tradebot.strategies as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
    _DISCOVERED = True


def available_strategies() -> dict[str, type[Strategy]]:
    """Name -> class for every registered strategy."""
    _discover()
    return dict(_REGISTRY)


def get_strategy(name: str) -> Strategy:
    """Instantiate a registered strategy by name."""
    _discover()
    try:
        return _REGISTRY[name]()
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown strategy {name!r}; available: {known}") from None
