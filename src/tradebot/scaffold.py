"""Generate a new strategy file: ``tradebot new my_strategy``."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TEMPLATE = '''"""{title}: describe the idea in one line (shown in reports)."""

import pandas as pd

from tradebot.indicators import crossed_above, crossed_below, ema, macd, rsi
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class {cls}(Strategy):
    """One-line description shown by `tradebot list` and in reports."""

    name = "{name}"
    warmup = 150  # bars skipped before the first on_bar call

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        # Parameters with defaults, so the registry can instantiate the
        # strategy without arguments.
        self.fast, self.slow = fast, slow

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Called once with the full OHLCV frame; add indicator columns.
        # MUST be causal: row i may only depend on rows <= i (rolling /
        # ewm / shift are fine). A framework test verifies this for every
        # registered strategy, on spot and futures.
        df["fast_ema"] = ema(df["close"], self.fast)
        df["slow_ema"] = ema(df["close"], self.slow)
        return df

    def on_bar(self, ctx: Context) -> None:
        # Called at every bar close; orders fill at the NEXT bar's open.
        # ctx.bar["col"] reads this bar (incl. prepare() columns);
        # ctx.position / ctx.equity / ctx.can_short describe the account.
        bar = ctx.bar
        if bar["fast_ema"] > bar["slow_ema"] and ctx.position <= 0:
            ctx.order_target(1.0)  # fully long (fraction of equity x leverage)
        elif bar["fast_ema"] < bar["slow_ema"] and ctx.position >= 0:
            # short on futures, flat on spot
            ctx.order_target(-1.0 if ctx.can_short else 0.0)
'''


def new_strategy(name: str, strategies_dir: Path | None = None) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise SystemExit(
            f"invalid strategy name {name!r}: use lowercase letters, digits and _ "
            "(it becomes the module and registry name)")
    if strategies_dir is None:
        strategies_dir = Path(__file__).resolve().parent / "strategies"
    path = strategies_dir / f"{name}.py"
    if path.exists():
        raise SystemExit(f"{path} already exists")

    from tradebot.registry import available_strategies

    if name in available_strategies():
        raise SystemExit(f"a strategy named {name!r} is already registered")

    cls = "".join(part.capitalize() for part in name.split("_"))
    title = name.replace("_", " ").capitalize()
    path.write_text(TEMPLATE.format(name=name, cls=cls, title=title))
    print(f"created {path}", file=sys.stderr)
    print("next steps:", file=sys.stderr)
    print(f"  - edit the indicators, rules and DOCSTRING (the idea) in {path.name}",
          file=sys.stderr)
    print("  - pytest                        # incl. automatic no-lookahead check",
          file=sys.stderr)
    print(f"  - tradebot run --strategies {name} buy_and_hold --max-bars 100000  # quick look",
          file=sys.stderr)
    print("  - tradebot run                  # full matrix; refreshes the README",
          file=sys.stderr)
    print("    (CI requires every strategy to appear in the README comparison table)",
          file=sys.stderr)
    return path
