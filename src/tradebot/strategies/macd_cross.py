"""Classic MACD signal-line crossover."""

import pandas as pd

from tradebot.indicators import crossed_above, crossed_below, macd
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class MacdCross(Strategy):
    """Long when MACD crosses above its signal line; flat (spot) or short (futures) on the cross below."""

    name = "macd_cross"
    warmup = 150  # let the EMAs stabilize

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast, self.slow, self.signal = fast, slow, signal

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        m = macd(df["close"], self.fast, self.slow, self.signal)
        df["macd_up"] = crossed_above(m["macd"], m["macd_signal"])
        df["macd_dn"] = crossed_below(m["macd"], m["macd_signal"])
        return df

    def on_bar(self, ctx: Context) -> None:
        if ctx.bar["macd_up"]:
            ctx.order_target(1.0)
        elif ctx.bar["macd_dn"]:
            ctx.order_target(-1.0 if ctx.can_short else 0.0)
