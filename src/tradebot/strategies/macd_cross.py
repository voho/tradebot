"""Classic MACD signal-line crossover."""

import pandas as pd

from tradebot.indicators import crossed_above, crossed_below, macd
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class MacdCross(Strategy):
    """Trend-following: long when MACD crosses above its signal line, flat/short on the cross below.

    The idea: MACD (the gap between a fast and a slow EMA) turns positive
    when momentum shifts upward; crossing its own signal line marks the
    shift early. Ride the new trend until the opposite cross. Classic
    12/26/9 parameters. Weakness: on a 5-minute chart crosses fire
    constantly in sideways chop, so fees and whipsaws eat the edge —
    which is exactly what the baseline comparison should expose.
    """

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
