"""RSI mean-reversion: fade oversold/overbought extremes."""

import pandas as pd

from tradebot.indicators import rsi
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class RsiReversion(Strategy):
    """Mean-reversion: buy oversold dips (RSI < 30), exit on recovery; mirror short overbought on futures.

    The idea: sharp moves overshoot; when the 14-period RSI drops under 30
    the sell-off is statistically stretched and price tends to snap back.
    Enter long on the dip, exit once RSI recovers past 55 (don't wait for
    overbought). On futures the mirror applies: short RSI > 70, cover
    below 45. Works in ranging markets; bleeds in strong trends, where
    "oversold" keeps getting more oversold.
    """

    name = "rsi_reversion"
    warmup = 150

    def __init__(self, period: int = 14, oversold: float = 30.0, exit_long: float = 55.0,
                 overbought: float = 70.0, exit_short: float = 45.0) -> None:
        self.period = period
        self.oversold, self.exit_long = oversold, exit_long
        self.overbought, self.exit_short = overbought, exit_short

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = rsi(df["close"], self.period)
        return df

    def on_bar(self, ctx: Context) -> None:
        r = float(ctx.bar["rsi"])
        pos = ctx.position
        if pos == 0.0:
            if r < self.oversold:
                ctx.order_target(1.0)
            elif r > self.overbought and ctx.can_short:
                ctx.order_target(-1.0)
        elif pos > 0.0 and r > self.exit_long:
            ctx.close_position()
        elif pos < 0.0 and r < self.exit_short:
            ctx.close_position()
