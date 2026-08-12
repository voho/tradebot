"""RSI mean-reversion: fade oversold/overbought extremes."""

import pandas as pd

from tradebot.indicators import rsi
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class RsiReversion(Strategy):
    """Buy when RSI drops below the oversold level, exit on recovery; mirror short on futures."""

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
