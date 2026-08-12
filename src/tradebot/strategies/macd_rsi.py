"""MACD trend filter combined with RSI pullback timing."""

import pandas as pd

from tradebot.indicators import crossed_above, crossed_below, macd, rsi
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class MacdRsi(Strategy):
    """Enter longs on RSI pullback recoveries while MACD trend is up; mirror short on futures."""

    name = "macd_rsi"
    warmup = 150

    def __init__(self, rsi_period: int = 14, entry_long: float = 45.0, exit_long: float = 75.0,
                 entry_short: float = 55.0, exit_short: float = 25.0) -> None:
        self.rsi_period = rsi_period
        self.entry_long, self.exit_long = entry_long, exit_long
        self.entry_short, self.exit_short = entry_short, exit_short

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        m = macd(df["close"])
        r = rsi(df["close"], self.rsi_period)
        df["trend_up"] = m["macd_hist"] > 0
        df["rsi"] = r
        df["rsi_up"] = crossed_above(r, pd.Series(self.entry_long, index=df.index))
        df["rsi_dn"] = crossed_below(r, pd.Series(self.entry_short, index=df.index))
        return df

    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar
        trend_up = bool(bar["trend_up"])
        r = float(bar["rsi"])
        pos = ctx.position

        if pos == 0.0:
            if trend_up and bar["rsi_up"]:
                ctx.order_target(1.0)
            elif not trend_up and bar["rsi_dn"] and ctx.can_short:
                ctx.order_target(-1.0)
        elif pos > 0.0 and (not trend_up or r > self.exit_long):
            ctx.close_position()
        elif pos < 0.0 and (trend_up or r < self.exit_short):
            ctx.close_position()
