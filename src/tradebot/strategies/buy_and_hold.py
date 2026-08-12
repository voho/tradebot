"""Buy and hold: the baseline every strategy must beat."""

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class BuyAndHold(Strategy):
    """Buy everything on the first bar and never trade again.

    The idea: BTC has historically rewarded simply holding through both
    bull and bear markets. This is the benchmark every active strategy
    must beat after fees — if a strategy can't outgrow doing nothing, its
    signal adds no value. On leveraged futures it doubles as a stress
    test: a deep enough drawdown liquidates the position, showing what
    leverage does to a passive long.
    """

    name = "buy_and_hold"
    warmup = 0

    def on_bar(self, ctx: Context) -> None:
        if ctx.i == self.warmup and not ctx.in_market:
            ctx.order_target(1.0)
