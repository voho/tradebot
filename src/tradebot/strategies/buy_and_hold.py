"""Buy and hold: the baseline every strategy must beat."""

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class BuyAndHold(Strategy):
    """Buy 100% on the first bar and hold to the end."""

    name = "buy_and_hold"
    warmup = 0

    def on_bar(self, ctx: Context) -> None:
        if ctx.i == self.warmup and not ctx.in_market:
            ctx.order_target(1.0)
