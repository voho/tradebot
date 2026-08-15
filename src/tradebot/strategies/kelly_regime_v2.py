"""kelly_regime with a convex confidence response to the regime vote."""

from tradebot.registry import register
from tradebot.strategies.kelly_regime import KellyRegime


@register
class KellyRegimeV2(KellyRegime):
    """Like kelly_regime, but treats partial anchor agreement as low confidence, not half a signal.

    The change is one line: exposure scales with `vote_fraction ** 1.75`
    instead of `vote_fraction`. Everything else - the 30/50/100-day latched
    anchors, the fractional-Kelly volatility target, the 2x cap, the
    deadband - is identical.

    Why: the vote fraction estimates P(bull), but the growth-optimal
    (Kelly) exposure scales with expected drift over variance, and that
    relationship is convex in agreement rather than linear. The partial
    states (1/3, 2/3) are exactly the transitional ones - a fast anchor has
    flipped, a slow one has not - where drift is near zero and variance is
    elevated. Wood, Roberts & Zohren (2022, JFDS) identify the period right
    after a turning point as where momentum strategies do worst; a linear
    response over-invests precisely there. Shrinking those states
    super-linearly is also the standard fractional-Kelly prescription under
    parameter uncertainty (MacLean, Thorp & Ziemba 2010): when the edge
    estimate is least reliable, shrink toward zero faster than linearly.

    Evidence and honesty: gamma anywhere in 1.25-4.0 improved return,
    Sharpe, drawdown AND turnover simultaneously, in both the in-sample and
    out-of-sample windows - a plateau rather than a lucky spike, which is
    the opposite of the overfitting signature. It remains a searched
    parameter and the measured gains sit inside the +/-0.2 Sharpe
    path-noise band on any single metric; the reason to credit it is that
    many metrics move together across both windows. See docs/VALIDATION.md.
    """

    name = "kelly_regime_v2"

    def __init__(self, vote_gamma: float = 1.75, **kwargs) -> None:
        super().__init__(vote_gamma=vote_gamma, **kwargs)
