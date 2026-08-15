"""kelly_regime_v3 with faster regime anchors (20/40/80 days)."""

from tradebot.registry import register
from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3


@register
class KellyRegimeV4(KellyRegimeV3):
    """v3 on a doubling anchor ladder (20/40/80 days) instead of 30/50/100.

    Same conditional volatility targeting as v3; only the regime anchors
    change, to a clean doubling ladder - each anchor covers twice the
    horizon of the last, the same a-priori structure MACD (12/26) and
    HAR (daily/weekly/monthly) use, rather than the ad-hoc 30/50/100.

    What the evidence supports, stated precisely: across nine anchor sets
    in the 18-28 day range, EVERY variant cut max drawdown to 35-39% from
    the base's 41.8%, and seven of nine scored Sharpe >= 1.52. That
    drawdown reduction is the robust finding. The Sharpe spread across
    the plateau (1.52-1.60) sits inside the +/-0.2 path-noise band
    measured by block bootstrap, so the return improvement should NOT be
    read as established - only the risk reduction is. Below ~18 days the
    plateau breaks sharply (16/32/64 scores 1.46), so this is a region,
    not a peak that was tuned to.

    Beta-tested and promoted: it beats v3 on full period, drawdown,
    out-of-sample and the Monte Carlo left tail. See docs/VALIDATION.md.
    """

    name = "kelly_regime_v4"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
