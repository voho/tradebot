"""Risk management: volatility targeting, drawdown governor, exposure caps.

The sizing game in :mod:`gtbot.game.equilibrium` decides *conviction*; this
layer decides how much capital conviction is allowed to command.  Keeping the
two separate matters, because they answer different questions and fail in
different ways: a conviction error costs one trade, a risk error ends the
account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    #: Annualised volatility the strategy targets on its own equity curve.
    target_vol_annual: float = 0.15
    #: Hard cap on gross exposure as a multiple of equity.
    max_leverage: float = 2.0
    #: Position change below this is not worth the spread.
    min_position: float = 0.02
    #: Drawdown at which exposure starts being cut back.
    drawdown_soft: float = 0.06
    #: Drawdown at which the strategy stops taking risk entirely.
    drawdown_hard: float = 0.15
    #: Bars over which realised strategy volatility is estimated.
    vol_halflife: float = 1000.0
    #: Cap on the vol-targeting multiplier, so a quiet patch cannot lever up.
    max_vol_scalar: float = 12.0


class RiskManager:
    """Stateful risk governor, updated once per bar."""

    def __init__(self, cfg: RiskConfig, bars_per_year: float):
        self.cfg = cfg
        self.bars_per_year = bars_per_year
        self._var = 0.0
        self._alpha = 1.0 - math.exp(-math.log(2.0) / max(cfg.vol_halflife, 1.0))
        self._peak_equity = 0.0
        self._n = 0

    def observe(self, bar_return: float, equity: float) -> None:
        self._var += self._alpha * (bar_return * bar_return - self._var)
        self._peak_equity = max(self._peak_equity, equity)
        self._n += 1

    @property
    def realized_vol_annual(self) -> float:
        return math.sqrt(max(self._var, 0.0) * self.bars_per_year)

    @property
    def drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return 0.0  # filled in by :meth:`current_drawdown`

    def current_drawdown(self, equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self._peak_equity)

    def vol_scalar(self) -> float:
        """Multiplier that steers realised volatility toward the target.

        Until enough history has accumulated the estimate is unreliable, so the
        multiplier stays at 1 rather than levering up on a small sample.
        """
        if self._n < 200 or self._var <= 0.0:
            return 1.0
        realized = self.realized_vol_annual
        if realized <= 1e-9:
            return self.cfg.max_vol_scalar
        return min(self.cfg.target_vol_annual / realized, self.cfg.max_vol_scalar)

    def drawdown_scalar(self, equity: float) -> float:
        """Linear de-risking between the soft and hard drawdown thresholds."""
        dd = self.current_drawdown(equity)
        if dd <= self.cfg.drawdown_soft:
            return 1.0
        if dd >= self.cfg.drawdown_hard:
            return 0.0
        span = max(self.cfg.drawdown_hard - self.cfg.drawdown_soft, 1e-9)
        return float(1.0 - (dd - self.cfg.drawdown_soft) / span)

    def apply(self, raw_position: float, equity: float, *, vol_target: bool = True) -> float:
        """Turn a conviction-scaled position into a permitted position.

        ``vol_target=False`` skips the volatility scalar, for the fixed-size
        mode where the caller has already decided the exposure and only wants
        the drawdown governor and the leverage cap applied.
        """
        scale = self.vol_scalar() if vol_target else 1.0
        pos = raw_position * scale * self.drawdown_scalar(equity)
        pos = max(-self.cfg.max_leverage, min(self.cfg.max_leverage, pos))
        if abs(pos) < self.cfg.min_position:
            return 0.0
        return float(pos)
