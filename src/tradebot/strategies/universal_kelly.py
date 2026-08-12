"""Cover's universal portfolio over an exposure grid with a fractional-Kelly cap."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class UniversalKelly(Strategy):
    """Universal-portfolio exposure: wealth-weighted mixture over fixed exposures, half-Kelly capped.

    Game-theoretic grounding: Cover (1991, Mathematical Finance) - the
    wealth-weighted average over all constant exposures achieves the growth
    rate of the best fixed exposure in hindsight to within O(log T) on ANY
    price sequence (minimax-regret optimal per Ordentlich & Cover 1998,
    Math. OR; Dirichlet(1/2) refinement Cover & Ordentlich 1996, IEEE IT).
    Bell & Cover (1980, Math. OR) show log-optimal (Kelly) investment is
    the equilibrium of the two-investor zero-sum game - growth-optimal play
    is competitively unbeatable, not just asymptotic. MacLean, Thorp &
    Ziemba (2010) document full Kelly's fragility to estimation error, so
    the mixture is scaled by a Kelly fraction.

    Mechanism: track discounted log-wealth of 41 fixed exposures b in
    [-1, 1]; the played exposure is the wealth-softmax mean times a 0.5
    Kelly fraction, targeted as a fraction of EQUITY notional (leverage
    independent). The posterior drifts slowly by construction, so
    re-targets are rare and fee-friendly.
    """

    name = "universal_kelly"
    warmup = 2000

    def __init__(self, grid_points: int = 41, memory_bars: int = 8640,
                 kappa: float = 0.5, hysteresis: float = 0.05) -> None:
        self.grid_points = grid_points
        self.memory_bars = memory_bars
        self.kappa = kappa
        self.hysteresis = hysteresis

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy()
        n = len(df)
        grid = np.linspace(-1.0, 1.0, self.grid_points)
        log_wealth = np.zeros(self.grid_points)
        gamma = 1.0 - 1.0 / self.memory_bars

        target = np.zeros(n)
        pos = 0.0
        for i in range(1, n):
            ret = close[i] / close[i - 1] - 1.0
            ret = min(max(ret, -0.5), 0.5)
            log_wealth = gamma * log_wealth + np.log1p(np.clip(grid * ret, -0.95, None))
            w = np.exp(log_wealth - log_wealth.max())
            w /= w.sum()
            b_hat = float(w @ grid)
            x = min(max(self.kappa * b_hat, -1.0), 1.0)
            if abs(x - pos) > self.hysteresis or (x > 0) != (pos > 0) or (x < 0) != (pos < 0):
                pos = x
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity, leverage-independent
