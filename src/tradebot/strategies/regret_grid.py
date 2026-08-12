"""Discounted regret-matching+ over a discrete position grid."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class RegretGrid(Strategy):
    """Regret-matching+ over a position grid: correlated-equilibrium play against the market.

    Game-theoretic grounding: Hart & Mas-Colell (2000, Econometrica) - playing
    each action with probability proportional to its positive cumulative
    regret drives average regret to zero at O(1/sqrt(T)) and empirical play
    to the set of correlated equilibria; the proof rests on Blackwell (1956,
    Pacific J. Math.) approachability. The RM+ clipping (regrets floored at
    zero, from Tammelin 2014's CFR+ work) speeds adaptation. Framed as a
    repeated zero-sum game against the market (von Neumann 1928), no-regret
    play earns at least the game value (Freund & Schapire 1999, GEB) - and
    with a flat action on the grid that value is at least "do nothing".

    Mechanism: actions are positions {-1,-0.6,-0.3,0,0.3,0.6,1}. Each bar,
    every grid action's counterfactual vol-normalized payoff (market PnL
    minus the fee its adoption would have cost) updates a discounted
    positive-regret vector; the played position is the regret-weighted mean
    action (payoffs are linear in the action, so the mean earns the mixed
    strategy's expectation while, by Jensen, paying no more in fees).
    A hysteresis band keeps re-targets sparse.
    """

    name = "regret_grid"
    warmup = 2000

    def __init__(self, discount: float = 0.9995, hysteresis: float = 0.05,
                 fee_rate: float = 0.0005) -> None:
        self.discount = discount
        self.hysteresis = hysteresis
        self.fee_rate = fee_rate

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        r_a = r.to_numpy()
        sig_a = sig1.shift(1).to_numpy()

        grid = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
        reg = np.zeros(len(grid))
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        pos_prev = 0.0
        pos_prev2 = 0.0
        for i in range(1, n):
            s = sig_a[i]
            if not np.isfinite(s) or s <= 0:
                target[i] = pos
                continue
            z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
            fee_n = min(self.fee_rate / (3.0 * s), 0.25)
            u = grid * z_t - fee_n * np.abs(grid - pos_prev)
            u_play = pos_prev * z_t - fee_n * abs(pos_prev - pos_prev2)
            reg = np.maximum(self.discount * reg + (u - u_play), 0.0)
            total = reg.sum()
            x = float(reg @ grid / total) if total > 1e-12 else 0.0
            if abs(x - pos) > self.hysteresis or (x > 0) != (pos > 0) or (x < 0) != (pos < 0):
                pos = x
            target[i] = pos
            pos_prev2 = pos_prev
            pos_prev = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
