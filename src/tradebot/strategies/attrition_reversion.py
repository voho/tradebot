"""Reservation-price mean reversion with war-of-attrition time-cost exits."""

import numpy as np
import pandas as pd

from tradebot.indicators import ema
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class AttritionReversion(Strategy):
    """Fade deviations from an inventory-shifted fair value; quit when waiting costs exceed the prize.

    Game-theoretic grounding: Avellaneda & Stoikov (2008, Quantitative
    Finance) - a market maker's fair price is the market's fair price
    shifted against inventory, r = m - q*gamma*sigma^2; the optimal spread
    around it is exactly a fee-aware no-trade band, and the inventory term
    demands ever-larger dislocations to add exposure (an automatic
    anti-martingale governor). Holding an underwater reversion position is
    a war of attrition (Maynard Smith 1974, J. Theor. Biol.): each bar has
    a waiting cost and the ESS quits when accumulated cost matches the
    prize (the expected snap-back). Fudenberg & Tirole (1986, Econometrica)
    add that NON-reversion is itself information - every bar the deviation
    persists, raise the posterior that the move is structural and shorten
    remaining patience.

    Mechanism: fair value = 1-day EMA of typical price; deviation measured
    in ATR units from the inventory-shifted reservation price; entries need
    a 2.5-ATR dislocation whose half-reversion clears 3x fees; exits on
    convergence, on attrition (waiting cost + extension pessimism > prize),
    with at most one reservation-gated add.
    """

    name = "attrition_reversion"
    warmup = 1300

    def __init__(self, inv_gamma: float = 0.6, z_entry: float = 2.5, z_exit: float = 0.5,
                 c_wait: float = 0.04, q_step: float = 0.5) -> None:
        self.inv_gamma = inv_gamma
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.c_wait = c_wait
        self.q_step = q_step

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        fair = ema(typical, 288).to_numpy()

        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 48, min_periods=48).mean().to_numpy()

        c = close.to_numpy()
        r = np.log(close).diff().to_numpy()

        n = len(df)
        target = np.zeros(n)
        q = 0.0
        bars_held = 0
        attrition = 0.0
        for i in range(n):
            a = atr[i]
            if not np.isfinite(a) or not np.isfinite(fair[i]) or a <= 0:
                q = 0.0
                target[i] = 0.0
                continue
            reservation = fair[i] - q * self.inv_gamma * a
            dev = (c[i] - reservation) / a

            if q == 0.0:
                bars_held = 0
                attrition = 0.0
                edge = 0.5 * abs(dev) * a / c[i]  # assume half-reversion
                if dev > self.z_entry and edge > 3.0 * 0.001:
                    q = -self.q_step  # fade rich price
                elif dev < -self.z_entry and edge > 3.0 * 0.001:
                    q = self.q_step  # fade cheap price
            else:
                bars_held += 1
                prize = 0.5 * abs(dev)
                extension = max(0.0, -np.sign(q) * r[i] / (a / c[i]))
                attrition += self.c_wait + 0.1 * extension
                if abs(dev) < self.z_exit:
                    q = 0.0  # prize collected
                elif attrition > prize:
                    q = 0.0  # opponent is strong: quit the war
                elif (abs(dev) > self.z_entry + self.inv_gamma and abs(q) < 1.0
                      and bars_held > 6):
                    q = min(max(q + np.sign(q) * self.q_step, -1.0), 1.0)  # add once
            target[i] = q

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
