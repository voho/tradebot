"""Mean-field-game equilibrium inventory: own belief net of the crowd's transient impact."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY
from tradebot.strategy import Context, Strategy


@register
class MfgCrowding(Strategy):
    """Hold the slow-trend belief minus ``gamma`` times the crowd's recent chase - the MFG equilibrium inventory.

    Game-theoretic grounding: in the mean-field game of Casgrain &
    Jaimungal (2020), "Mean-field games with differing beliefs for
    algorithmic trading", Mathematical Finance 30(3):995-1034 (building on
    Cardaliaguet & Lehalle 2018, "Mean field game of controls and an
    application to trade crowding", Mathematics and Financial Economics
    12:335-363), each agent's equilibrium trading is linear in the gap
    between *its own* drift belief and the *crowd's* aggregate belief, whose
    trading moves the price transiently. The equilibrium therefore leans
    into a trend the agent believes in and against the part of the recent
    move that is just the crowd's own impact - it is trend-following with a
    built-in anti-crowding correction.

    Mechanism. Own belief: the ``slow_days`` log return in daily-sigma
    units, squashed by tanh. Crowd chase: the ``fast_days`` return, same
    units. Exposure ``= clip(tanh(slow) - gamma * tanh(fast), -1, 1)``
    times ``exposure`` of equity, re-targeted only outside ``deadband``.
    ``gamma = 0`` is the pure slow-trend control; the whole test is whether
    ``gamma > 0`` helps. Shorts on futures only. Not ``kelly_regime``: no
    volatility target and no latched vote - the crowd term subtracts rather
    than gates.
    """

    name = "mfg_crowding"
    warmup = 60 * BARS_PER_DAY

    def __init__(self, slow_days: int = 20, fast_days: int = 2, gamma: float = 0.5,
                 exposure: float = 1.0, deadband: float = 0.10) -> None:
        self.slow_days = slow_days
        self.fast_days = fast_days
        self.gamma = gamma
        self.exposure = exposure
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        logc = np.log(df["close"])
        r = logc.diff()
        daily_vol = (r.ewm(span=30 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
                     * np.sqrt(BARS_PER_DAY)).shift(1)
        z_slow = (logc - logc.shift(self.slow_days * BARS_PER_DAY)) / (daily_vol * np.sqrt(self.slow_days))
        z_fast = (logc - logc.shift(self.fast_days * BARS_PER_DAY)) / (daily_vol * np.sqrt(self.fast_days))
        desired = np.clip(np.tanh(z_slow) - self.gamma * np.tanh(z_fast), -1.0, 1.0) * self.exposure
        desired = np.nan_to_num(desired.to_numpy(), nan=0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            d = desired[i]
            if abs(d - pos) > self.deadband:
                pos = d
            target[i] = pos
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
