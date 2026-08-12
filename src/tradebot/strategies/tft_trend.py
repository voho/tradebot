"""Generous tit-for-tat truce with the trend, with a grim trigger for betrayals."""

import numpy as np
import pandas as pd

from tradebot.indicators import ema
from tradebot.registry import register
from tradebot.strategy import Context, Strategy

PEACE, LONG, SHORT, GRIM = 0, 1, 2, 3


@register
class TitForTatTrend(Strategy):
    """Repeated-game trend truce: hold while the market cooperates, forgive one defection, punish two.

    Game-theoretic grounding: Axelrod (1984, The Evolution of Cooperation) -
    tit-for-tat won the repeated prisoner's dilemma tournaments by being
    nice, retaliatory, forgiving and clear. Strict TFT is brittle under
    noise, so we play GENEROUS tit-for-tat (Nowak & Sigmund 1992, Nature):
    a single adverse close beyond the noise scale is forgiven; two
    defections inside the forgiveness window are punished by exiting. A
    grim trigger (Friedman 1971, Rev. Econ. Studies) handles catastrophic
    single-bar betrayals: go flat and refuse to play until the market
    proves calm again. Forgiveness is turnover control - each state change
    costs fees, so the strategy only "defects" on confirmed betrayal.

    Mechanism: enter a truce (long/short) when a fast EMA clears a slow EMA
    by an ATR-scaled hurdle with near-unanimous closes; ratchet an ATR
    trail; a close through the trail is a defection (forgiven once per
    window); a G_GRIM*ATR adverse bar triggers GRIM until N_RESET calm bars.
    """

    name = "tft_trend"
    warmup = 1300

    def __init__(self, n_conf: int = 6, k_defect: float = 1.25, w_forgive: int = 12,
                 g_grim: float = 3.0, n_reset: int = 96, fee_hurdle: float = 0.003) -> None:
        self.n_conf = n_conf
        self.k_defect = k_defect
        self.w_forgive = w_forgive
        self.g_grim = g_grim
        self.n_reset = n_reset
        self.fee_hurdle = fee_hurdle

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        c = close.to_numpy()

        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 48, min_periods=48).mean().to_numpy()

        e_fast = ema(close, 24).to_numpy()
        e_slow = ema(close, 96).to_numpy()
        up = (close.diff() > 0).rolling(self.n_conf).sum().to_numpy()
        dn = (close.diff() < 0).rolling(self.n_conf).sum().to_numpy()

        n = len(df)
        target = np.zeros(n)
        mode = PEACE
        trail = 0.0
        entry = 0.0
        last_defect = -10**9
        calm = 0
        for i in range(n):
            a = atr[i]
            if not np.isfinite(a) or not np.isfinite(e_slow[i]) or a <= 0:
                target[i] = 0.0
                continue
            hurdle = max(3.0 * 0.001, self.fee_hurdle, a / c[i])

            if mode == GRIM:
                calm = calm + 1 if (high[i] - low[i]) < 4.0 * a else 0
                if calm >= self.n_reset:
                    mode = PEACE
                target[i] = 0.0
            elif mode == PEACE:
                if e_fast[i] > e_slow[i] * (1.0 + hurdle) and up[i] >= self.n_conf - 1:
                    mode, entry = LONG, c[i]
                    trail = c[i] - self.k_defect * a
                    last_defect = -10**9
                    target[i] = 1.0
                elif e_fast[i] < e_slow[i] * (1.0 - hurdle) and dn[i] >= self.n_conf - 1:
                    mode, entry = SHORT, c[i]
                    trail = c[i] + self.k_defect * a
                    last_defect = -10**9
                    target[i] = -1.0
                else:
                    target[i] = 0.0
            elif mode == LONG:
                trail = max(trail, c[i] - self.k_defect * a)
                if (entry - c[i]) > self.g_grim * a:
                    mode, calm = GRIM, 0
                    target[i] = 0.0
                elif c[i] < trail:
                    if i - last_defect <= self.w_forgive:
                        mode = PEACE  # second defection: punish by exiting
                        target[i] = 0.0
                    else:
                        last_defect = i  # forgive the first defection
                        target[i] = 1.0
                else:
                    target[i] = 1.0
            else:  # SHORT
                trail = min(trail, c[i] + self.k_defect * a)
                if (c[i] - entry) > self.g_grim * a:
                    mode, calm = GRIM, 0
                    target[i] = 0.0
                elif c[i] > trail:
                    if i - last_defect <= self.w_forgive:
                        mode = PEACE
                        target[i] = 0.0
                    else:
                        last_defect = i
                        target[i] = -1.0
                else:
                    target[i] = -1.0

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
