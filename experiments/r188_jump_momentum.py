"""Post-jump continuation: follow Lee-Mykland (2008) jumps for a fixed number of bars."""

import math

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY
from tradebot.strategy import Context, Strategy


def lee_mykland_threshold(n_per_day: int, alpha: float) -> float:
    """Critical value of the Lee-Mykland (2008) jump statistic, in local sigmas.

    ``|L| > beta* S_n + C_n`` rejects "no jump" at size ``alpha``, with the
    Gumbel-limit constants from the paper, ``c = sqrt(2/pi)``.
    """
    c = math.sqrt(2.0 / math.pi)
    root = math.sqrt(2.0 * math.log(n_per_day))
    c_n = root / c - (math.log(math.pi) + math.log(math.log(n_per_day))) / (2.0 * c * root)
    s_n = 1.0 / (c * root)
    beta = -math.log(-math.log(1.0 - alpha))
    return beta * s_n + c_n


@register
class JumpMomentum(Strategy):
    """After a statistically flagged 5-minute jump, hold its direction for ``hold_bars`` bars.

    Sources: Lee & Mykland (2008), "Jumps in financial markets: a new
    nonparametric test and jump dynamics", Review of Financial Studies
    21(6):2535-2563 - the per-observation jump test used here, a return
    divided by the local bipower-variation volatility with a Gumbel
    critical value; Scaillet, Treccani & Trevisan (2020), "High-frequency
    jump analysis of the Bitcoin market", Journal of Financial Econometrics
    18(2):209-232, who find Bitcoin jumps frequent, clustered, and tied to
    order-flow imbalance rather than reversed by it. This repo's
    ``overshoot_fade`` (L-13) traded the *reversal* after a forced move and
    lost; this is the untested complement, continuation.

    Mechanism. ``L_i = r_i / sigma_i`` with ``sigma_i`` the square root of
    the bipower variation over the previous ``window`` bars. When ``|L_i|``
    exceeds the Lee-Mykland critical value at size ``alpha`` (about 5.4
    local sigmas at 1% for 288 bars a day), the position is set to the
    jump's sign for ``hold_bars`` bars; a same-direction jump extends the
    hold, an opposite one flips it. One to a few trades a day, depending on
    how often BTC jumps.
    """

    name = "jump_momentum"
    warmup = 2 * BARS_PER_DAY

    def __init__(self, window: int = 270, alpha: float = 0.01, hold_bars: int = 12,
                 exposure: float = 1.0) -> None:
        self.window = window
        self.alpha = alpha
        self.hold_bars = hold_bars
        self.exposure = exposure

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        r = np.log(df["close"]).diff()
        # bipower variation over the previous `window` bars, excluding r_i itself
        bp = (r.abs() * r.abs().shift(1)).rolling(self.window - 2, min_periods=50).mean().shift(1)
        sigma = np.sqrt(bp * (math.pi / 2.0))
        stat = (r / sigma).to_numpy()
        crit = lee_mykland_threshold(BARS_PER_DAY, self.alpha)

        n = len(df)
        target = np.zeros(n)
        pos, until = 0.0, -1
        for i in range(n):
            s = stat[i]
            if np.isfinite(s) and abs(s) > crit:
                pos = math.copysign(self.exposure, s)
                until = i + self.hold_bars
            elif i >= until:
                pos = 0.0
            target[i] = pos
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
