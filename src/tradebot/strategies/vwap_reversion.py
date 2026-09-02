"""Fade standardized deviations from the session VWAP, only when the gap pays the fees."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY, day_id, session_vwap, slot
from tradebot.strategy import Context, Strategy


@register
class VwapReversion(Strategy):
    """Enter against a z-scored deviation from the UTC-session VWAP, exit at the VWAP or day end.

    Grounding: Hendershott & Menkveld (2014), "Price pressures", Journal of
    Financial Economics 114(3):405-423, measure transitory price pressure
    from intermediaries' inventory that mean-reverts over hours, and VWAP
    reversion is the standard intraday expression of it (Kakushadze & Serur
    2018, "151 Trading Strategies", the VWAP section). This repo's
    ``attrition_reversion`` (L-24) already showed a bar-close mean-reversion
    rule paying two taker fees per round trip loses; this variant adds the
    two things that rule lacked - an anchor that resets every session and
    an entry that is refused unless the deviation itself exceeds
    ``min_edge_mult`` round-trip fees.

    Mechanism. Deviation ``d = log(close / vwap)`` is standardized by its
    own typical size at this time of day - the mean ``|d|`` at the same
    5-minute slot over the previous ``lookback_days`` sessions (a random-walk
    ``sigma * sqrt(elapsed)`` scaling was tried first and made entries
    all but impossible late in the day). When ``z < -entry_z`` (and ``|d|``
    clears the fee hurdle) the strategy goes long; when ``z > entry_z`` it
    goes short on futures (flat on spot). It exits when the deviation
    changes sign or at the end of the day. Under one round trip a day at
    the frozen ``entry_z``.
    """

    name = "vwap_reversion"
    warmup = 25 * BARS_PER_DAY

    def __init__(self, entry_z: float = 3.0, min_edge_mult: float = 1.5,
                 lookback_days: int = 20, fee_rate: float = 0.001,
                 exposure: float = 1.0) -> None:
        self.entry_z = entry_z
        self.min_edge_mult = min_edge_mult
        self.lookback_days = lookback_days
        self.fee_rate = fee_rate
        self.exposure = exposure

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        day = day_id(df.index)
        slots = slot(df.index)
        close = df["close"]
        vwap = session_vwap(df, day)
        dev_s = np.log(close) - np.log(vwap)
        lb = self.lookback_days
        # typical |deviation| at this slot over the previous sessions only
        scale = dev_s.abs().groupby(slots).transform(
            lambda s: s.rolling(lb, min_periods=max(5, lb // 2)).mean().shift(1))
        # mean absolute deviation -> standard deviation for a normal
        z = (dev_s / (1.2533 * scale)).to_numpy()
        dev = dev_s.to_numpy()
        min_edge = self.min_edge_mult * 2.0 * self.fee_rate

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        last_slot = BARS_PER_DAY - 1
        for i in range(n):
            if slots[i] == last_slot or not np.isfinite(z[i]):
                pos = 0.0
            elif pos == 0.0:
                if z[i] < -self.entry_z and -dev[i] > min_edge:
                    pos = self.exposure
                elif z[i] > self.entry_z and dev[i] > min_edge:
                    pos = -self.exposure
            elif (pos > 0 and dev[i] >= 0.0) or (pos < 0 and dev[i] <= 0.0):
                pos = 0.0  # the deviation closed: take the reversion
            target[i] = pos
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
