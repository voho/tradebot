"""Market intraday momentum (Gao, Han, Li & Zhou 2018): the session's first hours predict its last."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY, day_id, session_open, slot
from tradebot.strategy import Context, Strategy


@register
class IntradayMomentum(Strategy):
    """Hold the last hours of the UTC day in the direction of its first hours; one trade a day.

    Source: Gao, Han, Li & Zhou (2018), "Market intraday momentum", Journal
    of Financial Economics 129(2):394-414. On SPY the first half-hour return
    predicts the last half-hour return with the same sign; the effect is
    strongest on volatile, high-volume days and is attributed to infrequent
    rebalancers who trade late in the day and to late-informed traders.
    Follow-ups (Zhang, Ma & Zhu 2019; Jin, Kearney, Li & Yang 2020) report
    the same shape on Chinese and Australian index futures.

    Mechanism, on UTC sessions. Let ``r_first`` be the log return over the
    first ``first_hours`` of the day. Over the last ``last_hours`` of the
    day the position is sign(r_first) - long, or short on futures (flat on
    spot) - provided ``|r_first|`` exceeds ``threshold`` trailing standard
    deviations of that opening return, so that quiet openings are not
    traded. The position is closed at the day's end. One round trip a day
    at most, each costing two taker fees.
    """

    name = "intraday_momentum"
    warmup = 25 * BARS_PER_DAY

    def __init__(self, first_hours: int = 4, last_hours: int = 4,
                 threshold: float = 0.5, exposure: float = 1.0,
                 vol_days: int = 20) -> None:
        self.first_hours = first_hours
        self.last_hours = last_hours
        self.threshold = threshold
        self.exposure = exposure
        self.vol_days = vol_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        day = day_id(df.index)
        slots = slot(df.index)
        close = df["close"]
        open_ = session_open(df, day)

        first_end = self.first_hours * 12 - 1  # last slot of the opening window
        logret = np.log(close) - np.log(open_)
        at_end = slots == first_end
        # opening return, known from the end of the opening window onward
        r_first = logret.where(at_end)
        # its trailing dispersion across previous days (one observation per day)
        opening = r_first[at_end]
        sigma_daily = opening.rolling(self.vol_days, min_periods=10).std().shift(1)
        sigma = pd.Series(np.nan, index=df.index)
        sigma[at_end] = sigma_daily
        r_first = r_first.groupby(day).ffill()
        sigma = sigma.groupby(day).ffill()

        entry_slot = BARS_PER_DAY - self.last_hours * 12 - 1  # signal bar
        in_window = (slots >= entry_slot) & (slots < BARS_PER_DAY - 1)
        strong = (r_first.abs() > self.threshold * sigma).to_numpy()
        sign = np.sign(r_first.to_numpy())
        target = np.where(in_window & strong & np.isfinite(sign), sign * self.exposure, 0.0)
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
