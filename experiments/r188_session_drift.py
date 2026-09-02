"""Adaptive hour-of-day seasonality: hold the hours whose trailing drift is significant."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY, day_id, slot
from tradebot.strategy import Context, Strategy


@register
class SessionDrift(Strategy):
    """Long (short) during the UTC hours whose trailing mean return is significantly positive (negative).

    Sources: Eross, McGroarty, Urquhart & Wolfe (2019), "The intraday
    dynamics of bitcoin", Research in International Business and Finance
    49:71-81, document intraday seasonality in Bitcoin returns, volume and
    volatility keyed to the US and European sessions; Baur, Cahill, Godfrey
    & Liu (2019), "Bitcoin time-of-day, day-of-week and month-of-year
    effects in returns and trading volume", Finance Research Letters
    31:78-92, find the volume effects robust and the return effects weak.
    This repo's R-75 found BTC's hour-of-day *volatility* pattern real but
    its day-of-week *return* pattern indistinguishable from noise; the
    hour-of-day return pattern, traded directly, is the untested cell.

    Mechanism. At the end of every day the strategy computes, for each of
    the 24 UTC hours, the mean 5-minute return over the past
    ``lookback_days`` days and its t-statistic. During the next day it is
    long in hours with t above ``t_min``, short (futures only) in hours
    below ``-t_min``, and flat otherwise, moving at hour boundaries. The
    turnover is set by how many hours change sign - typically one to a few
    round trips a day - and there are no other parameters.
    """

    name = "session_drift"
    warmup = 100 * BARS_PER_DAY

    def __init__(self, lookback_days: int = 90, t_min: float = 2.5,
                 exposure: float = 1.0) -> None:
        self.lookback_days = lookback_days
        self.t_min = t_min
        self.exposure = exposure

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        day = day_id(df.index)
        hour = df.index.hour.to_numpy()
        r = np.log(df["close"]).diff().fillna(0.0)

        # per (day, hour) sufficient statistics, rolled over the past days
        key = pd.MultiIndex.from_arrays([day, hour], names=["day", "hour"])
        stats = pd.DataFrame({"s": r.to_numpy(), "q": r.to_numpy() ** 2, "n": 1.0},
                             index=key).groupby(level=["day", "hour"]).sum()
        wide = stats.unstack("hour").fillna(0.0)
        # reindex to every day so a missing (day, hour) cell counts as zero
        all_days = np.arange(day.max() + 1)
        wide = wide.reindex(all_days).fillna(0.0)
        lb = self.lookback_days
        roll = wide.rolling(lb, min_periods=max(10, lb // 2)).sum()
        s, q, n = roll["s"], roll["q"], roll["n"]
        mean = s / n
        var = (q / n - mean ** 2).clip(lower=1e-18)
        t = (mean / np.sqrt(var / n)).shift(1)  # yesterday's table decides today
        t.columns = t.columns.astype(int)

        # decide for the hour the NEXT bar falls in: fills happen at the next open
        slots = slot(df.index)
        next_hour = np.where(slots % 12 == 11, (hour + 1) % 24, hour)
        next_day = np.where((slots % 12 == 11) & (hour == 23), day + 1, day)
        next_day = np.minimum(next_day, day.max())
        t_vals = t.reindex(columns=range(24)).to_numpy()
        tv = t_vals[next_day, next_hour]
        target = np.where(tv > self.t_min, self.exposure,
                          np.where(tv < -self.t_min, -self.exposure, 0.0))
        df["target"] = np.where(np.isfinite(tv), target, 0.0)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
