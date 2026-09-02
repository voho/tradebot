"""Intraday noise-area breakout (Zarattini, Barbon & Aziz 2024) on UTC sessions."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import (BARS_PER_DAY, day_id, previous_session_close,
                              session_open, slot)
from tradebot.strategy import Context, Strategy


@register
class NoiseAreaBreakout(Strategy):
    """Break out of a time-of-day-scaled noise band around the session open; flat by day end.

    Source: Zarattini, Barbon & Aziz (2024), "Beat the Market: An Effective
    Intraday Momentum Strategy for the S&P500 ETF (SPY)", SSRN 4824172. On
    SPY (2007-2024, ~$0.0035/share commissions) the rule earned a Sharpe of
    1.33 against 0.60 for buy-and-hold, at roughly one trade a day.

    Mechanism. For every time-of-day slot the *noise area* is the average
    absolute move from the session open at that slot over the previous
    ``lookback_days`` sessions. The upper band is the higher of the session
    open and the previous close, times (1 + noise); the lower band the lower
    of the two, times (1 - noise). A close above the upper band opens a long
    (below the lower band, a short on futures; spot stays flat), the band
    itself is the trailing stop - a close back inside the area exits - and
    everything is closed at the end of the UTC day. Position size targets
    ``target_vol`` of daily volatility, capped at 1x equity.

    Intraday, 1-3 round trips a day. The paper's market has commissions two
    orders of magnitude below this repo's 0.10% taker, which is the number
    this test is really about.
    """

    name = "noise_area_breakout"
    warmup = 20 * BARS_PER_DAY

    def __init__(self, lookback_days: int = 14, band_mult: float = 1.5,
                 check_every: int = 6, target_vol: float = 0.02,
                 max_exposure: float = 1.0) -> None:
        self.lookback_days = lookback_days
        self.band_mult = band_mult
        self.check_every = check_every  # bars between decisions (6 = 30 min)
        self.target_vol = target_vol  # daily volatility target
        self.max_exposure = max_exposure

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        day = day_id(df.index)
        slots = slot(df.index)
        close = df["close"]
        open_ = session_open(df, day)
        prev_close = previous_session_close(df, day)

        move = (close / open_ - 1.0).abs()
        lb = self.lookback_days
        # same-slot average over the previous sessions only (shift excludes today)
        noise = move.groupby(slots).transform(
            lambda s: s.rolling(lb, min_periods=max(3, lb // 2)).mean().shift(1))
        upper = np.maximum(open_, prev_close) * (1.0 + self.band_mult * noise)
        lower = np.minimum(open_, prev_close) * (1.0 - self.band_mult * noise)

        r = np.log(close).diff()
        daily_vol = (r.ewm(span=14 * BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
                     * np.sqrt(BARS_PER_DAY)).shift(1)
        size = np.minimum(self.target_vol / daily_vol, self.max_exposure).to_numpy()

        c = close.to_numpy(dtype=float)
        ub = upper.to_numpy(dtype=float)
        lo = lower.to_numpy(dtype=float)
        last_slot = BARS_PER_DAY - 1
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            if slots[i] == last_slot:
                pos = 0.0  # flat over the session boundary
            elif slots[i] % self.check_every == 0 and np.isfinite(ub[i]) and np.isfinite(size[i]):
                if c[i] > ub[i]:
                    pos = size[i]
                elif c[i] < lo[i]:
                    pos = -size[i]
                elif pos > 0 and c[i] < ub[i]:
                    pos = 0.0  # back inside the noise area: trailing stop
                elif pos < 0 and c[i] > lo[i]:
                    pos = 0.0
            target[i] = pos
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
