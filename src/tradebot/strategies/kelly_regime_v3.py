"""kelly_regime that re-sizes only in volatility extremes (conditional targeting)."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR, KellyRegime


@register
class KellyRegimeV3(KellyRegime):
    """Hold notional steady through normal volatility; re-size only when volatility breaks out.

    The incumbent re-sizes continuously: exposure is always
    target_vol / realized_vol. This variant keeps a CONSTANT notional
    while volatility sits in its normal band, and switches to full
    inverse-volatility sizing only when volatility breaks out high or
    low - latching that state until it retraces, exactly the hysteresis
    the regime gate already uses, applied to the risk axis.

    Why, specifically for crypto: Bongaerts, Kang & van Dijk (2020,
    Financial Analysts Journal 76(4)) show conventional continuous
    volatility targeting fails to consistently improve performance and
    can deepen drawdowns, while adjusting exposure only in the
    volatility extremes improves Sharpe and cuts tails at low turnover.
    The reason it bites here is an asset-class fact: Baur & Dimpfl
    (2018, Economics Letters 173) document that crypto has an INVERSE
    leverage effect - positive shocks raise volatility more than
    negative ones. Measured on this data, high-volatility states carry
    the HIGHEST forward Sharpe, so continuous targeting de-levers into
    precisely the best states. Moreira & Muir's (2017, J. Finance)
    volatility-managed alpha, which depends on high volatility
    forecasting low returns, is absent-to-inverted for BTC; what
    survives is Harvey et al.'s (2018, JPM) mechanical tail protection,
    which extreme-only targeting keeps.

    Everything else - the 30/50/100-day latched anchors, fractional
    Kelly, the 2x cap, the 10% deadband - is unchanged.
    """

    name = "kelly_regime_v3"

    def __init__(self, anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55,
                 low_out: float = 0.85, **kwargs) -> None:
        super().__init__(**kwargs)
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df
