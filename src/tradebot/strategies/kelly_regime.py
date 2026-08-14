"""Fractional-Kelly exposure gated by a multi-horizon crowd-regime vote."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


@register
class KellyRegime(Strategy):
    """Size growth-optimally (fractional Kelly, vol-targeted) while the crowd regime stays bullish.

    Game-theoretic grounding: Bell & Cover (1980, Math. OR) prove the
    log-optimal (Kelly) portfolio is the equilibrium strategy of the
    two-investor zero-sum investment game - no rule beats it with
    probability above one half - and Kelly (1956) / Breiman (1961)
    establish its growth optimality. Evstigneev, Hens & Schenk-Hoppe
    (2009) show Kelly-proportional rules are the survivors, the
    evolutionarily stable strategies, of the market-selection game. Full
    Kelly is famously fragile to estimation error (MacLean, Thorp &
    Ziemba 2010), so exposure is a FRACTION of it, expressed as a
    volatility target and hard-capped.

    The regime gate comes from the mean-field view of Cardaliaguet &
    Lehalle (2018, Math. Fin. Econ.): trend drift IS the crowd's net
    flow, so positive expected drift - the precondition for any positive
    Kelly fraction - holds while the mean field is still accumulating.
    Price above its slow anchors is the bar-visible signature of that
    accumulation; when the crowd turns distributor, the Kelly fraction of
    a negative-drift bet is zero, so the strategy stands flat rather than
    shorting a historically upward-drifting asset.

    Mechanism: three slow anchors (30/50/100-day means with a 1% band and
    latching hysteresis) vote on the regime; the vote fraction scales an
    exposure of target_vol / realized_vol, capped at max_leverage, sized
    as a fraction of EQUITY notional so spot and futures risk the same.
    A 10% deadband keeps turnover to a few trades a month.
    """

    name = "kelly_regime"
    warmup = 100 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (30, 50, 100), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10) -> None:
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # Crowd-regime vote: latch bullish above the anchor, bearish below,
        # hold the previous verdict inside the band (hysteresis, not chop).
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

        # Fractional-Kelly sizing: exposure ~ target_vol / realized_vol.
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            v = vol[i]
            scale = min(self.target_vol / v, self.max_leverage) if np.isfinite(v) and v > 0 else 0.0
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
