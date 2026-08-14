"""Hedge council over the profitable games, risk-shaped by fractional Kelly."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


@register
class ChampionsCouncil(Strategy):
    """Combine the games that actually pay: Hedge over their signals, sized by fractional Kelly.

    Game-theoretic grounding: this is a game of games. Its members are the
    equilibrium plays that survived paper testing - growth-optimal regime
    exposure (kelly_regime), the no-regret expert blend (hedge_experts),
    replicator-dynamics capital reallocation (replicator_book) and Cover's
    universal portfolio (universal_kelly) - plus buy-and-hold, the
    benchmark, and a flat action. Which member is right in the next regime
    is unknowable in advance, so the council allocates with Hedge
    (Freund & Schapire 1997, JCSS; Arora, Hazan & Kale 2012), whose
    external-regret bound guarantees performance within O(sqrt(T ln N)) of
    the best member in hindsight; fixed-share mixing (Herbster & Warmuth
    1998) lets leadership drift between regimes, and the flat action
    anchors the zero-sum maximin floor.

    On top of the blend sits the same fractional-Kelly volatility target
    the winners share (Bell & Cover 1980; MacLean, Thorp & Ziemba 2010),
    so the council's RISK is controlled centrally rather than inherited
    from whichever member happens to lead. Targets are equity-notional
    fractions, so spot and futures take the same risk.
    """

    name = "champions_council"
    warmup = 100 * BARS_PER_DAY + 10

    def __init__(self, eta: float = 0.06, fixed_share: float = 1e-4,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10) -> None:
        self.eta = eta
        self.fixed_share = fixed_share
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband

    @staticmethod
    def _members() -> list[Strategy]:
        from tradebot.strategies.hedge_experts import HedgeExperts
        from tradebot.strategies.kelly_regime import KellyRegime
        from tradebot.strategies.replicator_book import ReplicatorBook
        from tradebot.strategies.universal_kelly import UniversalKelly

        return [KellyRegime(), HedgeExperts(), ReplicatorBook(), UniversalKelly()]

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[["open", "high", "low", "close", "volume"]]
        signals = []
        for member in self._members():
            prepared = member.prepare(base.copy())
            signals.append(np.clip(np.nan_to_num(
                prepared["target"].to_numpy(dtype=np.float64)), -1.0, 1.0))
        signals.append(np.ones(len(df)))   # buy-and-hold: the benchmark member
        signals.append(np.zeros(len(df)))  # flat: the maximin floor
        a = np.column_stack(signals)

        r = np.log(df["close"]).diff()
        r_a = r.to_numpy()
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        n, num = a.shape
        target = np.zeros(n)
        logw = np.zeros(num)
        pos = 0.0
        for i in range(1, n):
            v = vol[i]
            if not np.isfinite(v) or v <= 0 or not np.isfinite(r_a[i]):
                target[i] = pos
                continue
            # per-bar vol-normalized payoff of each member; Hedge update
            bar_vol = v / np.sqrt(BARS_PER_YEAR)
            g = np.clip(a[i - 1] * r_a[i] / (3.0 * bar_vol), -1.0, 1.0)
            logw += self.eta * g
            logw -= logw.max()
            p = np.exp(logw)
            p /= p.sum()
            p = (1.0 - self.fixed_share) * p + self.fixed_share / num
            logw = np.log(p)

            blend = float(p @ a[i])
            desired = blend * min(self.target_vol / v, self.max_leverage)
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
