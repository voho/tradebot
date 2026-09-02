"""Parameter-free coin betting (Krichevsky-Trofimov; Orabona & Pal 2016) as a position sizer."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY, slot
from tradebot.strategy import Context, Strategy


@register
class CoinBetting(Strategy):
    """Bet the KT fraction of wealth on the sign of the next round's return; no learning rate anywhere.

    Game-theoretic grounding: Orabona & Pal (2016), "Coin Betting and
    Parameter-Free Online Learning", NeurIPS, reduce online learning to a
    betting game against an adversarial coin. A bettor who wagers the
    Krichevsky-Trofimov fraction ``beta_t = (sum of past outcomes) / t`` of
    current wealth has wealth within a ``sqrt(t)`` factor of the best
    constant bet in hindsight, on *every* sequence, with no learning rate
    or horizon to tune (Krichevsky & Trofimov 1981; Cover 1991 is the
    portfolio version). Cutkosky & Orabona (2018) extend the guarantee to
    unbounded settings. Applied to returns, the KT fraction is a momentum
    estimate shrunk by its own sample size - the sizing this repo's ledger
    says is the only kind of decision that has paid here (the SIZE axis).

    Mechanism. Each round of ``round_bars`` bars yields an outcome
    ``g = clip(round return / (clip_mult * trailing round volatility), -1,
    1)``. With forgetting factor ``discount`` the running sums are
    ``S = discount*S + g`` and ``N = discount*N + 1``, and the position for
    the next round is ``scale * S / (N + 1)`` of equity notional (KT's
    ``+1`` is the Bayesian prior that keeps the first bets small). Shorts on
    futures only. Re-targets inside ``deadband`` are suppressed, so a daily
    round trades a few times a month; a 4-hour round is intraday.
    """

    name = "coin_betting"
    warmup = 40 * BARS_PER_DAY

    def __init__(self, round_bars: int = BARS_PER_DAY, discount: float = 0.99,
                 scale: float = 3.0, clip_mult: float = 2.0, vol_rounds: int = 30,
                 deadband: float = 0.10) -> None:
        if BARS_PER_DAY % round_bars:
            raise ValueError("round_bars must divide a day so rounds align with the clock")
        self.round_bars = round_bars
        self.discount = discount
        self.scale = scale
        self.clip_mult = clip_mult
        self.vol_rounds = vol_rounds
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        slots = slot(df.index)
        round_end = (slots % self.round_bars) == self.round_bars - 1
        logc = np.log(df["close"])
        ends = logc[round_end]
        g_raw = ends.diff()
        vol = g_raw.ewm(span=self.vol_rounds, min_periods=5).std().shift(1)
        g = (g_raw / (self.clip_mult * vol)).clip(-1.0, 1.0)

        s_sum, n_sum = 0.0, 0.0
        bets = pd.Series(np.nan, index=ends.index)
        for ts, gi in g.items():
            if np.isfinite(gi):
                s_sum = self.discount * s_sum + gi
                n_sum = self.discount * n_sum + 1.0
                bets[ts] = self.scale * s_sum / (n_sum + 1.0)
        desired = bets.reindex(df.index).ffill().fillna(0.0).to_numpy()

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            if round_end[i]:
                d = desired[i]
                if abs(d - pos) > self.deadband or (d > 0) != (pos > 0) or (d < 0) != (pos < 0):
                    pos = d
            target[i] = pos
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
