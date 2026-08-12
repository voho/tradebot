"""A game of games: Hedge meta-allocation across the game-theory strategies themselves."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class GameCouncil(Strategy):
    """Combination of games: no-regret Hedge allocation over the game strategies' own signals.

    Game-theoretic grounding: each member strategy is the equilibrium play
    of a different game against the market - Kyle-flow back-running
    (camouflage_flow), depth-gated momentum (stealth_trend), predatory
    overshoot fading (overshoot_fade), fictitious play over history states
    (game_switch), a repeated-game trend truce (tft_trend), attrition-
    priced reversion (attrition_reversion) and Bayesian-type beliefs
    (harsanyi_crowd). Which game the market is currently playing is
    unknowable in advance, so the council treats the members as experts in
    a Hedge/multiplicative-weights meta-game (Freund & Schapire 1997,
    JCSS; Arora, Hazan & Kale 2012): weights grow with each member's
    realized fee-charged vol-normalized PnL, guaranteeing performance
    within O(sqrt(T ln N)) of the best member in hindsight - without
    knowing which one that is. A flat expert anchors the maximin floor.

    Mechanism: run every member's prepare() on a private copy of the
    candles, harvest its target column, then blend with discounted Hedge
    plus fixed-share tracking and a hysteresis band.
    """

    name = "game_council"
    warmup = 3100

    # Hysteresis and quantization retuned after the first paper test:
    # a continuously drifting blended target with a 0.05 band produced
    # 15k-43k trades and died to fees. Quantizing the played position to
    # a coarse grid keeps it piecewise-constant like its members.
    def __init__(self, eta: float = 0.08, fixed_share: float = 1e-4,
                 hysteresis: float = 0.20, quantum: float = 0.25,
                 fee_rate: float = 0.0005) -> None:
        self.eta = eta
        self.fixed_share = fixed_share
        self.hysteresis = hysteresis
        self.quantum = quantum
        self.fee_rate = fee_rate

    @staticmethod
    def _members() -> list[Strategy]:
        from tradebot.strategies.attrition_reversion import AttritionReversion
        from tradebot.strategies.camouflage_flow import CamouflageFlow
        from tradebot.strategies.game_switch import GameSwitch
        from tradebot.strategies.harsanyi_crowd import HarsanyiCrowd
        from tradebot.strategies.overshoot_fade import OvershootFade
        from tradebot.strategies.stealth_trend import StealthTrend
        from tradebot.strategies.tft_trend import TitForTatTrend

        return [CamouflageFlow(), StealthTrend(), OvershootFade(), GameSwitch(),
                TitForTatTrend(), AttritionReversion(), HarsanyiCrowd()]

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[["open", "high", "low", "close", "volume"]]
        signals = []
        for member in self._members():
            prepared = member.prepare(base.copy())
            signals.append(prepared["target"].to_numpy(dtype=np.float64))
        signals.append(np.zeros(len(df)))  # flat expert: the maximin floor
        a = np.nan_to_num(np.column_stack(signals), nan=0.0)  # (n, N)

        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        r_a = r.to_numpy()
        sig_a = sig1.shift(1).to_numpy()

        n, num = a.shape
        target = np.zeros(n)
        logw = np.zeros(num)
        pos = 0.0
        for i in range(2, n):
            s = sig_a[i]
            if not np.isfinite(s) or s <= 0:
                target[i] = pos
                continue
            z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
            fee_n = min(self.fee_rate / (3.0 * s), 0.25)
            g = np.clip(a[i - 1] * z_t - fee_n * np.abs(a[i - 1] - a[i - 2]), -1.0, 1.0)
            logw += self.eta * g
            logw -= logw.max()
            p = np.exp(logw)
            p /= p.sum()
            p = (1.0 - self.fixed_share) * p + self.fixed_share / num
            logw = np.log(p)
            x = float(p @ a[i])
            x = round(x / self.quantum) * self.quantum  # piecewise-constant play
            if abs(x - pos) > self.hysteresis or (x > 0) != (pos > 0) or (x < 0) != (pos < 0):
                pos = x
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
