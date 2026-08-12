"""Microstructure game combo: follow informed flow, defer to liquidation events, respect beliefs."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class FlowRegime(Strategy):
    """Combine the two sides of the microstructure game: follow flow, but fade liquidation overshoots.

    Game-theoretic grounding: in Kyle (1985)-type equilibria, following
    detected informed flow is correct EXCEPT at the moment of a forced-
    liquidation overshoot, when the equilibrium continuation is reversion
    (Brunnermeier & Pedersen 2005, J. Finance) - the two trades are
    complementary sides of one game. This combo therefore takes the
    consensus of the two flow followers (camouflage_flow's BVC imbalance
    and stealth_trend's depth-gated momentum), lets overshoot_fade's rare
    liquidation-event trade OVERRIDE them (its event flag is the regime
    where following flow is exactly wrong), and vetoes consensus entries
    that fight the harsanyi_crowd Bayesian regime belief - a Harsanyi
    (1967-68) type-check that the crowd regime agrees.

    Mechanism: per bar, target = overshoot_fade's target when it is
    active (rare); otherwise the mean of the two flow followers' targets,
    zeroed when below a floor or when the belief strategy actively
    disagrees in sign.
    """

    name = "flow_regime"
    warmup = 2300

    def __init__(self, consensus_floor: float = 0.15) -> None:
        self.consensus_floor = consensus_floor

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        from tradebot.strategies.camouflage_flow import CamouflageFlow
        from tradebot.strategies.harsanyi_crowd import HarsanyiCrowd
        from tradebot.strategies.overshoot_fade import OvershootFade
        from tradebot.strategies.stealth_trend import StealthTrend

        base = df[["open", "high", "low", "close", "volume"]]
        t_cam = CamouflageFlow().prepare(base.copy())["target"].to_numpy()
        t_stealth = StealthTrend().prepare(base.copy())["target"].to_numpy()
        t_fade = OvershootFade().prepare(base.copy())["target"].to_numpy()
        t_belief = HarsanyiCrowd().prepare(base.copy())["target"].to_numpy()

        consensus = 0.5 * (np.nan_to_num(t_cam) + np.nan_to_num(t_stealth))
        consensus = np.where(np.abs(consensus) < self.consensus_floor, 0.0, consensus)
        # belief veto: an actively opposing regime belief blocks the flow trade
        opposing = (t_belief != 0.0) & (np.sign(t_belief) != np.sign(consensus))
        consensus = np.where(opposing, 0.0, consensus)
        # liquidation events override: that regime is where following flow is wrong
        target = np.where(np.nan_to_num(t_fade) != 0.0, np.nan_to_num(t_fade), consensus)

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
