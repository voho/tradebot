"""R-190: fixed execution variations of the three promoted Kelly parents.

Parent signal defaults are unchanged. Nine candidates revisit those prepared
targets at four-hour UTC closes when actual equity-notional exposure drifts
outside a 5%, 10% or 20% band. The tenth averages the three parent targets and
uses the central 10% band. Orders retain ordinary next-open execution and the
market broker's own deadband, leverage cap, minimum size, fees and funding.

These experiments are deliberately unregistered. Six daily decision slots
are an upper bound, not a promise or a requirement to manufacture trades.
"""

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import KellyRegime
from tradebot.strategies.kelly_regime_v3 import KellyRegimeV3
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context, Strategy


PARENTS = {
    "kelly_regime_v4": KellyRegimeV4,
    "kelly_regime_v3": KellyRegimeV3,
    "kelly_regime": KellyRegime,
}
CONFIGS = (
    ("r190_v4_b05", "kelly_regime_v4", 0.05),
    ("r190_v4_b10", "kelly_regime_v4", 0.10),
    ("r190_v4_b20", "kelly_regime_v4", 0.20),
    ("r190_v3_b05", "kelly_regime_v3", 0.05),
    ("r190_v3_b10", "kelly_regime_v3", 0.10),
    ("r190_v3_b20", "kelly_regime_v3", 0.20),
    ("r190_base_b05", "kelly_regime", 0.05),
    ("r190_base_b10", "kelly_regime", 0.10),
    ("r190_base_b20", "kelly_regime", 0.20),
    ("r190_blend_b10", "blend", 0.10),
)


class RebalanceVariation(Strategy):
    """Revisit promoted Kelly targets at UTC slots using actual account drift."""

    def __init__(self, name: str, parent: str, deadband: float) -> None:
        self.name, self.parent, self.deadband = name, parent, deadband
        self.parents = tuple(PARENTS.values()) if parent == "blend" else (PARENTS[parent],)
        self.warmup = max(cls.warmup for cls in self.parents)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        targets = [cls().prepare(df.copy())["target"].to_numpy() for cls in self.parents]
        df["target"] = np.clip(np.mean(targets, axis=0), 0.0, 2.0)
        ts = pd.DatetimeIndex(df.index)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        df["r190_decision"] = (
            (ts.hour % 4 == 0) & (ts.minute == 0) & (ts.second == 0)
            & (ts.microsecond == 0) & (ts.nanosecond == 0)
        )
        return df

    def on_bar(self, ctx: Context) -> None:
        if not bool(ctx.bar["r190_decision"]) or ctx.equity <= 0:
            return
        target = float(ctx.bar["target"])
        held = ctx.position * ctx.close / ctx.equity
        if abs(target - held) > self.deadband or (target == 0 and ctx.position != 0):
            ctx.order_notional(target)


def make_strategy(name: str) -> RebalanceVariation:
    return RebalanceVariation(*next(config for config in CONFIGS if config[0] == name))
