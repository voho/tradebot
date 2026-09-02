"""Level-k / cognitive-hierarchy play: best-respond to the level the crowd is currently at."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY
from tradebot.strategy import Context, Strategy


@register
class LevelK(Strategy):
    """Beauty-contest trading: play one reasoning level above the level whose signal is currently paying.

    Game-theoretic grounding: Nagel (1995), "Unraveling in Guessing Games",
    American Economic Review 85(5):1313-1326, and Camerer, Ho & Chong
    (2004), "A Cognitive Hierarchy Model of Games", Quarterly Journal of
    Economics 119(3):861-898. Players reason a bounded number of steps:
    level-0 acts naively, level-1 best-responds to level-0, level-2 to the
    mix below it. Keynes' beauty contest is the market version - profit
    goes to whoever is one step ahead of the crowd, not to whoever is
    "right". Hommes (2011, JEDC) surveys the learning-to-forecast evidence
    that markets cycle through these levels.

    Mechanism. Three levels, each a position in {-1, 0, +1}: level-0 chases
    the slow trend (``slow_bars`` return, in local sigmas); level-1
    front-runs the chasers with the fast trend (``fast_bars``); level-2
    fades level-1's crowding (minus the fast trend). Each level's
    fee-charged PnL is tracked with an exponential memory. The level with
    the best recent PnL is where the crowd is heading, so with
    ``anticipate=True`` the strategy plays its best response - one level
    up, cyclically (level-2's fading is best answered by the slow trend);
    ``anticipate=False`` is the control that simply plays the currently
    best level (follow-the-leader) - and it is the frozen default, because
    on inner-validation the control beat the anticipating rule on both
    markets (R-188). Fast levels flip a few times a day.
    """

    name = "level_k"
    warmup = 3 * BARS_PER_DAY

    def __init__(self, fast_bars: int = 12, slow_bars: int = BARS_PER_DAY,
                 anticipate: bool = False, memory_bars: int = 3 * BARS_PER_DAY,
                 entry_z: float = 1.0, fee_rate: float = 0.001,
                 exposure: float = 1.0) -> None:
        self.fast_bars = fast_bars
        self.slow_bars = slow_bars
        self.anticipate = anticipate
        self.memory_bars = memory_bars
        self.entry_z = entry_z
        self.fee_rate = fee_rate
        self.exposure = exposure

    def _level_signals(self, df: pd.DataFrame) -> np.ndarray:
        logc = np.log(df["close"])
        r = logc.diff()
        sigma = r.ewm(span=BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).std().shift(1)
        z_fast = (logc - logc.shift(self.fast_bars)) / (sigma * np.sqrt(self.fast_bars))
        z_slow = (logc - logc.shift(self.slow_bars)) / (sigma * np.sqrt(self.slow_bars))
        l0 = np.sign(z_slow).where(z_slow.abs() > 0.5 * self.entry_z, 0.0)
        l1 = np.sign(z_fast).where(z_fast.abs() > self.entry_z, 0.0)
        levels = np.column_stack([l0.to_numpy(), l1.to_numpy(), -l1.to_numpy()])
        return np.nan_to_num(levels, nan=0.0)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        levels = self._level_signals(df)  # (n, 3)
        r = np.log(df["close"]).diff().fillna(0.0).to_numpy()
        held = np.vstack([np.zeros((1, 3)), levels[:-1]])  # position during bar t
        turnover = np.abs(np.vstack([np.zeros((1, 3)), np.diff(held, axis=0)]))
        pnl = held * r[:, None] - self.fee_rate * turnover
        score = pd.DataFrame(pnl).ewm(span=self.memory_bars, min_periods=BARS_PER_DAY).mean()
        best = score.to_numpy().argmax(axis=1)
        play = (best + 1) % 3 if self.anticipate else best
        target = levels[np.arange(len(df)), play] * self.exposure
        target[~np.isfinite(score.to_numpy()).all(axis=1)] = 0.0
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
