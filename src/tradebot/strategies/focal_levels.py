"""Round-number focal points (Schelling 1960; Osler 2003): trade the crowd's coordination on round prices."""

import math

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY
from tradebot.strategy import Context, Strategy


def round_spacing(price: float, frac: float) -> float:
    """The 1-2-5 round-number step nearest to ``frac`` of ``price``.

    $45,000 with frac 0.02 -> $1,000 steps; $8,000 -> $200; $100,000 -> $2,000.
    """
    if not np.isfinite(price) or price <= 0:
        return math.nan
    want = math.log10(price * frac)
    best, err = 1.0, math.inf
    for k in range(int(math.floor(want)) - 1, int(math.floor(want)) + 2):
        for m in (1.0, 2.0, 5.0):
            cand = m * 10.0 ** k
            e = abs(math.log10(cand) - want)
            if e < err:
                best, err = cand, e
    return best


@register
class FocalLevels(Strategy):
    """Round-number levels as focal points: ride the break through one (``breakout``) or fade the test of one (``bounce``).

    Grounding: Schelling (1960, *The Strategy of Conflict*) - when many
    players must coordinate without communicating, they converge on focal
    points, and round numbers are the canonical ones. Osler (2003),
    "Currency Orders and Exchange Rate Dynamics: An Explanation for the
    Predictive Success of Technical Analysis", Journal of Finance
    58(5):1791-1819, shows from a dealer's order book that take-profit
    orders cluster *at* round numbers (so rates tend to reverse there) and
    stop-loss orders cluster just *beyond* them (so a break tends to
    accelerate). Both predictions are testable from prices alone once the
    round levels are defined.

    Mechanism. The level grid is the 1-2-5 step nearest ``spacing_frac`` of
    a slow (30-day median) reference price, so $2,000 steps around $45K.
    ``breakout``: when consecutive closes straddle a level and the close is
    beyond it by more than ``buffer`` local sigmas, take the break's
    direction for ``hold_bars`` bars (the stop-loss cascade). ``bounce``:
    when the bar's extreme touches a level and the close is back across it
    by ``buffer`` sigmas without a straddle, take the rejection's direction
    for ``hold_bars`` bars (the take-profit wall). Shorts on futures only.
    A signal every day or two at the frozen $2,000-scale spacing; several a
    day at $1,000.
    """

    name = "focal_levels"
    warmup = 31 * BARS_PER_DAY

    def __init__(self, mode: str = "breakout", spacing_frac: float = 0.05,
                 buffer: float = 0.5, hold_bars: int = 24, exposure: float = 1.0) -> None:
        if mode not in ("breakout", "bounce"):
            raise ValueError("mode must be 'breakout' or 'bounce'")
        self.mode = mode
        self.spacing_frac = spacing_frac
        self.buffer = buffer  # in local 5-minute sigmas
        self.hold_bars = hold_bars
        self.exposure = exposure

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()
        sigma = (r.ewm(span=BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).std().shift(1)
                 .to_numpy())
        ref = close.rolling(30 * BARS_PER_DAY, min_periods=BARS_PER_DAY).median().shift(1)
        ref_a = ref.to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)

        n = len(df)
        target = np.zeros(n)
        pos, until = 0.0, -1
        step = math.nan
        for i in range(1, n):
            if i % BARS_PER_DAY == 0 or not np.isfinite(step):
                step = round_spacing(ref_a[i], self.spacing_frac)  # re-fit once a day
            s = sigma[i]
            if np.isfinite(step) and np.isfinite(s) and s > 0:
                buf = self.buffer * s * c[i]
                k_prev, k_now = math.floor(c[i - 1] / step), math.floor(c[i] / step)
                if self.mode == "breakout":
                    if k_now > k_prev and c[i] - k_now * step > buf:
                        pos, until = self.exposure, i + self.hold_bars
                    elif k_now < k_prev and (k_now + 1) * step - c[i] > buf:
                        pos, until = -self.exposure, i + self.hold_bars
                else:  # bounce
                    below = k_now * step  # nearest level at or below the close
                    above = below + step
                    if k_now == k_prev and lo[i] <= below and c[i] - below > buf:
                        pos, until = self.exposure, i + self.hold_bars
                    elif k_now == k_prev and h[i] >= above and above - c[i] > buf:
                        pos, until = -self.exposure, i + self.hold_bars
            if i >= until:
                pos = 0.0
            target[i] = pos
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
