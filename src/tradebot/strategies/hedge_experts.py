"""Discounted multiplicative-weights (Hedge) blend over fee-charged technical experts."""

import numpy as np
import pandas as pd

from tradebot.indicators import macd, rsi
from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class HedgeExperts(Strategy):
    """No-regret Hedge blend of technical experts, each charged its own turnover.

    Game-theoretic grounding: Hedge / multiplicative weights (Freund &
    Schapire 1997, JCSS) earns within 2*sqrt(T ln N) of the best expert in
    hindsight on fully adversarial sequences (survey: Arora, Hazan & Kale
    2012, Theory of Computing). Fixed-share mixing (Herbster & Warmuth
    1998, Machine Learning) tracks a drifting best expert. The expert set
    includes always-flat, so the blend inherits the zero-sum maximin
    floor - asymptotically it cannot do much worse than staying out.

    Mechanism: ten causal experts (vol-scaled momentum at 1h/6h/1d/1w,
    MACD, an RSI ramp, 1-bar reversion, a Donchian breakout, flat, and
    buy-and-hold) each emit a position in [-1, 1]. Every bar each expert's
    vol-normalized PnL minus the fees its own position changes imply
    updates log-weights; the played position is the weight-blended expert
    mix behind a hysteresis band so re-targets stay sparse.
    """

    name = "hedge_experts"
    warmup = 2500

    def __init__(self, eta: float = 0.05, fixed_share: float = 1e-4,
                 hysteresis: float = 0.05, fee_rate: float = 0.0005) -> None:
        self.eta = eta
        self.fixed_share = fixed_share
        self.hysteresis = hysteresis
        self.fee_rate = fee_rate

    def _experts(self, df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
        close = df["close"]
        logc = np.log(close)
        cols = []
        for h in (12, 72, 288, 2016):
            z = (logc - logc.shift(h)) / (sig1.shift(1) * np.sqrt(h))
            cols.append(np.tanh(z))
        m = macd(close)
        cols.append(np.tanh(m["macd_hist"] / (sig1.shift(1) * close * np.sqrt(26.0))))
        rs = rsi(close)
        ramp = ((50.0 - rs) / 40.0).clip(-0.5, 0.5)
        cols.append(pd.Series(np.where(rs < 30, 0.75, np.where(rs > 70, -0.75, ramp)),
                              index=df.index))
        cols.append(-(r / (3.0 * sig1.shift(1))).clip(-0.5, 0.5))

        hi = close.rolling(288).max().shift(1).to_numpy()
        lo = close.rolling(288).min().shift(1).to_numpy()
        c = close.to_numpy()
        donch = np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            if np.isfinite(hi[i]):
                if c[i] > hi[i]:
                    state = 1.0
                elif c[i] < lo[i]:
                    state = -1.0
                else:
                    state *= 0.99
            donch[i] = state
        cols.append(pd.Series(donch, index=df.index))

        cols.append(pd.Series(0.0, index=df.index))  # always flat: the maximin floor
        cols.append(pd.Series(1.0, index=df.index))  # buy and hold
        a = np.column_stack([np.asarray(col, dtype=np.float64) for col in cols])
        return np.nan_to_num(a, nan=0.0)  # NaN warmup rows act as flat experts

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        a = self._experts(df, r, sig1)  # (n, N)
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
