"""Bayesian belief over the market's hidden type with a mean-field crowding haircut."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


@register
class HarsanyiCrowd(Strategy):
    """Trade the belief margin over hidden market types, sized down when the trend is crowded.

    Game-theoretic grounding: Harsanyi (1967-68, Management Science) -
    games of incomplete information are played against an opponent whose
    TYPE is unknown; the rational player keeps a posterior over types
    (here: up-trend, down-trend, chop) updated by Bayes' rule from
    observed play (each bar's ATR-normalized move), with a sticky
    transition prior since regimes persist. Position follows the belief
    margin behind wide hysteresis bands, so fees are paid only when
    beliefs move decisively. Cardaliaguet & Lehalle (2018, Math. Fin.
    Econ.) model trade crowding as a mean-field game: trend drift is
    crowd flow, and an aged trend whose volume efficiency decays (more
    volume per unit of progress = the mean field saturating) carries a
    strategic crowding cost - so exposure is hair-cut in old, saturated
    trends instead of chasing them.

    Mechanism: Gaussian likelihoods for the three types on winsorized
    ATR-moves; posterior mixed with a 2% switching prior; crowding =
    sigmoid of trend age x a volume-efficiency-decay flag; target =
    hysteresis(belief margin) x (1 - 0.7 x crowding), with a deadband.
    """

    name = "harsanyi_crowd"
    warmup = 1300

    # Bands widened after the first paper test: beliefs oscillating around
    # b_in paid entry fees for micro-trends; entries now need conviction.
    def __init__(self, mu: float = 0.15, stick: float = 0.985, b_in: float = 0.70,
                 b_out: float = 0.40, age_scale: float = 150.0, lam_crowd: float = 0.7,
                 deadband: float = 0.25) -> None:
        self.mu = mu
        self.stick = stick
        self.b_in, self.b_out = b_in, b_out
        self.age_scale = age_scale
        self.lam_crowd = lam_crowd
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 48, min_periods=48).mean()

        x = (close.diff() / atr.shift(1)).clip(-3.0, 3.0).to_numpy()
        progress = ((close - close.shift(24)).abs() / atr).to_numpy()
        vol_ratio = (df["volume"].rolling(24).sum()
                     / df["volume"].rolling(288).sum()).to_numpy()

        n = len(df)
        target = np.zeros(n)
        b = np.full(3, 1.0 / 3.0)  # P(up), P(down), P(chop)
        eff_ema = np.nan
        decl = 0
        trend_age = 0
        prev_margin_sign = 0
        pos = 0.0
        alpha_eff = 2.0 / 49.0
        for i in range(n):
            if not np.isfinite(x[i]):
                target[i] = pos
                continue
            lik = np.array([
                np.exp(-0.5 * (x[i] - self.mu) ** 2),
                np.exp(-0.5 * (x[i] + self.mu) ** 2),
                np.exp(-0.5 * (x[i] / 0.8) ** 2) / 0.8,
            ])
            b = b * lik
            b = self.stick * b + (1.0 - self.stick) / 3.0
            b /= b.sum()
            margin = b[0] - b[1]

            # mean-field crowding proxy: aged trend + decaying volume efficiency
            if np.isfinite(progress[i]) and np.isfinite(vol_ratio[i]):
                eff = progress[i] / max(vol_ratio[i], 1e-9)
                if np.isfinite(eff_ema):
                    decl = decl + 1 if eff < eff_ema else 0
                    eff_ema += alpha_eff * (eff - eff_ema)
                else:
                    eff_ema = eff
            sign = 1 if margin > 0 else (-1 if margin < 0 else 0)
            trend_age = trend_age + 1 if sign == prev_margin_sign and sign != 0 else 0
            prev_margin_sign = sign
            crowd = (1.0 / (1.0 + np.exp(-(trend_age / self.age_scale - 1.0)))) \
                * (1.0 if decl >= 12 else 0.3)

            band = self.b_out if pos != 0.0 else self.b_in
            if abs(margin) < band:
                raw = 0.0
            else:
                raw = np.sign(margin) * min(1.0, (abs(margin) - self.b_out)
                                            / (1.0 - self.b_out))
            t = raw * (1.0 - self.lam_crowd * crowd)
            if abs(t - pos) >= self.deadband or (t == 0.0 and pos != 0.0):
                pos = t
            target[i] = pos

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
