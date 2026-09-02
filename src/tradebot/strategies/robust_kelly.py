"""Distributionally robust Kelly: size on the worst-case drift over a confidence set of models."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.session import BARS_PER_DAY
from tradebot.strategy import Context, Strategy


@register
class RobustKelly(Strategy):
    """Fractional Kelly on the lower confidence bound of drift, minimized over several estimation windows.

    Game-theoretic grounding: Kelly sizing is the equilibrium of the
    two-player investment game (Bell & Cover 1980), but it presumes the
    drift is known. Against *nature* choosing the distribution inside an
    ambiguity set, the growth-optimal play is the Kelly fraction for the
    worst distribution in the set: Rujeerapaiboon, Kuhn & Wiesemann (2016),
    "Robust Growth-Optimal Portfolios", Management Science 62(7):2090-2109;
    Sun & Boyd (2018), "Distributional Robust Kelly Gambling",
    arXiv:1812.10371. Baker & McHale (2013), "Optimal Betting Under
    Parameter Uncertainty: Improving the Kelly Criterion", Decision
    Analysis 10(3):189-199, derive the same shrinkage from estimation
    error: bet on the edge you can defend, not the edge you measured.

    Mechanism. For each window W in ``windows_days`` the drift ``mu_W`` and
    daily volatility ``sigma_W`` are estimated from the trailing W days.
    The worst-case drift for a long is ``min_W (mu_W - kappa sigma_W /
    sqrt(W))``; when it is positive the exposure is ``fraction`` of the
    Kelly ratio, ``mu_wc / sigma^2``, capped at ``max_exposure`` of equity.
    A short (futures only) is taken symmetrically when the best-case drift
    is negative. Otherwise the strategy is flat: it refuses to bet on a
    drift it cannot distinguish from zero at ``kappa`` standard errors -
    which, with 10-90 day windows on an asset whose daily volatility is
    about twenty times its daily drift, is most of the time (``kappa=1``
    sat flat on 90% of bars in development, so the frozen radius is small
    by construction). Re-targets inside a ``deadband`` are suppressed. Not a variant of
    ``kelly_regime`` (a latched price-vs-anchor vote times a volatility
    target) nor of R-123 (which shrank v4's own vote): the whole decision
    here is the confidence bound.
    """

    name = "robust_kelly"
    warmup = 90 * BARS_PER_DAY + 10

    def __init__(self, windows_days: tuple[int, ...] = (10, 30, 90), kappa: float = 0.5,
                 fraction: float = 0.5, max_exposure: float = 2.0,
                 deadband: float = 0.10) -> None:
        self.windows_days = windows_days
        self.kappa = kappa
        self.fraction = fraction
        self.max_exposure = max_exposure
        self.deadband = deadband

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        logc = np.log(df["close"])
        r = logc.diff()
        lcb, ucb, variances = [], [], []
        for days in self.windows_days:
            w = days * BARS_PER_DAY
            mu = (logc - logc.shift(w)) / days  # mean daily log return
            sigma = r.rolling(w, min_periods=w // 2).std() * np.sqrt(BARS_PER_DAY)
            se = sigma / np.sqrt(days)
            lcb.append(mu - self.kappa * se)
            ucb.append(mu + self.kappa * se)
            variances.append(sigma ** 2)
        worst_long = pd.concat(lcb, axis=1).min(axis=1).to_numpy()
        best_short = pd.concat(ucb, axis=1).max(axis=1).to_numpy()
        var = variances[0].to_numpy()  # the shortest window's variance sizes the bet

        with np.errstate(divide="ignore", invalid="ignore"):
            long_f = np.where(worst_long > 0, self.fraction * worst_long / var, 0.0)
            short_f = np.where(best_short < 0, self.fraction * best_short / var, 0.0)
        desired = np.clip(np.nan_to_num(long_f + short_f), -self.max_exposure, self.max_exposure)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            d = desired[i]
            if abs(d - pos) > self.deadband or (d == 0.0 and pos != 0.0):
                pos = d
            target[i] = pos
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
