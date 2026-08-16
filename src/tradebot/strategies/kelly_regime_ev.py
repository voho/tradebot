"""kelly_regime_v4 with a no-trade band derived from expected profit."""

import numpy as np

from tradebot.registry import register
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context


@register
class KellyRegimeEV(KellyRegimeV4):
    """Rebalance only when the expected gain exceeds the fee it costs.

    Every other variant uses a *fixed* 10% deadband — a number, not a
    decision. This one asks the actual question: is moving from the
    current exposure to the desired one worth what the venue charges?

    **The derivation.** For a growth-optimal (Kelly) sizer the expected
    log-growth at exposure ``f`` is ``g(f) = f·mu - f²·sigma²/2``, a
    parabola peaking at the desired exposure ``f*``. So the growth given
    up by sitting at ``f`` instead of ``f*`` is exactly

        g(f*) - g(f) = (sigma²/2)·(f - f*)²

    per unit time. Holding the wrong exposure for a horizon ``H`` costs
    ``H·(sigma²/2)·(Δf)²``; correcting it costs ``fee·|Δf|`` in taker
    fees on the traded notional. Trading is worth it only when the first
    exceeds the second, which reduces to a threshold on the size of the
    move:

        |Δf| > 2·fee / (H·sigma²)

    That is the whole strategy. The band is not tuned; it falls out of
    the fee, the volatility and the expected time to the next rebalance —
    and it reproduces the classic transaction-cost result (Constantinides
    1986; Davis & Norman 1990) that the no-trade region widens with cost
    and narrows with volatility and horizon.

    **What it says about a 0.40% venue.** At a 0.10% fee, 55% vol and a
    weekly horizon the band is about 0.34 — already 3x the hand-set 10%.
    At 0.40% it exceeds 1.0, i.e. *no* rebalance is ever worth its cost
    and the growth-optimal policy collapses to buy-and-hold. That is a
    derivation of the result the fee study found empirically, and it is
    the honest reason turnover reduction never rescued the strategy: the
    optimum was not a smaller trade, it was no trade.

    ``horizon_days`` is the expected time until the next rebalance, and
    the one judgement call left. It is set from an **observable** — the
    measured spacing of ``kelly_regime_v4``'s own fills, 1,056 over 9.6
    years, about one every 3.3 days — rather than fitted to returns.

    **What is robust here and what is not.** Sweeping the horizon from 1
    to 30 days at a 0.40% fee, max drawdown is a genuine region: 30-38%
    for every horizon from 1 to 5 days, against 45-48% beyond it, and
    against buy-and-hold's 84%. The *return* over the same sweep swings
    3x between adjacent values ($35K at 1 day, $69K at 3, $24K at 7),
    which is noise of exactly the kind ``scripts/fee_study.py`` showed is
    not tradable. Out-of-sample the whole 1-5 day region lands on the
    same number and still trails holding. So read this strategy as a
    turnover and drawdown result, not a return result — and note that
    the measured 3.3-day default happens to sit near the in-sample best,
    a coincidence that cannot be ruled out as luck.
    """

    name = "kelly_regime_ev"

    def __init__(self, horizon_days: float = 3.3, min_band: float = 0.02,
                 max_band: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.horizon_days = horizon_days
        self.min_band = min_band
        self.max_band = max_band

    def _band(self, fee: float, vol: float) -> float:
        """Threshold on |Δexposure| below which trading destroys value."""
        horizon_years = self.horizon_days / 365.25
        variance = max(vol, 1e-6) ** 2
        band = 2.0 * fee / (horizon_years * variance)
        return float(np.clip(band, self.min_band, self.max_band))

    def on_bar(self, ctx: Context) -> None:
        desired = float(ctx.bar["target"])
        vol = float(ctx.bar["_ev_vol"])
        if not np.isfinite(vol) or vol <= 0:
            return

        equity = ctx.equity
        if equity <= 0:
            return
        current = ctx.position * ctx.close / equity

        band = self._band(ctx.market.fee_rate, vol)
        # Always allow a full exit: standing flat is the one move whose
        # benefit is not captured by the quadratic (it removes the whole
        # position's risk, and the regime gate asked for it).
        if desired == 0.0 and abs(current) > 1e-9:
            ctx.order_notional(0.0)
            return
        if abs(desired - current) > band:
            ctx.order_notional(desired)

    def prepare(self, df):
        df = super().prepare(df)
        returns = np.log(df["close"]).diff()
        df["_ev_vol"] = (returns.ewm(span=self.vol_span, min_periods=BARS_PER_DAY)
                         .std() * np.sqrt(BARS_PER_YEAR)).shift(1)
        return df


@register
class KellyRegimeEVFast(KellyRegimeEV):
    """The same rule at a 1-day horizon, i.e. a wider band and less trading.

    Registered so the sensitivity of the one free parameter is visible in
    the comparison table rather than argued about in a docstring. Its
    drawdown sits in the same 30-38% region as the default; its final
    balance does not, which is the point.
    """

    name = "kelly_regime_ev_fast"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("horizon_days", 1.0)
        super().__init__(**kwargs)


