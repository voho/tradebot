"""kelly_regime_v4 with a continuous funding-z-score exposure discount.

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
Perpetual futures funding is not incidental noise on top of price — it is
the mechanism that enforces perp-spot convergence, a continuously realized
cost-of-carry analogous to the classical futures cost-of-carry (Working
1949, "The Theory of Price of Storage"), formalized for perpetuals by
Ackerer, Hugonnier & Jermann (2026, Mathematical Finance, "Perpetual
Futures Pricing"), who characterize the funding leg as an economically
meaningful carry object rather than a friction to be modeled away. Two
ledger findings motivate treating it as a *continuous* signal rather than a
discrete trigger:

- R-14: funding on this data is adversely timed against ``kelly_regime_v4``
  — richest exactly while the strategy holds — costing a leveraged long
  ~15-20%/yr.
- R-16: high funding predicts negative forward returns, but the effect is
  "weaker than the tables suggest" (VALIDATION.md) — soft, noisy and
  non-monotone across the middle quintiles. A hard percentile trigger would
  treat that noise as a clean switch; a continuous scale does not have to.

Mechanism
---------
Take ``kelly_regime_v4``'s ``target`` column unchanged (the vote and the
volatility sizer are the parent's job, untouched here). Compute a rolling
z-score of EWM-smoothed funding: how anomalous is *current* richness
relative to its own trailing 180-day distribution, not a fixed percentile.
Only the positive half of the z-score — funding richer than usual — is
allowed to shrink exposure; cheap or negative funding never boosts exposure
above the base v4 target. That asymmetry is deliberate: R-08 is the
project's cautionary tale that a strategy which de-levers *more promptly*
on a *better* forecast can make *less* money, because it de-levers out of
BTC's best (high-vol, high-forward-Sharpe) states (R-10). Boosting exposure
on cheap funding would risk the mirror-image trap — levering up on ordinary
variation in a noisy, non-monotone signal (R-16's own honest assessment).
Discounting only the high tail, clipped at ``z_cap`` standard deviations so
one extreme settlement cannot wipe exposure to zero, is the conservative
half of the idea.

Not a duplicate of B-05 (``NEXT`` in the backlog, "funding as a gate ...
stand flat in the top decile"): B-05 is a discrete percentile trigger,
0/1 in effect. This is the continuous alternative the backlog item's own
text gestures at without specifying — a smooth discount driven by a
rolling z-score, never binary, never a full stand-down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context


class FundingGateNovel(KellyRegimeV4):
    """v4's target, continuously discounted by a one-sided funding z-score.

    ``drag = clip(z, 0, z_cap) / z_cap * k`` scales exposure down toward
    ``(1 - k)`` of the base target as funding gets anomalously rich versus
    its own trailing 180-day distribution; ordinary or cheap funding leaves
    the target untouched. Not registered — see module docstring.
    """

    name = "funding_gate_novel"

    def __init__(self, k: float = 0.5, z_cap: float = 3.0,
                 funding_halflife_days: float = 3.0,
                 funding_zscore_window_days: int = 180,
                 funding: pd.Series | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.z_cap = z_cap
        self.funding_halflife_days = funding_halflife_days
        self.funding_zscore_window_days = funding_zscore_window_days
        self.funding = funding

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        df["target_base"] = df["target"]

        funding = self.funding
        if funding is None:
            df["target_funded"] = df["target_base"]
            return df

        # Causal alignment: only the most recent *known* settlement at or
        # before this bar is visible, and only within the real data's own
        # coverage window — no indefinite forward-fill past 2023-12-31 into
        # a period the funding file says nothing about.
        funding = funding.sort_index()
        known = funding.reindex(df.index, method="ffill")
        known = known.mask((df.index < funding.index[0]) | (df.index > funding.index[-1]))
        df["funding_known"] = known

        funding_ewm = df["funding_known"].ewm(
            halflife=f"{self.funding_halflife_days}D", times=df.index).mean()
        df["funding_ewm"] = funding_ewm

        roll_mean = funding_ewm.rolling(f"{self.funding_zscore_window_days}D").mean()
        roll_std = funding_ewm.rolling(f"{self.funding_zscore_window_days}D").std()
        z = (funding_ewm - roll_mean) / roll_std.replace(0, np.nan)
        df["z"] = z

        drag = (z.clip(lower=0, upper=self.z_cap) / self.z_cap) * self.k
        drag = drag.fillna(0.0)
        discount = (1.0 - drag).clip(lower=0.0)

        df["target_funded"] = df["target_base"] * discount
        return df

    def on_bar(self, ctx: Context) -> None:
        col = "target_base" if ctx.market.name == "spot" else "target_funded"
        t = float(ctx.bar[col])
        prev = float(ctx.prev[col]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
