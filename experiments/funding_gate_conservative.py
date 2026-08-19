"""Funding-gated `kelly_regime_v4` (backlog B-05, conservative variant).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea
--------
R-14 measured perpetual funding as a first-class cost: a leveraged BTC
long pays ~15-20%/yr, and the cost is *adversely timed* — funding is
richest exactly while `kelly_regime_v4` is holding (mean rate while
holding is worse than the unconditional mean). R-16 then asked whether
funding, beyond being a cost, is also a *signal*: sorted into quintiles,
richly-funded periods forecast negative forward returns — but only when
price momentum is not also confirming. Funding-high + past-low/mid gives
-1.5% to -1.7% forward 7d return; funding-high + past-high still gives
+1.2%. A blanket "flatten on rich funding" rule would cut exactly the
genuine bull continuations that are this strategy's entire edge (the
mechanism R-08 already burned a session establishing: a strategy that
de-levers more promptly on a better signal made LESS money, because it
de-levered out of BTC's best states).

Mechanism, exactly
-------------------
This is deliberately the smallest possible deviation from the incumbent.
`kelly_regime_v4`'s vote/vol-targeting logic is untouched. On top of it,
a single bounded multiplicative haircut is applied ONLY on the small,
already-measured slice of state-space R-16 found genuinely negative:
funding richer than its trailing 365-day 90th percentile AND trailing
7-day momentum non-positive. Everywhere else — including the momentum-
confirmed bull continuations R-16 explicitly warned about protecting —
the position is exactly `kelly_regime_v4`'s.

The gate is a pure function of ``(df, funding)``: this class does no file
I/O. The caller (``run_funding_gate_conservative.py``) loads
``data/btcusdt_perp_funding_8h.csv.gz`` via ``tradebot.data.load_funding``
and passes the raw series in, matching the project's
``run_backtest(..., funding=...)`` convention and keeping this class
testable without touching the filesystem. If ``funding`` is ``None`` the
whole gate is a no-op.

Because funding only exists as a real cost on a perpetual (futures
market in this simulator, never spot), the discount is applied on a
*second* column, ``target_funded``, while ``target_base`` carries
`kelly_regime_v4`'s untouched target through unchanged. ``on_bar`` reads
whichever column matches the market it is actually trading, so a spot
backtest of this class is bit-identical to plain `kelly_regime_v4`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context


class FundingGateConservative(KellyRegimeV4):
    """v4 with a bounded discount on funding-rich, momentum-unconfirmed bars."""

    name = "funding_gate_conservative"

    def __init__(
        self,
        funding: pd.Series | None = None,
        funding_pctile_threshold: float = 0.90,
        discount_factor: float = 0.5,
        momentum_days: int = 7,
        funding_halflife_days: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.funding_pctile_threshold = funding_pctile_threshold
        self.discount_factor = discount_factor
        self.momentum_days = momentum_days
        self.funding_halflife_days = funding_halflife_days

    # ------------------------------------------------------------------ pieces

    @staticmethod
    def _rolling_pctile(a: np.ndarray) -> float:
        """Fraction of the trailing window's valid values <= the last one.

        Called by a time-based ``rolling('365D')`` window, so ``a`` only
        ever contains bars up to and including the current one — never a
        future bar. NaNs inside the window (funding not yet known, or the
        gate having been explicitly masked outside the funding series'
        coverage) are excluded from both the numerator and the
        denominator rather than counted as "low"; the current bar's own
        value drives the answer, so if it is NaN the result is NaN (no
        signal), never a spurious low percentile.
        """
        last = a[-1]
        if np.isnan(last):
            return np.nan
        valid = a[~np.isnan(a)]
        if valid.size == 0:
            return np.nan
        return float(np.mean(valid <= last))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        df["target_base"] = df["target"]

        if self.funding is None:
            # No funding data supplied: the gate is a strict no-op.
            df["target_funded"] = df["target_base"]
            return df

        funding = self.funding

        # Causal alignment: ffill only, and only inside the funding
        # series' own observed range. Outside [funding.index[0],
        # funding.index[-1]] the aligned value is NaN — never carried
        # forward past the last real settlement, never backfilled before
        # the first one — so the gate is inert wherever funding was never
        # observed (2017-2019, and 2024+ if this is ever run past the
        # committed file's coverage).
        known = funding.reindex(df.index, method="ffill")
        outside = (df.index < funding.index[0]) | (df.index > funding.index[-1])
        known = known.where(~outside)
        df["funding_known"] = known

        # Causal EWM smoothing of the funding rate.
        ewm = known.ewm(halflife=f"{self.funding_halflife_days}D", times=df.index).mean()
        # ewm() carries the last computed value forward through NaN
        # inputs (it is not itself lookahead — it only uses past data —
        # but it would quietly un-mask the "no coverage" region with a
        # stale number). Re-mask explicitly so "no signal" stays "no
        # signal" for the percentile step below.
        ewm = ewm.where(~outside)
        df["funding_ewm"] = ewm

        # Causal rolling 365-day percentile rank: for each bar, the
        # fraction of the past year's (bar's-own-included) funding_ewm
        # values that are <= the current one. Time-based rolling windows
        # in pandas are right-closed on the current timestamp, so this
        # window never includes a bar after the current one.
        pctile = ewm.rolling("365D", min_periods=1).apply(self._rolling_pctile, raw=True)
        df["funding_pctile"] = pctile

        # Causal trailing momentum: uses only past bars (288 bars/day).
        mom = df["close"].pct_change(self.momentum_days * BARS_PER_DAY)
        df["mom"] = mom

        # Both conditions must hold; NaN in either comparison evaluates
        # to False in numpy/pandas, so a missing pctile or a missing
        # momentum (e.g. inside the first momentum_days) means no
        # discount, exactly the "NaN -> no signal" requirement.
        rich = (pctile.to_numpy() >= self.funding_pctile_threshold) & (mom.to_numpy() <= 0.0)
        discount = np.where(rich, self.discount_factor, 1.0)
        df["discount"] = discount

        target_base = df["target_base"].to_numpy()
        target_funded = target_base * discount
        # Defensive clip: discount only ever shrinks exposure, never
        # grows it or flips its sign, regardless of what discount_factor
        # is passed as.
        target_funded = np.sign(target_base) * np.minimum(
            np.abs(target_funded), np.abs(target_base)
        )
        df["target_funded"] = target_funded
        return df

    def on_bar(self, ctx: Context) -> None:
        # Funding is only ever charged on the futures/perp market in this
        # simulator; spot never sees it, so spot always trades the
        # unmodified v4 target.
        col = "target_base" if ctx.market.name == "spot" else "target_funded"
        t = float(ctx.bar[col])
        prev = float(ctx.prev[col]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
