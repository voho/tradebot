"""Funding as a gate on kelly_regime_v4: stand flat in the top decile (B-05, conservative).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

Prior research in this repo (docs/LEDGER.md R-14, R-16) established two
facts about Binance BTCUSDT perpetual funding over the period it is
observed (2020-01-01 .. 2023-12-31):

1. funding runs ~+20%/yr while ``kelly_regime_v4`` holds a position vs
   ~+2.8%/yr while flat — the crowding the strategy's own trend vote
   exploits is exactly what sets the rich rate the strategy then pays; and
2. sorting spot forward returns by funding-rate decile, the richest
   decile's 14-day forward return is only +0.56% against the cheapest
   decile's +4.13% — rich funding is a (weak) predictor of soft forward
   returns in its own right, not merely a cost line item.

The conservative reading of that evidence, and the ONLY thing this file
does: when funding is unusually rich relative to its own recent history,
stand flat. Nothing else about ``kelly_regime_v4`` changes — same vote,
same conditional-vol sizer, same deadband. This is a pure multiplicative
gate on the existing target, applied after v3/v4's own hysteresis, with
its own (much slower) hysteresis so it does not chatter in and out at
every 8-hourly settlement.

Mechanism
---------
1. The raw 8-hourly funding series is reindexed onto bar frequency
   causally: ``funding.reindex(df.index, method="ffill")`` — for any bar,
   the value used is the most recently SETTLED rate at or before that
   bar's timestamp. Bars before the first settlement (2020-01-01 03:00
   UTC) get 0.0 (reads as "not crowded" — the least restrictive default,
   and both splits below start exactly at the first settlement so this
   only ever touches the very first few bars of inner-train).

2. A CAUSAL rolling percentile rank of the (still 8-hourly, still lagged)
   funding series is computed with ``.rolling(window).rank(pct=True)`` —
   the window ending at settlement i only sees settlements <= i, so this
   is causal by construction. Ranking at 8-hourly frequency and then
   ffilling onto bar frequency is equivalent to ranking at bar frequency
   (the same 8-hourly value is repeated for 96 5-minute bars either way)
   and considerably cheaper.

3. A latched threshold, exactly analogous to ``KellyRegimeV3.prepare``'s
   volatility-extreme ``state`` loop: enter "stand flat" once the
   percentile rank reaches ``decile_in``, stay there until it drops back
   below ``decile_out`` (``decile_out < decile_in``, hysteresis so the
   gate does not flip on every settlement near the threshold).

4. ``df["target"] = v4_target * gate`` where ``gate in {0, 1}``. Nothing
   else about the v4 target is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context, Strategy


class FundingGateConservative(Strategy):
    """kelly_regime_v4, gated flat when funding sits in the top decile of its own history."""

    name = "_experiment_funding_gate_conservative"  # deliberately unregistered
    warmup = KellyRegimeV4.warmup  # 80*288 + 10, identical to v4

    def __init__(self, funding: pd.Series, decile_in: float = 0.90,
                 decile_out: float = 0.80, funding_lookback_days: int = 180,
                 **v4_kwargs) -> None:
        if decile_out >= decile_in:
            raise ValueError("decile_out must be < decile_in (hysteresis)")
        self.funding = funding
        self.decile_in = decile_in
        self.decile_out = decile_out
        self.funding_lookback_days = funding_lookback_days
        self._v4 = KellyRegimeV4(**v4_kwargs)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._v4.prepare(df)

        # Causal rolling percentile rank on the raw 8-hourly settlement
        # series (3 settlements/day) — computed BEFORE reindexing to bar
        # frequency, since ranking after would just repeat the same 96
        # values with no extra information.
        window = max(int(round(3 * self.funding_lookback_days)), 2)
        pct_8h = self.funding.sort_index().rolling(window, min_periods=window).rank(pct=True)

        # Reindex onto bars: for each 5m bar, the most recently SETTLED
        # value at or before that bar (causal by construction — never
        # reaches into a settlement that hasn't happened yet).
        funding_lag = self.funding.reindex(df.index, method="ffill").fillna(0.0)
        pct_lag = pct_8h.reindex(df.index, method="ffill")
        # Before the rolling window has filled (start of history) or
        # before the first settlement, percentile is undefined -> treated
        # as "not crowded" (0.0), the least restrictive default.
        pct = pct_lag.fillna(0.0).to_numpy()

        n = len(df)
        gate = np.ones(n)
        flat = False
        for i in range(n):
            if flat:
                if pct[i] < self.decile_out:
                    flat = False
            else:
                if pct[i] >= self.decile_in:
                    flat = True
            gate[i] = 0.0 if flat else 1.0

        df["funding_lag"] = funding_lag.to_numpy()
        df["funding_pct"] = pct
        df["funding_gate"] = gate
        df["target"] = df["target"].to_numpy() * gate
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
