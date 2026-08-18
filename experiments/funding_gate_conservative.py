"""Conservative funding gate on kelly_regime_v4 (backlog B-05).

Idea, one sentence: when the CURRENT settled funding rate sits in the top
`decile` of its own trailing `lookback_days` history (a causal,
backward-looking rolling percentile - never a settlement that has not
yet happened, never a statistic computed over the whole series), force
the position flat on markets that pay funding; otherwise behave exactly
like `kelly_regime_v4`. On markets that do not pay funding (spot) the
gate is a structural no-op - `on_bar` only ever consults the gate when
`ctx.market.pays_funding` is true.

Constraint attacked: COST (funding scales with the signal - R-14 found
it runs ~+20%/yr while the strategy holds vs +2.8%/yr flat, because the
same crowding that produces the trend vote sets the funding rate). This
is the deliberately unclever branch of B-05: a hard gate, not a sizing
continuum, so a separate "novel" variant has room to do something
smarter with the same R-16 finding (elevated funding predicts *lower*
14-day forward returns unless price is simultaneously still rising).

Not `@register`ed and not discovered by `tradebot run` - this lives only
in `experiments/`. See docs/LEDGER.md rows R-14, R-16 and backlog B-05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context

SETTLEMENTS_PER_DAY = 3  # Binance perp funding settles every 8h


def rolling_funding_percentile(funding: pd.Series, window: int,
                                min_periods: int) -> pd.Series:
    """Causal rolling percentile-rank of each settlement vs. its own trailing window.

    At settlement i, ``rolling(window)`` only ever spans settlements
    ``[i - window + 1, i]`` - strictly indices <= i - so this is causal
    by construction: it never touches a value that has not settled yet,
    and it is a fresh statistic per row rather than one full-series stat
    (mean/std/quantile) fit once and broadcast onto early rows (the
    lookahead class `test_causality_strict.py` checks for).

    NaN until `min_periods` trailing settlements exist.
    """
    def _pct(arr: np.ndarray) -> float:
        return float((arr <= arr[-1]).mean())

    return funding.sort_index().rolling(window, min_periods=min_periods).apply(
        _pct, raw=True
    )


class FundingGateConservative(KellyRegimeV4):
    """kelly_regime_v4, hard-gated flat when trailing funding is in the top decile.

    `funding` is the raw settlement-level series (`tradebot.data.load_funding`)
    used only to compute the causal gate signal - independent of whatever
    funding series (if any) the backtest itself charges as a cost.
    `lookback_days` sets the trailing window the percentile is measured
    against (in days; converted to settlements at 3/day); `decile` is the
    percentile threshold above which the position is forced flat (default
    0.90 = top decile).

    Outside the committed 2020-01-01..2023-12-31 funding window - before
    the first settlement, after the last, or before `lookback_days` of
    history has accumulated - there is no funding information, so the
    gate defaults OPEN (identical to unmodified kelly_regime_v4) rather
    than inventing a percentile from missing data.
    """

    name = "funding_gate_conservative"

    def __init__(self, funding: pd.Series | None = None,
                 lookback_days: int = 60, decile: float = 0.90, **kwargs) -> None:
        super().__init__(**kwargs)
        self._funding = funding
        self.lookback_days = lookback_days
        self.decile = decile

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # unmodified kelly_regime_v4 "target" column

        funding = self._funding
        if funding is None or len(funding) == 0:
            df["gate_open"] = 1.0
            return df

        window = max(2, int(round(self.lookback_days * SETTLEMENTS_PER_DAY)))
        pct = rolling_funding_percentile(funding, window, min_periods=window)

        # Causal alignment onto the 5m bar grid: each bar sees only the
        # most recently *settled* funding rate as of its own timestamp
        # (backward as-of join), and only within one settlement interval
        # (8h) of it. A bar more than 8h past the last known settlement
        # - i.e. outside the committed funding window entirely, or before
        # its first settlement - gets NaN rather than a stale carried-
        # forward value, which is what keeps the gate from firing on
        # invented data outside 2020-2023.
        left = pd.DataFrame({"ts": df.index})
        right = (pd.DataFrame({"ts": pct.index, "pct": pct.to_numpy()})
                 .dropna(subset=["ts"]).sort_values("ts"))
        # merge_asof requires identical datetime64 resolution on both keys;
        # the bar index and the funding index can differ (ms vs us) despite
        # both being tz-aware UTC.
        right["ts"] = right["ts"].astype(left["ts"].dtype)
        aligned = pd.merge_asof(left, right, on="ts", direction="backward",
                                 tolerance=pd.Timedelta(hours=8))
        pct_on_bars = aligned["pct"].to_numpy()

        df["gate_open"] = np.where(
            np.isnan(pct_on_bars), 1.0,
            np.where(pct_on_bars >= self.decile, 0.0, 1.0),
        )
        return df

    def on_bar(self, ctx: Context) -> None:
        def _effective(row) -> float:
            t = float(row["target"])
            if ctx.market.pays_funding and float(row["gate_open"]) < 0.5:
                return 0.0
            return t

        t = _effective(ctx.bar)
        prev = _effective(ctx.prev) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
