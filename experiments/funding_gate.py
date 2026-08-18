"""Funding as a gate on the existing strategy (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime_v4`` decides *how much* to hold from price alone. Two
ledger rows say price alone is leaving money on the table specifically
around funding:

* **R-14** — funding is not a flat tax on this strategy family, it is an
  *adversely timed* one: the strategy pays a mean rate of +20.05%/yr
  while it holds, against +2.78%/yr while it is flat, because the same
  crowding that produces the trend signal is what sets the funding rate.
* **R-16** — funding itself forecasts something price does not already
  say: the 14-day forward return spread between the cheapest and richest
  funding quintile is +3.57pp, and the correlation between funding and
  trailing return is only 0.39 — so it is *not* simply a momentum proxy
  restated. Controlling for trailing-return tercile, funding still
  separates outcomes within each momentum bucket (see the report for the
  full table), and critically: high funding paired with strongly rising
  price does **not** predict negative forward returns (+1.22% in
  ``VALIDATION.md``'s table) — only high funding *without* strong
  momentum does (-1.68% / -1.54%). That is the empirical basis for the
  "unless price is also strongly bullish" override below.

Mechanism, one sentence
------------------------
Multiply ``kelly_regime_v4``'s existing target exposure by a scale factor
that falls when trailing perpetual funding sits in its own top decile
(crowded longs) — unless trailing price momentum is also strongly
positive, in which case the scale factor is left at 1.0, because R-16
found the negative forward-return signal in rich funding disappears when
momentum confirms it.

Literature grounding beyond R-16 (gathered before any variant was coded)
--------------------------------------------------------------------
- **He, Manela, Ross & von Wachter (2024, "Fundamentals of Perpetual
  Futures", J. Finance)** — the no-arbitrage relation tying funding to
  the perp-spot basis; the theoretical reason funding is a persistent,
  economically meaningful rate rather than noise. Already cited in this
  repo's R-15/B-03.
- **Zhang (2026, SSRN 6185958, "Funding Rate Mechanism in Perpetual
  Futures")** — models the funding rate as an algorithmic feedback rule;
  in continuous time with risk-constrained arbitrageurs *and* momentum
  speculators, a linear funding rule induces an **endogenous
  mean-reverting basis**. This is a theoretical account of exactly the
  asymmetry R-16 measured empirically: funding pulls price back toward
  fair value except when momentum traders are strong enough to overwhelm
  the arbitrageurs, which is when the override case applies.
- **Nimmagadda & Sasanka (2019)** — the earliest systematic study of
  funding-rate dynamics (BitMEX), documenting heteroskedastic funding and
  Granger-causal links from funding to the perp price, i.e. funding is
  not merely contemporaneous with price, it carries directional
  information forward in time.
- **Presto Research, "Can Funding Rate Predict Price Change?" (industry
  research note, funding-rate variance-decomposition study)** — funding
  rate changes account for roughly 12.5% of the variance in price change
  over the *following* 7 days, decaying fast at longer horizons, and the
  authors judge the signal more reliable cross-sectionally (relative
  funding across many perpetuals) than for single-asset timing. This is
  a caution this design takes seriously: it is the reason the variants
  below stay conservative (a partial-scale gate, not a reversal) and why
  the pre-registered falsification bar does not expect a large effect.
- **Crowding-and-decay evidence for the carry side of the same trade**
  (already in R-15/``VALIDATION.md``): the funding-harvest Sharpe the
  literature reports for 2020-2025 falls from 6.45 to 4.06 in 2024 and
  negative in 2025 as the trade crowded (Ethena and exchange-native
  delta-neutral products absorbing the premium) — independently
  reconfirmed by a 2026 web search for this report. It does not bear on
  this specific gate (which only trades funding-as-signal, not
  funding-as-carry) but it is the reason B-02/B-03 stay blocked and why
  this design does not lean on funding levels persisting unchanged
  post-2023.

Constraint attacked: **COST**. This is the low-turnover use of R-16 the
ledger names: multiply an existing sizing decision by a gate rather than
trade the signal standalone (the high-turnover version is where
strategies go to die, R-12).

Not a duplicate of
-------------------
- **L-04/L-01/L-02/L-03** (``kelly_regime`` family) — same sizer and
  regime vote, unchanged; this file only adds a multiplicative funding
  overlay on top.
- **R-14** — measured the cost; this is the first strategy that tries to
  *avoid* the adversely-timed part of it rather than merely reporting it.
- **R-16** — measured the correlation; this is the first attempt to turn
  it into a trading rule, using the ledger's own prescribed low-turnover
  form (a gate on existing exposure, not a standalone reversal).
- **L-12 (harsanyi_crowd)** — also used a crowding intuition, but as a
  *direction* signal (long/short) rather than a *sizing* input, and lost.
  This file only ever scales an existing position down, never flips it.
- **B-03 (funding harvest / cash-and-carry)** — that is a market-neutral
  carry strategy (long spot, short perp); this is a directional
  strategy's sizing overlay. Different mechanism, same data file.

Data constraint, respected explicitly
--------------------------------------
Real Binance BTCUSDT funding covers 2020-01-01 03:00 UTC through
2023-12-31 19:00 UTC (confirmed by loading the file, not assumed). For
any bar outside that coverage the gate must have **NO EFFECT** — it must
fall back to exactly ``kelly_regime_v4``'s target, unmultiplied — so the
overlay's effect stays cleanly attributable and no bar silently trades on
a stale, indefinitely-extrapolated funding value. See ``_funding_state``.

Causality, respected explicitly
--------------------------------
All funding-derived quantities are computed on the funding series ALONE,
at settlement granularity, using ``.rolling(window).rank(pct=True)`` — a
trailing, causal percentile that only ever looks backward from each
settlement, never a full-series quantile. That rolling percentile is then
shifted by one further settlement (``.shift(1)``) before being aligned
onto the 5-minute bar grid with a backward ``merge_asof``, so a bar's
decision uses only the funding percentile as of the settlement *before*
the one that would otherwise be the most recent as-of value — one extra
settlement (8h) of conservatism beyond what "no lookahead" strictly
requires, matching this repo's habit of shipping causal quantities one
step more cautious than the minimum (see ``kelly_regime``'s
``vol...shift(1)``). The momentum override is a trailing return ending at
the current bar (``close / close.shift(N) - 1``), which by construction
uses only rows ``<= i``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

FUNDING_SETTLEMENTS_PER_DAY = 3  # every 8 hours


def _funding_state(
    close: pd.Series,
    funding: pd.Series,
    *,
    window_days: float,
    momentum_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bar-aligned (funding_active, funding_pctile, momentum_override).

    ``funding_active[i]`` is False for every bar outside the funding
    file's settlement coverage — the design constraint that makes the
    gate a no-op there.

    ``funding_pctile[i]`` is the trailing rolling percentile (0..1) of
    the funding rate as of the settlement *before* the one most recently
    known at bar i (one extra settlement of caution, see module
    docstring). NaN where inactive or where the trailing window has not
    filled.

    ``momentum_override[i]`` is a plain boolean array, independent of
    funding: trailing ``momentum_days``-day log return exceeds the
    caller-supplied threshold is NOT decided here (kept as a raw return
    so the caller can apply any threshold without recomputing).
    """
    window = int(round(window_days * FUNDING_SETTLEMENTS_PER_DAY))
    min_periods = max(10, window // 3)
    pct = funding.rolling(window, min_periods=min_periods).rank(pct=True)
    pct = pct.shift(1)  # one extra settlement of caution beyond as-of alignment

    # Backward as-of alignment: bar i gets the value from the most recent
    # settlement at or before its timestamp. Bars before the first
    # settlement get NaN (from merge_asof) -> inactive.
    # merge_asof requires matching datetime64 resolutions on both keys;
    # the funding file and the OHLCV frame can be parsed to different
    # units, so normalize before merging.
    settle_index = pd.DatetimeIndex(funding.index).as_unit(close.index.unit)
    settle_df = pd.DataFrame({"pctile": pct.to_numpy()}, index=settle_index)
    aligned = pd.merge_asof(
        pd.DataFrame(index=close.index), settle_df,
        left_index=True, right_index=True, direction="backward",
    )
    funding_pctile = aligned["pctile"].to_numpy()

    first_settlement = funding.index[0]
    last_settlement = funding.index[-1]
    # A bar stays "covered" for up to one settlement interval (8h) past
    # the last observed settlement -- the rate that settlement set is
    # still the live one -- and never beyond that, so post-2023-12-31
    # bars fall back to no-effect rather than freezing 2023's last value
    # forever.
    coverage_end = last_settlement + pd.Timedelta(hours=8)
    idx = close.index
    funding_active = (idx >= first_settlement) & (idx <= coverage_end)
    funding_active &= np.isfinite(funding_pctile)

    bars_per_window = int(round(momentum_days * BARS_PER_DAY))
    trailing_return = (close / close.shift(bars_per_window) - 1.0).to_numpy()

    return funding_active, funding_pctile, trailing_return


class FundingGateV4(KellyRegimeV4):
    """``kelly_regime_v4``, scaled down when funding is crowded and price is not confirming.

    ``target = v4_target * scale``, where ``scale`` is 1.0 everywhere the
    funding file has no coverage (design constraint) or funding is not
    unusually rich, and drops to ``floor`` when the trailing rolling
    percentile of funding is at or above ``pctile_threshold`` (top decile
    by default) — UNLESS trailing ``momentum_days``-day price return
    exceeds ``momentum_override``, in which case the scale stays at 1.0
    even though funding is rich (R-16: crowded longs only forecast
    negative forward returns when price is not also confirming them).

    ``floor=0.0`` is a hard gate (stand flat); ``floor>0`` is a partial
    scale-down. ``momentum_override=None`` disables the override
    entirely, so the gate fires purely on funding.
    """

    name = "funding_gate_v4"

    def __init__(
        self,
        pctile_threshold: float = 0.90,
        floor: float = 0.0,
        momentum_days: float = 7.0,
        momentum_override: float | None = 0.05,
        window_days: float = 180.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 < pctile_threshold < 1.0:
            raise ValueError(f"pctile_threshold must be in (0,1), got {pctile_threshold!r}")
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be in [0,1], got {floor!r}")
        self.pctile_threshold = pctile_threshold
        self.floor = floor
        self.momentum_days = momentum_days
        self.momentum_override = momentum_override
        self.window_days = window_days
        self._funding = None  # injected by the driver before prepare()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's own target, unmodified
        base_target = df["target"].to_numpy().copy()

        if self._funding is None or len(self._funding) == 0:
            # No funding file at all: gate is a pure no-op everywhere.
            df["funding_scale"] = 1.0
            df["funding_active"] = False
            df["funding_pctile"] = np.nan
            return df

        active, pctile, trailing_ret = _funding_state(
            df["close"], self._funding,
            window_days=self.window_days, momentum_days=self.momentum_days,
        )

        rich = active & (pctile >= self.pctile_threshold)
        if self.momentum_override is not None:
            overridden = rich & np.isfinite(trailing_ret) & (trailing_ret > self.momentum_override)
            gated = rich & ~overridden
        else:
            gated = rich

        scale = np.where(gated, self.floor, 1.0)
        df["target"] = base_target * scale
        df["funding_scale"] = scale
        df["funding_active"] = active
        df["funding_pctile"] = pctile
        return df
