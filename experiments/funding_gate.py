"""Funding-rate gate on kelly_regime_v4 (backlog B-05).

Not registered: this class lives under ``experiments/`` so it is not
auto-discovered by the strategy registry (ROUTINE.md step 5). This is
the first strategy-level implementation of the funding-as-signal
hypothesis; R-15/R-16 (docs/LEDGER.md) only measured it correlationally.

Mechanism, one sentence
------------------------
Haircut (or zero out) kelly_regime_v4's base long exposure whenever the
trailing perpetual funding rate sits above its own recent upper decile,
because rich funding prices exactly the crowded-long state that R-16
found predicts weak forward returns and R-14 found the strategy pays for
disproportionately.

Why, with citations
--------------------
kelly_regime_v4 answers "how much to hold" with a vote-gated,
volatility-targeted Kelly fraction; it never asks what the crowd it is
riding is paying to stay crowded. Cardaliaguet & Lehalle (2018,
Math. Fin. Econ.) ground the vote in a mean-field view where trend drift
IS the crowd's net flow; a perpetual's funding rate is the periodic
payment, set by the long/short open-interest imbalance, that keeps the
perp pinned to spot -- i.e. it is a direct price on that same crowd
imbalance, observable independently of the OHLCV price series the rest
of the strategy is built from. R-14 (docs/LEDGER.md) measured that
mechanism as a cost: funding on BTC ran +20.05%/yr while
``kelly_regime_v4`` held a position vs +2.78%/yr while flat -- the
crowding the strategy's vote detects is exactly what it pays for.
R-16 measured it as a signal, independently of price: BTC's forward
14-day return in the richest funding quintile was +0.56% against +4.13%
in the cheapest (+3.57pp spread), and high funding predicts *negative*
forward returns unless price is also rising strongly (correlation with
trailing return only 0.39, so this is not simply a momentum proxy). The
same row also warns the middle quintiles are non-monotone (Q3 beat Q4 at
tied clamped rates) -- evidence this may be noisier than the headline
spread suggests, which is why this file evaluates hard AND soft
haircuts rather than assuming the cleanest (zero-exposure) version wins.

Data honesty
------------
Real Binance BTCUSDT funding is committed for 2020-01-01 -> 2023-12-31
only (``data/btcusdt_perp_funding_8h.csv.gz``, 4,383 settlements, no
gaps). Outside that window there is no real funding data, and none is
synthesized or extrapolated here (the standing "never proxy unavailable
data out of price" rule) -- the gate defaults to fully inactive
(haircut multiplier == 1.0, identical to plain kelly_regime_v4)
whenever a bar has no real settlement to look up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import load_funding
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

ROOT = Path(__file__).resolve().parents[1]
FUNDING = load_funding(ROOT / "data")  # None if the committed file is somehow absent


class FundingGateV4(KellyRegimeV4):
    """kelly_regime_v4, haircut when trailing perp funding sits above its own upper decile (Cardaliaguet & Lehalle 2018; R-14; R-16)."""

    name = "funding_gate_v4"

    def __init__(
        self,
        settlement_span: float = 5.0,
        funding_quantile: float = 0.90,
        haircut: float = 0.0,
        min_settlements: int = 60,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= haircut <= 1.0:
            raise ValueError(f"haircut must be in [0, 1], got {haircut!r}")
        if not 0.0 < funding_quantile < 1.0:
            raise ValueError(f"funding_quantile must be in (0, 1), got {funding_quantile!r}")
        self.settlement_span = settlement_span
        self.funding_quantile = funding_quantile
        self.haircut = haircut
        self.min_settlements = min_settlements

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # base v4 exposure, in df["target"]
        base_target = df["target"].to_numpy(dtype=float)
        n = len(df)

        gate = np.ones(n, dtype=float)
        active = np.zeros(n, dtype=bool)

        if FUNDING is not None and len(FUNDING):
            # Trailing smoothing over SETTLEMENTS, not bars: an EWM over
            # the sparse 8-hourly series. Causal because the value at
            # settlement k depends only on settlements 0..k.
            smoothed = FUNDING.ewm(span=self.settlement_span, min_periods=1).mean()
            # Rolling/expanding quantile THRESHOLD: at settlement k this
            # depends only on settlements 0..k -- never a quantile taken
            # over the whole series and applied to early rows (the exact
            # lookahead class ROUTINE.md calls out by name).
            thresh = smoothed.expanding(min_periods=self.min_settlements).quantile(
                self.funding_quantile
            )
            settle = pd.DataFrame({
                "ts": smoothed.index,
                "smoothed": smoothed.to_numpy(),
                "thresh": thresh.to_numpy(),
            }).dropna(subset=["thresh"])

            if len(settle):
                # For each bar, the applicable rate is the most recent
                # settlement AT OR BEFORE that bar's own timestamp.
                # merge_asof(direction="backward") is exactly that lookup:
                # it can only match a settlement that has already
                # happened relative to the bar being decided.
                bars = pd.DataFrame({"ts": pd.DatetimeIndex(df.index).as_unit("us")})
                settle["ts"] = pd.DatetimeIndex(settle["ts"]).as_unit("us")
                merged = pd.merge_asof(bars, settle, on="ts", direction="backward")
                has_settlement = merged["thresh"].notna().to_numpy()

                # Do not carry a settlement forward past the committed
                # file's own span: ffill-forever would silently
                # extrapolate 2023 funding onto 2024-2026 bars the data
                # was never measured on. Restrict activity to bars inside
                # [first, last] settlement actually in the file -- outside
                # it the gate is a no-op by construction, matching plain
                # kelly_regime_v4 exactly.
                in_window = ((df.index >= FUNDING.index.min())
                             & (df.index <= FUNDING.index.max()))
                active = has_settlement & in_window

                smoothed_at_bar = merged["smoothed"].to_numpy()
                thresh_at_bar = merged["thresh"].to_numpy()
                high = active & (smoothed_at_bar > thresh_at_bar)
                gate = np.where(high, self.haircut, 1.0)

        df["funding_gate_active"] = active
        df["funding_haircut"] = gate
        df["target"] = base_target * gate
        return df

    # on_bar is inherited unchanged from KellyRegime: it reads
    # ctx.bar["target"] / ctx.prev["target"] and calls ctx.order_notional,
    # the same order-target pattern as every strategy in this family. This
    # class does not override it -- the gate only ever changes what ends
    # up written into the "target" column above.
