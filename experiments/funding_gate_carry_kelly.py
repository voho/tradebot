"""R-33 novel variant: kelly_regime_v4 with a carry-adjusted Kelly numerator.

R-14 found ``kelly_regime_v4`` pays perpetual funding at +20%/yr while
holding a long versus +2.8%/yr while flat, because the same crowding the
strategy's price-only regime vote detects late is exactly what sets the
funding rate. This variant does not threshold or gate on that fact; it
derives the adjustment from the Kelly formula itself. A Kelly sizer's
target notional is ``target_vol / vol`` where ``target_vol`` stands in
for the estimated Sharpe ratio (``target_vol / vol`` is ``Sharpe / vol``,
i.e. ``mu / sigma**2`` in the shipped sizer's own units - see the
docstring of ``experiments/eprocess_regime.py`` for the derivation that
full Kelly sets the vol target equal to the estimated Sharpe ratio). The
standard treatment of a financing cost in a Kelly framework subtracts it
from the excess-return numerator before dividing by variance::

    f* = (mu - r_f) / sigma**2

so funding, converted to the same annualized units as ``target_vol``, is
subtracted from ``target_vol`` before the ``full`` / ``steady`` division
in both branches of ``kelly_regime_v3``'s conditional-volatility sizer -
every other mechanism (the anchor vote, the hysteresis latch, the
deadband, the leverage cap) is untouched.

One-sided by design (R-33's pre-registration, "Method, fixed in
advance"): this only ever *reduces* exposure relative to
``kelly_regime_v4``, never increases it, matching R-09's standing warning
against any change that could silently raise effective leverage. A
negative settled rate (shorts paying) is clamped to zero before use, so
it can never boost the target past the incumbent's own tuned value.

Not registered (no ``@register``, no import of ``tradebot.registry``):
this is an experiment living entirely under ``experiments/``, per R-33's
pre-registration in docs/LEDGER.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


class CarryAdjustedKelly(KellyRegimeV4):
    """kelly_regime_v4 with the Kelly target-vol numerator carry-adjusted by settled funding.

    ``funding_halflife_days=30.0`` is frozen for the primary run - fixed
    in R-33's pre-registration before any data was read and must not be
    quietly retuned; see the ``neighbours`` command in
    ``run_funding_gate_carry_kelly.py`` for the reported (not selected
    from) plateau check around it.

    Causal, by construction: ``self.funding`` is shifted by one
    settlement before use, so at any bar the adjustment only ever reflects
    a funding rate that has already settled, never one that applies to or
    postdates the bar's own settlement window. Before enough settlement
    history exists (``funding`` is None/empty, or a bar precedes the
    first settlement), the adjustment defaults to no derate
    (``eff_target_vol = target_vol``), matching the "open gate" default
    named in the pre-registration.
    """

    def __init__(self, funding: pd.Series | None = None,
                 funding_halflife_days: float = 30.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.funding_halflife_days = funding_halflife_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # --- identical to KellyRegimeV3.prepare(): anchor vote, realized
        # vol, and the slow (anchor_span_days) vol anchor. Byte-identical
        # to the incumbent - only what happens to target_vol changes.
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        frac = (sum(votes) / len(votes)).to_numpy()
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        # --- funding-adjusted target_vol, causal, one-sided ---
        if self.funding is not None and not self.funding.empty:
            # index k of `settled` carries the rate that settled at index
            # k-1 of the raw series (same timestamps, values shifted back
            # one settlement) - so at settlement time t_k only funding that
            # settled strictly BEFORE t_k has been incorporated, and every
            # bar in [t_k, t_{k+1}) that later ffill's from t_k inherits
            # exactly that: never the rate about to apply at t_k itself.
            settled = self.funding.shift(1)
            ann = settled * 3 * 365.25  # funding_study.py's convention (8h rate -> annualized)
            ann_ewm = ann.ewm(halflife=f"{self.funding_halflife_days}D",
                              times=ann.index).mean()
            ann_on_bars = ann_ewm.reindex(df.index, method="ffill").fillna(0.0)
            funding_cost = np.maximum(ann_on_bars.to_numpy(), 0.0)  # one-sided: never a subsidy
        else:
            funding_cost = np.zeros(len(df))

        with np.errstate(divide="ignore", invalid="ignore"):
            eff_target_vol = np.maximum(
                self.target_vol - funding_cost / np.where(vol > 0, vol, np.nan), 0.0)
            full = np.minimum(eff_target_vol / vol, self.max_leverage)
            steady = np.minimum(eff_target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        # Diagnostic columns only (not used by on_bar, which is inherited
        # from KellyRegime and reads only "target"); kept so the causality
        # check and the regime_check diagnostic can inspect the
        # intermediate funding-adjustment math directly, not just orders.
        df["vol"] = vol
        df["eff_target_vol"] = np.nan_to_num(eff_target_vol, nan=0.0)
        df["funding_cost"] = funding_cost
        df["frac_vote"] = frac
        return df
