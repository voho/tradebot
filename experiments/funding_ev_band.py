"""kelly_regime_ev with a funding-carry-aware haircut and no-trade band (R-35, novel branch, 08-19).

Unregistered experiment: lives under ``experiments/`` so it is NOT
auto-discovered (docs/ROUTINE.md step 5). Do NOT decorate with
``@register``.

Idea in one sentence
---------------------
``kelly_regime_ev`` (L-05/L-06) derives its no-trade band purely from the
one-time taker **fee**: the growth given up by sitting at exposure ``f``
instead of the desired ``f*`` is ``(sigma^2/2)*(f-f*)^2`` per unit time
(a Kelly growth-rate argument), and correcting it is worth the trade only
when that exceeds ``fee*|Delta f|``. It never touches perpetual funding —
a *running* cost proportional to notional held over time
(``f * funding_rate`` per settlement), not a one-time fee on the trade.
R-14 measured `kelly_regime_v4` paying +20.05%/yr in funding while it
holds (against +2.78%/yr while flat) precisely because the crowding it
trades is the crowding that sets the rate; R-16 found funding itself
forecasts forward returns. This file prices the running cost into the
Kelly sizer on the one axis (SIZE) this project's twenty-five strategies
have ever found to work, per docs/LEDGER.md's R-35 pre-registration.

Mechanism (two complementary pieces, both implemented)
--------------------------------------------------------
1. **Haircut the growth-optimal target itself.** A Kelly sizer maximizing
   ``E[log(1+f*r)]`` nets out a *known* holding cost before computing the
   growth-optimal fraction: a carrying cost ``c`` per unit exposure per
   unit time shrinks the Merton/Kelly optimum from ``f* = mu/sigma^2`` to
   ``f* = (mu-c)/sigma^2``. ``kelly_regime_v4`` has no explicit ``mu``
   estimate (it is a vote+vol-target heuristic, not a literal Kelly
   formula), so this file treats v4's own ``frac*scale`` as the ``f*``
   term of that formula and subtracts the forecast funding drag in the
   SAME ``1/sigma^2`` units the rest of the family already uses for
   ``kelly_regime_ev``'s own EV band (``sigma`` = the shared ``_ev_vol``
   column both files compute identically):

       haircut  = funding_haircut_scale * max(funding_forecast_annual, 0) / sigma^2
       target   = max(frac * scale - haircut, 0)

   The ``max(..., 0)`` floor is safe *only* because this whole strategy
   family never goes short (the vote fraction lives in [0, 1] and the
   vol-targeting scale is non-negative — see ``kelly_regime.KellyRegime``'s
   own docstring: "the strategy stands flat rather than shorting"), so
   flooring the haircut result at zero can only ever remove exposure, not
   flip its sign. ``max(funding_forecast_annual, 0)`` means a *negative*
   forecast rate (longs being paid, i.e. a carry subsidy) is clamped to
   exactly zero drag rather than being allowed to enlarge the target —
   carrying cost must never make holding MORE attractive, per the
   pre-registration's hard requirement.

2. **Widen the no-trade band on the side that would ADD exposure**, reusing
   ``KellyRegimeEV._band()``'s literal formula
   (``2*fee/(horizon_years*sigma^2)``, Constantinides 1986 / Davis & Norman
   1990) but inflating its ``fee`` numerator with the *expected* funding
   bill over the same horizon the fee itself is amortized over:

       fee_up  = fee + funding_band_scale * max(funding_forecast_annual, 0) * horizon_years
       band_up   = _band(fee_up, sigma)      # only when INCREASING |exposure|
       band_down = _band(fee,    sigma)      # unchanged when CUTTING exposure

   This is the same spirit as Dumas & Luciano's (1991, J. Finance)
   two-barrier portfolio-choice framework under a continuous holding cost
   alongside proportional transaction costs: a continuous carry cost makes
   the strategy less eager to walk INTO a position it expects to pay carry
   on, without also making it more reluctant to walk OUT of one — cutting
   exposure keeps the unmodified (tighter) band so a richness signal never
   traps the strategy in a position longer than v4/EV would have held it.
   Controlled by ``widen_band`` (default True); set False to isolate the
   effect of (1) alone.

Built-in correctness check (the ``lam=0`` idiom, ``experiments/kelly_regime_v5_damp.py``)
-------------------------------------------------------------------------------------------
Both pieces are additive corrections gated on
``max(funding_forecast_annual, 0)``. When the forecast is exactly zero for
every bar (``funding_haircut_scale=0``, or no funding data at all, or every
bar outside the 2020-01-01..2023-12-31 coverage window — see below) then
``haircut == 0`` and ``fee_up == fee`` identically, for every row, by
construction — not by tuning. ``target`` then reduces to EXACTLY
``frac*scale`` (the same quantity ``kelly_regime_v4``/``kelly_regime_v3``
compute) and ``on_bar`` reduces to EXACTLY ``kelly_regime_ev.KellyRegimeEV``'s
symmetric ``abs(desired-current) > band`` check (see ``on_bar`` below: when
``fee_up==fee``, ``band_up==band_down``, and the two-sided check collapses
to the one-sided one bit-for-bit). This is verified empirically in
``experiments/reports/funding_ev_band_report.md`` section 1, not merely
asserted.

Funding data: loading, forecasting, alignment (the lookahead-risk part)
--------------------------------------------------------------------------
Real Binance BTCUSDT funding (``tradebot.data.load_funding``) is committed
for 2020-01-01..2023-12-31 only, 8h resolution. Per the standing
"never proxy unavailable data out of price" rule (docs/LEDGER.md /
docs/ROUTINE.md), bars outside that window get NO funding value
substituted — not a fill-forward of the last real settlement, not a period
mean. Concretely:

1. Reindex the sparse 8h series onto the 5m OHLCV index and forward-fill
   *within* coverage (``aligned``).
2. Explicitly overwrite with NaN any bar whose timestamp is after
   ``funding.index.max()`` — plain ``ffill`` would otherwise silently
   extend 2023-12-31's last settlement forever into 2024-2026 once the
   real series runs out, which is exactly the proxy-out-of-price mistake
   this project's rules forbid.
3. EWMA-smooth the (already NaN-bounded) ALIGNED series — not the raw 8h
   series before alignment — with a causal ``.ewm(span=..., min_periods=1)``.
   ``span`` is expressed in 5m bars (``funding_forecast_span_days *
   BARS_PER_DAY``) so it smooths across settlement boundaries at a
   comparable cadence to the days-denominated knob.
4. **Critical correction, found and fixed before any evaluation ran** (kept
   here because this project's culture records the near-miss, not just the
   final code): pandas' ``.ewm(...).mean()`` does NOT emit NaN at rows
   where the input is NaN once any real value has been seen upstream — it
   silently carries the last computed smoothed value forward through the
   NaN gap (verified empirically: ``pd.Series([nan,nan,1,2,nan,nan]).ewm(
   span=3,min_periods=1).mean()`` keeps outputting ``1.6667`` through the
   trailing NaNs rather than NaN). Left unguarded, this would have made the
   forecast — and therefore the haircut and the widened band — nonzero for
   every single bar from 2024 onward, silently violating the "inert outside
   coverage" requirement and re-introducing the exact proxy-out-of-price bug
   the alignment step was written to avoid. Fixed by re-masking the EWMA
   output to exactly 0.0 wherever the pre-EWMA aligned value was NaN
   (``forecast = forecast.where(aligned.notna(), 0.0)``), not merely
   trusting the EWMA's own NaN behaviour. This is the "easiest place to
   introduce a lookahead/proxy bug" the round's brief named, and it was
   caught here rather than left in.
5. ``funding_forecast_annual = forecast * 3 * 365.25`` (three 8h
   settlements/day) puts the forecast in the same annualized-rate units as
   ``target_vol``/``sigma`` so the ``.../sigma^2`` haircut and the
   ``...*horizon_years`` band inflation are dimensionally coherent with the
   rest of the Kelly/EV machinery.

Wherever ``aligned`` is NaN (no funding data at all, or outside
2020-2023), ``forecast`` — and therefore both the haircut and the band
inflation — is EXACTLY 0.0, by construction, matching kelly_regime_v4/
kelly_regime_ev bit-for-bit outside the funding-covered window.

Class shape
-----------
Subclasses ``KellyRegimeEV`` to reuse ``_band()`` literally (no
re-derivation of the fee/vol/horizon threshold) and its EV-band
``on_bar`` structure. ``prepare()`` is NOT calling
``super().prepare()`` — it reproduces v3/v4's vote + conditional
vol-targeting math byte-for-byte (the same pattern
``experiments/kelly_regime_v5_damp.py`` used) in one causal forward pass,
inserting the haircut before v3/v4's own 0.10 position-deadband latch, so
that when the haircut is zero the loop is bit-identical to
``KellyRegimeV3.prepare``/``KellyRegimeV4``, and therefore so is the
``_ev_vol`` column (same formula ``KellyRegimeEV.prepare`` uses, computed
once here rather than twice).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import load_funding
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_ev import KellyRegimeEV
from tradebot.strategy import Context

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SETTLEMENTS_PER_YEAR = 3 * 365.25  # 8h funding cadence


class FundingEVBand(KellyRegimeEV):
    """``kelly_regime_ev`` with a forecast-funding-drag haircut on the target and (optionally) on the band.

    See module docstring for the full derivation, the causality-critical
    NaN-masking fix, and the built-in ``forecast==0`` correctness check.
    """

    name = "funding_ev_band"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4 / kelly_regime_ev

    def __init__(self,
                 funding_forecast_span_days: float = 3.0,
                 funding_haircut_scale: float = 0.5,
                 funding_band_scale: float = 0.5,
                 widen_band: bool = True,
                 funding: pd.Series | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # New knobs only; every kelly_regime/v3/v4/ev knob (horizons, band,
        # target_vol, max_leverage, vol_span, deadband, vote_gamma,
        # anchor_span_days, high_in/out, low_in/out, horizon_days, min_band,
        # max_band) is inherited unchanged via **kwargs, matching v4/EV
        # defaults exactly when not overridden.
        self.funding_forecast_span_days = funding_forecast_span_days
        self.funding_haircut_scale = funding_haircut_scale
        self.funding_band_scale = funding_band_scale
        self.widen_band = widen_band
        # Injectable for the causality tamper test / unit checks; defaults
        # to the committed real series so the strategy is runnable with no
        # arguments, matching Strategy's own convention.
        self._funding_override = funding

    # ------------------------------------------------------------- funding

    def _funding_forecast_annual(self, df: pd.DataFrame) -> pd.Series:
        """Causal, coverage-bounded, EWMA-smoothed annualized funding forecast.

        Exactly 0.0 wherever no real funding value is available (no file,
        or the bar falls outside 2020-01-01..2023-12-31) — see module
        docstring step 4 for why this cannot simply trust pandas' own EWMA
        NaN-handling.
        """
        funding = self._funding_override
        if funding is None:
            funding = load_funding(DATA_DIR)
        if funding is None or len(funding) == 0:
            return pd.Series(0.0, index=df.index)

        aligned = (funding.reindex(df.index.union(funding.index))
                   .sort_index().ffill().reindex(df.index))
        # Never extend the last real settlement past the real data's end.
        aligned = aligned.where(df.index <= funding.index.max())

        span = max(1.0, self.funding_forecast_span_days * BARS_PER_DAY)
        forecast = aligned.ewm(span=span, min_periods=1).mean()
        # The critical fix (module docstring step 4): re-zero, do not trust
        # ewm to emit NaN through the coverage gap on its own.
        forecast = forecast.where(aligned.notna(), 0.0)
        return forecast * SETTLEMENTS_PER_YEAR

    # ------------------------------------------------------------- prepare

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # ---- byte-for-byte kelly_regime_v3/v4: latched multi-anchor vote -
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

        # ---- byte-for-byte v3/v4: conditional vol-targeting scale --------
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        # ---- new: forecast funding drag, converted to a haircut on f* ----
        annual_funding = self._funding_forecast_annual(df).to_numpy()
        annual_funding_pos = np.maximum(annual_funding, 0.0)  # never a bonus
        sigma2 = np.maximum(vol, 1e-6) ** 2
        haircut = self.funding_haircut_scale * annual_funding_pos / sigma2

        # ---- single causal forward pass: byte-for-byte v3/v4 breakout
        # hysteresis on the vol-targeting state, plus the new haircut -----
        n = len(df)
        target = np.zeros(n)
        pre_haircut_target = np.zeros(n)  # diagnostic: what v4's own loop would latch to
        pos = 0.0
        pos_pre = 0.0
        state = 0  # 0 normal vol band, +1 high-vol breakout, -1 low-vol breakout
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

            # f* = frac*scale (v4's growth-optimal target), haircut by the
            # forecast funding drag in 1/sigma^2 units; floored at 0 because
            # this family never goes short, so the floor can only remove
            # exposure, never flip its sign (never RAISES exposure).
            desired = max(frac[i] * scale - haircut[i], 0.0)
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

            # Diagnostic-only shadow: v4's own latch with NO haircut applied
            # (haircut[i] == 0 collapses `desired_pre` to `desired` exactly,
            # which is the built-in correctness check this column exists to
            # let a caller verify directly, bar by bar, without running a
            # second strategy instance).
            desired_pre = frac[i] * scale
            if abs(desired_pre - pos_pre) > self.deadband:
                pos_pre = desired_pre
            pre_haircut_target[i] = pos_pre

        df["target"] = target
        # Same formula as KellyRegimeEV.prepare's "_ev_vol" (same r,
        # vol_span, shift) -- computed once here rather than twice.
        df["_ev_vol"] = pd.Series(vol, index=df.index)
        # Diagnostics: mean-exposure / causality / plateau checks.
        df["_funding_annual"] = annual_funding
        df["_haircut"] = haircut
        df["_pre_haircut_target"] = pre_haircut_target
        return df

    # -------------------------------------------------------------- on_bar

    def on_bar(self, ctx: Context) -> None:
        """``KellyRegimeEV.on_bar``'s pattern, with an asymmetric band.

        Increasing |exposure| is judged against a band inflated by the
        forecast funding drag over the expected holding horizon; cutting
        exposure keeps the unmodified (tighter) band, so a richness signal
        never makes the strategy slower to leave a position, only slower
        to enter/add to one. When ``widen_band=False`` or the forecast is
        zero, ``fee_up == fee`` and both branches use the identical
        ``_band(fee, vol)``, which is exactly ``KellyRegimeEV.on_bar``'s
        single symmetric check.
        """
        desired = float(ctx.bar["target"])
        vol = float(ctx.bar["_ev_vol"])
        if not np.isfinite(vol) or vol <= 0:
            return

        equity = ctx.equity
        if equity <= 0:
            return
        current = ctx.position * ctx.close / equity

        fee = ctx.market.fee_rate
        if self.widen_band:
            annual_funding = float(ctx.bar["_funding_annual"])
            horizon_years = self.horizon_days / 365.25
            fee_up = fee + self.funding_band_scale * max(annual_funding, 0.0) * horizon_years
        else:
            fee_up = fee
        band_up = self._band(fee_up, vol)
        band_down = self._band(fee, vol)

        # Always allow a full exit, exactly as KellyRegimeEV does.
        if desired == 0.0 and abs(current) > 1e-9:
            ctx.order_notional(0.0)
            return
        if desired >= current:
            if desired - current > band_up:
                ctx.order_notional(desired)
        else:
            if current - desired > band_down:
                ctx.order_notional(desired)
