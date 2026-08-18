"""Asymmetric (downside-only) risk sizing on top of the kelly_regime_v4 mechanism.

Not registered: lives under ``experiments/`` per ROUTINE.md step 5, so it
is not auto-discovered by ``tradebot run``.

The question
------------
R-10 measured BTC's inverse leverage effect (Baur & Dimpfl 2018): positive
shocks raise realized volatility MORE than negative ones, and empirically
on this data high-*total*-volatility states carry the HIGHEST forward
Sharpe (+1.08 all bars, +2.06 when the regime gate is bullish). That is
why ``kelly_regime_v3``/``v4``'s conditional (extremes-only) targeting
beats continuous targeting. But R-08 found that a *more accurate*
forecast of that same symmetric/total-variance quantity made results
worse ($52K vs $115K): a better estimate of total variance de-levers even
more promptly into the good high-vol states.

The hypothesis here is that the problem was never estimation accuracy —
it is that total variance conflates two different things: vol from a
violent up-move (R-10 says stay levered into this) and vol from a genuine
downside crash (the tail risk ``kelly_regime`` exists to avoid). A
downside-only risk measure should distinguish these where symmetric
variance cannot.

Grounding: Rockafellar & Uryasev (2000, J. Risk) on CVaR as a coherent,
optimizable downside risk measure; MacLean, Sanegre, Zhao & Ziemba (2004,
J. Economic Dynamics & Control, "Capital growth with security") combine
Kelly growth with an explicit drawdown/tail-risk security constraint —
exactly the shape being tested: keep the Kelly/vol-target growth engine,
swap only the risk denominator that gates it.

This stays a SIZE mechanism: the risk measures below are computed
causally from OHLCV only (no new data source), and they change how much
is held, never a directional forecast.

Mechanism, one sentence each
-----------------------------
``risk_mode="vol"``            control: identical to ``kelly_regime_v4``
                                (total EWM volatility drives both the
                                breakout hysteresis and the scale).
``risk_mode="semidev_full"``   pure swap, no hysteresis: continuous
                                ``target_vol / semidev`` sizing (the
                                ``kelly_regime`` v1 mechanism) with total
                                volatility replaced everywhere by realized
                                downside semi-deviation (std of NEGATIVE
                                log returns only).
``risk_mode="semidev_hyst"``   v4's exact breakout-hysteresis structure,
                                but both the trigger ratio and the
                                full/steady scale are driven by
                                semi-deviation instead of total vol, so
                                the strategy only re-sizes on a rise in
                                genuine downside risk.
``risk_mode="cvar_hyst"``      same v4 hysteresis structure, risk axis is
                                a rolling historical CVaR (mean of the
                                worst ``cvar_q`` tail of trailing daily
                                returns over ``cvar_window_days``,
                                annualized) instead of total vol.
``risk_mode="blend"``          v4 hysteresis structure, risk axis is
                                ``alpha * vol + (1 - alpha) * downside``
                                for a downside measure chosen by
                                ``downside_measure`` ("semidev" or
                                "cvar") — tests whether a full swap is
                                too aggressive.
``risk_mode="state_only"``     v4's scale (total-vol inverse targeting)
                                is unchanged; only the breakout
                                hysteresis TRIGGER (when to re-size, not
                                how much) is driven by the downside
                                measure — so a bullish high-total-vol,
                                low-downside-risk regime stays in the
                                cheap constant-notional "steady" state
                                instead of de-levering.

Causality
---------
Semi-deviation: ``r.where(r < 0)`` fed to ``ewm(...).mean()`` — pandas
skips NaN entries and re-weights around gaps, so this is the EWM of
*only* the negative-return observations, weighted by recency among them,
then ``.shift(1)`` exactly like the incumbent's ``vol``. A rolling/EWM
statistic, never a whole-series statistic (R-21).

CVaR: computed once per DAY on a trailing window of daily log returns
(rolling ``.apply``, a rolling window statistic, never global), then
``.shift(1)`` at daily granularity (a full calendar day of conservatism)
before being forward-filled onto the 5-minute grid — so every 5m bar uses
only days strictly before the bar's own day, on top of the bar-level
shift used everywhere else in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

RISK_MODES = ("vol", "semidev_full", "semidev_hyst", "cvar_hyst", "blend", "state_only")


def _vote(close: pd.Series, horizons: tuple[int, ...], band: float) -> np.ndarray:
    """Latched multi-anchor vote, identical to kelly_regime_v4."""
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def _total_vol(r: pd.Series, vol_span: int) -> np.ndarray:
    """The incumbent's risk axis: EWM std of ALL log returns, annualized, lagged."""
    return (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
            * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def _semidev(r: pd.Series, vol_span: int) -> np.ndarray:
    """Realized downside semi-deviation: EWM std of NEGATIVE log returns only.

    ``r.where(r < 0)`` is NaN wherever the bar's return was >= 0; pandas'
    ewm skips those observations (adjusts weights around the gap) rather
    than treating them as zero, so this is the EWM standard deviation of
    the negative-return subsequence, weighted by recency among negative
    bars only — not the "downside deviation vs. zero" (Sortino) measure,
    which would silently shrink whenever bull runs dominate.
    """
    neg = r.where(r < 0)
    mean_neg = neg.ewm(span=vol_span, min_periods=BARS_PER_DAY // 4).mean()
    var_neg = (neg - mean_neg).pow(2).ewm(span=vol_span, min_periods=BARS_PER_DAY // 4).mean()
    return (np.sqrt(var_neg) * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()


def _cvar(close: pd.Series, index: pd.DatetimeIndex, window_days: int,
          q: float) -> np.ndarray:
    """Rolling historical CVaR of daily log returns, annualized, broadcast to 5m bars.

    ``cvar_of`` on window W returns the mean of the worst ``q`` tail of
    daily log returns in that trailing window (a negative number; more
    negative = more downside risk). Computed once per day (a rolling
    window statistic on ~3,500 daily observations, not the 1M-bar
    series), then shifted a full day and forward-filled onto the 5m
    index so no bar ever sees its own day's return.
    """
    daily_close = close.resample("1D").last().dropna()
    daily_r = np.log(daily_close).diff()

    def cvar_of(window: np.ndarray) -> float:
        window = window[~np.isnan(window)]
        if len(window) < max(10, window_days // 3):
            return np.nan
        var = np.quantile(window, q)
        tail = window[window <= var]
        return float(tail.mean()) if len(tail) else float(var)

    cvar_daily = daily_r.rolling(window_days, min_periods=max(10, window_days // 3)).apply(
        cvar_of, raw=True)
    cvar_daily = cvar_daily.shift(1)  # extra full-day lag: today's bars see yesterday's CVaR
    risk = (-cvar_daily) * np.sqrt(365.25)  # loss magnitude, annualized like `vol`
    risk_5m = risk.reindex(index, method="ffill")
    return risk_5m.to_numpy()


class CVaRRegime(Strategy):
    """kelly_regime_v4 with the risk axis swapped for a downside-only measure.

    Same 20/40/80-day latched anchor vote, same deadband, same v3/v4
    breakout-hysteresis skeleton as the incumbent; only ``risk_mode``
    changes which quantity plays the role of "realized volatility" in
    ``target_vol / risk``. See module docstring for the mechanism of
    each mode.
    """

    name = "cvar_regime"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(
        self,
        risk_mode: str = "vol",
        horizons: tuple[int, ...] = (20, 40, 80),
        band: float = 0.01,
        target_vol: float = 0.55,
        max_leverage: float = 2.0,
        vol_span: int = 8 * BARS_PER_DAY,
        deadband: float = 0.10,
        anchor_span_days: int = 180,
        high_in: float = 1.70,
        high_out: float = 1.20,
        low_in: float = 0.55,
        low_out: float = 0.85,
        cvar_window_days: int = 60,
        cvar_q: float = 0.05,
        alpha: float = 0.5,
        downside_measure: str = "semidev",
    ) -> None:
        if risk_mode not in RISK_MODES:
            raise ValueError(f"risk_mode must be one of {RISK_MODES}, got {risk_mode!r}")
        if downside_measure not in ("semidev", "cvar"):
            raise ValueError("downside_measure must be 'semidev' or 'cvar'")
        self.risk_mode = risk_mode
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.cvar_window_days = cvar_window_days
        self.cvar_q = cvar_q
        self.alpha = alpha
        self.downside_measure = downside_measure

    # ------------------------------------------------------------- risk axis

    def _risk(self, close: pd.Series, r: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Returns (risk_for_scale, risk_for_trigger) — equal except in state_only."""
        vol = _total_vol(r, self.vol_span)
        if self.risk_mode in ("vol",):
            return vol, vol
        if self.risk_mode in ("semidev_full", "semidev_hyst"):
            sd = _semidev(r, self.vol_span)
            return sd, sd
        if self.risk_mode == "cvar_hyst":
            cv = _cvar(close, close.index, self.cvar_window_days, self.cvar_q)
            return cv, cv
        if self.risk_mode == "blend":
            down = (_semidev(r, self.vol_span) if self.downside_measure == "semidev"
                    else _cvar(close, close.index, self.cvar_window_days, self.cvar_q))
            blended = self.alpha * vol + (1.0 - self.alpha) * down
            return blended, blended
        if self.risk_mode == "state_only":
            down = (_semidev(r, self.vol_span) if self.downside_measure == "semidev"
                    else _cvar(close, close.index, self.cvar_window_days, self.cvar_q))
            return vol, down  # scale from total vol, trigger from downside measure
        raise AssertionError(self.risk_mode)  # pragma: no cover

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()
        frac = _vote(close, self.horizons, self.band)

        risk_scale, risk_trigger = self._risk(close, r)

        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.minimum(self.target_vol / risk_scale, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)

        if self.risk_mode == "semidev_full":
            # No hysteresis: continuous inverse-risk targeting (kelly_regime's
            # own mechanism), risk axis swapped for semi-deviation.
            n = len(df)
            target = np.zeros(n)
            pos = 0.0
            for i in range(n):
                desired = frac[i] * full[i]
                if abs(desired - pos) > self.deadband:
                    pos = desired
                target[i] = pos
            df["target"] = target
            df["risk"] = risk_scale
            return df

        # All other modes: v4's breakout-hysteresis skeleton. `risk_trigger`
        # decides WHEN to switch state; `risk_scale` (== risk_trigger except
        # in state_only) decides the full-breakout scale.
        slow = (pd.Series(risk_trigger).ewm(
            span=self.anchor_span_days * BARS_PER_DAY,
            min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, risk_trigger / slow, np.nan)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        steady = np.where(np.isfinite(steady), steady, 0.0)
        # steady must use the SAME risk axis as `full` (risk_scale), not
        # necessarily the trigger's slow anchor, when they differ (state_only).
        if self.risk_mode == "state_only":
            slow_scale = (pd.Series(risk_scale).ewm(
                span=self.anchor_span_days * BARS_PER_DAY,
                min_periods=BARS_PER_DAY).mean().to_numpy())
            with np.errstate(divide="ignore", invalid="ignore"):
                steady = np.minimum(self.target_vol / slow_scale, self.max_leverage)
            steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high breakout, -1 low breakout
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
        df["risk_scale"] = risk_scale
        df["risk_trigger"] = risk_trigger
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
