"""Conditional panic-state exposure shrink on top of ``kelly_regime_v4``.

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The idea, one sentence
-----------------------
Add a conditional exposure shrink on top of ``kelly_regime_v4``'s existing
sizing, active only in a "panic state" defined as the *joint* condition of
(a) a recent market decline AND (b) elevated realized volatility -- rather
than reacting to volatility level alone.

Literature
----------
Daniel & Moskowitz (2016, NBER WP 20439 / J. Financial Economics):
momentum strategies suffer their worst losses in "panic states" --
following market declines, when ex-ante variance is high -- and a dynamic
strategy that de-risks specifically in panic states roughly doubles the
Sharpe of a static momentum strategy. Barroso & Santa-Clara (2015,
"Momentum Has Its Moments"): simple realized-vol scaling captures much of
this, but the crash risk is driven by the *interaction* of prior decline
and volatility, not vol alone -- the reason a joint indicator, not a
vol-only one, is the thing worth testing. Bianchi, De Polis & Petrella
("Taming Momentum Crashes") -- same family, more recent. Baker & McHale
(2013, Decision Analysis, "Optimal Betting Under Parameter Uncertainty"):
bet size should shrink with uncertainty about the edge, the general
principle a panic-state shrink instantiates for crypto trend-following.

Constraint attacked
--------------------
SIZE -- refined by conditioning on the *joint* (decline x vol) state
rather than the vol level alone, which R-08/R-10 already showed loses on
this data if done the naive way.

Not a duplicate of
-------------------
L-01/L-02/L-03/L-04 (heuristic latched vote + inverse-vol sizing, no
panic interaction term); R-08 (a better volatility *point forecast* --
negative, sign-inverting, because it de-levered more promptly into BTC's
high-vol, high-forward-Sharpe states); R-10 (documents that vol level
ALONE forecasts *higher*, not lower, forward Sharpe on this market -- the
trap this idea must avoid, see failure mode (a) below); R-11
(Grossman-Zhou drawdown cushion, keyed to the account's OWN equity
drawdown, not to market state).

Mechanism, concretely
----------------------
``decline = 1`` if the trailing N-day cumulative return of the market is
negative (N swept, e.g. 60/90/120 days). ``high_vol = 1`` if the trailing
realized volatility (EWM std over an M-day window, M swept, e.g.
10/20/30 days) sits above the top-tercile threshold of its own trailing
distribution, computed on a *rolling* lookback (1-2y, swept), never a
full-series fit. ``panic = decline AND high_vol``. When ``panic``, the
sizer's usual target exposure (v3/v4's vote-gated, breakout-conditional
inverse-vol size) is multiplied by an additional shrink factor
``shrink < 1`` (swept e.g. 0.25/0.5/0.75); otherwise the strategy behaves
exactly as ``kelly_regime_v4``.

Pre-registered failure modes (named before any code ran)
-----------------------------------------------------------
(a) **The important one.** The panic indicator ends up dominated by its
    ``high_vol`` leg and effectively reduces to "cut exposure whenever
    vol is high" -- which R-10 already showed loses here (BTC's inverse
    leverage effect: high vol forecasts the HIGHEST forward Sharpe,
    opposite of equities). Tested explicitly below via a ``use_decline``
    ablation flag: with it False, ``panic`` degenerates to ``high_vol``
    alone, giving a vol-only control arm run on the identical inner
    split as every joint-condition configuration.
(b) Adds complexity/turnover without moving any metric outside the +/-0.2
    Sharpe noise floor or a >=10pp drawdown improvement.
(c) Fails to replicate on the ETH falsification test (ordering/drawdown
    reversal, the same failure mode that retired R-28's headline in
    R-31).

Causality, and the one bug class this file exists to avoid
------------------------------------------------------------
The rolling tercile threshold (``.rolling(lookback).quantile(2/3)``) is
computed causally by construction -- the window at row i only ever looks
at rows <= i -- but it is exactly the shape of computation ("a quantile
fit over a stretch of the series") that a full-series ``.quantile()`` bug
would masquerade as if someone later "simplified" it. Two things make
this file's version robust to that: the vol series feeding the quantile
is itself lagged one bar (``.shift(1)``, matching ``kelly_regime``'s own
convention for its sizing vol), and the threshold is shifted one bar
*again* after the rolling quantile, so the value used to classify bar i
is fully determined before bar i's own realized vol enters the window.
Never call ``.quantile()`` on the whole series here -- see
``run_panic_scaling.py``'s ``causality()`` for the by-hand probe that
would catch a regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


class PanicScaledKelly(KellyRegimeV4):
    """``kelly_regime_v4`` with an extra exposure shrink in a joint panic state.

    Everything about the vote gate and the conditional (extreme-only)
    inverse-vol sizer is inherited unchanged from ``kelly_regime_v3`` /
    ``kelly_regime_v4`` -- see those docstrings. The only addition is a
    multiplicative ``shrink`` applied to the desired exposure, before the
    deadband, whenever the panic indicator is on.

    Set ``use_decline=False`` to run the vol-only ablation (failure mode
    (a)'s control arm): ``panic`` then equals ``high_vol`` alone, with the
    decline leg dropped.
    """

    name = "panic_scaled_kelly"  # experiment only -- NOT @register'd

    def __init__(
        self,
        decline_window_days: float = 90.0,
        vol_window_days: float = 20.0,
        tercile_lookback_days: float = 365.0,
        shrink: float = 0.5,
        use_decline: bool = True,
        horizons: tuple[int, ...] = (20, 40, 80),
        **kwargs,
    ) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.decline_window_days = float(decline_window_days)
        self.vol_window_days = float(vol_window_days)
        self.tercile_lookback_days = float(tercile_lookback_days)
        if not 0.0 < shrink <= 1.0:
            raise ValueError(f"shrink must be in (0, 1], got {shrink!r}")
        self.shrink = float(shrink)
        self.use_decline = bool(use_decline)
        # Warmup must cover the slowest of: v4's own 80-day anchor, the
        # decline window, and the vol-window-plus-tercile-lookback chain
        # that feeds the rolling quantile threshold. Recomputed per
        # instance because these are swept parameters, not fixed ones.
        needed_days = max(
            80.0,
            self.decline_window_days,
            self.vol_window_days + self.tercile_lookback_days,
        )
        self.warmup = int(needed_days * BARS_PER_DAY) + 10

    # --------------------------------------------------------------- panic

    def _panic(self, close: pd.Series, r: pd.Series) -> np.ndarray:
        """Joint (decline AND high_vol) indicator, or vol-only when ``use_decline=False``.

        Fully causal. The decline leg uses only close prices through the
        current bar -- the same convention the multi-anchor vote already
        uses (bar i's signal is decided with bar i's close, and orders
        fill at bar i+1's open, so nothing here reads ahead of that).
        The vol leg is lagged one bar before the rolling tercile is taken,
        and the tercile threshold itself is lagged a second time, so the
        threshold used to classify bar i is fixed before bar i's own vol
        contributes to it.
        """
        decline_bars = max(1, int(round(self.decline_window_days * BARS_PER_DAY)))
        cum_ret = np.log(close) - np.log(close.shift(decline_bars))
        decline = (cum_ret < 0.0).fillna(False).to_numpy()

        vol_bars = max(1, int(round(self.vol_window_days * BARS_PER_DAY)))
        vol_m = (r.ewm(span=vol_bars, min_periods=BARS_PER_DAY).std()
                 * np.sqrt(BARS_PER_YEAR)).shift(1)

        lookback_bars = max(1, int(round(self.tercile_lookback_days * BARS_PER_DAY)))
        min_p = min(lookback_bars, 60 * BARS_PER_DAY)
        thresh = (vol_m.rolling(lookback_bars, min_periods=min_p)
                  .quantile(2.0 / 3.0).shift(1))
        high_vol = (vol_m > thresh).fillna(False).to_numpy()

        return (decline & high_vol) if self.use_decline else high_vol

    # ------------------------------------------------------------- prepare

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # --- kelly_regime_v3/v4's vote, unchanged ---
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

        # --- kelly_regime_v3/v4's conditional (extreme-only) sizer, unchanged ---
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

        # --- the addition: panic-state shrink, applied before the deadband ---
        panic = self._panic(close, r)
        shrink_mult = np.where(panic, self.shrink, 1.0)

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
            desired = frac[i] * scale * shrink_mult[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["panic"] = panic
        return df
