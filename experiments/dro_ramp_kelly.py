"""Regime-age-conditioned fractional-Kelly shrink (backlog: none — new idea).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5.

The idea
--------
``kelly_regime_v4`` answers "how much of full Kelly should I bet" with one
constant, ``target_vol=0.55``, applied identically at every bar forever.
The strategy's own docstring calls this the fractional-Kelly prescription
under estimation error (MacLean, Thorp & Ziemba 2010) and admits the
fraction is hand-picked.

Distributionally-robust Kelly gambling (Sun & Boyd 2018, arXiv:1812.10371,
"Distributional Robust Kelly Gambling: Optimal Strategy under Uncertainty
in the Long-Run") replaces the plug-in return distribution with a ball of
distributions around the estimate and bets to maximize the *worst-case*
expected log-growth over that ball; the bet is provably more conservative
than plug-in Kelly, and the radius of the ball controls exactly how much
more conservative. Hsieh (2024, arXiv:2408.07879, "On Accelerating
Large-Scale Robust Portfolio Optimization") gives a tractable polyhedral-
ambiguity version of the same idea at S&P-500 scale. Both leave open how
to *set* the radius; Sun & Zou (2025, JUSTC 55(8), "Data-driven
distributionally robust Kelly portfolio optimization based on coherent
Wasserstein metrics") address exactly that — picking the ball size from
the data rather than by hand.

This experiment borrows the shape of that idea, not the convex program:
instead of a hand-set radius (equivalently, ``target_vol``'s constant
0.55), let the ambiguity radius be large exactly when within-regime
evidence is thin — immediately after the latched vote transitions — and
let it shrink back to zero (full ``target_vol``) as the *current* regime
accumulates confirming bars. This is a straight application of
MacLean-Thorp-Ziemba's own logic (shrink more when the drift estimate is
less trustworthy) along an axis the four prior sizing rounds never tried:
calendar time since the model's own signal last changed its mind, rather
than realized or forecast volatility.

Mechanism
---------
Reuse ``kelly_regime_v4``'s three-anchor latched vote exactly as shipped
(20/40/80-day anchors, 1% band) and its conditional (extremes-only)
volatility targeting exactly as shipped. The only new quantity is a
per-bar ramp multiplier on ``target_vol``::

    age_bars[i]  = 0                          if a transition fires at i
                 = age_bars[i-1] + 1           otherwise
    ramp[i]      = ramp_floor + (1 - ramp_floor) * (1 - exp(-age_days[i] / tau))
    target_vol_t = target_vol * ramp[i]

fed into the *same* ``full``/``steady`` formulas v3/v4 already compute, so
the anchor vote, the hysteresis state machine, the deadband and the
leverage cap are all untouched — only the number that used to be a
constant 0.55 becomes a causal function of one new thing: how long the
current vote value has held.

Three trigger definitions, i.e. three answers to "what counts as a
transition":

- ``any``      — the vote value changes at all (0 -> 1/3, 2/3 -> 1, ...).
- ``net``      — the vote's net *lean* changes sign (net-bullish <->
  net-bearish; ignores wobbles between e.g. 2/3 and 1 that do not cross
  1/2).
- ``down_only``— only a *decrease* in the vote fires the ramp; an
  increase resets confidence to 1.0 immediately (no shrink at all on
  entry into more bullish states) and only a subsequent decrease starts
  the clock again. This is the one variant designed to try to dodge the
  R-08/R-28/R-31/R-32 trap on purpose: see "predicted failure mode" below.

Not a duplicate of
-------------------
- **R-08** (better volatility *forecasting*, NEGATIVE — de-levered more
  promptly into BTC's high-vol/high-forward-Sharpe states). That round
  changed the *volatility estimator* feeding ``target_vol/vol``. This
  round never touches the volatility estimator at all; the new quantity
  is elapsed time since the vote last moved, which is not a function of
  the realized-vol level (a vote transition can happen quietly during a
  low-vol grind through the 1% band, or violently during a crash — the
  ramp does not know which).
- **R-28/R-31/R-32** (e-process evidence gate replacing the latched vote,
  all NEGATIVE at matched risk). Those rounds replaced *which fraction of
  the vote to believe* — a continuous, anytime-valid measure of evidence
  for "drift > 0" that stands in for the discrete 0/1/3/2/3/1 latch
  itself. This round leaves the latch completely alone (bit-for-bit
  identical ``frac`` to v4) and only rescales the *Kelly fraction
  magnitude* once the vote's value is already decided. The two mechanisms
  are answering different questions ("is the regime on?" vs "how hard do
  I trust betting the regime's implied size *right now*?").
- **`kelly_regime_v2`** (vote raised to a power > 1, NOT PROMOTED,
  -6.5% OOS). v2's discount is a function of the vote's *instantaneous
  value* (2/3 always gets the same haircut as any other time the vote
  reads 2/3, no matter how long it has read 2/3). This round's discount is
  a function of *elapsed time at the current value*, including full
  agreement (frac=1): a vote that just latched to 1 is shrunk hard here
  and a vote that has read 1 for eight months is not shrunk at all, a
  distinction v2 cannot express because it never looks at a clock.
- **`kelly_regime_v3`/`v4` conditional vol targeting**. Orthogonal axis —
  that machinery decides *when* to re-price the volatility denominator;
  this multiplies a separate ramp into the numerator (``target_vol``) and
  is compatible with it turned on or off.
- **R-11** (Grossman-Zhou drawdown cushion). Triggered by drawdown-from-
  peak, not by vote age; different state variable entirely.
- Not a re-tread of the fee/turnover rows (L-05/L-06/R-12/R-13): the
  deadband is untouched at 0.10 throughout.

Simulable here?
----------------
Yes. ``age_bars`` is a strictly backward-looking counter over the causal
``frac`` series already computed from OHLCV closes; no new data.

Predicted failure mode — written before any code ran
------------------------------------------------------
Regime transitions in this data cluster near volatility bursts (a flip
needs price to cross a 1% band around a 20/40/80-day mean, which happens
fastest during sharp moves). So the ``any`` and ``net`` triggers are, I
predict, at real risk of repeating the R-08/R-28/R-31/R-32 shape: bars
tagged "low confidence, recently flipped" will correlate with the
high-realized-vol bars that R-10 showed carry the *highest* forward
Sharpe in this asset, so shrinking them will cost more return than it
saves in risk, and average exposure will fall into a bull holdout the
same way it did in all four prior rounds. I expect this failure mode to
be real but *smaller* in magnitude than those rounds, because the
discount here decays away on a fixed calendar clock (weeks) regardless of
whether volatility stays elevated, rather than persisting for the whole
time volatility (or lack of evidence) remains high — it discounts a
regime's opening days, not its duration.

``down_only`` is the one variant built with a specific, stated reason to
expect it partially escapes the trap rather than a hope that it might:
it never discounts a fresh bullish entry (upward vote transitions reset
confidence to 1.0 immediately), so it should not delever the breakout
bars that R-10's inverse-leverage effect makes the best ones. It only
intervenes on downward transitions — the "known false-signal
neighbourhood" the incumbent's own hysteresis/latching already exists to
protect against (a fresh bearish flip in a historically-upward-drifting
asset is the higher-suspicion call). If ``down_only`` mostly preserves
v4's mean exposure through the bull holdout while still cutting the
drawdown around real bear onsets, that is the mechanism working as
designed; if it still loses to buy-and-hold, the honest reading is that
even asymmetric, decaying, transition-triggered shrinkage is still
shrinkage, and this asset's history punishes all of it alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


class RampedKellyRegime(KellyRegimeV4):
    """v4 with target_vol multiplied by a ramp keyed on time since the vote last moved."""

    name = "dro_ramp_kelly"

    def __init__(self, ramp_floor: float = 0.5, ramp_tau_days: float = 20.0,
                 ramp_trigger: str = "any", **kwargs) -> None:
        if ramp_trigger not in ("any", "net", "down_only"):
            raise ValueError(f"ramp_trigger must be any/net/down_only, got {ramp_trigger!r}")
        if not (0.0 <= ramp_floor <= 1.0):
            raise ValueError("ramp_floor must be in [0, 1]")
        super().__init__(**kwargs)
        self.ramp_floor = ramp_floor
        self.ramp_tau_days = ramp_tau_days
        self.ramp_trigger = ramp_trigger

    # ------------------------------------------------------------------ ramp

    def _ramp(self, raw_frac: np.ndarray) -> np.ndarray:
        """Causal per-bar multiplier on target_vol, in (0, 1].

        ``raw_frac[i]`` may only be compared with ``raw_frac[i-1]``: no
        window, no expanding statistic, nothing that sees its own future.
        """
        n = len(raw_frac)
        ramp = np.ones(n)
        floor, tau = self.ramp_floor, self.ramp_tau_days
        age = 0          # bars since the last triggering transition
        relaxed = True   # down_only only: True once an up-move has fired
        for i in range(n):
            if i == 0:
                ramp[i] = 1.0
                continue
            delta = raw_frac[i] - raw_frac[i - 1]
            if self.ramp_trigger == "any":
                trig = delta != 0.0
            elif self.ramp_trigger == "net":
                trig = np.sign(raw_frac[i] - 0.5) != np.sign(raw_frac[i - 1] - 0.5)
            else:  # down_only
                trig = delta < 0.0
                if delta > 0.0:
                    relaxed = True

            if self.ramp_trigger == "down_only" and relaxed and not trig:
                ramp[i] = 1.0
                continue

            if trig:
                age = 0
                if self.ramp_trigger == "down_only":
                    relaxed = False
            else:
                age += 1
            age_days = age / BARS_PER_DAY
            ramp[i] = floor + (1.0 - floor) * (1.0 - np.exp(-age_days / tau))
        return ramp

    # ----------------------------------------------------------------- prepare

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=df.index,
            )
            votes.append(v.ffill().fillna(0.0))
        raw_frac = (sum(votes) / len(votes)).to_numpy()
        frac = raw_frac ** self.vote_gamma if self.vote_gamma != 1.0 else raw_frac

        ramp = self._ramp(raw_frac)

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        tv = self.target_vol * ramp  # the only line that differs from kelly_regime_v3.prepare
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(tv / vol, self.max_leverage)
            steady = np.minimum(tv / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0
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
        df["ramp"] = ramp
        df["raw_frac"] = raw_frac
        return df
