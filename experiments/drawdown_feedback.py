"""Smooth own-drawdown feedback multiplier on kelly_regime_v4 (parallel branch B).

Not registered: lives under ``experiments/`` so it is not auto-discovered,
per ROUTINE.md step 5. This file and ``run_drawdown_feedback.py`` are
branch B's alone.

The mechanism, in one sentence
-------------------------------
Multiply ``kelly_regime_v4``'s existing target exposure by a smooth,
continuous, monotonically-decreasing function of the STRATEGY'S OWN
current drawdown from its running peak (notional-implied) equity, so
exposure de-levers gradually as the strategy's own equity curve worsens
and re-levers gradually as it recovers - no hard threshold, no ratchet,
no floor on wealth.

    multiplier(dd) = clip(1 - (dd / dd_cap) ** power, floor, 1.0)
    target[i]      = kelly_regime_v4.target[i] * multiplier(dd[i])

Why this form and not something else
-------------------------------------
Chekhlov, Uryasev & Zabarankin's Conditional Drawdown-at-Risk line, and
the "Drawdown-Modulated Feedback Control" lemma (arXiv:1710.01503), frame
drawdown control as continuous feedback rather than a hard floor: the
control variable (here, the exposure multiplier) is a smooth function of
the state (current drawdown), so it can be made to barely respond in the
interior of its domain and respond sharply only near the boundary, by
choice of the exponent ``power``. Busseti, Boyd et al.'s risk-constrained
Kelly framework (2016) and its 2026 descendant (arXiv:2603.01298,
"Single-Asset Adaptive Leveraged Volatility Control") are in the same
family: a transaction-cost-aware, continuously adjusted leverage control
around a Kelly-type base sizer, rather than a discrete regime switch.
``power=1`` gives linear (CDaR-style) feedback; ``power>1`` gives a
convex response that is close to flat for small/moderate drawdowns and
steepens only as ``dd`` approaches ``dd_cap`` - the "barely bites in
normal drawdowns, meaningfully de-levers in the tail" calibration this
branch's assignment asked for. ``floor`` keeps the multiplier bounded
away from zero so the position is never hard-cut to nothing; there is no
wealth floor and no cushion ratchet anywhere in this file.

Why this is NOT R-11's Grossman-Zhou cushion (must be argued explicitly)
--------------------------------------------------------------------------
R-11 (docs/LEDGER.md) sizes off the REMAINING RISK BUDGET relative to a
floor on wealth: exposure is proportional to the distance between current
wealth and a floor that ratchets up as wealth grows, so once the cushion
is spent the rule is forced toward zero exposure and stays there until
wealth rebuilds past the (ratcheted, path-dependent) floor - Klass &
Nowicki (2005)'s prediction, reproduced in R-11, is that this SELLS LOW
in a mean-reverting-drawdown asset because the floor does not release
until price has already recovered past it.

This mechanism has no floor and no ratchet. ``multiplier(dd)`` is a pure
function of the CURRENT drawdown level, evaluated fresh every bar: if
drawdown recovers, the multiplier recovers on the same bar, symmetrically
and without memory of how deep it got. There is no wealth level the
strategy must reclimb before re-levering, only a drawdown percentage it
must reduce. Whether that distinction actually changes behaviour on this
data, rather than merely reading differently on paper, is exactly what
step 3-7 below measure - and if it converges on R-11's failure mode
anyway (bites during ordinary, mean-reverting drawdowns and de-levers
into the recovery), the report says so plainly rather than arguing the
mechanism away.

Keeping the drawdown computation causal
-----------------------------------------
``prepare()`` is called once, up front, with the whole OHLCV frame and no
access to the broker's realized equity path (fills, fees and funding are
simulated by the engine afterward, per bar, and are not visible inside
``prepare``). So "this strategy's own equity" cannot be read off the
broker; it has to be built as a NOTIONAL-IMPLIED proxy inside a single
strict left-to-right forward pass over the array, exactly the pattern
``kelly_regime``/``v3``/``v4`` already use for their own latching state
machine:

  1. ``raw[i]`` = kelly_regime_v4's own (already-causal, already
     deadbanded) target column, obtained by calling
     ``KellyRegimeV4.prepare()`` unmodified and reading its ``target``.
  2. A second forward loop over ``i = 0..n-1`` carries three pieces of
     state: ``equity`` (the notional-implied proxy), ``peak`` (its running
     maximum) and ``pos`` (the final, gated position this strategy is
     actually holding). At step ``i``:
       a. ``equity *= 1 + pos * r[i]`` - ``pos`` is the position DECIDED
          at step ``i-1`` (held through bar ``i``), and ``r[i]`` is bar
          ``i``'s own close-to-close return, known at bar ``i``'s close -
          exactly the information a live strategy has when ``on_bar`` is
          called (mark-to-market equity through the current close). This
          mirrors how the real engine marks ``ctx.equity``: the fill that
          set ``pos`` happened at bar ``i-1``'s open, so bar ``i``'s
          return is realized P&L on a position already held.
       b. ``peak = max(peak, equity)`` and ``dd = (peak-equity)/peak`` -
          both use only ``equity`` values through step ``i``.
       c. ``m = multiplier(dd)``, ``desired = raw[i] * m``, and the SAME
          kind of deadband v4 already uses is applied before updating
          ``pos`` (a continuously wiggling multiplier would otherwise
          reorder every bar; the deadband exists to keep turnover
          comparable to v4's, not to change the mechanism).
       d. ``target[i] = pos``.

  Every quantity at step ``i`` is a function of ``close[0..i]`` and of
  ``pos`` decisions made at steps ``< i`` only. There is no expanding
  statistic computed over the whole series and indexed backward (the
  lookahead class a truncation test alone cannot catch): ``peak`` is a
  running max over a strictly-causal ``equity`` array, not a max over the
  full series. This is the single most important property of the file
  and it is checked by hand in ``run_drawdown_feedback.py causality``,
  the same two-opposite-tampers procedure R-28/R-31 used for their own
  unregistered strategies.

  This proxy deliberately ignores fees and funding (it exists only to
  drive the feedback signal, not to report P&L - the real engine still
  computes the actual account) and it is SELF-referential: the drawdown
  that throttles this strategy is measured on ITS OWN post-throttle
  notional path, not on plain v4's. A live implementation of this rule
  would observe exactly that - its own realized equity, which already
  reflects every past throttling decision - so using a different
  (ungated) strategy's equity to drive the throttle would itself be a
  look-elsewhere signal foreign to the mechanism being tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context


class DrawdownFeedbackKelly(KellyRegimeV4):
    """kelly_regime_v4 exposure, multiplied by a smooth own-drawdown feedback penalty."""

    name = "drawdown_feedback_kelly"  # experiment only - never @register-ed

    def __init__(
        self,
        dd_cap: float = 0.50,
        power: float = 2.0,
        floor: float = 0.0,
        gate_deadband: float = 0.10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if dd_cap <= 0.0:
            raise ValueError(f"dd_cap must be positive, got {dd_cap!r}")
        if power <= 0.0:
            raise ValueError(f"power must be positive, got {power!r}")
        if not 0.0 <= floor < 1.0:
            raise ValueError(f"floor must be in [0, 1), got {floor!r}")
        self.dd_cap = dd_cap
        self.power = power
        self.floor = floor
        self.gate_deadband = gate_deadband

    def multiplier(self, dd: np.ndarray) -> np.ndarray:
        """The feedback function itself, exposed for inspection/plotting."""
        dd = np.asarray(dd, dtype=float)
        m = 1.0 - (dd / self.dd_cap) ** self.power
        return np.clip(m, self.floor, 1.0)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's own causal target (incl. its own deadband)
        raw = df["target"].to_numpy(dtype=float)
        close = df["close"]
        r = np.log(close).diff().fillna(0.0).to_numpy(dtype=float)

        n = len(df)
        gated = np.zeros(n)
        dd_arr = np.zeros(n)
        mult_arr = np.ones(n)

        equity = 1.0
        peak = 1.0
        pos = 0.0  # the gated position HELD during the bar about to be processed
        for i in range(n):
            if i > 0:
                factor = 1.0 + pos * r[i]
                equity *= factor if factor > 1e-6 else 1e-6
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0.0 else 0.0
            dd_arr[i] = dd
            m = 1.0 - (dd / self.dd_cap) ** self.power
            m = min(1.0, max(self.floor, m))
            mult_arr[i] = m
            desired = raw[i] * m
            if abs(desired - pos) > self.gate_deadband:
                pos = desired
            gated[i] = pos

        df["raw_target"] = raw
        df["dd_proxy"] = dd_arr
        df["dd_multiplier"] = mult_arr
        df["target"] = gated
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
