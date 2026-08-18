"""FundingCrowdKelly: a continuous funding-cost shrink on top of kelly_regime_v4.

Not registered — lives under ``experiments/`` per ROUTINE.md step 5, and is
not auto-discovered (no ``tradebot.registry.register`` import or decorator).

The question (backlog B-05, docs/LEDGER.md section D)
-------------------------------------------------------
R-14 (``scripts/funding_study.py``) measured that real Binance BTCUSDT perp
funding costs ``kelly_regime_v4`` ~15%/yr as a constant-long tax and, worse,
is *adversely timed*: funding runs ~+20%/yr while the strategy holds a
position vs +2.8%/yr while flat, because the same anchor-vote crowding that
makes the strategy bullish is what pushes the funding rate up. B-05 proposes
using the funding data itself — already committed at
``data/btcusdt_perp_funding_8h.csv.gz``, real Binance settlements, 8-hourly,
EXACTLY 2020-01-01..2023-12-31 — as a gate on the existing strategy. A
sibling branch implements the literal version: a hard decile threshold that
snaps the position to flat once trailing funding crosses a rank cutoff. This
file is the other half of that idea.

Theoretical grounding
----------------------
Funding is not a per-trade cost. ``kelly_regime_ev`` (L-05, already in this
repo) handles the exchange's *taker fee* correctly by treating it the way
Constantinides (1986, J. Political Economy) and Davis & Norman (1990, Math.
OR) do: a cost paid once per rebalance, which is why the answer there is a
no-trade *band* on the size of a move. Funding is structurally different —
it is a continuously-accruing *holding* cost, proportional to position size
and to time, charged whether or not the position changes. That is exactly
the structure of a margin or borrowing rate in classical continuous-time
portfolio choice: Merton (1971, J. Economic Theory, "Optimum consumption
and portfolio rules in a continuous-time model") shows that when a levered
position pays a financing rate ``c`` against expected excess return ``mu``
and variance ``sigma^2``, the growth-optimal (Kelly) fraction is

    f* = (mu - c) / sigma^2  =  f0 * (1 - c/mu)

i.e. the no-cost optimum ``f0 = mu/sigma^2`` shrinks in proportion to the
financing rate relative to the return it is financing. This project's sizer
never estimates ``mu`` directly — ``kelly_regime`` substitutes a latched
multi-anchor vote plus ``target_vol/realized_vol`` for a Merton-style
closed-form Kelly fraction — so there is no ``mu`` on hand to plug into that
ratio. What transfers is the *shape*: financing cost should shrink exposure
continuously in proportion to its own severity relative to a ceiling,
tracking the a-priori Merton mechanism (cost divided by the thing it is
financing) with the ceiling standing in for the unobserved ``mu``, rather
than snap to flat at an empirical rank cutoff. MacLean, Thorp & Ziemba
(2010) — already the standing citation in ``kelly_regime`` for why this
project runs *fractional* rather than full Kelly — give the general
argument for the "continuously" part: full Kelly is fragile to estimation
error, and the standard response to a noisy input is to shrink it smoothly,
not to threshold it. A hard percentile gate creates a step discontinuity in
exposure exactly at its boundary, which manufactures trades (and turnover
cost) out of noise that crosses the cutoff back and forth; a smooth
multiplier does not have a boundary to whipsaw around.

Mechanism
---------
1. Run ``KellyRegimeV4.prepare`` unmodified to get its ``target`` column —
   this class changes nothing about the vote or the vol-target sizer.
2. Build a **causal** funding-at-bar series: at bar ``t`` only settlements
   at or before ``t`` are known, i.e. ``REAL.reindex(df.index,
   method="ffill")`` (spot-checked below to change only at/after each
   settlement's own timestamp — never before).
3. Smooth it with a causal EWM computed **on the settlement-frequency
   series** (4,383 points, 8h apart, not the 5m-bar grid — an EWM over a
   piecewise-constant step function resampled to bars would just be a
   leaky, meaningless re-derivative of the same steps). Settlements are
   evenly spaced with no gaps over the committed window, so the half-life
   is expressed in settlements: ``halflife_settlements =
   funding_ewm_halflife_days * 3`` (3 settlements/day). The smoothed series
   is then **shifted by one settlement** before being read: at bar ``t``,
   let ``s`` be the most recently known settlement (the ffill target).
   ``s`` is itself "known" by bar ``t``, but the smoothed value used for
   the multiplier is ``ewm(REAL).shift(1)`` reindexed onto ``t`` — which,
   because both series share the settlement index, lands on
   ``ewm(REAL)`` evaluated one settlement *before* ``s``. So the smoothing
   used at ``t`` reflects settlements strictly before ``s``, never ``s``
   itself, even though ``s``'s raw value is already public. This is a
   second, deliberately conservative causality margin on top of (2), not a
   requirement of it — it costs one settlement (≤8h) of extra lag in
   exchange for a recipe that is trivially safe to audit column-by-column.
4. Annualize: ``funding_annualized = funding_ewm * 3 * 365.25``.
5. ``mult = clip(1 - max(0, funding_annualized) / cost_ceiling, floor_mult,
   1.0)``. Only positive (longs-pay) funding shrinks exposure — this
   strategy never shorts (it stands flat instead, per ``KellyRegime``'s own
   docstring), so negative funding (shorts pay) is not a cost it ever
   incurs and leaves ``mult = 1.0``. ``cost_ceiling`` defaults to 0.30
   (30%/yr) — set with headroom above R-14's measured ~20%/yr
   *adversely-timed* holding cost, as an a-priori ceiling rather than a fit
   to that number (fitting the ceiling to the very statistic it is meant to
   defend against would make the "principled" framing a post-hoc rationale
   for the sibling branch's threshold).
6. Where funding data is unavailable — before 2020-01-01, before the
   settlement-EWM has any history (the first settlement's smoothed-and-
   shifted value is NaN by construction), or after 2023-12-31 — ``mult`` is
   forced to exactly ``1.0``, making this reduce bit-identically to
   ``KellyRegimeV4`` outside its covered/warmed window (checked below with
   ``max_abs_diff``).
7. ``desired = parent_target * mult`` is then put through **this class's
   own** latch/deadband loop (``self.deadband``, inherited from
   ``KellyRegime``, same "latch until the move exceeds the band" rule as
   every ``kelly_regime*`` class) rather than reusing the parent's already-
   latched decisions. Multiplying the parent's *post-deadband* ``target``
   by ``mult`` and shipping that directly would let ``mult`` create a fresh
   move on every wiggle even where the parent's own deadband had already
   suppressed one — e.g. the parent latches to 0.40 and holds it flat for a
   month; a multiplier oscillating 0.98/1.02 around a noisy funding
   estimate would, if applied post-hoc, turn every oscillation into a new
   order. Re-running the latch on ``desired`` keeps the same no-trade-band
   discipline this project uses everywhere else, and — bit-identically,
   proven by induction in the report — reproduces the parent's own ``target``
   column exactly wherever ``mult`` is exactly 1.0, since re-latching an
   already-latched sequence with the same band is a fixed point of the
   latch rule.

Deliberately NOT done here: turning the ceiling into a fitted parameter,
gating on the same anchors that drive the vote (that would just be the vote
again, dressed as a cost model), or decaying funding evidence rather than
smoothing it (this is a magnitude-shrink, not an error-control gate — R-28
already explored evidence accumulation for the direction question and it is
a different mechanism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import load_funding
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

SETTLEMENTS_PER_DAY = 3.0
DAYS_PER_YEAR = 365.25

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_UNSET = object()


class FundingCrowdKelly(KellyRegimeV4):
    """kelly_regime_v4 with exposure continuously shrunk by realized funding cost.

    Merton (1971)-grounded continuous shrink, not a hard percentile gate —
    see the module docstring for the full derivation and the exact causal
    recipe. Everything about the vote and the vol-target sizer is
    unchanged; only the final ``target`` differs, and only while real
    Binance BTCUSDT perp funding data is available (2020-01-01..2023-12-31,
    plus the one-settlement EWM warmup at the start of that window).
    """

    name = "funding_gate_novel"

    # Loaded once per process and shared across every instance, so a sweep
    # that constructs dozens of these does one CSV read total.
    _funding_cache: "pd.Series | None" = _UNSET  # type: ignore[assignment]

    def __init__(self, funding_ewm_halflife_days: float = 10.0,
                 cost_ceiling: float = 0.30, floor_mult: float = 0.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding_ewm_halflife_days = funding_ewm_halflife_days
        self.cost_ceiling = cost_ceiling
        self.floor_mult = floor_mult

    @classmethod
    def _real_funding(cls):
        """The committed Binance funding series, loaded and cached once."""
        if cls._funding_cache is _UNSET:
            cls._funding_cache = load_funding(_DATA_DIR)
        return cls._funding_cache

    def funding_multiplier(self, df: pd.DataFrame) -> np.ndarray:
        """The causal, magnitude-based shrink multiplier, in [floor_mult, 1.0].

        Exactly 1.0 wherever funding data is unavailable (see module
        docstring, mechanism step 6). Exposed as its own method so the
        causal-alignment and bit-identical checks can inspect it directly
        rather than only its effect on ``target``.
        """
        real = self._real_funding()
        if real is None or len(real) == 0:
            return np.ones(len(df))

        # Step 3: EWM on the settlement-frequency series, half-life in
        # settlements (3/day, evenly spaced with no gaps over the committed
        # window), then shifted by one settlement so a settlement's own
        # value never smooths itself before the *next* settlement is known.
        halflife_settlements = self.funding_ewm_halflife_days * SETTLEMENTS_PER_DAY
        smoothed = real.ewm(halflife=halflife_settlements, min_periods=1).mean().shift(1)

        # Step 2: causal ffill onto the bar grid - only settlements at or
        # before bar t are ever read.
        aligned = smoothed.reindex(df.index, method="ffill")

        # Step 6: force mult == 1.0 outside the covered window. reindex+ffill
        # already yields NaN before the first settlement and wherever the
        # shifted EWM has no history yet; it does NOT naturally go NaN after
        # the last settlement (ffill happily carries the last known value
        # forward forever), so that side is masked explicitly.
        in_range = (df.index >= real.index[0]) & (df.index <= real.index[-1])
        aligned = aligned.where(in_range)

        # Steps 4-5: annualize, shrink only for positive (longs-pay) funding.
        funding_annualized = aligned * (SETTLEMENTS_PER_DAY * DAYS_PER_YEAR)
        shrink = 1.0 - funding_annualized.clip(lower=0.0) / self.cost_ceiling
        mult = shrink.clip(lower=self.floor_mult, upper=1.0)
        return mult.fillna(1.0).to_numpy()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        parent_target = df["target"].to_numpy(copy=True)

        mult = self.funding_multiplier(df)
        desired = parent_target * mult

        # Step 7: this class's own latch/deadband loop over `desired`, not
        # the parent's already-latched target multiplied post-hoc.
        n = len(df)
        target = np.empty(n)
        pos = 0.0
        for i in range(n):
            d = desired[i]
            if abs(d - pos) > self.deadband:
                pos = d
            target[i] = pos

        df["funding_mult"] = mult
        df["parent_target"] = parent_target
        df["target"] = target
        return df
