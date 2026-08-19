"""Continuous cost-of-carry-adjusted Kelly exposure (novel funding branch).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per docs/ROUTINE.md step 5 (see ``experiments/matched_risk.py``'s
module docstring for the same convention). This is one of two independent,
parallel branches on the same research question (R-14: ``kelly_regime_v4``
pays real funding at roughly +20%/yr while long, vs +2.8%/yr while flat, and
that cost is not modeled in the strategy's own sizing decisions — only
charged after the fact by the backtest engine). The other branch is a hard
gate; this one is a continuous correction, derived the same way
``kelly_regime_ev.py`` derives its no-trade band — from the growth-optimal
trade-off, not tuned by grid search on returns.

The derivation
---------------
For a bettor with instantaneous return drift ``mu`` and variance
``sigma**2``, expected log-growth from holding fraction ``f`` is
(Kelly 1956; the same quadratic ``kelly_regime_ev.py`` builds its band
from)::

    g(f) ~= f*mu - 0.5*f**2*sigma**2

maximized at ``f* = mu / sigma**2`` — the classical Kelly fraction. This is
also exactly Merton's (1971, J. Economic Theory, "Optimum consumption and
portfolio rules in a continuous-time model") continuous-time optimal
portfolio fraction, ``mu / (gamma * sigma**2)`` at unit relative risk
aversion.

A continuously-running holding cost ``c`` — funding, positive meaning a
long pays it — subtracts ``f*c`` from that growth per unit time (it is a
running drag on the position, not a one-off friction), so the objective
becomes::

    g(f) ~= f*(mu - c) - 0.5*f**2*sigma**2

maximized at::

    f* = (mu - c) / sigma**2 = mu/sigma**2 - c/sigma**2

This project's sizer never estimates ``mu`` directly. It substitutes a
volatility-target proxy — ``scale = target_vol / vol``, gated by the
latched anchor vote as a stand-in for "is mu positive" — for the
``mu/sigma**2`` term. Under that substitution the funding correction is a
subtraction, in units of exposure, of::

    k * funding_drag_annualized / vol**2

applied to the vote-gated target, where ``vol`` is the identical realized
volatility series the base class's own sizer divides by (recomputed here
rather than exposed by the base class, and checked to match its exact
expression) and ``funding_drag_annualized`` is a causal, EWMA-smoothed
estimate of the current annualized funding rate. ``k=1.0`` is the literal
coefficient from the derivation; ``k`` is swept at 0.5x and 2x as a
robustness/plateau check on the strength of the correction, not because
the functional form is in doubt.

The result is clamped to ``[0, max_leverage]``: funding cost only ever
*reduces* a long (this strategy is never short — the anchor vote gates it
to flat, never short, exactly as the base class does), and negative
funding (shorts pay longs, i.e. a *subsidy* to a long) correctly
*increases* the allowed exposure through the same formula, capped as
usual by ``max_leverage``.

Why funding is worth modeling as a first-class carry cost rather than
noise the backtest happens to charge: Schmeling, Schrimpf & Todorov,
"Crypto Carry" (BIS Working Paper No. 1087, 2022; published Management
Science, 2026) document that crypto perpetual funding-driven carry is
large, persistent, and priced — a real, economically meaningful cost/return
variable that has been decaying/crowding since 2024 as more capital
arbitraged it, not a zero-mean nuisance. The transaction-cost lineage this
derivation mirrors is Constantinides (1986, J. Political Economy) and Davis
& Norman (1990, Math. Operations Research), already cited in
``kelly_regime_ev.py`` for the one-off-cost case; this file is the
continuous-running-cost analogue of the same idea.

Deadband: two passes, deliberately
-----------------------------------
``KellyRegimeV3.prepare`` (via ``KellyRegimeV4``) already runs its own 10%
deadband to produce ``base_target``. Subtracting a smoothly-varying
``correction`` from it reopens fine-grained wiggle the first deadband was
built to suppress — the funding drag changes every bar (it is reindexed
off an 8-hourly step function, but interacts with a continuously-updating
``vol**2`` in the denominator). Two options were on the table:

(a) accept the extra turnover — arguably correct, since the entire point
    of this strategy is to react continuously to a continuously-updating
    cost signal, and suppressing that reaction with a wide band defeats
    the purpose; or
(b) re-apply a deadband, of the same width and shape as the base class's
    own state-machine loop, to the corrected target.

This file takes **(b)**. Reasoning: this whole derivation lineage
(``kelly_regime_ev.py``, and Constantinides/Davis-Norman before it) exists
precisely because trading on every infinitesimal change in a growth-optimal
target is not growth-optimal once trading has a cost — that is the point
of a no-trade region, and it applies here exactly as it does to the base
vote. Skipping the second deadband would make this strategy internally
inconsistent with its own cited justification: modeling funding as a cost
worth correcting for, while ignoring that correcting for it also costs
something to execute. The resulting trade count is reported by the driver
script precisely so this choice is checkable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


class FundingGateNovel(KellyRegimeV4):
    """``kelly_regime_v4`` with exposure reduced by a derived cost-of-carry term.

    See the module docstring for the full derivation. Passing
    ``funding=None`` (the default) makes this class behave identically to
    ``kelly_regime_v4`` — the correction is a no-op when funding is
    unknown, never a guess.
    """

    name = "funding_gate_novel"

    def __init__(self, funding: pd.Series | None = None, k: float = 1.0,
                 funding_halflife_days: float = 3.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.k = k
        self.funding_halflife_days = funding_halflife_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # existing vote-gated, deadbanded df["target"]
        base_target = df["target"].to_numpy().copy()

        # Realized vol, recomputed to match kelly_regime.KellyRegime.prepare's
        # exact expression: span=self.vol_span, EWM std of log-returns,
        # annualized by sqrt(BARS_PER_YEAR), shift(1) so bar i's vol uses
        # only returns through bar i-1 (causal: the base class's sizer
        # divides by this exact series, and this file must divide by the
        # same sigma the sizer used, not a lookahead-tainted recomputation).
        close = df["close"]
        r = np.log(close).diff()
        vol_arr = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
                   * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        if self.funding is not None and len(self.funding) > 0:
            # 1. Causal EWMA smoothing on the funding series' OWN (8-hourly)
            #    index -- an ewm only ever looks backward including the
            #    current point, so this is causal by construction. pandas
            #    requires an explicit `times=` to interpret `halflife` as a
            #    wall-clock duration rather than a bar count (the settlement
            #    spacing is 8h, not 1 bar, so a bar-count halflife would be
            #    the wrong unit).
            smoothed = self.funding.ewm(
                halflife=pd.Timedelta(days=self.funding_halflife_days),
                times=self.funding.index,
            ).mean()
            annualized = smoothed * 3 * 365.25  # 3 settlements/day -> per-year
            # 2. Reindex onto the 5m bar grid (ffill = causal: each bar sees
            #    the last known settlement at or before it).
            drag = annualized.reindex(df.index, method="ffill")
            # 3. Extra one-bar shift so no bar acts on information dated to
            #    its own exact timestamp (matches this repo's convention,
            #    e.g. kelly_regime.py's vol .shift(1)).
            drag = drag.shift(1)
        else:
            drag = pd.Series(0.0, index=df.index)  # unknown funding -> no correction
        drag_arr = drag.fillna(0.0).to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            correction = np.where(vol_arr > 0, self.k * drag_arr / (vol_arr ** 2), 0.0)
        correction = np.where(np.isfinite(correction), correction, 0.0)
        raw_adjusted = np.clip(base_target - correction, 0.0, self.max_leverage)

        # Second deadband pass (subtlety #1, option (b) — see module docstring):
        # re-apply the base class's own state-machine loop to the corrected
        # series so this strategy's turnover stays controlled the same way
        # the base vote's turnover is controlled, instead of trading on
        # every bar-to-bar wiggle the funding correction reintroduces.
        n = len(df)
        adjusted = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = raw_adjusted[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            adjusted[i] = pos

        df["target"] = adjusted
        df["funding_drag_annualized"] = drag_arr
        return df
