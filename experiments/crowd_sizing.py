"""Crowd-sizing haircut on kelly_regime_v4 (backlog item drawn from L-12's own lesson).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Do not edit
``harsanyi_crowd.py`` or any ``kelly_regime*`` file to build this.

The question
------------
L-12 (``harsanyi_crowd``) recorded, verbatim: *"The crowding intuition
was right - it is what ``kelly_regime`` later exploited - but as a
DIRECTION signal rather than a SIZING input it loses."* Nobody has yet
taken that sentence literally: graft ``harsanyi_crowd``'s mean-field
crowding measure (Cardaliaguet & Lehalle 2018, Math. Fin. Econ.: an aged
trend whose volume efficiency is decaying - more volume buying less
price progress - carries a strategic crowding cost) onto
``kelly_regime_v4``'s exposure as a continuous MULTIPLICATIVE sizing
haircut, instead of using it to pick direction. This attacks the SIZE
constraint - the one axis this project has repeatedly found works
("every strategy that decides how much to hold makes money").

Caveat carried in on purpose: Lee (2025, arXiv:2512.11913, preprint later
withdrawn by its author for insufficient empirical validation - suggestive
only) reported crowded MOMENTUM factors show markedly lower crash risk
than crowded reversal factors. ``kelly_regime_v4`` is a trend/momentum
strategy, so there is a real prior reason this haircut might not earn its
keep even if the mechanism is sound; see the falsification design below.

Mechanism, in one sentence
---------------------------
Take ``kelly_regime_v4``'s own base exposure unmodified, and multiply it
by ``(1 - lam_crowd * crowd)`` where ``crowd`` is ``harsanyi_crowd``'s
exact sigmoid-of-trend-age-times-decay-flag crowding score, but with
"trend" now keyed to v4's own 3-anchor vote sign (majority of the
20/40/80-day anchors bullish vs bearish) rather than harsanyi's separate
Bayesian belief-margin sign - so the crowding measure is keyed to the
same regime the sizer is already gated on, not a second, independent
read of the market.

All of ``progress``, ``vol_ratio``, ``eff``, the decay-streak flag and
the sigmoid form are copied verbatim from ``harsanyi_crowd.py``'s
``prepare()`` - same causal primitives (rolling/ewm/shift only), not
re-derived. ``lam_crowd`` and ``age_scale`` are the free parameters this
experiment sweeps; ``harsanyi_crowd``'s defaults (0.7, 150.0) are used as
one grid point, not as a prior belief about what will win here - v4's
vote is far stickier (latched for weeks) than harsanyi's per-bar Bayesian
margin, so an age_scale tuned to the latter (150 bars = 12.5 hours) is
very likely too fast for this signal, hence the wide sweep in
``run_crowd_sizing.py``.

Citations: Cardaliaguet, P. & Lehalle, C.-A. (2018), "Mean field game of
controls and an application to trade crowding", Math. Fin. Econ. 12(3).
Harsanyi, J. (1967-68), "Games with incomplete information played by
'Bayesian' players", Management Science 14. This repo: L-12
(``docs/LEDGER.md``), whose recorded lesson is the direct motivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

# harsanyi_crowd's volume-efficiency EMA smoothing constant, copied
# verbatim (alpha = 2/49, i.e. an EMA span of 48 bars).
_ALPHA_EFF = 2.0 / 49.0
# harsanyi_crowd's decay-streak gate and its "not yet declining" haircut.
_DECL_STREAK = 12
_DECL_LOW_MULT = 0.3


class CrowdSizedKellyV4(KellyRegimeV4):
    """kelly_regime_v4 with harsanyi_crowd's mean-field crowding cost applied as a sizing haircut, not a direction signal."""

    def __init__(self, lam_crowd: float = 0.7, age_scale: float = 150.0,
                 haircut_deadband: float | None = None,
                 horizons: tuple[int, ...] = (20, 40, 80), **kwargs) -> None:
        super().__init__(horizons=horizons, **kwargs)
        self.lam_crowd = lam_crowd
        self.age_scale = age_scale
        # Outer hysteresis on the haircut-adjusted target, mirroring the
        # pattern harsanyi_crowd.py itself uses on its own `pos` update
        # (deadband on the *smoothed* signal, inside prepare(), so on_bar
        # can stay the simple "order whenever target changed" pattern).
        # Defaults to the inherited kelly_regime deadband (0.10) rather
        # than a new free parameter.
        self.haircut_deadband = (self.deadband if haircut_deadband is None
                                  else haircut_deadband)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # v4's own vote + vol-target exposure, untouched. This is the
        # thing being sized down, not re-derived.
        df = super().prepare(df)
        base_target = df["target"].to_numpy(dtype=float)

        close = df["close"]

        # --- harsanyi_crowd.py primitives, copied verbatim -----------------
        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 48, min_periods=48).mean()

        progress = ((close - close.shift(24)).abs() / atr).to_numpy()
        vol_ratio = (df["volume"].rolling(24).sum()
                     / df["volume"].rolling(288).sum()).to_numpy()

        # --- v4's own 3-anchor vote, recomputed to get its SIGN -----------
        # Identical formula to kelly_regime.py / kelly_regime_v3.py's
        # `frac`: latched hysteresis vote per anchor, averaged. Not a
        # second independent read of the market - the same regime v4's
        # sizer is already gated on. frac in [0, 1]; 0.5 is the tie point
        # between "majority of anchors bullish" and "majority bearish".
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

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        eff_ema = np.nan
        decl = 0
        trend_age = 0
        prev_sign = 0
        for i in range(n):
            # Volume-efficiency decay streak - exact harsanyi_crowd logic.
            if np.isfinite(progress[i]) and np.isfinite(vol_ratio[i]):
                eff = progress[i] / max(vol_ratio[i], 1e-9)
                if np.isfinite(eff_ema):
                    decl = decl + 1 if eff < eff_ema else 0
                    eff_ema += _ALPHA_EFF * (eff - eff_ema)
                else:
                    eff_ema = eff

            # Trend age keyed to v4's own vote sign, not harsanyi's belief
            # margin: sign of (frac - 0.5).
            sign = 1 if frac[i] > 0.5 else (-1 if frac[i] < 0.5 else 0)
            trend_age = trend_age + 1 if sign == prev_sign and sign != 0 else 0
            prev_sign = sign

            crowd = (1.0 / (1.0 + np.exp(-(trend_age / self.age_scale - 1.0)))) \
                * (1.0 if decl >= _DECL_STREAK else _DECL_LOW_MULT)

            raw = base_target[i] * (1.0 - self.lam_crowd * crowd)
            if abs(raw - pos) >= self.haircut_deadband:
                pos = raw
            target[i] = pos

        df["target"] = target
        return df

    # on_bar is inherited unchanged from KellyRegime (order_notional on the
    # smoothed `target` column) - same order-target pattern as the parent.
