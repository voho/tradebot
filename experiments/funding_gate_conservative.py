"""R-33 conservative variant: kelly_regime_v4 derated by a funding-decile gate.

R-16 found the top decile of trailing funding-rate percentile predicts
negative forward spot returns (not a momentum proxy - correlation with
trailing return only 0.39). R-14 found kelly_regime_v4 pays funding at
+20%/yr while holding a long versus +2.8%/yr while flat, because the same
crowding the strategy's price-only regime vote detects late is exactly what
sets the funding rate. This variant adds a second, non-price signal on top
of the unchanged v4 regime/sizing mechanism: stand flat on the long side
whenever trailing funding-rate percentile sits in its richest decile.

Not registered (no ``@register``, no import of ``tradebot.registry``): this
is an experiment living entirely under ``experiments/``, per R-33's
pre-registration in docs/LEDGER.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


def gate_multiplier(funding: pd.Series, bar_index: pd.DatetimeIndex,
                     window_days: int, gate_in: float, gate_out: float
                     ) -> tuple[np.ndarray, pd.Series]:
    """The causal 0/1 multiplier, exposed standalone so it can be inspected
    and re-used without re-deriving the mechanism.

    Returns ``(multiplier, pct_on_bars)`` - the per-bar 0/1 array and the
    trailing funding percentile it was computed from (also per-bar, for
    diagnostics). See :class:`FundingDecileGate` for the causality
    argument in full.
    """
    # 1. Shift by one settlement: only an already-settled rate is ever
    #    visible, never the one settling at (or after) the current point.
    settled = funding.shift(1)

    # 2. Causal trailing percentile on the settlement grid: each point
    #    ranked strictly against the window that precedes it (its own
    #    value excluded), then forward-filled onto the 5m bar index.
    #    Before a full window of settled observations exists, NaN - which
    #    step 3 treats as "no information -> gate stays open".
    win = window_days * 3
    pct = settled.rolling(win, min_periods=win).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean(), raw=False)
    pct_on_bars = pct.reindex(bar_index, method="ffill")

    # 3. Hysteresis latch.
    n = len(bar_index)
    mult = np.ones(n)
    shut = False
    pvals = pct_on_bars.to_numpy()
    for i in range(n):
        p = pvals[i]
        if np.isfinite(p):
            if p >= gate_in:
                shut = True
            elif p <= gate_out:
                shut = False
        mult[i] = 0.0 if shut else 1.0
    return mult, pct_on_bars


class FundingDecileGate(KellyRegimeV4):
    """kelly_regime_v4 with a causal funding-rate crowding derate on long exposure.

    Mechanism, in order:

    1. ``funding`` (settlement-indexed, 8-hourly) is shifted by one
       settlement so that at settlement time ``t_k`` only the rate that
       settled at ``t_{k-1}`` (or earlier) is ever visible - a full
       settlement period of safety margin beyond "already happened",
       since bars inside ``[t_k, t_{k+1})`` will use it.
    2. A trailing percentile rank of that shifted series is computed on
       its own settlement grid, over a rolling ``window_days``-day
       history (``window_days * 3`` settlements), each point ranked
       against the window that *precedes* it (current point excluded).
       This is forward-filled onto the 5-minute bar index - each bar
       sees the percentile as of the most recent completed settlement.
    3. A hysteresis latch shuts the gate (multiplier 0) once the
       percentile reaches ``gate_in`` and only reopens it (multiplier 1)
       once the percentile falls to ``gate_out``; NaN or in-between
       holds the previous state, defaulting OPEN.
    4. The multiplier derates only LONG exposure (``target > 0``) - R-16's
       relationship is about being long into rich funding, not about
       shorts, and L-12's lesson is that this project's crowding signal
       loses as a *direction* input but works as a *sizing* one.

    Frozen parameters (window_days=180, gate_in=0.90, gate_out=0.75) were
    fixed in R-33's pre-registration before any data was read and must not
    be quietly retuned for the primary/frozen run.
    """

    name = "funding_decile_gate_conservative"

    def __init__(self, funding: pd.Series | None = None,
                 window_days: int = 180, gate_in: float = 0.90,
                 gate_out: float = 0.75, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.window_days = window_days
        self.gate_in = gate_in
        self.gate_out = gate_out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's "target" column, unchanged mechanism
        if self.funding is None or self.funding.empty:
            return df

        mult, _ = gate_multiplier(self.funding, df.index, self.window_days,
                                  self.gate_in, self.gate_out)

        # Only derate LONG exposure - R-16's relationship is about being
        # long into rich funding, not about shorts.
        target = df["target"].to_numpy()
        gated = np.where(target > 0, target * mult, target)
        df["target"] = gated
        return df
