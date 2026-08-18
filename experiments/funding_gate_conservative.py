"""kelly_regime_v4 + a hard top-decile funding gate (backlog B-05, conservative variant).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered by ``tradebot.registry`` and cannot corrupt the
comparison table or trip CI, per ``docs/ROUTINE.md`` step 5 (see
``experiments/matched_risk.py``'s module docstring for the same reasoning).

The idea, literally
--------------------
R-14 measured that ``kelly_regime_v4`` pays real funding at roughly
+20%/yr while it holds a long, vs +2.8%/yr while flat, and that this cost
is invisible to the strategy's own decisions -- it is only charged after
the fact by the backtest engine when ``funding=...`` is passed in. B-05
asks for the minimal fix: stand flat whenever funding is unusually
expensive to be long into. "Unusually expensive" is operationalised as
the top decile of the funding rate's own trailing history, exited on a
lower threshold so the gate does not chatter around the 90th percentile
-- the identical hysteresis-latch pattern ``kelly_regime_v3.prepare()``
already uses for its volatility-breakout state machine (``state``,
``high_in``/``high_out``). No continuous adjustment, no new sizing
theory: a hard override on top of the existing mechanism.

Everything else -- the 20/40/80-day latched vote, the v3 conditional
vol-targeting sizer, the 2x cap, the 10% deadband -- is unchanged,
inherited unmodified from ``KellyRegimeV4``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


class FundingGateConservative(KellyRegimeV4):
    """v4, unchanged, except forced flat in the top decile of funding cost.

    ``funding`` is the real Binance BTCUSDT perp funding series (8-hourly
    settlements), or ``None`` to disable the gate entirely (in which case
    this class behaves exactly like ``KellyRegimeV4``). The gate is a
    causal, rolling percentile-rank of the CURRENT funding rate against
    its own trailing ``pct_window_days`` of history: >= ``decile_in``
    latches the gate closed (target forced to 0); it stays closed until
    the percentile drops to <= ``decile_out`` (hysteresis, exactly like
    v3's vol state machine). Outside the funding data's coverage (pre
    2020, or if the settlement history is too short: < 20 observed
    settlements), the percentile is NaN and the gate never engages --
    the strategy reduces to plain v4 wherever funding is unobserved.
    """

    name = "funding_gate_conservative"

    def __init__(self, funding: pd.Series | None = None, decile_in: float = 0.90,
                 decile_out: float = 0.75, pct_window_days: int = 180,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if decile_out >= decile_in:
            raise ValueError(
                f"decile_out ({decile_out}) must be < decile_in ({decile_in}) "
                "or the latch never releases")
        self.funding = funding
        self.decile_in = decile_in
        self.decile_out = decile_out
        self.pct_window_days = pct_window_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # existing df["target"] from v4/v3, deadband applied
        base_target = df["target"].to_numpy().copy()

        if self.funding is not None and len(self.funding) > 0:
            # 1. Causal rolling percentile-rank, computed on the funding
            #    series' OWN (8-hourly) index -- NOT on the 5m df, which
            #    would be ~1M rows and pointlessly slow/noisy. A
            #    settlement's percentile only ever looks at settlements
            #    at or before it (pandas' rolling window is trailing by
            #    construction: window "180D" ending at label i includes
            #    only index values in (i - 180D, i]).
            window = f"{self.pct_window_days}D"
            pct = self.funding.rolling(window, min_periods=20).apply(
                lambda x: (x <= x[-1]).mean(), raw=True)
            # 2. Reindex onto the 5m bar grid: each bar sees the last
            #    KNOWN settlement at or before it (ffill is causal here --
            #    it never pulls a future settlement backward).
            pct_bars = pct.reindex(df.index, method="ffill")
            # 3. Extra one-bar shift so a bar cannot act on information
            #    dated to its own exact timestamp (matches this repo's
            #    convention elsewhere, e.g. kelly_regime.py's vol
            #    .shift(1)). This costs at most one 5-minute bar of
            #    staleness relative to step 2, which is already
            #    conservative (ffill), so it is a deliberate belt-and-
            #    braces margin, not a correction of a real leak.
            pct_bars = pct_bars.shift(1)
        else:
            pct_bars = pd.Series(np.nan, index=df.index)
        pct_arr = pct_bars.to_numpy()

        # 4. Hysteresis gate, bar by bar, same pattern as kelly_regime_v3's
        #    volatility state machine (state / high_in / high_out).
        #
        #    Forcing target[i] = 0.0 directly, rather than routing the
        #    override through the base class's deadband check, is
        #    intentional and deliberately bypasses that deadband: the
        #    deadband exists to avoid churning size in and out on noise
        #    while the underlying signal is continuous, but a funding-cost
        #    latch closing is a discrete risk decision -- when gated, the
        #    position should go to exactly flat immediately, not drift
        #    there over several bars because the move didn't clear 10% of
        #    notional. The deadband still governs ordinary v4/v3 sizing
        #    both before the gate closes and after it reopens (line above:
        #    `base_target` is v4's own deadbanded output, untouched).
        n = len(df)
        target = base_target.copy()
        gated = False
        for i in range(n):
            p = pct_arr[i]
            if np.isfinite(p):
                if not gated and p >= self.decile_in:
                    gated = True
                elif gated and p <= self.decile_out:
                    gated = False
            # if p is NaN (funding unknown, e.g. outside 2020-2023, or
            # fewer than 20 settlements observed), never gate -- reduces
            # to plain v4/v3 wherever funding cost cannot be measured.
            if gated:
                target[i] = 0.0
        df["target"] = target
        df["funding_pct"] = pct_arr
        return df
