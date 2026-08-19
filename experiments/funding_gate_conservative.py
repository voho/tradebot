"""kelly_regime_v4, gated flat whenever funding sits in its own top decile.

Backlog **B-05** ("Funding as a gate on the existing strategy — stand flat
in the top decile"), the conservative variant. A second, independent
branch (``experiments/funding_crowding_novel*.py``, not read or touched
here) explores a continuous/analytically-derived treatment of the same
idea; this file is deliberately the minimal, discrete one.

Mechanism, stated precisely: whenever the current bar's causal funding
rate sits in the top decile of its own TRAILING distribution (a rolling
percentile computed only from funding observed so far — never a
whole-series quantile, which would be lookahead), force the target
exposure to exactly 0.0 for that bar, overriding whatever
``kelly_regime_v4``'s vote + vol-target sizer would otherwise produce.
Everywhere else this strategy is byte-for-byte ``kelly_regime_v4``.

Why this should make money: R-14 (``docs/LEDGER.md``) measured funding as
a large, adversely-timed cost on ``kelly_regime_v4`` — it runs +20%/yr
while the strategy holds vs +2.8%/yr while flat, because the crowding the
strategy's own regime vote detects is exactly what sets the funding rate.
R-16 found the genuine (if noisy) forward signal in the same data: high
funding predicts low forward returns, i.e. a richly-positive rate is a
*directly observed* crowding measurement, not the price-inferred proxy
the anchor vote uses. Cardaliaguet & Lehalle's (2018, Math. Fin. Econ.)
mean-field view is what both this gate and the base regime vote share:
positive expected drift is a precondition for a positive Kelly fraction,
and holds only while the crowd is still accumulating. The vote reads
accumulation off price; funding reads the price the crowd is currently
paying to stay long — a second, independent instrument pointed at the
same latent quantity, and one that (per R-16) goes stale in the *opposite*
direction from price momentum, so it should not simply duplicate the vote.
The base strategy already dodges some of this cost by going flat in bear
regimes (R-14's ``timing`` study); this gate targets what is left: bars
where the vote says "hold" but funding says "the crowd you'd be joining is
already paying up for it".

This is the CONSERVATIVE treatment of B-05: a hard decile threshold and a
binary override, not a continuous down-weighting. It changes nothing about
sizing, the vote, the deadband or the vol target — the gate can only ever
turn an otherwise-long bar flat, never add exposure. Not registered:
``experiments/`` per ROUTINE.md step 5, until/unless promoted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


class FundingGateConservative(KellyRegimeV4):
    """kelly_regime_v4, forced flat whenever funding is in its own rolling top decile.

    See module docstring for the mechanism and the citation. ``prepare()``
    calls ``KellyRegimeV4.prepare`` first (unchanged vote + vol-target
    sizing), then overrides ``target`` to 0.0 on any bar whose causal
    funding rate (``df["funding"]``, expected to already be merged in by
    the caller via ``experiments.funding_signal.causal_funding_column``)
    ranks at or above ``threshold`` within its own trailing
    ``lookback_days``-day window. The rank is computed with
    ``pandas.Series.rolling(...).rank(pct=True)``, which by construction
    only ever looks backward from each row — verified in
    ``run_funding_gate_conservative.py causality``.
    """

    name = "_funding_gate_conservative"  # unregistered experiment, not a table row

    def __init__(self, lookback_days: int = 90, threshold: float = 0.90, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback_days = lookback_days
        self.threshold = threshold
        # Give the rolling percentile a full lookback window of history
        # before trading starts, on top of whatever the base sizer needs.
        gate_warmup = int(lookback_days * BARS_PER_DAY) + 10
        self.warmup = max(self.warmup, gate_warmup)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # kelly_regime_v4's target column, unmodified
        if "funding" not in df.columns:
            raise ValueError(
                "FundingGateConservative requires a 'funding' column - merge it "
                "in first with experiments.funding_signal.causal_funding_column"
            )

        lookback_bars = int(self.lookback_days * BARS_PER_DAY)
        # Require at least a quarter of the window before trusting the rank;
        # earlier bars (partial window) simply don't gate rather than gating
        # on a handful of points, which is not "no lookahead" so much as
        # "not enough evidence yet" - a NaN percentile never satisfies the
        # >= threshold test below.
        min_periods = max(BARS_PER_DAY, lookback_bars // 4)
        pct = df["funding"].rolling(lookback_bars, min_periods=min_periods).rank(pct=True)

        gate = (pct >= self.threshold).fillna(False).to_numpy()
        target = df["target"].to_numpy().copy()
        target[gate] = 0.0

        df["funding_pct"] = pct
        df["funding_gate"] = gate
        df["target"] = target
        return df
