"""B-05: funding as a gate on kelly_regime_v4 (stand flat / haircut in the funding top decile).

Idea, one sentence: kelly_regime_v4 pays real funding at +20.05%/yr while
it holds a position against +2.78%/yr while flat (ledger row R-14) because
rich funding is the price of standing on the crowded side of the perp -
information the price-only anchor vote cannot see - so multiply v4's own
(price-only) target by a second, independent gate that cuts exposure
whenever the trailing percentile rank of the most recently settled real
funding rate sits at or above a threshold (e.g. its own top decile).

NOT REGISTERED. This module deliberately has no ``@register`` decorator
and is not auto-discovered by the CLI or the comparison table - it is a
step-3 experiment per docs/ROUTINE.md, backlog item B-05. See
``experiments/run_funding_decile_gate.py`` for the driver, and
docs/LEDGER.md rows R-14 / R-16 and backlog B-05 for the motivation and
what has and has not been tried before this.

Data constraint, stated up front: the real Binance BTCUSDT funding series
committed at ``data/btcusdt_perp_funding_8h.csv.gz`` covers only
2020-01-01 -> ~2023-12-31 (4,383 settlements). Outside that span the gate
has no information and defaults OPEN (multiplier 1.0, i.e. defers
entirely to v4's own price-based target) - "no funding data" must never
be read as "funding is rich", which would be exactly the kind of
proxy-from-price mistake that sank camouflage_flow / stealth_trend /
flow_regime (L-14/L-15/L-16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


def trailing_funding_percentile(funding: pd.Series, lookback: int,
                                 min_periods: int) -> pd.Series:
    """Causal percentile rank of each settlement within its own trailing window.

    ``pct[t]`` = the fraction of the last ``lookback`` settlements up to and
    including ``t`` that are ``<= funding[t]``. Only settlements at or
    before ``t`` are ever read - a rolling window by construction cannot
    see a future settlement, and there is no expanding statistic computed
    over the whole series and broadcast backwards (the lookahead class
    ``test_causality_strict.py`` / R-28's by-hand check exists to catch).
    """
    funding = funding.sort_index()

    def _rank_of_last(window: np.ndarray) -> float:
        return float(np.mean(window <= window[-1]))

    return funding.rolling(lookback, min_periods=min_periods).apply(_rank_of_last, raw=True)


def funding_gate_on_bars(funding: pd.Series, bar_index: pd.DatetimeIndex,
                          lookback: int, min_periods: int, threshold: float,
                          haircut: float) -> pd.Series:
    """Step-function gate (1.0 open, ``haircut`` when gated) forward-filled onto bars.

    Causal by construction: ``merge_asof(..., direction="backward")``
    attaches to bar *i* only the last funding settlement whose timestamp is
    ``<=`` bar *i*'s own timestamp - never a later one. Before the first
    settlement, or before ``min_periods`` settlements have accumulated, the
    percentile is undefined and the gate defaults OPEN (1.0).
    """
    pct = trailing_funding_percentile(funding, lookback, min_periods)
    pct_values = pct.to_numpy()
    gated = pct_values >= threshold
    gate_value = np.where(np.isnan(pct_values), 1.0, np.where(gated, haircut, 1.0))

    settle = pd.DataFrame({"gate": gate_value}, index=pct.index)
    settle.index.name = "timestamp"
    bars = pd.DataFrame(index=pd.DatetimeIndex(bar_index, name="timestamp"))

    # merge_asof requires identical datetime64 resolutions on both keys;
    # the OHLCV index and the funding index are not guaranteed to share one.
    left = bars.reset_index()
    right = settle.reset_index()
    unit = left["timestamp"].dtype.unit if hasattr(left["timestamp"].dtype, "unit") else "ns"
    right["timestamp"] = right["timestamp"].dt.as_unit(unit)

    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    out = merged.set_index("timestamp")["gate"].fillna(1.0)
    out.index = bar_index
    return out


class FundingDecileGate(KellyRegimeV4):
    """kelly_regime_v4, gated flat/haircut whenever trailing funding sits in its own top decile.

    Wraps v4's price-only target with a second, independent multiplier:
    ``haircut`` (0.0 = fully flat) whenever the trailing percentile rank of
    the most recently settled real funding rate, computed within its own
    trailing window, is at or above ``threshold``; 1.0 (defer to v4)
    otherwise. Everything else - the 20/40/80-day anchor vote, the
    conditional volatility targeting, the 10% deadband - is v4 unchanged,
    inherited rather than reimplemented.

    NOT registered - experimental, per docs/ROUTINE.md step 3. Requires a
    real funding series at construction time.
    """

    name = "funding_decile_gate_experiment"

    def __init__(self, funding: pd.Series, lookback_settlements: int = 90,
                 threshold: float = 0.90, haircut: float = 0.0,
                 min_periods: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        if funding is None or len(funding) == 0:
            raise ValueError("FundingDecileGate requires a non-empty real funding series")
        self.funding = funding.sort_index()
        self.lookback_settlements = lookback_settlements
        self.threshold = threshold
        self.haircut = haircut
        self.min_periods = min_periods

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = super().prepare(df)
        gate = funding_gate_on_bars(self.funding, prepared.index,
                                     self.lookback_settlements, self.min_periods,
                                     self.threshold, self.haircut)
        prepared["funding_gate"] = gate.to_numpy()
        prepared["target"] = prepared["target"].to_numpy() * gate.to_numpy()
        return prepared
