"""Funding as a gate on kelly_regime_v4: momentum-conditioned, continuous fade (B-05, novel).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The conservative variant in this same backlog item
(``experiments/funding_gate_conservative.py``) reads R-16's plain
funding-decile sort literally: rich funding predicts weak forward
returns, so stand flat in the top decile. That reading collides with
R-14: funding is richest in exactly the bullish regimes
``kelly_regime_v4``'s own vote wants to be long in (mean funding while
holding ~= +20%/yr vs ~= +2.8%/yr while flat), so a blind funding-decile
gate risks cutting exposure during precisely the strong bull runs that
make a trend follower its money.

R-16's own momentum-conditioned sort (funding tercile x trailing-7-day
return tercile, mean 7-day forward spot return) shows the naive reading
is incomplete:

    |               | past ret LOW | past ret MID | past ret HIGH |
    |---------------|-------------:|-------------:|--------------:|
    | funding LOW   |       +2.83% |       +1.74% |        +2.16% |
    | funding MID   |       +0.55% |       +1.36% |        +3.24% |
    | funding HIGH  |       -1.68% |       -1.54% |        +1.22% |

Rich funding only predicts weak forward returns when trailing momentum
is ALSO weak (-1.5% to -1.7%, bottom-left/mid). When rich funding
coincides with strong trailing momentum, forward returns stay positive
(+1.22%), nearly as good as the low-funding/high-momentum cell. The
naive univariate gate cannot see this distinction; this file's whole
reason to exist is to act on it.

Mechanism
---------
Reduce ``kelly_regime_v4``'s target continuously, in proportion to how
deep into "rich funding" territory the (smoothed, causal) funding rate
is, but only to the extent trailing price momentum is ALSO weak. When
momentum is strong, funding richness is not held against the position
at all - the interaction is multiplicative on a [0, 1] "fade" factor,
not a second independent gate.

1. Lagged, causal funding: ``funding.reindex(df.index, method="ffill")``
   - the most recently SETTLED rate at or before each bar. Bars before
   the first settlement (2020-01-01 03:00 UTC) get 0.0.
2. Smoothed funding: an EWM smooth (``span=funding_ewm_span``,
   settlements) of the RAW 8-hourly series, computed BEFORE reindexing
   to bar frequency - smoothing after reindexing would just weight
   repeated ffilled values, not new information. Then ffilled onto bars.
3. Funding percentile: a CAUSAL rolling percentile rank
   (``.rolling(window).rank(pct=True)``, window = 3 * funding_lookback_days
   settlements) of the smoothed 8-hourly series against its own trailing
   history, computed at 8-hourly frequency then ffilled onto bars.
4. Momentum percentile: a trailing ``momentum_days``-day return
   (``close.pct_change(momentum_days*288)``, causal), then a CAUSAL
   rolling percentile rank of THAT series (window =
   momentum_lookback_days*288 bars) against its own trailing history.
5. Interaction discount, multiplicative on kelly_regime_v4's target:

     funding_excess  = clip((funding_pct - funding_threshold)
                             / (1 - funding_threshold), 0, 1)
     momentum_shield = clip(momentum_pct / momentum_threshold, 0, 1)
     fade            = funding_excess * (1 - momentum_shield)
     discount        = 1 - fade_strength * fade
     target          = v4_target * discount

``discount`` is continuous and never fully zeroes exposure while
momentum is strong, regardless of how rich funding is - a genuine
departure from a hard decile on/off gate, chosen because the table above
supports an interaction, not a second univariate threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288


class FundingMomentumGate(Strategy):
    """kelly_regime_v4, continuously faded for rich funding only when momentum is weak."""

    name = "_experiment_funding_momentum_gate"  # deliberately unregistered

    # v4's own warmup is 80*288+10 (80-day anchor). This variant also needs
    # momentum_lookback_days of price history for its OWN rolling percentile
    # rank to be filled, and momentum_days more before that for the trailing
    # return itself to be defined. Use whichever is larger, plus a small
    # margin so the causal rolling ops are never cold at bar 0 of the
    # measured period.
    warmup = KellyRegimeV4.warmup

    def __init__(self, funding: pd.Series, funding_threshold: float = 0.90,
                 momentum_threshold: float = 0.50, fade_strength: float = 1.0,
                 funding_ewm_span: int = 3, funding_lookback_days: int = 180,
                 momentum_days: int = 7, momentum_lookback_days: int = 180,
                 **v4_kwargs) -> None:
        if not (0.0 < funding_threshold < 1.0):
            raise ValueError(f"funding_threshold must be in (0,1), got {funding_threshold!r}")
        if not (0.0 < momentum_threshold <= 1.0):
            raise ValueError(f"momentum_threshold must be in (0,1], got {momentum_threshold!r}")
        if not (0.0 <= fade_strength <= 1.0):
            raise ValueError(f"fade_strength must be in [0,1], got {fade_strength!r}")
        self.funding = funding
        self.funding_threshold = funding_threshold
        self.momentum_threshold = momentum_threshold
        self.fade_strength = fade_strength
        self.funding_ewm_span = funding_ewm_span
        self.funding_lookback_days = funding_lookback_days
        self.momentum_days = momentum_days
        self.momentum_lookback_days = momentum_lookback_days
        self._v4 = KellyRegimeV4(**v4_kwargs)

        # Effective warmup: v4's own, or enough bars for the momentum
        # lookback's rolling percentile rank to be filled (momentum_days to
        # define the trailing return, plus momentum_lookback_days more for
        # the percentile rank window over it), whichever is larger, plus a
        # small margin.
        mom_bars = (self.momentum_days + self.momentum_lookback_days) * BARS_PER_DAY
        self.warmup = max(KellyRegimeV4.warmup, mom_bars + BARS_PER_DAY)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._v4.prepare(df)  # sets df["target"] causally, unchanged

        # --- step 1+2: causal, lagged, smoothed funding, computed at
        # 8-hourly settlement frequency BEFORE reindexing onto bars.
        raw_8h = self.funding.sort_index()
        smoothed_8h = raw_8h.ewm(span=self.funding_ewm_span,
                                  min_periods=1).mean()

        # --- step 3: causal rolling percentile rank of the smoothed
        # 8-hourly series against its own trailing history, at settlement
        # frequency (ranking after reindexing to bars would just repeat the
        # same 96 values with no extra information, and be far more
        # expensive).
        window_f = max(int(round(3 * self.funding_lookback_days)), 2)
        fpct_8h = smoothed_8h.rolling(window_f, min_periods=window_f).rank(pct=True)

        funding_lag = self.funding.reindex(df.index, method="ffill").fillna(0.0)
        funding_pct = (fpct_8h.reindex(df.index, method="ffill")
                       .fillna(0.0).to_numpy())

        # --- step 4: trailing N-day return (causal - only looks backward),
        # then its own causal rolling percentile rank against its trailing
        # history, at bar frequency (this signal has no natural coarser
        # frequency the way funding does).
        close = df["close"]
        mom_bars = int(round(self.momentum_days * BARS_PER_DAY))
        trailing_ret = close.pct_change(mom_bars)
        window_m = int(round(self.momentum_lookback_days * BARS_PER_DAY))
        momentum_pct = (trailing_ret.rolling(window_m, min_periods=window_m)
                        .rank(pct=True).fillna(0.0).to_numpy())

        # --- step 5: interaction discount.
        ft, mt, fs = self.funding_threshold, self.momentum_threshold, self.fade_strength
        funding_excess = np.clip((funding_pct - ft) / (1.0 - ft), 0.0, 1.0)
        momentum_shield = np.clip(momentum_pct / mt, 0.0, 1.0)
        fade = funding_excess * (1.0 - momentum_shield)
        discount = 1.0 - fs * fade

        df["funding_lag"] = funding_lag.to_numpy()
        df["funding_percentile"] = funding_pct
        df["momentum_percentile"] = momentum_pct
        df["fade"] = fade
        df["discount"] = discount
        df["target"] = df["target"].to_numpy() * discount
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
