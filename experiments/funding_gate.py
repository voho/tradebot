"""Funding-rate flat-gate on kelly_regime_v4 (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
R-14 (``docs/LEDGER.md``) found that ``kelly_regime_v4``'s own crowding
gate and the perpetual funding rate are detecting the same thing:
funding runs **+20%/yr while the strategy holds** vs +2.8%/yr while flat,
because the crowding the regime vote reacts to is exactly what sets the
funding rate. R-16 found high funding predicts *negative* forward
returns over the following ~14 days (Q1-Q5 spread +3.57pp), independent
of trailing return (correlation 0.39, so not a momentum proxy).

This variant is deliberately the conservative one: it adds a **second,
independent gate** on top of v4's own regime vote, rather than replacing
or continuously re-weighting anything. When funding sits in its own
trailing top decile the position is forced flat; it re-opens (full v4
behaviour) only once funding relaxes back below a lower release
threshold. Everything else about v4 (the anchor vote, conditional vol
targeting, v4's own internal deadband) is untouched -- the gate is a
multiplier applied to v4's *target*, with its own outer deadband applied
to the *final* position, exactly the way ``kelly_regime_v3`` layers its
own state machine over ``kelly_regime``'s vote rather than editing it.

Mechanism, precisely
---------------------
1. ``df["target"]`` = ``KellyRegimeV4.prepare(df)["target"]`` -- the base
   v4 position, unmodified.
2. A causal funding gate, computed independently:

   - align the 8-hourly funding series onto the OHLCV index with
     ``ffill`` (a settlement's rate is known from the moment it settles
     until the next one -- forward-filling is the causal direction);
   - compute a **trailing** decile threshold using only *past*
     settlements: ``funding.rolling(window, min_periods=window//3)
     .quantile(pct).shift(1)`` -- the mandatory ``.shift(1)`` means the
     threshold applied to settlement *t* is built only from settlements
     strictly before *t*, never including *t*'s own reading;
   - a latched hysteresis state machine (mirroring
     ``kelly_regime_v3``'s steady/breakout latch): gate turns ON (force
     flat, multiplier 0) when the current known funding rate exceeds the
     upper (e.g. 90th percentile) threshold, and OFF (multiplier 1) only
     once funding drops back below the lower (e.g. 75th percentile)
     release threshold. In between, it holds -- no chattering on every
     8h settlement.

3. **Coverage guard.** The committed funding file
   (``data/btcusdt_perp_funding_8h.csv.gz``) covers 2020-01-01 through
   2023-12-31 only. ``pandas.Series.reindex(..., method="ffill")`` does
   not know that -- left unguarded it would silently repeat 2023's last
   settled rate forward through the entire 2024-2026 holdout (more than
   80% of the out-of-sample period), fabricating a signal where none
   exists. Every bar with ``index > funding.index[-1]`` or
   ``index < funding.index[0]`` is therefore explicitly forced to
   multiplier 1.0 (pass-through, identical to unmodified v4), regardless
   of what the latch state happened to be when coverage ended. This
   experiment is consequently *silent* (gate always off) for essentially
   the whole holdout period; see the report for what that implies about
   trusting any holdout number produced with this code.

4. Final target = ``v4_target * multiplier``, then the strategy's own
   outer deadband (``self.deadband``, inherited from ``KellyRegime``,
   default 0.10) is applied against the *previous final position* -- a
   fresh position loop, exactly like ``kelly_regime_v3.prepare()`` runs
   its own loop rather than reusing the one inside ``super().prepare()``.
   v4's internal deadband (already applied once, inside
   ``super().prepare()``) is not re-applied a second time to the same
   pre-gate value; only the final, gated series gets a deadband, and it
   is this class's own.

Causal-computation checklist (see the report for the full statement):
the only two ``.rolling``/``.shift`` calls in this file
(``threshold_hi``/``threshold_lo``) are both trailing-window quantiles
immediately followed by ``.shift(1)``. Nothing here is an
expanding-window or full-series statistic (no ``.mean()``, ``.std()``,
``.quantile()`` computed over the whole series and broadcast backward).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

SETTLEMENTS_PER_DAY = 3  # Binance BTCUSDT perp funding settles every 8h


class KellyRegimeFundingGate(KellyRegimeV4):
    """kelly_regime_v4, forced flat while funding sits in its own trailing top decile.

    Not ``@register``-ed -- see module docstring.
    """

    name = "kelly_regime_funding_gate"

    def __init__(self, funding: pd.Series | None = None,
                 window_days: int = 90, upper_pct: float = 0.90,
                 lower_pct: float = 0.75, **kwargs) -> None:
        super().__init__(**kwargs)
        self.funding = funding
        self.window_days = window_days
        self.upper_pct = upper_pct
        self.lower_pct = lower_pct

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4 target, unmodified, in df["target"]
        v4_target = df["target"].to_numpy()

        multiplier = self._funding_multiplier(df.index)

        # This class's OWN deadband loop against the FINAL position --
        # not a second pass over v4's already-deadbanded value; v4's
        # internal deadband ran once, inside super().prepare(), above.
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = v4_target[i] * multiplier[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["funding_gate_multiplier"] = multiplier  # diagnostic only
        return df

    def _funding_multiplier(self, index: pd.DatetimeIndex) -> np.ndarray:
        """1.0 = v4 pass-through, 0.0 = forced flat. Causal; see module docstring."""
        n = len(index)
        multiplier = np.ones(n)  # default: no data -> no gate -> unchanged v4
        if self.funding is None or len(self.funding) == 0:
            return multiplier

        funding = self.funding.sort_index()
        window = max(int(self.window_days * SETTLEMENTS_PER_DAY), 3)
        min_periods = max(window // 3, 1)

        # Trailing quantile thresholds, shifted by one settlement: the
        # threshold that applies AT settlement t is built only from
        # settlements strictly BEFORE t.
        threshold_hi = (funding.rolling(window, min_periods=min_periods)
                         .quantile(self.upper_pct).shift(1))
        threshold_lo = (funding.rolling(window, min_periods=min_periods)
                         .quantile(self.lower_pct).shift(1))

        # Forward-fill (causal: a settlement's rate/threshold is known
        # from the instant it settles until the next one) onto the
        # 5-minute OHLCV index.
        rate = funding.reindex(index, method="ffill").to_numpy()
        hi = threshold_hi.reindex(index, method="ffill").to_numpy()
        lo = threshold_lo.reindex(index, method="ffill").to_numpy()

        # Coverage guard: the committed funding file is 2020-01-01 ->
        # 2023-12-31 only. Bars outside that range must NOT inherit a
        # forward-filled 2023 rate -- force them to "no info" explicitly.
        covered = np.asarray((index >= funding.index[0]) &
                             (index <= funding.index[-1]))

        state = 0  # 0 = open (multiplier 1), 1 = gated flat (multiplier 0)
        for i in range(n):
            if not covered[i]:
                # Outside funding-data coverage: gate never fires here,
                # regardless of what the latch state was when coverage
                # ended. Multiplier is forced to 1 (unchanged v4).
                multiplier[i] = 1.0
                continue
            r, h, lw = rate[i], hi[i], lo[i]
            if np.isfinite(r) and np.isfinite(h) and np.isfinite(lw):
                if state == 0 and r > h:
                    state = 1
                elif state == 1 and r < lw:
                    state = 0
            # else: inside coverage but threshold not warmed up yet
            # (min_periods not met) -- treat as no info, latch unchanged.
            multiplier[i] = 0.0 if state == 1 else 1.0

        return multiplier
