"""Funding-momentum-conditioned exposure discount on kelly_regime_v4 (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5 mechanics. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
R-14 found that real Binance BTCUSDT funding costs ``kelly_regime_v4``
about 15%/yr on futures, and that the cost lands worst exactly when the
strategy is exposed — the naive fix is a hard cutoff (B-05's sibling
variant, ``experiments/funding_decile_gate.py``, being built in a
parallel branch). This is the *continuous, momentum-conditioned*
alternative.

R-16 (funding as a positioning signal) found two things worth combining:

1. crowded funding predicts negative forward returns, and it is not just
   a repackaging of trailing momentum (correlation with trailing return
   only 0.39, so it is not safe to treat "funding is rich" as "the trend
   is confirmed" and fade both at once — they are different information);
2. but the naive "fade funding" story is *weaker* specifically when price
   is *also* confirming a rally — high funding is bad news for forward
   returns mainly when the trend backing it is not fully confirmed.

Mechanism
---------
``kelly_regime_v4`` already computes ``frac`` — the multi-anchor
(20/40/80-day) latched regime vote, in [0, 1], where 1.0 means every
anchor is latched bullish (a fully confirmed trend). This variant reuses
that same vote as the "is this rally already confirmed" term, rather than
inventing a new momentum signal from price — this project has repeatedly
found that manufacturing signal that is not really there is where new
indicators go to die (see the bottom of the README comparison table).

A rolling (never expanding-from-start, never whole-series) percentile
rank of the causal funding rate, ``fpct``, feeds a smooth ramp
``smoothstep(fpct, lo, hi)`` — 0 below ``lo``, 1 above ``hi``, cubic
Hermite in between, so there is no hard decile cliff to game or overfit
to a boundary. The discount is::

    discount = 1 - weight * smoothstep(fpct, lo, hi) * (1 - frac)

so when ``frac == 1`` (every anchor latched bullish) the discount is
forced to 1.0 — no penalty — regardless of how crowded funding is; the
penalty only bites while the trend is weak or mixed AND funding is
elevated. ``weight`` caps how much exposure the discount can ever remove.
The discount is folded into ``desired`` *before* the deadband check
``kelly_regime``'s own loop already runs, so it rides the existing
latch rather than adding a second, independent one that would create
extra turnover the deadband was supposed to prevent.

Falls back exactly to plain ``kelly_regime_v4`` (``discount == 1``
everywhere) when ``funding`` is ``None``, or on any bar the funding file
does not cover (before 2020-01-01, after 2023-12-31, or before enough
settlements have accumulated to rank against) — missing funding
coverage is never silently treated as "funding is cheap."

Pre-registered falsification (this session): does the momentum
conditioning actually do anything a hard funding-only gate would not —
i.e. does the discount demonstrably bind less often while ``frac == 1``
than while ``frac < 1``? If the conditioning term is not measurably
doing that, the "R-16-motivated conditionality" claim is not supported
and the variant is redundant with a simpler unconditional gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4


def smoothstep(x, lo: float, hi: float):
    """Cubic Hermite ramp: 0 for x<=lo, 1 for x>=hi, smooth (3t^2-2t^3) between.

    NaN in ``x`` (no funding coverage) propagates to NaN in the output;
    callers must handle that explicitly rather than let it silently clip
    to an endpoint.
    """
    x = np.asarray(x, dtype=float)
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    # np.clip on a NaN with finite bounds returns NaN (neither comparison
    # is True), so t is NaN wherever x is NaN and the cubic below stays NaN.
    return t * t * (3.0 - 2.0 * t)


class FundingMomentumTilt(KellyRegimeV4):
    """kelly_regime_v4, exposure discounted by crowded funding UNLESS the trend is already confirmed.

    See module docstring for the full mechanism and citations (R-14, R-16).
    """

    name = "funding_momentum_tilt"

    def __init__(
        self,
        funding: pd.Series | None = None,
        weight: float = 0.5,
        lo: float = 0.70,
        hi: float = 0.95,
        lookback_days: int = 180,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # KellyRegimeV4 defaults horizons=(20,40,80)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0,1], got {weight!r}")
        if not lo < hi:
            raise ValueError(f"lo must be < hi, got lo={lo!r} hi={hi!r}")
        self.funding = funding
        self.weight = weight
        self.lo = lo
        self.hi = hi
        self.lookback_days = lookback_days

    # ------------------------------------------------------------- funding

    def _causal_funding_pct(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Rolling percentile-rank of the causal funding rate, aligned onto bars.

        Two independent causality guards, both applied on the funding
        series itself (never on the OHLCV bars, so truncating the bar
        frame cannot change this):

        1. ``known = funding.shift(1)`` — a bar decision never uses a rate
           that settles exactly at (or after) its own timestamp; the
           *previous* settlement is used instead, so any same-instant
           ambiguity between a bar's timestamp and a settlement's
           timestamp errs conservative rather than risking a peek.
        2. The percentile rank at settlement ``t`` is computed from a
           trailing, TIME-based rolling window ending at ``t`` — never an
           expanding-from-start or whole-series quantile, which is
           exactly the lookahead class ``test_causality_strict.py``
           exists to catch (see run_eprocess.py's causality() comment).

        Bars the funding file does not cover (before 2020-01-01, after
        its last settlement, or too early in the file to fill
        ``lookback_days`` at all) come back NaN; ``prepare()`` turns NaN
        here into ``discount = 1.0``, never into a penalty.
        """
        if self.funding is None or len(self.funding) == 0:
            return np.full(len(index), np.nan)

        f = self.funding.sort_index()
        known = f.shift(1)
        min_obs = max(8, int(round(self.lookback_days * 3 * 0.3)))

        def _rank(w: np.ndarray) -> float:
            return float(np.mean(w <= w[-1]))

        pct = known.rolling(f"{self.lookback_days}D", min_periods=min_obs).apply(
            _rank, raw=True)

        # The bar index and the funding index can carry different datetime64
        # resolutions (e.g. ms vs us) depending on how each file was parsed;
        # merge_asof requires an exact dtype match, so normalize both to ns.
        bar_ts = index.as_unit("ns")
        fund_ts = pct.index.as_unit("ns")
        left = pd.DataFrame({"ts": bar_ts})
        right = pd.DataFrame({"ts": fund_ts, "pct": pct.to_numpy()})
        merged = pd.merge_asof(left, right, on="ts", direction="backward")
        out = merged["pct"].to_numpy(copy=True)
        # No extrapolation past the file's own coverage (asof otherwise
        # carries the last known value forward forever).
        out[bar_ts > fund_ts[-1]] = np.nan
        return out

    # ------------------------------------------------------------- prepare

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # --- byte-identical copy of KellyRegime/V3's vote + vol-regime math,
        #     duplicated (not called via super()) so this stays independent
        #     of the base classes' internal shape and can insert the extra
        #     discount factor before the deadband loop, per design. ---
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
        if self.vote_gamma != 1.0:
            frac = frac ** self.vote_gamma

        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            full = np.minimum(self.target_vol / vol, self.max_leverage)
            steady = np.minimum(self.target_vol / slow, self.max_leverage)
        full = np.where(np.isfinite(full), full, 0.0)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        # --- the new part: momentum-conditioned funding discount ---
        fpct = self._causal_funding_pct(df.index)
        smooth = smoothstep(fpct, self.lo, self.hi)
        discount = 1.0 - self.weight * smooth * (1.0 - frac)
        discount = np.where(np.isfinite(fpct), discount, 1.0)

        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            scale = full[i] if state != 0 else steady[i]
            # The discount is folded in HERE, before the deadband check, so
            # it rides the existing latch instead of adding a second one.
            desired = frac[i] * scale * discount[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["frac"] = frac
        df["discount"] = discount
        df["funding_pct"] = fpct
        return df
