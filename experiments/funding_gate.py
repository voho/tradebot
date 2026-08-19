"""Funding-rate gate on kelly_regime_v4's exposure (backlog B-05).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The question
------------
R-14 measured that perpetual funding on BTC runs ~+20%/yr while
``kelly_regime_v4`` holds a leveraged long, against ~+2.8%/yr while it is
flat — the cost is adversely timed against exactly this strategy, because
the crowding the funding rate prices *is* the trend the regime vote is
riding. R-16 found that high trailing funding predicts negative 14-day
forward returns (Q1-Q5 spread +3.57pp) unless price is also rising, with
only 0.39 correlation to trailing return — so it is not a momentum proxy
in disguise. Neither row ever wired funding into the strategy as an
actual gate; this file does that, as the single most conservative version
of the idea: modify an already-promoted strategy with one new,
well-motivated input, rather than invent a new mechanism.

Mechanism, in one sentence
---------------------------
Downweight (or zero out) the position kelly_regime_v4 would otherwise
hold whenever the trailing mean funding rate is in the top decile of its
own history to date — crowded-long territory, where R-16 says forward
returns are worst and R-14 says the cost is highest.

The gate, and how it stays causal
----------------------------------
1. ``trailing``: rolling mean of the raw funding series over
   ``funding_lookback_days``, computed at the funding series' own
   8-hourly settlement frequency (not smeared across 5m bars first).
2. ``threshold``: an **expanding** (not full-series, not a fixed rolling
   window either) quantile of ``trailing`` at ``funding_quantile``
   (default the 90th percentile / top decile). At any settlement this
   uses only what has actually settled by then — "its own trailing
   top-decile", where "trailing" describes the threshold too, not just
   the rate. A full-series quantile applied to early rows is exactly the
   lookahead class ROUTINE.md's skeptic protocol calls out (a scaler,
   quantile, mean or std fit over the whole series and applied to early
   rows); an expanding quantile cannot see its own future by
   construction.
3. Both series are reindexed onto the strategy's bar index with a
   forward-fill (a settlement's information holds until the next one
   posts, never before) and then shifted by one bar — the same causality
   margin ``kelly_regime_v3`` applies to its own volatility estimate
   (``vol...shift(1)``) — so a bar's decision cannot see even its own
   settlement in the boundary case where a settlement and a bar close
   land on the same timestamp.
4. Outside the committed funding file's 2020-01 to 2023-12-31 coverage,
   or before enough history has accumulated to form a stable quantile,
   the multiplier is 1.0 (no gating). This is the project's standing
   rule about proxying unavailable data: absent information leaves the
   strategy exactly as it was, never guessed at.

Two gate styles
----------------
``gate_style="hard"``    multiplier snaps from 1.0 to ``gate_floor`` the
                        instant trailing funding crosses the top-decile
                        threshold (the backlog item's literal wording:
                        "stand flat in the top decile").
``gate_style="smooth"``  multiplier ramps linearly from 1.0 at the
                        threshold down to ``gate_floor`` at the trailing
                        distribution's own 99th percentile, so the
                        strategy downweights exposure in proportion to
                        how crowded funding is, rather than flipping a
                        single switch. This is preferred a priori for a
                        SIZE-adjacent COST attack (continuous downweight
                        of exposure, not an on/off flag) and is compared
                        against the hard cutoff on the inner split before
                        either is frozen.

Everything else — the 20/40/80-day latched anchors, the v3 conditional
volatility sizer, fractional Kelly, the 2x cap, the 10% deadband — is
kelly_regime_v4's, unchanged. The gate multiplies the SIZING SCALE before
the deadband decision, not the finished target position, so the
deadband's turnover control governs the gated position the same way it
governs the ungated one, instead of the gate silently doubling turnover
by moving the target after the hysteresis has already decided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
SETTLEMENTS_PER_DAY = 3  # perpetuals settle funding every 8 hours


class FundingGatedKellyV4(Strategy):
    """kelly_regime_v4, downweighted flat when trailing funding is crowded-long (B-05)."""

    name = "funding_gate"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(
        self,
        funding: pd.Series | None = None,
        # --- kelly_regime_v4 / v3, unchanged defaults ---
        horizons: tuple[int, ...] = (20, 40, 80),
        band: float = 0.01,
        target_vol: float = 0.55,
        max_leverage: float = 2.0,
        vol_span: int = 8 * BARS_PER_DAY,
        deadband: float = 0.10,
        anchor_span_days: int = 180,
        high_in: float = 1.70,
        high_out: float = 1.20,
        low_in: float = 0.55,
        low_out: float = 0.85,
        # --- the funding gate ---
        funding_lookback_days: float = 7.0,
        funding_quantile: float = 0.90,
        funding_quantile_min_days: float = 60.0,
        gate_style: str = "hard",
        gate_floor: float = 0.0,
    ) -> None:
        if gate_style not in ("hard", "smooth"):
            raise ValueError(f"gate_style must be 'hard' or 'smooth', got {gate_style!r}")
        if not 0.0 <= gate_floor <= 1.0:
            raise ValueError(f"gate_floor must be in [0, 1], got {gate_floor!r}")
        self.funding = funding
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        self.funding_lookback_days = funding_lookback_days
        self.funding_quantile = funding_quantile
        self.funding_quantile_min_days = funding_quantile_min_days
        self.gate_style = gate_style
        self.gate_floor = gate_floor

    # ------------------------------------------------------------ the gate

    def _funding_multiplier(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Bar-aligned, causal exposure multiplier from trailing funding.

        Returns 1.0 everywhere when no funding series was supplied, or
        wherever the expanding quantile has not yet warmed up — never a
        guess, per the project's standing rule against proxying missing
        data.
        """
        if self.funding is None or len(self.funding) == 0:
            return np.ones(len(index))

        f = self.funding.sort_index()
        win = max(1, int(round(self.funding_lookback_days * SETTLEMENTS_PER_DAY)))
        trailing = f.rolling(win, min_periods=win).mean()

        min_periods = max(2, int(round(self.funding_quantile_min_days
                                        * SETTLEMENTS_PER_DAY)))
        threshold = trailing.expanding(min_periods=min_periods).quantile(
            self.funding_quantile)
        p99 = trailing.expanding(min_periods=min_periods).quantile(0.99)

        bar_trailing = trailing.reindex(index, method="ffill").shift(1)
        bar_threshold = threshold.reindex(index, method="ffill").shift(1)
        bar_p99 = p99.reindex(index, method="ffill").shift(1)

        bt = bar_trailing.to_numpy()
        th = bar_threshold.to_numpy()
        p9 = bar_p99.to_numpy()

        excess = bt - th
        spread = p9 - th
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_over = np.where(spread > 0, excess / spread,
                                  np.where(excess > 0, 1.0, 0.0))
        frac_over = np.clip(frac_over, 0.0, 1.0)

        if self.gate_style == "hard":
            mult = np.where(excess > 0, self.gate_floor, 1.0)
        else:
            mult = 1.0 - frac_over * (1.0 - self.gate_floor)

        warm = np.isfinite(bt) & np.isfinite(th)
        mult = np.where(warm, mult, 1.0)
        return mult

    # --------------------------------------------------------- the strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # kelly_regime_v4's regime vote: three latched anchors on a
        # doubling 20/40/80-day ladder, unchanged.
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

        # v3's conditional volatility sizer, unchanged.
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

        gate = self._funding_multiplier(df.index)

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
            # The gate multiplies the SIZING SCALE, before the deadband
            # decision - so the existing hysteresis governs the gated
            # position too, instead of the gate silently adding its own
            # extra round trips on top of it.
            scale = (full[i] if state != 0 else steady[i]) * gate[i]
            desired = frac[i] * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["funding_gate_mult"] = gate
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
