"""Continuous, analytically-derived funding correction on kelly_regime_v4 (backlog B-05, "novel" branch).

This is one of two independent B-05 branches (see ROUTINE.md's "Running
directions in parallel"). The other branch (different, disjoint files,
not read or imported here) implements a conservative discrete decile
gate. This one derives a *continuous* correction to the sizer's existing
target exposure instead of thresholding a quantile — same spirit as
``kelly_regime_ev``'s (L-05/L-06) fee-aware no-trade band: a threshold
that falls out of a growth-rate argument rather than one that is swept
and picked.

**The derivation.** For a fractional-Kelly sizer, expected log-growth as
a function of exposure ``f`` over one period with edge ``mu`` and
variance ``sigma^2`` is the familiar second-order expansion

    g(f) = f*mu - (sigma^2/2)*f^2

maximized at ``f* = mu/sigma^2`` — this is exactly what
``kelly_regime``'s ``target_vol/realized_vol`` (scaled by the regime
vote) already stands in for empirically (Kelly 1956; Breiman 1961;
fractionalized per MacLean, Thorp & Ziemba 2010, and gated by the
mean-field crowding argument of Cardaliaguet & Lehalle 2018 — see
``kelly_regime.py``'s docstring for the full grounding).

He et al. (2024) formalize the perpetual funding rate as the
no-arbitrage market-clearing price of holding one side of a crowded
perpetual position: it is the same crowding cost Cardaliaguet & Lehalle
describe, except *directly observed* every 8 hours rather than inferred
from price. Holding exposure ``f`` for one period therefore carries an
expected drag ``r_t * f``, where ``r_t`` is the (causal, annualized)
funding rate current at that bar. Growth becomes

    g(f) = f*mu - (sigma^2/2)*f^2 - r_t*f

Re-maximizing: ``f*_adjusted = (mu - r_t)/sigma^2 = f*_before - r_t/sigma^2``.

This is the continuously-accruing-cost analogue of Constantinides (1986)
and Davis & Norman (1990), whose transaction-cost no-trade band
(``kelly_regime_ev``'s L-05/L-06) treats a *one-off* friction the same
way this treats a *running* one: both derive a correction to the
growth-optimal target from a cost that is already observed, rather than
tune a knob against realized returns.

**What is and is not a free parameter.** ``r_t`` (the causal, annualized
funding rate) and ``sigma_t^2`` (``kelly_regime_v4``'s own realized-vol
estimate, squared — the same array the incumbent already computes,
reused rather than re-estimated) are both already-observed quantities at
every bar; nothing here is fit to returns. ``funding_scale`` exists only
to check the correction is not a knife-edge at exactly 1.0 (a robustness
check, run at 0.5/1.0/2.0 — never swept for a best value).

**Where this can fail, named before running anything (the pre-registered
falsification test in run_funding_crowding_novel.py):** (a) the
correction term may be too small, relative to the sizer's 0-2x exposure
range, to move any decision — a well-derived mechanism that is
numerically vacuous is a legitimate negative result; (b) re-targeting on
every funding print may add turnover that a 0.40% taker tier eats
whole, the general finding of R-13; (c) the point of the mechanism is
lowering the dollar cost paid in funding, not raising the funding-free
return, so it can be correctly derived and still fail to help — that is
checked directly by comparing dollar funding paid, not just balances.

**Design choices kept deliberately conservative, stated so a skeptic does
not have to reverse-engineer them:** the corrected exposure is clipped to
``[0, max_leverage]`` — the same range the uncorrected sizer already
respects — rather than let a very high funding print push the position
short. Extending the mechanism into short-selling-for-carry is a
different, more aggressive strategy (closer to R-15's funding harvest)
and is out of scope here; this file only ever *reduces* an existing long,
never opens a new position on funding's say-so alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR
from tradebot.strategy import Context, Strategy

# 8-hourly perpetual settlements per year: 3/day * 365.25.
FUNDING_SETTLEMENTS_PER_YEAR = 3 * 365.25


class FundingCrowdingKelly(Strategy):
    """kelly_regime_v4 with target exposure reduced by r_t / sigma_t^2 (causal funding / own realized variance).

    UNREGISTERED experiment (see module docstring for the derivation and
    the falsification plan). Not a subclass of ``KellyRegimeV4`` because
    the correction has to be inserted *inside* ``prepare()``, between the
    vote/vol-state sizing and the deadband — there is no seam in the
    parent class to hook that does not require copying the loop anyway.
    Everything except the correction step is a verbatim copy of
    ``kelly_regime_v3.KellyRegimeV3.prepare`` with V4's default anchors.

    Requires the caller to have merged a causal ``funding`` column onto
    the frame before backtesting (see ``experiments/funding_signal.py``);
    the engine does not wire funding into ``prepare()`` on its own. If no
    ``funding`` column is present, the correction is identically zero and
    this reduces exactly to kelly_regime_v4 — a useful sanity check, not
    the intended mode of use.
    """

    name = "funding_crowding_novel"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 vote_gamma: float = 1.0, anchor_span_days: int = 180,
                 high_in: float = 1.70, high_out: float = 1.20,
                 low_in: float = 0.55, low_out: float = 0.85,
                 funding_scale: float = 1.0) -> None:
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.vote_gamma = vote_gamma
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # Multiplier on the derived correction term r_t/sigma_t^2. 1.0 is
        # the un-fit, purely-derived strategy; 0.5/2.0 exist only as a
        # sensitivity check on the same frozen structural form, never
        # searched for a best value.
        self.funding_scale = funding_scale

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # --- verbatim kelly_regime_v3 vote + vol-state sizing ---
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

        # --- the novel step: r_t / sigma_t^2, subtracted from the target ---
        # sigma_t^2 reuses the exact `vol` array above (no new estimator).
        # r_t is the causal, annualized funding rate (raw column is the
        # per-8h settlement rate; see experiments/funding_signal.py for
        # how it got onto this frame, causally, before prepare() ran).
        if "funding" in df.columns:
            r_t = df["funding"].to_numpy(dtype=float) * FUNDING_SETTLEMENTS_PER_YEAR
        else:
            r_t = np.zeros(len(df))
        with np.errstate(divide="ignore", invalid="ignore"):
            correction = self.funding_scale * r_t / (vol ** 2)
        correction = np.where(np.isfinite(correction), correction, 0.0)

        n = len(df)
        target = np.zeros(n)
        correction_applied = np.zeros(n)
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
            base_desired = frac[i] * scale
            desired = base_desired - correction[i]
            # Re-apply the existing leverage clamp. The uncorrected sizer
            # is long-only (scale >= 0, frac in [0, 1]); the correction can
            # in principle push `desired` negative (very high funding) or
            # above max_leverage (very negative funding). Clip to the same
            # [0, max_leverage] range rather than let funding alone open a
            # short (see module docstring).
            desired = float(np.clip(desired, 0.0, self.max_leverage))
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos
            correction_applied[i] = correction[i]

        df["target"] = target
        df["funding_correction"] = correction_applied
        df["kelly_vol"] = vol
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
