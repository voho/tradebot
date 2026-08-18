"""Matched-risk frontier: e-process gate vs latched anchor vote (backlog B-11).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

The question, and why R-28 could not answer it
----------------------------------------------
R-28 ran an e-process evidence gate (``experiments/eprocess_regime.py``)
against the incumbent ``kelly_regime_v4`` and found *better risk, worse
return*. That comparison is partly a tautology: correctly-calibrated
anytime-valid evidence justified only **0.27x** the incumbent's mean
exposure, so the two strategies were measured at two different points on
an exposure/risk trade-off. Anything holding a third of the notional will
draw down less and return less. The finding was real; the *comparison*
was not controlled.

The controlled version is this file. One sizer, one deadband, one warmup,
one exposure knob — and the only thing that varies is **which quantity
opens the gate**:

``gate="vote"``      the incumbent's latched multi-anchor vote: price 1%
                     above a 20/40/80-day mean, latched, averaged.
``gate="evidence"``  R-28's e-process: accumulated log-wealth of a
                     GRAPA-Kelly bet against "drift is zero", floored at
                     zero and capped at ``log(1/alpha)``, divided by that
                     threshold.

Both produce ``conf`` in [0, 1]; both multiply the *same* sizer; both are
scaled by the *same* exposure multiplier ``k``.

The exposure knob, and the one way not to build it
--------------------------------------------------
``k`` multiplies ``target_vol``, ``max_leverage`` and ``deadband``
together. Because ``min(k*tv/vol, k*ml) == k*min(tv/vol, ml)`` exactly,
this is a pure rescaling of the position: it changes how much is held and
nothing about when. Scaling the deadband with it keeps the *relative*
rebalancing tolerance — and therefore turnover — comparable across the
frontier instead of making high-exposure configurations trade more.

R-28 left an explicit warning about the wrong way to do this: raising the
e-process arm's exposure through ``evidence_cap_mult`` lets stale
evidence persist, and drawdown then grows superlinearly (49% DD, −22% on
inner-validation at cap 2). That knob is fixed at 1.0 here. Exposure is a
sizing decision; the cap is part of the test's definition.

The sizer, which is held fixed and reported both ways
-----------------------------------------------------
``sizer="plain"``        ``min(target_vol / vol, max_leverage)`` — the
                         ``kelly_regime`` sizer, and the one R-28's E1
                         variant used, so its numbers stay comparable.
``sizer="conditional"``  ``kelly_regime_v3``'s extreme-only targeting:
                         constant notional through normal volatility,
                         inverse-vol sizing only on a latched breakout.

Matching is done on **realized annualized volatility of the equity
curve**, measured on inner-validation only, and the resulting ``k`` is
frozen before the holdout is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


def realized_vol(equity: pd.Series | np.ndarray) -> float:
    """Annualized standard deviation of per-bar equity returns.

    The matching axis. A level, not a difference, so it needs no
    benchmark — which is the point: two strategies can be placed at the
    same risk without either one being the reference.
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 3:
        return 0.0
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    return float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR))


class GatedKelly(Strategy):
    """Fractional-Kelly sizing behind an interchangeable regime gate."""

    name = "gated_kelly"
    # Deliberately identical for both gates: the vote's slowest anchor is
    # 80 days and the e-process needs a settled volatility estimate, so a
    # common 100-day prefix means neither arm enters a measured period
    # warmer than the other.
    warmup = 100 * BARS_PER_DAY + 10

    def __init__(
        self,
        gate: str = "vote",
        exposure: float = 1.0,
        sizer: str = "plain",
        # --- the vote gate
        horizons: tuple[int, ...] = (20, 40, 80),
        band: float = 0.01,
        # --- the evidence gate
        bet_halflife_days: float = 20.0,
        alpha: float = 0.05,
        clip: float = 5.0,
        evidence_cap_mult: float = 1.0,
        # --- the shared sizer
        target_vol: float = 0.55,
        max_leverage: float = 2.0,
        vol_span: int = 8 * BARS_PER_DAY,
        deadband: float = 0.10,
        # --- the conditional sizer (kelly_regime_v3), used when sizer="conditional"
        anchor_span_days: int = 180,
        high_in: float = 1.70,
        high_out: float = 1.20,
        low_in: float = 0.55,
        low_out: float = 0.85,
    ) -> None:
        if gate not in ("vote", "evidence"):
            raise ValueError(f"gate must be 'vote' or 'evidence', got {gate!r}")
        if sizer not in ("plain", "conditional"):
            raise ValueError(f"sizer must be 'plain' or 'conditional', got {sizer!r}")
        if exposure <= 0.0:
            raise ValueError(f"exposure must be positive, got {exposure!r}")
        self.gate = gate
        self.exposure = exposure
        self.sizer = sizer
        self.horizons = horizons
        self.band = band
        self.bet_halflife_days = bet_halflife_days
        self.alpha = alpha
        self.clip = clip
        self.evidence_cap_mult = evidence_cap_mult
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out

    # --------------------------------------------------------------- the gates

    def _vote(self, close: pd.Series) -> np.ndarray:
        """Latched multi-anchor vote — ``kelly_regime``'s gate, unchanged."""
        votes = []
        for days in self.horizons:
            anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
            v = pd.Series(
                np.where(close > anchor * (1.0 + self.band), 1.0,
                         np.where(close < anchor * (1.0 - self.band), 0.0, np.nan)),
                index=close.index,
            )
            votes.append(v.ffill().fillna(0.0))
        return (sum(votes) / len(votes)).to_numpy()

    def _evidence(self, r: pd.Series, vol: np.ndarray) -> np.ndarray:
        """E-process gate — ``experiments/eprocess_regime.py``'s, unchanged.

        ``z`` uses bar ``i``'s return against volatility through ``i-1``;
        the bet ``lam`` uses ``z`` through ``i-1`` only. Both are known at
        the close of bar ``i``, one bar before anything fills.
        """
        sigma_bar = pd.Series(vol, index=r.index) / np.sqrt(BARS_PER_YEAR)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (r / sigma_bar).clip(-self.clip, self.clip)
        z = z.where(np.isfinite(z))

        hl = self.bet_halflife_days * BARS_PER_DAY
        m1 = z.ewm(halflife=hl, min_periods=BARS_PER_DAY).mean().shift(1)
        m2 = (z * z).ewm(halflife=hl, min_periods=BARS_PER_DAY).mean().shift(1)
        lam_max = 0.9 / self.clip  # keeps 1 + lam*z > 0, so wealth stays nonnegative
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (m1 / m2).clip(0.0, lam_max)

        zv = np.nan_to_num(z.to_numpy())
        lv = np.nan_to_num(lam.to_numpy())

        thr = np.log(1.0 / self.alpha)
        cap = self.evidence_cap_mult * thr
        wealth = 0.0
        conf = np.zeros(len(zv))
        for i in range(len(zv)):
            wealth = min(cap, max(0.0, wealth + np.log1p(lv[i] * zv[i])))
            conf[i] = min(1.0, wealth / thr)
        return conf

    # -------------------------------------------------------------- the sizer

    def _scale(self, vol: np.ndarray) -> np.ndarray:
        """Exposure before the gate, scaled by ``exposure``.

        ``min(k*tv/vol, k*ml) == k*min(tv/vol, ml)``, so ``k`` rescales the
        position without touching its shape.
        """
        k = self.exposure
        tv, ml = k * self.target_vol, k * self.max_leverage
        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.minimum(tv / vol, ml)
        full = np.where(np.isfinite(full), full, 0.0)
        if self.sizer == "plain":
            return full

        slow = (pd.Series(vol).ewm(span=self.anchor_span_days * BARS_PER_DAY,
                                   min_periods=BARS_PER_DAY).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(slow > 0, vol / slow, np.nan)
            steady = np.minimum(tv / slow, ml)
        steady = np.where(np.isfinite(steady), steady, 0.0)

        out = np.empty(len(vol))
        state = 0  # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        for i in range(len(vol)):
            x = ratio[i]
            if np.isfinite(x):
                if state == 0:
                    state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif state == 1 and x < self.high_out:
                    state = 0
                elif state == -1 and x > self.low_out:
                    state = 0
            out[i] = full[i] if state != 0 else steady[i]
        return out

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

        conf = self._vote(close) if self.gate == "vote" else self._evidence(r, vol)
        scale = self._scale(vol)

        band = self.exposure * self.deadband
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        for i in range(n):
            desired = conf[i] * scale[i]
            if abs(desired - pos) > band:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["conf"] = conf
        df["scale"] = scale
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
