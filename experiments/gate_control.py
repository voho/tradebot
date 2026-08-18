"""The ungated control, and a second reading of the gate comparison (R-32).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5.

**Read `matched_risk.py` first.** That file is R-31, the primary record for
backlog B-11, and it carries the validity gate this one does not. This
module was written independently and in parallel — same day, same backlog
row, no knowledge of the other until both had been run — so it is kept for
two reasons and no others:

1. it runs the arm R-31 does not, a **third gate that is no gate at all**,
   which is the only way to ask whether gating earns its keep rather than
   which gate is better; and
2. it is an independent implementation, so where the two agree the
   agreement is worth something.

Where they disagree, R-31 wins: its exposure solver matches to within 2%
in both directions and voids cells where the match fails, and applying its
validity rule here voids this file's spot cell too (the notional cap binds
on 21–41% of holdout bars). See docs/LEDGER.md rows R-31 and R-32.

The question
------------
R-28 measured an e-process evidence gate against the incumbent's latched
anchor vote and found "better risk, worse return". That comparison is
partly a tautology: the two ran at *different exposure levels* — the
e-process held 0.27x the incumbent's mean notional — and holding less of
a rising asset is guaranteed to produce both a smaller drawdown and a
smaller return. The comparison says nothing about the gates themselves.

This module strips the confound. One sizer, one deadband, one cap, one
fee model; the **only** thing that varies is the gate:

``none``      no gate at all — pure inverse-volatility targeting.
``vote``      the incumbent's latched multi-anchor crowd vote
              (``kelly_regime_v4``: 20/40/80-day means, 1% band,
              hysteresis), a value in {0, 1/3, 2/3, 1}.
``evidence``  the R-28 e-process: accumulated log-wealth of a
              GRAPA/Kelly bet against "drift is zero", floored at 0,
              capped at log(1/alpha), read as ``L_t / log(1/alpha)``.

and a scalar ``multiplier`` that moves each arm along its own risk axis::

    target_t = min(multiplier * gate_t * min(target_vol / vol_t, max_lev),
                   exposure_cap)

Sweeping the multiplier traces a **frontier** for each gate. Comparing
the frontiers at equal realized risk — rather than comparing two points
at different risk — is the question B-11 asks: *which gate delivers more
return per unit of risk?*

Why the multiplier and not ``evidence_cap_mult``
------------------------------------------------
R-28 left an explicit warning: raising the e-process exposure by lifting
the evidence cap keeps *stale* evidence alive, and drawdown then grows
superlinearly (49% DD, −22% on inner-validation at cap 2). The multiplier
here is a pure scalar on the position and does not touch the evidence
process, so the gate's *shape* is preserved and only its amplitude moves.

Nothing here is fitted to the data beyond the multiplier, and the
multiplier is frozen on inner-validation before the holdout is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

GATES = ("none", "vote", "evidence")


def vote_gate(df: pd.DataFrame, horizons=(20, 40, 80), band: float = 0.01
              ) -> np.ndarray:
    """The incumbent's latched crowd-regime vote, in [0, 1].

    Identical construction to ``kelly_regime``: price above the anchor by
    more than ``band`` latches bullish, below latches bearish, inside the
    band holds the previous verdict.
    """
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def evidence_gate(df: pd.DataFrame, bet_halflife_days: float = 20.0,
                  alpha: float = 0.05, clip: float = 5.0,
                  vol_span: int = 8 * BARS_PER_DAY,
                  evidence_cap_mult: float = 1.0) -> np.ndarray:
    """The R-28 e-process gate, in [0, 1].

    ``z_t`` standardizes bar returns by a *predictable* volatility
    estimate, ``lam_t`` is the exponentially-decayed GRAPA/Kelly bet
    (Waudby-Smith & Ramdas 2024) computed from data strictly before the
    bar, and the accumulated log-wealth of the resulting nonnegative
    supermartingale — floored at zero, capped at ``log(1/alpha)`` — is
    the evidence against "drift is zero" (Shafer 2021; Shin, Ramdas &
    Rinaldo 2024). The gate is that evidence as a fraction of the alpha
    threshold.
    """
    r = np.log(df["close"]).diff()
    vol = (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1)
    sigma_bar = vol / np.sqrt(BARS_PER_YEAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (r / sigma_bar).clip(-clip, clip)
    z = z.where(np.isfinite(z))

    hl = bet_halflife_days * BARS_PER_DAY
    m1 = z.ewm(halflife=hl, min_periods=BARS_PER_DAY).mean().shift(1)
    m2 = (z * z).ewm(halflife=hl, min_periods=BARS_PER_DAY).mean().shift(1)
    lam_max = 0.9 / clip  # keeps 1 + lam*z > 0, so wealth stays nonnegative
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (m1 / m2).clip(0.0, lam_max)

    z_a = np.nan_to_num(z.to_numpy())
    lam_a = np.nan_to_num(lam.to_numpy())
    thr = np.log(1.0 / alpha)
    cap = evidence_cap_mult * thr

    wealth = 0.0
    out = np.empty(len(df))
    for i in range(len(df)):
        wealth = min(cap, max(0.0, wealth + np.log1p(lam_a[i] * z_a[i])))
        out[i] = min(1.0, wealth / thr)
    return out


class GatedKelly(Strategy):
    """Inverse-volatility Kelly sizing behind an interchangeable regime gate."""

    name = "gated_kelly"
    warmup = 100 * BARS_PER_DAY + 10

    def __init__(self, gate: str = "vote", multiplier: float = 1.0,
                 horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 exposure_cap: float = 3.0, vol_span: int = 8 * BARS_PER_DAY,
                 deadband: float = 0.10, bet_halflife_days: float = 20.0,
                 alpha: float = 0.05, clip: float = 5.0,
                 evidence_cap_mult: float = 1.0) -> None:
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
        self.gate = gate
        self.multiplier = multiplier
        self.horizons = horizons
        self.band = band
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.exposure_cap = exposure_cap
        self.vol_span = vol_span
        self.deadband = deadband
        self.bet_halflife_days = bet_halflife_days
        self.alpha = alpha
        self.clip = clip
        self.evidence_cap_mult = evidence_cap_mult

    # ------------------------------------------------------------------ pieces

    def gate_series(self, df: pd.DataFrame) -> np.ndarray:
        if self.gate == "none":
            return np.ones(len(df))
        if self.gate == "vote":
            return vote_gate(df, self.horizons, self.band)
        return evidence_gate(df, self.bet_halflife_days, self.alpha, self.clip,
                             self.vol_span, self.evidence_cap_mult)

    def sizer(self, df: pd.DataFrame) -> np.ndarray:
        """Inverse-volatility exposure, predictable at each bar close."""
        r = np.log(df["close"]).diff()
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.minimum(self.target_vol / vol, self.max_leverage)
        return np.where(np.isfinite(scale), scale, 0.0)

    # ---------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        gate = self.gate_series(df)
        scale = self.sizer(df)
        desired = np.minimum(self.multiplier * gate * scale, self.exposure_cap)

        n = len(df)
        target = np.empty(n)
        pos = 0.0
        for i in range(n):
            if abs(desired[i] - pos) > self.deadband:
                pos = desired[i]
            target[i] = pos

        df["gate"] = gate
        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        # The incumbent's rule — order only when the target moves — plus one
        # repair. A target that last changed *inside the warmup prefix*, where
        # trading is disabled, would otherwise never be acted on: at large
        # multipliers the ungated arm pins itself to the exposure cap on its
        # first finite bar and then sits flat forever, entering the frontier
        # as a $1,000 straight line. Same class of silent corpse as R-29's
        # sliced-holdout bug, so it is fixed rather than footnoted.
        if abs(t - prev) > 1e-9 or (t != 0.0 and not ctx.in_market):
            ctx.order_notional(t)
