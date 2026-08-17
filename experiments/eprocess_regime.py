"""E-process regime detection with unified Kelly sizing (backlog B-01).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime`` answers two questions with two unrelated mechanisms:

1. *is the market in a bullish regime?* — a hypothesis test, answered by
   "price is 1% above a moving average, latched";
2. *how much should I hold?* — a Kelly sizing problem, answered by
   ``target_vol / realized_vol``.

Game-theoretic statistics (Shafer 2021, JRSS-A "Testing by betting";
Ramdas, Grunwald, Vovk & Shafer 2023, Statist. Sci. "Game-theoretic
statistics and safe anytime-valid inference") says these are the *same*
question. Evidence against a null is measured by how much money a bettor
would have made betting against it at fair odds; the test statistic is a
nonnegative martingale — a wealth process — and the growth-optimal bet
size is the Kelly fraction. So the wealth accumulated against "drift is
zero" IS the regime evidence, and the Kelly bet that grows it IS the
position.

The construction
----------------
Let ``r_t`` be the bar log return and ``s_{t-1}`` a predictable EWMA
volatility estimate. Define the standardized, clipped return::

    z_t = clip(r_t / s_{t-1}, -c, +c)

Null ``H0: E[z_t | F_{t-1}] <= 0`` (no positive drift). For any
*predictable* bet ``lam_t in [0, 1/c)`` the wealth process::

    W_t = prod_{s<=t} (1 + lam_s * z_s)

is a nonnegative supermartingale under H0 with ``W_0 = 1``, so Ville's
inequality gives ``P(sup_t W_t >= 1/alpha) <= alpha`` — anytime-valid,
non-asymptotic, and valid at arbitrary stopping times. That last property
is what this project actually needs: with an effective sample size of ~3
regime events and a holdout that has been read dozens of times, classical
error control was never available.

The bet is the empirical-Kelly (GRAPA) rule of Waudby-Smith & Ramdas
(2024, JRSS-B "Estimating means of bounded random variables by betting"),
computed on exponentially-decayed sums so it tracks the current regime::

    lam_t = clip( EW[z]_{t-1} / EW[z^2]_{t-1}, 0, lam_max )

which is the one-step Kelly optimum ``E[z] / E[z^2]`` for the
standardized bet. Two things follow, and they are the whole point:

**The bet is the position.** ``lam * z = (lam / s) * r``, so betting
``lam`` on the standardized return is exactly holding ``lam / s`` of
wealth in the asset. In annualized terms ``lam / s`` = (annualized
Sharpe estimate) / (annualized vol) — that is, *full Kelly sets the
volatility target equal to the Sharpe ratio*. The ``target_vol`` this
repo has been setting by hand stops being a free parameter.

**The evidence is the gate.** Accumulated log-wealth, floored at zero and
capped, is a CUSUM of betting increments — the e-detector form of Shin,
Ramdas & Rinaldo (2024, Ann. Statist. "E-detectors: a nonparametric
framework for sequential change detection"). Exposure scales with
``L_t / log(1/alpha)``: evidence, not a 0/1/3/2/3/1 latched vote with a
hand-set 1% band.

Why this is not R-03 (Bayesian online changepoint detection, which lost)
-----------------------------------------------------------------------
BOCPD's run-length posterior collapses on *volatility* bursts, and in BTC
large **up** moves are volatility bursts (R-10), so it fired with the
wrong sign. Here volatility sits in the *denominator* of the bet: a
volatility burst shrinks ``z`` and therefore shrinks the position, but it
does not by itself destroy evidence. Only realized *drift* moves the
e-process.

Variants
--------
``sizing="fixed"``, ``gate=True``  — evidence replaces the anchor vote,
incumbent sizer unchanged (E1: isolates the gate).
``sizing="kelly"``, ``gate=False`` — position IS the Kelly bet, no
``target_vol`` at all (E2: isolates the sizer).
``sizing="kelly"``, ``gate=True``  — both endogenous (E3: the unified
object).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


class EProcessRegime(Strategy):
    """Size by the Kelly bet of an e-process tested against "drift is zero"."""

    name = "eprocess_regime"
    warmup = 100 * BARS_PER_DAY + 10

    def __init__(
        self,
        bet_halflife_days: float = 60.0,
        alpha: float = 0.05,
        clip: float = 5.0,
        gate: bool = True,
        sizing: str = "fixed",
        kelly_fraction: float = 0.5,
        target_vol: float = 0.55,
        max_leverage: float = 2.0,
        vol_span: int = 8 * BARS_PER_DAY,
        deadband: float = 0.10,
        evidence_cap_mult: float = 1.0,
        evidence_halflife_days: float | None = None,
    ) -> None:
        if sizing not in ("fixed", "kelly"):
            raise ValueError(f"sizing must be 'fixed' or 'kelly', got {sizing!r}")
        self.bet_halflife_days = bet_halflife_days
        self.alpha = alpha
        self.clip = clip
        self.gate = gate
        self.sizing = sizing
        self.kelly_fraction = kelly_fraction
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.evidence_cap_mult = evidence_cap_mult
        self.evidence_halflife_days = evidence_halflife_days

    # ------------------------------------------------------------------ pieces

    def _bet(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (z, lam, annualized vol), all predictable at each row.

        ``z[i]`` uses the return of bar ``i`` and volatility through
        ``i-1``; ``lam[i]`` uses ``z`` through ``i-1`` only. Both are
        therefore known at the close of bar ``i``, which is when the
        decision is taken and one bar before it is filled.
        """
        r = np.log(df["close"]).diff()
        vol = (r.ewm(span=self.vol_span, min_periods=BARS_PER_DAY).std()
               * np.sqrt(BARS_PER_YEAR)).shift(1)
        sigma_bar = vol / np.sqrt(BARS_PER_YEAR)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (r / sigma_bar).clip(-self.clip, self.clip)
        z = z.where(np.isfinite(z))

        hl = self.bet_halflife_days * BARS_PER_DAY
        m1 = z.ewm(halflife=hl, min_periods=BARS_PER_DAY).mean().shift(1)
        m2 = (z * z).ewm(halflife=hl, min_periods=BARS_PER_DAY).mean().shift(1)
        lam_max = 0.9 / self.clip  # keeps 1 + lam*z > 0, so W stays nonnegative
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (m1 / m2).clip(0.0, lam_max)

        return (np.nan_to_num(z.to_numpy()),
                np.nan_to_num(lam.to_numpy()),
                vol.to_numpy())

    # ----------------------------------------------------------------- strategy

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        z, lam, vol = self._bet(df)

        thr = np.log(1.0 / self.alpha)
        cap = self.evidence_cap_mult * thr
        gamma = (1.0 if self.evidence_halflife_days is None else
                 0.5 ** (1.0 / (self.evidence_halflife_days * BARS_PER_DAY)))

        # Fixed sizer: the incumbent's inverse-volatility target.
        with np.errstate(divide="ignore", invalid="ignore"):
            fixed_scale = np.minimum(self.target_vol / vol, self.max_leverage)
            # Kelly sizer: exposure = kelly_fraction * lam / sigma_bar, i.e. an
            # endogenous volatility target equal to the estimated Sharpe.
            kelly_vol = self.kelly_fraction * lam * np.sqrt(BARS_PER_YEAR)
            kelly_scale = np.minimum(kelly_vol / vol, self.max_leverage)
        fixed_scale = np.where(np.isfinite(fixed_scale), fixed_scale, 0.0)
        kelly_scale = np.where(np.isfinite(kelly_scale), kelly_scale, 0.0)
        scale = kelly_scale if self.sizing == "kelly" else fixed_scale

        n = len(df)
        target = np.zeros(n)
        evidence = np.zeros(n)
        pos = 0.0
        wealth = 0.0  # log e-process wealth, floored at 0 and capped
        for i in range(n):
            wealth = gamma * wealth + np.log1p(lam[i] * z[i])
            wealth = min(cap, max(0.0, wealth))
            evidence[i] = wealth
            if self.gate:
                conf = min(1.0, wealth / thr)
            else:
                conf = 1.0 if lam[i] > 0.0 else 0.0
            desired = conf * scale[i]
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["evidence"] = evidence
        df["lam"] = lam
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)
