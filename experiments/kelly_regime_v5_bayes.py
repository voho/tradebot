"""R-34: replace kelly_regime_v4's discrete anchor vote with a continuous,
hysteresis-latched Bayesian regime-confidence signal, on the SAME
conditional-vol-targeting sizer v4 already uses.

Idea in one sentence: `harsanyi_crowd` (L-12) computes a Bayesian belief
margin P(bull)-P(bear) over three hidden market types (Harsanyi 1967-68) on
an hours-to-days timescale and trades it *directionally*, and loses; L-12's
own recorded lesson is that "the crowding intuition was right... but as a
direction signal rather than a sizing input it loses" -- a stated, never
tested hypothesis. `kelly_regime_v4` (L-01) is the strategy that DID work,
and its regime input is a *discrete* vote (0, 1/3, 2/3, 1) from three
latched weeks-to-months moving-average anchors (20/40/80 day). This module
is the novel variant of the R-34 sizing round: swap v4's discrete,
slow anchor vote for a continuous, hysteresis-latched transform of the
Bayesian margin, feeding the identical conditional-vol-targeting scale code
v4 uses unchanged, so any measured difference is attributable to the
regime-confidence mechanism and not to the risk axis.

Not a duplicate of: L-12 (margin traded directionally, can go short, no
vol-targeting sizer); L-01/L-02/L-03/L-04 (discrete weeks-to-months anchor
vote, not a continuous hours-to-days Bayesian one); a sibling experiment in
this same round builds a bounded DAMPENER multiplied onto v4's existing
discrete vote (conservative variant, different file) -- this module instead
REPLACES the vote entirely (the "deeper redesign" variant).

Mechanism, precisely:

1. `scale = full/steady` breakout-hysteresis conditional-vol-targeting code,
   copied verbatim from `kelly_regime_v3.KellyRegimeV3.prepare` (constant
   notional through normal volatility, re-size only on a volatility
   breakout, latched). This is the risk axis and it is UNCHANGED from v4.
2. `margin = bayesian_margin(df, mu, stick)` from `experiments/bayes_confidence.py`
   -- the byte-identical, already-causality-probed posterior recursion
   `harsanyi_crowd` uses, imported rather than re-derived.
3. A hysteresis latch shaped exactly like `harsanyi_crowd`'s own `b_in`/
   `b_out` bands (enter a confident state only above `b_in`, release it
   only once margin falls back through the looser `b_out`) but mapped to a
   CONTINUOUS, NEVER-NEGATIVE fraction instead of harsanyi's directional
   hysteresis-then-full-size jump: once latched "in",
   `frac = clip((margin - b_out) / (1 - b_out), 0, 1)`, else `frac = 0`.
   This is causal (state depends only on bars <= i), always in [0, 1] (this
   project's strategies never short a historically-upward-drifting asset --
   see `kelly_regime.py`'s own docstring), and has a genuine no-trade
   deadband: while un-latched, `frac` is pinned at 0 regardless of small
   wiggles below `b_in`.
4. `frac` REPLACES v4's discrete vote entirely in `desired = frac[i] *
   scale[i]`. Everything else -- the final deadband on `desired` vs the
   held position, `target_vol`, `max_leverage`, `vol_span` -- matches v4's
   defaults exactly.

Falsification (pre-registered per ROUTINE.md step 2): if this variant's
ordering against v4 (which is better on return, which on drawdown) is not
the same on the ETH/BTC Bitfinex control pair as it is on BTC/Bitstamp, the
result does not survive falsification, exactly as happened to R-28 (retired
by R-31) and warned against by R-33. If the best-looking config merely runs
at a different mean exposure than v4, that is the R-31/R-32/R-33 arithmetic
artifact, not a regime-detection improvement, and must be flagged plainly
rather than reported as an edge.

UNREGISTERED experiment (ROUTINE.md): no `@register`, not in the README
comparison table, not in the CI-enforced inference set. Kept here as a
frozen, reviewable record of the R-34 novel-variant branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.bayes_confidence import bayesian_margin  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402


class KellyRegimeV5Bayes(Strategy):
    """v4's conditional-vol-targeting sizer, fed by a continuous latched Bayesian margin instead of the discrete anchor vote.

    See module docstring for the full mechanism and pre-registered
    falsification test. UNREGISTERED: `experiments/kelly_regime_v5_bayes.py`
    is not auto-discovered and carries no CI-enforced inference interval.
    """

    name = "kelly_regime_v5_bayes"
    # Bayesian margin warmup mirrors harsanyi_crowd's own (ATR(48) burn-in +
    # a little slack for the sticky posterior to leave its uniform prior) --
    # deliberately much shorter than v4's 80-day anchor warmup, because this
    # signal operates on an hours-to-days timescale, not weeks-to-months.
    warmup = 1300

    def __init__(self, mu: float = 0.15, stick: float = 0.985,
                 b_in: float = 0.15, b_out: float = 0.05,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55,
                 low_out: float = 0.85, exposure_mult: float = 1.0) -> None:
        self.mu = mu
        self.stick = stick
        self.b_in, self.b_out = b_in, b_out
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_span = vol_span
        self.deadband = deadband
        self.anchor_span_days = anchor_span_days
        self.high_in, self.high_out = high_in, high_out
        self.low_in, self.low_out = low_in, low_out
        # Diagnostic-only knob (default 1.0, never swept as a "performance"
        # parameter): a constant multiplier on the confidence fraction, used
        # solely to re-run a frozen config at a mean exposure matched to
        # v4's, per ROUTINE.md's "match risk before comparing anything" and
        # the R-31/R-32/R-33 exposure-level artifact it warns about.
        self.exposure_mult = exposure_mult

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # --- regime-confidence input: continuous, hysteresis-latched Bayesian margin ---
        margin = bayesian_margin(df, mu=self.mu, stick=self.stick)

        # --- risk axis: kelly_regime_v3's conditional-vol-targeting scale, UNCHANGED ---
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

        n = len(df)
        target = np.zeros(n)
        frac_series = np.zeros(n)  # exposed for the vote-correlation diagnostic
        pos = 0.0
        vol_state = 0   # 0 normal band, +1 high-vol breakout, -1 low-vol breakout
        conf_state = 0  # 0 = not latched (no confidence), 1 = latched-in
        for i in range(n):
            x = ratio[i]
            if np.isfinite(x):
                if vol_state == 0:
                    vol_state = 1 if x > self.high_in else (-1 if x < self.low_in else 0)
                elif vol_state == 1 and x < self.high_out:
                    vol_state = 0
                elif vol_state == -1 and x > self.low_out:
                    vol_state = 0
            scale = full[i] if vol_state != 0 else steady[i]

            m = margin[i]
            if conf_state == 0:
                if m > self.b_in:
                    conf_state = 1
            elif m < self.b_out:
                conf_state = 0
            frac = (0.0 if conf_state == 0
                    else float(np.clip((m - self.b_out) / (1.0 - self.b_out), 0.0, 1.0)))
            frac_series[i] = frac  # unscaled, for the vote-correlation diagnostic

            desired = (frac * self.exposure_mult) * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["confidence_frac"] = frac_series
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
