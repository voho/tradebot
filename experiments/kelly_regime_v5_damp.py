"""kelly_regime_v4 with a bounded confidence dampener from the Harsanyi posterior (R-34, 08-19).

Unregistered experiment: lives under ``experiments/`` so it is NOT
auto-discovered (docs/ROUTINE.md step 5). Do not decorate with
``@register``.

Idea in one sentence
---------------------
``harsanyi_crowd`` (L-12, NEGATIVE) builds a Bayesian posterior over three
hidden market types (bull/bear/chop, Harsanyi 1967-68) from bar-return
likelihoods and trades the belief margin P(bull)-P(bear) *directionally* --
and loses. L-12's own recorded lesson: "the crowding intuition was right --
it is what kelly_regime later exploited -- but as a direction signal rather
than a sizing input it loses." That is a stated, never-tested hypothesis.
This module tests it the other way: feed the same posterior margin into
``kelly_regime_v4``'s exposure (the SIZE axis, the only axis that has ever
worked in this project, per the standing diagnosis) as a strict,
never-increase-only dampener, instead of a new predictor.

Mechanism
---------
Everything v4 already does -- the 20/40/80-day latched anchor vote that
produces ``frac``, and the conditional (extreme-only) volatility-targeting
scale from v3 -- is reproduced here byte-for-byte (see ``prepare`` below,
copied from ``kelly_regime_v3.KellyRegimeV3.prepare`` /
``kelly_regime_v4.KellyRegimeV4``). The only new ingredient is a
confidence multiplier:

1. ``raw_margin = bayesian_margin(df)`` -- the shared, already causality-
   verified Bayesian posterior margin (``experiments/bayes_confidence.py``,
   byte-identical recursion to the registered, CI-passing
   ``harsanyi_crowd``), P(bull) - P(bear) in [-1, 1].
2. Floor at 0 (only bullish confidence ever counts -- this project's
   stated "never short a historically-upward-drifting asset" stance,
   ``kelly_regime``'s own docstring), then smooth with a causal EMA
   (``conf_span_days``). The EMA is needed because a per-bar Bayesian
   update is exactly the kind of twitchy series L-12 found too fast to
   trade directly (mu=0.15, stick=0.985 moves within a handful of bars);
   smoothing over a multi-day span is what makes the confidence axis a
   *regime* read rather than a bar-to-bar wiggle.

   A first version of this file ALSO latched the smoothed confidence with
   its own deadband, matching v4's own position-hysteresis idiom
   (``pos`` only updates when it moves by more than ``deadband``) --
   reusing the house style literally. That was wrong in a way worth
   recording: measured on this data, the floored-and-smoothed margin's
   *entire* range is small (3-day-EMA max ~=0.11, std ~=0.012 over
   2017-2022), so a deadband of v4's own size (0.10) exceeds the signal's
   whole dynamic range and the latch never leaves its zero start state --
   silently turning the "confidence-driven dampener" into a near-constant
   multiplier of ``(1-lam)``, which would have tested nothing but a flat
   de-lever. The fix kept here is simpler than the thing it replaced: use
   the EMA-smoothed value directly, with NO second latch. Re-trading on
   every wiggle is already prevented two ways -- the EMA itself, and the
   *existing* deadband on the final position (``frac * mult * scale`` only
   updates ``pos`` when it moves by more than ``deadband``, exactly as in
   v3/v4) -- so a redundant intermediate latch buys nothing except the
   failure mode above if its scale is ever mismatched to the signal it
   gates. ``conf_span_days`` is the one new smoothing knob this leaves;
   chosen a-priori at 3 days (short relative to the 8-20/40/80-day scales
   v3/v4 already use, since the margin needs less averaging than realized
   volatility to stop being single-bar noise), not fit for performance.
3. ``mult = 1 - lam * (1 - conf)`` -- ranges in ``[1-lam, 1]``. Since
   ``conf`` is floored at 0, ``mult`` can only ever REDUCE exposure,
   never raise it above what v4 alone would hold. ``lam=0`` is
   `mult == 1`` identically, for every bar, regardless of ``conf`` --
   which is the built-in correctness check: this file reduces to v4
   exactly at ``lam=0`` by construction, not by tuning.
4. ``frac_final = frac_vote * mult``, fed into the identical v4 sizing
   scale (``frac_final * scale``, same deadband-latched position loop).

Causality
---------
``bayesian_margin`` is already verified causal (two-opposite-tampers probe,
max diff 0.0 before the cut -- see its own docstring). The EMA smoothing
(``pandas.Series.ewm``) uses only rows <= i. The deadband latch runs
inside the same single forward pass v4 already uses for its position, so
row i's ``target`` depends only on rows <= i. No ``.shift(-1)``, no
full-series statistic (mean/std/quantile) fit over the whole frame and
applied to early rows -- every estimator here is either an ``ewm`` or a
``rolling`` window, both causal by construction. Signals are read off the
bar-close columns in ``on_bar`` and filled at the next open via
``ctx.order_notional`` (identical pattern to ``kelly_regime.KellyRegime``),
so no lookahead enters through execution either.

Falsifiable prediction (recorded before evaluation, see
``experiments/reports/kelly_regime_v5_damp_report.md``): the dampener is
redundant with the vote it multiplies -- the Bayesian margin over
bull/bear/chop and the latched 20/40/80-day price-vs-anchor vote are both
reading the same trend information out of the same one price series
(INFO constraint), so ``conf`` should correlate highly with ``frac`` on
inner-validation, and any drawdown change should trace back to the
mean-exposure-level artifact this project has been burned by three times
(R-28/R-31, R-32, R-33/L-04) rather than to a genuine gate-quality effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.bayes_confidence import bayesian_margin
from tradebot.strategy import Context, Strategy

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


class KellyRegimeV5Damp(Strategy):
    """v4's vote + conditional vol-targeting exposure, dampened (never raised) by Bayesian confidence.

    See module docstring for the full mechanism and the falsification
    prediction. Defaults for every v4-inherited parameter match
    ``kelly_regime_v4`` exactly; ``lam`` and ``conf_span_days`` are the
    only new knobs.
    """

    name = "kelly_regime_v5_damp"
    warmup = 80 * BARS_PER_DAY + 10

    def __init__(self, horizons: tuple[int, ...] = (20, 40, 80), band: float = 0.01,
                 target_vol: float = 0.55, max_leverage: float = 2.0,
                 vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10,
                 vote_gamma: float = 1.0,
                 anchor_span_days: int = 180, high_in: float = 1.70,
                 high_out: float = 1.20, low_in: float = 0.55, low_out: float = 0.85,
                 lam: float = 0.3, conf_span_days: float = 3.0) -> None:
        # ---- identical to kelly_regime / v3 / v4 -------------------------
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
        # ---- new: the confidence dampener --------------------------------
        self.lam = lam                        # mult in [1-lam, 1]; 0 = exact v4
        self.conf_span_days = conf_span_days  # causal EMA smoothing of the margin

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        r = np.log(close).diff()

        # ---- byte-for-byte v3/v4: latched multi-anchor vote -> frac ------
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

        # ---- byte-for-byte v3/v4: conditional vol-targeting scale --------
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

        # ---- new: Bayesian confidence dampener ---------------------------
        # Shared, already causal, already verified helper -- reused exactly,
        # not re-derived (module docstring).
        raw_margin = bayesian_margin(df)
        # Floor at 0 BEFORE smoothing: only bullish confidence ever counts,
        # matching kelly_regime's own "never short a historically-upward-
        # drifting asset" stance, and it keeps a smoothed negative margin
        # from leaking through as a nonzero (still floored) confidence.
        floored = np.clip(raw_margin, 0.0, 1.0)
        conf_span = max(1.0, self.conf_span_days * BARS_PER_DAY)
        conf_smooth = (pd.Series(floored, index=df.index)
                       .ewm(span=conf_span, min_periods=1).mean().to_numpy())

        # ---- single causal forward pass: byte-for-byte v3/v4 breakout
        # hysteresis on the vol-targeting state, plus the new dampener ----
        n = len(df)
        target = np.zeros(n)
        pos = 0.0
        state = 0  # 0 normal vol band, +1 high-vol breakout, -1 low-vol breakout
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

            # mult in [1-lam, 1]; never raises exposure above v4's. conf_smooth
            # is already EMA-smoothed (causal, no extra latch needed -- see
            # module docstring for why a second deadband here was wrong).
            mult = 1.0 - self.lam * (1.0 - conf_smooth[i])
            desired = frac[i] * mult * scale
            if abs(desired - pos) > self.deadband:
                pos = desired
            target[i] = pos

        df["target"] = target
        df["conf"] = conf_smooth       # diagnostics: mean exposure / correlation checks
        df["vote_frac"] = frac         # diagnostics: correlation of conf vs the discrete vote
        return df

    def on_bar(self, ctx: Context) -> None:
        # Identical execution pattern to kelly_regime.KellyRegime.on_bar:
        # signal at bar close, fill at next open via order_notional.
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_notional(t)  # fraction of equity: same risk on spot and futures
