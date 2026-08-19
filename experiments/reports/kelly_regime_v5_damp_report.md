# kelly_regime_v5_damp — R-34 sizing round (08-19)

Unregistered experiment. Code: `experiments/kelly_regime_v5_damp.py`. Not
`@register`ed, not auto-discovered, nothing committed. All BTC evaluation
below is restricted to `<= 2022-12-31` (inner-train / inner-validation
only); the 2023+ holdout was never read. ETH/BTC-control evaluation uses
the Bitfinex falsification files (2016-03 → 2019-12), which are not the
project's holdout.

## Idea, mechanism, falsifiable prediction (pre-registered before running)

L-12 (`harsanyi_crowd`, NEGATIVE) built a Bayesian posterior margin
P(bull)−P(bear) over three hidden market types and traded it *directionally*
— and lost. Its own recorded lesson: "the crowding intuition was right...
but as a direction signal rather than a sizing input it loses." Untested
until now. `kelly_regime_v5_damp` reproduces `kelly_regime_v4`'s vote and
conditional vol-targeting scale byte-for-byte, and multiplies the vote
fraction by a bounded dampener `mult = 1 - lam*(1-conf) ∈ [1-lam, 1]` built
from the same margin (via `experiments/bayes_confidence.py`) — floored at
0, smoothed with a causal EMA, never able to raise exposure above v4's.

**Prediction, stated before any evaluation:** the dampener is redundant
with the vote it multiplies (both read trend out of the same one price
series — the INFO constraint), so `conf` should correlate highly with
`frac` on inner-validation, and any drawdown change should trace back to
the mean-exposure-level artifact this project has been burned by three
times already (L-04/R-33, R-31, R-32) rather than a genuine gate-quality
effect.

## An implementation correction, made before any result was trusted

The first version latched the smoothed confidence with its own deadband
(0.10, copied from v4's own position-deadband size) — "reuse the same
style of latching kelly_regime already uses." That was wrong: the
floored-and-smoothed margin's entire dynamic range on 2017–2022 is tiny
(3-day EMA max ≈0.11, std ≈0.012), far inside a 0.10 deadband, so the
latch never left its zero start state and the "confidence-driven"
multiplier silently became a near-constant `(1-lam)` — testing nothing but
a flat de-lever. Fixed by dropping the second latch: the EMA smoothing
plus the *existing* position-level deadband (already in v3/v4) is enough
to stop re-trading on every wiggle, and the module docstring now records
both the bug and the fix. This was caught and corrected entirely inside
the inner-train/inner-validation split; the holdout was never touched.

## Causality probe (unregistered strategies get no CI coverage)

Two-opposite-tampers procedure (R-28/R-31's method, by hand), on a
200,000-bar BTC-spot slice ending at 2022-12-31 (holdout untouched), bars
after the cut multiplied by 3 in one copy / divided by 3 in the other:

| check | max\|diff\| before cut | result |
|---|---|---|
| `target` column | 0.000e+00 | PASS |
| `conf` column | 0.000e+00 | PASS |
| `vote_frac` column | 0.000e+00 | PASS |
| equity curve (full backtest) | 0.000e+00 | PASS |

No lookahead detected.

## 1. Correctness check: lam=0 must reproduce kelly_regime_v4 exactly

By construction `mult ≡ 1` when `lam=0` regardless of `conf`, so this is a
build check, not a fit:

| split (spot) | strategy | final | trades | DD | Sharpe |
|---|---|---|---|---|---|
| inner-train | `kelly_regime_v4` | $18,477 | 72 | 43.3% | 2.03 |
| inner-train | `v5_damp(lam=0)` | $18,477 | 72 | 43.3% | 2.03 |
| inner-validation | `kelly_regime_v4` | $998 | 52 | 33.2% | 0.14 |
| inner-validation | `v5_damp(lam=0)` | $998 | 52 | 33.2% | 0.14 |

**Exact match on final balance, trade count and max drawdown, both
splits. PASS.**

## 2. Lam sweep, inner-validation (2021-01-01 → 2022-12-31), spot and futures 5x

| market | lam | final | Sharpe | DD | trades | mean\|target\| | exp. ratio vs v4 | exposure level |
|---|---|---|---|---|---|---|---|---|
| spot | v4 | $998 | 0.14 | 33.2% | 52 | 0.289 | 1.000 | — |
| spot | 0.1 | $982 | 0.10 | 31.2% | 52 | 0.261 | 0.903 | LOWER |
| spot | 0.2 | $983 | 0.08 | 28.9% | 52 | 0.233 | 0.804 | LOWER |
| spot | 0.3 | $985 | 0.07 | 26.2% | 52 | 0.205 | 0.707 | LOWER |
| spot | 0.4 | $987 | 0.06 | 23.2% | 52 | 0.177 | 0.611 | LOWER |
| spot | 0.5 | $983 | 0.02 | 19.9% | 42 | 0.141 | 0.488 | LOWER |
| futures5x | v4 | $1,064 | 0.25 | 32.3% | 52 | 0.289 | 1.000 | — |
| futures5x | 0.1 | $1,019 | 0.17 | 31.3% | 52 | 0.261 | 0.903 | LOWER |
| futures5x | 0.2 | $1,126 | 0.36 | 29.2% | 52 | 0.233 | 0.804 | LOWER |
| futures5x | 0.3 | $987 | 0.08 | 27.2% | 52 | 0.205 | 0.707 | LOWER |
| futures5x | 0.4 | $1,055 | 0.24 | 23.6% | 52 | 0.177 | 0.611 | LOWER |
| futures5x | 0.5 | $1,057 | 0.26 | 16.9% | 42 | 0.141 | 0.488 | LOWER |

**Mean exposure vs v4: LOWER at every single `lam` tried, on both
markets, monotonically with `lam`.** This is not incidental — it is
structural: `mult ∈ [1-lam, 1]` can only reduce exposure, so "matched"
exposure is impossible by construction at any `lam > 0`. Every drawdown
reduction in this table is therefore, by the project's own repeated
finding (L-04/R-33, R-31, R-32), the primary suspect for *why* drawdown
falls — before any claim about gate quality is examined.

On spot the return response is a clean, monotone decline with `lam`
(Sharpe 0.14 → 0.10 → 0.08 → 0.07 → 0.06 → 0.02) tracking the monotone
exposure decline almost exactly — consistent with a pure de-lever, not a
regime-conditional effect. On futures the same is mostly true except a
local spike at `lam=0.2` (Sharpe 0.36 against neighbours 0.17 and 0.08) —
exactly the kind of single-point spike the routine warns not to select
on; it is not surrounded by a plateau and does not appear on spot.

## 3. Recommended lam and why

**lam = 0.3.** Not the best point in the sweep (that would be a form of
selecting on the very noise the routine warns about — see the `lam=0.2`
futures spike above, and `lam=0.5`'s best absolute Sharpe, which also
carries the largest exposure cut and is the most mechanically "just holds
less"). `lam=0.3` is: (a) the pre-registered default named in this task's
own design; (b) the middle of the swept grid, away from both extremes;
(c) on the axis that actually matters here (mean-exposure ratio), it sits
on the same smooth, monotone curve as every other `lam`, so there is no
sense in which 0.3 is a peak being cherry-picked — the whole sweep is one
plateau-that-isn't, moving together with exposure. No `lam` in this sweep
is defensible as "the gate found something," which is itself the finding
(see §6 verdict).

## 4. Full comparison at lam=0.3: inner-train + inner-validation, both markets

| split | market | strategy | final | (%) | trades | DD | Sharpe |
|---|---|---|---|---|---|---|---|
| inner-train | spot | `kelly_regime_v4` | $18,477 | +1747.7% | 72 | 43.3% | 2.03 |
| inner-train | spot | `v5_damp(0.3)` | $9,805 | +880.5% | 62 | 31.6% | 2.06 |
| inner-train | spot | `buy_and_hold` | $29,803 | +2880.3% | 1 | 84.1% | 1.38 |
| inner-train | futures5x | `kelly_regime_v4` | $30,344 | +2934.4% | 72 | 35.3% | 2.28 |
| inner-train | futures5x | `v5_damp(0.3)` | $9,762 | +876.2% | 62 | 26.4% | 2.06 |
| inner-train | futures5x | `buy_and_hold` | $18 | −98.2% | 1 | 99.0% | −0.29 (LIQUIDATED) |
| inner-validation | spot | `kelly_regime_v4` | $998 | −0.2% | 52 | 33.2% | 0.14 |
| inner-validation | spot | `v5_damp(0.3)` | $985 | −1.5% | 52 | 26.2% | 0.07 |
| inner-validation | spot | `buy_and_hold` | $574 | −42.6% | 1 | 77.3% | 0.08 |
| inner-validation | futures5x | `kelly_regime_v4` | $1,064 | +6.4% | 52 | 32.3% | 0.25 |
| inner-validation | futures5x | `v5_damp(0.3)` | $987 | −1.3% | 52 | 27.2% | 0.08 |
| inner-validation | futures5x | `buy_and_hold` | $18 | −98.2% | 1 | 99.8% | 0.43 (LIQUIDATED) |

`v5_damp` never beats `kelly_regime_v4` on return in any of the four
cells; it beats it on drawdown in all four (by an amount matched almost
exactly by its exposure cut — see §6).

## 5. ETH/BTC falsification (Bitfinex, R-17's window, lam=0.3)

| asset | market | strategy | final | (%) | trades | DD | Sharpe |
|---|---|---|---|---|---|---|---|
| BTC (control) | spot | `kelly_regime_v4` | $12,278 | +1127.8% | 62 | 40.1% | 1.86 |
| BTC (control) | spot | `v5_damp(0.3)` | $8,033 | +703.3% | 53 | 33.2% | 1.95 |
| BTC (control) | futures5x | `kelly_regime_v4` | $25,681 | +2468.1% | 62 | 32.1% | 2.19 |
| BTC (control) | futures5x | `v5_damp(0.3)` | $11,184 | +1018.4% | 53 | 27.0% | 2.10 |
| ETH (test) | spot | `kelly_regime_v4` | $5,482 | +448.2% | 75 | 36.5% | 1.48 |
| ETH (test) | spot | `v5_damp(0.3)` | $2,956 | +195.6% | 60 | 27.6% | 1.27 |
| ETH (test) | futures5x | `kelly_regime_v4` | $4,263 | +326.3% | 75 | 35.1% | 1.25 |
| ETH (test) | futures5x | `v5_damp(0.3)` | $2,624 | +162.4% | 60 | 33.1% | 1.09 |

Ordering vs v4, all four cells:

| asset | market | return | drawdown |
|---|---|---|---|
| BTC (control) | spot | worse | better |
| BTC (control) | futures5x | worse | better |
| ETH (test) | spot | worse | better |
| ETH (test) | futures5x | worse | better |

**The ordering is identical on BTC control and ETH test — it does not
flip.** That is expected and not much comfort: a scalar de-lever (§6)
would replicate its risk/return trade-off on any asset by construction,
so this falsification test cannot by itself distinguish "the gate works
and generalizes" from "it's a flat haircut and haircuts generalize
trivially."

## 6. Turnover

Trade counts, v4 vs `v5_damp(0.3)`, everywhere measured:

| window | market | v4 trades | v5_damp trades |
|---|---|---|---|
| inner-train | spot | 72 | 62 |
| inner-train | futures5x | 72 | 62 |
| inner-validation | spot | 52 | 52 |
| inner-validation | futures5x | 52 | 52 |
| BTC control (Bitfinex) | spot/futures | 62 | 53 |
| ETH test (Bitfinex) | spot/futures | 75 | 60 |

**Turnover never increases; it is equal or lower everywhere (≈0.79–1.0x
v4).** No fee-drag flag (L-14/L-15/L-16 territory not entered here).

## The redundancy / artifact check — the honest headline result

Two separate tests, both on inner-validation spot, both requested by the
task:

**(a) Correlation between the confidence signal and v4's discrete vote.**
`corr(conf, vote_frac) = 0.097` over 209,953 bars — **low**, not the
`>0.8` that would flag naive redundancy with the discrete anchor vote.

**(b) But the confidence signal has almost no independent dynamic range
once smoothed enough to be usable.** `conf` (post floor + 3-day EMA) has
mean 0.025, std 0.012, max 0.087 on inner-validation — i.e. it lives in a
band about a tenth as wide as `[0,1]`. That collapses the dampener's
practical effect: with `mult = 1 - lam*(1-conf)`, `mult`'s own std is only
`lam * std(conf) ≈ 0.3 * 0.012 = 0.0036` around a mean of `≈ 1-lam`.
Checked directly against `kelly_regime_v4`'s own `target` series on the
same window:

| quantity | value |
|---|---|
| corr(`v5_damp.target`, `v4.target`) | 0.9986 |
| corr(`v5_damp.target`, `0.7 × v4.target`) | 0.9986 |
| R² of `v5_damp.target` ~ `0.7 × v4.target` | 0.9971 |
| mean\|`v5_damp.target` − `0.7 × v4.target`\| | 0.0043 |
| fraction of bars differing from `0.7 × v4.target` by >1% | 5.9% |

**`v5_damp` at `lam=0.3` is, to R²=0.997, indistinguishable from a flat
`0.7×` rescaling of `kelly_regime_v4`'s own exposure.** The confidence
axis is not redundant with the discrete vote in the naive
high-correlation sense (a) — but it fails a stricter and more damning
test: once smoothed enough to avoid whipsaw, the Bayesian margin barely
moves relative to its own mean on this data, so multiplying by it is
statistically almost the same operation as multiplying by a constant.
This is a *third* outcome, distinct from both "redundant with the vote"
and "a genuine regime-conditional gate": the confidence *variable* is not
duplicating information already in `frac`, but the *dampener built from
it*, at the smoothing needed to be tradeable, degenerates to a scalar —
which is exactly the L-04/R-33/R-31/R-32 exposure-level artifact, self-
manufactured rather than merely suspected.

## Verdict

**This is not a promotion candidate, and it is not obviously worth
spending the holdout on.** The pre-registered falsification test (ETH
ordering) did not kill it — the ordering replicates — but that test
cannot discriminate a real gate from a flat de-lever, and the direct check
built to answer that question (§ above) shows the actual mechanism *is*
close to a flat de-lever: R²=0.997 against `0.7 × v4`. Every metric that
"improves" (drawdown, in every cell, on both real assets) moves in lockstep
with mean exposure, which is strictly lower at every `lam` by construction
— never matched, because the dampener is architecturally incapable of
matching (it can only shrink). Given this project's own standing
diagnosis — three prior findings (L-04's headline via R-33, R-28/R-31,
R-32) that turned out to be exposure levels rather than gate quality — the
honest reading is that `kelly_regime_v5_damp` at any `lam` tested here
reproduces that same artifact a fourth time, using a new source signal
that happens not to add anything once it is smoothed enough to be usable.
It also never beats `kelly_regime_v4` on return in any of the 12 cells
measured (4 inner-split cells, 4 ETH/BTC cells, 4 sweep cells beyond
lam=0.3), so it does not clear this project's own promotion bar (beat the
incumbent, not just the benchmark) even before reaching the holdout
question.

What would make this worth revisiting: a confidence source with more
independent dynamic range after the smoothing a tradeable signal requires
— i.e. not this specific (mu=0.15, stick=0.985) Harsanyi parametrization,
whose posterior on 5-minute BTC bars is apparently too close to flat once
it survives multi-day smoothing to modulate anything. Absent that, L-12's
stated hypothesis — "as a sizing input, not a direction signal, it might
work" — is **not supported** by this test: it was given every structural
advantage (bounded, strictly risk-reducing, matched to v4's own vote and
scale, causally verified) and what it does with that advantage is
reproduce a scalar.

**Configurations evaluated this session: 1 correctness check + 10
(lam sweep, 5 values × 2 markets) + 1 chosen-lam full comparison (4 cells)
+ 1 ETH falsification (4 cells) = 5 distinct `lam` values swept, plus the
`lam=0` build check = 6 distinct configurations of this strategy in
total.** (No parameter other than `lam` was searched; `conf_span_days=3`
was fixed a priori and not swept.)

---

## Appendix: full `experiments/kelly_regime_v5_damp.py`

```python
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
```
