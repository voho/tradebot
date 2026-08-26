"""R-152 frozen pre-registration: Conditional Drawdown-at-Risk (CDaR)
sizing for `kelly_regime_v4`'s SCALE/CAP axis — two branches.

**The idea.** Chekhlov, Uryasev & Zabarankin (2005, *Int. J. Theor. Appl.
Finance* 8(1)) define CDaR_beta as the mean of the worst (1-beta) fraction
of drawdowns on an equity path — a convex, LP-representable risk measure
computed on the PATH, not on the return distribution (that is CVaR,
already tried and rejected: R-125). Krokhmal, Uryasev & Zrazhevsky (2003,
*J. Risk*) give the companion optimization machinery. Unlike every prior
regime-timing round in this ledger (HMM through LPPLS, all closed by the
six-episode detection-lag / N=3 problem), CDaR never tries to *detect* a
rare event: it is a rolling statistic of an already-observed, continuously
sampled quantity, so it is well-posed at any sample size. It sits on the
SCALE/CAP slot only — the vote (`frac`, i.e. WHEN to be exposed) is
untouched in both branches, preserving R-62's isolation discipline.

**Constraint attacked: N≈3.** Converts a scarce-event-detection problem
(how many regime changes has this asset had) into a continuous-estimation
problem (how has the strategy's own drawdown path behaved), which is the
only lever this ledger has found that does not need more regime events to
work.

**Not a duplicate of:**
- R-11 (Grossman & Zhou 1993 continuous-time cushion, PARTIAL) / R-46
  (CPPI fixed & Hurst-adaptive, NEGATIVE) — both are floor-plus-multiplier
  heuristics with hand-set free parameters; CDaR is a risk statistic
  computed from the empirical drawdown distribution, no floor/multiplier.
- R-125 (CVaR sizing, NEGATIVE both branches) — CVaR operates on the
  *return* distribution's tail; CDaR operates on the *drawdown path*, a
  different, path-dependent object. A strategy can have benign return-tail
  CVaR and severe CDaR or the reverse.
- R-97 (Wasserstein-DRO, NEGATIVE) — minimax over a distribution family,
  no drawdown-path component at all.
- Not a regime-timing round (BOCPD/HMM/HSMM/LPPLS family, all closed
  today, 08-26) — no six-episode detection-lag gate applies here, because
  nothing is being detected; it is a rolling statistic.

**Simulability (step 1, q3).** Fully simulable on this repo's 5m OHLCV:
both branches derive their CDaR series from a *reference return stream*
computed inside `prepare()` from columns already available there
(`close`, and `kelly_regime_v4`'s own vote/vol columns) — no order book,
no new data channel. Both are causal by construction: the CDaR at bar `i`
uses only the reference series through bar `i-1` (`shift(1)` applied to
the *inputs* of the rolling window, not to its output — see
`rolling_cdar` below for exactly where the shift sits).

**Falsification, named now, before any code ran.** If a rolling empirical
CDaR is in practice just a smoothed, lagged transform of realized
volatility, then per R-62's scale-factor isolation finding the SCALE/CAP
slot carries none of v4's edge regardless of what feeds it, and both
branches should reproduce the four-times-replicated SIZE-axis null
(matched-exposure drawdown property 0/2 on BTC/ETH, near-zero cross-asset
panel score). This is measured directly as **diagnostic B2** below,
BEFORE either branch's headline number is read, and reported regardless
of what it shows.

---

**THE TWO BRANCHES (frozen; only the target of the CDaR construction
differs — everything else is v4's own unmodified vote and hysteresis).**

- **Conservative — dynamic leverage cap.** Replace v4's fixed
  `max_leverage = 2.0` constant with a rolling causal CDaR_0.95, computed
  over a REFERENCE wealth path built from v4's own unmodified vote-scaled
  return series (`frac[i] * r[i]`, i.e. the raw signal before any sizing
  is applied — not the executed, capped position, so there is no
  circularity between the derived cap and the path used to derive it).
  The derived cap is rescaled so its INNER-TRAIN mean equals 2.0 (matches
  v4's own constant, so the branch changes *when* leverage tightens, not
  the average amount of it). Falsification test: **survives on ETH**
  (`scripts/build_bitfinex_dataset.py`) — this ledger's cheapest, most
  decisive discriminator for a cap/scale-slot construction (R-33, R-57,
  R-125 all failed here).

- **Novel — CDaR-budgeted exposure.** Replace v4's vol-target ratio
  (`target_vol / realized_vol`) with a direct solve: the largest exposure
  fraction `f` such that the rolling-window CDaR_0.95 of `f * (unit vote-
  scaled returns)` does not exceed a fixed CDaR budget. CDaR is
  homogeneous of degree 1 in `f` for a fixed return path
  (Chekhlov/Uryasev/Zabarankin 2005, Prop. 3), so this is closed-form:
  `f* = budget / CDaR_0.95(unit_returns)`, no iterative optimizer needed.
  `budget` is calibrated on INNER-TRAIN so the branch's mean exposure
  matches v4's own inner-train mean exposure (R-33's matched-risk
  discipline, paid up front rather than argued about after the fact).
  Falsification test: **survives the Monte Carlo stress windows**
  (`scripts/stress_test.py`) — the relevant test for a path-dependent, MC-
  window-varying mechanism; showing no differentiated behaviour there
  would mean the branch never actually engaged its own mechanism.

Both branches keep `frac` (the vote), the anchors, the hysteresis state
machine, the 10% deadband and `target_vol` byte-identical to
`kelly_regime_v4`. Only the leverage-limiting term changes.

---

**Shared machinery: `rolling_cdar`.**

Given a 1-D array of per-bar simple returns `x` (already representing
whatever unit exposure is being risk-measured) and a trailing window of
`window_bars`:

1. Build the window-local wealth path `W(t) = cumprod(1 + x)` over a
   rolling buffer (a plain expanding cumprod reset every `window_bars` is
   NOT used — see the note below on why a full rolling-window peak is
   required, not a periodically-reset one).
2. `peak(t) = rolling_max(W, window_bars)` — the running peak *within*
   the trailing window only, so a single old high does not suppress the
   measure forever (which is exactly what a global running peak would do
   over a 9-year series).
3. `dd(t) = (peak(t) - W(t)) / peak(t)`, clipped to `>= 0`.
4. `CDaR_beta(t)` = mean of the values of `dd` in the trailing
   `window_bars` window that are `>=` the window's own `beta`-quantile of
   `dd` (the mean of the worst `(1-beta)` fraction, the definition in
   Chekhlov/Uryasev/Zabarankin 2005 Def. 2). Implemented as a rolling
   quantile (`pandas.Series.rolling(window_bars).quantile(beta)`) followed
   by a masked rolling mean — both operations are causal (`.rolling`
   never reads ahead of the current row) and this is the ONLY place
   `beta` enters.
5. The whole result is additionally `.shift(1)`-ed before being used by
   any strategy `target` computation, so bar `i`'s cap/budget uses only
   information available at the close of bar `i-1` — the same discipline
   `kelly_regime.KellyRegime.prepare` already applies to its own `vol`
   column (`.shift(1)` at line ~91 of `kelly_regime.py`).

`CDAR_BETA = 0.95` (worst 5% of drawdowns) and `CDAR_WINDOW_DAYS = 365`
(one trailing year) are frozen now, not tuned; a plateau check across
`CDAR_WINDOW_DAYS in (180, 365, 545)` is REQUIRED before any promotion
claim per ROUTINE.md's "plateau, not a peak" bar, and is part of the
declared configs count below.

---

**PRE-REGISTERED DECISION RULE (frozen before either branch runs).**

All development happens on data before `OOS_START` (2023-01-01) only.
Iterate freely against inner-train (2017-01-01 -> 2020-12-31); select
between the window-length sweep on inner-validation
(2021-01-01 -> 2022-12-31). Both markets use futures 5x (v4's own
comparison market — see `docs/STRATEGIES.md`).

**Diagnostic B2 (run first, reported regardless of outcome).** Pearson
correlation, on inner-train, between the branch's own CDaR-derived
series (the dynamic cap for conservative; the derived `f*` for novel) and
v4's own realized-volatility series (`vol`, the same column
`kelly_regime.KellyRegime.prepare` computes). `|r| >= 0.85` is flagged as
"likely a smoothed relabeling of realized volatility" in the write-up —
this does not by itself REJECT the branch (a construction can be
correlated with vol and still add something at the margin), but any
result under a flagged B2 must say so explicitly rather than claim a
qualitatively new mechanism.

**Selection rule (inner-validation, 2021-01-01 -> 2022-12-31, futures
5x, vs the unmodified `kelly_regime_v4` control at its own registered
defaults).** A branch is **eligible for holdout** iff ALL of:

1. `time_in_market_pct` and mean realized exposure are within 15
   percentage points of v4's own control (R-33's matched-risk gate — an
   unmatched arm is voided, not scored, per the standing rule).
2. Sharpe does not fall by more than the ±0.2 noise floor (R-20) relative
   to the control, OR max-drawdown improves by more than 3 percentage
   points at matched exposure.
3. The window-length sweep (180/365/545 days) is a plateau: at least 2 of
   3 window lengths agree in the SIGN of the Sharpe delta vs control.

A branch that fails any of (1)-(3) is **NEGATIVE at inner-validation**;
its holdout counter contribution is **+0** and no holdout bar is read for
it. This is decided now, not after seeing inner-validation numbers, so
that "eligible for holdout" cannot be redefined post hoc.

**Promotion rule (holdout, 2023-01-01 ->, read ONLY for a branch that
passed the selection rule above).** Promote a branch iff ALL of
ROUTINE.md's standing bar:

- beats `buy_and_hold` out-of-sample on futures 5x after real costs
  (0.10% tier; the 0.40% Bitstamp tier per `scripts/fee_study.py` is
  reported as a caveat, not a promotion gate, matching how v4 itself was
  promoted);
- the Sharpe improvement over the unmodified `kelly_regime_v4` control
  exceeds the ±0.2 noise floor, OR is a drawdown/tail improvement at
  matched exposure (R-33 gate again, on the holdout slice);
- survives its own named falsification test (ETH for conservative, MC
  stress windows for novel);
- the parameter neighbourhood (the window-length sweep) is a plateau on
  holdout too, not just inner-validation.

Anything else on a branch that reached holdout is **NEGATIVE**, reported
with the same care as a promotion, per ROUTINE.md step 4.

**n-check (R-78's discipline).** The selection rule's Sharpe-delta gate
(2) is measured on 730 inner-validation days — the same window every
SIZE-axis round in this ledger uses (R-33, R-57, R-125, R-145, R-150,
R-151) — so no new power calculation is needed; it is already the
project's standard-powered comparison.

**Configs evaluated (declared before running, for the trials count).**
Per branch: 3 window lengths (180/365/545 days) x 1 CDaR beta (0.95,
fixed) x {inner-train fit, inner-validation score} = 3 scored
configurations, plus 1 control re-run (`kelly_regime_v4` unmodified) =
**4 per branch, 8 total** at the selection stage. Any branch reaching
holdout adds 1 more run (its selected window length only — no further
search) x {holdout score, ETH or stress-test falsification} = up to 2
more per promoted-eligible branch. **Ceiling: 12 total** if both branches
reach holdout; **8 total** if neither does.

**Holdout counter: +0 at this file's freeze.** Only incremented, and only
by exactly the branches that pass the selection rule, when the operator
runs the promotion-rule step.

**What would make this whole direction fail (named before any code
ran).** (a) Diagnostic B2 shows both branches are >=0.85-correlated with
plain realized volatility, in which case any headline effect likely
reproduces the already-closed vol-targeting result rather than something
new; (b) the window-length sweep is not a plateau (a spike at one window
length only) — read as overfitting to that specific horizon; (c) the
matched-exposure gate fails, meaning any apparent drawdown improvement is
R-33's already-diagnosed "holding less" artifact; (d) `rolling_cdar`
cannot be made both correct and fast enough to run the full 2017-2026
series in reasonable time, in which case this is filed as a BLOCKED
methodology round, not silently narrowed to a shorter window without
saying so.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402

#: Frozen constants — see the module docstring for why these values.
CDAR_BETA = 0.95
CDAR_WINDOW_DAYS_DEFAULT = 365
CDAR_WINDOW_DAYS_SWEEP = (180, 365, 545)

#: Selection-rule thresholds (inner-validation, 2021-01-01 -> 2022-12-31).
EXPOSURE_MATCH_TOL_PP = 15.0
SHARPE_NOISE_FLOOR = 0.2
DRAWDOWN_IMPROVEMENT_PP = 3.0
B2_CORRELATION_FLAG = 0.85

#: Splits, matching scripts/experiment.py and docs/ROUTINE.md exactly.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"


def rolling_cdar(x: np.ndarray, window_bars: int, beta: float = CDAR_BETA,
                  recompute_every: int = BARS_PER_DAY) -> np.ndarray:
    """Causal rolling CDaR_beta of the wealth path built from returns ``x``.

    ``x`` is a 1-D array of per-bar simple returns for whatever unit
    exposure is being risk-measured (e.g. ``frac * r`` for v4's own vote-
    scaled, unsized return stream). Returns an array the same length as
    ``x``; entry ``i`` uses only ``x[:i]`` (see the ``.shift(1)`` at the
    end) so it is safe to assign directly into a ``prepare()`` column.

    **Definition, and why the exact form matters.** For the window ending
    at bar ``i``, CDaR_beta(i) is the mean of the values of ``dd`` (the
    window-local drawdown series — running peak WITHIN the window, not a
    global one, so one old high does not suppress the measure forever)
    that fall at or above THAT SAME WINDOW's own beta-quantile threshold
    (Chekhlov, Uryasev & Zabarankin 2005, Def. 2). An earlier version of
    this function computed the beta-quantile with one rolling call and
    the tail-mean with a second, independent rolling call — each position
    ``j``'s quantile came from the window ENDING AT ``j``, so summing a
    per-``j`` pointwise "is dd[j] above its own trailing quantile" flag
    over yet another window silently mixed observations governed by
    different reference windows. Caught by exactly the smoke test
    ROUTINE.md step 2 asks for before any variant runs on real data: the
    "tail" flag rate came out at 19% (should be ~5%, no crash) and then
    at exactly 0% (should still be ~5%) once a large synthetic drawdown
    entered the window — a silently wrong risk measure, not a crash.

    **Fix: exact per-window computation, recomputed every
    ``recompute_every`` bars (default: daily) and forward-filled between
    recomputes.** CDaR of a slowly-decaying window statistic does not
    move meaningfully bar-to-bar (a 5-minute-bar window shift changes one
    of ``window_bars`` ~= 100k+ observations), so recomputing daily and
    holding the value between recomputes costs a negligible amount of
    reintroduced staleness against an ~365x reduction in the number of
    full-window sorts (`O(n / recompute_every)` full sorts of size
    ``window_bars`` each, via ``np.partition`` — no pandas rolling call
    with a non-associative reduction is used at all, closing off the
    class of bug above by construction).
    """
    n = len(x)
    wealth = np.cumprod(1.0 + np.nan_to_num(x, nan=0.0))
    cdar = np.full(n, np.nan)
    k = max(1, int(round((1.0 - beta) * window_bars)))
    for i in range(window_bars - 1, n, recompute_every):
        w = wealth[i - window_bars + 1: i + 1]
        peak = np.maximum.accumulate(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.clip((peak - w) / peak, 0.0, None)
        # np.partition is O(w); the last k elements after partitioning at
        # -k are exactly "some ordering of the k largest", which is all a
        # mean of the tail needs (no full sort required).
        tail = np.partition(dd, -k)[-k:]
        cdar[i] = tail.mean()
    # ffill holds the last exact recompute between recompute points; bars
    # before the first recompute (index < window_bars - 1) have no prior
    # valid value to fill from and correctly stay NaN.
    filled = pd.Series(cdar).ffill()
    return filled.shift(1).to_numpy()


def _rolling_cdar_bruteforce(x: np.ndarray, window_bars: int, beta: float = CDAR_BETA) -> np.ndarray:
    """Unvectorized reference implementation, one full sort per bar.

    Used only by the test in this module to check ``rolling_cdar``
    against ground truth on a short synthetic series — too slow to run
    on the real ~1M-bar dataset, which is exactly why the vectorized
    version above exists.
    """
    n = len(x)
    wealth = np.cumprod(1.0 + np.nan_to_num(x, nan=0.0))
    out = np.full(n, np.nan)
    k = max(1, int(round((1.0 - beta) * window_bars)))
    for i in range(window_bars - 1, n):
        w = wealth[i - window_bars + 1: i + 1]
        peak = np.maximum.accumulate(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.clip((peak - w) / peak, 0.0, None)
        out[i] = np.sort(dd)[-k:].mean()
    return pd.Series(out).shift(1).to_numpy()


def window_bars(days: int) -> int:
    return int(days * BARS_PER_DAY)


__all__ = [
    "B2_CORRELATION_FLAG",
    "CDAR_BETA",
    "CDAR_WINDOW_DAYS_DEFAULT",
    "CDAR_WINDOW_DAYS_SWEEP",
    "DRAWDOWN_IMPROVEMENT_PP",
    "EXPOSURE_MATCH_TOL_PP",
    "INNER_TRAIN_END",
    "INNER_VAL_END",
    "INNER_VAL_START",
    "OOS_START",
    "SHARPE_NOISE_FLOOR",
    "rolling_cdar",
    "window_bars",
]
