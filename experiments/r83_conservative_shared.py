"""R-83 conservative branch — shared, frozen infrastructure for B-38.

Committed BEFORE any number in this round was computed, per ROUTINE.md's
own discipline and the pattern R-78 set for this exact backlog item
(``experiments/r78_shared.py``). This module carries the round's
pre-registration verbatim and the machinery both the frozen-arm test and
the rolling-arm test read, so neither can quietly redefine the
measurement after seeing a result.

=====================================================================
BACKLOG ITEM — B-38, filed verbatim in docs/LEDGER.md section D
=====================================================================

"Pre-register and record a risk-matched forward comparison instead of the
raw one B-06 has been recording: pair `kelly_regime_v4` against a passive
long carrying v4's own mean notional (R-33's matched benchmark,
per-window matched, not a fully-invested hold), so the paired daily
difference stops carrying ~0.6-0.7 of BTC's own move as common-mode
variance. Decide in advance what horizon would make it worth continuing."

Constraints attacked: **N≈3** (effective sample size — B-06 is this
project's only route to more of it) and **ERR** (no error control in the
signal path — the anytime-valid tool is the error control; this round
asks whether it can ever fire on what B-06 currently records).

Not a duplicate of:

- **R-33 / `experiments/matched_hold.py`** — built the matched-benchmark
  machinery for the *backtest* comparison (B-13/B-14) and found freezing
  an exposure match on one period fails out-of-window (5 of 6 holdout
  cells voided by its own V1 gate; see ``reports/matched_hold/holdout.csv``
  row-by-row: the frozen ``notional-matched`` arm's vol gap on 2023+ is
  58.0% (spot) / 63.6% (futures) against a 20% bar). This round applies
  that same lesson to the *forward* comparison B-06 records, which R-33
  never touched.
- **R-78 / `experiments/r78_matched_arm_sizing.py`** — the addendum this
  item was filed from. Explicitly labelled "not the pre-registered B-38
  round" in its own docstring. Its construction computes v4's mean
  notional *separately inside each measurement window* — inner-train and
  inner-validation each get their own ``frac`` — which is a look-behind
  match: it uses the window's own future exposure history to size the
  benchmark for that same window. That is a legitimate backtest
  question (it costed the idea before filing it) but it is **not
  deployable as a live forward arm**, because a live recorder cannot
  know its own trailing multi-year mean notional before that time has
  elapsed. This round tests a construction that IS deployable: a benchmark
  whose exposure at time `t` is a function of information strictly before
  `t`.
- **R-71** (built the anytime-valid tool and scheduled B-06; never
  proposed a matched arm).

=====================================================================
PRE-REGISTRATION 1a — the matched-benchmark construction
=====================================================================

Three arms are built and compared against `kelly_regime_v4` on spot, at
both fee tiers (0.10% table / 0.40% live), on both inner splits:

**Arm A — `buy_and_hold` (the status quo).** What B-06 currently
records. Reproduces the ~0.6–0.7 common-mode-variance problem R-78 named.

**Arm B — frozen single mean-notional match (the addendum's shortcut, made
deployable).** A `ConstantExposureHold` (``experiments/matched_hold.py``,
R-33's construction, unmodified, imported not copied) at a SINGLE constant
`c_global`, computed once from `kelly_regime_v4`'s mean notional over
inner-train ONLY (2017-2020 — the only span that is chronologically prior
to inner-validation, i.e. the only span a real deployment could have
calibrated on before 2021). This is deliberately the naive, literal
reading of "hardcode v4's mean notional into the recorder": calibrate once
on history available at deploy time, then never touch it again. R-33's own
finding (freezing an exposure match on one period fails 5 of 6 holdout
cells) predicts this arm mismatches risk once the regime it was calibrated
on ends — tested directly in step 3 below by asking it to transfer from
inner-train into inner-validation, the sharpest regime change in the
dataset short of the holdout itself.

**Arm C — rolling causal match (this round's actual proposal for B-38).**
A new `RollingMatchedHold` strategy (built in this file): at every bar its
target exposure is a causal EWM (90-day halflife, one extra bar of lag
beyond the causality the underlying `target` column already carries) of
`kelly_regime_v4`'s own realized exposure, using only bars strictly before
the current one. It requires no calibration period, no freeze date and no
re-solve — it is the same computation a live recorder could run forever,
continuously, off the exact CSV history `scripts/paper_trade.py` already
writes for `kelly_regime_v4` (its `new_target` column). This is
deliberately chosen over interpolating R-33's `insplit`/`windows`
machinery (which resolves a fresh constant per 90–730 day *backtest*
window using a probe search) because that machinery is a research tool
for measuring a completed window, not a rule a forward recorder can run
online. A rolling causal average is the online equivalent of "match inside
each window" — R-33's own better-performing construction — rather than of
the frozen single constant its own ledger entry (B-13) found failed.

Decision made deliberately: **build (ii), the rolling design**, per the
task's instruction to attempt a real improvement rather than re-read the
addendum. The 90-day halflife is chosen, not fitted: it sits inside R-33's
own already-validated 90–730 day window range, close to the short end for
responsiveness to the ~4–5 day mean gap between v4's own rebalances
(332 fills / ~4 years on the addendum's data), and it is fixed here,
before any noise or horizon number in this file is computed, exactly once.

=====================================================================
PRE-REGISTRATION 1b — the decision rule (frozen before any horizon is read)
=====================================================================

The statistic is the median first-exclusion horizon `n50` (trading days,
paired daily-return difference `kelly_regime_v4 − arm`) from
`tradebot.inference.empirical_bernstein_confidence_sequence` +
`anytime_valid_first_exclusion`, read off 400 stationary-bootstrap paths
(30-day mean block, this project's own established convention) built from
the inner-validation, 0.40%-live-fee difference series for **Arm C** (the
rolling design — the arm actually being proposed for the recorder). The
25-year horizon and the 400-path count are R-78's own choices, reused
unchanged for comparability rather than re-derived, since nothing about
which benchmark is paired changes what the tool needs to resolve a mean
against a given noise level.

**Why this project's own history sets the bar, not a number that merely
sounds reasonable (ROUTINE.md step 2's rule, applied here explicitly).**
This project's forward evidence source is one scheduled GitHub Actions
workflow polling a public exchange feed — it accrues on real calendar
days, not on research-session count (82 rounds have run in the ~9
elapsed calendar days since R-1, but B-06 itself is "~2 days old" per
R-78 regardless of how many sessions ran in that time). The quantity that
has to be fundable is therefore wall-clock years of a human research
program still checking in on an unattended recorder, not sessions. R-78
derived 3 years / 25 years for the *raw* comparison by asking what a
single-operator repo could plausibly still be reading in on: 3 years is
inside a typical multi-year research program's own lifespan (this repo
crossed from R-1 to R-83 in under two weeks of session time but the
underlying strategies it studies were built to run for years); 25 years
is close to the outer edge of "worth a standing recommendation" at all —
beyond it, the honest description is "will not resolve inside anyone's
tenure on this project." Nothing about matching the benchmark's risk
changes that argument, so the same two thresholds are re-used rather than
re-invented:

- **ON TRACK** — `n50 <= 1095` (3 years) AND ≥50% of paths exclude zero
  within 5 years.
- **SLOW BUT VIABLE** — `n50 <= 9125` (25 years).
- **NOT VIABLE AS SPECIFIED** — `n50 > 9125`, or fewer than 50% of paths
  exclude zero within 25 years.

**"Worth continuing" (the literal question B-38 was filed to answer) is
read off this classification directly:** ON TRACK or SLOW BUT VIABLE
means yes — describe how the arm should be added to the recorder. NOT
VIABLE AS SPECIFIED means no, and the honest conclusion is that no
forward comparison of this family is settleable on any horizon a session,
or a series of sessions, can wait for.

=====================================================================
PRE-REGISTRATION 1c — what would make this fail, named now
=====================================================================

Three ways, all named before any number below is computed:

1. **The addendum's own worry reproduces exactly**: the rolling match cuts
   sd but the *mean* of `v4 − Arm C` shrinks by a similar or larger
   fraction, because most of what looks like "timing edge" against a
   fully-invested hold is itself an artifact of the exposure gap and
   evaporates once exposure is matched — leaving `n50` unchanged or worse
   despite the lower noise. This is the addendum's own finding on
   inner-train (7.1y → 6.2y, i.e. almost no gain) generalizing to
   inner-validation as well.
2. **R-33's transfer failure reproduces on the notional axis too, for
   BOTH arms, not just the frozen one** — i.e. even the rolling design's
   90-day EWM cannot track a regime change fast enough to keep the vol gap
   inside a usable band, so "genuinely rolling" turns out not to rescue
   anything R-33's frozen construction lost. Tested directly in step 3.
3. **The falsification test (chosen now, this round's pre-registered
   one, per ROUTINE.md step 2): does Arm B (frozen on inner-train) fail
   to transfer into inner-validation while Arm C (rolling) tracks it
   acceptably?** If Arm C's vol gap on the inner-train→inner-validation
   transition is not materially smaller than Arm B's, the "genuinely
   rolling" claim has nothing behind it beyond a good story, and this
   round's contribution over the addendum is zero.

=====================================================================
HOLDOUT DISCIPLINE
=====================================================================

Every frame this module hands to either script is truncated at
2022-12-31 23:59:59 (``r78_shared.load_truncated`` / ``assert_no_holdout``,
imported and reused, not re-implemented, so the truncation logic cannot
drift between rounds). Expected holdout increment for this round: **+0**.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.r78_shared import (  # noqa: E402
    FEE_LIVE,
    FEE_TABLE,
    TRADING_DAYS,
    W_TRAIN,
    W_VAL,
    assert_no_holdout,
    bootstrap_paths,
    load_truncated,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.inference import (  # noqa: E402
    anytime_valid_first_exclusion,
    daily_returns,
    empirical_bernstein_confidence_sequence,
)
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# The anytime-valid tool's parameters — R-78's, reused unchanged (see 1b).
N_PATHS = 400
ALPHA = 0.05
BOUND = 0.5   # two unleveraged spot arms differ by at most ~a day's own move
HORIZON_DAYS = TRADING_DAYS["25y"]

INCUMBENT = "kelly_regime_v4"

__all__ = [
    "ALPHA", "BARS_PER_DAY", "BARS_PER_YEAR", "BOUND", "FEE_LIVE",
    "FEE_TABLE", "HORIZON_DAYS", "INCUMBENT", "N_PATHS", "TRADING_DAYS",
    "W_TRAIN", "W_VAL", "RollingMatchedHold", "assert_no_holdout",
    "bootstrap_paths", "cs_horizon", "diff_stats", "load_truncated",
    "paired_diff", "v4_exposure_series",
]


# --------------------------------------------------------- exposure helpers

def v4_exposure_series(df: pd.DataFrame) -> pd.Series:
    """`kelly_regime_v4`'s own causal target, clipped to spot's [0, 1] cap.

    ``prepare()`` is required to be causal (row i uses only rows <= i,
    enforced project-wide by ``tests/test_causality_strict.py`` for
    registered strategies); this reads that column back rather than
    re-deriving it, so any future change to v4's own gate is picked up
    automatically instead of silently drifting out of sync.
    """
    v4 = get_strategy(INCUMBENT)
    prepared = v4.prepare(df.copy())
    if "target" not in prepared.columns:
        raise AssertionError(f"{INCUMBENT} lost its target column")
    return prepared["target"].clip(lower=0.0, upper=1.0)


# --------------------------------------------------------------- Arm C: rolling

class RollingMatchedHold(Strategy):
    """A passive long whose exposure tracks v4's own TRAILING mean notional.

    The deployable analogue of R-33's "match inside each window": rather
    than a constant solved once (Arm B) or a look-behind constant solved
    from the very window being measured (the R-78 addendum), this
    strategy's target at bar ``i`` is a causal EWM of
    ``kelly_regime_v4``'s own exposure using only bars strictly before
    ``i`` — the same computation a live recorder could run forever off
    the CSV history it already writes for v4.

    No forecast, no gate of its own: it carries no opinion about the
    market beyond "hold what v4 has recently tended to hold." A
    difference against it is therefore a statement about v4's *timing*
    within a matched risk envelope, the same interpretation R-33 gives
    its own matched arms.
    """

    name = "_r83_rolling_matched_hold"

    def __init__(self, halflife_days: float = 90.0, deadband_rel: float = 0.10,
                 deadband_floor: float = 0.02, lag_bars: int = 1) -> None:
        if halflife_days <= 0:
            raise ValueError("halflife_days must be positive")
        self.halflife_days = float(halflife_days)
        self.deadband_rel = float(deadband_rel)
        self.deadband_floor = float(deadband_floor)
        self.lag_bars = int(lag_bars)
        # v4's own warmup, plus room for the EWM to leave its initial
        # transient behind before the measured period starts. run_period
        # gives this many bars of real prefix when available (R-22); when
        # the dataset does not go back far enough (only possible at the
        # very start of inner-train) the run is cold, same caveat every
        # arm in this project's matched-benchmark work already carries.
        v4_warmup = get_strategy(INCUMBENT).warmup
        self.warmup = v4_warmup + int(2 * halflife_days * BARS_PER_DAY)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        exposure = v4_exposure_series(df)
        hl = self.halflife_days * BARS_PER_DAY
        rolling = (exposure.ewm(halflife=hl, min_periods=BARS_PER_DAY)
                   .mean().shift(self.lag_bars))
        # Before the EWM has any real history it is undefined; back/forward
        # fill with the first available value rather than 0.0, so a cold
        # start does not manufacture a spurious "target zero" episode that
        # would not occur once the arm has run for a while.
        rolling = rolling.bfill().fillna(0.0)
        df["target"] = rolling.to_numpy()
        return df

    def on_bar(self, ctx: Context) -> None:
        equity = ctx.equity
        if not np.isfinite(equity) or equity <= 0.0:
            return
        price = ctx.close
        c = float(ctx.bar["target"])
        current = abs(ctx.position) * price / equity
        band = max(self.deadband_rel * c, self.deadband_floor)
        if abs(current - c) <= band:
            return
        desired = c * equity / price
        delta = desired - abs(ctx.position)
        if delta > 0:
            ctx.buy(delta)
        else:
            ctx.sell(-delta)


# --------------------------------------------------------- paired difference

def paired_diff(df: pd.DataFrame, label: str, window: tuple, fee: float,
                a, b) -> pd.Series:
    """``a``'s daily return minus ``b``'s over ``window``, on spot at ``fee``.

    Unlike ``r78_shared.paired_daily_diff`` (name-based, registry lookup
    only), ``a``/``b`` here may be either a registered strategy name (str)
    or a ready ``Strategy`` instance — Arms B and C are not registered
    (``experiments/`` strategies are deliberately kept out of
    ``tradebot run``'s comparison table per ROUTINE.md step 5).
    """
    market = MarketSpec.spot(fee_rate=fee)
    out = []
    for arm in (a, b):
        strat = get_strategy(arm) if isinstance(arm, str) else arm
        result = run_period(strat, df, window[0], window[1], market=market,
                            data_label=label)
        out.append(daily_returns(result.equity))
    joined = pd.concat(out, axis=1, join="inner")
    joined.columns = ["a", "b"]
    return (joined["a"] - joined["b"]).dropna()


def diff_stats(d: pd.Series) -> dict:
    """Mean/sd/fixed-n-floor of a paired daily-difference series.

    The fixed-n floor is the look-once, invalid-for-a-growing-record lower
    bound R-78 also reports: no honest sequential test can beat it, so if
    even it needs decades the anytime-valid horizon computed alongside it
    is not an artifact of the tool's own conservatism.
    """
    mu, sd = float(d.mean()), float(d.std(ddof=1))
    n_fixed = (1.96 * sd / abs(mu)) ** 2 if mu != 0 else float("inf")
    return {"n": len(d), "mean_per_day": mu, "sd_per_day": sd,
            "ann_pct": mu * 365.0, "fixed_n_days": n_fixed,
            "fixed_n_years": n_fixed / 365.0}


# ------------------------------------------------------- anytime-valid horizon

def _first_exclusions(paths: np.ndarray) -> tuple:
    firsts = np.full(len(paths), np.nan)
    signs = np.zeros(len(paths))
    for p, path in enumerate(paths):
        cs = empirical_bernstein_confidence_sequence(path, bound=BOUND, alpha=ALPHA)
        n = anytime_valid_first_exclusion(cs)
        if n is not None:
            firsts[p] = n
            signs[p] = 1.0 if float(cs["lower"].to_numpy()[n - 1]) > 0 else -1.0
    return firsts, signs


def cs_horizon(d: pd.Series, *, horizon_days: int = HORIZON_DAYS,
              n_paths: int = N_PATHS, seed: int = 83) -> dict:
    """Median first-exclusion horizon (n50) and related summary stats.

    Same construction as ``r78_conservative_b06_horizon.py``'s
    ``_summarize`` (stationary-bootstrap paths -> real anytime-valid CS on
    each -> first exclusion, if any), pulled out here as shared,
    reusable machinery rather than re-implemented per script.
    """
    paths = bootstrap_paths(d.to_numpy(), horizon_days, n_paths, seed=seed)
    firsts, signs = _first_exclusions(paths)
    fired = np.isfinite(firsts)
    row = {
        "n50_days": float(np.median(firsts[fired])) if fired.any() else float("inf"),
        "fired_pct": 100.0 * fired.mean(),
        "for_pct": 100.0 * (signs > 0).mean(),
        "against_pct": 100.0 * (signs < 0).mean(),
    }
    for name, days in TRADING_DAYS.items():
        row[f"by_{name}"] = 100.0 * np.mean(fired & (firsts <= days))
    return row


def classify(n50_days: float, by_5y: float, by_25y: float) -> str:
    """The frozen 1b classification rule, both named clauses enforced.

    **Bug fixed here, before any verdict was read (ROUTINE.md's allowed
    kind of do-over — a bug fix, not a moved goalpost).** The
    pre-registration's own prose requires NOT VIABLE when fewer than 50%
    of paths exclude zero within 25 years, *regardless of `n50`* — a
    median computed over a tiny firing subset (e.g. 1% of paths) is not a
    horizon anyone should read as 8-or-so years just because that is the
    midpoint of the few paths that happened to resolve. R-78's own
    reference implementation (`r78_conservative_b06_horizon.py`) carries
    the identical gap in its code (only its prose states the `by_25y`
    clause) but it never surfaced there because R-78's own decisive cell
    fired 0% of paths, so `n50` was already infinite and the missing
    clause was never exercised. This round's decisive cell fires 1.0% of
    paths — precisely the case the gap hides — so it is fixed here rather
    than silently inherited.
    """
    if by_25y < 50.0:
        return "NOT VIABLE AS SPECIFIED"
    if n50_days <= TRADING_DAYS["3y"] and by_5y >= 50.0:
        return "ON TRACK"
    if n50_days <= TRADING_DAYS["25y"]:
        return "SLOW BUT VIABLE"
    return "NOT VIABLE AS SPECIFIED"
