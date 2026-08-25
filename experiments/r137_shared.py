"""Shared, read-only utilities and pre-registration for the R-137 round (08-25).

DIRECTION, in one sentence: **R-127 found that excising a handful of
ETH-idiosyncratic structural-event days narrows one construction's
BTC-pass/ETH-invert `d_sharpe` gap by 60-78% without flipping its sign --
this round asks whether that finding GENERALIZES to the other five
constructions that show the identical inversion signature, and whether it
survives a random-day-exclusion placebo control that R-127 itself never ran.**

**Which constraint this attacks.** Primarily **N≈3** (same framing as
R-127): does removing a small number of ETH-idiosyncratic days move the
five other BTC/ETH comparisons the way it moved R-126-novel's. Secondarily
**ERR**: R-127 established that named-event excision does *something* on
one construction: it never checked whether *any* comparably-sized random
exclusion would do the same thing, which is exactly the alternative
hypothesis that would make "idiosyncratic-event excision" mean nothing
more than "shorter, noisier daily series have wider bootstrap intervals."
This round adds that control -- tightening the bar per `ROUTINE.md`'s own
rule that post-freeze additions may only tighten, never loosen, it.

**Why this and not a fresh mechanism.** Per R-136's own re-ranking (the
most recent backlog state): "the backlog remains empty of anything but
B-06 ... A future session otherwise has: `hedge_experts`'s EXPERT
COMPOSITION axis (closed alongside R-132/R-135); R-127's own named
follow-on (does idiosyncratic-event excision generalize past the one
construction it tested); `champions_council`'s own allocation mechanism
(closed, R-126); the multi-asset panel's own closed list (eleven rounds);
or B-28's breadth clause (blocked on data this project cannot fetch or
simulate)." R-127's follow-on is the only unclosed, unblocked, genuinely
non-duplicate lead on the ranked backlog.

**Not a duplicate of:**
- R-127 itself: tested exactly ONE construction (`champions_council`'s
  R-126-novel CVaR-budgeted reallocation), with no random-day placebo
  control and no causal/algorithmic (as opposed to hand-picked, hindsight
  named) event detector. This round tests the other five and adds both.
- R-109, R-113, R-115-conservative, R-125-conservative, R-126-conservative
  themselves: each measured its own BTC/ETH `d_sharpe` gap and reported
  whether it passed or failed B4. None excised a single day from its own
  ETH series afterward.

**Constructions in scope**, five total, all previously found to clear
their BTC-side gate and then either invert sign on ETH's B4 test, or (one
case) reproduce the same shape one gate earlier:

| id | file(s) | object | mechanism | BTC/ETH shape |
|---|---|---|---|---|
| R-109-novel | `r109_shared.py` + `r109_novel_knn_novelty_brake.py` | `kelly_regime_v4` | k-NN distributional-novelty ERR brake | B1 clears both markets, B4 spot sign-inverts |
| R-113 | `r113_shared.py` + `r113_conservative_mahalanobis_panel.py` / `r113_novel_knn_panel.py` | 8-asset multi-asset panel (`xsmom_entry_band`) | same brake family, panel-level | ETH is one of 8 basket members, not an isolated cell -- see caveat below |
| R-115-conservative | `r115_conservative_shared.py` + `r115_conservative_pooled_eth_coinbase.py` | `kelly_regime_v4` | CORAL-pooled k-NN brake, Coinbase-sourced ETH | B1 clears, B4 inverts |
| R-125-conservative | `r125_shared.py` + `r125_conservative_cvar_scale.py` | `kelly_regime_v4` | CVaR substituted for std-dev in `scale` | fails B1 (BTC) before reaching B4 -- reproduces the *shape*, not a clean inversion; included for completeness, reported separately |
| R-126-conservative | `r126_shared.py` + `r126_conservative_erc_council.py` | `champions_council` | Equal Risk Contribution reallocation | B1 clears BTC spot, B4 inverts on ETH |

**R-113 caveat, disclosed before any code ran.** R-113's own B4-equivalent
cell (`cell_cmp`) scores an 8-asset basket equity curve, not an isolated
ETH return series -- there is no per-asset `d_sharpe` cell to excise days
from. This round still includes it, but at the BASKET level: the same
event/placebo excision is applied to the panel's own realized daily return
series (the same one `cell_cmp` already scores), using event windows and a
BTC/ETH correlation filter computed independently of the panel's own
composition. A null result here is weak evidence (7 of 8 members are
unaffected by an ETH-specific event, diluting any signal); a real,
placebo-beating result would be strong evidence, since it would have to
move a basket despite that dilution. Report this distinction explicitly,
never average R-113's cell in with the four isolated-ETH cells un-flagged.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r137_conservative_fixed_battery_generalization.py`):
  mechanically re-applies R-127's own already-frozen fixed battery --
  Terra/Luna window, The Merge window, their union, and the trailing-14-day
  BTC/ETH correlation < 0.30 filter, all imported verbatim from
  `r127_shared` with no new constant chosen after seeing any R-137 number
  -- to the five constructions above, using each construction's own native
  harness to reproduce its already-published daily candidate/baseline
  return series, then re-scoring under exclusion with the placebo control
  below.
- NOVEL (`r137_novel_causal_cusum_excision.py`): R-127's named events were
  chosen by 20/20 hindsight knowledge of crypto history, which is a
  reasonable basis for a *diagnostic* about why a backtest differs, but
  could never be a deployable rule (nobody can excise "the Merge" from a
  live strategy before it has happened). This branch instead builds a
  CAUSAL, trailing-window CUSUM changepoint detector (Page, E.S. 1954,
  "Continuous inspection schemes," Biometrika 41(1-2), 100-115; textbook
  two-sided formulation per Hawkins & Olwell 1998; recent financial
  applications e.g. the CUSUM-based unsupervised changepoint literature,
  IEEE Access 2022/2025) on the daily spread `d_t = eth_ret_t - btc_ret_t`:
  a trailing `CUSUM_TRAIL_DAYS`-day window estimates `mu_hat`/`sigma_hat`
  causally (data strictly before day t), a two-sided CUSUM statistic with
  reference value `k = CUSUM_K_MULT * sigma_hat` and control limit
  `h = CUSUM_H_MULT * sigma_hat` (both standard textbook multipliers, not
  fit to this data) flags day t as an "ETH-idiosyncratic shock day" and
  resets. Uses only information available AT OR BEFORE day t -- a stronger
  causality bar than the fixed battery's own low-correlation filter, which
  the branch may also sanity-check for overlap. Report, before scoring
  anything: what fraction of the CUSUM-flagged days overlap the hand-picked
  named events (a check that the detector is finding structurally similar
  things, not a promotion criterion), then apply the identical
  excise-and-regap-and-placebo procedure the conservative branch uses.

**What would make each branch fail, named in advance:**

- Conservative: FAILS to generalize if fewer than `MAJORITY_K` of the five
  constructions clear the decision rule below, OR if the real excision's
  narrowing does not beat the random-day placebo control on a majority of
  the constructions that do show narrowing (meaning the fixed battery's
  apparent effect is explainable by "removing any days shortens/thins the
  daily series," not by the specific days chosen) -- either outcome would
  also retroactively narrow R-127's own claim to "one construction, not
  examined for a sample-size confound."
- Novel: FAILS if the CUSUM detector's flagged days have near-zero overlap
  with the hand-picked named events (it is not finding the same structural
  breaks the fixed battery is) or if its excision set does not clear the
  same placebo bar on a majority of constructions.

**Decision rule, pre-registered before either branch is dispatched.** No
excision result of any kind in this round -- narrowing, or even an outright
sign flip -- makes anything here a promotable strategy candidate: excising
already-known-bad calendar days from an already-completed backtest is not
a live, causally-deployable trading rule by construction (this is true of
the conservative branch's hindsight-dated events on its face, and, more
subtly, of the novel branch's detector too, since day t's flag can only be
acted on starting day t+1, a distinction this round measures but does not
implement or trade). The deliverable is a documented finding about the
BTC-pass/ETH-invert pattern's origin, exactly as R-127 scoped itself:

- Per-construction verdict: **GENERALIZES** if excision materially narrows
  the gap (relative narrowing >= `MATERIALITY_REL`, or the raw gap moves by
  >= `NOISE_FLOOR`, whichever is reached first) with NO sign flip, AND the
  real narrowing beats the random-day placebo control at `PLACEBO_ALPHA`
  (one-sided: fraction of `N_PLACEBO_DRAWS` random-day-matched-count draws
  narrowing at least as much as the real excision is < `PLACEBO_ALPHA`).
  **SIGN_FLIP** if the gap's sign reverses after excision (reported and
  discussed on its own terms, not folded into GENERALIZES, per the
  no-promotion rule above). **NOT_GENERALIZE** otherwise.
- Round-level verdict per branch: **CONFIRMED** if >= `MAJORITY_K` of the
  in-scope constructions score GENERALIZES or SIGN_FLIP; **REFUTED** if <=1
  do; **MIXED** otherwise. R-125-conservative's B1-level (not B4-level)
  shape is reported in the same table but excluded from the majority count,
  since it never reached a clean inversion to narrow in the first place --
  named here, before any number is read, specifically so it cannot be
  quietly included or excluded after the fact depending on which reading
  favors the round's own outcome.

No bar at or after `OOS_START = 2023-01-01` is read by either branch --
every construction's own ETH series here is already restricted to
`INNER_VAL` by its own harness.

---

**ADDENDUM, written after `experiments/r137_loaders.py` was built and run,
before any excision was applied to any of the five series -- an allowed
post-freeze addition per `ROUTINE.md` ("they may tighten the bar... never
loosen it"): a scope narrowing discovered from the data itself, not from
any performance number.**

Building the five constructions' loaders (`experiments/r137_loaders.py`,
run standalone, no excision code involved) surfaced two facts that were not
visible from the ledger text alone:

1. **R-109-novel's own ETH cell is not calendar-matched to `INNER_VAL` at
   all.** Its `compare()` call passes `start=None, end=None`, scoring the
   ENTIRE Bitfinex ETH file (2016-03-09..2019-12-31) -- a window that ends
   more than a year before `INNER_VAL_START` and therefore contains neither
   `TERRA_LUNA_WINDOW` nor `THE_MERGE_WINDOW` at all (those dates simply do
   not exist in the series). Excising them would be vacuous by
   construction, not a null result.
2. **Recomputed on daily returns (the statistic this round's whole
   bootstrap/placebo apparatus requires), R-109-novel's own gap does not
   just move in magnitude, it flips sign** (published bar-level d_sharpe
   -0.009 vs. this round's own daily-resampled gap_sharpe +0.0021) -- a
   near-zero, resampling-unstable number, the weakest of the six citations
   this signature was ever built from.

**R-109-novel is therefore EXCLUDED from the in-scope set for both
branches**, reported once here with the reason rather than run through
excision and reported as a misleading "NOT_GENERALIZE." `IN_SCOPE =
["R-113", "R-115-conservative", "R-125-conservative", "R-126-conservative"]`,
four constructions, not five. `MAJORITY_K = 3` (of 4) and the `<=1 ->
REFUTED` / `==2 -> MIXED` thresholds in `round_verdict` below need no
change for `n=4` and are used as originally written.

**A second, separate discrepancy, disclosed but NOT scope-narrowing**:
every construction's own historically-published `d_sharpe` (the number in
`docs/LEDGER.md`) is `tradebot.metrics.sharpe_ratio` on PER-BAR (5-minute)
returns; this round's `gap_sharpe` (below) is deliberately DAILY-resampled,
matching R-126/R-127's own `tradebot.inference.annualized_sharpe` /
`paired_bootstrap` convention -- the one this whole round's bootstrap and
placebo-control machinery requires (block-bootstrapping 5m bars would
treat ~100,000 highly autocorrelated intraday observations as independent
trials, understating every interval). For R-125-conservative and R-113 the
two statistics land within 1% and 0% of each other; for R-115-conservative
and R-126-conservative they diverge by 12.12% and 3.04% respectively
(`experiments/r137_loaders.py`'s own run, both root-caused to the bar-vs-
day resampling difference, not a wiring bug -- independently confirmed by
recomputing each with the CONSTRUCTION'S OWN native bar-level metric and
matching the published number to the printed digit). This round reports
and reasons about the DAILY number throughout, since it is the one
internally consistent with the placebo control every verdict below depends
on, and flags every case where that differs materially from the historical
headline number, rather than treating the two as interchangeable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r127_shared import (  # noqa: E402
    CORR_WINDOW_DAYS,
    INNER_VAL_END,
    INNER_VAL_START,
    LOW_CORR_THRESHOLD,
    THE_MERGE_WINDOW,
    TERRA_LUNA_WINDOW,
    excise_days,
    low_correlation_days,
    rolling_btc_eth_daily_corr,
)
from tradebot.inference import annualized_sharpe, paired_bootstrap, total_log_return  # noqa: E402

# ----------------------------------------------------------------------
# Decision-rule constants, fixed before any R-137 number is read.
# ----------------------------------------------------------------------

NOISE_FLOOR = 0.2          # this project's standard Sharpe noise floor
MATERIALITY_REL = 0.40     # relative-narrowing bar (R-127 measured 60-78%)
PLACEBO_ALPHA = 0.05       # placebo control significance level
N_PLACEBO_DRAWS = 1000
PLACEBO_SEED = 137
MAJORITY_K = 3              # of the 4 in-scope constructions, for a round-level CONFIRMED

# R-109-novel excluded per this module's own addendum above: its ETH cell
# predates INNER_VAL entirely (no overlap with either named 2022 event
# window) and its daily-resampled gap sign-flips vs. its published
# bar-level number. Reported once, not run through excision.
IN_SCOPE = ["R-113", "R-115-conservative", "R-125-conservative", "R-126-conservative"]
EXCLUDED = {"R-109-novel": "ETH cell predates INNER_VAL (2016-2019 Bitfinex); "
                           "no overlap with TERRA_LUNA_WINDOW/THE_MERGE_WINDOW; "
                           "daily-resampled gap sign-flips vs. its own published number."}


def align_daily(candidate: pd.Series, baseline: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two calendar-daily return series on a tz-naive day index.

    Every construction below builds its candidate/baseline series through a
    different harness; this is the one common step needed before excision
    or `paired_bootstrap` (which requires equal-length, positionally
    aligned arrays) can be applied.
    """
    c = candidate.copy()
    c.index = c.index.tz_localize(None) if c.index.tz is not None else c.index
    b = baseline.copy()
    b.index = b.index.tz_localize(None) if b.index.tz is not None else b.index
    df = pd.DataFrame({"candidate": c, "baseline": b}).dropna()
    return df["candidate"], df["baseline"]


def gap_sharpe(candidate: pd.Series, baseline: pd.Series) -> float:
    """`d_sharpe` in this round's own units: candidate Sharpe minus
    baseline Sharpe, annualized -- the same statistic every prior B1/B4
    cell in this project reports (`a.sharpe - b.sharpe`, R-105_shared)."""
    c, b = align_daily(candidate, baseline)
    return float(annualized_sharpe(c.to_numpy()) - annualized_sharpe(b.to_numpy()))


def excise_and_regap(candidate: pd.Series, baseline: pd.Series,
                      excluded_days: pd.DatetimeIndex) -> dict:
    """Excise `excluded_days` from both series (aligned first, so both
    arms drop the same calendar days) and recompute the gap plus a paired
    bootstrap CI on the post-excision series."""
    c, b = align_daily(candidate, baseline)
    c2 = excise_days(c, excluded_days)
    b2 = excise_days(b, excluded_days)
    c2, b2 = c2.align(b2, join="inner")
    gap_after = float(annualized_sharpe(c2.to_numpy()) - annualized_sharpe(b2.to_numpy()))
    n_excised = len(c) - len(c2)
    boot = paired_bootstrap(c2.to_numpy(), b2.to_numpy(), stat=annualized_sharpe,
                             seed=PLACEBO_SEED)
    return dict(gap_after=gap_after, n_before=len(c), n_after=len(c2),
                n_excised=n_excised, boot_lo=boot.diff.lo, boot_hi=boot.diff.hi)


def random_day_placebo(candidate: pd.Series, baseline: pd.Series, n_exclude: int,
                        n_draws: int = N_PLACEBO_DRAWS, seed: int = PLACEBO_SEED) -> np.ndarray:
    """Draw `n_draws` random subsets of `n_exclude` calendar days (without
    replacement, single days -- matching how the real excisions remove
    named/flagged single days, not contiguous blocks) from the aligned
    series' own date range, excise each, and return the array of resulting
    gaps. Used to test whether the REAL excision narrows the gap more than
    an arbitrary same-sized exclusion would."""
    c, b = align_daily(candidate, baseline)
    rng = np.random.default_rng(seed)
    n = len(c)
    if n_exclude <= 0 or n_exclude >= n:
        return np.full(n_draws, gap_sharpe(c, b))
    out = np.empty(n_draws)
    idx_all = np.arange(n)
    for i in range(n_draws):
        drop = rng.choice(idx_all, size=n_exclude, replace=False)
        keep = np.setdiff1d(idx_all, drop, assume_unique=False)
        out[i] = float(annualized_sharpe(c.to_numpy()[keep]) - annualized_sharpe(b.to_numpy()[keep]))
    return out


def placebo_pvalue(gap_before: float, real_gap_after: float,
                    placebo_gap_afters: np.ndarray) -> float:
    """One-sided empirical p-value: fraction of placebo (random-day)
    exclusions that narrow the gap AT LEAST as much as the real,
    pre-registered exclusion did. Narrowing is measured toward zero from
    `gap_before`'s own sign, so a placebo draw that WIDENS the gap or
    overshoots past zero and out the other side counts as narrowing less
    than a draw that lands closer to zero, never as "more."""
    real_narrow = abs(gap_before) - abs(real_gap_after)
    placebo_narrow = np.abs(gap_before) - np.abs(placebo_gap_afters)
    return float(np.mean(placebo_narrow >= real_narrow))


def classify_movement(gap_before: float, gap_after: float, placebo_p: float) -> str:
    if np.sign(gap_after) != np.sign(gap_before) and gap_after != 0.0:
        return "SIGN_FLIP"
    narrowing = abs(gap_before) - abs(gap_after)
    rel = narrowing / abs(gap_before) if gap_before != 0 else 0.0
    material = (rel >= MATERIALITY_REL) or (narrowing >= NOISE_FLOOR)
    if material and placebo_p < PLACEBO_ALPHA:
        return "GENERALIZES"
    return "NOT_GENERALIZE"


def round_verdict(per_construction: dict[str, str], in_scope: list[str]) -> str:
    hits = sum(1 for k in in_scope if per_construction.get(k) in ("GENERALIZES", "SIGN_FLIP"))
    if hits >= MAJORITY_K:
        return "CONFIRMED"
    if hits <= 1:
        return "REFUTED"
    return "MIXED"


# Re-exported for convenience so branch scripts need one import line.
__all__ = [
    "CORR_WINDOW_DAYS", "INNER_VAL_START", "INNER_VAL_END", "LOW_CORR_THRESHOLD",
    "THE_MERGE_WINDOW", "TERRA_LUNA_WINDOW", "excise_days", "low_correlation_days",
    "rolling_btc_eth_daily_corr", "total_log_return", "annualized_sharpe",
    "NOISE_FLOOR", "MATERIALITY_REL", "PLACEBO_ALPHA", "N_PLACEBO_DRAWS",
    "PLACEBO_SEED", "MAJORITY_K", "IN_SCOPE", "EXCLUDED", "align_daily", "gap_sharpe",
    "excise_and_regap", "random_day_placebo", "placebo_pvalue", "classify_movement",
    "round_verdict",
]
