"""R-107 NOVEL: does correlation-aware risk-parity weighting across R-63's own
eligible set raise the panel's realized effective breadth, and does that
translate into money, on the SAME cross-sectional trend signal five prior
rounds have already tested to exhaustion on the equal-weight allocation step?

Shared, frozen infrastructure for this branch. Per ROUTINE.md's parallelism
rules this file is neutral ground within the "novel" branch of R-107: it
carries the pre-registration, committed before any candidate number is read,
and the machinery `experiments/r107_novel_risk_parity.py` (the only other
file this branch writes) imports rather than duplicates. The "conservative"
branch of this round is a *different* R-107 session doing infrastructure
work under `src/tradebot/` -- disjoint scope, disjoint files, not read or
touched here.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **INFO** is not reopened -- the cross-sectional score
that decides WHICH assets are eligible is R-63's own, imported unmodified.
What this round attacks is a different, never-yet-touched joint of the same
mechanism: the **ALLOCATION** step. R-63 through R-68 (five rounds) all
retuned WHEN the portfolio acts on the score (R-65's holding period, R-67's
hysteresis, R-68's band decomposition) or WHETHER an asset is eligible at
all -- never HOW MUCH of the total notional a newly-eligible asset gets once
it clears the bar. Every one of those five rounds split the desired total
notional 1/m across the m eligible names, unconditionally. R-63's own
breadth measurement (mean pairwise daily correlation 0.634, Grinold
equal-correlation breadth 1.47 of 8, 1.41 of 6) was computed on RAW asset
returns, unweighted by any portfolio construction -- it measures what the
panel offers, not what any of the five constructions actually captured of
it. An equal-weight split captures none of that structure: an asset 90%
correlated with the rest of the held set gets the same slice as a genuine
diversifier.

**Mechanism, one sentence:** replace the 1/m equal split across R-63's
eligible set with a causal, rolling risk-parity (equal risk contribution)
solve over those assets' realized covariance, so a correlated latecomer is
sized down and a genuine diversifier is sized up -- raising the panel's
REALIZED effective breadth toward its 1.47-of-8 ceiling by construction,
which (Grinold's `IR = IC*sqrt(BR)`) should raise the risk-adjusted return
of the identical directional signal without any change to the forecast
itself.

**Why this is testable as a real prediction and not just asserted.** Baltas
(2015) directly compares risk-parity weighting against naive/equal weighting
in a long-short trend-following context and finds risk-parity wins
specifically when the traded universe is meaningfully correlated -- and this
project has ALREADY MEASURED its own panel at 0.634 mean pairwise
correlation, squarely inside the regime Baltas's result requires. That is
what makes this a prediction with a stated precondition rather than a
generic "try weighting differently": if this panel's correlation were near
zero, Baltas's own paper would predict risk-parity buys nothing over equal
weight, and this round's premise would already be dead on arrival. It is
not, because R-63 already measured the precondition holds.

**Not a duplicate of:**

- R-63 (`r63_novel_xsmom_rank.py`). Same score, same eligibility rule
  (`score > 0`, rank `< k`), same conditional-volatility total-notional
  scale, same 0.10 deadband on that total. The ONLY change is how the total
  is split across the eligible set: R-63 splits it 1/m; this round solves
  for risk parity. `k` is swept afresh (see below, and see the explicit
  justification for why that is not "re-tuning the signal") but the SCORE,
  the POSITIVITY FILTER and the VOLATILITY-TARGET SCALE are byte-for-byte
  R-63's, imported, never redefined.
- R-65 (holding period / aim portfolio), R-67 (hysteresis), R-68 (band
  decomposition). All three retune WHEN membership changes (entry/exit
  timing). This round changes neither entry nor exit timing at all --
  membership is decided by the identical rule R-63 uses, at the identical
  cadence. It changes only the WEIGHT assigned once a bar's eligible set is
  already decided, which is a disjoint step from anything those three
  rounds touched.
- R-57/R-59/R-60/R-61/R-62 (single-asset replications, no cross-section) and
  R-33 (exposure-level artifacts on a single instrument). Neither forms or
  reweights a cross-sectional selection.
- L-05/L-06 (`kelly_regime_ev`/`_fast`, a no-trade band on a *continuous
  single-asset* fraction). Different object entirely.

**Is it simulable here?** Yes, with zero new data: the same eight committed
5m spot series R-63 uses, through the identical `simulate_portfolio` engine
(`r63_shared.py`), so a risk-parity candidate and R-63's own equal-weight
one differ ONLY in the target-weight matrix handed to that one simulator --
exactly the property that made R-63/65/67/68's D1/D2 comparisons
interpretable at all.

**What would make it fail (named now, before any code ran).**

  (F1) **The premise itself fails.** Risk-parity does not in fact raise
       realized diversification on THIS panel's actual selected sequence
       (as opposed to the panel in the abstract) -- e.g. because the
       eligible set is usually too small (`k` too low) for weighting to
       matter, or because whichever asset the trend score currently
       prefers is *also* the most correlated with the rest, so
       down-weighting it costs signal rather than buying diversification.
       This is checked FIRST, on inner-train only, before any decisive
       read -- see the falsification test below.
  (F2) **The weighting mechanism is real but the signal was never the
       binding constraint.** R-63->R-68's standing diagnosis is that the
       interval, not any mechanism, is what has bound every variant on this
       axis for five rounds running ("no mechanism can narrow an interval
       -- only more data, more breadth, or forward evidence can", R-67).
       Raising realized breadth is a genuine, different kind of move than
       the five prior ones (which all touched timing, never weighting), but
       R-67/R-68's own finding is that even an 80x turnover-economics
       improvement left every 95% interval containing zero. A risk-parity
       construction that clears (F1) may still die exactly there, for a
       different, disclosed reason: raising effective breadth from ~1.4 to
       something closer to 2 buys `sqrt(2/1.4) ~ 1.2x`, a small multiple
       against noise this dataset has shown five times running it cannot
       resolve at the ±0.2 Sharpe floor.
  (F3) **Concentration, not diversification.** Because ERC systematically
       DOWN-weights the most correlated (often also currently-strongest)
       name and UP-weights minor diversifiers, it could push weight toward
       the panel's noisier, thinner-liquidity members (five of six U6
       assets are exactly the "smaller-cap, lower-liquidity" segment
       Fieberg et al. 2024 warn deteriorates a crypto trend factor). If so
       the mechanism could raise measured breadth on paper while
       *increasing*, not decreasing, realized cost/slippage-adjacent risk
       -- the falsification test below only measures the breadth premise,
       not this one, so it is watched for qualitatively in the D-cell
       turnover/notional diagnostics rather than gated on.

=====================================================================
LITERATURE
=====================================================================

- Baltas, N. (2015), "Trend-Following, Risk-Parity and the Influence of
  Correlations," working paper (also circulated as a chapter in *Risk-Based
  and Factor Investing*, Elsevier, 2015). Compares risk-parity weighting
  (uses the full pairwise correlation matrix, penalizing assets correlated
  with the rest of the selected set) against naive inverse-volatility
  weighting (uses only individual vols, ignores correlation) across a
  long-short trend-following universe, and finds risk-parity's advantage is
  concentrated specifically in periods/universes of elevated correlation.
  Read directly for the construction and the qualitative claim; NOT for a
  transferable magnitude -- the paper's universe is liquid futures at
  institutional cost and long-short, this round's is 6-8 crypto spot pairs
  at retail cost and long-only.
- Bruder, B., & Roncalli, T. (2012), "Managing Risk Exposures Using the
  Risk Budgeting Approach," SSRN (doi:10.2139/ssrn.2009778). The general
  risk-budgeting framework this round implements a special case of: solve
  for `w` such that every asset's contribution to total portfolio variance
  is equalized, `w_i (Sigma w)_i = w_j (Sigma w)_j` for all `i, j` in the
  selected set (equal risk contribution, ERC, is risk budgeting with equal
  budgets). Cited for the risk-contribution definition and the budgeting
  frame this round's `b_i = 1/m` is the special case of.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010), "The Properties of
  Equally Weighted Risk Contribution Portfolios," Journal of Portfolio
  Management 36(4), 60-70. Establishes existence and uniqueness of the ERC
  solution for a positive-definite covariance matrix and gives the convex
  reformulation this round's solver actually optimizes: minimize
  `0.5 y'Sigma y - sum_i b_i * ln(y_i)` over `y > 0`, then `w = y / sum(y)`.
  The solver below (`solve_erc`) is a cyclical coordinate descent on this
  convex objective's own first-order condition, closed-form per coordinate
  because each one-variable subproblem is a quadratic in `y_i` (see
  `solve_erc`'s docstring for the derivation). Cited for the formulation;
  the coordinate-descent implementation is this round's own, verified by a
  unit check (`check_erc_converges`) rather than claimed to reproduce any
  paper's specific numerical algorithm.
- Ledoit, O., & Wolf, M. (2004), "Honey, I Shrunk the Sample Covariance
  Matrix," Journal of Portfolio Management 30(4), 110-119. With N assets and
  a rolling window of length T such that T is not overwhelmingly larger than
  N, the raw sample covariance is noisy and shrinking it toward a
  structured target (their paper's own choice: the constant-correlation
  model, off-diagonal correlations all equal to the sample's own average
  pairwise correlation, diagonal variances left alone) reduces estimation
  error. THIS ROUND'S DISCLOSED DEPARTURE FROM THE PAPER: Ledoit & Wolf
  derive a CLOSED-FORM OPTIMAL shrinkage intensity from the data (their
  Theorem 1); fitting that closed form well needs more machinery and,
  honestly, more data-mining latitude than this round's budget justifies for
  a first test of the WEIGHTING mechanism itself. Instead, shrinkage
  intensity `lambda` is treated as a construction parameter and SWEPT over a
  small pre-registered grid on inner-train/inner-validation, selected the
  same way R-63's `k` was -- exactly the project's standing convention for
  "one new free parameter" (R-65's `buffer`/`hold_days`, R-67's `delta`,
  R-68's `delta_in`/`delta_out`). The target model (constant off-diagonal
  correlation, individual variances untouched) is Ledoit & Wolf's own; only
  the choice of intensity is a grid rather than their derived optimum. If
  the grid ever prefers `lambda=0` (no shrinkage) on inner-validation, that
  is itself reported honestly as evidence the departure from the paper's
  formula did not matter here.
- Choueifaty, Y., & Coignard, Y. (2008), "Toward Maximum Diversification,"
  Journal of Portfolio Management 35(1), 40-51; and Choueifaty, Y.,
  Froidure, T., & Reynier, J. (2013), "Properties of the Most Diversified
  Portfolio," Journal of Investment Strategies 2(2), 49-70. Define the
  Diversification Ratio `DR(w) = (w . sigma) / sqrt(w' Sigma w)` (weighted
  average of individual vols over portfolio vol) and formally connect its
  square to Grinold's independent-bet count under a portfolio's own realized
  covariance -- DR(w)=1 for a single asset or a set of unit-correlated
  assets, and DR(w)^2 rises as the SAME asset set is weighted to capture
  more of its own diversification potential. This is the statistic this
  round's falsification test uses: it is Grinold breadth's own portfolio-
  construction-level analogue (R-63's `equal_corr_breadth` measured it on
  the RAW panel; `DR^2` here measures it on a WEIGHTED, TIME-VARYING
  selected subset, which is what R-63->R-68's constructions actually held).
- Grinold, R. C. (1989), JPM 15(3), 30-37; Clarke, de Silva & Thorley
  (2002), FAJ 58(5), 48-66. `IR = IC * sqrt(BR)`, restated here rather than
  re-derived: R-63 already measured this panel's raw breadth at 1.47 of 8
  (1.41 of 6). This round's premise is that the CONSTRUCTION, not the panel,
  has been leaving realized breadth on the table -- a claim about the gap
  between 1.47 and whatever an equal-weight construction actually realizes,
  which is exactly what the falsification test below measures directly.
- Fieberg, C., et al. (2024), JFQA (doi:10.1017/S0022109024000747). Cited
  again here (as in R-63) for named failure mode (F3): a crypto trend factor
  deteriorates on smaller-cap, lower-liquidity coins, which risk-parity's
  own down-weighting of the correlated/liquid leader could push weight
  toward.

=====================================================================
UNIVERSE, WINDOWS, COSTS, BENCHMARKS -- INHERITED, NOT RESTATED
=====================================================================

Every one of these is imported from `r63_shared.py` / `r63_novel_xsmom_rank.py`
/ `r65_shared.py` / `r68_shared.py`, unchanged, so this round cannot drift
from the numbers it extends:

  UNIVERSE_6, UNIVERSE_8            R-63's panel definitions
  W_TRAIN, W_VAL, W_FULL6, W_HOLD   R-63's windows
  SPOT_BASE (0.10%), SPOT_REAL (0.40%), SPOT_FREE (0 bps, D5 only)
  cross_sectional_score, conditional_vol_scale, basket_log_returns, DEADBAND
                                     R-63's score and scale, byte for byte
  simulate_portfolio, matched_hold_targets, static_hold_equity
                                     R-63's one simulator and its arms
  volmatched_hold_equity            R-65's risk-matched primary benchmark
  d1_pass, d2_pass, d3_pass, d5_pass, D5_BAR_R68
                                     R-68's frozen gates, imported UNMODIFIED
                                     (this is the ACTUAL current state of
                                     "r63_shared's own d1/d2/d3" after two
                                     corrections R-63 and R-65 filed against
                                     themselves -- both D1 and D2 compare
                                     against VOLMATCH_HOLD, the risk-matched
                                     arm, per R-63's own disclosed correction
                                     that MATCHED_HOLD is not a risk match
                                     for a concentrated candidate; MATCHED_HOLD
                                     is still reported for continuity, as
                                     R-65 did)
  scramble_fixed_perm               R-68's single correct-for-continuous-
                                     weight scramble control implementation
  further_work                      R-68's bar WITHOUT the M1' membership
                                     clause (M1' is specific to R-67/R-68's
                                     eligibility-timing axis -- this round
                                     changes neither entry nor exit timing,
                                     so there is nothing for a membership-
                                     change-rate gate to measure; using
                                     R-65's four-clause form, (D1 or D2) and
                                     D3 and D5 and scramble_survived, since
                                     that is the form actually keyed to what
                                     this round can change)

The `BTC_HOLD` context cell R-63/65/67/68 each built (uncounted, until R-72
found and corrected the omission, +9 across those four rounds) is
DELIBERATELY NOT built here: it feeds no gate, and R-72's own standing
instruction is "retire or explicitly count it" -- retiring it is the
correct choice for a round that does not need it.

=====================================================================
THIS ROUND'S OWN NEW MACHINERY
=====================================================================

Everything below is new: the covariance estimator, its shrinkage, the ERC
solver, and the falsification statistic. None of it existed anywhere in
this repo before this round.

COVARIANCE. Daily (not bar-level) log returns of the closing price, over
the FULL universe passed to `build_targets` (U6 or U8, whichever the caller
aligned), a rolling window of `WINDOW_DAYS` calendar days, `MIN_DAYS`
minimum periods. THIS IS A DELIBERATE, DISCLOSED CHOICE, not a sweep: bar-
level (5-minute) covariance would be dominated by microstructure noise and
would need its own realized-covariance estimator (a materially bigger
project on its own); daily returns are the standard unit in this exact
literature (Baltas 2015, Ledoit & Wolf 2004) and match what R-63's own
`r63_breadth.py` measured its panel's correlation structure on. `WINDOW_DAYS
= 60`: at up to 8 assets that is a T/N ratio of >= 7.5, comfortably inside
the regime Ledoit & Wolf study (their own simulations go down to T/N < 1),
so shrinkage here is a stabilizing precaution rather than a strict
necessity -- which is exactly why sweeping `lambda` from 0 (no shrinkage) to
1 (fully shrunk to the constant-correlation target) is the informative
thing to do, rather than also sweeping the window length. CAUSALITY: the
covariance used to size any bar within calendar day D is computed from
daily returns STRICTLY BEFORE day D (the window ending at day D-1's close);
day D's own not-yet-realized return never enters its own day's sizing. This
is verified by the truncation causality check on the full `build_targets`
output, not merely asserted.

SHRINKAGE. Ledoit & Wolf's constant-correlation target: diagonal
(individual variances) untouched, every off-diagonal correlation replaced
by the window's own average pairwise correlation, shrunk covariance =
`(1-lambda)*raw + lambda*target`.

RISK PARITY (ERC) SOLVER. Given the covariance sub-matrix of whichever
assets are eligible at a bar (R-63's own `score > 0, rank < k` mask,
unmodified), solve for weights whose contribution to total portfolio
variance is equal across the selected set (`solve_erc`), verified by a
unit check that recomputes each asset's realized share of variance and
requires it within a stated tolerance of `1/m` (`check_erc_converges`).

`k`, the size of the eligible set. R-63's own frozen `k=1` (one asset held
at a time) makes risk-parity WEIGHTING MATHEMATICALLY VACUOUS: with exactly
one eligible asset, both equal-weight and risk-parity assign it the entire
total notional, byte for byte, on every bar. Testing the allocation
mechanism at all THEREFORE REQUIRES `k >= 2`, which is why `k` is swept
here rather than inherited frozen -- and it is why this is not "re-tuning
the directional signal": the SCORE and its POSITIVITY FILTER (which
assets qualify at all) never change; `k` only ever governs how many of the
already-qualified assets the allocation step is asked to weight, exactly
the same status R-65's `hold_days`, R-67's `delta` and R-68's
`delta_in`/`delta_out` each had as "one new free parameter belonging to the
mechanism under test, not to the signal."

TOTAL-NOTIONAL SCALE, PRESERVED EXACTLY. R-63's own formula for the desired
TOTAL notional, `scale(t) * m(t) / k`, latched through the 0.10 deadband
(`conditional_vol_scale`, `DEADBAND`, both imported unmodified) is computed
IDENTICALLY here -- this round changes nothing about how much total
exposure the portfolio wants, only how that total is split across the
eligible names. Consequently, for a FIXED `k`, R-63's own equal-weight
construction (re-run here at that `k` via `r63_baseline_targets`, imported
unmodified) and this round's risk-parity construction share the identical
`total(t)` series, bar for bar, by construction -- not merely approximately
matched by a downstream MATCHED_HOLD/VOLMATCH_HOLD benchmark (which also
still applies, computed from EACH candidate's own realized notional/vol per
the standing R-33/R-31 convention). This is the strongest exposure-match
this project's SIZE-axis rounds have ever had between a candidate and its
nearest same-mechanism-family comparator, and it is reported explicitly in
the results.

=====================================================================
PRE-REGISTERED FALSIFICATION TEST (run BEFORE any W_FULL6/holdout read)
=====================================================================

On W_TRAIN (inner-train) ONLY, U8, at the frozen `(k, lambda)` selected by
the sweep below: build BOTH R-63's equal-weight target matrix and this
round's risk-parity target matrix, from the IDENTICAL eligible-asset
sequence (same score, same positivity filter, same `k` -- verified by an
identity check that the two matrices' nonzero SUPPORT is bar-for-bar
identical before any statistic is computed). For every bar with `m >= 2`
eligible assets (the only bars where the two constructions can differ at
all), compute the Diversification Ratio `DR(w) = (w . sigma) / sqrt(w'
Sigma w)` using the SAME causal covariance estimate for both (so the two
numbers differ only in `w`, never in `Sigma`). Aggregate as `DR^2`, the
portfolio-level analogue of Grinold breadth.

PASS/FAIL, decided now: the mechanism's own named effect is that risk-parity
should raise realized effective breadth relative to equal-weight on the
IDENTICAL eligible sequence. **FAIL (falsified) iff `mean(DR^2_riskparity)
<= mean(DR^2_equalweight)` over those bars on W_TRAIN.** This is a
structural, not a noisy, comparison -- both statistics are deterministic
functions of the SAME realized covariance path and the SAME bar-by-bar
eligible-set sequence, differing only in the weight-solving rule, so no
bootstrap interval is needed to read a sign; ERC is not guaranteed a priori
to beat equal weight on `DR^2` (it solves for equal RISK contribution, not
for maximum diversification, which is a related but different convex
program -- see Choueifaty et al. above), so this is a genuine, falsifiable
question, not a tautology. If it fails, the round STOPS HERE: no W_FULL6
read, no W_HOLD read, reported as a clean negative on inner-train alone.

If it PASSES, proceed to the full pre-registered battery on W_FULL6/U6
(decisive) and W_VAL/U8 (D3), at 0.10% and 0.40%, exactly as R-63/65/67/68
each ran theirs, using the gates listed above, imported unmodified.

FURTHER-WORK BAR (this pre-registration does NOT by itself authorize a
holdout read): `(D1 or D2) and D3 and D5 and scramble_survived`, R-65's
form (see note above on why M1' is not inherited). Clearing it authorizes
exactly ONE holdout read on `W_HOLD`, per R-63's own convention, counted
honestly in the report regardless of outcome.

PROMOTION BAR (only reachable after an authorized holdout read; default is
REJECT): R-63's own, unmodified -- beats EW_HOLD and BTC_HOLD out-of-sample
after real costs; improvement exceeds the +/-0.2 Sharpe noise floor (R-20)
or is a drawdown/tail improvement; survives the scramble control on the
holdout cell too; the `(k, lambda)` neighbourhood is a plateau, not a peak.
Nothing found by this branch can be REGISTERED regardless of verdict --
R-67/R-68's own finding stands: this is a bar-by-bar cross-asset allocator
and B-32 (multi-asset registration infrastructure) is still OPEN.

CONFIGURATIONS EVALUATED. Counted by `r63_shared.config_count()`, shared
process-wide with every prior round descended from `r63_shared.py`, so the
number printed at the end of a run of this branch's script is cumulative
across whatever else has run in the same process -- report the DELTA
(before/after this branch's own run) as this round's contribution, exactly
as R-63/65/67/68 each did.

=====================================================================
HOLDOUT-READ NOTE (Step 4 Note 3 in this round's own dispatch instructions)
=====================================================================

This axis has an established, disclosed departure from the rest of the
project's holdout discipline: its decisive `W_FULL6` cell already spans
2020-04-01 through the last committed bar, so D1/D2/D5/scramble on W_FULL6
already include 2023+ U6 data, counted as "+0" holdout consultations by an
explicit, project-wide convention (R-47/R-57/R-63 and onward) on the
grounds that the RESERVED BTC/ETH 2023+ holdout, not the U6 panel, is the
thing "+0" refers to. This round follows the SAME convention, for
comparability with R-63/65/67/68's own published numbers -- it does not
invent a stricter split. Every W_FULL6/W_VAL read this branch performs is
counted honestly below; a genuine `W_HOLD` (U8, 2023-01-01 onward, BTC/ETH
included) read, if the further-work bar is ever cleared, is the only kind
that increments the running total.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r63_shared import (  # noqa: E402,F401
    BOOT_KW,
    SCRAMBLE_SEEDS,
    SPOT_BASE,
    SPOT_REAL,
    START_BALANCE,
    TOTAL_NOTIONAL_DEADBAND,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_HOLD,
    W_TRAIN,
    W_VAL,
    align_frames,
    check_against_engine,
    check_causality,
    compare,
    config_count,
    excludes_zero,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
    scramble_targets,
    simulate_portfolio,
    static_hold_equity,
)
from experiments.r63_novel_xsmom_rank import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    DEADBAND,
    HORIZONS,
    WARM_DAYS,
    basket_log_returns,
    conditional_vol_scale,
    cross_sectional_score,
    warm_window,
)
from experiments.r63_novel_xsmom_rank import build_targets as r63_baseline_targets  # noqa: E402,F401
from experiments.r65_shared import (  # noqa: E402,F401
    D5_BAR_CORRECTED,
    R63_GROSS_EDGE,
    R63_GROSS_EDGE_VS_VOLMATCH,
    R63_NET_D1,
    R63_TURNOVER_PER_DAY,
    SPOT_FREE,
    frontier_row,
    realized_vol,
    turnover_stats,
    volmatched_hold_equity,
)
from experiments.r68_shared import (  # noqa: E402,F401
    D5_BAR_R68,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    scramble_fixed_perm,
)

OUT_DIR = ROOT / "reports" / "r107_risk_parity"

# ---------------------------------------------------------------- constants

WINDOW_DAYS = 60   # rolling causal covariance window, calendar days
MIN_DAYS = 30      # minimum periods before a covariance estimate is trusted

K_GRID = (2, 3, 4, 6)
LAMBDA_GRID = (0.0, 0.5, 1.0)

# R-65's four-clause further-work bar (no M1' -- see docstring above for why
# the membership-timing clause does not apply to a weighting-only mechanism).
def further_work(d1: bool, d2: bool, d3: bool, d5: bool,
                  scramble_survived: bool) -> bool:
    return (d1 or d2) and d3 and d5 and scramble_survived


# ---------------------------------------------------------------- covariance


def daily_log_returns(aligned: dict[str, pd.DataFrame], universe) -> pd.DataFrame:
    """Daily close-to-close log returns for ``universe``, in column order.

    Built from the ALIGNED (forward-filled, causal) frame, so a day an
    exchange did not print trades contributes a zero return rather than a
    gap -- consistent with `align_frames`'s own convention.
    """
    closes = pd.DataFrame({t: aligned[t]["close"] for t in universe})
    daily = closes.resample("1D").last()
    return np.log(daily).diff()


def shrink_to_constant_corr(cov: np.ndarray, lam: float) -> np.ndarray:
    """Ledoit & Wolf's (2004) constant-correlation shrinkage target.

    Diagonal (individual variances) is left untouched; every off-diagonal
    correlation is replaced by the window's own average pairwise
    correlation. ``lam=0`` returns ``cov`` unchanged; ``lam=1`` returns the
    fully-shrunk target.
    """
    n = cov.shape[0]
    if n < 2 or lam <= 0.0:
        return cov
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    outer = np.outer(d, d)
    corr = cov / outer
    mask = ~np.eye(n, dtype=bool)
    rho_bar = float(corr[mask].mean())
    target_corr = np.full((n, n), rho_bar)
    np.fill_diagonal(target_corr, 1.0)
    target_cov = target_corr * outer
    return (1.0 - lam) * cov + lam * target_cov


def build_cov_lookup(aligned: dict[str, pd.DataFrame], universe,
                      lam: float, window_days: int = WINDOW_DAYS,
                      min_days: int = MIN_DAYS) -> dict:
    """Causal daily covariance, one matrix per calendar day, keyed by the
    NORMALIZED TIMESTAMP OF THE DAY IT APPLIES TO (not the day it was
    measured through).

    ``cov_by_day[D]`` is estimated from daily returns over the ``window_days``
    calendar days STRICTLY BEFORE ``D`` (i.e. through day ``D-1``'s close) --
    day ``D``'s own not-yet-realized return never enters its own lookup
    value. Returns ``None`` for a day before ``min_days`` of prior history
    exist (the eligible universe carries no covariance-derived weight there;
    callers must fall back, but this never happens in practice because every
    evaluation window is preceded by R-63's own 91-day warm buffer).
    """
    ret = daily_log_returns(aligned, universe)
    n = ret.shape[1]
    dates = ret.index
    raw = np.full((len(dates), n, n), np.nan)
    vals = ret.to_numpy(dtype=float)
    for i in range(len(dates)):
        lo = max(0, i - window_days)
        window = vals[lo:i]  # STRICTLY before date i -- day i's own row excluded
        window = window[np.all(np.isfinite(window), axis=1)] if window.size else window
        if len(window) < min_days:
            continue
        c = np.cov(window, rowvar=False)
        if n == 1:
            c = np.array([[c]])
        raw[i] = shrink_to_constant_corr(c, lam)

    out = {}
    for i, d in enumerate(dates):
        out[d] = raw[i] if np.isfinite(raw[i]).all() else None
    return out


# ---------------------------------------------------------------- ERC solver


def solve_erc(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Equal-risk-contribution weights over ``cov`` (m x m, PD).

    Solves, by cyclical coordinate descent, the convex program of Maillard,
    Roncalli & Teiletche (2010): minimize ``0.5 y'Sigma y - sum_i b_i ln(y_i)``
    over ``y > 0`` with equal budgets ``b_i = 1/m``, then normalize
    ``w = y / sum(y)``. The first-order condition for coordinate ``i`` holding
    the rest fixed is ``Sigma_ii * y_i^2 + c_i * y_i - b_i = 0`` with
    ``c_i = sum_{j!=i} Sigma_ij y_j``, a quadratic in ``y_i`` with a unique
    positive root (since ``b_i > 0``, the two roots have opposite sign):
    ``y_i = (-c_i + sqrt(c_i^2 + 4*Sigma_ii*b_i)) / (2*Sigma_ii)``.
    """
    m = cov.shape[0]
    if m == 0:
        return np.zeros(0)
    if m == 1:
        return np.array([1.0])
    cov = np.array(cov, dtype=float, copy=True)
    diag = np.diag(cov).copy()
    diag[diag <= 0.0] = 1e-12
    np.fill_diagonal(cov, diag)
    # Tiny ridge for numerical safety on a near-singular subset.
    cov = cov + 1e-10 * np.eye(m) * np.mean(diag)

    b = np.full(m, 1.0 / m)
    y = np.full(m, 1.0 / m)
    for _ in range(max_iter):
        y_prev = y.copy()
        for i in range(m):
            c_i = float(cov[i] @ y - cov[i, i] * y[i])
            a = float(cov[i, i])
            disc = c_i * c_i + 4.0 * a * b[i]
            y[i] = max((-c_i + np.sqrt(max(disc, 0.0))) / (2.0 * a), 1e-12)
        if np.max(np.abs(y - y_prev)) < tol:
            break
    return y / y.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's realized SHARE of total portfolio variance (sums to 1)."""
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return np.full(len(w), np.nan)
    mrc = cov @ w
    rc = w * mrc
    return rc / port_var


def check_erc_converges(tol: float = 1e-4) -> dict:
    """Unit check: build several structured covariance matrices, solve, and
    verify the realized risk-contribution shares equalize to within ``tol``
    of ``1/m``. Returns a dict of per-case max deviation and an overall
    pass/fail, printed by the caller.
    """
    cases = {}

    rng = np.random.default_rng(0)

    # Case 1: identity (uncorrelated, equal variance) -- ERC must equal 1/m.
    cases["identity_m4"] = np.eye(4)

    # Case 2: unequal variances, zero correlation.
    cases["unequal_var_m3"] = np.diag([0.01, 0.04, 0.25])

    # Case 3: one highly-correlated pair plus a diversifier.
    c = np.array([[1.0, 0.9, 0.1],
                  [0.9, 1.0, 0.1],
                  [0.1, 0.1, 1.0]])
    d = np.array([0.20, 0.22, 0.18])
    cases["corr_pair_plus_diversifier_m3"] = np.outer(d, d) * c

    # Case 4: random PD matrix, m=6 (the U6 panel's own size).
    a = rng.normal(size=(6, 6))
    cases["random_pd_m6"] = a @ a.T + 0.05 * np.eye(6)

    results = {}
    ok = True
    for name, cov in cases.items():
        w = solve_erc(cov)
        rc = risk_contributions(w, cov)
        m = len(w)
        dev = float(np.max(np.abs(rc - 1.0 / m)))
        results[name] = {"weights": w.tolist(), "max_rc_dev_from_equal": dev,
                          "converged": dev <= tol}
        ok = ok and dev <= tol
    results["all_pass"] = ok
    return results


# ---------------------------------------------------------------- DR^2 stat


def diversification_ratio_sq(w_row: np.ndarray, cov: np.ndarray) -> float:
    """`DR(w)^2 = ((w . sigma) / sqrt(w' Sigma w))^2`, restricted to the
    support of ``w_row`` (nonzero entries) against the matching sub-matrix of
    ``cov``. Returns NaN if the support has fewer than 2 assets (DR is
    trivially 1 there for any construction, uninformative for this round's
    comparison) or if the portfolio variance is non-positive.
    """
    idx = np.flatnonzero(w_row > 0.0)
    if len(idx) < 2:
        return float("nan")
    sub = cov[np.ix_(idx, idx)]
    wv = w_row[idx]
    wv = wv / wv.sum()
    sigma = np.sqrt(np.clip(np.diag(sub), 0.0, None))
    port_var = float(wv @ sub @ wv)
    if port_var <= 0:
        return float("nan")
    dr = float(wv @ sigma) / np.sqrt(port_var)
    return dr * dr
