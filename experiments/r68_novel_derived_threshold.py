"""R-68 NOVEL branch -- the band threshold DERIVED, not fitted. Zero free
parameters, no grid, no selection.

The pre-registration for this round is the module docstring of
`experiments/r68_shared.py`, which is FROZEN and is NOT edited by this file.
This file implements one derivation (plus a declared second and a declared
variant), measures it, and reports it. It does not define, relax or re-derive
a decision rule; `git diff` on `r68_shared.py`, `r67_shared.py`,
`r65_shared.py`, `r63_shared.py`, `r65_conservative_rank_buffer.py` and
`r63_novel_xsmom_rank.py` is empty for this branch. Any flaw found in them is
REPORTED, never fixed (the R-63 process violation, not repeated).

=====================================================================
WHY THIS ARM EXISTS
=====================================================================

R-67's conservative arm chose `delta = 0.080` by sweeping a six-cell grid and
taking the best W_VAL cell. The winner landed on the grid's TOP CORNER, which
is R-68's named failure mode (F1) and which cost six trials in the
deflated-Sharpe accounting. R-65's and R-67's single strongest pieces of
evidence were, by contrast, both *derived* rates that landed where an
independent measurement said they should: R-67's `a_GP = 0.0073088` is
computed from the fee, R-63's published turnover and a measured per-bar sigma
-- nothing fitted -- and reproduces R-65's independently published value to
eight significant figures on a completely different aim construction.

This branch asks the same question of the band, and it is a question that can
be answered whether or not the arm wins: **does a threshold nobody fitted
land where the data independently prefers?**

There is NO selection in this branch. The derived number IS the
configuration. A neighbourhood of fixed multipliers is reported afterwards so
a plateau-vs-peak statement can be made, and it is REPORTED, NOT SELECTED ON:
the derived point's rank inside it is stated honestly, including when it
ranks last.

=====================================================================
THE FORMULAE, FROZEN BEFORE THE FIRST NUMBER WAS COMPUTED
=====================================================================

Everything below -- the expressions, every symbol's definition, every
imported assumption, every assumption this setting VIOLATES, the numerical
guards, the cap, and which arm is PRIMARY -- was written into this docstring
before any of it was executed. The code honours it exactly as written.

Common symbols
--------------
    s_i(t)     R-63's composite cross-sectional score for asset i at bar t,
               `mean_h (close_i(t)/anchor_{i,h}(t) - 1)` over h in {20,40,80}
               days. Dimensionless, raw units, never standardized.
               (`r63_novel_xsmom_rank.cross_sectional_score`, unmodified.)
    c          the proportional transaction cost rate actually charged on the
               decision cells: c = 0.001 (SPOT_BASE, 0.10% taker). Paid on
               every buy and on every sell, so a round trip costs 2c per unit
               of notional.
    gamma      relative risk aversion. gamma = 1 (log utility) -- this
               repo's standing convention, identical to `GAMMA` in
               `r65_novel_aim_portfolio.py` and `r67_novel_smoothed_score.py`.
               NOT fitted here and NOT chosen here.
    BARS_PER_DAY = 288 (5-minute bars).
    sigma_s(t) the CAUSAL, EXPANDING, ONE-BAR-SHIFTED pooled standard
               deviation of s across all assets over every bar strictly
               before t. Population sd (divide by n; n is ~1e5, so ddof is
               immaterial and the choice is declared rather than tuned).
    sigma_ds(t) the same estimator applied to the per-bar score INCREMENT
               `Delta s_i(t) = s_i(t) - s_i(t-1)`.
    theta(t)   R-63's own conditional volatility scale at bar t, capped at
               1.0 and shifted ONE EXTRA BAR: `theta(t) = min(scale(t-1),1)`.
               This is the total notional fraction the arm intends to hold
               when it holds anything (at k=1 the whole desired notional goes
               to the single held slot), so it is this mechanism's analogue of
               a Merton proportion. It is already causal inside
               `conditional_vol_scale` (its EWM vol carries a `.shift(1)`);
               the extra shift is applied to satisfy this round's
               pre-registration to the letter.

(D-A) SMALL-COST ASYMPTOTIC NO-TRADE BAND -- THE CUBE-ROOT LAW
--------------------------------------------------------------
Sources. Constantinides, G. M. (1986), "Capital Market Equilibrium with
Transaction Costs", *Journal of Political Economy* 94(4), 842-862 -- the
original demonstration that with proportional costs the optimal policy is a
no-trade interval around the frictionless target and that its width is
O(cost^(1/3)) rather than O(cost). Janecek, K., & Shreve, S. E. (2004),
"Asymptotic analysis for optimal investment and consumption with transaction
costs", *Finance and Stochastics* 8, 181-206 -- the rigorous asymptotic
expansion and the constant. Muhle-Karbe, J., Reppen, M., & Soner, H. M.
(2017), "A Primer on Portfolio Choice with Small Transaction Costs",
*Annual Review of Financial Economics* 9, 301-331 -- the modern statement in
the form adopted here.

THE EXPRESSION ADOPTED, in position-fraction units:

    Delta_theta(t) = ( (3 / (2*gamma)) * theta(t)^2 * (1-theta(t))^2 * c )^(1/3)

i.e. the half-width of the no-trade interval around the frictionless target,
to leading order in c^(1/3), for a CRRA investor in a Black-Scholes market
with proportional cost c.

THE BRIDGE TO SCORE UNITS. This is MINE, not the cited papers', and it is
declared here as the weakest link in D-A. The papers give a width in units of
the risky-asset FRACTION; this mechanism's band is on the SCORE. The
conversion adopted is the single dimensionally-consistent one available from
quantities already measured causally:

    ds/dtheta = sigma_s(t) / theta(t)     [score units per unit of exposure]

read as: one standard deviation of the cross-sectional score is what it takes
to move this arm from flat to its intended exposure theta(t). Hence

    delta_A(t) = Delta_theta(t) * sigma_s(t) / theta(t)
               = sigma_s(t) * (3c/(2*gamma))^(1/3)
                            * theta(t)^(-1/3) * (1-theta(t))^(2/3)

REGIME OF VALIDITY, and what this setting VIOLATES.
  Valid when: (i) c is small (c = 0.001, and c^(1/3) = 0.1, so the leading
  order term is the right order of magnitude but the expansion's neglected
  terms are only one decade down -- this is at the edge, not deep inside, the
  asymptotic regime); (ii) the risky asset is a geometric Brownian motion
  with constant mu and sigma; (iii) the investor holds a CONTINUOUS position
  and trades only at the boundaries of the interval; (iv) the target is the
  static Merton proportion.
  VIOLATED HERE, all four, and stated plainly rather than buried:
   1. **The position is BINARY, not continuous.** This arm holds theta or 0.
      The map from score to position is a step function whose derivative is a
      delta function, so a "no-trade interval in position space" does not
      pull back to a score interval by any exact route. The bridge above is a
      linearization of a map that is not differentiable. This is the single
      largest reason to distrust D-A and it is why D-A is NOT this branch's
      primary arm (see below).
   2. **The target is not static.** The frictionless target here is a
      cross-sectional selection driven by a mean-reverting score
      (half-life ~5.99 days, R-67's measurement), not a constant Merton
      proportion. Gerhold, Muhle-Karbe & Schachermayer's and Martin's
      extensions to a moving target change the constant, not the exponent.
   3. **Prices are not GBM at 5-minute frequency**, and the arm's exposure is
      volatility-targeted, so theta is itself stochastic. The formula is
      evaluated per bar with the contemporaneous theta, which is a
      quasi-static approximation.
   4. **The costs are not the only friction.** R-63's 0.10 deadband on
      desired total notional and the simulator's 5% band are additional
      no-trade regions already present in the substrate; D-A ignores them.

(D-B) COST-MATCHED FIRST-PASSAGE THRESHOLD -- IN THIS REPO'S OWN MEASUREMENTS
----------------------------------------------------------------------------
Every input is measured elsewhere, in this repository, before this branch
existed. Nothing is fitted.

  1. **First passage.** At 5-minute bars the score is near-diffusive. For a
     driftless diffusion with per-bar increment variance sigma_ds^2, started
     at the CENTRE of a symmetric band of half-width delta, the mean exit
     time is exactly

         T(delta) = (delta / sigma_ds)^2   bars

     (the standard two-sided first-passage identity E[tau] = a*b/sigma^2 with
     a = b = delta). This is the form named in this round's pre-registration
     and it is the form used.

  2. **Cost rate.** One exit is one round trip (out of the incumbent and,
     when a fresh entrant clears +delta, into it), costing 2c per unit
     notional. So the mechanism's cost rate is

         C(delta) = 2c / T(delta)   per bar,    = 2c / h   per day
                                                 with h = T/288 in days.

  3. **Value rate.** The value of re-deciding at frequency 1/h is bounded by
     the signal's own decay curve, which R-65 measured and committed:
     `reports/r65_holding_period/decay_pre_holdout.csv`, column
     `top1_spread_per_day` -- the mean forward log return of the
     highest-scoring asset minus the equal-weight basket's, per day, when the
     selection is refreshed every h days, on 2020-04-01 -> 2022-12-31, prices
     only. Call it V(h).

  4. **Marginal value equals marginal cost.** Maximise

         Net(h) = V(h) - 2c/h            (per day)

     over h. At an interior optimum V'(h*) = -2c/h*^2 -- marginal value of
     deciding less often equals marginal cost saved -- which is the
     first-order condition this derivation is built on. V is measured on a
     grid, so h* is found by piecewise-linear interpolation of V in log h on
     a fine grid over h in [0.25, 30] days. The h = 1-BAR ROW IS EXCLUDED, on
     `r65_decay.py`'s own written instruction ("the h=1-bar row is a
     microstructure artifact ... must not be quoted"), and the grid is
     truncated at 30 days because every longer row in that table has
     V(h) < 0. Neither bound is a tuned parameter: both are transcribed from
     the source measurement's own stated caveats.

  5. **Invert.** T* = h* * 288 bars, and

         delta_B(t) = sigma_ds(t) * sqrt(T*)

DECLARED VARIANT (D-B'), stated now so it cannot be produced later to taste.
T(delta) above is the mean exit time from the CENTRE of the band. A full
re-decision cycle of the mechanism -- exit at -delta, then re-entry at
+delta -- traverses the band's full width 2*delta, whose mean first passage
is 4*delta^2/sigma_ds^2. Under that reading the same h* implies

    delta_B'(t) = delta_B(t) / 2

D-B is PRIMARY (it is the pre-registered form); D-B' is reported beside it as
a declared factor-of-two sensitivity, in `novel_derived.csv`, and is NOT
separately backtested.

REGIME OF VALIDITY for D-B, and what this setting violates.
  Valid when: the score is a martingale over the relevant horizon; increments
  are homoskedastic; V(h) is the value actually available at re-decision
  frequency 1/h; and the band's two crossings are the only trading events.
  VIOLATED HERE: (i) the score is mean-reverting, not a martingale, so the
  true exit time is SHORTER than the diffusive one and delta_B therefore
  OVERSTATES the band needed -- stated before the number was computed;
  (ii) V(h) is measured with the incumbent re-chosen on a CALENDAR clock,
  while this mechanism re-chooses on a BARRIER clock, and the two are only
  equal in distribution for a martingale; (iii) `decay_pre_holdout.csv` is
  measured on W_TRAIN **and W_VAL** together, so h* has a mild in-sample
  dependence on the W_VAL evaluation window. That is DISCLOSED, not repaired:
  the alternative table (`decay_full.csv`) includes the reserved holdout and
  is therefore off limits to this branch entirely. (iv) Every interval in
  that table's `top1_spread` column contains zero.

(D-C) B-35's PRICED GRACE CURVE -- A BOUNDING DIAGNOSTIC, NOT AN ARM
--------------------------------------------------------------------
This round's own prices-only pre-measurement priced the mechanism directly:
`reports/r68_band/r68_grace_cost.csv` gives, per delta, the log-unit cost of
the grace periods the band buys and the log-unit fee saving of the round
trips it avoids. Their difference `net_log_units` is a measured objective in
delta. Its argmax over the measured grid is reported in `novel_derived.csv`
as D-C. It is NOT evaluated as an arm and NOT counted as a derivation,
because the measured grid stops at delta = 0.160 and the column is still
rising there -- an EDGE solution, which is the very artifact this round
exists to avoid. It is reported because dropping an unfavourable diagnostic
is the one thing this branch must not do.

THE CAP -- DERIVED, NOT CHOSEN, AND EVALUATED CAUSALLY
------------------------------------------------------
de Lataillade, J., & Chaouki, A. (2020), "Equations and Shape of the Optimal
Band Strategy", arXiv:2003.04646, Eq. (11): the optimal tolerance around zero
SATURATES at approximately 1.6 * sigma_signal, so no cost justifies an
arbitrarily wider band. `r68_shared.DLC_SATURATION` fixes the round's static
grid cap at 1.6 * 0.2295 = 0.3672 using a whole-window sigma, which that file
states may be used ONLY to bound a grid of constants and never inside a
per-bar decision. This branch therefore applies the SAME saturation with a
CAUSAL sigma:

    delta(t) <- min( delta(t), 1.6 * sigma_s(t) )

applied to D-A, to D-B and to every neighbourhood multiplier. The fraction of
bars on which the cap binds is measured and reported. The static
`DLC_SATURATION` is quoted only as a reference number.

A FLAW IN THE ROUND'S OWN FROZEN CAP, FOUND BY READING THE CITED PAPER AND
REPORTED RATHER THAN FIXED (the R-63 process violation, not repeated). This
paragraph was added AFTER the derivation had been computed; it changes no
formula and no number, and the arm's cap is left exactly as pre-registered.
dLC (2020) Eq. (11) reads

    lim_{Gamma -> inf}  u(0) - l(0)  =  sqrt(8/pi) * sigma_p  =  1.5958 * sigma_p

and `u(0) - l(0)` is the band's **FULL WIDTH**, not its half-width: u(0) and
l(0) are the upper and lower edges, and the paper's own small-cost limit on
the facing page is stated as the symmetric pair
`l(0) = -(3*Gamma*beta^2/2)^(1/3)`, `u(0) = +(3*Gamma*beta^2/2)^(1/3)`.
`r68_shared.DLC_SATURATION = 1.6 * sigma_score = 0.3672` therefore applies
the paper's saturated FULL width as a bound on a HALF width, and is
2x too generous against the source. Read literally, Eq. (11) caps the
half-width at `0.5 * sqrt(8/pi) * sigma = 0.7979 * sigma`, i.e. **0.1831** at
the round's own `sigma_score = 0.2295`.

NOTHING IN THIS BRANCH MOVES AS A RESULT. The pre-registered cap
`1.6 * sigma_s(t)` is kept, it is what the code applies, and it never binds on
any bar of any window this branch evaluates -- so both readings produce the
identical arm. Both ratios are reported side by side in `novel_derived.csv`
(`ratio_to_dlc_*_halfwidth`) so the operator can see where the derived
thresholds sit against the source's literal statement as well as against the
round's frozen constant. The other branch's grid, which is bounded BY that
constant, is not this branch's business and is not commented on.

Incidental corroboration from the same paper, worth recording because it is
independent of D-A's own sources: dLC's small-linear-cost limit for the band
edge is `(3/2 * Gamma * beta^2)^(1/3)` -- the SAME 1/3 exponent and the SAME
3/2 prefactor as the Janecek-Shreve form adopted in D-A, credited there to
Rogers (2004) and to de Lataillade, Deremble, Potters & Bouchaud (2012). The
constant in D-A is therefore not an artifact of one derivation.

A SECOND CAVEAT ON THE CAP, stated for completeness: dLC's band is a band in
predictor space around a CONTINUOUS target position proportional to the
predictor, in the Gamma -> infinity limit. This mechanism's band is on a
BINARY membership decision around zero. The cap is adopted because the round
adopted it, not because the two settings are the same problem.

NUMERICAL GUARDS -- DECLARED, NOT TUNED
---------------------------------------
  * `MIN_FINITE_BARS = 30 * 288`. Until the expanding estimator has seen 30
    days of bars with a finite score, delta(t) := 0.0 exactly -- which makes
    the arm identically R-65's frozen winner on those bars. 30 days is five
    half-lives of the composite score (half-life 5.99 days, R-67's
    measurement), i.e. the point at which an expanding sd has ~5 independent
    observations. It is a warm-up floor, not a fitted parameter, and it is
    reported (it costs W_VAL its first ~20 days, because that window's warm
    frame starts only 91 days before it).
  * Non-finite delta (warm-up, theta = 0, sigma undefined) := 0.0.
  * delta is clipped below at 0.0. It is never allowed negative.

CONSEQUENCE OF THE PER-WINDOW WARM FRAME, disclosed. Every arm in this
lineage builds its targets on a frame that starts `WARM_DAYS = 91` days
before the evaluation window. An EXPANDING estimator therefore sees a
different amount of history on W_VAL (warm start 2021-10-02) than it does on
W_FULL6 (warm start 2020-01-01), so the same calendar bar can carry a
different delta in two different cells. That is inherent to the substrate,
not introduced here, and it is reported rather than engineered around --
engineering around it would mean changing the frame the identity checks
depend on.

=====================================================================
THE MECHANISM -- IDENTICAL TO R-67's, WITH THE SCALAR MADE A SERIES
=====================================================================

    enter_eligible(t) = isfinite(s) & (s >  +delta(t))
    hold_eligible(t)  = isfinite(s) & (s >  -delta(t))

`r67_conservative_hysteresis.hysteresis_selection`'s loop, copied
byte-for-byte, with the two scalar comparisons broadcast against a per-bar
column instead. R-67's file is NOT edited and NOT imported for this: the loop
is reproduced here so that generalising the threshold cannot mutate a
committed arm. Everything else is frozen at R-65's selected winner: k = 1,
buffer = 0.05, hold_days = 1, R-63's composite score and conditional vol
scale, R-63's 0.10 deadband on desired total notional, equal weights among
held slots, long-only spot, unlevered. The sizing block below the selection
is R-63's, copied unmodified.

STRICTLY CAUSAL BY CONSTRUCTION: a forward loop whose state at bar i depends
on rows <= i, driven by a delta series each of whose inputs is an expanding
statistic over rows STRICTLY BEFORE i. Proven, not asserted, by three probes
in `checks`.

=====================================================================
THE IDENTITY CHECKS -- RUN BEFORE ANY OTHER NUMBER
=====================================================================

  (I1) delta(t) == 0.0 for all t  ->  the target matrix must be BIT-IDENTICAL
       to `r68_shared.r65_winner_targets`. max|diff| is reported whatever it
       is. At delta = 0 the two predicates `s > +0.0` and `s > -0.0` are the
       identical IEEE-754 comparison, so this is expected to be exact.
  (I2) delta(t) == 0.080 for all t  ->  must reproduce R-67's published
       W_TRAIN cell, read from `reports/r67_gate/conservative_frontier.csv`
       (R-67's committed record, opened read-only).

If either fails, the branch STOPS and reports that as its result.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- INHERITED, AND RESTATED
=====================================================================

(F1) another corner -- not applicable to a branch with no grid, but the
     neighbourhood report can still show the derived point at an end of its
     own multiplier ladder, and that is reported as such.
(F4) **theory and data may disagree** -- this branch's own named failure
     mode. The derived delta may land far from where R-67's sweep selected,
     or the two derivations may disagree with each other. R-65's and R-67's
     derived rates landed INSIDE independently measured windows, which is why
     that evidence was strong; a miss is equally informative and is reported
     as a miss.
(F5) the round's base case: `(D1 or D2)` fails again, because three rounds
     have improved this signal's economics by 10-80x and every one died on
     the same interval.

THE RESERVED HOLDOUT (2023-01-01 onward) IS NOT READ BY THIS BRANCH UNDER ANY
OUTCOME. Its window constant is never imported, never named and never
sliced; the D-cells are the W_FULL6/U6 and W_VAL/U8 cells R-63, R-65 and
R-67 all use. Verified by grep over this file, and reported.

Run as:
    .venv/bin/python experiments/r68_novel_derived_threshold.py derive
    .venv/bin/python experiments/r68_novel_derived_threshold.py identity
    .venv/bin/python experiments/r68_novel_derived_threshold.py checks
    .venv/bin/python experiments/r68_novel_derived_threshold.py frontier
    .venv/bin/python experiments/r68_novel_derived_threshold.py m1
    .venv/bin/python experiments/r68_novel_derived_threshold.py run
    .venv/bin/python experiments/r68_novel_derived_threshold.py scramble
    .venv/bin/python experiments/r68_novel_derived_threshold.py all
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tradebot.inference import (  # noqa: E402
    annualized_sharpe,
    daily_returns,
    deflated_sharpe_ratio,
    deflation_breakeven_sd,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
)

from experiments.r68_shared import (  # noqa: E402
    BARS_PER_DAY,
    DEADBAND,
    DLC_SATURATION,
    M1_MIN_REDUCTION,
    OUT_DIR,
    R63_TURNOVER_PER_DAY,
    R65_BUFFER,
    R65_HOLD_DAYS,
    R65_K,
    R65_OUT_DIR,
    R67_DELTA_WINNER,
    SCRAMBLE_SEEDS,
    SIGMA_SCORE_W_TRAIN,
    SPOT_BASE,
    SPOT_FREE,
    SPOT_REAL,
    UNIVERSE_6,
    UNIVERSE_8,
    W_FULL6,
    W_TRAIN,
    W_VAL,
    align_frames,
    basket_log_returns,
    check_causality,
    compare,
    conditional_vol_scale,
    config_count,
    cross_sectional_score,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    frontier_row,
    further_work,
    held_indicator,
    holding_period_days,
    load_universe,
    m1_pass,
    matched_hold_targets,
    mean_total_notional,
    membership_change_rate_thresholded,
    r63_baseline_targets,
    r65_winner_targets,
    realized_vol,
    scramble_fixed_perm,
    simulate_portfolio,
    static_hold_equity,
    turnover_stats,
    volmatched_hold_equity,
    warm_window,
)
from experiments.r68_shared import D5_BAR_R68  # noqa: E402

ARM = "derived_threshold"

# ---- frozen substrate: R-65's selected winner, inherited, never re-selected
K_FIXED = R65_K            # 1
BUFFER_FIXED = R65_BUFFER  # 0.05
HOLD_FIXED = R65_HOLD_DAYS  # 1

# ---- derivation constants, all declared in the docstring above
FEE = 0.001          # c, SPOT_BASE's 0.10% taker tier
GAMMA = 1.0          # log utility, this repo's standing convention
DLC_K = 1.6          # de Lataillade & Chaouki (2020) Eq. (11) saturation, as
                     # the round froze it. UNCHANGED, and it never binds.
# The same equation read literally: sqrt(8/pi) is the saturated FULL width, so
# the implied HALF width is half of it. Reporting only -- see the docstring.
DLC_EQ11_FULL_WIDTH = math.sqrt(8.0 / math.pi)          # 1.59577
DLC_EQ11_HALF_WIDTH = 0.5 * DLC_EQ11_FULL_WIDTH         # 0.79788
DLC_LITERAL_HALFWIDTH_CAP = DLC_EQ11_HALF_WIDTH * SIGMA_SCORE_W_TRAIN  # 0.1831
MIN_FINITE_BARS = 30 * BARS_PER_DAY   # warm-up floor: 5 score half-lives

# R-65's committed decay table -- the value side of D-B. Read-only.
DECAY_CSV = R65_OUT_DIR / "decay_pre_holdout.csv"
DECAY_H_LO, DECAY_H_HI = 0.25, 30.0   # both bounds transcribed from r65_decay.py

# R-67's committed record, for identity check (I2). Read-only.
R67_FRONTIER_CSV = ROOT / "reports" / "r67_gate" / "conservative_frontier.csv"

# This round's prices-only pre-measurement, for D-C. Read-only.
GRACE_CSV = OUT_DIR / "r68_grace_cost.csv"

# The context neighbourhood. REPORTED, NOT SELECTED ON.
NEIGHBOURHOOD = (0.5, 0.75, 1.0, 1.5, 2.0)

# Every distinct threshold specification this file backtests, for the trials
# accounting. Populated by `evaluate`; never used to choose anything.
_SPECS: set[str] = set()


# ======================================================================
# 1. THE DERIVATION -- causal estimators
# ======================================================================


def _shift1(a: np.ndarray) -> np.ndarray:
    """One-bar shift with a zero/NaN-safe leading value."""
    out = np.empty_like(a, dtype=float)
    out[0] = 0.0
    out[1:] = a[:-1]
    return out


def expanding_pooled_sd(x: np.ndarray, min_finite_bars: int) -> np.ndarray:
    """Pooled sd of ``x`` across assets over every row STRICTLY BEFORE t.

    EXPANDING and SHIFTED BY ONE BAR, which is this round's pre-registered
    causality requirement. Population sd (divide by n). Returns NaN until the
    estimator has seen ``min_finite_bars`` bars carrying at least one finite
    entry.
    """
    finite = np.isfinite(x)
    v = np.where(finite, x, 0.0)
    cnt = np.cumsum(finite.sum(axis=1, dtype=float))
    s1 = np.cumsum(v.sum(axis=1))
    s2 = np.cumsum((v * v).sum(axis=1))
    rows = np.cumsum(finite.any(axis=1).astype(float))

    cnt, s1, s2, rows = _shift1(cnt), _shift1(s1), _shift1(s2), _shift1(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1.0), np.nan)
        var = np.where(cnt > 1, s2 / np.maximum(cnt, 1.0) - mean ** 2, np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(rows >= min_finite_bars, sd, np.nan)
    return sd


def sigma_signal_series(aligned: dict[str, pd.DataFrame]) -> np.ndarray:
    """sigma_s(t): expanding, one-bar-shifted pooled sd of the score."""
    s = cross_sectional_score(aligned).to_numpy(dtype=float)
    return expanding_pooled_sd(s, MIN_FINITE_BARS)


def sigma_dscore_series(aligned: dict[str, pd.DataFrame]) -> np.ndarray:
    """sigma_ds(t): the same estimator on the per-bar score increment."""
    s = cross_sectional_score(aligned).to_numpy(dtype=float)
    d = np.full_like(s, np.nan)
    d[1:] = s[1:] - s[:-1]
    return expanding_pooled_sd(d, MIN_FINITE_BARS)


def theta_series(aligned: dict[str, pd.DataFrame]) -> np.ndarray:
    """theta(t) = min(scale(t-1), 1.0): the exposure the arm intends to hold.

    R-63's own `conditional_vol_scale`, capped at the long-only unlevered 1.0
    and shifted one EXTRA bar beyond the shift already inside that function.
    """
    scale = conditional_vol_scale(basket_log_returns(aligned))
    th = np.minimum(np.asarray(scale, dtype=float), 1.0)
    out = np.full_like(th, np.nan)
    out[1:] = th[:-1]
    return out


def _finalize(raw: np.ndarray, sigma_s: np.ndarray) -> np.ndarray:
    """Apply the dLC (2020) Eq. (11) saturation CAUSALLY, then the guards."""
    cap = DLC_K * sigma_s
    d = np.minimum(raw, cap)
    d = np.where(np.isfinite(d), d, 0.0)
    return np.maximum(d, 0.0)


def delta_A_series(aligned: dict[str, pd.DataFrame], mult: float = 1.0) -> np.ndarray:
    """(D-A) the cube-root band, in score units. See the module docstring."""
    sig = sigma_signal_series(aligned)
    th = theta_series(aligned)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (sig
               * (3.0 * FEE / (2.0 * GAMMA)) ** (1.0 / 3.0)
               * np.power(th, -1.0 / 3.0)
               * np.power(np.maximum(1.0 - th, 0.0), 2.0 / 3.0))
    return _finalize(float(mult) * raw, sig)


def decay_optimum() -> dict:
    """(D-B) step 4: h* = argmax_h [ V(h) - 2c/h ], from R-65's frozen table.

    Piecewise-linear interpolation of V in log h over the table's own usable
    range. Also returns the marginal-condition residual at h*.
    """
    df = pd.read_csv(DECAY_CSV)
    use = df[(df["hold_days"] >= DECAY_H_LO) & (df["hold_days"] <= DECAY_H_HI)]
    h = use["hold_days"].to_numpy(dtype=float)
    v = use["top1_spread_per_day"].to_numpy(dtype=float)

    grid = np.exp(np.linspace(math.log(h[0]), math.log(h[-1]), 200_001))
    vg = np.interp(np.log(grid), np.log(h), v)
    net = vg - 2.0 * FEE / grid
    j = int(np.argmax(net))
    hstar = float(grid[j])

    # Marginal condition V'(h*) = -2c/h*^2. The interpolant is piecewise
    # linear, so if h* lands on a table NODE the derivative does not exist
    # there and the condition holds in the SUBGRADIENT sense instead:
    #     V'(h*-) >= -2c/h*^2 >= V'(h*+)
    # Both one-sided slopes are reported; the central difference is reported
    # too but must NOT be quoted as "the" derivative at a kink.
    eps = hstar * 1e-4
    def _v(x):
        return float(np.interp(math.log(x), np.log(h), v))
    vprime = (_v(hstar + eps) - _v(hstar - eps)) / (2 * eps)
    vprime_l = (_v(hstar) - _v(hstar - eps)) / eps
    vprime_r = (_v(hstar + eps) - _v(hstar)) / eps
    mc = -2.0 * FEE / hstar ** 2
    on_node = bool(np.min(np.abs(h - hstar)) < 1e-3)
    return {
        "V_prime_left": vprime_l, "V_prime_right": vprime_r,
        "h_star_on_table_node": on_node,
        "subgradient_condition_holds":
            bool(vprime_l >= mc >= vprime_r),
        "h_star_days": hstar,
        "T_star_bars": hstar * BARS_PER_DAY,
        "net_at_h_star_per_day": float(net[j]),
        "V_at_h_star_per_day": float(vg[j]),
        "cost_at_h_star_per_day": float(2.0 * FEE / hstar),
        "V_prime_at_h_star": vprime,
        "marginal_cost_at_h_star": mc,
        "marginal_residual": vprime - mc,
        "grid_h_lo": float(h[0]), "grid_h_hi": float(h[-1]),
        "n_table_rows_used": int(len(use)),
        "interior": bool(h[0] < hstar < h[-1]),
    }


_DECAY_CACHE: dict = {}


def t_star_bars() -> float:
    if "d" not in _DECAY_CACHE:
        _DECAY_CACHE["d"] = decay_optimum()
    return float(_DECAY_CACHE["d"]["T_star_bars"])


def delta_B_series(aligned: dict[str, pd.DataFrame], mult: float = 1.0) -> np.ndarray:
    """(D-B) the cost-matched first-passage band. See the module docstring."""
    sig = sigma_signal_series(aligned)
    sdd = sigma_dscore_series(aligned)
    raw = float(mult) * sdd * math.sqrt(t_star_bars())
    return _finalize(raw, sig)


def delta_series(aligned: dict[str, pd.DataFrame], kind: str,
                 mult: float = 1.0) -> np.ndarray:
    """The one entry point every backtest goes through.

    ``kind`` is "D_A", "D_B", or "const:<x>". The whole derivation lives
    INSIDE this call, so the truncation and perturbation probes exercise the
    estimator and not merely the selection loop.
    """
    if kind.startswith("const:"):
        n = len(next(iter(aligned.values())))
        return np.full(n, float(kind.split(":", 1)[1]))
    if kind == "D_A":
        return delta_A_series(aligned, mult)
    if kind == "D_B":
        return delta_B_series(aligned, mult)
    raise ValueError(f"unknown delta kind {kind!r}")


# ======================================================================
# 2. THE MECHANISM -- R-67's loop, threshold generalised to a series
# ======================================================================


def derived_selection(s: np.ndarray, k: int, buffer: float, hold_days: float,
                      delta: np.ndarray):
    """`r67_conservative_hysteresis.hysteresis_selection`, byte-for-byte,
    with the scalar `delta` replaced by a per-bar column.

    R-67:   enter = s > +delta        hold = s > -delta      (delta scalar)
    Here:   enter = s > +delta[t]     hold = s > -delta[t]   (delta series)

    With `delta` a constant series this is exactly R-67's function; with the
    constant 0.0 both predicates collapse to `s > 0.0` (IEEE-754: -0.0 == 0.0)
    and this is R-65's `buffered_selection`.

    STRICTLY CAUSAL: a forward loop whose state at bar i depends on rows <= i.
    `delta` is supplied by :func:`delta_series`, every input to which is an
    expanding statistic over rows strictly before i.
    """
    n, n_assets = s.shape
    d = np.asarray(delta, dtype=float)
    if d.ndim == 0:
        d = np.full(n, float(d))
    if d.shape != (n,):
        raise ValueError(f"delta series shape {d.shape} != ({n},)")

    finite = np.isfinite(s)
    enter_eligible = finite & (s > d[:, None])
    hold_eligible = finite & (s > -d[:, None])
    hold_bars = int(round(float(hold_days) * BARS_PER_DAY))
    buf = float(buffer)

    sel = np.zeros((n, n_assets), dtype=bool)
    held: list[int] = []
    last_change = -(1 << 60)
    keys = ("forced_exit", "entry", "swap", "blocked_by_timer",
            "blocked_by_buffer", "flat_bars")
    ev = {key: 0 for key in keys}

    for i in range(n):
        row = s[i]
        elig_in = enter_eligible[i]
        elig_hold = hold_eligible[i]
        changed = False

        # (a) forced exits -- never blocked by the timer.
        if held:
            keep = [a for a in held if elig_hold[a]]
            if len(keep) != len(held):
                ev["forced_exit"] += 1
                held = keep
                changed = True

        # entries into empty slots, allowed immediately (R-65's rule).
        if len(held) < k:
            free = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if free:
                free.sort(key=lambda a: -row[a])
                for a in free[: k - len(held)]:
                    held.append(a)
                    changed = True
                ev["entry"] += 1

        # (b) voluntary swap -- buffered AND time-gated.
        elif not changed:
            worst = min(held, key=lambda a: row[a])
            out = [a for a in range(n_assets) if elig_in[a] and a not in held]
            if out:
                best = max(out, key=lambda a: row[a])
                if row[best] > row[worst] + buf:
                    if (i - last_change) >= hold_bars:
                        held.remove(worst)
                        held.append(best)
                        changed = True
                        ev["swap"] += 1
                    else:
                        ev["blocked_by_timer"] += 1
                elif row[best] > row[worst]:
                    ev["blocked_by_buffer"] += 1

        if changed:
            last_change = i
        if held:
            sel[i, held] = True
        else:
            ev["flat_bars"] += 1

    return sel, ev


def build_derived_targets(aligned: dict[str, pd.DataFrame], delta: np.ndarray,
                          k: int = K_FIXED, buffer: float = BUFFER_FIXED,
                          hold_days: float = HOLD_FIXED) -> pd.DataFrame:
    """R-65's `build_buffered_targets` with :func:`derived_selection` in place
    of `buffered_selection`. The sizing block is R-63's, copied unmodified."""
    score = cross_sectional_score(aligned)
    assets = list(score.columns)
    s = score.to_numpy(dtype=float)
    n = s.shape[0]

    sel, _ = derived_selection(s, k, buffer, hold_days, delta)

    # ---- sizing: R-63's, untouched -------------------------------------
    m = sel.sum(axis=1)
    scale = conditional_vol_scale(basket_log_returns(aligned))

    desired = scale * (m / float(k))
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        d = desired[i]
        if abs(d - cur) > DEADBAND:
            cur = d
        pos[i] = cur

    total = np.minimum(pos, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(m > 0, total / np.maximum(m, 1), 0.0)
    w = sel * per[:, None]
    return pd.DataFrame(w, index=score.index, columns=assets)


def targets_fn(kind: str, mult: float = 1.0):
    """A pure `aligned -> targets` closure. The derivation runs inside it."""
    return lambda aligned: build_derived_targets(
        aligned, delta_series(aligned, kind, mult))


def spec_name(kind: str, mult: float) -> str:
    if kind.startswith("const:"):
        return kind
    return f"{kind}x{mult:g}"


# ======================================================================
# 3. cells / io
# ======================================================================


_WARM_CACHE: dict = {}


def _warm_frames(frames, universe, window):
    key = (tuple(universe), window)
    if key not in _WARM_CACHE:
        _WARM_CACHE[key] = align_frames({t: frames[t] for t in universe},
                                        warm_window(window))
    return _WARM_CACHE[key]


def _slice_index(warm: dict[str, pd.DataFrame], window):
    """STRICT right-exclusive slice, independent of the shared `_hi` helper."""
    idx = next(iter(warm.values())).index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    if window[1] is not None:
        idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    return idx


def build_cell(frames, universe, window, kind, mult=1.0,
               baseline=False, r65=False):
    warm = _warm_frames(frames, universe, window)
    if baseline:
        targets = r63_baseline_targets(warm, K_FIXED)
    elif r65:
        targets = r65_winner_targets(warm)
    else:
        targets = build_derived_targets(warm, delta_series(warm, kind, mult))
    idx = _slice_index(warm, window)
    score = cross_sectional_score(warm).loc[idx]
    first_warm = bool(np.isfinite(score.iloc[0].to_numpy()).all())
    aligned_eval = {t: df.loc[idx] for t, df in warm.items()}
    return aligned_eval, targets.loc[idx], first_warm


def membership_stats(targets: pd.DataFrame) -> dict:
    w = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0)
    n = len(w)
    days = max(n / BARS_PER_DAY, 1e-9)
    ind = held_indicator(targets)
    starts = int(ind[0].sum() + (ind[1:] & ~ind[:-1]).sum())
    return {
        "membership_changes_gt1pct_per_day":
            membership_change_rate_thresholded(targets),
        "mean_n_held_gt1pct": float(ind.sum(axis=1).mean()),
        "mean_tenure_gt1pct_days":
            float(ind.sum()) / max(starts, 1) / BARS_PER_DAY,
        "frac_bars_flat": float((w.sum(axis=1) <= 0.0).mean()),
        "days": days,
    }


def delta_stats(warm, idx, kind, mult) -> dict:
    """Descriptive statistics of the delta series ON THE EVALUATION SLICE."""
    if kind.startswith("const:"):
        d = delta_series(warm, kind, mult)
        ser = pd.Series(d, index=next(iter(warm.values())).index).loc[idx]
        arr = ser.to_numpy(dtype=float)
        return {"delta_mean": float(arr.mean()), "delta_median": float(np.median(arr)),
                "delta_min": float(arr.min()), "delta_max": float(arr.max()),
                "delta_final": float(arr[-1]), "delta_cap_frac": 0.0,
                "delta_zero_frac": float((arr <= 0.0).mean())}
    full_idx = next(iter(warm.values())).index
    sig = pd.Series(sigma_signal_series(warm), index=full_idx).loc[idx].to_numpy()
    d = pd.Series(delta_series(warm, kind, mult), index=full_idx).loc[idx].to_numpy()
    cap = DLC_K * sig
    binding = np.isfinite(cap) & (d >= cap - 1e-15) & (d > 0)
    return {
        "delta_mean": float(np.mean(d)), "delta_median": float(np.median(d)),
        "delta_min": float(np.min(d)), "delta_max": float(np.max(d)),
        "delta_final": float(d[-1]),
        "delta_cap_frac": float(np.mean(binding)),
        "delta_zero_frac": float(np.mean(d <= 0.0)),
        "sigma_s_final": float(sig[-1]) if np.isfinite(sig[-1]) else float("nan"),
        "delta_over_sigma_mean":
            float(np.nanmean(np.where(sig > 0, d / np.where(sig > 0, sig, np.nan),
                                      np.nan))),
    }


def evaluate(targets, aligned, assets, window_name, universe_name, params,
             spec: str):
    """One frontier row: 0.10% and 0 bps, both against VOLMATCH_HOLD."""
    _SPECS.add(f"{spec}@{window_name}")
    out, cmps = {}, {}
    for tag, market in (("net", SPOT_BASE), ("gross", SPOT_FREE)):
        cand = simulate_portfolio(targets, aligned, market)
        bench, c, vol, matched = volmatched_hold_equity(cand, aligned, assets,
                                                        market)
        if bench is None:
            raise RuntimeError(f"{spec} {window_name}: volmatch gave no benchmark")
        cmps[tag] = compare(cand, bench)
        out[f"{tag}_volmatch_c"] = c
        out[f"{tag}_volmatch_vol"] = vol
        out[f"{tag}_volmatch_matched"] = matched
        out[f"{tag}_cand_vol"] = realized_vol(cand)
    row = frontier_row(ARM, params, targets, cmps["net"], cmps["gross"],
                       "VOLMATCH_HOLD", window_name, universe_name, **out)
    row["spec"] = spec
    row.update(membership_stats(targets))
    return row


def write_csv(path, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")
    return path


def fmt_front(r):
    return (f"    hold {r['hold_days']:8.3f}d | turn {r['turnover_per_day']:8.4f}/d"
            f" | mtn {r['mean_notional']:.3f}"
            f" | GROSS {r['gross_growth_diff']:+8.3f}"
            f" [{r['gross_growth_lo']:+7.3f},{r['gross_growth_hi']:+7.3f}]"
            f" | NET {r['net_growth_diff']:+8.3f}"
            f" [{r['net_growth_lo']:+7.3f},{r['net_growth_hi']:+7.3f}]"
            f" | ddiff {r['net_dd_diff']:+7.2f}")


# ======================================================================
# 4. commands
# ======================================================================


def cmd_derive(frames):
    """Both derivations, their inputs, and where each sits versus the cap."""
    print("== DERIVATION (formulae frozen in the module docstring) ==")
    rows = []

    dec = decay_optimum()
    _DECAY_CACHE["d"] = dec
    print(f"  (D-B) value side, from {DECAY_CSV.relative_to(ROOT)}"
          f"  ({dec['n_table_rows_used']} usable rows, h in "
          f"[{dec['grid_h_lo']:g},{dec['grid_h_hi']:g}] days)")
    print(f"    h*        = {dec['h_star_days']:.4f} days "
          f"({dec['T_star_bars']:.1f} bars)   interior={dec['interior']}")
    print(f"    V(h*)     = {dec['V_at_h_star_per_day']:+.8f} /day")
    print(f"    2c/h*     = {dec['cost_at_h_star_per_day']:+.8f} /day")
    print(f"    Net(h*)   = {dec['net_at_h_star_per_day']:+.8f} /day")
    print(f"    marginal condition: -2c/h*^2 = "
          f"{dec['marginal_cost_at_h_star']:+.4e}")
    print(f"      h* lands on a table node: {dec['h_star_on_table_node']}"
          f"  -> the interpolant has a KINK there and V' does not exist;"
          f" the condition is a SUBGRADIENT inclusion")
    print(f"      V'(h*-) = {dec['V_prime_left']:+.4e}  >=  -2c/h*^2 = "
          f"{dec['marginal_cost_at_h_star']:+.4e}  >=  V'(h*+) = "
          f"{dec['V_prime_right']:+.4e}   holds: "
          f"{dec['subgradient_condition_holds']}")
    print(f"      (central difference across the kink, NOT the derivative: "
          f"{dec['V_prime_at_h_star']:+.3e})")
    rows.append({"item": "D_B_value_side", **dec, "source": str(DECAY_CSV)})

    for uni_name, uni, window in (("U8", UNIVERSE_8, W_TRAIN),
                                  ("U8", UNIVERSE_8, W_VAL),
                                  ("U6", UNIVERSE_6, W_FULL6)):
        warm = _warm_frames(frames, uni, window)
        idx = _slice_index(warm, window)
        full_idx = next(iter(warm.values())).index
        wname = ("W_TRAIN" if window is W_TRAIN
                 else "W_VAL" if window is W_VAL else "W_FULL6")

        sig = pd.Series(sigma_signal_series(warm), index=full_idx).loc[idx].to_numpy()
        sdd = pd.Series(sigma_dscore_series(warm), index=full_idx).loc[idx].to_numpy()
        th = pd.Series(theta_series(warm), index=full_idx).loc[idx].to_numpy()

        print(f"  -- {wname} {uni_name}  ({len(idx):,} bars, "
              f"{idx[0].date()} -> {idx[-1].date()}) --")
        print(f"    sigma_s   (expanding, shifted): final {sig[-1]:.6f}"
              f"  mean {np.nanmean(sig):.6f}  min {np.nanmin(sig):.6f}")
        print(f"    sigma_ds  (expanding, shifted): final {sdd[-1]:.8f}"
              f"  mean {np.nanmean(sdd):.8f}")
        print(f"    theta     (min(scale,1), shifted): mean {np.nanmean(th):.4f}"
              f"  median {np.nanmedian(th):.4f}")

        for kind, label in (("D_A", "D-A cube-root"), ("D_B", "D-B first-passage")):
            st = delta_stats(warm, idx, kind, 1.0)
            dfin = st["delta_final"]
            print(f"    {label:22s} delta*: mean {st['delta_mean']:.6f}"
                  f"  median {st['delta_median']:.6f}"
                  f"  final {dfin:.6f}"
                  f"  max {st['delta_max']:.6f}")
            print(f"        cap binds on {st['delta_cap_frac']:.2%} of bars;"
                  f"  delta==0 on {st['delta_zero_frac']:.2%};"
                  f"  mean delta/sigma_s = {st['delta_over_sigma_mean']:.4f}"
                  f"  (dLC saturation is 1.6)")
            print(f"        vs R-67's fitted {R67_DELTA_WINNER}: "
                  f"mean ratio {st['delta_mean'] / R67_DELTA_WINNER:.3f}x"
                  f"  | vs the round's frozen dLC cap {DLC_SATURATION:.4f}: "
                  f"{st['delta_mean'] / DLC_SATURATION:.3f}x")
            over = st["delta_mean"] > DLC_LITERAL_HALFWIDTH_CAP
            verdict = ("ABOVE the literal cap -- NOT DEFENSIBLE under that "
                       "reading" if over else "inside the literal cap")
            print(f"        vs dLC Eq.(11) read LITERALLY as a full width "
                  f"(half-width cap {DLC_LITERAL_HALFWIDTH_CAP:.4f}): "
                  f"{st['delta_mean'] / DLC_LITERAL_HALFWIDTH_CAP:.3f}x"
                  f"  -> {verdict}")
            rows.append({"item": f"delta_{kind}", "window": wname,
                         "universe": uni_name, "n_bars": len(idx),
                         "mult": 1.0, **st,
                         "r67_fitted_delta": R67_DELTA_WINNER,
                         "ratio_to_r67_fitted": st["delta_mean"] / R67_DELTA_WINNER,
                         "dlc_static_cap_as_frozen": DLC_SATURATION,
                         "ratio_to_dlc_frozen_halfwidth":
                             st["delta_mean"] / DLC_SATURATION,
                         "dlc_literal_halfwidth_cap": DLC_LITERAL_HALFWIDTH_CAP,
                         "ratio_to_dlc_literal_halfwidth":
                             st["delta_mean"] / DLC_LITERAL_HALFWIDTH_CAP,
                         "above_dlc_literal_halfwidth": bool(over),
                         "sigma_s_final": st.get("sigma_s_final"),
                         "sigma_ds_final": float(sdd[-1]),
                         "theta_mean": float(np.nanmean(th)),
                         "fee_c": FEE, "gamma": GAMMA,
                         "h_star_days": dec["h_star_days"],
                         "T_star_bars": dec["T_star_bars"]})

        # the declared factor-of-two sensitivity, reported not backtested
        stb = delta_stats(warm, idx, "D_B", 1.0)
        rows.append({"item": "delta_D_B_prime_DECLARED_VARIANT", "window": wname,
                     "universe": uni_name, "mult": 0.5,
                     "delta_mean": stb["delta_mean"] / 2.0,
                     "delta_median": stb["delta_median"] / 2.0,
                     "delta_final": stb["delta_final"] / 2.0,
                     "note": "full-width 2*delta traverse; NOT backtested"})

    # (D-C) B-35's priced grace curve -- bounding diagnostic only
    try:
        g = pd.read_csv(GRACE_CSV)
        g = g[g["window"] == "PRE_HOLDOUT"]
        j = int(np.argmax(g["net_log_units"].to_numpy()))
        best = g.iloc[j]
        edge = bool(j == len(g) - 1)
        print(f"  (D-C) B-35 priced grace curve, PRE_HOLDOUT: argmax "
              f"net_log_units at delta={best['delta']:g} "
              f"({best['net_log_units']:+.4f} log units)  EDGE={edge}")
        print("        NOT evaluated as an arm: the measured grid stops at "
              "0.160 and the column is still rising -> an edge solution.")
        rows.append({"item": "D_C_grace_curve_DIAGNOSTIC",
                     "delta_argmax": float(best["delta"]),
                     "net_log_units": float(best["net_log_units"]),
                     "fee_saving_log_units": float(best["fee_saving_log_units"]),
                     "total_grace_log_ret": float(best["total_grace_log_ret"]),
                     "is_grid_edge": edge, "source": str(GRACE_CSV),
                     "note": "bounding diagnostic; NOT evaluated as an arm"})
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  (D-C) unavailable: {exc}")

    write_csv(OUT_DIR / "novel_derived.csv", rows)
    return rows


def cmd_identity(frames):
    """(I1) delta==0 reproduces R-65's frozen winner bit-for-bit.
       (I2) delta==0.080 reproduces R-67's published W_TRAIN cell."""
    print("== IDENTITY CHECKS (before any other number) ==")
    rows = []

    for uni_name, uni, window, wname in (("U8", UNIVERSE_8, W_TRAIN, "W_TRAIN"),
                                         ("U6", UNIVERSE_6, W_FULL6, "W_FULL6")):
        warm = _warm_frames(frames, uni, window)
        base = r65_winner_targets(warm)
        mine = build_derived_targets(warm, delta_series(warm, "const:0.0"))
        b = np.nan_to_num(base.to_numpy(dtype=float), nan=0.0)
        m = np.nan_to_num(mine.to_numpy(dtype=float), nan=0.0)
        dmax = float(np.abs(m - b).max())
        exact = bool(np.array_equal(m, b))
        print(f"  (I1) {wname} {uni_name} ({len(b):,} bars): "
              f"max|diff| = {dmax:.3e}   bit-identical = {exact}")
        rows.append({"check": "I1_delta0_vs_r65_winner", "window": wname,
                     "universe": uni_name, "n_bars": len(b),
                     "max_abs_diff": dmax, "bit_identical": exact,
                     "passed": exact})
        if not exact:
            print("    !! IDENTITY (I1) FAILED -- STOPPING per the "
                  "pre-registration.")
            write_csv(OUT_DIR / "novel_checks.csv", rows)
            raise SystemExit("I1 failed")

    # (I2) R-67's published W_TRAIN cell at delta = 0.080
    ref = pd.read_csv(R67_FRONTIER_CSV)
    ref = ref[(ref["window"] == "W_TRAIN") & (ref["config_kind"] == "grid")
              & (np.abs(ref["p_delta"] - R67_DELTA_WINNER) < 1e-12)]
    if len(ref) != 1:
        raise RuntimeError(f"expected exactly one R-67 W_TRAIN delta=0.080 row,"
                           f" got {len(ref)}")
    ref = ref.iloc[0]

    aligned, tg, warm_ok = build_cell(frames, UNIVERSE_8, W_TRAIN,
                                      f"const:{R67_DELTA_WINNER}")
    row = evaluate(tg, aligned, UNIVERSE_8, "W_TRAIN", "U8",
                   {"delta": R67_DELTA_WINNER, "kind": "const"},
                   f"const:{R67_DELTA_WINNER}")
    checks = {
        "turnover_per_day": (row["turnover_per_day"], float(ref["turnover_per_day"])),
        "hold_days": (row["hold_days"], float(ref["hold_days"])),
        "mean_notional": (row["mean_notional"], float(ref["mean_notional"])),
        "net_growth_diff": (row["net_growth_diff"], float(ref["net_growth_diff"])),
        "gross_growth_diff": (row["gross_growth_diff"], float(ref["gross_growth_diff"])),
        "net_dd_diff": (row["net_dd_diff"], float(ref["net_dd_diff"])),
        "cand_final": (row["cand_final"], float(ref["cand_final"])),
    }
    print(f"  (I2) delta=0.080 vs R-67's published W_TRAIN cell "
          f"({R67_FRONTIER_CSV.relative_to(ROOT)}):")
    ok = True
    out = {"check": "I2_delta0p080_vs_r67_published", "window": "W_TRAIN",
           "universe": "U8", "first_bar_warm": warm_ok}
    for name, (mine_v, ref_v) in checks.items():
        rel = abs(mine_v - ref_v) / max(abs(ref_v), 1e-12)
        good = rel < 1e-6
        ok = ok and good
        print(f"       {name:20s} mine {mine_v:+14.6f}  R-67 {ref_v:+14.6f}"
              f"   rel {rel:.2e}  {'OK' if good else 'MISMATCH'}")
        out[f"{name}_mine"] = mine_v
        out[f"{name}_r67"] = ref_v
        out[f"{name}_rel_diff"] = rel
    out["passed"] = bool(ok)
    rows.append(out)
    print(f"  (I2) reproduced: {ok}")
    if not ok:
        print("    !! IDENTITY (I2) FAILED -- reported as the branch result.")
    return rows


def cmd_checks(frames, identity_rows=None):
    """Causality probes on the DERIVED (time-varying) threshold."""
    print("== CAUSALITY PROBES (on the derived, time-varying threshold) ==")
    rows = list(identity_rows or [])
    warm = _warm_frames(frames, UNIVERSE_8, W_TRAIN)
    n = len(next(iter(warm.values())))

    for kind in ("D_A", "D_B"):
        fn = targets_fn(kind, 1.0)
        out = {"check": f"probes_{kind}", "window": "W_TRAIN", "universe": "U8",
               "n_bars": n}

        # (a) truncation probe at 60%: overlapping prefix must be unchanged
        cut = int(n * 0.6)
        full = fn(warm)
        trunc = fn({t: df.iloc[:cut] for t, df in warm.items()})
        af = np.nan_to_num(full.iloc[:cut].to_numpy(dtype=float), nan=0.0)
        bf = np.nan_to_num(trunc.to_numpy(dtype=float), nan=0.0)
        exact60 = bool(np.array_equal(af, bf))
        dmax = float(np.abs(af - bf).max())
        print(f"  [{kind}] truncation @60% ({cut:,}/{n:,} bars): "
              f"bit-identical={exact60}  max|diff|={dmax:.3e}")
        out.update(trunc60_bit_identical=exact60, trunc60_max_abs_diff=dmax,
                   trunc60_cut_row=cut)

        # the same probe on the DELTA SERIES itself, which is the quantity at
        # risk -- a target matrix can agree by luck where the threshold does not
        d_full = delta_series(warm, kind, 1.0)[:cut]
        d_tr = delta_series({t: df.iloc[:cut] for t, df in warm.items()}, kind, 1.0)
        dd = float(np.abs(d_full - d_tr).max())
        print(f"  [{kind}] truncation @60% on the DELTA SERIES: "
              f"max|diff|={dd:.3e}  bit-identical={bool(np.array_equal(d_full, d_tr))}")
        out.update(trunc60_delta_max_abs_diff=dd,
                   trunc60_delta_bit_identical=bool(np.array_equal(d_full, d_tr)))

        # (b) tail-perturbation probe: multiply the last 40% of closes by a
        #     constant; the early prefix must be untouched.
        cutp = int(n * 0.6)
        bad = {}
        for t, df in warm.items():
            dfx = df.copy()
            for col in ("open", "high", "low", "close"):
                v = dfx[col].to_numpy(dtype=float).copy()
                v[cutp:] *= 10.0
                dfx[col] = v
            bad[t] = dfx
        pa = np.nan_to_num(fn(warm).to_numpy(dtype=float)[:cutp], nan=0.0)
        pb = np.nan_to_num(fn(bad).to_numpy(dtype=float)[:cutp], nan=0.0)
        perturb = bool(np.allclose(pa, pb, atol=1e-12, rtol=0.0))
        da = delta_series(warm, kind, 1.0)[:cutp]
        db = delta_series(bad, kind, 1.0)[:cutp]
        perturb_d = bool(np.allclose(da, db, atol=1e-15, rtol=0.0))
        print(f"  [{kind}] tail x10 perturbation, early prefix unchanged: "
              f"targets={perturb}  delta series={perturb_d}")
        out.update(perturbation_targets=perturb, perturbation_delta=perturb_d,
                   perturbation_delta_max_abs_diff=float(np.abs(da - db).max()))

        # (c) r63_shared's own truncation probe
        c1 = check_causality(fn, warm)
        print(f"  [{kind}] check_causality (shared probe): {c1}")
        out["check_causality_shared"] = c1

        # (d) the delta series is genuinely time-varying, not a constant
        d = delta_series(warm, kind, 1.0)
        idx = _slice_index(warm, W_TRAIN)
        st = delta_stats(warm, idx, kind, 1.0)
        out.update({f"eval_{k}": v for k, v in st.items()})
        print(f"  [{kind}] eval-slice delta: mean {st['delta_mean']:.6f} "
              f"min {st['delta_min']:.6f} max {st['delta_max']:.6f} "
              f"(constant={bool(np.nanmax(d) == np.nanmin(d))})")
        rows.append(out)

    # (e) monotone response of turnover to the multiplier
    idx = _slice_index(warm, W_TRAIN)
    resp = []
    print("  turnover / membership response to the multiplier (D-B, W_TRAIN):")
    for mlt in NEIGHBOURHOOD:
        tg = build_derived_targets(warm, delta_series(warm, "D_B", mlt)).loc[idx]
        ts = turnover_stats(tg)
        ms = membership_stats(tg)
        resp.append(ts["turnover_per_day"])
        print(f"    {mlt:>5g}x  turn {ts['turnover_per_day']:8.4f}/d "
              f"hold {holding_period_days(tg):8.3f}d "
              f"mtn {mean_total_notional(tg):.3f} "
              f"chg>1% {ms['membership_changes_gt1pct_per_day']:.4f}/d")
    mono = all(resp[i] >= resp[i + 1] for i in range(len(resp) - 1))
    print(f"  turnover non-increasing as the multiplier rises: {mono}")
    rows.append({"check": "turnover_monotone_in_multiplier", "value": mono,
                 "multipliers": ";".join(f"{m:g}" for m in NEIGHBOURHOOD),
                 "turnovers": ";".join(f"{v:.6f}" for v in resp)})

    write_csv(OUT_DIR / "novel_checks.csv", rows)
    return rows


def cmd_frontier(frames):
    """The derived points and the CONTEXT NEIGHBOURHOOD, on W_TRAIN and W_VAL.

    NOTHING IS SELECTED HERE. The derived point is the configuration; the
    neighbourhood exists only so a plateau-vs-peak statement can be made.
    """
    print("== FRONTIER (NO SELECTION -- the derived point IS the config) ==")
    rows = []
    for wname, window in (("W_TRAIN", W_TRAIN), ("W_VAL", W_VAL)):
        print(f"  -- {wname} --")
        warm = _warm_frames(frames, UNIVERSE_8, window)
        idx = _slice_index(warm, window)

        aligned, tg, ok = build_cell(frames, UNIVERSE_8, window, None, r65=True)
        r = evaluate(tg, aligned, UNIVERSE_8, wname, "U8",
                     {"delta": 0.0, "kind": "R65_WINNER"}, "R65_WINNER")
        r["kind"] = "reference"
        r["first_bar_warm"] = ok
        rows.append(r)
        print("  R65_WINNER (reference, delta == 0)")
        print(fmt_front(r))

        specs = [("D_A", 1.0, "derived D-A")]
        specs += [("D_B", m, ("DERIVED POINT (D-B)" if m == 1.0
                              else f"context {m:g}x")) for m in NEIGHBOURHOOD]

        for kind, mlt, label in specs:
            aligned, tg, ok = build_cell(frames, UNIVERSE_8, window, kind, mlt)
            st = delta_stats(warm, idx, kind, mlt)
            r = evaluate(tg, aligned, UNIVERSE_8, wname, "U8",
                         {"kind": kind, "mult": mlt,
                          "delta_mean": st["delta_mean"]},
                         spec_name(kind, mlt))
            r["kind"] = ("derived" if mlt == 1.0 else "context")
            r["label"] = label
            r["first_bar_warm"] = ok
            r.update(st)
            rows.append(r)
            print(f"  {label}  (mean delta {st['delta_mean']:.5f}, "
                  f"cap binds {st['delta_cap_frac']:.1%})")
            print(fmt_front(r))
            print(f"      volmatch net={r['net_volmatch_matched']} "
                  f"(c={r['net_volmatch_c']:.3f}) gross="
                  f"{r['gross_volmatch_matched']} | chg>1% "
                  f"{r['membership_changes_gt1pct_per_day']:.4f}/d")

    write_csv(OUT_DIR / "novel_frontier.csv", rows)
    cmd_neighbourhood_report(rows)
    return rows


def cmd_neighbourhood_report(rows):
    """The derived point's HONEST rank in its own multiplier ladder."""
    print("== CONTEXT NEIGHBOURHOOD -- REPORTED, NOT SELECTED ON ==")
    for wname in ("W_TRAIN", "W_VAL"):
        cand = [r for r in rows if r["window"] == wname
                and r.get("p_kind") == "D_B"]
        cand.sort(key=lambda r: -r["net_growth_diff"])
        print(f"  {wname}, ordered by net growth vs VOLMATCH_HOLD @0.10%:")
        for i, r in enumerate(cand):
            mark = "   <-- DERIVED POINT" if r["p_mult"] == 1.0 else ""
            print(f"    {i+1}. {r['p_mult']:>5g}x (mean delta "
                  f"{r['delta_mean']:.5f})  net {r['net_growth_diff']:+8.4f}"
                  f"  gross {r['gross_growth_diff']:+8.4f}"
                  f"  dd {r['net_dd_diff']:+7.2f}"
                  f"  turn {r['turnover_per_day']:.4f}/d{mark}")
        rank = 1 + next(i for i, r in enumerate(cand) if r["p_mult"] == 1.0)
        print(f"    -> the derived point ranks {rank} of {len(cand)} on {wname}."
              f"  THIS IS NOT A SELECTION; the derived point is the arm "
              f"whatever its rank.")


def cmd_m1(frames):
    """M1' on W_TRAIN: BOTH the thresholded-membership and turnover reductions
    against R-65's frozen winner."""
    print("== M1' (mechanism gate), W_TRAIN, U8 ==")
    warm = _warm_frames(frames, UNIVERSE_8, W_TRAIN)
    idx = _slice_index(warm, W_TRAIN)
    base = r65_winner_targets(warm).loc[idx]
    base_turn = turnover_stats(base)["turnover_per_day"]

    rows = []
    specs = [("D_A", 1.0)] + [("D_B", m) for m in NEIGHBOURHOOD]
    for kind, mlt in specs:
        tg = build_derived_targets(warm, delta_series(warm, kind, mlt)).loc[idx]
        ts = turnover_stats(tg)
        m1 = m1_pass(tg, base, ts["turnover_per_day"], base_turn)
        st = delta_stats(warm, idx, kind, mlt)
        row = {"arm": ARM, "spec": spec_name(kind, mlt), "kind": kind,
               "mult": mlt, "window": "W_TRAIN", "universe": "U8",
               "delta_mean": st["delta_mean"], "delta_max": st["delta_max"],
               "is_derived_point": bool(kind == "D_B" and mlt == 1.0),
               "is_secondary_derivation": bool(kind == "D_A"),
               **m1,
               "hold_days": holding_period_days(tg),
               "mean_notional": mean_total_notional(tg),
               **membership_stats(tg)}
        rows.append(row)
        tag = ("  <-- DERIVED POINT" if row["is_derived_point"]
               else "  <-- D-A" if row["is_secondary_derivation"] else "")
        print(f"  {spec_name(kind, mlt):>10s}  mem {m1['cand_membership_per_day']:8.4f}/d"
              f" red {m1['membership_reduction']:+7.2%} pass={str(m1['membership_passed']):<5}"
              f" | turn {m1['cand_turnover_per_day']:8.4f}/d"
              f" red {m1['turnover_reduction']:+7.2%} pass={str(m1['turnover_passed']):<5}"
              f" | M1'={m1['passed']}{tag}")
    print(f"  BASELINE r65_winner: mem "
          f"{rows[0]['baseline_membership_per_day']:.4f}/d  turn {base_turn:.4f}/d"
          f"   (M1' bar: >= {M1_MIN_REDUCTION:.0%} on BOTH)")
    rows.append({"arm": "R65_WINNER_baseline", "spec": "R65_WINNER",
                 "window": "W_TRAIN", "universe": "U8",
                 "cand_membership_per_day":
                     membership_change_rate_thresholded(base),
                 "cand_turnover_per_day": base_turn,
                 "hold_days": holding_period_days(base),
                 "mean_notional": mean_total_notional(base),
                 **membership_stats(base)})
    write_csv(OUT_DIR / "novel_m1.csv", rows)
    sel = [r for r in rows if r.get("is_derived_point")][0]
    da = [r for r in rows if r.get("is_secondary_derivation")][0]
    return {"primary": sel, "D_A": da, "rows": rows}


def cmd_run(frames):
    """The D-cells, for the DERIVED point (D-B) and for D-A."""
    print("== DECISION CELLS at the DERIVED thresholds (no selection) ==")
    rows = []
    state = {}

    # substrate reproduction, through this branch's own path
    aligned6, rt, _ = build_cell(frames, UNIVERSE_6, W_FULL6, None, baseline=True)
    rts = turnover_stats(rt)
    rc = mean_total_notional(rt)
    r_cand = simulate_portfolio(rt, aligned6, SPOT_BASE)
    r_mh = simulate_portfolio(matched_hold_targets(rt.index, UNIVERSE_6, rc),
                              aligned6, SPOT_BASE)
    r_cmp = compare(r_cand, r_mh)
    print("  [R-63 REFERENCE REPRODUCTION, W_FULL6 U6 vs MATCHED_HOLD]")
    print(f"    turnover {rts['turnover_per_day']:.4f}/d (R-63 published "
          f"{R63_TURNOVER_PER_DAY})  net growth_diff {r_cmp['growth_diff']:+.4f}"
          f" (published -7.537)  cand_final {r_cmp['cand_final']:,.4f}"
          f" (published 1.4419)  mtn {rc:.4f} (published 0.5249)")
    rows.append({"arm": "R63_BASELINE_k1", "window": "W_FULL6", "universe": "U6",
                 "bench": "MATCHED_HOLD", "kind": "substrate reproduction",
                 "turnover_per_day": rts["turnover_per_day"],
                 "hold_days": holding_period_days(rt), "mean_notional": rc,
                 "net_growth_diff": r_cmp["growth_diff"],
                 "net_growth_lo": r_cmp["growth_lo"],
                 "net_growth_hi": r_cmp["growth_hi"],
                 "cand_final": r_cmp["cand_final"],
                 "bench_final": r_cmp["bench_final"],
                 "cand_dd": r_cmp["cand_dd"], "bench_dd": r_cmp["bench_dd"],
                 "n_days": r_cmp["n_days"]})

    for kind, label in (("D_B", "DERIVED POINT (D-B)"), ("D_A", "D-A cube-root")):
        print(f"\n  ===== {label} =====")
        warm6 = _warm_frames(frames, UNIVERSE_6, W_FULL6)
        idx6 = _slice_index(warm6, W_FULL6)
        aligned, targets, warm_ok = build_cell(frames, UNIVERSE_6, W_FULL6, kind)
        st6 = delta_stats(warm6, idx6, kind, 1.0)
        print(f"  W_FULL6 bars {len(targets):,}  {targets.index[0]} -> "
              f"{targets.index[-1]}   first bar warm: {warm_ok}")
        print(f"  delta on this cell: mean {st6['delta_mean']:.6f} "
              f"median {st6['delta_median']:.6f} max {st6['delta_max']:.6f}  "
              f"cap binds {st6['delta_cap_frac']:.1%}  zero {st6['delta_zero_frac']:.1%}")
        if not warm_ok:
            raise RuntimeError("W_FULL6 first evaluated bar not warm")

        d12 = evaluate(targets, aligned, UNIVERSE_6, "W_FULL6", "U6",
                       {"kind": kind, "mult": 1.0,
                        "delta_mean": st6["delta_mean"]}, spec_name(kind, 1.0))
        d12["kind_row"] = "D1/D2/D5 primary"
        d12["label"] = label
        d12["first_bar_warm"] = warm_ok
        d12.update(st6)
        print("  [D1/D2/D5] W_FULL6 U6 vs VOLMATCH_HOLD")
        print(fmt_front(d12))
        print(f"    volmatch matched: net={d12['net_volmatch_matched']} "
              f"(c={d12['net_volmatch_c']:.3f}, bench vol "
              f"{d12['net_volmatch_vol']:.3f} vs cand {d12['net_cand_vol']:.3f}) "
              f"| gross={d12['gross_volmatch_matched']} "
              f"(c={d12['gross_volmatch_c']:.3f})")

        if d12["net_volmatch_matched"]:
            d1, d2 = d1_pass(d12), d2_pass(d12)
        else:
            d1 = d2 = False
            print("    !! VOLMATCH did not match @0.10% -- D1/D2 VOIDED")
        d5 = d5_pass(d12) if d12["gross_volmatch_matched"] else False
        if not d12["gross_volmatch_matched"]:
            print("    !! VOLMATCH did not match @0bps -- D5 VOIDED")
        d12["d1_pass"], d12["d2_pass"], d12["d5_pass"] = d1, d2, d5
        rows.append(d12)
        print(f"    D1 PASS={d1}  D2 PASS={d2}  D5 PASS={d5} "
              f"(gross {d12['gross_growth_diff']:+.3f} vs bar {D5_BAR_R68:+.3f})")

        # context benchmarks
        cand = simulate_portfolio(targets, aligned, SPOT_BASE)
        c = mean_total_notional(targets)
        mh = simulate_portfolio(matched_hold_targets(targets.index, UNIVERSE_6, c),
                                aligned, SPOT_BASE)
        ew = static_hold_equity(aligned, UNIVERSE_6, SPOT_BASE)
        btc = frames["BTC"]
        btc_on = (btc.reindex(btc.index.union(targets.index)).ffill()
                  .reindex(targets.index))
        btc_eq = static_hold_equity({"BTC": btc_on}, ["BTC"], SPOT_BASE)
        for blabel, bench in (("MATCHED_HOLD", mh), ("EW_HOLD", ew),
                              ("BTC_HOLD", btc_eq)):
            cm = compare(cand, bench)
            rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                         "bench": blabel, "kind_row": "context", "label": label,
                         "p_kind": kind, "mean_notional": c,
                         "hold_days": holding_period_days(targets),
                         "turnover_per_day":
                             turnover_stats(targets)["turnover_per_day"],
                         "net_growth_diff": cm["growth_diff"],
                         "net_growth_lo": cm["growth_lo"],
                         "net_growth_hi": cm["growth_hi"],
                         "net_dd_diff": cm["dd_diff"], "net_dd_lo": cm["dd_lo"],
                         "net_dd_hi": cm["dd_hi"], "cand_final": cm["cand_final"],
                         "bench_final": cm["bench_final"],
                         "cand_dd": cm["cand_dd"], "bench_dd": cm["bench_dd"],
                         "n_days": cm["n_days"]})
            print(f"  [context vs {blabel}] cand {cm['cand_final']:,.1f} vs "
                  f"{cm['bench_final']:,.1f} | growth {cm['growth_diff']:+.3f}"
                  f" [{cm['growth_lo']:+.3f},{cm['growth_hi']:+.3f}] | dd "
                  f"{cm['cand_dd']:.1f}% vs {cm['bench_dd']:.1f}% "
                  f"({cm['dd_diff']:+.2f})")

        # D4: W_FULL6 @0.40% vs EW_HOLD
        cand40 = simulate_portfolio(targets, aligned, SPOT_REAL)
        ew40 = static_hold_equity(aligned, UNIVERSE_6, SPOT_REAL)
        d4 = bool(cand40.iloc[-1] > ew40.iloc[-1])
        rows.append({"arm": ARM, "window": "W_FULL6", "universe": "U6",
                     "bench": "EW_HOLD", "kind_row": "D4 @0.40%", "label": label,
                     "p_kind": kind, "cand_final": float(cand40.iloc[-1]),
                     "bench_final": float(ew40.iloc[-1]), "d4_pass": d4})
        print(f"  [D4 @0.40%] cand {cand40.iloc[-1]:,.1f} vs EW_HOLD "
              f"{ew40.iloc[-1]:,.1f} -> D4 PASS={d4}")

        # D3: W_VAL, U8
        aligned3, targets3, warm3 = build_cell(frames, UNIVERSE_8, W_VAL, kind)
        if not warm3:
            raise RuntimeError("W_VAL first evaluated bar not warm")
        warm8 = _warm_frames(frames, UNIVERSE_8, W_VAL)
        st3 = delta_stats(warm8, _slice_index(warm8, W_VAL), kind, 1.0)
        d3row = evaluate(targets3, aligned3, UNIVERSE_8, "W_VAL", "U8",
                         {"kind": kind, "mult": 1.0,
                          "delta_mean": st3["delta_mean"]}, spec_name(kind, 1.0))
        d3row["kind_row"] = "D3 inner-validation"
        d3row["label"] = label
        d3row["first_bar_warm"] = warm3
        d3row.update(st3)
        print("  [D3] W_VAL U8 vs VOLMATCH_HOLD")
        print(fmt_front(d3row))
        d3 = d3_pass(d3row) if d3row["net_volmatch_matched"] else False
        if not d3row["net_volmatch_matched"]:
            print("    !! VOLMATCH did not match on W_VAL -- D3 VOIDED")
        d3row["d3_pass"] = d3
        rows.append(d3row)
        print(f"    D3 PASS={d3}  (matched={d3row['net_volmatch_matched']})")

        state[kind] = {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5,
                       "row": d12, "targets": targets, "aligned": aligned,
                       "cand": cand, "label": label}

    write_csv(OUT_DIR / "novel_cells.csv", rows)
    return state


def cmd_scramble(frames, state=None, kind="D_B"):
    """The round's fixed-permutation control, on the derived point's D1 cell."""
    print(f"== FALSIFICATION: fixed-permutation scramble, {kind} D1 cell ==")
    if state is None or kind not in state:
        aligned, targets, _ = build_cell(frames, UNIVERSE_6, W_FULL6, kind)
    else:
        aligned, targets = state[kind]["aligned"], state[kind]["targets"]
    cand = simulate_portfolio(targets, aligned, SPOT_BASE)
    bench, c, vol, matched = volmatched_hold_equity(cand, aligned, UNIVERSE_6,
                                                    SPOT_BASE)
    real = compare(cand, bench)["growth_diff"]
    cand_turn = turnover_stats(targets)["turnover_per_day"]
    print(f"  candidate real growth_diff {real:+.4f}  turnover {cand_turn:.4f}/d"
          f"  (volmatch matched={matched}, c={c:.3f})")

    rows, diffs = [], []
    for seed in SCRAMBLE_SEEDS:
        st = scramble_fixed_perm(targets, seed)
        eq = simulate_portfolio(st, aligned, SPOT_BASE)
        r = compare(eq, bench)
        diffs.append(r["growth_diff"])
        st_turn = turnover_stats(st)["turnover_per_day"]
        ident = bool(np.allclose(st.to_numpy(), targets.to_numpy()))
        rows.append({"arm": f"{ARM}_fixedperm", "seed": seed, "p_kind": kind,
                     "window": "W_FULL6", "universe": "U6", "fee": 0.001,
                     "bench": "VOLMATCH_HOLD", "identity_perm": ident,
                     "mean_notional": mean_total_notional(st),
                     "turnover_per_day": st_turn,
                     **{k: r[k] for k in ("cand_final", "bench_final", "cand_dd",
                                          "bench_dd", "growth_diff", "growth_lo",
                                          "growth_hi", "dd_diff", "dd_lo",
                                          "dd_hi", "n_days")}})
        print(f"    seed {seed}: growth_diff {r['growth_diff']:+.4f}  "
              f"turnover {st_turn:.4f}/d{'  (IDENTITY)' if ident else ''}")
    p90 = float(np.percentile(diffs, 90))
    surv = bool(real > p90)
    n_beat = int(sum(1 for d in diffs if real > d))
    print(f"  real {real:+.4f} vs fixed-perm p90 {p90:+.4f} -> "
          f"SURVIVED={surv}  (beats {n_beat}/{len(diffs)})")
    rows.append({"arm": ARM, "seed": -1, "p_kind": kind, "window": "W_FULL6",
                 "universe": "U6", "fee": 0.001, "bench": "VOLMATCH_HOLD",
                 "growth_diff": real, "scramble_p90": p90,
                 "scramble_survived": surv, "turnover_per_day": cand_turn,
                 "n_beaten": n_beat,
                 "control": "fixed-permutation (r68_shared.scramble_fixed_perm)"})
    write_csv(OUT_DIR / "novel_scramble.csv", rows)
    return {"survived": surv, "real": real, "p90": p90}


def cmd_trials(state):
    """The trials accounting, and the adversarial reading of it."""
    print("== TRIALS ACCOUNTING ==")
    rows = []
    n_backtests = config_count()
    specs_all = sorted(_SPECS)
    # what a reader would call a "trial": one threshold SPECIFICATION whose
    # decision statistic was read, per window.
    n_derived_points = 2      # D-A and D-B, both fixed by formula
    n_context = len(NEIGHBOURHOOD) - 1   # 1.0x IS the derived point
    n_identity = 2            # const:0.0 and const:0.080 -- identities, not searches
    swept_arm = 6             # R-67's delta grid
    swept_ext = 11            # r68_shared.DELTA_GRID_EXT, this round's other branch

    print(f"  simulate/static-hold calls in this process: {n_backtests}")
    print(f"  distinct threshold specifications x window backtested: "
          f"{len(specs_all)}")
    print(f"    {', '.join(specs_all)}")
    print(f"  SEARCH trials attributable to this arm: {n_derived_points} "
          f"(two formulae, neither fitted; the multiplier ladder is reported, "
          f"not selected on, and the two constants are identity checks)")
    print(f"  a SWEPT arm on the same axis would carry {swept_arm} (R-67) or "
          f"{swept_ext} (this round's extended grid) selection trials.")

    row = {"n_simulate_calls": n_backtests,
           "n_spec_window_backtests": len(specs_all),
           "specs": ";".join(specs_all),
           "n_search_trials_claimed": n_derived_points,
           "n_context_cells_reported_not_selected": n_context,
           "n_identity_cells": n_identity,
           "swept_arm_trials_r67": swept_arm,
           "swept_arm_trials_r68_ext": swept_ext}

    prim = state.get("D_B")
    if prim is not None:
        eq = prim["cand"]
        r = daily_returns(eq).to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        sr = float(annualized_sharpe(r))
        sk, ku = moments(r)
        row.update(sharpe_ann=sr, n_obs=len(r), skew=sk, kurtosis=ku)
        print(f"  primary D1 cell (W_FULL6 @0.10%): Sharpe {sr:+.4f} on "
              f"{len(r)} daily obs (skew {sk:+.3f}, kurt {ku:.3f})")
        psr = probabilistic_sharpe_ratio(sr, len(r), sk, ku)
        row["psr_vs_zero"] = psr
        print(f"    PSR vs 0: {psr:.4f}")
        for n_tr, tag in ((n_derived_points, "this arm, as claimed"),
                          (len(specs_all), "this arm, every cell it read"),
                          (swept_arm, "a swept arm (R-67's grid)"),
                          (swept_ext, "a swept arm (extended grid)")):
            for sd_tr in (0.25, 0.5):
                dsr = deflated_sharpe_ratio(sr, len(r), sk, ku, n_tr, sd_tr)
                srstar = expected_max_sharpe(n_tr, sd_tr)
                print(f"    n_trials={n_tr:<3d} sd_trials={sd_tr}: SR* "
                      f"{srstar:+.4f}  DSR {dsr:.4f}   [{tag}]")
                row[f"dsr_n{n_tr}_sd{sd_tr}"] = dsr
                row[f"srstar_n{n_tr}_sd{sd_tr}"] = srstar
        be = deflation_breakeven_sd(sr, len(r), sk, ku, n_derived_points)
        row["deflation_breakeven_sd_at_claimed_trials"] = be
        print(f"    deflation breakeven sd_trials at n_trials="
              f"{n_derived_points}: {be:.4f}")
        mtrl = min_track_record_length(sr, sk, ku)
        row["min_track_record_days"] = mtrl
        print(f"    minimum track record length vs 0: {mtrl:,.0f} daily obs")

    print("  ADVERSARIAL READING, stated by this branch against itself: the "
          "trials count above is NOT 0 and is NOT honestly 2. The FORM of the "
          "rule -- a symmetric band on this exact eligibility test -- was "
          "chosen after R-64, R-66 and R-67 had each reported on this axis, "
          "and the cap, the substrate and the gates were all fixed by rounds "
          "that saw results. A formula selected after three rounds of "
          "results on the same axis carries the selection of those rounds, "
          "which is a program-level count in the dozens, not a branch-level "
          "count of two.")
    row["adversarial_note"] = ("form of the rule chosen after 3 prior rounds on "
                               "this axis; branch-level count of 2 understates "
                               "the program-level count")
    rows.append(row)
    write_csv(OUT_DIR / "novel_configcount.csv", rows)
    return row


# ======================================================================
# main
# ======================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["derive", "identity", "checks", "frontier",
                                    "m1", "run", "scramble", "trials", "all"])
    args = ap.parse_args()

    frames = load_universe(UNIVERSE_8)

    if args.cmd == "derive":
        cmd_derive(frames)
    elif args.cmd == "identity":
        cmd_identity(frames)
    elif args.cmd == "checks":
        cmd_checks(frames, cmd_identity(frames))
    elif args.cmd == "frontier":
        cmd_frontier(frames)
    elif args.cmd == "m1":
        cmd_m1(frames)
    elif args.cmd == "run":
        cmd_run(frames)
    elif args.cmd == "scramble":
        cmd_scramble(frames)
    elif args.cmd == "trials":
        cmd_trials({})
    else:
        ident = cmd_identity(frames)
        cmd_derive(frames)
        cmd_checks(frames, ident)
        cmd_frontier(frames)
        m1 = cmd_m1(frames)
        st = cmd_run(frames)
        sc = cmd_scramble(frames, st, "D_B")
        prim = st["D_B"]
        fw = further_work(bool(m1["primary"]["passed"]), prim["d1"], prim["d2"],
                          prim["d3"], prim["d5"], sc["survived"])
        print(f"\n== DERIVED POINT (D-B): further_work("
              f"m1'={m1['primary']['passed']}, d1={prim['d1']}, "
              f"d2={prim['d2']}, d3={prim['d3']}, d5={prim['d5']}, "
              f"scramble={sc['survived']}) = {fw} ==")
        da = st["D_A"]
        fw_a = further_work(bool(m1["D_A"]["passed"]), da["d1"], da["d2"],
                            da["d3"], da["d5"], sc["survived"])
        print(f"== D-A (secondary derivation): further_work("
              f"m1'={m1['D_A']['passed']}, d1={da['d1']}, d2={da['d2']}, "
              f"d3={da['d3']}, d5={da['d5']}, scramble=<D-B's control>) "
              f"= {fw_a} ==")
        print("  -> the reserved 2023+ holdout is NOT read by this branch "
              "under any outcome.")
        cmd_trials(st)

    print(f"\nconfig_count() = {config_count()}")


if __name__ == "__main__":
    main()
