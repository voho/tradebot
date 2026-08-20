"""R-68: attack B-34 -- the band's SHAPE, and the grid edges R-67 leaned on.

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it (and from `r63_shared.py` / `r65_shared.py` / `r67_shared.py`,
which it does not duplicate), NEITHER BRANCH EDITS IT, and it does not itself
define a candidate strategy or compute a verdict.

Committed BEFORE either branch was written, together with this round's
prices-only pre-measurement (`experiments/r68_stopping_premium.py`, backlog
item **B-35**), whose numbers are quoted below as the round's own prior and
are the reason two of the failure modes are named the way they are.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **COST**. Backlog item **B-34**, filed by R-67 and top
of the ranked, actionable list (LEDGER.md D, "NEXT").

R-67 broke R-65's forced-exit floor 30-fold with one new parameter and still
failed on `(D1 or D2)`. It left two things unfinished, and B-34 is both:

1. **Both winners sit against a grid edge.** The conservative winner is
   delta=0.080, the top corner of a grid stopping there, with only a lower
   neighbour tested. The novel winner is `a=0.02`, second-slowest on a grid
   stopping at 0.01 and still improving on W_TRAIN. Neither branch extended
   its grid after seeing a number -- correctly, since that would have moved
   the goalposts -- so whether either curve turns over past its edge is
   simply **untested**. R-67 filed this as its own live failure mode (F3).
2. **The band and the asymmetry were changed together.** R-67's rule moves
   the entry threshold up to `+delta` and the exit threshold down to
   `-delta` in one step, so it cannot say which half did the work. That is
   the identical confound R-64 and R-66 each died of on the single-asset
   axis, and R-67's own commissioned literature says the asymmetric ordering
   is licensed only conditionally while the band itself is not.

=====================================================================
WHY THIS IS NOT A DUPLICATE
=====================================================================

- **R-67 conservative (gate_hysteresis).** Sweeps ONE coupled parameter with
  `delta_in = delta_out = delta` over `delta <= 0.080`. This round's
  conservative branch **decomposes that single parameter into two** and
  sweeps each alone, over a grid extended to a theory-derived cap. R-67's
  arm is the constrained diagonal of this round's plane and is re-run here
  as a named reference cell, not re-selected.
- **R-67 novel (smoothed_score).** An EWMA partial adjustment on a
  continuous target, no threshold anywhere. This round's novel branch keeps
  the discrete band and removes the *grid* instead, deriving the threshold
  from measurable quantities with no fitted parameter at all.
- **R-65 (rank buffer / aim portfolio).** Retunes `buffer` and `hold_days`
  with the eligibility test frozen at `s > 0`. Both are frozen here at
  R-65's own selected winner (0.05, 1), as in R-67.
- **L-05 / L-06 (`kelly_regime_ev`, `kelly_regime_ev_fast`).** A no-trade
  band on a *continuous single-asset exposure fraction*. Both branches here
  band a *discrete cross-sectional membership* decision.
- **R-64 / R-66 (B-29).** A single-asset SIZE-axis deadband's *destination*
  on `kelly_regime_v4`, a different signal in a different codepath. Used
  below as the named precedent for the confound this round exists to break.

=====================================================================
THE TWO BRANCHES
=====================================================================

Both start from R-65's selected winner byte-for-byte where not explicitly
varied: `k=1`, `buffer=0.05`, `hold_days=1`, R-63's composite cross-sectional
score and conditional volatility scale unmodified, R-63's 0.10 deadband on
desired total notional, equal weighting among held slots, long-only spot,
unlevered. Only the eligibility thresholds change.

**CONSERVATIVE -- decompose the band, and extend the grid to its
theoretical cap** (`experiments/r68_conservative_band_decomposition.py`).
R-67's single `delta` becomes two:

    enter_eligible  =  s >  +delta_in      (a new entrant must clear this)
    hold_eligible   =  s >  -delta_out     (an incumbent is kept above this)

Three named sub-arms on one extended grid, all reported for every cell
whatever the verdict:

    EXIT-ONLY   delta_in = 0, delta_out = d   hold through the crossing,
                                              enter exactly as R-63 does
    ENTRY-ONLY  delta_in = d, delta_out = 0   demand a stronger entrant,
                                              exit exactly as R-63 does
    COUPLED     delta_in = delta_out = d      R-67's own arm, the diagonal

`d=0` is all three at once and must reproduce R-65's frozen winner exactly;
`COUPLED, d=0.080` must reproduce R-67's published cell exactly. Both
identities are required before any other number is reported.

**NOVEL -- remove the grid: a threshold derived, not fitted**
(`experiments/r68_novel_derived_threshold.py`). R-65's and R-67's single
strongest pieces of evidence were both *derived* rates that landed inside a
window measured independently of them (`a_GP`, reproduced to 8 significant
figures across two different aim constructions). This branch asks the same
question of the band: does a threshold computed from the cost rate and the
score's own scale -- with **zero fitted parameters** -- land where the
conservative sweep selects? The formula, its citation and its regime of
validity are frozen in that branch's docstring before its first number, and
the branch reports the derived point WHETHER OR NOT it wins.

**CAUSALITY, stated here because it is this round's specific lookahead
risk.** The conservative branch's thresholds are constants and cannot leak.
The novel branch's threshold is a function of measured score scale, and a
whole-series standard deviation applied to early rows is exactly the
full-series fit ROUTINE's skeptic rule hunts for. It is therefore a
REQUIREMENT of this pre-registration, not a preference, that any data-derived
quantity in the novel branch be computed on an EXPANDING or ROLLING window
and SHIFTED by at least one bar, and that the branch demonstrate this with a
truncation probe before reporting anything.

=====================================================================
THE ROUND'S OWN PRIOR -- FROM B-35, MEASURED BEFORE EITHER BRANCH
=====================================================================

`experiments/r68_stopping_premium.py` was run and committed first. On
W_TRAIN+W_VAL, prices only:

- The incumbent's forward return after a downward zero crossing is
  **+0.00504 at H=1 day** (unconditional +0.00048), and the mean grace span
  the band actually buys is **0.17 days at delta=0.080**. At the horizon the
  mechanism acts on, the forced exit therefore has a **negative stopping
  premium**: it sells something that on average goes UP next.
- Return autocorrelation at 1 / 5 / 14 days is **-0.078 / +0.059 / -0.061**
  -- a random walk to two decimal places, which is Kaminski & Lo's (2014)
  own condition for a negative premium.
- Priced in log units on the same window, delta=0.080's grace periods are
  worth **+0.311** while the round trips they avoid save **+1.044**.
- **The fee saving asymptotes**: +1.044 at delta=0.080, +1.062 at 0.120,
  +1.074 at 0.160, while the grace span keeps growing (0.17 -> 0.29 ->
  0.39 days).

That last line is a **prediction about this round**, recorded before either
branch exists: the extended grid should FLATTEN rather than turn over, the
marginal value of widening past 0.080 should be small, and a materially
better cell at the new top corner is more likely a second grid-edge artifact
than a discovery. See (F1).

Every interval in that pre-measurement contains zero. It constrains the
round's interpretation; it establishes nothing on its own.

=====================================================================
THE GRID CAP -- DERIVED, NOT CHOSEN
=====================================================================

de Lataillade, J., & Chaouki, A. (2020), "Equations and Shape of the Optimal
Band Strategy," arXiv:2003.04646, Eq. (11): the optimal tolerance around
zero **saturates at approximately 1.6 sigma_signal**, so a larger fee does
not justify an arbitrarily wider band. Measured on W_TRAIN, the R-63
composite score has pooled sigma = **0.2295** (mean per-asset 0.2187), so the
cap is **1.6 x 0.2295 = 0.367**.

R-67's winner, delta=0.080, is only **0.35 sigma** -- well inside the cap,
which is the specific reason extending the grid is defensible rather than
fishing. `DELTA_GRID_EXT` below stops at 0.350, just under the cap, and it is
frozen here before either branch runs.

The cap is computed from a whole-window standard deviation and is used ONLY
to bound a grid of constants. It never enters a per-bar decision, so it
cannot leak; a branch that uses sigma inside its rule must derive it causally
(see CAUSALITY above).

=====================================================================
GATES, FROZEN
=====================================================================

D1/D2/D3/D5, the bootstrap, the benchmarks (`MATCHED_HOLD`, `VOLMATCH_HOLD`,
`EW_HOLD`), the windows and the cost tiers are inherited BY IMPORT from
`r65_shared.py` / `r63_shared.py`, unchanged, so this round cannot drift from
the numbers it extends. D4 (the 0.40% tier vs `EW_HOLD`) is inherited from
R-67. This round changes exactly three things, all of them corrections
R-67 disclosed against itself, and all three make the bar HARDER:

**(1) D5 uses R-65's corrected bar.** R-67 disclosed that it inherited
`D5_BAR = +0.240`, the value R-65 had already retired in favour of
`D5_BAR_CORRECTED = +0.342` (the like-for-like number against VOLMATCH_HOLD)
with the note "use this next". This round uses **+0.342**. Both R-67 arms
cleared both bars by 4-8x, so no prior verdict moves.

**(2) M1 is replaced by M1', on a thresholded statistic.** R-67 disclosed
that its M1 counts `weight > 0` and is therefore structurally invalid for a
continuous-weight arm -- an unfloored recursion that never fully releases an
asset scored a 98.53% "reduction" identically at `a=0.5` (which turns over
4.8x the baseline) and at `a=0.01` (half of it). It also disclosed a latent
bug: `membership_change_rate` round-trips through `holding_period_days` and
returns `1/days` rather than `0` for an arm that never changes its held set.
:func:`membership_change_rate_thresholded` below counts changes DIRECTLY, on
a held-set indicator thresholded at **1% of equity**, and M1' requires the
reduction on **BOTH** that statistic **AND** raw turnover, so a branch cannot
pass on the flattering one alone.

**(3) The fixed-permutation scramble control is exported here.** R-67
disclosed that its shared file named the control in the bar but exported only
`scramble_targets`, so each branch re-implemented it -- an invitation to two
incomparable controls in one round. :func:`scramble_fixed_perm` below is the
single implementation both branches must use.

M1' threshold, unchanged in spirit from R-67: **>= 25% fewer thresholded
membership changes per day AND >= 25% lower turnover** than R-65's frozen
winner on W_TRAIN.

FURTHER-WORK BAR (R-67's, with M1'):
    M1' AND (D1 or D2) AND D3 AND D5 AND scramble_survived
Clearing it authorizes exactly ONE holdout read on `W_HOLD`; it is NOT the
promotion bar. Nothing here is registrable regardless of verdict -- both
branches are bar-by-bar cross-asset allocators and **B-32** (multi-asset
registration) is still OPEN. Recorded so a positive result is not
overclaimed.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- NAMED NOW, BEFORE ANY CODE
=====================================================================

**(F1) The extension finds another corner.** B-35's fee-saving curve is
already asymptoting, so the honest expectation is a FLAT extended frontier.
If a branch's winner again lands on the new top corner, the round reports it
as a corner and does not extend a second time; two corners in a row is
evidence about the selection criterion, not about the mechanism.

**(F2) The far end is a concentrated buy-and-hold** (R-67's F3, inherited
verbatim). As `d -> large` both eligibility tests stop firing and the arm
converges on "hold whatever was strongest in 2020-04". D5, at its corrected
+0.342 bar, is what catches this, and at the cap it SHOULD catch it.

**(F3) Both halves may be inert.** The decomposition can show that neither
`delta_in` alone nor `delta_out` alone reproduces R-67's coupled result --
i.e. that the effect requires both and the round learns only that the
confound cannot be separated on this data. That is a real outcome and it is
an acceptable answer; it closes B-34's second half rather than the axis.

**(F4) Theory and data may disagree.** The derived threshold may land far
outside the region the conservative sweep prefers. R-65 and R-67 both had
the derived rate land INSIDE an independently measured window, which is why
that evidence was strong; a miss is equally informative and is reported as a
miss, not quietly dropped.

**(F5) And the one this round expects, stated plainly.** R-67's own
one-line lesson is that three rounds have improved this signal's economics
by 10-80x and every one died on the same interval, "which is no longer a
fact about any mechanism but a fact about how much this dataset can
resolve." **The base case for this round is that `(D1 or D2)` fails again.**
It is run anyway because B-34's two questions -- does the curve turn over,
and which half of the band works -- are answerable independently of the
interval, and because leaving a named confound unresolved is how R-64 and
R-66 both lost a round. A round that answers both and still fails
`(D1 or D2)` is a successful round by this repo's own standard, and
"nothing" is an explicitly acceptable answer here.

Configurations evaluated are counted by `r63_shared.config_count()`, shared
process-wide; each branch reports its own count and the round's total is the
sum, per ROUTINE's parallelism rules.
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
    check_causality,
    compare,
    config_count,
    excludes_zero,
    load_universe,
    matched_hold_targets,
    mean_total_notional,
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
    OUT_DIR as R65_OUT_DIR,
    R63_GROSS_EDGE,
    R63_GROSS_EDGE_VS_VOLMATCH,
    R63_NET_D1,
    R63_TURNOVER_PER_DAY,
    SPOT_FREE,
    d1_pass,
    d2_pass,
    d3_pass,
    frontier_row,
    holding_period_days,
    realized_vol,
    turnover_stats,
    volmatched_hold_equity,
)
from experiments.r67_shared import (  # noqa: E402,F401
    R65_BUFFER,
    R65_HOLD_DAYS,
    R65_K,
    r65_winner_targets,
)

OUT_DIR = ROOT / "reports" / "r68_band"

# ---------------------------------------------------------------------------
# Correction (1): D5's bar. R-65 measured the like-for-like gross edge against
# VOLMATCH_HOLD at +0.683 and recorded `D5_BAR_CORRECTED = +0.342` with the
# note "use this next"; R-67 inherited the retired +0.240 instead and
# disclosed it. This round uses the corrected, HARDER bar.
D5_BAR_R68 = D5_BAR_CORRECTED  # +0.342

# The dLC (2020) Eq. (11) cap, and the grid it licenses. sigma_score is the
# pooled standard deviation of the R-63 composite score over W_TRAIN, measured
# once, before either branch existed, and used ONLY to bound a grid of
# constants -- never inside a per-bar decision.
SIGMA_SCORE_W_TRAIN = 0.2295
DLC_SATURATION = 1.6 * SIGMA_SCORE_W_TRAIN  # 0.3672

# R-67's grid, extended to just under the cap. Frozen here, before any run.
DELTA_GRID_R67 = (0.000, 0.005, 0.010, 0.020, 0.040, 0.080)
DELTA_GRID_EXT = DELTA_GRID_R67 + (0.120, 0.160, 0.220, 0.280, 0.350)

# R-67's published reference cells, cited for identity checks rather than
# re-derived. Source: LEDGER.md R-67 and reports/r67_gate/.
R67_DELTA_WINNER = 0.080
R67_CONSERVATIVE_TURNOVER_PER_DAY = 0.174   # W_TRAIN, delta=0.080
R67_CONSERVATIVE_CHANGES_PER_DAY = 0.142    # W_TRAIN, delta=0.080
R65_BREAKEVEN_TURNOVER_PER_DAY = 0.641
R65_ACHIEVED_TURNOVER_PER_DAY = 0.900

# B-35's measured prior, cited so a branch does not re-measure it and so the
# round's own prediction is quotable from code.
B35_FEE_SAVING_LOG = {0.080: 1.0440, 0.120: 1.0620, 0.160: 1.0740}
B35_GRACE_SPAN_DAYS = {0.080: 0.17, 0.120: 0.29, 0.160: 0.39}

M1_MIN_REDUCTION = 0.25
MEMBERSHIP_WEIGHT_FLOOR = 0.01  # 1% of equity -- correction (2)


# ------------------------------------------------------------------ M1'


def held_indicator(targets: pd.DataFrame,
                   floor: float = MEMBERSHIP_WEIGHT_FLOOR) -> np.ndarray:
    """Thresholded held-set indicator: `weight > floor`, not `weight > 0`.

    Correction (2). R-67's M1 counted any strictly positive weight, which a
    continuous-weight arm satisfies on essentially every bar for essentially
    every asset, making the statistic report a large "reduction" for a
    configuration that turns over five times the baseline. A 1%-of-equity
    floor is the smallest position this project would call a holding.
    """
    return np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0) > float(floor)


def membership_change_rate_thresholded(
        targets: pd.DataFrame,
        floor: float = MEMBERSHIP_WEIGHT_FLOOR) -> float:
    """Changes to the thresholded held set, per day, counted DIRECTLY.

    Correction (2), second half. R-67's `membership_change_rate` obtained the
    count by inverting `holding_period_days`, which returns `1/days` instead
    of `0` for an arm whose held set never changes -- a latent bug it
    disclosed but did not trigger. This counts the changes and divides by the
    span, so an arm that never changes scores exactly 0.0.
    """
    ind = held_indicator(targets, floor)
    if len(ind) < 2:
        return 0.0
    changes = int(np.any(ind[1:] != ind[:-1], axis=1).sum())
    days = max(len(targets) / BARS_PER_DAY, 1e-9)
    return changes / days


def m1_pass(cand_targets: pd.DataFrame, baseline_targets: pd.DataFrame,
            cand_turnover: float, baseline_turnover: float) -> dict:
    """M1': the reduction must hold on the thresholded membership statistic
    AND on raw turnover. Both rates are returned so both are reported even
    when the gate fails."""
    cand_rate = membership_change_rate_thresholded(cand_targets)
    base_rate = membership_change_rate_thresholded(baseline_targets)
    mem_red = 1.0 - (cand_rate / base_rate) if base_rate > 0 else float("nan")
    turn_red = (1.0 - (cand_turnover / baseline_turnover)
                if baseline_turnover > 0 else float("nan"))
    return {
        "cand_membership_per_day": cand_rate,
        "baseline_membership_per_day": base_rate,
        "membership_reduction": mem_red,
        "cand_turnover_per_day": cand_turnover,
        "baseline_turnover_per_day": baseline_turnover,
        "turnover_reduction": turn_red,
        "membership_passed": bool(np.isfinite(mem_red)
                                  and mem_red >= M1_MIN_REDUCTION),
        "turnover_passed": bool(np.isfinite(turn_red)
                                and turn_red >= M1_MIN_REDUCTION),
        "passed": bool(np.isfinite(mem_red) and mem_red >= M1_MIN_REDUCTION
                       and np.isfinite(turn_red)
                       and turn_red >= M1_MIN_REDUCTION),
    }


# ------------------------------------------------------------------ D5


def d5_pass(row: dict) -> bool:
    """Signal retention against the CORRECTED bar (correction 1)."""
    return row["gross_growth_diff"] >= D5_BAR_R68


# ------------------------------------------------------------------ control


def scramble_fixed_perm(targets: pd.DataFrame, seed: int) -> pd.DataFrame:
    """ONE fixed column permutation for the whole series (correction 3).

    R-65's own correction to R-63's redraw-per-change control, exported here
    so both branches use the identical implementation rather than two
    incomparable re-implementations. The candidate's total notional path,
    turnover and holding periods are preserved EXACTLY -- only the
    asset->signal assignment is destroyed -- so a candidate that beats this
    control is using cross-sectional information rather than exposure.

    R-63's redraw-per-change form is deliberately NOT offered: R-65 and R-67
    both measured it as a 739x turnover over-charge for a smoothed arm, a
    bias running in the candidate's favour.
    """
    rng = np.random.default_rng(seed)
    w = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0)
    perm = rng.permutation(w.shape[1])
    return pd.DataFrame(w[:, perm], index=targets.index, columns=targets.columns)


def further_work(m1: bool, d1: bool, d2: bool, d3: bool, d5: bool,
                 scramble_survived: bool) -> bool:
    """R-67's bar with M1'. NOT the promotion bar; clearing it authorizes
    exactly one holdout read on W_HOLD."""
    return m1 and (d1 or d2) and d3 and d5 and scramble_survived


__all__ = [
    "B35_FEE_SAVING_LOG", "B35_GRACE_SPAN_DAYS", "BARS_PER_DAY",
    "BARS_PER_YEAR", "BOOT_KW", "D5_BAR_R68", "DEADBAND", "DELTA_GRID_EXT",
    "DELTA_GRID_R67", "DLC_SATURATION", "HORIZONS", "M1_MIN_REDUCTION",
    "MEMBERSHIP_WEIGHT_FLOOR", "OUT_DIR", "R63_GROSS_EDGE",
    "R63_GROSS_EDGE_VS_VOLMATCH", "R63_NET_D1", "R63_TURNOVER_PER_DAY",
    "R65_ACHIEVED_TURNOVER_PER_DAY", "R65_BREAKEVEN_TURNOVER_PER_DAY",
    "R65_BUFFER", "R65_HOLD_DAYS", "R65_K", "R65_OUT_DIR",
    "R67_CONSERVATIVE_CHANGES_PER_DAY", "R67_CONSERVATIVE_TURNOVER_PER_DAY",
    "R67_DELTA_WINNER", "SCRAMBLE_SEEDS", "SIGMA_SCORE_W_TRAIN", "SPOT_BASE",
    "SPOT_FREE", "SPOT_REAL", "START_BALANCE", "TOTAL_NOTIONAL_DEADBAND",
    "UNIVERSE_6", "UNIVERSE_8", "WARM_DAYS", "W_FULL6", "W_HOLD", "W_TRAIN",
    "W_VAL", "align_frames", "basket_log_returns", "check_causality",
    "compare", "conditional_vol_scale", "config_count",
    "cross_sectional_score", "d1_pass", "d2_pass", "d3_pass", "d5_pass",
    "excludes_zero", "frontier_row", "further_work", "held_indicator",
    "holding_period_days", "load_universe", "m1_pass", "matched_hold_targets",
    "mean_total_notional", "membership_change_rate_thresholded",
    "r63_baseline_targets", "r65_winner_targets", "realized_vol",
    "scramble_fixed_perm", "simulate_portfolio", "static_hold_equity",
    "turnover_stats", "volmatched_hold_equity", "warm_window",
]
