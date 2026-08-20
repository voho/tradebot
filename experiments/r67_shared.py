"""R-67: attack B-31 -- the long/flat gate's forced zero-crossing exits.

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it (and from `r63_shared.py` / `r65_shared.py`, which it does
not duplicate), NEITHER BRANCH EDITS IT, and it does not itself define a
candidate strategy or compute a verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **COST**. Backlog item **B-31**, filed by R-65 and
top of the ranked, actionable list (LEDGER.md D, "NEXT").

R-65's conservative branch (rank buffering + a minimum holding period on
R-63's cross-sectional trend selection) bought a 3.9x turnover cut
essentially for free -- gross edge did not fall -- and still fell short of
its own break-even by 1.38:1 (needed turnover <= 0.641/day, reached
0.900). It named the exact reason: R-63's frozen selection rule is
`eligible = (score > 0)`, "hold only positive-scoring assets, flat
otherwise", and every crossing of that boundary forces an exit -- **never
blocked by the buffer or the minimum-hold timer**, because a forced exit
is deliberately exempt from both (holding a decaying asset for 30 days
regardless was rejected as a worse hypothesis by R-65's own
pre-registration). That channel measured **invariant at 0.386 forced
exits/day across all 20 cells of R-65's buffer x hold_days grid**, while
voluntary swaps fell 16-fold (0.270 -> 0.017/day) over the same grid. It is
a hard floor the round could not touch by retuning either of its two free
parameters, because neither parameter governs it.

This round attacks that floor directly, the way B-31 names it: by
softening the `s > 0` gate itself rather than the buffer or timer wrapped
around it.

=====================================================================
WHY THIS IS NOT A DUPLICATE
=====================================================================

- **R-65 conservative (rank_buffer).** Retunes `buffer` (voluntary-swap
  margin) and `hold_days` (minimum tenure) with the eligibility test itself
  frozen at `s > 0.0`. This round freezes `buffer`/`hold_days` at R-65's own
  selected winner (0.05, 1) and retunes the eligibility test instead --
  the orthogonal axis, explicitly named as untouched by R-65's own
  diagnosis ("a hard floor the round could not touch").
- **R-65 novel (aim_portfolio).** Smooths a *different* aim construction
  (per-horizon top-1 one-hot, GP-persistence-weighted, then partially
  adjusted) and already runs at low turnover (0.19/day) without touching
  R-63's gate at all -- which is exactly why B-31's own filing says this
  question "may turn out to matter only for the discrete-selection
  family." This round's novel arm applies partial adjustment to a
  *different* base signal: R-63's own composite cross-sectional score
  directly, with k fixed at 1, rather than a multi-horizon one-hot blend.
  Not a re-run of R-65's novel arm under a new name.
- **L-05 / L-06 (`kelly_regime_ev`, `kelly_regime_ev_fast`).** A no-trade
  band on a *continuous single-asset exposure fraction*. This round bands
  (conservative) or smooths (novel) a *discrete cross-sectional
  membership* decision -- the same distinction R-65's own docstring drew
  against L-05/L-06.
- **R-64 / R-66 (trade-to-the-boundary / snap-to-flat, B-29).** Those
  attack a single-asset SIZE-axis deadband's *destination* (does it reach
  exactly flat) on `kelly_regime_v4` itself. This round attacks a
  *different* rule's (R-63's cross-sectional selector) *entry/exit
  threshold*, on a different signal, in a different (unregistered,
  multi-asset) codepath. The mechanisms rhyme -- both are about a discrete
  jump to/from zero -- and R-64/R-66's finding is used below as the named
  risk this round predicts for itself (see F1).

=====================================================================
THE TWO BRANCHES
=====================================================================

Both start from R-65's own selected winner **byte-for-byte** where not
explicitly varied: `k=1` (R-63's frozen concentration), R-63's composite
cross-sectional score and conditional volatility scale (unmodified),
R-63's 0.10 deadband on desired total notional, equal weighting among held
slots, long-only spot, unlevered. Only the membership mechanism changes.

**CONSERVATIVE -- asymmetric hysteresis around the selection boundary**
(`experiments/r67_conservative_hysteresis.py`). R-65's `buffer` parameter
already bands the *voluntary swap* decision (challenger vs incumbent); it
does nothing to the *involuntary exit* decision, which fires the instant
score crosses exactly zero. This branch adds the literal fix B-31 names:
an asset already held stays eligible until its score falls below `-delta`
(not `<= 0`), while a new entrant still needs `score > +delta`. `buffer`
stays frozen at R-65's winning 0.05, `hold_days` at 1 -- one new
parameter, `delta`, swept on W_TRAIN, selected on W_VAL by the same
criterion R-65 used.

**NOVEL -- partial adjustment on R-63's own composite score**
(`experiments/r67_novel_smoothed_score.py`). B-31's third named candidate:
"letting the partial-adjustment recursion carry the position through a
crossing instead of resetting it." Garleanu & Pedersen's (2013) closed
form -- already used by R-65's novel arm on a *different* aim -- applied
here directly to R-63's `k=1` target vector: `x_t = x_{t-1} + a * (aim_t -
x_{t-1})`, with `aim_t` R-63's own byte-for-byte discrete target
(unmodified score, vol scale, deadband) and `a` the one free parameter,
swept on W_TRAIN, selected on W_VAL. `a=1.0` is a built-in no-op check --
it must reproduce R-63's own k=1 arm exactly, which the branch is required
to verify before reporting anything else. Structurally different from the
conservative arm: no boolean membership exists anywhere in this
construction, so there is no threshold to cross, only a continuous chase.

=====================================================================
THE MECHANISM CHECK -- THIS ROUND'S OWN FALSIFICATION TEST
=====================================================================

Named now, before any candidate number is read, because a round that
merely retunes a knob without ever reducing forced-exit frequency has not
tested the hypothesis at all.

**M1 (mechanism, gate).** At the branch's selected configuration, on
W_TRAIN, the change frequency of the held-set indicator (`weight > 0`),
measured by :func:`membership_change_rate` below (an application of
`r65_shared.holding_period_days`'s own counting rule -- "time between
changes to the held set" -- to the SAME quantity R-65 used to name the
floor), must be materially lower than the SAME statistic computed on
**R-65's own frozen conservative winner** (`k=1, buffer=0.05,
hold_days=1`), supplied by :func:`r65_winner_targets` and shared by both
branches so the two reports are commensurable. "Materially" is fixed here,
before any run, at **>= 25% fewer membership changes per day**. A branch
that fails M1 has not touched the mechanism B-31 named and its D-cells are
reported as a diagnostic only, not as a test of the hypothesis.

This is a coarser instrument than R-65's own `forced_exit` event ledger
(it counts every held-set change, forced or voluntary, because the novel
arm has no discrete "forced" event to count at all) but it is the one
quantity both branches' target matrices share by construction, and it is
exactly the column R-65 used to diagnose the floor in the first place.

=====================================================================
DECISION RULES, FROZEN -- REUSED, NOT RESTATED
=====================================================================

`experiments/r65_shared.py` already carries D1/D2/D3/D5/`further_work`,
the bootstrap machinery, the benchmarks (`MATCHED_HOLD`, `VOLMATCH_HOLD`,
`EW_HOLD`), the windows (`W_TRAIN`, `W_VAL`, `W_FULL6`, `W_HOLD`) and the
cost tiers, all inherited unchanged from R-63. Re-deriving them here would
risk a silent drift from the number this round extends, so both branches
import them directly and this round adds ONLY the M1 gate above and D4
(the 0.40% cost tier vs `EW_HOLD`, imported as `SPOT_REAL`).

FURTHER-WORK BAR (unchanged from R-65, plus M1):
    M1 AND (D1 or D2) AND D3 AND D5 AND scramble_survived (fixed-permutation
    form -- R-65's own correction, since neither arm here is guaranteed to
    have R-63's original arm's high turnover).
Clearing it authorizes exactly ONE holdout read on `W_HOLD`; it is NOT the
promotion bar. Even a cleared bar does not by itself put anything in the
comparison table -- B-32 (multi-asset registration) remains a separate,
OPEN backlog item, and neither branch here is registrable as-is regardless
of verdict (recorded so a positive result is not overclaimed).

=====================================================================
WHAT WOULD MAKE THIS FAIL -- NAMED NOW, BEFORE ANY CODE
=====================================================================

**(F1) The residual-long precedent (R-64/R-66) applies here by the same
logic, on a different codepath.** R-66 found, twice, from opposite
directions, that reaching exactly flat was not what cost `kelly_regime_v4`'s
boundary-destination arm growth on ETH -- and that removing a residual
long (conservative) or reaching flat as a smooth consequence (novel)
either failed to help or actively cost more. Both of this round's arms
predict the SAME risk in mirror image: widening the exit threshold
(conservative) or smoothing through a crossing (novel) means holding a
GENUINELY declining asset a little longer before de-risking, in a
long-only universe where "declining" often means "about to keep
declining." If the forced-exit floor exists because those exits are
informative rather than because the threshold is degenerate, softening it
trades turnover for drawdown at unfavourable odds -- cheaper, not better.
Named now so that a pass reads as a genuine surprise.

**(F2) The floor may not be a threshold artifact at all.** R-65 measured
it as invariant across an entire buffer x hold_days grid, which is
consistent with either "the threshold itself is the problem" (this
round's hypothesis) or "score crosses zero this often regardless of any
downstream rule" (a property of the composite score's own noise, which no
downstream softening fixes -- it would only relabel the crossings as slow
fades instead of fast exits, at the same underlying frequency of true
sign changes). M1 is what tells these two apart: if M1 fails at every
swept parameter short of a delta/a extreme enough to gut D5, (F2) is
confirmed and the floor was never a gate-design problem.

**(F3) The frontier's far end is a hold, again.** As `delta -> large` or
`a -> 0`, both arms converge on "buy the initially-strongest asset and
hold it indefinitely," which is a concentrated buy-and-hold and may beat
`VOLMATCH_HOLD` by luck of which asset was strongest in 2020-04 rather
than by signal. D5 (signal retention, >= half of R-63's own +0.480 gross
edge) is what R-65 built to catch exactly this, inherited unchanged.

None of the three is a reason not to run it. R-65 named this floor as a
specific, attackable target with a known invariant value; this round is
the direct, pre-registered attempt, and a clean failure that confirms (F1)
or (F2) closes B-31 with a mechanism attached rather than merely a number,
exactly as R-66 did for B-29.

Configurations evaluated are counted by `r63_shared.config_count()`,
shared process-wide; each branch reports its own count and the round's
total is the sum, per ROUTINE's parallelism rules.
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
from experiments.r65_conservative_rank_buffer import (  # noqa: E402,F401
    build_buffered_targets as _r65_build_buffered_targets,
)
from experiments.r65_shared import (  # noqa: E402,F401
    D5_BAR,
    OUT_DIR as R65_OUT_DIR,
    R63_GROSS_EDGE,
    R63_NET_D1,
    R63_TURNOVER_PER_DAY,
    SPOT_FREE,
    d1_pass,
    d2_pass,
    d3_pass,
    d5_pass,
    frontier_row,
    further_work as further_work_r65,
    holding_period_days,
    realized_vol,
    turnover_stats,
    volmatched_hold_equity,
)

OUT_DIR = ROOT / "reports" / "r67_gate"

# R-65's own selected winner on the buffer/hold_days axis -- the substrate
# both branches here start from, frozen. Cited from
# `experiments/r65_conservative_rank_buffer.py` (FROZEN_BUFFER,
# FROZEN_HOLD_DAYS), not re-derived.
R65_K = 1
R65_BUFFER = 0.05
R65_HOLD_DAYS = 1

# R-65's own measured invariant, cited for the M1 comparison's context (the
# branches measure their OWN baseline cell rather than trusting this
# number blind, since it was measured on a slightly different quantity --
# see M1's docstring above).
R65_FORCED_EXIT_PER_DAY = 0.386
R65_BREAKEVEN_TURNOVER_PER_DAY = 0.641
R65_ACHIEVED_TURNOVER_PER_DAY = 0.900

M1_MIN_REDUCTION = 0.25  # >=25% fewer membership changes/day than baseline


def r65_winner_targets(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """R-65's own frozen conservative winner (k=1, buffer=0.05, hold_days=1),
    imported rather than reconstructed, and re-exposed here as the ONE fixed
    M1 reference point both branches compare against -- deliberately not
    each branch's own no-op/identity setting (delta=0 or a=1.0), which are
    two DIFFERENT constructions (the conservative arm's delta=0 coincides
    with this; the novel arm's a=1.0 is R-63's raw, unbuffered k=1 arm, a
    higher-turnover point). Using one shared baseline keeps M1 comparable
    across both branches' reports.
    """
    return _r65_build_buffered_targets(aligned, R65_K, R65_BUFFER, R65_HOLD_DAYS)


def membership_change_rate(targets: pd.DataFrame) -> float:
    """Changes to the held-set indicator (`weight > 0`) per day.

    Identical counting rule to `r65_shared.holding_period_days`'s
    numerator -- re-exposed here under the name this round's M1 gate
    reasons about, so a reader does not have to infer that "time between
    changes" and "changes per day" are reciprocals of the same count.
    """
    days = max(len(targets) / BARS_PER_DAY, 1e-9)
    hp = holding_period_days(targets)
    changes = days / hp if hp > 0 else 0.0
    return changes / days


def m1_pass(cand_targets: pd.DataFrame, baseline_targets: pd.DataFrame) -> dict:
    """The round's own mechanism gate: did the branch actually reduce
    membership-change frequency vs R-65's frozen baseline, on the SAME
    window? Returns the rates and the pass/fail so both are reported even
    when the gate fails.
    """
    cand_rate = membership_change_rate(cand_targets)
    base_rate = membership_change_rate(baseline_targets)
    reduction = 1.0 - (cand_rate / base_rate) if base_rate > 0 else float("nan")
    return {
        "cand_rate_per_day": cand_rate,
        "baseline_rate_per_day": base_rate,
        "reduction": reduction,
        "passed": bool(np.isfinite(reduction) and reduction >= M1_MIN_REDUCTION),
    }


def further_work(m1: bool, d1: bool, d2: bool, d3: bool, d5: bool,
                 scramble_survived: bool) -> bool:
    """This round's further-work bar: R-65's bar, gated additionally on M1.
    NOT the promotion bar. Clearing it authorizes exactly one holdout read
    on W_HOLD."""
    return m1 and (d1 or d2) and d3 and d5 and scramble_survived


__all__ = [
    "BARS_PER_DAY", "BARS_PER_YEAR", "BOOT_KW", "D5_BAR", "DEADBAND",
    "HORIZONS", "M1_MIN_REDUCTION", "OUT_DIR", "R63_GROSS_EDGE",
    "R63_NET_D1", "R63_TURNOVER_PER_DAY", "R65_ACHIEVED_TURNOVER_PER_DAY",
    "R65_BREAKEVEN_TURNOVER_PER_DAY", "R65_BUFFER",
    "R65_FORCED_EXIT_PER_DAY", "R65_HOLD_DAYS", "R65_K", "R65_OUT_DIR",
    "SCRAMBLE_SEEDS", "SPOT_BASE", "SPOT_FREE", "SPOT_REAL",
    "START_BALANCE", "TOTAL_NOTIONAL_DEADBAND", "UNIVERSE_6", "UNIVERSE_8",
    "WARM_DAYS", "W_FULL6", "W_HOLD", "W_TRAIN", "W_VAL", "align_frames",
    "basket_log_returns", "check_causality", "compare",
    "conditional_vol_scale", "config_count", "cross_sectional_score",
    "d1_pass", "d2_pass", "d3_pass", "d5_pass", "excludes_zero",
    "frontier_row", "further_work", "further_work_r65",
    "holding_period_days", "load_universe", "matched_hold_targets",
    "m1_pass", "matched_hold_targets", "mean_total_notional",
    "membership_change_rate", "r63_baseline_targets", "r65_winner_targets",
    "realized_vol",
    "scramble_targets", "simulate_portfolio", "static_hold_equity",
    "turnover_stats", "volmatched_hold_equity", "warm_window",
]
