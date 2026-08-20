"""R-69: does the entry-only edge survive with the R-65/67/68 machinery
removed entirely, leaving one parameter bolted onto R-63's ORIGINAL rule?

Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it (and from `r63_shared.py` / `r63_novel_xsmom_rank.py` /
`r65_shared.py` / `r68_shared.py`, none of which it duplicates), NEITHER
BRANCH EDITS IT, and it does not itself define a candidate strategy or
compute a verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **COST**. Backlog item **B-37**, filed by R-68,
immediately below B-36 (a methodology item, not a candidate) on the ranked
list.

R-68's conservative branch decomposed R-67's single coupled threshold into
`(delta_in, delta_out)` and found ENTRY_ONLY (`delta_in=d, delta_out=0`)
dominates both EXIT_ONLY and the coupled diagonal: W_VAL winner at d=0.080,
net +0.8480 [-0.132, +1.843] (the pre-registered selection criterion), and
on W_FULL6/U6 net +1.0662 [-2.134, +4.303] vs VOLMATCH_HOLD (D1 FAIL),
dd_diff -9.54pp [-26.16, +18.05] (D2 FAIL) -- `further_work=False`, the
round's own predicted base case, but the entry/exit decomposition itself was
a genuine, clean finding: the exit half (B-31's original target) carries
none of the edge.

But R-68's ENTRY_ONLY sub-arm was never run in isolation from the OTHER
machinery it inherited unchanged from R-65: `buffer=0.05` (a score-margin
challengers must clear before a voluntary swap is allowed) and `hold_days=1`
(a minimum-tenure timer gating swaps). Both are still present in every R-68
cell, "frozen ... byte-for-byte where not explicitly varied" (r68_shared.py
docstring). B-37 asks the cheap follow-up directly, in its own words: **is a
one-parameter entry-only gate -- no buffer, no hold_days retuning, no exit
threshold at all -- sufficient, or does the coupling with R-65's machinery
still matter for reasons ENTRY_ONLY's isolated read cannot see?**

This round answers exactly that, and nothing more. It changes ONE thing
relative to R-63's ORIGINAL, unmodified selection rule
(`r63_novel_xsmom_rank.build_targets`: `sel = (s > 0) & (rank < k)`,
recomputed fresh every bar, no persistence, no buffer, no timer): a newly
entering asset must additionally clear `s > delta_in` on the bar it enters.
An asset already held needs only `s > 0` and `rank < k` to remain, exactly
as R-63's original rule already required of every asset, held or not. No
buffer margin gates a swap; no timer blocks one. The only state this
introduces is "was asset a held at bar i-1", which is the minimum needed to
distinguish "entering" from "remaining" at all.

=====================================================================
WHY THIS IS NOT A DUPLICATE
=====================================================================

- **R-68 conservative (band_decomposition), ENTRY_ONLY sub-arm.** Same
  predicate (`s > delta_in` to enter, `s > 0` to remain), but computed
  inside R-65's slot/buffer/hold_days machinery: incumbents are held via an
  explicit `held` list, voluntary swaps require clearing a `buffer` margin
  AND a `hold_days` timer, and entries into a full k=1 slot only happen via
  that gated swap path. This round removes buffer and the timer entirely and
  goes back to R-63's original PER-BAR recomputed top-k-among-eligible
  selection, so an asset can be displaced by a higher-ranked eligible
  entrant on the very next bar with no margin and no minimum tenure. If the
  two constructions score alike, R-65's buffer/timer contributed nothing
  beyond what the entry bar alone buys, on this axis; if they diverge, the
  buffer/timer combination is doing real work ENTRY_ONLY's R-68 read could
  not isolate.
- **R-68 novel (derived_threshold).** Derives a threshold and applies it
  inside the SAME buffer/hold_days band construction as R-68 conservative
  (`delta_in = delta_out = delta_B(t)` there, the coupled diagonal, not an
  entry-only read). This round's novel branch applies the identical D-B
  formula (`sigma_ds(t) * sqrt(T*)`, R-68's already-causal, already-derived
  estimator, imported read-only from `r68_novel_derived_threshold.py` rather
  than re-derived) as the entry-only threshold on THIS round's bufferless
  construction -- same formula, different codepath, so a match or a mismatch
  is informative about the codepath rather than about the formula.
- **R-65 (rank buffer / aim portfolio).** Introduces `buffer` and
  `hold_days` as the axis under test. This round holds neither: both are
  simply absent, not frozen at R-65's selected values.
- **R-63 novel (`xsmom_rank`).** The k=1 baseline this round's candidate
  reduces to at `delta_in=0`, required as an exact identity below.

=====================================================================
THE TWO BRANCHES
=====================================================================

Both start from R-63's frozen constants unmodified: `k=1`
(`K_FROZEN` in `r63_novel_xsmom_rank.py`), the composite cross-sectional
score and conditional volatility scale, R-63's own 0.10 deadband on desired
TOTAL notional (part of the original construction, not R-65's addition, and
therefore kept), equal weighting among held slots (moot at k=1), long-only
spot, unlevered. The ONLY axis either branch touches is `delta_in`.

**CONSERVATIVE** (`experiments/r69_conservative_entry_gate.py`): sweep
`delta_in` over R-68's own extended grid, `r68_shared.DELTA_GRID_EXT`
(0.000 .. 0.350, capped at the dLC 2020 Eq. (11) saturation already derived
and frozen by that round -- not re-derived here), select on W_VAL exactly as
R-68 did, then read the W_FULL6/U6 D-cells at the selected point.
`delta_in=0.000` must reproduce `r63_baseline_targets(aligned, k=1)`
(R-63's ORIGINAL, unmodified rule) EXACTLY -- this round's own identity
check, analogous to R-68's `d=0` identity against R-65's winner.

**NOVEL** (`experiments/r69_novel_derived_entry.py`): zero fitted
parameters. `delta_in(t) = delta_B(t) = sigma_ds(t) * sqrt(T*)`, R-68's own
cost-matched first-passage threshold (Kaminski & Lo 2014's stopping-time
identity applied to R-65's measured decay table; causal by construction,
expanding, one-bar-shifted -- see `r68_novel_derived_threshold.py`'s module
docstring for the full derivation and its stated regime of validity, which
this branch inherits unchanged and does not re-argue). Imported read-only
from that file, not re-derived, so a numerical disagreement is a bug report
rather than a second derivation. The neighbourhood multipliers R-68 reported
(0.5, 0.75, 1.0, 1.5, 2.0) are reported here too, for the same
plateau-vs-peak reason, and the derived point (mult=1.0) is the one D-cells
are read at -- not the best-scoring multiplier, exactly as R-68's own rule
required.

**CAUSALITY.** The conservative branch's threshold is a swept constant and
cannot leak. The novel branch's threshold is a function of measured score
statistics and MUST be computed on an EXPANDING or ROLLING window, SHIFTED
by at least one bar -- inherited directly from `r68_novel_derived_threshold`,
which already satisfies this and already carries its own truncation and
perturbation probes. Both branches additionally run `check_causality` on
their own `build_targets`-equivalent function, because the membership state
(`held at bar i-1`) is new machinery this round introduces and R-68's own
causality probes never exercised it.

=====================================================================
GATES, FROZEN -- INHERITED, NOT RE-DERIVED
=====================================================================

D1/D2/D5's bar, the bootstrap, `VOLMATCH_HOLD`/`MATCHED_HOLD`/`EW_HOLD`/
`BTC_HOLD`, the windows and the two cost tiers are inherited BY IMPORT from
`r65_shared.py` / `r63_shared.py`, unchanged. M1' and the fixed-permutation
scramble control are inherited BY IMPORT from `r68_shared.py`, unchanged.
This round changes exactly one thing: **M1's baseline.**

**M1' is measured against R-63's ORIGINAL rule (`r63_baseline_targets`,
k=1), not R-65's winner.** R-68's M1' asked "does the band reduce turnover
relative to R-65's already-reduced baseline". This round's own question is
narrower and harder: does the entry gate ALONE, with no buffer and no
timer, reduce turnover relative to the UNMODIFIED R-63 rule it is bolted
onto. Both branches compute the baseline on the SAME aligned frame and
window they evaluate the candidate on (`raw_baseline_cell` below), rather
than importing a hardcoded number, so a baseline mismatch cannot silently
enter through stale constants.

FURTHER-WORK BAR (unchanged in shape from R-65/R-67/R-68):
    M1' AND (D1 or D2) AND D3 AND D5 AND scramble_survived
Clearing it authorizes exactly ONE holdout read on `W_HOLD`; it is NOT the
promotion bar. Nothing here is registrable regardless of verdict -- both
branches are bar-by-bar cross-asset allocators and **B-32** (multi-asset
registration) is still OPEN, three rounds running. Recorded so a positive
result is not overclaimed.

=====================================================================
WHAT WOULD MAKE THIS FAIL -- NAMED NOW, BEFORE ANY CODE
=====================================================================

**(F1) The buffer/timer were doing real work.** R-65's buffer prevents a
one-bar rank flicker from causing a swap; its timer prevents two swaps in
quick succession. Neither exists here. The predicted failure mode: turnover
and membership-change rate come back HIGHER than R-68's ENTRY_ONLY cell at
the same delta, because a challenger clearing `delta_in` can now displace an
incumbent on literally the next bar with no margin and no cooldown -- i.e.
M1' fails even at deltas where R-68's banded construction passed it.

**(F2) The edge was never in the entry bar at all, but in the interaction.**
If turnover comes back comparable to R-68's ENTRY_ONLY cell (F1 does not
fire) and D1/D2 still fail by a similar margin, that reproduces R-68's
result cleanly and answers B-37's question "yes, one parameter is
sufficient, and it still isn't enough to clear the interval." That is a
genuine, informative NEGATIVE, not a failure of this round.

**(F3) The base case, restated from R-67/R-68.** Three rounds running have
improved this signal's economics by a large factor without moving
`(D1 or D2)`, because the binding constraint is the interval's width, not
any mechanism on it (R-67's own lesson). **The base case for this round is
that `(D1 or D2)` fails again**, on both branches, regardless of what M1'
says. It is run anyway because B-37's question -- does the entry bar alone,
context-free, do what R-68's coupled construction did -- is answerable
independently of the interval, and because "not tested" is not the same as
"ruled out" (ROUTINE.md's parallelism section).

**(F4) Theory and data may disagree.** D-B may land far from the region the
conservative sweep prefers, exactly as D-A did (missed by 6-11x) in R-68.
Reported as a miss if it is one, not dropped.

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
    check_against_engine,
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
    K_FROZEN,
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
    R63_TURNOVER_PER_DAY,
    d1_pass,
    d2_pass,
    d3_pass,
    frontier_row,
    holding_period_days,
    realized_vol,
    turnover_stats,
    volmatched_hold_equity,
)
from experiments.r68_shared import (  # noqa: E402,F401
    DELTA_GRID_EXT,
    DLC_SATURATION,
    MEMBERSHIP_WEIGHT_FLOOR,
    M1_MIN_REDUCTION,
    SIGMA_SCORE_W_TRAIN,
    held_indicator,
    m1_pass,
    membership_change_rate_thresholded,
    scramble_fixed_perm,
)

OUT_DIR = ROOT / "reports" / "r69_entry_only_raw"

# D5 uses R-65's own corrected, harder bar (like-for-like against
# VOLMATCH_HOLD) -- the same choice R-68 made and disclosed.
D5_BAR_R69 = D5_BAR_CORRECTED  # +0.342

# R-68's own published reference cell, for context only (a different
# codepath, not an identity this round is required to reproduce).
R68_ENTRY_ONLY_DELTA_WINNER = 0.080
R68_ENTRY_ONLY_WVAL_NET = 0.8480          # [-0.132, +1.843]
R68_ENTRY_ONLY_WFULL6_NET_VOLMATCH = 1.0662   # [-2.134, +4.303], D1 FAIL
R68_ENTRY_ONLY_WFULL6_DD_VOLMATCH = -9.54     # pp, [-26.16, +18.05], D2 FAIL
R68_ENTRY_ONLY_TURNOVER_PER_DAY = 0.1183
R68_ENTRY_ONLY_MEMBERSHIP_PER_DAY = 0.1163


# ------------------------------------------------------------------ M1' baseline


def raw_baseline_cell(aligned: dict, k: int = K_FROZEN):
    """R-63's ORIGINAL rule on the SAME aligned frame a branch is scoring its
    candidate on -- the baseline M1' is measured against in this round.

    Returns (targets, turnover_per_day, membership_per_day). Computed fresh
    every call (not memoized) so a branch cannot accidentally score a
    candidate against a baseline built on a different window.
    """
    targets = r63_baseline_targets(aligned, k)
    turnover = turnover_stats(targets)["turnover_per_day"]
    membership = membership_change_rate_thresholded(targets)
    return targets, turnover, membership


def m1_pass_vs_raw(cand_targets: pd.DataFrame, aligned: dict, k: int = K_FROZEN) -> dict:
    """M1', baseline = R-63's ORIGINAL rule on this cell's own aligned frame
    (see module docstring: this round's M1' baseline, not R-68's)."""
    _, base_turnover, _ = raw_baseline_cell(aligned, k)
    cand_turnover = turnover_stats(cand_targets)["turnover_per_day"]
    baseline_targets, _, _ = raw_baseline_cell(aligned, k)
    return m1_pass(cand_targets, baseline_targets, cand_turnover, base_turnover)


# ------------------------------------------------------------------ D5


def d5_pass(row: dict) -> bool:
    """Signal retention against the CORRECTED, like-for-like bar."""
    return row["gross_growth_diff"] >= D5_BAR_R69


def further_work(m1: bool, d1: bool, d2: bool, d3: bool, d5: bool,
                 scramble_survived: bool) -> bool:
    """This round's bar, identical in shape to R-68's. NOT the promotion
    bar; clearing it authorizes exactly one holdout read on W_HOLD."""
    return m1 and (d1 or d2) and d3 and d5 and scramble_survived


__all__ = [
    "BARS_PER_DAY", "BARS_PER_YEAR", "BOOT_KW", "D5_BAR_R69", "DEADBAND",
    "DELTA_GRID_EXT", "DLC_SATURATION", "HORIZONS", "K_FROZEN",
    "MEMBERSHIP_WEIGHT_FLOOR", "M1_MIN_REDUCTION", "OUT_DIR",
    "R63_GROSS_EDGE", "R63_GROSS_EDGE_VS_VOLMATCH", "R63_TURNOVER_PER_DAY",
    "R65_OUT_DIR", "R68_ENTRY_ONLY_DELTA_WINNER",
    "R68_ENTRY_ONLY_MEMBERSHIP_PER_DAY", "R68_ENTRY_ONLY_TURNOVER_PER_DAY",
    "R68_ENTRY_ONLY_WFULL6_DD_VOLMATCH", "R68_ENTRY_ONLY_WFULL6_NET_VOLMATCH",
    "R68_ENTRY_ONLY_WVAL_NET", "SCRAMBLE_SEEDS", "SIGMA_SCORE_W_TRAIN",
    "SPOT_BASE", "SPOT_REAL", "START_BALANCE", "TOTAL_NOTIONAL_DEADBAND",
    "UNIVERSE_6", "UNIVERSE_8", "WARM_DAYS", "W_FULL6", "W_HOLD", "W_TRAIN",
    "W_VAL", "align_frames", "basket_log_returns", "check_against_engine",
    "check_causality", "compare", "conditional_vol_scale", "config_count",
    "cross_sectional_score", "d1_pass", "d2_pass", "d3_pass", "d5_pass",
    "excludes_zero", "frontier_row", "further_work", "held_indicator",
    "holding_period_days", "load_universe", "m1_pass", "m1_pass_vs_raw",
    "matched_hold_targets", "mean_total_notional",
    "membership_change_rate_thresholded", "r63_baseline_targets",
    "raw_baseline_cell", "realized_vol", "scramble_fixed_perm",
    "simulate_portfolio", "static_hold_equity", "turnover_stats",
    "volmatched_hold_equity", "warm_window",
]
