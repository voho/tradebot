"""R-70: shared, frozen infrastructure for B-36 -- formalize the Ledoit &
Wolf (2008)-style paired difference test in `tradebot.inference` and apply
it retroactively to the COST axis's near-clearing arms.

Committed BEFORE either branch was written, per ROUTINE.md's parallelism
rules: both branches import this file, NEITHER EDITS IT, and it computes no
test statistic and reaches no verdict of its own. Its only job is to
reproduce, bit-for-bit, the three paired daily-return series the round
compares -- by calling each source round's OWN frozen construction
functions, never reimplementing them.

=====================================================================
WHY THIS ROUND
=====================================================================

Constraint attacked: **ERR** (no error control in the signal path) --
methodology, per the standing diagnosis, not a new mechanism. Backlog item
**B-36**, filed by R-68 and top of the ranked list after R-69 closed B-37:

    "Formalize a Ledoit & Wolf (2008)-style paired difference test between
    a candidate arm and the frozen arm it's meant to improve on, as a
    reusable function in tradebot.inference rather than a one-off script,
    and apply it retroactively to every surviving (further_work-clearing
    or near-clearing) arm on the COST axis before a fifth mechanism round
    is run."

R-68's own `experiments/r68_inference.py` ran one such comparison as a
bespoke script (R-67's delta=0.080 vs R-65's delta=0.000) using this
project's existing `paired_bootstrap` -- a PLAIN percentile stationary
bootstrap of the difference statistic. That is not what Ledoit & Wolf
(2008) actually propose: their paper studentizes the Sharpe-ratio
difference by an estimate of its own asymptotic standard error (a
delta-method gradient against the long-run covariance of the return
moments) before resampling, which is what gives their test its improved
finite-sample coverage over a naive percentile interval on heavy-tailed,
autocorrelated returns -- exactly this project's data. Nothing in this
repo computes that studentized statistic today. This round builds it.

Two independent estimators of the same long-run covariance are used,
because B-36 was filed on the strength of a single near-miss (+0.4525
[-0.069, +1.105]) and one estimator cannot say whether "just short of
significance" is a property of the data or an artifact of how the
variance was estimated:

- **Conservative** (`experiments/r70_conservative_ledoit_wolf.py`): the
  literal L&W (2008) construction -- a Parzen-kernel HAC estimate of the
  long-run covariance matrix of the four moments (mu_1, mu_2, gamma_1,
  gamma_2) with Andrews (1991)/Newey-West-style automatic bandwidth
  selection, exactly as implemented in the reference `PeerPerformance` R
  package's `sharpeTesting()` (Ardia & Boudt), which this project's
  literature review confirms is the standard practical realization of the
  paper.
- **Novel** (`experiments/r70_novel_bootstrap_studentized.py`): the SAME
  studentized statistic, but with the long-run covariance estimated
  nonparametrically via this project's own `stationary_bootstrap_indices`
  (Politis & Romano 1994) at its own established 30-day mean block --
  the convention every other bootstrap interval in this repo already uses
  (R-20's noise floor, `paired_bootstrap`, R-68's own difference test) --
  rather than a kernel with an analytically-chosen bandwidth. This is a
  real methodological fork, not a relabeling: a kernel HAC estimator
  assumes a specific decay shape and picks its bandwidth from the data's
  own autocorrelation, while a block bootstrap at a literature-external,
  already-adopted block length makes no assumption about the covariance's
  functional form at all. If the two disagree, the disagreement is itself
  informative about which one to trust going forward.

=====================================================================
WHAT THIS FILE DOES NOT DO
=====================================================================

- It builds no new strategy and sweeps no new parameter. Every targets
  function it calls is imported, not reimplemented, from the round that
  froze it.
- **Holdout: +0.** Both windows are W_TRAIN and W_VAL, exactly R-68's own
  restriction (`experiments/r68_inference.py`) -- W_FULL6 is deliberately
  excluded because B-33 (whether a W_FULL6 read is genuinely free of the
  reserved BTC/ETH holdout) is still open and load-bearing on three prior
  rounds; this round declines to add a fourth.
- It does not select a winner among the three arms below. All three are
  reported, including any that come back unfavourably -- reporting only
  the favourable cells is the operator-performed version of the exact
  bias ROUTINE.md's parallelism section warns against.

=====================================================================
THE THREE ARM-VS-BASELINE PAIRS
=====================================================================

Baseline, all three: `r65_winner_targets` (k=1, buffer=0.05, hold_days=1,
delta=0 -- R-65's frozen winner, and the same "unbanded" reference
R-68's own difference test used).

1. **R-67's arm** (`experiments.r67_conservative_hysteresis.build_hysteresis_targets`,
   delta=0.080) -- reproduces R-68's own published difference-test pair,
   as a validation that the new machinery agrees with the old one's sign
   and rough magnitude before trusting it on the other two.
2. **R-68 conservative's ENTRY_ONLY arm** (`build_band_targets`,
   delta_in=0.080, delta_out=0.0) -- the round's own selected winner,
   never previously difference-tested against the baseline directly (R-68
   only ran the coupled-diagonal pair above).
3. **R-68 novel's derived-threshold arm** (`targets_fn("D_B", mult=1.0)`)
   -- the zero-fitted-parameter construction, likewise never
   difference-tested.

Six cells total (3 arms x 2 windows), at the 0.10% base fee tier, matching
R-68's own inference script's tier.

    .venv/bin/python experiments/r70_shared.py     # smoke-test: prints shapes
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.inference import daily_returns  # noqa: E402

from experiments.r68_shared import (  # noqa: E402
    SPOT_BASE,
    UNIVERSE_8,
    W_TRAIN,
    W_VAL,
    align_frames,
    config_count,
    load_universe,
    r65_winner_targets,
    simulate_portfolio,
    volmatched_hold_equity,
    warm_window,
)
from experiments.r67_conservative_hysteresis import (  # noqa: E402
    BUFFER_FIXED,
    HOLD_FIXED,
    K_FIXED,
    build_hysteresis_targets,
)
from experiments.r68_conservative_band_decomposition import (  # noqa: E402
    build_band_targets,
)
from experiments.r68_novel_derived_threshold import (  # noqa: E402
    delta_series,
    build_derived_targets,
)

OUT_DIR = ROOT / "reports" / "r70_inference"

WINDOWS = (("W_TRAIN", W_TRAIN), ("W_VAL", W_VAL))

# name -> aligned -> targets. Each closure is a direct, unmodified call into
# the round that froze it; nothing here re-derives a threshold or a rule.
ARMS = {
    "r67_hysteresis_0.080": lambda aligned: build_hysteresis_targets(
        aligned, K_FIXED, BUFFER_FIXED, HOLD_FIXED, 0.080),
    "r68_entry_only_0.080": lambda aligned: build_band_targets(
        aligned, K_FIXED, BUFFER_FIXED, HOLD_FIXED, 0.080, 0.0),
    "r68_novel_derived_mult1.0": lambda aligned: build_derived_targets(
        aligned, delta_series(aligned, "D_B", 1.0), K_FIXED, BUFFER_FIXED,
        HOLD_FIXED),
}


def _arm_daily(frames, window, targets_fn, market=SPOT_BASE):
    """One arm's own daily returns on one window. Mirrors
    `r68_inference.cell_series`'s candidate leg, generalised to take a
    targets closure instead of a fixed delta -- the vol-matched-hold leg is
    deliberately NOT built here, because the comparison this round runs is
    arm vs UNBANDED ARM (R-68's own `series[delta]` construction), not arm
    vs a synthetic hold. `matched` is carried through only as context, from
    R-68's own risk-match diagnostic on the candidate leg."""
    sub = {t: frames[t] for t in UNIVERSE_8}
    warm = align_frames(sub, warm_window(window))
    targets = targets_fn(warm)

    idx = warm[UNIVERSE_8[0]].index
    idx = idx[idx >= pd.Timestamp(window[0], tz="UTC")]
    idx = idx[idx < pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)]
    assert not (idx >= pd.Timestamp("2023-01-01", tz="UTC")).any(), \
        "holdout hygiene: this file evaluates W_TRAIN and W_VAL only"

    aligned = {t: df.loc[idx] for t, df in warm.items()}
    targets = targets.loc[idx]

    eq = simulate_portfolio(targets, aligned, market)
    _, _, _, matched = volmatched_hold_equity(eq, aligned, UNIVERSE_8, market)
    return daily_returns(eq).to_numpy(dtype=float), matched


def build_all_cells() -> dict:
    """{(arm_name, window_name): (candidate_daily, baseline_daily, matched, n)}.

    ``baseline_daily`` is `r65_winner_targets`'s own daily-return series on
    the same window and market -- the "unbanded" reference every arm here
    is meant to improve on, exactly R-68's `DELTA_UNBANDED=0.000` leg."""
    frames = load_universe(UNIVERSE_8)
    cells = {}
    for window_name, window in WINDOWS:
        base, base_matched = _arm_daily(frames, window, r65_winner_targets)
        for arm_name, targets_fn in ARMS.items():
            cand, cand_matched = _arm_daily(frames, window, targets_fn)
            n = min(len(cand), len(base))
            cells[(arm_name, window_name)] = (
                cand[:n], base[:n], bool(cand_matched and base_matched), n)
    return cells


def main():
    cells = build_all_cells()
    for (arm, window), (a, b, matched, n) in cells.items():
        print(f"{arm:28s} {window:8s}  n_days={n:4d}  matched={matched}  "
              f"cand_mean={np.mean(a):+.6f}  base_mean={np.mean(b):+.6f}")
    print(f"\nconfig_count() = {config_count()}")
    print("Holdout consultations: +0 (W_TRAIN and W_VAL only; see B-33).")


if __name__ == "__main__":
    main()
