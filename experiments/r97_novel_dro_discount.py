"""R-97 novel branch: independent robustness/sensitivity check on the
Step-0 gate computed in `experiments/r97_shared.py` -- NOT a from-scratch
re-derivation (the conservative branch is doing that in parallel with the
same causal logic; this branch's job is structurally different: stress-test
whether the pre-registered FAIL is fragile to the two a-priori constants
that were merely "reasonable round numbers" rather than derived from data).

--------------------------------------------------------------------------
THE GATE, AS ALREADY RUN BY THE OPERATOR (r97_shared.py, frozen a priori):
--------------------------------------------------------------------------
N(t) at the six dated BTC stress episodes = [30, 48, 68, 97, 103, 109]
  -> kill switch A (>=4 distinct N values): PASS (6 distinct).
discount(N) at those six episodes = [0.7597, 0.8000, 0.8264, 0.8504,
  0.8542, 0.8577] (computed at BETA_CONF=0.10, N_REF=3, KAPPA=1.0)
  -> ratio max/min = 1.129
  -> kill switch B (ratio >= 1.3x): FAIL.

Per the pre-registered rule in r97_shared.py and docs/ROUTINE.md step 4,
this FAIL is the round's decision for the novel branch too: the gate
failed at its frozen a-priori parameters, so **no strategy or backtest
code is built, and the holdout (>= OOS_START = 2023-01-01) is never
touched.** Nothing below reopens that decision.

--------------------------------------------------------------------------
WHAT THIS FILE ADDS: a diagnostic sensitivity sweep, run AFTER seeing the
FAIL above, and reported as exploratory/in-sample color for the ledger
write-up only -- per docs/ROUTINE.md step 4's rule that any threshold
inspection that happens after seeing a result must be disclosed as such
and downgraded to in-sample, never used to overturn a pre-registered gate.
--------------------------------------------------------------------------

Question asked: is kill switch B's FAIL an artifact of the two fixed
constants BETA_CONF=0.10 and N_REF=3 (both chosen as defensible round
numbers, not fitted), or does the FAIL hold up across a small, defensible
neighborhood of those choices? `KAPPA` is deliberately NOT swept: it is a
pure multiplicative scale on the radius `eps(N) = kappa * sqrt(log(1/beta)
/ N)`, and the discount ratio `max(d)/min(d)` is invariant to any rescaling
of `eps` that is common across all six episodes' N values -- kappa cancels
out of the ratio algebraically (see `_kappa_invariance_check` below, which
verifies this numerically rather than just asserting it). Sweeping it
could not change the verdict, so it is excluded from the grid to keep the
"small, pre-specified" character of the check honest.

Grid (9 cells, all evaluated on inner-train + inner-validation only, i.e.
strictly before OOS_START -- the holdout is never read by this file):
  BETA_CONF in {0.05, 0.10, 0.20}  (Li 2023's own reported range is 5-20%)
  N_REF     in {2, 3, 5}           (neighbors of the project's own
                                     N-approx-3 diagnosis)

Result (see stdout of running this file): the discount ratio stays in a
narrow band across all 9 cells and DOES NOT clear 1.3x anywhere in the
grid. The FAIL is robust to this neighborhood of the a-priori constants,
not an artifact of the particular round numbers chosen -- it is driven by
the shape of the N(t) trajectory across the six episodes (all six
episodes sit at N >= 30, deep in the sqrt(1/N) radius formula's flat
region, where further increases in N buy very little additional radius
shrinkage) rather than by BETA_CONF or N_REF specifically.

STOP: this sweep is diagnostic only. It does not promote, does not build
`kelly_regime_v4 * discount(N)` as a registered or even experimental
strategy, and does not run `scripts/experiment.py` or touch the holdout.
Per the pre-registered decision rule, kill switch B FAILED and the round
stops here for the novel branch. Had the gate PASSED (at the frozen
a-priori parameters, not at any cell found by this post-hoc sweep), the
construction this branch would have built is: leave `kelly_regime_v4`'s
existing `frac x scale` position pipeline completely untouched, and apply
`position_final = position_v4 * dro_discount(regime_cycle_count(t))` as a
separate, bounded ([0,1]-valued by construction) multiplicative overlay
computed causally at every bar -- isolating the DRO-ambiguity effect as a
pure add-on layer on top of v4's frozen sizing machinery, in the spirit of
the factor-isolation R-62 established as good practice, rather than a
full scale replacement. That strategy file was never written.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.r97_shared import (  # noqa: E402
    INNER_VAL_END,
    KAPPA,
    STRESS_EPISODES,
    assert_no_holdout,
    episode_pre_window,
    regime_cycle_count,
    wasserstein_radius,
    dro_discount,
)

# Pre-specified, small grid. Chosen for coverage of the literature's own
# reported range (Li 2023: beta in 5-20%) and the project's own N-approx-3
# neighborhood, NOT swept to find a passing cell -- reported in full below
# regardless of outcome.
BETA_GRID = (0.05, 0.10, 0.20)
NREF_GRID = (2, 3, 5)


def _kappa_invariance_check() -> None:
    """Numeric confirmation that KAPPA cancels out of the discount ratio,
    justifying its exclusion from the swept grid. Checks two arbitrary
    kappa values against a small synthetic N array and asserts the ratio
    of resulting discounts is identical."""
    n_fake = np.array([30.0, 48.0, 68.0, 97.0, 103.0, 109.0])
    for kappa_test in (0.3, 1.0, 5.0):
        d = dro_discount(n_fake, n_ref=3, kappa=kappa_test, beta=0.10)
        ratio = d.max() / d.min()
        d_base = dro_discount(n_fake, n_ref=3, kappa=KAPPA, beta=0.10)
        ratio_base = d_base.max() / d_base.min()
        assert np.isclose(ratio, ratio_base, atol=1e-9), (
            f"kappa={kappa_test} changed the ratio ({ratio} vs {ratio_base}); "
            "the claimed invariance does not hold, KAPPA must be swept too.")
    print("kappa-invariance check: PASS (ratio unchanged for kappa in "
          "{0.3, 1.0, 5.0} at fixed beta/n_ref) -- confirms KAPPA is "
          "correctly excluded from the swept grid.\n")


def episode_n_values(df) -> list[float]:
    """Causal N(t) at each of the six stress episodes -- identical logic
    to r97_shared.step0_gate, factored out here only so it is computed
    once and reused across every grid cell (N(t) does not depend on
    BETA_CONF or N_REF at all -- only discount(N) does)."""
    cycles = regime_cycle_count(df)
    ns = []
    for _label, onset_str in STRESS_EPISODES:
        ts = episode_pre_window(df, onset_str)
        if ts is None:
            ns.append(None)
            continue
        ns.append(float(cycles.loc[ts]))
    return ns


def grid_cell_ratio(n_values: list[float], beta: float, n_ref: float) -> float:
    """Discount ratio (max/min) at one (beta, n_ref) grid cell, given the
    already-computed episode N(t) values. KAPPA is held at the module's
    fixed a-priori value throughout (irrelevant to the ratio; see
    `_kappa_invariance_check`)."""
    valid_n = [n for n in n_values if n is not None]
    d_values = [float(dro_discount(n, n_ref=n_ref, kappa=KAPPA, beta=beta))
                for n in valid_n]
    if not d_values or min(d_values) <= 0:
        return float("inf")
    return max(d_values) / min(d_values)


def main() -> None:
    from tradebot.data import load_dataset

    df, label = load_dataset(ROOT / "data", "spot")
    inner = df.loc[:INNER_VAL_END]
    assert_no_holdout(inner)  # hard guard: this file never reads the holdout

    print(f"data: {label}, inner bars (<= {INNER_VAL_END}): {len(inner):,}\n")

    _kappa_invariance_check()

    n_values = episode_n_values(inner)
    print("causal N(t) at the six episodes (identical to r97_shared.py, "
          "recomputed independently here via the same imported function):")
    for (elabel, onset), n in zip(STRESS_EPISODES, n_values):
        print(f"  {elabel:42s} {onset}  N={n}")
    print()

    print("SENSITIVITY GRID -- discount ratio (max/min) across the six "
          "episodes, at BETA_CONF x N_REF cells (KAPPA fixed, provably "
          "irrelevant to the ratio):")
    print(f"{'':>12s}" + "".join(f"N_REF={nr:<10d}" for nr in NREF_GRID))
    any_pass = False
    results = {}
    for beta in BETA_GRID:
        row_vals = []
        for n_ref in NREF_GRID:
            ratio = grid_cell_ratio(n_values, beta, n_ref)
            results[(beta, n_ref)] = ratio
            row_vals.append(ratio)
            if ratio >= 1.3:
                any_pass = True
        row_str = "".join(f"{v:<15.4f}" for v in row_vals)
        print(f"beta={beta:<7.2f}{row_str}")

    print(f"\nfrozen a-priori cell (BETA_CONF=0.10, N_REF=3) ratio: "
          f"{results[(0.10, 3)]:.4f}  "
          f"(matches r97_shared.py's reported 1.129: "
          f"{'yes' if abs(results[(0.10, 3)] - 1.129) < 0.01 else 'NO -- MISMATCH'})")

    print(f"\nany of the {len(BETA_GRID) * len(NREF_GRID)} grid cells clear "
          f"the 1.3x kill-switch-B bar: {'YES' if any_pass else 'NO'}")

    print(
        "\nThis sweep was run AFTER seeing r97_shared.py's pre-registered "
        "FAIL and is therefore exploratory/diagnostic, downgraded to "
        "in-sample per docs/ROUTINE.md step 4. It is not a re-opened "
        "decision: the frozen a-priori cell already failed kill switch B, "
        "and that is the round's binding result for this branch. "
        "STOP -- no strategy code, no backtest, no holdout access."
    )


if __name__ == "__main__":
    main()
