"""R-148 NOVEL branch: per-species fractional-Kelly sizing for
`replicator_book`, run against the frozen `experiments/r148_shared.py`
pre-registration.

This file is intentionally thin: `novel_target()`, `per_species_kelly_
fraction()`, `compare()`, `print_rows()`, `r_squared()` and every constant
are already defined in the frozen, read-only `r148_shared.py` (written by
the operator before this branch ran). This script only SEQUENCES the
pre-registered checks against that machinery and reports results -- it adds
no new sizing logic of its own.

Order of operations, matching the pre-registration's PROMOTION BAR:
  0. A2 kill switch (R^2 of novel_target vs replicator_target, inner-train).
     If this trips, STOP -- do not spend time on B1-B5.
  1. compare(novel_target, label="novel_kelly") over inner_train, inner_val,
     eth_replication, both markets -- full table via print_rows().
  2. B3 plateau: kelly_cap in {1.0, 1.5, 2.0}, inner-validation, both
     markets -- sign stability of d_sharpe / boot_d_loggrowth.
  3. B5: 0.40% spot fee tier re-run, inner-validation, BTC only.
  4. Turnover (num_trades) honestly reported for both branches, both
     markets, inner-validation -- named failure risk for this branch.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.r148_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    SPOT,
    compare,
    fee_at,
    load_btc,
    novel_target,
    print_rows,
    replicator_target,
    r_squared,
)


def step0_kill_switch() -> float:
    """A2: R^2 of novel_target vs replicator_target on inner-train only."""
    import pandas as pd

    btc = load_btc()
    inner_train = btc[btc.index < pd.Timestamp(INNER_TRAIN_END, tz="UTC")]
    return r_squared(novel_target(inner_train), replicator_target(inner_train))


def b3_plateau_sweep() -> list[dict]:
    """kelly_cap in {1.0, 1.5, 2.0}, inner-validation, both markets."""
    rows = []
    for kelly_cap in (1.0, 1.5, 2.0):
        candidate = functools.partial(novel_target, kelly_cap=kelly_cap)
        r = compare(
            candidate, label=f"novel_kcap{kelly_cap:g}",
            markets=(SPOT, FUTURES), include_eth=False,
        )
        # keep only inner_val rows for the plateau report
        rows.extend(row for row in r if row["slice"] == "inner_val")
    return rows


def b5_fee_tier_check() -> list[dict]:
    """0.40% spot fee tier, inner-validation only, BTC only (fast)."""
    return compare(
        novel_target, label="novel_kelly_fee40bp",
        markets=(fee_at(SPOT, 0.004),), include_eth=False,
    )


def main() -> int:
    print("=" * 100)
    print("R-148 NOVEL: per-species fractional-Kelly sizing (replicator_book)")
    print("=" * 100)

    # ---------------------------------------------------------- Step-0 (A2)
    print("\n--- Step-0 A2 kill switch: R^2(novel_target, replicator_target), inner-train ---")
    r2 = step0_kill_switch()
    print(f"R^2 = {r2:.6f}")
    if r2 > 0.98:
        print("\n*** A2 KILL SWITCH TRIPPED (R^2 > 0.98) -- STOPPING. VERDICT: NEGATIVE ***")
        return 1
    print("A2 kill switch: PASS (R^2 <= 0.98) -- proceeding.")

    # -------------------------------------------------------------- main
    print("\n--- Main comparison: compare(novel_target, label='novel_kelly') ---")
    print("    (use_notional left at default False -- novel_target's output is")
    print("     bounded to [-1, 1] by construction, same execution convention")
    print("     as the control; see TargetStrategy's own docstring.)")
    rows = compare(novel_target, label="novel_kelly")
    print()
    print_rows(rows)

    # -------------------------------------------------------------- B3
    print("\n--- B3 plateau: kelly_cap sweep {1.0, 1.5, 2.0}, inner-validation, both markets ---")
    b3_rows = b3_plateau_sweep()
    print_rows(b3_rows)
    print("\n  sign-stability summary (d_sharpe / boot_d_loggrowth), inner_val:")
    for r in b3_rows:
        print(f"    kelly_cap in label={r['label']:22s} market={r['market']:11s} "
              f"d_sharpe={r['d_sharpe']:+.4f} boot_d_loggrowth={r['boot_d_loggrowth']:+.4f} "
              f"excludes_zero={r['excludes_zero']}")

    # -------------------------------------------------------------- B5
    print("\n--- B5: 0.40% spot fee tier, inner-validation + inner-train, BTC only ---")
    b5_rows = b5_fee_tier_check()
    print_rows(b5_rows)

    # ---------------------------------------------------------- turnover
    print("\n--- Turnover (num_trades), candidate vs control, inner-validation, both markets ---")
    inner_val_rows = [r for r in rows if r["slice"] == "inner_val"]
    for r in inner_val_rows:
        print(f"  market={r['market']:11s} cand_trades={r['cand_trades']:>6d} "
              f"ctrl_trades={r['ctrl_trades']:>6d} "
              f"ratio={ (r['cand_trades']/r['ctrl_trades']) if r['ctrl_trades'] else float('nan'):.2f}x")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
