"""R-110 NOVEL -- continuous convex blend between R-63's equal-weight split
and R-107's pure equal-risk-contribution (ERC) risk-parity split, across
R-63's own top-k eligible-asset set.

    w = (1 - alpha) * w_equal_weight + alpha * w_risk_parity,  alpha in [0, 1]

See `r110_shared.py` for the full pre-registration: mechanism, literature
(DeMiguel, Garlappi & Uppal 2009's shrinkage-to-1/N robustness check;
Baltas 2015 / Bruder & Roncalli 2012 for the ERC half), the three named
failure modes (F1/F2/F3), and the machinery this file calls but does not
redefine.

One sentence: this file changes only the SPLIT of already-decided total
notional across R-63's own eligible set, continuously between 1/m
(alpha=0) and causal risk-parity (alpha=1), rather than R-107's own
all-or-nothing substitution.

Run as:
    python experiments/r110_novel_blend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.r110_shared import (  # noqa: E402
    ALPHA_GRID,
    K_GRID_MILD,
    UNIVERSE_8,
    W_TRAIN,
    W_VAL,
    build_cell,
    config_count,
    decisive_battery,
    evaluate,
    falsification_test,
    identity_checks,
    load_universe,
    perturbation_probe,
    write_csv,
)
from experiments.r110_shared import OUT_DIR  # noqa: E402

LAMBDA_FROZEN = 1.0  # R-107's own selected shrinkage intensity; held fixed
                      # here so alpha is the ONE new free parameter under test.


def cmd_sweep(frames):
    print("== (k, alpha) grid: k in {2, 3} x alpha in "
          f"{ALPHA_GRID}, lambda={LAMBDA_FROZEN:.2f} (frozen) ==")
    print("   W_TRAIN then W_VAL, U8, spot 0.10% vs VOLMATCH_HOLD\n")
    rows = []
    for wname, window in (("W_TRAIN", W_TRAIN), ("W_VAL", W_VAL)):
        for k in K_GRID_MILD:
            for alpha in ALPHA_GRID:
                aligned, targets, warm_ok = build_cell(
                    frames, UNIVERSE_8, window, k, LAMBDA_FROZEN, alpha)
                if not warm_ok:
                    raise RuntimeError(
                        f"{wname} k={k} alpha={alpha}: first bar not warm")
                row = evaluate(targets, aligned, UNIVERSE_8, wname, "U8",
                               "blend", {"k": k, "lambda": LAMBDA_FROZEN,
                                         "alpha": alpha})
                row["first_bar_warm"] = warm_ok
                rows.append(row)
                print(f"  {wname} k={k} alpha={alpha:.2f}  net_growth "
                      f"{row['net_growth_diff']:+.4f} [{row['net_growth_lo']:+.4f},"
                      f" {row['net_growth_hi']:+.4f}]  net_dd {row['net_dd_diff']:+.2f}"
                      f"  mean_notional={row['mean_notional']:.4f}"
                      f"  gross_growth {row['gross_growth_diff']:+.4f}")
    write_csv(OUT_DIR / "blend_sweep.csv", rows)
    return rows


def check_f3(sweep_rows):
    """F3, pre-registered in r110_shared's docstring: for each k, does ANY
    interior alpha (0.25, 0.5, 0.75) beat BOTH its own k's endpoints
    (alpha=0 AND alpha=1) on W_VAL net_growth_diff? Report per k."""
    print("\n== F3 check: does any interior alpha beat both its own k's "
          "endpoints on W_VAL net_growth_diff? ==")
    val_rows = [r for r in sweep_rows if r["window"] == "W_VAL"]
    result = {}
    for k in K_GRID_MILD:
        by_alpha = {float(r["p_alpha"]): r["net_growth_diff"]
                    for r in val_rows if int(r["p_k"]) == k}
        g0 = by_alpha[0.0]
        g1 = by_alpha[1.0]
        lo, hi = min(g0, g1), max(g0, g1)
        interiors = {a: g for a, g in by_alpha.items() if a not in (0.0, 1.0)}
        beats_both = {a: (g > hi) for a, g in interiors.items()}
        any_beats_both = any(beats_both.values())
        result[k] = any_beats_both
        print(f"  k={k}: alpha=0.00 net_growth={g0:+.4f}  "
              f"alpha=1.00 net_growth={g1:+.4f}  "
              f"(endpoint envelope [{lo:+.4f}, {hi:+.4f}])")
        for a in sorted(interiors):
            print(f"    alpha={a:.2f} net_growth={interiors[a]:+.4f}  "
                  f"beats_both_endpoints={beats_both[a]}")
        print(f"  F3[k={k}] = ANY interior alpha beats both endpoints: "
              f"{any_beats_both}")
    return result


def select_frozen(sweep_rows):
    """Select (k, alpha) on W_VAL's net growth_diff vs VOLMATCH_HOLD,
    tie-broken by net_dd_diff (more negative = better) -- R-63/R-107's own
    criterion, applied here to the (k, alpha) plane with lambda held fixed."""
    val_rows = [r for r in sweep_rows if r["window"] == "W_VAL"]
    val_rows.sort(key=lambda r: (-r["net_growth_diff"], r["net_dd_diff"]))
    winner = val_rows[0]
    k = int(winner["p_k"])
    alpha = float(winner["p_alpha"])
    print(f"\n  SELECTED on W_VAL: k={k} alpha={alpha:.2f}  "
          f"net_growth={winner['net_growth_diff']:+.4f}  "
          f"net_dd={winner['net_dd_diff']:+.2f}")
    print("  full 10-cell W_VAL ranking (plateau check):")
    for r in val_rows:
        print(f"    k={int(r['p_k'])} alpha={float(r['p_alpha']):.2f}  "
              f"net_growth={r['net_growth_diff']:+.4f}  net_dd={r['net_dd_diff']:+.2f}")
    train_rows = {(int(r["p_k"]), float(r["p_alpha"])): r
                  for r in sweep_rows if r["window"] == "W_TRAIN"}
    tr = train_rows.get((k, alpha))
    if tr is not None:
        print(f"  same cell on W_TRAIN: net_growth={tr['net_growth_diff']:+.4f}  "
              f"(rank transfer check)")
    return k, alpha


def main():
    # [bug fix, disclosed] `write_csv` (imported from r107_novel_risk_parity)
    # mkdir's ITS OWN module's OUT_DIR (reports/r107_risk_parity), not the
    # r110_shared OUT_DIR (reports/r110_blend) that paths passed to it here
    # live under -- so the very first write_csv call crashes with
    # FileNotFoundError unless reports/r110_blend already exists. Ensure it
    # exists here; no shared file touched.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_universe(UNIVERSE_8)
    start_configs = config_count()

    print("== sanity gate: identity checks + perturbation probe ==")
    ident_ok = identity_checks(frames)
    if not ident_ok:
        print("\n== IDENTITY CHECKS FAILED. r110_shared.py is broken. STOP. ==")
        return
    pert_ok = perturbation_probe(frames)
    print(f"  perturbation_probe (k=3, lambda=0.5, alpha=0.5, tail x10): {pert_ok}")
    if not pert_ok:
        print("\n== PERTURBATION PROBE FAILED. r110_shared.py is broken. STOP. ==")
        return
    print("  sanity gate PASSED.\n")

    sweep_rows = cmd_sweep(frames)
    f3 = check_f3(sweep_rows)
    k, alpha = select_frozen(sweep_rows)

    print()
    fal = falsification_test(frames, k, LAMBDA_FROZEN, alpha, tag="novel_blend")
    if not fal["passed"]:
        print("\n== FALSIFICATION TEST FAILED on inner-train. STOP. ==")
        print("   No W_FULL6 read, no W_HOLD read. Reporting the negative.")
        print(f"\nF3 = {f3}")
        print(f"config_count() this run = {config_count() - start_configs}  "
              f"(cumulative process total = {config_count()})")
        return

    print("\n== falsification test PASSED. Proceeding to the decisive "
          "battery on W_FULL6/W_VAL. ==\n")
    battery = decisive_battery(frames, k, LAMBDA_FROZEN, alpha, tag="novel_blend")
    fw = battery["further_work"]
    print(f"\n== further_work(d1={battery['d1']}, d2={battery['d2']}, "
          f"d3={battery['d3']}, d5={battery['d5']}, "
          f"scramble={battery['scramble']}) = {fw} ==")
    if fw:
        print("  -> STOP. Report to the operator; the W_HOLD read is theirs "
              "to authorize (+1 holdout consultation).")
    else:
        print("  -> DONE. W_HOLD is NOT read. Reporting the negative.")

    print(f"\nF3 = {f3}")
    print(f"\nconfig_count() this run = {config_count() - start_configs}  "
          f"(cumulative process total = {config_count()})")


if __name__ == "__main__":
    main()
