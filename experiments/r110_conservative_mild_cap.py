"""R-110 CONSERVATIVE -- does R-107's own pure-ERC construction (alpha=1,
this round's notation) survive the FULL decisive battery at a milder rank
cap, k=2 or k=3, rather than the k=6 cell R-107 selected purely by W_VAL
growth across its whole grid?

This is R-107's own unfiled next step, read verbatim (see r110_shared.py's
docstring for the full quote and rationale). It is NOT a new mechanism: it
reuses R-107's exact `build_targets` construction (verified an identity to
r107_novel_risk_parity.build_targets at alpha=1 by r110_shared's own
identity_checks) and simply reads k=2 and k=3 each through the SAME full
pre-registered battery R-107 ran only on its k=6 winner.

Run as:
    python experiments/r110_conservative_mild_cap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import r110_shared as sh  # noqa: E402

ALPHA = 1.0  # pure ERC -- R-107's own construction, unmodified. Blend
             # (alpha in (0,1)) is the novel branch's question, not this one's.


def sweep_k(frames, k):
    """lambda sweep at fixed k, alpha=1.0, on W_TRAIN and W_VAL, U8."""
    print(f"\n== lambda sweep at k={k}, alpha={ALPHA:.2f} "
          f"(W_TRAIN then W_VAL, U8, spot 0.10% vs VOLMATCH_HOLD) ==")
    rows = []
    for wname, window in (("W_TRAIN", sh.W_TRAIN), ("W_VAL", sh.W_VAL)):
        for lam in sh.LAMBDA_GRID:
            aligned, targets, warm_ok = sh.build_cell(
                frames, sh.UNIVERSE_8, window, k, lam, ALPHA)
            if not warm_ok:
                raise RuntimeError(f"{wname} k={k} lam={lam}: first bar not warm")
            row = sh.evaluate(targets, aligned, sh.UNIVERSE_8, wname, "U8",
                              f"cons_k{k}", {"k": k, "lambda": lam, "alpha": ALPHA})
            row["first_bar_warm"] = warm_ok
            rows.append(row)
            print(f"  {wname} k={k} lam={lam:.2f}  net_growth "
                  f"{row['net_growth_diff']:+.4f} [{row['net_growth_lo']:+.4f},"
                  f" {row['net_growth_hi']:+.4f}]  net_dd {row['net_dd_diff']:+.2f}"
                  f"  mean_notional={row['mean_notional']:.4f}"
                  f"  gross_growth {row['gross_growth_diff']:+.4f}")
    sh.write_csv(sh.OUT_DIR / f"cons_sweep_k{k}.csv", rows)
    return rows


def select_lambda(sweep_rows, k):
    """Select lambda maximizing net_growth_diff on W_VAL, at fixed k --
    mirrors R-107's own select_frozen but restricted to one k at a time
    (since this branch's question is per-k, not a joint (k,lambda) pick
    across the whole mild-cap grid)."""
    val_rows = [r for r in sweep_rows if r["window"] == "W_VAL"]
    val_rows.sort(key=lambda r: (-r["net_growth_diff"], r["net_dd_diff"]))
    winner = val_rows[0]
    lam = float(winner["p_lambda"])
    print(f"\n  SELECTED at k={k}: lambda={lam:.2f}  "
          f"net_growth={winner['net_growth_diff']:+.4f}  "
          f"net_dd={winner['net_dd_diff']:+.2f}")
    print(f"  full W_VAL lambda ranking at k={k} (plateau check):")
    for r in val_rows:
        print(f"    lam={float(r['p_lambda']):.2f}  "
              f"net_growth={r['net_growth_diff']:+.4f}  net_dd={r['net_dd_diff']:+.2f}")
    train_rows = {float(r["p_lambda"]): r for r in sweep_rows if r["window"] == "W_TRAIN"}
    tr = train_rows.get(lam)
    if tr is not None:
        print(f"  same cell on W_TRAIN: net_growth={tr['net_growth_diff']:+.4f}  "
              f"(rank transfer check)")
    return lam


def run_for_k(frames, k):
    sweep_rows = sweep_k(frames, k)
    lam = select_lambda(sweep_rows, k)

    fal = sh.falsification_test(frames, k, lam, ALPHA, tag=f"cons_k{k}")
    result = {"k": k, "lambda": lam, "alpha": ALPHA, "sweep": sweep_rows,
              "falsification": fal, "battery": None}

    if not fal["passed"]:
        print(f"\n== [k={k}] FALSIFICATION TEST FAILED on inner-train. STOP. ==")
        print(f"   No W_FULL6 read, no W_HOLD read, for k={k}. Reporting NEGATIVE.")
        return result

    print(f"\n== [k={k}] falsification test PASSED. Proceeding to the decisive "
          f"battery on W_FULL6/W_VAL. ==")
    battery = sh.decisive_battery(frames, k, lam, ALPHA, tag=f"cons_k{k}")
    result["battery"] = battery
    return result


def main():
    print("== R-110 CONSERVATIVE: mild rank cap (k=2, k=3), pure ERC "
          "(alpha=1.0), full decisive battery per k ==\n")

    frames = sh.load_universe(sh.UNIVERSE_8)
    start_configs = sh.config_count()

    print("== pre-flight sanity gate ==")
    ok_id = sh.identity_checks(frames)
    warm = sh.align_frames({t: frames[t] for t in sh.UNIVERSE_8}, sh.warm_window(sh.W_TRAIN))
    ok_probe = sh.perturbation_probe(frames)
    print(f"  perturbation_probe (tail x10, early rows unchanged, "
          f"k=3 lam=0.5 alpha=0.5): {ok_probe}")
    ok_causal = sh.check_causality(lambda a: sh.build_targets(a, 3, 0.5, 1.0), warm)
    print(f"  check_causality(k=3, lambda=0.5, alpha=1.0): {ok_causal}")

    gate = ok_id and ok_probe and ok_causal
    print(f"\n  PRE-FLIGHT GATE: {'PASSED' if gate else 'FAILED'}")
    if not gate:
        print("\n== PRE-FLIGHT GATE FAILED. The shared module is not behaving "
              "as its own smoke tests claimed. STOPPING -- reporting this to "
              "the operator instead of proceeding on unverified infrastructure. ==")
        print(f"\nconfig_count() this run = {sh.config_count() - start_configs}  "
              f"(cumulative process total = {sh.config_count()})")
        return

    results = {}
    for k in sh.K_GRID_MILD:
        results[k] = run_for_k(frames, k)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY -- R-110 CONSERVATIVE (mild rank cap, pure ERC)")
    print("=" * 78)
    any_further_work = False
    for k in sh.K_GRID_MILD:
        r = results[k]
        fal = r["falsification"]
        print(f"\n  k={k}  selected lambda={r['lambda']:.2f}  alpha={ALPHA:.2f}")
        print(f"    falsification: mean_DR2_blend={fal['mean_dr2_blend']:.4f}  "
              f"mean_DR2_eq={fal['mean_dr2_eq']:.4f}  "
              f"PASSED={fal['passed']}")
        if r["battery"] is None:
            print(f"    -> STOPPED at falsification. No decisive battery run. "
                  f"VERDICT: NEGATIVE (k={k})")
            continue
        b = r["battery"]
        d1r, d3r = b["d1_row"], b["d3_row"]
        print(f"    D1 (W_FULL6/U6 growth vs VOLMATCH_HOLD): {b['d1']}  "
              f"net_growth={d1r['net_growth_diff']:+.4f} "
              f"[{d1r['net_growth_lo']:+.4f}, {d1r['net_growth_hi']:+.4f}]")
        print(f"    D2 (drawdown vs VOLMATCH_HOLD): {b['d2']}  "
              f"net_dd={d1r['net_dd_diff']:+.2f} "
              f"[{d1r['net_dd_lo']:+.2f}, {d1r['net_dd_hi']:+.2f}]")
        print(f"    D3 (W_VAL/U8 directional check): {b['d3']}  "
              f"net_growth={d3r['net_growth_diff']:+.4f}  "
              f"net_dd={d3r['net_dd_diff']:+.2f}")
        print(f"    D4 (0.40% fee tier vs EW_HOLD): {b['d4']}")
        print(f"    D5 (gross signal retention, bar {sh.D5_BAR_R68:+.4f}): "
              f"{b['d5']}  gross_growth={d1r['gross_growth_diff']:+.4f}")
        print(f"    scramble (real > seed-p90): {b['scramble']}")
        print(f"    further_work (R-65 four-clause bar): {b['further_work']}")
        if b["further_work"]:
            any_further_work = True
            print(f"    *** further_work=True at k={k} -- a holdout (W_HOLD) "
                  f"read is now authorized per the pre-registered decision "
                  f"rule. This agent is NOT performing that read. The "
                  f"operator should perform and record it. ***")
        else:
            print(f"    -> VERDICT: NEGATIVE (k={k}), further_work bar not cleared.")

    print("\n" + "=" * 78)
    print(f"ANY k CLEARS further_work: {any_further_work}")
    print("=" * 78)

    print(f"\nconfig_count() this run = {sh.config_count() - start_configs}  "
          f"(cumulative process total = {sh.config_count()})")


if __name__ == "__main__":
    main()
