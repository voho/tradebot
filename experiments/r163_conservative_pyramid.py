#!/usr/bin/env python
"""R-163 CONSERVATIVE branch: literal Faith (2007) Turtle-style discrete
unit-adding stack on top of kelly_regime_v4's own exposure. Direction,
citations, non-duplication argument, kill switches, and the pre-registered
decision rule all live in `experiments/r163_shared.py`'s module docstring
(read there first -- this file does not repeat that reasoning and does not
edit that module, which is frozen/read-only for both branches).

THE MECHANISM, exactly: `r163_shared.build_conservative_target(df,
num_units_cap)` adds a discrete 0..num_units_cap unit stack (Faith 2007's
0.5N-add / 2N-from-last-add-stop state machine, reset at every new bullish
episode) in UNIT_SIZE=0.5 increments on top of v4's own unmodified
`frac*scale` (`v4_raw_desired`), clips to v4's own max_leverage envelope,
then applies v4's own unmodified 10% deadband -- the ONLY change from v4
anywhere in this file. Every number below comes from `r163_shared`'s frozen
primitives; this file only sequences the pre-registered steps and prints
the results.

This is the CONSERVATIVE branch only. The NOVEL branch (continuous,
two-sided tanh excursion multiplier) is a sibling agent's disjoint file;
`novel_multiplier` / `build_novel_target` / `excursion_atr_units` are
NOVEL-branch-only and are not used here.

Run with: python experiments/r163_conservative_pyramid.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r163_shared import (  # noqa: E402
    CONSERVATIVE_GRID,
    CONSERVATIVE_PRIMARY,
    CONST_CAP_R2_THRESH,
    ETH_SLICE_NAME,
    FEE_TIER,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SHARPE_NOISE_FLOOR,
    SPOT,
    UNIT_SIZE,
    V4_MAX_LEVERAGE,
    build_conservative_target,
    bullish_episode_state,
    causal_truncation_probe_series,
    clears_bar,
    compare,
    conservative_unit_path,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    pyramid_activity_diagnostic,
    r_squared,
    v4_raw_desired,
    v4_target,
)

CONFIGS_EVALUATED = 0  # incremented per compare()-cell (Step 3: 6/grid value; Step 4: per fee-tier cell)


# ================================================================== STEP 1
# Sanity / causality kill switches (A1 + causal truncation probe) plus the
# disclosed pyramid-activity diagnostic (failure mode (4)).
# ==================================================================

def step1_sanity_checks(btc_full: pd.DataFrame) -> dict:
    print("\n" + "=" * 88)
    print("STEP 1 -- Sanity / causality kill switches + activity diagnostic")
    print("=" * 88)
    # A1: num_units_cap=0 must reproduce v4_target bit-for-bit, on a subset for speed.
    sub = btc_full.iloc[:150_000]  # ~520 days: matches R-162's own convention.
    cand0 = build_conservative_target(sub, 0)
    ctrl0 = v4_target(sub)
    identical = np.array_equal(cand0, ctrl0)
    max_abs_diff = float(np.max(np.abs(cand0 - ctrl0)))
    r2_identity = r_squared(cand0, ctrl0)
    print(f"A1 identity check (num_units_cap=0 vs v4_target, n={len(sub)} bars):")
    print(f"    bit-for-bit equal : {identical}")
    print(f"    max |diff|        : {max_abs_diff:.3e}")
    print(f"    R^2               : {r2_identity:.10f}")
    assert identical, "A1 KILL SWITCH FAILED: num_units_cap=0 does not reproduce v4_target bit-for-bit"

    # Causal truncation probe at PRIMARY num_units_cap, on the FULL BTC
    # pre-holdout frame (631,008 bars): timed at ~9s in a standalone check,
    # well within budget, so no subset fallback was needed here.
    t0 = time.time()
    probe_ok = causal_truncation_probe_series(
        lambda d: build_conservative_target(d, CONSERVATIVE_PRIMARY), btc_full)
    probe_dt = time.time() - t0
    print(f"\nCausal truncation probe (num_units_cap={CONSERVATIVE_PRIMARY}), "
          f"full BTC pre-holdout frame (n={len(btc_full)} bars): PASS={probe_ok} "
          f"[{probe_dt:.1f}s]")
    assert probe_ok, "CAUSALITY PROBE FAILED at PRIMARY num_units_cap"

    # Pyramid activity/saturation diagnostic (failure mode (4)) -- reported
    # BEFORE any Sharpe number, on the full BTC pre-holdout frame, at PRIMARY.
    bullish, _entry = bullish_episode_state(btc_full)
    units_path = conservative_unit_path(btc_full, CONSERVATIVE_PRIMARY)
    diag = pyramid_activity_diagnostic(units_path, bullish)
    print(f"\nPyramid activity diagnostic (num_units_cap={CONSERVATIVE_PRIMARY}, "
          f"full BTC pre-holdout frame, n={len(btc_full)} bars):")
    print(f"    bullish_bars                    = {diag['bullish_bars']}")
    print(f"    bars_with_any_units             = {diag['bars_with_any_units']}")
    print(f"    fraction_bullish_with_units     = {diag['fraction_bullish_with_units']:.4f}")
    print(f"    bars_at_full_stack              = {diag['bars_at_full_stack']}")
    print(f"    fraction_bullish_at_full_stack  = {diag['fraction_bullish_at_full_stack']:.4f}")

    return {
        "a1_identical": identical, "a1_max_abs_diff": max_abs_diff,
        "truncation_probe_ok": probe_ok, "truncation_probe_seconds": probe_dt,
        "activity_diagnostic": diag,
    }


# ================================================================== STEP 2
# A2 kill switch: non-inertness / non-collinearity with v4's own raw
# exposure. Reproduces build_conservative_target's own pre-deadband
# computation (frozen r163_shared.py is never edited).
# ==================================================================

def conservative_raw_exposure(df: pd.DataFrame, num_units_cap: int) -> np.ndarray:
    """v4's own raw_desired plus the discrete unit stack, clipped to v4's
    own max_leverage envelope -- the exact pre-deadband computation inside
    `build_conservative_target`, reproduced here (not edited there) so
    Step 2 can compare it directly against `v4_raw_desired` unclipped."""
    raw = v4_raw_desired(df)
    if num_units_cap <= 0:
        return raw
    units = conservative_unit_path(df, num_units_cap)
    combined = raw + units * UNIT_SIZE
    return np.clip(combined, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)


def step2_a2_kill_switch(btc_full: pd.DataFrame) -> dict:
    print("\n" + "=" * 88)
    print("STEP 2 -- A2 kill switch (non-collinearity with v4's own raw exposure)")
    print("=" * 88)
    train = btc_full[(btc_full.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc_full.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC"))]
    raw_v4 = v4_raw_desired(train)
    results = {}
    for cap in CONSERVATIVE_GRID:
        cand_raw = conservative_raw_exposure(train, cap)
        r2 = r_squared(cand_raw, raw_v4)
        results[cap] = r2
        flag = "OK (below cap)" if r2 < CONST_CAP_R2_THRESH else "FAIL (>= cap -- relabeling)"
        print(f"    num_units_cap={cap:<3} R^2(candidate raw exposure vs v4 raw exposure) "
              f"= {r2:.6f}  [{flag}]")

    cand_primary = conservative_raw_exposure(train, CONSERVATIVE_PRIMARY)
    diff = cand_primary - raw_v4
    print(f"\n    (candidate_raw - v4_raw) at PRIMARY (num_units_cap={CONSERVATIVE_PRIMARY}) "
          f"on inner-train:")
    print(f"        min={diff.min():.4f} max={diff.max():.4f} mean={diff.mean():.4f} "
          f"std={diff.std():.4f} fraction_nonzero={float(np.mean(np.abs(diff) > 1e-9)):.4f}")

    a2_pass = results[CONSERVATIVE_PRIMARY] < CONST_CAP_R2_THRESH
    print(f"\n    A2 kill switch (PRIMARY num_units_cap={CONSERVATIVE_PRIMARY}): "
          f"R^2={results[CONSERVATIVE_PRIMARY]:.6f} < {CONST_CAP_R2_THRESH} -> "
          f"{'PASS (candidate is NOT a relabeling of v4)' if a2_pass else 'FAIL'}")
    if not a2_pass:
        print("    ==> A2 FAILS. Per pre-registration, this construction never meaningfully")
        print("        moves exposure relative to v4's own raw path at the primary setting --")
        print("        it is a relabeling of v4, not a tested mechanism. Continuing through the")
        print("        remaining pre-registered steps regardless (a null result is still reported")
        print("        in full), but no Sharpe/growth number below should be read as a real effect.")
    return {"r2_by_cap": results, "a2_pass": a2_pass}


# ================================================================== STEP 3
# Sweep: 4 num_units_cap values x 2 markets x 3 slices = 24 cells.
# ==================================================================

def step3_sweep(btc_full: pd.DataFrame, eth_full: pd.DataFrame) -> list[dict]:
    global CONFIGS_EVALUATED
    print("\n" + "=" * 88)
    print("STEP 3 -- Sweep: num_units_cap in CONSERVATIVE_GRID x {spot, futures_5x} x "
          "{inner_train, inner_val, eth_replication}")
    print("=" * 88)
    all_rows: list[dict] = []
    for cap in CONSERVATIVE_GRID:
        t0 = time.time()
        rows = compare(lambda df, c=cap: build_conservative_target(df, c),
                        label=f"cons_units{cap}", btc=btc_full, eth=eth_full,
                        include_eth=True)
        CONFIGS_EVALUATED += len(rows)  # 6 cells/grid value, matching R-162's own convention
        print(f"\n-- num_units_cap={cap} [{time.time() - t0:.0f}s, {len(rows)} cells] --")
        print_rows(rows)
        all_rows.extend(rows)
    return all_rows


# ================================================================== STEP 4
# Fee-tier robustness: PRIMARY num_units_cap at FEE_TIER=0.40% on both
# markets, inner-validation slice only -- 2 cells (matches the
# pre-registration's "+2 cells" fee-tier re-run of the finalist config).
# ==================================================================

def step4_fee_tier(btc_full: pd.DataFrame, cap: int = CONSERVATIVE_PRIMARY) -> list[dict]:
    global CONFIGS_EVALUATED
    print("\n" + "=" * 88)
    print(f"STEP 4 -- Fee-tier robustness: num_units_cap={cap} @ FEE_TIER={FEE_TIER:.2%}, "
          "inner_val, both markets (2 cells)")
    print("=" * 88)
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    t0 = time.time()
    rows_all = compare(lambda df, c=cap: build_conservative_target(df, c),
                        label=f"cons_units{cap}_fee{FEE_TIER:.2%}", btc=btc_full,
                        markets=fee_markets, include_eth=False)
    rows = [r for r in rows_all if r["slice"] == "inner_val"]
    CONFIGS_EVALUATED += len(rows)
    print(f"[{time.time() - t0:.0f}s, {len(rows_all)} cells computed "
          f"(inner_train + inner_val x 2 markets), {len(rows)} kept (inner_val only)]")
    print_rows(rows)
    return rows


# ================================================================== STEP 5
# Decision rule, applied exactly as written in r163_shared's module
# docstring, plus the pre-registered A3 exposure-match disclosure.
# ==================================================================

def apply_decision_rule(sweep_rows: list[dict]) -> dict:
    print("\n" + "=" * 88)
    print("STEP 5 -- Decision rule (r163_shared.py's frozen table)")
    print("=" * 88)

    inner_val_rows = [r for r in sweep_rows if r["slice"] == "inner_val"]
    eth_rows = [r for r in sweep_rows if r["slice"] == ETH_SLICE_NAME]

    def cap_of(row: dict) -> int:
        # label is "cons_units<cap>"
        return int(row["label"].rsplit("units", 1)[1])

    clear_by_market: dict[str, list[tuple[int, dict]]] = {"spot": [], "futures_5x": []}
    for row in inner_val_rows:
        cap = cap_of(row)
        if cap == 0:
            continue  # only non-zero grid values count toward CLEAR(m)
        if clears_bar(row):
            clear_by_market[row["market"]].append((cap, row))

    clear_spot = len(clear_by_market["spot"]) > 0
    clear_futures = len(clear_by_market["futures_5x"]) > 0
    print(f"CLEAR(spot)        = {clear_spot}  "
          f"(clearing num_units_cap: {[c for c, _ in clear_by_market['spot']]})")
    print(f"CLEAR(futures_5x)  = {clear_futures}  "
          f"(clearing num_units_cap: {[c for c, _ in clear_by_market['futures_5x']]})")

    # ETH sign-replication falsification test, at PRIMARY only, on whichever
    # market cleared (pre-registration's falsification test is stated for
    # the PRIMARY config specifically).
    sign_ok = True
    plateau_ok = True
    replication_notes = []

    def eth_row_for(cap: int, market: str) -> dict | None:
        for r in eth_rows:
            if r["market"] == market and cap_of(r) == cap:
                return r
        return None

    cleared_markets = [m for m, ok in (("spot", clear_spot), ("futures_5x", clear_futures)) if ok]
    for market in cleared_markets:
        primary_row = next((r for c, r in clear_by_market[market] if c == CONSERVATIVE_PRIMARY), None)
        if primary_row is None:
            replication_notes.append(
                f"market={market}: PRIMARY (num_units_cap={CONSERVATIVE_PRIMARY}) did not itself "
                f"clear on this market (a different non-zero cap did) -- falsification test is "
                f"pre-registered specifically for the PRIMARY config, so this market contributes "
                f"no sign-replication check.")
            continue
        eth_row = eth_row_for(CONSERVATIVE_PRIMARY, market)
        if eth_row is None:
            replication_notes.append(f"no ETH row for num_units_cap={CONSERVATIVE_PRIMARY}/{market}")
            continue
        btc_sign = np.sign(primary_row["d_log_growth"])
        eth_sign = np.sign(eth_row["d_log_growth"])
        note = (f"market={market} num_units_cap={CONSERVATIVE_PRIMARY}: BTC d_log_growth="
                f"{primary_row['d_log_growth']:+.4f} (sign {btc_sign:+.0f}), "
                f"ETH d_log_growth={eth_row['d_log_growth']:+.4f} (sign {eth_sign:+.0f})")
        replication_notes.append(note)
        if btc_sign != 0 and eth_sign != 0 and btc_sign != eth_sign:
            sign_ok = False

    # Plateau check: non-zero num_units_cap values that clear (on any
    # market) must share the same sign of d_log_growth on inner-validation.
    nonzero_clearing_signs = set()
    for market in ("spot", "futures_5x"):
        for cap, row in clear_by_market[market]:
            nonzero_clearing_signs.add(np.sign(row["d_log_growth"]))
    if len(nonzero_clearing_signs) > 1:
        plateau_ok = False

    gate_ok = sign_ok and plateau_ok
    print(f"\nETH sign-replication notes (PRIMARY config only, per pre-registration):")
    for n in replication_notes:
        print(f"    {n}")
    print(f"\nsign_ok (no inversion on the market that cleared, at PRIMARY) = {sign_ok}")
    print(f"plateau_ok (clearing non-zero num_units_cap values share one sign) = {plateau_ok}")
    print(f"GATE_OK = sign_ok AND plateau_ok = {gate_ok}")

    if not gate_ok:
        verdict = "REJECT"
    elif not clear_spot and not clear_futures:
        verdict = "REJECT"
    elif clear_spot and clear_futures:
        verdict = "PROMOTE"
    else:
        verdict = "PARTIAL"

    print(f"\nDecision table row matched:")
    print(f"    | GATE_OK={gate_ok!s:<5} | CLEAR(spot)={clear_spot!s:<5} "
          f"| CLEAR(futures)={clear_futures!s:<5} | Verdict = {verdict} |")

    # A3 exposure-match disclosure (NOT a kill switch -- pre-registered as
    # the EXPECTED failure mode for this branch, per r163_shared's failure
    # mode (1) / Zarattini 2026's own headline).
    print("\nA3 exposure-match disclosure (NOT expected to pass by design -- failure mode (1)):")
    primary_inner_val = [r for r in inner_val_rows if cap_of(r) == CONSERVATIVE_PRIMARY]
    a3_notes = []
    for row in primary_inner_val:
        note = (f"    market={row['market']:<11} exposure_ratio={row['exposure_ratio']:.4f} "
                f"vol_ratio={row['vol_ratio']:.4f} risk_matched={row['risk_matched']} "
                f"d_log_growth={row['d_log_growth']:+.4f} d_sharpe={row['d_sharpe']:+.3f} "
                f"d_dd={row['d_dd']:+.2f}")
        print(note)
        a3_notes.append(row)
    predicted_pattern_held = all(
        (not r["risk_matched"]) and (r["exposure_ratio"] > 1.0) for r in a3_notes
    ) if a3_notes else False
    print(f"    Predicted pattern (elevated, unmatched exposure/vol; risk_matched=False) "
          f"held on ALL inner-val rows at PRIMARY: {predicted_pattern_held}")

    return {
        "clear_spot": clear_spot, "clear_futures": clear_futures,
        "sign_ok": sign_ok, "plateau_ok": plateau_ok, "gate_ok": gate_ok,
        "verdict": verdict, "a3_predicted_pattern_held": predicted_pattern_held,
        "a3_rows": a3_notes,
    }


# ================================================================== main

if __name__ == "__main__":
    t_start = time.time()
    print("R-163 CONSERVATIVE branch (Faith 2007 discrete unit-stack pyramid) -- full run")
    print(f"CONSERVATIVE_GRID={CONSERVATIVE_GRID}  CONSERVATIVE_PRIMARY={CONSERVATIVE_PRIMARY}  "
          f"UNIT_SIZE={UNIT_SIZE}  FEE_TIER={FEE_TIER:.2%}  "
          f"SHARPE_NOISE_FLOOR={SHARPE_NOISE_FLOOR}  CONST_CAP_R2_THRESH={CONST_CAP_R2_THRESH}")

    btc_full = load_btc()
    eth_full = load_eth()
    print(f"BTC pre-holdout bars: {len(btc_full)} ({btc_full.index[0]} .. {btc_full.index[-1]})")
    print(f"ETH replication bars: {len(eth_full)} ({eth_full.index[0]} .. {eth_full.index[-1]})")
    assert pd.Timestamp(btc_full.index[-1]) < pd.Timestamp(OOS_START, tz="UTC"), (
        "btc_full reaches OOS_START -- holdout must not be touched by this branch")

    step1 = step1_sanity_checks(btc_full)
    a2 = step2_a2_kill_switch(btc_full)
    sweep_rows = step3_sweep(btc_full, eth_full)
    fee_rows = step4_fee_tier(btc_full, cap=CONSERVATIVE_PRIMARY)

    print("\n" + "=" * 88)
    print("FULL RESULT TABLE (sweep + fee-tier robustness)")
    print("=" * 88)
    print_rows(sweep_rows + fee_rows)

    decision = apply_decision_rule(sweep_rows)

    print("\n" + "=" * 88)
    print("STEP 6 -- Holdout")
    print("=" * 88)
    print("NOT READ. Per instructions, this branch stops after Step 5 regardless of verdict --")
    print(f"the holdout (start={OOS_START}) is left for a human/operator to authorize separately,")
    print(f"even though this run's verdict is {decision['verdict']!r}.")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"Configs evaluated: {CONFIGS_EVALUATED} (expected 26: 4 num_units_cap values x 2 "
          f"markets x 3 slices = 24, + 2 fee-tier cells)")
    print(f"A2 kill switch (PRIMARY num_units_cap={CONSERVATIVE_PRIMARY}): "
          f"R^2={a2['r2_by_cap'][CONSERVATIVE_PRIMARY]:.6f} vs cap {CONST_CAP_R2_THRESH} -> "
          f"{'PASS' if a2['a2_pass'] else 'FAIL -- relabeling of v4, not a tested mechanism'}")
    print(f"Decision rule verdict (inner-validation): {decision['verdict']}")
    print(f"  GATE_OK={decision['gate_ok']}  CLEAR(spot)={decision['clear_spot']}  "
          f"CLEAR(futures_5x)={decision['clear_futures']}")
    print(f"A3 predicted unmatched-exposure pattern held at PRIMARY: "
          f"{decision['a3_predicted_pattern_held']}")
    if not a2["a2_pass"]:
        print("NOTE: A2 failed at PRIMARY -- per pre-registration, any apparent Sharpe/growth")
        print("      effect above is not attributable to this mechanism moving exposure; it is")
        print("      indistinguishable from v4 itself at this setting. This does not change the")
        print("      mechanical decision-rule verdict above, but it is the honest headline.")
    print("Holdout read: NO (not authorized at this stage)")
    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")
