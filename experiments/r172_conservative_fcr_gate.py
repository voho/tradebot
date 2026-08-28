#!/usr/bin/env python
"""R-172 CONSERVATIVE branch: literal binary FCR gate on kelly_regime_v4's
vote. Direction, citations, the non-duplication argument, the shared FCR
primitive's provenance, and the pre-registered promotion rule/kill switches
all live in `experiments/r172_direction.md` (S5 "Conservative", S6, S7) and
`experiments/r172_shared.py`'s module docstring -- read those first. This
file does not repeat that reasoning and does not edit either frozen
document.

THE MECHANISM, exactly (design doc S5 "Conservative"): test, at the
FCR-corrected level, the null "the currently-active pattern's true mean
forward edge is <= 0"; trust the vote fully when the null is rejected on the
positive side (`lcb_p(t) > 0`), force the position flat otherwise.

    gated_frac(t) = frac(t)   if lcb_broadcast(t) > 0 OR n_used_broadcast(t) < MIN_N
                  = 0.0       otherwise
    desired = gated_frac * v4_scale(df)
    target  = apply_deadband(desired)      # v4's own unmodified 10% deadband

`frac` is v4's own unmodified `v4_vote_frac` -- byte-identical to v4 wherever
the gate does not bind, per R-62's isolation discipline (this file never
touches `scale` or the deadband). `lcb`/`n_used` come from
`experiments.r172_shared.fcr_lower_bounds`, the frozen, causally-embargoed
primitive -- not reimplemented here.

Q_FCR/HORIZON_DAYS/MIN_N are NOT swept for the PRIMARY result (the module's
own defaults: 0.10 / 5 / 30). The only sweep in this file is the
pre-registered PLATEAU check (design doc S7 clause 3): a 3-point bracket of
`q` (0.05, 0.10, 0.20) at fixed `horizon_days`/`min_n` -- a structural
robustness check on the FCR correction level, not a performance search; all
3 points are reported, none cherry-picked.

Run: `uv run python3 experiments/r172_conservative_fcr_gate.py` (repo root).
This file never reads a bar at or after OOS_START (2023-01-01, enforced by
`load_btc()`/`load_eth()`/`run_slice()`/`compare()` in the shared chain) --
this branch's job stops at inner-validation/ETH/fee-tier; the holdout is the
operator's call, not this file's.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r172_shared as shared  # noqa: E402

# ================================================================== (1)
# The candidate factor and full target builder, parameterised by `q` only
# (the plateau-bracket knob). horizon_days/min_n stay at the module's own
# PRIMARY defaults throughout -- not part of this branch's sweep.
# ==================================================================


def gate_arrays(df: pd.DataFrame, q: float = shared.Q_FCR):
    """Return (frac, gated_frac, trust) -- all length len(df), bar frequency.

    `trust(t)` is True ("no evidence yet" OR the FCR-corrected lower bound
    is positive) -- v4's own vote is used unmodified wherever `trust` is
    True, forced to 0.0 otherwise. NaN `lcb` (no bound reported, either
    because the bucket hasn't reached MIN_N yet or because no daily value
    exists yet at this bar) evaluates False in `lcb_bc > 0.0` by IEEE
    comparison semantics, so a NaN lcb never trusts the vote on its own --
    it only trusts via the explicit `n_used_bc < MIN_N` "no evidence yet"
    clause, which is exactly the intended default per r172_direction.md S3.1.

    WIRING NOTE (this file's own responsibility, not `r172_shared.py`'s):
    `fcr_lower_bounds`'s `idx` labels each day D's `(lcb, n_used)` by D
    itself, but `daily_pattern` (which `fcr_lower_bounds` uses to pick the
    active bucket) is built from day D's OWN LAST bar -- i.e. day D's
    `(lcb, n_used)` is only fully KNOWN at day D's close, not at its open.
    Broadcasting it onto day D's own bars unshifted (as a naive
    `broadcast_daily(lcb, idx, ...)` would) is a one-day lookahead that a
    smooth/synthetic price path rarely exposes but real BTC data does
    (caught by `causal_truncation_probe_series` below at cuts=(0.5,0.7,0.9)
    on real BTC inner-train data -- a mid-day cut lets the probe's tail
    perturbation flip a boundary day's own end-of-day pattern, which then
    visibly changes bars *earlier that same day*, before the perturbed
    region). The fix used everywhere else in this codebase for an
    end-of-period value (e.g. `v4_symmetric_vol`'s own `.shift(1)`) is to
    lag the daily index by one full calendar day before broadcasting, so
    day D's bound is applied starting day D+1, once it is actually known.
    """
    idx, lcb, _ucb, n_used = shared.fcr_lower_bounds(df, q=q)
    idx_lagged = idx + pd.Timedelta(days=1)
    lcb_bc = shared.broadcast_daily(np.asarray(lcb, dtype=float), idx_lagged, df.index,
                                     fill_value=np.nan)
    n_used_bc = shared.broadcast_daily(n_used.astype(float), idx_lagged, df.index,
                                        fill_value=0.0)
    frac = shared.v4_vote_frac(df).to_numpy()
    trust = (lcb_bc > 0.0) | (n_used_bc < shared.MIN_N)
    gated_frac = np.where(trust, frac, 0.0)
    return frac, gated_frac, trust


def make_build(q: float = shared.Q_FCR):
    """Candidate `build_target(df) -> np.ndarray`, matching compare()'s
    expected `candidate_build` signature. Pure function of `df` alone --
    no live-equity dependency, matching `TargetStrategy`'s contract."""

    def build(df: pd.DataFrame) -> np.ndarray:
        _frac, gated_frac, _trust = gate_arrays(df, q=q)
        desired = gated_frac * shared.v4_scale(df)
        return shared.apply_deadband(desired)

    build.__name__ = f"r172_cons_fcr_gate_q{q:.4g}"
    return build


BUILD_PRIMARY = make_build(shared.Q_FCR)
LABEL_PRIMARY = "r172_conservative_fcr_gate"

# Plateau bracket (design doc S7 clause 3): q in {0.05, 0.10, 0.20}.
Q_BRACKET = (0.05, shared.Q_FCR, 0.20)
assert Q_BRACKET[1] == shared.Q_FCR

FEE_TIER_040 = shared.FEE_TIER  # 0.40% taker, already the module's own constant
assert abs(FEE_TIER_040 - 0.0040) < 1e-12


# ================================================================== (2)
# KS-C sample-size disclosure (design doc S6, mandatory report, not a kill
# switch): resolved bucket size n_p for each of the 8 patterns, reached by
# INNER_TRAIN_END and by INNER_VAL_END. Reports the bucket size as of each
# pattern's own last occurrence inside the cumulative window (the causal
# n_used value in force immediately before that occurrence) -- a direct
# read of `fcr_lower_bounds`'s own `n_used` output, not a reimplementation.
# ==================================================================

def ks_c_disclosure(df_upto: pd.DataFrame, label: str) -> None:
    idx, _lcb, _ucb, n_used = shared.fcr_lower_bounds(df_upto)
    pattern = shared.daily_pattern(df_upto).reindex(idx).to_numpy()
    print(f"    KS-C ({label}, window end {df_upto.index[-1]}):")
    for p in range(8):
        mask = pattern == p
        occurrences = int(mask.sum())
        n_p = int(n_used[mask][-1]) if occurrences else 0
        bits = f"{p:03b}"
        print(f"        pattern {p} ({bits}, v20={bits[2]},v40={bits[1]},v80={bits[0]}): "
              f"occurrences={occurrences:5d}  n_used as of last occurrence={n_p:5d}"
              f"{'  (< MIN_N, no evidence)' if n_p < shared.MIN_N else '  (>= MIN_N, bound reported)'}")


# ================================================================== (3)
# Reporting helpers over compare()'s row dicts.
# ==================================================================

def cell(rows: list[dict], label: str, slice_name: str, market: str) -> dict | None:
    for r in rows:
        if r["label"] == label and r["slice"] == slice_name and r["market"] == market:
            return r
    return None


def hr(title: str = "") -> None:
    print("\n" + "=" * 92)
    if title:
        print(title)
        print("=" * 92)


def main() -> None:
    t0 = time.time()
    configs_evaluated = 0
    backtest_cells = 0
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-172 CONSERVATIVE -- literal binary FCR gate on kelly_regime_v4's "
       "vote. See r172_direction.md S5 /\nr172_shared.py for the full mechanism.")

    # ========================================================== STEP 0
    hr("STEP 0 -- load data (pre-holdout truncated), causal truncation probe "
       "on the FULL strategy build (PRIMARY q)")
    btc = shared.load_btc()
    eth = shared.load_eth()
    max_ts_seen += [btc.index.max(), eth.index.max()]
    shared.assert_no_holdout(btc, "BTC full")
    shared.assert_no_holdout(eth, "ETH full")
    print(f"BTC (truncated < {shared.OOS_START}): {len(btc):,} bars, {btc.index[0]} -> {btc.index[-1]}")
    print(f"ETH (truncated < {shared.OOS_START}): {len(eth):,} bars, {eth.index[0]} -> {eth.index[-1]}")

    btc_train = btc.loc[shared.INNER_TRAIN_START:shared.INNER_TRAIN_END]
    shared.assert_no_holdout(btc_train, "BTC inner-train")

    causal_ok = True
    try:
        ok = shared.causal_truncation_probe_series(BUILD_PRIMARY, btc_train, cuts=(0.5, 0.7, 0.9))
        print(f"\ncausal_truncation_probe_series(build_target, BTC inner-train, "
              f"cuts=(0.5,0.7,0.9)): {'PASS' if ok else 'FAIL'}")
    except AssertionError as e:
        ok = False
        print(f"\ncausal_truncation_probe_series: FAIL ({e})")
    causal_ok = causal_ok and ok

    if not causal_ok:
        hr("VERDICT")
        print("    VERDICT: NEGATIVE (causal truncation probe FAILED end-to-end "
              "through the strategy wiring -- a lookahead bug -- stopping before "
              "any promotion-bar evaluation.)")
        print(f"    Configurations evaluated: {configs_evaluated} (stopped before any comparison).")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    # ========================================================== STEP 1
    hr("STEP 1 -- Kill switches KS-A (non-triviality) / KS-B (relabeling), "
       "BTC inner-train, PRIMARY q")
    frac_train, gated_frac_train, trust_train = gate_arrays(btc_train, q=shared.Q_FCR)
    differs = gated_frac_train != frac_train
    binding_frac = shared.binding_fraction(differs)
    ks_a_ok = binding_frac >= shared.GATE_MIN_BINDING_FRACTION
    print(f"\nKS-A (non-triviality): binding_fraction = {binding_frac:.6f}  "
          f"(threshold: >= {shared.GATE_MIN_BINDING_FRACTION})  -> {'PASS' if ks_a_ok else 'FAIL'}")
    print(f"    (fraction of BTC inner-train bars where the gate forced flat: "
          f"gated_frac != frac)")

    cand_target_train = BUILD_PRIMARY(btc_train)
    v4_raw_train = shared.v4_raw_desired(btc_train)
    r2 = shared.relabeling_r2(cand_target_train, v4_raw_train)
    ks_b_ok = r2 < shared.R2_KILL_THRESH
    print(f"\nKS-B (relabeling): R^2(candidate final target, v4 raw frac*scale) "
          f"= {r2:.6f}  (threshold: < {shared.R2_KILL_THRESH})  -> {'PASS' if ks_b_ok else 'FAIL'}")

    print("\nKS-C sample-size disclosure (design doc S6, mandatory report, not a kill switch):")
    ks_c_disclosure(btc.loc[:shared.INNER_TRAIN_END], "reached by INNER_TRAIN_END")
    ks_c_disclosure(btc.loc[:shared.INNER_VAL_END], "reached by INNER_VAL_END")

    if not (ks_a_ok and ks_b_ok):
        hr("VERDICT")
        tripped = []
        if not ks_a_ok:
            tripped.append(f"KS-A (binding_fraction={binding_frac:.6f} < {shared.GATE_MIN_BINDING_FRACTION})")
        if not ks_b_ok:
            tripped.append(f"KS-B (R^2={r2:.6f} >= {shared.R2_KILL_THRESH})")
        print(f"    VERDICT: NEGATIVE. Kill switch(es) tripped on BTC inner-train, "
              f"PRIMARY q: {', '.join(tripped)}.")
        print("    Per the design doc's own convention, a tripped kill switch stops "
              "evaluation here -- the main sweep, falsification test, plateau check "
              "and fee-tier check are NOT run.")
        print(f"    Configurations evaluated: {configs_evaluated} (stopped before the sweep).")
        print("    The holdout (>= 2023-01-01) was NOT read.")
        return

    print("\nNeither kill switch tripped on BTC inner-train (PRIMARY q) -- "
          "proceeding to the main comparison.")

    # ========================================================== STEP 2
    hr("STEP 2 -- PRIMARY compare(): inner_train / inner_val / eth_replication "
       "x SPOT/futures_5x, q=0.10 (PRIMARY, module default)")
    primary_rows = shared.compare(BUILD_PRIMARY, label=LABEL_PRIMARY, btc=btc, eth=eth)
    configs_evaluated += 1
    backtest_cells += len(primary_rows)
    shared.print_rows(primary_rows)

    # ========================================================== STEP 3
    hr("STEP 3 -- Falsification test (design doc S5, frozen): sign(dSharpe "
       "BTC inner_val) vs sign(dSharpe ETH replication), PRIMARY q, both markets")
    falsification_notes = []
    falsification_ok = True
    for market_name in (shared.SPOT.name, shared.FUTURES.name):
        btc_row = cell(primary_rows, LABEL_PRIMARY, "inner_val", market_name)
        eth_row = cell(primary_rows, LABEL_PRIMARY, shared.ETH_SLICE_NAME, market_name)
        btc_sign = np.sign(btc_row["d_sharpe"])
        eth_sign = np.sign(eth_row["d_sharpe"])
        note = (f"market={market_name:11s} BTC inner_val d_sharpe={btc_row['d_sharpe']:+.4f} "
                f"(sign {btc_sign:+.0f})   ETH replication d_sharpe={eth_row['d_sharpe']:+.4f} "
                f"(sign {eth_sign:+.0f})")
        print(f"    {note}")
        falsification_notes.append(note)
        if btc_sign != 0 and eth_sign != 0 and btc_sign != eth_sign:
            falsification_ok = False
    print(f"\nFALSIFICATION TEST (same sign, BTC inner_val vs ETH replication, on "
          f"every market -- design doc's exact kill outcome): "
          f"{'PASS' if falsification_ok else 'FAIL -- FALSIFIED'}")

    # ========================================================== STEP 4
    hr("STEP 4 -- Robustness bracket / plateau check (design doc S7 clause 3): "
       f"q in {Q_BRACKET}, horizon_days/min_n fixed at PRIMARY defaults")
    bracket_rows: dict[float, list[dict]] = {shared.Q_FCR: primary_rows}
    for q in Q_BRACKET:
        if q == shared.Q_FCR:
            continue
        label = f"r172_cons_fcr_gate_q{q:.4g}"
        build_q = make_build(q)
        rows_q = shared.compare(build_q, label=label, btc=btc, eth=eth)
        configs_evaluated += 1
        backtest_cells += len(rows_q)
        bracket_rows[q] = rows_q
        print(f"\n  -- q={q:.4g} --")
        shared.print_rows(rows_q)

    print("\n(4a) dSharpe across the q bracket, promotion-relevant cells (inner_val, "
          "eth_replication, both markets), no cherry-picking:")
    bracket_signs: dict[tuple[str, str], set] = {}
    for q in Q_BRACKET:
        label = f"r172_cons_fcr_gate_q{q:.4g}" if q != shared.Q_FCR else LABEL_PRIMARY
        rows_q = bracket_rows[q]
        for slice_name in ("inner_val", shared.ETH_SLICE_NAME):
            for market_name in (shared.SPOT.name, shared.FUTURES.name):
                row = cell(rows_q, label, slice_name, market_name)
                key = (slice_name, market_name)
                bracket_signs.setdefault(key, set()).add(np.sign(row["d_sharpe"]))
                print(f"    q={q:.4g}  {slice_name:16s} {market_name:11s} "
                      f"d_sharpe={row['d_sharpe']:+.4f}")

    plateau_ok = all(len(signs - {0.0}) <= 1 for signs in bracket_signs.values())
    print(f"\nPLATEAU CHECK (sign of dSharpe consistent across the q bracket on every "
          f"promotion-relevant cell: inner_val x {{spot,futures_5x}}, "
          f"eth_replication x {{spot,futures_5x}}): "
          f"{'PLATEAU (consistent)' if plateau_ok else 'PEAK (sign flips within the bracket)'}")
    for key, signs in bracket_signs.items():
        flips = len(signs - {0.0}) > 1
        print(f"    {key}: signs seen across bracket = {sorted(signs)}"
              f"{'  <-- FLIPS' if flips else ''}")

    # ========================================================== STEP 5
    hr("STEP 5 -- Fee-tier check (design doc S7 clause 4): BTC inner_val, "
       f"futures_5x, standard fee vs FEE_TIER={FEE_TIER_040:.2%} taker")
    fee_market = shared.fee_at(shared.FUTURES, FEE_TIER_040)
    cand_strat = shared.TargetStrategy(BUILD_PRIMARY, name=f"r102_{LABEL_PRIMARY}")
    ctrl_strat = shared.TargetStrategy(shared.v4_target, name="kelly_regime_v4")

    a_fee = shared.run_slice(cand_strat, btc, shared.INNER_VAL_START, shared.INNER_VAL_END,
                              "inner_val_fee040", fee_market)
    b_fee = shared.run_slice(ctrl_strat, btc, shared.INNER_VAL_START, shared.INNER_VAL_END,
                              "inner_val_fee040", fee_market)
    d_sharpe_fee = a_fee.sharpe - b_fee.sharpe
    configs_evaluated += 1
    backtest_cells += 1

    std_row = cell(primary_rows, LABEL_PRIMARY, "inner_val", shared.FUTURES.name)
    d_sharpe_std = std_row["d_sharpe"]
    print(f"\n    futures_5x @ standard fee ({shared.FUTURES.fee_rate:.4%}): "
          f"d_sharpe={d_sharpe_std:+.4f}  (cand_sharpe={std_row['cand_sharpe']:.4f} "
          f"ctrl_sharpe={std_row['ctrl_sharpe']:.4f})")
    print(f"    futures_5x @ {FEE_TIER_040:.2%} taker:      "
          f"d_sharpe={d_sharpe_fee:+.4f}  (cand_sharpe={a_fee.sharpe:.4f} "
          f"ctrl_sharpe={b_fee.sharpe:.4f})")
    sign_std = np.sign(d_sharpe_std)
    sign_fee = np.sign(d_sharpe_fee)
    clause4_ok = bool(sign_std == 0 or sign_fee == 0 or sign_std == sign_fee)
    print(f"\nFEE-TIER CHECK (sign does not reverse at {FEE_TIER_040:.2%} taker): "
          f"{'PASS' if clause4_ok else 'FAIL -- sign reverses'}")

    # ========================================================== STEP 6
    hr("STEP 6 -- Promotion rule (design doc S7), ALL FOUR clauses, mechanically")
    val_spot = cell(primary_rows, LABEL_PRIMARY, "inner_val", shared.SPOT.name)
    val_fut = cell(primary_rows, LABEL_PRIMARY, "inner_val", shared.FUTURES.name)
    eth_spot = cell(primary_rows, LABEL_PRIMARY, shared.ETH_SLICE_NAME, shared.SPOT.name)
    eth_fut = cell(primary_rows, LABEL_PRIMARY, shared.ETH_SLICE_NAME, shared.FUTURES.name)

    def clause1_sharpe(row_btc, row_eth) -> bool:
        return row_btc["d_sharpe"] >= shared.SHARPE_DELTA_PROMOTE and \
            row_eth["d_sharpe"] >= shared.SHARPE_DELTA_PROMOTE

    def clause1_dd(row_btc, row_eth) -> bool:
        return (row_btc["risk_matched"] and row_btc["d_dd"] <= -shared.DD_REDUCTION_PROMOTE_PP
                and row_eth["risk_matched"] and row_eth["d_dd"] <= -shared.DD_REDUCTION_PROMOTE_PP)

    clause1_spot = clause1_sharpe(val_spot, eth_spot) or clause1_dd(val_spot, eth_spot)
    clause1_fut = clause1_sharpe(val_fut, eth_fut) or clause1_dd(val_fut, eth_fut)
    clause1_ok = clause1_spot or clause1_fut
    print(f"\nClause 1 (dSharpe>=+{shared.SHARPE_DELTA_PROMOTE} both BTC&ETH inner-val, OR "
          f"matched-exposure DD reduction>={shared.DD_REDUCTION_PROMOTE_PP}pp both), "
          f"satisfied on >=1 market:")
    print(f"    spot:       BTC dSharpe={val_spot['d_sharpe']:+.4f}  ETH dSharpe={eth_spot['d_sharpe']:+.4f}  "
          f"BTC dDD={val_spot['d_dd']:+.2f} (RM={val_spot['risk_matched']})  "
          f"ETH dDD={eth_spot['d_dd']:+.2f} (RM={eth_spot['risk_matched']})  -> {clause1_spot}")
    print(f"    futures_5x: BTC dSharpe={val_fut['d_sharpe']:+.4f}  ETH dSharpe={eth_fut['d_sharpe']:+.4f}  "
          f"BTC dDD={val_fut['d_dd']:+.2f} (RM={val_fut['risk_matched']})  "
          f"ETH dDD={eth_fut['d_dd']:+.2f} (RM={eth_fut['risk_matched']})  -> {clause1_fut}")
    print(f"    Clause 1 satisfied on >=1 market: {clause1_ok}")

    print(f"\nClause 2 (survives falsification test): {falsification_ok}")
    print(f"Clause 3 (plateau, not peak, across q in {Q_BRACKET}): {plateau_ok}")
    print(f"Clause 4 (sign does not reverse at 0.40% taker fee): {clause4_ok}")

    all_four = clause1_ok and falsification_ok and plateau_ok and clause4_ok
    print(f"\nALL FOUR CLAUSES MET: {all_four}")

    hr("VERDICT")
    if all_four:
        print("    VERDICT: PROMOTE-CANDIDATE (worth carrying to the holdout) -- all "
              "four design-doc S7 clauses are met on the inner-validation/ETH/"
              "fee-tier evidence above. The holdout decision itself is the "
              "operator's, not this file's.")
    else:
        failed = []
        if not clause1_ok:
            failed.append("clause 1 (Sharpe/DD promotion bar)")
        if not falsification_ok:
            failed.append("clause 2 (falsification -- BTC/ETH sign disagreement)")
        if not plateau_ok:
            failed.append("clause 3 (plateau -- sign flips across the q bracket)")
        if not clause4_ok:
            failed.append("clause 4 (0.40% fee-tier sign reversal)")
        print(f"    VERDICT: NEGATIVE. Clause(s) not met: {', '.join(failed)}.")
        print("    Per r172_direction.md S7: 'Any other outcome is NEGATIVE, including "
              "partial passes' -- a partial pass is reported as a partial pass here, "
              "not rounded up to the nearest favorable label.")

    hr("SUMMARY")
    print(f"Causality probe: {'PASS' if causal_ok else 'FAIL'}")
    print(f"KS-A (binding_fraction): {binding_frac:.6f}  (threshold >= {shared.GATE_MIN_BINDING_FRACTION})  "
          f"-> {'PASS' if ks_a_ok else 'FAIL'}")
    print(f"KS-B (relabeling R^2): {r2:.6f}  (threshold < {shared.R2_KILL_THRESH})  "
          f"-> {'PASS' if ks_b_ok else 'FAIL'}")
    print(f"Falsification test: {'PASS' if falsification_ok else 'FAIL -- FALSIFIED'}")
    print(f"Plateau check: {'PLATEAU' if plateau_ok else 'PEAK'}")
    print(f"Fee-tier check: {'PASS' if clause4_ok else 'FAIL'}")
    print(f"\nConfigurations evaluated (distinct build_target parameterisations): "
          f"{configs_evaluated}  (PRIMARY q=0.10: 1 config / 6 cells; "
          f"q-bracket 0.05 & 0.20: 2 configs / 6 cells each = 12 cells; "
          f"fee-tier BTC inner_val futures_5x @ 0.40%: 1 config / 1 cell)")
    print(f"Total backtest cells run (candidate+control paired, each cell = 1 compare() "
          f"row or 1 run_slice pair): {backtest_cells}")
    print(f"Max timestamp read anywhere in this run: {max(max_ts_seen)}  "
          f"(must be < {shared.OOS_START})")
    assert max(max_ts_seen) < pd.Timestamp(shared.OOS_START, tz="UTC"), \
        "holdout was read -- this must never happen in this file"
    print(f"Total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
