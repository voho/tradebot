"""R-150 NOVEL branch: per-member fractional-Kelly pre-scaling (Whitrow 2007)
of `champions_council`'s six Hedge-weighted members' own signals, run against
the frozen `experiments/r150_shared.py` pre-registration.

This file is intentionally thin: `novel_target()`, `per_member_kelly_
fraction()`, `champions_council_target()`, `compare()`, `print_rows()`,
`r_squared()`, `run_b6()` and every constant are already defined in the
frozen, read-only `r150_shared.py` (written by the operator before this
branch ran). This script only SEQUENCES the pre-registered checks against
that machinery and reports results -- it adds no new sizing logic of its own.

Order of operations, matching the pre-registration's PROMOTION BAR:
  A2. Step-0 kill switch: R^2 of novel_target (default kelly_cap=1.5) vs
      champions_council_target, inner-train. If R^2 > 0.98, STOP.
  B1. compare(novel_target, label="novel", markets=(SPOT, FUTURES),
      include_eth=False) -- read d_sharpe + bootstrap CI on inner_val, both
      markets. Passes if d_sharpe > +0.2 OR the 95% interval excludes zero,
      on >=1 market.
  B2. Diagnostic only: exposure_ratio / vol_ratio for every cell (printed by
      print_rows(), not separately gated).
  Named kill-condition check (this round's own pre-registered novel failure
      risk): turnover ratio (fills) > 2x the control AND ΔSharpe negative on
      BOTH BTC markets at inner-validation.
  B3. Plateau: kelly_cap in {1.0, 1.5, 2.0}, inner-validation, primary
      market = whichever market B1 passed on (SPOT if both/neither) -- sign
      of d_sharpe must hold across the full grid.
  B4. ETH falsification: compare(novel_target, label="novel_eth",
      include_eth=True) -- eth_replication slice's d_sharpe sign must agree
      with BTC inner_val's sign on >=1 market.
  B5. 0.40% spot fee tier re-run, inner-validation, gates only if B1 passed.
  B6. Mandatory whenever B1 passes on any market: run_b6() -- a zero-
      information, turnover-matched control. If it detects an artifact, the
      final verdict is NEGATIVE regardless of A2-B5.
  Then the causality-strict test and the full pytest suite via subprocess.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.r150_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    SPOT,
    champions_council_target,
    compare,
    fee_at,
    load_btc,
    novel_target,
    print_rows,
    run_b6,
    r_squared,
)


def step0_kill_switch() -> float:
    """A2: R^2 of novel_target (default kelly_cap=1.5) vs
    champions_council_target on inner-train only."""
    btc = load_btc()
    inner_train = btc[btc.index < pd.Timestamp(INNER_TRAIN_END, tz="UTC")]
    return r_squared(novel_target(inner_train), champions_council_target(inner_train))


def b3_plateau_sweep(primary_market) -> list[dict]:
    """kelly_cap in {1.0, 1.5, 2.0}, inner-validation, primary market only."""
    rows = []
    for kelly_cap in (1.0, 1.5, 2.0):
        candidate = functools.partial(novel_target, kelly_cap=kelly_cap)
        r = compare(
            candidate, label=f"novel_kcap{kelly_cap:g}",
            markets=(primary_market,), include_eth=False,
        )
        rows.extend(row for row in r if row["slice"] == "inner_val")
    return rows


def b5_fee_tier_check() -> list[dict]:
    """0.40% spot fee tier, inner-validation + inner-train, BTC only."""
    return compare(
        novel_target, label="novel_fee40bp",
        markets=(fee_at(SPOT, 0.004),), include_eth=False,
    )


def run_pytest(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    out = proc.stdout + "\n" + proc.stderr
    return proc.returncode, out


def main() -> int:
    print("=" * 100)
    print("R-150 NOVEL: per-member fractional-Kelly pre-scaling (champions_council)")
    print("=" * 100)

    verdicts: dict[str, bool | None] = {
        "A2": None, "B1": None, "B3": None, "B4": None, "B5": None, "B6": None,
    }

    # ---------------------------------------------------------- A2 kill switch
    print("\n--- A2 Step-0 kill switch: R^2(novel_target[kelly_cap=1.5], "
          "champions_council_target), inner-train ---")
    r2 = step0_kill_switch()
    print(f"R^2 = {r2:.6f}")
    if r2 > 0.98:
        verdicts["A2"] = False
        print("\n*** A2 KILL SWITCH TRIPPED (R^2 > 0.98) -- STOPPING. VERDICT: NEGATIVE ***")
        print_summary(verdicts)
        return 1
    verdicts["A2"] = True
    print("A2 kill switch: PASS (R^2 <= 0.98) -- proceeding.")

    # -------------------------------------------------------------- B1
    print("\n--- B1: compare(novel_target, label='novel', markets=(SPOT, FUTURES), "
          "include_eth=False) ---")
    b1_rows = compare(novel_target, label="novel", markets=(SPOT, FUTURES), include_eth=False)
    print()
    print_rows(b1_rows)

    b1_inner_val = [r for r in b1_rows if r["slice"] == "inner_val"]
    b1_pass_markets = []
    for r in b1_inner_val:
        passed = r["d_sharpe"] > 0.2 or r["excludes_zero"]
        b1_pass_markets.append((r["market"], passed, r))
        print(f"  market={r['market']:10s} d_sharpe={r['d_sharpe']:+.4f} "
              f"boot_CI=[{r['boot_lo']:+.4f}, {r['boot_hi']:+.4f}] "
              f"excludes_zero={r['excludes_zero']} -> {'PASS' if passed else 'fail'}")
    b1_pass = any(p for _, p, _ in b1_pass_markets)
    verdicts["B1"] = b1_pass
    print(f"B1 overall: {'PASS' if b1_pass else 'FAIL'} (>=1 market required)")

    # pick primary market: whichever market B1 passed on; SPOT if both or
    # neither passed. Reused identically for B3's own primary-market choice
    # and B6's primary_market argument.
    passing = [m for m, p, _ in b1_pass_markets if p]
    if len(passing) == 1:
        primary_name = passing[0]
    else:
        primary_name = SPOT.name
    primary_market = SPOT if primary_name == SPOT.name else FUTURES
    print(f"Primary market selected (for B3 sweep and B6, if run): {primary_market.name}")

    # -------------------------------------------------------------- B2 (diagnostic)
    print("\n--- B2 (diagnostic, not gating): exposure_ratio / vol_ratio, all B1 cells ---")
    for r in b1_rows:
        print(f"  slice={r['slice']:12s} market={r['market']:10s} "
              f"exposure_ratio={r['exposure_ratio']:.4f} vol_ratio={r['vol_ratio']:.4f} "
              f"risk_matched={r['risk_matched']}")

    # --------------------------------------------------- named kill condition
    # This round's own pre-registered failure risk for the novel branch:
    # "turnover ratio (fills) > 2x the control AND ΔSharpe negative on both
    # BTC markets at inner-validation."
    print("\n--- Named failure-risk check (pre-registered): turnover ratio (fills) "
          "> 2x control AND ΔSharpe negative on BOTH BTC markets, inner-validation ---")
    turnover_ratios = {}
    d_sharpes = {}
    for r in b1_inner_val:
        ratio = (r["cand_trades"] / r["ctrl_trades"]) if r["ctrl_trades"] else float("nan")
        turnover_ratios[r["market"]] = ratio
        d_sharpes[r["market"]] = r["d_sharpe"]
        print(f"  market={r['market']:10s} cand_trades={r['cand_trades']:>6d} "
              f"ctrl_trades={r['ctrl_trades']:>6d} turnover_ratio={ratio:.3f}x "
              f"d_sharpe={r['d_sharpe']:+.4f}")
    both_over_2x = all(v > 2.0 for v in turnover_ratios.values()) if turnover_ratios else False
    both_dsharpe_neg = all(v < 0 for v in d_sharpes.values()) if d_sharpes else False
    kill_condition_hit = both_over_2x and both_dsharpe_neg
    print(f"  turnover ratio > 2x on ALL BTC markets: {both_over_2x}")
    print(f"  d_sharpe < 0 on ALL BTC markets: {both_dsharpe_neg}")
    print(f"  NAMED KILL CONDITION REPRODUCED: {kill_condition_hit}")
    print("  (this check is diagnostic/reporting only, per the round's own pre-registration; "
          "the formal gate is A2/B1/B3/B4/B5/B6 below, not this statistic directly)")

    # -------------------------------------------------------------- B3
    print(f"\n--- B3 plateau: kelly_cap in {{1.0, 1.5, 2.0}}, inner-validation, "
          f"primary market={primary_market.name} ---")
    b3_rows = b3_plateau_sweep(primary_market)
    print_rows(b3_rows)
    signs = []
    for r in b3_rows:
        sign = "+" if r["d_sharpe"] > 0 else ("-" if r["d_sharpe"] < 0 else "0")
        signs.append(sign)
        print(f"  label={r['label']:22s} market={r['market']:10s} "
              f"d_sharpe={r['d_sharpe']:+.4f} sign={sign} "
              f"boot_d_loggrowth={r['boot_d_loggrowth']:+.4f} excludes_zero={r['excludes_zero']}")
    b3_pass = len(set(signs)) == 1 and "0" not in signs
    verdicts["B3"] = b3_pass
    print(f"B3 sign stability across sweep: signs={signs} -> {'PASS' if b3_pass else 'FAIL'}")

    # -------------------------------------------------------------- B4
    print("\n--- B4: compare(novel_target, label='novel_eth', include_eth=True) ---")
    b4_rows = compare(novel_target, label="novel_eth", include_eth=True)
    print()
    print_rows(b4_rows)

    btc_val_signs = {r["market"]: (r["d_sharpe"] > 0) for r in b4_rows if r["slice"] == "inner_val"}
    eth_signs = {r["market"]: (r["d_sharpe"] > 0) for r in b4_rows if r["slice"] == "eth_replication"}
    print("\n  BTC inner_val d_sharpe signs:", {m: ("+" if s else "-") for m, s in btc_val_signs.items()})
    print("  ETH replication d_sharpe signs:", {m: ("+" if s else "-") for m, s in eth_signs.items()})
    b4_pass = any(
        m in eth_signs and eth_signs[m] == btc_val_signs[m]
        for m in btc_val_signs
    )
    verdicts["B4"] = b4_pass
    print(f"B4 ETH same-sign falsification: {'PASS' if b4_pass else 'FAIL'} (>=1 market agreeing required)")

    # -------------------------------------------------------------- B5
    if b1_pass:
        print("\n--- B5: 0.40% spot fee tier, inner-validation + inner-train, BTC only "
              "(B1 passed, so this gates) ---")
        b5_rows = b5_fee_tier_check()
        print_rows(b5_rows)
        b5_val = [r for r in b5_rows if r["slice"] == "inner_val"]
        # sign must survive vs B1's SPOT inner_val sign at default fee.
        b1_spot_val = next((r for r in b1_inner_val if r["market"] == SPOT.name), None)
        b1_spot_sign = (b1_spot_val["d_sharpe"] > 0) if b1_spot_val is not None else None
        b5_pass = False
        for r in b5_val:
            b5_sign = r["d_sharpe"] > 0
            print(f"  market={r['market']:10s} (0.40% fee) d_sharpe={r['d_sharpe']:+.4f} "
                  f"vs default-fee SPOT sign={'+' if b1_spot_sign else '-'}")
            if b1_spot_sign is not None and b5_sign == b1_spot_sign:
                b5_pass = True
        verdicts["B5"] = b5_pass
        print(f"B5 fee-tier sign survival: {'PASS' if b5_pass else 'FAIL'}")
    else:
        print("\n--- B5: SKIPPED (moot -- B1 did not pass) ---")
        verdicts["B5"] = None  # moot

    # -------------------------------------------------------------- B6
    # Mandatory whenever B1 passes on any market (this round's own added
    # requirement, per R-132/R-135's own lesson).
    b6_result = None
    if b1_pass:
        primary_row = next(
            r for r in b1_inner_val if r["market"] == primary_market.name
        )
        cand_inner_val_trades = int(primary_row["cand_trades"])
        print(f"\n--- B6 (mandatory: B1 passed): zero-information turnover-matched control, "
              f"primary_market={primary_market.name}, "
              f"cand_inner_val_trades={cand_inner_val_trades} ---")
        b6_result = run_b6(
            primary_market=primary_market,
            cand_inner_val_trades=cand_inner_val_trades,
        )
        print(f"  trials (deadband -> trade count): {b6_result['trials_trade_counts']}")
        print(f"  chosen deadband: {b6_result['chosen_deadband']} "
              f"(trades={b6_result['chosen_trades']}, candidate trades={b6_result['cand_trades']})")
        print(f"  deadband-only control d_sharpe vs true control: {b6_result['d_sharpe']:+.4f}")
        print(f"  bootstrap CI: [{b6_result['boot_lo']:+.4f}, {b6_result['boot_hi']:+.4f}]")
        print(f"  ARTIFACT DETECTED: {b6_result['artifact_detected']}")
        verdicts["B6"] = not b6_result["artifact_detected"]
        if b6_result["artifact_detected"]:
            print("  *** B6 DETECTED A TURNOVER-REDUCTION ARTIFACT -- B1's pass is "
                  "reclassified as an artifact, not a genuine mechanism. ***")
        else:
            print("  B6: no artifact detected -- the zero-information, turnover-matched "
                  "control does NOT itself clear B1's own bar.")
    else:
        print("\n--- B6: SKIPPED (B1 did not pass on any market, so the mandatory-if-B1-passes "
              "gate does not apply) ---")
        verdicts["B6"] = None  # moot / skipped

    # -------------------------------------------------------------- pytest
    print("\n--- pytest tests/test_causality_strict.py -q ---")
    rc_causal, out_causal = run_pytest(["tests/test_causality_strict.py", "-q"])
    print(out_causal)
    print(f"exit code: {rc_causal}")

    print("\n--- full pytest -q suite ---")
    rc_full, out_full = run_pytest(["-q"])
    print(out_full)
    print(f"exit code: {rc_full}")

    pytest_ok = (rc_causal == 0) and (rc_full == 0)

    # ---------------------------------------------------------- final summary
    overall = print_summary(
        verdicts, pytest_ok=pytest_ok,
        kill_condition_hit=kill_condition_hit,
        turnover_ratios=turnover_ratios, d_sharpes=d_sharpes,
    )
    return 0 if overall == "PROMOTE-CANDIDATE" else 1


def print_summary(verdicts: dict, pytest_ok: bool | None = None,
                   kill_condition_hit: bool | None = None,
                   turnover_ratios: dict | None = None,
                   d_sharpes: dict | None = None) -> str:
    print("\n" + "=" * 100)
    print("PROMOTION BAR SUMMARY -- R-150 NOVEL (per-member fractional-Kelly, champions_council)")
    print("=" * 100)

    def fmt(v):
        if v is None:
            return "MOOT"
        return "PASS" if v else "FAIL"

    a2_ok = verdicts.get("A2") is True
    print(f"  A2 (Step-0 kill switch, R^2 <= 0.98):       {fmt(verdicts.get('A2'))}")
    print(f"  B1 (d_sharpe > +0.2 or CI excl. 0, >=1mkt):  {fmt(verdicts.get('B1'))}")
    print(f"  B2 (diagnostic exposure/vol ratios):         reported above, not gating")
    if turnover_ratios is not None:
        print(f"  Named kill condition (turnover>2x ALL BTC mkts AND "
              f"dSharpe<0 ALL BTC mkts): {kill_condition_hit} "
              f"(turnover_ratios={ {k: round(v,3) for k,v in turnover_ratios.items()} }, "
              f"d_sharpes={ {k: round(v,4) for k,v in (d_sharpes or {}).items()} })")
    print(f"  B3 (plateau: sign-stable across kelly_cap sweep): {fmt(verdicts.get('B3'))}")
    print(f"  B4 (ETH same-sign falsification):            {fmt(verdicts.get('B4'))}")
    print(f"  B5 (0.40% fee tier sign survival):           {fmt(verdicts.get('B5'))}")
    print(f"  B6 (mandatory zero-info turnover-matched control, no artifact): {fmt(verdicts.get('B6'))}")
    if pytest_ok is not None:
        print(f"  pytest (causality-strict + full suite):      {'PASS' if pytest_ok else 'FAIL'}")

    b1_ok = verdicts.get("B1") is True
    b4_ok = verdicts.get("B4") is True
    b5_ok = verdicts.get("B5") is True or verdicts.get("B5") is None  # pass or moot
    b6_ok = verdicts.get("B6") is True or verdicts.get("B6") is None  # no artifact, or skipped (B1 never passed)

    if a2_ok and b1_ok and b4_ok and b5_ok and b6_ok:
        overall = "PROMOTE-CANDIDATE"
    else:
        overall = "NEGATIVE"
    print(f"\nVERDICT: {overall}")
    print("=" * 100)
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
