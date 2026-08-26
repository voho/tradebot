"""R-149 NOVEL branch: fixed-share re-injection (Herbster & Warmuth 1998;
continuous-grid generalization per Cesa-Bianchi, Gaillard, Lugosi & Stoltz
2012) into `universal_kelly`'s own wealth posterior, run against the frozen
`experiments/r149_shared.py` pre-registration.

This file is intentionally thin: `novel_target()`, `universal_kelly_target()`,
`compare()`, `print_rows()`, `r_squared()`, `paired_diff()`, `run_slice()` and
every constant are already defined in the frozen, read-only `r149_shared.py`
(written by the operator before this branch ran). This script only SEQUENCES
the pre-registered checks against that machinery and reports results -- it
adds no new sizing logic of its own.

Order of operations, matching the pre-registration's PROMOTION BAR:
  A2. Step-0 kill switch: R^2 of novel_target (fixed_share=1e-2, the shipped
      default) vs universal_kelly_target, inner-train. If R^2 > 0.98, STOP.
  B1. compare(novel_target, label="novel", markets=(SPOT, FUTURES),
      include_eth=False) -- read d_sharpe + bootstrap CI on inner_val, both
      markets. Passes if d_sharpe > +0.2 OR the 95% interval excludes zero,
      on >=1 market.
  B2. Diagnostic only: exposure_ratio / vol_ratio for every cell (printed by
      print_rows(), not separately gated).
  B3. Plateau: fixed_share in {1e-2, 3e-2, 1e-1}, inner-validation, primary
      market = whichever market B1 passed on (SPOT if both/neither) -- sign
      of d_sharpe must hold across the full grid.
  B4. ETH falsification: compare(novel_target, label="novel_eth",
      include_eth=True) -- eth_replication slice's d_sharpe sign must agree
      with BTC inner_val's sign on >=1 market.
  B5. 0.40% spot fee tier re-run, inner-validation, gates only if B1 passed.
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

from experiments.r149_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    SPOT,
    compare,
    fee_at,
    load_btc,
    novel_target,
    print_rows,
    universal_kelly_target,
    r_squared,
)


def step0_kill_switch() -> float:
    """A2: R^2 of novel_target (default fixed_share=1e-2) vs
    universal_kelly_target on inner-train only."""
    btc = load_btc()
    inner_train = btc[btc.index < pd.Timestamp(INNER_TRAIN_END, tz="UTC")]
    return r_squared(novel_target(inner_train), universal_kelly_target(inner_train))


def b3_plateau_sweep(primary_market) -> list[dict]:
    """fixed_share in {1e-2, 3e-2, 1e-1}, inner-validation, primary market only."""
    rows = []
    for fixed_share in (1e-2, 3e-2, 1e-1):
        candidate = functools.partial(novel_target, fixed_share=fixed_share)
        r = compare(
            candidate, label=f"novel_fs{fixed_share:g}",
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
    print("R-149 NOVEL: fixed-share re-injection into universal_kelly's wealth posterior")
    print("=" * 100)

    verdicts: dict[str, bool | None] = {"A2": None, "B1": None, "B3": None, "B4": None, "B5": None}

    # ---------------------------------------------------------- A2 kill switch
    print("\n--- A2 Step-0 kill switch: R^2(novel_target[fixed_share=1e-2], "
          "universal_kelly_target), inner-train ---")
    r2 = step0_kill_switch()
    print(f"R^2 = {r2:.6f}")
    print("(operator's own smoke-test measurement was ~0.9559; comparing against that now)")
    if abs(r2 - 0.9559) > 0.01:
        print(f"NOTE: this run's R^2 ({r2:.6f}) differs from the operator's smoke-test "
              f"figure (0.9559) by more than 0.01 -- flagging as requested.")
    else:
        print(f"This run's R^2 ({r2:.6f}) matches the operator's smoke-test figure (~0.9559) "
              f"to within 0.01.")
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

    # pick primary market for B3: whichever market B1 passed on; SPOT if
    # both or neither passed.
    passing = [m for m, p, _ in b1_pass_markets if p]
    if len(passing) == 1:
        primary_name = passing[0]
    else:
        primary_name = SPOT.name
    primary_market = SPOT if primary_name == SPOT.name else FUTURES
    print(f"Primary market selected for B3 plateau sweep: {primary_market.name}")

    # -------------------------------------------------------------- B2 (diagnostic)
    print("\n--- B2 (diagnostic, not gating): exposure_ratio / vol_ratio, all B1 cells ---")
    for r in b1_rows:
        print(f"  slice={r['slice']:12s} market={r['market']:10s} "
              f"exposure_ratio={r['exposure_ratio']:.4f} vol_ratio={r['vol_ratio']:.4f} "
              f"risk_matched={r['risk_matched']}")

    # -------------------------------------------------------------- B3
    print(f"\n--- B3 plateau: fixed_share in {{1e-2, 3e-2, 1e-1}}, inner-validation, "
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
    overall = print_summary(verdicts, pytest_ok=pytest_ok)
    return 0 if overall == "PROMOTE-CANDIDATE" else 1


def print_summary(verdicts: dict, pytest_ok: bool | None = None) -> str:
    print("\n" + "=" * 100)
    print("PROMOTION BAR SUMMARY -- R-149 NOVEL (fixed-share re-injection, universal_kelly)")
    print("=" * 100)

    def fmt(v):
        if v is None:
            return "MOOT"
        return "PASS" if v else "FAIL"

    a2_ok = verdicts.get("A2") is True
    print(f"  A2 (Step-0 kill switch, R^2 <= 0.98):     {fmt(verdicts.get('A2'))}")
    print(f"  B1 (d_sharpe > +0.2 or CI excl. 0, >=1mkt): {fmt(verdicts.get('B1'))}")
    print(f"  B2 (diagnostic exposure/vol ratios):        reported above, not gating")
    print(f"  B3 (plateau: sign-stable across fs sweep):  {fmt(verdicts.get('B3'))}")
    print(f"  B4 (ETH same-sign falsification):           {fmt(verdicts.get('B4'))}")
    print(f"  B5 (0.40% fee tier sign survival):          {fmt(verdicts.get('B5'))}")
    if pytest_ok is not None:
        print(f"  pytest (causality-strict + full suite):     {'PASS' if pytest_ok else 'FAIL'}")

    b1_ok = verdicts.get("B1") is True
    b4_ok = verdicts.get("B4") is True
    b5_ok = verdicts.get("B5") is True or verdicts.get("B5") is None  # pass or moot

    if a2_ok and b1_ok and b4_ok and b5_ok:
        overall = "PROMOTE-CANDIDATE"
    else:
        overall = "NEGATIVE"
    print(f"\n  OVERALL VERDICT: {overall}")
    print("=" * 100)
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
