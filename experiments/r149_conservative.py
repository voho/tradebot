"""R-149 CONSERVATIVE branch: bolt kelly_regime_v3/v4's own (unretuned)
conditional-volatility-target scale onto universal_kelly, replacing its
fixed ``kappa=0.5`` scalar.

Executes the *conservative* half of the frozen pre-registration in
``experiments/r149_shared.py`` (read that file's module docstring for the
full pre-registration -- direction, non-duplicate argument, literature,
named failure risks and the exact promotion bar; this file does not restate
any of it beyond what is needed to run the checks).

Per that freeze:
- mechanism: ``conservative_target`` (imported, not reimplemented) --
  universal_kelly's own posterior mean ``b_hat``, byte-identical to the
  control, scaled by v4's own conditional-volatility-target factor instead
  of the fixed ``kappa=0.5``.
- Step-0 A2 kill switch: R^2 of ``conservative_target`` vs
  ``universal_kelly_target`` on inner-train. If > 0.98, STOP.
- B1: bootstrap paired Delta-log-growth / d_sharpe, inner-validation, both
  markets (spot, 5x futures).
- B2 (diagnostic only): exposure_ratio / vol_ratio reported for every cell.
- B3: target_vol +/-20% plateau sweep (0.44, 0.55, 0.66), sign stability on
  the market B1 passed on (SPOT if both/neither passed).
- B4: ETH replication slice, sign agreement with BTC inner-validation on
  at least one market.
- B5: 0.40% spot fee tier, sign survival. Gates only if B1 passed.
- pytest: causality-strict + full suite, to confirm nothing outside this
  file broke (this branch touches no file except this one).

This file does not modify ``r149_shared.py`` and does not commit anything,
touch ``docs/LEDGER.md``, or read/print/compute anything at or after
``OOS_START`` -- per ROUTINE.md's parallel-branch rules, the operator
merges and writes the ledger entry once after both branches (this one and
the "novel" branch) report.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.r149_shared import (  # noqa: E402
    FUTURES,
    INNER_TRAIN_END,
    SPOT,
    V4_TARGET_VOL,
    compare,
    conservative_target,
    fee_at,
    load_btc,
    print_rows,
    r_squared,
    universal_kelly_target,
)

# --------------------------------------------------------------- A2: Step-0 kill switch

def run_a2() -> float:
    btc = load_btc()
    inner_train = btc[btc.index < INNER_TRAIN_END]
    cand = conservative_target(inner_train)
    ctrl = universal_kelly_target(inner_train)
    return r_squared(cand, ctrl)


# --------------------------------------------------------------- B3: target_vol +/-20% sweep

def target_vol_variant(target_vol: float):
    """A candidate build_fn with target_vol overridden, via functools.partial
    over conservative_target's own `target_vol` kwarg -- no new mechanism,
    just a parameter probe for the plateau check."""
    return functools.partial(conservative_target, target_vol=target_vol)


def run_b3(primary_market) -> dict:
    sweep = {}
    for tv in (0.44, V4_TARGET_VOL, 0.66):
        rows = compare(
            target_vol_variant(tv),
            label=f"cons_tv{tv:.2f}",
            markets=(primary_market,),
            include_eth=False,
        )
        sweep[tv] = rows
    return sweep


# --------------------------------------------------------------- B5: 0.40% fee tier

def run_b5() -> list[dict]:
    return compare(
        conservative_target,
        label="cons_v4scale_fee40",
        markets=(fee_at(SPOT, 0.004),),
        include_eth=False,
    )


# --------------------------------------------------------------- reporting helpers

def _inner_val_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["slice"] == "inner_val"]


def _eth_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["slice"] == "eth_replication"]


def fmt_b3_table(sweep: dict) -> str:
    lines = [f"{'target_vol':>10s} {'market':>10s} {'d_sharpe':>9s} "
             f"{'d_loggrowth':>12s} {'boot_lo':>9s} {'boot_hi':>9s} {'excl0':>6s} {'sign':>5s}"]
    lines.append("-" * len(lines[0]))
    for tv, rows in sweep.items():
        for r in _inner_val_rows(rows):
            sign = "+" if r["d_sharpe"] > 0 else ("-" if r["d_sharpe"] < 0 else "0")
            lines.append(
                f"{tv:10.3f} {r['market']:>10s} "
                f"{r['d_sharpe']:+9.3f} {r['d_log_growth']:+12.4f} "
                f"{r['boot_lo']:+9.3f} {r['boot_hi']:+9.3f} "
                f"{'YES' if r['excludes_zero'] else 'no':>6s} {sign:>5s}")
    return "\n".join(lines)


def run_pytest(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest"] + args,
        cwd=str(ROOT), capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


if __name__ == "__main__":
    print("=" * 100)
    print("R-149 CONSERVATIVE branch: conditional-vol-target scale on universal_kelly")
    print("=" * 100)

    # ---- A2 kill switch ----
    print("\n--- Step-0 A2 kill switch (R^2 vs universal_kelly_target, inner-train) ---")
    r2 = run_a2()
    print(f"R^2(conservative_target, universal_kelly_target) on inner-train = {r2:.6f}")
    if r2 > 0.98:
        print(f"\n*** A2 KILL SWITCH TRIPPED (R^2={r2:.6f} > 0.98) -- STOPPING. ***")
        print("VERDICT: NEGATIVE (Step-0 kill: candidate is a disguised no-op).")
        sys.exit(0)
    print("A2 does not trip -- proceeding.")

    # ---- primary comparison (both markets, BTC only) ----
    print("\n--- Primary comparison: compare(conservative_target, markets=(SPOT, FUTURES), "
          "include_eth=False) ---")
    rows = compare(conservative_target, label="conservative", markets=(SPOT, FUTURES),
                    include_eth=False)
    print_rows(rows)

    inner_val = _inner_val_rows(rows)

    # ---- B1 gate evaluation (diagnostic computation here; verdict below) ----
    b1_pass_by_market = {}
    for r in inner_val:
        b1_pass_by_market[r["market"]] = (r["d_sharpe"] > 0.2) or r["excludes_zero"]
    b1_pass = any(b1_pass_by_market.values())

    # primary market for B3: whichever market B1 passed on; SPOT if both/neither
    passing_markets = [m for m, p in b1_pass_by_market.items() if p]
    if len(passing_markets) == 1:
        primary_market_name = passing_markets[0]
    else:
        primary_market_name = SPOT.name
    primary_market = SPOT if primary_market_name == SPOT.name else FUTURES
    print(f"\nB1 primary market for B3 sweep: {primary_market_name} "
          f"(b1_pass_by_market={b1_pass_by_market})")

    # ---- B2 diagnostic (already printed above via print_rows: exposure_ratio, vol_ratio) ----
    print("\n--- B2 (diagnostic only, not gating): exposure_ratio / vol_ratio per cell ---")
    for r in rows:
        print(f"  {r['slice']:12s} {r['market']:>10s}  exposure_ratio={r['exposure_ratio']:.3f}  "
              f"vol_ratio={r['vol_ratio']:.3f}  risk_matched={r['risk_matched']}")

    # ---- B3 plateau sweep ----
    print(f"\n--- B3: target_vol {{0.44, {V4_TARGET_VOL}, 0.66}} plateau sweep "
          f"(inner-validation, {primary_market_name}) ---")
    sweep = run_b3(primary_market)
    b3_table = fmt_b3_table(sweep)
    print(b3_table)

    # sign stability across the three sweep points, on the primary market
    b3_signs = []
    for tv, srows in sweep.items():
        for r in _inner_val_rows(srows):
            b3_signs.append(1 if r["d_sharpe"] > 0 else (-1 if r["d_sharpe"] < 0 else 0))
    b3_pass = len(set(b3_signs)) == 1 and b3_signs[0] != 0 if b3_signs else False
    print(f"\nB3 sign stability on {primary_market_name}: signs={b3_signs} stable={b3_pass}")

    # ---- B5: 0.40% fee tier (gates only if B1 passed) ----
    print("\n--- B5: 0.40% spot fee tier (inner-validation, SPOT only) ---")
    if b1_pass:
        b5_rows = run_b5()
        print_rows(b5_rows)
        b5_inner_val = _inner_val_rows(b5_rows)

        primary_spot_dsharpe = next((r["d_sharpe"] for r in inner_val if r["market"] == "spot"),
                                     float("nan"))
        b5_spot_dsharpe = next((r["d_sharpe"] for r in b5_inner_val), float("nan"))
        b5_survives = (
            np.isfinite(primary_spot_dsharpe) and np.isfinite(b5_spot_dsharpe)
            and np.sign(primary_spot_dsharpe) == np.sign(b5_spot_dsharpe) and primary_spot_dsharpe != 0
        )
        print(f"\nSpot d_sharpe @0.10% fee = {primary_spot_dsharpe:+.4f}   "
              f"@0.40% fee = {b5_spot_dsharpe:+.4f}   sign survives = {b5_survives}")
    else:
        b5_survives = None
        print("B1 did not pass on any market -- B5 is moot (skipped).")

    b5_pass = (not b1_pass) or bool(b5_survives)  # B5 gates only if B1 passed

    # ---- B4: ETH sign agreement ----
    print("\n--- B4: ETH replication slice sign agreement with BTC inner-validation ---")
    eth_compare_rows = compare(conservative_target, label="conservative_eth",
                                markets=(SPOT, FUTURES), include_eth=True)
    eth_rows = _eth_rows(eth_compare_rows)
    btc_inner_val_from_eth_run = _inner_val_rows(eth_compare_rows)

    eth_signs = {r["market"]: (1 if r["d_sharpe"] > 0 else (-1 if r["d_sharpe"] < 0 else 0))
                 for r in eth_rows}
    btc_signs = {r["market"]: (1 if r["d_sharpe"] > 0 else (-1 if r["d_sharpe"] < 0 else 0))
                 for r in inner_val}
    print_rows(eth_rows)
    b4_pass = False
    for mkt, eth_sign in eth_signs.items():
        btc_sign = btc_signs.get(mkt)
        agree = (btc_sign is not None and eth_sign == btc_sign and eth_sign != 0)
        print(f"  {mkt:>10s}: BTC inner_val sign={btc_sign:+d}  ETH sign={eth_sign:+d}  agree={agree}")
        b4_pass = b4_pass or agree

    # ---- pytest ----
    print("\n--- pytest tests/test_causality_strict.py -q ---")
    rc1, out1 = run_pytest(["tests/test_causality_strict.py", "-q"])
    print(out1)
    print("\n--- pytest -q (full suite) ---")
    rc2, out2 = run_pytest(["-q"])
    print(out2)

    # ---- promotion bar ----
    print("\n" + "=" * 100)
    print("PROMOTION BAR SUMMARY")
    print("=" * 100)
    print(f"A2 (kill switch):        R^2={r2:.6f}  tripped={r2 > 0.98}")
    print(f"B1 (>=1 market passes):  {b1_pass_by_market}  -> pass={b1_pass}")
    print(f"B3 (sign stable +/-20% on {primary_market_name}): signs={b3_signs}  -> pass={b3_pass}")
    print(f"B4 (ETH sign agrees):    -> pass={b4_pass}")
    print(f"B5 (0.40% fee survives): sign_survives={b5_survives} (gates only if B1 passed) -> pass={b5_pass}")
    print(f"pytest causality-strict: returncode={rc1}")
    print(f"pytest full suite:       returncode={rc2}")

    verdict = "PROMOTE-CANDIDATE" if (r2 <= 0.98 and b1_pass and b4_pass and b5_pass) else "NEGATIVE"
    print(f"\nVERDICT: {verdict}")
