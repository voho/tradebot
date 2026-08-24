"""R-104 CONSERVATIVE branch: static split-conformal Conformal Risk Control
(Angelopoulos et al. 2024, Algorithm 1) wrapped around kelly_regime_v4's raw
exposure, calibrated ONCE on inner-train and then frozen.

Reads experiments/r104_shared.py (READ-ONLY pre-registration + harness) and
implements the 10-step pre-registered plan for this branch. This file does
not modify r104_shared.py and does not touch the OOS holdout (>= 2023-01-01);
every data access goes through r104_shared's already-truncated load_btc() /
load_eth() and its own assert_no_holdout() calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r104_shared import (  # noqa: E402
    CRC_ALPHA,
    CRC_D_GRID,
    CRC_TAU_QUANTILE,
    FUTURES,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    apply_deadband,
    assert_no_holdout,
    bar_forward_loss,
    causal_truncation_probe_series,
    compare,
    crc_static_lambda,
    exceedance_rate,
    fee_at,
    load_btc,
    print_rows,
    r_squared,
    v4_raw_desired,
)

BARS_PER_DAY = 288


def calibrate_tau_fixed(df: pd.DataFrame, cal_end: str, q: float = CRC_TAU_QUANTILE) -> float:
    """Reimplementation of r104_shared.calibrate_tau's exact documented
    algorithm, working around a bug in the shared (READ-ONLY, not editable)
    module: `df.index < pd.Timestamp(cal_end, tz="UTC")` already returns a
    plain numpy bool ndarray (DatetimeIndex comparison never returns a
    pandas Series), so the shared function's own `mask.to_numpy()` raises
    `AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'` on
    every real call. This local copy is byte-for-byte the same recipe minus
    that one bad `.to_numpy()` call; see the calling code and the final
    report for the reproduction and the exact traceback."""
    e = v4_raw_desired(df)
    loss = bar_forward_loss(e, df["close"])
    mask = df.index < pd.Timestamp(cal_end, tz="UTC")
    cal_loss = loss[mask]
    assert len(cal_loss) > BARS_PER_DAY * 30, "calibration window too short"
    return float(np.quantile(cal_loss, q))

CAL_START = "2017-01-01"
CAL_END_EXCLUSIVE = "2019-01-01"   # same boundary calibrate_tau uses (cal_end)
CHECK_START = "2019-01-01"
CHECK_END_EXCLUSIVE = "2021-01-01"  # covers [2019-01-01, 2020-12-31]

print("=" * 78)
print("R-104 CONSERVATIVE: static split-conformal CRC over kelly_regime_v4")
print("=" * 78)

# --------------------------------------------------------------------- (1)
# tau: shared, fixed measurement gate. v4's own 99th-pct single-bar loss on
# the 2017-01-01 -> 2018-12-31 window, independently reproduced here via the
# identical recipe the novel/online branch uses (not imported cross-branch).
btc_df = load_btc()
assert_no_holdout(btc_df, "btc_df")

tau = calibrate_tau_fixed(btc_df, cal_end=CAL_END_EXCLUSIVE, q=0.99)
print(f"\n[1] tau = {tau:.6f}  "
      f"(v4's own 99th-pct single-bar log-loss on [{CAL_START}, {CAL_END_EXCLUSIVE}))")
print(f"    economic meaning: a bar counts as an 'exceedance' when v4's raw "
      f"exposure realizes a loss > {tau*100:.3f}% (log-return terms) on that bar.")

# --------------------------------------------------------------------- (2)
# Calibrate d (Angelopoulos et al. 2024, Algorithm 1). Calibration set = same
# window as tau: bars strictly before 2019-01-01. e = v4_raw_desired(btc_df)
# computed on the FULL df (causal, full rolling-window context) and THEN
# restricted to the calibration window, per the pre-registered recipe.
idx = btc_df.index
e_full = v4_raw_desired(btc_df)
close_full = btc_df["close"]

cal_mask = (idx >= pd.Timestamp(CAL_START, tz="UTC")) & (idx < pd.Timestamp(CAL_END_EXCLUSIVE, tz="UTC"))
e_cal = e_full[cal_mask]
close_cal = close_full[cal_mask]
print(f"\n[2] calibration window bars: {int(cal_mask.sum())} "
      f"({idx[cal_mask][0]} .. {idx[cal_mask][-1]})")


def loss_fn(d: float) -> np.ndarray:
    loss = bar_forward_loss(e_cal * (1.0 - d), close_cal)
    return (loss > tau).astype(float)


d_star = crc_static_lambda(loss_fn, grid=CRC_D_GRID, alpha=CRC_ALPHA)
print(f"    d_star = {d_star:.3f}")
if d_star == 0.0:
    print("    -> INERT by construction: v4's unmodified exposure already "
          "clears the target risk rate on the calibration set with room to "
          "spare (a legitimate negative result per the pre-registration, not a bug).")
else:
    print(f"    -> BINDING: calibration required discounting raw exposure by "
          f"{d_star*100:.1f}% to clear alpha={CRC_ALPHA} on the calibration set.")

# --------------------------------------------------------------------- (3)
# A0 gate: does the frozen (tau, d_star) construction hold on the CHECK
# window [2019-01-01, 2020-12-31] (inside inner-train, held out from
# calibration) and on inner-validation [2021-01-01, 2022-12-31]? Computed on
# the full-df v4_raw_desired (preserving rolling-window warmup context, same
# convention calibrate_tau itself uses) then masked to each window -- NOT by
# recomputing v4_raw_desired on a truncated df, which would introduce a
# warmup artifact unrelated to the CRC construction itself.
check_mask = (idx >= pd.Timestamp(CHECK_START, tz="UTC")) & (idx < pd.Timestamp(CHECK_END_EXCLUSIVE, tz="UTC"))
val_mask = (idx >= pd.Timestamp(INNER_VAL_START, tz="UTC")) & (idx < pd.Timestamp(OOS_START, tz="UTC"))

exposure_full_frozen = e_full * (1.0 - d_star)

check_rate = exceedance_rate(exposure_full_frozen[check_mask], close_full[check_mask], tau)
val_rate = exceedance_rate(exposure_full_frozen[val_mask], close_full[val_mask], tau)

print(f"\n[3] A0 measurement gate (target alpha = {CRC_ALPHA}):")
print(f"    CHECK window  [{CHECK_START}, {CHECK_END_EXCLUSIVE}) "
      f"({int(check_mask.sum())} bars): exceedance rate = {check_rate:.5f} "
      f"({check_rate/CRC_ALPHA:.2f}x alpha)")
print(f"    inner_val     [{INNER_VAL_START}, {INNER_VAL_END}] "
      f"({int(val_mask.sum())} bars): exceedance rate = {val_rate:.5f} "
      f"({val_rate/CRC_ALPHA:.2f}x alpha)")

for name, rate in (("CHECK", check_rate), ("inner_val", val_rate)):
    ratio = rate / CRC_ALPHA if CRC_ALPHA else float("nan")
    if ratio > 3.0 or ratio < (1.0 / 3.0):
        print(f"    -> A0 FINDING: {name} exceedance rate is {ratio:.2f}x alpha "
              f"(outside the [0.33x, 3x] pre-registered tolerance band) -- the "
              f"static calibration does NOT hold on this window even though it "
              f"is inside inner-train. This is diagnostic (not a hard kill for "
              f"this branch); proceeding to report the rest as pre-registered.")
    else:
        print(f"    -> A0 OK: {name} exceedance rate within [0.33x, 3x] alpha tolerance.")

# --------------------------------------------------------------------- (4)
# Build the candidate. d_star is the ONE frozen value from step 2; NOT
# recomputed per-slice.


def build_target(df: pd.DataFrame) -> np.ndarray:
    return apply_deadband(v4_raw_desired(df) * (1.0 - d_star))


# --------------------------------------------------------------------- (5)
# Causal truncation probe.
print("\n[5] causal truncation probe...")
try:
    ok = causal_truncation_probe_series(build_target, btc_df)
    print(f"    PASS: {ok}")
except AssertionError as exc:
    ok = False
    print(f"    FAIL: {exc}")

# --------------------------------------------------------------------- (6)
# A2 non-inertness kill switch, over the FULL pre-holdout BTC series.
v4_raw = v4_raw_desired(btc_df)
cand_raw = build_target(btc_df)
a2_r2 = r_squared(cand_raw, v4_raw)
print(f"\n[6] A2 kill switch: R^2(build_target(btc_df), v4_raw_desired(btc_df)) = {a2_r2:.6f}")
if np.isfinite(a2_r2) and a2_r2 >= 0.98:
    print("    -> VERDICT: this is the 27th SIZE-axis construction to collapse "
          "into 'a near-constant rescale of v4's own path'.")
else:
    print("    -> VERDICT: does NOT collapse into a near-constant rescale of "
          "v4's own path by the project's R^2 >= 0.98 convention.")

# --------------------------------------------------------------------- (7)
print("\n[7] compare() -- inner_train, inner_val, eth_replication x {spot, futures_5x}")
rows = compare(build_target, label="conservative_static_crc")
print_rows(rows)

# --------------------------------------------------------------------- (8)
print("\n[8] 0.40% taker fee tier, inner_val only")
rows_fee40 = compare(
    build_target,
    label="conservative_static_crc_fee40",
    markets=(fee_at(SPOT, 0.004), fee_at(FUTURES, 0.004)),
    include_eth=False,
)
print_rows(rows_fee40)

# --------------------------------------------------------------------- (9)
# Independent re-derivation of d_star: manually confirm it is the smallest
# grid point clearing the finite-sample-corrected risk bound, and that the
# grid point immediately below it fails the bound.
print("\n[9] independent re-derivation of d_star against CRC_D_GRID")


def manual_risk(d: float) -> float:
    L = loss_fn(d)
    n = len(L)
    return (float(np.sum(L)) + 1.0) / (n + 1.0)


grid = list(CRC_D_GRID)
star_idx = grid.index(d_star)
risk_at_star = manual_risk(d_star)
print(f"    risk_hat(d_star={d_star:.3f}) = {risk_at_star:.6f}  "
      f"(<= alpha={CRC_ALPHA}: {risk_at_star <= CRC_ALPHA})")
if star_idx > 0:
    d_below = grid[star_idx - 1]
    risk_below = manual_risk(d_below)
    print(f"    risk_hat(d_below={d_below:.3f}) = {risk_below:.6f}  "
          f"(<= alpha={CRC_ALPHA}: {risk_below <= CRC_ALPHA})")
    verify_ok = (risk_at_star <= CRC_ALPHA) and (risk_below > CRC_ALPHA)
    print(f"    VERIFY: d_star clears the bound AND the grid point below it "
          f"fails the bound -> {verify_ok}")
else:
    verify_ok = risk_at_star <= CRC_ALPHA
    print(f"    d_star is the smallest grid point (0.0) -- no lower grid point "
          f"exists to check; confirming risk_hat(0.0) <= alpha directly -> {verify_ok}")

# --------------------------------------------------------------------- summary
print("\n" + "=" * 78)
print("PROMOTION-BAR CHECK (mechanical, per pre-registration)")
print("=" * 78)


def cell(rows_, slice_name, market_name):
    for r in rows_:
        if r["slice"] == slice_name and r["market"] == market_name:
            return r
    return None


val_spot = cell(rows, "inner_val", SPOT.name)
val_fut = cell(rows, "inner_val", FUTURES.name)
eth_spot = cell(rows, "eth_replication", SPOT.name)
eth_fut = cell(rows, "eth_replication", FUTURES.name)
fee_val_spot = cell(rows_fee40, "inner_val", SPOT.name)
fee_val_fut = cell(rows_fee40, "inner_val", FUTURES.name)


def clears(r):
    if r is None:
        return False
    sharpe_favor = r["d_sharpe"] > 0.2
    boot_favor = r["excludes_zero"] and r["boot_d_loggrowth"] > 0
    return sharpe_favor or boot_favor


def consistent_sign(val_r, eth_r):
    if val_r is None or eth_r is None:
        return False
    val_sign = np.sign(val_r["d_sharpe"]) if val_r["d_sharpe"] != 0 else np.sign(val_r["boot_d_loggrowth"])
    eth_sign = np.sign(eth_r["d_sharpe"]) if eth_r["d_sharpe"] != 0 else np.sign(eth_r["boot_d_loggrowth"])
    return val_sign == 0 or eth_sign == 0 or val_sign == eth_sign


def not_worse_at_fee(r):
    if r is None:
        return False
    return r["d_sharpe"] >= -0.05 and r["boot_d_loggrowth"] >= -0.01


spot_clears = clears(val_spot)
fut_clears = clears(val_fut)
eth_consistent_spot = consistent_sign(val_spot, eth_spot)
eth_consistent_fut = consistent_sign(val_fut, eth_fut)
fee_ok_spot = not_worse_at_fee(fee_val_spot)
fee_ok_fut = not_worse_at_fee(fee_val_fut)

print(f"inner_val SPOT clears |dSharpe|>0.2-or-CI-excl-0 in candidate's favor: {spot_clears}")
print(f"inner_val FUTURES_5X clears same bar: {fut_clears}")
print(f"ETH replication sign consistent with inner_val (SPOT): {eth_consistent_spot}")
print(f"ETH replication sign consistent with inner_val (FUTURES_5X): {eth_consistent_fut}")
print(f"0.40% fee tier not worse than v4 (SPOT): {fee_ok_spot}")
print(f"0.40% fee tier not worse than v4 (FUTURES_5X): {fee_ok_fut}")

overall = (spot_clears and fut_clears and eth_consistent_spot and eth_consistent_fut
          and fee_ok_spot and fee_ok_fut)
print(f"\nOVERALL PROMOTION: {'YES' if overall else 'NO'}")
