"""R-164 NOVEL branch: Daniel & Moskowitz (2016) "momentum crashes"
panic/calm-bull two-sided multiplier applied to `kelly_regime_v4`'s SCALE
slot. All mechanism, constants and decision rule are frozen in
``experiments/r164_shared.py`` (read-only, never edited by this file); this
file only drives that pre-registration end to end and reports the result.

CONSTRUCTION UNDER TEST: ``build_novel_target(df, kappa)`` /
``dm_multiplier(df, kappa)`` (r164_shared.py):

    score = calm_bull_component - panic_component     (dm_panic_calm_score)
    mult  = 1 + kappa * tanh(score / DM_SCORE_REF), clipped to NOVEL_MULT_CLIP
    combined = clip(v4_raw_desired(df) * mult, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)

kappa=0.0 is the identity check (mult==1.0 everywhere -> bit-for-bit v4).
kappa=1.0 (NOVEL_PRIMARY) is the falsification-test / would-be-holdout
config.

The mechanism's own docstring (`trailing_vol_ratio` in r164_shared.py)
claims an earlier draft had a same-day lookahead bug (only the median leg
was shifted, not the raw vol numerator) that was caught by
`causal_truncation_probe_series` and fixed before freezing. That claim is
RE-VERIFIED here by running the probe again, not trusted on the docstring's
word alone (Step 2 below).

STEPS (numbered per the dispatch instructions):
  1. Imports (this section).
  2. Causal-truncation probe on `build_novel_target` at PRIMARY -- re-verify
     the disclosed lookahead bug is actually fixed.
  3. A1 kill switch: kappa=0.0 reproduces v4_target bit-for-bit.
  4. A2 kill switch: R^2 of PRIMARY's raw pre-deadband exposure path vs
     v4_raw_desired must be < CONST_CAP_R2_THRESH (0.98).
  5. Full sweep: NOVEL_KAPPA_GRID x 2 markets x 3 slices via compare() = 24
     cells, printed table, CLEAR(spot)/CLEAR(futures) via clears_bar().
  6. ETH falsification test at PRIMARY, on whichever market cleared.
  7. 0.40% fee-tier re-run of PRIMARY on both markets (inner_val only, 2
     cells) -> 26 configurations total.
  8. Disclosed activity diagnostic (`dm_activity_diagnostic`, failure mode
     6) on BTC pre-holdout -- reported, not folded into the headline.
  9. Closing summary applying the pre-registered decision table verbatim,
     plus the A3 exposure/vol-ratio diagnostic for every cell.

Holdout is NOT read by this file under any circumstances -- that is an
operator-authorized step after both R-164 branches report in.

USAGE
-----
    python experiments/r164_novel_dm.py
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

from experiments.r164_shared import (  # noqa: E402
    CONST_CAP_R2_THRESH,
    FEE_TIER,
    FUTURES,
    NOVEL_KAPPA_GRID,
    NOVEL_PRIMARY,
    OOS_START,
    SPOT,
    V4_MAX_LEVERAGE,
    broadcast_daily_lambda,
    build_novel_target,
    causal_truncation_probe_series,
    clears_bar,
    compare,
    dm_activity_diagnostic,
    dm_multiplier,
    fee_at,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    v4_raw_desired,
    v4_target,
)

# 4 kappa values x 2 markets x 3 slices (24) + fee-tier re-run x 2 markets,
# inner_val only (2) = 26, matching r164_shared.py's own pre-registered count.
CONFIGS_EVALUATED = 26


# ================================================================== (2)
# Step 2: causal-truncation re-verification. The module docstring for
# `trailing_vol_ratio` claims a prior draft's same-day lookahead bug was
# caught and fixed -- re-run the probe here rather than trust that claim.
# ==================================================================

def step2_causal_probe() -> dict:
    btc_full = load_btc()
    t0 = time.time()
    causal_ok = causal_truncation_probe_series(
        lambda df: build_novel_target(df, NOVEL_PRIMARY), btc_full)
    elapsed = time.time() - t0
    return dict(causal_ok=causal_ok, elapsed=elapsed, n_bars=len(btc_full))


# ================================================================== (3)
# Step 3: A1 identity kill switch -- kappa=0.0 must reproduce v4_target
# bit-for-bit (R^2==1.0, max abs diff==0.0).
# ==================================================================

def step3_a1_identity(btc: pd.DataFrame) -> dict:
    novel_k0 = np.asarray(build_novel_target(btc, 0.0), dtype=float)
    v4 = np.asarray(v4_target(btc), dtype=float)
    identical = bool(np.array_equal(novel_k0, v4))
    max_abs_diff = float(np.max(np.abs(novel_k0 - v4))) if len(novel_k0) else float("nan")
    r_sq = r_squared(novel_k0, v4)
    return dict(identical=identical, max_abs_diff=max_abs_diff, r_sq=r_sq, n_bars=len(btc))


# ================================================================== (4)
# Step 4: A2 non-collinearity kill switch, PRIMARY (kappa=1.0), over BTC's
# full pre-holdout history (the pre-registration names no particular slice
# for this check; the full non-holdout series is the least ambiguous
# choice, disclosed here, matching r163's own precedent).
# ==================================================================

def step4_a2_kill_switch(btc: pd.DataFrame) -> dict:
    raw = v4_raw_desired(btc)
    mult_daily = dm_multiplier(btc, NOVEL_PRIMARY)
    mult_bars = broadcast_daily_lambda(mult_daily, btc.index)
    cand_raw = np.clip(np.asarray(raw, dtype=float) * mult_bars, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)
    r_sq = r_squared(cand_raw, raw)
    passes = bool(np.isfinite(r_sq) and r_sq < CONST_CAP_R2_THRESH)
    return dict(r_sq=r_sq, passes=passes, n_bars=len(btc))


# ================================================================== (5)
# Step 5: full sweep, via the shared compare().
# ==================================================================

def step5_sweep(btc: pd.DataFrame, eth: pd.DataFrame) -> list[dict]:
    rows = []
    for kappa in NOVEL_KAPPA_GRID:
        rows.extend(compare(lambda df, k=kappa: build_novel_target(df, k),
                            label=f"novel_kappa{kappa}", btc=btc, eth=eth,
                            include_eth=True))
    return rows


def _kappa_of(row: dict) -> float:
    return float(row["label"].split("novel_kappa", 1)[1])


def clear_flags(sweep_rows: list[dict]) -> dict:
    val_rows = [r for r in sweep_rows if r["slice"] == "inner_val"]
    nonzero_val_clearing = [r for r in val_rows if _kappa_of(r) != 0.0 and clears_bar(r)]
    clear_spot = any(r["market"] == "spot" for r in nonzero_val_clearing)
    clear_futures = any(r["market"] == "futures_5x" for r in nonzero_val_clearing)
    nonzero_clear_signs = {np.sign(r["d_log_growth"]) for r in nonzero_val_clearing
                            if r["d_log_growth"] != 0}
    plateau_ok = len(nonzero_clear_signs) <= 1
    return dict(val_rows=val_rows, nonzero_val_clearing=nonzero_val_clearing,
                clear_spot=clear_spot, clear_futures=clear_futures,
                plateau_ok=plateau_ok)


# ================================================================== (6)
# Step 6: ETH falsification test, PRIMARY config only, per the pre-
# registration verbatim: on whichever market PRIMARY clears clears_bar() on
# inner-validation, does eth_replication at the SAME config carry the SAME
# SIGN of d_log_growth?
# ==================================================================

def step6_eth_falsification(sweep_rows: list[dict]) -> dict:
    val_rows = [r for r in sweep_rows if r["slice"] == "inner_val"]
    eth_rows = [r for r in sweep_rows if r["slice"] == "eth_replication"]
    primary_val_rows = [r for r in val_rows if _kappa_of(r) == NOVEL_PRIMARY]
    primary_clears = [r for r in primary_val_rows if clears_bar(r)]

    def eth_row_for(market: str):
        for r in eth_rows:
            if _kappa_of(r) == NOVEL_PRIMARY and r["market"] == market:
                return r
        return None

    checks = []
    eth_pass = True  # vacuously true if PRIMARY clears nowhere
    for r in primary_clears:
        eth_r = eth_row_for(r["market"])
        same_sign = (eth_r is not None and
                     np.sign(eth_r["d_log_growth"]) == np.sign(r["d_log_growth"]) and
                     np.sign(r["d_log_growth"]) != 0)
        checks.append(dict(market=r["market"],
                           btc_d_log_growth=r["d_log_growth"],
                           eth_d_log_growth=eth_r["d_log_growth"] if eth_r is not None else None,
                           same_sign=bool(same_sign)))
        if not same_sign:
            eth_pass = False
    return dict(primary_clears=primary_clears, checks=checks, eth_pass=eth_pass)


# ================================================================== (7)
# Step 7: 0.40% fee-tier re-run of PRIMARY, both markets, inner_val only.
# ==================================================================

def step7_fee_tier(btc: pd.DataFrame) -> list[dict]:
    """compare() always computes inner_train and inner_val internally
    (include_eth=False here so it does not also run ETH); the inner_train
    cells it produces as a side effect are discarded, disclosed here rather
    than hidden, matching r163's own precedent."""
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    label = f"novel_kappa{NOVEL_PRIMARY}_fee{FEE_TIER:.4f}"
    rows_all = compare(lambda df: build_novel_target(df, NOVEL_PRIMARY),
                       label=label, btc=btc, markets=fee_markets, include_eth=False)
    rows = [r for r in rows_all if r["slice"] == "inner_val"]
    return rows


# ================================================================== (9)
# Step 9: decision-table verdict, applied verbatim from r164_shared.py's own
# docstring:
#   GATE_OK = ETH falsification passes AND plateau_ok
#   | GATE_OK | CLEAR(spot) | CLEAR(futures) | Verdict  |
#   | false   | --          | --              | REJECT   |
#   | true    | false       | false           | REJECT   |
#   | true    | true        | false (or v.v.) | PARTIAL  |
#   | true    | true        | true            | PROMOTE  |
# ==================================================================

def step9_decision(cf: dict, eth: dict) -> str:
    gate_ok = bool(eth["eth_pass"] and cf["plateau_ok"])
    if not gate_ok:
        return "REJECT"
    if not cf["clear_spot"] and not cf["clear_futures"]:
        return "REJECT"
    if cf["clear_spot"] and cf["clear_futures"]:
        return "PROMOTE"
    return "PARTIAL"


# ================================================================== main
# ==================================================================

def main() -> None:
    print("=" * 100)
    print("R-164 NOVEL: Daniel & Moskowitz (2016) panic/calm-bull two-sided multiplier (kelly_regime_v4)")
    print("=" * 100)

    btc = load_btc()

    print("\n--- Step 2: causal-truncation re-verification (re-check the disclosed "
          "trailing_vol_ratio lookahead-fix claim, not trust the docstring) ---")
    s2 = step2_causal_probe()
    print(f"  causal_truncation_probe_series(build_novel_target, kappa={NOVEL_PRIMARY}) "
          f"over full pre-holdout BTC (n_bars={s2['n_bars']:,}): "
          f"{'PASS' if s2['causal_ok'] else 'FAIL'} in {s2['elapsed']:.1f}s")
    if not s2["causal_ok"]:
        print("  CAUSALITY PROBE FAILED -- stopping. This branch cannot be trusted further.")
        return

    print("\n--- Step 3: A1 identity kill switch (kappa=0.0 vs v4_target) ---")
    s3 = step3_a1_identity(btc)
    print(f"  identical (bit-for-bit): {s3['identical']}  max_abs_diff={s3['max_abs_diff']:.3e}  "
          f"R^2={s3['r_sq']:.10f}  n_bars={s3['n_bars']:,}")
    a1_ok = s3["identical"] and s3["max_abs_diff"] == 0.0 and s3["r_sq"] == 1.0
    assert a1_ok, "A1 kill switch FAILED: kappa=0.0 does not reproduce v4_target bit-for-bit"
    print("  A1: PASS")

    print("\n--- Step 4: A2 non-collinearity kill switch (PRIMARY, BTC full pre-holdout) ---")
    s4 = step4_a2_kill_switch(btc)
    print(f"  R^2(clip(v4_raw_desired*dm_multiplier(PRIMARY)), v4_raw_desired) = "
          f"{s4['r_sq']:.6f} (n_bars={s4['n_bars']:,}); threshold < {CONST_CAP_R2_THRESH}")
    assert s4["passes"], "A2 kill switch FAILED: PRIMARY is a near-exact rescale of v4_raw_desired"
    print("  A2: PASS (non-collinear)")

    print("\n--- Step 5: sweep (NOVEL_KAPPA_GRID x 2 markets x 3 slices = 24 cells) ---")
    eth = load_eth()
    sweep_rows = step5_sweep(btc, eth)
    print_rows(sweep_rows)
    cf = clear_flags(sweep_rows)
    print(f"\n  CLEAR(spot)    = {cf['clear_spot']}")
    print(f"  CLEAR(futures) = {cf['clear_futures']}")
    print(f"  plateau_ok (clearing non-zero-kappa cells share one sign of d_log_growth) = "
          f"{cf['plateau_ok']}")
    print("  inner-val cells, non-zero kappa, clears_bar()==True:")
    if cf["nonzero_val_clearing"]:
        for r in cf["nonzero_val_clearing"]:
            print(f"    label={r['label']:20s} market={r['market']:11s} "
                  f"d_sharpe={r['d_sharpe']:+.4f} d_log_growth={r['d_log_growth']:+.6f} "
                  f"excludes_zero={r['excludes_zero']} risk_matched={r['risk_matched']}")
    else:
        print("    (none)")

    print(f"\n--- Step 6: ETH falsification test (PRIMARY kappa={NOVEL_PRIMARY} only) ---")
    eth_res = step6_eth_falsification(sweep_rows)
    print(f"  PRIMARY clears clears_bar() on inner-val markets: "
          f"{[r['market'] for r in eth_res['primary_clears']] if eth_res['primary_clears'] else '(none)'}")
    if eth_res["checks"]:
        for c in eth_res["checks"]:
            print(f"    market={c['market']:11s} btc_d_log_growth={c['btc_d_log_growth']:+.6f} "
                  f"eth_d_log_growth={c['eth_d_log_growth']:+.6f} same_sign={c['same_sign']}")
    else:
        print("    (n/a -- PRIMARY did not clear on any market)")
    print(f"  eth_pass (no sign inversion on any market PRIMARY cleared) = {eth_res['eth_pass']}")

    print(f"\n--- Step 7: fee-tier robustness (kappa={NOVEL_PRIMARY}, FEE_TIER={FEE_TIER:.2%}, "
          f"inner_val only, 2 cells) ---")
    fee_rows = step7_fee_tier(btc)
    print_rows(fee_rows)

    all_rows = sweep_rows + fee_rows
    print(f"\nTotal configurations evaluated: {len(all_rows)} "
          f"(pre-registered count: {CONFIGS_EVALUATED})")
    assert len(all_rows) == CONFIGS_EVALUATED, (
        f"config count mismatch: got {len(all_rows)}, expected {CONFIGS_EVALUATED}")

    print("\n--- Step 8: disclosed activity diagnostic (failure mode 6, BTC pre-holdout, "
          "NOT folded into any headline number) ---")
    act = dm_activity_diagnostic(btc)
    print(f"  days_scored={act['days_scored']:,}  days_nonzero={act['days_nonzero']:,}  "
          f"fraction_nonzero={act['fraction_nonzero']:.4f}")
    print(f"  days_panic={act['days_panic']:,}  days_calm_bull={act['days_calm_bull']:,}")

    print("\n--- Step 9: decision-table verdict (r164_shared.py's own frozen table, verbatim) ---")
    gate_ok = bool(eth_res["eth_pass"] and cf["plateau_ok"])
    print(f"  GATE_OK = ETH-falsification passes AND plateau_ok = "
          f"{eth_res['eth_pass']} AND {cf['plateau_ok']} = {gate_ok}")
    verdict = step9_decision(cf, eth_res)
    print(f"  | GATE_OK={gate_ok!s:5s} | CLEAR(spot)={cf['clear_spot']!s:5s} | "
          f"CLEAR(futures)={cf['clear_futures']!s:5s} | -> VERDICT = {verdict} |")

    print(f"\n  A3 exposure/vol-ratio diagnostic (disclosed, NOT calibrated) -- every cell:")
    for r in sweep_rows + fee_rows:
        print(f"    label={r['label']:26s} slice={r['slice']:16s} market={r['market']:11s} "
              f"exposure_ratio={r['exposure_ratio']:.4f} vol_ratio={r['vol_ratio']:.4f} "
              f"risk_matched={r['risk_matched']}")

    print("\n--- Holdout ---")
    print("  NOT read by this file, regardless of the Step 9 verdict above. If the verdict is "
          "PARTIAL or PROMOTE, the frozen PRIMARY config (kappa="
          f"{NOVEL_PRIMARY}) is ready for an operator-authorized holdout read "
          f"(ev(..., start={OOS_START!r})) as the next step -- not run here.")

    print("\n" + "=" * 100)
    print(f"FINAL VERDICT: {verdict}  |  A1: PASS  |  A2 kill switch: PASS  |  "
          f"causal probe: PASS  |  configs evaluated: {len(all_rows)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
