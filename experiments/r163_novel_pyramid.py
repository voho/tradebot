"""R-163 NOVEL branch: continuous, two-sided, episode-relative excursion
multiplier on `kelly_regime_v4`'s own raw_desired exposure -- a tanh-shaped
pyramiding factor keyed on how far price has moved (favorably OR adversely)
since the CURRENT bullish episode began, with no discrete unit-stack or
stop-ratchet state machine (that is the sibling CONSERVATIVE branch, a
different file). All mechanism, constants and decision rule are frozen in
``experiments/r163_shared.py`` (read-only, never edited by this file); this
file only drives that pre-registration end to end and reports the result.

CONSTRUCTION UNDER TEST: ``build_novel_target(df, kappa)`` /
``novel_multiplier(df, kappa)`` (r163_shared.py):

    mult = 1 + kappa * tanh(excursion_atr_units / NOVEL_REF_ATR_UNITS)
    combined = clip(v4_raw_desired(df) * mult, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)

kappa=0.0 is the identity check (mult==1.0 everywhere -> bit-for-bit v4).
kappa=1.0 (NOVEL_PRIMARY) is the falsification-test / would-be-holdout
config. Genuinely two-sided: excursion_atr_units is signed (favorable OR
adverse move since episode start), so mult can push exposure both above
AND below v4's own raw_desired, unlike the conservative branch's add-only
unit stack.

STEPS (mirrors the dispatch instructions verbatim):
  1. Sanity/causality: kappa=0.0 reproduces v4_target bit-for-bit (A1) on a
     150k-bar subset; causal_truncation_probe_series passes at kappa=1.0 on
     the full pre-holdout BTC series (timed); novel_multiplier(kappa=1.0)'s
     own min/max/mean/std over BTC's full pre-holdout history, reported as
     a genuine finding about whether the two-sided construction is actually
     symmetric in practice.
  2. A2 kill switch: R^2 of clip(v4_raw_desired*novel_multiplier(PRIMARY),
     +-V4_MAX_LEVERAGE) vs v4_raw_desired, over BTC's full pre-holdout
     history, must be < CONST_CAP_R2_THRESH (0.98).
  3. Sweep: NOVEL_GRID x 2 markets x 3 slices via compare() = 24 cells.
  4. Fee-tier robustness: PRIMARY kappa, FEE_TIER=0.40%, both markets,
     inner_val only (compare() run then filtered) = 2 more cells. Total 26.
  5. Decision rule from r163_shared.py's own frozen table, applied to the
     inner-validation rows, plus the PRIMARY-config ETH sign-replication
     falsification test and an explicit A3 (exposure-match) discussion.
  6. Holdout is NOT read by this file under any circumstances -- that is
     an operator-authorized step after both branches report in.

USAGE
-----
    python experiments/r163_novel_pyramid.py
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
    CONST_CAP_R2_THRESH,
    FEE_TIER,
    FUTURES,
    NOVEL_GRID,
    NOVEL_PRIMARY,
    OOS_START,
    SPOT,
    V4_MAX_LEVERAGE,
    build_novel_target,
    causal_truncation_probe_series,
    clears_bar,
    compare,
    fee_at,
    load_btc,
    load_eth,
    novel_multiplier,
    print_rows,
    r_squared,
    v4_raw_desired,
    v4_target,
)

# 4 kappa values x 2 markets x 3 slices (24) + fee-tier re-run x 2 markets,
# inner_val only (2) = 26, matching r163_shared.py's own pre-registered count.
CONFIGS_EVALUATED = 26


# ================================================================== (1)
# Step 1: A1 identity + causality kill switches, and the multiplier's own
# raw summary statistics (a real finding, not just a diagnostic).
# ==================================================================

def step1_identity_and_causality() -> dict:
    btc_full = load_btc()

    # A1: kappa=0.0 must reproduce v4_target bit-for-bit. Smaller subset
    # for speed, per dispatch instructions.
    n_a1 = min(len(btc_full), 150_000)
    btc_a1 = btc_full.iloc[:n_a1].copy()
    novel_k0 = np.asarray(build_novel_target(btc_a1, 0.0), dtype=float)
    v4 = np.asarray(v4_target(btc_a1), dtype=float)
    identical = bool(np.array_equal(novel_k0, v4))
    max_abs_diff = float(np.max(np.abs(novel_k0 - v4))) if len(novel_k0) else float("nan")

    # Causal truncation probe at PRIMARY, on the FULL pre-holdout BTC series
    # (load_btc() is already truncated before OOS_START, i.e. exactly the
    # "full BTC inner-period" -- timed, since the dispatch instructions ask
    # for a fallback to a 300k+ subset if it proves too slow).
    t0 = time.time()
    causal_ok = causal_truncation_probe_series(
        lambda df: build_novel_target(df, NOVEL_PRIMARY), btc_full)
    causal_elapsed = time.time() - t0
    causal_n_bars = len(btc_full)

    # The multiplier's OWN summary stats over BTC's full pre-holdout
    # history -- this is a two-sided construction by design, so whether it
    # actually pushes exposure both up AND down in practice (vs leaning
    # one way) is a real, disclosed finding.
    mult_full = np.asarray(novel_multiplier(btc_full, NOVEL_PRIMARY), dtype=float)
    mult_stats_full = dict(
        min=float(np.min(mult_full)), max=float(np.max(mult_full)),
        mean=float(np.mean(mult_full)), std=float(np.std(mult_full)),
        frac_above_1=float(np.mean(mult_full > 1.0 + 1e-12)),
        frac_below_1=float(np.mean(mult_full < 1.0 - 1e-12)),
        frac_at_1=float(np.mean(np.abs(mult_full - 1.0) <= 1e-12)),
    )
    # Bonus, not required: the same stats restricted to bars where the
    # multiplier can actually move at all (bullish episodes) -- the
    # non-bullish majority is mechanically pinned at 1.0 and would dilute
    # the "is it symmetric" question if left in.
    bullish_mask = mult_full != 1.0  # mult==1.0 off-episode by construction
    # (note: mult can also legitimately equal 1.0 exactly AT an episode's
    # first bar, where excursion==0 -- this mask slightly undercounts that,
    # disclosed here rather than silently corrected.)
    if bullish_mask.any():
        mult_active = mult_full[bullish_mask]
        mult_stats_active = dict(
            min=float(np.min(mult_active)), max=float(np.max(mult_active)),
            mean=float(np.mean(mult_active)), std=float(np.std(mult_active)),
            n_bars=int(bullish_mask.sum()),
        )
    else:
        mult_stats_active = dict(min=float("nan"), max=float("nan"),
                                  mean=float("nan"), std=float("nan"), n_bars=0)

    return dict(
        identical=identical, max_abs_diff=max_abs_diff, n_bars_a1=n_a1,
        causal_ok=causal_ok, causal_elapsed=causal_elapsed, causal_n_bars=causal_n_bars,
        mult_stats_full=mult_stats_full, mult_stats_active=mult_stats_active,
    )


# ================================================================== (2)
# Step 2: A2 non-collinearity kill switch, over BTC's full pre-holdout
# history (the pre-registration names no particular slice for this check;
# the full non-holdout series is used here as the least ambiguous choice,
# disclosed explicitly).
# ==================================================================

def step2_a2_kill_switch(btc: pd.DataFrame) -> dict:
    raw = v4_raw_desired(btc)
    mult = novel_multiplier(btc, NOVEL_PRIMARY)
    cand_raw = np.clip(raw * mult, -V4_MAX_LEVERAGE, V4_MAX_LEVERAGE)
    r_sq = r_squared(cand_raw, raw)
    passes = bool(np.isfinite(r_sq) and r_sq < CONST_CAP_R2_THRESH)
    return dict(r_sq=r_sq, passes=passes, n_bars=len(btc))


# ================================================================== (3)+(4)
# Sweep + fee-tier robustness, both via the shared compare().
# ==================================================================

def step3_sweep(btc: pd.DataFrame, eth: pd.DataFrame) -> list[dict]:
    rows = []
    for kappa in NOVEL_GRID:
        rows.extend(compare(lambda df, k=kappa: build_novel_target(df, k),
                            label=f"novel_kappa{kappa}", btc=btc, eth=eth,
                            include_eth=True))
    return rows


def step4_fee_tier(btc: pd.DataFrame) -> list[dict]:
    """Re-run the PRIMARY kappa at FEE_TIER=0.40% on both markets, via the
    shared compare() (per dispatch instructions), then keep only the
    inner_val rows. compare() always runs both inner_train and inner_val
    internally (include_eth=False here so it does not also run ETH) -- the
    inner_train cells it computes as a side effect are discarded, disclosed
    here rather than hidden."""
    fee_markets = (fee_at(SPOT, FEE_TIER), fee_at(FUTURES, FEE_TIER))
    label = f"novel_kappa{NOVEL_PRIMARY}_fee{FEE_TIER:.4f}"
    rows_all = compare(lambda df: build_novel_target(df, NOVEL_PRIMARY),
                       label=label, btc=btc, markets=fee_markets, include_eth=False)
    rows = [r for r in rows_all if r["slice"] == "inner_val"]
    return rows


# ================================================================== (5)
# Decision rule, exactly as frozen in r163_shared.py's own module
# docstring, plus the PRIMARY-specific falsification test as spelled out in
# the dispatch instructions, plus an explicit A3 exposure-match discussion.
# ==================================================================

def _kappa_of(row: dict) -> float:
    # sweep labels are f"novel_kappa{kappa}" (kappa in NOVEL_GRID); fee-tier
    # rows are never passed into this helper.
    return float(row["label"].split("novel_kappa", 1)[1])


def step5_decision_rule(sweep_rows: list[dict]) -> dict:
    val_rows = [r for r in sweep_rows if r["slice"] == "inner_val"]
    eth_rows = [r for r in sweep_rows if r["slice"] == "eth_replication"]

    # CLEAR(m): at least one NON-ZERO kappa in the grid clears clears_bar()
    # on inner-validation, on market m.
    nonzero_val_clearing = [r for r in val_rows if _kappa_of(r) != 0.0 and clears_bar(r)]
    clear_spot = any(r["market"] == "spot" for r in nonzero_val_clearing)
    clear_futures = any(r["market"] == "futures_5x" for r in nonzero_val_clearing)
    clear_any = clear_spot or clear_futures

    # Plateau check: the non-zero kappa values that clear (any market) share
    # one sign of d_log_growth on inner-validation.
    nonzero_clear_signs = {np.sign(r["d_log_growth"]) for r in nonzero_val_clearing
                            if r["d_log_growth"] != 0}
    plateau_ok = len(nonzero_clear_signs) <= 1

    # Falsification test, PRIMARY-config specific, per the dispatch
    # instructions verbatim: "for the PRIMARY config, on whichever market
    # cleared on inner-validation, does the ETH-replication slice's
    # d_log_growth at the SAME config carry the SAME SIGN?" -- this checks
    # only the market(s) where kappa=NOVEL_PRIMARY itself cleared, not every
    # market where some OTHER non-zero kappa happened to clear.
    primary_val_rows = [r for r in val_rows if _kappa_of(r) == NOVEL_PRIMARY]
    primary_clears = [r for r in primary_val_rows if clears_bar(r)]

    def eth_row_for(market: str):
        for r in eth_rows:
            if _kappa_of(r) == NOVEL_PRIMARY and r["market"] == market:
                return r
        return None

    eth_checks = []
    eth_pass = True  # vacuously true if PRIMARY clears nowhere
    for r in primary_clears:
        eth_r = eth_row_for(r["market"])
        same_sign = (eth_r is not None and
                     np.sign(eth_r["d_log_growth"]) == np.sign(r["d_log_growth"]) and
                     np.sign(r["d_log_growth"]) != 0)
        eth_checks.append(dict(market=r["market"],
                               btc_d_log_growth=r["d_log_growth"],
                               eth_d_log_growth=eth_r["d_log_growth"] if eth_r is not None else None,
                               same_sign=bool(same_sign)))
        if not same_sign:
            eth_pass = False

    gate_ok = bool(clear_any and eth_pass and plateau_ok)

    if not gate_ok:
        verdict = "REJECT"
    elif not clear_spot and not clear_futures:
        verdict = "REJECT"
    elif clear_spot and clear_futures:
        verdict = "PROMOTE"
    else:
        verdict = "PARTIAL"

    # A3 (disclosed, not a kill switch): risk_matched / exposure_ratio /
    # vol_ratio for PRIMARY's own inner-val rows, both markets -- explicit,
    # since this branch is two-sided by design and might self-normalize
    # exposure better than the conservative (add-only) branch, or might not.
    primary_val_risk = [
        dict(market=r["market"], exposure_ratio=r["exposure_ratio"],
             vol_ratio=r["vol_ratio"], risk_matched=r["risk_matched"],
             d_sharpe=r["d_sharpe"], d_dd=r["d_dd"], d_log_growth=r["d_log_growth"])
        for r in primary_val_rows
    ]

    return dict(
        val_rows=val_rows, nonzero_val_clearing=nonzero_val_clearing,
        clear_spot=clear_spot, clear_futures=clear_futures, clear_any=clear_any,
        plateau_ok=plateau_ok, primary_clears=primary_clears,
        eth_checks=eth_checks, eth_pass=eth_pass,
        gate_ok=gate_ok, verdict=verdict,
        primary_val_risk=primary_val_risk,
    )


# ================================================================== main
# ==================================================================

def main() -> None:
    print("=" * 100)
    print("R-163 NOVEL: continuous two-sided episode-relative excursion multiplier (kelly_regime_v4)")
    print("=" * 100)

    print("\n--- Step 1: A1 identity + causality kill switches, multiplier own-stats ---")
    s1 = step1_identity_and_causality()
    print(f"  kappa=0.0 bit-for-bit v4_target: {s1['identical']} "
          f"(max abs diff={s1['max_abs_diff']:.3e}, n_bars={s1['n_bars_a1']:,})")
    print(f"  causal_truncation_probe_series(build_novel_target, kappa={NOVEL_PRIMARY}) "
          f"over full pre-holdout BTC (n_bars={s1['causal_n_bars']:,}): "
          f"{'PASS' if s1['causal_ok'] else 'FAIL'} in {s1['causal_elapsed']:.1f}s")
    a1_ok = s1["identical"] and s1["causal_ok"]
    if not a1_ok:
        print("  A1/causality FAILED -- stopping. This branch cannot be trusted further.")
        return

    mf = s1["mult_stats_full"]
    ma = s1["mult_stats_active"]
    print(f"  novel_multiplier(kappa={NOVEL_PRIMARY}) over BTC's FULL pre-holdout history "
          f"(all bars, including non-bullish bars pinned at mult=1.0):")
    print(f"    min={mf['min']:.6f} max={mf['max']:.6f} mean={mf['mean']:.6f} std={mf['std']:.6f}")
    print(f"    frac(mult>1)={mf['frac_above_1']:.4f}  frac(mult<1)={mf['frac_below_1']:.4f}  "
          f"frac(mult==1, exactly)={mf['frac_at_1']:.4f}")
    print(f"  Same, restricted to bars where mult != 1.0 (i.e. inside an active bullish "
          f"episode away from its own start, n_bars={ma['n_bars']:,}):")
    print(f"    min={ma['min']:.6f} max={ma['max']:.6f} mean={ma['mean']:.6f} std={ma['std']:.6f}")

    print("\n--- Step 2: A2 non-collinearity kill switch (BTC full pre-holdout history) ---")
    btc = load_btc()
    s2 = step2_a2_kill_switch(btc)
    print(f"  R^2(clip(v4_raw_desired*novel_multiplier(PRIMARY)), v4_raw_desired) = "
          f"{s2['r_sq']:.6f} (n_bars={s2['n_bars']:,}); threshold < {CONST_CAP_R2_THRESH}")
    print(f"  A2 kill switch: {'PASS (non-collinear)' if s2['passes'] else 'FAIL (relabeling of v4)'}")
    if not s2["passes"]:
        print("  NOTE: proceeding to run and report the sweep honestly per instructions, "
              "but any Sharpe/growth differences below must NOT be interpreted as a real "
              "mechanism effect -- A2 failure means the candidate's exposure path is a "
              "near-exact rescale of v4's own.")

    print("\n--- Step 3: sweep (NOVEL_GRID x 2 markets x 3 slices = 24 cells) ---")
    eth = load_eth()
    sweep_rows = step3_sweep(btc, eth)
    print_rows(sweep_rows)

    print(f"\n--- Step 4: fee-tier robustness (kappa={NOVEL_PRIMARY}, FEE_TIER={FEE_TIER:.2%}, "
          f"inner_val only, 2 cells) ---")
    fee_rows = step4_fee_tier(btc)
    print_rows(fee_rows)

    all_rows = sweep_rows + fee_rows
    print(f"\nTotal configurations evaluated: {len(all_rows)} "
          f"(pre-registered count: {CONFIGS_EVALUATED})")
    assert len(all_rows) == CONFIGS_EVALUATED, (
        f"config count mismatch: got {len(all_rows)}, expected {CONFIGS_EVALUATED}")

    print("\n--- Step 5: decision rule (r163_shared.py's own frozen table) ---")
    dr = step5_decision_rule(sweep_rows)
    print(f"  CLEAR(spot)    = {dr['clear_spot']}")
    print(f"  CLEAR(futures) = {dr['clear_futures']}")
    print(f"  inner-val cells, non-zero kappa, clears_bar()==True:")
    if dr["nonzero_val_clearing"]:
        for r in dr["nonzero_val_clearing"]:
            print(f"    label={r['label']:20s} market={r['market']:11s} "
                  f"d_sharpe={r['d_sharpe']:+.4f} d_log_growth={r['d_log_growth']:+.6f} "
                  f"excludes_zero={r['excludes_zero']} risk_matched={r['risk_matched']}")
    else:
        print("    (none)")
    print(f"  plateau_ok (clearing non-zero-kappa cells share one sign of d_log_growth) = "
          f"{dr['plateau_ok']}")

    print(f"\n  Falsification test (PRIMARY config kappa={NOVEL_PRIMARY} only, per dispatch "
          f"instructions): on whichever market(s) PRIMARY itself clears clears_bar() on "
          f"inner-val, does eth_replication at the SAME kappa carry the SAME SIGN of "
          f"d_log_growth?")
    print(f"  PRIMARY clears on inner-val: "
          f"{[r['market'] for r in dr['primary_clears']] if dr['primary_clears'] else '(none)'}")
    if dr["eth_checks"]:
        for c in dr["eth_checks"]:
            print(f"    market={c['market']:11s} btc_d_log_growth={c['btc_d_log_growth']:+.6f} "
                  f"eth_d_log_growth={c['eth_d_log_growth']:+.6f} same_sign={c['same_sign']}")
    else:
        print("    (n/a -- PRIMARY did not clear on any market)")
    print(f"  eth_pass (no sign inversion on any market PRIMARY cleared) = {dr['eth_pass']}")
    print(f"  GATE_OK = {dr['gate_ok']}")
    print(f"  VERDICT = {dr['verdict']}")

    print(f"\n  A3 exposure-match discussion (disclosed, NOT a kill switch -- this "
          f"construction is NOT expected to pass A3 per the pre-registration's failure "
          f"mode (1), but as a genuinely two-sided/symmetric construction it might "
          f"self-normalize better than the conservative add-only branch):")
    for r in dr["primary_val_risk"]:
        print(f"    market={r['market']:11s} exposure_ratio={r['exposure_ratio']:.4f} "
              f"vol_ratio={r['vol_ratio']:.4f} risk_matched={r['risk_matched']} "
              f"d_sharpe={r['d_sharpe']:+.4f} d_dd={r['d_dd']:+.4f} "
              f"d_log_growth={r['d_log_growth']:+.6f}")
    ratios = [r["exposure_ratio"] for r in dr["primary_val_risk"] if np.isfinite(r["exposure_ratio"])]
    if ratios:
        mean_dev = float(np.mean([abs(x - 1.0) for x in ratios]))
        print(f"    mean |exposure_ratio - 1| across PRIMARY's inner-val cells = {mean_dev:.4f} "
              f"-- {'closer to' if mean_dev < 0.10 else 'still materially off'} exposure-matched "
              f"(A3 band is [0.9, 1.1], i.e. |ratio-1|<=0.10); the full-history multiplier mean "
              f"reported in Step 1 (mean={mf['mean']:.6f}) is the mechanical reason: a mean near "
              f"1.0 would predict the two-sided symmetry partially cancels out the mechanical "
              f"exposure inflation that an add-only construction cannot avoid, a mean above 1.0 "
              f"would predict it still inflates, just less than pure add-only would.")

    print("\n--- Step 6: holdout ---")
    print("  NOT read by this file, regardless of the Step 5 verdict above. If the verdict is "
          "PARTIAL or PROMOTE, the frozen PRIMARY config (kappa="
          f"{NOVEL_PRIMARY}) is ready for an operator-authorized holdout read "
          f"(ev(..., start={OOS_START!r})) as the next step -- not run here.")

    print("\n" + "=" * 100)
    print(f"FINAL VERDICT: {dr['verdict']}  |  A1: {'PASS' if a1_ok else 'FAIL'}  |  "
          f"A2 kill switch: {'PASS' if s2['passes'] else 'FAIL (relabeling risk)'}  |  "
          f"configs evaluated: {len(all_rows)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
