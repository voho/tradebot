#!/usr/bin/env python
"""R-117 NOVEL branch: a 5-member Donchian-channel BREAKOUT ensemble
(Zarattini, Pagani & Barbon 2025's own "aggregate multiple lookback periods
into one signal" construction), substituted into `kelly_regime_v4`'s `frac`
slot only. `scale` (v4's own volatility-target factor) is left completely
UNTOUCHED, per R-62's finding that `scale` carries none of v4's signature --
only `frac`, the directional vote, does.

EXACT CONSTRUCTION (pre-registered, not fitted). LOOKBACKS = (10, 20, 40, 60,
80) calendar days -- 5 members spanning v4's own 20-80 day scope down to a
faster 10-day member. All <= 80 days, so `TargetStrategy`'s default 80-day
warmup (already correct for v4 itself) covers every member without
override. The candidate target is built by `r117_shared.make_donchian_target
(LOOKBACKS)`, which composes `donchian_ensemble_frac(df, LOOKBACKS) *
v4_scale(df)` (v4's own unchanged scale factor) and then applies v4's own
10% re-target deadband -- the identical slot convention every SIZE-axis
round since R-89 has used (substitute the vote/frac input only).

Full citation trail, non-duplication argument against the nine formal
regime-timing estimators (R-82...R-60), against R-105's anchor-ladder
ensemble (same SIZE-axis role, mean-crossing detector family, flipped axis
of variation), and against `hedge_experts`'s own single-lookback Donchian
expert, all live in `experiments/r117_shared.py`'s own module docstring
(read in full before this file was written); not re-derived here. This file
never edits, and never reads a bar at or after `r117_shared.OOS_START`
from, `r117_shared.py` or any other file under `experiments/` or `src/`.

=====================================================================
PRE-REGISTRATION (frozen before any real-data r_sq, exposure, disagreement,
or backtest number in this file was computed -- docs/ROUTINE.md steps 1-2).
Anything below later contradicted by what actually happened is stated in
the results section, not edited back into this banner.
=====================================================================

STEP-0 NON-DEGENERACY GATE: on BTC inner-train only (`INNER_TRAIN_START`..
`INNER_TRAIN_END`), `r_sq = r_squared(candidate_target, v4_target)` on that
slice. QUALIFY iff `r_sq < R2_THRESH` (0.98) -- i.e. the candidate is NOT
simply a near-exact relabeling of v4's own path. If `r_sq >= R2_THRESH`:
STOP, report NEGATIVE (Step-0 kill switch, "near-exact rescale of v4's own
path"), no promotion bar, no ETH, no holdout bar touched. Mean exposure
(candidate's own breakout-membership frac) and the sign-disagreement
fraction against v4's own target path are reported alongside as diagnostic
context only -- neither gates. A causal-truncation probe on
`make_donchian_target(LOOKBACKS)` runs before any other number in this file
is trusted; a failure is investigated as a bug first, per this project's
own precedence, never reported around.

PROMOTION BAR (only if Step-0 qualifies): the standard SIZE/ERR-axis bar
used since R-89, via `compare()` (inner_train + inner_val + eth_replication,
both SPOT and FUTURES).
  B1 (gating): `b1_from_inner_val` on inner_val, both markets.
  B2 (diagnostic ONLY, never gates): `b2_diagnostic`.
  B3 (plateau, gating): two alternative 4-member ensembles as robustness
     context -- (10, 20, 40, 80) and (20, 40, 60, 80) -- via `inner_val_rows`,
     both markets, alongside the primary 5-member cell's own inner_val rows
     (reused, not recomputed). PASS requires a directionally consistent
     (same-sign) majority across the resulting 6 cells (3 ensembles x 2
     markets), matching R-105 NOVEL's own B3 majority rule.
  B4 (ETH falsification, gating, pre-registered): `b4_eth_falsification`,
     require FULL pass (both markets same sign as BTC inner_val).
  B5 (fee-tier robustness, gating): `b5_fee_tier` at 0.40% taker, primary
     cell, BTC inner_val, both markets -- no sign reversal.
PROMOTE-candidate only if the causal-truncation probe passed AND B1 AND B3
AND B4 (full) AND B5 all hold (B2 is diagnostic-only). Default: NEGATIVE.
No threshold or decision rule is changed after seeing a number.

WHAT WOULD MAKE THIS FAIL, named in full in `r117_shared.py`'s own module
docstring: even if the Donchian ensemble's frac is not collinear with v4's
own vote, its disagreement with v4 could concentrate in ordinary chop
rather than in the historical stress episodes that actually move the
promotion bar (the R-87/R-104/R-105-shaped "real but inert" pattern).
Given zero of 26+ prior SIZE-axis constructions and zero of 5 prior
ERR-axis constructions have promoted, the pre-registered expectation is
another NEGATIVE, reported with the same honesty as every prior round
regardless of outcome.

CONFIGURATIONS EVALUATED IN THIS FILE (IF Step-0 qualifies): 6 (primary
cell's full `compare()`: inner_train x2 markets + inner_val x2 markets +
eth_replication x2 markets) + 4 (B3's two alternative ensembles x 2
markets, freshly computed; the 2 primary-ensemble inner_val cells are
reused from the primary `compare()`, not recomputed) + 2 (B5's 0.40% fee
tier, 2 markets) = 12 total. IF Step-0 does not qualify, this file stops
after the Step-0 diagnostics (0 backtests run -- Step-0 is pure path
arithmetic, no `compare()`/`run_slice()` call).

USAGE
-----
    python experiments/r117_novel_donchian_ensemble_size.py
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

from experiments.r117_shared import (  # noqa: E402
    BARS_PER_DAY,
    FUTURES,
    INNER_TRAIN_START,
    INNER_TRAIN_END,
    INNER_VAL_START,
    INNER_VAL_END,
    OOS_START,
    R2_THRESH,
    SHARPE_NOISE_FLOOR,
    SPOT,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    causal_truncation_probe_series,
    compare,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    print_plateau_table,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_target,
    v4_vote_frac,
    v4_scale,
    donchian_ensemble_frac,
    donchian_ensemble_target,
    make_donchian_target,
)

# ---------------------------------------------------------- pre-registered
LOOKBACKS = (10, 20, 40, 60, 80)          # primary 5-member ensemble (calendar days)
ALT_B3_LOOKBACKS = (
    (10, 20, 40, 80),                     # B3 robustness alt #1 (4-member)
    (20, 40, 60, 80),                     # B3 robustness alt #2 (4-member)
)
assert max(LOOKBACKS) <= 80, "every member must fit inside the default 80-day warmup"
for _lb in ALT_B3_LOOKBACKS:
    assert max(_lb) <= 80, _lb

PRIMARY_LABEL = f"donchian_ens_{'_'.join(str(l) for l in LOOKBACKS)}d"


# ================================================================== (1)
# Step-0 non-degeneracy gate: r_sq of the candidate's target path against
# v4's own target path, on BTC inner-train only. Mean exposure and sign-
# disagreement are diagnostic context only -- neither gates.
# ==================================================================

def step0_diagnostics(btc: pd.DataFrame, build_primary) -> dict:
    mask = np.asarray((btc.index >= pd.Timestamp(INNER_TRAIN_START, tz="UTC")) &
                      (btc.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")))
    n_bars = int(mask.sum())

    candidate_target = np.asarray(build_primary(btc), dtype=float)
    v4t = np.asarray(v4_target(btc), dtype=float)
    candidate_frac = donchian_ensemble_frac(btc, LOOKBACKS).to_numpy()

    r_sq = r_squared(candidate_target[mask], v4t[mask])

    mean_exposure = float(np.mean(candidate_frac[mask]))

    cand_pos = candidate_target[mask] > 1e-12
    v4_pos = v4t[mask] > 1e-12
    sign_disagree_frac = float(np.mean(cand_pos != v4_pos))

    qualifies = bool(np.isfinite(r_sq) and r_sq < R2_THRESH)

    return dict(n_bars=n_bars, r_sq=r_sq, mean_exposure=mean_exposure,
               sign_disagree_frac=sign_disagree_frac, qualifies=qualifies,
               candidate_target=candidate_target, v4_target=v4t)


def print_step0_table(step0: dict) -> None:
    print(f"\nSTEP-0 NON-DEGENERACY GATE (inner-train slice, {INNER_TRAIN_START} -> "
          f"{INNER_TRAIN_END}, {step0['n_bars']:,} bars)")
    print(f"QUALIFY = r_sq < {R2_THRESH} (candidate target path is NOT a near-exact "
          f"rescale of v4's own path)")
    print(f"  r_sq (candidate target vs v4 target, inner-train)   : {step0['r_sq']:.4f}")
    print(f"  qualifies                                            : "
          f"{'YES' if step0['qualifies'] else 'no'}")
    print(f"  [diagnostic only, does not gate]")
    print(f"  candidate mean exposure (breakout-membership frac)  : "
          f"{step0['mean_exposure']:.4f}")
    print(f"  sign-disagreement fraction (cand>0 vs v4>0)          : "
          f"{step0['sign_disagree_frac']:.4f}")


# ================================================================== (2)
# Promotion bar: B1 (gating), B2 (diagnostic only), B3 (gating plateau,
# alternative-ensemble sweep), B4 (gating falsification), B5 (gating fee
# tier). Structurally identical machinery to R-105 NOVEL's own, imported
# unchanged from r117_shared (itself re-exporting r105_shared).
# ==================================================================

def run_b3(inner_val_primary_rows: list[dict], btc: pd.DataFrame) -> tuple[dict, bool]:
    plateau_rows: dict[str, list[dict]] = {}
    plateau_rows[PRIMARY_LABEL] = [
        dict(label=PRIMARY_LABEL, market=r["market"], d_sharpe=r["d_sharpe"], d_dd=r["d_dd"],
             exposure_ratio=r["exposure_ratio"], vol_ratio=r["vol_ratio"],
             risk_matched=r["risk_matched"], boot_d_loggrowth=r["boot_d_loggrowth"],
             boot_lo=r["boot_lo"], boot_hi=r["boot_hi"], excludes_zero=r["excludes_zero"])
        for r in inner_val_primary_rows
    ]
    for alt in ALT_B3_LOOKBACKS:
        label = f"donchian_ens_{'_'.join(str(l) for l in alt)}d"
        build_alt = make_donchian_target(alt)
        plateau_rows[label] = inner_val_rows(build_alt, label, btc)

    same_sign_flags = [r["d_sharpe"] > 0 for rows in plateau_rows.values() for r in rows]
    b3_pass = (sum(same_sign_flags) >= len(same_sign_flags) / 2.0) if same_sign_flags else False
    return plateau_rows, b3_pass


def run_promotion_bar(build_primary, btc: pd.DataFrame, eth: pd.DataFrame) -> dict:
    hr(f"PROMOTION BAR -- PRIMARY CELL: {PRIMARY_LABEL}")
    print("compare() over inner_train / inner_val / eth_replication, SPOT + FUTURES:")
    rows = compare(build_primary, label=PRIMARY_LABEL, btc=btc, eth=eth,
                  markets=(SPOT, FUTURES), include_eth=True)
    print_rows(rows)

    inner_val_primary = [r for r in rows if r["slice"] == "inner_val"]
    eth_primary = [r for r in rows if r["slice"] == "eth_replication"]

    b1_pass, b1_cells = b1_from_inner_val(inner_val_primary)
    b2_pass, b2_cells = b2_diagnostic(inner_val_primary)
    b3_rows, b3_pass = run_b3(inner_val_primary, btc)
    b4_partial, b4_full, b4_cells = b4_eth_falsification(eth_primary, inner_val_primary)
    b5_pass, b5_cells = b5_fee_tier(build_primary, PRIMARY_LABEL, btc, inner_val_primary)

    all_pass = b1_pass and b3_pass and b4_full and b5_pass

    return dict(
        label=PRIMARY_LABEL, compare_rows=rows,
        inner_val_primary=inner_val_primary, eth_primary=eth_primary,
        b1_pass=b1_pass, b1_cells=b1_cells,
        b2_pass=b2_pass, b2_cells=b2_cells,
        b3_pass=b3_pass, b3_rows=b3_rows,
        b4_partial=b4_partial, b4_full=b4_full, b4_cells=b4_cells,
        b5_pass=b5_pass, b5_cells=b5_cells,
        all_pass=all_pass,
        n_configs_promotion_bar=6 + 4 + 2,
    )


# --------------------------------------------------------------------- main

def main() -> dict:
    t0 = time.time()
    max_ts_seen: list[pd.Timestamp] = []

    hr("R-117 NOVEL: DonchianEnsembleKellyV4 -- 5-member Donchian breakout "
       "ENSEMBLE substituted into v4's frac slot")
    print("mechanism: Donchian-channel range-breakout is a structurally different detector")
    print("family from v4's own mean-crossing anchor vote (range extremity vs distance-from-")
    print("mean). A pre-registered 5-member ensemble (10/20/40/60/80 calendar-day lookbacks,")
    print("Zarattini/Pagani/Barbon 2025's own multi-lookback aggregation) replaces v4's `frac`")
    print("(directional vote) input only; v4's own `scale` (volatility-target factor) is left")
    print("completely UNTOUCHED, per R-62's finding that scale carries none of v4's signature.")
    print(f"\nLOOKBACKS (primary, traded): {LOOKBACKS}")
    print(f"B3 alternative ensembles (robustness context only): {ALT_B3_LOOKBACKS}")
    print(f"TargetStrategy.warmup: {TargetStrategy.warmup:,} bars "
          f"({TargetStrategy.warmup / BARS_PER_DAY:g} calendar days) -- shared default, "
          f"unmodified; covers every ensemble member (max lookback = {max(LOOKBACKS)}d <= 80d).")

    btc = load_btc()
    max_ts_seen.append(btc.index.max())
    assert_no_holdout(btc, "main(): btc")
    print(f"\nBTC (spot dataset, truncated < {OOS_START}): {len(btc):,} bars, "
          f"{btc.index[0]} -> {btc.index[-1]}")

    build_primary = make_donchian_target(LOOKBACKS)
    assert build_primary.__name__ == PRIMARY_LABEL, (build_primary.__name__, PRIMARY_LABEL)

    # ============================================================= STEP 0
    hr("STEP 0 -- NON-DEGENERACY KILL SWITCH (run BEFORE any Sharpe/compare() number)")
    step0 = step0_diagnostics(btc, build_primary)
    print_step0_table(step0)

    hr("CAUSAL TRUNCATION PROBE (composed build_target, real BTC data, "
       "run before trusting any other number in this file)")
    print(f"causal_truncation_probe_series({build_primary.__name__}, btc):")
    try:
        probe_ok = causal_truncation_probe_series(build_primary, btc)
        print("  PASS")
    except AssertionError as e:
        probe_ok = False
        print(f"  FAIL: {e}")
    print(f"\ncausal-truncation probe PASS: {probe_ok}")

    if not step0["qualifies"]:
        hr("STEP-0 GATE: r_sq >= threshold -- STOPPING HERE")
        print(f"r_sq = {step0['r_sq']:.4f} >= {R2_THRESH}: the Donchian ensemble's target path")
        print("is a near-exact rescale of v4's own path on BTC inner-train -- it carries no")
        print("information beyond a relabeling of the shipped vote. Per this file's own")
        print("pre-registration, this Step-0 table (plus the causal-truncation probe above) is")
        print("the branch's ENTIRE product, reported NEGATIVE / stopped-at-Step-0. No")
        print("promotion-bar code runs, and no ETH data or bar on/after OOS_START is ever")
        print("touched.")

        hr("VERDICT")
        print(f"Step-0 (r_sq < {R2_THRESH}): FAIL (r_sq={step0['r_sq']:.4f})")
        print(f"causal-truncation probe: {probe_ok}")
        print("B1: NOT COMPUTED (Step-0 kill switch)")
        print("B2: NOT COMPUTED (Step-0 kill switch)")
        print("B3: NOT COMPUTED (Step-0 kill switch)")
        print("B4: NOT COMPUTED (Step-0 kill switch)")
        print("B5: NOT COMPUTED (Step-0 kill switch)")
        print("VERDICT: NEGATIVE (Step-0 kill switch, near-exact rescale of v4's own path)")

        n_configs = 0
        max_ts = max(max_ts_seen)
        print(f"\nconfigurations evaluated (total): {n_configs} "
              f"(Step-0 is pure path arithmetic, no compare()/run_slice() call ran)")
        print(f"max timestamp read anywhere in this branch: {max_ts}  "
              f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")
        print(f"\n[{time.time() - t0:.0f}s]")
        return dict(btc=btc, step0=step0, passed_step0=False, probe_ok=probe_ok,
                   n_configs=n_configs, max_ts=max_ts,
                   verdict="NEGATIVE (Step-0 kill switch)")

    print(f"\nSTEP-0 GATE: QUALIFIES (r_sq={step0['r_sq']:.4f} < {R2_THRESH}) -- "
          f"proceeding to the promotion bar.")

    eth = load_eth()
    max_ts_seen.append(eth.index.max())
    assert_no_holdout(eth, "main(): eth")
    print(f"\nETH: {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}  (< {OOS_START})")

    bar = run_promotion_bar(build_primary, btc, eth)

    hr("B1 -- inner-validation, both markets (dSharpe > +0.2 OR bootstrap excludes zero positively)")
    for c in bar["b1_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  PASS={c['passes']}")
    print(f"B1 PASS (both markets): {bar['b1_pass']}")

    hr("B2 -- diagnostic only (drawdown change / risk-matched), inner-validation, both markets")
    for c in bar["b2_cells"]:
        status = "VALID" if c["risk_matched"] else "VOID (not risk-matched)"
        print(f"  {c['market']:>9s}  d_dd={c['d_dd']:+.2f}pp  risk_matched={c['risk_matched']}  [{status}]")

    hr("B3 -- plateau: primary 5-member ensemble + 2 alternative 4-member ensembles, "
       "inner-validation, both markets")
    print_plateau_table(bar["b3_rows"])
    print(f"\nB3 (directionally consistent majority across the 6-cell grid): {bar['b3_pass']}")

    hr("B4 -- ETH falsification (pre-registered)")
    for c in bar["b4_cells"]:
        print(f"  {c['market']:>9s}  d_sharpe={c['d_sharpe']:+.4f}  "
              f"boot=[{c['boot_lo']:+.4f},{c['boot_hi']:+.4f}]  "
              f"same_sign_as_btc_inner_val={c['same_sign_as_btc']}")
    print(f"B4 PARTIAL PASS (>=1 market): {bar['b4_partial']}   B4 FULL PASS (both markets): {bar['b4_full']}")

    hr("B5 -- fee-tier survival summary (0.40% taker vs. standard-fee sign)")
    for c in bar["b5_cells"]:
        print(f"  {c['market']:>9s}  fee-tier d_sharpe={c['d_sharpe']:+.4f}  "
              f"standard-fee d_sharpe={c['base_d_sharpe']:+.4f}  "
              f"fee-tier boot_d_loggrowth={c['boot_d_loggrowth']:+.4f}  "
              f"standard-fee boot_d_loggrowth={c['base_boot_d_loggrowth']:+.4f}  "
              f"no_reversal={c['no_reversal']}")
    print(f"B5 PASS: {bar['b5_pass']}")

    hr("VERDICT")
    print(f"causal-truncation probe: {probe_ok}")
    print(f"B1={bar['b1_pass']}  B2=diagnostic-only  B3={bar['b3_pass']}  "
          f"B4_full={bar['b4_full']}  B5={bar['b5_pass']}")
    all_applicable_pass = (probe_ok and bar["b1_pass"] and bar["b3_pass"] and
                          bar["b4_full"] and bar["b5_pass"])
    verdict = "PROMOTE-candidate" if all_applicable_pass else "NEGATIVE"
    print(f"ALL APPLICABLE CLAUSES PASS: {all_applicable_pass}")
    print(f"VERDICT: {verdict}")
    if not probe_ok:
        print("NOTE: verdict driven (at least in part) by a causal-truncation probe failure -- "
              "a lookahead is a bug report first, per docs/ROUTINE.md's own precedence.")

    n_configs = bar["n_configs_promotion_bar"]
    max_ts = max(max_ts_seen)
    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(6 primary-cell compare() + 4 B3 alt-ensemble sweep [2 ensembles x 2 markets, "
          f"fresh] + 2 B5 fee-tier; primary's own 2 inner_val cells reused for B3, not "
          f"recounted)")
    print(f"max timestamp read anywhere in this branch: {max_ts}  "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz='UTC')})")

    print(f"\n[{time.time() - t0:.0f}s]")

    return dict(btc=btc, eth=eth, step0=step0, passed_step0=True, probe_ok=probe_ok,
               promotion_bar=bar, verdict=verdict, n_configs=n_configs, max_ts=max_ts)


if __name__ == "__main__":
    main()
