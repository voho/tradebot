#!/usr/bin/env python
"""R-138 NOVEL branch: Nguyen-Wolf (2026) permutation test on `kelly_regime_v4`'s
edge-concentration claim, with the event set produced by a CAUSAL two-sided
CUSUM changepoint detector on each asset's OWN daily log-return series, rather
than the conservative branch's hand-picked `STRESS_EPISODES`.

The complete pre-registration for this round -- mechanism, four-question
justification, named failure modes, and the exact decision rule (C1/C2/C3) --
lives in `experiments/r138_shared.py`'s own module docstring, written before
this branch was dispatched. Read that file in full first. This file imports
ONLY from `experiments.r138_shared` (read-only), never edits it, never
coordinates with the conservative branch's file, and never reads a bar at or
after `r138_shared.OOS_START` (2023-01-01).

MECHANISM, exactly as frozen in `r138_shared.py`: run `causal_cusum_breaks`
on BTC's own `price_daily_logret(df)` (the raw close-price series, NOT the
strategy equity curve) at the frozen constants `CUSUM_TRAIL_DAYS=90,
CUSUM_K_MULT=0.5, CUSUM_H_MULT=5.0` to get BTC's event-date list; feed that
event set (and its actual count) into the identical `empirical_type1_rate`
(C1) and `permutation_test` (C2) machinery the conservative branch uses; then
repeat the WHOLE procedure independently on ETH (ETH's own CUSUM breaks on
ETH's own price series, ETH's own AR series) as the pre-registered C3
falsification test.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import r138_shared as shared  # noqa: E402


def _overlap_diagnostic(cusum_dates: list, tolerance_days: int = 3) -> list:
    """Pre-registered diagnostic only (does not gate anything): for each
    `STRESS_EPISODES` entry, report the nearest CUSUM date within
    `tolerance_days` calendar days, if any."""
    cusum_ts = sorted(pd.Timestamp(d).tz_localize(None) for d in cusum_dates)
    matches = []
    for name, date_str in shared.STRESS_EPISODES:
        target = pd.Timestamp(date_str)
        best = None
        best_gap = None
        for c in cusum_ts:
            gap = abs((c - target).days)
            if gap <= tolerance_days and (best_gap is None or gap < best_gap):
                best = c
                best_gap = gap
        if best is not None:
            matches.append((name, date_str, str(best.date()), best_gap))
    return matches


def run_market(df: pd.DataFrame, label: str, market: shared.MarketSpec,
               market_name: str) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  {market_name}")
    print(f"{'=' * 70}")

    # --- CUSUM event set on this asset's OWN daily log-return series ---
    logret = shared.price_daily_logret(df)
    breaks = shared.causal_cusum_breaks(
        logret,
        trail_days=shared.CUSUM_TRAIL_DAYS,
        k_mult=shared.CUSUM_K_MULT,
        h_mult=shared.CUSUM_H_MULT,
    )
    n_breaks = len(breaks)
    print(f"CUSUM breaks found: {n_breaks}")
    if breaks:
        print("  dates: " + ", ".join(str(pd.Timestamp(b).date()) for b in breaks))

    degenerate = None
    if n_breaks == 0:
        degenerate = "ZERO breaks -- CUSUM detector produced an empty event set"
    elif n_breaks == 1:
        degenerate = "ONE break -- N too small for a meaningful CAAR statistic"
    elif n_breaks > 100:
        degenerate = f"{n_breaks} breaks -- implausibly large, likely noise-dominated"
    if degenerate:
        print(f"NAMED FAILURE MODE: {degenerate}")

    # --- AR series: candidate minus vol-matched hold, on kelly_regime_v4 ---
    cand_log, matched_log, c, achieved_vol = shared.candidate_and_matched_daily_logret(
        df, market, label)
    ar = cand_log - matched_log
    print(f"AR series length: {len(ar)} days; matched hold c={c:.4f}, "
          f"achieved_vol={achieved_vol:.4f}")

    result = {
        "market_name": market_name,
        "n_breaks": n_breaks,
        "breaks": breaks,
        "degenerate": degenerate,
        "ar": ar,
    }

    if degenerate:
        result["c1"] = None
        result["c2"] = None
        return result

    # --- C1: calibration, using the ACTUAL CUSUM event count ---
    c1 = shared.empirical_type1_rate(ar, n_events=n_breaks)
    c1_pass = (np.isfinite(c1["rate"])
               and shared.CALIBRATION_BAND[0] <= c1["rate"] <= shared.CALIBRATION_BAND[1])
    print(f"C1 (calibration): rate={c1['rate']:.4f} "
          f"(n_trials={c1['n_trials']}, rejects={c1.get('rejects')}), "
          f"band={shared.CALIBRATION_BAND} -> {'PASS' if c1_pass else 'FAIL'}")
    result["c1"] = c1
    result["c1_pass"] = c1_pass

    # --- C2 / C3-equivalent: permutation test on the CUSUM event dates ---
    c2 = shared.permutation_test(ar, breaks)
    c2_sig = (np.isfinite(c2["pvalue"]) and c2["pvalue"] < shared.ALPHA
              and c2["n_exceed"] > 2)
    print(f"C2 (permutation): observed CAAR={c2['observed']:.6f}, "
          f"p={c2['pvalue']:.5f}, n_exceed={c2['n_exceed']}, "
          f"n_perm={c2['n_perm']}, pool_size={c2['pool_size']} "
          f"-> {'SIGNIFICANT' if c2_sig else 'not significant'} "
          f"(resolution-aware: needs p<{shared.ALPHA} AND n_exceed>2)")
    result["c2"] = c2
    result["c2_sig"] = c2_sig

    return result


def main() -> None:
    # ------------------------------------------------------------------
    # BTC
    # ------------------------------------------------------------------
    btc_df, btc_label = shared.load_btc_train()
    btc = run_market(btc_df, btc_label, shared.PRIMARY_MARKET, "BTC (primary)")

    print(f"\n{'-' * 70}")
    print("BTC CUSUM vs STRESS_EPISODES overlap diagnostic (+/-3 days, does not gate)")
    print(f"{'-' * 70}")
    overlap = _overlap_diagnostic(btc["breaks"])
    if overlap:
        for name, target, matched, gap in overlap:
            print(f"  MATCH: '{name}' ({target}) <-> CUSUM {matched} (gap {gap}d)")
    else:
        print("  No STRESS_EPISODES date is within +/-3 days of any CUSUM break.")
    print(f"  {len(overlap)}/{len(shared.STRESS_EPISODES)} STRESS_EPISODES overlapped.")

    # ------------------------------------------------------------------
    # ETH (C3 falsification -- independent CUSUM breaks, independent AR)
    # ------------------------------------------------------------------
    eth_df = shared.load_eth_train()
    eth = run_market(eth_df, "eth_spot", shared.PRIMARY_MARKET, "ETH (C3 falsification)")

    # ------------------------------------------------------------------
    # Decision rule, evaluated exactly as pre-registered in r138_shared.py
    # ------------------------------------------------------------------
    btc_c1_pass = bool(btc.get("c1_pass"))
    btc_c2_pass = bool(btc.get("c2_sig"))
    eth_c1_pass = bool(eth.get("c1_pass"))

    eth_c3_pass = False
    if eth_c1_pass and eth.get("c2") is not None and btc.get("c2") is not None:
        eth_pv = eth["c2"]["pvalue"]
        btc_sign = np.sign(btc["c2"]["observed"])
        eth_sign = np.sign(eth["c2"]["observed"])
        eth_c3_pass = (np.isfinite(eth_pv) and eth_pv < 0.10
                       and btc_sign != 0 and btc_sign == eth_sign)

    n_configs = 2  # one frozen procedure, applied to BTC and ETH -- no sweep.

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"BTC CUSUM breaks:            {btc['n_breaks']}"
          + (f"  [DEGENERATE: {btc['degenerate']}]" if btc["degenerate"] else ""))
    print(f"BTC C1 (calibration):        {'PASS' if btc_c1_pass else 'FAIL/VOID'}"
          + (f"  rate={btc['c1']['rate']:.4f}" if btc.get("c1") else ""))
    print(f"BTC C2 (significance):       {'PASS' if btc_c2_pass else 'FAIL'}"
          + (f"  p={btc['c2']['pvalue']:.5f}, n_exceed={btc['c2']['n_exceed']}"
             if btc.get("c2") else ""))
    print(f"ETH CUSUM breaks:            {eth['n_breaks']}"
          + (f"  [DEGENERATE: {eth['degenerate']}]" if eth["degenerate"] else ""))
    print(f"ETH C1 (calibration):        {'PASS' if eth_c1_pass else 'FAIL/VOID'}"
          + (f"  rate={eth['c1']['rate']:.4f}" if eth.get("c1") else ""))
    print(f"ETH C3 (falsification):      {'PASS' if eth_c3_pass else 'FAIL'}"
          + (f"  p={eth['c2']['pvalue']:.5f}" if eth.get("c2") else ""))
    print(f"Configurations evaluated:    {n_configs} "
          f"(one frozen procedure x {{BTC, ETH}}, no sweep)")

    if not btc_c1_pass:
        verdict = "VOIDED on BTC (C1 calibration failed) -- per pre-registration, VOID not score"
    elif not eth_c1_pass:
        verdict = ("BTC scored, but ETH is VOIDED (C1 calibration failed on ETH) -- "
                   "C3 cannot be evaluated, so overall NOT PROMOTABLE")
    elif btc_c1_pass and btc_c2_pass and eth_c3_pass:
        verdict = "PROMOTABLE (all of C1(BTC), C2, C3 pass)"
    else:
        verdict = "NEGATIVE/METHOD (decision rule not cleared -- see C1/C2/C3 above)"

    print(f"\nOVERALL VERDICT: {verdict}")


if __name__ == "__main__":
    main()
