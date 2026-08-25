#!/usr/bin/env python
"""R-138 CONSERVATIVE branch: the Nguyen-Wolf (2026) small-N permutation test
applied off the shelf to the already-frozen, narrative-selected
``STRESS_EPISODES`` list (used verbatim -- not re-selected for this round)
as the event set, on ``kelly_regime_v4`` vs. its own realized-volatility-
matched constant-exposure hold, 5x futures (``PRIMARY_MARKET``).

Full literature grounding, the four-question justification, the named
failure modes, the falsification test, and the pre-registered C1/C2/C3
decision rule all live in ``experiments/r138_shared.py``'s own module
docstring (read in full before this file was written); not re-derived here
beyond the summary above. This file NEVER edits ``r138_shared.py`` (frozen,
shared with the parallel NOVEL branch, a disjoint file this session does not
touch), never edits ``src/tradebot/strategies/kelly_regime_v4.py``, and
never reads a bar at or after ``r138_shared.OOS_START`` (2023-01-01) from any
data source -- the shared loaders' own ``_assert_no_holdout`` guards this,
and this file constructs no bespoke slice that could bypass it.

MECHANISM (exact, per ``r138_shared.py``'s own frozen procedure -- nothing
new invented here):

1. BTC primary case. ``candidate_and_matched_daily_logret(df, PRIMARY_MARKET,
   label)`` on ``load_btc_train()``'s dataframe gives ``(cand_log,
   matched_log, c, achieved_vol)``. ``ar = cand_log - matched_log`` is the
   abnormal-return series the permutation machinery watches.
   - C1: ``empirical_type1_rate(ar, n_events=6)`` at the module's frozen
     defaults; must land in ``CALIBRATION_BAND = (0.02, 0.09)`` or BTC is
     VOIDED (not scored) per the pre-registration's void-don't-score rule.
   - C2: ``permutation_test(ar, [d for _, d in STRESS_EPISODES])`` at the
     module's frozen ``N_PERM``; passes only if ``pvalue < 0.05`` AND
     ``n_exceed > 2`` (both conditions, resolution-aware per the
     pre-registration -- guards against reading a floor-resolution result as
     more informative than it is).

2. ETH falsification case (C3), identical procedure, ``PRIMARY_MARKET`` on
   ``load_eth_train()``'s dataframe for consistency with BTC. ETH's data
   starts 2019-03, so the two 2018 ``STRESS_EPISODES`` entries (bear onset,
   bear bottom) are not coverable and are DROPPED (reported explicitly, not
   padded or substituted) -- the remaining four dates (2020-03 COVID crash,
   2021-11 top, 2022-05 Terra/Luna, 2022-11 FTX) are used verbatim. ETH's own
   C1 is run at ``n_events=4`` (the actual usable count, per the shared
   file's own convention for a detector/event-set whose count differs from
   6). C3 passes only if ETH's own C1 lands in the calibration band AND the
   ETH permutation p-value is ``< 0.10`` with the SAME SIGN as BTC's observed
   CAAR.

3. Diagnostics only (never gate anything): the vol-match solve's ``c`` and
   achieved vol on both BTC and ETH, and a count of configurations evaluated
   -- this round sweeps no parameter, it is one frozen procedure applied
   twice (BTC, ETH), so that count is deliberately small.

CONFIGURATIONS EVALUATED: 2 (BTC: one ``candidate_and_matched_daily_logret``
call, i.e. one ``kelly_regime_v4`` backtest + one vol-matched constant-
exposure-hold backtest, on ``PRIMARY_MARKET`` + ETH: the identical single
call, same market). C1's 200 calibration trials and C2's permutation draws
are NOT separate backtest configurations -- they are statistical resampling
of the SAME already-computed per-asset ``ar`` series (no strategy is re-run
to produce them), so they are not counted in this tally, matching how
``r131_conservative_turnover_band.py``'s own tally excludes its pure-numpy
wiring self-test.

DECISION RULE (pre-registered, verbatim from ``r138_shared.py``, unaltered
after seeing any number): promotable (worth writing up as a genuine
methodological result) only if ALL of C1(BTC), C2, C3 pass. If C1 fails on
either market, THAT MARKET IS VOIDED (not scored as negative -- a broken
instrument, per ROUTINE.md's void-don't-score rule) rather than folded into
a plain NEGATIVE. Anything else not clearing all three is NEGATIVE or
METHOD. This round produces no strategy code change regardless of outcome.

USAGE
-----
    python experiments/r138_conservative_stress_permutation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r138_shared as shared  # noqa: E402

PRIMARY_MARKET = shared.PRIMARY_MARKET
STRESS_EPISODES = shared.STRESS_EPISODES
CALIBRATION_BAND = shared.CALIBRATION_BAND
ALPHA = shared.ALPHA
N_PERM = shared.N_PERM
N_CALIBRATION_TRIALS = shared.N_CALIBRATION_TRIALS


# ================================================================== (1)
# Per-market run: candidate_and_matched_daily_logret -> ar -> C1 -> C2.
# ==================================================================

def run_market(name: str, df: pd.DataFrame, event_dates: list, n_events: int,
                label: str = "") -> dict:
    print(f"\n  [{name}] running kelly_regime_v4 + vol-matched constant-exposure "
          f"hold on {PRIMARY_MARKET.name if hasattr(PRIMARY_MARKET, 'name') else PRIMARY_MARKET}...")
    t0 = time.time()
    cand_log, matched_log, c, achieved_vol = shared.candidate_and_matched_daily_logret(
        df, PRIMARY_MARKET, label)
    ar = cand_log - matched_log
    print(f"    vol-match solve: c={c:.6f}  achieved_vol={achieved_vol:.6f}  "
          f"n_days(ar)={len(ar)}  [{time.time() - t0:.1f}s]")

    print(f"  [{name}] C1 -- empirical_type1_rate(ar, n_events={n_events}) "
          f"({N_CALIBRATION_TRIALS} trials x {N_PERM} perms)...")
    t0 = time.time()
    c1 = shared.empirical_type1_rate(ar, n_events=n_events)
    c1_pass = bool(np.isfinite(c1["rate"]) and CALIBRATION_BAND[0] <= c1["rate"] <= CALIBRATION_BAND[1])
    print(f"    C1 rate={c1['rate']:.4f}  (n_trials valid={c1['n_trials']}, "
          f"rejects={c1.get('rejects')})  band={CALIBRATION_BAND}  PASS={c1_pass}  "
          f"[{time.time() - t0:.1f}s]")

    print(f"  [{name}] C2 -- permutation_test(ar, {n_events} event dates) "
          f"(N_PERM={N_PERM})...")
    t0 = time.time()
    c2 = shared.permutation_test(ar, event_dates)
    c2_sig = bool(np.isfinite(c2["pvalue"]) and c2["pvalue"] < ALPHA)
    c2_resolved = bool(c2["n_exceed"] > 2)
    c2_pass = bool(c2_sig and c2_resolved)
    print(f"    observed CAAR={c2['observed']:+.6f}  pvalue={c2['pvalue']:.6f}  "
          f"n_exceed={c2['n_exceed']}  n_perm(used)={c2['n_perm']}  pool_size={c2['pool_size']}  "
          f"[{time.time() - t0:.1f}s]")
    print(f"    C2 significant (p<{ALPHA})={c2_sig}  resolution-aware (n_exceed>2)={c2_resolved}  "
          f"C2 PASS={c2_pass}")

    return dict(name=name, c=c, achieved_vol=achieved_vol, n_days=len(ar),
                c1=c1, c1_pass=c1_pass, c2=c2, c2_pass=c2_pass, ar=ar)


def main() -> dict:
    t_start = time.time()
    n_configs = 0

    print("=" * 78)
    print("R-138 CONSERVATIVE: Nguyen-Wolf permutation test on kelly_regime_v4's")
    print("edge over its vol-matched constant-exposure hold, at STRESS_EPISODES.")
    print("=" * 78)
    print(f"PRIMARY_MARKET: {PRIMARY_MARKET}")
    print(f"STRESS_EPISODES (verbatim, {len(STRESS_EPISODES)} dates): "
          f"{[d for _, d in STRESS_EPISODES]}")
    print(f"N_PERM={N_PERM}  N_CALIBRATION_TRIALS={N_CALIBRATION_TRIALS}  "
          f"ALPHA={ALPHA}  CALIBRATION_BAND={CALIBRATION_BAND}")

    # -------------------------------------------------------------- BTC
    print("\n" + "=" * 78)
    print("STEP 1 -- BTC primary case (all 6 STRESS_EPISODES)")
    print("=" * 78)
    btc_df, btc_label = shared.load_btc_train()
    print(f"BTC train frame: {len(btc_df):,} bars, {btc_df.index[0]} -> {btc_df.index[-1]}")
    btc_events = [d for _, d in STRESS_EPISODES]
    btc = run_market("BTC", btc_df, btc_events, n_events=6, label=btc_label)
    n_configs += 1

    # -------------------------------------------------------------- ETH
    print("\n" + "=" * 78)
    print("STEP 2 -- ETH falsification case (C3): STRESS_EPISODES restricted to")
    print("ETH's available history")
    print("=" * 78)
    eth_df = shared.load_eth_train()
    print(f"ETH train frame: {len(eth_df):,} bars, {eth_df.index[0]} -> {eth_df.index[-1]}")

    eth_start = eth_df.index.min()
    eth_end = eth_df.index.max()

    def _in_range(date_str: str) -> bool:
        ts = pd.Timestamp(date_str)
        if ts.tzinfo is None and eth_start.tzinfo is not None:
            ts = ts.tz_localize(eth_start.tzinfo)
        return bool(eth_start <= ts <= eth_end)

    eth_kept = [(name, d) for name, d in STRESS_EPISODES if _in_range(d)]
    eth_dropped = [(name, d) for name, d in STRESS_EPISODES if not _in_range(d)]
    print(f"  ETH available range: {eth_start} -> {eth_end}")
    print(f"  STRESS_EPISODES dropped (predate ETH data start, not coverable, "
          f"NOT padded/substituted):")
    for name, d in eth_dropped:
        print(f"      DROPPED  {d}  ({name})")
    print(f"  STRESS_EPISODES kept ({len(eth_kept)}/{len(STRESS_EPISODES)}):")
    for name, d in eth_kept:
        print(f"      KEPT     {d}  ({name})")
    eth_events = [d for _, d in eth_kept]
    n_events_eth = len(eth_events)

    eth = run_market("ETH", eth_df, eth_events, n_events=n_events_eth, label="eth-spot")
    n_configs += 1

    # -------------------------------------------------------------- C3
    print("\n" + "=" * 78)
    print("STEP 3 -- C3: ETH replication check")
    print("=" * 78)
    btc_sign = float(np.sign(btc["c2"]["observed"])) if np.isfinite(btc["c2"]["observed"]) else 0.0
    eth_sign = float(np.sign(eth["c2"]["observed"])) if np.isfinite(eth["c2"]["observed"]) else 0.0
    eth_p = eth["c2"]["pvalue"]
    c3_sig = bool(np.isfinite(eth_p) and eth_p < 0.10)
    c3_same_sign = bool(btc_sign != 0.0 and eth_sign == btc_sign)
    c3_pass = bool(eth["c1_pass"] and c3_sig and c3_same_sign)
    print(f"  BTC observed CAAR={btc['c2']['observed']:+.6f}  sign={btc_sign:+.0f}")
    print(f"  ETH observed CAAR={eth['c2']['observed']:+.6f}  sign={eth_sign:+.0f}  "
          f"pvalue={eth_p:.6f}")
    print(f"  ETH C1 pass={eth['c1_pass']}  ETH p<0.10={c3_sig}  same sign as BTC={c3_same_sign}")
    print(f"  C3 PASS: {c3_pass}")

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    btc_voided = not btc["c1_pass"]
    eth_voided = not eth["c1_pass"]

    if btc_voided:
        print("  BTC: C1 FAILED -- market VOIDED (broken instrument on this series, "
              "not scored as negative), per the pre-registration's void-don't-score rule.")
    if eth_voided:
        print("  ETH: C1 FAILED -- market VOIDED (broken instrument on this series, "
              "not scored as negative), per the pre-registration's void-don't-score rule.")

    all_pass = bool(btc["c1_pass"] and btc["c2_pass"] and c3_pass)
    if btc_voided or eth_voided:
        verdict = "VOIDED"
    elif all_pass:
        verdict = "PROMOTABLE (method result -- proposed for tradebot/inference.py, no strategy change)"
    else:
        verdict = "NEGATIVE"

    print(f"\n  BTC C1 (calibration, band={CALIBRATION_BAND}): "
          f"rate={btc['c1']['rate']:.4f}  PASS={btc['c1_pass']}")
    print(f"  BTC C2 (significance, resolution-aware): "
          f"pvalue={btc['c2']['pvalue']:.6f}  n_exceed={btc['c2']['n_exceed']}  "
          f"PASS={btc['c2_pass']}")
    print(f"  ETH C1 (calibration, band={CALIBRATION_BAND}, n_events={n_events_eth}): "
          f"rate={eth['c1']['rate']:.4f}  PASS={eth['c1_pass']}")
    print(f"  ETH C3 (replication, p<0.10 same sign as BTC): PASS={c3_pass}")
    print(f"\n  OVERALL VERDICT: {verdict}")

    print(f"\nconfigurations evaluated (total): {n_configs} "
          f"(1 BTC candidate_and_matched_daily_logret call + "
          f"1 ETH candidate_and_matched_daily_logret call -- C1/C2/C3's "
          f"resampling trials are statistics on the already-computed `ar` "
          f"series, not separate backtest configurations, and are not "
          f"counted here)")
    max_ts = max(btc_df.index.max(), eth_df.index.max())
    print(f"max timestamp read anywhere in this branch: {max_ts} "
          f"(< {shared.OOS_START}: {max_ts < pd.Timestamp(shared.OOS_START, tz=max_ts.tzinfo)})")
    print("NO bar at or after 2023-01-01 was ever read by this file.")
    print(f"\n[{time.time() - t_start:.0f}s]")

    return dict(
        verdict=verdict, n_configs=n_configs,
        btc=btc, eth=eth, eth_kept=eth_kept, eth_dropped=eth_dropped,
        c3_pass=c3_pass, btc_sign=btc_sign, eth_sign=eth_sign,
        btc_voided=btc_voided, eth_voided=eth_voided, all_pass=all_pass,
    )


if __name__ == "__main__":
    main()
