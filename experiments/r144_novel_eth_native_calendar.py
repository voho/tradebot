#!/usr/bin/env python
"""R-144 NOVEL branch: the Nguyen-Wolf (2026) small-N permutation test,
reused verbatim from ``experiments/r138_shared.py``, applied to ETH's OWN
independently-dated protocol/monetary-policy calendar
(``r144_shared.ETH_NATIVE_EPISODES`` -- Beacon Chain genesis, EIP-1559/
London, The Merge) instead of BTC's borrowed six-episode narrative
calendar, on ``kelly_regime_v4`` vs. its own realized-volatility-matched
constant-exposure hold, 5x futures (``r138_shared.PRIMARY_MARKET``).

Full literature grounding, the four-question justification, the named
failure modes, the falsification test, and the pre-registered C1/C2
decision rule all live in ``experiments/r144_shared.py``'s own module
docstring (read in full before this file was written); not re-derived here
beyond the summary above. This file NEVER edits ``r144_shared.py`` or
``r138_shared.py`` (both frozen), never edits
``src/tradebot/strategies/kelly_regime_v4.py``, and never reads a bar at or
after ``r144_shared.OOS_START`` (2023-01-01) from any data source -- the
shared loader's own ``_assert_no_holdout`` guards this, and this file
constructs no bespoke slice that could bypass it.

MECHANISM (exact, per ``r138_shared.py``'s own frozen procedure -- nothing
new invented here beyond swapping the event calendar):

1. ``eth_df = r138_shared.load_eth_train()`` (holdout-safe, up to
   ``INNER_VAL_END = 2022-12-31``).
2. ``candidate_and_matched_daily_logret(eth_df, PRIMARY_MARKET, label)``
   gives ``(cand_log, matched_log, c, achieved_vol)`` exactly as R-138's own
   ETH falsification case used it (5x futures, not spot -- ETH futures
   existed throughout this period, so the conservative branch's forced
   BTC-spot situation does not apply here). ``ar = cand_log - matched_log``
   is the abnormal-return series the permutation machinery watches.
3. Per-event CAR (``car_for_event``) for each of the 3
   ``r144_shared.ETH_NATIVE_EPISODES`` dates, reported individually
   alongside the pooled CAAR so a reader can see whether the effect is
   broad across the three dates or driven by one.
4. C1: ``empirical_type1_rate(ar, n_events=3)`` at the module's frozen
   defaults; must land in ``CALIBRATION_BAND = (0.02, 0.09)`` or the branch
   is VOIDED (not scored) per the pre-registration's void-don't-score rule.
   The base eligible-pseudo-date pool size is also reported and sanity
   checked explicitly (task flags this: with only ~3.75 years of ETH
   training data and 3 events, the pool is smaller than BTC's 6-year one,
   and must still be checked, not assumed, to be large enough to trust).
5. C2: ``permutation_test(ar, [d for _, d in ETH_NATIVE_EPISODES])`` at the
   module's frozen ``N_PERM``; passes only if ``pvalue < 0.05`` AND
   ``n_exceed > 2`` (both conditions, resolution-aware per the
   pre-registration).
6. Diagnostics only (never gate anything, R-33's standing warning): the
   vol-match solve's ``c``, achieved vs. target realized volatility, and
   time-in-market for BOTH the candidate and the matched-hold arm (read off
   ``compute_metrics(result).time_in_market_pct``), so a reader can confirm
   any effect is not just an exposure-level artifact.
7. The observed CAAR's SIGN is reported explicitly (positive = v4
   outperforms the matched hold around ETH's own regime-transition dates) --
   this branch cannot itself check it against the conservative branch's BTC
   CAAR sign (parallel branch, no shared state at report time); that
   cross-branch check is for the operator's synthesis.

CONFIGURATIONS EVALUATED: 1 (ETH: one ``candidate_and_matched_daily_logret``
call, i.e. one ``kelly_regime_v4`` backtest + one vol-matched
constant-exposure-hold backtest, on ``PRIMARY_MARKET``). ``solve_matched_c``'s
internal bisection iterations to find ``c``, this file's own re-run of the
candidate/matched-hold pair purely to read ``compute_metrics`` diagnostics
(time-in-market, achieved vol confirmation) at the ALREADY-solved ``c``, and
C1's 200 calibration trials / C2's permutation draws, are NOT separate
backtest configurations -- matching the counting convention
``r138_conservative_stress_permutation.py`` already used.

DECISION RULE (pre-registered, verbatim from ``r144_shared.py``, unaltered
after seeing any number): promotable (worth writing up as a genuine
methodological, partial result) only if BOTH C1 and C2 pass. If C1 fails,
the branch is VOIDED (not scored as negative -- a broken instrument, per
ROUTINE.md's void-don't-score rule) rather than folded into a plain
NEGATIVE. Anything else not clearing both is NEGATIVE. This round produces
no strategy code change regardless of outcome.

INTERPRETIVE CAVEAT (disclosed, not gated on, per the pre-registration's own
guardrail against voiding a pass for "the wrong kind of event" after the
fact): two of the three ETH-native dates (Beacon Chain genesis, EIP-1559)
were scheduled weeks in advance and could have been pre-positioned around,
unlike a sudden crash -- a genuine disanalogy with the sudden-shock
character of most of BTC's six-episode calendar. Reported as context on any
verdict, never as a post-hoc reason to discard a result either way.

USAGE
-----
    python experiments/r144_novel_eth_native_calendar.py
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

import r138_shared as shared138  # noqa: E402
import r144_shared as shared144  # noqa: E402
from experiments.matched_hold import ConstantExposureHold  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

PRIMARY_MARKET = shared138.PRIMARY_MARKET
ETH_NATIVE_EPISODES = shared144.ETH_NATIVE_EPISODES
CALIBRATION_BAND = shared138.CALIBRATION_BAND
ALPHA = shared138.ALPHA
N_PERM = shared138.N_PERM
N_CALIBRATION_TRIALS = shared138.N_CALIBRATION_TRIALS
WINDOW_PRE_DAYS = shared138.WINDOW_PRE_DAYS
WINDOW_POST_DAYS = shared138.WINDOW_POST_DAYS
OOS_START = shared144.OOS_START
INNER_TRAIN_START = shared138.INNER_TRAIN_START
INNER_VAL_END = shared138.INNER_VAL_END


def _run(strategy, df, market, start, end, label=""):
    """Identical wiring to ``r138_shared._run`` / ``r144_shared._run`` --
    duplicated locally (not imported as a private name) purely to obtain the
    ``BacktestResult`` objects needed for ``compute_metrics`` diagnostics."""
    return run_period(strategy, df, start=start, end=end, market=market,
                       start_balance=1_000.0, data_label=label)


def main() -> dict:
    t_start = time.time()

    print("=" * 78)
    print("R-144 NOVEL: Nguyen-Wolf permutation test on kelly_regime_v4's edge")
    print("over its vol-matched constant-exposure hold, at ETH's OWN native")
    print("protocol/monetary-policy calendar (not BTC's borrowed dates).")
    print("=" * 78)
    print(f"PRIMARY_MARKET: {PRIMARY_MARKET}")
    print(f"ETH_NATIVE_EPISODES ({len(ETH_NATIVE_EPISODES)} dates):")
    for name, d in ETH_NATIVE_EPISODES:
        print(f"    {d}   {name}")
    print(f"N_PERM={N_PERM}  N_CALIBRATION_TRIALS={N_CALIBRATION_TRIALS}  "
          f"ALPHA={ALPHA}  CALIBRATION_BAND={CALIBRATION_BAND}")
    print(f"WINDOW_PRE_DAYS={WINDOW_PRE_DAYS}  WINDOW_POST_DAYS={WINDOW_POST_DAYS}")

    # -------------------------------------------------------------- load
    print("\n" + "=" * 78)
    print("STEP 1 -- load ETH training frame, build AR series")
    print("=" * 78)
    eth_df = shared138.load_eth_train()
    print(f"ETH train frame: {len(eth_df):,} bars, {eth_df.index[0]} -> {eth_df.index[-1]}")
    assert eth_df.index.max() < pd.Timestamp(OOS_START, tz=eth_df.index.max().tzinfo), (
        "holdout breach in loaded ETH frame")

    events = [d for _, d in ETH_NATIVE_EPISODES]
    n_events = len(events)
    eth_start, eth_end = eth_df.index.min(), eth_df.index.max()

    def _in_range(date_str: str) -> bool:
        ts = pd.Timestamp(date_str)
        if ts.tzinfo is None and eth_start.tzinfo is not None:
            ts = ts.tz_localize(eth_start.tzinfo)
        return bool(eth_start <= ts <= eth_end)

    for name, d in ETH_NATIVE_EPISODES:
        print(f"    coverage check: {d} in [{eth_start}, {eth_end}]: {_in_range(d)}")
    assert all(_in_range(d) for d in events), (
        "an ETH-native episode falls outside ETH's available training range")

    t0 = time.time()
    cand_log, matched_log, c, achieved_vol = shared138.candidate_and_matched_daily_logret(
        eth_df, PRIMARY_MARKET, "eth-spot")
    ar = cand_log - matched_log
    print(f"  candidate_and_matched_daily_logret: n_days(ar)={len(ar)}  "
          f"[{time.time() - t0:.1f}s]")

    # -------------------------------------------------------- diagnostics
    print("\n" + "=" * 78)
    print("STEP 2 -- R-33 standing diagnostics: vol match + time-in-market,")
    print("both arms (never gates the verdict, reported so a reader can rule")
    print("out an exposure-level artifact)")
    print("=" * 78)
    t0 = time.time()
    cand_res = _run(get_strategy("kelly_regime_v4"), eth_df, PRIMARY_MARKET,
                     INNER_TRAIN_START, INNER_VAL_END, "eth-spot")
    matched_res = _run(ConstantExposureHold(c, static=False), eth_df, PRIMARY_MARKET,
                        INNER_TRAIN_START, INNER_VAL_END, "eth-spot")
    target_vol = shared138.realized_vol_daily(cand_res.equity)
    matched_vol_check = shared138.realized_vol_daily(matched_res.equity)
    cand_metrics = compute_metrics(cand_res)
    matched_metrics = compute_metrics(matched_res)
    print(f"  matched-hold solved c = {c:.6f}")
    print(f"  candidate (kelly_regime_v4) realized vol (target) = {target_vol:.6f}")
    print(f"  matched-hold achieved vol (from solver)           = {achieved_vol:.6f}")
    print(f"  matched-hold achieved vol (independent re-check)  = {matched_vol_check:.6f}")
    vol_ratio = achieved_vol / target_vol if target_vol else float("nan")
    print(f"  achieved/target vol ratio = {vol_ratio:.4f}  "
          f"(within 2% tolerance: {abs(vol_ratio - 1.0) <= 0.02})")
    print(f"  time-in-market  candidate (kelly_regime_v4)  = {cand_metrics.time_in_market_pct:.2f}%")
    print(f"  time-in-market  matched-hold (constant c={c:.4f}) = {matched_metrics.time_in_market_pct:.2f}%")
    print(f"  [{time.time() - t0:.1f}s]")

    # ------------------------------------------------------------- events
    print("\n" + "=" * 78)
    print("STEP 3 -- individual event CARs (each date's own cumulative")
    print("abnormal return, window [-%d, +%d] days), so the pooled CAAR is not" %
          (WINDOW_PRE_DAYS, WINDOW_POST_DAYS))
    print("read without seeing whether it is broad or driven by one date")
    print("=" * 78)
    event_cars = {}
    for name, d in ETH_NATIVE_EPISODES:
        car = shared138.car_for_event(ar, d)
        event_cars[d] = car
        print(f"    CAR({d})  = {car:+.6f}   ({name})")

    # ---------------------------------------------------------------- C1
    print("\n" + "=" * 78)
    print(f"STEP 4 -- C1: empirical_type1_rate(ar, n_events={n_events}) "
          f"({N_CALIBRATION_TRIALS} trials x {N_PERM} perms)")
    print("=" * 78)
    pool_base = shared138.eligible_pseudo_dates(ar, [], WINDOW_PRE_DAYS, WINDOW_POST_DAYS)
    pool_size_base = len(pool_base)
    span_days = (ar.index.max() - ar.index.min()).days
    print(f"  ar series span: {ar.index.min()} -> {ar.index.max()} "
          f"({span_days} days, ~{span_days / 365.25:.2f} years)")
    print(f"  base eligible pseudo-date pool size (no real-event exclusions): "
          f"{pool_size_base}")
    pool_sane = pool_size_base >= 200
    if pool_size_base < 200:
        print(f"  FLAG: pool_size_base={pool_size_base} looks too small to trust "
              f"the calibration/permutation draws (expected at least low "
              f"hundreds for a ~3.75yr ETH series with a {n_events}-date "
              f"exclusion buffer).")
    else:
        print(f"  pool_size_base={pool_size_base} is sane (not in the low "
              f"thousands BTC's 6-year series reaches, because ETH's training "
              f"window is only ~3.75 years, but ample relative to drawing "
              f"{n_events} dates at a time).")

    t0 = time.time()
    c1 = shared138.empirical_type1_rate(ar, n_events=n_events)
    c1_pass = bool(np.isfinite(c1["rate"]) and
                   CALIBRATION_BAND[0] <= c1["rate"] <= CALIBRATION_BAND[1])
    print(f"  C1 empirical Type-I rate = {c1['rate']:.4f}  "
          f"(n_trials valid={c1['n_trials']}, rejects={c1.get('rejects')})  "
          f"band={CALIBRATION_BAND}  PASS={c1_pass}  [{time.time() - t0:.1f}s]")

    # ---------------------------------------------------------------- C2
    print("\n" + "=" * 78)
    print(f"STEP 5 -- C2: permutation_test(ar, {n_events} ETH-native event "
          f"dates)  (N_PERM={N_PERM})")
    print("=" * 78)
    t0 = time.time()
    c2 = shared138.permutation_test(ar, events)
    observed_caar = c2["observed"]
    caar_sign = float(np.sign(observed_caar)) if np.isfinite(observed_caar) else 0.0
    c2_sig = bool(np.isfinite(c2["pvalue"]) and c2["pvalue"] < ALPHA)
    c2_resolved = bool(c2["n_exceed"] > 2)
    c2_pass = bool(c2_sig and c2_resolved)
    print(f"  observed CAAR = {observed_caar:+.6f}   sign = {caar_sign:+.0f}  "
          f"({'kelly_regime_v4 OUTPERFORMS matched hold' if caar_sign > 0 else 'kelly_regime_v4 UNDERPERFORMS matched hold' if caar_sign < 0 else 'exactly zero'} "
          f"around ETH-native dates)")
    print(f"  pvalue = {c2['pvalue']:.6f}   n_exceed = {c2['n_exceed']}   "
          f"n_perm(used) = {c2['n_perm']}   pool_size(excl. real events) = {c2['pool_size']}")
    print(f"  C2 significant (p<{ALPHA}) = {c2_sig}   "
          f"resolution-aware (n_exceed>2) = {c2_resolved}   C2 PASS = {c2_pass}")
    print(f"  [{time.time() - t0:.1f}s]")

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not c1_pass:
        verdict = "VOIDED"
        print("  C1 FAILED -- branch VOIDED (broken instrument on this series, "
              "not scored as negative), per the pre-registration's "
              "void-don't-score rule. The permutation p-value below is NOT "
              "trustworthy and must not be read as evidence either way.")
    elif c2_pass:
        verdict = ("PASS (C1+C2 clear) -- first-ever population of C3 with a "
                   "calendar not borrowed from BTC; a genuinely new, partial "
                   "result. Whether this counts as GENUINE ETH replication of "
                   "the edge-concentration property additionally requires the "
                   "conservative branch's BTC CAAR sign to match this branch's "
                   f"sign ({caar_sign:+.0f}) -- routed to the operator's "
                   "cross-branch synthesis, not resolvable by this branch alone.")
        print(f"  C1 PASS (rate={c1['rate']:.4f} in {CALIBRATION_BAND})")
        print(f"  C2 PASS (p={c2['pvalue']:.6f} < {ALPHA}, n_exceed={c2['n_exceed']} > 2)")
        print(f"  {verdict}")
    else:
        verdict = "NEGATIVE"
        print(f"  C1 PASS (rate={c1['rate']:.4f} in {CALIBRATION_BAND})")
        print(f"  C2 FAIL (p={c2['pvalue']:.6f}, sig={c2_sig}, "
              f"n_exceed={c2['n_exceed']}, resolved={c2_resolved})")
        print("  NEGATIVE: ETH's edge does not concentrate significantly around "
              "its own native regime-transition calendar either, at the "
              "pre-registered C1+C2 bar. This closes the 'borrowed BTC dates' "
              "escape hatch as an explanation for R-138's ETH failure -- the "
              "effect does not appear to be a property of the calendar's "
              "origin, on this evidence.")

    print("\n  INTERPRETIVE CAVEAT (disclosed, not gating): 2 of the 3 "
          "ETH-native dates (Beacon Chain genesis, EIP-1559/London) were "
          "scheduled weeks in advance and could have been pre-positioned "
          "around, unlike a sudden crash -- a genuine disanalogy with the "
          "sudden-shock character of most of BTC's six-episode calendar. "
          "Context only, per the pre-registration's guard against voiding a "
          "pass for 'the wrong kind of event' after the fact.")

    max_ts = eth_df.index.max()
    print(f"\nmax timestamp read anywhere in this branch: {max_ts} "
          f"(< {OOS_START}: {max_ts < pd.Timestamp(OOS_START, tz=max_ts.tzinfo)})")
    print("NO bar at or after 2023-01-01 was ever read by this file.")
    print("configurations evaluated (total): 1 (ETH: one "
          "candidate_and_matched_daily_logret call on PRIMARY_MARKET; the "
          "solver's internal bisection iterations, this file's own metrics "
          "re-run at the already-solved c, and C1/C2's resampling trials are "
          "not separate backtest configurations, matching "
          "r138_conservative_stress_permutation.py's counting convention)")

    print("\n" + "=" * 78)
    print("FULL NUMERIC SUMMARY")
    print("=" * 78)
    print(f"  observed CAAR             : {observed_caar:+.6f}  (sign {caar_sign:+.0f})")
    for name, d in ETH_NATIVE_EPISODES:
        print(f"    CAR[{d}]              : {event_cars[d]:+.6f}  ({name})")
    print(f"  p-value                   : {c2['pvalue']:.6f}")
    print(f"  n_exceed                  : {c2['n_exceed']}")
    print(f"  n_perm                    : {c2['n_perm']}")
    print(f"  pool_size (C2, excl real) : {c2['pool_size']}")
    print(f"  pool_size (base, C1 ref)  : {pool_size_base}  (sane: {pool_sane})")
    print(f"  C1 empirical Type-I rate  : {c1['rate']:.4f}  in-band {CALIBRATION_BAND}: {c1_pass}")
    print(f"  matched-hold c            : {c:.6f}")
    print(f"  target vol (candidate)    : {target_vol:.6f}")
    print(f"  achieved vol (matched)    : {achieved_vol:.6f}")
    print(f"  time-in-market candidate  : {cand_metrics.time_in_market_pct:.2f}%")
    print(f"  time-in-market matched    : {matched_metrics.time_in_market_pct:.2f}%")
    print(f"  C1 pass                   : {c1_pass}")
    print(f"  C2 pass                   : {c2_pass}")
    print(f"  VERDICT                   : {verdict}")
    print(f"\n[{time.time() - t_start:.0f}s]")

    return dict(
        verdict=verdict, c1=c1, c1_pass=c1_pass, c2=c2, c2_pass=c2_pass,
        observed_caar=observed_caar, caar_sign=caar_sign, event_cars=event_cars,
        c=c, target_vol=target_vol, achieved_vol=achieved_vol,
        time_in_market_candidate=cand_metrics.time_in_market_pct,
        time_in_market_matched=matched_metrics.time_in_market_pct,
        pool_size_base=pool_size_base, pool_sane=pool_sane,
    )


if __name__ == "__main__":
    main()
