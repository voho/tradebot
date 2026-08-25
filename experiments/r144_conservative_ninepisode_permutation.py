#!/usr/bin/env python
"""R-144 CONSERVATIVE branch: does the Nguyen-Wolf (2026) small-N permutation
test on ``kelly_regime_v4``'s edge over its own vol-matched constant-exposure
hold tighten (or loosen) when BTC's ``STRESS_EPISODES`` calendar is extended
from R-138's original six dates to R-143's nine (the original six plus the
three independently news-dated pre-2017 episodes: Mt. Gox 2014-02-07,
Bitstamp breach 2015-01-05, Bitfinex hack 2016-08-02)?

Full literature grounding, the four-question justification, the named
failure modes (including the ceiling stated BEFORE any run: this branch
cannot touch ETH replication -- see below), the falsification test and the
pre-registered C1/C2 decision rule all live in ``experiments/r144_shared.py``'s
own module docstring (read in full before this file was written); not
re-derived here beyond the summary above. This file NEVER edits
``r144_shared.py``, ``r138_shared.py``, ``r143_shared.py`` or
``r143_novel_extended_gate.py`` (all frozen / shared neutral ground), never
touches ``src/tradebot/strategies/kelly_regime_v4.py``, and never reads a bar
at or after ``r144_shared.OOS_START`` (2023-01-01) -- the shared loader's own
``_assert_no_holdout`` guards ``load_btc_extended_train()``, and an explicit
second assertion below re-checks the frame this file actually uses before any
statistic is computed on it.

============================================================================
THIS BRANCH CANNOT TEST ETH REPLICATION -- STATED HERE, BEFORE ANY NUMBER
============================================================================
None of the three new pre-2017 episodes (2014, 2015, 2016) fall inside ETH's
data (ETH spot starts 2019-03-14). Populating C3 (the ETH-replication check)
with a genuinely non-BTC-borrowed calendar is the PARALLEL, disjoint NOVEL
branch's job (``experiments/r144_novel_*.py``, ETH's own protocol-event
calendar) -- not attempted here. This file does not import ETH data and does
not touch ``r144_shared.ETH_NATIVE_EPISODES``. Any verdict below is a
BTC-only, method-only finding.

MECHANISM (exact, per ``r144_shared.py``'s own frozen procedure -- nothing
new invented here):

1. Load BTC's extended training frame, 2014-01-01 -> 2022-12-31, SPOT (forced:
   no BTC perpetual futures existed before 2017), via
   ``r144_shared.load_btc_extended_train()``.
2. ``r144_shared.candidate_and_matched_daily_logret_on(df, r144_shared.SPOT,
   r144_shared.BTC_EXTENDED_START, r144_shared.INNER_VAL_END, label)`` runs
   ``kelly_regime_v4`` and its own realized-volatility-matched
   ``ConstantExposureHold`` over the WHOLE 2014-2022 window in one shot
   (R-138's own convention: match once over the whole training period, not
   per sub-window) and returns aligned daily log-return series for both arms
   plus the solved match diagnostics ``(c, achieved_vol)``. ``ar = cand_log -
   matched_log`` is the abnormal-return series the permutation machinery
   watches.
3. Two event sets, both reported side by side, neither cherry-picked (a
   tightening-only addition ROUTINE.md explicitly allows: compute both,
   report both):
   (a) PRIMARY -- ``r144_shared.EXTENDED_STRESS_EPISODES_9`` (all 9 dates).
   (b) DIAGNOSTIC -- ``r143_novel_extended_gate.ORIGINAL_SIX`` (the original
       six R-138 dates), evaluated on THIS round's NEW 2014-2022/spot ``ar``
       series -- isolating whether any change in significance vs. R-138 comes
       from "more events" or from "different era + market" (this branch's
       forced era/market extension).
   ``r138_shared.caar_statistic`` / ``permutation_test`` at
   ``N_PERM=r138_shared.N_PERM`` (20000), ``ALPHA=r138_shared.ALPHA`` (0.05),
   both frozen, neither touched.
4. C1 calibration (``r138_shared.empirical_type1_rate``) at BOTH
   ``n_events=9`` and ``n_events=6`` -- checked separately because the pool of
   eligible pseudo-dates (``eligible_pseudo_dates``, which excludes windows
   around the REAL events) differs by event count, so an in-band result at
   one ``n`` does not imply the other.
5. Decision rule (pre-registered verbatim, ``r144_shared.py``'s docstring),
   applied separately to the n=9-primary and n=6-diagnostic event sets:
   promotable only if C1 (that set's own ``n_events`` calibration lands in
   ``CALIBRATION_BAND=(0.02,0.09)``) AND C2 (``pvalue < 0.05`` AND
   ``n_exceed > 2``). Anything else is NEGATIVE or METHOD for that cell (a
   failed C1 VOIDS that cell rather than scoring it negative, per
   ROUTINE.md's void-don't-score rule).
6. Standing-rule diagnostics (ROUTINE.md "Standing rules": match risk in
   controls too): the matched hold's solved constant exposure ``c``,
   achieved vs. target realized volatility, and time-in-market for both the
   candidate and the matched-hold arm (``tradebot.metrics.compute_metrics``'s
   ``time_in_market_pct`` field), computed by re-running the identical two
   backtests ``candidate_and_matched_daily_logret_on`` already ran internally
   (same strategy objects, same market/window/label) purely to recover the
   ``BacktestResult`` objects for these diagnostics -- NOT a second
   configuration (see CONFIGURATIONS EVALUATED below).

CONFIGURATIONS EVALUATED: 1 distinct backtest configuration
(``kelly_regime_v4`` vs. its vol-matched ``ConstantExposureHold``, BTC spot,
2014-01-01 -> 2022-12-31 -- one ``candidate_and_matched_daily_logret_on``
call). The diagnostic re-run in step 6 above re-executes that SAME call's two
backtests to expose ``BacktestResult`` objects for time-in-market reporting;
it is not a new configuration, exactly as R-138's own conservative script
excluded its C1/C2 resampling trials from its tally (pure statistics on an
already-computed ``ar`` series, no new strategy parameter or window chosen).

Distinct (event-set, statistic) evaluations on that one ``ar`` series: **4**
-- (9-episode, permutation_test/CAAR), (6-episode, permutation_test/CAAR),
(n_events=9, empirical_type1_rate/C1), (n_events=6, empirical_type1_rate/C1).
This is a fixed hypothesis test, not a parameter sweep: no threshold, window,
or model parameter is chosen or swept against this data at any point.

DECISION RULE (pre-registered, verbatim from ``r144_shared.py``'s docstring,
unaltered after seeing any number): for EACH event set independently,
promotable (a genuine BTC-only METHOD result) only if C1 AND C2 both pass on
that event set. This branch's finding, even if both cells pass, resolves
ONLY BTC significance under the extended calendar -- it structurally cannot
resolve C3 (ETH replication), which is reported here as an explicit ceiling,
never folded into a pass.

USAGE
-----
    python experiments/r144_conservative_ninepisode_permutation.py
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

import r144_shared as shared  # noqa: E402
from r143_novel_extended_gate import ORIGINAL_SIX  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.window import run_period  # noqa: E402

SPOT = shared.SPOT
CALIBRATION_BAND = shared.CALIBRATION_BAND
ALPHA = shared.ALPHA
N_PERM = shared.N_PERM
N_CALIBRATION_TRIALS = shared.N_CALIBRATION_TRIALS

EVENT_SETS = [
    ("9-episode PRIMARY", shared.EXTENDED_STRESS_EPISODES_9, 9),
    ("6-episode DIAGNOSTIC (original six, new series)", ORIGINAL_SIX, 6),
]

LABEL = "btc-extended-2014-2022-spot"


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(shared.OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {shared.OOS_START}")


def run_c1(ar: pd.Series, n_events: int) -> dict:
    print(f"\n  C1 -- empirical_type1_rate(ar, n_events={n_events}) "
          f"({N_CALIBRATION_TRIALS} trials x {N_PERM} perms)...")
    t0 = time.time()
    c1 = shared.empirical_type1_rate(ar, n_events=n_events)
    c1_pass = bool(np.isfinite(c1["rate"]) and CALIBRATION_BAND[0] <= c1["rate"] <= CALIBRATION_BAND[1])
    print(f"    rate={c1['rate']:.4f}  (n_trials valid={c1['n_trials']}, "
          f"rejects={c1.get('rejects')})  band={CALIBRATION_BAND}  PASS={c1_pass}  "
          f"[{time.time() - t0:.1f}s]")
    return dict(**c1, n_events=n_events, c1_pass=c1_pass)


def run_c2(ar: pd.Series, name: str, events: list) -> dict:
    dates = [d for _, d in events]
    print(f"\n  C2 [{name}] -- permutation_test(ar, {len(dates)} event dates) "
          f"(N_PERM={N_PERM})...")
    t0 = time.time()
    c2 = shared.permutation_test(ar, dates)
    c2_sig = bool(np.isfinite(c2["pvalue"]) and c2["pvalue"] < ALPHA)
    c2_resolved = bool(c2["n_exceed"] > 2)
    c2_pass = bool(c2_sig and c2_resolved)
    print(f"    observed CAAR={c2['observed']:+.6f}  pvalue={c2['pvalue']:.6f}  "
          f"n_exceed={c2['n_exceed']}  n_perm(used)={c2['n_perm']}  pool_size={c2['pool_size']}  "
          f"[{time.time() - t0:.1f}s]")
    print(f"    significant (p<{ALPHA})={c2_sig}  resolution-aware (n_exceed>2)={c2_resolved}  "
          f"C2 PASS={c2_pass}")
    return dict(**c2, name=name, c2_pass=c2_pass)


def main() -> dict:
    t_start = time.time()

    print("=" * 84)
    print("R-144 CONSERVATIVE: Nguyen-Wolf permutation test on kelly_regime_v4's")
    print("edge over its vol-matched constant-exposure hold, BTC 2014-2022 spot,")
    print("at the 9-episode extended calendar (primary) vs. the original 6 (diagnostic).")
    print("=" * 84)
    print("THIS BRANCH CANNOT TEST ETH REPLICATION (C3) -- none of the 3 new")
    print("pre-2017 episodes fall inside ETH's 2019-03-14+ data. That is the")
    print("parallel NOVEL branch's job; not attempted here.")
    print(f"\nN_PERM={N_PERM}  N_CALIBRATION_TRIALS={N_CALIBRATION_TRIALS}  "
          f"ALPHA={ALPHA}  CALIBRATION_BAND={CALIBRATION_BAND}")
    print(f"9-episode calendar: {[d for _, d in shared.EXTENDED_STRESS_EPISODES_9]}")
    print(f"6-episode calendar: {[d for _, d in ORIGINAL_SIX]}")

    # -------------------------------------------------------------- (1)
    print("\n" + "=" * 84)
    print("STEP 1 -- load BTC extended train frame, build AR series")
    print("=" * 84)
    df = shared.load_btc_extended_train()
    _assert_no_holdout(df)
    print(f"BTC extended train frame: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    cand_log, matched_log, c, achieved_vol = shared.candidate_and_matched_daily_logret_on(
        df, SPOT, shared.BTC_EXTENDED_START, shared.INNER_VAL_END, LABEL)
    ar = cand_log - matched_log
    print(f"vol-match solve: c={c:.6f}  achieved_vol={achieved_vol:.6f}  n_days(ar)={len(ar)}")

    # -------------------------------------------------------------- (2) diagnostics
    print("\n" + "=" * 84)
    print("STEP 2 -- standing-rule diagnostics: match risk in controls too")
    print("(re-running the identical two backtests already run inside step 1's")
    print(" candidate_and_matched_daily_logret_on call -- NOT a new configuration,")
    print(" purely to recover BacktestResult objects for time-in-market)")
    print("=" * 84)
    from tradebot.registry import get_strategy
    from experiments.matched_hold import ConstantExposureHold

    cand_res = run_period(get_strategy("kelly_regime_v4"), df,
                           start=shared.BTC_EXTENDED_START, end=shared.INNER_VAL_END,
                           market=SPOT, start_balance=1_000.0, data_label=LABEL)
    matched_res = run_period(ConstantExposureHold(c, static=False), df,
                              start=shared.BTC_EXTENDED_START, end=shared.INNER_VAL_END,
                              market=SPOT, start_balance=1_000.0, data_label=LABEL)
    target_vol = shared.realized_vol_daily(cand_res.equity)
    cand_m = compute_metrics(cand_res)
    matched_m = compute_metrics(matched_res)
    print(f"  candidate (kelly_regime_v4):  target_vol={target_vol:.6f}  "
          f"time_in_market={cand_m.time_in_market_pct:.2f}%")
    print(f"  matched hold (c={c:.6f}):     achieved_vol={achieved_vol:.6f}  "
          f"time_in_market={matched_m.time_in_market_pct:.2f}%")
    vol_gap_pct = 100.0 * abs(achieved_vol - target_vol) / target_vol if target_vol else float("nan")
    print(f"  |achieved - target| / target = {vol_gap_pct:.2f}%  "
          f"(risk match diagnostic, R-33/R-131's standing warning)")

    # -------------------------------------------------------------- (3) C1 at n=9, n=6
    print("\n" + "=" * 84)
    print("STEP 3 -- C1 calibration, checked separately at n_events=9 and n_events=6")
    print("=" * 84)
    c1_by_n = {n: run_c1(ar, n) for _, _, n in EVENT_SETS}

    # -------------------------------------------------------------- (4) C2 at both event sets
    print("\n" + "=" * 84)
    print("STEP 4 -- C2 permutation test, both event sets, side by side")
    print("=" * 84)
    c2_by_name = {}
    for name, events, n in EVENT_SETS:
        c2_by_name[name] = run_c2(ar, name, events)

    # -------------------------------------------------------------- (5) decision rule
    print("\n" + "=" * 84)
    print("STEP 5 -- decision rule (pre-registered verbatim), per event set")
    print("=" * 84)
    results = {}
    for name, events, n in EVENT_SETS:
        c1 = c1_by_n[n]
        c2 = c2_by_name[name]
        voided = not c1["c1_pass"]
        if voided:
            verdict = "VOIDED (C1 failed -- broken instrument on this series/n, not scored negative)"
        elif c1["c1_pass"] and c2["c2_pass"]:
            verdict = "PROMOTABLE (BTC-only method result; C3/ETH untouched by this branch)"
        else:
            verdict = "NEGATIVE"
        results[name] = dict(n_events=n, c1=c1, c2=c2, voided=voided, verdict=verdict)
        print(f"\n  [{name}]  n_events={n}")
        print(f"    C1: rate={c1['rate']:.4f}  PASS={c1['c1_pass']}")
        print(f"    C2: pvalue={c2['pvalue']:.6f}  n_exceed={c2['n_exceed']}  PASS={c2['c2_pass']}")
        print(f"    VERDICT: {verdict}")

    # -------------------------------------------------------------- (6) summary table
    print("\n" + "=" * 84)
    print("SUMMARY TABLE")
    print("=" * 84)
    header = (f"  {'event set':42s} {'n':>2s} {'obs CAAR':>10s} {'pvalue':>9s} "
              f"{'n_exc':>6s} {'pool':>6s} {'C1 rate':>8s} {'C1':>5s} {'C2':>5s}  verdict")
    print(header)
    for name, events, n in EVENT_SETS:
        r = results[name]
        c1, c2 = r["c1"], r["c2"]
        print(f"  {name:42s} {n:>2d} {c2['observed']:>+10.6f} {c2['pvalue']:>9.6f} "
              f"{c2['n_exceed']:>6d} {c2['pool_size']:>6d} {c1['rate']:>8.4f} "
              f"{str(c1['c1_pass']):>5s} {str(c2['c2_pass']):>5s}  {r['verdict']}")

    print(f"\n  matched-hold c={c:.6f}  target_vol={target_vol:.6f}  achieved_vol={achieved_vol:.6f}"
          f"  gap={vol_gap_pct:.2f}%")
    print(f"  time-in-market: candidate={cand_m.time_in_market_pct:.2f}%  "
          f"matched-hold={matched_m.time_in_market_pct:.2f}%")

    print("\n  ETH REPLICATION (C3): NOT TESTED by this branch (structural ceiling, "
          "stated before any run -- see novel branch for the ETH-native calendar).")

    n_configs_backtest = 1
    n_configs_eventstat = len(EVENT_SETS) * 2  # (event-set, C2) + (n_events, C1), 2 event sets x 2 stats
    print(f"\nconfigurations evaluated (backtest): {n_configs_backtest} "
          f"(kelly_regime_v4 vs. vol-matched ConstantExposureHold, BTC spot, "
          f"2014-2022 -- one candidate_and_matched_daily_logret_on call; the "
          f"step-2 diagnostic re-run repeats this same call's two backtests, "
          f"not a new configuration)")
    print(f"distinct (event-set, statistic) evaluations on that one ar series: "
          f"{n_configs_eventstat} "
          f"(9-episode CAAR/permutation, 6-episode CAAR/permutation, "
          f"n=9 C1 calibration, n=6 C1 calibration)")

    max_ts = df.index.max()
    within = max_ts < pd.Timestamp(shared.OOS_START, tz=max_ts.tzinfo)
    print(f"\nmax timestamp read anywhere in this branch: {max_ts} "
          f"(< {shared.OOS_START}: {within})")
    assert within, "holdout breach"
    print("NO bar at or after 2023-01-01 was ever read by this file.")
    print(f"\n[{time.time() - t_start:.0f}s]")

    return dict(
        c=c, achieved_vol=achieved_vol, target_vol=target_vol,
        cand_tim=cand_m.time_in_market_pct, matched_tim=matched_m.time_in_market_pct,
        results=results, n_configs_backtest=n_configs_backtest,
        n_configs_eventstat=n_configs_eventstat,
    )


if __name__ == "__main__":
    main()
