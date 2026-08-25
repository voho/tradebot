#!/usr/bin/env python
"""R-141 CONSERVATIVE branch: LPPLS confidence indicator (Johansen, Ledoit
& Sornette 2000; Filimonov & Sornette 2013) run against the SAME
six-episode Step-A detection-lag gate that sank HMM (R-01), BOCPD (R-82),
Kalman LLT (R-83), critical slowing down (R-85), transfer entropy (R-86)
and CUSUM (R-139) -- see `experiments/r141_shared.py`'s module docstring
for the full theoretical-basis argument, not-a-duplicate-of trail, the
pre-registered Step-B specification for this branch (a
`confirming_vote_frac` combination swept over weight in {0.5, 1.0, 1.5}),
and the pre-registered stop rule this file follows verbatim.

PRE-REGISTERED STOP RULE (frozen in `r141_shared.py`'s docstring before
any of today's numbers existed): if no `conf_thresh` in the sweep
{0.4, 0.6, 0.8} reaches >=4/6 episodes passing Step-A, this branch STOPS
HERE -- no Step-B implementation, no holdout read, reported as NEGATIVE
at Step-A.

INDEX-ALIGNMENT NOTE (read before trusting any number below): the task
brief for this branch initially named `lppls_daily_signals`'s
DAILY-indexed frame as the `lppls` argument to `step_a_gate`, with an
explicit instruction to check the function's actual expectation and
adapt. `step_a_gate` calls `confidence.reindex(window)` where `window` is
a slice of `bars.index` -- a 5-MINUTE-frequency DatetimeIndex, identical
in shape to how R-139's conservative branch fed `cusum_daily_causal_signals`
(bar-aligned, not the raw daily frame) into the analogous gate. Feeding
the raw daily frame directly reindexes onto 5-minute timestamps and
silently drops to ~0.4% coverage (measured below) because almost no bar
timestamp lands exactly on a calendar-day boundary. The correct argument
is therefore `lppls_bar_signals`'s bar-aligned, forward-filled output, and
that is what this file uses; the mismatch is demonstrated explicitly in
the printed diagnostic below rather than fixed silently.

=====================================================================
RESULT (filled in by running this file; see printed report below for the
authoritative numbers)
=====================================================================

CONFIGURATIONS EVALUATED IN THIS FILE: 3 (conf_thresh in {0.4, 0.6, 0.8},
the pre-registered Step-A sweep; Step-B does not trigger -- see below).
The LPPLS signal itself (800 fits/window-length x 5 window lengths x 234
calibration dates = 936,000 lstsq fits) is shared, byte-identical
machinery computed ONCE in `r141_shared.py` and counted once for the
round's trials total per that module's own docstring, not per branch.
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

from experiments.r141_shared import (  # noqa: E402
    CONF_MAJORITY,
    GENUINE_TOP_EPISODES,
    OOS_START,
    STRESS_EPISODES,
    lppls_bar_signals,
    lppls_daily_signals,
    load_btc_train,
    step_a_gate,
)

LPPLS_CACHE = ROOT / "experiments" / "r141_lppls_btc_cache.csv"

# Pre-registered stop-rule bar (identical to the shared gate's own bar and
# to every predecessor round's bar on this gate): >= 4/6 episodes must
# pass, for ANY reported threshold, to proceed past Step A.
STOP_RULE_MIN_PASS = 4

# Pre-registered sweep, fixed in the task brief before any real-data
# number existed for this branch (n_qualify >= 2, >=3, >=4 out of 5).
CONF_THRESH_GRID = (0.4, 0.6, 0.8)


def assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


def causal_truncation_probe(bars: pd.DataFrame, check_pos: int = 885,
                             extra_days: int = 51) -> dict:
    """Causal-truncation probe against THIS branch's own new code path:
    the full pre-registered grid `lppls_daily_signals` call (all of
    M_GRID x OMEGA_GRID x TC_OFFSET_FRACTIONS x WINDOW_LENGTHS_DAYS -- no
    restricted single-cell grid the way `r141_shared.py`'s own `__main__`
    self-test uses for speed). Compares the FULL cached run's calibration
    output at `check_pos` against an INDEPENDENT, freshly computed run on
    a truncated copy of the series that ends only `extra_days` past
    `check_pos` (cache_path=None, forcing full recomputation, not a
    cache read). `check_pos=885` (2019-06-05) is a stride-grid calibration
    date with `lppls_n_qualify=2` in the cached run -- a genuinely
    qualifying date, not a degenerate all-zero one, so the probe checks
    that the QUALIFYING flag itself, not just an all-NaN/zero fit, survives
    truncation unchanged."""
    daily_close = bars["close"].resample("1D").last().dropna()
    full = lppls_daily_signals(bars, cache_path=LPPLS_CACHE, verbose=False)
    trunc_bars = bars.loc[bars.index <= daily_close.index[check_pos + extra_days]].copy()
    t0 = time.time()
    trunc = lppls_daily_signals(trunc_bars, cache_path=None, verbose=False)
    elapsed = time.time() - t0

    cols = ["lppls_n_qualify", "lppls_confidence", "lppls_best_m",
            "lppls_best_omega", "lppls_best_tc_offset_days"]
    a = full.iloc[check_pos][cols].to_numpy(dtype=float)
    b = trunc.iloc[check_pos][cols].to_numpy(dtype=float)
    ok = bool(np.allclose(a, b, equal_nan=True, rtol=1e-9))
    return dict(ok=ok, check_date=str(full.index[check_pos].date()),
                full_vals=dict(zip(cols, a.tolist())),
                trunc_vals=dict(zip(cols, b.tolist())),
                trunc_n_days=len(trunc_bars.index.normalize().unique()),
                elapsed=elapsed)


def index_mismatch_diagnostic(bars: pd.DataFrame, lppls_daily: pd.DataFrame) -> dict:
    """Demonstrates, on real data, why the DAILY-indexed frame cannot be
    fed directly into `step_a_gate` (see module docstring's
    INDEX-ALIGNMENT NOTE): reindexing the daily confidence series onto a
    5-minute bar window drops to near-zero coverage."""
    from experiments.r141_shared import episode_window
    label, onset_str = STRESS_EPISODES[0]
    onset, window = episode_window(bars, onset_str, 60)
    sub = lppls_daily["lppls_confidence"].reindex(window)
    coverage = int(sub.notna().sum())
    total = len(sub)
    return dict(episode=label, coverage=coverage, total=total, frac=coverage / total)


def main() -> dict:
    print("=" * 78)
    print("R-141 CONSERVATIVE branch: LPPLS confidence vs v4 anchor")
    print("STEP A detection-lag gate (identical machinery to R-01/R-82/R-83/R-85/R-86/R-139)")
    print("=" * 78)

    bars, label = load_btc_train("spot")
    assert_no_holdout(bars)
    print(f"\nBTC ({label}): {len(bars):,} bars  {bars.index[0]} -> {bars.index[-1]}  "
          f"(< {OOS_START})")

    print("\n--- LPPLS signal (shared, cached) ---")
    lppls_daily = lppls_daily_signals(bars, cache_path=LPPLS_CACHE, verbose=True)
    lppls_bar = lppls_bar_signals(bars, cache_path=LPPLS_CACHE, verbose=True)
    assert_no_holdout(pd.DataFrame(index=lppls_daily.index))

    print("\n--- Index-alignment diagnostic (daily-indexed frame fed directly) ---")
    mism = index_mismatch_diagnostic(bars, lppls_daily)
    print(f"  episode window: {mism['episode']}")
    print(f"  reindexing DAILY confidence onto the 5-min bar window: "
          f"{mism['coverage']}/{mism['total']} bars covered "
          f"({mism['frac']:.4%}) -- silently near-empty.")
    print("  => using lppls_bar_signals (bar-aligned, ffilled) as the `lppls` "
          "argument to step_a_gate instead, matching r139's own precedent "
          "(cusum_daily_causal_signals, bar-aligned) for the same gate shape.")

    print("\n" + "=" * 78)
    print(f"STEP-A SWEEP: conf_thresh in {CONF_THRESH_GRID}")
    print("=" * 78)

    sweep_results = []
    for thresh in CONF_THRESH_GRID:
        n_qualify_min = round(thresh * 5)
        print(f"\n--- conf_thresh={thresh} (n_qualify >= {n_qualify_min}/5) ---")
        gate = step_a_gate(bars, lppls_bar, conf_thresh=thresh, verbose=True)
        assert_no_holdout(bars)
        sweep_results.append(gate)
        print(f"  n_pass={gate['n_pass']}/6  passed={gate['passed']}  "
              f"genuine_top={gate['n_pass_genuine_top']}/{gate['n_genuine_top']}")

    print("\n" + "=" * 78)
    print("FULL PER-EPISODE TABLE, EVERY conf_thresh SWEPT")
    print("=" * 78)
    header = f"{'episode':42s} {'conf_thresh':>11s} {'lead(d)':>9s} {'null_median(d)':>15s} {'PASS':>6s}"
    print(header)
    print("-" * len(header))
    for gate in sweep_results:
        for r in gate["results"]:
            lead = r.get("lead", float("nan"))
            null_median = r.get("null_median", float("nan"))
            print(f"{r['label']:42s} {gate['conf_thresh']:>11.2f} {lead:>+9.2f} "
                  f"{null_median:>+15.2f} {str(r['pass_b']):>6s}")

    print("\n" + "=" * 78)
    print("SUMMARY BY THRESHOLD")
    print("=" * 78)
    any_passed = False
    for gate in sweep_results:
        print(f"  conf_thresh={gate['conf_thresh']:.2f}  n_pass={gate['n_pass']}/6  "
              f"GATE={'PASS' if gate['passed'] else 'FAIL'}  "
              f"genuine_top_subset={gate['n_pass_genuine_top']}/{gate['n_genuine_top']}")
        any_passed = any_passed or gate["passed"]

    print("\n" + "=" * 78)
    print("CAUSAL TRUNCATION PROBE (own new code path: full-grid lppls_daily_signals)")
    print("=" * 78)
    probe = causal_truncation_probe(bars)
    print(f"  check date: {probe['check_date']}  (a genuinely qualifying calibration date, "
          f"n_qualify={probe['full_vals']['lppls_n_qualify']:.0f}/5 in the full run)")
    print(f"  full-series values:      {probe['full_vals']}")
    print(f"  truncated-series values: {probe['trunc_vals']}  "
          f"(recomputed from scratch on {probe['trunc_n_days']} days, {probe['elapsed']:.1f}s, "
          f"cache_path=None)")
    print(f"  PROBE RESULT: {'PASS' if probe['ok'] else 'FAIL'} "
          f"(bit-identical={'yes' if probe['ok'] else 'NO -- LOOKAHEAD SUSPECTED'})")
    assert probe["ok"], "LPPLS causal-truncation probe failed on this branch's own code path"

    print("\n" + "=" * 78)
    if not any_passed:
        print("FINAL VERDICT (pre-registered stop rule, r141_shared.py docstring):")
        print(f"  No conf_thresh in {CONF_THRESH_GRID} reached n_pass >= {STOP_RULE_MIN_PASS}/6.")
        print("  STOP AT STEP A. NEGATIVE. No Step-B combination logic implemented.")
        print("  No OOS/holdout data touched.")
        print("  This CONSERVATIVE branch's LPPLS confidence indicator, swept over its")
        print("  pre-registered threshold grid, does not detect the six dated historical")
        print("  BTC regime transitions with lead time at or better than v4's own")
        print("  anchor-crossing reaction, at a rate clearing this project's pre-registered")
        print("  >=4/6 bar -- the same failure mode every one of the six prior structurally")
        print("  distinct detectors (HMM, BOCPD, Kalman LLT, critical slowing down, transfer")
        print("  entropy, CUSUM) hit against this identical gate.")
    else:
        print("FINAL VERDICT: some conf_thresh reaches n_pass >= 4 -- Step-A gate PASSES.")
        print("  Proceeding to Step B (confirming_vote_frac combination, weight swept over")
        print("  {0.5, 1.0, 1.5}), per r141_shared.py's pre-registered specification.")

    print(f"\nconfigurations evaluated in this file: {len(CONF_THRESH_GRID)} "
          f"(conf_thresh sweep at Step A)")
    max_ts_touched = max(bars.index.max(), lppls_daily.index.max().tz_convert(bars.index.tz)
                          if lppls_daily.index.tz else lppls_daily.index.max())
    print(f"max timestamp read anywhere in this file's run: {max_ts_touched}  (< {OOS_START})")

    return dict(sweep_results=sweep_results, any_passed=any_passed, probe=probe,
                max_ts=max_ts_touched)


if __name__ == "__main__":
    main()
