#!/usr/bin/env python
"""Bootstrap / path-resample the CONSERVATIVE dual-book drawdown claim (B-16, step 1).

``experiments/kelly_regime_dual_fixed.py`` found that ``kelly_regime_v4``
run independently on BTC and ETH with a fixed capital split, summed,
improves max drawdown by -4pp to -7pp versus ``kelly_regime_v4`` on
BTC alone -- but on exactly ONE window, the mandated 2021-01-01..2022-12-31
inner-validation slice, which happens to contain the 2022 BTC/ETH joint
bear. R-42's ledger row calls that "the same N~3 fragility this branch set
out to attack" and B-16 reopens it with one instruction for this half of
the pair: bootstrap it.

This file does NOT re-implement the dual-book runner. It imports
``run_dual``, ``run_baseline_v4_btc``, ``SPLITS``, ``exposure_artifact_check``,
``BTC``, ``ETH``, and the market/window constants from
``kelly_regime_dual_fixed.py`` UNCHANGED, and does not modify that file (or
``kelly_regime_covkelly.py``, or anything under ``src/tradebot/strategies/``)
in any way. It only adds a NEW axis on top: many resampled calendar
windows, drawn the way this project's own ``scripts/stress_test.py``
already draws them for R-19/R-33/R-36 (random start, random length, a
seeded RNG, a warmup prefix so short windows are not penalised for a cold
start, trading disabled until the window's own start) -- extended here to
the two-asset case.

Mechanism (one sentence, written before any code ran)
-------------------------------------------------------
If the dual book's inner-validation drawdown improvement is a genuine
diversification effect and not a one-window accident, it should survive
being measured on many *different* resampled windows drawn from the same
pre-2023 history, not just the one the project happened to pick as
"inner-validation".

Pre-registered failure mode (written before any code ran)
-------------------------------------------------------------
Per R-42's own falsification result, BTC/ETH correlation spikes
specifically during crashes. If that is the dominant effect, the
drawdown-delta should be small or even reversed on the windows that
contain the worst BTC-alone drawdowns (the "bear" windows, defined below
as a proxy via BTC-alone severity) and only look good on calm windows --
which would mean the R-42 headline is not "diversification helps in a
crash", it is "diversification helps when there is no crash to help
with", which is a much weaker and much less useful claim. This is
measured directly below (see ``bear_calm_report``), not asserted.

Constraint attacked
--------------------
N~3 -- same as R-42's conservative branch. This file adds nothing new on
that axis by itself; it only asks whether R-42's own N~3-attacking result
is itself resting on N=1.

Not a duplicate of
-------------------
- ``kelly_regime_dual_fixed.py`` (R-42, conservative): that file computes
  the fixed-split dual book on exactly two fixed periods (inner-train,
  inner-validation) and one bear window. This file resamples ~40 NEW
  windows from the same pre-2023 calendar range and asks whether the
  headline finding replicates in DISTRIBUTION -- a different question,
  answered with the SAME runner functions, imported unchanged.
- ``scripts/stress_test.py`` (R-19/R-33/R-36): that script resamples
  windows for SINGLE-asset strategies on ONE price series, and implements
  the warmup-prefix / trade-start-disable logic by hand via
  ``run_backtest(..., trade_start=eval_start)``. This file needs the
  identical property for TWO assets at once and gets it for free by
  calling ``run_period`` (via the sibling file's ``run_leg``/``run_dual``),
  which already implements the same warmup-prefix / trade-disable
  contract as a general calendar-window primitive -- so the window here is
  specified as (start date, end date) rather than (bar position, bar
  length), and no new warmup-handling code is written in this file at all.

Simulable here?
----------------
Yes -- every window is backtested via the same ``run_period`` call every
other experiment file in this repo uses, on committed real OHLCV. No
engine change, no new strategy code, no re-implementation of the dual
book.

What would make this fail (falsification, restated as a decision question)
----------------------------------------------------------------------------
Does the median per-window drawdown delta (dual minus BTC-alone) stay
negative (an improvement) with a bootstrap 95% CI that excludes zero, AND
does that hold specifically on the top-quartile-severity ("bear") windows,
not just the calm ones? If either fails, the R-42 headline does not
survive contact with a distribution and should be downgraded from "a
result" to "one lucky window".

Window-resampling convention
-----------------------------
Same convention this project already uses for Monte Carlo window
resampling (``scripts/stress_test.py``'s CLI defaults, used for
R-19/R-33/R-36): ``trials=40``, ``min_days=90``, ``max_days=730``,
``seed=42``. This is NOT the identical `windows.csv` those rows publish
(that file is drawn from bar positions on the single BTC series; here the
window bounds are calendar dates drawn from ETH's own bar grid, since ETH
is the shorter/binding series for this two-asset case and its own warmup
must fit before the window start) -- so exact window-for-window
comparability to `windows.csv` is not claimed, only convention
comparability (same trial count, same day-length range, same seed
philosophy). This is stated plainly per this task's instruction rather
than implied.

Every window's start position is drawn no earlier than ``kelly_regime_v4``'s
own warmup (23,050 bars, ~80 days) into ETH's pre-2023 history, and every
window's end is capped at 2021-01-01..2022-12-31's own right edge,
``VALID[1] = "2022-12-31"`` -- i.e. strictly inside the calendar range this
whole B-16 round is restricted to. NOTHING in this file reads, slices, or
prints a number derived from 2023-01-01 onward: grep this file for
"2023"/"2024"/"2025"/"2026" to confirm -- only variable names, comments
describing that boundary, and citation years appear.

Usage
-----
    python experiments/kelly_regime_dual_bootstrap.py run         # step 3/4: the full resample, spot
    python experiments/kelly_regime_dual_bootstrap.py run --futures  # + 5x futures secondary check
    python experiments/kelly_regime_dual_bootstrap.py artifact    # exposure-artifact re-check, vol_weighted
    python experiments/kelly_regime_dual_bootstrap.py causality   # window-boundary self-check
    python experiments/kelly_regime_dual_bootstrap.py all
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

from tradebot.registry import get_strategy  # noqa: E402

# Everything dual-book-related is imported UNCHANGED from the sibling
# conservative-branch file -- this file adds a resampling harness on top,
# it does not re-derive the runner.
from experiments.kelly_regime_dual_fixed import (  # noqa: E402
    BTC,
    ETH,
    FUTURES,
    INCUMBENT,
    SPOT,
    TRAIN,
    VALID,
    exposure_artifact_check,
    run_baseline_v4_btc,
    run_dual,
)

BARS_PER_DAY = 288

# --------------------------------------------------------------- resampling convention
# Same trial count / day-length range / seed philosophy as
# scripts/stress_test.py's own CLI defaults (--trials 40 --min-days 90
# --max-days 730 --seed 42), extended to calendar-date windows on ETH's
# grid. See the "Window-resampling convention" docstring section above for
# why this is convention-comparable rather than window-for-window
# identical to windows.csv.
WINDOW_TRIALS = 40
WINDOW_MIN_DAYS = 90
WINDOW_MAX_DAYS = 730
WINDOW_SEED = 42

# The two split candidates this task asks for: the flagship 50/50 and the
# train-only vol_weighted split, both already members of SPLITS in the
# sibling file -- neither is a new configuration invented here.
CANDIDATE_SPLITS = ("50_50", "vol_weighted")

MARKETS = (("spot", SPOT), ("futures", FUTURES))

OUT = ROOT / "reports" / "kelly_regime_dual_bootstrap"

# Raw backtest-execution counter (leg-level run_backtest calls made via
# run_period, through run_dual/run_baseline_v4_btc). This is DISTINCT from
# the ROUTINE.md "configurations searched" trials count: resampling
# ALREADY-CHOSEN configs (50_50, vol_weighted, BTC-alone control -- all
# three already evaluated in R-42) onto new windows is a robustness check,
# not a parameter search, exactly the distinction R-33/R-36 already draw
# between "N configs" and "40 windows" in their own ledger rows. See the
# end-of-run report for both numbers, reported separately and honestly.
N_BACKTESTS = 0


# --------------------------------------------------------------------- window generation


def build_windows() -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """~40 resampled (start, end, days) windows, calendar-bounded to pre-2023.

    Mirrors scripts/stress_test.py's own generation loop (length drawn
    first, then a start position, same RNG call order) but on ETH's own
    bar grid truncated at VALID[1] ("2022-12-31") -- ETH is the shorter,
    binding series here, and its own kelly_regime_v4 warmup (not BTC's)
    is what determines how early a window may start with a full,
    non-cold prefix. BTC has far more history before any of these start
    dates, so its own leg is never cold; only ETH's floor matters.

    No price content of any kind (open/high/low/close/volume) is read by
    this function -- only ETH's DatetimeIndex length and the RNG. There is
    therefore no channel through which a window boundary could depend on
    future price data; the causality risk this file introduces is only
    "does a window ever creep past 2022-12-31", checked by the assertion
    below on every single generated window, not by price-tamper probes
    (which are already covered, unchanged, by the sibling file's own
    ``causality()`` for run_leg/run_dual/combine_equity themselves).
    """
    warmup = get_strategy(INCUMBENT).warmup
    pool = ETH.loc[:VALID[1]]  # strictly <= 2022-12-31; nothing later is ever touched
    boundary = pd.Timestamp(VALID[1], tz="UTC") + pd.Timedelta(hours=23, minutes=55)

    rng = np.random.default_rng(WINDOW_SEED)
    windows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for _ in range(WINDOW_TRIALS):
        length_days = int(rng.integers(WINDOW_MIN_DAYS, WINDOW_MAX_DAYS + 1))
        length_bars = length_days * BARS_PER_DAY
        start_pos = int(rng.integers(warmup, len(pool) - length_bars))
        start_ts = pool.index[start_pos]
        end_ts = pool.index[start_pos + length_bars - 1]
        assert end_ts <= boundary, f"window end {end_ts} crosses the pre-2023 boundary"
        windows.append((start_ts, end_ts, length_days))
    return windows


# --------------------------------------------------------------------- per-window evaluation


def run_window(start: pd.Timestamp, end: pd.Timestamp, market) -> dict:
    """Control (BTC-alone v4) + both split candidates, on one resampled window."""
    global N_BACKTESTS
    ctl = run_baseline_v4_btc(start, end, market)
    N_BACKTESTS += 1  # one BTC leg
    out = {
        "control_final": ctl["final"], "control_sharpe": ctl["sharpe"], "control_dd": ctl["max_dd"],
    }
    for split in CANDIDATE_SPLITS:
        d = run_dual(split, start, end, market, count=False)
        N_BACKTESTS += 2  # BTC leg + ETH leg
        out[f"{split}_final"] = d["final"]
        out[f"{split}_sharpe"] = d["sharpe"]
        out[f"{split}_dd"] = d["max_dd"]
        out[f"{split}_delta_dd"] = d["max_dd"] - ctl["max_dd"]  # negative = improvement
        out[f"{split}_delta_final"] = d["final"] - ctl["final"]
    return out


def run_all_windows(windows: list[tuple[pd.Timestamp, pd.Timestamp, int]],
                     markets=(("spot", SPOT),)) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for k, (start, end, days) in enumerate(windows, 1):
        for mname, market in markets:
            r = run_window(start, end, market)
            r.update({"trial": k, "start": start, "end": end, "days": days, "market": mname})
            rows.append(r)
        print(f"[{k}/{len(windows)}] {start:%Y-%m-%d}..{end:%Y-%m-%d} (+{days}d)  "
              f"control_dd={rows[-1]['control_dd']:.1f}%  "
              f"50_50_delta={rows[-1]['50_50_delta_dd']:+.1f}pp  "
              f"vol_w_delta={rows[-1]['vol_weighted_delta_dd']:+.1f}pp  "
              f"[{time.time() - t0:.0f}s, {N_BACKTESTS} backtests so far]")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- bootstrap CI


def percentile_bootstrap_median(deltas: np.ndarray, n_boot: int = 10_000,
                                 seed: int = 123, level: float = 0.95) -> tuple[float, float, float]:
    """95% percentile bootstrap CI on the MEDIAN of a set of per-window deltas.

    Deliberately NOT ``inference.py``'s ``stationary_bootstrap_indices`` /
    ``paired_bootstrap``: those resample an autocorrelated DAILY RETURN
    series within a single equity curve (they need a block length because
    consecutive days are dependent). Here the unit of observation is
    already a single scalar summary per window (that window's own
    max-drawdown delta) -- there is no within-unit time series left to
    block-bootstrap. Resampling WITH REPLACEMENT across the N window-level
    deltas is the direct empirical analogue of "what would the median look
    like under a different draw of resampled windows", which is exactly
    the quantity this task asks for. This is a plain percentile bootstrap,
    not the studentized/BCa variety -- adequate for a diagnostic check, not
    claimed to be more than that.

    Caveat stated plainly: the WINDOWS themselves can overlap in calendar
    time (this file does not deduplicate overlapping windows, matching
    scripts/stress_test.py's own convention), so the 40 deltas are not
    fully independent draws and the true effective N is somewhat below 40.
    This inflates the bootstrap's apparent precision to an unknown degree;
    it is not corrected for here, and the report says so.
    """
    rng = np.random.default_rng(seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.median(deltas[idx], axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.percentile(draws, [100 * tail, 100 * (1 - tail)])
    return float(np.median(deltas)), float(lo), float(hi)


def iqr(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))


# --------------------------------------------------------------------- reporting


def distribution_report(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Median/IQR/CI/win-fraction of the drawdown delta, per split candidate."""
    sub = df[df.market == market]
    rows = []
    print(f"\n=== drawdown-delta distribution, {market}, n={sub['trial'].nunique()} windows ===")
    for split in CANDIDATE_SPLITS:
        deltas = sub[f"{split}_delta_dd"].to_numpy(dtype=float)
        med, lo, hi = percentile_bootstrap_median(deltas)
        q25, q75 = iqr(deltas)
        frac_improved = float(np.mean(deltas < 0)) * 100.0
        frac_worse = float(np.mean(deltas > 0)) * 100.0
        frac_flat = 100.0 - frac_improved - frac_worse
        sig = "excludes zero" if (lo > 0 or hi < 0) else "CONTAINS ZERO"
        print(f"  {split:14s} median={med:+.2f}pp  IQR=[{q25:+.2f}, {q75:+.2f}]  "
              f"95% CI=[{lo:+.2f}, {hi:+.2f}] ({sig})  "
              f"improved={frac_improved:.0f}%  worse={frac_worse:.0f}%  flat={frac_flat:.0f}%")
        rows.append({"market": market, "split": split, "median_delta_dd": med,
                     "ci_lo": lo, "ci_hi": hi, "q25": q25, "q75": q75,
                     "frac_improved": frac_improved, "frac_worse": frac_worse})
    return pd.DataFrame(rows)


def bear_calm_report(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Same distribution, split by BTC-alone severity (top quartile vs rest).

    Severity proxy = that window's OWN ``control_dd`` (kelly_regime_v4,
    BTC-alone, max drawdown realized IN THAT WINDOW) -- not an external
    label, so it needs no calendar lookup and cannot leak 2023+ information.
    Top-quartile-severity windows stand in for "bear/crash-heavy"; the rest
    stand in for "calm", per this task's explicit instruction to segment
    this way.
    """
    sub = df[df.market == market].copy()
    thresh = float(sub["control_dd"].quantile(0.75))
    bear = sub[sub["control_dd"] >= thresh]
    calm = sub[sub["control_dd"] < thresh]
    print(f"\n=== bear-vs-calm split, {market} (top-quartile BTC-alone DD >= {thresh:.1f}%, "
          f"n_bear={len(bear)}, n_calm={len(calm)}) ===")
    rows = []
    for label, part in (("bear (top quartile)", bear), ("calm (rest)", calm)):
        for split in CANDIDATE_SPLITS:
            deltas = part[f"{split}_delta_dd"].to_numpy(dtype=float)
            if len(deltas) < 2:
                print(f"  {label:20s} {split:14s} n={len(deltas)} (too few for a CI)")
                continue
            med, lo, hi = percentile_bootstrap_median(deltas)
            print(f"  {label:20s} {split:14s} n={len(deltas):2d}  median={med:+.2f}pp  "
                  f"95% CI=[{lo:+.2f}, {hi:+.2f}]")
            rows.append({"market": market, "segment": label, "split": split,
                         "n": len(deltas), "median_delta_dd": med, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main pipeline


def run(include_futures: bool = False) -> None:
    windows = build_windows()
    print(f"{len(windows)} resampled windows: {windows[0][0]:%Y-%m-%d}..{windows[0][1]:%Y-%m-%d} "
          f"... spanning {min(w[0] for w in windows):%Y-%m-%d} to {max(w[1] for w in windows):%Y-%m-%d}, "
          f"seed={WINDOW_SEED}, {WINDOW_MIN_DAYS}-{WINDOW_MAX_DAYS} days each")

    markets = [("spot", SPOT)] + ([("futures", FUTURES)] if include_futures else [])
    df = run_all_windows(windows, markets=markets)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "windows_results.csv", index=False)
    print(f"\nwritten: {OUT / 'windows_results.csv'}")

    dist_rows = []
    bearcalm_rows = []
    for mname, _ in markets:
        dist_rows.append(distribution_report(df, mname))
        bearcalm_rows.append(bear_calm_report(df, mname))
    pd.concat(dist_rows, ignore_index=True).to_csv(OUT / "distribution_summary.csv", index=False)
    pd.concat(bearcalm_rows, ignore_index=True).to_csv(OUT / "bear_calm_summary.csv", index=False)
    print(f"written: {OUT / 'distribution_summary.csv'}")
    print(f"written: {OUT / 'bear_calm_summary.csv'}")

    n_configs = len(CANDIDATE_SPLITS) + 1  # 2 split candidates + BTC-alone control
    print(f"\ndistinct configurations resampled: {n_configs} (control BTC-alone v4, "
          f"{', '.join(CANDIDATE_SPLITS)}) -- ALL three already members of R-42's evaluated set, "
          "so this branch adds 0 NEW configurations to the project's running "
          "trials-search count (same convention R-33/R-36 use for window-resampling "
          "an already-chosen config: the search count tracks distinct PARAMETERIZATIONS, "
          "not repeated resampled evaluations of one).")
    print(f"total leg-level backtests actually executed in this run: {N_BACKTESTS} "
          f"({len(windows)} windows x {len(markets)} market(s) x 5 leg-runs/window-market: "
          "1 control BTC leg + 2 split candidates x 2 legs each)")


def causality() -> None:
    """Window-boundary self-check: every generated window obeys the pre-2023 cutoff.

    See build_windows()'s own docstring for why a generic price-tamper
    probe is not the relevant check for THIS file's own additions (window
    selection reads no price content at all) -- the actual risk this file
    could introduce is a window silently crossing the 2023-01-01 boundary,
    which is what this asserts, on every one of the 40 windows, not just
    spot-checked.
    """
    windows = build_windows()
    boundary = pd.Timestamp(VALID[1], tz="UTC") + pd.Timedelta(hours=23, minutes=55)
    worst_end = max(end for _, end, _ in windows)
    ok = worst_end <= boundary
    print(f"windows generated: {len(windows)}")
    print(f"latest window end across all {len(windows)}: {worst_end}  boundary: {boundary}  "
          f"{'PASS' if ok else 'FAIL'}")
    earliest_start = min(start for start, _, _ in windows)
    print(f"earliest window start: {earliest_start}  (ETH real data starts {ETH.index[0]})")
    warmup = get_strategy(INCUMBENT).warmup
    pool = ETH.loc[:VALID[1]]
    min_pos = int(pool.index.searchsorted(earliest_start))
    ok2 = min_pos >= warmup
    print(f"earliest window's ETH bar position: {min_pos}  required warmup: {warmup}  "
          f"{'PASS' if ok2 else 'FAIL'} (a full, non-cold ETH warmup precedes every window)")
    print(f"\noverall: {'PASS' if ok and ok2 else 'FAIL'}")


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"BTC: {len(BTC):,} bars {BTC.index[0]:%Y-%m-%d} -> {BTC.index[-1]:%Y-%m-%d}", file=sys.stderr)
    print(f"ETH: {len(ETH):,} bars {ETH.index[0]:%Y-%m-%d} -> {ETH.index[-1]:%Y-%m-%d}", file=sys.stderr)
    args = sys.argv[1:]
    choice = args[0] if args else ""
    include_futures = "--futures" in args
    if choice == "run":
        run(include_futures=include_futures)
    elif choice == "artifact":
        exposure_artifact_check("vol_weighted")
        exposure_artifact_check("50_50")
    elif choice == "causality":
        causality()
    elif choice == "all":
        run(include_futures=True)
        exposure_artifact_check("vol_weighted")
        exposure_artifact_check("50_50")
        causality()
    else:
        print("usage: python experiments/kelly_regime_dual_bootstrap.py "
              "[run [--futures]|artifact|causality|all]")
