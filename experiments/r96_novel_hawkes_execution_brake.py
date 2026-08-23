#!/usr/bin/env python
"""R-96 NOVEL branch: Step-0 sub-claim gate for a Hawkes-clustering-intensity
EXECUTION-TIMING BRAKE on ``kelly_regime_v4`` -- "does a Hawkes-intensity
spike predict elevated near-term turbulence?" -- run BEFORE any delay/
execution-brake strategy code, in the same "bounded delay, then force
through" architecture R-77 (closed NEGATIVE) and R-88's novel branch
(``r88_novel_taker_flow_delay.py``) already validated for other signals,
but keyed on Hawkes CLUSTERING INTENSITY (the conditional rate of further
price jumps given recent jump timing) rather than volatility level (R-77)
or order-flow direction (R-88).

=====================================================================
PRE-REGISTRATION (frozen before any Hawkes-intensity or forward-turbulence
number in this file was computed -- docs/ROUTINE.md steps 1-2). Anything
below later contradicted by what actually happened is stated in the
results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence, full citation trail and "not a duplicate of"
   list already established in ``r96_shared.py``'s own module docstring,
   not re-derived here): a self-exciting Hawkes process's conditional
   intensity (Hawkes 1971; Bacry-Mastromatteo-Muzy 2015) rising after a
   cluster of price jumps means further jumps are, by the model's own
   construction, more likely SOON -- if that is also true of realized
   turbulence in this project's own data (not merely assumed from the
   literature), a scheduled ``kelly_regime_v4`` rebalance that lands
   during such a spike could be postponed up to K bars (re-checking each
   bar whether the intensity has fallen back below threshold -- execute
   immediately -- or the deadline has been reached -- force the trade
   through regardless, so signal timing is never permanently overridden),
   exactly the bounded-delay-then-force shape R-88's novel branch used for
   ``tv_z``, reused here with a structurally different driving signal (a
   univariate, price-only conditional EVENT RATE, not a bivariate
   order-flow imbalance).

   THIS FILE DOES NOT TEST THE 6-EPISODE LEAD-TIME GATE. That is the
   CONSERVATIVE branch's own, separate, independent pre-registration
   (regime-timing ALARM role, tested against ``r96_shared.STRESS_EPISODES``
   exactly as R-82/83/84/85/86 were). This file's own, separate,
   independent Step-0 gate is below -- a measurement/falsification test in
   the same family as R-82 through R-86's Step-A gates, testing whether
   the EXECUTION-BRAKE mechanism's own premise (elevated near-term
   turbulence after a spike) holds at all, before any delay/execution
   code is built. Passing this gate does not imply the 6-episode alarm
   gate would also pass, or vice versa -- the two branches' gates are
   independent falsification tests of independent claims about the same
   underlying signal, exactly as this round's dispatch specifies.

2. STEP-0 SUB-CLAIM TEST -- "does a Hawkes-intensity spike predict
   elevated near-term turbulence?"

   a. PRIMARY PRE-REGISTERED CONFIGURATION: ``n=0.5, halflife_days=7`` --
      the CENTER of ``r96_shared.N_GRID=(0.3,0.5,0.7)`` and
      ``r96_shared.HALFLIFE_DAYS_GRID=(3,7,14)``, the same center-of-grid
      choice the CONSERVATIVE branch uses, for direct comparability and to
      avoid an undisclosed multi-cell search. THIS IS THE ONE
      DECISION-BEARING CELL evaluated by this file. The other 8 grid
      cells (``N_GRID`` x ``HALFLIFE_DAYS_GRID`` minus the center) are NOT
      tested at Step-0 at all, by design -- this branch's own trial count
      stays at exactly 1 decision-bearing configuration.

   b. DATA WINDOW: the FULL pre-holdout BTC series, ``< OOS_START``
      (2017 through 2022 inclusive) -- the same window
      ``r96_shared.STRESS_EPISODES`` are drawn from, NOT only the
      2017-2020 inner-train slice. This is a measurement/falsification
      gate in the same family as R-82 through R-86's Step-A gates, which
      this project has consistently run against the full pre-holdout
      window rather than the narrower inner-train slice -- inner-train /
      inner-validation is for parameter FITTING, which this gate does
      none of (every constant used below -- ``n``, ``halflife_days``,
      ``Z_THRESH``, ``K``, ``block_days``, ``n_draws``, ``seed`` -- is
      fixed a priori, before any real-data number was computed, and none
      of them is retuned after seeing a result).

   c. SPIKE-DAY DEFINITION: a calendar day ``t`` where the daily Hawkes-
      intensity z-score (``r96_shared.hawkes_intensity_zscore`` of
      ``r96_shared.hawkes_intensity_daily`` of
      ``r96_shared.intraday_relative_jump(bars)``, with ``n=0.5,
      halflife_days=7``) crosses UP through ``Z_THRESH=2.0`` (value < 2.0
      on day ``t-1``, >= 2.0 on day ``t``, BOTH values required to be
      finite -- a day immediately following the z-score's own warmup
      period, where day ``t-1`` is NaN, is not counted as a crossing by
      construction, since "value < 2.0 on day t-1" is undefined, not
      satisfied, when ``t-1`` has no z-score yet). EVERY such day across
      the whole pre-holdout series is collected, not just the 6 named
      ``STRESS_EPISODES``.

   d. FORWARD OUTCOME, daily cadence:
      - ``RV_fwd(t, K)`` = realized variance (sum of squared 5-minute log
        returns, i.e. daily RV summed across days) over the STRICTLY
        FOLLOWING K calendar days ``t+1 .. t+K`` (excluding day ``t``
        itself, so there is no overlap with the jump/spike-detection
        day's own statistics). ``K=3`` is THE SINGLE PRE-REGISTERED
        PRIMARY DECISION WINDOW. ``K=1`` and ``K=7`` are also computed and
        reported as secondary/robustness context only.
      - ``Whipsaw_fwd(t, K)`` = count of ``anchor_majority`` value changes
        (any direction), among the K forward days themselves (i.e. up to
        K-1 adjacent-day transitions; the transition from day t to day
        t+1 is not counted, matching ``RV_fwd``'s own exclusion of day
        t) -- a SECONDARY / CORROBORATING measure, reported but NOT part
        of the pass/fail decision (point f below).
      - Any day ``t`` whose forward window would require a bar dated on
        or after ``OOS_START`` is NaN by construction (the truncated
        bars frame this file loads never contains such a bar in the
        first place, so the forward-window lookup simply finds no data
        rather than needing an explicit extra guard -- verified with
        ``assert_no_holdout`` at every stage regardless).

   e. NULL DISTRIBUTION: ``r96_shared.block_bootstrap_shifts`` circularly
      block-shifts the spike-day 0/1 indicator series (``block_days=5,
      n_draws=500, seed=9601`` -- reusing the identical seed the
      CONSERVATIVE branch uses, fixed a priori, disclosed, never altered
      after seeing a result) over the FULL pre-holdout DAILY index (one
      entry per calendar day, ~2,191 rows for 2017-2022); for each of the
      500 shifted copies, the mean ``RV_fwd(., K)`` is recomputed at the
      (shifted) "spike" days. This gives a null distribution for "mean
      forward RV conditional on an arbitrary day being flagged", against
      which the true conditional mean is compared.

      DISCLOSED IMPLEMENTATION NOTE, found while wiring this up, not
      assumed from the docstring: ``block_bootstrap_shifts``'s
      ``block_days`` parameter is converted to a raw-row block size via
      ``block = int(block_days * BARS_PER_DAY)`` (``BARS_PER_DAY=288``),
      i.e. it is calibrated for 5-MINUTE BAR cadence (as R-82 through
      R-86 and the CONSERVATIVE branch of this round all use it). Called
      here with ``n_bars`` = number of DAYS (~2,191), ``block=5*288=1440``
      exceeds half of ``n_bars``, so every draw falls into the function's
      own documented fallback (``n_bars <= 2*block``): a single uniform
      random circular shift of the WHOLE daily array, rather than a
      genuine multi-block reshuffle at 5-day granularity. This is still a
      legitimate circular-shift null (it preserves the real spike-day
      clustering structure exactly, and only randomizes its PHASE against
      the calendar), just not the finer block structure the identical
      call produces at 5-minute-bar cadence -- disclosed here, not
      silently assumed to behave identically, and NOT worked around by
      writing a second, un-reviewed block-bootstrap routine: the task
      calls for reusing ``r96_shared.block_bootstrap_shifts`` on the
      daily index exactly as specified, so that is what this file does.

   f. PRE-REGISTERED PASS BAR (frozen now): the TRUE mean ``RV_fwd(., 3)``
      at real spike days must exceed the null distribution's 95th
      percentile (one-sided: real spike days show significantly MORE
      forward turbulence than an arbitrary day would). THIS IS THE ONLY
      QUANTITY THE PASS/FAIL DECISION DEPENDS ON. ``Whipsaw_fwd`` and the
      ``K=1``/``K=7`` windows are reported as corroborating context only,
      explicitly NOT allowed to override this one pre-registered primary
      test -- named now, to prevent an "OR of several tests" p-hacking
      multiplicity after seeing results.

   g. STOP RULE: if the primary test in (f) does NOT clear its bar, STOP
      -- this file's result is the novel branch's ENTIRE product, written
      up NEGATIVE, and no delay/execution-brake strategy code is built.
      If it DOES clear, this file still does not build full backtest/
      strategy code: the passing result is reported, and the decision of
      whether to proceed to a full strategy build belongs to the operator
      synthesizing both branches -- this keeps this branch's own trial
      count at exactly 1 decision-bearing configuration, per this
      project's "count every configuration evaluated" discipline.

3. WHAT WOULD MAKE THIS FAIL, named now (copied near-verbatim from
   ``r96_shared.py``'s own docstring, last paragraph): if realized
   volatility/whipsaw frequency in the bars immediately following a
   Hawkes-intensity spike is NOT significantly elevated relative to the
   unconditional baseline, delaying execution during a cluster buys
   nothing and the branch must stop at this pre-registered gate before
   any delay mechanism is built or backtested.

CONFIGURATIONS EVALUATED IN THIS FILE: 1 (the single pre-registered
primary cell, ``n=0.5, halflife_days=7``, K=3 RV test). K=1/K=7 and the
whipsaw diagnostic are reported context computed alongside that one cell,
not separate swept configurations competing for selection.

USAGE
-----
    python experiments/r96_novel_hawkes_execution_brake.py
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

from tradebot.data import load_dataset  # noqa: E402

from experiments.r96_shared import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    DETECTION_WINDOW_DAYS,
    OOS_START,
    Z_THRESH,
    anchor_majority,
    assert_no_holdout,
    block_bootstrap_shifts,
    hawkes_intensity_daily,
    hawkes_intensity_zscore,
    intraday_relative_jump,
)

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------- pre-registered
N_PRIMARY = 0.5
HALFLIFE_PRIMARY = 7
K_GRID = (1, 3, 7)
K_PRIMARY = 3
BLOCK_DAYS = 5
N_DRAWS = 500
NULL_SEED = 9601


# ---------------------------------------------------------------- data load


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


# ---------------------------------------------------------- daily builders


def daily_rv(df: pd.DataFrame) -> pd.Series:
    """Daily realized variance (sum of squared 5-minute log returns),
    identical calendar-day grouping to ``r96_shared.intraday_relative_jump``
    (``day = index.floor('D')``, UTC-tz daily index) so this series lines
    up exactly with ``jump_flag`` / ``lam`` / ``z``'s own index."""
    close = df["close"]
    r = np.log(close).diff()
    day = df.index.floor("D")
    frame = pd.DataFrame({"r": r.to_numpy(), "day": day})
    frame = frame.dropna(subset=["r"])
    rv_by_day = frame.groupby("day")["r"].apply(lambda g: float(np.sum(g.to_numpy() ** 2)))
    rv_by_day.index = pd.DatetimeIndex(rv_by_day.index, tz="UTC")
    return rv_by_day.rename("rv")


def daily_majority_last(bars: pd.DataFrame) -> pd.Series:
    """``anchor_majority`` resampled to daily cadence: the value at each
    day's LAST 5-minute bar."""
    majority = anchor_majority(bars)
    day = majority.index.floor("D")
    daily = majority.groupby(day).last()
    daily.index = pd.DatetimeIndex(daily.index, tz="UTC")
    return daily.rename("majority")


def forward_window_sum(daily: pd.Series, k: int) -> pd.Series:
    """``RV_fwd(t, k)`` = sum of ``daily`` over the strictly-following k
    calendar days ``t+1..t+k``. NaN if any of those k days is missing from
    ``daily``'s own index (insufficient forward history OR the day would
    fall on/after OOS_START, which this file's truncated input never
    contains in the first place)."""
    idx = daily.index
    full_range = pd.date_range(idx.min(), idx.max(), freq="D", tz="UTC")
    vals = daily.reindex(full_range).to_numpy()
    n = len(vals)
    pos = pd.Series(np.arange(n), index=full_range)
    out = pd.Series(np.nan, index=idx)
    idx_pos = pos.reindex(idx).to_numpy()
    for j, i in enumerate(idx_pos):
        i = int(i)
        if i + k < n:
            window = vals[i + 1: i + 1 + k]
            if not np.any(np.isnan(window)):
                out.iloc[j] = float(np.sum(window))
    return out.rename(f"fwd_sum_{k}d")


def forward_whipsaw_count(daily_majority: pd.Series, k: int) -> pd.Series:
    """``Whipsaw_fwd(t, k)`` = count of value changes among the k forward
    days ``t+1..t+k`` themselves (up to k-1 adjacent transitions). NaN
    under the identical missing-forward-data condition as
    ``forward_window_sum``."""
    idx = daily_majority.index
    full_range = pd.date_range(idx.min(), idx.max(), freq="D", tz="UTC")
    vals = daily_majority.reindex(full_range).to_numpy()
    n = len(vals)
    pos = pd.Series(np.arange(n), index=full_range)
    out = pd.Series(np.nan, index=idx)
    idx_pos = pos.reindex(idx).to_numpy()
    for j, i in enumerate(idx_pos):
        i = int(i)
        if i + k < n:
            window = vals[i + 1: i + 1 + k]
            if not np.any(np.isnan(window)) and k >= 2:
                out.iloc[j] = float(np.sum(window[1:] != window[:-1]))
            elif not np.any(np.isnan(window)):
                out.iloc[j] = 0.0  # k==1: no adjacent pair to compare, 0 by construction
    return out.rename(f"fwd_whipsaw_{k}d")


# ------------------------------------------------------------- spike days


def find_spike_days(z: pd.Series, z_thresh: float = Z_THRESH) -> pd.DatetimeIndex:
    """Every day ``t`` where ``z`` crosses UP through ``z_thresh``: value
    < thresh on day t-1 AND value >= thresh on day t, both required finite
    (a day right after warmup, where t-1 is NaN, is not counted -- "value
    < thresh on day t-1" is undefined there, not satisfied)."""
    vals = z.to_numpy()
    finite = np.isfinite(vals)
    high = finite & (vals >= z_thresh)
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = high[1:] & ~high[:-1] & finite[:-1]
    return z.index[cross]


# --------------------------------------------------------------- null test


def null_mean_rv(flag: np.ndarray, rv_fwd: np.ndarray, block_days: int,
                  n_draws: int, seed: int) -> np.ndarray:
    """500 circular-block-shifted copies of the spike-day 0/1 indicator
    (via ``r96_shared.block_bootstrap_shifts``, see the disclosed
    implementation note in this file's module docstring re: block
    granularity at daily cadence); for each, the mean ``rv_fwd`` at the
    (shifted) flagged days."""
    n_days = len(flag)
    shifts = block_bootstrap_shifts(n_bars=n_days, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    out = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted_flag = flag[shift]
        pos = np.where(shifted_flag == 1)[0]
        vals = rv_fwd[pos]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[k] = float(np.mean(vals))
    return out


# ------------------------------------------------------------ causal probe


def causality_probe(bars: pd.DataFrame) -> bool:
    """Does the daily z-score pipeline (``intraday_relative_jump`` ->
    ``hawkes_intensity_daily`` -> ``hawkes_intensity_zscore``) at a fixed
    check date change if bars strictly after it are dropped?

    DISCLOSED, not a call to ``r96_shared.truncation_causality_probe``
    directly: that helper's ``check_at``/``shorter_by`` convention
    (``short = build_target_fn(df.iloc[:check_at + shorter_by])``, then
    ``full[check_at]``) assumes ``build_target_fn`` returns an array
    aligned 1:1 with ``df``'s own 5-minute-bar rows -- true for every
    prior round's usage (the CONSERVATIVE branch of this round upsamples
    onto bars via ``align_daily_causal`` first). This file's Step-0 test
    works entirely in DAILY cadence (per this file's own spec), so its z
    array has far fewer rows than bars and ``check_at`` cannot
    simultaneously index both a bar-slice bound and a daily-array
    position (verified by hand before writing this function: forcing
    ``check_at``/``shorter_by`` to satisfy both roles at once requires
    truncating away almost nothing, defeating the point of the probe).
    This function asks the IDENTICAL question -- does the causal target at
    a fixed point change under tail truncation -- directly against the
    daily-indexed output, the same adaptation R-86's NOVEL branch made
    explicit ("a bespoke ... version of the identical idea") when the
    generic bar-cadence helper did not fit its own bivariate pipeline.
    """
    check_date = pd.Timestamp("2020-06-01", tz="UTC")
    keep_until = check_date + pd.Timedelta(days=200)

    def build(df: pd.DataFrame) -> pd.Series:
        jump_flag = intraday_relative_jump(df)
        lam = hawkes_intensity_daily(jump_flag, n=N_PRIMARY, halflife_days=HALFLIFE_PRIMARY)
        return hawkes_intensity_zscore(lam, detection_window_days=DETECTION_WINDOW_DAYS,
                                        baseline_window_days=BASELINE_WINDOW_DAYS)

    full = build(bars)
    trunc = bars.loc[bars.index <= keep_until].copy()
    short = build(trunc)

    def get(series: pd.Series, date: pd.Timestamp) -> float:
        if date not in series.index:
            return float("nan")
        return float(series.loc[date])

    a_full, a_short = get(full, check_date), get(short, check_date)
    ok = bool(np.isclose(a_full, a_short, equal_nan=True))
    print(f"  check_date={check_date.date()}  keep_until={keep_until.date()}  "
          f"(full series max {bars.index.max().date()})")
    print(f"  z(check_date) on full series:      {a_full:.6f}")
    print(f"  z(check_date) on truncated series: {a_short:.6f}")
    print(f"  CAUSAL-TRUNCATION PROBE: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------- main


def main() -> dict:
    t0 = time.time()
    print("=" * 90)
    print("R-96 NOVEL: Hawkes execution-brake -- STEP 0 sub-claim gate")
    print("  'does a Hawkes-intensity spike predict elevated near-term turbulence?'")
    print("=" * 90)

    bars = load_btc_bars()

    jump_flag = intraday_relative_jump(bars)
    lam = hawkes_intensity_daily(jump_flag, n=N_PRIMARY, halflife_days=HALFLIFE_PRIMARY)
    z = hawkes_intensity_zscore(lam, detection_window_days=DETECTION_WINDOW_DAYS,
                                 baseline_window_days=BASELINE_WINDOW_DAYS)
    assert_no_holdout(z.to_frame())

    rv = daily_rv(bars)
    assert_no_holdout(rv.to_frame())
    daily_maj = daily_majority_last(bars)
    assert_no_holdout(daily_maj.to_frame())

    n_jump_days = int(jump_flag.sum())
    print(f"\nprimary config: n={N_PRIMARY}, halflife_days={HALFLIFE_PRIMARY}  "
          f"(center of N_GRID={(0.3, 0.5, 0.7)} x HALFLIFE_DAYS_GRID={(3, 7, 14)})")
    print(f"daily index: {len(z):,} days  {z.index[0].date()} -> {z.index[-1].date()}")
    print(f"jump-event days (RJ_THRESH crossed, all-time): {n_jump_days:,} / {len(jump_flag):,}")

    spike_days = find_spike_days(z, Z_THRESH)
    n_spikes = len(spike_days)
    print(f"\nspike days (z crosses UP through Z_THRESH={Z_THRESH}): {n_spikes}")
    if n_spikes:
        print("  " + ", ".join(d.date().isoformat() for d in spike_days))

    flag = np.zeros(len(z), dtype=float)
    flag[z.index.isin(spike_days)] = 1.0

    fwd_rv = {k: forward_window_sum(rv, k) for k in K_GRID}
    fwd_whip = {k: forward_whipsaw_count(daily_maj, k) for k in K_GRID}

    print("\n" + "-" * 90)
    print("forward-turbulence results by K (days), spike days vs. block-bootstrap null")
    print(f"null: r96_shared.block_bootstrap_shifts(block_days={BLOCK_DAYS}, "
          f"n_draws={N_DRAWS}, seed={NULL_SEED})")
    print("-" * 90)

    results = {}
    for k in K_GRID:
        rv_fwd_k = fwd_rv[k].reindex(z.index).to_numpy()
        whip_fwd_k = fwd_whip[k].reindex(z.index).to_numpy()

        spike_pos = np.where(flag == 1.0)[0]
        spike_rv_vals = rv_fwd_k[spike_pos]
        spike_rv_vals_valid = spike_rv_vals[~np.isnan(spike_rv_vals)]
        true_mean_rv = float(np.mean(spike_rv_vals_valid)) if len(spike_rv_vals_valid) else float("nan")
        n_spikes_with_data = len(spike_rv_vals_valid)

        spike_whip_vals = whip_fwd_k[spike_pos]
        spike_whip_vals_valid = spike_whip_vals[~np.isnan(spike_whip_vals)]
        true_mean_whip = float(np.mean(spike_whip_vals_valid)) if len(spike_whip_vals_valid) else float("nan")

        null = null_mean_rv(flag, rv_fwd_k, BLOCK_DAYS, N_DRAWS, NULL_SEED)
        valid_null = null[~np.isnan(null)]
        null_p95 = float(np.percentile(valid_null, 95)) if len(valid_null) else float("nan")
        null_median = float(np.median(valid_null)) if len(valid_null) else float("nan")

        clears = (not np.isnan(true_mean_rv)) and (not np.isnan(null_p95)) and (true_mean_rv > null_p95)

        tag = "PRIMARY (decision-bearing)" if k == K_PRIMARY else "secondary/context only"
        print(f"\nK={k}d  [{tag}]")
        print(f"    spike days with valid forward data: {n_spikes_with_data}/{n_spikes}")
        print(f"    TRUE mean RV_fwd(.,{k}) at spike days:      {true_mean_rv:.8f}")
        print(f"    null mean RV_fwd  median / p95 ({len(valid_null)} valid draws): "
              f"{null_median:.8f} / {null_p95:.8f}")
        print(f"    true mean > null p95: {clears}")
        print(f"    (context) TRUE mean Whipsaw_fwd(.,{k}) at spike days: {true_mean_whip:.4f}")

        results[k] = dict(true_mean_rv=true_mean_rv, null_p95=null_p95,
                           null_median=null_median, clears=clears,
                           true_mean_whip=true_mean_whip,
                           n_spikes_with_data=n_spikes_with_data,
                           n_null_valid=len(valid_null))

    primary = results[K_PRIMARY]
    passed = primary["clears"]

    print("\n" + "=" * 90)
    print("PRE-REGISTERED PASS BAR (K=3 RV test ONLY -- Whipsaw and K=1/K=7 are")
    print("corroborating context, explicitly NOT allowed to override this test):")
    print(f"  TRUE mean RV_fwd(.,3) = {primary['true_mean_rv']:.8f}   "
          f"null p95 = {primary['null_p95']:.8f}")
    print(f"STEP-0 GATE VERDICT: {'PASS' if passed else 'FAIL (NEGATIVE)'}")
    print("=" * 90)

    print("\n" + "-" * 90)
    print("CAUSAL-TRUNCATION PROBE")
    print("-" * 90)
    probe_ok = causality_probe(bars)

    print(f"\nconfigurations evaluated (decision-bearing): 1  (n={N_PRIMARY}, "
          f"halflife_days={HALFLIFE_PRIMARY}, K={K_PRIMARY})")
    print(f"max timestamp read anywhere in this file: {bars.index.max()}  (< {OOS_START})")

    if not passed:
        print("\n" + "#" * 90)
        print("# STEP-0 GATE FAILED ITS PRE-REGISTERED PASS BAR.")
        print("# Per this file's own pre-registration: STOP HERE. This gate result is")
        print("# this branch's ENTIRE product, written up NEGATIVE. No delay/")
        print("# execution-brake strategy code is built. No data on/after 2023-01-01")
        print("# is touched.")
        print("#" * 90)
    else:
        print("\n" + "#" * 90)
        print("# STEP-0 GATE PASSED. Per this file's own pre-registration, this file")
        print("# still does NOT build any delay/execution-brake strategy code -- that")
        print("# decision belongs to the operator synthesizing both branches. This")
        print("# passing measurement result is this file's entire output.")
        print("#" * 90)

    print(f"\n[{time.time()-t0:.0f}s]")

    return dict(bars=bars, z=z, spike_days=spike_days, results=results,
                passed=passed, probe_ok=probe_ok)


if __name__ == "__main__":
    main()
