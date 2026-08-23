#!/usr/bin/env python
"""R-99 NOVEL branch: Step-0 sub-claim gate for a bipower-variation
JUMP-DAY forward-loss kill switch/de-risking overlay on ``kelly_regime_v4``
-- "does forward realized loss over the days immediately following an
unusually large realized JUMP component differ from the unconditional
baseline enough to justify de-risking afterward?" -- run BEFORE any
kill-switch/de-risking strategy code, in the same "Step-0 sub-claim gate
before any strategy is built" architecture R-96's novel branch
(``r96_novel_hawkes_execution_brake.py``) and R-98's novel branch
(``r98_novel_gpd_killswitch.py``) used, applied here to
Barndorff-Nielsen & Shephard bipower-variation jump detection instead of
Hawkes-intensity spikes or a POT/GPD tail-VaR breach.

=====================================================================
PRE-REGISTRATION (frozen before any real-data big-jump-day or forward-loss
number in this file was computed -- docs/ROUTINE.md steps 1-2). Anything
below later contradicted by what actually happened is stated in the
results section, not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence; full citation trail, mathematical
   construction, and "not a duplicate of" list already established in
   ``r99_shared.py``'s own module docstring, not re-derived here): does
   forward N-day realized loss after a day flagged as a "big jump day"
   (``RJ_t`` -- the Huang & Tauchen 2005 relative jump measure computed by
   ``r99_shared.daily_rv_bv_jump`` -- exceeding the CAUSAL trailing
   730-calendar-day 95th percentile of its own past, per
   ``r99_shared.JUMP_EVENT_QUANTILE = 0.95``) differ from the unconditional
   baseline enough to justify a kill-switch/de-risking overlay?

   THIS FILE DOES NOT TEST THE 6-EPISODE LEAD-TIME GATE. That is the
   CONSERVATIVE branch's own, separate, independent pre-registation
   (``experiments/r99_conservative_bv_jump_alarm.py``, a regime-timing
   ALARM role tested against ``r99_shared.STRESS_EPISODES``, exactly as
   R-82/83/84/85/86/96/98 tested their own alarms). This file's own Step-0
   gate below tests a genuinely different, adjacent claim: not "does jump
   activity LEAD a slow multi-week regime transition" but "does a SINGLE
   big-jump day predict elevated DAMAGE over the following 1-10 days" --
   an unconditional, day-by-day forward-loss measurement, before any
   kill-switch code is built. This is the identical Step-0-vs-Step-A
   separation R-96 and R-98 each used on their own underlying statistic.

2. THE BIG-JUMP-DAY FLAG CONSTRUCTION (exact, causal, verified below by a
   truncation probe before any downstream number is trusted):

       trailing_q_t  = rolling_quantile( RJ_{t-1}, RJ_{t-2}, ..., RJ_{t-W} ;
                                          q = JUMP_EVENT_QUANTILE = 0.95 )
       is_jump_day_t = 1  if RJ_t > trailing_q_t  else 0

   where ``W = r99_shared.PRIMARY_BASELINE_WINDOW_DAYS = 730`` calendar
   days (the SAME 730-day window the conservative branch's own PRIMARY
   cell uses for its baseline -- reused here, not re-picked, so both
   branches' "how much trailing history counts as the baseline" choice is
   the identical, non-degeneracy-gated value) with
   ``min_periods=JUMP_TRAILING_MIN_PERIODS=180`` calendar days -- an
   EXPANDING window before the full 730 days of history accumulate,
   disclosed here rather than swept: a day with fewer than 180 days of
   prior RJ history has no computable flag at all (excluded from the
   eligible set, not defaulted to "not a jump day" as a hidden zero -- see
   the "eligible days" count in the results section). ``trailing_q`` is
   computed on ``RJ.shift(1).rolling(730, min_periods=180)`` -- i.e.
   STRICTLY on RJ values dated before day ``t``, never including ``RJ_t``
   itself -- and this construction is verified by an explicit
   causal-truncation probe (section below) that the flag on a fixed
   check-point day is bit-for-bit identical whether or not bars dated
   after that check-point exist in the input at all.

3. FORWARD-LOSS HORIZONS, IDENTICAL GRID to R-98's novel branch for direct
   comparability: ``N in {1, 3, 5, 10}`` calendar days. For a jump day
   ``t``, ``Loss(t, N) = -(sum of daily log returns over the STRICTLY
   FOLLOWING days t+1 .. t+N)`` -- a positive number means price fell over
   the following ``N`` days. This is computed causally in the sense that
   matters for a leakage guard (the FLAG at day ``t`` never uses any
   day > t), even though the OUTCOME variable itself necessarily looks
   forward from ``t`` -- that is what "forward loss" means, and is not a
   causality violation: no information from ``t+1..t+N`` is fed back into
   the flag construction in section 2 above.

   HOLDOUT BOUNDARY HANDLING: because a forward-loss window near the very
   end of the pre-holdout period can require days on or after
   ``OOS_START = 2023-01-01``, and this branch is contractually forbidden
   from ever reading such a day, every jump day whose ``t+N`` window is
   not ENTIRELY contained inside the pre-holdout daily-return series is
   dropped from that horizon's sample (``NaN``, never filled) -- disclosed
   explicitly, per-horizon, as an exclusion count in the results section
   below, not silently absorbed into a shrunk denominator.

4. THE NULL: ``r99_shared.block_bootstrap_shifts_daily`` (a daily-cadence
   circular block-shift, unlike ``r99_shared.block_bootstrap_shifts``
   which is calibrated for 5-minute-bar cadence and hits a documented
   fallback at daily block sizes -- see R-98's own disclosed quirk; this
   round avoids that quirk entirely by using the daily-native helper),
   ``block_days=5`` (a working-week block, matching R-98's own
   daily-cadence null block size), ``n_draws=500``,
   ``seed=9902`` (a FRESH seed for this file -- the conservative branch
   uses 9901, R-98's own novel branch used 9801; no seed is reused across
   branches or rounds). The flag's own 0/1 array is circularly shifted 500
   times (preserving its own temporal clustering, randomizing only its
   calendar phase against the fixed forward-loss series), and the mean
   ``Loss(., N)`` at the shifted "jump days" is recomputed each time,
   giving a null mean/std/95th-percentile for "mean forward loss
   conditional on an arbitrary-phase day being flagged".

5. PRE-REGISTERED STOP RULE, IDENTICAL bar to R-98's novel branch: at
   horizon ``N``, PASS if the TRUE mean forward loss on real jump days
   exceeds the null distribution's own empirical 95th percentile
   (``null_p95(N)``). Need >= 3 of the 4 horizons to PASS to justify
   proceeding to build a kill-switch/de-risking strategy. If < 3/4: STOP
   HERE, write NO strategy code -- this file's job is the Step-0
   measurement alone. (In the event Step-0 unexpectedly clears >= 3/4 --
   named now, before running, as a possibility this pre-registration does
   not foreclose -- this file still stops at the Step-0 measurement and
   reports PASS; building and evaluating an actual kill-switch strategy
   class, promotion-bar comparison, and ETH falsification, per R-98's own
   section-4 template, is explicitly out of scope for this file and is
   left as a follow-up round, not silently added here after seeing a
   favorable number.)

6. WHAT WOULD MAKE THIS FAIL, named now, before any real-data number
   exists (this section is the honest, pre-registered EXPECTED outcome,
   not a hedge added after the fact): Andersen, Bollerslev & Diebold
   (2007) found the JUMP component of realized variance behaves close to
   i.i.d. and is far less persistent/forecastable than the CONTINUOUS
   component (full citation and quote already given in
   ``r99_shared.py``'s own module docstring, "WHAT WOULD MAKE THIS FAIL"
   section -- not re-derived here). If that finding holds on this
   project's own BTC series too, a single big-jump day carries little
   memory of its own: it resolves within one day by construction (a jump
   is a discontinuity, not a multi-day drift), and there is no strong a
   priori reason it should predict elevated DAMAGE over the following 1-10
   days, independent of whether jump ACTIVITY clusters near regime
   transitions (the separate claim the conservative branch's own Step-A
   gate tests). The expected result, stated now, is that most or all of
   the four horizons FAIL to clear their null bar, mirroring R-96's novel
   branch (Hawkes: NO) and R-98's novel branch (POT/GPD: only 1 of 4
   horizons cleared) -- a third consecutive Step-0 sub-claim test on this
   axis coming back mostly negative would be additional convergent
   evidence that "a rare-event flag computed from this project's own
   price history predicts forward damage" is not a reliable premise on
   this series, independent of which formal statistic supplies the flag.

CONFIGURATIONS EVALUATED IN THIS FILE: Step-0 = 4 (the horizon grid
``N in {1,3,5,10}``, at the one fixed PRIMARY jump-flag construction --
not swept). No further configurations are evaluated regardless of
Step-0's outcome (see section 5's disclosed scope limit).

USAGE
-----
    python experiments/r99_novel_bv_jump_forward_loss.py
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

from experiments.r99_shared import (  # noqa: E402
    JUMP_EVENT_QUANTILE,
    OOS_START,
    PRIMARY_BASELINE_WINDOW_DAYS,
    align_daily_causal,
    assert_no_holdout,
    block_bootstrap_shifts_daily,
    daily_rv_bv_jump,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------- pre-registered
N_GRID = (1, 3, 5, 10)
NULL_BLOCK_DAYS = 5
NULL_DRAWS = 500
NULL_SEED = 9902  # fresh: conservative branch uses 9901, R-98 novel used 9801
STEP0_PASS_HORIZONS_NEEDED = 3  # of 4
JUMP_TRAILING_MIN_PERIODS = 180  # calendar days; expanding window before 730d accumulate


# ---------------------------------------------------------------- data load


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


# --------------------------------------------------------- Step-0 machinery


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily close-to-close log return, from this project's own 5-minute
    bars resampled to daily close (last bar of each UTC day). Mirrors
    ``r98_shared.daily_log_returns``'s own construction (not imported --
    this round's dependency graph is confined to ``r99_shared`` per
    dispatch) so both rounds' forward-loss numbers stay directly
    comparable. Entirely causal by construction: day t's value depends
    only on bars dated on or before day t."""
    daily_close = df["close"].resample("1D").last().dropna()
    r = np.log(daily_close).diff().dropna()
    return r.rename("daily_log_ret")


def forward_loss(daily_ret: pd.Series, n: int) -> pd.Series:
    """``Loss(t, n)`` = ``-(sum of daily_ret[t+1 .. t+n])``; NaN if the
    n-day forward window is not entirely available inside ``daily_ret``'s
    own index range (no peeking past what this series actually contains --
    since ``daily_ret`` never contains a day >= OOS_START, this also IS
    the holdout-boundary exclusion mechanism disclosed in section 3 of
    this file's pre-registration). Identical construction to
    ``r98_novel_gpd_killswitch.forward_loss``, for direct comparability."""
    idx = daily_ret.index
    full_range = pd.date_range(idx.min(), idx.max(), freq="D", tz="UTC")
    vals = daily_ret.reindex(full_range).to_numpy()
    m = len(vals)
    pos = pd.Series(np.arange(m), index=full_range)
    idx_pos = pos.reindex(idx).to_numpy()
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx_pos):
        i = int(i)
        if i + n < m:
            window = vals[i + 1: i + 1 + n]
            if not np.any(np.isnan(window)):
                out[j] = -float(np.sum(window))
    return pd.Series(out, index=idx, name=f"loss_{n}d")


def build_jump_flag_daily(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The causal big-jump-day flag itself, at daily cadence: returns
    ``(rj, trailing_q, is_jump)``, all indexed by UTC calendar day. See
    section 2 of this file's pre-registration for the exact formula."""
    daily = daily_rv_bv_jump(df)
    assert_no_holdout(daily)
    rj = daily["rj"]
    trailing_q = rj.shift(1).rolling(
        PRIMARY_BASELINE_WINDOW_DAYS, min_periods=JUMP_TRAILING_MIN_PERIODS
    ).quantile(JUMP_EVENT_QUANTILE)
    eligible = trailing_q.notna() & rj.notna()
    is_jump = pd.Series(
        np.where(eligible & (rj > trailing_q), 1.0, np.where(eligible, 0.0, np.nan)),
        index=daily.index,
    )
    return rj, trailing_q, is_jump


def build_jump_flag_aligned(df: pd.DataFrame) -> np.ndarray:
    """Bar-aligned version of the big-jump-day flag (via
    ``align_daily_causal``), for the causal-truncation probe, which
    requires an array indexable by the SAME integer bar positions as
    ``df``'s own 5-minute index (identical role to R-98's own
    ``build_breach_aligned``)."""
    _, _, is_jump = build_jump_flag_daily(df)
    aligned = align_daily_causal(is_jump.fillna(0.0), df)
    return aligned.to_numpy()


def causality_probe(bars: pd.DataFrame) -> bool:
    """Does the bar-aligned big-jump-day flag at a fixed check date change
    if bars strictly after it are dropped? Run BEFORE trusting any Step-0
    number, per this round's dispatch. ``check_date=2020-06-01`` is chosen
    with ~3.5 years of prior BTC history already available, i.e. well past
    the 730-day trailing window's own full accumulation point, so the
    check is exercised on a day where the flag is not trivially NaN/zero
    for lack of history."""
    check_date = pd.Timestamp("2020-06-01", tz="UTC")
    check_at = int(bars.index.searchsorted(check_date))
    ok = truncation_causality_probe(build_jump_flag_aligned, bars, check_at=check_at,
                                     shorter_by=20_000)
    print(f"  check_date={check_date.date()}  check_at bar index={check_at}  "
          f"(~{bars.index[check_at].date()})  shorter_by=20,000 bars")
    print(f"  CAUSAL-TRUNCATION PROBE (big-jump-day flag): {'PASS' if ok else 'FAIL'}")
    return ok


def null_mean_loss(flag: np.ndarray, loss_arr: np.ndarray, block_days: int,
                    n_draws: int, seed: int) -> np.ndarray:
    """500 circular-block-shifted copies of the jump-day 0/1 indicator
    (via ``r99_shared.block_bootstrap_shifts_daily``, a genuine daily-
    cadence block shift -- see section 4 of this file's pre-registration
    for why this avoids R-98's own disclosed bar-cadence-helper fallback
    quirk); for each, the mean ``loss_arr`` at the (shifted) flagged
    days."""
    n_days = len(flag)
    shifts = block_bootstrap_shifts_daily(n_days=n_days, block_days=block_days,
                                           n_draws=n_draws, seed=seed)
    out = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted_flag = flag[shift]
        pos = np.where(shifted_flag == 1.0)[0]
        vals = loss_arr[pos]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[k] = float(np.mean(vals))
    return out


def run_step0(bars: pd.DataFrame) -> dict:
    assert_no_holdout(bars)
    rj, trailing_q, is_jump = build_jump_flag_daily(bars)

    n_days_total = len(rj)
    n_days_valid_rj = int(rj.notna().sum())
    n_eligible = int(is_jump.notna().sum())
    n_jump_days = int((is_jump == 1.0).sum())

    print(f"\ndaily RV/BV/RJ frame: {n_days_total} calendar days "
          f"({rj.index[0].date()} -> {rj.index[-1].date()}); "
          f"{n_days_valid_rj} with valid RJ "
          f"(NaN days are short-bar-count days below MIN_BARS_PER_DAY)")
    print(f"causal trailing-quantile flag: window={PRIMARY_BASELINE_WINDOW_DAYS}d "
          f"min_periods={JUMP_TRAILING_MIN_PERIODS}d quantile={JUMP_EVENT_QUANTILE} "
          f"(reuses r99_shared.PRIMARY_BASELINE_WINDOW_DAYS, the conservative branch's "
          f"own non-degeneracy-chosen baseline window)")
    print(f"eligible days (trailing quantile computable): {n_eligible}/{n_days_total}")
    print(f"flagged big-jump days: {n_jump_days} "
          f"({100.0 * n_jump_days / max(n_eligible, 1):.2f}% of eligible days; "
          f"target is (1 - {JUMP_EVENT_QUANTILE}) = "
          f"{100.0 * (1.0 - JUMP_EVENT_QUANTILE):.1f}% by construction of the quantile)")

    daily_ret = daily_log_returns(bars)
    assert_no_holdout(daily_ret.to_frame())

    flag_aligned = is_jump.reindex(daily_ret.index).fillna(0.0)
    flag_arr = flag_aligned.to_numpy()
    jump_dates_all = daily_ret.index[flag_arr == 1.0]
    print(f"jump days re-aligned onto daily-return index: {int(flag_arr.sum())} "
          f"(should equal {n_jump_days} up to index intersection)")

    results = {}
    for n in N_GRID:
        loss_n = forward_loss(daily_ret, n)
        assert_no_holdout(loss_n.to_frame())
        loss_arr = loss_n.to_numpy()

        jump_pos = np.where(flag_arr == 1.0)[0]
        jump_vals_raw = loss_arr[jump_pos]
        valid_mask = ~np.isnan(jump_vals_raw)
        jump_vals = jump_vals_raw[valid_mask]
        n_valid = len(jump_vals)
        n_dropped = len(jump_pos) - n_valid
        dropped_dates = jump_dates_all[~valid_mask]
        true_mean = float(np.mean(jump_vals)) if n_valid else float("nan")

        null = null_mean_loss(flag_arr, loss_arr, NULL_BLOCK_DAYS, NULL_DRAWS, NULL_SEED)
        valid_null = null[~np.isnan(null)]
        null_mean = float(np.mean(valid_null)) if len(valid_null) else float("nan")
        null_std = float(np.std(valid_null)) if len(valid_null) else float("nan")
        null_p95 = float(np.percentile(valid_null, 95)) if len(valid_null) else float("nan")
        z = (true_mean - null_mean) / null_std if null_std else float("nan")

        clears = (not np.isnan(true_mean)) and (not np.isnan(null_p95)) and (true_mean > null_p95)

        print(f"\nN={n}d")
        print(f"  jump days with valid forward data: {n_valid}/{len(jump_pos)}  "
              f"(dropped {n_dropped} -- forward window incomplete, i.e. would require "
              f"reading a day >= {OOS_START} or otherwise past the series' own end)")
        if n_dropped:
            print(f"    dropped dates: {[d.date().isoformat() for d in dropped_dates]}")
        print(f"  TRUE mean Loss(.,{n}) at jump days:      {true_mean:+.6f}")
        print(f"  null mean/std/p95 ({len(valid_null)} valid draws): "
              f"{null_mean:+.6f} / {null_std:.6f} / {null_p95:+.6f}")
        print(f"  effect size (z = (true-null_mean)/null_std): {z:+.3f}")
        print(f"  true_mean > null_p95: {clears}")

        results[n] = dict(true_mean=true_mean, null_mean=null_mean, null_std=null_std,
                           null_p95=null_p95, clears=clears, n_valid=n_valid,
                           n_dropped=n_dropped, z=z)

    n_pass = sum(1 for r in results.values() if r["clears"])
    passed = n_pass >= STEP0_PASS_HORIZONS_NEEDED
    return dict(results=results, n_pass=n_pass, passed=passed, n_jump_days=n_jump_days,
                n_eligible=n_eligible, n_days_total=n_days_total,
                n_days_valid_rj=n_days_valid_rj)


# --------------------------------------------------------------------- main


def hdr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def main() -> dict:
    t0 = time.time()

    hdr("R-99 NOVEL: bipower-variation big-jump-day forward-loss gate -- STEP 0 sub-claim test")
    print("'does forward N-day realized loss after an unusually large realized JUMP")
    print(" component exceed the unconditional baseline enough to justify a kill-switch/")
    print(" de-risking overlay?'")
    print("\nPRE-REGISTERED EXPECTED FAILURE MODE (named before any number below was")
    print("computed): Andersen-Bollerslev-Diebold (2007) found the jump component of")
    print("realized variance is close to i.i.d. and far less persistent than the")
    print("continuous component -- if that holds here too, a single big-jump day should")
    print("show LITTLE OR NO significantly elevated forward loss vs. the unconditional")
    print("baseline. See this file's own module docstring, section 6, for the full")
    print("citation-grounded statement.")

    bars = load_btc_bars()

    hdr("CAUSAL-TRUNCATION PROBE (run before trusting any Step-0 number)")
    probe_ok = causality_probe(bars)

    hdr("STEP 0 -- forward-loss test, N in {1,3,5,10} calendar days, PRIMARY jump-flag config")
    step0 = run_step0(bars)

    print("\n" + "-" * 96)
    print(f"STEP-0 SUMMARY: {step0['n_pass']}/4 horizons clear (true_mean > null_p95); "
          f"need >= {STEP0_PASS_HORIZONS_NEEDED}/4 to pass. "
          f"big-jump days (pre-holdout, eligible period): {step0['n_jump_days']}")
    print(f"STEP-0 GATE VERDICT: {'PASS' if step0['passed'] else 'FAIL (NEGATIVE)'}")
    print("-" * 96)

    print(f"\nconfigurations evaluated in this file (decision-bearing): 4 "
          f"(N in {N_GRID}, PRIMARY jump-flag config only)")
    print(f"max timestamp read anywhere in this branch: {bars.index.max()}  (< {OOS_START})")

    if not step0["passed"]:
        print("\n" + "#" * 96)
        print("# STEP-0 GATE FAILED ITS PRE-REGISTERED PASS BAR.")
        print("# Per this file's own pre-registration: STOP HERE. This gate result is")
        print("# this branch's ENTIRE product, written up NEGATIVE. No kill-switch/")
        print("# de-risking strategy code is built. No data on/after 2023-01-01 is")
        print("# touched anywhere in this file.")
        print("#" * 96)
    else:
        print("\n" + "#" * 96)
        print("# STEP-0 GATE PASSED its pre-registered pass bar (>= 3/4 horizons).")
        print("# Per section 5 of this file's own pre-registration, building and")
        print("# evaluating an actual kill-switch/de-risking strategy class is OUT OF")
        print("# SCOPE for this file -- this file's job is the Step-0 measurement")
        print("# alone. No strategy code is built here; that is left as a deliberate")
        print("# follow-up round, not silently added after seeing a favorable number.")
        print("# No data on/after 2023-01-01 is touched anywhere in this file.")
        print("#" * 96)

    print(f"\n[{time.time()-t0:.0f}s]")
    return dict(bars=bars, step0=step0, probe_ok=probe_ok, passed=step0["passed"],
                max_ts=bars.index.max(), n_configs=4)


if __name__ == "__main__":
    main()
