#!/usr/bin/env python
"""R-98 NOVEL branch: Step-0 sub-claim gate for a POT/GPD tail-VaR-BREACH
KILL SWITCH on ``kelly_regime_v4`` -- "does forward realized loss in the
bars immediately following a live POT/GPD VaR breach differ from the
unconditional baseline enough to justify standing flat afterward?" -- run
BEFORE any kill-switch strategy code, in the same "Step-0 sub-claim gate
before any strategy is built" architecture R-96's novel branch
(``r96_novel_hawkes_execution_brake.py``) used, applied here to a
gate/kill-switch role (force flat after a tail event) rather than an
execution-timing delay.

=====================================================================
PRE-REGISTRATION (frozen before any GPD/VaR or forward-loss number in this
file was computed -- docs/ROUTINE.md steps 1-2). Anything below later
contradicted by what actually happened is stated in the results section,
not edited back into this banner.
=====================================================================

1. MECHANISM (one sentence, full citation trail and "not a duplicate of"
   list already established in ``r98_shared.py``'s own module docstring,
   not re-derived here): does forward realized loss in the bars
   immediately following a live POT/GPD VaR breach
   (``r98_shared.rolling_gpd_signal(...)['breach']``, at
   ``VAR_PROB=0.99``, using the PRIMARY config
   ``PRIMARY_THRESH_QUANTILE=0.90, PRIMARY_FIT_WINDOW_DAYS=730`` already
   chosen in ``r98_shared.py`` for non-degeneracy -- NOT re-picked here)
   differ from the unconditional baseline enough to justify standing flat
   afterward?

   THIS FILE DOES NOT TEST THE 6-EPISODE LEAD-TIME GATE. That is the
   CONSERVATIVE branch's own, separate, independent pre-registration
   (regime-timing ALARM role, tested against ``r98_shared.STRESS_EPISODES``
   exactly as R-82/83/84/85/86/96 tested their own alarms). This file's
   own Step-0 gate below is a measurement/falsification test of a
   genuinely different claim -- not "does the tail estimator lead a named
   historical transition" but "does a live breach carry forward-loss
   information at all, on an unconditional day-by-day basis" -- before any
   kill-switch code is built.

2. THE NAMED WHIPSAW RISK (B-41/R-90, read in full before writing this
   section): R-90 built a path-dependent trailing-stop/kill-switch overlay
   on ``kelly_regime_v4`` gated on RECLAIM-OF-THRESHOLD re-entry (``close[i]
   > exit_price`` required to re-arm) and found this construction is
   mechanically near-definitional with its own whipsaw diagnostic -- a
   whipsaw is *defined* as "re-entry at a close higher than the exit
   close", and a reclaim-gated re-entry condition requires exactly that
   same inequality to fire at all, so whenever the mechanism re-enters, it
   is close to guaranteed to whipsaw by construction (R-90's novel branch:
   whipsaw_rate = 1.000 exactly, independently reproduced by the
   operator). B-41 is CLOSED, NEGATIVE, specifically because of this trap
   (docs/LEDGER.md, R-90 section, "A genuinely reusable structural
   finding" paragraph).

   THIS BRANCH'S RE-ENTRY DESIGN, CHOSEN NOW TO AVOID THAT TRAP: re-entry
   is a FIXED K-CALENDAR-DAY COOLDOWN, timed from the breach day itself,
   with NO reclaim-of-any-threshold condition anywhere in the re-entry
   rule. Once K days have elapsed, ``kelly_regime_v4``'s own vote/scale
   resumes exactly as it would have computed with no override at all --
   regardless of whether the GPD breach/VaR condition has since cleared,
   regardless of price level, regardless of anything path-dependent. The
   trigger condition (``|daily return| > VaR_t``, a tail-magnitude
   comparison) and the re-entry condition (elapsed calendar days) do not
   share an inequality, a sign, or even a common variable, which is
   exactly the structural property R-90's finding says a safe re-entry
   rule needs to have. This does not guarantee low whipsaw -- it removes
   the *definitional* mechanism that made R-90's rate hit 1.000, and
   whipsaw is still measured and reported below as B-41 requires, not
   assumed safe.

3. STEP-0 SUB-CLAIM TEST -- "does forward realized loss after a live
   breach exceed the unconditional baseline?"

   a. CONFIG: the PRIMARY GPD cell only (``quantile=0.90,
      fit_window_days=730``, ``VAR_PROB=0.99``) -- ``r98_shared``'s own
      pre-chosen, non-degeneracy-gated PRIMARY, not re-picked or swept
      here. The only grid swept in THIS file is the forward-horizon grid,
      point (b) below.

   b. DATA WINDOW: inner-train only, BTC daily log returns, dated
      <= ``INNER_TRAIN_END`` (2020-12-31) -- per docs/ROUTINE.md step 3,
      Step-0 is a measurement gate that also doubles as this branch's own
      design iteration, so it stays inside the inner-train slice rather
      than the full pre-holdout window R-82 through R-96's Step-A gates
      used (those are the CONSERVATIVE branch's own convention; this
      branch's own dispatch explicitly names "on inner-train" for this
      test). Forward-window lookups that would require a day outside
      inner-train's own date range are NaN by construction (excluded, not
      filled) -- inner-validation and the holdout are never read by this
      computation.

   c. BREACH-DAY DEFINITION: every calendar day ``t`` in inner-train where
      ``rolling_gpd_signal(...)['breach'][t] == 1.0`` -- i.e. today's own
      ``|return|`` exceeded the GPD/VaR quantile fit from strictly-earlier
      days (causal by ``rolling_gpd_signal``'s own construction). EVERY
      such day is collected, not a subset.

   d. FORWARD OUTCOME: ``Loss(t, N) = -(sum of daily log returns over the
      STRICTLY FOLLOWING N calendar days t+1..t+N)`` -- positive means
      price fell over the forward window, i.e. this IS a signed loss
      statistic (not |return| or variance), matching this test's own
      question ("does the kill switch avoid *loss*", not "does volatility
      rise"). ``N`` is swept over the A-PRIORI GRID ``{1, 3, 5, 10}``
      calendar days -- no horizon is chosen after seeing results; ALL FOUR
      are reported, and the decision rule (point f) is fixed on how many
      of the four must clear, decided now, not after.

   e. NULL DISTRIBUTION: ``r98_shared.block_bootstrap_shifts`` circularly
      shifts the breach-day 0/1 indicator (``block_days=5, n_draws=500,
      seed=9808``, fixed a priori, never altered after seeing a result)
      over inner-train's own daily index (~1,461 rows, 2017-01-02 through
      2020-12-31); for each of the 500 shifted copies, the mean
      ``Loss(., N)`` at the (shifted) "breach" days is recomputed. This
      gives a null distribution for "mean forward loss conditional on an
      arbitrary day being flagged, preserving the true flag's own temporal
      clustering", against which the true conditional mean is compared --
      the identical methodology R-96's novel branch used for its own
      Step-0 gate (same shared helper, same block/draw/seed *shape*, a
      fresh seed disclosed above).

      DISCLOSED IMPLEMENTATION NOTE, matching R-96's own disclosure: the
      shared helper's ``block_days`` is converted to a raw-row block size
      via ``block = int(block_days * BARS_PER_DAY)`` (``BARS_PER_DAY=288``,
      calibrated for 5-minute-bar cadence). Called here with ``n_bars`` =
      number of inner-train DAYS (~1,461), ``block = 5*288 = 1,440``
      exceeds half of ``n_bars``, so every draw falls into the function's
      own documented fallback: a single uniform random circular shift of
      the whole daily array, not a genuine multi-block reshuffle at 5-day
      granularity. This is still a legitimate circular-shift null (exact
      breach-day clustering preserved, only its calendar phase
      randomized) -- disclosed, not silently assumed to be the finer-block
      version, and not worked around with a second, un-reviewed bootstrap
      routine.

   f. PRE-REGISTERED PASS BAR (frozen now, tied to the comparison's own
      noise per docs/ROUTINE.md step 2's explicit rule -- not a round
      number chosen because it "sounds big"): at horizon ``N``, the TRUE
      mean ``Loss(., N)`` at real breach days must exceed the null
      distribution's own empirical 95th percentile (``null_p95(N)``, from
      the 500 block-shifted draws above) -- i.e. the bar IS the
      comparison's own measured noise band, not a fixed magnitude.
      PASS CONDITION: this must hold at >= 3 of the 4 horizons in the
      grid. Failing that, the branch STOPS at this gate, NEGATIVE, no
      strategy code built.

      POWER SANITY CHECK (done now, per R-78's lesson that a threshold
      must be checked against the ``n`` it implies before being trusted):
      ``VAR_PROB=0.99`` means a well-calibrated live breach flag fires on
      the order of ~1% of days with a valid fit, i.e. roughly 7-15 breach
      days are expected across inner-train's ~1,000-1,300 days with a
      populated 730-day trailing window (the first ~1-2 years of
      inner-train have too little trailing history for
      ``MIN_EXCEEDANCES=15`` to be met, per ``r98_shared``'s own docstring
      on grid-corner degeneracy) -- named now: with single-digit-to-low-
      double-digit ``n``, only a LARGE conditional-mean shift will clear a
      95th-percentile null bar; a small or moderate true effect will not
      reach significance at this sample size, and that is a property of
      the test's own reachable power, disclosed before running, not a
      reason to loosen the bar after seeing a borderline number.

   g. STOP RULE: if (f) does NOT clear at >= 3/4 horizons, STOP here.
      This file's result is the novel branch's ENTIRE product, reported
      NEGATIVE. No kill-switch strategy code is built or backtested, and
      no bar dated >= 2023-01-01 is ever read. If (f) DOES clear, the file
      proceeds to build and evaluate the kill-switch strategy exactly as
      spec'd in section 4 below.

4. IF THE STEP-0 PREMISE HOLDS -- the kill-switch strategy (built only if
   section 3 passes): ``GPDKillSwitchV4``, a ``kelly_regime_v4`` subclass
   whose ``prepare()`` computes v4's own target UNMODIFIED (identical
   internal vote/scale/deadband computation, ``super().prepare(df)``),
   then overrides the OUTPUT to 0 (flat) for a fixed K-calendar-day window
   starting on any day the live breach flag fires (K swept over the
   a-priori grid ``{1, 3, 5, 10}`` days, disclosed before running), then
   lets v4's own already-computed vote/scale resume exactly as computed --
   never re-derived, never gated on a reclaim condition, per section 2
   above. Evaluated via ``scripts/experiment.py``'s ``ev()`` helper against
   unmodified ``kelly_regime_v4`` and ``buy_and_hold``, spot and futures,
   inner-train and inner-validation, plus an ETH falsification check
   restricted to ETH's own available pre-2020 history (Bitfinex ETH ends
   2019-12-31). Promotion bar (docs/ROUTINE.md step 4, default REJECT):
   beats v4/buy_and_hold beyond the +/-0.2 Sharpe noise floor (or a clear
   drawdown/tail improvement at matched exposure), passes ETH
   falsification, and the K-grid neighbourhood is a plateau, not a peak.
   The whipsaw rate (fraction of kill-switch activations followed by a
   same-direction re-entry inside a short horizon that loses money -- the
   named risk from section 2) is measured and reported regardless of the
   other results.

5. WHAT WOULD MAKE THIS FAIL, named now: the same pattern R-85 and R-87
   (see docs/LEDGER.md, "Closed by R-85" / "Confirmed a fourth independent
   way by R-87") already found on directly adjacent premise tests on this
   exact price series -- a statistic computed FROM price (here: whether
   today's |return| exceeded a trailing-fit tail quantile) can only be
   large once a large move has already happened, and by the time it fires,
   the loss the kill switch would be trying to avoid may already be mostly
   realized rather than still ahead. If forward loss after a breach is not
   distinguishable from an arbitrary day's forward loss, that is this
   round's own version of the same finding, and the honest conclusion is
   that the estimator fires too late (or not informatively) on this
   series, not that GPD/POT is wrong as a tail model in general.

CONFIGURATIONS EVALUATED IN THIS FILE: Step-0 = 4 (the horizon grid
N in {1,3,5,10}, at the one fixed PRIMARY GPD cell -- not swept). If Step-0
passes: + kill-switch sweep = 4 more (K in {1,3,5,10} days, each scored on
2 slices x 2 markets = 4 cells; v4/buy_and_hold baselines and the ETH/
whipsaw re-runs on the finalist are re-runs of already-defined comparators,
not additional swept configurations, matching R-90's counting convention).

USAGE
-----
    python experiments/r98_novel_gpd_killswitch.py
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

from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402

from experiments.r98_shared import (  # noqa: E402
    BARS_PER_DAY,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    PRIMARY_FIT_WINDOW_DAYS,
    PRIMARY_THRESH_QUANTILE,
    align_daily_causal,
    assert_no_holdout,
    block_bootstrap_shifts,
    daily_log_returns,
    rolling_gpd_signal,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------- pre-registered
N_GRID = (1, 3, 5, 10)
NULL_BLOCK_DAYS = 5
NULL_DRAWS = 500
NULL_SEED = 9808
STEP0_PASS_HORIZONS_NEEDED = 3  # of 4

COOLDOWN_GRID_DAYS = (1, 3, 5, 10)
WHIPSAW_HORIZON_DAYS = 10.0
SHARPE_NOISE_FLOOR = 0.2


# ---------------------------------------------------------------- data load


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


def load_eth_bars() -> pd.DataFrame:
    path = DATA_DIR / "ethusd_bitfinex_5m.csv.gz"
    df = load_ohlcv_csv(path)
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz="UTC")].copy()
    assert_no_holdout(df)
    return df


# --------------------------------------------------------- Step-0 machinery


def forward_loss(daily_ret: pd.Series, n: int) -> pd.Series:
    """``Loss(t, n)`` = ``-(sum of daily_ret[t+1 .. t+n])``; NaN if the
    n-day forward window is not entirely available inside ``daily_ret``'s
    own index range (no peeking past what this series actually contains)."""
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


def null_mean_loss(flag: np.ndarray, loss_arr: np.ndarray, block_days: int,
                    n_draws: int, seed: int) -> np.ndarray:
    """500 circular-block-shifted copies of the breach-day 0/1 indicator
    (via ``r98_shared.block_bootstrap_shifts``; see the disclosed
    implementation note in this file's module docstring re: block
    granularity at daily cadence); for each, the mean ``loss_arr`` at the
    (shifted) flagged days."""
    n_days = len(flag)
    shifts = block_bootstrap_shifts(n_bars=n_days, block_days=block_days,
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


def build_breach_aligned(df: pd.DataFrame) -> np.ndarray:
    """The bar-aligned breach signal: PRIMARY GPD config, causal daily fit,
    reindexed onto ``df``'s own 5-minute bars via ``align_daily_causal``.
    Returns a 1:1-aligned array, suitable for
    ``r98_shared.truncation_causality_probe`` directly."""
    daily_ret = daily_log_returns(df)
    gpd = rolling_gpd_signal(daily_ret, PRIMARY_THRESH_QUANTILE, PRIMARY_FIT_WINDOW_DAYS)
    aligned = align_daily_causal(gpd["breach"], df)
    return aligned.to_numpy()


def causality_probe(bars: pd.DataFrame) -> bool:
    """Does the bar-aligned breach signal at a fixed check date change if
    bars strictly after it are dropped? Run BEFORE trusting any Step-0
    number, per this round's dispatch."""
    check_date = pd.Timestamp("2020-06-01", tz="UTC")
    check_at = int(bars.index.searchsorted(check_date))
    ok = truncation_causality_probe(build_breach_aligned, bars, check_at=check_at,
                                     shorter_by=20_000)
    print(f"  check_date={check_date.date()}  check_at bar index={check_at}  "
          f"(~{bars.index[check_at].date()})  shorter_by=20,000 bars")
    print(f"  CAUSAL-TRUNCATION PROBE (breach pipeline): {'PASS' if ok else 'FAIL'}")
    return ok


def run_step0(bars: pd.DataFrame) -> dict:
    inner_train = bars.loc[:INNER_TRAIN_END]
    assert_no_holdout(inner_train)
    daily_ret = daily_log_returns(inner_train)
    assert_no_holdout(daily_ret.to_frame())

    gpd = rolling_gpd_signal(daily_ret, PRIMARY_THRESH_QUANTILE, PRIMARY_FIT_WINDOW_DAYS)
    assert_no_holdout(gpd)

    breach = gpd["breach"]
    flag = breach.fillna(0.0).to_numpy()
    n_breach = int(flag.sum())
    print(f"\ninner-train daily index: {len(daily_ret):,} days  "
          f"{daily_ret.index[0].date()} -> {daily_ret.index[-1].date()}")
    print(f"PRIMARY config: quantile={PRIMARY_THRESH_QUANTILE}, "
          f"fit_window_days={PRIMARY_FIT_WINDOW_DAYS}, VAR_PROB=0.99")
    print(f"breach days: {n_breach} / {len(daily_ret):,}")

    results = {}
    for n in N_GRID:
        loss_n = forward_loss(daily_ret, n)
        loss_arr = loss_n.to_numpy()
        spike_pos = np.where(flag == 1.0)[0]
        spike_vals = loss_arr[spike_pos]
        spike_vals = spike_vals[~np.isnan(spike_vals)]
        true_mean = float(np.mean(spike_vals)) if len(spike_vals) else float("nan")
        n_valid = len(spike_vals)

        null = null_mean_loss(flag, loss_arr, NULL_BLOCK_DAYS, NULL_DRAWS, NULL_SEED)
        valid_null = null[~np.isnan(null)]
        null_p95 = float(np.percentile(valid_null, 95)) if len(valid_null) else float("nan")
        null_mean = float(np.mean(valid_null)) if len(valid_null) else float("nan")
        null_std = float(np.std(valid_null)) if len(valid_null) else float("nan")
        z = (true_mean - null_mean) / null_std if null_std else float("nan")

        clears = (not np.isnan(true_mean)) and (not np.isnan(null_p95)) and (true_mean > null_p95)

        print(f"\nN={n}d")
        print(f"  breach days with valid forward data: {n_valid}/{n_breach}")
        print(f"  TRUE mean Loss(.,{n}) at breach days:   {true_mean:+.6f}")
        print(f"  null mean/std/p95 ({len(valid_null)} valid draws): "
              f"{null_mean:+.6f} / {null_std:.6f} / {null_p95:+.6f}")
        print(f"  effect size (z = (true-null_mean)/null_std): {z:+.3f}")
        print(f"  true_mean > null_p95: {clears}")

        results[n] = dict(true_mean=true_mean, null_mean=null_mean, null_std=null_std,
                           null_p95=null_p95, clears=clears, n_valid=n_valid, z=z)

    n_pass = sum(1 for r in results.values() if r["clears"])
    passed = n_pass >= STEP0_PASS_HORIZONS_NEEDED
    return dict(results=results, n_pass=n_pass, passed=passed, n_breach=n_breach)


# ------------------------------------------------------------ kill switch


def _build_cooldown_daily(breach: pd.Series, cooldown_days: int) -> pd.Series:
    """Daily 0/1 cooldown flag: 1 on the breach day itself and the
    following ``cooldown_days - 1`` days (``cooldown_days`` days total),
    re-extended (not reset) if a further breach occurs while already in
    cooldown. Purely a function of calendar days elapsed since the most
    recent breach -- no reclaim-of-threshold condition anywhere, per
    section 2 of this file's pre-registration."""
    vals = breach.fillna(0.0).to_numpy()
    n = len(vals)
    out = np.zeros(n)
    flat_until = -1
    for i in range(n):
        if vals[i] == 1.0:
            flat_until = max(flat_until, i + cooldown_days)
        out[i] = 1.0 if i < flat_until else 0.0
    return pd.Series(out, index=breach.index)


class GPDKillSwitchV4(KellyRegimeV4):
    """kelly_regime_v4 with a fixed K-day flat cooldown after a live POT/GPD
    VaR breach (R-98 novel branch, experiment only -- not registered)."""

    def __init__(self, cooldown_days: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cooldown_days = int(cooldown_days)
        # Generous warmup so the 730-day trailing GPD fit window has real
        # history available before the strategy is ever asked to trade,
        # on top of v4's own 80-day anchor warmup.
        self.warmup = max(KellyRegimeV4.warmup,
                           (PRIMARY_FIT_WINDOW_DAYS + 100) * BARS_PER_DAY)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)  # v4's own target, computed unmodified
        daily_ret = daily_log_returns(df)
        gpd = rolling_gpd_signal(daily_ret, PRIMARY_THRESH_QUANTILE, PRIMARY_FIT_WINDOW_DAYS)
        cooldown_daily = _build_cooldown_daily(gpd["breach"], self.cooldown_days)
        cooldown_bars = align_daily_causal(cooldown_daily, df)
        mask = (cooldown_bars.fillna(0.0).to_numpy() >= 0.5)
        target = df["target"].to_numpy(dtype=float)
        df["target"] = np.where(mask, 0.0, target)
        df["_gpd_cooldown"] = mask.astype(float)
        return df


def killswitch_whipsaw_rate(close: np.ndarray, final_target: np.ndarray,
                            activation_events: np.ndarray,
                            horizon_days: float = WHIPSAW_HORIZON_DAYS) -> dict:
    """Rate and cost of whipsaws following a kill-switch activation -- the
    round's named risk (section 2). For each bar where the cooldown just
    started, look forward up to ``horizon_days`` in the FINAL target path
    for the first bar where the position re-enters (target crosses from
    <=0 to >0). A whipsaw is a re-entry at a close HIGHER than the
    activation bar's own close. Same shape as r90_shared's
    ``stopout_whipsaw_rate`` (same horizon convention), applied to this
    branch's own activation events."""
    horizon = int(horizon_days * BARS_PER_DAY)
    n = len(close)
    events = np.flatnonzero(activation_events)
    total = 0
    whip = 0
    costs = []
    for i in events:
        exit_price = close[i]
        window_end = min(n, i + 1 + horizon)
        reentry = None
        for j in range(i + 1, window_end):
            if final_target[j] > 0 and final_target[j - 1] <= 0:
                reentry = j
                break
        if reentry is None:
            continue
        total += 1
        if close[reentry] > exit_price:
            whip += 1
            costs.append(float(np.log(close[reentry] / exit_price)))
    rate = (whip / total) if total else float("nan")
    return dict(activations=int(len(events)), events_with_reentry_in_horizon=total,
                whipsaws=whip, whipsaw_rate=rate,
                mean_whipsaw_log_cost=float(np.mean(costs)) if costs else 0.0)


def whipsaw_diagnostic(bars_train_val: pd.DataFrame, cooldown_days: int) -> dict:
    strat = GPDKillSwitchV4(cooldown_days=cooldown_days)
    frame = strat.prepare(bars_train_val.copy())
    mask = frame["_gpd_cooldown"].to_numpy()
    target = frame["target"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    activation = np.zeros(len(mask), dtype=bool)
    activation[1:] = (mask[1:] == 1.0) & (mask[:-1] == 0.0)
    activation[0] = mask[0] == 1.0
    return killswitch_whipsaw_rate(close, target, activation)


def mean_abs_exposure(strategy, df: pd.DataFrame, start: str | None, end: str | None) -> float:
    frame = strategy.prepare(df.copy())
    sl = frame.loc[start:end] if (start or end) else frame
    return float(np.mean(np.abs(sl["target"].to_numpy(dtype=float))))


# --------------------------------------------------------------------- main


def hdr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def main() -> dict:
    t0 = time.time()
    max_ts = []

    hdr("R-98 NOVEL: POT/GPD tail-VaR-breach kill switch -- STEP 0 sub-claim gate")
    print("'does forward realized loss after a live breach exceed the unconditional")
    print(" baseline enough to justify standing flat afterward?'")
    print(f"\nnamed whipsaw risk (B-41/R-90): re-entry gated on a fixed K-day cooldown,")
    print(f"NOT a reclaim-of-threshold condition -- see pre-registration section 2.")

    bars = load_btc_bars()
    max_ts.append(bars.index.max())

    hdr("CAUSAL-TRUNCATION PROBE (run before trusting any Step-0 number)")
    probe_ok = causality_probe(bars)

    hdr("STEP 0 -- forward-loss test, N in {1,3,5,10} days, PRIMARY GPD config")
    step0 = run_step0(bars)

    print("\n" + "-" * 96)
    print(f"STEP-0 SUMMARY: {step0['n_pass']}/4 horizons clear (true_mean > null_p95); "
          f"need >= {STEP0_PASS_HORIZONS_NEEDED}/4 to pass. breach days (inner-train): "
          f"{step0['n_breach']}")
    print(f"STEP-0 GATE VERDICT: {'PASS' if step0['passed'] else 'FAIL (NEGATIVE)'}")
    print("-" * 96)

    print(f"\nconfigurations evaluated so far (decision-bearing): 4 "
          f"(N in {N_GRID}, PRIMARY GPD config only)")
    print(f"max timestamp read anywhere so far: {max(max_ts)}  (< {OOS_START})")

    if not step0["passed"]:
        print("\n" + "#" * 96)
        print("# STEP-0 GATE FAILED ITS PRE-REGISTERED PASS BAR.")
        print("# Per this file's own pre-registration: STOP HERE. This gate result is")
        print("# this branch's ENTIRE product, written up NEGATIVE. No kill-switch")
        print("# strategy code is built. No data on/after 2023-01-01 is touched.")
        print("#" * 96)
        print(f"\n[{time.time()-t0:.0f}s]")
        return dict(bars=bars, step0=step0, probe_ok=probe_ok, passed=False,
                    max_ts=max(max_ts), n_configs=4)

    # ================================================================
    # STEP 0 PASSED -- build and evaluate the kill-switch strategy.
    # ================================================================
    print("\n" + "#" * 96)
    print("# STEP-0 GATE PASSED. Proceeding to build and evaluate the kill-switch")
    print("# strategy, per section 4 of this file's own pre-registration.")
    print("#" * 96)

    from scripts.experiment import SPOT, FUTURES, ev  # noqa: E402  (local import: only

    v4 = get_strategy("kelly_regime_v4")
    bh = get_strategy("buy_and_hold")

    SLICES = {
        "inner_train": dict(end=INNER_TRAIN_END),
        "inner_val": dict(start=INNER_VAL_START, end=INNER_VAL_END),
    }
    MARKETS = {"spot": SPOT, "futures": FUTURES}

    hdr("BASELINES -- kelly_regime_v4, buy_and_hold (inner-train / inner-val x spot / futures)")
    baseline = {}
    for label, strat in (("v4", v4), ("buy_and_hold", bh)):
        for slice_name, kw in SLICES.items():
            for mkt_name, mkt in MARKETS.items():
                m = ev(strat, market=mkt, tag=f"{label:12s} {slice_name} {mkt_name}", **kw)
                baseline[(label, slice_name, mkt_name)] = m

    hdr(f"KILL-SWITCH SWEEP -- cooldown_days in {COOLDOWN_GRID_DAYS}")
    ks = {}
    for k in COOLDOWN_GRID_DAYS:
        strat = GPDKillSwitchV4(cooldown_days=k)
        for slice_name, kw in SLICES.items():
            for mkt_name, mkt in MARKETS.items():
                m = ev(strat, market=mkt, tag=f"K={k:<3d}d       {slice_name} {mkt_name}", **kw)
                ks[(k, slice_name, mkt_name)] = m

    n_configs_ks = len(COOLDOWN_GRID_DAYS)
    total_configs = 4 + n_configs_ks  # Step-0 horizons + kill-switch sweep

    hdr("SELECTION -- delta-Sharpe vs v4, inner-validation, futures (primary) and spot")
    print(f"{'K(days)':>8s} {'dSharpe_fut':>12s} {'dSharpe_spot':>13s} "
          f"{'dSharpe_avg':>12s}")
    sel = {}
    for k in COOLDOWN_GRID_DAYS:
        d_fut = ks[(k, "inner_val", "futures")].sharpe - baseline[("v4", "inner_val", "futures")].sharpe
        d_spot = ks[(k, "inner_val", "spot")].sharpe - baseline[("v4", "inner_val", "spot")].sharpe
        sel[k] = (d_fut + d_spot) / 2.0
        print(f"{k:8d} {d_fut:+12.3f} {d_spot:+13.3f} {sel[k]:+12.3f}")

    finalist_k = max(sel, key=lambda k: sel[k])
    print(f"\nFINALIST: K={finalist_k} days  (avg dSharpe inner-val = {sel[finalist_k]:+.3f})")

    hdr(f"PROMOTION BAR -- finalist K={finalist_k}")
    d_sharpe_fut = ks[(finalist_k, "inner_val", "futures")].sharpe - baseline[("v4", "inner_val", "futures")].sharpe
    d_sharpe_spot = ks[(finalist_k, "inner_val", "spot")].sharpe - baseline[("v4", "inner_val", "spot")].sharpe
    d_dd_fut = ks[(finalist_k, "inner_val", "futures")].max_drawdown_pct - baseline[("v4", "inner_val", "futures")].max_drawdown_pct
    d_dd_spot = ks[(finalist_k, "inner_val", "spot")].max_drawdown_pct - baseline[("v4", "inner_val", "spot")].max_drawdown_pct

    exp_ks_fut = mean_abs_exposure(GPDKillSwitchV4(cooldown_days=finalist_k), bars, INNER_VAL_START, INNER_VAL_END)
    exp_v4 = mean_abs_exposure(v4, bars, INNER_VAL_START, INNER_VAL_END)
    exposure_ratio = exp_ks_fut / exp_v4 if exp_v4 else float("nan")
    risk_matched = 0.9 <= exposure_ratio <= 1.1

    print(f"dSharpe futures={d_sharpe_fut:+.3f}  spot={d_sharpe_spot:+.3f}  "
          f"(noise floor +/-{SHARPE_NOISE_FLOOR})")
    print(f"dMaxDD  futures={d_dd_fut:+.2f}pp  spot={d_dd_spot:+.2f}pp")
    print(f"mean|target| finalist={exp_ks_fut:.3f}  v4={exp_v4:.3f}  "
          f"exposure_ratio={exposure_ratio:.3f}  risk_matched={risk_matched}")

    sharpe_pass = (d_sharpe_fut > SHARPE_NOISE_FLOOR) and (d_sharpe_spot > SHARPE_NOISE_FLOOR)
    dd_pass = (d_dd_fut < 0.0) and (d_dd_spot < 0.0) and risk_matched
    b_perf = sharpe_pass or dd_pass
    print(f"B-perf (Sharpe or matched-drawdown): {'PASS' if b_perf else 'FAIL'} "
          f"(via {'Sharpe' if sharpe_pass else ('matched drawdown' if dd_pass else 'neither')})")

    hdr("PLATEAU CHECK -- finalist's immediate K-grid neighbours")
    idx_f = COOLDOWN_GRID_DAYS.index(finalist_k)
    neighbours = [COOLDOWN_GRID_DAYS[j] for j in (idx_f - 1, idx_f + 1) if 0 <= j < len(COOLDOWN_GRID_DAYS)]
    for k in [finalist_k] + neighbours:
        tag = "<-- FINALIST" if k == finalist_k else ""
        print(f"  K={k:<3d}d  avg dSharpe={sel[k]:+.3f}  {tag}")
    plateau = bool(neighbours) and all(np.sign(sel[k]) == np.sign(sel[finalist_k]) for k in neighbours)
    print(f"plateau (neighbours same sign as finalist): {'PASS' if plateau else 'FAIL'}")

    hdr("WHIPSAW DIAGNOSTIC -- all 4 K values (named risk, section 2)")
    bars_tv = bars.loc[:INNER_VAL_END]
    whip = {}
    for k in COOLDOWN_GRID_DAYS:
        w = whipsaw_diagnostic(bars_tv, k)
        whip[k] = w
        print(f"  K={k:<3d}d  activations={w['activations']:4d}  "
              f"reentries_in_horizon={w['events_with_reentry_in_horizon']:4d}  "
              f"whipsaws={w['whipsaws']:4d}  rate={w['whipsaw_rate']:.3f}  "
              f"mean_cost={w['mean_whipsaw_log_cost']:+.5f}")

    hdr("ETH FALSIFICATION -- finalist K, pre-2020 only (Bitfinex ETH ends 2019-12-31)")
    eth = load_eth_bars()
    max_ts.append(eth.index.max())
    print(f"ETH (Bitfinex): {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}")
    eth_finalist = GPDKillSwitchV4(cooldown_days=finalist_k)
    eth_results = {}
    for mkt_name, mkt in MARKETS.items():
        m_ks = ev(eth_finalist, df=eth, market=mkt, tag=f"ETH K={finalist_k}d {mkt_name}")
        m_v4 = ev(v4, df=eth, market=mkt, tag=f"ETH v4      {mkt_name}")
        eth_results[mkt_name] = (m_ks, m_v4)
    eth_ok = all(m_ks.sharpe >= m_v4.sharpe for m_ks, m_v4 in eth_results.values())
    print(f"\nETH falsification (finalist Sharpe >= v4 Sharpe, both markets): "
          f"{'PASS' if eth_ok else 'FAIL'}")
    for mkt_name, (m_ks, m_v4) in eth_results.items():
        print(f"  {mkt_name}: kill-switch sharpe={m_ks.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}")

    hdr("VERDICT")
    clauses = {"B-perf (Sharpe/drawdown)": b_perf, "plateau": plateau, "ETH falsification": eth_ok}
    for k, v in clauses.items():
        print(f"  {k:28s} {'PASS' if v else 'FAIL'}")
    promote = all(clauses.values())
    print(f"\nVERDICT: {'CANDIDATE FOR HOLDOUT (PROMOTE-CANDIDATE)' if promote else 'NEGATIVE'}")
    if not promote:
        print(f"Failing clause(s): {', '.join(k for k, v in clauses.items() if not v)}")

    print(f"\nConfigurations evaluated in this file: {total_configs} "
          f"(4 Step-0 horizons + {n_configs_ks} kill-switch cooldown sweep)")
    print(f"max timestamp read anywhere in this branch (BTC and ETH): "
          f"{max(max_ts)}  (< {OOS_START})")
    print(f"\n[{time.time()-t0:.0f}s]")

    return dict(bars=bars, step0=step0, probe_ok=probe_ok, passed=True,
                finalist_k=finalist_k, promote=promote, clauses=clauses,
                whip=whip, sel=sel, max_ts=max(max_ts), n_configs=total_configs)


if __name__ == "__main__":
    main()
