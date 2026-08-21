#!/usr/bin/env python
"""R-83 NOVEL operator measurement: the Step-A detection-lag gate for a
causal Kalman local-linear-trend (LLT) filter, run BEFORE any strategy
code is built -- same "operator-measurement-before-branch" convention
R-78/R-80/R-81/R-82 all used, and the same reason: this is a fixed,
non-swept measurement, not a strategy, and doing it once avoids paying
for the identical question twice.

=============================================================================
PRE-REGISTRATION (frozen before this file was ever run against real data)
=============================================================================

1. MECHANISM, one sentence: a causal Kalman filter's FILTERED slope state
   for a Harvey (1989) local-linear-trend model of daily log(close) is a
   continuously-updated estimate of the CURRENT trend's sign and rate built
   from the same price series `kelly_regime_v4`'s anchor ladder already
   reads, and because it updates by a bar-by-bar Kalman GAIN proportional
   to that bar's own surprise (rather than v4's fixed 20/40/80-day rolling
   window, which cannot move until enough NEW bars have entered the window
   to outweigh the OLD ones still inside it), its sign should cross down
   through zero at least as promptly as v4's own anchor-crossing heuristic
   reacts to the same six dated historical regime transitions.

2. HYPERPARAMETER CALIBRATION (done BEFORE any real BTC number was
   computed, on SYNTHETIC data only, never retuned against this gate's own
   real-market result -- the identical discipline R-82 used for BOCPD's
   `hazard_lambda`/`K_SHORT_DAYS`):

   The LLT filter has three scale parameters -- observation noise
   `SIGMA_EPS`, level process noise `SIGMA_ETA`, slope process noise
   `SIGMA_ZETA` (all defined and justified in
   `r83_novel_kalman_shared.py`). `SIGMA_ZETA` is the primary
   speed/smoothness knob named in this round's brief. A synthetic
   calibration harness (500 Monte Carlo draws per candidate, a step change
   from a +0.15%/day to a -0.25%/day daily drift with 3.5%/day noise
   added, both magnitudes chosen to be broadly representative of a BTC
   regime transition, at day 100 of a 250-day series) measured, for a
   grid of `(SIGMA_ETA, SIGMA_ZETA)` combinations at fixed
   `SIGMA_EPS=0.030`: (a) the median number of days after the step until
   the filtered slope first crosses down through zero, and (b) the mean
   number of spurious sign flips over a 220-day STABLE-drift synthetic
   control (no step at all) -- the direct synthetic analogue of the
   R-01 HMM "rapid switching" failure mode named in this round's brief.
   `SIGMA_ETA=0.0010, SIGMA_ZETA=0.00007` was selected as the combination
   whose median synthetic detection delay (~14 days) is comparable to
   v4's own FASTEST anchor (20 days -- the anchor that dominates
   `anchor_majority`'s first downward move, since the majority is an
   average of three latched votes and moves as soon as the fastest one
   flips) while keeping the stable-regime false-flip rate low (~3.2
   spurious sign changes per 220 trading days, i.e. roughly one every ten
   weeks) rather than in the double digits it reaches once `SIGMA_ZETA`
   is pushed materially higher. This calibration is reported in full in
   `calibrate()` below and is run again, reproducibly, every time this
   file executes -- it is not a one-off scratch computation whose numbers
   are merely asserted here. **This grid search never reads a single BTC
   or ETH bar; it is entirely synthetic**, exactly R-82's own convention
   for its BOCPD prior hyperparameters, and is not counted in this file's
   "configurations evaluated against real market data" tally.

3. DETECTION-LAG DEFINITION. For each of the six episodes in
   `r83_novel_kalman_shared.STRESS_EPISODES` (byte-for-byte identical to
   R-82's own six, for direct comparability), within a +/-60-day search
   window around its onset (`episode_window`):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority` to the onset (`nearest_transition`,
     `direction="down"` -- the same primary rule R-81/R-82 used).
   - Kalman's reaction: the nearest bar where the filtered slope crosses
     DOWN through zero, to the onset (`nearest_kalman_detection`).
   - LEAD = (v4_flip_time - kalman_detect_time) in days. Positive means
     the Kalman filter detected the break before v4's own gate reacted.

4. THE NULL. `block_bootstrap_shifts` circularly block-shifts the LOCAL
   (episode-window) filtered-slope series (block_days=5, N=500 draws,
   seed=83, fixed before running) and recomputes "nearest Kalman
   detection to the real, unshifted v4 flip time" against each shifted
   copy -- byte-for-byte the same construction R-82 used for BOCPD.

5. PRE-REGISTERED STOP RULE (fixed now, before any number below was
   computed, identical bar to R-82's): an episode counts as a PASS if BOTH
   (a) LEAD >= 0 (Kalman detected at or before v4's own nearest reaction),
   AND (b) the true LEAD is >= the null distribution's median. PROCEED TO
   STEP B (build a real strategy variant) only if >= 4 of the 6 episodes
   PASS. If fewer than 4 pass: STOP, report this file's result as the
   whole round's product, write it up as NEGATIVE, do not write any
   strategy code past this file. The bar is not relaxed after seeing the
   numbers.

6. WHAT WOULD MAKE THIS FAIL, named now, as two NAMED and DISTINCT failure
   modes (per this round's brief):

   (i) CHANGEPOINT-STYLE LAG: Kalman's detections cluster AFTER v4's own
       reaction on the sudden, violent episodes (COVID crash, 2021 top,
       Terra/Luna, FTX) even though it may lead on the two slow 2018
       episodes -- the exact pattern R-82 found for BOCPD.

   (ii) HMM-STYLE OVER-SWITCHING: once `SIGMA_ZETA` is tuned high enough
        to react fast enough to beat v4 on the sudden episodes, the slope
        starts flipping sign too rapidly on ordinary noise, reproducing
        R-01's HMM "rapid switching... fatal at a 0.1% round trip" failure
        mode in continuous-state form.

   PRE-REGISTERED EXPECTATION, stated before this file was ever run
   against real data: **(i) is expected, not (ii)**. Reasoning: the
   calibration in section 2 above fixes `SIGMA_ZETA` to a value that
   reacts SMOOTHLY (median synthetic delay ~14 days to a SUSTAINED step
   change in mean drift) rather than SHARPLY (in one or two bars) to any
   single day's return. A local-linear-trend filter's slope update at bar
   t is `nu_t = nu_pred + K_nu * (y_t - Z @ x_pred)` -- a SINGLE violent
   one-day move produces one large surprise term `v_t`, but `K_nu` (the
   Kalman gain on the slope) was calibrated to be small precisely so the
   filter does not chase one-bar noise (that is what keeps the stable-
   regime false-flip rate low in section 2's own numbers) -- so one huge
   down-bar should nudge the slope estimate but should generally need
   several corroborating bars before the slope's SIGN actually flips.
   That is structurally the same "needs corroborating evidence before
   committing" property that made BOCPD slow on the four sudden 2020-2022
   shocks in R-82, even though the underlying machinery (a continuous
   Kalman gain vs. a discrete run-length posterior) is completely
   different. Mode (ii) is judged the less likely primary failure here
   specifically BECAUSE `SIGMA_ZETA` was deliberately calibrated on the
   smooth/low-false-flip side of the tradeoff curve in section 2, at the
   cost of exactly the responsiveness mode (i) would need -- the two
   failure modes are not independent risks, they are the two ends of the
   one dial this round was told to reason about explicitly, and this round
   pre-commits to having picked the smooth end.

CONFIGURATIONS EVALUATED AGAINST REAL MARKET DATA IN THIS FILE: 0 (a
fixed, non-swept measurement gate, using `r83_novel_kalman_shared`'s three
named hyperparameters throughout -- no sweep against real BTC data occurs
here; that sweep, if the gate passes, belongs to Step B and is
pre-registered separately).

Run: ``python experiments/r83_novel_kalman_gate.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_dataset  # noqa: E402

from experiments.r83_novel_kalman_shared import (  # noqa: E402
    OOS_START,
    SIGMA_EPS,
    STRESS_EPISODES,
    anchor_majority,
    block_bootstrap_shifts,
    kalman_daily_causal_signals,
    kalman_llt_filter,
    episode_window,
    nearest_kalman_detection,
    nearest_transition,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"
WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 83


def assert_no_holdout(df: pd.DataFrame) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


def load_btc_bars() -> pd.DataFrame:
    df, label = load_dataset(DATA_DIR, "spot")
    df = df.loc[df.index < pd.Timestamp(OOS_START, tz=df.index.tz)].copy()
    assert_no_holdout(df)
    print(f"BTC ({label}): {len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)
    return df


# =============================================================================
# Section 2's synthetic calibration -- reproduced here in full, reads NO
# market data. Reproduces the exact grid and numbers cited in the module
# docstring above.
# =============================================================================

def _synthetic_step(n_days: int, cp: int, drift_before: float, drift_after: float,
                     daily_vol: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drift = np.where(np.arange(n_days) < cp, drift_before, drift_after)
    logret = drift + rng.normal(0.0, daily_vol, n_days)
    return np.cumsum(logret) + np.log(20_000.0)


def _median_detect_delay(sigma_eta: float, sigma_zeta: float, n_trials: int = 500,
                          sigma_eps: float = SIGMA_EPS) -> tuple[float, int]:
    delays = []
    misses = 0
    for seed in range(n_trials):
        y = _synthetic_step(250, 100, 0.0015, -0.0025, 0.035, seed)
        _, slope = kalman_llt_filter(y, sigma_eps, sigma_eta, sigma_zeta)
        post = slope[100:]
        idx = np.flatnonzero(post < 0.0)
        if len(idx) == 0:
            misses += 1
            continue
        delays.append(idx[0])
    med = float(np.median(delays)) if delays else float("nan")
    return med, misses


def _false_flip_rate(sigma_eta: float, sigma_zeta: float, n_trials: int = 500,
                      sigma_eps: float = SIGMA_EPS) -> float:
    rates = []
    for seed in range(10_000, 10_000 + n_trials):
        rng = np.random.default_rng(seed)
        n_days = 250
        logret = 0.0015 + rng.normal(0.0, 0.035, n_days)
        y = np.cumsum(logret) + np.log(20_000.0)
        _, slope = kalman_llt_filter(y, sigma_eps, sigma_eta, sigma_zeta)
        s = np.sign(slope[30:])
        rates.append(int(np.sum(s[1:] != s[:-1])))
    return float(np.mean(rates))


def calibrate() -> None:
    print("-" * 78)
    print("Section-2 synthetic calibration grid (NO market data read; 500 draws/cell)")
    print("-" * 78)
    grid = [
        (0.0010, 0.00006), (0.0010, 0.00007), (0.0010, 0.00008),
        (0.0008, 0.00006), (0.0012, 0.00006), (0.0006, 0.00004),
        (0.0006, 0.00005), (0.0006, 0.00006),
    ]
    print(f"{'sigma_eta':>10s} {'sigma_zeta':>11s} {'median_delay(d)':>17s} "
          f"{'misses/500':>11s} {'flips/220d(stable)':>19s}")
    for sig_eta, sig_zeta in grid:
        med, miss = _median_detect_delay(sig_eta, sig_zeta)
        flips = _false_flip_rate(sig_eta, sig_zeta)
        chosen = " <-- SELECTED" if (sig_eta, sig_zeta) == (0.0010, 0.00007) else ""
        print(f"{sig_eta:>10.5f} {sig_zeta:>11.6f} {med:>17.1f} {miss:>11d} "
              f"{flips:>19.2f}{chosen}")
    print(f"\n(this grid never reads BTC/ETH data; not counted toward the "
          f"real-market configuration tally below.)\n")


def null_leads(slope: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
               flip_time: pd.Timestamp, n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
               seed: int = NULL_SEED) -> np.ndarray:
    local = slope.reindex(window).to_numpy()
    n_bars = len(local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = local[shift]
        neg = shifted < 0.0
        cross = np.zeros(n_bars, dtype=bool)
        cross[1:] = neg[1:] & ~neg[:-1]
        cross[0] = bool(neg[0])
        idx = np.where(cross)[0]
        if len(idx) == 0:
            continue
        times = window[idx]
        deltas = np.abs((times - onset).to_numpy())
        detect_time = times[int(np.argmin(deltas))]
        leads[k] = (flip_time - detect_time).total_seconds() / 86400.0
    return leads


def causality_probe(bars: pd.DataFrame) -> None:
    print("-" * 78)
    print("Causality probe: does kalman_slope[check_at] change if bars after it are dropped?")
    print("-" * 78)

    def build_slope(df: pd.DataFrame) -> np.ndarray:
        return kalman_daily_causal_signals(df)["kalman_slope"].to_numpy()

    for check_at in (250_000, 350_000):
        ok = truncation_causality_probe(build_slope, bars, check_at)
        print(f"  check_at={check_at:>7d}: {'PASS (causal)' if ok else 'FAIL (LOOKAHEAD)'}")
        assert ok, "kalman_slope is not causal -- stop, do not trust this gate"


def gate() -> dict:
    print("=" * 78)
    print("R-83 NOVEL OPERATOR MEASUREMENT: Kalman LLT filter vs v4 anchor -- "
          "STEP A detection-lag gate")
    print("=" * 78)

    calibrate()

    bars = load_btc_bars()
    causality_probe(bars)

    majority = anchor_majority(bars)
    kalman = kalman_daily_causal_signals(bars)
    assert_no_holdout(kalman)
    slope = kalman["kalman_slope"]

    print(f"\nsearch window=+/-{WINDOW_DAYS}d  null: {N_DRAWS} draws, "
          f"block={BLOCK_DAYS}d, seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue

        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_kalman_detection(slope, window, onset)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no anchor-gate transition' if flip_time is None else 'no Kalman detection'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        leads_null = null_leads(slope, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 anchor nearest downward flip: {flip_time}")
        print(f"    Kalman nearest detection (slope crosses down through 0): {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time,
                             detect=detect_time, lead=lead, null_median=null_median,
                             pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 4

    print("\n" + "=" * 78)
    for r in results:
        print(f"  {r['label']:42s} lead={r.get('lead', float('nan')):+.2f}d  PASS={r['pass_b']}")
    print(f"\nEpisodes passing: {n_pass}/6")
    print(f"GATE VERDICT: {'PASS -> proceed to Step B (build a strategy variant)' if passed else 'FAIL -> STOP, no strategy built'}")
    print(f"\nconfigurations evaluated against real market data in this file: 0 "
          f"(fixed measurement gate)")
    print(f"max timestamp read anywhere in this session: "
          f"{max(bars.index.max(), kalman.index.max())}  (< {OOS_START})")

    return dict(results=results, n_pass=n_pass, passed=passed)


if __name__ == "__main__":
    gate()
