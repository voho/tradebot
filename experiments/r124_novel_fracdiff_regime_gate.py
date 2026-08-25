#!/usr/bin/env python
"""R-124 NOVEL branch: Step-A detection-lag gate for a "fractional-diff
momentum" regime-timing detector -- the eleventh structurally distinct
mechanism this project has thrown at the identical gate R-01/R-82/R-83/
R-85/R-86/R-96/R-98/R-84/R-60/R-117 all failed. See `r124_shared.py`'s
module docstring for the full citation trail (Lopez de Prado 2018 Ch.5;
Hosking 1981) and the not-a-duplicate-of list against those ten priors.

PRE-REGISTRATION (frozen before this file was ever run):

1. MECHANISM, one sentence: take `r124_shared.causal_ffd_log_close(df)`
   (the frozen Fixed-Window Fractional Differentiation of log(close) at
   FFD_D=0.85, a genuinely fast/short-memory ~4-hour-scale stationary
   series), build a MACD-style crossover ON TOP of it -- rolling mean over
   v4's own fastest anchor horizon (20 days) minus rolling mean over v4's
   own slowest anchor horizon (80 days) -- z-score that spread by its own
   80-day rolling standard deviation, and declare "detection" the nearest
   bar (to each episode onset, within a +/-60-day window) where that
   z-score crosses DOWN through a fixed, pre-registered threshold. This is
   a fundamentally different construction from v4's own MACD-style
   crossover (which operates on raw `close`) and from all ten prior
   regime-timing mechanisms (all computed on the raw price series' level
   or simple returns, none on a fractionally-differenced input). No new
   data channel: only the already-committed BTC OHLCV `close` series v4
   itself already reads.

2. THRESHOLD, fixed a priori, disclosed here BEFORE any episode-level
   number was computed: THRESH = -1.0 (one standard deviation below the
   spread's own rolling mean of ~0). This is the "reasonable starting
   choice" the task itself names, not tuned to any episode -- a round,
   conventional one-sigma downside-deviation threshold, the same kind of
   fixed, un-fit constant this project's other gates use (e.g. R-82's own
   K_SHORT_DAYS=5, ADF_CRIT_5PCT=-2.86; R-96's Hawkes alarm threshold).
   For DISCLOSED CONTEXT ONLY (not used as the decision threshold, not
   used to pick among candidates, computed once and reported alongside
   the primary result exactly like R-117's non-decision robustness grid):
   the 15th percentile of this same z-score series computed on
   INNER_TRAIN ONLY (2017-01-01..2020-12-31) is printed below so a reader
   can see whether -1.0 sits inside a similar range. The PRIMARY, DECIDING
   threshold is -1.0, fixed before this file read a single episode-level
   bar, never adjusted after seeing results.

3. DETECTION-LAG DEFINITION, identical machinery to
   `r82_shared.nearest_bocpd_detection`/`r117_conservative_donchian_gate
   .py`'s `gate()` -- for each of the six `STRESS_EPISODES`, within a
   +/-60-day search window around its onset (`episode_window`,
   WINDOW_DAYS=60):
   - v4's own reaction: the nearest DOWNWARD transition of
     `anchor_majority(df)` to the onset (`nearest_transition`,
     direction="down" -- the identical rule R-82 through R-117 used).
   - This detector's reaction: the nearest bar, within the window, where
     the fracdiff-momentum z-score crosses DOWN through THRESH=-1.0, to
     the onset (`nearest_fracdiff_detection` below -- written analogously
     to `nearest_bocpd_detection`'s "nearest crossing to onset" pattern).
   - LEAD = (v4_flip_time - fracdiff_detect_time) in days. Positive means
     this detector reacted BEFORE v4's own gate.

4. NULL. `block_bootstrap_shifts` (block_days=5, n_draws=500, seed=12401
   -- fixed now, disclosed, never altered after seeing results) circularly
   shifts the LOCAL (episode-window) z-score series and recomputes
   "nearest detection-to-onset" against each shifted copy, exactly
   R-117's `null_leads` pattern adapted to a threshold-crossing detector
   instead of a discrete vote-transition series.

5. PRE-REGISTERED PASS RULE per episode (fixed now, before any number
   below was computed, identical bar to R-82 through R-117): LEAD >= 0
   AND LEAD >= the null distribution's median.

6. PRE-REGISTERED STOP RULE (fixed now): proceed to Step B only if >= 4
   of 6 episodes PASS. If fewer than 4 pass -- the expected and fully
   successful outcome of a well-run negative test, given 10/10 prior
   mechanisms have failed this exact gate -- STOP immediately, do not
   write any further code, do not read any bar at or after OOS_START
   (2023-01-01), and report this as the whole round's product, NEGATIVE.
   The bar is not relaxed, narrowed, or otherwise adjusted after seeing
   the numbers.

7. WHAT WOULD MAKE THIS GATE FAIL, named now (copied from `r124_shared
   .py`'s own "WHAT WOULD MAKE EACH BRANCH FAIL" section, named there as
   this branch's pre-registered EXPECTATION, not a hoped-for result): a
   threshold crossing on a stationarized, ~4-hour-scale series reacts to
   LOCAL deviations -- a fundamentally noisier, more frequent event than
   v4's own slow 1%-past-an-80-day-mean crossings. The named, expected
   failure mode is FALSE-POSITIVE OVER-TRIGGERING (many small z-score
   crossings unrelated to the six historical stress episodes, inflating
   the null distribution's own median lead until the true lead can no
   longer clear it) -- a qualitatively DIFFERENT failure signature from
   the pure LAG failure mode all ten predecessors showed (in which the
   detector eventually reacted, just always after v4). Both failure
   shapes are treated identically as NEGATIVE by the pass rule above, but
   this file distinguishes and reports which one (if either) actually
   occurred.

CAUSAL-TRUNCATION PROBE: run first, unconditionally, before any
episode-level number is trusted (this project has had lookahead bugs
before -- one produced a $3.7e23 balance). Uses
`causal_truncation_probe_series` on the full composed
`fracdiff_zscore_series` builder (FFD -> rolling means -> spread ->
z-score), not just the frozen FFD primitive alone (which r124_shared.py
already self-tests independently).

CONFIGURATIONS EVALUATED: 0 swept. One fixed primary configuration
(SHORT_DAYS=20, LONG_DAYS=80, THRESH=-1.0), matching R-82/R-117's own
"single fixed non-swept measurement gate" convention -- 6 episode
measurements against that one configuration, no grid search, no
threshold or horizon tuned to any episode-level outcome. The inner-train
15th-percentile number in point 2 above is diagnostic context only, never
used to pick or override the primary threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r124_shared import (  # noqa: E402
    BARS_PER_DAY,
    INNER_TRAIN_END,
    OOS_START,
    STRESS_EPISODES,
    anchor_majority,
    assert_no_holdout,
    block_bootstrap_shifts,
    causal_ffd_log_close,
    causal_truncation_probe_series,
    episode_window,
    hr,
    load_btc,
    nearest_transition,
)

WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 12401

SHORT_DAYS = 20   # v4's own fastest anchor
LONG_DAYS = 80    # v4's own slowest anchor
THRESH = -1.0     # fixed a priori -- see pre-registration point 2 above


def fracdiff_zscore_series(df: pd.DataFrame) -> pd.Series:
    """The novel detection series: FFD(log close) -> 20d rolling mean minus
    80d rolling mean ("fractional-diff momentum") -> z-scored by its own
    80d rolling std. Causal by construction: `causal_ffd_log_close` is a
    causal FIR filter (verified independently in r124_shared._self_test),
    and every subsequent step is a plain backward-looking pandas
    `.rolling(...)` -- no operation here looks forward. Re-verified by this
    file's own `causal_truncation_probe_series` call below before any
    episode-level number is trusted.
    """
    ffd = causal_ffd_log_close(df)
    short_ma = ffd.rolling(SHORT_DAYS * BARS_PER_DAY).mean()
    long_ma = ffd.rolling(LONG_DAYS * BARS_PER_DAY).mean()
    spread = short_ma - long_ma
    spread_std = spread.rolling(LONG_DAYS * BARS_PER_DAY).std()
    return spread / spread_std


def nearest_fracdiff_detection(z: pd.Series, window: pd.DatetimeIndex,
                                onset: pd.Timestamp, thresh: float = THRESH
                                ) -> pd.Timestamp | None:
    """Timestamp, within `window`, of the z-score's downward crossing
    through `thresh` closest to `onset` -- the fracdiff-momentum analogue
    of `r82_shared.nearest_bocpd_detection`. NaN z-score values (rolling
    warmup) never count as "below threshold" (NaN comparisons are False in
    numpy), matching every other gate file's warmup handling."""
    vals = z.reindex(window).to_numpy()
    below = vals < thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = below[1:] & ~below[:-1]
    cross[0] = bool(below[0]) if len(below) else False
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def null_leads(z_local: np.ndarray, window: pd.DatetimeIndex, onset: pd.Timestamp,
                flip_time: pd.Timestamp, n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                seed: int = NULL_SEED) -> np.ndarray:
    """Adapted from r117_conservative_donchian_gate's `null_leads`: same
    circular block-shift construction, applied here to the continuous
    z-score series via `nearest_fracdiff_detection` instead of a discrete
    vote transition."""
    n_bars = len(z_local)
    shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                     n_draws=n_draws, seed=seed)
    leads = np.full(n_draws, np.nan)
    for k, shift in enumerate(shifts):
        shifted = pd.Series(z_local[shift], index=window)
        detect_time = nearest_fracdiff_detection(shifted, window, onset)
        if detect_time is None:
            continue
        leads[k] = (flip_time - detect_time).total_seconds() / 86400.0
    return leads


def causality_probe(bars: pd.DataFrame) -> bool:
    """Independent causal-truncation check of the FULL composed detection
    series (FFD -> rolling means -> spread -> z-score), run before any
    headline result is trusted, per this project's rule that every round
    re-runs the probe itself rather than trusting a prior claim. Probes
    well inside the 2018 episode window so the check exercises real,
    non-degenerate signal values, not warmup NaNs."""
    ok = causal_truncation_probe_series(fracdiff_zscore_series, bars)
    print(f"\ncausal truncation probe (fracdiff_zscore_series, "
          f"SHORT={SHORT_DAYS}d LONG={LONG_DAYS}d): {'PASS' if ok else 'FAIL'}")
    return ok


def crossing_frequency_stats(z: pd.Series, thresh: float = THRESH) -> dict:
    """How often, and for what fraction of time, the detector is actually
    'triggered' over the WHOLE pre-OOS series -- the direct evidence for
    (or against) the over-triggering failure mode, independent of the
    per-episode null-vs-signal comparison in `run()` below."""
    vals = z.to_numpy()
    finite = np.isfinite(vals)
    below = np.zeros(len(vals), dtype=bool)
    below[finite] = vals[finite] < thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = below[1:] & ~below[:-1]
    n_valid_days = float(finite.sum()) / BARS_PER_DAY
    n_cross = int(cross.sum())
    return dict(
        n_valid_days=n_valid_days,
        n_crossings=n_cross,
        mean_days_between_crossings=(n_valid_days / n_cross) if n_cross else float("nan"),
        frac_time_triggered=float(below[finite].sum()) / float(finite.sum()) if finite.sum() else float("nan"),
    )


def inner_train_threshold_context(bars: pd.DataFrame) -> float:
    """DISCLOSED CONTEXT ONLY (pre-registration point 2): the 15th
    percentile of the z-score series on INNER_TRAIN ONLY, reported
    alongside -- never substituted for -- the fixed a priori THRESH=-1.0
    decision threshold."""
    inner = bars[bars.index <= pd.Timestamp(INNER_TRAIN_END, tz="UTC")]
    assert_no_holdout(inner, "r124_novel: inner-train threshold context")
    z = fracdiff_zscore_series(inner).to_numpy()
    z = z[np.isfinite(z)]
    return float(np.percentile(z, 15)) if len(z) else float("nan")


def gate(bars: pd.DataFrame, v4_frac: pd.Series, z: pd.Series) -> dict:
    hr(f"R-124 NOVEL: fracdiff-momentum z-score (SHORT={SHORT_DAYS}d, LONG={LONG_DAYS}d, "
       f"THRESH={THRESH}) vs v4 anchor vote -- STEP A detection-lag gate")
    print(f"search window=+/-{WINDOW_DAYS}d  null: {N_DRAWS} draws, block={BLOCK_DAYS}d, "
          f"seed={NULL_SEED}\n")

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, WINDOW_DAYS)
        if len(window) == 0:
            print(f"[{label}] onset={onset_str}: window has ZERO bars -- outside data coverage.")
            results.append(dict(label=label, onset=onset_str, lead=float("nan"), pass_b=False))
            continue

        flip_time = nearest_transition(v4_frac, window, onset, direction="down")
        detect_time = nearest_fracdiff_detection(z, window, onset)

        if flip_time is None or detect_time is None:
            print(f"[{label}] onset={onset_str}: "
                  f"{'no v4 vote transition' if flip_time is None else 'no fracdiff-momentum crossing'} "
                  f"found in +/-{WINDOW_DAYS}d window. FAIL by construction.")
            results.append(dict(label=label, onset=onset_str, flip=flip_time, detect=detect_time,
                                 lead=float("nan"), null_median=float("nan"), pass_b=False))
            continue

        lead = (flip_time - detect_time).total_seconds() / 86400.0
        z_local = z.reindex(window).to_numpy()
        leads_null = null_leads(z_local, window, onset, flip_time)
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = pass_a and (not np.isnan(null_median)) and (lead >= null_median)

        print(f"[{label}] onset={onset_str}")
        print(f"    v4 vote nearest downward flip:              {flip_time}")
        print(f"    fracdiff-momentum nearest down-crossing:    {detect_time}")
        print(f"    LEAD = {lead:+.2f} days  null median={null_median:+.2f}d "
              f"(valid draws {len(valid)}/{N_DRAWS})")
        print(f"    PASS (a) lead>=0: {pass_a}   PASS (b) lead>=null median: {pass_b}")

        results.append(dict(label=label, onset=onset_str, flip=flip_time, detect=detect_time,
                             lead=lead, null_median=null_median, pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    passed = n_pass >= 4

    hr("R-124 NOVEL: PER-EPISODE SUMMARY TABLE")
    print(f"{'episode':42s} {'LEAD(d)':>9s} {'null_med(d)':>12s} {'PASS':>6s}")
    print("-" * 74)
    for r in results:
        lead_s = f"{r.get('lead', float('nan')):+9.2f}" if not np.isnan(r.get("lead", float("nan"))) else f"{'n/a':>9s}"
        nm = r.get("null_median", float("nan"))
        nm_s = f"{nm:+12.2f}" if not np.isnan(nm) else f"{'n/a':>12s}"
        print(f"{r['label']:42s} {lead_s} {nm_s} {str(r['pass_b']):>6s}")
    print(f"\nEpisodes passing: {n_pass}/6")

    return dict(results=results, n_pass=n_pass, passed=passed)


def run() -> dict:
    bars = load_btc()
    assert_no_holdout(bars, "r124_novel: full BTC frame")
    print(f"BTC (spot): {len(bars):,} bars  {bars.index[0]} -> {bars.index[-1]}  "
          f"(< {OOS_START})", file=sys.stderr)

    v4_frac = anchor_majority(bars)
    assert_no_holdout(v4_frac, "r124_novel: v4 anchor_majority")

    z = fracdiff_zscore_series(bars)
    assert_no_holdout(z, "r124_novel: fracdiff_zscore_series")

    ctx_thresh = inner_train_threshold_context(bars)
    print(f"\n[disclosed context only, NOT the decision threshold] "
          f"inner-train (<= {INNER_TRAIN_END}) 15th percentile of z-score: {ctx_thresh:+.3f} "
          f"(primary/decision THRESH remains {THRESH:+.3f}, fixed a priori)")

    freq = crossing_frequency_stats(z)
    print(f"\n[crossing-frequency diagnostic, whole pre-OOS series] "
          f"{freq['n_crossings']} down-crossings of THRESH={THRESH} over "
          f"{freq['n_valid_days']:.0f} valid days "
          f"(one every {freq['mean_days_between_crossings']:.1f}d on average); "
          f"fraction of time in the triggered (below-threshold) state: "
          f"{freq['frac_time_triggered']:.1%}")

    result = gate(bars, v4_frac, z)

    max_ts_seen = max(bars.index.max(), z.index.max(), v4_frac.index.max())

    hr("R-124 NOVEL: VERDICT")
    print(f"PRIMARY (and only) configuration: SHORT={SHORT_DAYS}d LONG={LONG_DAYS}d "
          f"THRESH={THRESH}: {result['n_pass']}/6 episodes pass")
    print(f"DECISION RULE (>=4/6): {'PASS' if result['passed'] else 'FAIL'}")

    if result["passed"]:
        print("BRANCH VERDICT: PROMOTE-CANDIDATE for Step B. Per dispatch scope, Step-B "
              "strategy/backtest code is NOT built in this file -- STOPPING here anyway "
              "and reporting this as a live follow-on for a future round.")
    else:
        # Distinguish failure mode. The decisive diagnostic is not the raw
        # LEAD sign alone (that only shows the detector reacts after v4 --
        # "lag") but whether the null (block-bootstrap-shifted) copy of the
        # SAME detector does just as well, or better, at landing near the
        # real onset as the true, unshifted detection does. If a randomly
        # time-shifted copy of the detector's own crossings is AS GOOD OR
        # BETTER than the real one in most/all episodes, the true timing
        # carries no real information about the historical onset beyond
        # what its own base crossing-frequency would give by chance -- the
        # OVER-TRIGGERING / non-specific-alarm signature named a priori in
        # this file's pre-registration (point 7), qualitatively distinct
        # from PURE LAG (where the real detector would still clearly beat
        # its own null baseline, just arrive after v4).
        finite = [r for r in result["results"]
                  if np.isfinite(r.get("lead", float("nan"))) and np.isfinite(r.get("null_median", float("nan")))]
        n_finite = len(finite)
        n_noise_beats_signal = sum(1 for r in finite if r["null_median"] >= r["lead"])
        mean_lead = float(np.mean([r["lead"] for r in finite])) if finite else float("nan")
        mean_null = float(np.mean([r["null_median"] for r in finite])) if finite else float("nan")
        print(f"\nFailure-mode diagnostics: mean true LEAD over episodes with a detection "
              f"= {mean_lead:+.2f}d; mean null-median LEAD = {mean_null:+.2f}d.")
        print(f"episodes where the null (randomly time-shifted) detector's own median LEAD "
              f">= the real detector's true LEAD: {n_noise_beats_signal}/{n_finite} "
              f"(i.e. random placement matched or beat the real, correctly-timed detection).")
        if n_finite == 0:
            mode = "NO DETECTION: the z-score never crossed THRESH inside any episode window."
        elif n_noise_beats_signal >= 4:
            mode = ("OVER-TRIGGERING / NON-SPECIFIC ALARM (the pre-registered expected failure "
                    "mode for this branch): in the great majority of episodes a randomly "
                    "time-shifted copy of this detector's own crossings lands as close to, or "
                    "closer to, the true onset than the real, correctly-timed detection does. "
                    "That is only possible if the detector fires often enough, and "
                    f"non-specifically enough (measured directly above: {freq['n_crossings']} "
                    f"down-crossings of THRESH={THRESH} over the full "
                    f"{freq['n_valid_days']:.0f}-day pre-OOS BTC series, one every "
                    f"{freq['mean_days_between_crossings']:.1f} days on average, spending "
                    f"{freq['frac_time_triggered']:.1%} of all time in the 'triggered' "
                    "below-threshold state), that landing near ANY fixed date by chance is not "
                    "rare. This is a qualitatively DIFFERENT failure signature from the pure "
                    "lag pattern of the ten prior mechanisms: it is not merely slow, its "
                    "specific timing carries essentially no information about the six "
                    "historical stress onsets beyond its own base crossing rate.")
        else:
            mode = ("PURE LAG (like the ten predecessors): the detector's real timing clearly "
                    "beats its own null baseline in most episodes, it simply arrives "
                    "systematically after v4's own anchor vote.")
        print(f"Observed failure mode: {mode}")
        print(f"\nBRANCH VERDICT: NEGATIVE. Fractional-diff momentum, substituted into the "
              f"identical Step-A detection-lag gate, is the eleventh mechanism to fail it -- "
              f"consistent with this round's own pre-registered expectation (r124_shared.py). "
              f"No Step-B strategy/backtest code was built; no bar on or after {OOS_START} "
              f"was read.")

    print(f"\nconfigurations evaluated: 0 swept (1 fixed primary configuration x 6 episodes "
          f"= 6 measurements; inner-train 15th-percentile context above is diagnostic only, "
          f"not a second configuration)")
    print(f"max timestamp read anywhere in this session: {max_ts_seen}  (< {OOS_START})")
    assert max_ts_seen < pd.Timestamp(OOS_START, tz="UTC"), "holdout bar read"

    return dict(result=result, max_ts_seen=max_ts_seen, ctx_thresh=ctx_thresh)


if __name__ == "__main__":
    bars_for_probe = load_btc()
    assert_no_holdout(bars_for_probe, "r124_novel: probe frame")
    hr("R-124 NOVEL: CAUSAL TRUNCATION PROBE (run before any headline result is trusted)")
    probe_ok = causality_probe(bars_for_probe)
    if not probe_ok:
        print("\n*** CAUSALITY PROBE FAILED -- STOPPING. Results below would NOT be "
              "trustworthy; a lookahead bug must be investigated, not reported around. ***",
              file=sys.stderr)
        sys.exit(1)

    final = run()
