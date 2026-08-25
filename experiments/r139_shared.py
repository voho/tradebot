"""R-139: does a causal CUSUM changepoint detector serve as a regime-timing
input to `kelly_regime_v4`, evaluated against the SAME six-episode
detection-lag Step-A gate that HMM (R-01), BOCPD (R-82), Kalman LLT (R-83),
critical slowing down (R-85) and transfer entropy (R-86) were all measured
against and all failed (0-2/6 passes each)? Shared, frozen infrastructure
for a two-branch parallel round. Per ROUTINE.md's parallelism rules this
file is neutral ground: both branches import from it, NEITHER BRANCH EDITS
IT, and it does not itself compute a verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **INFO**, in the same narrow sense R-82's own module
docstring used it -- not a new external data channel (both branches read
only the committed BTC OHLCV close series `kelly_regime_v4` already uses),
but a structurally different ESTIMATOR extracting a regime-timing signal
from that same series. CUSUM (Page 1954, "Continuous inspection schemes",
Biometrika 41(1/2); Hawkins & Olwell 1998, *Cumulative Sum Charts and
Charting for Quality Improvement*) is a sixth, structurally distinct
theoretical basis: sequential statistical-process-control theory, with NO
generative probabilistic model of the regime process at all -- unlike
discrete-state Markov switching (HMM), Bayesian generative changepoint
estimation (BOCPD), linear state-space filtering (Kalman LLT), dynamical-
systems fluctuation statistics (CSD) and information-theoretic directed
flow (transfer entropy), the five bases R-86's own closing verdict named
as exhausted on this exact gate. Secondary: this round also closes the one
concretely-named, still-open thread from R-137/R-138's own re-ranking --
"sweeping the CUSUM detector's own textbook parameters
(CUSUM_TRAIL_DAYS=90, k=0.5 sigma, h=5 sigma) ... still small, still not
comparable in weight to B-06" -- by making that sweep the NOVEL branch's
whole content, pre-registered and reported in full rather than fished.

**Not a duplicate of:**
- R-01 (HMM), R-82 (BOCPD), R-83 (Kalman LLT), R-85 (CSD), R-86 (transfer
  entropy): same six-episode Step-A detection-lag gate, structurally
  different detector each time; CUSUM has never been run against it.
- R-137 / R-138: both used a causal CUSUM detector, but only as an EVENT
  SET feeding a permutation test on a different question (does excising
  ETH-idiosyncratic dates rescue an ETH replication bar; does the edge
  concentrate at ~3 events at all) -- neither tested CUSUM as a
  regime-timing INPUT to v4's own vote, and neither swept the detector's
  own parameters against real data (R-137/R-138 both used the fixed
  textbook constants verbatim, and R-138's own Verdict names the sweep as
  future work). This round is that follow-on, reframed against the
  detection-lag gate rather than the permutation-test machinery, because
  the two ask different questions: R-137/R-138 ask "is 2022-07-18 a real
  break that explains an ETH replication failure"; this round asks "can a
  CUSUM-derived regime signal detect KNOWN historical BTC regime breaks
  fast enough to be useful as a v4 input at all."
- R-91 (four-state GHM classification of v4's OWN anchor-agreement
  structure, no external detector, never touches the six-episode gate).
- B-42 / R-92 (deriving the anchor SPAN from a fitted generative Sharpe
  model -- a different question about the existing anchors' timescale,
  not a new detector).

**Is it simulable here?** Yes. Entirely computable from the already-
committed BTC OHLCV series, reusing `experiments/r82_shared.py`'s
episode table, gate machinery and confirming-vote combination rule
verbatim (imported, not re-derived), and `experiments/r138_shared.py`'s
`causal_cusum_breaks` verbatim. Zero new data, zero new fetch.

**What would make this fail, named now, before any code beyond this
shared module was run:**

(a) CONSERVATIVE branch (fixed textbook constants): CUSUM's detections
    cluster AFTER v4's own anchor reaction -- the identical failure mode
    every one of the five prior mechanisms hit against this gate -- or the
    detections are not distinguishable from an arbitrary block-shift of
    the same run-length series (the block-bootstrap null every predecessor
    round used, reused here unchanged).
(b) NOVEL branch (swept constants): no cell in the pre-registered grid
    clears the >=4/6 bar, OR the best-passing cell is an isolated peak
    rather than a plateau (neighbouring cells failing sharply) -- which
    would mean any apparent pass is a six-episode fitting artifact rather
    than a property of CUSUM detection at a sensible parameterization, per
    the promotion bar's own "plateau, not a peak" requirement
    (`docs/ROUTINE.md` step 4). Guarded explicitly: the grid is reported
    in FULL (every cell, not just the best), and `plateau_ok()` below is
    the pre-registered mechanical check for this, run before any Step-B
    number exists.

If a branch's Step-A gate fails its own pre-registered bar, that branch
STOPS at Step-A (the identical convention R-82/R-85/R-86 used) -- no Step-B
implementation, no holdout read, reported as NEGATIVE at Step-A. Both
branches disclose this decision now, before any real-data number exists.

**Step-B, pre-registered now for both branches, contingent on their own
Step-A gate passing (fixed before any Step-A number was computed):**

- CONSERVATIVE, if triggered: `confirming_vote_frac` (R-53/R-55's
  combination rule, imported verbatim above) with `meta_vote` = 1 on any
  bar where CUSUM's run length is `<= K_SHORT_DAYS` (a live "recent break"
  flag, discrete per R-80/R-81's lesson), `weight` swept over {0.5, 1.0,
  1.5}. Compare the resulting `kelly_regime_v4` variant against (i) frozen
  `kelly_regime_v4` and (ii) a realized-volatility-matched
  `ConstantExposureHold` (`experiments/matched_hold.py`), on `W_TRAIN`
  (<=`INNER_TRAIN_END`) via `tradebot.inference.paired_bootstrap`
  (growth-log-return diff = D1, max-drawdown-pct diff = D2, both with a
  block-bootstrap 95% CI). `further_work = (D1 or D2 excludes zero in
  the improving direction) and the same sign holds directionally on
  `W_VAL` (`INNER_VAL_START`..`INNER_VAL_END`)`. Only if `further_work` is
  True does the round reach Step 4 (freeze, then read `W_HOLD`).
- NOVEL, if triggered: a TRIGGER-OVERRIDE combination, structurally
  distinct from the confirming vote per R-73's own recommendation ("try a
  structurally different combination rule ... rather than a fifth
  brake") -- on a bar where the winning grid cell's CUSUM run length
  crosses `<= K_SHORT_DAYS` AND the break's own local trailing mean
  shifted negative (computed locally in the novel branch's own file, not
  in this shared module, from the same trailing window
  `causal_cusum_breaks` already scans), `kelly_regime_v4`'s exposure is
  forced toward flat immediately regardless of the anchor vote, latching
  until the anchor vote's own OWN latch next changes (i.e. the override
  never fights the anchor gate once it has caught up, only pre-empts it).
  Same D1/D2/`further_work` battery as the conservative branch, same
  `W_TRAIN`/`W_VAL` split, same holdout gate.

Neither Step-B path is expected to be reached, given the five prior
mechanisms' 0-2/6 base rate on this exact gate -- named now so a reader
can check the round did not invent a rule after seeing whether it would be
needed.

=====================================================================
STEP-A: the detection-lag gate (shared machinery)
=====================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_onchain_causal  # noqa: E402

from experiments.r82_shared import (  # noqa: E402
    BARS_PER_DAY,
    INNER_TRAIN_END,
    INNER_VAL_END,
    INNER_VAL_START,
    K_SHORT_DAYS,
    OOS_START,
    STRESS_EPISODES,
    V4_BAND,
    V4_HORIZONS,
    anchor_majority,
    block_bootstrap_shifts,
    confirming_vote_frac,
    episode_window,
    nearest_bocpd_detection as nearest_run_length_detection,  # generic on any run-length series
    nearest_transition,
)
from experiments.r138_shared import (  # noqa: E402
    CUSUM_H_MULT,
    CUSUM_K_MULT,
    CUSUM_TRAIL_DAYS,
    causal_cusum_breaks,
    price_daily_logret,
)

WINDOW_DAYS = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 139

# CONSERVATIVE branch: identical textbook constants R-137/R-138 already
# used, unedited. NOVEL branch: the pre-registered sweep grid, fixed here
# before any real-data number was computed. 4 x 3 x 3 = 36 cells -- the
# round's total trials count for deflated Sharpe, per ROUTINE.md's
# parallelism rule (trials are counted across BOTH branches).
NOVEL_TRAIL_GRID = (30, 60, 90, 120)
NOVEL_K_GRID = (0.25, 0.5, 1.0)
NOVEL_H_GRID = (3.0, 5.0, 7.0)


def cusum_run_length_daily(daily_logret: pd.Series, *, trail_days: int = CUSUM_TRAIL_DAYS,
                            k_mult: float = CUSUM_K_MULT, h_mult: float = CUSUM_H_MULT
                            ) -> pd.Series:
    """Daily 'days since last CUSUM-flagged break' series, the CUSUM
    analogue of BOCPD's MAP run length -- 0 on a break day, incrementing
    otherwise. Causal by construction: `causal_cusum_breaks` only ever
    flags day t using `daily_logret[:t+1]` (a trailing window ending at
    t), so this derived series inherits the same causality.
    """
    x = daily_logret.dropna()
    breaks = set(causal_cusum_breaks(x, trail_days=trail_days, k_mult=k_mult, h_mult=h_mult))
    run_length = np.zeros(len(x), dtype=int)
    since = trail_days  # unknown/warmup state before any break is observed
    for i, ts in enumerate(x.index):
        if ts in breaks:
            since = 0
        else:
            since = since + 1 if i > 0 else since
        run_length[i] = since
    return pd.Series(run_length, index=x.index, name="cusum_run_length")


def cusum_daily_causal_signals(df: pd.DataFrame, *, trail_days: int = CUSUM_TRAIL_DAYS,
                                k_mult: float = CUSUM_K_MULT, h_mult: float = CUSUM_H_MULT
                                ) -> pd.DataFrame:
    """Resample to daily, run the CUSUM run-length series, and align onto
    `df`'s 5-minute index with the same full-calendar-day causal shift
    every daily-cadence signal in this project uses
    (`tradebot.data.align_onchain_causal`)."""
    daily_ret = price_daily_logret(df)
    rl = cusum_run_length_daily(daily_ret, trail_days=trail_days, k_mult=k_mult, h_mult=h_mult)
    daily = pd.DataFrame({"cusum_run_length": rl})
    return align_onchain_causal(daily, df)


def assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


def step_a_gate(bars: pd.DataFrame, *, trail_days: int, k_mult: float, h_mult: float,
                 k_short_days: int = K_SHORT_DAYS, window_days: int = WINDOW_DAYS,
                 n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS,
                 seed: int = NULL_SEED, verbose: bool = True) -> dict:
    """The R-82-identical Step-A detection-lag gate, CUSUM in place of
    BOCPD. An episode PASSES if (a) CUSUM detects at or before v4's own
    nearest downward anchor-flip (lead >= 0), AND (b) that lead beats the
    block-bootstrap null's median. Gate PASSES overall at >= 4/6 episodes.
    """
    majority = anchor_majority(bars)
    cusum = cusum_daily_causal_signals(bars, trail_days=trail_days, k_mult=k_mult, h_mult=h_mult)
    assert_no_holdout(cusum)
    run_length = cusum["cusum_run_length"]

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, window_days)
        if len(window) == 0:
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue
        flip_time = nearest_transition(majority, window, onset, direction="down")
        detect_time = nearest_run_length_detection(run_length, window, onset, k_short_days)
        if flip_time is None or detect_time is None:
            results.append(dict(label=label, pass_b=False, lead=float("nan")))
            continue
        lead = (flip_time - detect_time).total_seconds() / 86400.0
        local = run_length.reindex(window).to_numpy()
        n_bars = len(local)
        shifts = block_bootstrap_shifts(n_bars=n_bars, block_days=block_days,
                                         n_draws=n_draws, seed=seed)
        leads_null = np.full(n_draws, np.nan)
        for k, shift in enumerate(shifts):
            shifted = local[shift]
            short = shifted <= k_short_days
            cross = np.zeros(n_bars, dtype=bool)
            cross[1:] = short[1:] & ~short[:-1]
            cross[0] = bool(short[0])
            idx = np.where(cross)[0]
            if len(idx) == 0:
                continue
            times = window[idx]
            deltas = np.abs((times - onset).to_numpy())
            dt = times[int(np.argmin(deltas))]
            leads_null[k] = (flip_time - dt).total_seconds() / 86400.0
        valid = leads_null[~np.isnan(leads_null)]
        null_median = float(np.median(valid)) if len(valid) else float("nan")
        pass_a = lead >= 0
        pass_b = bool(pass_a and not np.isnan(null_median) and lead >= null_median)
        if verbose:
            print(f"    [{label}] lead={lead:+.2f}d null_median={null_median:+.2f}d PASS={pass_b}")
        results.append(dict(label=label, onset=onset_str, lead=lead,
                             null_median=null_median, pass_b=pass_b))

    n_pass = sum(1 for r in results if r["pass_b"])
    return dict(trail_days=trail_days, k_mult=k_mult, h_mult=h_mult,
                results=results, n_pass=n_pass, passed=n_pass >= 4)


def plateau_ok(grid_results: list[dict], winner: dict, min_neighbor_pass: int = 4) -> bool:
    """Pre-registered plateau check for the NOVEL branch's swept grid: a
    passing winner only counts if at least one immediate neighbour in EACH
    swept dimension (one grid step away, when it exists) also clears
    `min_neighbor_pass`. An isolated single-cell pass is reported as a
    peak, not promoted to 'the grid clears the gate'.
    """
    def neighbors(cell):
        wt, wk, wh = cell["trail_days"], cell["k_mult"], cell["h_mult"]
        out = []
        for grid, key in ((NOVEL_TRAIL_GRID, "trail_days"),
                          (NOVEL_K_GRID, "k_mult"), (NOVEL_H_GRID, "h_mult")):
            vals = sorted(grid)
            i = vals.index(cell[key])
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    probe = dict(trail_days=wt, k_mult=wk, h_mult=wh)
                    probe[key] = vals[j]
                    out.append(probe)
        return out

    lookup = {(r["trail_days"], r["k_mult"], r["h_mult"]): r for r in grid_results}
    near = neighbors(winner)
    if not near:
        return False
    hits = 0
    for n in near:
        key = (n["trail_days"], n["k_mult"], n["h_mult"])
        r = lookup.get(key)
        if r is not None and r["n_pass"] >= min_neighbor_pass:
            hits += 1
    return hits >= 1
