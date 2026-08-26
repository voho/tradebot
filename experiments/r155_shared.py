"""R-155: does topological data analysis -- persistent homology (H0) of a
causal Takens-embedded window of daily log returns -- serve as a
regime-timing input to `kelly_regime_v4`, evaluated against the SAME
six-episode detection-lag Step-A gate that HMM (R-01), BOCPD (R-82),
Kalman LLT (R-83), critical slowing down / CSD (R-85), transfer entropy
(R-86), Hawkes (R-96), POT/GPD (R-98), bipower-variation jump/QV (R-99),
CUSUM (R-139) and LPPLS (R-141) were all measured against and all failed
(0-3/6 passes each)? Shared, frozen infrastructure for a two-branch
parallel round. Per ROUTINE.md's parallelism rules this file is neutral
ground: both branches import from it, NEITHER BRANCH EDITS IT, and it
does not itself compute a verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Step 0 (08-26, eleventh same-day pass): no undispatched frozen
pre-registration (`r99_shared.py` was the newest `_shared.py` file before
this one and already has its matching R-99 section-B entry); origin/main
and this working branch reconciled before writing this file. Backlog-table
grep: unchanged live set (B-06, B-09, B-10, B-17, B-28), none actionable
on their own merits (see R-154's own close-out). Rather than re-run the
literature categories the ten prior 08-26 passes (through R-154) already
swept and closed, a dedicated research-only sub-agent was dispatched with
the ten prior passes' own closed-candidate list and told to grep the
ledger by name before treating anything as promising. It found exactly one
literal non-duplicate: topological data analysis. Grepped
"topological", "persistent homology", "persistence landscape", "TDA",
"Takens" against docs/LEDGER.md -- zero hits anywhere before this round.

Constraint attacked: **INFO**, in the identical narrow sense every
predecessor on this gate used it -- not a new external data channel (both
branches read only the committed BTC OHLCV close series `kelly_regime_v4`
already uses), but a structurally different ESTIMATOR of regime state
extracted from that same series. Persistent homology (Carlsson 2009,
"Topology and data", Bull. AMS 46(2), 255-308; Edelsbrunner & Harer 2010,
*Computational Topology*) applied to a Takens (1981) time-delay embedding
of a return series is an eleventh theoretical basis with no shared
machinery against any of the ten prior attempts: discrete-state Markov
switching (HMM), Bayesian generative changepoint estimation (BOCPD),
linear state-space filtering (Kalman LLT), dynamical-systems fluctuation
statistics (CSD), information-theoretic directed flow (transfer entropy),
self-exciting point processes (Hawkes), extreme-value tail theory
(POT/GPD), bipower-variation jump/quadratic-variation decomposition,
sequential statistical-process-control (CUSUM), log-periodic bubble
models (LPPLS) -- and now algebraic-topological persistent homology of a
reconstructed phase space. Financial/crypto-specific motivation, both
dated 2026 (verified live by the research sub-agent via WebSearch before
this file was written): Bhatia et al., "Topological Complexity and Phase
Space Stability: A Persistent Homology Approach to Cryptocurrency Risk"
(arXiv:2604.13311, BTC futures 2019-2026) and "Null-Validated Topological
Signatures of Financial Market Dynamics" (arXiv:2602.00383) -- both argue
rising topological complexity/instability in a delay-embedded price point
cloud precedes regime breaks.

**Not a duplicate of:** R-01 (HMM), R-82 (BOCPD), R-83 (Kalman LLT), R-85
(CSD), R-86 (transfer entropy), R-96 (Hawkes), R-98 (POT/GPD), R-99
(jump/QV), R-139 (CUSUM), R-141 (LPPLS) -- same six-episode Step-A
detection-lag gate, structurally different detector each time; TDA has
never been run against it, and is absent from the ledger entirely by
direct grep.

**Is it simulable here?** Yes. Entirely computable from the already-
committed BTC OHLCV close series, daily-resampled exactly like every
predecessor on this gate (R-82's own disclosed and reused design choice:
v4's anchors operate on a 20-80 calendar-day horizon, so a daily cadence
matches the horizon the mechanism is meant to describe), reusing
`experiments/r82_shared.py`'s episode table, gate machinery
(`nearest_transition`, `episode_window`, `block_bootstrap_shifts`,
`nearest_bocpd_detection` -- generic on any run-length series, per R-139's
own precedent) and anchor gate (`anchor_majority`) verbatim. No order
book, no external data, no proxying.

**What would make this fail, named now** -- this round's explicit prior,
restated from R-85's own closing diagnosis and R-86's own closing lesson:
the six-episode gate is dominated by SUDDEN, news-driven shocks (COVID,
Terra/Luna, FTX) where no mechanism tried so far, across ten structurally
distinct theoretical bases, has beaten v4's own crude fixed-window anchor
average; only the one slow-building 2018 episode has ever been
anticipated early, by any detector. A topological-complexity statistic
computed from the same daily-resampled OHLCV window is, like CSD, a
geometry-of-recent-price-action statistic, and there is no a priori
reason to expect it escapes the same shock-suddenness problem.
**Falsification, pre-registered:** fewer than 4/6 episodes pass the
identical Step-A gate defined below -- STOP, no Step-B, report NEGATIVE.
The bar is not relaxed after seeing the numbers.

=====================================================================
THE CONSTRUCTION
=====================================================================

1. **Causal Takens embedding.** For a 1-D series `x` and day index `t`,
   the delay-embedded point at `t` is
   `(x[t], x[t-delay], x[t-2*delay], ..., x[t-(dim-1)*delay])` -- a point
   in R^dim built only from `x[<=t]`, causal by construction. A rolling
   window of `window_days` consecutive embedded points (each itself built
   only from data at or before its own day) forms the point cloud whose
   topology is measured at day `t`.

2. **H0 persistent homology via minimum spanning tree.** For a
   Vietoris-Rips filtration on a finite point cloud, the 0-dimensional
   (connected-component) persistence diagram is exactly the sorted edge
   weights of the point cloud's Euclidean minimum spanning tree (MST) --
   a standard equivalence (e.g. Chazal & Michel 2021, "An introduction to
   Topological Data Analysis", Frontiers in Artificial Intelligence 4,
   Section 3.2): every H0 feature is born at filtration value 0 and dies
   exactly at the weight of the MST edge that merges its component, so
   total H0 persistence = sum of MST edge weights. This project has no
   TDA library dependency (`gudhi`/`ripser` are not in `pyproject.toml`
   and are not added here); the MST-equivalence lets H0 total persistence
   be computed with a plain O(n^2) Prim's algorithm in numpy, appropriate
   given each window's point cloud has at most a few dozen points.
   `mst_total_weight` below is that computation, self-contained.

3. **Daily topological-instability scalar.** `h0_total_persistence_daily`
   slides the window across the whole daily-return series (causal: day
   t's value depends only on `x[<=t]`) and returns one scalar per day --
   the TDA analogue of BOCPD's MAP run length / CUSUM's run-length-since-
   last-break, except this scalar is a continuous complexity measure, not
   itself a run length. It is converted into a run-length-shaped series
   (the shape the shared `nearest_bocpd_detection` gate helper expects)
   by `causal_alarm_run_length`: a trailing `trail_days`-day rolling
   z-score of the scalar is computed causally (mean/std over
   `[t-trail_days, t)`, excluding day t itself, matching R-138/R-139's own
   causal-CUSUM convention of a trailing, not centered, baseline), an
   "alarm" fires the first day the z-score exceeds `z_thresh`, and
   `run_length[t]` counts days since the most recent alarm (0 on an alarm
   day), exactly `cusum_run_length_daily`'s shape from `r139_shared.py`
   with CUSUM's break-set swapped for a topological-instability alarm.

=====================================================================
BOTH BRANCHES, PARAMETER FREEDOM
=====================================================================

**CONSERVATIVE** (one configuration, no sweep): `window_days=20` (matches
v4's own shortest/fastest anchor, `V4_HORIZONS[0]`, the most literal
reading of "does a topological alarm fire before v4's fastest anchor
reacts" and the choice least favourable to a slow, over-smoothed
detector), `embed_dim=3` (the standard default dimension in the cited
TDA-for-finance literature and in the broader delay-embedding literature
when no dimension is separately estimated), `embed_delay=1` (one day,
literal), `trail_days=90` (identical to BOCPD/CUSUM's own baseline
convention on this project, for direct comparability), `z_thresh=2.0`
(a standard two-sigma alarm threshold, fixed a priori, not tuned).

**NOVEL** (pre-registered 3x3 grid, fixed here before any real-data
number was computed): `window_days in {10, 20, 30}` x
`embed_dim in {2, 3, 4}`, holding `embed_delay=1`, `trail_days=90`,
`z_thresh=2.0` fixed at the conservative branch's own values -- the two
swept parameters are the ones specific to the topological construction
itself (how much history the point cloud sees, how many lags reconstruct
its phase space), matching R-139's own novel-branch convention of
sweeping the new detector's own construction parameters rather than the
shared alarm/gate machinery. 9 cells. **Configurations evaluated in this
round: 10** (1 conservative + 9 novel), the round's trials count for
deflated Sharpe per ROUTINE.md's parallelism rule (trials are counted
across BOTH branches).

Both branches stop at Step A if the gate does not clear >=4/6 -- per
every predecessor round on this gate (R-82, R-85, R-96, R-98, R-99,
R-139, R-141), a Step-A failure is reported as the round's whole result;
no Step-B (vote combination, holdout) is attempted. Neither branch may
read any bar at or after `OOS_START`.
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
    K_SHORT_DAYS,
    OOS_START,
    STRESS_EPISODES,
    V4_HORIZONS,
    anchor_majority,
    block_bootstrap_shifts,
    episode_window,
    nearest_bocpd_detection as nearest_run_length_detection,  # generic on any run-length series
    nearest_transition,
)

WINDOW_DAYS_GATE = 60   # +/- search window around each episode onset
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 155

# Conservative branch: one fixed configuration.
CONS_WINDOW_DAYS = V4_HORIZONS[0]   # 20, v4's fastest anchor
CONS_EMBED_DIM = 3
CONS_EMBED_DELAY = 1
CONS_TRAIL_DAYS = 90
CONS_Z_THRESH = 2.0

# Novel branch: pre-registered 3x3 sweep grid, fixed before any real-data
# number was computed. Delay/trail/threshold held at the conservative
# branch's own values throughout.
NOVEL_WINDOW_GRID = (10, 20, 30)
NOVEL_DIM_GRID = (2, 3, 4)


# --------------------------------------------------------- TDA construction


def mst_total_weight(points: np.ndarray) -> float:
    """Sum of Euclidean minimum-spanning-tree edge weights over a point
    cloud -- equals total H0 persistence of its Vietoris-Rips filtration
    (every 0-dim feature is born at 0 and dies at the MST edge weight that
    merges its component). Plain O(n^2) Prim's algorithm, appropriate for
    the small (<=30-point) clouds used here; no TDA library dependency.
    """
    n = points.shape[0]
    if n < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    min_edge = dist[0].copy()
    total = 0.0
    for _ in range(n - 1):
        min_edge_masked = np.where(in_tree, np.inf, min_edge)
        j = int(np.argmin(min_edge_masked))
        total += float(min_edge_masked[j])
        in_tree[j] = True
        min_edge = np.minimum(min_edge, dist[j])
    return total


def causal_takens_embedding(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Rows `i >= (dim-1)*delay` of the delay embedding
    `(x[i], x[i-delay], ..., x[i-(dim-1)*delay])`. Row i depends only on
    `x[<=i]` -- causal by construction. Returns shape
    `(len(x) - (dim-1)*delay, dim)`; the first `(dim-1)*delay` rows of
    `x` have no embedded point.
    """
    n = len(x)
    lag = (dim - 1) * delay
    if n <= lag:
        return np.empty((0, dim))
    cols = [x[lag - k * delay: n - k * delay] for k in range(dim)]
    return np.column_stack(cols)


def h0_total_persistence_daily(daily_logret: pd.Series, *, window_days: int,
                                embed_dim: int, embed_delay: int) -> pd.Series:
    """One H0-total-persistence scalar per day, using only the trailing
    `window_days` embedded points ending at that day (causal: day t's
    value depends only on `daily_logret[<=t]`)."""
    x = daily_logret.dropna().to_numpy()
    idx = daily_logret.dropna().index
    lag = (embed_dim - 1) * embed_delay
    out = np.full(len(x), np.nan)
    min_points_needed = max(3, embed_dim + 1)
    for t in range(len(x)):
        lo = max(0, t - window_days + 1)
        seg = x[lo:t + 1]
        emb = causal_takens_embedding(seg, embed_dim, embed_delay)
        if emb.shape[0] < min_points_needed:
            continue
        out[t] = mst_total_weight(emb)
    return pd.Series(out, index=idx, name="h0_total_persistence")


def causal_alarm_run_length(scalar_daily: pd.Series, *, trail_days: int,
                             z_thresh: float) -> pd.Series:
    """Days-since-last-alarm run length, alarm = trailing (excludes day t
    itself) rolling z-score of `scalar_daily` crossing above `z_thresh`.
    Same shape as R-139's `cusum_run_length_daily`, generalised to any
    scalar instability series."""
    x = scalar_daily.dropna()
    vals = x.to_numpy()
    n = len(vals)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - trail_days)
        hist = vals[lo:t]  # excludes t itself -- trailing, not centered
        if len(hist) >= max(5, trail_days // 3):
            mean[t] = np.mean(hist)
            std[t] = np.std(hist)
    z = np.where(std > 1e-12, (vals - mean) / np.where(std > 1e-12, std, 1.0), np.nan)
    alarm = z > z_thresh
    run_length = np.zeros(n, dtype=float)
    since = float(trail_days)  # warmup state before any alarm/valid z observed
    for i in range(n):
        if np.isnan(z[i]):
            run_length[i] = since
            continue
        if alarm[i]:
            since = 0.0
        else:
            since = since + 1.0 if i > 0 else since
        run_length[i] = since
    return pd.Series(run_length, index=x.index, name="tda_run_length")


def tda_daily_causal_signals(df: pd.DataFrame, *, window_days: int, embed_dim: int,
                              embed_delay: int, trail_days: float, z_thresh: float
                              ) -> pd.DataFrame:
    """Resample to daily, compute the H0-persistence scalar and its
    alarm-derived run length, and align onto `df`'s 5-minute index with
    the same full-calendar-day causal shift every daily-cadence signal in
    this project uses (`tradebot.data.align_onchain_causal`)."""
    daily_close = df["close"].resample("1D").last().dropna()
    daily_ret = np.log(daily_close).diff().dropna()
    scalar = h0_total_persistence_daily(daily_ret, window_days=window_days,
                                         embed_dim=embed_dim, embed_delay=embed_delay)
    run_length = causal_alarm_run_length(scalar, trail_days=trail_days, z_thresh=z_thresh)
    daily = pd.DataFrame({"tda_run_length": run_length})
    return align_onchain_causal(daily, df)


def assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


# --------------------------------------------------------- Step-A gate infra


def step_a_gate(bars: pd.DataFrame, *, window_days: int, embed_dim: int,
                 embed_delay: int = CONS_EMBED_DELAY, trail_days: float = CONS_TRAIL_DAYS,
                 z_thresh: float = CONS_Z_THRESH, k_short_days: int = K_SHORT_DAYS,
                 window_days_gate: int = WINDOW_DAYS_GATE, n_draws: int = N_DRAWS,
                 block_days: int = BLOCK_DAYS, seed: int = NULL_SEED,
                 verbose: bool = True) -> dict:
    """The R-82-identical Step-A detection-lag gate, the TDA alarm run
    length in place of BOCPD's MAP run length / CUSUM's break-derived run
    length. An episode PASSES if (a) the TDA alarm detects at or before
    v4's own nearest downward anchor-flip (lead >= 0), AND (b) that lead
    beats the block-bootstrap null's median. Gate PASSES overall at
    >= 4/6 episodes."""
    majority = anchor_majority(bars)
    tda = tda_daily_causal_signals(bars, window_days=window_days, embed_dim=embed_dim,
                                    embed_delay=embed_delay, trail_days=trail_days,
                                    z_thresh=z_thresh)
    assert_no_holdout(tda)
    run_length = tda["tda_run_length"]

    results = []
    for label, onset_str in STRESS_EPISODES:
        onset, window = episode_window(bars, onset_str, window_days_gate)
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
    return dict(window_days=window_days, embed_dim=embed_dim, embed_delay=embed_delay,
                trail_days=trail_days, z_thresh=z_thresh,
                results=results, n_pass=n_pass, passed=n_pass >= 4)


def plateau_ok(grid_results: list[dict], winner: dict, min_neighbor_pass: int = 4) -> bool:
    """Pre-registered plateau check for the NOVEL branch's swept grid,
    identical rule to R-139's own: a passing winner only counts if at
    least one immediate neighbour in EACH swept dimension (one grid step
    away, when it exists) also clears `min_neighbor_pass`."""
    w_grid = sorted(NOVEL_WINDOW_GRID)
    d_grid = sorted(NOVEL_DIM_GRID)
    by_key = {(r["window_days"], r["embed_dim"]): r for r in grid_results}
    w, d = winner["window_days"], winner["embed_dim"]
    for grid, key_fn in ((w_grid, lambda v: (v, d)), (d_grid, lambda v: (w, v))):
        i = grid.index(w if grid is w_grid else d)
        neighbours = [grid[i - 1]] if i > 0 else []
        if i < len(grid) - 1:
            neighbours.append(grid[i + 1])
        if not neighbours:
            continue
        ok = any(by_key.get(key_fn(v), {}).get("n_pass", 0) >= min_neighbor_pass
                 for v in neighbours)
        if not ok:
            return False
    return True
