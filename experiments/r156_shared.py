"""Shared, read-only utilities for the R-156 round (08-26/08-27).

Idea in one sentence, restated from the IN-PROGRESS stub this file
completes (docs/LEDGER.md, "R-156 deterministic Elliott Wave counter
(B-10)"): implement Elliott Wave counting as a fully mechanical, causal,
no-discretion ZigZag+Fibonacci pivot machine, then test it two ways --
(CONSERVATIVE) as a standalone directional strategy, exactly B-10's own
brief, and (NOVEL) as a SIZE-axis regime-timing input to `kelly_regime_v4`,
run through the IDENTICAL six-episode Step-A detection-lag gate that
R-82 (BOCPD), R-83 (Kalman LLT), R-85 (CSD), R-86 (transfer entropy),
R-96 (Hawkes), R-98 (POT/GPD), R-99 (jump/QV), R-139 (CUSUM), R-141
(LPPLS) and R-155 (TDA/persistent homology) all failed (0-3/6 passes
each). This file is neutral ground per ROUTINE.md's parallelism rules:
both branches import from it, NEITHER BRANCH EDITS IT, and it computes no
verdict of its own.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Step 0 (08-26/27): the prior IN-PROGRESS stub ("R-156 ... dispatched
conservative+novel") named a direction but froze no `r156_shared.py` and
no branch file exists on disk (`ls experiments/*_shared.py` finds none
newer than r155) -- so nothing was actually dispatched, and the strict
"execute the frozen file verbatim" clause of Step 0 does not apply (there
is no frozen file to execute). Treated here as the direction having been
SELECTED but not yet designed: this file is that design, written before
any real-data number from it exists, honouring the prior session's stated
split (conservative = literal ZigZag+Fibonacci mechanical counter per
B-10 as filed; novel = wave-count confidence as a SIZE-axis regime
multiplier on `kelly_regime_v4`) rather than discarding it. `origin/main`
and this working branch were reconciled to the same tip
(`dbca7d3`) before this file was written.

Backlog: **B-10**, "Deterministic Elliott wave counter -- LOW -- Only as
a documented negative result, per R-18. ZigZag pivots, mechanical
impulse/corrective rules, no discretion. About a day, converts an
unfalsifiable debate into a table row." R-18 (08-16, NOT PURSUED) found
Elliott Wave "not falsifiable as practised" (counts re-labelled after the
fact -- exactly the leak class `test_causality_strict.py` exists to
catch) and its one quantitative component refuted: Batchelor & Ramyar
(2005, working paper, cited via Wikipedia/CFA-level summaries; "the idea
that prices retrace to a Fibonacci ratio ... clearly lacks any scientific
rationale", no significant difference between Fibonacci-ratio frequencies
in the Dow and frequencies expected at random) -- reconfirmed by this
round's own literature check (WebSearch, 08-26): no 2024-2026 study
overturns Batchelor & Ramyar, and the one positive recent result cited in
R-18 (ElliottAgents, Applied Sciences 14(24) 2024) is 14/19 vs 11/19 over
a single monotonic 2022-2024 BTC bull with no walk-forward -- not
independent evidence the ratios themselves carry information. B-10's own
prescription is therefore executed literally: build the falsifiable,
no-discretion version and let the comparison table -- not a debate about
whether any given wave count was "right" -- carry the verdict.

Constraint attacked:
- CONSERVATIVE attacks nothing structural (an EXPECTED negative, per the
  standing diagnosis's own one-line summary: "every strategy that tries
  to predict what happens next loses" -- this is another pure predictor).
  Its value is procedural, exactly as B-10 states: replacing an
  unfalsifiable debate with one falsifiable, mechanical, causal
  implementation and a table row.
- NOVEL attacks **INFO** in the identical narrow sense every Step-A-gate
  predecessor used it: not a new external data channel (reads only the
  committed BTC OHLCV close series `kelly_regime_v4` already uses), but a
  structurally different ESTIMATOR of "did the crowd's structure just
  break" -- an algebraic/combinatorial rule-violation event (a ZigZag
  pivot breaking one of Frost & Prechter's (1978, *Elliott Wave
  Principle*) three hard impulse rules) rather than a probabilistic,
  spectral, information-theoretic, extreme-value or topological one.

**Not a duplicate of:**
- R-18 (assessment only, no code, no backtest -- this round is the
  implementation R-18 recommended against pursuing as a *return* thesis
  but that B-10 later filed as worth building anyway, as a negative-result
  table row).
- R-82/83/85/86/96/98/99/139/141/155 (nine prior Step-A-gate mechanisms):
  same six-episode detection-lag gate, structurally different detector
  each time (Markov switching, linear state-space, dynamical-systems
  fluctuation statistics, information-theoretic flow, self-exciting point
  processes, extreme-value theory, jump/quadratic-variation decomposition,
  sequential process control, log-periodic bubbles, algebraic topology);
  a deterministic combinatorial rule-violation counter on ZigZag pivots
  has never been run against it and is absent from the ledger entirely by
  direct grep (`grep -i elliott docs/LEDGER.md` before this round: R-18
  and its two backlog-table mentions only).
- R-62 (factored v4 into vote x scale; vote carries the whole
  matched-exposure drawdown signature, scale carries none): the same
  reason r82_shared.py gives for confining every Step-A-gate mechanism's
  vote to a SIZE-axis regime multiplier rather than touching v4's
  conditional-vol-targeting scale factor, inherited here unchanged.

**Is it simulable here?** Yes, entirely from the already-committed BTC
5-minute OHLCV close series -- no order book, no discretion, no external
data. Conservative operates directly on 5-minute closes (a directional
strategy needs bar-native resolution to place real orders). Novel
resamples to daily, matching v4's own 20-80 CALENDAR-DAY anchor horizon --
the identical, disclosed design choice r82_shared.py made and every
Step-A-gate predecessor since has reused.

**What would make this fail, named now, before any real-data number:**
- CONSERVATIVE: it is EXPECTED to fail -- underperform `buy_and_hold`
  out-of-sample after real costs, most likely via the failure mode every
  other pure directional predictor in this project's Section A registry
  has shown (L-19 through L-25): repeated count invalidation (the ZigZag
  reconfirms a new candidate wave 0 every time price reverses) churns
  entries/exits and pays the taker fee on both sides more often than the
  wave-3/wave-5 moves it is trying to capture pay back. That expectation
  does not exempt it from the standard promotion bar -- it is reported
  either way, per B-10's own design, and registered as a table row either
  way.
- NOVEL: this round's explicit prior, restated from R-155's own closing
  diagnosis (echoing R-85's and R-86's): the six-episode gate is
  dominated by SUDDEN, news-driven shocks (COVID, Terra/Luna, FTX) where
  no mechanism tried so far, across ten structurally distinct theoretical
  bases, has beaten v4's own crude fixed-window anchor average. An
  Elliott rule-violation event computed from the same daily-resampled
  OHLCV window is, like CSD/TDA, a geometry-of-recent-price-action
  statistic, and there is no a priori reason to expect it escapes the
  same shock-suddenness problem. **Falsification, pre-registered: fewer
  than 4/6 episodes pass the identical Step-A gate defined below -- STOP,
  no Step-B, report NEGATIVE.** The bar is not relaxed after seeing the
  numbers.

=====================================================================
THE CONSTRUCTION (shared by both branches, duplicated nowhere else)
=====================================================================

1. **Causal percentage ZigZag.** A single forward pass over the price
   series. State: the current swing direction ('up' search-for-high or
   'down' search-for-low), the running extreme price/index since the last
   CONFIRMED pivot. A pivot is CONFIRMED -- and only then written to the
   output, at the CONFIRMING bar's own index, never backdated to the
   extreme's own earlier index -- the moment price reverses from the
   running extreme by more than `pct`. This is causal by construction:
   the array value at row i depends only on prices[<=i], and
   `truncation_causality_probe` (below, identical contract to every prior
   round's) verifies it mechanically. No repainting: once written, a
   pivot's type/price/index never change.

2. **Elliott impulse rule-checking (Frost & Prechter 1978, canonical
   three hard rules for a 5-wave bull impulse; a long-only simplification
   -- bear/short counting and diagonal-triangle exceptions are explicitly
   OUT of scope, disclosed, not proxied):** pivots alternate low/high by
   ZigZag construction. Track the current candidate impulse anchored at
   the most recent "start" pivot P0 (a low) with subsequent confirmed
   pivots P1 (high, wave 1 top) .. P5 (high, wave 5 top):
   - Rule 1 (wave 2 retrace): at P2 (a low), invalid if `P2 <= P0`
     (wave 2 fully retraces wave 1). `require_fib_band=True` additionally
     requires the retracement `(P1-P2)/(P1-P0)` to land in Elliott
     International's canonical [0.382, 0.786] band (the literal,
     practitioner-standard Fibonacci reading B-10 asks for); with
     `require_fib_band=False` only the hard >100% rule applies -- this
     ablation directly tests Batchelor & Ramyar's claim that the specific
     Fibonacci ratio carries no information beyond the structural rule.
   - Rule 2 (wave 4 overlap): at P4 (a low), invalid if `P4 <= P1`
     (wave 4's low re-enters wave 1's price territory).
   - Rule 3 (wave 3 not shortest): at P5 (a high), invalid if the wave-3
     leg `P3-P2` is shorter than BOTH the wave-1 leg `P1-P0` and the
     wave-5 leg `P5-P4`.
   Any rule violation INVALIDATES the count: the violating pivot becomes
   the new candidate P0 and the search restarts. A clean, un-violated
   completion of P5 also starts a fresh search from P5 (P5 becomes the
   next P0) -- expected end-of-cycle, not an invalidation.

3. **Two outputs, one engine (`run_wave_engine`):**
   - `long_signal[i]` (bool): True from the bar wave 2 CONFIRMS VALID
     (passing rule 1, and the Fibonacci band if `require_fib_band`) --
     anticipating wave 3, canonically the longest/strongest impulse leg --
     until wave 5 confirms OR an invalidation fires, whichever is first.
     Feeds the CONSERVATIVE branch's `on_bar` target directly.
   - `invalidated[i]` (bool): True on bars where rule 1, 2 or 3 fires.
     Feeds the NOVEL branch's regime-timing signal: the count of days
     since the most recent invalidation, in the identical run-length
     shape `nearest_bocpd_detection`-style gate helpers already consume
     (`cusum_run_length_daily` in R-139, `tda_run_length` in R-155).

=====================================================================
BOTH BRANCHES, PARAMETER FREEDOM (fixed here, before any number)
=====================================================================

**CONSERVATIVE** (one configuration, no sweep -- B-10's own "no
discretion" brief and this project's established conservative-branch
convention, e.g. R-155's single fixed config): `pct=0.05` (a round,
standard percentage-ZigZag threshold; not tuned, not swept),
`require_fib_band=True` (the literal practitioner reading -- "ZigZag
pivots... Fibonacci" as B-10 names the item), operating on raw 5-minute
closes (bar-native, needed to place real orders). Long-only (spot and
futures both use `target in {0, 1}`; no short/bear-impulse counting --
disclosed simplification, not proxied data). **1 configuration.**

**NOVEL** (pre-registered 3x2 grid, fixed here before any real-data
number is computed): `pct in {0.03, 0.05, 0.08}` x
`require_fib_band in {True, False}`, operating on DAILY-resampled closes
(matching v4's own anchor horizon, identical reasoning to every Step-A
predecessor). The Fibonacci-band ablation is the scientifically
motivated half of the grid: it directly tests, on this project's own
data, whether Batchelor & Ramyar's "no scientific rationale" finding
extends from retracement PRICE LEVELS (their subject) to retracement-rule
STRUCTURE (this round's). `trail`/alarm machinery matches every
predecessor exactly: `k_short_days=K_SHORT_DAYS` (5, this project's
established "a break likely just happened" horizon). **6 configurations.**
Total configurations evaluated this round: **1 + 6 = 7** (deflated-Sharpe
trials count, ROUTINE.md's parallelism rule: total across both branches).

Both branches stop at their own gate before any holdout read: NOVEL stops
at Step A if the gate does not clear >=4/6 (per every Step-A predecessor,
a Step-A failure is the round's whole result for that branch -- no Step-B
vote-combination, no holdout). CONSERVATIVE follows the standard
ROUTINE.md Step 3/4 procedure (inner-train/inner-validation sanity check,
freeze the promotion bar, THEN read the holdout once) since it is a
standalone directional strategy, not a Step-A-gated regime input. Neither
branch may read any bar at or after `OOS_START` before its own gate/freeze
point.
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
    anchor_majority,
    block_bootstrap_shifts,
    episode_window,
    nearest_bocpd_detection as nearest_run_length_detection,  # generic on any run-length series
    nearest_transition,
)

WINDOW_DAYS_GATE = 60
N_DRAWS = 500
BLOCK_DAYS = 5
NULL_SEED = 156

# Conservative branch: one fixed configuration.
CONS_PCT = 0.05
CONS_FIB = True

# Novel branch: pre-registered 3x2 sweep grid, fixed before any real-data
# number was computed.
NOVEL_PCT_GRID = (0.03, 0.05, 0.08)
NOVEL_FIB_GRID = (True, False)

FIB_LO, FIB_HI = 0.382, 0.786  # canonical wave-2 retracement band


# --------------------------------------------------------- the wave engine


def causal_zigzag_pivots(prices: np.ndarray, pct: float) -> list[tuple[int, int, float, str]]:
    """Standard causal percentage ZigZag: one forward pass. Returns a list
    of confirmed pivots `(extreme_idx, confirm_idx, price, kind)`, `kind`
    in {'H', 'L'}, in confirmation order. `extreme_idx <= confirm_idx`
    always: a pivot's PRICE is set at the bar where the extreme actually
    occurred, but it only enters this list -- and may only affect any
    bar's output -- at `confirm_idx`, the bar where price has moved `pct`
    away from it. Never revised after being appended (no repainting)."""
    n = len(prices)
    pivots: list[tuple[int, int, float, str]] = []
    if n < 2:
        return pivots

    hi, hi_idx = prices[0], 0
    lo, lo_idx = prices[0], 0
    mode: str | None = None  # None = undetermined, 'up' = tracking toward a high,
    # 'down' = tracking toward a low

    for i in range(1, n):
        p = prices[i]
        if mode is None:
            if p > hi:
                hi, hi_idx = p, i
            if p < lo:
                lo, lo_idx = p, i
            if p <= hi * (1.0 - pct) and hi_idx < i:
                pivots.append((hi_idx, i, hi, "H"))
                mode = "down"
                lo, lo_idx = p, i
            elif p >= lo * (1.0 + pct) and lo_idx < i:
                pivots.append((lo_idx, i, lo, "L"))
                mode = "up"
                hi, hi_idx = p, i
        elif mode == "up":
            if p > hi:
                hi, hi_idx = p, i
            elif p <= hi * (1.0 - pct):
                pivots.append((hi_idx, i, hi, "H"))
                mode = "down"
                lo, lo_idx = p, i
        else:  # mode == "down"
            if p < lo:
                lo, lo_idx = p, i
            elif p >= lo * (1.0 + pct):
                pivots.append((lo_idx, i, lo, "L"))
                mode = "up"
                hi, hi_idx = p, i
    return pivots


def run_wave_engine(prices: np.ndarray, pct: float, require_fib_band: bool) -> dict:
    """Two-phase, both causal: (1) `causal_zigzag_pivots` finds confirmed
    swing pivots; (2) a single pass over that pivot stream applies Frost &
    Prechter's three hard impulse rules, anchoring a fresh candidate P0 at
    every confirmed LOW (long-only, bear-leg counting out of scope).
    Returns dict of length-`len(prices)` arrays: `long_signal` (bool,
    forward-filled from the pivot-confirmation bar where each state
    change occurs), `invalidated` (bool, True only ON the bar a rule
    fires), `pivot_type` (+1 high confirmed this bar, -1 low, 0
    otherwise, diagnostics only). Every write lands at a pivot's
    `confirm_idx`, never its (earlier) `extreme_idx` -- so the whole
    engine depends on `prices[<=i]` at every row i, by construction."""
    n = len(prices)
    invalidated = np.zeros(n, dtype=bool)
    pivot_type = np.zeros(n, dtype=np.int8)
    if n < 2:
        return dict(long_signal=np.zeros(n, dtype=bool), invalidated=invalidated,
                     pivot_type=pivot_type)

    pivots = causal_zigzag_pivots(prices, pct)
    for _extreme_idx, confirm_idx, _price, kind in pivots:
        pivot_type[confirm_idx] = 1 if kind == "H" else -1

    long_events: list[tuple[int, bool]] = []  # (bar_idx, new_state), in order
    window: list[float] = []  # confirmed prices since the current candidate P0

    for _extreme_idx, confirm_idx, price, kind in pivots:
        if not window:
            if kind == "L":
                window = [price]
            continue  # a lone H before any L anchors nothing (bear leg, out of scope)
        window.append(price)
        stage = len(window) - 1  # 1 = P1 just appended, 2 = P2, ...
        if stage == 2:
            p0, p1, p2 = window[0], window[1], window[2]
            retrace = (p1 - p2) / (p1 - p0) if (p1 - p0) != 0 else float("inf")
            hard_bad = p2 <= p0
            fib_bad = require_fib_band and not (FIB_LO <= retrace <= FIB_HI)
            if hard_bad or fib_bad:
                invalidated[confirm_idx] = True
                long_events.append((confirm_idx, False))
                window = [price]  # P2 (a low) becomes the new candidate P0
            else:
                long_events.append((confirm_idx, True))
        elif stage == 4:
            p1, p4 = window[1], window[4]
            if p4 <= p1:
                invalidated[confirm_idx] = True
                long_events.append((confirm_idx, False))
                window = [price]  # P4 (a low) becomes the new candidate P0
        elif stage == 5:
            p0, p1, p2, p3, p4, p5 = window[:6]
            len1, len3, len5 = p1 - p0, p3 - p2, p5 - p4
            if len3 < len1 or len3 < len5:
                invalidated[confirm_idx] = True
            long_events.append((confirm_idx, False))
            window = []  # fresh search; the NEXT confirmed low becomes the new P0
        # stage 1 or 3 (P1 or P3 just appended): nothing to check or emit yet.

    long_signal = np.zeros(n, dtype=bool)
    state = False
    ei = 0
    for i in range(n):
        while ei < len(long_events) and long_events[ei][0] == i:
            state = long_events[ei][1]
            ei += 1
        long_signal[i] = state

    return dict(long_signal=long_signal, invalidated=invalidated, pivot_type=pivot_type)

    return dict(long_signal=long_signal, invalidated=invalidated, pivot_type=pivot_type)


# ---------------------------------------------------- conservative outputs


def conservative_target_series(df: pd.DataFrame, pct: float = CONS_PCT,
                                require_fib_band: bool = CONS_FIB) -> np.ndarray:
    """0/1 causal target-position array on `df`'s own (5-minute) index."""
    out = run_wave_engine(df["close"].to_numpy(), pct, require_fib_band)
    return out["long_signal"].astype(float)


# --------------------------------------------------------------- novel gate


def invalidation_run_length_daily(daily_close: pd.Series, pct: float,
                                   require_fib_band: bool) -> pd.Series:
    out = run_wave_engine(daily_close.to_numpy(), pct, require_fib_band)
    invalidated = out["invalidated"]
    n = len(invalidated)
    run_length = np.zeros(n, dtype=float)
    since = float(K_SHORT_DAYS) * 4.0  # warmup state before any invalidation observed
    for i in range(n):
        if invalidated[i]:
            since = 0.0
        else:
            since = since + 1.0 if i > 0 else since
        run_length[i] = since
    return pd.Series(run_length, index=daily_close.index, name="wave_run_length")


def wave_daily_causal_signal(df: pd.DataFrame, pct: float, require_fib_band: bool) -> pd.DataFrame:
    daily_close = df["close"].resample("1D").last().dropna()
    run_length = invalidation_run_length_daily(daily_close, pct, require_fib_band)
    daily = pd.DataFrame({"wave_run_length": run_length})
    return align_onchain_causal(daily, df)


def assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")


def step_a_gate(bars: pd.DataFrame, *, pct: float, require_fib_band: bool,
                 k_short_days: int = K_SHORT_DAYS, window_days_gate: int = WINDOW_DAYS_GATE,
                 n_draws: int = N_DRAWS, block_days: int = BLOCK_DAYS, seed: int = NULL_SEED,
                 verbose: bool = True) -> dict:
    """R-82-identical Step-A detection-lag gate: the Elliott invalidation
    run length in place of BOCPD's MAP run length / TDA's alarm run
    length. An episode PASSES if (a) the invalidation alarm detects at or
    before v4's own nearest downward anchor-flip, AND (b) that lead beats
    the block-bootstrap null's median. Gate PASSES overall at >=4/6."""
    majority = anchor_majority(bars)
    wave = wave_daily_causal_signal(bars, pct=pct, require_fib_band=require_fib_band)
    assert_no_holdout(wave)
    run_length = wave["wave_run_length"]

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
    return dict(pct=pct, require_fib_band=require_fib_band, results=results,
                n_pass=n_pass, passed=n_pass >= 4)


def plateau_ok(grid_results: list[dict], winner: dict, min_neighbor_pass: int = 4) -> bool:
    """Pre-registered plateau check for the NOVEL branch's grid: at least
    one immediate neighbour in the `pct` dimension (holding `fib` fixed at
    the winner's own value) must also clear `min_neighbor_pass`."""
    grid = sorted(NOVEL_PCT_GRID)
    by_key = {(r["pct"], r["require_fib_band"]): r for r in grid_results}
    w_pct, w_fib = winner["pct"], winner["require_fib_band"]
    i = grid.index(w_pct)
    neighbours = [grid[i - 1]] if i > 0 else []
    if i < len(grid) - 1:
        neighbours.append(grid[i + 1])
    if not neighbours:
        return True
    return any(by_key.get((v, w_fib), {}).get("n_pass", 0) >= min_neighbor_pass for v in neighbours)


def truncation_causality_probe(build_target_fn, df: pd.DataFrame,
                                check_at: int, shorter_by: int = 20_000) -> bool:
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))
