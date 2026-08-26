"""Mechanical ZigZag+Fibonacci Elliott Wave impulse counter, long-only."""

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy

FIB_LO, FIB_HI = 0.382, 0.786  # canonical wave-2 retracement band


def _causal_zigzag_pivots(prices: np.ndarray, pct: float) -> list[tuple[int, int, float, str]]:
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


def _run_wave_engine(prices: np.ndarray, pct: float, require_fib_band: bool) -> np.ndarray:
    """Two-phase, both causal: (1) `_causal_zigzag_pivots` finds confirmed
    swing pivots; (2) a single pass over that pivot stream applies Frost &
    Prechter's three hard impulse rules, anchoring a fresh candidate P0 at
    every confirmed LOW (long-only, bear-leg counting out of scope).
    Returns a length-`len(prices)` bool array, `long_signal`, forward-filled
    from the pivot-confirmation bar where each state change occurs. Every
    write lands at a pivot's `confirm_idx`, never its (earlier)
    `extreme_idx` -- so the whole engine depends on `prices[<=i]` at every
    row i, by construction (identical to `experiments/r156_shared.py`'s
    frozen `run_wave_engine`, duplicated here per this project's convention
    that registered strategies do not import from `experiments/`)."""
    n = len(prices)
    if n < 2:
        return np.zeros(n, dtype=bool)

    pivots = _causal_zigzag_pivots(prices, pct)

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
                long_events.append((confirm_idx, False))
                window = [price]  # P2 (a low) becomes the new candidate P0
            else:
                long_events.append((confirm_idx, True))
        elif stage == 4:
            p1, p4 = window[1], window[4]
            if p4 <= p1:
                long_events.append((confirm_idx, False))
                window = [price]  # P4 (a low) becomes the new candidate P0
        elif stage == 5:
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

    return long_signal


@register
class ElliottWaveZigzag(Strategy):
    """Mechanical ZigZag+Fibonacci Elliott Wave impulse counter (Frost & Prechter 1978); long from a valid wave-2 retracement to wave-5 completion or rule invalidation.

    Backlog B-10, filed as a documented negative result per R-18 (2026-08-16,
    "not falsifiable as practised" -- wave counts get re-labelled after the
    fact, exactly the leak class ``test_causality_strict.py`` exists to
    catch -- and its one quantitative component, the Fibonacci retracement
    ratio, refuted by Batchelor & Ramyar 2005: no significant difference
    between Fibonacci-ratio frequencies in the Dow and frequencies expected
    at random). Rather than debate any individual wave count, this builds
    the falsifiable, no-discretion version B-10 asks for and lets the
    comparison table carry the verdict.

    Mechanism: a single causal forward pass computes a standard percentage
    ZigZag (5% reversal threshold) over 5-minute closes; confirmed pivots
    alternate low/high by construction. Starting from each confirmed low as
    a candidate wave-0 (P0), track P1..P5 and apply Frost & Prechter's three
    hard impulse rules for a 5-wave bull count: wave 2 may not fully retrace
    wave 1 (and, per the Fibonacci reading this strategy tests literally,
    must retrace into the canonical [0.382, 0.786] band); wave 4 may not
    re-enter wave 1's price territory; wave 3 may not be the shortest of
    waves 1/3/5. Any violation invalidates the count and restarts the search
    from the violating pivot; a clean completion of wave 5 also restarts the
    search (expected end-of-cycle, not an invalidation). Long-only: enter at
    the bar wave 2 confirms valid (anticipating wave 3, canonically the
    strongest leg), exit at wave 5 completion or invalidation, whichever
    comes first. Bear-leg counting and diagonal-triangle exceptions are
    explicitly out of scope (disclosed simplification, not proxied). One
    frozen configuration (pct=0.05, Fibonacci band required) -- no
    parameter search, per B-10's own "no discretion" brief.
    """

    name = "elliott_wave_zigzag"
    warmup = 0  # no rolling-window lookback; the pivot state machine warms
    # up on its own as pivots confirm during the run.

    def __init__(self, pct: float = 0.05, require_fib_band: bool = True) -> None:
        self.pct = pct
        self.require_fib_band = require_fib_band

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        long_signal = _run_wave_engine(df["close"].to_numpy(), self.pct, self.require_fib_band)
        df["target"] = long_signal.astype(float)
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
