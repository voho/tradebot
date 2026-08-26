"""Deterministic, causal ZigZag/Fibonacci Elliott Wave counter, traded as a directional signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.registry import register
from tradebot.strategy import Context, Strategy


def _gauss(ratio: float, target: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((ratio - target) / sigma) ** 2))


def _zigzag_pivots(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    atr: np.ndarray, k: float) -> list[tuple[int, int, float, int]]:
    """Causal ZigZag: one forward pass, pivots frozen at confirmation.

    Returns ``(confirm_idx, extreme_idx, price, kind)`` tuples in increasing
    ``confirm_idx`` order (``kind`` +1 = swing high, -1 = swing low). A
    pivot's price and index are fixed the instant price reverses from the
    running extreme by ``k * ATR / price``; nothing later ever revises an
    already-confirmed pivot (verified by a truncation check — see R-157).
    """
    n = len(close)
    ready = np.isfinite(atr)
    if not ready.any():
        return []
    start = int(np.argmax(ready))

    ext_price = float(high[start])
    ext_idx = start
    trend = 1  # 1 = extending a swing high, -1 = extending a swing low

    pivots: list[tuple[int, int, float, int]] = []

    for i in range(start + 1, n):
        a = atr[i]
        if not np.isfinite(a) or close[i] <= 0.0:
            continue
        thresh = k * a / close[i]
        hi = float(high[i])
        lo = float(low[i])

        if trend == 1:
            if hi >= ext_price:
                ext_price, ext_idx = hi, i
                continue
            retr = (ext_price - lo) / ext_price
            if retr >= thresh:
                pivots.append((i, ext_idx, ext_price, 1))
                trend = -1
                ext_price, ext_idx = lo, i
        else:
            if lo <= ext_price:
                ext_price, ext_idx = lo, i
                continue
            retr = (hi - ext_price) / ext_price
            if retr >= thresh:
                pivots.append((i, ext_idx, ext_price, -1))
                trend = 1
                ext_price, ext_idx = hi, i

    return pivots


def _check_impulse(window: list[tuple[int, int, float, int]], sigma: float):
    """HARD-gate a 6-pivot window as a 5-wave impulse; return (direction, confidence) or None."""
    p0, p1, p2, p3, p4, p5 = window
    direction = -1 if p0[3] == 1 else 1  # p0 kind H -> down-impulse, L -> up-impulse
    P0, P1, P2, P3, P4, P5 = (p[2] for p in (p0, p1, p2, p3, p4, p5))

    len1, len2, len3, len4, len5 = (abs(P1 - P0), abs(P2 - P1), abs(P3 - P2),
                                     abs(P4 - P3), abs(P5 - P4))
    if len1 <= 0.0 or len3 <= 0.0 or len5 <= 0.0:
        return None

    retrace2 = len2 / len1
    if not (0.382 <= retrace2 < 1.0):
        return None
    if len3 < len1 and len3 < len5:  # wave 3 is the shortest of 1/3/5
        return None
    if direction == 1 and not (P4 > P1):   # wave 4 overlaps wave 1's territory
        return None
    if direction == -1 and not (P4 < P1):
        return None

    score3 = _gauss(len3 / len1, 1.618, sigma)
    score5 = max(_gauss(len5 / len1, 0.618, sigma), _gauss(len5 / len1, 1.0, sigma))
    return direction, 0.5 * (score3 + score5)


def _check_abc(window: list[tuple[int, int, float, int]], sigma: float):
    """HARD-gate a 4-pivot window as an A-B-C correction; return confidence or None."""
    p0, p1, p2, p3 = window
    P0, P1, P2, P3 = (p[2] for p in (p0, p1, p2, p3))
    lenA, lenB, lenC = abs(P1 - P0), abs(P2 - P1), abs(P3 - P2)
    if lenA <= 0.0:
        return None
    retraceB = lenB / lenA
    if not (0.382 <= retraceB <= 0.786):
        return None
    ratioC = lenC / lenA
    return max(_gauss(ratioC, 1.0, sigma), _gauss(ratioC, 1.618, sigma))


def _wave_events(pivots: list[tuple[int, int, float, int]], sigma: float,
                  conf_threshold: float) -> list[tuple[int, float]]:
    """Scan the confirmed pivot list and return (confirm_idx, target) events.

    Every trailing 6-window is checked for an impulse and every trailing
    4-window for an ABC, at each new pivot; an impulse takes priority over
    an ABC sharing the same terminal pivot (a fixed, documented tie-break,
    not a per-case judgment call). Only windows clearing both the hard
    gates and the confidence gate produce an event.
    """
    events: list[tuple[int, float]] = []
    for j in range(len(pivots)):
        fired = False
        if j >= 5:
            res = _check_impulse(pivots[j - 5:j + 1], sigma)
            if res is not None:
                direction, confidence = res
                if confidence >= conf_threshold:
                    events.append((pivots[j][0], float(direction)))
                    fired = True
        if not fired and j >= 3:
            confidence = _check_abc(pivots[j - 3:j + 1], sigma)
            if confidence is not None and confidence >= conf_threshold:
                events.append((pivots[j][0], 0.0))
    return events


@register
class ElliottWave(Strategy):
    """Deterministic, causal ZigZag/Fibonacci Elliott Wave counter, traded directionally (B-10).

    Backlog item B-10, R-18's literature rejection converted into a run:
    classical Elliott Wave counting is unfalsifiable as practiced (Aronson
    2006 — counts are relabeled after the fact once they fail), and its one
    quantitative claim, Fibonacci retracement ratios, was empirically
    refuted (Batchelor & Ramyar, "Magic numbers in the Dow"). This is the
    falsifiable version B-10 asks for: a deterministic, causal, ZigZag-pivot
    impulse/corrective counter with no discretion, no relabeling, evaluated
    against this project's own holdout like every other strategy here.

    Mechanism: a Wilder-style ATR feeds a causal ZigZag (a pivot is
    permanently frozen the instant price reverses `k * ATR` from the
    running extreme since the last confirmed pivot — never repainted).
    Every trailing 6-pivot window is HARD-gated as a 5-wave impulse (wave 2
    retraces 38.2-100% of wave 1 without exceeding wave 0's start; wave 3 is
    never the shortest of 1/3/5; wave 4 does not enter wave 1's territory —
    the diagonal-triangle exception is deliberately ignored, per B-10's "no
    discretion") and every trailing 4-pivot window as an A-B-C correction
    (B retraces 38.2-78.6% of A). A pattern clearing its hard gates gets a
    soft Fibonacci-ratio confidence score (Gaussian kernel around the
    canonical 1.618/0.618/1.0 ratios); only a completed pattern with
    confidence >= `conf_threshold` moves the target: impulse up -> long,
    impulse down -> short (clamped flat on spot by the broker automatically),
    ABC either direction -> flatten (a correction says the larger trend has
    resumed, not a new directional call). No Kelly sizing, no partial
    positions — flat/long/short at fixed notional, per B-10's scope.

    R-157 (08-26): NEGATIVE. The frozen configuration lost to buy_and_hold
    on both spot (-65.0% vs +283.9%) and futures_5x (-99.6% vs +1417.6%,
    funding-free upper bound) on the 2023+ holdout, and the same failure
    magnitude replicated almost exactly on ETH (-66.8% spot, -99.9%
    futures) — ruling out "unlucky on BTC" as an explanation. Registered
    despite the negative verdict, per B-10's own framing ("converts an
    unfalsifiable debate into a table row") and this project's convention
    for instructive negatives (`minority_oracle`, `game_switch`): turnover
    stayed high (543-1,065 trades in the holdout alone) despite the ATR
    threshold and confidence gate, so most of the loss is fee bleed from
    frequent small round-trips — the same COST-scales-with-signal failure
    mode as every other directional predictor in this ledger, now measured
    rather than merely predicted from the literature.
    """

    name = "elliott_wave"
    warmup = 3000  # comfortably past atr_n's ready point plus room for the first few pivots

    def __init__(self, atr_n: int = 50, k: float = 3.0, sigma: float = 0.5,
                 conf_threshold: float = 0.4) -> None:
        self.atr_n = atr_n
        self.k = k
        self.sigma = sigma
        self.conf_threshold = conf_threshold

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / self.atr_n, min_periods=self.atr_n).mean().to_numpy()

        pivots = _zigzag_pivots(high, low, close, atr, self.k)
        events = _wave_events(pivots, self.sigma, self.conf_threshold)

        n = len(df)
        raw = np.full(n, np.nan)
        for idx, decision in events:
            raw[idx] = decision
        target = pd.Series(raw, index=df.index).ffill().fillna(0.0).to_numpy()

        df["target"] = target
        return df

    def on_bar(self, ctx: Context) -> None:
        t = float(ctx.bar["target"])
        prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
        if abs(t - prev) > 1e-9:
            ctx.order_target(t)
