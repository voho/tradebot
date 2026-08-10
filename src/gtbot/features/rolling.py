"""Causal rolling primitives.

Every function here maps an array ``x`` to an array ``y`` of the same length
where ``y[t]`` depends only on ``x[:t+1]``.  Nothing peeks forward.  This is the
single most important invariant in the codebase — ``tests/test_causality.py``
verifies it by perturbing the future and asserting the past is unchanged.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import deque

import numpy as np
from scipy.signal import lfilter


def ewma(x: np.ndarray, halflife: float, *, init: float | None = None) -> np.ndarray:
    """Exponentially weighted moving average with a half-life in samples."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    x = np.asarray(x, dtype=float)
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    seed = float(x[0]) if init is None else float(init)
    # lfilter's zi carries the recursion's initial condition, so the result is
    # identical to the scalar loop but runs at C speed.
    zi = np.array([(1.0 - alpha) * seed])
    out, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=zi)
    return out


def ewm_var(x: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted variance (biased, zero-lag against its own mean)."""
    m = ewma(x, halflife)
    dev = (x - m) ** 2
    return ewma(dev, halflife)


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum over ``window`` samples; partial windows at the start."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    idx = np.arange(x.size) + 1
    lo = np.maximum(idx - window, 0)
    return c[idx] - c[lo]


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    idx = np.arange(x.size) + 1
    n = np.minimum(idx, window)
    return rolling_sum(x, window) / n


def rolling_std(x: np.ndarray, window: int, *, ddof: int = 0) -> np.ndarray:
    idx = np.arange(x.size) + 1
    n = np.minimum(idx, window).astype(float)
    s1 = rolling_sum(x, window)
    s2 = rolling_sum(x * x, window)
    var = (s2 - s1 * s1 / n) / np.maximum(n - ddof, 1.0)
    return np.sqrt(np.maximum(var, 0.0))


def _rolling_extreme(x: np.ndarray, window: int, *, maximum: bool) -> np.ndarray:
    """O(n) monotonic-deque rolling max/min."""
    out = np.empty_like(x, dtype=float)
    dq: deque[int] = deque()
    for i in range(x.size):
        while dq and dq[0] <= i - window:
            dq.popleft()
        if maximum:
            while dq and x[dq[-1]] <= x[i]:
                dq.pop()
        else:
            while dq and x[dq[-1]] >= x[i]:
                dq.pop()
        dq.append(i)
        out[i] = x[dq[0]]
    return out


def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling_extreme(x, window, maximum=True)


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling_extreme(x, window, maximum=False)


def shift(x: np.ndarray, k: int = 1, fill: float = np.nan) -> np.ndarray:
    """Shift forward by ``k`` (``y[t] = x[t-k]``)."""
    out = np.full_like(x, fill, dtype=float)
    if k <= 0:
        raise ValueError("shift must be positive; the future is off limits")
    if k < x.size:
        out[k:] = x[:-k]
    return out


def rolling_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing percentile rank of the current value within its own window.

    Returns values in [0, 1].  Robust to the heavy tails that make z-scores
    unreliable on crypto data.
    """
    out = np.empty_like(x, dtype=float)
    window = max(int(window), 1)
    order: deque[float] = deque()  # insertion order, for eviction
    ordered: list[float] = []  # kept sorted, for O(log n) rank lookup
    for i in range(x.size):
        v = float(x[i])
        if len(order) == window:
            old = order.popleft()
            ordered.pop(bisect_left(ordered, old))
        insort(ordered, v)
        order.append(v)
        out[i] = bisect_right(ordered, v) / len(ordered)
    return out


def rolling_beta(y: np.ndarray, x: np.ndarray, window: int, *, ridge: float = 1e-12) -> np.ndarray:
    """Trailing OLS slope of ``y`` on ``x`` through the origin.

    Through the origin because every use here (price impact per unit of signed
    flow) has a structural zero: no flow, no impact.
    """
    sxx = rolling_sum(x * x, window)
    sxy = rolling_sum(x * y, window)
    return sxy / np.maximum(sxx, ridge)


def zscore(x: np.ndarray, window: int, *, clip: float = 8.0) -> np.ndarray:
    m = rolling_mean(x, window)
    s = rolling_std(x, window)
    z = (x - m) / np.maximum(s, 1e-12)
    return np.clip(z, -clip, clip)
