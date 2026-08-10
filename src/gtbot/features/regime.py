"""Regime classification.

The meta-learner in :mod:`gtbot.game.regret` is *contextual*: it keeps a
separate weight vector per regime cell, because which player is right depends
on the state of the market.  Trend-following works when informed flow dominates;
fading works when it does not.  Regimes here are deliberately coarse and
computed from robust trailing ranks, so cell membership is stable and each cell
accumulates enough observations to learn from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rolling import ewma, rolling_mean, rolling_rank, rolling_sum


@dataclass(frozen=True)
class RegimeConfig:
    vr_window: int = 288
    vr_lag: int = 6
    trend_fast: float = 12.0
    trend_slow: float = 96.0
    rank_window: int = 2016
    # Context cells learn independently, so every extra cell divides the
    # evidence.  At an information coefficient near 0.03 a cell needs O(10k)
    # non-overlapping observations before its weights mean anything; at 5m bars
    # that is over a year of data *per cell*.  The default is therefore a
    # single global cell, and the contextual machinery is left switched on for
    # deployments with enough history to support it.  Raising this without the
    # data to back it makes the high-volatility cell — where the tradeable
    # dislocations actually live — learn its sign from noise.
    n_vol_states: int = 1
    n_flow_states: int = 1


def variance_ratio(log_ret: np.ndarray, lag: int, window: int) -> np.ndarray:
    """Lo-MacKinlay variance ratio over a trailing window.

    ``VR > 1`` means multi-bar moves are larger than the sum of their parts —
    trending.  ``VR < 1`` means they cancel — mean reverting.  This is the
    single most useful summary of which side of the game we are playing.
    """
    agg = rolling_sum(log_ret, lag)  # trailing k-bar return, causal by construction
    var1 = np.maximum(rolling_mean(log_ret**2, window), 1e-18)
    var_k = np.maximum(rolling_mean(agg**2, window), 1e-18)
    return np.clip(var_k / (lag * var1), 0.0, 5.0)


def trend_strength(log_close: np.ndarray, fast: float, slow: float, atr: np.ndarray) -> np.ndarray:
    """Signed EMA spread normalised by ATR."""
    ef = ewma(log_close, fast)
    es = ewma(log_close, slow)
    return np.clip((ef - es) / np.maximum(atr, 1e-12), -20.0, 20.0)


def session_bucket(ts_ms: np.ndarray) -> np.ndarray:
    """UTC session index: 0 = Asia, 1 = Europe, 2 = US, 3 = late US/overnight."""
    hour = (ts_ms // 3_600_000) % 24
    return np.select(
        [hour < 7, hour < 13, hour < 20],
        [0.0, 1.0, 2.0],
        default=3.0,
    )


def compute(
    ts_ms: np.ndarray,
    log_close: np.ndarray,
    log_ret: np.ndarray,
    atr: np.ndarray,
    vpin_rank: np.ndarray,
    vol_rank: np.ndarray,
    cfg: RegimeConfig = RegimeConfig(),
) -> dict[str, np.ndarray]:
    vr = variance_ratio(log_ret, cfg.vr_lag, cfg.vr_window)
    ts_strength = trend_strength(log_close, cfg.trend_fast, cfg.trend_slow, atr)

    vol_state = np.floor(np.clip(vol_rank, 0.0, 0.999) * cfg.n_vol_states)
    flow_state = np.floor(np.clip(vpin_rank, 0.0, 0.999) * cfg.n_flow_states)
    # Composite context cell used by the contextual no-regret learner.
    cell = (vol_state * cfg.n_flow_states + flow_state).astype(int)

    return {
        "variance_ratio": vr,
        "vr_rank": rolling_rank(vr, cfg.rank_window),
        "trend_strength": ts_strength,
        "session": session_bucket(ts_ms),
        "vol_state": vol_state,
        "flow_state": flow_state,
        "regime_cell": cell.astype(float),
    }


def n_cells(cfg: RegimeConfig = RegimeConfig()) -> int:
    return cfg.n_vol_states * cfg.n_flow_states
