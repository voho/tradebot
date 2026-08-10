"""Liquidity-pool geometry and sweep detection.

Resting stop and liquidation orders cluster just beyond recent swing extremes.
When price trades through such a cluster the resulting cascade is *uninformed*
flow, so its price impact is transient by construction — the market maker who
absorbed it wants out of the inventory.  Detecting the cascade in real time and
distinguishing it from a genuine informed breakout is the core inference problem
this module supports.

A sweep is defined as: the bar traded beyond a prior swing extreme, and closed
back inside it.  The close-back-inside condition is what separates a failed
break (liquidity grab) from a real one, and it is known at bar close, so the
signal is available for execution on the next bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rolling import rolling_max, rolling_mean, rolling_min, rolling_rank, shift


@dataclass(frozen=True)
class LiquidityConfig:
    swing_lookback: int = 36  # bars used to define the pool level
    rank_window: int = 2016
    vol_ma: int = 48


def swing_levels(high: np.ndarray, low: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Prior swing high/low, excluding the current bar.

    Excluding the current bar matters: a level that includes today's own high
    can never be swept by today's own high, and the feature would be vacuous.
    """
    prior_high = shift(rolling_max(high, lookback), 1, fill=np.nan)
    prior_low = shift(rolling_min(low, lookback), 1, fill=np.nan)
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    return prior_high, prior_low


def compute(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    cfg: LiquidityConfig = LiquidityConfig(),
) -> dict[str, np.ndarray]:
    prior_high, prior_low = swing_levels(high, low, cfg.swing_lookback)

    rng = np.maximum(high - low, 1e-12)
    upper_wick = (high - np.maximum(open_, close)) / rng
    lower_wick = (np.minimum(open_, close) - low) / rng
    body = np.abs(close - open_) / rng

    # --- sweep geometry -------------------------------------------------
    broke_high = high > prior_high
    broke_low = low < prior_low
    rejected_high = close < prior_high
    rejected_low = close > prior_low

    sweep_up = broke_high & rejected_high  # ran buy-stops, closed back below
    sweep_down = broke_low & rejected_low  # ran sell-stops, closed back above

    # Penetration depth beyond the pool, in ATR units: how far into the stop
    # cluster price actually traded.
    log_hi = np.log(np.maximum(high, 1e-12))
    log_lo = np.log(np.maximum(low, 1e-12))
    pen_up = (log_hi - np.log(np.maximum(prior_high, 1e-12))) / np.maximum(atr, 1e-12)
    pen_down = (np.log(np.maximum(prior_low, 1e-12)) - log_lo) / np.maximum(atr, 1e-12)
    penetration = np.where(sweep_up, pen_up, np.where(sweep_down, pen_down, 0.0))
    penetration = np.clip(np.nan_to_num(penetration), 0.0, 20.0)

    # --- cascade intensity ------------------------------------------------
    # Range expansion times volume expansion.  A stop cascade prints an
    # abnormally wide bar on abnormally high volume; ordinary drift does not.
    range_expansion = np.log(np.maximum(high, 1e-12) / np.maximum(low, 1e-12)) / np.maximum(atr, 1e-12)
    volume_expansion = volume / np.maximum(rolling_mean(volume, cfg.vol_ma), 1e-12)
    cascade = range_expansion * volume_expansion

    direction = np.where(sweep_up, -1.0, np.where(sweep_down, 1.0, 0.0))

    return {
        "prior_swing_high": prior_high,
        "prior_swing_low": prior_low,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "body_ratio": body,
        "sweep_up": sweep_up.astype(float),
        "sweep_down": sweep_down.astype(float),
        "sweep_dir": direction,  # +1 = fade upward, -1 = fade downward
        "sweep_penetration": penetration,
        "range_expansion": range_expansion,
        "volume_expansion": volume_expansion,
        "cascade_intensity": cascade,
        "cascade_rank": rolling_rank(cascade, cfg.rank_window),
        # Distance to the nearest untouched pool, in ATR units: how much room
        # there is before the next cluster of resting orders.
        "dist_to_high_pool": np.clip(
            (np.log(np.maximum(prior_high, 1e-12)) - np.log(np.maximum(close, 1e-12)))
            / np.maximum(atr, 1e-12),
            -20.0,
            20.0,
        ),
        "dist_to_low_pool": np.clip(
            (np.log(np.maximum(close, 1e-12)) - np.log(np.maximum(prior_low, 1e-12)))
            / np.maximum(atr, 1e-12),
            -20.0,
            20.0,
        ),
    }
