"""Bar-level microstructure estimators.

These are the observables the strategy's players reason about: how much of the
flow is informed, how expensive it is to move price, and where the market
maker's inventory probably sits.  Every estimator is causal and uses only
columns present in the canonical bar schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rolling import ewma, rolling_beta, rolling_mean, rolling_rank, rolling_sum, shift, zscore


@dataclass(frozen=True)
class MicroConfig:
    kyle_window: int = 288  # one day of 5m bars
    vpin_buckets: int = 50
    flow_halflife: float = 6.0
    # The maker hedges fast, so the inventory displacement it creates unwinds
    # over roughly one bar.  An 18-bar half-life measures a different, much
    # weaker quantity — getting this timescale right is worth more than any
    # amount of signal blending.
    inventory_halflife: float = 0.5
    slow_inventory_halflife: float = 6.0
    roll_window: int = 288
    rank_window: int = 2016  # one week


def order_flow(volume: np.ndarray, taker_buy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed volume and normalised order-flow imbalance.

    ``signed`` is in base units (BTC); ``imbalance`` is in [-1, 1] and is the
    quantity most comparable across volatility regimes.
    """
    signed = 2.0 * taker_buy - volume
    imbalance = signed / np.maximum(volume, 1e-12)
    return signed, np.clip(imbalance, -1.0, 1.0)


def kyle_lambda(log_ret: np.ndarray, signed_volume: np.ndarray, window: int) -> np.ndarray:
    """Rolling estimate of Kyle's lambda: price impact per unit of signed flow.

    Fitted through the origin over a trailing window.  Because ``log_ret[t]``
    and ``signed_volume[t]`` are contemporaneous, the estimate at ``t`` uses
    bar ``t`` itself — which is legitimate: both are known once bar ``t`` has
    closed, and the decision it feeds is executed at ``t+1``.
    """
    lam = rolling_beta(log_ret, signed_volume, window)
    return np.maximum(lam, 0.0)


def impact_residual(
    log_ret: np.ndarray, signed_volume: np.ndarray, lam: np.ndarray, atr: np.ndarray
) -> np.ndarray:
    """Move not explained by contemporaneous order flow, in ATR units.

    A large positive residual means price rose far more than the flow that
    arrived can justify — the hallmark of a thin-book dislocation rather than
    genuine repricing, and therefore a reversion candidate.
    """
    explained = lam * signed_volume
    return (log_ret - explained) / np.maximum(atr, 1e-12)


def vpin(signed_volume: np.ndarray, volume: np.ndarray, buckets: int) -> np.ndarray:
    """Volume-synchronised probability of informed trading.

    The canonical VPIN uses equal-volume buckets; at bar resolution we use a
    trailing window of ``buckets`` bars, which is the standard bar-level
    approximation.  High VPIN means the flow is one-sided, i.e. the market
    maker is being adversely selected and continuation is more likely than
    reversion.
    """
    num = rolling_sum(np.abs(signed_volume), buckets)
    den = rolling_sum(volume, buckets)
    return np.clip(num / np.maximum(den, 1e-12), 0.0, 1.0)


def flow_persistence(imbalance: np.ndarray, window: int) -> np.ndarray:
    """Trailing autocorrelation of order-flow imbalance.

    Persistent same-signed flow is the observable signature of a Hawkes-
    clustered informed trader working a large order over many bars.
    """
    prev = shift(imbalance, 1, fill=0.0)
    m = rolling_mean(imbalance, window)
    cov = rolling_mean((imbalance - m) * (prev - m), window)
    var = np.maximum(rolling_mean((imbalance - m) ** 2, window), 1e-12)
    return np.clip(cov / var, -1.0, 1.0)


def mm_inventory_proxy(signed_volume: np.ndarray, halflife: float) -> np.ndarray:
    """Proxy for the market maker's inventory.

    The maker is the counterparty to aggressive flow, so its inventory is minus
    the cumulative taker flow, decayed at the rate it hedges.  Sign convention:
    positive means the maker is *long* and therefore wants to sell, which is
    downward pressure on quotes.
    """
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    # EWMA of -signed flow scaled back up to "accumulated units".
    return -ewma(signed_volume, halflife) / max(alpha, 1e-9)


def amihud_illiquidity(log_ret: np.ndarray, quote_volume: np.ndarray, window: int) -> np.ndarray:
    """Amihud's |return| per unit of traded value, smoothed."""
    raw = np.abs(log_ret) / np.maximum(quote_volume, 1e-9)
    return rolling_mean(raw, window)


def roll_spread(log_ret: np.ndarray, window: int) -> np.ndarray:
    """Roll's effective-spread estimator, ``2*sqrt(-cov(r_t, r_{t-1}))``.

    Where the serial covariance is non-negative the estimator is undefined; we
    return zero there, which is the conventional treatment.
    """
    prev = shift(log_ret, 1, fill=0.0)
    m = rolling_mean(log_ret, window)
    cov = rolling_mean((log_ret - m) * (prev - m), window)
    return 2.0 * np.sqrt(np.maximum(-cov, 0.0))


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range in log terms, so it is directly comparable to log returns."""
    prev_close = shift(close, 1, fill=np.nan)
    prev_close[0] = close[0]
    hi = np.maximum(high, prev_close)
    lo = np.minimum(low, prev_close)
    return np.log(np.maximum(hi, 1e-12) / np.maximum(lo, 1e-12))


def compute(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    quote_volume: np.ndarray,
    taker_buy: np.ndarray,
    cfg: MicroConfig = MicroConfig(),
) -> dict[str, np.ndarray]:
    """Compute the full microstructure feature block."""
    log_close = np.log(np.maximum(close, 1e-12))
    log_ret = np.diff(log_close, prepend=log_close[0])

    tr = true_range(high, low, close)
    atr = ewma(tr, 48.0)
    rv = np.sqrt(ewma(log_ret**2, 48.0))

    signed, imbalance = order_flow(volume, taker_buy)
    lam = kyle_lambda(log_ret, signed, cfg.kyle_window)

    feats = {
        "log_ret": log_ret,
        "true_range": tr,
        "atr": atr,
        "realized_vol": rv,
        "vol_rank": rolling_rank(rv, cfg.rank_window),
        "signed_volume": signed,
        "ofi": imbalance,
        "ofi_ewma": ewma(imbalance, cfg.flow_halflife),
        "ofi_persistence": flow_persistence(imbalance, cfg.kyle_window),
        "kyle_lambda": lam,
        "kyle_lambda_rank": rolling_rank(lam, cfg.rank_window),
        "impact_residual": impact_residual(log_ret, signed, lam, atr),
        "vpin": vpin(signed, volume, cfg.vpin_buckets),
        "mm_inventory": mm_inventory_proxy(signed, cfg.inventory_halflife),
        "mm_inventory_slow": mm_inventory_proxy(signed, cfg.slow_inventory_halflife),
        "amihud": amihud_illiquidity(log_ret, quote_volume, cfg.kyle_window),
        "roll_spread": roll_spread(log_ret, cfg.roll_window),
        "volume_ratio": volume / np.maximum(rolling_mean(volume, 48), 1e-12),
    }
    feats["vpin_rank"] = rolling_rank(feats["vpin"], cfg.rank_window)
    feats["mm_inventory_z"] = zscore(feats["mm_inventory"], cfg.rank_window, clip=6.0)
    feats["mm_inventory_slow_z"] = zscore(feats["mm_inventory_slow"], cfg.rank_window, clip=6.0)
    return feats
