"""Feature construction for the BTCUSD 5m game-theoretic bot.

``build`` turns a canonical bar frame into a :class:`FeatureSet`: a dict of
named, strictly causal arrays aligned one-to-one with the input bars.

Causality contract
------------------
``features[name][t]`` may depend on bars ``0..t`` inclusive and on nothing
else.  The backtester consumes the feature row at ``t`` to decide a position
that is executed at the **open of bar t+1**, so using bar ``t``'s own close is
legitimate and is not lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import liquidity, microstructure, regime
from .rolling import ewma

__all__ = ["FeatureSet", "build", "FeatureConfig"]


@dataclass(frozen=True)
class FeatureConfig:
    micro: microstructure.MicroConfig = microstructure.MicroConfig()
    liq: liquidity.LiquidityConfig = liquidity.LiquidityConfig()
    reg: regime.RegimeConfig = regime.RegimeConfig()
    #: Bars to discard at the start while trailing windows fill up.
    warmup: int = 2016


@dataclass
class FeatureSet:
    """Aligned feature arrays plus the bar arrays they were derived from."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    data: dict[str, np.ndarray]
    warmup: int

    def __len__(self) -> int:
        return int(self.ts.size)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    @property
    def names(self) -> list[str]:
        return sorted(self.data)

    def row(self, t: int) -> dict[str, float]:
        return {k: float(v[t]) for k, v in self.data.items()}

    def to_frame(self) -> pd.DataFrame:
        base = {
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        return pd.DataFrame({**base, **self.data})


def build(bars: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> FeatureSet:
    """Build the full feature set from a canonical bar frame."""
    ts = bars["ts"].to_numpy(dtype="int64")
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    v = bars["volume"].to_numpy(dtype=float)
    qv = bars["quote_volume"].to_numpy(dtype=float)
    tb = bars["taker_buy_base"].to_numpy(dtype=float)

    feats: dict[str, np.ndarray] = {}
    feats.update(microstructure.compute(h, lo, c, v, qv, tb, cfg.micro))
    feats.update(liquidity.compute(o, h, lo, c, v, feats["atr"], cfg.liq))

    log_close = np.log(np.maximum(c, 1e-12))
    feats.update(
        regime.compute(
            ts,
            log_close,
            feats["log_ret"],
            feats["atr"],
            feats["vpin_rank"],
            feats["vol_rank"],
            cfg.reg,
        )
    )

    # A couple of cross-block composites that several players share.
    feats["sweep_score"] = feats["sweep_dir"] * feats["cascade_rank"] * np.tanh(
        feats["sweep_penetration"]
    )
    feats["flow_pressure"] = np.tanh(3.0 * feats["ofi_ewma"]) * feats["vpin_rank"]
    feats["overshoot"] = np.tanh(feats["impact_residual"])
    feats["ret_ewma_fast"] = ewma(feats["log_ret"], 6.0) / np.maximum(feats["atr"], 1e-12)

    for name, arr in feats.items():
        feats[name] = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    return FeatureSet(
        ts=ts, open=o, high=h, low=lo, close=c, volume=v, data=feats, warmup=cfg.warmup
    )
