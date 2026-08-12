"""Data loading: canonical CSVs, with a clearly-labeled synthetic fallback.

Canonical files (committed to the repo when available):

- ``data/btcusdt_perp_5m.csv``          USDT-margined BTCUSDT perp, 5m klines
- ``data/btcusdt_spot_aligned_5m.csv``  BTCUSDT spot 5m, aligned to the perp index

Format: columns ``timestamp,open,high,low,close,volume`` with ``timestamp``
as milliseconds since epoch (UTC). ``python -m tradebot fetch`` produces them.

If a canonical file is missing, ``load_dataset`` falls back to a seeded
synthetic series (generated once into ``data/synthetic_*.csv``, gitignored)
so the framework stays runnable; every report is labeled with the source.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL = {
    "perp": "btcusdt_perp_5m.csv",
    "spot": "btcusdt_spot_aligned_5m.csv",
}
SYNTHETIC = {
    "perp": "synthetic_perp_5m.csv",
    "spot": "synthetic_spot_5m.csv",
}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV into a UTC-indexed DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = _epoch_unit(float(ts.iloc[0]))
        idx = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True, format="mixed")
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(idx, name="timestamp")
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def _epoch_unit(sample: float) -> str:
    """Guess epoch resolution by magnitude (seconds / ms / us)."""
    if sample > 1e14:
        return "us"
    if sample > 1e11:
        return "ms"
    return "s"


def save_ohlcv_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = df.copy()
    out.insert(0, "timestamp", (out.index.view("int64") // 10**6))  # ms epoch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_dataset(data_dir: str | Path, kind: str) -> tuple[pd.DataFrame, str]:
    """Load 'perp' or 'spot' data; returns (df, source_label).

    source_label is "real" for the canonical CSVs, "SYNTHETIC" otherwise.
    """
    if kind not in CANONICAL:
        raise ValueError(f"kind must be one of {sorted(CANONICAL)}")
    data_dir = Path(data_dir)
    canonical = data_dir / CANONICAL[kind]
    if canonical.exists():
        return load_ohlcv_csv(canonical), "real"

    synth = data_dir / SYNTHETIC[kind]
    if not synth.exists():
        print(
            f"WARNING: {canonical} not found - generating SYNTHETIC data at {synth}.\n"
            "         Run 'python -m tradebot fetch' (with network access) to get real data.",
            file=sys.stderr,
        )
        perp, spot = generate_synthetic_pair()
        save_ohlcv_csv(perp, data_dir / SYNTHETIC["perp"])
        save_ohlcv_csv(spot, data_dir / SYNTHETIC["spot"])
    return load_ohlcv_csv(synth), "SYNTHETIC"


def generate_synthetic_pair(
    n_bars: int = 60_000,
    start: str = "2025-01-01",
    seed: int = 7,
    start_price: float = 94_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded BTC-like 5m series: regime-switching drift, clustered vol, jumps.

    Returns (perp, spot); spot shares the price path with a slow basis
    offset, mimicking the aligned real dataset. Deterministic per seed.
    """
    rng = np.random.default_rng(seed)
    steps = 4  # sub-steps per bar for consistent OHLC
    n = n_bars * steps
    dt_years = (5 * 60 / steps) / (365.25 * 24 * 3600)

    # Markov regimes: 0=bull, 1=bear, 2=chop
    drifts = np.array([2.2, -2.0, 0.0])         # annualized log drift
    vols = np.array([0.55, 0.75, 0.40])         # annualized vol
    stay = 0.99985                              # ~ multi-day regimes at sub-step scale
    regime = np.zeros(n, dtype=np.int8)
    switches = rng.random(n) > stay
    choices = rng.integers(0, 3, size=n)
    r = 0
    for i in range(n):
        if switches[i]:
            r = choices[i]
        regime[i] = r

    # clustered volatility multiplier (slow AR(1) on log-vol)
    lv = np.zeros(n)
    eps = rng.normal(0.0, 0.02, size=n)
    for i in range(1, n):
        lv[i] = 0.999 * lv[i - 1] + eps[i]
    vol_mult = np.exp(lv)

    z = rng.normal(size=n)
    jumps = (rng.random(n) < 2e-5) * rng.normal(0.0, 0.02, size=n)
    sigma = vols[regime] * vol_mult
    rets = (drifts[regime] - 0.5 * sigma**2) * dt_years + sigma * np.sqrt(dt_years) * z + jumps
    log_price = np.log(start_price) + np.cumsum(rets)
    price = np.exp(log_price)

    sub = price.reshape(n_bars, steps)
    idx = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    base = pd.DataFrame(
        {
            "open": np.concatenate([[start_price], sub[:-1, -1]]),
            "high": sub.max(axis=1),
            "low": sub.min(axis=1),
            "close": sub[:, -1],
        },
        index=idx,
    )
    base["high"] = base[["open", "high", "close"]].max(axis=1)
    base["low"] = base[["open", "low", "close"]].min(axis=1)

    bar_ret = np.abs(np.diff(np.log(sub[:, -1]), prepend=np.log(start_price)))
    volume = 50.0 * (1.0 + 400.0 * bar_ret) * np.exp(rng.normal(0, 0.5, n_bars))

    # slow AR(1) basis: perp trades a few bps around spot
    basis = np.zeros(n_bars)
    beps = rng.normal(0.0, 3e-5, size=n_bars)
    for i in range(1, n_bars):
        basis[i] = 0.999 * basis[i - 1] + beps[i]

    spot = base.copy()
    spot["volume"] = volume
    perp = base.mul(1.0 + basis, axis=0)
    perp["volume"] = volume * 2.4
    return perp, spot


def align(perp: pd.DataFrame, spot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to their common timestamps."""
    common = perp.index.intersection(spot.index)
    return perp.loc[common], spot.loc[common]
