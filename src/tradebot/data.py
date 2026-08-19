"""Data loading: committed real data, optional fetched data, synthetic fallback.

Canonical files, in loading priority order:

- ``data/btcusdt_perp_5m.csv``          USDT-margined BTCUSDT perp 5m klines
  (produced by ``python -m tradebot fetch``; needs Binance network access)
- ``data/btcusdt_spot_aligned_5m.csv``  BTCUSDT spot 5m aligned to the perp
  (produced by ``python -m tradebot fetch``)
- ``data/btcusd_spot_5m.csv.gz``        Bitstamp BTC/USD 5m, 2017 -> present,
  committed to the repo (built by ``scripts/build_bitstamp_dataset.py``)

Format: columns ``timestamp,open,high,low,close,volume`` with ``timestamp``
as milliseconds since epoch (UTC); ``.gz`` files are read transparently.

When no perp file exists, the futures market runs on the spot series and
every report is labeled "spot (perp proxy)" — the perp basis is small, but
the label keeps it honest. If no real data exists at all, ``load_dataset``
falls back to a seeded synthetic series (generated once into
``data/synthetic_*.csv``, gitignored) so the framework stays runnable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL = {
    "perp": ["btcusdt_perp_5m.csv"],
    "spot": ["btcusdt_spot_aligned_5m.csv", "btcusd_spot_5m.csv.gz"],
}
SYNTHETIC = {
    "perp": "synthetic_perp_5m.csv",
    "spot": "synthetic_spot_5m.csv",
}

LABEL_REAL = "real"
LABEL_PROXY = "spot (perp proxy)"
LABEL_SYNTH = "SYNTHETIC"


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
    # as_unit("ms") makes this correct for any index resolution (pandas can
    # hold datetime64 in s/ms/us/ns depending on how the index was built)
    out.insert(0, "timestamp", out.index.as_unit("ms").asi8)  # ms epoch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


FUNDING_FILE = "btcusdt_perp_funding_8h.csv.gz"


def load_funding(data_dir: str | Path) -> pd.Series | None:
    """Historical perpetual funding rates, or None when the file is absent.

    Indexed by settlement time (8-hourly), values are the per-settlement
    rate as a decimal: positive means longs pay shorts. Real Binance
    BTCUSDT data, 2020-2023; see docs/VALIDATION.md for what it does to
    the futures results and for the periods it does not cover.
    """
    path = Path(data_dir) / FUNDING_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    s = df["funding_rate"].astype(float).sort_index()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


DERIBIT_FUNDING_FILE = "btcusdt_deribit_perp_funding_8h.csv.gz"


def _load_raw_funding_csv(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    s = df["funding_rate"].astype(float).sort_index()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def load_funding_deribit(data_dir: str | Path) -> pd.Series | None:
    """Deribit BTC-PERPETUAL funding, 2020-01 -> 2026-08 (R-39 / B-02).

    Fetched by ``scripts/fetch_deribit_funding.py`` from Deribit's public
    API (reachable where Binance is not, as of 08-19). Deribit charges
    funding continuously; each row here is the sum of hourly
    ``interest_1h`` charges over a UTC-aligned 8-hour bucket
    [00:00, 08:00, 16:00), timestamped at the bucket's close -- a
    *different* settlement convention from Binance's discrete 8-hourly
    rate (settled at 03/11/19 UTC in the committed Binance series), so
    the two are comparable in shape and rough magnitude but are not the
    same instrument. On the 2020-2023 overlap the two series correlate at
    r=0.69 (daily-summed) but the level ratio is unstable year to year
    (0.21x-1.24x), so this is NOT rescaled to "look like" Binance -- see
    ``load_funding_extended`` for how the two are combined, and
    docs/VALIDATION.md for the cross-venue check in full.
    """
    return _load_raw_funding_csv(Path(data_dir) / DERIBIT_FUNDING_FILE)


def load_funding_extended(data_dir: str | Path) -> tuple[pd.Series | None, pd.Series | None]:
    """Real Binance funding (2020-2023) concatenated with Deribit funding
    for the genuine post-2023 gap only -- never the reverse, and never
    blended inside the overlap. Returns ``(rate, source)`` where
    ``source`` is the string "binance" or "deribit" per settlement, or
    ``(None, None)`` if neither file is present.

    This is deliberately not a single silently-spliced number: any
    analysis that cares about the venue should read ``source`` rather
    than assume every row means the same thing. See ``load_funding_deribit``
    for why the two series are not rescaled onto a common level.
    """
    binance = load_funding(data_dir)
    deribit = load_funding_deribit(data_dir)
    if binance is None and deribit is None:
        return None, None
    if binance is None:
        return deribit, pd.Series("deribit", index=deribit.index)
    if deribit is None:
        return binance, pd.Series("binance", index=binance.index)
    cutoff = binance.index.max()
    extension = deribit[deribit.index > cutoff]
    combined = pd.concat([binance, extension]).sort_index()
    source = pd.concat([
        pd.Series("binance", index=binance.index),
        pd.Series("deribit", index=extension.index),
    ]).sort_index()
    return combined, source


DERIBIT_PERP_PRICE_FILES = {
    "BTC": "btcusdt_deribit_perp_5m.csv.gz",
    "ETH": "ethusdt_deribit_perp_5m.csv.gz",
}
COINBASE_ETH_SPOT_FILE = "ethusd_coinbase_spot_5m.csv.gz"


def load_deribit_perp_price(data_dir: str | Path, asset: str = "BTC") -> pd.DataFrame | None:
    """Real Deribit ``{asset}-PERPETUAL`` 5m OHLCV, or None if not fetched (B-15).

    Unlike every other "futures" series in this repository, this is a
    genuinely distinct, independently-transacted price series -- not the
    spot series relabeled. Fetched by ``scripts/fetch_deribit_perp_price.py``.
    Coverage is NOT the full spot history: BTC-PERPETUAL chart data starts
    2018-08-14 (probed empirically -- Deribit returns ``no_data`` before
    that), and ETH-PERPETUAL was created 2019-03-14. Bars before those
    dates simply do not exist here; nothing is back-filled or proxied.
    """
    if asset not in DERIBIT_PERP_PRICE_FILES:
        raise ValueError(f"asset must be one of {sorted(DERIBIT_PERP_PRICE_FILES)}")
    path = Path(data_dir) / DERIBIT_PERP_PRICE_FILES[asset]
    if not path.exists():
        return None
    return load_ohlcv_csv(path)


def load_coinbase_eth_spot(data_dir: str | Path) -> pd.DataFrame | None:
    """Real Coinbase ETH-USD 5m spot OHLCV, 2019-03 -> present, or None if absent.

    The committed ``ethusd_bitfinex_5m.csv.gz`` (R-17's falsification file)
    stops in 2019-12, which would leave under a year of overlap with
    ETH-PERPETUAL's basis history. This is fetched separately
    (``scripts/fetch_coinbase_spot.py``) to give the ETH basis a spot
    reference that spans its full available window.
    """
    path = Path(data_dir) / COINBASE_ETH_SPOT_FILE
    if not path.exists():
        return None
    return load_ohlcv_csv(path)


def compute_basis(spot: pd.DataFrame, perp: pd.DataFrame) -> pd.Series:
    """Log basis ``log(perp_close / spot_close)`` on the perp series' own index.

    Positive means the perpetual trades at a premium to spot (crowded
    longs paying to stay levered); negative means a discount. Both frames
    are reindexed onto ``perp``'s timestamps with a causal (as-of, not
    interpolated) join against ``spot`` so no future spot bar can leak
    into an earlier perp bar. Bars where the join has no spot observation
    at or before the perp timestamp are dropped (NaN), never filled.
    """
    spot_aligned = (
        spot["close"]
        .reindex(spot.index.union(perp.index))
        .sort_index()
        .ffill()
        .reindex(perp.index)
    )
    spot_aligned = spot_aligned.where(perp.index >= spot.index.min())
    return np.log(perp["close"] / spot_aligned).rename("basis")


def load_dataset(data_dir: str | Path, kind: str) -> tuple[pd.DataFrame, str]:
    """Load 'perp' or 'spot' data; returns (df, source_label).

    source_label is "real" for canonical files, "spot (perp proxy)" when
    the futures market falls back to the spot series, "SYNTHETIC" otherwise.
    """
    if kind not in CANONICAL:
        raise ValueError(f"kind must be one of {sorted(CANONICAL)}")
    data_dir = Path(data_dir)
    for name in CANONICAL[kind]:
        path = data_dir / name
        if path.exists():
            return load_ohlcv_csv(path), LABEL_REAL

    if kind == "perp":
        for name in CANONICAL["spot"]:
            path = data_dir / name
            if path.exists():
                print(
                    "NOTE: no perp data found - futures runs use the spot series "
                    "(labeled 'spot (perp proxy)'). Run 'python -m tradebot fetch' "
                    "to get real Binance perp data.",
                    file=sys.stderr,
                )
                return load_ohlcv_csv(path), LABEL_PROXY

    synth = data_dir / SYNTHETIC[kind]
    if not synth.exists():
        print(
            f"WARNING: no real data in {data_dir} - generating SYNTHETIC data at {synth}.\n"
            "         Run 'python -m tradebot fetch' (with network access) to get real data.",
            file=sys.stderr,
        )
        perp, spot = generate_synthetic_pair()
        save_ohlcv_csv(perp, data_dir / SYNTHETIC["perp"])
        save_ohlcv_csv(spot, data_dir / SYNTHETIC["spot"])
    return load_ohlcv_csv(synth), LABEL_SYNTH


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
