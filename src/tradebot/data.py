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
    return load_coinbase_spot(data_dir, "ETH")


def coinbase_spot_file(asset: str) -> str:
    """Filename convention for a Coinbase USD 5m spot series (R-57's panel).

    ``ETH`` -> ``ethusd_coinbase_spot_5m.csv.gz``, matching the file
    ``scripts/fetch_coinbase_spot.py`` already wrote and the names
    ``scripts/fetch_coinbase_panel.py`` writes for the rest of the panel.
    """
    return f"{asset.lower()}usd_coinbase_spot_5m.csv.gz"


def load_coinbase_spot(data_dir: str | Path, asset: str) -> pd.DataFrame | None:
    """Real Coinbase ``{asset}-USD`` 5m spot OHLCV, or None if not fetched.

    The generic form of :func:`load_coinbase_eth_spot`. R-57 fetched a panel
    of six further Coinbase USD series (2020-01 -> 2026-08) to ask whether
    ``kelly_regime_v4``'s drawdown property replicates on instruments it was
    never fitted on; this is the loader those files share. Nothing is
    computed over the whole series here (no scaler, mean or std), so the
    loading path carries no full-series-fit lookahead risk.
    """
    path = Path(data_dir) / coinbase_spot_file(asset)
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


ONCHAIN_FILES = {"BTC": "btc_onchain_daily.csv.gz", "ETH": "eth_onchain_daily.csv.gz"}
ONCHAIN_METRICS = ["AdrActCnt", "TxCnt", "HashRate"]


def load_onchain_metrics(data_dir: str | Path, asset: str = "BTC") -> pd.DataFrame | None:
    """Daily on-chain metrics (active addresses, tx count, hash rate), or None if absent.

    Fetched by ``scripts/fetch_onchain_metrics.py`` from CoinMetrics'
    free community API (B-07). This is the first price-independent
    information channel in this project -- every other feature is derived
    from the same OHLCV series it is meant to trade. Indexed by UTC day
    (midnight). ``asset="BTC"`` covers 2017-01-01 -> present;
    ``asset="ETH"`` covers 2019-01-01 -> present, with ``HashRate`` NaN
    from the 2022-09-15 Merge onward (ETH is proof-of-stake from there;
    the column is not back-filled or proxied).
    """
    if asset not in ONCHAIN_FILES:
        raise ValueError(f"asset must be one of {sorted(ONCHAIN_FILES)}")
    path = Path(data_dir) / ONCHAIN_FILES[asset]
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = df.index.tz_localize("UTC")
    return df[ONCHAIN_METRICS].astype(float).sort_index()


def align_onchain_causal(onchain: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex daily on-chain metrics onto ``bars``' index, causally.

    CoinMetrics reports day D's metric only after day D has closed, so a
    bar at time T may only see the on-chain row for the most recent day
    that closed strictly before T's own day -- i.e. shifted one full day
    later than a naive as-of join would allow. Concretely: a metric dated
    2024-01-05 (meaning "day 2024-01-05") only becomes visible at
    2024-01-06T00:00 UTC. Bars before the first visible row get NaN, never
    filled or back-cast.
    """
    shifted = onchain.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


MACRO_FILES = {
    "spx": "spx_daily.csv.gz",
    "vix": "vix_daily.csv.gz",
    "dxy": "dxy_daily.csv.gz",
}


def load_macro_metrics(data_dir: str | Path) -> pd.DataFrame | None:
    """Daily S&P 500 close, VIX close and Fed broad dollar index, or None if absent.

    Fetched by ``scripts/fetch_macro_data.py`` from FRED's free public CSV
    endpoint (R-53). Unlike on-chain metrics (B-07), which describe the
    traded asset's own network, these three describe the rest of the
    financial system -- equity risk appetite, equity-implied fear and
    dollar strength -- the channel the VIX/DXY-Bitcoin spillover literature
    argues leads crypto risk-off moves rather than merely coinciding with
    them. Indexed by UTC day (midnight). ``spx`` only carries a trailing
    ~10-year window (a FRED platform limit on that specific series, not a
    fetch gap); ``vix`` and ``dxy`` go back decades. Columns with no
    observation for a given day (weekends, market holidays) are simply
    absent rows, not zero-filled -- causal alignment handles that with
    ``ffill``, same as the on-chain loader.
    """
    frames = {}
    for col, filename in MACRO_FILES.items():
        path = Path(data_dir) / filename
        if not path.exists():
            return None
        raw = pd.read_csv(path, parse_dates=["date"], index_col="date")
        frames[col] = raw.iloc[:, 0]
    df = pd.DataFrame(frames)
    df.index = df.index.tz_localize("UTC")
    return df.astype(float).sort_index()


def align_macro_causal(macro: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex daily macro series onto ``bars``' index, causally.

    FRED publishes day D's close only after day D has ended, so -- exactly
    like ``align_onchain_causal`` -- a bar at time T may only see the row
    for the most recent day that closed strictly before T's own day. Bars
    before the first visible row get NaN, never filled or back-cast.
    """
    shifted = macro.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


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


STABLECOIN_FILE = "stablecoin_supply_daily.csv.gz"


def load_stablecoin_supply(data_dir: str | Path) -> pd.DataFrame | None:
    """Daily aggregate stablecoin (USDT) circulating supply, or None if absent.

    Fetched by ``scripts/fetch_stablecoin_supply.py`` from CoinMetrics'
    free community API (R-54 NOVEL branch), the same endpoint family
    ``load_onchain_metrics`` already uses for BTC/ETH chain metrics
    (B-07/R-44). Unlike on-chain activity metrics (which describe BTC's own
    network) or VIX/DXY macro stress (which describe the rest of the
    financial system, R-53), this describes dollar capital actually inside
    the crypto trading system: stablecoin issuance is the on-ramp for new
    dollar capital entering crypto, redemption the off-ramp. USDT alone
    (2017-01-01 -> present, 0 NaN, 0 gaps as committed); USDC's community-tier
    history is real but starts materially later (placeholder/near-zero
    until 2018-09-25) and is deliberately NOT combined in here -- see
    ``experiments/_stablecoin_signal.py``'s module docstring for the reason
    stated plainly rather than silently blended. Indexed by UTC day
    (midnight), single column ``supply``.
    """
    path = Path(data_dir) / STABLECOIN_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = df.index.tz_localize("UTC")
    out = df.rename(columns={"usdt_SplyCur": "supply"})[["supply"]].astype(float)
    return out.sort_index()


def align_stablecoin_causal(supply: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex daily stablecoin supply onto ``bars``' index, causally.

    CoinMetrics reports day D's supply only after day D has closed, so --
    exactly like ``align_onchain_causal``/``align_macro_causal`` -- a bar at
    time T may only see the row for the most recent day that closed
    strictly before T's own day. Bars before the first visible row get NaN,
    never filled or back-cast.
    """
    shifted = supply.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


DVOL_FILE = "btc_dvol_daily.csv.gz"


def load_dvol_index(data_dir: str | Path) -> pd.DataFrame | None:
    """Daily BTC DVOL (Deribit's official 30-day implied-volatility index), or None if absent.

    Fetched by ``scripts/fetch_deribit_dvol_novel.py`` from Deribit's public
    ``get_volatility_index_data`` endpoint (R-73). Unlike every other
    INFO-axis signal tried in this project so far -- on-chain activity
    (B-07/R-44, describes BTC's own network), VIX/DXY macro stress (R-53,
    describes the rest of the financial system), stablecoin supply
    (R-54/R-55/R-58, a spot/balance-sheet flow proxy) -- DVOL is a
    forward-looking, PRICED market expectation: option writers' 30-day-ahead
    volatility view, set today. Columns ``open, high, low, close`` (index
    close is the value used everywhere in this round, matching the
    convention "today's DVOL close" the way VIX close is used in
    ``load_macro_metrics``). Indexed by UTC day (midnight).

    Hard data limitation, stated plainly rather than proxied around: history
    starts ~2021-03-24 (options markets did not exist at scale before then).
    There is no way to backfill this -- any strategy or study built on DVOL
    is confined to that window, materially shorter than this project's usual
    2017-> history.
    """
    path = Path(data_dir) / DVOL_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = df.index.tz_localize("UTC")
    return df.astype(float).sort_index()


def align_dvol_causal(dvol: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex daily DVOL onto ``bars``' index, causally.

    Deribit's index_data endpoint's daily bar for day D only finishes
    forming once day D has closed, so -- exactly like
    ``align_onchain_causal``/``align_macro_causal``/``align_stablecoin_causal``
    -- a bar at time T may only see the row for the most recent day that
    closed strictly before T's own day. Bars before the first visible row
    get NaN, never filled or back-cast.
    """
    shifted = dvol.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


MVRV_FILES = {
    "BTC": "btc_mvrv_daily.csv.gz",
    "ETH": "eth_mvrv_daily.csv.gz",
}


def load_mvrv_ratio(data_dir: str | Path, asset: str = "BTC") -> pd.DataFrame | None:
    """Daily MVRV ratio (market cap / realized cap), or None if absent.

    Fetched by ``scripts/fetch_coinmetrics_mvrv.py`` from CoinMetrics'
    free community API (R-74), the same endpoint family
    ``load_stablecoin_supply`` already uses. Realized cap marks every coin
    at the price it last moved on-chain rather than at today's price
    (Mahmudov & Puell 2018, building on Carter & Le Calvez's realized-cap
    concept), so MVRV is a valuation signal -- aggregate holder
    profit/loss -- genuinely distinct from every other INFO-axis signal
    tried so far: not a flow (stablecoin supply, R-54), not a priced
    volatility expectation (DVOL/VRP, R-73), not a spillover from the rest
    of the financial system (VIX/DXY, R-53), and not the traded asset's
    own activity (active-address growth, B-07/R-44). Single column
    ``mvrv``. Indexed by UTC day (midnight). ``BTC`` covers 2016-01-01 ->
    present, ``ETH`` 2018-01-01 -> present (both requested from before
    this project's own OHLCV start so no warmup bar is starved).
    """
    if asset not in MVRV_FILES:
        raise ValueError(f"asset must be one of {sorted(MVRV_FILES)}")
    path = Path(data_dir) / MVRV_FILES[asset]
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = df.index.tz_localize("UTC")
    return df[["mvrv"]].astype(float).sort_index()


def align_mvrv_causal(mvrv: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex daily MVRV onto ``bars``' index, causally.

    CoinMetrics reports day D's MVRV only after day D has closed, so --
    exactly like ``align_onchain_causal``/``align_macro_causal``/
    ``align_stablecoin_causal``/``align_dvol_causal`` -- a bar at time T
    may only see the row for the most recent day that closed strictly
    before T's own day. Bars before the first visible row get NaN, never
    filled or back-cast.
    """
    shifted = mvrv.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


METRICS_FILES = {
    "BTC": "btcusdt_perp_metrics_5m.csv.gz",
    "ETH": "ethusdt_perp_metrics_5m.csv.gz",
}


def load_binance_metrics(data_dir: str | Path, asset: str = "BTC") -> pd.DataFrame | None:
    """Binance USDⓈ-M futures positioning metrics at native 5-minute
    cadence, or ``None`` if absent.

    Fetched by ``scripts/fetch_binance_metrics.py`` from the static
    ``data.binance.vision`` history host (R-81) -- reachable even when the
    live ``fapi.binance.com`` API 451s under this project's network
    policy. Columns: ``sum_open_interest`` (base units),
    ``sum_open_interest_value`` (quote), ``count_toptrader_long_short_ratio``
    / ``sum_toptrader_long_short_ratio`` (Binance's largest-account
    positioning ratio, by count and by size), ``count_long_short_ratio``
    (all-account count ratio), ``sum_taker_long_short_vol_ratio`` (taker
    buy/sell volume ratio). Unlike every INFO signal this project has tried
    before -- on-chain activity (B-07/R-44), macro VIX/DXY (R-53),
    stablecoin supply (R-54), DVOL/VRP (R-73), MVRV (R-74), calendar
    structure (R-75/R-79) -- this is a derivatives-positioning ("crowding")
    measure at the SAME 5-minute cadence as this project's own bars, not a
    daily-or-coarser feed. It measures how levered/one-sided the market
    currently is, not a claim about future price.

    Hard data limitation, stated plainly: ``BTC`` history starts
    2020-09-01 (the earliest daily file Binance publishes for this
    endpoint); ``ETH`` only from 2021-12-01, materially shorter, a
    DVOL-like coverage caveat. Indexed by UTC timestamp; each on-the-wire
    row is duplicated (verified: every 5-minute timestamp appears twice,
    byte-identical) and deduplicated by the fetch script already, not
    here.
    """
    if asset not in METRICS_FILES:
        raise ValueError(f"asset must be one of {sorted(METRICS_FILES)}")
    path = Path(data_dir) / METRICS_FILES[asset]
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["create_time"], index_col="create_time")
    df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"
    cols = ["sum_open_interest", "sum_open_interest_value",
            "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    return df[cols].astype(float).sort_index()


def align_metrics_causal(metrics: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex 5-minute metrics onto ``bars``' index, causally.

    Unlike the daily signals above (which shift by a full day because the
    source only finalizes once the calendar day closes), this feed shares
    ``bars``' own 5-minute cadence -- so the causal contract is the same
    one every OHLCV bar itself uses: a decision at bar i's close may see
    metrics timestamped at or before bar i's own timestamp, never after.
    Forward-fills across any gap (a day the venue never published, or a
    momentary feed outage) rather than leaving it NaN mid-series, then
    reindexes onto ``bars`` exactly; bars strictly before the first
    metrics row get NaN, never back-cast.
    """
    return metrics.reindex(metrics.index.union(bars.index)).sort_index().ffill().reindex(bars.index)
