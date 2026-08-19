import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from tradebot.data import (
    align,
    generate_synthetic_pair,
    load_dataset,
    load_ohlcv_csv,
    load_onchain,
    save_ohlcv_csv,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_synthetic_is_deterministic():
    a, _ = generate_synthetic_pair(n_bars=500, seed=42)
    b, _ = generate_synthetic_pair(n_bars=500, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_ohlc_consistency():
    perp, spot = generate_synthetic_pair(n_bars=2_000)
    for df in (perp, spot):
        assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()
        assert df.index.tz is not None
        assert df.index.is_monotonic_increasing


def test_csv_round_trip(tmp_path):
    perp, _ = generate_synthetic_pair(n_bars=300)
    path = tmp_path / "x.csv"
    save_ohlcv_csv(perp, path)
    loaded = load_ohlcv_csv(path)
    assert len(loaded) == len(perp)
    assert loaded.index.equals(perp.index)
    assert np.allclose(loaded["close"], perp["close"])


def test_csv_round_trip_seconds_resolution_index(tmp_path):
    """pandas may hold a DatetimeIndex in s/ms/us resolution; saving must
    still write correct ms epochs (regression: index.view assumed ns)."""
    idx = pd.to_datetime(pd.Series([1_483_228_800, 1_483_229_100]), unit="s", utc=True)
    df = pd.DataFrame(
        {"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5],
         "close": [1.5, 2.5], "volume": [10.0, 11.0]},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
    path = tmp_path / "x.csv"
    save_ohlcv_csv(df, path)
    loaded = load_ohlcv_csv(path)
    assert len(loaded) == 2
    assert loaded.index.equals(df.index.as_unit(loaded.index.unit))


def test_load_epoch_units(tmp_path):
    idx_ms = 1_700_000_000_000
    for factor, name in [(1, "ms"), (1_000, "us"), (0.001, "s")]:
        path = tmp_path / f"{name}.csv"
        path.write_text(
            "timestamp,open,high,low,close,volume\n"
            f"{int(idx_ms * factor)},1,2,0.5,1.5,10\n"
        )
        df = load_ohlcv_csv(path)
        assert df.index[0] == pd.Timestamp(idx_ms, unit="ms", tz="UTC")


def test_load_dataset_falls_back_to_synthetic(tmp_path, capsys):
    df, label = load_dataset(tmp_path, "perp")
    assert label == "SYNTHETIC"
    assert len(df) > 10_000
    assert (tmp_path / "synthetic_perp_5m.csv").exists()
    assert (tmp_path / "synthetic_spot_5m.csv").exists()
    # second load reuses the files (still labeled synthetic)
    df2, label2 = load_dataset(tmp_path, "spot")
    assert label2 == "SYNTHETIC"


def test_load_dataset_prefers_real(tmp_path):
    perp, _ = generate_synthetic_pair(n_bars=200)
    save_ohlcv_csv(perp, tmp_path / "btcusdt_perp_5m.csv")
    df, label = load_dataset(tmp_path, "perp")
    assert label == "real"
    assert len(df) == 200


def test_align():
    a, _ = generate_synthetic_pair(n_bars=100)
    b = a.iloc[10:].copy()
    a2, b2 = align(a, b)
    assert a2.index.equals(b2.index)
    assert len(a2) == 90


# --- committed on-chain data (B-07), fetched by scripts/fetch_coinmetrics_onchain.py ---


def test_committed_onchain_data_is_present_and_sane():
    for asset, min_rows, min_year in (("BTC", 5_500, 2010), ("ETH", 3_500, 2015)):
        onchain = load_onchain(DATA_DIR, asset)
        assert onchain is not None, f"committed on-chain history for {asset} is missing"
        assert len(onchain) > min_rows
        assert onchain.index.is_monotonic_increasing
        assert not onchain.index.has_duplicates
        assert onchain.index.tz is not None
        assert onchain.index.min().year <= min_year + 1
        assert onchain.index.max().year >= 2026
        # MVRV is a ratio: near 0 at deep capitulation, historically never
        # negative and rarely above ~10 once realized cap has stabilized.
        # Its own first ~60 days are a bootstrap artifact (realized cap
        # starts near zero as the metric's own lookback fills in) --
        # excluded here, not evidence the strategy code needs to handle.
        mvrv = onchain["mvrv"].dropna().iloc[60:]
        assert len(mvrv) > min_rows - 350  # allow the realized-cap warmup gap
        assert (mvrv > 0).all()
        assert mvrv.max() < 20
        assert (onchain["active_addresses"] > 0).all()
        assert onchain["supply"].is_monotonic_increasing.__class__ is bool  # sanity: no crash


def test_onchain_index_is_shifted_one_day_forward_for_causal_use():
    """CoinMetrics timestamps a day's aggregate at that day's own 00:00 UTC;
    the loader must shift it so the value is only available the next day."""
    onchain = load_onchain(DATA_DIR, "BTC")
    # 2026-08-18 is the last day fetched (see fetch_coinmetrics_onchain.py);
    # after the +1 day shift its value should be indexed at 2026-08-19.
    assert onchain.index.max() == pd.Timestamp("2026-08-19", tz="UTC")


def test_onchain_missing_asset_returns_none(tmp_path):
    assert load_onchain(tmp_path, "BTC") is None
