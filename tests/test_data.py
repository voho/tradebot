import numpy as np
import pandas as pd
import pytest

from tradebot.data import (
    align,
    coinbase_spot_file,
    generate_synthetic_pair,
    load_coinbase_eth_spot,
    load_coinbase_spot,
    load_dataset,
    load_ohlcv_csv,
    save_ohlcv_csv,
)


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


def test_coinbase_spot_file_naming():
    assert coinbase_spot_file("ETH") == "ethusd_coinbase_spot_5m.csv.gz"
    assert coinbase_spot_file("bch") == "bchusd_coinbase_spot_5m.csv.gz"


def test_load_coinbase_spot_missing_returns_none(tmp_path):
    assert load_coinbase_spot(tmp_path, "LTC") is None
    assert load_coinbase_eth_spot(tmp_path) is None


def test_load_coinbase_spot_reads_panel_file(tmp_path):
    df, _ = generate_synthetic_pair(n_bars=300)
    save_ohlcv_csv(df, tmp_path / coinbase_spot_file("LTC"))
    out = load_coinbase_spot(tmp_path, "LTC")
    assert out is not None and len(out) == 300
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_load_coinbase_eth_spot_uses_the_same_path(tmp_path):
    df, _ = generate_synthetic_pair(n_bars=120)
    save_ohlcv_csv(df, tmp_path / coinbase_spot_file("ETH"))
    pd.testing.assert_frame_equal(load_coinbase_eth_spot(tmp_path),
                                  load_coinbase_spot(tmp_path, "ETH"))
