"""Bar-by-bar cross-asset portfolio engine (backlog **B-32**).

``tradebot.multiasset`` composes already-independent single-asset legs at a
fixed capital split; its own module docstring says that design cannot
express a strategy that needs a shared risk or leverage budget decided
*during* the run. This module is the native multi-instrument engine that
family needs.

It is a promoted, production copy of the simulator five research rounds
(R-63, R-65, R-67, R-68) built and validated inside ``experiments/`` before
any registration path existed for the strategy family it runs — most
directly ``experiments/r63_shared.py``'s ``simulate_portfolio`` /
``align_frames`` / ``load_universe`` (lines ~388-548 of that file at the time
of porting). ``experiments/`` is intentionally never imported from
``src/tradebot/`` (no existing module does; see e.g.
``tradebot.data``/``tradebot.multiasset`` docstrings, which only ever cite
experiment files, never import them), so the logic is duplicated here as a
clean, type-hinted copy rather than re-imported. ``experiments/r63_shared.py``
and its descendants are untouched and remain independently reproducible.

Contract, unchanged from the experiment:

    targets[t, a] = desired fraction of PORTFOLIO EQUITY in asset ``a``,
                    decided at bar ``t``'s CLOSE.

``simulate_portfolio`` shifts that matrix by one bar and fills at the next
bar's OPEN (this project's standing fill convention). Long-only: weights are
clipped to ``[0, 1]`` and rescaled down if they sum above 1.0. Fees are
charged on traded notional at every rebalance, and a rebalance is skipped
unless the change in total notional exceeds ``deadband`` fraction of equity,
mirroring the single-asset engine's own rebalance band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.data import load_coinbase_spot, load_dataset

#: R-57's frozen six-asset panel (mechanical liquidity rule; never fitted).
UNIVERSE_6: tuple[str, ...] = ("BCH", "LTC", "ETC", "DASH", "LINK", "XTZ")
#: UNIVERSE_6 + the two BTC/ETH series this project's single-asset axis
#: fits on. BTC is the Bitstamp canonical series, ETH the Coinbase one, so
#: this universe mixes venues -- noted rather than hidden.
UNIVERSE_8: tuple[str, ...] = ("BTC", "ETH") + UNIVERSE_6

#: Mirrors ``tradebot.broker.REBALANCE_DEADBAND`` so this engine charges
#: turnover on comparable terms to every other number in this repo.
TOTAL_NOTIONAL_DEADBAND = 0.05

START_BALANCE = 1_000.0


def load_universe(tickers: tuple[str, ...], data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Raw 5m OHLCV per ticker.

    ``BTC`` is loaded as the Bitstamp canonical series via
    :func:`tradebot.data.load_dataset`; every other ticker is the committed
    Coinbase USD spot file via :func:`tradebot.data.load_coinbase_spot`.
    """
    data_dir = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if t == "BTC":
            df, label = load_dataset(data_dir, "spot")
            if "SYNTH" in label.upper():
                raise RuntimeError(f"refusing to run on synthetic BTC data ({label})")
        else:
            df = load_coinbase_spot(data_dir, t)
        if df is None or df.empty:
            raise RuntimeError(f"no data for {t}")
        frames[t] = df
    return frames


def _lo(df: pd.DataFrame, start: str | None) -> pd.Timestamp:
    return df.index[0] if start is None else pd.Timestamp(start, tz="UTC")


def _hi(df: pd.DataFrame, end: str | None) -> pd.Timestamp:
    """Right edge of a window, EXCLUSIVE of the following day's first bar."""
    if end is None:
        return df.index[-1]
    return pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def align_frames(frames: dict[str, pd.DataFrame],
                 window: tuple[str | None, str | None]) -> dict[str, pd.DataFrame]:
    """Align every frame onto the union of their 5m grids inside ``window``,
    forward-filling bars an exchange did not print.

    Forward-fill only ever copies a PAST bar forward, so this cannot leak.
    Leading rows before an asset's first real bar are dropped from the shared
    index, which is why the returned index starts at the latest first-bar
    across the universe.
    """
    start, end = window
    idx = None
    for df in frames.values():
        sub = df.loc[_lo(df, start):_hi(df, end)]
        idx = sub.index if idx is None else idx.union(sub.index)
    if idx is None or len(idx) == 0:
        raise ValueError(f"empty window {window!r}")

    first_real = max(
        frames[t].loc[_lo(frames[t], start):_hi(frames[t], end)].index[0]
        for t in frames
    )
    idx = idx[idx >= first_real]

    out: dict[str, pd.DataFrame] = {}
    for t, df in frames.items():
        # Reindex against the FULL frame (not the windowed slice) so the
        # forward-fill at the window's left edge uses that asset's own last
        # real bar rather than inventing one.
        sub = df.reindex(df.index.union(idx)).ffill().reindex(idx)
        if sub[["open", "high", "low", "close"]].isna().any().any():
            raise RuntimeError(f"{t}: NaNs survive alignment on {window!r}")
        out[t] = sub
    return out


def simulate_portfolio(
    targets: pd.DataFrame,
    aligned: dict[str, pd.DataFrame],
    market: MarketSpec,
    start_balance: float = START_BALANCE,
    deadband: float = TOTAL_NOTIONAL_DEADBAND,
) -> pd.Series:
    """Long-only unlevered portfolio backtest over a target-weight matrix.

    ``targets.iloc[t]`` is the desired fraction of portfolio equity per
    asset, DECIDED at bar ``t``'s close. It is filled at bar ``t+1``'s OPEN
    -- the standing convention of this project's engine -- so row ``t`` can
    never influence anything at or before bar ``t``.

    Weights are clipped to ``[0, 1]`` and rescaled down if they sum above
    1.0. A rebalance is skipped entirely unless the requested change in
    total traded notional exceeds ``deadband`` x equity, mirroring the
    broker's own 5% band so turnover is charged comparably to every other
    number in this repo.
    """
    assets = list(targets.columns)
    idx = targets.index
    for t in assets:
        if not aligned[t].index.equals(idx):
            raise ValueError(f"{t}: price index does not match the target matrix")

    w = np.clip(targets.to_numpy(dtype=float), 0.0, 1.0)
    w = np.nan_to_num(w, nan=0.0)
    gross = w.sum(axis=1)
    over = gross > 1.0
    if over.any():
        w[over] = w[over] / gross[over][:, None]

    opens = np.column_stack([aligned[t]["open"].to_numpy(dtype=float) for t in assets])
    closes = np.column_stack([aligned[t]["close"].to_numpy(dtype=float) for t in assets])

    n, k = w.shape
    cash = float(start_balance)
    qty = np.zeros(k)
    equity = np.empty(n)
    equity[0] = cash
    fee_rate = float(market.fee_rate)

    for i in range(1, n):
        po = opens[i]
        eq_open = cash + float(qty @ po)
        if eq_open <= 0.0:
            equity[i:] = 0.0
            break

        want_q = (w[i - 1] * eq_open) / po
        dq = want_q - qty
        traded = float(np.abs(dq) @ po)
        if traded > deadband * eq_open:
            fee = fee_rate * traded
            cash -= float(dq @ po) + fee
            qty = want_q

        equity[i] = cash + float(qty @ closes[i])
        if equity[i] < 0.0:
            equity[i] = 0.0

    return pd.Series(equity, index=idx, name="equity")


def mean_total_notional(targets: pd.DataFrame) -> float:
    """The candidate's own realized mean total notional fraction."""
    w = np.clip(np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0), 0.0, 1.0)
    return float(np.mean(np.minimum(w.sum(axis=1), 1.0)))


def static_hold_equity(aligned: dict[str, pd.DataFrame], assets: tuple[str, ...],
                       market: MarketSpec, c: float = 1.0,
                       start_balance: float = START_BALANCE) -> pd.Series:
    """Buy ``c``/N of equity in each asset on the first bar and never trade
    again -- the passive equal-weight-hold benchmark, weights allowed to
    drift."""
    assets = list(assets)
    idx = aligned[assets[0]].index
    opens = np.column_stack([aligned[t]["open"].to_numpy(dtype=float) for t in assets])
    closes = np.column_stack([aligned[t]["close"].to_numpy(dtype=float) for t in assets])

    per = c * start_balance / len(assets)
    qty = per / opens[1] if len(idx) > 1 else np.zeros(len(assets))
    fee = market.fee_rate * per * len(assets)
    cash = start_balance - per * len(assets) - fee

    equity = cash + closes @ qty
    equity[0] = start_balance
    return pd.Series(equity, index=idx, name="equity")


def matched_hold_targets(idx: pd.Index, assets: tuple[str, ...], c: float) -> pd.DataFrame:
    """Equal-weight at a CONSTANT total notional ``c``, rebalanced every bar
    -- the exposure-matched benchmark (standing R-33 rule: compare only
    arms that carry the same average notional)."""
    n = len(assets)
    c = float(np.clip(c, 1e-6, 1.0))
    return pd.DataFrame(c / n, index=idx, columns=list(assets))
