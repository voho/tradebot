"""Shared, read-only utilities for the R-81 exchange-net-flow round (08-21).

Idea in one sentence: CoinMetrics' ``FlowInExNtv``/``FlowOutExNtv`` (native
BTC/ETH moving onto/off known exchange addresses) is an eighth structurally
distinct INFO signal this project has not tried -- a capital-custody /
selling-pressure flow, not a network-activity count (B-07/R-44's
``AdrActCnt``/``TxCnt``/``HashRate``), a valuation ratio (R-74's MVRV), a
macro spillover level (R-53/54's VIX/DXY), a stablecoin-supply capital-flow
proxy (R-54/55/58), a priced expectation (R-73's DVOL), a calendar feature
(R-75), a cyclical phase (R-79), or a confidence signal derived from the
strategy's own vote (R-80). Mechanism: coins flowing onto exchanges are
read as increased latent selling pressure, coins flowing off as
accumulation / reduced free float -- the standard reading behind
CryptoQuant's and Glassnode's netflow products, and studied academically
in Ren, Wu & Liu (2024, arXiv:2411.06327) "Return and Volatility
Forecasting Using On-Chain Flows in Cryptocurrency Markets", which found
(their words, at intraday 1-6h horizons): BTC's own net exchange inflow
*generally lacks return-forecasting power* (except at a 4h horizon) but
*negatively forecasts volatility* at every intraday interval tested; ETH's
own net inflow negatively forecasts both ETH returns and ETH volatility;
and USDT net inflow to exchanges (dry powder) positively forecasts both
BTC and ETH returns. That last channel (stablecoin flows) is close enough
to R-54/55/58's stablecoin-supply-growth signal that this round does not
re-test it; the two branches below split the two *native-asset* findings
instead -- one on the DIRECTION axis (which the paper's own result should
make the operator skeptical of, going in) and one on the SIZE/volatility
axis (which the paper's own result is actually optimistic about).

Data: ``scripts/fetch_exchange_flows.py`` (CoinMetrics free community API,
no key). BTC coverage 2016-01-01 -> present, ETH 2019-01-01 -> present,
zero missing days in the pulled range (both checked by the operator before
dispatch). Load with ``tradebot.data.load_exchange_flow`` /
``align_onchain_causal`` (the latter's one-day-late causal shift applies
unchanged -- CoinMetrics reports day D's flow only after D closes).

This module is read-only utility, written by the operator before dispatch.
Neither branch edits it. It contains: (1) shared date constants matching
every prior lead-time round in this project (R-53/54/55/73/74/75/79/80);
(2) the same 5-episode stress-onset list R-74/R-73 used, for direct
comparability across INFO-axis rounds; (3) a byte-for-byte duplicate of
`kelly_regime_v4`'s own 3-anchor vote and the R-53/R-55 confirming-vote
formula (re-exported from `experiments.r80_shared`, not re-implemented, to
avoid a second independent chance at the same bug); (4) small causal-safe
helpers (`assert_no_holdout`, `daily_transitions`, `nearest_onset`) with
the exact semantics every prior lead-time study in this project has used.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_onchain_causal, load_dataset, load_exchange_flow  # noqa: E402

# Re-exported, not re-implemented -- see experiments/r80_shared.py's own docstring.
from experiments.r80_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    V4_BAND,
    V4_HORIZONS,
    anchor_votes,
    confirming_vote_frac,
    placebo_offset_indices,
    truncation_causality_probe,
)

DATA_DIR = ROOT / "data"

INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"
TRAIN = ("2017-01-01", INNER_TRAIN_END)
VALID = (INNER_VAL_START, INNER_VAL_END)

# Identical to R-73/R-74's own list, for direct cross-round comparability.
EPISODES = [
    ("2018 bear onset", pd.Timestamp("2018-01-17", tz="UTC")),
    ("2020-03 COVID crash", pd.Timestamp("2020-03-12", tz="UTC")),
    ("2021 top / 2022 bear transition", pd.Timestamp("2021-11-10", tz="UTC")),
    ("2022-05 Terra/Luna", pd.Timestamp("2022-05-09", tz="UTC")),
    ("2022-11 FTX collapse", pd.Timestamp("2022-11-08", tz="UTC")),
]
SEARCH_WINDOW_DAYS = 90


def assert_no_holdout(df: pd.DataFrame) -> None:
    """Every prior round's second, independent holdout guard, unchanged."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")


def load_net_flow(asset: str = "BTC") -> pd.DataFrame | None:
    """Daily net exchange flow, RAW (no z-score, no causal shift yet).

    Columns: FlowInExNtv, FlowOutExNtv, net_flow (= In - Out, positive =
    more flowing ONTO exchanges = read as latent selling pressure).
    """
    raw = load_exchange_flow(DATA_DIR, asset=asset)
    if raw is None:
        return None
    out = raw.copy()
    out["net_flow"] = out["FlowInExNtv"] - out["FlowOutExNtv"]
    return out


def net_flow_on_bars(asset: str, bars: pd.DataFrame) -> pd.Series:
    """Causally-aligned `net_flow`, reindexed onto `bars`' timestamps.

    Uses the same one-day-late shift as `load_onchain_metrics`/MVRV/DVOL
    (CoinMetrics reports day D's flow only after D closes -- a bar at time
    T may only see the flow row for the most recent day that closed
    strictly before T's own day).
    """
    daily = load_net_flow(asset)
    if daily is None:
        raise FileNotFoundError(f"no exchange-flow data for {asset}")
    aligned = align_onchain_causal(daily[["net_flow"]], bars)
    return aligned["net_flow"]


def daily_transitions(series: pd.Series, target_value: float, min_gap_days: int = 14) -> list:
    """Daily-resampled transitions INTO target_value, de-duplicated within
    min_gap_days. Identical logic to every prior lead-time study here."""
    daily = series.resample("1D").last().ffill()
    is_target = (daily == target_value)
    prev = is_target.shift(fill_value=False)
    onsets = daily.index[is_target & (~prev)]
    kept = []
    for d in onsets:
        if not kept or (d - kept[-1]).days >= min_gap_days:
            kept.append(d)
    return kept


def nearest_onset(target_date, candidates, window_days=SEARCH_WINDOW_DAYS):
    best, best_dist = None, None
    for c in candidates:
        dist = (c - target_date).days
        if abs(dist) <= window_days and (best_dist is None or abs(dist) < abs(best_dist)):
            best, best_dist = c, dist
    return best, best_dist


def build_spot_with_net_flow(asset_flow: str = "BTC") -> tuple[pd.DataFrame, str]:
    """Load the spot BTC bars used everywhere else in this project, with a
    `net_flow` column attached (causal, from `asset_flow`'s CoinMetrics
    series). `asset_flow="ETH"` is for the ETH falsification leg only --
    call with `load_dataset(DATA_DIR, "spot")`'s ETH equivalent externally
    if a branch needs ETH price bars too; this helper always returns BTC
    price bars, matching every prior INFO-axis round's primary construction.
    """
    spot, label = load_dataset(DATA_DIR, "spot")
    spot = spot.copy()
    spot["net_flow"] = net_flow_on_bars(asset_flow, spot)
    return spot, label
