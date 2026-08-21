"""R-74 CONSERVATIVE branch: causal MVRV-ratio Z-score feature.

Private helper for ``experiments/r74_conservative_mvrv_level.py``. Not
shared with, or coordinated with, the parallel NOVEL branch running the
same round on MVRV's *rate of change* instead of its *level* -- neither
file is read by the other.

Data: ``data/btc_mvrv_daily.csv.gz`` / ``data/eth_mvrv_daily.csv.gz``,
already fetched and committed (CoinMetrics free community API,
``CapMVRVCur``), loaded read-only via ``tradebot.data.load_mvrv_ratio`` /
``align_mvrv_causal`` (both already added to ``src/tradebot/data.py`` --
not modified here or anywhere in this branch).

Mechanism (why MVRV LEVEL should carry signal, stated in one sentence per
ROUTINE.md step 2): realized cap marks every coin at the price it last
moved on-chain rather than today's price (Nic Carter's realized-cap
concept, Carter & Le Calvez, Honeybadger Capital 2018), so MVRV =
market_cap / realized_cap is an aggregate holder profit/loss ratio, and
Mahmudov & Puell (2018, "Bitcoin Market-Value-to-Realized-Value (MVRV)
Ratio", introducing the companion MVRV-Z-Score) show its historical
extremes have marked market-cycle tops (aggregate unrealized profit far
above its own norm -- euphoria, distribution risk) and bottoms
(aggregate unrealized loss -- capitulation).

**Directionality decision, made before any code ran:** this branch is
ONE-DIRECTIONAL -- only the EUPHORIA extreme (high MVRV Z) is used, as a
risk-off DILUTING vote, never a bullish override. Three reasons, stated
up front:
  1. It matches every other INFO-axis signal already tried in this
     project (VIX/DXY stress, R-53; stablecoin-supply deceleration,
     R-54/R-55; VRP, R-73) -- all one-directional, all diluting-only,
     never manufacturing bullish exposure the anchors would not already
     grant.
  2. The confirming-vote architecture itself (R-55's validated
     ``frac=(anchor_sum+weight*vote)/(3+weight)``) is structurally built
     to only ever pull ``frac`` DOWN when the extra vote reads 0; using
     the bottom extreme as a symmetric bullish trigger would require a
     different architecture never validated by this project's own R-55
     ablation, and would risk exactly the "fight the trend" failure mode
     kelly_regime's own design docstring explicitly avoids (a
     negative-drift bet is never sized up).
  3. It keeps this branch's falsification test clean: a signal that can
     only ever REDUCE exposure cannot be blamed for an unmatched-risk
     comparison in the direction that has burned this project before
     (R-33) -- diluting-only structurally reduces exposure only, whose
     comparison against v4 must be read on a risk-matched or return
     basis, exactly as the standard battery below does.

**Z-score window decision, made before any code ran (structural, NOT
swept for fit):** the classic MVRV-Z-Score construction (Puell/Mahmudov
2018; the LookIntoBitcoin/Coin Metrics convention this project's own
data-loader docstring cites) z-scores the raw valuation-gap level against
ALL AVAILABLE HISTORY up to each day -- an EXPANDING window since
inception, not a fixed lookback tuned to any particular cycle length.
This branch reproduces that literal construction on the MVRV RATIO (the
one column the committed data provides): ``mvrv_z_t = (mvrv_t -
expanding_mean(mvrv)_t) / expanding_std(mvrv)_t``, computed causally (row
t uses only rows <= t) with ``min_periods=365`` (~1yr) so the earliest,
noisiest stub of history does not vote -- before that the feature is NaN
and the confirming vote defaults to neutral (no dilution), the same
convention every prior INFO signal in this project uses for missing or
insufficient data. There is therefore no window LENGTH to sweep: the
window is "since inception" by construction, exactly as the cited
indicator defines it. What IS swept below (in the strategy file, not
here) is the EXTREME THRESHOLD applied to this Z-score and the
confirming-vote WEIGHT -- a small, structural, a-priori grid, not a
fitted window.

Every rolling statistic here is computed on the RAW DAILY MVRV frame,
strictly backward-looking, before ``align_mvrv_causal`` additionally
shifts the finished daily Z-score one more day forward for CoinMetrics'
own publication lag -- exactly the discipline ``_stablecoin_signal.py``
and ``_macro_signal.py`` already use for their own rolling features.

**Holdout discipline:** the raw MVRV frame is truncated to strictly
before ``OOS_START`` at the moment it is loaded, before any expanding
statistic is computed on it -- belt-and-braces on top of the fact that an
expanding statistic at day t already only depends on days <= t, so
truncating rows on/after the holdout start changes nothing for any
in-bounds day; it only guarantees no 2023+ MVRV value is ever held in
memory by this module, matching this round's explicit hard rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data import align_mvrv_causal, load_mvrv_ratio

OOS_START = "2023-01-01"
MIN_PERIODS_DAYS = 365  # ~1yr warmup before the expanding z-score is trusted


def _truncate_before_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any row on/after OOS_START. No-op if the frame is already clear."""
    if len(df) == 0:
        return df
    cutoff = pd.Timestamp(OOS_START, tz=df.index.tz)
    return df.loc[df.index < cutoff]


def compute_mvrv_z(bars_df: pd.DataFrame, data_dir: str | Path, asset: str = "BTC") -> pd.Series:
    """Causal, EXPANDING-window MVRV Z-score aligned onto ``bars_df``'s
    index, or all-NaN if the underlying data is absent. See module
    docstring for the full construction and its citation."""
    mvrv = load_mvrv_ratio(data_dir, asset=asset)
    if mvrv is None:
        return pd.Series(index=bars_df.index, dtype=float)
    mvrv = _truncate_before_holdout(mvrv)

    s = mvrv["mvrv"]
    mean = s.expanding(min_periods=MIN_PERIODS_DAYS).mean()
    std = s.expanding(min_periods=MIN_PERIODS_DAYS).std()
    z = ((s - mean) / std).rename("mvrv_z").to_frame()

    aligned = align_mvrv_causal(z, bars_df)["mvrv_z"]
    return aligned
