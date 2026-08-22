"""Shared, read-only utilities and pre-registration for the R-93 round (08-22):
Wikipedia "Bitcoin" article pageviews as the twelfth structurally distinct
INFO-axis signal this project has tried.

IDEA IN ONE SENTENCE. Daily English Wikipedia pageviews for the "Bitcoin"
article are a retail-ATTENTION/demand proxy -- independent of any exchange,
chain or derivatives venue -- fetched via `scripts/fetch_wikipedia_pageviews.py`
from the Wikimedia Foundation's free, unauthenticated Pageviews REST API and
loaded via `tradebot.data.load_wikipedia_pageviews`/`align_wikipedia_causal`.

CITATIONS.
- Da, Z., Engelberg, J. & Gao, P. (2011), "In Search of Attention",
  *Journal of Finance* 66(5), 1461-1499. Abnormal Google search volume
  (their SVI) predicts short-horizon RETURN CONTINUATION for retail-attention
  stocks, but also produces a partial REVERSAL over the following weeks --
  their own paper reports both a momentum and a mean-reverting consequence
  of the same attention shock, which is why this round's two branches test
  opposite readings of the same underlying data rather than two unrelated
  ideas.
- Kristoufek, L. (2013), "BitCoin meets Google Trends and Wikipedia:
  Quantifying the relationship between phenomena of the Internet era",
  *Scientific Reports* 3, 3415. The first paper to apply search/pageview
  attention specifically to Bitcoin: finds bidirectional Granger causality
  between search/pageview interest and price, and specifically that periods
  of the largest positive price DEVIATION from a fundamental trend
  (bubble-like run-ups) are preceded and accompanied by extreme pageview/
  search spikes that historically marked exhaustion rather than continuation
  -- the citation the novel (contrarian) branch below is built on.

WHICH CONSTRAINT THIS ATTACKS: INFO. Retail attention is not reconstructed
from price/volume (the L-14/15/16 ruling on order-flow proxies does not
apply -- pageviews are a genuinely external human-behavior signal, not a
price transform) and is not a duplicate of any of the eleven INFO signals
already closed NEGATIVE in this project: on-chain network activity/hash rate
(B-07/R-44), VIX/DXY macro spillover (R-53), MVRV valuation/cost-basis
(R-74), aggregate USDT stablecoin supply (R-54/R-55/R-58, five mechanism
variants), Deribit DVOL implied volatility (R-73), halving-cycle phase
(R-79), day-of-week/session timing (R-75), Binance futures positioning --
open interest and top-trader long/short ratio (R-81), raw traded volume
(R-84, two architectures), Binance taker buy/sell order-flow imbalance
(R-88, two architectures). All eleven describe either the traded asset's
own on-chain/exchange/derivatives state or the macro financial system;
Wikipedia pageviews describe how many humans are reading about Bitcoin on a
given day, sourced from neither a crypto venue nor a financial data
provider. "Another indicator" framing does not apply for the same reason it
did not apply to R-84's raw volume: this is a materially different *kind*
of information, not a retune of price/volume/vote parameters.

DATA COVERAGE, verified before any branch was dispatched (see
`scripts/fetch_wikipedia_pageviews.py`'s own docstring): the Pageviews API's
documented data start is 2015-07-01, before this project's 2017-01-01
dataset start. Spot-checked directly against the raw committed file: ZERO
missing days across the full 2015-07-01 -> 2026-08-20 range, and ZERO NaN
bars after `align_wikipedia_causal` onto the full 2017-01-01 -> 2026-08
5-minute BTC series (1,010,889 bars). This is the cleanest coverage any
INFO-axis signal in this project has had -- even R-84's raw volume, the
previous best, only matched this project's OWN 2017-01-01 start rather than
predating it -- and is the first INFO round able to use the FULL six-episode
table with no forced-fail/truncated-coverage caveat on any episode.

NOT SIMULABLE BEYOND THIS: pageviews are daily, not 5-minute; every feature
built from them is causally aligned via `align_wikipedia_causal` (shift by
1 day, ffill, never back-cast), exactly like the on-chain/macro/stablecoin
loaders already in `tradebot.data`. No order book, no queue model, no
information this project cannot already simulate.

This module is read-only utility, written by the operator before dispatch
(same convention as r84_shared.py/r88_shared.py). Neither branch edits it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.data import align_wikipedia_causal, load_wikipedia_pageviews

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated, not imported
# (same convention as every prior INFO-axis shared.py).
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch until Step 4's pre-registered decision, if reached.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# The full R-82/R-83/R-84 six-episode table -- usable here, as it was for
# R-84's raw volume, because pageviews have no external coverage-start
# caveat (present for the full committed 2017-01 -> 2026-08 history).
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]


# ----------------------------------------------------------------- v4 vote

def anchor_votes(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                  band: float = V4_BAND) -> list[pd.Series]:
    """The three latched 0/1 anchor votes `kelly_regime`/`_v3`/`_v4` use internally.

    Causal: row i depends only on rows <= i (rolling mean + ffill/latch).
    """
    close = df["close"]
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=df.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return votes


def anchor_majority(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                     band: float = V4_BAND) -> pd.Series:
    """`frac` = mean of the three anchor votes, in {0, 1/3, 2/3, 1} -- v4's
    own gate, exactly, for use as the Step-A comparison baseline."""
    votes = anchor_votes(df, horizons, band)
    return sum(votes) / len(votes)


def confirming_vote_frac(anchor_sum: np.ndarray, meta_vote: np.ndarray,
                          weight: float) -> np.ndarray:
    """R-53/R-55's combination rule.

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``

    ``anchor_sum`` in [0, 3], ``meta_vote`` in {0, 1} (keep it DISCRETE so
    the formula can still reach exactly flat/exactly full -- R-80's lesson).
    ``weight == 0`` recovers `kelly_regime_v4` exactly -- the required
    identity-recovery check every confirming-vote round has run.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


# ------------------------------------------------------------ pageview features

def load_pageviews_5m(data_dir) -> pd.DataFrame | None:
    """Daily pageviews causally aligned onto the 5-minute BTC bar grid, or
    None if the fetch script has not been run. Callers must still truncate
    to their own step's cutoff (`INNER_VAL_END` in step 3) themselves --
    this loader does not enforce the holdout boundary."""
    from tradebot.data import load_ohlcv_csv
    pv = load_wikipedia_pageviews(data_dir)
    if pv is None:
        return None
    bars = load_ohlcv_csv(f"{data_dir}/btcusd_spot_5m.csv.gz")
    return align_wikipedia_causal(pv, bars)


def attention_z(views_5m: pd.Series, window_days: int = 20) -> pd.Series:
    """Causal log-pageview z-score against its own trailing `window_days`
    mean/std -- the natural unit-free construction given pageviews' own
    secular trend across 2017-2026 (Bitcoin's Wikipedia readership in 2026
    is not comparable in raw level to 2017's; a fixed threshold would
    conflate "unusual right now" with "the encyclopedia has grown"). Same
    treatment as `r84_shared.volume_z` and `r81_shared.crowding_z`.
    """
    log_v = np.log(views_5m.replace(0.0, np.nan).ffill())
    w = int(window_days * BARS_PER_DAY)
    mean = log_v.rolling(w, min_periods=w // 4).mean()
    std = log_v.rolling(w, min_periods=w // 4).std()
    return (log_v - mean) / std.replace(0.0, np.nan)


# ------------------------------------------------------- Step-A lead-time gate

def nearest_transition(anchor_majority_series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp) -> pd.Timestamp | None:
    """First bar in `window` where v4's own anchor-vote flips (crosses 0.5
    from either side), taken as "v4's own reaction" to the episode -- same
    definition R-81/R-84 use."""
    seg = anchor_majority_series.loc[window]
    above = seg > 0.5
    flips = above.ne(above.shift()).fillna(False)
    flips = flips[flips.index >= onset]
    if not flips.any():
        return None
    return flips[flips].index[0]


def nearest_crossing(z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                      thresh: float, direction: str) -> pd.Timestamp | None:
    """First bar in `window`, at or after `onset`, where the z-scored
    attention signal crosses `thresh` in the given direction ('above' or
    'below'). Same definition R-84's `nearest_crossing` uses."""
    seg = z.loc[window]
    seg = seg[seg.index >= onset]
    if direction == "above":
        hit = seg[seg > thresh]
    else:
        hit = seg[seg < thresh]
    if hit.empty:
        return None
    return hit.index[0]


def episode_window(bars: pd.DataFrame, onset_str: str,
                    pre_days: int = 60, post_days: int = 30) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    start = onset - pd.Timedelta(days=pre_days)
    end = onset + pd.Timedelta(days=post_days)
    window = bars.index[(bars.index >= start) & (bars.index <= end)]
    return onset, window


def episode_null_leads(z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                        flip_time: pd.Timestamp, thresh: float, direction: str,
                        n_draws: int = 500, block_days: int = 5, seed: int = 93) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the z-score series within `window` by a random offset (block
    size `block_days`) and recompute the crossing time against the SAME
    fixed `flip_time`, `n_draws` times. Same construction as R-84's
    `episode_null_leads`."""
    rng = np.random.default_rng(seed)
    seg = z.loc[window].dropna()
    if seg.empty:
        return np.array([])
    n = len(seg)
    block = max(1, int(block_days * BARS_PER_DAY))
    leads = np.full(n_draws, np.nan)
    values = seg.values
    idx = seg.index
    for k in range(n_draws):
        shift = rng.integers(0, n)
        shifted = np.roll(values, shift)
        s = pd.Series(shifted, index=idx)
        post_onset = s[idx >= onset]
        if direction == "above":
            hit = post_onset[post_onset > thresh]
        else:
            hit = post_onset[post_onset < thresh]
        if hit.empty:
            continue
        cross_time = hit.index[0]
        leads[k] = (flip_time - cross_time).total_seconds() / 86400.0
    return leads
