"""Shared, read-only utilities and pre-registration for the R-95 round
(08-22): the Crypto Fear & Greed Index (alternative.me) as the thirteenth
structurally distinct INFO-axis signal this project has tried.

IDEA IN ONE SENTENCE. alternative.me's daily Fear & Greed Index (FGI) is a
proprietary MULTI-SOURCE COMPOSITE crowd-sentiment score (0=Extreme Fear,
100=Extreme Greed; publicly documented weights: Volatility 25%, Market
Momentum/Volume 25%, Social Media 15%, Surveys 15%, Bitcoin Dominance 10%,
Google Trends 10%) -- structurally different in KIND from every one of the
twelve prior INFO signals, which are each a single raw metric from one data
domain, because FGI blends several of those domains (plus survey and social
data this project cannot fetch individually) into one third-party number.
Fetched via ``scripts/fetch_fear_greed_index.py`` from alternative.me's free,
unauthenticated public API and loaded via
``tradebot.data.load_fear_greed_index``/``align_fear_greed_causal``.

CITATIONS.
- Baker, M. & Wurgler, J. (2006), "Investor Sentiment and the Cross-Section
  of Stock Returns," *Journal of Finance* 61(4), 1645-1680. Sentiment is a
  CONTRARIAN predictor: broad sentiment waves correct, so extreme optimism
  (greed) predicts LOWER subsequent returns and extreme pessimism (fear)
  predicts HIGHER ones -- particularly for hard-to-arbitrage assets. The
  citation the novel (contrarian) branch below is built on.
- He, M., Shen, L., Zhang, Y. & Zhang, Y. (2023), "Predicting cryptocurrency
  returns for real-world investments: A daily updated and accessible
  predictor," *Finance Research Letters* 58(PA). Using the same
  alternative.me FGI series this round fetches, finds significant
  in-sample AND out-of-sample return predictability at 1-day-to-1-week
  horizons, and economically material gains net of risk aversion -- the
  citation the conservative (continuation) branch below is built on, and
  the reason this is not a settled null before any number is computed.
- Anonymous / ScienceDirect (2026), "Do bitcoin returns move sentiment?
  Evidence from the crypto fear & greed index" (VAR model, 2018-2025 daily
  BTC data, the *same* FGI series this round fetches). Finds the opposite:
  FGI does NOT Granger-cause returns and adds no out-of-sample forecasting
  gain, while returns DO Granger-cause FGI -- sentiment is reactive, not
  predictive, on this exact signal. This is the round's own named risk
  (item 3 below), not suppressed: two 2023/2026 papers using the identical
  data source disagree about its own headline claim, which is exactly the
  situation a Step-A measurement gate exists to resolve empirically rather
  than by literature vote.

WHICH CONSTRAINT THIS ATTACKS: INFO. FGI is not reconstructed from this
project's own price/volume (the L-14/15/16 ruling on order-flow proxies
does not apply) and is not a duplicate of any of the twelve INFO signals
already closed NEGATIVE in this project: on-chain network activity/hash
rate (B-07/R-44), VIX/DXY macro spillover (R-53), aggregate USDT stablecoin
supply (R-54/R-55/R-58, five mechanism variants), Deribit DVOL implied
volatility (R-73), MVRV valuation/cost-basis (R-74), day-of-week/session
timing (R-75), Binance futures positioning -- open interest and top-trader
long/short ratio (R-81), raw traded volume (R-84, two architectures),
Binance taker buy/sell order-flow imbalance (R-88, two architectures),
Bitcoin halving-cycle phase (R-79), Wikipedia "Bitcoin" pageview attention
(R-94, two architectures). All twelve are single-domain raw metrics; FGI is
a third-party composite blending several of those domains (and survey/
social data none of them cover individually) into one proprietary number.
"Another indicator" framing does not apply for the same reason it did not
apply to R-94's pageviews or R-84's raw volume: this is a materially
different *construction* of information, not a retune of price/volume/vote
parameters, and its own literature is genuinely split on whether it leads
or lags -- unlike several prior signals where the literature already
predicted a lag before any number was run.

DATA COVERAGE, verified before any branch was dispatched (see
`scripts/fetch_fear_greed_index.py`'s own docstring, cross-checked directly
against the raw committed file): 3,121 daily rows, 2018-02-01 -> 2026-08-22,
with exactly two short gaps (2018-04-13->17, a real 3-day gap; and
2024-10-25->27, entirely inside the holdout and therefore irrelevant to
this round). The API's own history starts AFTER this project's 2017-01-01
dataset start and after the first (2018-01-17) stress episode in the
project's standing six-episode table -- so, like DVOL (2021-03-24 start)
and Binance futures positioning (2020-09-01 BTC start), that one episode is
a disclosed, automatic Step-A coverage fail, not a measured pass or fail;
the other five episodes (2018-12-15 onward) are fully covered.

NOT SIMULABLE BEYOND THIS: FGI is daily, not 5-minute; every feature built
from it is causally aligned via `align_fear_greed_causal` (shift by 1 day,
ffill, never back-cast), exactly like the on-chain/macro/stablecoin/DVOL/
MVRV/pageview loaders already in `tradebot.data`. No order book, no queue
model, no information this project cannot already simulate.

This module is read-only utility, written by the operator before dispatch
(same convention as r84_shared.py/r88_shared.py/r94_shared.py). Neither
branch edits it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.data import align_fear_greed_causal, load_fear_greed_index

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

# The full R-82/R-83/R-84/R-94 six-episode table. Episode 1 (2018-01-17) is
# a disclosed, automatic coverage fail for this signal (FGI starts
# 2018-02-01) -- included in the table for comparability with every prior
# round, scored as a forced FAIL, never silently dropped.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

FGI_COVERAGE_START = pd.Timestamp("2018-02-01", tz="UTC")


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


# ------------------------------------------------------------ FGI features

def load_fgi_5m(data_dir) -> pd.DataFrame | None:
    """Daily FGI causally aligned onto the 5-minute BTC bar grid, or None if
    the fetch script has not been run. Callers must still truncate to their
    own step's cutoff (`INNER_VAL_END` in step 3) themselves -- this loader
    does not enforce the holdout boundary."""
    from tradebot.data import load_ohlcv_csv
    fgi = load_fear_greed_index(data_dir)
    if fgi is None:
        return None
    bars = load_ohlcv_csv(f"{data_dir}/btcusd_spot_5m.csv.gz")
    return align_fear_greed_causal(fgi, bars)


def fgi_level(value_5m: pd.Series) -> pd.Series:
    """The raw 0-100 index, already unit-free by construction (unlike
    pageviews/volume/on-chain metrics, FGI needs no z-scoring against its
    own trailing history -- the provider already normalizes it to a fixed
    0-100 scale every day, so a fixed threshold is meaningful across the
    whole 2018-2026 span without conflating "unusual today" with a secular
    trend the way a raw count would)."""
    return value_5m.ffill()


# ------------------------------------------------------- Step-A lead-time gate

def nearest_transition(anchor_majority_series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp) -> pd.Timestamp | None:
    """First bar in `window` where v4's own anchor-vote flips (crosses 0.5
    from either side), taken as "v4's own reaction" to the episode -- same
    definition R-81/R-84/R-94 use."""
    seg = anchor_majority_series.loc[window]
    above = seg > 0.5
    flips = above.ne(above.shift()).fillna(False)
    flips = flips[flips.index >= onset]
    if not flips.any():
        return None
    return flips[flips].index[0]


def nearest_crossing(level: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                      thresh: float, direction: str) -> pd.Timestamp | None:
    """First bar in `window`, at or after `onset`, where the FGI level
    crosses `thresh` in the given direction ('above' or 'below'). Same
    definition R-84/R-94's `nearest_crossing` uses."""
    seg = level.loc[window]
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


def episode_coverage_ok(onset_str: str, pre_days: int = 60) -> bool:
    """True iff FGI has data for the whole pre-onset lookback window (i.e.
    the episode is not a forced coverage fail)."""
    onset = pd.Timestamp(onset_str, tz="UTC")
    return (onset - pd.Timedelta(days=pre_days)) >= FGI_COVERAGE_START


def episode_null_leads(level: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                        flip_time: pd.Timestamp, thresh: float, direction: str,
                        n_draws: int = 500, block_days: int = 5, seed: int = 95) -> np.ndarray:
    """Block-bootstrap null lead distribution for one episode: circularly
    shift the FGI level series within `window` by a random offset (block
    size `block_days`) and recompute the crossing time against the SAME
    fixed `flip_time`, `n_draws` times. Same construction as R-84/R-94's
    `episode_null_leads`."""
    rng = np.random.default_rng(seed)
    seg = level.loc[window].dropna()
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
