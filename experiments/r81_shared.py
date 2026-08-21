"""Shared, read-only utilities for the R-81 crowding-signal round (08-21).

Idea in one sentence: every INFO-axis signal tried so far in this project
-- on-chain activity (B-07/R-44), macro VIX/DXY (R-53/R-54), stablecoin
supply (R-54/R-55/R-58), Deribit DVOL/VRP (R-73), MVRV (R-74), calendar/
session structure (R-75/R-79), the halving cycle (R-79), cross-instrument
pairs (R-76) and `kelly_regime_v4`'s own meta-labeled vote (R-80) -- is
either daily-or-coarser cadence or a transform of price/the vote itself,
and every one of them, when measured, LAGGED the anchor-vote gate rather
than leading it (R-53 median -5.5d; R-73 -2.0/-9.0d; R-74 -4.0/-35.0d).
Binance's futures "metrics" feed -- open interest, top-trader long/short
ratio, taker buy/sell volume ratio -- updates at this project's own
5-minute native cadence, which no prior INFO signal has done, and it
measures a causal, mechanical quantity (how crowded/levered the market
currently is) rather than a claim about future price: forced deleveraging
of a one-sided, over-levered position is a real driver of the next move,
not a correlate of it (Hirshleifer 1988; Kang, Rouwenhorst & Tang
2021/2023, "Crowding and Factor Returns", FAJ 79(1), SSRN 3803954).
Palazzi, Junior & Klotzle (2025, SSRN 6725492) report funding rate + open
interest predicting BTC returns "in every regime" of a 2014-2025 sample --
the direct citation motivating this round.

Not a duplicate of:
- R-35/R-39 (B-05, closed): raw Binance funding ALONE as a binary
  top-decile FLAT GATE (COST axis). This round uses funding jointly with
  open interest and account long/short ratios as a multivariate crowding
  score, fed as a DIRECTIONAL confirming vote or an event-triggered exit
  override (both SIZE/N-approx-3/ERR axis constructions), at 5-minute
  cadence rather than the 8-hour funding-settlement cadence R-35/39 used.
- R-73 (DVOL): closest methodological template (measurement-gate-before-
  strategy; prefer the confirming-vote architecture over a brake) but a
  different data type (derivatives POSITIONING, not options-implied vol)
  and a materially faster native cadence than DVOL's daily update.
- R-53/R-54 (VIX/DXY): same validated confirming-vote combination rule,
  different channel (this market's own derivatives book, not an external
  asset class) and cadence.
- R-80 (meta-labeling): reuses its hard-won lesson (keep the vote
  DISCRETE 0/1, not continuous, so `confirming_vote_frac` retains its
  exact-flat state) but this signal is genuinely external market data,
  not a self-referential function of v4's own hit-rate.
- "Recovering order flow from OHLCV" (ruled out, L-14/15/16/L-12): that
  ruled out reconstructing flow FROM PRICE ALONE (BVC/VPIN). This is not
  a price transform -- it is a separately reported, real exchange feed
  (account positions, open interest), which that ruling does not cover.

This module is read-only utility, written by the operator before dispatch
(same convention as r79_shared.py/r80_shared.py). Neither branch edits it.
Contains: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s 3-anchor
vote construction, so both branches compose against the true incumbent
inputs without importing the registered strategy module (R-54/R-55's
convention: duplicate the combination rule, don't import); (2) the
R-53/R-55 confirming-vote formula; (3) the crowding-metric construction
both branches share (z-scored deviation of top-trader long/short ratio
and open interest from their own trailing baselines); (4) the dated
stress-episode table both branches' Step-A lead-time gates test against;
(5) a block-bootstrap null generator for that gate; (6) shared date
constants and the causality truncation probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.data import align_metrics_causal, load_binance_metrics

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated, not imported.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# Metrics feed coverage (scripts/fetch_binance_metrics.py, verified by
# direct fetch, not guessed): BTC from 2020-09-01, ETH only from
# 2021-12-01 (a DVOL-like short-history caveat -- named up front, not
# discovered after the fact). Both fetched only through INNER_VAL_END:
# this step must not read the holdout, so there was no reason to fetch
# metrics data past it.
METRICS_START = {"BTC": "2020-09-01", "ETH": "2021-12-01"}
METRICS_END = INNER_VAL_END

# Dated stress/regime-transition episodes inside the metrics feed's
# coverage window, the same episode-table discipline R-73/R-74 used
# (there: DVOL's 2021-03-24 start excluded the 2018 bear and the 2020-03
# COVID crash from that round's table; here, BTC metrics' 2020-09-01
# start excludes those same two). Three episodes remain observable
# pre-holdout, honestly n=3 -- on par with this project's usual N-approx-3
# power limitation, stated here rather than discovered after computing
# anything.
STRESS_EPISODES = [
    # (label, onset date the episode is anchored to)
    ("2021-top / 2022-bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]


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
    own gate, exactly, for use as the Step-A lead-time comparison target."""
    votes = anchor_votes(df, horizons, band)
    return sum(votes) / len(votes)


def confirming_vote_frac(anchor_sum: np.ndarray, meta_vote: np.ndarray,
                          weight: float) -> np.ndarray:
    """R-53/R-55's combination rule.

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``

    ``anchor_sum`` in [0, 3], ``meta_vote`` in {0, 1} (per R-80's lesson:
    keep it DISCRETE so the formula can still reach exactly flat/exactly
    full). ``weight == 0`` recovers `kelly_regime_v4` exactly -- the
    required identity-recovery check every confirming-vote round has run.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


def load_crowding_inputs(data_dir, asset: str = "BTC") -> pd.DataFrame | None:
    """Raw Binance metrics for ``asset``, truncated to METRICS_END, or
    ``None`` if the file is absent. Callers are responsible for their own
    ``assert_no_holdout`` re-check after aligning onto bars -- this
    function only enforces the truncation, not the guard."""
    df = load_binance_metrics(data_dir, asset)
    if df is None:
        return None
    cutoff = pd.Timestamp(METRICS_END, tz="UTC") + pd.Timedelta(days=1)
    return df.loc[df.index < cutoff].copy()


def crowding_z(metrics: pd.DataFrame, bars: pd.DataFrame,
               window_days: int = 14) -> pd.DataFrame:
    """Two causal, bar-aligned crowding features from the raw metrics feed.

    - ``ls_z``: top-trader long/short ratio (``sum_toptrader_long_short_ratio``),
      z-scored against its own trailing ``window_days`` mean/std. Positive
      = more long-crowded than recently normal.
    - ``oi_chg_z``: the ROLLING-window log-change in open interest
      (``sum_open_interest``) over the same window, z-scored the same way.
      Positive = open interest expanding faster than recently normal
      (new leverage entering); a sharp NEGATIVE value is the signature of
      an active deleveraging cascade (positions being closed/liquidated).

    Both features are causal by construction (`rolling` only looks
    backward) and use only fields aligned onto ``bars`` via
    ``align_metrics_causal`` (already a same-cadence, no-future-peek
    alignment -- see that function's own docstring).

    Data wrinkle handled here: 127 BTC bars (a reporting gap around the
    2021-05-19/23 crash) carry a literal ``sum_open_interest`` of exactly
    0.0 rather than a missing row -- a bad reading, not a real zero
    open-interest venue. Masked to NaN and forward-filled (same causal
    treatment ``align_metrics_causal`` already gives an outright missing
    timestamp) before the log is taken, so one bad reading cannot poison
    ``log(0) = -inf`` through every subsequent bar in its rolling window.
    """
    aligned = align_metrics_causal(metrics, bars)
    w = int(window_days * BARS_PER_DAY)

    ls = aligned["sum_toptrader_long_short_ratio"]
    ls_mean = ls.rolling(w, min_periods=w // 4).mean()
    ls_std = ls.rolling(w, min_periods=w // 4).std()
    ls_z = (ls - ls_mean) / ls_std.replace(0.0, np.nan)

    oi = aligned["sum_open_interest"].replace(0.0, np.nan).ffill()
    oi_logchg = np.log(oi).diff(w)
    oi_mean = oi_logchg.rolling(w, min_periods=w // 4).mean()
    oi_std = oi_logchg.rolling(w, min_periods=w // 4).std()
    oi_chg_z = (oi_logchg - oi_mean) / oi_std.replace(0.0, np.nan)

    return pd.DataFrame({"ls_z": ls_z, "oi_chg_z": oi_chg_z}, index=bars.index)


def block_bootstrap_lead_null(event_offsets_days: np.ndarray, n_bars: int,
                               block_days: int, n_draws: int,
                               seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n_bars`
    index, for a block-bootstrap null on the Step-A lead-time gate (this
    signal is a level/positioning measure, not a cyclical phase partition
    of a trending series, so the R-79-style placebo-OFFSET null -- built
    specifically to control for spurious trend-confound under an
    arbitrary cyclical partition -- is not the applicable template here;
    a standard block bootstrap of the crowding series against the fixed,
    real anchor-vote flip dates is). Kept separate from any particular
    branch's file so both branches draw from the identical null
    construction and seed.
    """
    rng = np.random.default_rng(seed)
    block = int(block_days * BARS_PER_DAY)
    draws = []
    for _ in range(n_draws):
        shift = int(rng.integers(block, n_bars - block)) if n_bars > 2 * block else int(rng.integers(1, n_bars))
        draws.append((np.arange(n_bars) + shift) % n_bars)
    return draws


def truncation_causality_probe(build_target_fn, df: pd.DataFrame,
                                check_at: int, shorter_by: int = 20_000) -> bool:
    """Standard truncation probe: does `target[check_at]` change if bars
    after it are dropped? Returns True if causal (identical both ways)."""
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))
