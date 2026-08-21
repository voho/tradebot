"""Shared, read-only utilities for the R-88 taker-flow-imbalance round (08-21).

Idea in one sentence: Binance's `sum_taker_long_short_vol_ratio` -- the
aggregated ratio of taker BUY volume to taker SELL volume, at this
project's own 5-minute native cadence -- is a genuine, exchange-reported
order-FLOW signal (which side is currently paying to cross the spread),
distinct from every INFO signal this project has tried so far, including
R-81's own use of the same metrics feed: R-81 built `ls_z` from
`sum_toptrader_long_short_ratio` (a STOCK -- how many large accounts
currently hold long vs. short) and `oi_chg_z` from open-interest change (a
LEVERAGE signal). Taker volume ratio is a FLOW quantity -- the direction of
currently-EXECUTING aggressive trades -- the closest thing this project's
committed data gets to real order flow, and the one column in the metrics
file `docs/LEDGER.md`'s L-14/15/16 ruling (no reconstructing flow from
price alone) does not cover, because it is not reconstructed: it is
reported directly by the venue.

Citations (both verified to exist via WebFetch before being relied on):
- Vafin (2026), SSRN 6938742 (posted 2026-06-14) -- order-flow imbalance
  and short-horizon crypto return predictability, with an explicit
  transaction-cost model and data-snooping control.
- Bieganowski & Ślepaczuk (2026), arXiv:2602.00776 -- order-flow imbalance
  and adverse-selection cost driving short-horizon price moves across
  Binance-futures instruments, with feature effects stable across assets.

Not a duplicate of:
- R-81 (crowding/positioning, closed NEGATIVE): different economic
  quantity (a stock of who is currently long/short vs. a flow of who is
  currently trading which direction) and, measured directly against the
  raw file, materially better coverage on the exact episode R-81's own
  signal could not see (`sum_toptrader_long_short_ratio` is 37.6% missing,
  blanketing the entire FTX window; `sum_taker_long_short_vol_ratio` is
  15.2% missing on BTC and 32.7% on ETH, and its only two BTC gaps >= 1 day
  both fall in Dec 2021 - May 2022, so the FTX episode window is fully
  covered -- verified directly against the raw file, not assumed).
- R-84 (raw traded volume, closed NEGATIVE): a magnitude-only signal
  (how MUCH traded, no direction) reconstructed from the OHLCV file's own
  sixth column. This is a DIRECTIONAL flow ratio, reported by the venue,
  not derived from price/volume co-movement.
- L-14/L-15/L-16 (BVC/VPIN order-flow reconstruction from price alone,
  ruled out): that ruling covers signals derived FROM price. This is not a
  price transform -- it is a separately reported exchange feed.
- R-53/R-55 (confirming-vote architecture): reused verbatim here (the
  validated combination rule), applied to a new channel.
- B-24/R-77 (execution-model rounds: patient-limit N-sweep, regime-
  adaptive urgency): both are VOLATILITY-driven execution timing. The
  novel branch here conditions execution on FLOW DIRECTION, a mechanism
  neither tried, and attacks COST (adverse-selection cost of trading
  against strong contrary flow) rather than INFO/regime-timing.

DISCLOSED COVERAGE CAVEAT, named before any number below was computed: BTC
`sum_taker_long_short_vol_ratio` has a second gap, 2022-01-31 -> 2022-05-09,
which ends EXACTLY on the Terra/Luna episode's onset date (2022-05-09).
The pre-episode 60-day baseline window for that one episode
(2022-03-10 -> 2022-05-09) therefore has NO usable data at all -- unlike
the FTX episode, which is fully clean. This is named here, not discovered
after running the gate; both branches must report the Terra/Luna episode
as a construction-forced FAIL (no crossing computable in its baseline
window) rather than silently drop it or search outside the pre-registered
window to rescue it.

This module is read-only utility, written by the operator before dispatch
(same convention as r79_shared.py/r81_shared.py). Neither branch edits it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.data import align_metrics_causal, load_binance_metrics

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated, not imported
# (same convention as r81_shared.py / r84_shared.py).
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# Metrics feed coverage (scripts/fetch_binance_metrics.py). Same as R-81:
# BTC from 2020-09-01, ETH only from 2021-12-01. Both truncated at
# INNER_VAL_END -- this step must not read the holdout.
METRICS_START = {"BTC": "2020-09-01", "ETH": "2021-12-01"}
METRICS_END = INNER_VAL_END

# Dated stress/regime-transition episodes, identical to R-81's table (same
# 3 episodes inside the metrics feed's coverage window). The Terra/Luna
# episode's forced-FAIL caveat is named above, not here, to keep the
# episode table itself a byte-for-byte reusable constant.
STRESS_EPISODES = [
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

    ``anchor_sum`` in [0, 3], ``meta_vote`` in {0, 1} (R-80's lesson: keep it
    DISCRETE so the formula can still reach exactly flat/exactly full).
    ``weight == 0`` recovers `kelly_regime_v4` exactly -- the required
    identity-recovery check every confirming-vote round has run.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


def load_flow_inputs(data_dir, asset: str = "BTC") -> pd.DataFrame | None:
    """Raw Binance metrics for ``asset``, truncated to METRICS_END, or
    ``None`` if the file is absent. Callers re-check `assert_no_holdout`
    themselves after aligning onto bars."""
    df = load_binance_metrics(data_dir, asset)
    if df is None:
        return None
    cutoff = pd.Timestamp(METRICS_END, tz="UTC") + pd.Timedelta(days=1)
    return df.loc[df.index < cutoff].copy()


def taker_flow_z(metrics: pd.DataFrame, bars: pd.DataFrame,
                  window_days: int = 14) -> pd.Series:
    """Causal, bar-aligned taker-flow-imbalance z-score.

    ``sum_taker_long_short_vol_ratio`` (taker BUY volume / taker SELL
    volume) z-scored against its own trailing ``window_days`` mean/std.
    Positive = more buy-flow-imbalanced than recently normal; negative =
    more sell-flow-imbalanced. Causal by construction (`rolling` only
    looks backward), aligned via `align_metrics_causal` (no future peek).
    """
    aligned = align_metrics_causal(metrics, bars)
    w = int(window_days * BARS_PER_DAY)

    tv = aligned["sum_taker_long_short_vol_ratio"]
    tv_mean = tv.rolling(w, min_periods=w // 4).mean()
    tv_std = tv.rolling(w, min_periods=w // 4).std()
    tv_z = (tv - tv_mean) / tv_std.replace(0.0, np.nan)
    return tv_z.rename("tv_z")


def block_bootstrap_lead_null(n_bars: int, block_days: int, n_draws: int,
                               seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n_bars`
    index, for a block-bootstrap null on the Step-A lead-time gate
    (identical construction to r81_shared.py's, duplicated here so this
    round has no import dependency on a prior round's experiment file)."""
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
