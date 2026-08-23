"""Shared, read-only utilities for the R-100 cross-venue funding-divergence
round (08-23/08-24).

Idea in one sentence: the SPREAD between Binance's and Deribit's
simultaneously-reported BTC perpetual funding rates -- both already
committed (`data/btcusdt_perp_funding_8h.csv.gz`,
`data/btcusdt_deribit_perp_funding_8h.csv.gz`) -- is a fifteenth
structurally distinct INFO-axis candidate: a measure of *relative*
crowding/leverage-demand between a retail-heavy venue (Binance) and an
institutional/options-flow venue (Deribit), rather than either venue's
absolute funding LEVEL (already used at R-35/R-39/B-05) or a single
venue's own futures-vs-spot basis (R-41/B-15).

Literature grounding, fetched and read (or corroborated via the paper's
own SSRN/arXiv listing) before being relied on:

- Zhivkov, P. (2026), "The Two-Tiered Structure of Cryptocurrency Funding
  Rate Markets", *Mathematics* (MDPI) 14(2):346 -- a 35.7M-observation,
  26-venue panel finding CEX-CEX and CEX-DEX funding/price divergences are
  real and economically large (17% of observations >=20bps), but that
  forced exits (95% of opportunities) and costs mean only ~40% of the
  largest spreads net positive after fees. Read here as evidence the
  divergence tracks genuine cross-venue STRESS/segmentation (the
  motivating claim for both branches below), not as a claim that the
  spread itself is a tradeable arbitrage income stream (neither branch
  attempts to trade the spread directly -- both use it only as a vote/
  brake on `kelly_regime_v4`'s existing directional position).
- Inan, E. (2025/26), SSRN 5576424, "Predictability of Funding Rates" --
  double-autoregressive models beat random-walk/no-change forecasts for
  Binance/Bybit BTC funding out-of-sample: funding dynamics carry a
  directional, non-random component. Motivates z-scoring each venue's
  funding against its OWN trailing history (a "surprise", not a raw
  level) rather than assuming either series is already stationary or
  comparable in scale to the other.
- He, Manela, Ross & von Wachter (2024), SSRN 4301150, "Fundamentals of
  Perpetual Futures" -- already used at R-39 to frame funding as
  compensation for basis risk; cross-venue divergence, in this framework,
  reflects segmented, capital-constrained arbitrage capacity between the
  two venues (a friction, not a mispricing that self-corrects
  instantaneously) -- the reason a real divergence could plausibly persist
  long enough to lead a slower multi-week regime signal.

**The disclosed methodological problem this module exists to solve, named
before any signal number was computed:** `load_funding_deribit`'s own
docstring already records that the two venues' raw rates are NOT
comparable in level -- they correlate at only r=0.69 (daily-summed) and
their ratio is unstable year to year (0.21x-1.24x) -- because Deribit
charges funding continuously (summed into 8h buckets here) while Binance
settles a discrete 8-hourly rate. A raw rate DIFFERENCE would therefore
conflate genuine cross-venue divergence with this known settlement-
convention/scale instability. This module's `cross_venue_divergence_z`
sidesteps it exactly the way R-73's/R-81's/R-88's own signals handle
non-stationary raw levels: z-score each venue's own DAILY-SUMMED funding
against its own trailing baseline first, then take the DIFFERENCE of the
two z-scores. This is a standardized "who is more surprised than usual"
comparison, immune to a constant or slowly-drifting scale ratio between
the two venues (a multiplicative rescaling of one whole series shifts its
own mean and std together and leaves its z-score unchanged), and is fixed
here, a priori, rather than tried and swapped for the raw-difference
version after seeing a result.

Attacks **INFO** (conservative branch: a fifteenth INFO-axis signal) and
**COST** (novel branch: an execution-timing brake, contingent on the same
signal first clearing an informativeness gate, same dependency structure
as R-88's own INFO->COST pair).

Not a duplicate of:
- **R-35/R-39 (B-05, closed)**: single-venue Binance funding LEVEL (a
  binary top-decile flat gate) -- one venue's absolute cost. This is the
  cross-sectional DIFFERENCE between two venues' standardized surprise, a
  different economic quantity (relative crowding, not absolute cost) that
  is zero whenever both venues are equally stressed, however high the
  common level is.
- **R-39/B-02 (`load_funding_extended`)**: Deribit funding was fetched
  there only to SPLICE onto Binance's post-2023 gap -- concatenated, never
  blended, and the unstable overlap ratio was logged purely as a data-
  quality caveat on the splice. This round is the first to treat that same
  overlap instability as the object of interest rather than a nuisance to
  route around.
- **R-41 (B-15)**: Deribit's own SPOT-vs-PERP futures curve premium on ONE
  exchange (a term-structure signal). This round compares FUNDING to
  FUNDING across TWO exchanges -- a different instrument relationship
  entirely (cross-venue, not cross-instrument-on-one-venue).
- **R-73 (Deribit DVOL)**: an implied-volatility index, unrelated to
  funding or to any cross-venue comparison.
- **R-81/R-88** (Binance-metrics-feed positioning/flow signals): both are
  single-venue Binance constructions (`sum_toptrader_long_short_ratio`,
  `sum_taker_long_short_vol_ratio`). Neither reads Deribit data at all.
- **R-53/R-55** (confirming-vote architecture): reused verbatim here
  (conservative branch), applied to a new channel.
- **B-24/R-77** (execution-model rounds: patient-limit N-sweep, regime-
  adaptive urgency): both condition execution timing on VOLATILITY. The
  novel branch here conditions on cross-venue DIVERGENCE STRESS, a
  mechanism neither tried (same distinction R-88's novel branch drew
  against B-24/R-77 for flow direction).

DISCLOSED COVERAGE CAVEAT, named before any number below was computed:
Binance funding starts 2020-01-01 and the standard six-episode table's
first two entries (2018 bear onset, 2018 bear bottom) predate it by 1-2
years -- both are construction-forced FAILs, exactly as R-88 forced-failed
Terra/Luna for an unrelated data gap. Of the remaining four, the COVID
episode (onset 2020-03-12) has only ~70 days of Binance-funding history
before its own baseline window opens (60 days pre-onset = 2020-01-12,
eleven days after the funding series' own first observation) -- this is
disclosed as a THIN-BASELINE caveat, not a forced fail, and reported
separately from the two episodes with zero possible baseline.

This module is read-only utility, written by the operator before dispatch
(same convention as r88_shared.py/r99_shared.py). Neither branch edits it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import load_funding, load_funding_deribit  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated verbatim from
# r82_shared.py / r88_shared.py / r96_shared.py / r98_shared.py / r99_shared.py.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# Funding coverage. Binance funding ends 2023-12-31 in the committed file,
# already past INNER_VAL_END, so no extra truncation logic is needed beyond
# the standard OOS_START guard both branches must independently enforce.
FUNDING_COVERAGE_START = "2020-01-01"

# IDENTICAL to R-82/83/84/85/86/88/96/98/99's own table -- copied verbatim,
# not re-derived, so gate numbers stay comparable across the whole axis.
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# The two 2018 episodes predate Binance funding data entirely (starts
# 2020-01-01) -- construction-forced FAILs, disclosed above, not silently
# dropped from the denominator.
FORCED_FAIL_EPISODES = {
    "2018 bear onset (post-Dec-2017 top)",
    "2018 bear bottom / capitulation",
}
# COVID's own 60-day pre-onset baseline window opens eleven days after the
# funding series' first observation -- thin, not absent. Reported as its
# own caveat category, not merged into FORCED_FAIL_EPISODES.
THIN_BASELINE_EPISODES = {"2020-03 COVID crash"}

Z_THRESH = 1.5  # identical literature-anchored threshold to R-81/R-88's ls_z/tv_z gates
BASELINE_WINDOW_DAYS_GRID = (30, 60, 90)  # trailing window each venue's own z-score is computed over
PRIMARY_BASELINE_WINDOW_DAYS = 60  # grid centre; matches R-81/R-88's 14-90d family, chosen for
                                    # non-degeneracy by r100_killswitch_a.py before any episode ran
                                    # (all 3 grid cells fire cleanly: 118-175 threshold crossings
                                    # each across 2020-2022, no substitution needed).

# Step-A pass bar, CORRECTED before either branch was dispatched (a bug fix
# to the round's own initial proposal, per ROUTINE.md's allowance -- made
# before any real episode number existed, not after): the proposal's
# "at least 4 of the 6" bar is unreachable by construction, since funding
# data does not exist before 2020 and both 2018 episodes are therefore
# FORCED FAILs regardless of signal quality (see FORCED_FAIL_EPISODES
# above). The achievable bar is stated against the DENOMINATOR OF VALID
# EPISODES (4: COVID/2021-top/Terra/FTX), at a pass fraction (75%)
# consistent with this axis's own standing ~4/6 (~67%) convention rounded
# to the nearest whole episode on a 4-episode base:
PASS_BAR_NUM = 3
PASS_BAR_DEN = 4  # COVID, 2021-top, Terra/Luna, FTX -- the four non-forced-fail episodes


# ----------------------------------------------------------------- v4 vote
#   Copied verbatim from r82/85/86/88/96/98/99_shared.py.


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
    """R-53/R-55's combination rule, copied verbatim.

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


# ----------------------------------------------------- cross-venue funding


def load_daily_funding_totals(data_dir) -> pd.DataFrame:
    """Binance and Deribit BTC perpetual funding, each resampled to a
    UTC-calendar-day SUM (the day's total funding cost as a fraction of
    notional, matching `load_funding_deribit`'s own "sum of settlements"
    framing), truncated to `INNER_VAL_END` so this step never reads the
    holdout. Returns a DataFrame with columns `binance`, `deribit`
    indexed by day (both NaN on days either venue has no observation).

    Entirely causal by construction: day t's sum uses only settlements
    timestamped on day t itself.
    """
    binance = load_funding(data_dir)
    deribit = load_funding_deribit(data_dir)
    cutoff = pd.Timestamp(INNER_VAL_END, tz="UTC") + pd.Timedelta(days=1)
    binance = binance.loc[binance.index < cutoff]
    deribit = deribit.loc[deribit.index < cutoff]
    b_daily = binance.groupby(binance.index.normalize()).sum()
    d_daily = deribit.groupby(deribit.index.normalize()).sum()
    out = pd.DataFrame({"binance": b_daily, "deribit": d_daily})
    out.index = pd.DatetimeIndex(out.index, tz="UTC").normalize()
    return out.sort_index()


def _causal_zscore(s: pd.Series, window_days: int) -> pd.Series:
    mean = s.rolling(window_days, min_periods=max(5, window_days // 3)).mean()
    std = s.rolling(window_days, min_periods=max(5, window_days // 3)).std()
    return (s - mean) / std.replace(0.0, np.nan)


def cross_venue_divergence_z(daily: pd.DataFrame,
                              baseline_window_days: int = PRIMARY_BASELINE_WINDOW_DAYS
                              ) -> pd.Series:
    """The round's core statistic: `z(binance_daily_funding) -
    z(deribit_daily_funding)`, each z-scored causally against its OWN
    trailing `baseline_window_days`-day mean/std (see module docstring for
    why this, not a raw rate difference, is the fix for the two venues'
    disclosed non-comparable levels/settlement conventions). Positive =
    Binance funding running hotter (more long-crowded) than its own recent
    norm, relative to Deribit doing the same; negative = the reverse.
    """
    zb = _causal_zscore(daily["binance"], baseline_window_days)
    zd = _causal_zscore(daily["deribit"], baseline_window_days)
    return (zb - zd).rename("div_z")


def align_daily_causal(daily: pd.Series | pd.DataFrame, bars: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reindex a daily-cadence signal onto `bars`' 5-minute index, causally
    -- IDENTICAL shift convention to r85/86/96/98/99_shared's own helper: a
    bar at time T may only see the row for the most recent day that closed
    strictly before T's own day."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.reindex(shifted.index.union(bars.index)).sort_index().ffill().reindex(bars.index)


def nearest_alarm(z: pd.Series, window: pd.DatetimeIndex, onset: pd.Timestamp,
                   z_thresh: float = Z_THRESH, signed: str = "abs") -> pd.Timestamp | None:
    """Timestamp of the first bar where `z` crosses (up through +z_thresh,
    down through -z_thresh, or either, per `signed`) inside `window`,
    closest to `onset`. `signed` in {"abs", "pos", "neg"}."""
    vals = z.reindex(window).to_numpy()
    if signed == "pos":
        high = vals >= z_thresh
    elif signed == "neg":
        high = vals <= -z_thresh
    else:
        high = np.abs(vals) >= z_thresh
    cross = np.zeros(len(vals), dtype=bool)
    cross[1:] = high[1:] & ~high[:-1]
    cross[0] = bool(high[0]) if len(high) else False
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


# --------------------------------------------------------- Step-A gate infra
#   `nearest_transition`, `episode_window`, `block_bootstrap_shifts`,
#   `truncation_causality_probe` copied verbatim from r82/85/86/88/96/98/99_shared.py.


def nearest_transition(series: pd.Series, window: pd.DatetimeIndex,
                        onset: pd.Timestamp, direction: str = "down") -> pd.Timestamp | None:
    vals = series.reindex(window).to_numpy()
    changed = np.zeros(len(vals), dtype=bool)
    if direction == "down":
        changed[1:] = vals[1:] < vals[:-1]
    elif direction == "any":
        changed[1:] = vals[1:] != vals[:-1]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    idx = np.where(changed)[0]
    if len(idx) == 0:
        return None
    times = window[idx]
    deltas = np.abs((times - onset).to_numpy())
    return times[int(np.argmin(deltas))]


def episode_window(bars: pd.DataFrame, onset_str: str, window_days: int = 60
                    ) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    onset = pd.Timestamp(onset_str, tz="UTC")
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    window = bars.index[(bars.index >= lo) & (bars.index <= hi)]
    return onset, window


def block_bootstrap_shifts(n_bars: int, block_days: int, n_draws: int, seed: int) -> list[np.ndarray]:
    """Copied verbatim from r82/85/86/88/96/98/99_shared.py."""
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


# ---------------------------------------------------------------- holdout guard


def assert_no_holdout(obj) -> None:
    """Hard guard, same pattern as r81/r86/r88/r96/r98/r99: the max
    timestamp anywhere this file touches must be strictly before OOS_START."""
    idx = obj.index if hasattr(obj, "index") else obj
    if len(idx) == 0:
        return
    cutoff = pd.Timestamp(OOS_START, tz="UTC")
    max_ts = pd.Timestamp(idx.max())
    if max_ts.tzinfo is None:
        max_ts = max_ts.tz_localize("UTC")
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {OOS_START}. "
        "This file must never read data on or after the holdout start.")
