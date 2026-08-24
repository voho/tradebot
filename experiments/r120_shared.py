"""Shared, read-only utilities for the R-120 calendar-basis (term-structure)
round (08-24/08-25).

Idea in one sentence: Deribit's DATED (quarterly) BTC/ETH futures give this
project a second, genuinely new INFO channel -- calendar (roll-yield)
basis between the front-quarter dated future and spot -- distinct from the
spot-vs-PERPETUAL basis R-41/R-53 already tried, because a dated future
resolves only at its fixed expiry while the perpetual's basis is reset by
funding every 8 hours and is mechanically mean-reverting by construction
(Zhang, T. 2026, "Funding Rate Mechanism in Perpetual Futures", SSRN
6185958). This is the "crypto carry" quantity the literature actually
studies:

Citations (all checked to exist -- see docs/RESEARCH.md conventions):
- Schmeling, Schrimpf & Todorov (2023, rev. 2025), "Crypto carry", BIS
  Working Papers No. 1087 -- documents crypto futures basis averaging
  >10% annualized across venues/maturities, time-varying.
- Chi, Hong, ... (2023), "An empirical investigation on risk factors in
  cryptocurrency futures", Journal of Futures Markets -- basis, momentum
  and basis-momentum are the three persistent cross-sectional factors in
  crypto futures, adapting Erb & Harvey (2006), "The Strategic and
  Tactical Value of Commodity Futures", Financial Analysts Journal 62(2).
- Zhang (2026), SSRN 6185958 -- the funding-reset mean-reversion argument
  above; the structural reason this is a different statistical object
  than R-41's spot-perp basis, not a re-parameterization of it.

Not a duplicate of:
- R-41/kelly_regime_v9_basis_lead (spot-vs-PERPETUAL basis, as a vote-
  latch accelerant): different instrument pair. That basis resets every
  8h at funding settlement; this one resolves only at a fixed quarterly
  expiry. R-41's own file is `experiments/kelly_regime_v9_basis_lead.py`.
- B-05/R-35/R-39 (raw Binance funding rate, a COST-axis flat gate):
  funding is a per-8h payment, not a term-structure/roll-yield quantity.
- R-73 (DVOL level/ROC): implied volatility, not a forward price.
- R-81 (OI + top-trader long/short crowding): positioning/leverage
  stocks, not a priced term-structure quantity.
- R-63/R-76 (cross-instrument, e.g. BTC-vs-ETH spot pairs): those pair
  two DIFFERENT COINS' spot series; this pairs the SAME coin across two
  MATURITIES (spot vs. its own dated future).
- Grepped docs/LEDGER.md, docs/RESEARCH.md, docs/STRATEGIES.md for "term
  structure", "quarterly future", "roll yield", "contango",
  "backwardation", "calendar future", "dated future": zero hits before
  this round.

DISCLOSED COVERAGE CAVEAT, named before any Step-A GATE number was
computed (the coverage measurement itself -- running `front_quarter_basis`
and checking where it is non-NaN -- is not a gate number, it is the same
kind of up-front data-shape check R-73/R-81/R-88 ran before naming their
own usable-episode subset): Deribit's quarterly market is thin before
~2019 (probed empirically -- BTC-29MAR19 has real chart data from
>= 2018-10-01 and none before 2018-06-01, roughly a 6-9 month listed
life). The fetcher (`scripts/fetch_deribit_quarterly_futures.py`) pulled
contracts expiring 2018-12-28 -> 2023-03-31 (BTC, 18 contracts, 18/18
returned data) and 2020-09-25 -> 2023-03-31 (ETH, 11 contracts, 11/11
returned data; ETH's quarterly market opened later -- probed, no data
before 2020-06-01 for ETH-25SEP20). Measured directly against
`front_quarter_basis`: BTC's `ann_basis` is 100% non-NaN from
2019-01-01 onward (front contract BTC-29MAR19 was already listed by
then), giving **4 of 6** usable stress episodes -- COVID (2020-03-12),
2021-top (2021-11-10), Terra/Luna (2022-05-09), FTX (2022-11-08); the two
2018 episodes are unreachable, no quarterly contract existed yet. ETH is
100% non-NaN from 2020-09-01 onward, giving **3 of 6** usable episodes
(2021-top, Terra/Luna, FTX) for the B4 falsification leg. Like R-73's
DVOL and R-81's Binance metrics before it, the pass bar below is
therefore MAJORITY OF USABLE episodes, not literal >=4/6 -- see
`USABLE_EPISODES_BTC`/`MIN_EPISODES_PASS_BTC` below, frozen from this
measurement, not tuned after seeing any lead/lag number.

This module is read-only utility, written by the operator before dispatch
(same convention as r81_shared.py / r84_shared.py / r88_shared.py).
Neither branch edits it. Contains: (1) a raw-quarterly-contract loader;
(2) a causal front-quarter selector and annualized-basis computation;
(3) a byte-for-byte duplicate of kelly_regime_v4's 3-anchor vote
construction; (4) the R-53/R-55 confirming-vote combination rule; (5) the
full six-episode stress table (R-82/R-83/R-84/R-85's own table, reused
verbatim) plus the coverage-truncated subset actually usable this round;
(6) a block-bootstrap null generator for the Step-A lead-time gate,
identical construction to r81_shared.py/r84_shared.py; (7) the causal
truncation probe.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

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

QUARTERLY_FILES = {
    "BTC": "btcusd_deribit_quarterly_5m.csv.gz",
    "ETH": "ethusd_deribit_quarterly_5m.csv.gz",
}

# Full six-episode table, byte-for-byte R-82/R-83/R-84/R-85's own
# STRESS_EPISODES (docs/ROUTINE.md's "cite by ID, don't retype a new one"
# convention -- this is the identical construction, duplicated per this
# project's no-cross-round-import rule).
STRESS_EPISODES_FULL = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# Frozen coverage-truncated subsets, measured directly against
# `front_quarter_basis` before any gate/lead-lag number was computed (see
# module docstring's DISCLOSED COVERAGE CAVEAT). BTC front-quarter basis
# is 100% non-NaN from 2019-01-01; ETH from 2020-09-01. Both branches use
# these frozen lists, not a live `usable_episodes()` call, so a
# late-session change to either coverage window cannot silently move the
# episode set after dispatch.
USABLE_EPISODES_BTC = [
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]
USABLE_EPISODES_ETH = [
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]
# Majority-of-usable pass bar (R-81's convention when the full six-episode
# table is coverage-truncated): >=3 of 4 for BTC's Step-A gate.
MIN_EPISODES_PASS_BTC = 3
BTC_BASIS_COVERAGE_START = "2019-01-01"
ETH_BASIS_COVERAGE_START = "2020-09-01"


def load_deribit_quarterly(data_dir: str | Path, asset: str = "BTC") -> pd.DataFrame | None:
    """Raw multi-contract quarterly futures bars, or None if not fetched.

    Columns: ``instrument`` (str), ``expiry`` (Timestamp, UTC midnight),
    ``open/high/low/close/volume``. Index: bar timestamp (UTC). Multiple
    rows CAN share a timestamp (two quarterlies simultaneously listed) --
    callers must select, never average, per :func:`front_quarter_basis`.
    """
    if asset not in QUARTERLY_FILES:
        raise ValueError(f"asset must be one of {sorted(QUARTERLY_FILES)}")
    path = Path(data_dir) / QUARTERLY_FILES[asset]
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    return df.set_index("timestamp").sort_index()


def front_quarter_basis(spot: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Causal front-quarter close + annualized calendar basis, on ``spot``'s index.

    For each spot bar at time t, "front quarter" is the LISTED contract
    whose expiry is the smallest expiry still > t (i.e. the next contract
    to expire) -- not "nearest by calendar date" but "nearest by actual
    resolution date", the standard futures-curve definition (Erb & Harvey
    2006's front-month convention, applied to Deribit's quarterly cycle).
    Within a (previous_expiry, this_expiry] segment, the front contract's
    OWN price series is joined onto spot causally (``merge_asof``,
    direction="backward" -- only bars at or before t, never after), so a
    bar before that specific contract's real listing date is NaN, not
    back-filled from a later contract or from spot itself (this project's
    standing "never proxy unavailable data out of price" rule).

    Returns a DataFrame on ``spot.index`` with columns:
      - ``front_close``: the front-quarter contract's own close (NaN pre-listing)
      - ``days_to_expiry``: calendar days from bar t to that segment's expiry
      - ``calendar_basis``: ``log(front_close / spot_close)``
      - ``ann_basis``: ``calendar_basis * (365.25 / days_to_expiry)`` --
        the annualized roll yield / cash-and-carry basis (Schmeling,
        Schrimpf & Todorov 2023's own annualization convention), clipping
        ``days_to_expiry`` at a 3-day floor so the last trading days
        before expiry (where a tiny basis / tiny denominator would
        otherwise blow up) don't dominate the series.
    """
    expiries = sorted(quarterly["expiry"].unique())
    idx = spot.index
    front_close = pd.Series(np.nan, index=idx)
    days_to_expiry = pd.Series(np.nan, index=idx)

    prev_expiry = pd.Timestamp("1970-01-01", tz="UTC")
    for expiry in expiries:
        seg_mask = (idx > prev_expiry) & (idx <= expiry)
        if not seg_mask.any():
            prev_expiry = expiry
            continue
        seg_idx = idx[seg_mask]
        contract = quarterly.loc[quarterly["expiry"] == expiry, ["close"]].sort_index()
        if contract.empty:
            prev_expiry = expiry
            continue
        joined = pd.merge_asof(
            pd.DataFrame(index=seg_idx).reset_index().rename(columns={"index": "timestamp"}),
            contract.reset_index().rename(columns={"timestamp": "timestamp"}),
            on="timestamp", direction="backward",
        ).set_index("timestamp")
        front_close.loc[seg_idx] = joined["close"].to_numpy()
        days_to_expiry.loc[seg_idx] = (expiry - seg_idx).days
        prev_expiry = expiry

    spot_close = spot["close"].reindex(idx)
    calendar_basis = np.log(front_close / spot_close)
    dte_floor = days_to_expiry.clip(lower=3.0)
    ann_basis = calendar_basis * (365.25 / dte_floor)

    return pd.DataFrame({
        "front_close": front_close,
        "days_to_expiry": days_to_expiry,
        "calendar_basis": calendar_basis,
        "ann_basis": ann_basis,
    }, index=idx)


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


def nearest_transition(anchor_frac: np.ndarray, index: pd.DatetimeIndex,
                        onset: pd.Timestamp, window_days: int,
                        direction: str = "down") -> pd.Timestamp | None:
    """The anchor-gate 'flip' nearest ``onset`` within +/- window_days.

    Reused verbatim from R-81's disclosed, bug-fixed convention
    (`experiments/r81_conservative_crowding_vote.py`'s own
    `nearest_transition`): "flip" means a transition in ``direction``
    only (default "down" -- the gate de-risking), never "any direction",
    which R-81 found could silently pick a spurious opposite-sign blip.
    """
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    mask = (index >= lo) & (index <= hi)
    if not mask.any():
        return None
    seg = pd.Series(anchor_frac, index=index)[mask]
    diffs = seg.diff()
    if direction == "down":
        cross = diffs < -1e-9
    else:
        cross = diffs > 1e-9
    candidates = seg.index[cross]
    if len(candidates) == 0:
        return None
    return min(candidates, key=lambda t: abs((t - onset).total_seconds()))


def block_bootstrap_lead_null(n_bars: int, block_days: int, n_draws: int,
                               seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n_bars`
    index, for a block-bootstrap null on the Step-A lead-time gate
    (identical construction to r81_shared.py's / r84_shared.py's,
    duplicated here per this project's no-cross-round-import convention).
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


def usable_episodes(coverage_start: pd.Timestamp, window_days: int = 60) -> list[tuple[str, str]]:
    """STRESS_EPISODES_FULL entries whose [onset-window, onset+window]
    search window falls entirely at/after ``coverage_start`` -- named
    before any gate number is computed, per this round's disclosed
    coverage caveat above."""
    out = []
    for label, onset in STRESS_EPISODES_FULL:
        lo = pd.Timestamp(onset, tz="UTC") - pd.Timedelta(days=window_days)
        if lo >= coverage_start:
            out.append((label, onset))
    return out
