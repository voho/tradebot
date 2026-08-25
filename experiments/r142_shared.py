"""Shared, read-only utilities for the R-142 futures TERM-STRUCTURE-SLOPE
round (08-25).

Idea in one sentence: R-120 (08-24, NEGATIVE, ruled out -- see
docs/LEDGER.md) tested the Deribit front-quarter calendar basis's own
LEVEL and MOMENTUM (time-derivative of one point on the curve) against
spot. Neither branch ever looked at the CURVE's own SHAPE: the two
nearest quarterlies are simultaneously listed and simultaneously
traded for roughly six months at every roll (verified empirically below),
so the SLOPE between them -- front-quarter annualized basis minus
next-quarter annualized basis -- is a cross-sectional quantity a single
maturity point cannot express, structurally distinct from both of
R-120's statistics the same way a yield curve's slope is a different
object from its level or its day-over-day change.

Citations (checked to exist; see docs/RESEARCH.md conventions):
- Erb & Harvey (2006), "The Strategic and Tactical Value of Commodity
  Futures", Financial Analysts Journal 62(2) -- term-structure basis as a
  commodity risk-premium signal (R-120's own citation, reused here for
  the shared curve-construction background, not re-cited for a new claim).
- Bianchi, Fan, Miffre & Zhang (2023), "Exploiting the dynamics of
  commodity futures curves", Journal of Banking & Finance (arXiv
  2308.00383) -- Nelson-Siegel-decomposed LEVEL, SLOPE and CURVATURE of
  commodity futures curves are separately profitable, uncorrelated
  factors; slope-momentum strategies "generate significant profits...
  unrelated to previously documented risk factors" including the LEVEL
  (basis) factor R-120 already tested here. This is the direct citation
  for why slope is not a re-parameterization of R-120's level/momentum.
- Schmeling, Schrimpf & Todorov (2023, rev. 2025), "Crypto carry", BIS
  Working Papers No. 1087 -- crypto futures basis literature background
  (R-120's own citation).
- Chi et al. (2023), "An empirical investigation on risk factors in
  cryptocurrency futures", Journal of Futures Markets -- basis/momentum/
  basis-momentum as the three persistent crypto-futures factors (R-120's
  own citation; note this paper's own "basis-momentum" is a single
  maturity's own time-derivative, still a LEVEL-derived statistic, not a
  cross-maturity slope -- the distinction this round's own citation trail
  rests on).
- Context only, NOT used to select or motivate the Step-A gate's episode
  list (see the discipline note below): CF Benchmarks (2025), "Revisiting
  the Bitcoin Basis", and contemporaneous reporting (CoinDesk, 2025-11-18
  and 2025-12-03) on BTC futures backwardation preceding the 2025-11-21
  local bottom. This round's own literature search (WebSearch, this
  session) surfaced this real, very recent (within-holdout) example of
  curve inversion accompanying a market bottom -- exactly the kind of
  episode this mechanism is meant to catch. IT IS DELIBERATELY NOT ADDED
  to STRESS_EPISODES_FULL below or to either branch's falsification test:
  having read a news account of this specific instance before writing any
  gate code makes it exactly the kind of after-the-fact episode selection
  R-81 warned against ("any direction" transitions silently picking a
  spurious blip) and the routine's own Step 2 discipline over "what would
  make it fail, named in advance" forbids. The pre-registered gate below
  uses only the SAME six/four/three-episode table R-53 through R-141 have
  used throughout this project, chosen years before this session existed.

NOT A DUPLICATE OF:
- R-120 (conservative: basis LEVEL; novel: basis MOMENTUM) -- both are
  single-maturity (front-quarter-only) statistics; this round's slope
  needs a SECOND, simultaneously-listed maturity R-120's own module never
  reads. R-120's own conservative-branch docstring explicitly scopes
  itself to "front-quarter" and its shared module's own coverage note
  never mentions a second contract.
- R-41/kelly_regime_v9_basis_lead (spot-vs-PERPETUAL basis): different
  instrument entirely (no fixed expiry, funding-reset every 8h).
- Every prior SIZE-axis dampener (R-59, R-97, R-102, R-109, R-123, R-125,
  R-141, 28+ total) -- none has used a futures-curve-shape input; all
  transform either the vote, the volatility estimate, or a model-derived
  confidence/hazard indicator computed from spot alone.
- R-141's LPPLS dampener specifically: that construction's kappa
  calibration was EQUALITY mean-exposure-matched against a ONE-SIDED
  (damp <= 1 always) multiplier, which is why it was mathematically
  forced to kappa=0 (R-141, ruled out; see docs/LEDGER.md). This round's
  own novel branch is deliberately TWO-SIDED (damp can be above or below
  1) and deliberately NOT equality-calibrated -- kappa is swept on a fixed
  pre-registered grid and every grid point is reported (a B3-style
  plateau sweep), exactly to avoid re-running into the identical trap.
  See kappa_grid_response() below.

DISCLOSED INFRASTRUCTURE WORK THIS ROUND REQUIRED, done BEFORE any
gate/backtest number was computed: `data/btcusd_deribit_quarterly_5m.csv.gz`
and `data/ethusd_deribit_quarterly_5m.csv.gz` as fetched for R-120 stopped
at 2023-03-31 -- three months into the holdout (OOS_START=2023-01-01) --
which would have made this round's own novel (SIZE-axis, holdout-requiring)
branch unevaluable past Q1 2023, the same coverage-ceiling defect R-135
found for `hedge_experts`'s DVOL/positioning experts. Re-ran
`scripts/fetch_deribit_quarterly_futures.py` (unmodified, same
`--first-expiry` as the original R-120 invocation, `--last-expiry` extended
to 2026-12-31) for both assets before writing any code in this file, so the
full holdout through the present is now covered. See the module-level
COVERAGE constants below for the exact measured coverage after re-fetch,
recorded once, before any gate number was computed.

This module is read-only utility, written by the operator before dispatch
(same convention as r81_shared.py / r84_shared.py / r120_shared.py).
Neither branch edits it.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# kelly_regime_v4's own anchor ladder and band -- duplicated, not imported
# (this project's no-cross-round-import convention; see r120_shared.py).
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in step 3; the novel branch's Step-4 holdout
# consultation is pre-registered separately, after both branches' Step-3
# results are in (see r142_novel's own module docstring).
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

QUARTERLY_FILES = {
    "BTC": "btcusd_deribit_quarterly_5m.csv.gz",
    "ETH": "ethusd_deribit_quarterly_5m.csv.gz",
}

# Byte-for-byte R-82/R-83/R-84/R-85/R-120's own STRESS_EPISODES table,
# duplicated per this project's no-cross-round-import rule. NOT edited to
# add the 2025-11 episode named in the module docstring's discipline note.
STRESS_EPISODES_FULL = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]


def load_deribit_quarterly(data_dir: str | Path, asset: str = "BTC") -> pd.DataFrame | None:
    """Raw multi-contract quarterly futures bars, or None if not fetched.

    Columns: ``instrument`` (str), ``expiry`` (Timestamp, UTC midnight),
    ``open/high/low/close/volume``. Index: bar timestamp (UTC). Multiple
    rows CAN share a timestamp (two or more quarterlies simultaneously
    listed) -- callers must select per-contract, never average.
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


def dual_quarter_slope(spot: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Causal front- and next-quarter annualized basis, and their SLOPE.

    For each spot bar at time t: among all quarterly contracts with
    ``expiry > t`` (or ``== t``'s own settlement bar) that are ALREADY
    LISTED at t (their own real data exists at or before t), rank by
    expiry ascending. "front" is the nearest-to-expire listed contract,
    "next" is the second-nearest. Each contract's own price series is
    joined onto ``spot`` causally (``merge_asof``, direction="backward"),
    so a bar before that specific contract's real listing date is NaN for
    it, never back-filled from another contract or from spot (this
    project's "never proxy unavailable data out of price" rule).

    Column selection is fully vectorized and row-local: for row i, take
    the non-NaN columns of a (bars x contracts) matrix, in expiry order,
    and keep the first two. This depends only on row i's own data, so it
    is causal by construction (verified by `causal_truncation_probe`
    below, not merely assumed).

    Returns a DataFrame on ``spot.index`` with columns:
      - ``front_close``, ``next_close``: each contract's own close (NaN
        if fewer than 1 or 2 contracts are listed at that bar)
      - ``front_dte``, ``next_dte``: calendar days to that contract's expiry
      - ``ann_basis_front``, ``ann_basis_next``: ``log(contract_close /
        spot_close) * (365.25 / max(dte, 3))`` -- R-120's own annualization
        convention (3-day floor on days-to-expiry), reused verbatim.
      - ``slope``: ``ann_basis_next - ann_basis_front`` -- positive means
        the curve steepens further out (typical contango-getting-steeper);
        negative means the front is relatively more expensive than the
        next point (an inversion/kink), NaN whenever fewer than 2
        contracts are simultaneously listed.
    """
    idx = spot.index
    spot_close = spot["close"].reindex(idx).to_numpy()
    expiries = sorted(quarterly["expiry"].unique())

    close_cols: dict = {}
    dte_cols: dict = {}
    for expiry in expiries:
        contract = quarterly.loc[quarterly["expiry"] == expiry, ["close"]].sort_index()
        if contract.empty:
            continue
        joined = pd.merge_asof(
            pd.DataFrame(index=idx).reset_index().rename(columns={"index": "timestamp"}),
            contract.reset_index().rename(columns={"timestamp": "timestamp"}),
            on="timestamp", direction="backward",
        ).set_index("timestamp")["close"]
        first_ts = contract.index.min()
        valid = (idx >= first_ts) & (idx <= expiry)
        close_cols[expiry] = joined.where(valid).to_numpy()
        dte_cols[expiry] = np.where(valid, (expiry - idx).days.to_numpy().astype(float), np.nan)

    if not close_cols:
        n = len(idx)
        nan = np.full(n, np.nan)
        return pd.DataFrame({"front_close": nan, "next_close": nan, "front_dte": nan,
                              "next_dte": nan, "ann_basis_front": nan, "ann_basis_next": nan,
                              "slope": nan}, index=idx)

    # Columns already in expiry-ascending order (dict preserves insertion
    # order, and `expiries` was sorted before the loop).
    close_arr = np.column_stack(list(close_cols.values()))
    dte_arr = np.column_stack(list(dte_cols.values()))
    n = close_arr.shape[0]
    rows = np.arange(n)

    isnan = np.isnan(close_arr)
    # Stable sort on the boolean mask puts non-NaN (False=0) columns
    # first, preserving their original (expiry-ascending) relative order
    # among ties -- so order[:, 0] is the nearest-expiry listed contract,
    # order[:, 1] the second-nearest, purely from row i's own data.
    order = np.argsort(isnan, axis=1, kind="stable")

    front_idx = order[:, 0]
    front_valid = ~isnan[rows, front_idx]
    front_close = np.where(front_valid, close_arr[rows, front_idx], np.nan)
    front_dte = np.where(front_valid, dte_arr[rows, front_idx], np.nan)

    if close_arr.shape[1] >= 2:
        next_idx = order[:, 1]
        next_valid = ~isnan[rows, next_idx]
        next_close = np.where(next_valid, close_arr[rows, next_idx], np.nan)
        next_dte = np.where(next_valid, dte_arr[rows, next_idx], np.nan)
    else:
        next_close = np.full(n, np.nan)
        next_dte = np.full(n, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ann_basis_front = np.log(front_close / spot_close) * (365.25 / np.clip(front_dte, 3.0, None))
        ann_basis_next = np.log(next_close / spot_close) * (365.25 / np.clip(next_dte, 3.0, None))
    slope = ann_basis_next - ann_basis_front

    return pd.DataFrame({
        "front_close": front_close, "next_close": next_close,
        "front_dte": front_dte, "next_dte": next_dte,
        "ann_basis_front": ann_basis_front, "ann_basis_next": ann_basis_next,
        "slope": slope,
    }, index=idx)


def slope_zscore(slope: pd.Series, window_days: int = 20) -> pd.Series:
    """Causal rolling z-score of the slope, `window_days` matching v4's
    own fastest anchor (20 days) -- R-120's own window-choice convention
    for its `basis_z`, reused here for the same reason."""
    window = int(window_days * BARS_PER_DAY)
    mean = slope.rolling(window, min_periods=window // 2).mean()
    std = slope.rolling(window, min_periods=window // 2).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (slope - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def anchor_votes(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                  band: float = V4_BAND) -> list[pd.Series]:
    """The three latched 0/1 anchor votes `kelly_regime`/`_v3`/`_v4` use
    internally. Causal: row i depends only on rows <= i."""
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
    """`frac` = mean of the three anchor votes, in {0, 1/3, 2/3, 1}."""
    votes = anchor_votes(df, horizons, band)
    return sum(votes) / len(votes)


def confirming_vote_frac(anchor_sum: np.ndarray, meta_vote: np.ndarray,
                          weight: float) -> np.ndarray:
    """R-53/R-55's combination rule, reused verbatim (duplicated, not
    imported, per this project's convention).

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``
    ``weight == 0`` recovers `kelly_regime_v4` exactly.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


def nearest_transition(anchor_frac: np.ndarray, index: pd.DatetimeIndex,
                        onset: pd.Timestamp, window_days: int,
                        direction: str = "down") -> pd.Timestamp | None:
    """The anchor-gate 'flip' nearest ``onset`` within +/- window_days.
    Reused verbatim from R-81/R-120's disclosed, bug-fixed convention:
    "flip" means a transition in ``direction`` only, never "any
    direction"."""
    lo = onset - pd.Timedelta(days=window_days)
    hi = onset + pd.Timedelta(days=window_days)
    mask = (index >= lo) & (index <= hi)
    if not mask.any():
        return None
    seg = pd.Series(anchor_frac, index=index)[mask]
    diffs = seg.diff()
    cross = diffs < -1e-9 if direction == "down" else diffs > 1e-9
    candidates = seg.index[cross]
    if len(candidates) == 0:
        return None
    return min(candidates, key=lambda t: abs((t - onset).total_seconds()))


def block_bootstrap_lead_null(n_bars: int, block_days: int, n_draws: int,
                               seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n_bars`
    index, for a block-bootstrap null on the Step-A lead-time gate
    (identical construction to r81_shared.py/r84_shared.py/r120_shared.py,
    duplicated here)."""
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
    search window falls entirely at/after ``coverage_start``."""
    out = []
    for label, onset in STRESS_EPISODES_FULL:
        lo = pd.Timestamp(onset, tz="UTC") - pd.Timedelta(days=window_days)
        if lo >= coverage_start:
            out.append((label, onset))
    return out


# Pre-registered kappa grid for the novel branch's SIZE-axis dampener --
# frozen here, before any backtest number, so neither branch can silently
# widen or narrow it after seeing a result. kappa=0 is the identity check;
# each grid point is a B3 plateau cell, none is "the" selected candidate.
NOVEL_KAPPA_GRID = (0.0, 0.10, 0.20, 0.30)

# Disclosed coverage measurement, run BEFORE any gate/backtest number in
# either branch (docs/ROUTINE.md step 2's "disclosed coverage caveat, not
# a gate number" convention -- identical in kind to r120_shared.py's own
# DISCLOSED COVERAGE CAVEAT, re-measured here because `slope` needs TWO
# simultaneously-listed contracts, a strictly stronger requirement than
# R-120's front-quarter-only `ann_basis`). Measured directly against
# `dual_quarter_slope` on the re-fetched (through 2026-12-31) data, once,
# by the operator, before dispatch:
#   BTC: slope is non-NaN for 79.50% of all spot bars, first continuously
#     non-NaN from 2018-12-21, and 100.00% of the holdout (>= OOS_START)
#     -- the coverage-extension fetch closed R-120's own truncation at
#     2023-03-31 entirely; the novel branch's holdout consultation is no
#     longer coverage-limited.
#   ETH: slope is non-NaN for 85.00% of all spot bars, first continuously
#     non-NaN from 2020-04-24, and 100.00% of the holdout.
# Applying the identical [onset-60d, onset+60d]-window-must-fall-entirely-
# at-or-after-coverage-start rule R-120 used (see `usable_episodes` above)
# to these coverage-start dates yields the SAME usable-episode subsets
# R-120 already published and froze: BTC 4/6 (COVID, 2021-11 top,
# Terra/Luna, FTX), ETH 3/6 (2021-11 top, Terra/Luna, FTX). Adopted
# verbatim rather than re-derived with new names, since the sets agree
# exactly and R-81's majority-of-usable convention is already
# established for this exact subset:
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
# Majority-of-usable pass bar (R-81/R-120's convention when the full
# six-episode table is coverage-truncated): >=3 of 4 for BTC's Step-A gate.
MIN_EPISODES_PASS_BTC = 3
