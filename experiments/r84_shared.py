"""Shared, read-only utilities for the R-84 round (08-21).

Idea in one sentence: every INFO-axis signal tried in this project so far
(nine of them: on-chain activity B-07/R-44, macro VIX/DXY R-53/R-54,
stablecoin supply R-54/R-55/R-58, Deribit DVOL/VRP R-73, MVRV R-74,
calendar/session structure R-75, the halving cycle R-79, Binance futures
crowding R-81, cross-instrument pairs R-76) either came from an EXTERNAL
feed with its own coverage-start date, or was a signed transform of price
alone (BVC/VPIN, L-14/L-15/L-16, ruled out: "a price transform, not order
flow"). **Trading volume is neither.** It is the sixth column of the exact
OHLCV file `kelly_regime_v4` already reads (`data/btcusd_spot_5m.csv.gz`
etc. -- confirmed present, `timestamp,open,high,low,close,volume`), used
raw (unsigned, not BVC-classified into buy/sell), so this round proxies
nothing out of price and has ZERO coverage-gap risk across the full
2017-01 -> 2026-08 history -- the first INFO-axis signal in this project's
history with that property, and the first able to use the FULL six-episode
table (R-82/R-83's set) rather than a truncated one.

Theoretical basis: the Mixture-of-Distributions Hypothesis (Clark 1973,
Econometrica 41(1); Tauchen & Pitts 1983, Econometrica 51(2); Ane & Geman
2000, J. Finance 55(5)) models volume as a proxy for the latent rate of
information arrival, with returns a subordinated process driven by that
same clock. Easley & O'Hara (1992, J. Finance 47(2)) show informed trading
concentrates volume around genuine information events in a sequential-
trade model. Llorente, Michaely, Saar & Wang (2002, Rev. Fin. Studies
15(4)) find high-volume price moves carry different continuation/reversal
signatures than low-volume ones, depending on whether the volume behind a
move is information- or hedging/liquidity-motivated -- the closest
existing literature to "does elevated volume make a price crossing more
trustworthy," which is the question both branches below ask, two different
ways.

Which constraint this attacks: **INFO**, via a data channel that is
present in the file but that no strategy in this project's registry has
used as an unsigned activity/confidence measure (only ever as a signed
flow-classification input, which failed for a different, disclosed reason
-- see below). Also touches **ERR**, in the novel branch only: v4's vote
is a fixed-band latch with no notion of how much evidence a crossing
carries; modulating the effective confirmation delay by information
intensity is a (small, informal) error-control step on that latch.

Not a duplicate of:
- L-14/L-15/L-16 (`camouflage_flow`/`stealth_trend`/`flow_regime`) and
  L-12 (`harsanyi_crowd`): those classify volume into a SIGNED buy/sell
  flow proxy via Bulk Volume Classification and use it DIRECTIONALLY.
  Ruled-out reason on record: "BVC from OHLCV is a price transform, not
  order flow" -- the classification step (inferring trade side from a
  price move) is what was proxied and what failed. This round never
  classifies a side; it uses raw traded volume MAGNITUDE only, as an
  activity-intensity feature, fed as a confirming vote (conservative) or
  a latch-speed modulator (novel) on v4's existing vote -- not as its own
  direction signal.
- The nine external INFO signals: all had an external feed with its own
  coverage start (DVOL 2021-03, MVRV inception-dependent, Binance metrics
  2020-09/2021-12, etc.) and all LAGGED the anchor gate when measured
  (R-53 median -5.5d; R-73 -2.0d/-9.0d; R-74 -4.0d/-35.0d; R-81 3/3
  episodes lag on its primary metric). Volume has no coverage gap and is
  not fetched from anywhere -- a structurally different risk profile,
  tested here on the full six-episode table rather than a three-to-four
  episode subset.
- R-62/B-27 (factored v4 into vote x scale; found the vote alone
  reproduces v4's whole matched-exposure drawdown signature and the scale
  factor alone reproduces neither): motivates keeping v4's conditional
  volatility-target SCALE factor untouched in both branches below and
  confining the change to the DIRECTION/VOTE side, where R-62 showed the
  signature actually lives.
- R-34/R-41-conservative/R-53-conservative/R-73-conservative (four
  independent confirmations that a never-increase-only bounded
  multiplicative BRAKE on exposure degenerates into a flat rescale of
  v4's own exposure path, R² > 0.95, regardless of the feeding signal):
  motivates that NEITHER branch below builds a brake. The conservative
  branch reuses R-53/R-55's validated CONFIRMING-VOTE combination rule
  instead (a genuinely different combination architecture, already shown
  in R-55 not to be the source of that failure mode). The novel branch
  uses a combination rule not yet tried in this ledger at all: modulating
  the anchor vote's OWN latch/confirmation speed, rather than adding a
  vote or multiplying exposure.
- R-80 (causal meta-labeling): reuses its hard-won lesson that any
  confirming vote fed into `confirming_vote_frac` must be DISCRETE (0/1),
  not continuous, so the formula keeps its ability to reach exactly flat
  / exactly full. Preserved here.

This module is read-only utility, written by the operator before dispatch
(same convention as r79_shared.py/r80_shared.py/r81_shared.py/
r82_shared.py). Neither branch edits it. Contains: (1) a byte-for-byte
duplicate of `kelly_regime_v4`'s 3-anchor vote construction (duplicated,
not imported -- R-54/R-55's convention); (2) the R-53/R-55 confirming-vote
combination rule; (3) causal volume-activity features (log-volume z-score
against its own trailing baseline, i.e. "how unusual is right-now's
participation," the natural unit-free construction given raw BTC volume's
own secular trend over 2017-2026); (4) the full six dated stress episodes
from R-82/R-83, usable here for the first time on an INFO-axis round
because volume has no coverage-start caveat; (5) a block-bootstrap null
generator for a lead-time gate; (6) shared date constants and the
causality truncation probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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

# The full R-82/R-83 six-episode table -- usable here for the first time
# on an INFO-axis round because volume has no external coverage-start
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

    ``anchor_sum`` in [0, 3], ``meta_vote`` in {0, 1} (per R-80/R-81's
    lesson: keep it DISCRETE so the formula can still reach exactly
    flat/exactly full). ``weight == 0`` recovers `kelly_regime_v4` exactly
    -- the required identity-recovery check every confirming-vote round
    has run.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


# ------------------------------------------------------------- volume features

def volume_z(df: pd.DataFrame, window_days: int = 20) -> pd.Series:
    """Causal log-volume z-score against its own trailing `window_days`
    mean/std. Positive = participation running hotter than recently
    normal; the natural unit-free construction given raw BTC volume's own
    secular trend across 2017-2026 (using a raw level or a fixed
    percentile threshold would conflate "unusual right now" with "the
    market has grown since 2017"). Zero/negative raw volume (should not
    occur but guarded) is masked to NaN before the log, forward-filled,
    exactly the treatment `r81_shared.crowding_z` gives its own zero-
    reading gap so this feature cannot silently manufacture `log(0)`.
    """
    vol = df["volume"].replace(0.0, np.nan).ffill()
    log_vol = np.log(vol)
    w = int(window_days * BARS_PER_DAY)
    mean = log_vol.rolling(w, min_periods=w // 4).mean()
    std = log_vol.rolling(w, min_periods=w // 4).std()
    return (log_vol - mean) / std.replace(0.0, np.nan)


# --------------------------------------------------------------------- null

def block_bootstrap_shifts(n_bars: int, block_days: int, n_draws: int,
                            seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n_bars`
    index, for a block-bootstrap null on a Step-A lead-time gate (this
    signal is a level/activity measure, not a cyclical phase partition of
    a trending series, so a standard block bootstrap of the local feature
    series against the fixed, real anchor-vote flip dates is the
    applicable null -- same construction and rationale as
    `r81_shared.block_bootstrap_lead_null`, generalized here to return raw
    shift arrays so both r84 branches can build their own crossing/rate
    statistics on top without re-deriving the shift logic).
    """
    rng = np.random.default_rng(seed)
    block = int(block_days * BARS_PER_DAY)
    draws = []
    for _ in range(n_draws):
        shift = int(rng.integers(block, n_bars - block)) if n_bars > 2 * block else int(rng.integers(1, n_bars))
        draws.append((np.arange(n_bars) + shift) % n_bars)
    return draws


# ---------------------------------------------------------------- causality

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


def assert_no_holdout(df: pd.DataFrame, oos_start: str = OOS_START) -> None:
    """Hard guard: the max timestamp in any frame this file touches must be
    strictly before `oos_start`. Same convention as r79/r81's own guard."""
    if len(df) == 0:
        return
    cutoff = pd.Timestamp(oos_start, tz=df.index.tz)
    max_ts = df.index.max()
    assert max_ts < cutoff, (
        f"holdout bar read: max timestamp {max_ts} >= {oos_start}. "
        "This file must never read data on or after the holdout start.")
