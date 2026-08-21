"""Shared, read-only utilities for the R-80 meta-labeling round (08-21).

Idea in one sentence: `kelly_regime_v4` applies its 3-anchor directional
vote with no error control at all -- the vote is acted on identically
whether it has recently been reliable or not (the ERR constraint in
`docs/LEDGER.md`'s standing diagnosis). Lopez de Prado's (2018, *Advances
in Financial Machine Learning*, ch. 3) meta-labeling separates "which way"
from "how much/whether to act", training a secondary model on whether the
primary bet would have paid off. This project tried a version of that once,
very early (R-04, before `docs/ROUTINE.md` existed): "Walk-forward, purged
and embargoed logistic secondary model... Hurt in-sample, neutral
out-of-sample. Their trend-scanning label looks *forward* and is
inadmissible here." That is a specific, diagnosed lookahead bug in the
LABEL construction, not a refutation of meta-labeling as a mechanism -- and
R-04 predates every methodology tool this repo now has (purged K-fold is
`tradebot.inference`, the confirming-vote architecture is R-53/R-55, the
truncation causality probe is now standard practice). This round redoes it
with both fixed.

Second fix, independent of the first: this project's own standing lesson
(the line below the standing diagnosis in `docs/LEDGER.md`) is that a
**never-increase-only bounded brake is 4-for-4 failed** (R-34, R-41,
R-53-conservative, R-73-conservative) regardless of what signal feeds it,
and the recommended alternative is R-55's validated CONFIRMING-VOTE
architecture, which can move exposure in either direction. Both R-80
branches use that combination rule, not a brake -- fed by a confidence
signal (this round's actual contribution) instead of an external dataset.

This module is read-only utility, written by the operator before dispatch.
Neither branch edits it. It contains no signal and no strategy logic --
only: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s own 3-anchor
vote construction, so both branches compose against the true incumbent
inputs without importing the registered strategy module (project
convention -- see R-54/R-55: duplicate the combination rule, don't import
a registered strategy); (2) the R-53/R-55 confirming-vote formula,
generalized to a continuous meta-vote in [0, 1] rather than a 0/1 override;
(3) a placebo/permutation-null generator for each branch's own
pre-registered falsification gate; (4) shared date constants.
"""

from __future__ import annotations

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


def anchor_votes(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                  band: float = V4_BAND) -> list[pd.Series]:
    """The three latched 0/1 anchor votes `kelly_regime`/`_v3`/`_v4` use internally.

    Causal: row i depends only on rows <= i (rolling mean + ffill/latch).
    Returned as a list of per-anchor 0/1 series (not yet averaged), so a
    branch can form `anchor_sum = sum(anchor_votes(df))` and combine it
    with a meta-vote via `confirming_vote_frac` below.
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


def confirming_vote_frac(anchor_sum: np.ndarray, meta_vote: np.ndarray,
                          weight: float) -> np.ndarray:
    """R-53/R-55's combination rule, generalized to a continuous meta-vote.

    ``frac = (anchor_sum + weight * meta_vote) / (3 + weight)``

    ``anchor_sum`` in [0, 3] (sum of the three 0/1 anchor votes),
    ``meta_vote`` in [0, 1] (this round's confidence signal -- may be
    continuous, unlike R-53/55's own 0/1 external vote). ``weight == 0``
    recovers `kelly_regime_v4` exactly (the required identity-recovery
    check every prior confirming-vote round has run). Result stays in
    [0, 1], so it drops straight into ``desired = frac * scale`` unchanged.
    """
    anchor_sum = np.asarray(anchor_sum, dtype=float)
    meta_vote = np.asarray(meta_vote, dtype=float)
    return (anchor_sum + weight * meta_vote) / (3.0 + weight)


def placebo_offset_indices(n: int, block_days: int, n_draws: int,
                            seed: int) -> list[np.ndarray]:
    """`n_draws` causal-safe circular block-shifts of a length-`n` index.

    Used to build a placebo/permutation null for a Step-A measurement
    gate: recompute a statistic against each shifted alignment and compare
    the true (zero-shift) statistic to that null distribution, exactly
    R-79's placebo-offset design (chosen there, and here, over a plain
    within-series block bootstrap, which does not control for a slowly
    trending series showing spurious structure under ANY partition).
    Shifts are circular over the full series so each draw reuses the same
    bars (no new data, no shortened sample) with a different time
    alignment between "confidence signal" and "realized outcome".
    """
    rng = np.random.default_rng(seed)
    block = int(block_days * BARS_PER_DAY)
    draws = []
    for _ in range(n_draws):
        shift = int(rng.integers(block, n - block)) if n > 2 * block else int(rng.integers(1, n))
        draws.append((np.arange(n) + shift) % n)
    return draws


def truncation_causality_probe(build_target_fn, df: pd.DataFrame,
                                check_at: int, shorter_by: int = 20_000) -> bool:
    """Standard truncation probe: does `target[check_at]` change if bars after it are dropped?

    ``build_target_fn(df) -> np.ndarray`` must be the branch's own
    `prepare()`-equivalent target-construction function. Returns True if
    causal (identical value both ways), False if it moved -- i.e. found a
    lookahead bug. This is the same check R-53 onward runs against every
    novel signal before trusting any number from it.
    """
    full = build_target_fn(df)
    short = build_target_fn(df.iloc[:check_at + shorter_by].copy())
    a, b = full[check_at], short[check_at]
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, equal_nan=True))
