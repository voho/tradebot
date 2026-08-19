"""Shared causal Bayesian regime-confidence helper for the R-34 sizing round (08-19).

Idea in one sentence: `harsanyi_crowd` (L-12) builds a Bayesian posterior
over {bull, bear, chop} market types (Harsanyi 1967-68) and trades its
belief margin *directionally* -- and loses. L-12's own recorded lesson is
that "the crowding intuition was right... but as a direction signal
rather than a sizing input it loses." That is a stated, untested
hypothesis: this round tests it by feeding the same posterior margin into
`kelly_regime_v4`'s exposure (the SIZE axis, the only one that has ever
worked in this project) instead of building a new predictor.

This module isolates just the belief-update loop from `harsanyi_crowd.py`
-- byte-for-byte the same recursion (bull/bear/chop likelihoods, sticky
transition prior) -- and returns the raw margin series, with none of that
strategy's hysteresis, crowding haircut or position logic. It is causal
by construction (row i depends only on rows <= i, identical to the
already-registered, already-CI-passing `harsanyi_crowd`), so both
sizing variants built on top of it inherit that property rather than
re-deriving it.
"""

import numpy as np
import pandas as pd


def bayesian_margin(df: pd.DataFrame, mu: float = 0.15, stick: float = 0.985) -> np.ndarray:
    """P(bull) - P(bear) at each bar, in [-1, 1]. Causal: bar i uses only bars <= i.

    Identical recursion to ``harsanyi_crowd.HarsanyiCrowd.prepare`` (same
    ATR normalisation, same three Gaussian type-likelihoods, same sticky
    mixing), with the hysteresis/crowding/position machinery stripped out.
    """
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 48, min_periods=48).mean()
    x = (close.diff() / atr.shift(1)).clip(-3.0, 3.0).to_numpy()

    n = len(df)
    margin = np.zeros(n)
    b = np.full(3, 1.0 / 3.0)  # P(up), P(down), P(chop)
    for i in range(n):
        if not np.isfinite(x[i]):
            margin[i] = margin[i - 1] if i > 0 else 0.0
            continue
        lik = np.array([
            np.exp(-0.5 * (x[i] - mu) ** 2),
            np.exp(-0.5 * (x[i] + mu) ** 2),
            np.exp(-0.5 * (x[i] / 0.8) ** 2) / 0.8,
        ])
        b = b * lik
        b = stick * b + (1.0 - stick) / 3.0
        b /= b.sum()
        margin[i] = b[0] - b[1]
    return margin
