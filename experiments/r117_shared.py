"""Shared, read-only utilities and pre-registration for the R-117 round (08-24).

DIRECTION, in one sentence: does a Donchian-channel BREAKOUT ensemble --
range-extremity detection (does price clear the recent high/low envelope),
not distance-from-a-moving-average -- carry a structurally different (and
possibly better/faster or genuinely complementary) regime signal than
`kelly_regime_v4`'s own 3-anchor mean-crossing vote, when substituted into
either (a) the SAME "formal regime-timing estimator" role nine prior
mechanisms have tried and failed at (HMM, BOCPD, Kalman LLT, CSD, transfer
entropy, Hawkes, POT/GPD, vote-latch/volume modulation, CUSUM -- all judged
by the identical Step-A 6-episode detection-lag gate this file reuses), or
(b) the SAME "alternative vote construction feeding v4's own frac*scale
slot" role R-105's anchor-ladder ensemble tried (judged by the standard
B1-B5 SIZE/ERR promotion bar this file also reuses)?

**Literature grounding, fetched via WebSearch this round:**

- Donchian, R. D. (1960s trading rules; formalized and popularized by the
  Turtle Traders, Dennis & Eckhardt, 1983-1985): the classical range-
  breakout rule -- go long when price clears the N-day high, short/flat
  when it breaks the N-day low. The archetypal "range extremity", not
  "distance from a central tendency", trend detector -- structurally
  distinct from every anchor-mean-crossing construction in this repo
  (`kelly_regime`/`_v2`/`_v3`/`_v4`, and R-105's own anchor-ladder ensemble)
  and from all nine formal regime-timing estimators tried in R-01/R-82/
  R-83/R-85/R-86/R-96/R-98/R-84/R-60 (state-space, point-process or
  distributional-statistic families, none of them a price-range construct).
- Zarattini, C., Pagani, A., & Barbon, A. (2025), "Catching Crypto Trends:
  A Tactical Approach for Bitcoin and Altcoins," SSRN 5209907 (posted April
  2025). The direct motivating citation, found by this round's own
  literature search: an ENSEMBLE of Donchian-channel breakout models at
  several different lookback periods, aggregated into one signal, combined
  with volatility-based position sizing, tested on a rotational portfolio
  of the top-20 most liquid coins since 2015 (net-of-fees Sharpe > 1.5,
  +10.8%/yr alpha vs BTC). Two properties of this paper motivate the
  two-branch split below: (1) the paper's OWN headline claim is about the
  ENSEMBLE, not a single lookback -- motivating the novel branch's
  multi-lookback construction; (2) the paper's edge is measured on a
  20-coin DAILY-rebalanced rotational book, not this project's single-BTC
  5-minute-bar, cost-realistic setting, so nothing about its reported
  Sharpe is assumed to transfer here -- both branches re-measure from
  scratch, on this project's own data, fee tier and promotion bar.
  Motivating, not load-bearing (same evidentiary status this project has
  given every other paper cited without a verified cost model on this
  data: R-116's Zaremba et al., R-105's Baltas & Kosowski, etc.)

**Which constraint each branch attacks:**

- Conservative: regime-timing (a TENTH structurally distinct formal
  estimator of "has a known historical regime break just happened",
  judged by the identical Step-A detection-lag gate R-82 through R-60 used
  -- reused verbatim, not re-derived, for direct comparability).
- Novel: SIZE (a 27th+ construction in that family, but -- like R-105's
  anchor-ladder ensemble -- substituting a structurally NEW detector
  FAMILY into the existing `frac` slot, not merely retuning a parameter of
  the shipped mean-anchor family; `scale` is left completely untouched in
  both branches, per R-62's finding that it carries none of v4's
  matched-exposure signature).

**Not a duplicate of:**

- R-01/R-82(BOCPD)/R-83(Kalman LLT)/R-85(CSD)/R-86(transfer entropy)/
  R-96(Hawkes)/R-98(POT/GPD)/R-84(vote-latch/volume)/R-60(CUSUM): nine
  regime-timing mechanisms, all state-space, point-process, information-
  theoretic or extreme-value-statistic estimators computed on RETURNS.
  Donchian breakout is computed on the PRICE LEVEL relative to its own
  recent high/low range -- a classical technical construct with no
  distributional or generative model anywhere, tried here for the first
  time in this ledger (confirmed by grep: no prior "donchian" mention).
- R-105 (anchor-ladder ensemble, MEAN-CROSSING detector family, varies only
  the LOOKBACK of the SAME construction v4 already uses -- five doubling
  ladders of the identical `close vs rolling-mean-plus-band` rule). This
  round varies the DETECTOR FAMILY itself (range-breakout vs mean-
  crossing), holding the family fixed at Donchian throughout, the flipped
  axis of variation from R-105.
- `hedge_experts`'s own registered Donchian-breakout expert: ONE single-
  lookback Donchian signal, mixed via Hedge/multiplicative-weights against
  nine STRUCTURALLY DIFFERENT signal types (MACD, RSI, momentum at four
  horizons, reversion, always-flat, buy-and-hold) as one vote among many in
  an already-registered, already-measured strategy. This round never
  touches `hedge_experts`, uses no Hedge/multiplicative-weights machinery,
  and both branches build a Donchian-ONLY ensemble (multiple lookbacks of
  the same family), substituted into `kelly_regime_v4`'s own architecture
  (vote x volatility-target scale) rather than blended by regret-matching
  against unrelated signal families.
- R-06/R-07/R-40/R-45 (anchor-ladder plateau search/bagging on the SHIPPED
  mean-anchor family): search for/blend a BETTER mean-anchor ladder. Both
  branches below leave the mean-anchor family alone entirely; Donchian
  channels are a different construction, not a re-tuned mean anchor.

Confirmed by grep of docs/LEDGER.md before this file was written: no prior
round mentions "donchian" in any form.

**Is it simulable here?** Yes, zero new data. Donchian channels need only
`high`/`low`/`close`, already present in every committed 5-minute OHLCV
file this project uses. Fully causal: the breakout reference for bar `t`
is `rolling(high, L).max()` / `rolling(low, L).min()` computed over bars
`< t` only (shift(1) before comparison), never including bar `t` itself.

**What would make each branch fail, named now, before any code ran:**

- Conservative (Step-A gate): a breakout detector requires price to have
  already cleared the recent extreme -- if anything, this is a STRUCTURALLY
  MORE lagging construction than v4's own "1% past a rolling mean" band,
  since a fresh N-day high/low is by construction a rarer, later event than
  a 1%-past-the-mean crossing. The single most likely outcome, named before
  any bar was read, is that this becomes the TENTH mechanism to fail the
  identical Step-A gate the same way its predecessors did -- lagging every
  sudden 2020-2022 shock and, at best, tying or narrowly leading only the
  slow 2018 build-up. A clean NEGATIVE closing regime-timing mechanism #10
  is the fully expected, fully successful outcome of this branch.
- Novel (SIZE-axis substitution): even if the Donchian ensemble's fraction-
  in-breakout signal is NOT collinear with v4's own vote (a real
  possibility, since it measures a geometrically different property of the
  same price series), Baltas & Kosowski's own finding (already cited by
  R-105) is that different-horizon trend signals normally carry LOW
  correlation with each other -- which cuts both ways: it could mean this
  ensemble genuinely adds information, or (the R-87/R-104/R-105-shaped
  "real but inert" pattern) that its disagreement with v4 concentrates in
  ordinary chop rather than in the six historical stress episodes that
  actually move the promotion bar. Given zero of 26+ prior SIZE-axis
  constructions and zero of 5 prior ERR-axis constructions have promoted,
  the pre-registered expectation is another NEGATIVE, reported with the
  same honesty as every prior round regardless of outcome.

This module is written by the operator BEFORE the branches are dispatched
and is READ-ONLY for both -- neither may edit it. Nothing here reads a bar
at or after OOS_START (2023-01-01); every function that walks a data frame
either restricts explicitly to a pre-OOS window or routes through
`assert_no_holdout`/`compare()`, inherited unmodified from the r102..r105
chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Re-exported verbatim from the r102..r105 chain: identical control
# machinery (compare(), the B1-B5 promotion bar, TargetStrategy, causal
# probes, fee tiers, data loaders) so every number this round produces is
# directly comparable to R-101 through R-105's own.
from experiments.r105_shared import (  # noqa: E402,F401
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BIND_FRAC_THRESH,
    ETH_SLICE_NAME,
    FUTURES,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    R2_THRESH,
    SHARPE_NOISE_FLOOR,
    SLICES,
    SPOT,
    STEP0_FLOOR_GRID,
    TargetStrategy,
    apply_deadband,
    assert_no_holdout,
    b1_from_inner_val,
    b2_diagnostic,
    b4_eth_falsification,
    b5_fee_tier,
    causal_truncation_probe_series,
    compare,
    fee_at,
    hr,
    inner_val_rows,
    load_btc,
    load_eth,
    paired_diff,
    print_plateau_table,
    print_rows,
    r_squared,
    run_slice,
    v4_raw_desired,
    v4_scale,
    v4_target,
    v4_vote_frac,
)
from experiments.r102_shared import V4_BAND, V4_HORIZONS, vote_frac  # noqa: E402,F401

# Step-A detection-lag gate machinery, reused verbatim from the regime-
# timing lineage (R-82 through R-96) -- fully generic over any candidate
# vote/alarm series, never re-derived per round.
from experiments.r82_shared import (  # noqa: E402,F401
    STRESS_EPISODES,
    block_bootstrap_shifts,
    episode_window,
    nearest_transition,
)

assert V4_HORIZONS == (20, 40, 80), V4_HORIZONS
assert abs(V4_BAND - 0.01) < 1e-12, V4_BAND

# ------------------------------------------------------------------------
# Donchian channel primitive -- causal, latched (same hysteresis shape v4's
# own anchor vote uses, for maximum structural comparability): bullish when
# close clears the PRIOR L-day high, bearish when it breaks the PRIOR
# L-day low, holding the previous verdict in between (neither a fresh high
# nor a fresh low). shift(1) on the rolling max/min excludes the current
# bar from its own reference range -- a fresh high cannot "break" itself.
# ------------------------------------------------------------------------


def donchian_vote(df: pd.DataFrame, lookback_days: int) -> pd.Series:
    """One latched Donchian breakout vote, in {0.0, 1.0}, for one lookback.

    Causal: `rolling(...).max()/.min()` over bars <= t, then `.shift(1)` so
    bar t's own high/low never appears in its own breakout reference --
    the reference range covers strictly-prior bars only.
    """
    bars = int(lookback_days * BARS_PER_DAY)
    upper = df["high"].rolling(bars).max().shift(1)
    lower = df["low"].rolling(bars).min().shift(1)
    close = df["close"]
    v = pd.Series(
        np.where(close > upper, 1.0, np.where(close < lower, 0.0, np.nan)),
        index=df.index,
    )
    return v.ffill().fillna(0.0)


def donchian_ensemble_frac(df: pd.DataFrame, lookbacks_days: tuple[int, ...]) -> pd.Series:
    """Mean of `donchian_vote` across `lookbacks_days` -- the ensemble
    "fraction of members currently in breakout", in [0, 1], the direct
    Donchian analogue of `vote_frac`'s anchor-vote average. Matches
    Zarattini/Pagani/Barbon's own "aggregate multiple models with
    different lookback periods into a single signal" construction.
    """
    votes = [donchian_vote(df, L) for L in lookbacks_days]
    return sum(votes) / len(votes)


def donchian_ensemble_target(df: pd.DataFrame, lookbacks_days: tuple[int, ...]) -> np.ndarray:
    """Drop-in replacement for `v4_raw_desired`: Donchian ensemble frac x
    v4's OWN UNCHANGED volatility-target scale (`v4_scale`), same
    composition order and deadband as `v4_target`/`v4_raw_desired`
    (R-62's slot convention: only the vote/frac input changes)."""
    frac = donchian_ensemble_frac(df, lookbacks_days).to_numpy()
    scale = v4_scale(df)
    return frac * scale


def make_donchian_target(lookbacks_days: tuple[int, ...]):
    def _build(df: pd.DataFrame) -> np.ndarray:
        return apply_deadband(donchian_ensemble_target(df, lookbacks_days))
    _build.__name__ = f"donchian_ens_{'_'.join(str(l) for l in lookbacks_days)}d"
    return _build


def _self_test() -> None:
    idx = pd.date_range("2017-01-01", periods=60_000, freq="5min", tz="UTC")
    rng = np.random.default_rng(117)
    innov = rng.normal(0, 0.0006, len(idx))
    drift = np.cumsum(np.full(len(idx), 0.00002))
    close = 10_000 * np.exp(np.cumsum(innov) + drift)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.0006, "low": close * 0.9994,
         "close": close, "volume": 1.0}, index=idx)

    v = donchian_vote(df, 20)
    assert set(np.unique(v.to_numpy())) <= {0.0, 1.0}
    # Causality: bar t's vote must not change if bars > t are perturbed.
    check_at = 45_000
    short = df.iloc[: check_at + 5_000].copy()
    v_short = donchian_vote(short, 20)
    assert np.isclose(v.iloc[check_at], v_short.iloc[check_at]), "donchian_vote is not causal"

    ens = donchian_ensemble_frac(df, (20, 40, 80))
    assert ens.between(0.0, 1.0).all()

    tgt = donchian_ensemble_target(df, (20, 40, 80))
    assert np.isfinite(tgt).all()


_self_test()
