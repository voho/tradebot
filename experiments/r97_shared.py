"""Shared, read-only utilities for the R-97 round (08-23).

Idea in one sentence: rather than sizing `kelly_regime_v4`'s bet from the
empirical regime-conditional return distribution as if it were known,
solve for the exposure that is optimal against the WORST-CASE distribution
inside a Wasserstein ball around that empirical distribution, with the
ball's radius set by the number of completed regime cycles actually
observed so far -- so the strategy's own diagnosed sample-size problem
(N-approx-3, R-57/R-62) becomes a literal state variable that shrinks bet
size when few independent regime realizations have been seen and relaxes
it as more accumulate, rather than being a background caveat nobody prices.

Literature grounding, fetched and read before being relied on (both
citations verified live via WebSearch by the operator this round, not
taken from the research subagent's report on trust alone):

- Mohajerin Esfahani, P., & Kuhn, D. (2018), "Data-driven distributionally
  robust optimization using the Wasserstein metric: performance guarantees
  and tractable reformulations", *Mathematical Programming* 171(1-2),
  115-166. The general finite-sample result: for N iid samples from an
  unknown light-tailed distribution, the smallest Wasserstein-1 ball
  around the empirical measure that contains the true distribution with
  confidence >= 1-beta has radius shrinking at rate N^{-1/max(m,2)} in
  the sample dimension m (their Theorem 3.5, case A(2)); for a single
  scalar quantity (m=1) that is N^{-1/2} up to a log(1/beta) factor -- the
  standard concentration rate this round's radius formula uses.
- Li, J. Y.-M. (2023), "Wasserstein-Kelly Portfolios: A Robust Data-Driven
  Solution to Optimize Portfolio Growth", arXiv:2302.13979 (SSRN 4372148).
  Applies exactly this ball to the Kelly/log-optimal betting problem: the
  robust bet is the one maximizing worst-case expected log-growth over all
  distributions within the ball, and is provably more conservative than
  and converges to the plain empirical-Kelly bet as the ball shrinks (i.e.
  as N grows or as the confidence level required is relaxed). Reports
  out-of-sample outperformance and lower variance of realized growth
  versus plain empirical Kelly on real portfolio data.

Attacks **N-approx-3** primarily: this is the first SIZE-axis construction
in the project's history to make the REGIME-CYCLE COUNT ITSELF the state
variable driving exposure, rather than treating "how many regimes have we
actually seen" as a footnote about confidence. Attacks **ERR** secondarily:
a minimax worst-case guarantee is a formal error-control device, not
another point-estimate signal -- it answers "no error control anywhere in
the signal path" directly rather than adding a new prediction. It is not
"another indicator": it adds zero new predictive information about
direction or timing; it only reshapes how hard the existing vote is
allowed to bet on the return distribution it has already selected.

Not a duplicate of:
- The 22 prior SIZE-axis constructions (R-34 through R-60, R-62, R-87,
  R-93): every one is either a point-estimate scale retune keyed on
  MARKET volatility (vol-target variants, OU half-life-adaptive anchor
  rescaling, CUSUM changepoint vote) or a function of the STRATEGY'S OWN
  realized drawdown (Grossman & Zhou 1993, R-93, fixed and Hedge-blended).
  This round's state variable is neither: it is the CAUSAL COUNT OF
  COMPLETED REGIME CYCLES, an exogenous, monotonically non-decreasing
  integer with no volatility or drawdown content at all, driving a
  MINIMAX reformulation of the sizing problem rather than a threshold,
  ratio, or blend of point estimates.
- R-87 (Adaptive Conformal Inference wrapped around vote confidence or
  Kelly-scale dispersion): ACI is an ONLINE CALIBRATION recursion that
  adjusts a coverage target from realized miscoverage; Wasserstein DRO is
  a ONE-SHOT MINIMAX optimization over a metric ball fixed by a
  finite-sample concentration bound. Different mathematical objects: one
  is a control-theoretic feedback loop, the other is a robust-optimization
  reformulation of the objective itself.
- R-93 (Grossman & Zhou 1993 drawdown-constrained sizing, fixed and
  Hedge-blended): a function of the strategy's OWN realized drawdown
  path. This round's radius is a function of REGIME-CYCLE COUNT, entirely
  independent of realized P&L or drawdown.
- The seven regime-timing/detection-lag mechanisms (R-01, R-82 through
  R-86, R-96): none of them touch sizing at all -- they are alarms/braks
  keyed on when a NEW regime is starting. This round does not attempt to
  detect regime changes earlier; it takes v4's own regime vote exactly as
  given and only reweights how much to trust the return distribution
  CONDITIONAL on that vote's own history.
- The thirteen INFO-axis rounds: this round introduces no new data
  channel. Both branches read nothing beyond the committed OHLCV close
  series `kelly_regime_v4` itself already consumes.
- R-62 (vote x scale factorization): this round DOES touch `scale`, which
  R-62 found carries none of v4's matched-exposure DRAWDOWN signature --
  disclosed explicitly, not glossed over. R-62's finding is about which
  factor reproduces a specific cross-asset drawdown property, not about
  whether a scale retune can ever improve the return/Sharpe frontier; 21
  prior scale retunes were tested and rejected on THAT basis (beats v4 or
  not, out-of-sample), which is the basis this round uses too.

This module is read-only utility, written by the operator before dispatch
(same convention as r82_shared.py through r96_shared.py). Neither branch
edits it. Contains: (1) a byte-for-byte duplicate of `kelly_regime_v4`'s
3-anchor vote construction; (2) causal regime-cycle counting; (3) a
dependency-free (numpy/pandas only) simplified Wasserstein finite-sample
radius and DRO discount factor, disclosed as a simplification of Li
(2023)'s full convex-dual solve, not a claim of reproducing it exactly --
in the same spirit as R-65/R-67/R-68/R-79/R-85/R-86/R-96's own hand-rolled
estimators; (4) the pre-registered Step-0 falsification gate; (5) the
holdout guard and truncation probe.

ALL PARAMETERS BELOW ARE FIXED A PRIORI, before any real-market number was
computed for this round, and are not retuned after seeing any result.

`BETA_CONF = 0.10`: the DRO confidence level (ball contains the true
distribution with probability >= 90%), the standard finance-DRO choice
(Li 2023 uses 5-20% across their experiments; 10% is the round midpoint).
`KAPPA = 1.0`: the radius formula's leading constant, fixed at the
simplest defensible round number (same convention as R-96's `RJ_THRESH
= 0.5`) rather than solved for the exact Esfahani-Kuhn constant, which
depends on a light-tail exponent this project has no prior estimate of.
`N_REF = 3`: the reference cycle count the discount is calibrated against
-- literally the project's own "N-approx-3" diagnosis (R-57/R-62), chosen
so `discount(N_REF) = 0.5` (half-Kelly at the project's own measured
regime count) rather than an arbitrary number.

WHAT WOULD MAKE THIS FAIL, named now, before any real-data number exists
(the pre-registered Step-0 gate both branches must clear before writing
any strategy or sizing code):

**Kill switch A (spread).** Compute, causally, the completed regime-cycle
count N(t) at each of the six dated historical BTC regime transitions this
project's regime-timing rounds (R-82 through R-96) all use as their
episode table. If N(t) takes fewer than 4 distinct values across the six
episodes, the "dynamic" radius/discount degenerates into two or three
discrete constant multipliers over the whole 2017-2026 history --
indistinguishable from just picking among a handful of fixed
fractional-Kelly constants, which is exactly the "exposure collapses to a
low, roughly-constant fraction" artifact R-33/R-57/R-62 diagnosed as not a
real mechanism in 22 prior SIZE-axis attempts. This round would then be a
23rd instance of the identical failure mode under new notation, not an
escape from it.

**Kill switch B (magnitude).** Even if N(t) is spread, compute the DRO
discount factor at each episode's N(t). If the ratio of the largest to
smallest discount factor across the six episodes is below 1.3x, the
variation is too small to matter economically relative to v4's own 10%
deadband and 2x cap -- the mechanism would be real but too weak to clear
the +/-0.2 Sharpe noise floor (R-20) even before any backtest is run, so
there is no reason to build one.

Both switches are computed from data alone, causally, on inner-train and
inner-validation only (before OOS_START), and checked BEFORE either
branch writes a single line of strategy or backtest code.
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

# kelly_regime_v4's own anchor ladder and band -- duplicated verbatim from
# r82_shared.py / r85_shared.py / r86_shared.py / r96_shared.py.
V4_HORIZONS = (20, 40, 80)
V4_BAND = 0.01

# Inner split per docs/ROUTINE.md step 3. Holdout (>= OOS_START) is never
# read by either branch in this step.
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

# IDENTICAL to R-82 through R-96's own table -- copied verbatim, not
# re-derived, so this round's gate numbers stay comparable to theirs even
# though this round uses the table for a different purpose (a sizing state
# variable, not a detection-lag race).
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# ------------------------------------------------------------ DRO params
BETA_CONF = 0.10
KAPPA = 1.0
N_REF = 3


# ----------------------------------------------------------------- v4 vote
#   Copied verbatim from r82_shared.py / r85_shared.py / r86_shared.py /
#   r96_shared.py.


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
    own gate, exactly."""
    votes = anchor_votes(df, horizons, band)
    return sum(votes) / len(votes)


# ------------------------------------------------------------ regime cycles


def regime_cycle_count(df: pd.DataFrame, horizons: tuple[int, ...] = V4_HORIZONS,
                        band: float = V4_BAND) -> pd.Series:
    """Causal, expanding count of COMPLETED regime cycles as of each bar.

    `frac` only ever takes values in {0, 1/3, 2/3, 1}, so its majority
    LEAN (frac - 0.5) never sits exactly at zero: {2/3, 1} are a bullish
    majority, {0, 1/3} a bearish one. A "regime transition" is a sign flip
    of that lean; a "completed cycle" is a pair of flips (bull-to-bear-
    to-bull, or the mirror). This is a coarse, causal proxy for "how many
    independent regime realizations has the vote actually lived through",
    the quantity R-57/R-62's own N-approx-3 diagnosis names informally.
    Strictly causal: row i's count depends only on flips at rows <= i.
    """
    frac = anchor_majority(df, horizons, band)
    lean_sign = np.sign(frac.to_numpy() - 0.5)
    flips = np.zeros(len(lean_sign), dtype=bool)
    flips[1:] = lean_sign[1:] != lean_sign[:-1]
    cum_flips = np.cumsum(flips)
    cycles = (cum_flips // 2).astype(float)
    return pd.Series(cycles, index=df.index, name="regime_cycle_count")


# --------------------------------------------------------------- DRO radius


def wasserstein_radius(n_cycles: np.ndarray | float, kappa: float = KAPPA,
                        beta: float = BETA_CONF) -> np.ndarray | float:
    """Simplified Esfahani-Kuhn (2018) finite-sample radius for a scalar
    (m=1) light-tailed quantity: `eps(N) = kappa * sqrt(log(1/beta) / N)`.

    N=0 (no completed cycle observed yet) maps to an arbitrarily large
    radius (total distrust of the empirical distribution -- the DRO bet
    collapses toward flat). This is a disclosed SIMPLIFICATION of the
    paper's exact constant (which depends on a light-tail exponent this
    project has no prior estimate of), not a literal reproduction of
    their Theorem 3.5 -- same convention as R-96's moment-matched Hawkes
    parametrization.
    """
    n = np.asarray(n_cycles, dtype=float)
    log_term = np.log(1.0 / beta)
    with np.errstate(divide="ignore"):
        radius = kappa * np.sqrt(log_term / np.maximum(n, 1e-9))
    radius = np.where(n <= 0, np.inf, radius)
    return radius if isinstance(n_cycles, np.ndarray) else float(radius)


def dro_discount(n_cycles: np.ndarray | float, n_ref: float = N_REF,
                  kappa: float = KAPPA, beta: float = BETA_CONF) -> np.ndarray | float:
    """Bounded (0, 1] multiplicative discount from the DRO radius:
    `discount = 1 / (1 + eps(N) / eps(N_ref))`, so `discount(N_ref) = 0.5`
    by construction (half-Kelly at the project's own measured regime
    count) and `discount -> 1` as `N -> infinity` (no distrust, full
    Kelly) and `discount -> 0` as `N -> 0` (no completed regime cycle yet,
    maximal distrust). Monotone increasing in N, as Li (2023)'s robust
    bet is provably monotone in ball radius.
    """
    ref_radius = wasserstein_radius(float(n_ref), kappa, beta)
    r = wasserstein_radius(n_cycles, kappa, beta)
    return 1.0 / (1.0 + r / ref_radius)


# --------------------------------------------------------- Step-0 gate infra


def episode_pre_window(df: pd.DataFrame, onset_str: str) -> pd.Timestamp:
    """Timestamp strictly before an episode onset, for causal N(t) reads."""
    onset = pd.Timestamp(onset_str, tz="UTC")
    idx = df.index[df.index < onset]
    return idx[-1] if len(idx) else None


def step0_gate(df: pd.DataFrame) -> dict:
    """Pre-registered Step-0 falsification gate (both kill switches).

    Computes N(episode) and discount(episode) causally (strictly before
    each of the six dated onsets) on the frame passed in (the caller must
    restrict `df` to inner-train/inner-validation only -- this function
    performs no date filtering of its own beyond "strictly before onset").
    Returns a dict with per-episode values and the two pass/fail booleans.
    """
    cycles = regime_cycle_count(df)
    rows = []
    for label, onset_str in STRESS_EPISODES:
        ts = episode_pre_window(df, onset_str)
        if ts is None:
            rows.append((label, onset_str, None, None))
            continue
        n = float(cycles.loc[ts])
        d = float(dro_discount(n))
        rows.append((label, onset_str, n, d))

    valid = [(n, d) for (_, _, n, d) in rows if n is not None]
    n_values = sorted({n for (n, _) in valid})
    d_values = [d for (_, d) in valid]
    spread_pass = len(n_values) >= 4
    ratio = (max(d_values) / min(d_values)) if d_values and min(d_values) > 0 else float("inf")
    magnitude_pass = ratio >= 1.3

    return {
        "rows": rows,
        "distinct_n_values": n_values,
        "spread_pass": spread_pass,
        "discount_ratio": ratio,
        "magnitude_pass": magnitude_pass,
        "overall_pass": bool(spread_pass and magnitude_pass),
    }


# --------------------------------------------------------------- causal probe


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
    """Hard guard, same pattern as r81/r86/r88/r96: the max timestamp
    anywhere this file touches must be strictly before OOS_START."""
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


if __name__ == "__main__":
    from tradebot.data import load_dataset

    df, label = load_dataset(ROOT / "data", "spot")
    inner = df.loc[:INNER_VAL_END]
    assert_no_holdout(inner)
    gate = step0_gate(inner)
    print(f"data: {label}, inner bars: {len(inner):,}\n")
    print("episode                                   onset        N     discount")
    for label_, onset, n, d in gate["rows"]:
        if n is None:
            print(f"{label_:42s} {onset}  (no pre-onset data)")
        else:
            print(f"{label_:42s} {onset}  N={n:5.0f}  discount={d:.4f}")
    print(f"\ndistinct N values across episodes: {gate['distinct_n_values']}")
    print(f"kill switch A (spread >=4 distinct N): "
          f"{'PASS' if gate['spread_pass'] else 'FAIL'}")
    print(f"discount ratio max/min: {gate['discount_ratio']:.3f}")
    print(f"kill switch B (ratio >=1.3x): "
          f"{'PASS' if gate['magnitude_pass'] else 'FAIL'}")
    print(f"\nSTEP-0 GATE: {'PASS -- proceed to branches' if gate['overall_pass'] else 'FAIL -- STOP, no branch proceeds past this gate'}")
