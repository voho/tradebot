"""R-140: a Synthetic Control Method (SCM) test of this project's own
central small-N claim -- that `kelly_regime_v4`'s edge over a risk-matched
hold concentrates in a handful of dated regime-transition episodes
(L-01/R-62's "roughly three sudden regime transitions"; the project's own
N approx 3 constraint; most recently re-tested by R-138's permutation
test). Shared, frozen infrastructure for a two-branch parallel round. Per
ROUTINE.md's parallelism rules this file is neutral ground: both branches
import from it, NEITHER BRANCH EDITS IT, and it does not itself compute a
verdict.

=====================================================================
WHY THIS ROUND, AND WHAT IT ATTACKS
=====================================================================

Constraint attacked: **N approx 3** (primary). This project's own backlog
has held nothing but B-06 since R-110 (reconfirmed by every re-ranking
through R-139), and R-139's own closing line names the shape of thread
still worth pursuing: not a Nth individual mechanism re-tried on an
existing gate, but "a structurally different question." R-138 already
brought one such structurally different tool to this exact claim -- a
formally-justified small-N permutation test (Nguyen & Wolf 2026) -- and
found the edge concentration real and well-calibrated on BTC alone, but
not replicating on ETH. This round brings a SECOND, methodologically
distinct small-N causal-inference tool to the SAME underlying claim:
**Synthetic Control Method (SCM)**, grounded in a body of literature this
project has never cited (grep-confirmed: zero prior mentions of
"synthetic control" or "Abadie" anywhere in `docs/LEDGER.md`).

Literature (WebSearch, this session, before either branch was written):
- Abadie, A., Diamond, A. & Hainmueller, J. (2010), "Synthetic Control
  Methods for Comparative Case Studies: Estimating the Effect of
  California's Tobacco Control Program," *JASA* 105(490), 493-505. The
  foundational method: approximate a single treated unit's counterfactual
  path as a convex (non-negative, sum-to-one) weighted combination of
  untreated "donor" units, fit to match the treated unit's own PRE-
  treatment path; the post-treatment gap between the real and synthetic
  path is the estimated effect. Inference is via IN-SPACE PLACEBO: re-run
  the identical procedure with each donor relabeled as the "treated" unit
  in turn, and rank the real gap within that placebo distribution.
- Cattaneo, M.D., Feng, Y. & Titiunik, R. et al., "Inference with Few
  Treated Units" (arXiv:2504.19841, 2025), the fresh survey that motivated
  this round: it names exactly this project's own situation -- few
  treated units/events, standard asymptotic inference unreliable however
  large the total bar count -- and reviews SCM's in-space placebo test
  alongside randomization/permutation approaches (i.e. what R-138 already
  used) as complementary, not interchangeable, tools for it.
- Chernozhukov, V., Wuthrich, K. & Zhu, Y. (2021), "An Exact and Robust
  Conformal Inference Method for Counterfactual and Synthetic Control,"
  *JASA* 116(536), 1-16 (arXiv:1712.09089). A structurally different
  inference route on the SAME synthetic-control point estimate: instead of
  permuting WHICH UNIT is treated (cross-sectional, Abadie's in-space
  placebo), it permutes/block-permutes the assignment of POST-period
  RESIDUALS under an exchangeability argument, giving finite-sample-exact
  (not merely asymptotic) p-values. This is the novel branch's mechanism.

**Not a duplicate of:**
- R-101 (delete-one-episode jackknife of the vote's own edge -- no donor
  pool, no counterfactual construction, feeds a sizing multiplier not a
  hypothesis test).
- R-104/R-105/R-106 (ERR-axis: sampling significance of the vote's OWN
  historical P&L, specification disagreement across anchor ladders, and
  cross-model-class disagreement across regime DETECTORS -- none of these
  five prior ERR attempts construct a donor-weighted counterfactual PRICE
  PATH from other instruments; R-106 in particular measures disagreement
  among detector OUTPUTS, never a fitted weighted combination of other
  ASSETS' own return paths).
- R-116 (cross-asset breadth/disagreement as a confirming signal -- a
  summary correlation/vote statistic over the panel, not a fitted convex
  weighting solved to minimize pre-period tracking error, and used there
  as a live SIZE-axis input; this round is an ex-post inference tool, not
  a live trading signal, and explicitly does not modify `kelly_regime_v4`
  regardless of outcome, matching R-138/R-101/R-104-106's convention).
- R-118/R-119/R-122 ("synthetic" branches: MODEL-BASED simulated price
  paths -- jump-augmented GBM, regime-switching generators -- used to
  stress-test v4's own PARAMETER selection. This round's "synthetic" is
  DATA-DRIVEN: a weighted combination of this project's own real,
  already-committed donor instruments, used for INFERENCE on already-
  realized episodes, never to generate a fake price series or select a
  parameter.)
- R-138 (the Nguyen-Wolf permutation test on the identical claim -- same
  target claim, deliberately, but a different tool: R-138 permutes EVENT
  DATES across the eligible calendar and needs no donor pool at all; this
  round constructs an explicit counterfactual PRICE PATH from other
  instruments and needs no calendar permutation. The two are complementary
  per the Cattaneo/Feng/Titiunik survey's own framing, not redundant.)
- R-57/R-63 (cross-sectional panel breadth/momentum across independent
  price PATHS, used to build a NEW trading signal; this round never trades
  the donor pool and never scores a new strategy).

**Is it simulable here?** Yes, and re-uses only already-committed data:
BTC (`load_dataset(..., "spot")`), ETH (`load_coinbase_eth_spot`), and the
same 6-instrument donor panel R-57/R-63/R-106/R-113/R-116 already use
(`experiments/r57_cross_asset_panel.py`'s `load_candidates`/
`select_panel`). No order book, no new data channel, no proxying.

**A disclosed, load-bearing data constraint, found before any branch ran:**
the donor panel's own committed files start **2020-01-01** (verified this
session: every donor CSV's first row is `1577836800000` ms = 2020-01-01
00:00 UTC). Of the six `STRESS_EPISODES` below, only the four from
2020-03 onward have ANY donor coverage at all -- the two 2018 episodes
(bear onset, bear bottom) cannot be given a donor-weighted counterfactual
by this data and are EXCLUDED from the SCM analysis by construction, not
by choice. This shrinks the already-small N from 6 to **4**, which is
itself informative about how much smaller this project's usable evidence
base gets once "does a plausible counterfactual instrument even exist"
is asked, rather than merely "is there a dated headline." Report both
counts; do not let the smaller N pass unremarked.

**What would make this fail, named now, before any code:**

(a) **Invalidity by construction.** R-63 already measured this exact donor
    pool's mean pairwise daily-return correlation with BTC/ETH at 0.634
    and its Grinold breadth at 1.47-of-8 -- a panel this correlated with
    its own target may simply be unable to produce a donor combination
    that both (i) tracks the target's PRE-event path well (the standard
    SCM pre-fit diagnostic) and (ii) diverges meaningfully from it during
    a market-wide crypto crash that plausibly hits every donor at once
    (violating SCM's implicit no-interference/no-common-shock assumption
    -- if the "control" units crash in sympathy with the "treated" one,
    there is no valid counterfactual to estimate against). This is the
    single most likely failure mode and is guarded by the pre-fit
    diagnostic below: a branch whose pre-fit RMSPE ratio fails the gate
    must say SCM is not a valid tool for this panel and stop, rather than
    force an inference reading through a bad fit.
(b) **Real but redundant.** Even a valid fit could simply reproduce
    R-138's own finding (significant on BTC, fails ETH replication) --
    informative as independent triangulation (the R-131/R-133 precedent:
    two different tools agreeing is evidence the finding is not an
    artifact of either tool), but not a new result.
(c) **N=4's own resolution.** With only 4 donor-covered episodes, even a
    well-fitted placebo test has coarse resolution (a 6-donor panel gives
    at most 6 placebo draws per episode from the in-space test; guarded by
    pooling across episodes per Abadie's own convention, and by the novel
    branch's structurally different, denser residual-permutation route).

**Falsification test, pre-registered:** the project's standard B4 --
does the sign and significance of the SCM-based finding replicate on ETH,
using the identical donor pool and procedure, ETH's own pre-fit window?

**Decision rule, pre-registered verbatim, evaluated identically by both
branches on the training period (`<= INNER_VAL_END`); NO bar at or after
`OOS_START = 2023-01-01` may be read by either branch during Step 3:**

Each branch reports one of three pre-registered outcomes, decided now,
before any real number exists:

1. **INVALID (Step-A stop).** The pre-fit diagnostic
   (`pre_fit_rmspe_ratio`, defined below) exceeds `RMSPE_GATE = 2.0` --
   i.e. the best-fit synthetic path tracks the real pre-event path more
   than 2x worse than the median donor-as-placebo synthetic tracks ITS
   own pre-event path -- for a MAJORITY (>=3 of 4) donor-covered episodes
   on BTC. If so: SCM is not a valid inference tool for this donor panel,
   full stop, report which failure mode (a) predicted, and do not report
   a p-value that a bad fit cannot support.
2. **VALID & CONFIRMS.** Pre-fit gate passes AND the pooled in-space (or,
   for the novel branch, conformal) p-value for kelly_regime_v4's excess
   log-return across the donor-covered BTC episodes is `< 0.10` (a p-value
   floor set by N=4's own coarse resolution, wider than R-138's N=6-based
   0.05) AND the identical procedure on ETH's own episodes replicates with
   the same sign at `< 0.20` (ETH's own wider tolerance, matching R-138's
   asymmetric convention). This upgrades confidence in the standing edge-
   concentration diagnosis via a second, independent tool -- a METHOD
   result, not a strategy promotion (this round changes no strategy code
   regardless of outcome, matching R-138/R-101/R-104-106's convention).
3. **VALID & DOES NOT CONFIRM.** Pre-fit gate passes but BTC significance
   or ETH replication fails -- NEGATIVE, and, if it reproduces R-138's own
   BTC-pass/ETH-fail shape, an independent triangulation of that finding
   by a structurally different tool (informative in its own right, per the
   R-131/R-133 precedent on the value of accidental independent
   replication).

No bar at or after `OOS_START = 2023-01-01` may be read by either branch
during Step 3. This round produces no strategy code change under any
outcome; if outcome 2 is reached, the reusable SCM machinery is proposed
for addition to `tradebot/inference.py`, parallel to how R-138's
permutation-test precedent was scoped (matching R-134's "fix adopted, no
promoted strategy" verdict shape).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold  # noqa: E402,F401
from experiments.r57_cross_asset_panel import (  # noqa: E402
    load_candidates,
    select_panel,
)
from tradebot.broker import MarketSpec  # noqa: E402,F401
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402,F401
from tradebot.metrics import compute_metrics  # noqa: E402,F401
from tradebot.registry import get_strategy  # noqa: E402,F401
from tradebot.window import run_period  # noqa: E402,F401

DATA_DIR = ROOT / "data"

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round: inner-train / inner-
# validation only. The holdout (>= OOS_START) is never read during Step 3.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
PRIMARY_MARKET = FUTURES

# ----------------------------------------------------------------------
# STRESS_EPISODES -- copied VERBATIM from experiments/r138_shared.py
# (itself unchanged since R-82). Not re-selected for this round.
# ----------------------------------------------------------------------
STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]

# The donor panel's own committed CSVs start 2020-01-01 (verified this
# session). Only episodes at or after this date can receive a donor-
# weighted counterfactual. Named explicitly rather than silently dropped.
DONOR_COVERAGE_START = pd.Timestamp("2020-01-01", tz="UTC")
DONOR_COVERED_EPISODES = [
    (name, date) for name, date in STRESS_EPISODES
    if pd.Timestamp(date, tz="UTC") >= DONOR_COVERAGE_START
]
assert len(DONOR_COVERED_EPISODES) == 4, (
    f"expected 4 donor-covered episodes, got {len(DONOR_COVERED_EPISODES)} "
    "-- STRESS_EPISODES or DONOR_COVERAGE_START changed unexpectedly"
)

# Event-window convention, reused verbatim from r138_shared (MacKinlay
# 1997) rather than re-fit for this round.
WINDOW_PRE_DAYS = 5
WINDOW_POST_DAYS = 20

# Pre-fit window: how much history before an episode's WINDOW_PRE_DAYS
# start is used to fit the donor weights. Chosen once, before any real
# number was computed, as "long enough to constrain a <=6-donor convex
# combination without reaching into the PRIOR stress episode" -- the
# shortest gap between any two donor-covered STRESS_EPISODES dates is
# 2021-11-10 -> 2022-05-09, 180 days; PRE_FIT_DAYS is set strictly below
# that gap so no episode's fit window ever overlaps another episode's
# post-window.
PRE_FIT_DAYS = 150

RMSPE_GATE = 2.0
BTC_P_GATE = 0.10
ETH_P_GATE = 0.20


# ------------------------------------------------------- data plumbing


def load_target_daily_returns(which: str) -> pd.Series:
    """Causal daily log-returns for 'btc' or 'eth' from committed 5m bars."""
    if which == "btc":
        df, _label = load_dataset(DATA_DIR, "spot")
    elif which == "eth":
        df = load_coinbase_eth_spot(DATA_DIR)
        if df is None:
            raise FileNotFoundError("ETH Coinbase spot data not found")
    else:
        raise ValueError(which)
    daily_close = df["close"].resample("1D").last().dropna()
    return np.log(daily_close).diff().dropna()


def load_donor_daily_returns() -> pd.DataFrame:
    """Causal daily log-returns for the same 6-instrument donor panel
    R-57/R-63/R-106/R-113/R-116 already use. Columns = tickers."""
    candidates = load_candidates()
    panel = select_panel(candidates)
    series = {}
    for asset in panel:
        daily_close = asset.df["close"].resample("1D").last().dropna()
        series[asset.ticker] = np.log(daily_close).diff().dropna()
    return pd.DataFrame(series).dropna(how="all")


# ------------------------------------------------------- SCM core (shared)
#
# The donor-WEIGHT-FITTING mechanism is genuinely shared machinery (both
# branches need the SAME point estimate; what differs between them is the
# INFERENCE built on top of it -- in-space cross-sectional placebo
# (conservative) vs. conformal residual permutation (novel)). No scipy is
# a project dependency (checked: not installed, not in pyproject.toml), so
# the convex (non-negative, sum-to-one) weight fit below uses projected
# gradient descent onto the simplex (Duchi, Shalev-Shwartz, Singer &
# Chandra 2008, "Efficient Projections onto the l1-Ball for Learning in
# High Dimensions," ICML) rather than adding a new dependency for one
# round -- a plain, easily-verified O(n log n) Euclidean simplex
# projection, not a new statistical assumption.


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto {w : w >= 0, sum(w) == 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.clip(v - theta, 0.0, None)


def fit_scm_weights(
    donor_returns: pd.DataFrame,
    target_returns: pd.Series,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    n_iter: int = 5000,
    seed: int = 140,
) -> pd.Series:
    """Non-negative, sum-to-one donor weights minimizing squared tracking
    error against `target_returns` over [fit_start, fit_end]. Abadie,
    Diamond & Hainmueller (2010)'s SCM objective, solved by projected
    gradient descent (Duchi et al. 2008's simplex projection)."""
    idx = donor_returns.index.intersection(target_returns.index)
    idx = idx[(idx >= fit_start) & (idx <= fit_end)]
    X = donor_returns.loc[idx].to_numpy()
    y = target_returns.loc[idx].to_numpy()
    n_donors = X.shape[1]
    rng = np.random.default_rng(seed)
    w = _project_to_simplex(rng.uniform(0, 1, n_donors))
    lr = 1.0 / (np.linalg.norm(X, ord=2) ** 2 + 1e-12)
    for _ in range(n_iter):
        resid = X @ w - y
        grad = X.T @ resid
        w = _project_to_simplex(w - lr * grad)
    return pd.Series(w, index=donor_returns.columns)


def synthetic_path(
    donor_returns: pd.DataFrame, weights: pd.Series, idx: pd.DatetimeIndex
) -> pd.Series:
    """Weighted-combination daily log-return path over `idx`."""
    aligned = donor_returns.reindex(idx)[weights.index]
    return aligned.mul(weights, axis=1).sum(axis=1)


def pre_fit_rmspe(
    target_returns: pd.Series, synth_returns: pd.Series,
    fit_start: pd.Timestamp, fit_end: pd.Timestamp,
) -> float:
    idx = target_returns.index.intersection(synth_returns.index)
    idx = idx[(idx >= fit_start) & (idx <= fit_end)]
    gap = target_returns.loc[idx] - synth_returns.loc[idx]
    return float(np.sqrt(np.mean(gap.to_numpy() ** 2)))


def pre_fit_rmspe_ratio(
    donor_returns: pd.DataFrame,
    target_returns: pd.Series,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
) -> tuple[float, float, float]:
    """Standard SCM validity diagnostic: the real target's own pre-fit
    RMSPE, divided by the MEDIAN pre-fit RMSPE achieved when each donor in
    turn is treated as the target and fit against the REMAINING donors.
    A ratio near or below 1 means the target is at least as fittable as a
    typical donor; a ratio >> 1 (this round's gate: RMSPE_GATE=2.0) means
    the donor pool cannot approximate the target even before any event is
    reached, and no post-event gap from this fit should be trusted.
    Returns (ratio, target_rmspe, median_donor_rmspe)."""
    w = fit_scm_weights(donor_returns, target_returns, fit_start, fit_end)
    synth = synthetic_path(donor_returns, w, donor_returns.index)
    target_rmspe = pre_fit_rmspe(target_returns, synth, fit_start, fit_end)

    donor_rmspes = []
    for placebo in donor_returns.columns:
        others = donor_returns.drop(columns=[placebo])
        placebo_target = donor_returns[placebo]
        pw = fit_scm_weights(others, placebo_target, fit_start, fit_end)
        psynth = synthetic_path(others, pw, others.index)
        donor_rmspes.append(pre_fit_rmspe(placebo_target, psynth, fit_start, fit_end))
    median_donor_rmspe = float(np.median(donor_rmspes))
    ratio = target_rmspe / median_donor_rmspe if median_donor_rmspe > 0 else float("inf")
    return ratio, target_rmspe, median_donor_rmspe


def load_v4_and_extended_donor_returns(
    which: str, market: MarketSpec = PRIMARY_MARKET
) -> tuple[pd.Series, pd.DataFrame]:
    """The actual SCM target and donor pool for this round.

    TARGET: `kelly_regime_v4`'s daily EXCESS log-return over its own
    realized-volatility-matched constant-exposure hold (`cand - matched`,
    reusing `experiments/r138_shared.py`'s existing
    `candidate_and_matched_daily_logret`/`solve_matched_c` machinery, the
    identical "abnormal return" object R-138's own permutation test used).
    Not raw v4 P&L, and not raw asset price: R-33's risk-matching is
    already baked into the target BY SUBTRACTION, so the donor fit below
    is never allowed to "win" merely by rediscovering that a matched
    passive hold explains v4 (an early smoke test, before either branch
    was dispatched, showed exactly that degenerate result when the
    matched-hold was left IN the donor pool alongside raw v4 returns as
    the target -- fit weight collapsed to 1.0 on the matched-hold donor,
    testing nothing new. Excess-over-matched-hold as the target closes
    that degenerate path by construction.)

    DONOR POOL: the 6 cross-asset instruments' raw daily log-returns only
    (`load_donor_daily_returns`). The question this SCM construction asks:
    can a weighted combination of OTHER crypto assets' own price action
    approximate v4's own edge over a risk-matched hold, well enough
    pre-event to trust a post-event gap as informative?
    """
    from experiments.r138_shared import candidate_and_matched_daily_logret

    df = load_dataset(DATA_DIR, "spot")[0] if which == "btc" else load_coinbase_eth_spot(DATA_DIR)
    df = df.loc[:INNER_VAL_END].copy()
    cand, matched, c, achieved_vol = candidate_and_matched_daily_logret(
        df, market, label=which
    )
    excess = (cand - matched).dropna()
    print(f"[r140] {which}/{market}: matched c={c:.4f} achieved_vol={achieved_vol:.4f} "
          f"n_obs(excess)={len(excess)}")

    donors = load_donor_daily_returns()
    return excess, donors


def event_gap(
    target_returns: pd.Series,
    synth_returns: pd.Series,
    event_date: pd.Timestamp,
    pre_days: int = WINDOW_PRE_DAYS,
    post_days: int = WINDOW_POST_DAYS,
) -> float:
    """Cumulative (target - synthetic) log-return gap over the event
    window [event_date - pre_days, event_date + post_days], MacKinlay
    (1997)'s standard event-window shape."""
    start = event_date - pd.Timedelta(days=pre_days)
    end = event_date + pd.Timedelta(days=post_days)
    idx = target_returns.index.intersection(synth_returns.index)
    idx = idx[(idx >= start) & (idx <= end)]
    gap = target_returns.loc[idx] - synth_returns.loc[idx]
    return float(gap.sum())
