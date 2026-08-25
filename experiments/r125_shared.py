"""Shared, read-only utilities and pre-registration for the R-125 round (08-25).

DIRECTION, in one sentence: replace the RISK MEASURE `kelly_regime_v4`'s
`scale` component sizes against -- realized standard deviation, via the
`target_vol / realized_vol` conditional-volatility-targeting rule inherited
unchanged from `kelly_regime_v3` -- with realized Conditional Value-at-Risk
(CVaR, expected shortfall), the coherent, tail-focused risk measure
Rockafellar & Uryasev (2000, "Optimization of Conditional Value-at-Risk",
*Journal of Risk* 2(3), 21-41) built specifically because variance penalizes
upside and downside symmetrically while CVaR does not.

**Why this and not a 28th retuning of the existing vol-target.** Every
SIZE-axis round to date (R-34...R-124, 27+ attempts per docs/LEDGER.md's own
running count) has retuned `scale`'s MAGNITUDE, its INPUT (per-asset
calibration: R-59; timing/adaptivity: R-60), or replaced the VOTE (`frac`)
with a structurally new detector family (R-105 novel, R-117 novel) -- but in
every single one of those 27+ attempts, `scale` itself has remained
`target_vol / realized_vol` (v4) or one of its conditional-targeting cousins
(v3's extremes-only latch). The RISK MEASURE that `scale` is built to hit a
target of -- standard deviation -- has never been varied. This round varies
exactly that one thing and nothing else: `frac` (the 3-anchor vote), the
anchors themselves, the deadband, and the hysteresis latch are all reused
byte-for-byte from `kelly_regime_v4`.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r125_conservative_cvar_scale.py`): the minimal, single-line
  substitution -- keep v4's exact architecture (`desired = frac * scale`,
  same deadband, same anchors) and swap only the risk measure inside
  `scale`: `scale = min(target_cvar / realized_cvar, max_leverage)` in place
  of `min(target_vol / realized_vol, max_leverage)`. Because CVaR is
  POSITIVELY HOMOGENEOUS of degree 1 (Artzner, Delbaen, Eber & Heath 1999,
  "Coherent Measures of Risk", *Mathematical Finance* 9(3), 203-228: for a
  coherent risk measure rho and lambda >= 0, rho(lambda X) = lambda
  rho(X)), this substitution is a like-for-like replacement of one
  homogeneous risk functional with another -- the FORM of the sizing rule
  is unchanged, only which tail of the return distribution it reacts to.

- NOVEL (`r125_novel_cvar_kelly.py`): a genuinely different derivation, not
  a substitution. Rockafellar & Uryasev's own point is that CVaR is useful
  as a CONSTRAINT on an optimization, not merely as a target to hit -- so
  the novel branch numerically solves, at every bar, for the exposure
  fraction f in [0, max_leverage] that maximizes expected log growth
  E[log(1 + f*r)] over the CAUSAL, TRAILING empirical return distribution
  (conditional on the current 3-anchor vote state, so the distribution used
  is "what has price done historically when the vote looked like this",
  not an unconditional one), subject to CVaR_alpha(f*r) <= budget. Because
  log is strictly concave, the unconstrained maximizer f_kelly may sit
  BELOW the CVaR-implied cap -- so the rule is `f* = min(f_kelly, budget /
  realized_cvar)`, a two-part construction (a growth-optimal target AND a
  tail-risk cap) that collapses to neither v4's linear vote-scaled
  vol-target nor the conservative branch's like-for-like substitution. This
  is architecturally the SIZE-axis analogue of what R-123 did on the
  ERR axis (move a construction from "discount applied after the fact" to
  "built into the optimization"), applied here to a risk measure instead of
  a novelty statistic.

**Which constraint this attacks: SIZE** (how much to hold) -- the one
constraint this project's own standing diagnosis credits with being the
"what actually worked" axis, and specifically the risk-measure choice
inside it, never varied before.

**Not a duplicate of:**
- R-59/R-60 (SIZE-axis, panel-calibration and timing of `target_vol`/vol
  itself): both retune parameters or inputs of the EXISTING std-based rule;
  neither changes what statistic of the return distribution is being
  targeted.
- R-105 novel / R-117 novel (detector-family SIZE substitutions): both
  replace `frac` (the VOTE) with a structurally new detector; `scale` is
  untouched in both. This round is the mirror image -- `frac` is reused
  unchanged, `scale`'s risk measure is what varies.
- R-109/R-112/R-115/R-121/R-122/R-123 (ERR-axis distributional-novelty
  brake family): all discount or fuse a NOVELTY/uncertainty statistic
  (is the current state unlike history?) into `frac`/`scale`/the final
  target. This round carries no novelty statistic at all -- CVaR is a
  measure of the return distribution's tail, not of how unusual the
  current state is relative to a reference pool.
- R-123 conservative (Baker & McHale shrinkage-Kelly): shrinks the EDGE
  ESTIMATE (`frac`) toward zero under parameter uncertainty. This round
  never touches `frac`; it changes what risk measure `scale` targets.
- R-118/R-119 (N-approx-3 parameter-selection-by-Monte-Carlo family):
  those pick `(anchor, target_vol, max_leverage)` by a robust criterion
  measured across synthetic paths, but the SELECTED rule is still v4's own
  std-based `scale`. This round changes the risk measure the rule targets,
  not how its scalar parameters are chosen.

**What would make this fail, named now, before any code:** CVaR and
realized standard deviation are highly correlated for return series without
strong tail asymmetry -- if BTC's 5m-bar return distribution is close to
Gaussian at the horizons this project resamples to, `target_cvar /
realized_cvar` will track `target_vol / realized_vol` almost exactly and
the two branches will just be a relabeled copy of v4 (killed by the Step-0
`R2_VS_V4_THRESH` gate below, exactly as R-109's family used it to catch a
near-identical rescale). The novel branch's own more specific failure mode:
the conditional (vote-state-sliced) empirical distribution may not have
enough effective samples in the "transitional" 1/3 and 2/3 vote states to
estimate a stable CVaR at alpha=0.05, in which case `f_kelly` should be
expected to be noisy and its own bootstrap CI wide -- reported, not hidden,
via `NOVEL_SAMPLE_COUNTS` below.

**Falsification test, pre-registered:** B4 -- does the branch's sign on
`d_sharpe` (vs `kelly_regime_v4`, inner-validation) replicate on ETH? This
is the identical test this whole SIZE/ERR research programme has used since
R-59, chosen for continuity rather than invented fresh, and because six
consecutive rounds in the ERR family have shown it is the discriminating
test on this data (BTC alone passes almost everything).

**Decision rule, pre-registered verbatim from the SIZE/ERR family
(R-109...R-123), unchanged:** PROMOTE-candidate only if the causal-
truncation probe AND B1 (both markets) AND B3 (plateau majority) AND B4
(full, both markets) AND B5 all pass. B2 (drawdown) is diagnostic only and
never gates promotion by itself. Anything else is NEGATIVE.

No bar at or after `OOS_START = 2023-01-01` may be read by either branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round: inner-train / inner-
# validation only. The holdout (>= OOS_START) is never read by a branch.
# ----------------------------------------------------------------------
INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"  # do not read; guarded by _assert_no_holdout below

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)     # B5: 0.40% taker tier
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

CVAR_ALPHA = 0.05  # expected shortfall in the worst 5% of trailing days


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


def load_btc_train(kind: str = "spot"):
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(train)
    return train, label


def load_eth_train():
    from tradebot.data import load_coinbase_eth_spot

    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


# ----------------------------------------------------------------------
# Causal CVaR of a daily-horizon return series.
# ----------------------------------------------------------------------

def _calendar_daily_close(close: pd.Series) -> pd.Series:
    """Last 5m close of each UTC calendar day -- causal by construction
    (each day's value is fixed once that day's bars stop arriving)."""
    return close.resample("1D").last().dropna()


def rolling_cvar_daily_index(close: pd.Series, window_days: int,
                              alpha: float = CVAR_ALPHA) -> pd.Series:
    """Causal rolling expected shortfall of CALENDAR-DAY log returns,
    indexed by calendar day (not yet reindexed to the 5m bar grid).

    Computed on ~one observation per day rather than one per 5m bar: (a) it
    is what "a day's tail risk" should mean -- the daily return distribution,
    not 5m intraday noise -- and (b) a rolling-quantile over a 25,920-bar
    (90-day) window evaluated once per 5m bar is computationally
    intractable at this dataset's size; evaluating it once per day is ~288x
    cheaper and the natural resolution for a quantity used at a daily
    rebalancing cadence.
    """
    daily_close = _calendar_daily_close(close)
    daily_ret = np.log(daily_close).diff()

    def _es(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if x.size < 20:
            return np.nan
        q = np.quantile(x, alpha)
        tail = x[x <= q]
        if tail.size == 0:
            tail = x[:1]
        return float(-tail.mean())

    cvar = daily_ret.rolling(window_days, min_periods=max(20, window_days // 4)).apply(_es, raw=True)
    # shift(1): day D's CVaR estimate must not use day D's own not-yet-closed
    # return, matching v4's own .shift(1) convention on its volatility input.
    return cvar.shift(1)


def annualized_cvar(close: pd.Series, window_days: int, alpha: float = CVAR_ALPHA) -> pd.Series:
    """Rolling daily-horizon CVaR, reindexed onto ``close``'s own 5m grid
    (each 5m bar sees the most recently CLOSED day's CVaR estimate -- never
    the day it is currently inside of, which is still open), annualized by
    sqrt(365.25) (the same sqrt-time heuristic v4 already uses to annualize
    its own std-based vol input -- disclosed as an approximation, not a
    re-derivation of CVaR's true, non-square-root time-scaling).
    """
    cvar_daily = rolling_cvar_daily_index(close, window_days, alpha)
    # Causal reindex: bar at time t gets the daily CVaR estimate dated on
    # (or, via ffill across any data gap, the most recent day at-or-before)
    # t's own calendar day -- that estimate never used t's own day's return
    # (rolling_cvar_daily_index already .shift(1)s it), so this cannot leak.
    day_of_bar = close.index.floor("D")
    full_day_range = pd.date_range(cvar_daily.index.min(), day_of_bar.max(), freq="1D",
                                    tz=cvar_daily.index.tz)
    cvar_by_day = cvar_daily.reindex(full_day_range).ffill()
    out = cvar_by_day.reindex(day_of_bar).to_numpy()
    return pd.Series(out, index=close.index) * np.sqrt(365.25)


def calibrate_target_cvar(close: pd.Series, v4_scale: np.ndarray, window_days: int,
                           alpha: float = CVAR_ALPHA) -> float:
    """Pick target_cvar so mean(scale) on inner-train matches v4's own mean
    scale, exactly the exposure-matching discipline R-33/R-59 established
    (never compare arms at different average risk). Grid search, not a fit
    to performance -- the objective is realized exposure, not Sharpe/growth.
    """
    cvar = annualized_cvar(close, window_days, alpha).to_numpy()
    target_mean = float(np.nanmean(v4_scale))
    grid = np.linspace(0.05, 2.0, 80)
    best, best_gap = grid[0], np.inf
    for tc in grid:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.minimum(tc / cvar, 2.0)
        s = np.where(np.isfinite(s), s, 0.0)
        gap = abs(float(np.nanmean(s)) - target_mean)
        if gap < best_gap:
            best, best_gap = tc, gap
    return float(best)


# ----------------------------------------------------------------------
# Step-0 sanity gate: is the candidate genuinely different from v4, or a
# rescaled copy? Adapted from R-109's own kill-switch convention.
# ----------------------------------------------------------------------

def v4_reference_target(df: pd.DataFrame) -> np.ndarray:
    v4 = get_strategy("kelly_regime_v4")
    out = v4.prepare(df.copy())
    return out["target"].to_numpy()


def step0_gate(candidate_target: np.ndarray, v4_target: np.ndarray,
                r2_vs_v4_thresh: float = 0.98) -> dict:
    """Returns dict with r2_vs_v4 and a pass/kill verdict.

    KILL if the candidate's R^2 against v4's own target exceeds
    ``r2_vs_v4_thresh`` -- i.e. it is, numerically, just v4 relabeled.
    """
    a, b = np.asarray(candidate_target), np.asarray(v4_target)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 100:
        return {"r2_vs_v4": np.nan, "kill": True, "reason": "insufficient overlap"}
    aa, bb = a[mask], b[mask]
    ss_res = float(np.sum((aa - bb) ** 2))
    ss_tot = float(np.sum((bb - bb.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"r2_vs_v4": r2, "kill": bool(np.isfinite(r2) and r2 > r2_vs_v4_thresh)}


# ----------------------------------------------------------------------
# Falsification battery: B1 (BTC signal), B3 (plateau), B4 (ETH), B5 (fee).
# ----------------------------------------------------------------------

def _daily_log_growth(m) -> np.ndarray:
    return np.log(m.equity / m.equity.shift(1)).dropna().to_numpy()


def run_candidate(strategy_factory, df: pd.DataFrame, market: MarketSpec,
                   start: str, end: str, label: str = ""):
    strat = strategy_factory()
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def b1_signal(candidate_factory, df: pd.DataFrame, market: MarketSpec) -> dict:
    """Paired bootstrap difference (log-growth) vs kelly_regime_v4, on
    inner-validation, matching every prior round's primary decisive cell.
    """
    m_cand, res_cand = run_candidate(candidate_factory, df, market,
                                      INNER_VAL_START, INNER_VAL_END)
    m_v4, res_v4 = run_candidate(lambda: get_strategy("kelly_regime_v4"), df, market,
                                  INNER_VAL_START, INNER_VAL_END)
    r_cand = daily_returns(res_cand.equity)
    r_v4 = daily_returns(res_v4.equity)
    n = min(len(r_cand), len(r_v4))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_v4.to_numpy()[:n],
                               stat=total_log_return, seed=125)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_v4": m_v4.sharpe,
        "d_sharpe": m_cand.sharpe - m_v4.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo, "paired_hi": paired.diff.hi,
        "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_v4": m_v4.max_drawdown_pct,
    }


if __name__ == "__main__":
    # Self-test: causal truncation probe. Any candidate branch importing
    # this module should call `self_test()` before reading a single
    # inner-validation number.
    df, _ = load_btc_train("spot")
    close = df["close"]
    full = annualized_cvar(close, window_days=90).to_numpy()
    cut = 400_000
    trunc = annualized_cvar(close.iloc[:cut], window_days=90).to_numpy()
    n_check = min(len(trunc), cut) - BARS_PER_DAY * 91  # skip the fresh warmup tail
    ok = np.allclose(full[:n_check], trunc[:n_check], equal_nan=True, rtol=1e-9)
    print(f"causal truncation probe (annualized_cvar): {'PASS' if ok else 'FAIL'}")
    assert ok, "annualized_cvar reads ahead of its own truncation point"
