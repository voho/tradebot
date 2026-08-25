"""Shared, read-only utilities and pre-registration for the R-130 round (08-25).

DIRECTION, in one sentence: constrain `kelly_regime_v4`'s realized rebalance
rate to an admissible turnover CORRIDOR, and only intervene once trailing
turnover has been running unusually heavy -- a rate/resource constraint on
top of v4's own already-decided target, not a new signal, and not another
application of the Kelly-quadratic no-trade-band algebra `kelly_regime_ev`
already uses.

Literature: Khubiev, Semenov, Podlipnova & Khubieva (2025/2026,
arXiv:2509.04541, "Finance-Grounded Optimization For Algorithmic Trading")
propose band turnover regularization -- a penalty that is zero while
trailing turnover sits inside a predefined admissible range and only bites
once it is exceeded, structurally different from a per-trade EV threshold.
Boyd, Busseti, Diamond, Kahn, Koh, Nystrup & Speth (2017, Foundations and
Trends in Optimization 3(1), "Multi-Period Trading via Convex Optimization")
frame turnover-penalized trading as a resource-constrained control problem;
the novel branch below is the causal, online, dual-variable analogue of that
control-theoretic framing rather than a closed-form solve.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r130_conservative_turnover_band.py`): track a causal
  trailing-turnover EWM of v4's own realized rebalances; while it sits
  inside a corridor `[0, TURNOVER_UPPER]`, rebalance exactly as v4 does
  today; once trailing turnover reaches the corridor's upper edge, DEFER
  (skip) the smallest-magnitude pending rebalance, with a pre-registered
  override -- a move never gets deferred if it is a full de-risking exit
  (`desired == 0`) or if `|desired - current|` exceeds
  `OVERRIDE_MULT * TURNOVER_UPPER`, so the mechanism can never indefinitely
  block a large de-risking trade.

- NOVEL (`r130_novel_turnover_throttle.py`): maintain a causal shadow price
  `lambda_t` on turnover, updated every bar by projected dual ascent,
  `lambda_{t+1} = clip(lambda_t + ETA * (turnover_ewm_t - TURNOVER_UPPER),
  0, LAMBDA_MAX)`; instead of a hard skip, SHRINK the size of the pending
  rebalance by `1 / (1 + lambda_t)` before executing -- a smooth,
  self-regulating control loop rather than a fixed threshold or a hard
  freeze. `lambda_t` decays back toward 0 on its own once trailing turnover
  falls back inside the corridor.

**Which constraint this attacks: COST** (costs that scale with the signal)
-- specifically a cost-model family this project has never tried: a
rate/resource constraint on the strategy's OWN trailing trading history,
rather than a per-trade marginal-value threshold.

**Not a duplicate of:**
- `kelly_regime_ev` / L-05 / L-06 (promoted): evaluates each rebalance's
  OWN marginal EV against a static fee/vol/horizon threshold
  (`|df| > 2*fee/(H*sigma^2)`), a per-decision equality. This round is
  inert unless MULTI-BAR trailing realized turnover has been running hot;
  a single large, isolated rebalance after a long quiet spell is untouched
  by either branch here, but would be judged solely on its own EV by
  `kelly_regime_ev`.
- Gârleanu-Pedersen partial adjustment (R-65/R-67/R-68): a closed-form LQ
  solution assuming a stationary, analytically-derived trading rate. Both
  branches here use an online, history-reactive dual variable / corridor,
  no closed-form solve.
- Almgren-Chriss adaptive execution urgency (R-56/R-77, B-24): changes HOW
  one already-decided trade fills (patience before a taker cross). Both
  branches here change WHETHER/HOW MUCH to rebalance at all, conditioned on
  multi-bar trailing history, not one trade's own fill schedule.
- Width-profile banding (R-66; Janeček & Shreve; Gerhold et al.): band
  width is a function of the CURRENT TARGET LEVEL `f`. Both branches here
  gate on TRAILING REALIZED TURNOVER HISTORY, a different state variable
  that does not depend on the level of `f` at all.
- `hedge_experts` pre/post-blend banding (R-128/R-129): those transplant
  the SAME Kelly-quadratic-cost algebra onto a different object
  (`hedge_experts`). This round keeps the object (`kelly_regime_v4`) and
  introduces the different cost-model family R-129's own closing line
  named as the open door.

**What would make this fail, named now, before any code:** v4's edge (per
L-01/R-62) concentrates around roughly three sudden regime transitions
(`STRESS_EPISODES` below spans six candidate dates). A throttle that damps
trading precisely when turnover is spiking -- which is exactly when a
regime transition also drives a burst of rebalances -- risks reproducing
the LAG failure that has now killed all 11 tried regime-timing mechanisms:
Sharpe and especially max drawdown could get WORSE around those dates.
Second, independent failure mode: v4 already trades rarely (174 trades over
9.6 years, full period), so a corridor derived from the fee-tier breakeven
turnover may sit far above v4's own natural rate and the mechanism may
simply never bind -- an INERTNESS failure (checked directly by A2 below,
before either branch's Step-B numbers are read).

**Falsification test, pre-registered:** B4 -- does the branch's sign on
`d_sharpe` (vs `kelly_regime_v4`, inner-validation) replicate on ETH? The
identical test this SIZE/ERR/COST research programme has used since R-59,
kept for continuity and because it has been the discriminating test on this
data. In addition, each branch carries its own named diagnostic (reported
regardless of B1-B5, never gates promotion by itself): the conservative
branch reports deferral behaviour AT each of the six `STRESS_EPISODES`
below (did the mechanism ever defer a rebalance within 3 days of a listed
episode date, and if so what the counterfactual v4 fill would have been);
the novel branch reports `lambda_t`'s own trajectory through the same six
episodes (does it spike WITH the turnover burst a transition causes, i.e.
does the throttle engage exactly when L-01's edge concentrates).

**Decision rule, pre-registered verbatim from the SIZE/ERR/COST family
(R-109...R-129), unchanged:** PROMOTE-candidate only if the non-inertness
gate (A2) AND B1 (both markets) AND B3 (plateau majority) AND B4 (full,
both markets) AND B5 all pass. B2 (drawdown) is diagnostic only and never
gates promotion by itself. Anything else is NEGATIVE.

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
from tradebot.strategies.kelly_regime import BARS_PER_DAY  # noqa: E402
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

# v4's own full-period fill spacing is ~1 rebalance / 3.3 days -> a natural
# per-bar rate of ~1/(3.3*BARS_PER_DAY). The corridor upper edge is set at
# 3x that natural rate (a deliberately loose corridor so the mechanism binds
# only on genuine turnover BURSTS, not on ordinary trading) -- derived from
# an observable (v4's own measured spacing, same source kelly_regime_ev's
# horizon_days uses), not fit to performance.
V4_NATURAL_TRADES_PER_DAY = 1.0 / 3.3
TURNOVER_EWM_SPAN_DAYS = 30
TURNOVER_UPPER = 3.0 * V4_NATURAL_TRADES_PER_DAY  # trades/day, EWM units
OVERRIDE_MULT = 2.0     # conservative: never defer a move > this x the corridor
ETA = 0.5                # novel: dual-ascent step size
LAMBDA_MAX = 20.0        # novel: throttle cap (shrink factor floor 1/21)

STRESS_EPISODES = [
    ("2018 bear onset (post-Dec-2017 top)", "2018-01-17"),
    ("2018 bear bottom / capitulation", "2018-12-15"),
    ("2020-03 COVID crash", "2020-03-12"),
    ("2021-11 top / 2022 bear transition", "2021-11-10"),
    ("2022-05 Terra/Luna collapse", "2022-05-09"),
    ("2022-11 FTX collapse", "2022-11-08"),
]


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
# Causal trailing turnover: an EWM of |rebalance size| per bar, expressed
# in trades/day units comparable to TURNOVER_UPPER above.
# ----------------------------------------------------------------------

def trailing_turnover_ewm(rebalance_events: np.ndarray, span_days: int = TURNOVER_EWM_SPAN_DAYS) -> np.ndarray:
    """`rebalance_events[i]` is 1.0 if bar i fired a rebalance, else 0.0.

    Returns a causal EWM (in units of events/day) using only events at or
    before i -- ``pandas.Series.ewm`` is itself causal (each output uses
    only current and past inputs), so no additional ``.shift`` is required
    here; the caller decides whether the strategy acts on bar i's own EWM
    value (this round's convention, since the event at i has already been
    decided by the time the EWM is read for bar i's own throttle check) or
    a lagged one.
    """
    s = pd.Series(rebalance_events).ewm(span=int(span_days * BARS_PER_DAY), min_periods=1).mean()
    return (s.to_numpy() * BARS_PER_DAY)  # events/bar -> events/day


# ----------------------------------------------------------------------
# Step-0 sanity gate: A2 non-inertness -- does the mechanism ever actually
# bind? If it never triggers, the branch is untested, not negative (per
# ROUTINE.md's "not tested is not a negative result").
# ----------------------------------------------------------------------

def a2_non_inertness(n_interventions: int) -> dict:
    return {"n_interventions": int(n_interventions), "pass": n_interventions > 0}


# ----------------------------------------------------------------------
# Falsification battery: B1 (BTC signal, inner-val), B3 (plateau), B4 (ETH).
# ----------------------------------------------------------------------

def run_candidate(strategy_factory, df: pd.DataFrame, market: MarketSpec,
                   start: str, end: str, label: str = ""):
    strat = strategy_factory()
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def b1_signal(candidate_factory, df: pd.DataFrame, market: MarketSpec, seed: int = 130) -> dict:
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
                               stat=total_log_return, seed=seed)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_v4": m_v4.sharpe,
        "d_sharpe": m_cand.sharpe - m_v4.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo, "paired_hi": paired.diff.hi,
        "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_v4": m_v4.max_drawdown_pct,
        "trades_cand": m_cand.num_trades, "trades_v4": m_v4.num_trades,
    }


if __name__ == "__main__":
    # Self-test: causal truncation probe on trailing_turnover_ewm. Any
    # candidate branch importing this module should call a version of this
    # before reading a single inner-validation number.
    rng = np.random.default_rng(130)
    events = (rng.random(500_000) < 0.0005).astype(float)
    full = trailing_turnover_ewm(events)
    cut = 300_000
    trunc = trailing_turnover_ewm(events[:cut])
    n_check = cut - BARS_PER_DAY * (TURNOVER_EWM_SPAN_DAYS + 1)
    ok = np.allclose(full[:n_check], trunc[:n_check], equal_nan=True, rtol=1e-9)
    print(f"causal truncation probe (trailing_turnover_ewm): {'PASS' if ok else 'FAIL'}")
    assert ok, "trailing_turnover_ewm reads ahead of its own truncation point"
