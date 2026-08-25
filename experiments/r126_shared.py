"""Shared, read-only utilities and pre-registration for the R-126 round (08-25).

DIRECTION, in one sentence: replace `champions_council`'s own capital
ALLOCATION rule across its six member signals -- Hedge / multiplicative
weights (Freund & Schapire 1997, JCSS; Arora, Hazan & Kale 2012), an
average-case *regret* guarantee with no explicit tail-risk objective --
with a risk-budgeted allocation that targets the members' *joint risk*
directly: Equal Risk Contribution (Maillard, Roncalli & Teiletche 2010,
"The Properties of Equally Weighted Risk Contribution Portfolios",
J. Portfolio Management 36(4), 60-70) for the conservative branch, and a
CVaR-budgeted convex reallocation (Rockafellar & Uryasev 2000,
"Optimization of Conditional Value-at-Risk", J. Risk 2(3), 21-41) for the
novel branch.

**Why this and not another `kelly_regime_v4` variant.** Per this round's
own Step-0 diligence (recorded in the R-126 ledger entry): the single-asset
`kelly_regime_v4` axis is closed across INFO (19 signals), SIZE (27+
attempts including R-125's own risk-measure substitution), ERR (5 notions
of uncertainty), regime-timing (11 mechanisms) and N-approx-3 calibration
(4 procedures); the multi-asset panel axis is closed across 11 rounds. Every
one of those ~125 rounds varied something INSIDE `kelly_regime_v4`'s own
vote/scale construction, or inside a panel of *assets*. `champions_council`
(L-08, 08-14) is a different object entirely -- a portfolio of *strategies*,
not of assets or of one strategy's internal signal -- and its own capital
allocation rule (Hedge, eta=0.06, fixed_share=1e-4) has never been touched
by any of the 125 subsequent rounds (grep-confirmed against
`docs/LEDGER.md`: the only two hits for "champions_council" outside its own
registration and description are R-29's blanket 25-strategy holdout sweep
and a walk-forward comparison in `scripts/experiment.py` -- neither varies
its mechanism). R-107's risk-parity work (Maillard-Roncalli-Teiletche, the
same paper) reweighted a panel of *assets* (BCH/LTC/ETC/DASH/LINK/XTZ);
R-125's CVaR work substituted CVaR into `kelly_regime_v4`'s *own* internal
`scale`. Neither touches the *between-strategy* allocation question this
round asks.

**Which constraint this attacks: SIZE** -- specifically, how much capital to
allocate to each of several already-profitable strategies, a portfolio-level
sizing decision this project's own standing diagnosis has never varied,
distinct from every prior SIZE-axis attempt (all internal to one strategy).

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r126_conservative_erc_council.py`): periodically (every
  `REBALANCE_DAYS` calendar days, structural not fit) set each member's
  weight inversely proportional to its own trailing realized volatility of
  daily vol-normalized payoff (`w_i proportional to 1/std_i`), the classical
  Equal Risk Contribution special case that is exact when members are
  uncorrelated and a documented approximation otherwise (Maillard, Roncalli
  & Teiletche 2010, Section 2; the same special case already used, on a
  different object, in this project's own ruled-out row "Inverse-trailing-
  volatility weighting ... of a periodically-rebalanced BTC+ETH
  kelly_regime_v4 portfolio"). Weights renormalized to the simplex
  (non-negative, sum to 1) each rebalance. Everything downstream of the
  weight vector -- the vol-target scale, deadband, member signals
  themselves -- is byte-identical to `champions_council`.

- NOVEL (`r126_novel_cvar_council.py`): periodically re-solves, on the
  trailing `LOOKBACK_DAYS` window of daily member payoffs only, for the
  simplex weight vector `w` that minimizes the blended portfolio's
  Conditional Value-at-Risk at `CVAR_ALPHA`, subject to a minimum expected
  daily-payoff floor (the trailing cross-member median), via the Rockafellar
  & Uryasev (2000) linear formulation `CVaR_alpha(L) = min_zeta{zeta +
  1/((1-alpha)*T) * sum_t max(0, L_t - zeta)}` where `L_t = -sum_i w_i
  payoff_i,t` -- solved by dependency-free projected subgradient descent
  over `(w, zeta)` (no scipy in this sandbox, matching R-125 novel's own
  "dependency-free golden-section search" precedent), the return floor
  enforced via a Lagrange multiplier found by a coarse grid + bisection over
  the dual (structural search bounds, not a fit to performance). This is a
  genuinely different derivation from the conservative branch, not a
  rescaling of it: ERC targets each member's *marginal* variance
  contribution equally; CVaR-budgeting targets the *joint* tail of the
  blended portfolio directly and can concentrate weight on members whose
  tails are uncorrelated with the rest even if their own standalone
  volatility is not the lowest.

**Not a duplicate of:**
- L-08 (`champions_council`'s own registration): defines the Hedge
  allocation this round replaces; never varied since.
- R-107 (risk-parity / Maillard-Roncalli-Teiletche): applied to a panel of
  six *asset* exposures (BCH/LTC/ETC/DASH/LINK/XTZ), not to
  `champions_council`'s four-strategy-plus-two-benchmark member set. Same
  paper, structurally different object -- disclosed explicitly rather than
  hidden, and the reason this round is titled "council" not "panel".
- R-125 (CVaR conservative/novel): substituted CVaR into `kelly_regime_v4`'s
  own internal `scale = min(target_x/realized_x, max_leverage)` term, a
  single-strategy sizing rule. This round applies CVaR-budgeting to a
  *cross-strategy allocation weight vector*, a different mathematical
  object (a simplex-constrained portfolio weight solved by convex
  optimization, not a scalar leverage multiplier).
- Every `kelly_regime_v4`-internal SIZE/ERR/regime-timing round (R-34
  through R-125): none touch `champions_council`'s allocation mechanism,
  which does not import or call `kelly_regime_v4` at all (it uses the base
  `kelly_regime`, per its own `_members()`).

**What would make this fail, named now, before any code:** three of
`champions_council`'s four active members (`kelly_regime`, `hedge_experts`,
`replicator_book`) are all directionally long-biased trend/momentum
strategies on the *same* underlying BTC or ETH price series -- if their
daily payoffs are highly cross-correlated (plausible, given R-63's own
finding that this project's asset panel carries a 0.634 mean pairwise
correlation and a Grinold breadth of 1.47 of 8 -- the analogous number for
*strategies* sharing one price series could be worse, not better), then (a)
Step-0 may show both candidates converge to weight paths close to Hedge's
own (killed by the `R2_VS_COUNCIL_THRESH` gate below), and (b) with only 6
members and short rebalance windows there may not be enough effective
independent bets to estimate a stable ERC/CVaR weight vector at all --
reported via `EFFECTIVE_MEMBER_COUNT` below, not hidden. The CVaR branch's
own more specific risk: a 6-asset LP solved by subgradient descent on
`LOOKBACK_DAYS`-worth of daily data (few hundred points) may not converge to
a stable interior solution and could degenerate to a corner (single-member)
weight, which is disclosed via the solver's own convergence diagnostics
(`NOVEL_SOLVER_DIAGNOSTICS`).

**Falsification test, pre-registered:** B4 -- does the branch's sign on
`d_sharpe` (candidate vs `champions_council`, inner-validation) replicate on
ETH? Chosen for continuity with the whole SIZE/ERR research programme since
R-59 rather than invented fresh.

**Decision rule, pre-registered, matching the SIZE/ERR family's own
convention (R-109...R-125):** PROMOTE-candidate only if the causal-
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
from tradebot.data import load_coinbase_eth_spot, load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.champions_council import BARS_PER_DAY, BARS_PER_YEAR  # noqa: E402
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

REBALANCE_DAYS = 30       # structural (monthly), matches R-107/B-19 convention
LOOKBACK_DAYS = 90        # trailing window for both branches' weight fits
CVAR_ALPHA = 0.05
N_MEMBERS = 6             # kelly_regime, hedge_experts, replicator_book,
                           # universal_kelly, buy_and_hold, flat


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
    eth = load_coinbase_eth_spot(ROOT / "data")
    assert eth is not None, "ETH spot data not committed"
    eth = eth.loc[:INNER_VAL_END].copy()
    _assert_no_holdout(eth)
    return eth


# ----------------------------------------------------------------------
# Member signal matrix -- byte-identical to champions_council's own
# _members() plus its own buy-and-hold/flat appends. Frozen here so both
# branches share the exact same member set and the same clipping.
# ----------------------------------------------------------------------

def member_names() -> list[str]:
    return ["kelly_regime", "hedge_experts", "replicator_book", "universal_kelly",
            "buy_and_hold", "flat"]


def member_signal_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_bars, N_MEMBERS) matrix of each member's clipped target signal,
    identical construction to ChampionsCouncil.prepare()."""
    from tradebot.strategies.hedge_experts import HedgeExperts
    from tradebot.strategies.kelly_regime import KellyRegime
    from tradebot.strategies.replicator_book import ReplicatorBook
    from tradebot.strategies.universal_kelly import UniversalKelly

    base = df[["open", "high", "low", "close", "volume"]]
    members = [KellyRegime(), HedgeExperts(), ReplicatorBook(), UniversalKelly()]
    signals = []
    for member in members:
        prepared = member.prepare(base.copy())
        signals.append(np.clip(np.nan_to_num(
            prepared["target"].to_numpy(dtype=np.float64)), -1.0, 1.0))
    signals.append(np.ones(len(df)))   # buy-and-hold
    signals.append(np.zeros(len(df)))  # flat
    return np.column_stack(signals)


def member_daily_payoffs(df: pd.DataFrame, a: np.ndarray) -> pd.DataFrame:
    """Causal per-bar payoff `a[i-1] * r[i]` for each member, resampled to
    one row per UTC calendar day (sum of that day's bar payoffs) -- the
    same "daily, not 5m-bar" resampling convention `tradebot.inference`
    uses throughout, needed because a block bootstrap or a risk fit over
    autocorrelated 5m bars is both intractable and dishonest about the
    effective sample size (see that module's own docstring).

    Strictly causal: bar i's payoff uses `a[i-1]` (yesterday's-or-earlier
    close's signal) and `r[i]` (the return realized *into* bar i's close),
    exactly mirroring champions_council's own per-bar Hedge payoff term.
    """
    r = np.log(df["close"]).diff().to_numpy()
    payoff = np.zeros_like(a)
    payoff[1:, :] = a[:-1, :] * r[1:, None]
    payoff_df = pd.DataFrame(payoff, index=df.index, columns=member_names())
    return payoff_df.resample("1D").sum()


# ----------------------------------------------------------------------
# Step-0 sanity gate: is the candidate genuinely different from
# champions_council's own Hedge blend, or a rescaled copy?
# ----------------------------------------------------------------------

def council_reference_target(df: pd.DataFrame) -> np.ndarray:
    council = get_strategy("champions_council")
    out = council.prepare(df.copy())
    return out["target"].to_numpy()


def step0_gate(candidate_target: np.ndarray, council_target: np.ndarray,
               r2_thresh: float = 0.98) -> dict:
    a, b = np.asarray(candidate_target), np.asarray(council_target)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 100:
        return {"r2_vs_council": np.nan, "kill": True, "reason": "insufficient overlap"}
    aa, bb = a[mask], b[mask]
    ss_res = float(np.sum((aa - bb) ** 2))
    ss_tot = float(np.sum((bb - bb.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"r2_vs_council": r2, "kill": bool(np.isfinite(r2) and r2 > r2_thresh)}


# ----------------------------------------------------------------------
# Weight -> position: identical downstream construction to
# champions_council.prepare()'s own vol-target/deadband tail, so only the
# weight vector `w(t)` differs between branches and the Hedge baseline.
# ----------------------------------------------------------------------

def weights_to_target(df: pd.DataFrame, a: np.ndarray, weight_schedule: pd.DataFrame,
                       target_vol: float = 0.55, max_leverage: float = 2.0,
                       vol_span: int = 8 * BARS_PER_DAY, deadband: float = 0.10) -> np.ndarray:
    """`weight_schedule`: DataFrame indexed by calendar day, columns =
    member_names(), the weight vector effective *starting* that day
    (computed causally from data strictly before that day -- enforced by
    each branch's own fit function, not here).
    """
    r = np.log(df["close"]).diff()
    vol = (r.ewm(span=vol_span, min_periods=BARS_PER_DAY).std()
           * np.sqrt(BARS_PER_YEAR)).shift(1).to_numpy()

    day_of_bar = df.index.floor("D")
    w_daily = weight_schedule.reindex(pd.date_range(
        weight_schedule.index.min(), day_of_bar.max(), freq="1D",
        tz=weight_schedule.index.tz)).ffill().bfill()
    w_full = w_daily.reindex(day_of_bar).to_numpy()  # (n_bars, N_MEMBERS)

    n = len(df)
    target = np.zeros(n)
    pos = 0.0
    for i in range(1, n):
        v = vol[i]
        if not np.isfinite(v) or v <= 0 or not np.all(np.isfinite(w_full[i])):
            target[i] = pos
            continue
        blend = float(w_full[i] @ a[i])
        desired = blend * min(target_vol / v, max_leverage)
        if abs(desired - pos) > deadband:
            pos = desired
        target[i] = pos
    return target


# ----------------------------------------------------------------------
# Falsification battery: B1 (BTC/ETH signal vs champions_council), B5 (fee).
# B3 (plateau) and B4 (ETH) are run by each branch over its own grid using
# these primitives.
# ----------------------------------------------------------------------

def run_target_series(target: np.ndarray, df: pd.DataFrame, market: MarketSpec,
                       start: str, end: str, label: str = ""):
    """Backtest a precomputed target-position series via a thin wrapper
    strategy (so both branches evaluate through the same engine
    champions_council itself uses, not a hand-rolled PnL calc)."""
    from tradebot.strategy import Context, Strategy

    class _Frozen(Strategy):
        name = "r126_frozen"
        warmup = 100 * BARS_PER_DAY + 10

        def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame["target"] = target[:len(frame)]
            return frame

        def on_bar(self, ctx: Context) -> None:
            t = float(ctx.bar["target"])
            prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
            if abs(t - prev) > 1e-9:
                ctx.order_notional(t)

    res = run_period(_Frozen(), df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def b1_signal(candidate_target: np.ndarray, df: pd.DataFrame, market: MarketSpec) -> dict:
    m_cand, res_cand = run_target_series(candidate_target, df, market,
                                          INNER_VAL_START, INNER_VAL_END)
    m_council, res_council = run_candidate_council(df, market)
    r_cand = daily_returns(res_cand.equity)
    r_council = daily_returns(res_council.equity)
    n = min(len(r_cand), len(r_council))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_council.to_numpy()[:n],
                               stat=total_log_return, seed=126)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_council": m_council.sharpe,
        "d_sharpe": m_cand.sharpe - m_council.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo, "paired_hi": paired.diff.hi,
        "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_council": m_council.max_drawdown_pct,
    }


def run_candidate_council(df: pd.DataFrame, market: MarketSpec, start: str = INNER_VAL_START,
                           end: str = INNER_VAL_END):
    strat = get_strategy("champions_council")
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label="")
    return compute_metrics(res), res


if __name__ == "__main__":
    # Self-test: causal truncation probe on member_daily_payoffs. Any
    # candidate branch importing this module should call self_test() before
    # reading a single inner-validation number.
    df, _ = load_btc_train("spot")
    a_full = member_signal_matrix(df)
    payoff_full = member_daily_payoffs(df, a_full)

    cut = 400_000
    df_trunc = df.iloc[:cut]
    a_trunc = member_signal_matrix(df_trunc)
    payoff_trunc = member_daily_payoffs(df_trunc, a_trunc)

    common_days = payoff_trunc.index[payoff_trunc.index.isin(payoff_full.index)]
    # drop the last couple of days of the truncated frame: a partial UTC day
    # at the cut point sums fewer bars than the full-series version of that
    # same day, which is expected (not a leak) -- exclude it from the check.
    common_days = common_days[:-2]
    ok = np.allclose(payoff_full.loc[common_days].to_numpy(),
                      payoff_trunc.loc[common_days].to_numpy(), atol=1e-12)
    print(f"causal truncation probe (member_daily_payoffs): {'PASS' if ok else 'FAIL'}")
    assert ok, "member_daily_payoffs reads ahead of its own truncation point"
