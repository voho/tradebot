"""Shared, read-only utilities and pre-registration for the R-128 round (08-25).

DIRECTION, in one sentence: replace `hedge_experts`'s own FIXED
`hysteresis=0.05` re-target threshold on its blended output -- a hand-set
number, never derived, of the exact kind this project's own methodology
finding 7 (`docs/RESEARCH.md`) flags as suspect ("the deadband should be
derived, not chosen") -- with a band derived from the fee/volatility/
horizon algebra that already earned a promotion once (`kelly_regime_ev`,
L-nn, Constantinides 1986; Davis & Norman 1990), applied for the first
time to a DIFFERENT strategy object.

**Why `hedge_experts` and not another `kelly_regime_v4`/`champions_council`
variant.** Per this round's own Step-0 diligence (recorded in the R-128
ledger entry): the single-asset `kelly_regime_v4` axis is closed across
INFO (19+ signals), SIZE (28+ attempts), ERR (5 notions of uncertainty),
regime-timing (11 mechanisms) and N-approx-3 calibration (4 procedures);
the multi-asset panel axis is closed across 11 rounds; `champions_council`'s
own cross-strategy allocation mechanism was tried and closed NEGATIVE by
R-126 (both branches); R-127 diagnosed the recurring BTC-pass/ETH-invert
signature itself rather than a new mechanism. Every one of those ~127
rounds varied something INSIDE `kelly_regime_v4`'s vote/scale, INSIDE
`champions_council`'s allocation weights, or INSIDE a panel of *assets*.
`hedge_experts` (L-09, registered 08-12) is a different OBJECT that has
never had its own internal re-target rule varied by any round --
grep-confirmed against `docs/LEDGER.md`: `hedge_experts` appears in only 6
lines total, none of them about its `hysteresis`/deadband/no-trade
mechanism (the hits are its own registration row, two mentions inside
`champions_council`/`r126_shared.py`'s member list, and two unrelated
holdout-accounting footnotes).

**Which constraint this attacks: COST** -- `hedge_experts` is profitable on
spot ($13.3K) but "over-trades on leverage" (STRATEGIES.md's own words,
4,103 trades on the full period) and loses to fees on futures ($258). Its
`on_bar` re-targets whenever `abs(x - pos) > self.hysteresis` for a FIXED
`hysteresis=0.05` -- structurally the same defect `kelly_regime`'s original
fixed 10% deadband had before `kelly_regime_ev` derived
`|delta_f| > 2*fee/(H*sigma^2)` from first principles and shipped it
(finding 7, `docs/RESEARCH.md`): "at a 0.10% fee the derived band is ~3x
the hand-set 10%... turnover, not signal, was the binding cost." Nobody has
asked whether the same defect, on this different strategy, has the same
fix.

**Mechanism, one sentence per branch, before any code was run:**

- CONSERVATIVE (`r128_conservative_ev_band.py`): the literal
  `kelly_regime_ev` transplant -- replace the fixed `hysteresis=0.05` test
  with `abs(x - pos) > 2*fee/(H*sigma^2)`, `sigma` = `hedge_experts`'s own
  already-computed `sig1` (EWM realized vol), `H` = a SINGLE frozen
  constant measured the same way `kelly_regime_ev`'s 3.3-day default was
  measured -- `hedge_experts`'s OWN current (fixed-hysteresis) realized
  fill spacing on inner-train, pooled across both markets (below). Minimal
  new code: same derivation, same algebra, a different host strategy.

- NOVEL (`r128_novel_adaptive_band.py`): the SAME quadratic-cost-vs-linear-
  fee algebra, but with `H` computed ONLINE per-bar from the blend signal
  `x`'s own trailing autocorrelation-implied persistence (a causal EWM
  AR(1) half-life on `x`, converted to a horizon in days) instead of one
  fixed constant measured once. This is the "state-dependence of the
  horizon" axis R-89's own literature commission (finding 11,
  `docs/RESEARCH.md`) named as one of only three legitimate ways to vary a
  single-instrument trend/regime rule (Levine & Pedersen 2016) -- and
  explicitly the one axis R-89 itself did not take (it varied nonlinearity
  and path-dependence). B-40 later closed ONE instantiation of horizon
  state-dependence (Goulding-Harvey-Mazzoleni's four-state vote reweighting,
  REJECTED at the sub-claim stage, R-108) -- a different quantity
  (the VOTE's own blend weights) made state-dependent by a different
  mechanism (a discrete 4-state classifier) than this round's (the
  REBALANCE BAND's horizon input, made state-dependent by a continuous
  AR(1)-persistence estimate). Not a re-run of B-40.

**Not a duplicate of:**
- `kelly_regime_ev`/`kelly_regime_ev_fast` (L-nn): same algebra, applied to
  `kelly_regime_v4`, a strategy with a single homogeneous-horizon vote, not
  to `hedge_experts`'s 10-expert multi-timescale blend.
- B-29/B-31/B-34/B-35/B-37/B-40's whole entry/exit-band research line
  (R-64 through R-69, R-108): all attack `xsmom_entry_band`'s
  cross-sectional multi-asset ENTRY/EXIT eligibility threshold on the
  `src/tradebot/multi_engine.py` panel path -- a structurally different
  object (a panel selection rule, not a single-instrument position-update
  rule) on a different engine.
- R-126 (`champions_council` ERC/CVaR allocation): varies the WEIGHT this
  project puts on each of several strategies; this round varies ONE
  strategy's own internal re-target rule and never touches allocation
  weights.
- R-42/R-50 (`kelly_regime_covkelly`'s rebalance-engine restart artifact):
  a different bug class (segment-restart NaN propagation) on a different,
  unregistered multi-asset construction.

**What would make this fail, named now, before any code:**
1. `hedge_experts`'s current 0.05 hysteresis might already be an accidental
   near-optimum (its own docstring already credits it with "sparse
   re-targets") -- a derived band could land close to 0.05 and show a null
   result, which is itself informative (the existing number was not
   costing much) rather than a bug.
2. Six independent prior mechanisms/objects on this project have passed a
   BTC promotion gate and inverted sign on ETH (R-109, R-113,
   R-115-conservative, R-125-conservative, R-126 both branches) -- there is
   a real, named prior that a seventh construction does the same. B4 below
   is the test built to catch exactly this, and an inversion here would be
   consistent with, not contradictory to, R-127's own finding.
3. `kelly_regime_ev`'s derivation assumes a single, roughly time-invariant
   Kelly-optimal exposure `f*` whose OWN dynamics do not need modeling --
   `hedge_experts` blends 10 experts across four momentum horizons
   (1h/6h/1d/1w) plus MACD/RSI/reversion/Donchian, so its own "target" is
   not one stationary process; a band-width algebra built for a single
   homogeneous bet may be structurally mismatched to a genuinely
   multi-timescale blend. This is the sharpest reason either branch could
   fail even before ETH is read.
4. A band wide enough to cut turnover materially could reproduce the LAG
   failure this project has now measured in every one of 11 regime-timing
   mechanisms on `kelly_regime_v4` -- slowing `hedge_experts` down enough
   to give back the responsiveness that makes it profitable on spot in the
   first place.

**Falsification test, pre-registered:** B4 -- does the candidate's
`d_sharpe` sign (candidate vs `hedge_experts`, inner-validation) replicate
on ETH? Chosen for continuity with the whole SIZE/ERR/COST research
programme since R-59/R-64.

**Decision rule, pre-registered, matching the SIZE/ERR/COST family's own
convention (R-64...R-126):** PROMOTE-candidate only if the causal-
truncation probe AND B1 (both markets, full period AND inner-validation)
AND B3 (plateau majority across a param-multiplier grid) AND B4 (sign
replicates on ETH) AND B5 (no sign flip at 0.40% fee) all pass. B2
(drawdown/turnover) is diagnostic only and never gates promotion by
itself. Anything else is NEGATIVE.

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
from tradebot.strategy import Context, Strategy  # noqa: E402
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
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)      # B5: 0.40% taker tier
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# ----------------------------------------------------------------------
# Frozen horizon input, measured (not fit) BEFORE either branch was
# dispatched: hedge_experts' OWN current fixed-hysteresis fill spacing on
# inner-train, pooled across both markets (spot 713 trades / 1460 days =
# 2.048 d/trade; futures 1543 trades / 1460 days = 0.946 d/trade; pooled
# 2256 trades / 2920 market-days = 1.294 d/trade). Mirrors kelly_regime_ev's
# own precedent of using the pre-existing strategy's realized fill spacing
# as the EV-band's horizon input, tightened to inner-train only (that
# strategy's 3.3-day figure was measured over its full 9.6-year history).
# ----------------------------------------------------------------------
HORIZON_DAYS_FROZEN = 1.294


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
# Baseline (unmodified hedge_experts) run/metric helpers, shared so both
# branches score the identical reference.
# ----------------------------------------------------------------------

def run_baseline(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    strat = get_strategy("hedge_experts")
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def run_target_series(target: np.ndarray, df: pd.DataFrame, market: MarketSpec,
                       start: str, end: str, label: str = ""):
    """Backtest a precomputed target-position series through the same
    engine every strategy uses (a thin frozen wrapper), so a branch only
    supplies a `target` column, not its own PnL accounting."""

    class _Frozen(Strategy):
        name = "r128_frozen"
        warmup = 2500

        def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame["target"] = target[: len(frame)]
            return frame

        def on_bar(self, ctx: Context) -> None:
            t = float(ctx.bar["target"])
            prev = float(ctx.prev["target"]) if ctx.prev is not None else 0.0
            if abs(t - prev) > 1e-9:
                ctx.order_target(t)

    res = run_period(_Frozen(), df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def b1_signal(candidate_target: np.ndarray, df: pd.DataFrame, market: MarketSpec,
              start: str = INNER_VAL_START, end: str = INNER_VAL_END) -> dict:
    m_cand, res_cand = run_target_series(candidate_target, df, market, start, end)
    m_base, res_base = run_baseline(df, market, start, end)
    r_cand = daily_returns(res_cand.equity)
    r_base = daily_returns(res_base.equity)
    n = min(len(r_cand), len(r_base))
    paired = paired_bootstrap(r_cand.to_numpy()[:n], r_base.to_numpy()[:n],
                               stat=total_log_return, seed=128)
    return {
        "sharpe_cand": m_cand.sharpe, "sharpe_base": m_base.sharpe,
        "d_sharpe": m_cand.sharpe - m_base.sharpe,
        "paired_diff": paired.diff.point, "paired_lo": paired.diff.lo,
        "paired_hi": paired.diff.hi, "significant": paired.significant,
        "dd_cand": m_cand.max_drawdown_pct, "dd_base": m_base.max_drawdown_pct,
        "trades_cand": m_cand.num_trades, "trades_base": m_base.num_trades,
        "final_cand": m_cand.final_balance, "final_base": m_base.final_balance,
    }


if __name__ == "__main__":
    # Self-test: causal truncation probe. Any candidate branch importing
    # this module should call self_test() before reading a single
    # inner-validation number. hedge_experts.prepare() is already covered
    # by tests/test_causality_strict.py at the framework level; this
    # checks THIS module's own baseline/target-series plumbing adds no
    # extra lookahead.
    df, label = load_btc_train("spot")
    m_full, _ = run_baseline(df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    df_trunc = df.loc[:INNER_VAL_END]
    m_trunc, _ = run_baseline(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe (r128_shared baseline plumbing): "
          f"{'PASS' if ok else 'FAIL'} ({m_full.final_balance} vs {m_trunc.final_balance})")
    assert ok, "run_baseline reads ahead of its own truncation point"

    print(f"HORIZON_DAYS_FROZEN = {HORIZON_DAYS_FROZEN}")
    m_spot, _ = run_baseline(df, SPOT, None, INNER_TRAIN_END, label)
    m_fut, _ = run_baseline(df, FUTURES, None, INNER_TRAIN_END, label)
    print(f"baseline inner-train spot: trades={m_spot.num_trades} "
          f"final={m_spot.final_balance:.1f} sharpe={m_spot.sharpe:.3f}")
    print(f"baseline inner-train futures: trades={m_fut.num_trades} "
          f"final={m_fut.final_balance:.1f} sharpe={m_fut.sharpe:.3f}")
