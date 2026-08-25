"""Shared, read-only utilities for the R-130 round (08-25), CONSERVATIVE branch.

DIRECTION, in one sentence: two prior rounds (R-128, R-129) both tried
bolting a Kelly quadratic-cost no-trade BAND (Constantinides 1986; Davis &
Norman 1990) onto `hedge_experts`'s OUTPUT -- on the already-blended signal
(R-128), on each raw pre-blend expert (R-129 conservative), and on three
timescale-bucket sub-blends (R-129 novel) -- and all four constructions
failed (BTC passes but ETH inverts, or futures reverses the sign). R-129's
own closing line named the untried alternative explicitly: "a future
attempt on this object needs a cost model outside the Kelly-quadratic-cost
family entirely... not a third application point of the same algebra."
This round is that attempt: it makes `hedge_experts`'s own Hedge
weight-update recursion (the `logw`/`p` loop inside `HedgeExperts.prepare`)
itself cost-aware, rather than gating what happens after it computes the
blended target `x`.

**This branch (CONSERVATIVE): a literal port of a published cost-aware
online weight-update algorithm** -- Das, P., Johnson, N., & Banerjee, A.
(2013), "Online Lazy Updates for Portfolio Selection with Transaction
Costs," AAAI 2013, pp. 202-208. See
`experiments/r130_conservative_lazy_hedge.py`'s own module docstring for
the mechanism, the exact equations ported, what was adapted and why, and
the pre-registered failure modes -- this file is strategy-agnostic
plumbing only (data loading, splits, the frozen baseline runner), so it
can in principle be reused by a parallel branch, but this round does not
coordinate with or wait for one.

Not a duplicate of R-128/R-129: neither ever touches the Hedge weight
recursion itself -- both apply a downstream no-trade band to `x` (R-128)
or to the raw experts/bucket sub-blends that feed `x` (R-129). This round
never constructs a Kelly quadratic-cost band anywhere; it regularizes the
weight-update objective directly, per Das/Johnson/Banerjee's own
formulation, which is a different algebra family entirely.

No bar at or after `OOS_START = 2023-01-01` may be read by this branch.
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
from tradebot.inference import daily_returns, paired_bootstrap, total_log_return  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

# ----------------------------------------------------------------------
# Splits. Identical convention to every prior round: inner-train / inner-
# validation only. The holdout (>= OOS_START) is never read by this branch.
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

B3_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)


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
# Baseline (unmodified hedge_experts) run/metric helper.
# ----------------------------------------------------------------------

def run_baseline(df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    strat = get_strategy("hedge_experts")
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def run_strategy(strat, df: pd.DataFrame, market: MarketSpec, start: str, end: str,
                  label: str = ""):
    res = run_period(strat, df, start=start, end=end, market=market,
                      start_balance=1000.0, data_label=label)
    return compute_metrics(res), res


def sharpe_diff(res_a, res_b, mean_block: float = 30.0, n_boot: int = 2000, seed: int = 7):
    """Paired-bootstrap Sharpe difference (a - b) on aligned daily returns."""
    ra = daily_returns(res_a.equity)
    rb = daily_returns(res_b.equity)
    n = min(len(ra), len(rb))
    ra = ra.iloc[-n:].to_numpy()
    rb = rb.iloc[-n:].to_numpy()

    def _sharpe(x):
        x = np.asarray(x, dtype=float)
        sd = x.std(axis=-1, ddof=1)
        sd = np.where(sd <= 0, np.nan, sd)
        return np.nan_to_num(x.mean(axis=-1) / sd * np.sqrt(365.25))

    return paired_bootstrap(ra, rb, _sharpe, mean_block=mean_block, n_boot=n_boot, seed=seed)


if __name__ == "__main__":
    # Self-test: causal truncation probe on THIS module's own baseline
    # plumbing (hedge_experts.prepare() itself is covered by
    # tests/test_causality_strict.py at the framework level).
    df, label = load_btc_train("spot")
    m_full, _ = run_baseline(df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    df_trunc = df.loc[:INNER_VAL_END]
    m_trunc, _ = run_baseline(df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)

    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe (r130_shared baseline plumbing): "
          f"{'PASS' if ok else 'FAIL'} ({m_full.final_balance} vs {m_trunc.final_balance})")
    assert ok, "run_baseline reads ahead of its own truncation point"

    m_spot, _ = run_baseline(df, SPOT, None, INNER_TRAIN_END, label)
    m_fut, _ = run_baseline(df, FUTURES, None, INNER_TRAIN_END, label)
    print(f"baseline inner-train spot: trades={m_spot.num_trades} "
          f"final={m_spot.final_balance:.1f} sharpe={m_spot.sharpe:.3f}")
    print(f"baseline inner-train futures: trades={m_fut.num_trades} "
          f"final={m_fut.final_balance:.1f} sharpe={m_fut.sharpe:.3f}")
