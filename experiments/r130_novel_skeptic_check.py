"""SKEPTIC audit scratch for R-130 NOVEL (CAPEHedge). Read-only re-derivation.

Nothing here is a research claim; it exists so the audit's numbers are
operator-reproducible. No bar at or after 2023-01-01 is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.data import load_coinbase_eth_spot, load_dataset
from tradebot.inference import annualized_sharpe, daily_returns, paired_bootstrap
from tradebot.metrics import compute_metrics
from tradebot.registry import get_strategy
from tradebot.strategies.hedge_experts import HedgeExperts
from tradebot.strategy import Context, Strategy
from tradebot.window import run_period

sys.path.insert(0, str(ROOT / "experiments"))
from r130_novel_commission_avoidant import CAPEHedge, LAMBDA_BASE  # noqa: E402

INNER_TRAIN_START = "2017-01-01"
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"
OOS_START = "2023-01-01"

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
SPOT_HIGH_FEE = MarketSpec.spot(fee_rate=0.0040)
FUTURES_HIGH_FEE = MarketSpec.futures(leverage=5.0, fee_rate=0.0040)


class FixedWHedge(Strategy):
    """Ablation: the SAME played rule x_t = w*g_t + (1-w)*x_{t-1} with w a
    CONSTANT (no learning, no lambda, no gradient, no A_t). If this matches
    CAPEHedge, the online-learning machinery is decorative and the
    mechanism is a fixed-alpha EMA / smooth partial adjustment."""

    name = "skeptic_fixed_w"
    warmup = 2500

    def __init__(self, w: float = 0.386) -> None:
        self.w = float(w)
        self._held = None
        self._last_target = None
        self.n_retarget_calls = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        base = HedgeExperts()
        prepared = base.prepare(df.copy())
        df = df.copy()
        df["hedge_target"] = prepared["target"].to_numpy()
        return df

    def on_bar(self, ctx: Context) -> None:
        g = float(ctx.bar["hedge_target"])
        if self._held is None:
            self._held = g
        x = self.w * g + (1.0 - self.w) * self._held
        if self._last_target is None or abs(x - self._last_target) > 1e-9:
            ctx.order_target(x)
            self._last_target = x
            self.n_retarget_calls += 1
        self._held = x


class RewardVariantCAPE(CAPEHedge):
    """CAPEHedge with the reward channel perturbed.

    ``mode``:
      ``"neg"``   -> r := -r  (reward channel deliberately inverted)
      ``"zero"``  -> r := 0   (reward channel removed; only lambda drift left)
      ``"lead"``  -> r := log_ret of bar i+1 (DELIBERATE 1-BAR LOOKAHEAD;
                     a sensitivity probe only, never a reported result)
    """

    name = "skeptic_reward_variant"

    def __init__(self, mode: str = "neg", **kw) -> None:
        super().__init__(**kw)
        self.mode = mode

    def on_bar(self, ctx: Context) -> None:  # mirrors CAPEHedge.on_bar exactly
        g = float(ctx.bar["hedge_target"])
        if self.mode == "zero":
            r = 0.0
        elif self.mode == "lead":
            arr = ctx._cols["log_ret"]
            j = min(ctx.i + 1, len(arr) - 1)
            rv = arr[j]
            r = float(rv) if np.isfinite(rv) else 0.0
        else:
            rv = ctx.bar["log_ret"]
            r = -float(rv) if np.isfinite(rv) else 0.0
        fee = ctx.market.fee_rate
        lev = ctx.market.leverage

        if self._w is None:
            self._w = 0.5
            self._A = self.a0
            self._held = g
            self._x_prev = g
            self._x_prevprev = g
            self._g_prev = g
            self._held_prev = g
        else:
            grad = ((self._g_prev - self._held_prev)
                    * (fee * lev * float(np.sign(self._x_prev - self._x_prevprev)) - r)
                    + self.lam)
            self._A += grad * grad
            step = self.eta / np.sqrt(self._A)
            self._w = float(np.clip(self._w - step * grad, 0.0, 1.0))

        self.w_history.append(self._w)
        x = self._w * g + (1.0 - self._w) * self._held
        if self._last_target is None or abs(x - self._last_target) > 1e-9:
            ctx.order_target(x)
            self._last_target = x
            self.n_retarget_calls += 1
        self._g_prev = g
        self._held_prev = self._held
        self._x_prevprev = self._x_prev
        self._x_prev = x
        self._held = x


def load_btc(kind="spot"):
    df, label = load_dataset(ROOT / "data", kind)
    train = df.loc[:INNER_VAL_END].copy()
    assert train.index[-1] < pd.Timestamp(OOS_START, tz=train.index[-1].tz)
    return train, label


def load_eth():
    eth = load_coinbase_eth_spot(ROOT / "data")
    eth = eth.loc[:INNER_VAL_END].copy()
    assert eth.index[-1] < pd.Timestamp(OOS_START, tz=eth.index[-1].tz)
    return eth


def dsharpe(res_a, res_b, label, seed=7):
    da = daily_returns(res_a.equity)
    db = daily_returns(res_b.equity)
    n = min(len(da), len(db))
    da, db = da.iloc[-n:].to_numpy(), db.iloc[-n:].to_numpy()
    pr = paired_bootstrap(da, db, annualized_sharpe, mean_block=30.0, seed=seed)
    print(f"  [{label} seed={seed}] a={pr.stat_a:+.3f} b={pr.stat_b:+.3f} "
          f"d={pr.diff.point:+.3f} CI=[{pr.diff.lo:+.3f},{pr.diff.hi:+.3f}] sig={pr.significant}")
    return pr


def describe(res, m, strat=None, tag=""):
    eq = res.equity.to_numpy(dtype=float)
    extra = ""
    if strat is not None and getattr(strat, "w_history", None):
        w = np.array(strat.w_history)
        extra = (f" w[min={w.min():.4f} p1={np.percentile(w,1):.4f} med={np.median(w):.4f} "
                 f"p99={np.percentile(w,99):.4f} max={w.max():.4f} mean={w.mean():.4f} std={w.std():.4f}]"
                 f" A_final={getattr(strat,'_A',float('nan')):.6f}"
                 f" max|dw|={np.abs(np.diff(w)).max():.2e} mean|dw|={np.abs(np.diff(w)).mean():.2e}")
    print(f"  {tag}: final={m.final_balance:.2f} sharpe(bar)={m.sharpe:.3f} trades={m.num_trades} "
          f"fees={m.fees_paid:.1f} liq={res.liquidated} min_eq={eq.min():.2f} "
          f"n_bars={len(eq)}{extra}")


def main():
    np.set_printoptions(suppress=True)
    df_btc, lab = load_btc("spot")
    print(f"BTC frame: {df_btc.index[0]} .. {df_btc.index[-1]}  n={len(df_btc)}")

    # ---------------------------------------------------------------- T0
    print("\n=== T0: is the truncation probe vacuous? (run_period slices to `end`) ===")
    full = df_btc
    trunc = df_btc.loc[:INNER_TRAIN_END]
    lo = int(full.index.searchsorted(INNER_TRAIN_START))
    hi_full = int(full.index.searchsorted(INNER_TRAIN_END, side="right"))
    hi_tr = int(trunc.index.searchsorted(INNER_TRAIN_END, side="right"))
    print(f"  full frame last={full.index[-1]}  trunc frame last={trunc.index[-1]}")
    print(f"  run_period hi(full)={hi_full}  hi(trunc)={hi_tr}  identical_slice="
          f"{full.iloc[lo-2500:hi_full].equals(trunc.iloc[lo-2500:hi_tr])}")
    print(f"  -> frames handed to run_backtest are identical: probe cannot fail for ANY strategy")

    # ---------------------------------------------------------------- T1
    print("\n=== T1: w=1 sanity (wrapper must be a no-op at w=1) ===")
    m_b, r_b = None, None
    sb = get_strategy("hedge_experts")
    r_b = run_period(sb, df_btc, INNER_VAL_START, INNER_VAL_END, market=SPOT,
                     start_balance=1000.0, data_label=lab)
    m_b = compute_metrics(r_b)
    describe(r_b, m_b, tag="baseline hedge_experts spot inner-val")
    s1 = FixedWHedge(w=1.0)
    r1 = run_period(s1, df_btc, INNER_VAL_START, INNER_VAL_END, market=SPOT,
                    start_balance=1000.0, data_label=lab)
    describe(r1, compute_metrics(r1), tag="FixedW(w=1.0)")

    # ---------------------------------------------------------------- T2
    cells = [
        ("BTC spot full-inner", df_btc, SPOT, INNER_TRAIN_START, INNER_VAL_END, lab),
        ("BTC spot inner-val", df_btc, SPOT, INNER_VAL_START, INNER_VAL_END, lab),
        ("BTC fut5x full-inner", df_btc, FUTURES, INNER_TRAIN_START, INNER_VAL_END, lab),
        ("BTC fut5x inner-val", df_btc, FUTURES, INNER_VAL_START, INNER_VAL_END, lab),
        ("BTC spot 0.40% inner-val", df_btc, SPOT_HIGH_FEE, INNER_VAL_START, INNER_VAL_END, lab),
        ("BTC fut5x 0.40% inner-val", df_btc, FUTURES_HIGH_FEE, INNER_VAL_START, INNER_VAL_END, lab),
    ]
    df_eth = load_eth()
    cells.append(("ETH spot inner-val", df_eth, SPOT, INNER_VAL_START, INNER_VAL_END, "eth_spot"))

    for name, dfx, mkt, s, e, lb in cells:
        print(f"\n=== {name} ===")
        sc = CAPEHedge(lam=LAMBDA_BASE)
        rc = run_period(sc, dfx, s, e, market=mkt, start_balance=1000.0, data_label=lb)
        mc = compute_metrics(rc)
        describe(rc, mc, sc, tag="CAPEHedge")
        sbase = get_strategy("hedge_experts")
        rbase = run_period(sbase, dfx, s, e, market=mkt, start_balance=1000.0, data_label=lb)
        mbase = compute_metrics(rbase)
        describe(rbase, mbase, tag="baseline")
        wbar = float(np.mean(sc.w_history))
        sf = FixedWHedge(w=wbar)
        rf = run_period(sf, dfx, s, e, market=mkt, start_balance=1000.0, data_label=lb)
        mf = compute_metrics(rf)
        describe(rf, mf, tag=f"FixedW(w={wbar:.4f}) ABLATION")
        variants = {}
        for mode in ("neg", "zero", "lead"):
            sv = RewardVariantCAPE(mode=mode, lam=LAMBDA_BASE)
            rv = run_period(sv, dfx, s, e, market=mkt, start_balance=1000.0, data_label=lb)
            describe(rv, compute_metrics(rv), sv, tag=f"reward={mode:5s} ABLATION")
            variants[mode] = rv

        dsharpe(rc, rbase, f"{name} CAPE-vs-base", seed=7)
        dsharpe(rc, rbase, f"{name} CAPE-vs-base", seed=20260825)
        dsharpe(rc, rbase, f"{name} CAPE-vs-base", seed=101)
        dsharpe(rf, rbase, f"{name} FIXEDW-vs-base", seed=7)
        for mode, rv in variants.items():
            dsharpe(rv, rbase, f"{name} reward={mode}-vs-base", seed=7)
        dsharpe(rc, rf, f"{name} CAPE-vs-FIXEDW", seed=7)

        # correlation of daily returns CAPE vs FixedW
        dc, dfw = daily_returns(rc.equity), daily_returns(rf.equity)
        n = min(len(dc), len(dfw))
        print(f"  corr(daily CAPE, daily FixedW) = "
              f"{np.corrcoef(dc.iloc[-n:], dfw.iloc[-n:])[0,1]:.6f}")


if __name__ == "__main__":
    main()
