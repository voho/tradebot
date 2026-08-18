"""B-05 pre-registered holdout evaluation. Single read, per
experiments/funding_gate_preregistration.md's decision rule.

Frozen configs (selected on funding-inner-validation 2022 ONLY, before
this file ever ran):
  Variant A (conservative): enter_pct=0.85, exit_pct=0.60
  Variant B (novel):        k=1.5, funding_span_days=1.0

Evaluated once, here, on funding-holdout 2023-01-01..2023-12-31,
futures 5x, funding charged, vs kelly_regime_v4 funding-charged.
"""
import sys, numpy as np, pandas as pd
from dataclasses import replace
sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from tradebot.broker import MarketSpec
from tradebot.data import load_dataset, load_funding
from tradebot.engine import run_backtest
from tradebot.metrics import compute_metrics
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
from tradebot.inference import (daily_returns, paired_bootstrap, total_log_return,
                                 max_drawdown_from_returns, annualized_sharpe)
from funding_gate_novel import FundingAdjustedKellyV4
from funding_gate_conservative import FundingGateConservative

DF, LABEL = load_dataset("data", "spot")
REAL = load_funding("data")
FUTURES = MarketSpec.futures(leverage=5.0)
START, END = "2023-01-01", "2023-12-31"
assert REAL.index[-1] >= pd.Timestamp(END, tz="UTC"), "funding data must cover the full holdout year"

class ScaledBaseline(KellyRegimeV4):
    name = "scaled_baseline"
    def __init__(self, scale=1.0, **kw):
        super().__init__(**kw); self.scale = scale
    def prepare(self, df):
        df = super().prepare(df); df["target"] = df["target"] * self.scale; return df

def period(strategy):
    lo = int(DF.index.searchsorted(START)); hi = int(DF.index.searchsorted(END, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo-pre:hi], FUTURES, 1000.0, trade_start=pre, funding=REAL, data_label=LABEL)
    return raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])

def daily_vol(eq):
    r = daily_returns(eq)
    return r.std() * np.sqrt(365.25), r

def summarize(name, raw):
    m = compute_metrics(raw)
    vol, rets = daily_vol(raw.equity)
    print(f"{name:30s} final=${m.final_balance:>10,.0f}  logG={np.log(max(m.final_balance,1e-9)/1000):+.3f}  "
          f"DD={m.max_drawdown_pct:5.1f}%  Sharpe={m.sharpe:5.2f}  trades={m.num_trades:4d}  "
          f"funding=${raw.funding_paid:>9,.0f}  realizedVol={vol:.3f}")
    return rets

print(f"HOLDOUT {START} .. {END}, futures 5x, funding CHARGED (single pre-registered read)\n")

base_raw = period(KellyRegimeV4())
base_rets = summarize("kelly_regime_v4 (baseline)", base_raw)

varA_raw = period(FundingGateConservative(funding=REAL, enter_pct=0.85, exit_pct=0.60))
varA_rets = summarize("Variant A (0.85/0.60 gate)", varA_raw)

varB_raw = period(FundingAdjustedKellyV4(funding=REAL, k=1.5, funding_span_days=1.0))
varB_rets = summarize("Variant B (k=1.5, 1d EWM)", varB_raw)

# matched-risk scaled baseline, for each variant's realized vol
for tag, rets in (("A", varA_rets), ("B", varB_rets)):
    target_vol = rets.std() * np.sqrt(365.25)
    lo_c, hi_c = 0.01, 1.5
    for _ in range(20):
        mid = (lo_c + hi_c) / 2
        v, _ = daily_vol(period(ScaledBaseline(scale=mid)).equity)
        if v > target_vol: hi_c = mid
        else: lo_c = mid
    c = (lo_c + hi_c) / 2
    mr_raw = period(ScaledBaseline(scale=c))
    summarize(f"  matched-risk baseline (c={c:.3f}, for {tag})", mr_raw)

print("\nPaired block-bootstrap (30-day block, 2000 resamples), vs kelly_regime_v4 baseline:\n")
for tag, rets in (("Variant A", varA_rets), ("Variant B", varB_rets)):
    n = min(len(rets), len(base_rets))
    a, b = rets.to_numpy()[-n:], base_rets.to_numpy()[-n:]
    pg = paired_bootstrap(a, b, total_log_return, mean_block=30.0, n_boot=2000, seed=7)
    pdd = paired_bootstrap(a, b, max_drawdown_from_returns, mean_block=30.0, n_boot=2000, seed=7)
    # note: diff is stat(a)-stat(b); for drawdown, a NEGATIVE diff means variant's DD is smaller (better)
    print(f"{tag}: n={n} obs")
    print(f"  log growth  diff = {pg.diff.point:+.3f} [{pg.diff.lo:+.3f}, {pg.diff.hi:+.3f}]  "
          f"significant={pg.significant}  P(diff>0)={pg.p_positive:.2f}")
    print(f"  max DD (pp) diff = {pdd.diff.point:+.2f} [{pdd.diff.lo:+.2f}, {pdd.diff.hi:+.2f}]  "
          f"significant={pdd.significant}  (negative = variant's DD is smaller/better)")
    print()
