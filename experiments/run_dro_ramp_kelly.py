#!/usr/bin/env python
"""Driver for the regime-age ramp experiment (docs/LEDGER.md has no ID yet).

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_dro_ramp_kelly.py inspect     # what the ramp does
    python experiments/run_dro_ramp_kelly.py sweep       # 3 triggers, inner splits
    python experiments/run_dro_ramp_kelly.py neighbours  # plateau check around winner
    python experiments/run_dro_ramp_kelly.py causality   # by-hand lookahead probe
    python experiments/run_dro_ramp_kelly.py holdout     # step 4, frozen config
    python experiments/run_dro_ramp_kelly.py eth         # falsification test
    python experiments/run_dro_ramp_kelly.py costs       # fee tier + funding
    python experiments/run_dro_ramp_kelly.py windows     # 40-window path sensitivity
    python experiments/run_dro_ramp_kelly.py deflated    # deflated Sharpe, this session's trials
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.dro_ramp_kelly import BARS_PER_YEAR, RampedKellyRegime  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

N_EVALUATED = 0  # every distinct configuration this file evaluates, for deflated Sharpe


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0, count=True):
    """One backtest, one line, counted (unless it is only a benchmark)."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# --------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the ramp actually does, before any backtest."""
    s = RampedKellyRegime(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger="any")
    prepared = s.prepare(DF.copy())
    ramp = prepared["ramp"]
    print(f"ramp quantiles (any-trigger, floor=0.5, tau=20d): "
          + "  ".join(f"q{int(q*100)}={ramp.quantile(q):.2f}"
                      for q in (0.05, 0.25, 0.5, 0.75, 0.95)))
    print(f"fraction of bars fully relaxed (ramp>=0.99): {(ramp >= 0.99).mean():.1%}")
    print(f"fraction of bars at/near the floor (ramp<=0.55): {(ramp <= 0.55).mean():.1%}")

    print("\nby trigger, mean ramp and correlation with realized volatility:")
    r = np.log(DF["close"]).diff()
    vol = (r.ewm(span=8 * 288, min_periods=288).std() * np.sqrt(BARS_PER_YEAR)).shift(1)
    for trig in ("any", "net", "down_only"):
        p = RampedKellyRegime(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger=trig).prepare(DF.copy())
        both = pd.DataFrame({"ramp": p["ramp"], "vol": vol}).dropna()
        print(f"  {trig:10s} mean ramp={p['ramp'].mean():.3f}  "
              f"corr(ramp, realized vol)={both['ramp'].corr(both['vol']):+.3f}")

    print("\nmean exposure vs kelly_regime_v4 (target column), by trigger:")
    v4 = get_strategy("kelly_regime_v4")
    v4p = v4.prepare(DF.copy())
    for trig in ("any", "net", "down_only"):
        p = RampedKellyRegime(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger=trig).prepare(DF.copy())
        print(f"  {trig:10s} mean|target|={p['target'].abs().mean():.3f}  "
              f"vs v4's {v4p['target'].abs().mean():.3f}  "
              f"ratio={p['target'].abs().mean() / v4p['target'].abs().mean():.2f}x")

    print("\nyearly mean |target|, ramp vs v4 (does the ramp bite disproportionately in the bull years?):")
    for trig in ("down_only", "any"):
        p = RampedKellyRegime(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger=trig).prepare(DF.copy())
        by_year_ramp = p["target"].abs().groupby(p.index.year).mean()
        by_year_v4 = v4p["target"].abs().groupby(v4p.index.year).mean()
        print(f"  trigger={trig}:")
        for year in by_year_ramp.index:
            if year in by_year_v4.index:
                print(f"    {year}  ramp={by_year_ramp[year]:.3f}  v4={by_year_v4[year]:.3f}  "
                      f"ratio={by_year_ramp[year] / max(by_year_v4[year], 1e-9):.2f}")


# ----------------------------------------------------------------------- sweep


def _trigger_variants(floor: float = 0.5, tau: float = 20.0):
    return [(f"{trig} floor={floor} tau={tau:g}d",
             dict(ramp_floor=floor, ramp_tau_days=tau, ramp_trigger=trig))
            for trig in ("any", "net", "down_only")]


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}", count=False)


def sweep() -> None:
    """Step 3, part 1: the three trigger variants at their a-priori defaults."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} ramp variants:")
            for tag, kw in _trigger_variants():
                ev(RampedKellyRegime(**kw), start, end, market=market, tag=tag)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


def neighbours(winner_trigger: str = "down_only", winner_floor: float = 0.5) -> None:
    """Step 3, part 2: plateau check around the selected trigger."""
    grid = [(f"floor={f}", dict(ramp_floor=f, ramp_tau_days=20.0, ramp_trigger=winner_trigger))
            for f in (0.3, 0.7, 0.9)]
    grid += [(f"tau={t:g}d", dict(ramp_floor=winner_floor, ramp_tau_days=t, ramp_trigger=winner_trigger))
             for t in (10.0, 40.0, 80.0)]
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nINNER-VALIDATION neighbourhood ({winner_trigger}) / {mname}:")
        for tag, kw in grid:
            ev(RampedKellyRegime(**kw), *VALID, market=market, tag=tag)
        print(f"INNER-TRAIN neighbourhood ({winner_trigger}) / {mname}:")
        for tag, kw in grid:
            ev(RampedKellyRegime(**kw), *TRAIN, market=market, tag=tag, count=False)
    print(f"\nconfigurations evaluated so far: {N_EVALUATED}")


def causality() -> None:
    """The strict on_bar peek check, run by hand (unregistered strategy).

    Same two-opposite-tampers procedure as R-28/R-31: bars after a cut are
    multiplied by 3 in one copy and divided by 3 in the other. Every order
    at or before the cut must be identical, and the prepared columns must
    match bit-for-bit before the cut too (catches a full-series statistic
    leaking into early rows, which a pure order-level truncation test can
    miss).
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    kw = dict(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger="down_only")

    def decisions(frame):
        s = RampedKellyRegime(**kw)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    a, b = decisions(up), decisions(down)
    bad = [bar for bar, oa, ob in zip(bars, a, b) if oa != ob]
    print(f"tampered from bar {cut:,} of {len(df):,}; checked bars {bars}")
    print("FAIL - reads the future at bars " + str(bad) if bad
          else "PASS - every decision at or before the cut is unchanged")

    pa = RampedKellyRegime(**kw).prepare(up.copy())
    pb = RampedKellyRegime(**kw).prepare(down.copy())
    for col in ("target", "ramp", "raw_frac"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        print(f"  column {col:10s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")

    # Also check the other two triggers, since they build the ramp differently.
    for trig in ("any", "net"):
        kw2 = dict(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger=trig)
        pa2 = RampedKellyRegime(**kw2).prepare(up.copy())
        pb2 = RampedKellyRegime(**kw2).prepare(down.copy())
        worst = float(np.nanmax(np.abs(pa2["target"].to_numpy()[:cut]
                                        - pb2["target"].to_numpy()[:cut])))
        print(f"  trigger={trig:10s} target max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


# --------------------------------------------------------------------- holdout

# Frozen before the holdout was read (see the ledger row's pre-registration
# text for the exact commit-ordering claim). Trigger and floor/tau selected
# on inner-validation only (see the report accompanying this file).
FROZEN = dict(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger="down_only")
ALSO = [("any-trigger",  dict(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger="any")),
        ("net-trigger",  dict(ramp_floor=0.5, ramp_tau_days=20.0, ramp_trigger="net"))]


def holdout() -> None:
    """Step 4. The configuration is frozen; the decision rule is in the report."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"  {name}", count=False)
        ev(RampedKellyRegime(**FROZEN), *OOS, market=market,
           tag="  down_only ramp (FROZEN)", count=False)
        for tag, kw in ALSO:
            ev(RampedKellyRegime(**kw), *OOS, market=market, tag=f"  {tag}", count=False)


def eth() -> None:
    """Pre-registered falsification: does the mechanism survive on ETH?

    Same venue (Bitfinex), same window as R-17/R-28's falsification test.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for market in (SPOT, FUTURES):
            for name in ("buy_and_hold", "kelly_regime_v4"):
                ev(get_strategy(name), None, None, df=df, market=market,
                   tag=f"  {name}", count=False)
            ev(RampedKellyRegime(**FROZEN), None, None, df=df, market=market,
               tag="  down_only ramp (frozen)", count=False)


def costs() -> None:
    """Step-4 cost checks: the real fee tier, and funding on the futures side."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ at Bitstamp's 0.40% entry taker tier (spot):")
    for tier, label in ((0.001, "0.10% (table assumption)"), (0.004, "0.40% (entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"    {name}", count=False)
        ev(RampedKellyRegime(**FROZEN), *OOS, market=market,
           tag="    down_only ramp", count=False)

    real = load_funding(ROOT / "data")
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr assumed after):")
    for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                        ("down_only ramp", RampedKellyRegime(**FROZEN))):
        lo = int(DF.index.searchsorted(OOS[0]))
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=blended, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:24s} final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>7,.0f}")


def windows(trials: int = 40, seed: int = 42) -> None:
    """Path sensitivity: same design as R-19/R-28/scripts/stress_test.py."""
    from tradebot.metrics import max_drawdown_pct

    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("down_only ramp", RampedKellyRegime(**FROZEN))]
    warmup = max(s.warmup for _, s in contenders) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(90, 731) * 288)
        specs.append((int(rng.integers(warmup, len(DF) - length)), length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for mname, market in (("spot", SPOT), ("futures", FUTURES)):
            for name, strat in contenders:
                res = run_backtest(strat, window, market, 1_000.0,
                                   trade_start=warmup, data_label=LABEL)
                eq = res.equity.to_numpy(dtype=float)
                base, seg = eq[warmup], eq[warmup:]
                ok = np.isfinite(base) and base > 0
                rows.append({"trial": k, "market": mname, "strategy": name,
                             "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                             "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                             "liquidated": res.liquidated})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    res = pd.DataFrame(rows)

    print(f"\n{trials} random windows (90-730 days), identical across strategies:\n")
    for mname in ("spot", "futures"):
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name in ("buy_and_hold", "kelly_regime_v4", "down_only ramp"):
            g = sub[sub.strategy == name].set_index("trial")
            beat = (g["return_pct"] > bench).mean()
            print(f"    {name:16s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"P(DD>50%) {(g.max_dd_pct > 50).mean():>5.0%}  "
                  f"beat hold {beat:>5.0%}  liq {g.liquidated.mean():>4.0%}")
        a = sub[sub.strategy == "down_only ramp"].set_index("trial")["max_dd_pct"]
        b = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["max_dd_pct"]
        d = (a - b).dropna()
        ra = sub[sub.strategy == "down_only ramp"].set_index("trial")["return_pct"]
        rb = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["return_pct"]
        dr = (ra - rb).dropna()
        print(f"    paired DD difference (ramp - v4): median {d.median():+.1f}pp, "
              f"deeper in {(d > 0).mean():.0%} of windows")
        print(f"    paired return difference (ramp - v4): median {dr.median():+.1f}pp, "
              f"higher in {(dr > 0).mean():.0%} of windows\n")


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation; good to ~1e-9, no SciPy dependency."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = (-2 * np.log(1 - p)) ** 0.5
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated() -> None:
    """Deflated Sharpe (Bailey & Lopez de Prado 2014) on this session's trials.

    Same narrow method as R-28/eprocess_regime.deflated(): one strategy,
    this session's trial count, not the systematic B-04/R-29 treatment.
    """
    from math import e

    gamma = 0.5772156649015329

    trials = [kw for _, kw in _trigger_variants()]
    for f in (0.3, 0.7, 0.9):
        trials.append(dict(ramp_floor=f, ramp_tau_days=20.0, ramp_trigger="down_only"))
    for t in (10.0, 40.0, 80.0):
        trials.append(dict(ramp_floor=0.5, ramp_tau_days=t, ramp_trigger="down_only"))

    sharpes = []
    for kw in trials:
        result = run_period(RampedKellyRegime(**kw), DF, *VALID, market=SPOT,
                            start_balance=1_000.0, data_label=LABEL)
        sharpes.append(compute_metrics(result).sharpe)
    sharpes = np.array(sharpes)
    n_trials = len(sharpes)
    var_trials = float(sharpes.var(ddof=1))

    result = run_period(RampedKellyRegime(**FROZEN), DF, *OOS, market=SPOT,
                        start_balance=1_000.0, data_label=LABEL)
    eq = result.equity.to_numpy(dtype=float)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    n = len(rets)
    sr = float(rets.mean() / rets.std(ddof=1))
    z = (rets - rets.mean()) / rets.std(ddof=1)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())

    scale = np.sqrt(BARS_PER_YEAR)
    sr_star = np.sqrt(var_trials / BARS_PER_YEAR) * (
        (1 - gamma) * _norm_ppf(1 - 1.0 / n_trials)
        + gamma * _norm_ppf(1 - 1.0 / (n_trials * e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2)
    dsr = _norm_cdf((sr - sr_star) * np.sqrt(n - 1) / denom)

    print(f"trials searched this session (distinct configurations): {n_trials}")
    print(f"  spread of trial Sharpes on inner-validation: sd={np.sqrt(var_trials):.3f} "
          f"(annualized), range {sharpes.min():.2f}..{sharpes.max():.2f}")
    print("\nholdout, spot, frozen config:")
    print(f"  observed Sharpe          {sr * scale:>6.2f} (annualized)")
    print(f"  skew {skew:+.2f}  kurtosis {kurt:.1f}  n={n:,} bars")
    print(f"  SR* (expected best of {n_trials} by luck) {sr_star * scale:>6.2f}")
    print(f"  deflated Sharpe (P(SR > SR*))            {dsr:>6.3f}")
    print("\nProgram-level caveat: the trials count that matters is the total across\n"
          "the whole project (~172 before this session), and the holdout has been\n"
          "read well over a hundred times. Read this as a lower bound on the\n"
          "deflation, not as a clean significance test.")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"inspect": inspect, "sweep": sweep, "neighbours": neighbours,
            "causality": causality, "holdout": holdout, "eth": eth,
            "costs": costs, "windows": windows, "deflated": deflated}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_dro_ramp_kelly.py [{'|'.join(cmds)}]")
