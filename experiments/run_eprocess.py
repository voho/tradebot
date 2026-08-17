#!/usr/bin/env python
"""Driver for backlog B-01 — e-process regime detection with Kelly sizing.

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_eprocess.py inspect     # what the e-process does
    python experiments/run_eprocess.py sweep       # inner-train + inner-val
    python experiments/run_eprocess.py neighbours  # plateau check
    python experiments/run_eprocess.py holdout     # step 4, frozen config
    python experiments/run_eprocess.py eth         # falsification test
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataclasses import replace  # noqa: E402

from experiments.eprocess_regime import BARS_PER_YEAR, EProcessRegime  # noqa: E402
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

N_EVALUATED = 0  # every configuration this file evaluates, for deflated Sharpe


def ev(strategy, start, end, df=None, market=SPOT, tag="", balance=1_000.0,
       count=True):
    """One backtest, one line, counted."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    # fills, not `trades`: a strategy whose exposure never returns to zero
    # books one open round-trip while rebalancing (and paying) many times.
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# --------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the e-process actually computes, before any backtest."""
    s = EProcessRegime(**FROZEN)
    prepared = s.prepare(DF.copy())
    thr = np.log(1.0 / s.alpha)

    sharpe_est = prepared["lam"] * np.sqrt(BARS_PER_YEAR)
    ev_ = prepared["evidence"]
    print(f"threshold log(1/alpha) = {thr:.2f}   (alpha={s.alpha})")
    print("\nendogenous target vol = kelly_fraction x annualized Sharpe estimate:")
    print(f"  lam*sqrt(BPY) quantiles: "
          + "  ".join(f"q{int(q*100)}={sharpe_est.quantile(q):.2f}"
                      for q in (0.05, 0.25, 0.5, 0.75, 0.95)))
    print(f"  half-Kelly target vol at the median: "
          f"{0.5 * sharpe_est.median():.2f}  (repo's hand-set value: 0.55)")
    print("\nevidence process:")
    print(f"  fraction of bars at the cap (gate fully open): "
          f"{(ev_ >= thr - 1e-9).mean():.1%}")
    print(f"  fraction of bars at zero (gate shut):          "
          f"{(ev_ <= 1e-9).mean():.1%}")

    print("\nyearly mean gate (fraction of full exposure the evidence allows):")
    gate = (ev_ / thr).clip(0, 1)
    per_year = gate.groupby(prepared.index.year).mean()
    ret = DF["close"].groupby(DF.index.year).last() / \
        DF["close"].groupby(DF.index.year).first() - 1.0
    for year, g in per_year.items():
        print(f"  {year}  gate={g:.2f}   BTC year return {ret.get(year, float('nan')):>+7.1%}")

    # Pre-registered failure mode (a): is the evidence process just a
    # smoothed momentum indicator wearing a martingale costume? Compare it
    # with the incumbent's latched anchor vote on the same bars.
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
    v4 = KellyRegimeV4()
    votes = []
    for days in v4.horizons:
        anchor = DF["close"].rolling(int(days * 288)).mean()
        v = pd.Series(np.where(DF["close"] > anchor * (1 + v4.band), 1.0,
                               np.where(DF["close"] < anchor * (1 - v4.band),
                                        0.0, np.nan)), index=DF.index)
        votes.append(v.ffill().fillna(0.0))
    vote = sum(votes) / len(votes)
    both = pd.DataFrame({"gate": gate, "vote": vote}).dropna()
    print("\ngate vs the incumbent's latched anchor vote:")
    print(f"  correlation                       {both['gate'].corr(both['vote']):.3f}")
    print(f"  bars where both are >0            {((both.gate > 0) & (both.vote > 0)).mean():.1%}")
    print(f"  evidence open while vote shut     {((both.gate > 0) & (both.vote == 0)).mean():.1%}")
    print(f"  vote open while evidence shut     {((both.gate == 0) & (both.vote > 0)).mean():.1%}")
    print(f"  mean exposure gate {both.gate.mean():.3f} vs vote {both.vote.mean():.3f}"
          f"  -> the e-process holds {both.gate.mean() / both.vote.mean():.2f}x the incumbent")


# ----------------------------------------------------------------------- sweep


def _variants():
    """The three variants, on a small grid. Every entry is one trial."""
    out = []
    for hl in (20.0, 60.0, 180.0):
        out.append((f"E1 gate hl={hl:g}d",
                    dict(bet_halflife_days=hl, gate=True, sizing="fixed")))
    for hl in (20.0, 60.0, 180.0):
        out.append((f"E2 kelly hl={hl:g}d",
                    dict(bet_halflife_days=hl, gate=False, sizing="kelly")))
    for hl in (20.0, 60.0, 180.0):
        out.append((f"E3 both  hl={hl:g}d",
                    dict(bet_halflife_days=hl, gate=True, sizing="kelly")))
    return out


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}",
           count=False)


def sweep() -> None:
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} variants:")
            for tag, kw in _variants():
                ev(EProcessRegime(**kw), start, end, market=market, tag=tag)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def neighbours() -> None:
    """Plateau, not peak: vary one knob at a time around the selection."""
    base = dict(bet_halflife_days=20.0, gate=True, sizing="fixed")
    grid = [("base E1 hl=20d", {})]
    grid += [(f"hl={h:g}d", dict(bet_halflife_days=h))
             for h in (10.0, 15.0, 30.0, 40.0, 90.0)]
    grid += [(f"alpha={a}", dict(alpha=a)) for a in (0.01, 0.20)]
    grid += [(f"clip={c:g}", dict(clip=c)) for c in (3.0, 8.0)]
    grid += [(f"cap_mult={m:g}", dict(evidence_cap_mult=m)) for m in (0.5, 2.0)]
    grid += [(f"deadband={d:g}", dict(deadband=d)) for d in (0.05, 0.20)]
    grid += [("decay hl=180d", dict(evidence_halflife_days=180.0))]
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nINNER-VALIDATION neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(EProcessRegime(**{**base, **kw}), *VALID, market=market, tag=tag)
        print(f"INNER-TRAIN neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(EProcessRegime(**{**base, **kw}), *TRAIN, market=market, tag=tag,
               count=False)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def causality() -> None:
    """The strict on_bar peek check, run by hand for an unregistered strategy.

    ``tests/test_causality_strict.py`` only parametrizes over *registered*
    strategies, so an experiment gets none of that protection. This is the
    same two-opposite-tampers procedure: bars after a cut are multiplied by
    3 in one copy and divided by 3 in the other, and every decision at or
    before the cut must be identical.
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

    def decisions(frame):
        s = EProcessRegime(**FROZEN)
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

    # A prepared column computed over the whole series would also move at
    # rows before the cut even when no order changes; check the target and
    # evidence columns directly, which is what the skeptic actually wants.
    pa = EProcessRegime(**FROZEN).prepare(up.copy())
    pb = EProcessRegime(**FROZEN).prepare(down.copy())
    for col in ("target", "evidence", "lam"):
        diff = np.abs(pa[col].to_numpy()[:cut] - pb[col].to_numpy()[:cut])
        worst = float(np.nanmax(diff))
        print(f"  column {col:9s} max |difference| before the cut = {worst:.3e}"
              f"  {'PASS' if worst < 1e-12 else 'FAIL'}")


# --------------------------------------------------------------------- holdout

# Frozen before the holdout was read. E1 (evidence gate, incumbent sizer) is
# the variant that isolates the e-process claim; half-life 20d was selected on
# inner-validation and coincides with the 18-28 day anchor region R-07 found
# robust, so it is not purely a validation fit. Every other knob sits at its
# a-priori default: alpha=0.05 is the conventional level, cap_mult=1.0 makes
# full exposure mean "evidence has reached the alpha threshold", clip=5 and
# deadband=0.10 are inherited unchanged from the incumbent.
FROZEN = dict(bet_halflife_days=20.0, gate=True, sizing="fixed")
# Reported too, because a parallel branch that stays silent is selection by
# the operator (ROUTINE.md, "Running directions in parallel").
ALSO = [("E2 kelly (unified sizer)", dict(bet_halflife_days=60.0, gate=False,
                                          sizing="kelly")),
        ("E3 both", dict(bet_halflife_days=60.0, gate=True, sizing="kelly"))]


def holdout() -> None:
    """Step 4. The configuration is frozen; the decision rule is in the ledger."""
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        for name in ("buy_and_hold", "kelly_regime", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"  {name}", count=False)
        ev(EProcessRegime(**FROZEN), *OOS, market=market,
           tag="E1 eprocess_regime (FROZEN)", count=False)
        for tag, kw in ALSO:
            ev(EProcessRegime(**kw), *OOS, market=market, tag=tag, count=False)


def eth() -> None:
    """Pre-registered falsification: does the mechanism survive on ETH?

    Same venue (Bitfinex), same window, only the asset varies - the design
    of R-17 / docs/CROSS_ASSET.md, so the incumbent's numbers there are
    the comparison.
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
            ev(EProcessRegime(**FROZEN), None, None, df=df, market=market,
               tag="  eprocess_regime (frozen)", count=False)


def costs() -> None:
    """The step-4 cost checks: the real fee tier, and funding on the futures side."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ at Bitstamp's 0.40% entry taker tier (spot):")
    for tier, label in ((0.001, "0.10% (table assumption)"), (0.004, "0.40% (entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"    {name}", count=False)
        ev(EProcessRegime(**FROZEN), *OOS, market=market,
           tag="    E1 eprocess_regime", count=False)

    real = load_funding(ROOT / "data")
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr assumed after):")
    for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                        ("E1 eprocess_regime", EProcessRegime(**FROZEN))):
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
    """Path sensitivity: is the drawdown reduction a property or one lucky path?

    Same design as ``scripts/stress_test.py`` - random start, random
    length, identical windows for every strategy, warmup prefix that
    cannot trade - so the numbers are comparable with R-19.
    """
    from tradebot.metrics import max_drawdown_pct

    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("E1 eprocess", EProcessRegime(**FROZEN))]
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
        for name in ("buy_and_hold", "kelly_regime_v4", "E1 eprocess"):
            g = sub[sub.strategy == name].set_index("trial")
            beat = (g["return_pct"] > bench).mean()
            print(f"    {name:16s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"P(DD>50%) {(g.max_dd_pct > 50).mean():>5.0%}  "
                  f"beat hold {beat:>5.0%}  liq {g.liquidated.mean():>4.0%}")
        # Paired, per window: the drawdown claim is a difference, not a level.
        a = sub[sub.strategy == "E1 eprocess"].set_index("trial")["max_dd_pct"]
        b = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")["max_dd_pct"]
        d = (a - b).dropna()
        print(f"    paired DD difference (eprocess - v4): median {d.median():+.1f}pp, "
              f"deeper in {(d > 0).mean():.0%} of windows\n")


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
    """Deflated Sharpe (Bailey & Lopez de Prado 2014) on the holdout Sharpe.

    R-25 has been open since the ledger was written: deflated Sharpe is
    cited in RESEARCH.md and computed nowhere. This is the narrow version
    - one strategy, this session's trial count - not the systematic
    treatment B-04 still needs, but it is the first time the number exists.

    SR* is the Sharpe the *best of N trials* would be expected to reach by
    luck alone when the true Sharpe is zero; the deflated Sharpe is the
    probability that the observed Sharpe beats that.
    """
    from math import e

    gamma = 0.5772156649015329  # Euler-Mascheroni

    # The trial set: every configuration this session searched over, scored
    # on the split the selection was made on. Re-run, not re-counted.
    trials = [kw for _, kw in _variants()]
    base = dict(bet_halflife_days=20.0, gate=True, sizing="fixed")
    for extra in ({}, dict(bet_halflife_days=10.0), dict(bet_halflife_days=15.0),
                  dict(bet_halflife_days=30.0), dict(bet_halflife_days=40.0),
                  dict(bet_halflife_days=90.0), dict(alpha=0.01), dict(alpha=0.20),
                  dict(clip=3.0), dict(clip=8.0), dict(evidence_cap_mult=0.5),
                  dict(evidence_cap_mult=2.0), dict(deadband=0.05),
                  dict(deadband=0.20), dict(evidence_halflife_days=180.0)):
        trials.append({**base, **extra})

    sharpes = []
    for kw in trials:
        result = run_period(EProcessRegime(**kw), DF, *VALID, market=SPOT,
                            start_balance=1_000.0, data_label=LABEL)
        sharpes.append(compute_metrics(result).sharpe)
    sharpes = np.array(sharpes)
    n_trials = len(sharpes)
    var_trials = float(sharpes.var(ddof=1))

    # The observed side, in per-bar units.
    result = run_period(EProcessRegime(**FROZEN), DF, *OOS, market=SPOT,
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
    print(f"\nholdout, spot, frozen config:")
    print(f"  observed Sharpe          {sr * scale:>6.2f} (annualized)")
    print(f"  skew {skew:+.2f}  kurtosis {kurt:.1f}  n={n:,} bars")
    print(f"  SR* (expected best of {n_trials} by luck) {sr_star * scale:>6.2f}")
    print(f"  deflated Sharpe (P(SR > SR*))            {dsr:>6.3f}")
    print("\nProgram-level caveat: the trials count that matters is the total\n"
          "across the whole project, not this session's, and the holdout has\n"
          "been read dozens of times. Read this as a lower bound on the\n"
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
        print(f"usage: python experiments/run_eprocess.py [{'|'.join(cmds)}]")
