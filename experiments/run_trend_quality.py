#!/usr/bin/env python
"""Driver for the trend-quality conviction experiment (Baltas & Kosowski 2013).

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_trend_quality.py inspect     # what the t-stat looks like
    python experiments/run_trend_quality.py sweep       # step 3 grid (counted)
    python experiments/run_trend_quality.py neighbours  # plateau check (counted)
    python experiments/run_trend_quality.py causality   # by-hand lookahead probe
    python experiments/run_trend_quality.py holdout     # step 4, frozen
    python experiments/run_trend_quality.py interval    # paired bootstrap vs v4
    python experiments/run_trend_quality.py eth         # falsification test
    python experiments/run_trend_quality.py costs       # 0.40% fee + funding
    python experiments/run_trend_quality.py windows     # 40-window path check
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

from experiments.trend_quality import TrendQualityKelly, rolling_trend_tstat  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
OOS = ("2023-01-01", None)

N_EVALUATED = 0  # every distinct configuration this file scores, for deflated Sharpe


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
    print(f"  {tag or strategy.name:34s} {market.name:9s} "
          f"final=${m.final_balance:>11,.0f} ({m.profit_pct:>+8.1f}%) "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}{'  LIQUIDATED' if m.liquidated else ''}")
    return m


# --------------------------------------------------------------------- inspect


def inspect() -> None:
    """What the trend-quality t-statistic looks like, before any backtest."""
    t20 = rolling_trend_tstat(DF["close"], 20)
    t80 = rolling_trend_tstat(DF["close"], 80)
    print("20-day OLS trend t-statistic quantiles:")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"  q{int(q*100)}={np.nanquantile(t20, q):>7.2f}")
    print("80-day OLS trend t-statistic quantiles:")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"  q{int(q*100)}={np.nanquantile(t80, q):>7.2f}")

    # Pre-registered failure mode (a): is this just a smoothed relabelling of
    # the latched vote? Compare on the same bars.
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4
    v4 = KellyRegimeV4()
    votes = []
    for days in v4.horizons:
        anchor = DF["close"].rolling(int(days * 288)).mean()
        v = pd.Series(np.where(DF["close"] > anchor * (1 + v4.band), 1.0,
                               np.where(DF["close"] < anchor * (1 - v4.band),
                                        0.0, np.nan)), index=DF.index)
        votes.append(v.ffill().fillna(0.0))
    vote = (sum(votes) / len(votes)).to_numpy()
    conv80 = np.clip(np.abs(t80) / 4.0, 0.0, 1.0)
    mask = np.isfinite(t80)
    corr = np.corrcoef(vote[mask], conv80[mask])[0, 1]
    print(f"\ncorrelation, latched vote vs 80d trend-quality conviction (t_clip=4): {corr:.3f}")
    print(f"fraction of bars where vote=1 but conviction<0.3 (flipped, but on chop): "
          f"{((vote == 1.0) & (conv80 < 0.3) & mask).mean():.1%}")
    print(f"fraction of bars where vote=0 but conviction<0.3 (flipped, but on chop): "
          f"{((vote == 0.0) & (conv80 < 0.3) & mask).mean():.1%}")


# ----------------------------------------------------------------------- sweep


def _variant_sweep():
    """Step 3 grid: mode x knob. Every entry here is one trial."""
    out = []
    for tc in (2.0, 3.0, 4.0, 6.0, 8.0):
        out.append((f"A overlay t_clip={tc:g}", dict(mode="overlay", t_clip=tc)))
    for tc in (2.0, 3.0, 4.0, 6.0, 8.0):
        out.append((f"B continuous t_clip={tc:g}", dict(mode="continuous_vote", t_clip=tc)))
    for w in (20.0, 40.0, 60.0, 80.0, 120.0):
        out.append((f"C single_window w={w:g}d t_clip=4",
                    dict(mode="single_window", window_days=w, t_clip=4.0)))
    for tc in (2.0, 3.0, 6.0, 8.0):
        out.append((f"C single_window w=60d t_clip={tc:g}",
                    dict(mode="single_window", window_days=60.0, t_clip=tc)))
    return out


def _benchmarks(start, end, market, label):
    print(f"\n{label} benchmarks:")
    for name in ("buy_and_hold", "kelly_regime_v4"):
        ev(get_strategy(name), start, end, market=market, tag=f"  {name}", count=False)


def sweep() -> None:
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        for (start, end), split in ((TRAIN, "INNER-TRAIN"), (VALID, "INNER-VALIDATION")):
            _benchmarks(start, end, market, f"{split} / {mname}")
            print(f"{split} / {mname} variants:")
            for tag, kw in _variant_sweep():
                ev(TrendQualityKelly(**kw), start, end, market=market, tag=tag)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


def neighbours() -> None:
    """Plateau, not peak: t_clip neighbourhood around t_clip=10 (mode=overlay),
    plus sizer/deadband robustness, exactly the grid that selected t_clip=10
    on inner-validation and confirmed it against inner-train.

    t_clip=2..8 is already scored by ``sweep()`` above (mode A rows); this
    function is the follow-up grid that extended the search once t_clip=8
    turned out to beat t_clip=6 on BOTH axes on inner-validation (a
    non-monotone step that is the signature either of a real plateau
    starting around t_clip~9-10, or of a lucky read on a 2-year sample —
    which is exactly what checking it against the 4-year inner-train
    disambiguates).
    """
    grid = [(f"t_clip={tc:g}", dict(mode="overlay", t_clip=tc))
             for tc in (5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0)]
    grid += [("t_clip=10 sizer=plain", dict(mode="overlay", t_clip=10.0, sizer="plain")),
             ("t_clip=10 deadband=0.05", dict(mode="overlay", t_clip=10.0, deadband=0.05)),
             ("t_clip=10 deadband=0.20", dict(mode="overlay", t_clip=10.0, deadband=0.20))]
    for market, mname in ((SPOT, "spot"), (FUTURES, "futures 5x")):
        print(f"\nINNER-VALIDATION neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(TrendQualityKelly(**kw), *VALID, market=market, tag=tag)
        print(f"INNER-TRAIN neighbourhood / {mname}:")
        for tag, kw in grid:
            ev(TrendQualityKelly(**kw), *TRAIN, market=market, tag=tag, count=False)
    print(f"\nconfigurations evaluated in this run: {N_EVALUATED}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict on_bar peek check, run by hand for an unregistered strategy.

    ``tests/test_causality_strict.py`` only parametrizes over *registered*
    strategies, so an experiment gets none of that protection. Same
    two-opposite-tampers procedure used throughout this repo's experiments:
    bars after a cut are multiplied by 3 in one copy and divided by 3 in the
    other, and every decision at or before the cut must be identical.
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

    ok = True
    for mode in ("overlay", "continuous_vote", "single_window"):
        def decisions(frame):
            s = TrendQualityKelly(mode=mode, t_clip=4.0)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down))
               if oa != ob]
        pa = TrendQualityKelly(mode=mode, t_clip=4.0).prepare(up.copy())
        pb = TrendQualityKelly(mode=mode, t_clip=4.0).prepare(down.copy())
        worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut]
                                           - pb[c].to_numpy()[:cut])))
                    for c in ("target", "conf", "scale"))
        good = not bad and worst < 1e-12
        ok &= good
        print(f"  mode={mode:16s} orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- holdout

# Frozen before the holdout was read (step 4). See the ledger row / final
# report for the full decision rule; this dict is the exact configuration it
# refers to.
#
# mode="overlay", t_clip=10.0: selected on inner-validation, where it gave
# the deepest and most consistent drawdown cut of the whole 31-configuration
# search (spot 33.2% -> 21.2%, -12.0pp; futures 32.3% -> 19.5%, -12.8pp),
# and confirmed directionally on the 4-year inner-train (spot 43.3% -> 24.6%,
# -18.7pp; futures 35.3% -> 26.7%, -8.6pp) -- the larger, less noisy sample,
# used here only to check the selection is not an inner-validation artifact,
# never to re-select. Everything else is v4's own shipped default:
# horizons=(20,40,80), band=0.01, sizer="conditional", target_vol=0.55,
# max_leverage=2.0, deadband=0.10, vol_span=8d, anchor_span_days=180.
FROZEN = dict(mode="overlay", t_clip=10.0)
# Reported alongside it, since a variant that stays silent is selection by
# the operator (ROUTINE.md, "Running directions in parallel").
ALSO = [
    ("B continuous_vote t_clip=4 (best of a uniformly worse mode)",
     dict(mode="continuous_vote", t_clip=4.0)),
    ("C single_window w=120d t_clip=4 (best C on futures)",
     dict(mode="single_window", window_days=120.0, t_clip=4.0)),
]


def holdout() -> None:
    """Step 4. The configuration is frozen; the decision rule is in the ledger."""
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}:")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"  {name}", count=False)
        ev(TrendQualityKelly(**FROZEN), *OOS, market=market,
           tag="A overlay (FROZEN)", count=False)
        for tag, kw in ALSO:
            ev(TrendQualityKelly(**kw), *OOS, market=market, tag=f"  {tag}", count=False)


# -------------------------------------------------------------------- interval


def interval() -> None:
    """Paired stationary block-bootstrap: frozen config vs `kelly_regime_v4`.

    Identical resamples for both arms, the R-29/R-30 method reused rather
    than reinvented.
    """
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    rows = []
    for mname, market in MARKETS:
        a = run_period(TrendQualityKelly(**FROZEN), DF, *OOS, market=market,
                       start_balance=1_000.0, data_label=LABEL)
        b = run_period(get_strategy("kelly_regime_v4"), DF, *OOS, market=market,
                       start_balance=1_000.0, data_label=LABEL)
        ra, rb = daily_returns(a.equity).to_numpy(), daily_returns(b.equity).to_numpy()
        n = min(len(ra), len(rb))
        ra, rb = ra[:n], rb[:n]
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
        print(f"\n{mname}: paired trend_quality(overlay) - kelly_regime_v4, "
              f"2023+ holdout ({n} daily observations)")
        for stat_name, stat in (("Δ log growth", total_log_return),
                                ("Δ max drawdown (pp)", max_drawdown_from_returns)):
            r = paired_bootstrap(ra, rb, stat, indices=idx)
            mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
            print(f"  {stat_name:22s} {mark} {r.diff.point:>+7.3f} "
                  f"[{r.diff.lo:>+7.3f}, {r.diff.hi:>+7.3f}]  P(>0)={r.p_positive:.2f}")
            rows.append({"market": mname, "stat": stat_name, "trend_quality": r.stat_a,
                         "v4": r.stat_b, "diff": r.diff.point, "lo": r.diff.lo,
                         "hi": r.diff.hi, "p_positive": r.p_positive,
                         "significant": r.significant})
    out = ROOT / "reports" / "trend_quality"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "intervals.csv", index=False)
    print(f"\nwritten: {out / 'intervals.csv'}")


# ------------------------------------------------------------------------- eth


def eth() -> None:
    """Pre-registered falsification: does the drawdown/return ordering vs v4
    survive on ETH (Bitfinex, the R-17 window)? BTC over the same window is
    the control.
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
            ev(TrendQualityKelly(**FROZEN), None, None, df=df, market=market,
               tag="  trend_quality (frozen)", count=False)


# ----------------------------------------------------------------------- costs


def costs() -> None:
    """Step 4 cost checks: the real fee tier on spot, funding on futures."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ spot, at both taker tiers:")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            ev(get_strategy(name), *OOS, market=market, tag=f"    {name}", count=False)
        ev(TrendQualityKelly(**FROZEN), *OOS, market=market,
           tag="    trend_quality (frozen)", count=False)

    real = load_funding(ROOT / "data")
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ futures 5x with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr assumed after):")
    for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                        ("trend_quality (frozen)", TrendQualityKelly(**FROZEN))):
        lo = int(DF.index.searchsorted(OOS[0]))
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=blended, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:24s} final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>7,.0f}"
              f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------- windows


def windows(trials: int = 40, seed: int = 42) -> None:
    """Path sensitivity, the R-19 design: identical random windows, both arms."""
    from tradebot.metrics import max_drawdown_pct

    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("trend_quality", TrendQualityKelly(**FROZEN))]
    warmup = max(s.warmup for _, s in contenders) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(90, 731) * 288)
        specs.append((int(rng.integers(warmup, len(DF) - length)), length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = DF.iloc[start - warmup: start + length]
        for mname, market in MARKETS:
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
    out = ROOT / "reports" / "trend_quality"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "windows.csv", index=False)

    print(f"\n{trials} random windows (90-730 days), identical across strategies:\n")
    for mname, _ in MARKETS:
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name, _ in contenders:
            g = sub[sub.strategy == name].set_index("trial")
            print(f"    {name:16s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"beat hold {(g['return_pct'] > bench).mean():>5.0%}  "
                  f"liq {g.liquidated.mean():>4.0%}")
        a = sub[sub.strategy == "trend_quality"].set_index("trial")
        b = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")
        d_ret = (a.return_pct - b.return_pct).dropna()
        d_dd = (a.max_dd_pct - b.max_dd_pct).dropna()
        print(f"    paired trend_quality - v4: "
              f"return median {d_ret.median():+.1f}pp, higher in "
              f"{(d_ret > 0).mean():.0%};  DD median {d_dd.median():+.1f}pp, "
              f"deeper in {(d_dd > 0).mean():.0%}")
        print()


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"inspect": inspect, "sweep": sweep, "neighbours": neighbours,
            "causality": causality, "holdout": holdout, "interval": interval,
            "eth": eth, "costs": costs, "windows": windows}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_trend_quality.py [{'|'.join(cmds)}]")
