#!/usr/bin/env python
"""Driver for the "novel" B-05 branch — continuous funding-crowding correction.

See ``experiments/funding_crowding_novel.py`` for the derivation and the
``FundingCrowdingKelly`` strategy. This file only evaluates it, following
ROUTINE.md step 3/4 on an INNER split restricted to the window the
committed funding file actually covers with headroom before the 2023+
holdout:

    inner-train        2020-01-01 -> 2021-12-31   fit / sweep / iterate
    inner-validation    2022-01-01 -> 2022-12-31   select between variants
    (holdout            2023-01-01 ->               NEVER read from this file)

Usage::

    python experiments/run_funding_crowding_novel.py materiality  # check #1
    python experiments/run_funding_crowding_novel.py results      # step 4
    python experiments/run_funding_crowding_novel.py feetier      # falsification (a)
    python experiments/run_funding_crowding_novel.py fundingcost  # falsification (b)
    python experiments/run_funding_crowding_novel.py causality    # lookahead probe
    python experiments/run_funding_crowding_novel.py all
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

from experiments.funding_crowding_novel import (  # noqa: E402
    FUNDING_SETTLEMENTS_PER_YEAR,
    FundingCrowdingKelly,
)
from experiments.funding_signal import causal_funding_column, funding_coverage  # noqa: E402
from tradebot.broker import MarketSpec, PaperBroker  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context  # noqa: E402

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

# The two inner slices, both strictly inside the funding file's 2020-2023
# coverage and strictly before the 2023-01-01 holdout boundary.
INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VALID = ("2022-01-01", "2022-12-31")
COMBINED = ("2020-01-01", "2022-12-31")
HOLDOUT_BOUNDARY = pd.Timestamp("2023-01-01", tz="UTC")


def _load():
    df, label = load_dataset(ROOT / "data", "spot")
    funding = load_funding(ROOT / "data")
    if funding is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    df = df.copy()
    df["funding"] = causal_funding_column(df.index, funding)
    return df, label, funding


DF, LABEL, REAL_FUNDING = _load()

N_EVALUATED = 0  # distinct (strategy-config, market, period, fee-tier) backtests
_CACHE: dict = {}


def _key(strategy, market, start, end, funding_charged: bool) -> tuple:
    params = tuple(sorted((k, v) for k, v in vars(strategy).items()))
    return (type(strategy).__name__, params, market.name, market.fee_rate,
            start, end, funding_charged)


def _period(strategy, market, start, end, *, funding=None, count=True):
    """One backtest over ``DF[start:end]`` with a real warmup prefix.

    Mirrors ``scripts/funding_study.py``'s ``_period`` (needed because
    ``tradebot.window.run_period`` does not forward a ``funding`` kwarg).
    Caches by config so a number computed once (e.g. the funding_scale=1.0
    inner-train run, used by both `materiality` and `results`) counts once
    toward the deflated-Sharpe bookkeeping, never twice.
    """
    global N_EVALUATED
    key = _key(strategy, market, start, end, funding is not None)
    if key in _CACHE:
        return _CACHE[key]
    lo = 0 if start is None else int(DF.index.searchsorted(start))
    hi = len(DF) if end is None else int(DF.index.searchsorted(end, side="right"))
    if end is not None:
        assert pd.Timestamp(end, tz="UTC") < HOLDOUT_BOUNDARY, \
            f"refusing to evaluate up to {end}: on/after the 2023-01-01 holdout"
    pre = min(lo, strategy.warmup)
    if count:
        N_EVALUATED += 1
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                       trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:],
                                           df=raw.df.iloc[pre:])
    m = compute_metrics(trimmed)
    result = (m, trimmed, raw.funding_paid)
    _CACHE[key] = result
    return result


def line(tag, m, result, funding_paid=0.0):
    print(f"  {tag:42s} final=${m.final_balance:>11,.0f} ({m.profit_pct:>+7.1f}%) "
          f"trades={m.num_trades:>4d} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f}"
          f"{f' funding_paid=${funding_paid:>8,.0f}' if funding_paid else ''}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------- materiality


def materiality() -> None:
    """Pre-registered check #1: is r_t / sigma_t^2 even the right order of magnitude?

    Written before looking at the numbers below: if |r_t/sigma_t^2| sits
    three-plus orders of magnitude below the sizer's 0-2x exposure range
    almost always, the mechanism is empirically vacuous even though it is
    correctly derived, and that is the headline, not a footnote.
    """
    print(f"funding file covers {funding_coverage(REAL_FUNDING)[0]:%Y-%m-%d} -> "
          f"{funding_coverage(REAL_FUNDING)[1]:%Y-%m-%d} (2020-2023, real Binance)\n")
    print(f"inner-train {INNER_TRAIN[0]} -> {INNER_TRAIN[1]}, spot, funding_scale=1.0\n")

    novel = FundingCrowdingKelly(funding_scale=1.0)
    _, res_novel, _ = _period(novel, SPOT, *INNER_TRAIN)
    corr = res_novel.df["funding_correction"].to_numpy()
    corr = corr[np.isfinite(corr)]
    abs_corr = np.abs(corr)

    baseline = get_strategy("kelly_regime_v4")
    _, res_base, _ = _period(baseline, SPOT, *INNER_TRAIN)
    tgt = res_base.df["target"].to_numpy()
    in_market_tgt = np.abs(tgt[tgt > 1e-9])
    typical_target = float(np.mean(in_market_tgt)) if len(in_market_tgt) else float("nan")

    print(f"|r_t / sigma_t^2|  (funding_scale=1.0, {len(abs_corr):,} bars)")
    print(f"  mean   {np.mean(abs_corr):.4f}")
    print(f"  median {np.median(abs_corr):.4f}")
    print(f"  90th pct {np.percentile(abs_corr, 90):.4f}")
    print(f"  max    {np.max(abs_corr):.4f}")
    print(f"\nkelly_regime_v4's own target exposure while in market (same window):")
    print(f"  mean |target|  {typical_target:.4f}   (0-{baseline.max_leverage:g}x range, "
          f"deadband={baseline.deadband:.2f})")
    ratio_mean = np.mean(abs_corr) / typical_target if typical_target else float("nan")
    ratio_median = np.median(abs_corr) / typical_target if typical_target else float("nan")
    print(f"\ncorrection / typical target: mean ratio {ratio_mean:.2%}, "
          f"median ratio {ratio_median:.2%}")
    frac_above_deadband = float(np.mean(abs_corr > baseline.deadband))
    print(f"correction exceeds the {baseline.deadband:.0%} deadband on "
          f"{frac_above_deadband:.1%} of bars")
    print("\n(materiality verdict is written into "
          "experiments/funding_crowding_novel_REPORT.md, not decided here — "
          "this command only measures.)")


# --------------------------------------------------------------------- results


def results() -> None:
    """Step 4: funding_scale in {0.5, 1.0, 2.0} (sensitivity, not a search) vs kelly_regime_v4."""
    for split_name, (start, end) in (("inner-train", INNER_TRAIN),
                                     ("inner-validation", INNER_VALID)):
        for mname, market in MARKETS:
            print(f"\n{split_name} / {mname}  ({start} -> {end})")
            m, res, _ = _period(get_strategy("kelly_regime_v4"), market, start, end)
            line("kelly_regime_v4 (baseline)", m, res)
            for scale in (0.5, 1.0, 2.0):
                s = FundingCrowdingKelly(funding_scale=scale)
                m, res, _ = _period(s, market, start, end)
                tag = f"funding_crowding_novel scale={scale:g}"
                tag += "  <- frozen (un-fit)" if scale == 1.0 else "  <- sensitivity only"
                line(tag, m, res)


# ------------------------------------------------------------------- feetier


def feetier() -> None:
    """Falsification (a): does turnover from re-targeting survive a 0.40% taker (R-13)?"""
    print(f"\ninner-train {INNER_TRAIN[0]} -> {INNER_TRAIN[1]}, spot, "
          f"Bitstamp entry taker tier (0.40%):\n")
    hi_fee_spot = MarketSpec.spot(fee_rate=0.004)
    m, res, _ = _period(get_strategy("kelly_regime_v4"), hi_fee_spot, *INNER_TRAIN)
    line("kelly_regime_v4 (baseline)", m, res)
    s = FundingCrowdingKelly(funding_scale=1.0)
    m, res, _ = _period(s, hi_fee_spot, *INNER_TRAIN)
    line("funding_crowding_novel scale=1.0", m, res)

    print(f"\nfor reference, the same pair at the 0.10% table-assumption tier:")
    m, res, _ = _period(get_strategy("kelly_regime_v4"), SPOT, *INNER_TRAIN)
    line("kelly_regime_v4 (baseline)", m, res)
    m, res, _ = _period(FundingCrowdingKelly(funding_scale=1.0), SPOT, *INNER_TRAIN)
    line("funding_crowding_novel scale=1.0", m, res)


# ---------------------------------------------------------------- fundingcost


def fundingcost() -> None:
    """Falsification (b): does the derived scaler cut the DOLLAR funding bill (the actual point)?

    Combined inner-train+inner-validation, futures 5x, real funding charged
    vs funding-free, for the frozen (funding_scale=1.0) strategy and the
    kelly_regime_v4 baseline.
    """
    start, end = COMBINED
    print(f"\ncombined inner split {start} -> {end}, futures 5x:\n")
    print(f"  {'':42s} {'funding-free':>13s} {'with funding':>13s} "
          f"{'cost':>7s} {'paid':>10s}")
    for name, strat in (("kelly_regime_v4 (baseline)", get_strategy("kelly_regime_v4")),
                        ("funding_crowding_novel scale=1.0", FundingCrowdingKelly(funding_scale=1.0))):
        free, res_free, _ = _period(strat, FUTURES, start, end)
        paid, res_paid, cost = _period(strat, FUTURES, start, end, funding=REAL_FUNDING)
        print(f"  {name:42s} ${free.final_balance:>12,.0f} "
              f"${paid.final_balance:>12,.0f} "
              f"{paid.final_balance / free.final_balance - 1:>6.0%} ${cost:>9,.0f}")
    print("\n(reported: does the novel arm's funding paid, in dollars, come in "
          "below the baseline's, without a materially worse funding-free number?)")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """By-hand lookahead probe (the R-28 / matched_risk.py design; no CI protection here.

    Two opposite tampers on every bar strictly after a cut (prices x3 / /3,
    funding column too) must leave every decision at or before the cut
    bit-identical. This substitutes for tests/test_causality_strict.py,
    which only parametrizes over the registry and never sees this
    unregistered strategy.
    """
    df = DF.iloc[-200_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0
    # funding column: opposite tampers too, same convention. Funding rates
    # can be negative, so *3 / /3 (not an absolute-value trick) is the
    # correct opposite-tamper pair, same as price.
    up.iloc[cut:, up.columns.get_loc("funding")] *= 3.0
    down.iloc[cut:, down.columns.get_loc("funding")] /= 3.0

    strategy_factory = lambda: FundingCrowdingKelly(funding_scale=1.0)

    def decisions(frame):
        s = strategy_factory()
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    pa = strategy_factory().prepare(up.copy())
    pb = strategy_factory().prepare(down.copy())
    worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                for c in ("target", "funding_correction", "kelly_vol"))
    ok = not bad and worst < 1e-9
    print(f"tampered from bar {cut:,} of {len(df):,} (prices x3//3, funding x3//3)")
    print(f"  orders {'match' if not bad else f'DIFFER at {bad}'}")
    print(f"  max |column difference| before the cut = {worst:.3e}")
    print(f"  {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------------ all


def all_() -> None:
    materiality()
    print("\n" + "=" * 78)
    results()
    print("\n" + "=" * 78)
    feetier()
    print("\n" + "=" * 78)
    fundingcost()
    print("\n" + "=" * 78)
    causality()
    print(f"\n{'=' * 78}\ndistinct (strategy-config, market, period, fee-tier) backtests "
          f"evaluated: {N_EVALUATED}")


COMMANDS = {"materiality": materiality, "results": results, "feetier": feetier,
            "fundingcost": fundingcost, "causality": causality, "all": all_}


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in COMMANDS:
        COMMANDS[choice]()
        if choice != "all":
            print(f"\ndistinct (strategy-config, market, period, fee-tier) backtests "
                  f"evaluated so far: {N_EVALUATED}")
    else:
        print(f"usage: python experiments/run_funding_crowding_novel.py "
              f"[{'|'.join(COMMANDS)}]")
