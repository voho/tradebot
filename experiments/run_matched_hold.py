#!/usr/bin/env python
"""Driver for backlog B-13 — kelly_regime_v4 against a de-levered buy-and-hold.

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants,
                                                 and set the matched exposure
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_matched_hold.py frontier    # the sweep (step 3)
    python experiments/run_matched_hold.py match       # solve for matched c
    python experiments/run_matched_hold.py insplit     # matched pairs inside each split
    python experiments/run_matched_hold.py causality   # by-hand lookahead probe
    python experiments/run_matched_hold.py holdout     # step 4, frozen
    python experiments/run_matched_hold.py interval    # paired bootstrap
    python experiments/run_matched_hold.py eth         # falsification test
    python experiments/run_matched_hold.py costs       # fee tier + funding
    python experiments/run_matched_hold.py windows     # 40-window path check
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.matched_hold import ConstantExposureHold, mean_notional  # noqa: E402
from experiments.matched_risk import realized_vol  # noqa: E402
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

OUT = ROOT / "reports" / "matched_hold"

N_EVALUATED = 0  # configurations searched in step 3, for deflated Sharpe

# The exposure grid the frontier is traced on. Not a performance search: c
# is the matching axis. It stops at 1.0 because that is spot's notional
# cap - above it the market, not the arm, sets the position (R-31's V).
C_GRID = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)

INCUMBENT = "kelly_regime_v4"


def hold(c: float, static: bool = False) -> ConstantExposureHold:
    return ConstantExposureHold(c=c, static=static)


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            count=False):
    """One backtest -> (metrics, realized vol, clamp fraction, result)."""
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    # How often the arm asked for more notional than the market allows.
    # Above the cap the two arms are no longer running the sizer they were
    # matched on, and the cell is not a controlled comparison (R-31's V).
    tgt = result.df["target"].to_numpy() if "target" in result.df else np.zeros(1)
    clamp = float(np.mean(np.abs(tgt) > market.leverage + 1e-9))
    return m, realized_vol(result.equity), clamp, result


def line(tag, m, vol, clamp, result):
    print(f"  {tag:34s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} fills={len(result.fills):>5d} "
          f"fees=${m.fees_paid:>8,.0f} clamp={clamp:>5.1%}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# -------------------------------------------------------------------- frontier


def _frontier_rows(start, end, split):
    rows = []
    for mname, market in MARKETS:
        for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                            (INCUMBENT, get_strategy(INCUMBENT))):
            m, vol, clamp, res = measure(strat, start, end, market=market)
            rows.append({"split": split, "market": mname, "arm": name,
                         "c": float("nan"), "final": m.final_balance, "vol": vol,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "fills": len(res.fills), "fees": m.fees_paid,
                         "clamp": clamp, "liquidated": m.liquidated,
                         "mean_notional": mean_notional(res)})
        for static in (False, True):
            arm = "static_hold" if static else "rebalanced_hold"
            for c in C_GRID:
                # A configuration is (arm, c). Scoring it on a second market
                # or a second split is another backtest, not another trial -
                # the R-28/R-31 convention.
                m, vol, clamp, res = measure(
                    hold(c, static), start, end, market=market,
                    count=(split == "inner-train" and mname == "spot"))
                rows.append({"split": split, "market": mname, "arm": arm, "c": c,
                             "final": m.final_balance, "vol": vol,
                             "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                             "fills": len(res.fills), "fees": m.fees_paid,
                             "clamp": clamp, "liquidated": m.liquidated,
                             "mean_notional": float("nan")})
    return rows


def frontier() -> None:
    """Step 3. Trace return-vs-risk for the passive arms, inner splits only."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
        rows += _frontier_rows(start, end, split)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "frontier.csv", index=False)

    for split in ("inner-train", "inner-validation"):
        for mname, _ in MARKETS:
            sub = df[(df.split == split) & (df.market == mname)]
            print(f"\n{split} / {mname}")
            print(f"  {'arm':17s} {'c':>5s} {'vol':>6s} {'final':>11s} "
                  f"{'DD':>6s} {'sharpe':>7s} {'fills':>6s} {'clamp':>6s}")
            for _, r in sub.iterrows():
                c = "  ---" if not np.isfinite(r.c) else f"{r.c:>5g}"
                print(f"  {r.arm:17s} {c} {r.vol:>6.3f} "
                      f"${r.final:>10,.0f} {r.max_dd:>5.1f}% {r.sharpe:>7.2f} "
                      f"{r.fills:>6.0f} {r.clamp:>5.1%}")
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'frontier.csv'}")


# ----------------------------------------------------------------------- match


def solve_c(target_vol: float, start, end, *, market=SPOT, static=False,
            df=None, tol: float = 0.02, max_iter: int = 8,
            c_max: float = 1.0) -> tuple[float, float, int]:
    """Find the constant exposure whose realized volatility equals ``target_vol``.

    A constant-exposure hold's realized volatility is very nearly ``c``
    times the asset's, so a proportional step converges in two or three
    backtests. The achieved volatility is returned rather than assumed:
    the residual is the reader's only check that the arms really were
    matched, and ``c`` is clipped at the market's cap so a match that is
    unreachable reports itself instead of silently pinning.
    """
    c = min(0.5, c_max)
    _, vol, _, _ = measure(hold(c, static), start, end, df=df, market=market)
    for it in range(1, max_iter + 1):
        if vol <= 0 or not np.isfinite(vol):
            return float("nan"), vol, it
        if abs(vol - target_vol) <= tol * target_vol:
            return c, vol, it
        c = float(np.clip(c * (target_vol / vol), 1e-3, c_max))
        _, vol, _, _ = measure(hold(c, static), start, end, df=df, market=market)
        if c >= c_max and vol < target_vol:  # the cap binds; no match exists
            return c, vol, it
    return c, vol, max_iter


def match(write: bool = True) -> dict:
    """Solve, on inner-validation only, for the exposures that equalize risk.

    Two axes, because "de-levered to match" is ambiguous and this project
    has not been careful about which one it means:

    **vol**       ``c`` such that the hold's realized volatility equals
                  v4's over the same span. The R-31 convention.
    **notional**  ``c`` = v4's own mean notional fraction. The literal
                  form of the "it just holds less" critique; no solver.
    """
    frozen = {}
    for mname, market in MARKETS:
        m, v4_vol, v4_clamp, res = measure(get_strategy(INCUMBENT), *VALID,
                                           market=market)
        c_notional = mean_notional(res)
        cell = {"v4_vol": v4_vol, "v4_clamp": v4_clamp,
                "v4_final": m.final_balance, "v4_max_dd": m.max_drawdown_pct,
                "c_notional": c_notional}
        print(f"\n{mname}")
        print(f"  inner-validation v4: vol {v4_vol:.3f}  DD {m.max_drawdown_pct:.1f}%"
              f"  mean notional {c_notional:.3f}  clamp {v4_clamp:.1%}")
        for static in (False, True):
            arm = "static" if static else "rebalanced"
            c, vol, it = solve_c(v4_vol, *VALID, market=market, static=static,
                                 c_max=market.leverage)
            cell[f"c_vol_{arm}"] = c
            cell[f"vol_{arm}"] = vol
            cell[f"iterations_{arm}"] = it
            print(f"  match on VOL, {arm:11s} hold: c={c:.3f} "
                  f"(vol {vol:.3f} vs v4 {v4_vol:.3f}, {it} backtests)")
        frozen[mname] = cell
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"\nwritten: {OUT / 'frozen.json'}")
    return frozen


def _frozen() -> dict:
    return json.loads((OUT / "frozen.json").read_text())


def _arms(mname: str):
    """The frozen passive arms for one market, as (label, strategy)."""
    f = _frozen()[mname]
    return [
        (f"vol-matched rebalanced c={f['c_vol_rebalanced']:.3f}",
         hold(f["c_vol_rebalanced"], static=False)),
        (f"vol-matched static     c={f['c_vol_static']:.3f}",
         hold(f["c_vol_static"], static=True)),
        (f"notional-matched reb.  c={f['c_notional']:.3f}",
         hold(f["c_notional"], static=False)),
    ]


def insplit() -> None:
    """The matched comparison *within* each inner split, before the holdout.

    R-31's lesson was that an exposure solved on one regime need not match
    risk in the next, so the ordering has to be read inside each split as
    well as across the freeze. Nothing here touches 2023+.
    """
    for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
        for mname, market in MARKETS:
            m, v4_vol, clamp, res = measure(get_strategy(INCUMBENT), start, end,
                                            market=market)
            print(f"\n{split} / {mname}  (exposures solved inside this split)")
            line(f"  {INCUMBENT}", m, v4_vol, clamp, res)
            line("  buy_and_hold", *measure(get_strategy("buy_and_hold"), start,
                                            end, market=market))
            for static in (False, True):
                c, vol, _ = solve_c(v4_vol, start, end, market=market,
                                    static=static, c_max=market.leverage)
                tag = "static" if static else "rebalanced"
                line(f"  vol-matched {tag} c={c:.3f}",
                     *measure(hold(c, static), start, end, market=market))
            cn = mean_notional(res)
            line(f"  notional-matched c={cn:.3f}",
                 *measure(hold(cn), start, end, market=market))


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict lookahead probe, by hand — experiments get no CI protection.

    ``tests/test_causality_strict.py`` parametrizes over the *registry*, so
    an unregistered strategy is unprotected. Same two-opposite-tampers
    procedure as R-28/R-31: bars after a cut are multiplied by 3 in one
    copy and divided by 3 in the other, and every decision at or before the
    cut must be identical.

    A passive constant-exposure arm has no indicator to leak through, which
    is exactly why the probe is worth running: the arm reads the *account*
    (position, equity, close) rather than a prepared column, and a bug
    there would be invisible to a column-difference test.
    """
    from tradebot.broker import PaperBroker
    from tradebot.orders import Order
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
    for static in (False, True):
        arm = "static" if static else "rebalanced"

        def decisions(frame):
            s = hold(0.5, static)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            # A real position, so the rebalance branch is actually exercised
            # rather than short-circuiting on a flat account.
            broker.execute(Order(target=0.1), prepared.index[0],
                           float(prepared["open"].iloc[0]))
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down))
               if oa != ob]
        pa = hold(0.5, static).prepare(up.copy())
        pb = hold(0.5, static).prepare(down.copy())
        worst = float(np.nanmax(np.abs(pa["target"].to_numpy()[:cut]
                                       - pb["target"].to_numpy()[:cut])))
        good = not bad and worst < 1e-12
        ok &= good
        print(f"  arm={arm:11s} orders {'match' if not bad else f'DIFFER at {bad}'}"
              f"   max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")

    # And the equity-path check that a column comparison cannot make: the
    # measured period must be bit-identical under both tampers.
    for static in (False, True):
        a = run_backtest(hold(0.5, static), up.iloc[:cut + 1], FUTURES, 1_000.0,
                         data_label=LABEL)
        b = run_backtest(hold(0.5, static), down.iloc[:cut + 1], FUTURES, 1_000.0,
                         data_label=LABEL)
        # up/down differ only at the cut bar itself, which fills the NEXT bar
        # and so cannot move any equity at or before it.
        worst = float(np.max(np.abs(a.equity.to_numpy()[:cut]
                                    - b.equity.to_numpy()[:cut])))
        ok &= worst < 1e-9
        print(f"  arm={'static' if static else 'rebalanced':11s} "
              f"max |equity difference| before the cut = {worst:.3e}   "
              f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- holdout


def holdout() -> None:
    """Step 4. Exposure frozen on inner-validation; decision rule in the ledger."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}")
        contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                      (INCUMBENT, get_strategy(INCUMBENT))] + _arms(mname)
        for name, strat in contenders:
            m, vol, clamp, res = measure(strat, *OOS, market=market)
            line(f"  {name}", m, vol, clamp, res)
            rows.append({"market": mname, "arm": name,
                         "final": m.final_balance, "vol": vol,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "fills": len(res.fills), "fees": m.fees_paid,
                         "clamp": clamp, "liquidated": m.liquidated,
                         "mean_notional": mean_notional(res)})
        # V, the validity gate, reported before any decision rule is read.
        # V1: the risk match frozen on inner-validation must survive into
        # 2023+, within 20% relative. V2: the passive arm must not be pinned
        # at the market's notional cap - if matching v4's volatility needs
        # more exposure than the market allows, no such benchmark exists.
        # v4's OWN clamp fraction is reported as a diagnostic and does not
        # void a cell: it is a property of the incumbent as registered, and
        # every published v4-vs-hold number in this repo carries it.
        v4 = [r for r in rows if r["market"] == mname and r["arm"] == INCUMBENT][0]
        print(f"  --- validity gate V (vs {INCUMBENT}, vol {v4['vol']:.3f}; "
              f"v4 clamp {v4['clamp']:.1%}, diagnostic only)")
        for r in rows:
            if r["market"] != mname or r["arm"] in ("buy_and_hold", INCUMBENT):
                continue
            gap = abs(r["vol"] - v4["vol"]) / v4["vol"]
            ok = gap <= 0.20 and r["clamp"] < 0.01
            why = "" if ok else (" (V1 risk match)" if gap > 0.20
                                 else " (V2 arm pinned at the cap)")
            print(f"      {r['arm']:34s} vol gap {gap:>5.1%}  "
                  f"arm clamp {r['clamp']:>5.1%}  "
                  f"{'VALID' if ok else 'VOID'}{why}")
            r["vol_gap"] = gap
            r["valid"] = ok
    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# -------------------------------------------------------------------- interval


def interval() -> None:
    """The decision statistic: paired block-bootstrap on holdout daily returns.

    Identical resamples for both arms of a pair, so the market's own
    variance cancels instead of swamping the gap — the R-29/R-30 method,
    reused rather than reinvented. Every comparison is stated as
    ``kelly_regime_v4 − benchmark``, so a negative drawdown difference
    means v4 draws down less, which is the direction this project claims.
    """
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    rows = []
    for mname, market in MARKETS:
        def curve(strat):
            res = run_period(strat, DF, *OOS, market=market,
                             start_balance=1_000.0, data_label=LABEL)
            return daily_returns(res.equity).to_numpy()

        v4 = curve(get_strategy(INCUMBENT))
        benches = [("buy_and_hold (R-29 reproduction)",
                    curve(get_strategy("buy_and_hold")))]
        benches += [(label, curve(strat)) for label, strat in _arms(mname)]

        idx = stationary_bootstrap_indices(len(v4), 30.0, 2_000,
                                           np.random.default_rng(7))
        print(f"\n{mname}: paired {INCUMBENT} − benchmark on the 2023+ holdout "
              f"({len(v4)} daily observations)")
        for label, b in benches:
            for stat_name, stat in (("Δ log growth", total_log_return),
                                    ("Δ max drawdown (pp)", max_drawdown_from_returns)):
                r = paired_bootstrap(v4, b, stat, indices=idx)
                mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
                print(f"  {label:34s} {stat_name:22s} {mark} {r.diff.point:>+8.3f} "
                      f"[{r.diff.lo:>+8.3f}, {r.diff.hi:>+8.3f}]  "
                      f"P(>0)={r.p_positive:.2f}")
                rows.append({"market": mname, "benchmark": label, "stat": stat_name,
                             "v4": r.stat_a, "benchmark_value": r.stat_b,
                             "diff": r.diff.point, "lo": r.diff.lo,
                             "hi": r.diff.hi, "p_positive": r.p_positive,
                             "significant": r.significant})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intervals.csv", index=False)
    print(f"\nwritten: {OUT / 'intervals.csv'}")


# ------------------------------------------------------------------------- eth


def eth() -> None:
    """Pre-registered falsification: does the ORDERING replicate on ETH?

    Same venue (Bitfinex), same window as R-17, R-28 and R-31, only the
    asset varies. The exposure is re-matched on each asset's own
    volatility — matching is a property of the risk axis, not a fitted
    parameter — and the question is whether v4 still draws down less than
    a passive arm carrying the same risk.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            m, v4_vol, clamp, res = measure(get_strategy(INCUMBENT), None, None,
                                            df=df, market=market)
            line(f"    {INCUMBENT}", m, v4_vol, clamp, res)
            line("    buy_and_hold",
                 *measure(get_strategy("buy_and_hold"), None, None, df=df,
                          market=market))
            c_notional = mean_notional(res)
            for static in (False, True):
                c, vol, _ = solve_c(v4_vol, None, None, df=df, market=market,
                                    static=static, c_max=market.leverage)
                if not np.isfinite(c):
                    print(f"    no constant exposure reaches vol {v4_vol:.3f}")
                    continue
                tag = "static" if static else "rebalanced"
                line(f"    vol-matched {tag} c={c:.3f}",
                     *measure(hold(c, static), None, None, df=df, market=market))
            line(f"    notional-matched c={c_notional:.3f}",
                 *measure(hold(c_notional), None, None, df=df, market=market))


# ----------------------------------------------------------------------- costs


def costs() -> None:
    """Step 4 cost checks: the real fee tier on spot, funding on futures."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ spot, at both taker tiers:")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name, strat in ([("buy_and_hold", get_strategy("buy_and_hold")),
                             (INCUMBENT, get_strategy(INCUMBENT))]
                            + _arms("spot")):
            line(f"    {name}", *measure(strat, *OOS, market=market))

    real = load_funding(ROOT / "data")
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ futures 5x with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr after):")
    contenders = ([("buy_and_hold", get_strategy("buy_and_hold")),
                   (INCUMBENT, get_strategy(INCUMBENT))] + _arms("futures"))
    lo = int(DF.index.searchsorted(OOS[0]))
    for name, strat in contenders:
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=blended, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:34s} final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>7,.0f}"
              f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------- windows


def windows(trials: int = 40, seed: int = 42) -> None:
    """Path sensitivity, the R-19 design: identical random windows, every arm.

    One thing here is *not* inherited from R-19. Both B-11 branches had to
    attach the same caveat to their window tables — "the exposures are
    frozen rather than re-matched per window, so the arms are only
    approximately equal-risk" — and on futures R-31's e-process arm was
    plainly running hot as a result. That caveat is avoidable: a
    constant-exposure arm's realized volatility is proportional to ``c`` to
    better than 1% (measured: c=0.1 -> 0.083, c=0.5 -> 0.412 on
    inner-validation spot), so one probe backtest per window fixes the
    exposure that matches v4 *in that window*, exactly. This run therefore
    carries a genuinely equal-risk arm alongside the frozen ones, and
    reports the residual volatility gap rather than assuming it away.
    """
    from tradebot.metrics import max_drawdown_pct

    # One set of strategies scored on both markets, as R-19 does: the SPOT
    # freeze, so the passive arms carry an exposure that is legal on both.
    contenders = ([("buy_and_hold", get_strategy("buy_and_hold")),
                   (INCUMBENT, get_strategy(INCUMBENT))] + _arms("spot"))
    PROBE_C = 0.35  # the probe exposure; only its measured vol is used

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
            v4_vol = float("nan")
            for name, strat in contenders:
                res = run_backtest(strat, window, market, 1_000.0,
                                   trade_start=warmup, data_label=LABEL)
                eq = res.equity.to_numpy(dtype=float)
                base, seg = eq[warmup], eq[warmup:]
                ok = np.isfinite(base) and base > 0
                vol = realized_vol(seg) if ok else float("nan")
                if name == INCUMBENT:
                    v4_vol = vol
                rows.append({"trial": k, "market": mname, "strategy": name,
                             "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                             "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                             "vol": vol, "c": float("nan"),
                             "liquidated": res.liquidated})
            # ... and the arm matched to v4 inside this window.
            probe = run_backtest(hold(PROBE_C), window, market, 1_000.0,
                                 trade_start=warmup, data_label=LABEL)
            probe_vol = realized_vol(probe.equity.to_numpy(dtype=float)[warmup:])
            c = (PROBE_C * v4_vol / probe_vol) if probe_vol > 0 else float("nan")
            c = float(np.clip(c, 1e-3, market.leverage)) if np.isfinite(c) else c
            if np.isfinite(c):
                res = run_backtest(hold(c), window, market, 1_000.0,
                                   trade_start=warmup, data_label=LABEL)
                eq = res.equity.to_numpy(dtype=float)
                base, seg = eq[warmup], eq[warmup:]
                ok = np.isfinite(base) and base > 0
                rows.append({"trial": k, "market": mname,
                             "strategy": "per-window matched hold",
                             "return_pct": 100.0 * (seg[-1] / base - 1.0) if ok else -100.0,
                             "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
                             "vol": realized_vol(seg) if ok else float("nan"),
                             "c": c, "liquidated": res.liquidated})
        print(f"[{k}/{trials}]", end=" ", flush=True, file=sys.stderr)
    res = pd.DataFrame(rows)
    contenders = contenders + [("per-window matched hold", None)]
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "windows.csv", index=False)

    print(f"\n{trials} random windows (90-730 days), identical across strategies:\n")
    for mname, _ in MARKETS:
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name, _ in contenders:
            g = sub[sub.strategy == name].set_index("trial")
            print(f"    {name:34s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"median vol {g.vol.median():>5.2f}  "
                  f"beat hold {(g['return_pct'] > bench).mean():>5.0%}  "
                  f"liq {g.liquidated.mean():>4.0%}")
        a = sub[sub.strategy == INCUMBENT].set_index("trial")
        m = sub[sub.strategy == "per-window matched hold"].set_index("trial")
        gap = ((m.vol - a.vol).abs() / a.vol).dropna()
        print(f"    per-window risk match: median |vol gap| {gap.median():.2%}, "
              f"worst {gap.max():.2%}, exposure c median {m.c.median():.3f} "
              f"[{m.c.min():.3f}, {m.c.max():.3f}]")
        for name, _ in contenders:
            if name == INCUMBENT:
                continue
            b = sub[sub.strategy == name].set_index("trial")
            d_ret = (a.return_pct - b.return_pct).dropna()
            d_dd = (a.max_dd_pct - b.max_dd_pct).dropna()
            print(f"    paired v4 − {name:28s} return median {d_ret.median():>+8.1f}pp, "
                  f"v4 higher in {(d_ret > 0).mean():>4.0%};  "
                  f"DD median {d_dd.median():>+7.1f}pp, "
                  f"v4 deeper in {(d_dd > 0).mean():>4.0%}")
        print()


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"frontier": frontier, "match": match, "insplit": insplit,
            "causality": causality,
            "holdout": holdout, "interval": interval, "eth": eth,
            "costs": costs, "windows": windows}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_matched_hold.py [{'|'.join(cmds)}]")
