#!/usr/bin/env python
"""Driver for backlog B-11 — matched-risk frontier, e-process gate vs latched vote.

Splits follow ROUTINE.md step 3::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select between variants,
                                                 and set the matched exposures
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_matched_risk.py parity      # reproduces R-28's E1
    python experiments/run_matched_risk.py frontier    # the sweep (step 3)
    python experiments/run_matched_risk.py match       # solve for matched k
    python experiments/run_matched_risk.py causality   # by-hand lookahead probe
    python experiments/run_matched_risk.py holdout     # step 4, frozen
    python experiments/run_matched_risk.py interval    # paired bootstrap on the holdout
    python experiments/run_matched_risk.py eth         # falsification test
    python experiments/run_matched_risk.py costs       # fee tier + funding
    python experiments/run_matched_risk.py windows     # 40-window path check
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

from experiments.matched_risk import GatedKelly, realized_vol  # noqa: E402
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

OUT = ROOT / "reports" / "matched_risk"

N_EVALUATED = 0  # configurations searched in step 3, for deflated Sharpe

# The exposure grid the frontier is traced on. Not a performance search:
# k is the matching axis, swept wide enough that the two gates' realized
# volatilities overlap. The e-process needs the top of it (R-28: it holds
# 0.27x the incumbent's exposure), the vote needs the bottom.
K_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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
    # How often the strategy asked for more notional than the market allows.
    # order_notional(t) is clamped at |t| > leverage, so above that the two
    # arms are no longer running the same sizer and the cell is not a
    # controlled comparison.
    tgt = result.df["target"].to_numpy() if "target" in result.df else np.zeros(1)
    clamp = float(np.mean(np.abs(tgt) > market.leverage + 1e-9))
    return m, realized_vol(result.equity), clamp, result


def line(tag, m, vol, clamp, result):
    print(f"  {tag:30s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} fills={len(result.fills):>5d} "
          f"fees=${m.fees_paid:>8,.0f} clamp={clamp:>5.1%}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# ---------------------------------------------------------------------- parity


def parity() -> None:
    """This file's evidence arm must BE R-28's E1, or nothing below transfers."""
    from experiments.eprocess_regime import EProcessRegime

    print("Inner-validation, spot. The two rows must be identical.")
    for tag, s in (("R-28 E1 (eprocess_regime)",
                    EProcessRegime(bet_halflife_days=20.0, gate=True, sizing="fixed")),
                   ("B-11 evidence gate, k=1",
                    GatedKelly(gate="evidence", exposure=1.0, sizer="plain"))):
        line(tag, *measure(s, *VALID))
    a = run_period(EProcessRegime(bet_halflife_days=20.0, gate=True, sizing="fixed"),
                   DF, *VALID, market=SPOT, start_balance=1_000.0, data_label=LABEL)
    b = run_period(GatedKelly(gate="evidence", exposure=1.0, sizer="plain"),
                   DF, *VALID, market=SPOT, start_balance=1_000.0, data_label=LABEL)
    worst = float(np.max(np.abs(a.equity.to_numpy() - b.equity.to_numpy())))
    print(f"\n  max |equity difference| over the period: {worst:.3e}  "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    print("\nAnd the tautology R-28 flagged, in one line — same sizer, same k,")
    print("wildly different risk, which is why 'better risk, worse return' was")
    print("not yet a comparison:")
    for gate in ("vote", "evidence"):
        line(f"{gate} gate, k=1", *measure(GatedKelly(gate=gate, exposure=1.0), *VALID))


# -------------------------------------------------------------------- frontier


def _frontier_rows(start, end, split):
    rows = []
    for mname, market in MARKETS:
        for sizer in ("plain", "conditional"):
            for gate in ("vote", "evidence"):
                for k in K_GRID:
                    s = GatedKelly(gate=gate, exposure=k, sizer=sizer)
                    # A configuration is (sizer, gate, k). Scoring it on a
                    # second market or a second split is another backtest,
                    # not another trial — the R-28 convention.
                    m, vol, clamp, res = measure(
                        s, start, end, market=market,
                        count=(split == "inner-train" and mname == "spot"))
                    rows.append({
                        "split": split, "market": mname, "sizer": sizer,
                        "gate": gate, "k": k, "final": m.final_balance,
                        "vol": vol, "max_dd": m.max_drawdown_pct,
                        "sharpe": m.sharpe, "fills": len(res.fills),
                        "fees": m.fees_paid, "clamp": clamp,
                        "liquidated": m.liquidated,
                        "mean_target": float(res.df["target"].mean()),
                    })
    return rows


def frontier() -> None:
    """Step 3. Trace return-vs-risk for both gates, on the inner splits only."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (start, end), split in ((TRAIN, "inner-train"), (VALID, "inner-validation")):
        rows += _frontier_rows(start, end, split)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "frontier.csv", index=False)

    for split in ("inner-train", "inner-validation"):
        for mname, _ in MARKETS:
            for sizer in ("plain", "conditional"):
                sub = df[(df.split == split) & (df.market == mname)
                         & (df.sizer == sizer)]
                print(f"\n{split} / {mname} / sizer={sizer}")
                print(f"  {'gate':9s} {'k':>5s} {'vol':>6s} {'final':>11s} "
                      f"{'DD':>6s} {'sharpe':>7s} {'fills':>6s} {'clamp':>6s}")
                for _, r in sub.iterrows():
                    print(f"  {r.gate:9s} {r.k:>5g} {r.vol:>6.3f} "
                          f"${r.final:>10,.0f} {r.max_dd:>5.1f}% {r.sharpe:>7.2f} "
                          f"{r.fills:>6.0f} {r.clamp:>5.1%}")
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'frontier.csv'}")


# ----------------------------------------------------------------------- match


def solve_k(gate: str, target: float, start, end, *, market=SPOT,
            sizer: str = "plain", df=None, tol: float = 0.02,
            max_iter: int = 8) -> tuple[float, float, int]:
    """Find the exposure whose realized volatility equals ``target``.

    Interpolating the frontier grid would be cheaper and is wrong: the
    grid's (k, vol) pairs are not monotone at the bottom of the range,
    where a low-exposure arm trades so rarely that its realized
    volatility is dominated by a handful of fills. Sorting such a grid by
    volatility scrambles k and silently returns a nonsense match.

    So this solves directly. ``k`` rescales the position exactly
    (``min(k*tv/vol, k*ml) == k*min(tv/vol, ml)``), so realized
    volatility is very nearly proportional to it and a secant step from
    that proportionality converges in a few backtests. Returns
    ``(k, achieved_vol, iterations)`` — the achieved volatility is
    reported rather than assumed, because the residual is the reader's
    only check that the two arms really were matched.
    """
    k = 1.0
    _, vol, _, _ = measure(GatedKelly(gate=gate, exposure=k, sizer=sizer),
                           start, end, df=df, market=market)
    for it in range(1, max_iter + 1):
        if vol <= 0 or not np.isfinite(vol):
            return float("nan"), vol, it
        if abs(vol - target) <= tol * target:
            return k, vol, it
        k = float(np.clip(k * (target / vol), 1e-3, 1e3))
        _, vol, _, _ = measure(GatedKelly(gate=gate, exposure=k, sizer=sizer),
                               start, end, df=df, market=market)
    return k, vol, max_iter


def match(write: bool = True) -> dict:
    """Solve, on inner-validation only, for the exposures that equalize risk.

    Two directions, because which arm is moved is itself a choice and
    moving only one of them would make the answer depend on that choice:

    **up**   the vote holds its shipped exposure (k=1); the e-process is
             levered until its realized volatility matches.
    **down** the e-process holds its shipped exposure (k=1); the vote is
             de-levered until its realized volatility matches.
    """
    df = pd.read_csv(OUT / "frontier.csv")
    val = df[df.split == "inner-validation"]
    frozen = {}
    for mname, market in MARKETS:
        for sizer in ("plain", "conditional"):
            sub = val[(val.market == mname) & (val.sizer == sizer)]
            v1 = float(sub[(sub.gate == "vote") & (sub.k == 1.0)]["vol"].iloc[0])
            e1 = float(sub[(sub.gate == "evidence") & (sub.k == 1.0)]["vol"].iloc[0])
            ek, ev_vol, ei = solve_k("evidence", v1, *VALID, market=market,
                                     sizer=sizer)
            vk, vt_vol, vi = solve_k("vote", e1, *VALID, market=market,
                                     sizer=sizer)
            frozen[f"{mname}/{sizer}"] = {
                "vote_vol_at_k1": v1, "evidence_vol_at_k1": e1,
                "up": {"vote_k": 1.0, "vote_vol": v1,
                       "evidence_k": ek, "evidence_vol": ev_vol,
                       "target_vol": v1, "iterations": ei},
                "down": {"evidence_k": 1.0, "evidence_vol": e1,
                         "vote_k": vk, "vote_vol": vt_vol,
                         "target_vol": e1, "iterations": vi},
            }
            print(f"\n{mname} / sizer={sizer}")
            print(f"  inner-validation realized vol: vote@k=1 {v1:.3f}   "
                  f"evidence@k=1 {e1:.3f}   ratio {v1 / e1:.2f}x")
            print(f"  match UP   to vol {v1:.3f}: vote k=1.000 (vol {v1:.3f}), "
                  f"evidence k={ek:.3f} (vol {ev_vol:.3f}, {ei} backtests)")
            print(f"  match DOWN to vol {e1:.3f}: evidence k=1.000 (vol {e1:.3f}), "
                  f"vote k={vk:.3f} (vol {vt_vol:.3f}, {vi} backtests)")
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"\nwritten: {OUT / 'frozen.json'}")
    return frozen


def _frozen() -> dict:
    return json.loads((OUT / "frozen.json").read_text())


def _pairs(mname: str, sizer: str = "plain"):
    """The frozen matched pairs for one market, as (label, vote_k, evidence_k)."""
    f = _frozen()[f"{mname}/{sizer}"]
    return [("match-up", f["up"]["vote_k"], f["up"]["evidence_k"],
             f["up"]["target_vol"]),
            ("match-down", f["down"]["vote_k"], f["down"]["evidence_k"],
             f["down"]["target_vol"])]


# ------------------------------------------------------------------- causality


def causality() -> None:
    """The strict lookahead probe, by hand — experiments get no CI protection.

    ``tests/test_causality_strict.py`` parametrizes over the *registry*, so
    an unregistered strategy is unprotected. Same two-opposite-tampers
    procedure as R-28: bars after a cut are multiplied by 3 in one copy and
    divided by 3 in the other, and every decision at or before the cut must
    be identical. The column comparison is the part that catches a
    full-series fit, which a truncation test cannot see.
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
    for gate in ("vote", "evidence"):
        for sizer in ("plain", "conditional"):
            def decisions(frame):
                s = GatedKelly(gate=gate, exposure=3.0, sizer=sizer)
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
            pa = GatedKelly(gate=gate, exposure=3.0, sizer=sizer).prepare(up.copy())
            pb = GatedKelly(gate=gate, exposure=3.0, sizer=sizer).prepare(down.copy())
            worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut]
                                               - pb[c].to_numpy()[:cut])))
                        for c in ("target", "conf", "scale"))
            good = not bad and worst < 1e-12
            ok &= good
            print(f"  gate={gate:9s} sizer={sizer:12s} "
                  f"orders {'match' if not bad else f'DIFFER at {bad}'}   "
                  f"max |column difference| before the cut = {worst:.3e}   "
                  f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- holdout


def holdout() -> None:
    """Step 4. Exposures frozen on inner-validation; decision rule in the ledger."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}   (sizer=plain)")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            m, vol, clamp, res = measure(get_strategy(name), *OOS, market=market)
            line(f"  {name}", m, vol, clamp, res)
            rows.append({"market": mname, "pair": "reference", "arm": name,
                         "k": float("nan"), "final": m.final_balance, "vol": vol,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "fills": len(res.fills), "fees": m.fees_paid,
                         "clamp": clamp, "liquidated": m.liquidated})
        for label, vk, ek, tvol in _pairs(mname):
            print(f"  --- {label}: matched to inner-validation vol {tvol:.3f}")
            for gate, k in (("vote", vk), ("evidence", ek)):
                s = GatedKelly(gate=gate, exposure=k, sizer="plain")
                m, vol, clamp, res = measure(s, *OOS, market=market)
                line(f"  {gate} k={k:.3f}", m, vol, clamp, res)
                rows.append({"market": mname, "pair": label, "arm": gate, "k": k,
                             "final": m.final_balance, "vol": vol,
                             "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                             "fills": len(res.fills), "fees": m.fees_paid,
                             "clamp": clamp, "liquidated": m.liquidated})
    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# -------------------------------------------------------------------- interval


def interval() -> None:
    """The decision statistic: paired block-bootstrap on holdout daily returns.

    Identical resamples for both arms of a pair, so the market's own
    variance cancels instead of swamping the gap — the R-29/R-30 method,
    reused rather than reinvented.
    """
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    rows = []
    for mname, market in MARKETS:
        curves = {}
        for label, vk, ek, tvol in _pairs(mname):
            for gate, k in (("vote", vk), ("evidence", ek)):
                res = run_period(GatedKelly(gate=gate, exposure=k, sizer="plain"),
                                 DF, *OOS, market=market, start_balance=1_000.0,
                                 data_label=LABEL)
                curves[(label, gate)] = daily_returns(res.equity).to_numpy()
        n = len(next(iter(curves.values())))
        idx = stationary_bootstrap_indices(n, 30.0, 2_000,
                                           np.random.default_rng(7))
        print(f"\n{mname}: paired evidence − vote on the 2023+ holdout "
              f"({n} daily observations)")
        for label, vk, ek, tvol in _pairs(mname):
            a, b = curves[(label, "evidence")], curves[(label, "vote")]
            for stat_name, stat in (("Δ log growth", total_log_return),
                                    ("Δ max drawdown (pp)", max_drawdown_from_returns)):
                r = paired_bootstrap(a, b, stat, indices=idx)
                mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
                print(f"  {label:11s} {stat_name:22s} {mark} {r.diff.point:>+7.3f} "
                      f"[{r.diff.lo:>+7.3f}, {r.diff.hi:>+7.3f}]  "
                      f"P(>0)={r.p_positive:.2f}")
                rows.append({"market": mname, "pair": label, "stat": stat_name,
                             "evidence": r.stat_a, "vote": r.stat_b,
                             "diff": r.diff.point, "lo": r.diff.lo,
                             "hi": r.diff.hi, "p_positive": r.p_positive,
                             "significant": r.significant})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intervals.csv", index=False)
    print(f"\nwritten: {OUT / 'intervals.csv'}")


# ------------------------------------------------------------------------- eth


def eth() -> None:
    """Pre-registered falsification: does the matched-risk ORDERING replicate?

    Same venue (Bitfinex), same window as R-17 and R-28's test, only the
    asset varies. The exposures are re-matched on ETH's own volatility —
    matching is a property of the risk axis, not a fitted parameter — and
    the question is whether the same gate wins.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            v1 = measure(GatedKelly(gate="vote"), None, None, df=df, market=market)[1]
            e1 = measure(GatedKelly(gate="evidence"), None, None, df=df,
                         market=market)[1]
            ek, _, _ = solve_k("evidence", v1, None, None, df=df, market=market)
            vk, _, _ = solve_k("vote", e1, None, None, df=df, market=market)
            for label, tvol, vote_k, ev_k in (("match-up", v1, 1.0, ek),
                                              ("match-down", e1, vk, 1.0)):
                print(f"    --- {label}: matched to vol {tvol:.3f}")
                for gate, k in (("vote", vote_k), ("evidence", ev_k)):
                    if not np.isfinite(k):
                        print(f"      {gate:9s} no exposure reaches this volatility")
                        continue
                    line(f"    {gate} k={k:.3f}",
                         *measure(GatedKelly(gate=gate, exposure=k), None, None,
                                  df=df, market=market))


# ----------------------------------------------------------------------- costs


def costs() -> None:
    """Step 4 cost checks: the real fee tier on spot, funding on futures."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ spot, at both taker tiers (sizer=plain):")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            line(f"    {name}", *measure(get_strategy(name), *OOS, market=market))
        for lbl, vk, ek, tvol in _pairs("spot"):
            for gate, k in (("vote", vk), ("evidence", ek)):
                line(f"    {lbl} {gate} k={k:.2f}",
                     *measure(GatedKelly(gate=gate, exposure=k), *OOS, market=market))

    real = load_funding(ROOT / "data")
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ futures 5x with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr after):")
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4"))]
    for lbl, vk, ek, tvol in _pairs("futures"):
        for gate, k in (("vote", vk), ("evidence", ek)):
            contenders.append((f"{lbl} {gate} k={k:.2f}",
                               GatedKelly(gate=gate, exposure=k)))
    lo = int(DF.index.searchsorted(OOS[0]))
    for name, strat in contenders:
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=blended, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:30s} final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>7,.0f}"
              f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------- windows


def windows(trials: int = 40, seed: int = 42) -> None:
    """Path sensitivity, the R-19 design: identical random windows, both arms."""
    from tradebot.metrics import max_drawdown_pct

    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4"))]
    for mname, _ in MARKETS:
        pass
    # One matched pair per direction, taken from the SPOT freeze so a single
    # set of strategies is scored on both markets, as R-19 does.
    for lbl, vk, ek, tvol in _pairs("spot"):
        contenders.append((f"{lbl} vote", GatedKelly(gate="vote", exposure=vk)))
        contenders.append((f"{lbl} evidence", GatedKelly(gate="evidence", exposure=ek)))

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
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "windows.csv", index=False)

    print(f"\n{trials} random windows (90-730 days), identical across strategies:\n")
    for mname, _ in MARKETS:
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name, _ in contenders:
            g = sub[sub.strategy == name].set_index("trial")
            print(f"    {name:22s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"beat hold {(g['return_pct'] > bench).mean():>5.0%}  "
                  f"liq {g.liquidated.mean():>4.0%}")
        for lbl, _, _, _ in _pairs("spot"):
            a = sub[sub.strategy == f"{lbl} evidence"].set_index("trial")
            b = sub[sub.strategy == f"{lbl} vote"].set_index("trial")
            d_ret = (a.return_pct - b.return_pct).dropna()
            d_dd = (a.max_dd_pct - b.max_dd_pct).dropna()
            print(f"    paired {lbl} (evidence − vote): "
                  f"return median {d_ret.median():+.1f}pp, evidence higher in "
                  f"{(d_ret > 0).mean():.0%};  DD median {d_dd.median():+.1f}pp, "
                  f"evidence deeper in {(d_dd > 0).mean():.0%}")
        print()


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"parity": parity, "frontier": frontier, "match": match,
            "causality": causality, "holdout": holdout, "interval": interval,
            "eth": eth, "costs": costs, "windows": windows}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_matched_risk.py [{'|'.join(cmds)}]")
