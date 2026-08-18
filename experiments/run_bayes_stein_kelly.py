#!/usr/bin/env python
"""Driver: Bayes-Stein shrinkage Kelly gate vs `vote` and `evidence`, matched risk.

Splits follow ROUTINE.md step 3, identical convention to
``experiments/run_matched_risk.py``::

    inner-train       2017-01-01 -> 2020-12-31   fit, sweep, iterate
    inner-validation  2021-01-01 -> 2022-12-31   select the frozen span,
                                                 solve the matched exposures
    holdout           2023-01-01 ->              step 4 only, pre-registered

Usage::

    python experiments/run_bayes_stein_kelly.py frontier    # step 3 sweep
    python experiments/run_bayes_stein_kelly.py plateau     # P4 neighbourhood
    python experiments/run_bayes_stein_kelly.py match       # solve matched k
    python experiments/run_bayes_stein_kelly.py causality   # by-hand lookahead probe
    python experiments/run_bayes_stein_kelly.py holdout     # step 4, frozen
    python experiments/run_bayes_stein_kelly.py interval    # paired bootstrap
    python experiments/run_bayes_stein_kelly.py eth         # falsification test
    python experiments/run_bayes_stein_kelly.py costs       # fee tier + funding
    python experiments/run_bayes_stein_kelly.py windows     # 40-window path check
    python experiments/run_bayes_stein_kelly.py memory      # gate-persistence probe
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

from experiments.bayes_stein_kelly import BayesSteinKelly  # noqa: E402
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

OUT = ROOT / "reports" / "bayes_stein_kelly"

N_EVALUATED = 0     # step-3 frontier configurations, for deflated Sharpe
N_PLATEAU = 0        # separate P4 neighbourhood configurations, counted apart

# Candidate shrinkage-window lengths (days). 20d is the a-priori favourite —
# it coincides with R-07's independently-found robust 18-28 day anchor
# region and matches the vote's fastest anchor and the evidence gate's
# default half-life, so all three gates share a timescale and a frontier
# difference cannot be blamed on "different lookback" alone. 10d and 60d
# bracket it as genuinely different mechanisms of window length.
SPAN_GRID = (10.0, 20.0, 60.0)
FROZEN_SPAN = 20.0   # overwritten by frontier()'s printed selection if it disagrees

# Same exposure grid matched_risk.py traces its frontier on.
K_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def gate_factory(name: str, k: float, *, span: float = FROZEN_SPAN):
    if name == "bayes_stein":
        return BayesSteinKelly(gate="bayes_stein", exposure=k, sizer="plain",
                               drift_span_days=span)
    return GatedKelly(gate=name, exposure=k, sizer="plain")


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0,
            count=None):
    """One backtest -> (metrics, realized vol, clamp fraction, result).

    ``count`` is ``"frontier"``, ``"plateau"`` or ``None`` — which counter
    (if any) this call adds to. Matching-solver and reference backtests
    pass ``None``: they select on a criterion (equalize volatility) or
    reproduce an already-registered strategy, neither of which is a
    trial searched for performance.
    """
    global N_EVALUATED, N_PLATEAU
    if count == "frontier":
        N_EVALUATED += 1
    elif count == "plateau":
        N_PLATEAU += 1
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
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
        for span in SPAN_GRID:
            for k in K_GRID:
                s = BayesSteinKelly(gate="bayes_stein", exposure=k, sizer="plain",
                                    drift_span_days=span)
                # A configuration is (span, k). A second market/split scores
                # it again, not a second trial — the R-28/R-31 convention.
                m, vol, clamp, res = measure(
                    s, start, end, market=market,
                    count="frontier" if (split == "inner-train" and mname == "spot")
                    else None)
                rows.append({
                    "split": split, "market": mname, "span_days": span, "k": k,
                    "final": m.final_balance, "vol": vol,
                    "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                    "fills": len(res.fills), "fees": m.fees_paid,
                    "clamp": clamp, "liquidated": m.liquidated,
                    "mean_conf": float(res.df["conf"].mean()),
                })
    return rows


def frontier() -> None:
    """Step 3. Trace return-vs-risk for the bayes_stein gate, inner splits only."""
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
            print(f"  {'span':>5s} {'k':>5s} {'vol':>6s} {'final':>11s} "
                  f"{'DD':>6s} {'sharpe':>7s} {'fills':>6s} {'conf':>6s} {'clamp':>6s}")
            for _, r in sub.iterrows():
                print(f"  {r.span_days:>5g} {r.k:>5g} {r.vol:>6.3f} "
                      f"${r.final:>10,.0f} {r.max_dd:>5.1f}% {r.sharpe:>7.2f} "
                      f"{r.fills:>6.0f} {r.mean_conf:>6.3f} {r.clamp:>5.1%}")
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'frontier.csv'}")


# --------------------------------------------------------------------- plateau


def plateau(span: float = FROZEN_SPAN, k: float | None = None) -> None:
    """P4. One-knob-at-a-time neighbourhood of the frozen config, inner-validation.

    Varies drift_span (days) and z_clip independently around the frozen
    point. Not a performance search for a better point — the frozen span
    and k come from ``frontier()``/``match()`` — but every backtest here is
    still a configuration evaluated in step 3 (R-28's convention: the
    15-point neighbourhood in R-28 was counted too), so it is tallied
    separately and added to the total.
    """
    if k is None:
        try:
            k = _frozen()["spot"]["vs_vote"]["up"]["bs_k"]
        except FileNotFoundError:
            k = 1.0
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"P4 neighbourhood around span={span}d, k={k:.3f} (spot, inner-validation)")
    print(f"  {'span':>6s} {'z_clip':>6s} {'vol':>6s} {'final':>11s} "
          f"{'DD':>6s} {'sharpe':>7s}")
    for sp in (span - 5, span - 2, span, span + 5, span + 10, span + 40):
        if sp <= 0:
            continue
        for zc in (5.0, 10.0, 20.0):
            s = BayesSteinKelly(gate="bayes_stein", exposure=k, sizer="plain",
                                drift_span_days=sp, z_clip=zc)
            m, vol, clamp, res = measure(s, *VALID, market=SPOT, count="plateau")
            rows.append({"span_days": sp, "z_clip": zc, "k": k, "final": m.final_balance,
                         "vol": vol, "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe})
            print(f"  {sp:>6.0f} {zc:>6.1f} {vol:>6.3f} ${m.final_balance:>10,.0f} "
                  f"{m.max_drawdown_pct:>5.1f}% {m.sharpe:>7.2f}")
    pd.DataFrame(rows).to_csv(OUT / "plateau.csv", index=False)
    print(f"\nplateau configurations evaluated (separate tally): {N_PLATEAU}")
    print(f"written: {OUT / 'plateau.csv'}")


# ----------------------------------------------------------------------- match


def solve_k(gate: str, target: float, start, end, *, market=SPOT,
            span: float = FROZEN_SPAN, df=None, tol: float = 0.02,
            max_iter: int = 8) -> tuple[float, float, int]:
    """Find the exposure whose realized volatility equals ``target``.

    Identical method to ``run_matched_risk.solve_k``: a direct secant
    search rather than interpolating the frontier grid, because the grid's
    (k, vol) pairs are not monotone at the bottom of the range where a
    low-exposure arm trades so rarely its realized volatility is dominated
    by a handful of fills.
    """
    k = 1.0
    _, vol, _, _ = measure(gate_factory(gate, k, span=span), start, end, df=df,
                           market=market)
    for it in range(1, max_iter + 1):
        if vol <= 0 or not np.isfinite(vol):
            return float("nan"), vol, it
        if abs(vol - target) <= tol * target:
            return k, vol, it
        k = float(np.clip(k * (target / vol), 1e-3, 1e3))
        _, vol, _, _ = measure(gate_factory(gate, k, span=span), start, end, df=df,
                               market=market)
    return k, vol, max_iter


def match(span: float = FROZEN_SPAN, write: bool = True) -> dict:
    """Solve, on inner-validation only, for exposures that equalize risk.

    Two reference arms, each matched in both directions (the ``vote`` gate
    is the primary comparison the assignment asks for; ``evidence`` is the
    secondary one, "ideally" per the brief):

    **vs vote**
      up    vote holds k=1; bayes_stein levered to match vote's vol.
      down  bayes_stein holds k=1; vote de-levered to match bayes_stein's vol.
    **vs evidence**
      up    evidence holds k=1; bayes_stein levered to match evidence's vol.
      down  bayes_stein holds k=1; evidence de-levered to match bayes_stein's vol.
    """
    frozen = {}
    for mname, market in MARKETS:
        vote1 = measure(gate_factory("vote", 1.0), *VALID, market=market)[1]
        ev1 = measure(gate_factory("evidence", 1.0), *VALID, market=market)[1]
        bs1 = measure(gate_factory("bayes_stein", 1.0, span=span), *VALID,
                     market=market)[1]

        bs_up_k, bs_up_vol, i1 = solve_k("bayes_stein", vote1, *VALID, market=market,
                                        span=span)
        vote_dn_k, vote_dn_vol, i2 = solve_k("vote", bs1, *VALID, market=market)

        bs_up2_k, bs_up2_vol, i3 = solve_k("bayes_stein", ev1, *VALID, market=market,
                                          span=span)
        ev_dn_k, ev_dn_vol, i4 = solve_k("evidence", bs1, *VALID, market=market)

        frozen[mname] = {
            "vote_vol_at_k1": vote1, "evidence_vol_at_k1": ev1,
            "bayes_stein_vol_at_k1": bs1,
            "vs_vote": {
                "up": {"vote_k": 1.0, "vote_vol": vote1,
                       "bs_k": bs_up_k, "bs_vol": bs_up_vol,
                       "target_vol": vote1, "iterations": i1},
                "down": {"bs_k": 1.0, "bs_vol": bs1,
                         "vote_k": vote_dn_k, "vote_vol": vote_dn_vol,
                         "target_vol": bs1, "iterations": i2},
            },
            "vs_evidence": {
                "up": {"evidence_k": 1.0, "evidence_vol": ev1,
                       "bs_k": bs_up2_k, "bs_vol": bs_up2_vol,
                       "target_vol": ev1, "iterations": i3},
                "down": {"bs_k": 1.0, "bs_vol": bs1,
                         "evidence_k": ev_dn_k, "evidence_vol": ev_dn_vol,
                         "target_vol": bs1, "iterations": i4},
            },
        }
        print(f"\n{mname}  (span={span}d)")
        print(f"  inner-validation vol at k=1: vote {vote1:.3f}  evidence {ev1:.3f}  "
              f"bayes_stein {bs1:.3f}")
        print(f"  vs vote      up   -> target {vote1:.3f}: bayes_stein k={bs_up_k:.3f} "
              f"(vol {bs_up_vol:.3f}, {i1} backtests)")
        print(f"  vs vote      down -> target {bs1:.3f}: vote k={vote_dn_k:.3f} "
              f"(vol {vote_dn_vol:.3f}, {i2} backtests)")
        print(f"  vs evidence  up   -> target {ev1:.3f}: bayes_stein k={bs_up2_k:.3f} "
              f"(vol {bs_up2_vol:.3f}, {i3} backtests)")
        print(f"  vs evidence  down -> target {bs1:.3f}: evidence k={ev_dn_k:.3f} "
              f"(vol {ev_dn_vol:.3f}, {i4} backtests)")
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"\nwritten: {OUT / 'frozen.json'}")
    return frozen


def _frozen() -> dict:
    return json.loads((OUT / "frozen.json").read_text())


def _pairs(mname: str):
    """Frozen matched pairs for one market: list of (label, ref_gate, ref_k, bs_k, target_vol)."""
    f = _frozen()[mname]
    out = []
    for ref in ("vote", "evidence"):
        key = f"vs_{ref}"
        up, dn = f[key]["up"], f[key]["down"]
        out.append((f"{ref}/match-up", ref, 1.0, up["bs_k"], up["target_vol"]))
        out.append((f"{ref}/match-down", ref, dn[f"{ref}_k"], 1.0, dn["target_vol"]))
    return out


# ------------------------------------------------------------------- causality


def causality() -> None:
    """Strict by-hand lookahead probe — experiments get no CI protection.

    Same two-opposite-tampers procedure as ``run_matched_risk.causality``:
    bars after a cut are multiplied by 3 in one copy and divided by 3 in
    the other, and every decision at or before the cut must be identical.
    The column comparison (target/conf/scale) is what catches a
    full-series fit a truncation test cannot see.
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
    for span in SPAN_GRID:
        def decisions(frame, span=span):
            s = BayesSteinKelly(gate="bayes_stein", exposure=3.0, sizer="plain",
                                drift_span_days=span)
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
        pa = BayesSteinKelly(gate="bayes_stein", exposure=3.0, sizer="plain",
                             drift_span_days=span).prepare(up.copy())
        pb = BayesSteinKelly(gate="bayes_stein", exposure=3.0, sizer="plain",
                             drift_span_days=span).prepare(down.copy())
        worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut]
                                           - pb[c].to_numpy()[:cut])))
                    for c in ("target", "conf", "scale"))
        good = not bad and worst < 1e-12
        ok &= good
        print(f"  span={span:>5g}d "
              f"orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# --------------------------------------------------------------------- holdout


def validity(vol_a: float, vol_b: float, clamp_a: float, clamp_b: float) -> str:
    """R-31's validity gate, applied verbatim: vol within 20%, both clamps < 1%."""
    if vol_a <= 0 or vol_b <= 0:
        return "VOID (degenerate vol)"
    gap = abs(vol_a - vol_b) / max(vol_a, vol_b)
    if gap > 0.20:
        return f"VOID (vol gap {gap:.1%})"
    if clamp_a >= 0.01 or clamp_b >= 0.01:
        return f"VOID (clamp {clamp_a:.1%}/{clamp_b:.1%})"
    return "VALID"


def holdout() -> None:
    """Step 4. Exposures frozen on inner-validation; decision rule in the report."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            m, vol, clamp, res = measure(get_strategy(name), *OOS, market=market)
            line(f"  {name}", m, vol, clamp, res)
            rows.append({"market": mname, "pair": "reference", "arm": name,
                         "k": float("nan"), "final": m.final_balance, "vol": vol,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "fills": len(res.fills), "fees": m.fees_paid,
                         "clamp": clamp, "liquidated": m.liquidated})
        for label, ref, ref_k, bs_k, tvol in _pairs(mname):
            print(f"  --- {label}: matched to inner-validation vol {tvol:.3f}")
            m1, v1, c1, r1 = measure(gate_factory(ref, ref_k), *OOS, market=market)
            line(f"  {ref} k={ref_k:.3f}", m1, v1, c1, r1)
            m2, v2, c2, r2 = measure(gate_factory("bayes_stein", bs_k), *OOS,
                                     market=market)
            line(f"  bayes_stein k={bs_k:.3f}", m2, v2, c2, r2)
            verdict = validity(v1, v2, c1, c2)
            print(f"    validity gate: {verdict}")
            for arm, kk, m, v, c, res in ((ref, ref_k, m1, v1, c1, r1),
                                          ("bayes_stein", bs_k, m2, v2, c2, r2)):
                rows.append({"market": mname, "pair": label, "ref_gate": ref,
                            "arm": arm, "k": kk, "final": m.final_balance, "vol": v,
                            "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                            "fills": len(res.fills), "fees": m.fees_paid,
                            "clamp": c, "liquidated": m.liquidated,
                            "validity": verdict})
    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# -------------------------------------------------------------------- interval


def interval() -> None:
    """Decision statistic: paired block-bootstrap on holdout daily returns.

    Reuses ``tradebot.inference`` directly (assignment step 5) rather than
    a registry-backed harness: identical resamples for both arms of a
    pair, so the market's own variance cancels instead of swamping the
    gap (the R-29/R-30/R-31 method).
    """
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    rows = []
    for mname, market in MARKETS:
        curves = {}
        for label, ref, ref_k, bs_k, tvol in _pairs(mname):
            for tag, gate, k in ((f"{label}/ref", ref, ref_k),
                                (f"{label}/bs", "bayes_stein", bs_k)):
                res = run_period(gate_factory(gate, k), DF, *OOS, market=market,
                                 start_balance=1_000.0, data_label=LABEL)
                curves[tag] = daily_returns(res.equity).to_numpy()
        n = len(next(iter(curves.values())))
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
        print(f"\n{mname}: paired bayes_stein - reference on the 2023+ holdout "
              f"({n} daily observations)")
        for label, ref, ref_k, bs_k, tvol in _pairs(mname):
            a, b = curves[f"{label}/bs"], curves[f"{label}/ref"]
            for stat_name, stat in (("Δ log growth", total_log_return),
                                    ("Δ max drawdown (pp)", max_drawdown_from_returns)):
                r = paired_bootstrap(a, b, stat, indices=idx)
                mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
                print(f"  {label:18s} {stat_name:22s} {mark} {r.diff.point:>+7.3f} "
                      f"[{r.diff.lo:>+7.3f}, {r.diff.hi:>+7.3f}]  P(>0)={r.p_positive:.2f}")
                rows.append({"market": mname, "pair": label, "stat": stat_name,
                             "bayes_stein": r.stat_a, "reference": r.stat_b,
                             "diff": r.diff.point, "lo": r.diff.lo, "hi": r.diff.hi,
                             "p_positive": r.p_positive, "significant": r.significant})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intervals.csv", index=False)
    print(f"\nwritten: {OUT / 'intervals.csv'}")


# ------------------------------------------------------------------------- eth


def eth() -> None:
    """Pre-registered falsification: does the matched-risk ORDERING replicate?

    Same venue (Bitfinex), same window as R-17/R-28/R-31's test, only the
    asset varies; exposures re-matched on ETH's own volatility.
    """
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            vote1 = measure(gate_factory("vote", 1.0), None, None, df=df,
                            market=market)[1]
            ev1 = measure(gate_factory("evidence", 1.0), None, None, df=df,
                          market=market)[1]
            bs1 = measure(gate_factory("bayes_stein", 1.0), None, None, df=df,
                         market=market)[1]
            bs_up_k, _, _ = solve_k("bayes_stein", vote1, None, None, df=df,
                                    market=market)
            vote_dn_k, _, _ = solve_k("vote", bs1, None, None, df=df, market=market)
            bs_up2_k, _, _ = solve_k("bayes_stein", ev1, None, None, df=df,
                                     market=market)
            ev_dn_k, _, _ = solve_k("evidence", bs1, None, None, df=df, market=market)
            for label, tvol, gates in (
                ("vs vote match-up", vote1, (("vote", 1.0), ("bayes_stein", bs_up_k))),
                ("vs vote match-down", bs1, (("vote", vote_dn_k), ("bayes_stein", 1.0))),
                ("vs evidence match-up", ev1,
                 (("evidence", 1.0), ("bayes_stein", bs_up2_k))),
                ("vs evidence match-down", bs1,
                 (("evidence", ev_dn_k), ("bayes_stein", 1.0))),
            ):
                print(f"    --- {label}: matched to vol {tvol:.3f}")
                for gate, k in gates:
                    if not np.isfinite(k):
                        print(f"      {gate:12s} no exposure reaches this volatility")
                        continue
                    line(f"    {gate} k={k:.3f}",
                         *measure(gate_factory(gate, k), None, None, df=df,
                                 market=market))


# ----------------------------------------------------------------------- costs


def costs() -> None:
    """Step 4 cost checks: the real fee tier on spot, funding on futures."""
    from tradebot.data import load_funding

    print("HOLDOUT 2023+ spot, at both taker tiers:")
    for tier, tag in ((0.001, "0.10% (table assumption)"),
                      (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {tag}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            line(f"    {name}", *measure(get_strategy(name), *OOS, market=market))
        for label, ref, ref_k, bs_k, tvol in _pairs("spot"):
            line(f"    {label} {ref} k={ref_k:.2f}",
                 *measure(gate_factory(ref, ref_k), *OOS, market=market))
            line(f"    {label} bayes_stein k={bs_k:.2f}",
                 *measure(gate_factory("bayes_stein", bs_k), *OOS, market=market))

    real = load_funding(ROOT / "data")
    grid = pd.date_range(DF.index[0].ceil("8h"), DF.index[-1], freq="8h", tz="UTC")
    filler = pd.Series(float(real.mean()), index=grid)
    blended = pd.concat([filler[~filler.index.isin(real.index)], real]).sort_index()
    print(f"\nHOLDOUT 2023+ futures 5x with funding CHARGED (real through "
          f"{real.index[-1]:%Y-%m}, mean {real.mean() * 3 * 365.25:+.1%}/yr after):")
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4"))]
    for label, ref, ref_k, bs_k, tvol in _pairs("futures"):
        contenders.append((f"{label} {ref} k={ref_k:.2f}", gate_factory(ref, ref_k)))
        contenders.append((f"{label} bayes_stein k={bs_k:.2f}",
                           gate_factory("bayes_stein", bs_k)))
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
    """Path sensitivity, the R-19/R-31 design: identical random windows, both arms."""
    from tradebot.metrics import max_drawdown_pct

    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4"))]
    for label, ref, ref_k, bs_k, tvol in _pairs("spot"):
        contenders.append((f"{label} {ref}", gate_factory(ref, ref_k)))
        contenders.append((f"{label} bayes_stein", gate_factory("bayes_stein", bs_k)))

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
        for label, ref, ref_k, bs_k, tvol in _pairs("spot"):
            a = sub[sub.strategy == f"{label} bayes_stein"].set_index("trial")
            b = sub[sub.strategy == f"{label} {ref}"].set_index("trial")
            d_ret = (a.return_pct - b.return_pct).dropna()
            d_dd = (a.max_dd_pct - b.max_dd_pct).dropna()
            print(f"    paired {label} (bayes_stein - {ref}): "
                  f"return median {d_ret.median():+.1f}pp, bs higher in "
                  f"{(d_ret > 0).mean():.0%};  DD median {d_dd.median():+.1f}pp, "
                  f"bs deeper in {(d_dd > 0).mean():.0%}")
        print()


# ---------------------------------------------------------------------- memory


def memory() -> None:
    """Direct check of pre-registered failure mode (d): does the gate have memory?

    Computes, on the full BTC spot series, the lag-1-day autocorrelation of
    each gate's ``conf`` series and the median number of consecutive days a
    gate stays "mostly open" (conf > 0.5) once it opens — the operational
    definition of "re-opens/closes on the timescale of one drift_span
    window" vs "accumulates like a multi-year e-process". Not a
    performance measurement, so not counted toward the trials tally.
    """
    for name, strat in (("vote", gate_factory("vote", 1.0)),
                        ("evidence", gate_factory("evidence", 1.0)),
                        ("bayes_stein (20d)", gate_factory("bayes_stein", 1.0)),
                        ("bayes_stein (60d)",
                         gate_factory("bayes_stein", 1.0, span=60.0))):
        res = run_period(strat, DF, "2017-01-01", None, market=SPOT,
                         start_balance=1_000.0, data_label=LABEL)
        conf = res.df["conf"].to_numpy()
        daily = pd.Series(conf, index=res.df.index).resample("1D").mean().dropna()
        ac1 = float(daily.autocorr(lag=1))
        open_ = (daily > 0.5).to_numpy()
        # run lengths of consecutive open days
        runs, cur = [], 0
        for o in open_:
            if o:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        med_run = float(np.median(runs)) if runs else 0.0
        print(f"  {name:20s} daily-mean-conf 1-day autocorr={ac1:.4f}   "
              f"median 'open' run length={med_run:>6.0f} days   "
              f"n_runs={len(runs):>4d}   mean_conf={float(daily.mean()):.3f}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"frontier": frontier, "plateau": plateau, "match": match,
            "causality": causality, "holdout": holdout, "interval": interval,
            "eth": eth, "costs": costs, "windows": windows, "memory": memory}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_bayes_stein_kelly.py [{'|'.join(cmds)}]")
