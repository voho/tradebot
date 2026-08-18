#!/usr/bin/env python
"""Driver for backlog B-05 -- funding as a gate on kelly_regime_v4.

Splits follow ROUTINE.md step 3, anchored to where the real funding file
actually starts (2020-01-01), not the dataset's 2017 start::

    inner-train       2020-01-01 -> 2020-12-31   fit, sanity-check
    inner-validation  2021-01-01 -> 2022-12-31   select between variants
    holdout           2023-01-01 ->              step 4 only, pre-registered

Funding itself only covers 2020-01-01 03:00 UTC -> 2023-12-31 19:00 UTC,
confirmed by loading the file (see ``rates()``). The gate is a no-op by
construction outside that window (``experiments/funding_gate.py``,
``_funding_state``), so most of the 2024+ tail of the holdout is
``kelly_regime_v4`` unmodified -- ``holdout()`` reports exactly what
fraction of holdout bars actually had the gate active.

Usage::

    python experiments/run_funding_gate.py rates       # confirm funding coverage
    python experiments/run_funding_gate.py sweep        # step 3, inner splits
    python experiments/run_funding_gate.py causality    # by-hand lookahead probe
    python experiments/run_funding_gate.py holdout      # step 4, frozen config
    python experiments/run_funding_gate.py interval     # paired bootstrap, holdout
    python experiments/run_funding_gate.py windows      # falsification test
    python experiments/run_funding_gate.py costs        # 0.40% fee tier + funding charged
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

from experiments.funding_gate import FundingGateV4  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
FUNDING = load_funding(ROOT / "data")
if FUNDING is None:
    raise SystemExit("no funding data committed; see docs/VALIDATION.md")

SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

TRAIN = ("2020-01-01", "2020-12-31")     # inner-train: funding-file start -> end 2020
VALID = ("2021-01-01", "2022-12-31")     # inner-validation: 2021 top + 2022 bear
OOS = ("2023-01-01", None)               # holdout: project standard OOS_START

OUT = ROOT / "reports" / "funding_gate"

N_EVALUATED = 0  # distinct configurations searched in step 3


def make(**kw) -> FundingGateV4:
    s = FundingGateV4(**kw)
    s._funding = FUNDING
    return s


def measure(strategy, start, end, *, market=SPOT, balance=1_000.0, count=False):
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    result = run_period(strategy, DF, start, end, market=market,
                        start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, result


def line(tag, m, result):
    print(f"  {tag:46s} final=${m.final_balance:>11,.0f} "
          f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
          f"fills={len(result.fills):>5d} fees=${m.fees_paid:>8,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------- rates


def rates() -> None:
    """Confirm the funding coverage this whole design leans on."""
    print(f"{len(FUNDING):,} settlements  {FUNDING.index[0]} -> {FUNDING.index[-1]}")
    print(f"{len(DF):,} price bars  {DF.index[0]} -> {DF.index[-1]}  (data: {LABEL})")
    covered_days = (FUNDING.index[-1] - FUNDING.index[0]).days
    total_days = (DF.index[-1] - DF.index[0]).days
    print(f"funding covers {covered_days} of {total_days} days "
          f"({covered_days / total_days:.1%} of the dataset span)")


# --------------------------------------------------------------------- sweep


# Main grid: 2 thresholds x 2 floors x 3 overrides = 12 distinct configs.
THRESHOLDS = (0.90, 0.95)
FLOORS = (0.0, 0.35)
OVERRIDES = (None, 0.05, 0.10)
# Window sensitivity, evaluated only for the eventually-selected (threshold,
# floor, override) triple, on inner-validation -- the P4 plateau check.
WINDOW_DAYS = (90.0, 180.0, 365.0)   # 180 is the a-priori default used above
# Selected on inner-validation (sweep): threshold=0.90, floor=0.0 (hard gate),
# override=None. The momentum override HURT inner-validation (2021 top ->
# 2022 bear) despite helping inner-train (2020) -- see funding_gate_report.md.
SELECTED_FOR_WINDOW_CHECK = (0.90, 0.0, None)


def _grid_rows(start, end, split, count_split):
    rows = []
    for threshold in THRESHOLDS:
        for floor in FLOORS:
            for override in OVERRIDES:
                s = make(pctile_threshold=threshold, floor=floor,
                         momentum_override=override)
                m, res = measure(s, start, end,
                                 count=(count_split and split == "inner-train"))
                active_frac = float(res.df["funding_active"].mean())
                gated_frac = float((res.df["funding_scale"] < 1.0).mean())
                rows.append({
                    "split": split, "threshold": threshold, "floor": floor,
                    "override": override, "final": m.final_balance,
                    "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                    "fills": len(res.fills), "fees": m.fees_paid,
                    "active_frac": active_frac, "gated_frac": gated_frac,
                    "liquidated": m.liquidated,
                })
    return rows


def _window_rows(start, end, split):
    rows = []
    for window_days in WINDOW_DAYS:
        threshold, floor, override = SELECTED_FOR_WINDOW_CHECK
        s = make(pctile_threshold=threshold, floor=floor,
                 momentum_override=override, window_days=window_days)
        m, res = measure(s, start, end, count=(window_days != 180.0))
        rows.append({
            "split": split, "window_days": window_days,
            "final": m.final_balance, "max_dd": m.max_drawdown_pct,
            "sharpe": m.sharpe, "fills": len(res.fills),
        })
    return rows


def sweep() -> None:
    """Step 3. Main grid on inner-train + inner-validation, spot only."""
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += _grid_rows(*TRAIN, "inner-train", count_split=True)
    rows += _grid_rows(*VALID, "inner-validation", count_split=False)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid.csv", index=False)

    print("=== main grid: threshold x floor x override (spot) ===")
    for split in ("inner-train", "inner-validation"):
        sub = grid[grid.split == split].copy()
        sub["_ovr_sort"] = sub["override"].fillna(-1.0)
        sub = sub.sort_values(["threshold", "floor", "_ovr_sort"])
        print(f"\n{split}")
        print(f"  {'thr':>5s} {'floor':>5s} {'ovr':>6s} {'final':>11s} "
              f"{'DD':>6s} {'sharpe':>7s} {'fills':>6s} {'active%':>8s} "
              f"{'gated%':>7s}")
        for _, r in sub.iterrows():
            ovr = "none" if pd.isna(r.override) else f"{r.override:.2f}"
            print(f"  {r.threshold:>5.2f} {r.floor:>5.2f} {ovr:>6s} "
                  f"${r.final:>10,.0f} {r.max_dd:>5.1f}% {r.sharpe:>7.2f} "
                  f"{r.fills:>6.0f} {r.active_frac:>7.1%} {r.gated_frac:>6.1%}")

    m, res = measure(get_strategy("kelly_regime_v4"), *TRAIN)
    line("baseline kelly_regime_v4 (inner-train)", m, res)
    m, res = measure(get_strategy("kelly_regime_v4"), *VALID)
    line("baseline kelly_regime_v4 (inner-validation)", m, res)
    m, res = measure(get_strategy("buy_and_hold"), *TRAIN)
    line("buy_and_hold (inner-train)", m, res)
    m, res = measure(get_strategy("buy_and_hold"), *VALID)
    line("buy_and_hold (inner-validation)", m, res)

    print(f"\nconfigurations evaluated so far (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'grid.csv'}")


def window_check() -> None:
    """P4 plateau check: window_days sensitivity for the selected triple."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += _window_rows(*TRAIN, "inner-train")
    rows += _window_rows(*VALID, "inner-validation")
    wdf = pd.DataFrame(rows)
    wdf.to_csv(OUT / "window_check.csv", index=False)
    print("=== window_days sensitivity for the selected (threshold, floor, override) ===")
    for split in ("inner-train", "inner-validation"):
        sub = wdf[wdf.split == split]
        print(f"\n{split}")
        for _, r in sub.iterrows():
            print(f"  window={r.window_days:>5.0f}d  final=${r.final:>10,.0f} "
                  f"DD={r.max_dd:>5.1f}% sharpe={r.sharpe:>6.2f} fills={r.fills:>5.0f}")
    print(f"\nconfigurations evaluated so far (distinct, counted once): {N_EVALUATED}")
    print(f"written: {OUT / 'window_check.csv'}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """Two-opposite-tampers probe, by hand -- see experiments/matched_risk.py.

    Uses a slice that actually falls inside the funding-covered window
    (2020-01-01 -> 2023-12-31); the tail of the dataset (2024+) has the
    gate permanently inactive by design, which would make this check
    trivially pass without exercising the gate logic at all.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    lo = int(DF.index.searchsorted("2021-01-01"))
    hi = int(DF.index.searchsorted("2022-06-01"))
    df = DF.iloc[lo:hi].copy()
    cut = len(df) // 2
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def decisions(frame):
        s = make(pctile_threshold=0.90, floor=0.0, momentum_override=None)
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    pa = make(pctile_threshold=0.90, floor=0.0, momentum_override=None).prepare(up.copy())
    pb = make(pctile_threshold=0.90, floor=0.0, momentum_override=None).prepare(down.copy())
    worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut].astype(float)
                                       - pb[c].to_numpy()[:cut].astype(float))))
                for c in ("target", "funding_scale", "funding_pctile"))
    good = not bad and worst < 1e-9
    print(f"slice: {df.index[0]} -> {df.index[-1]}  ({len(df):,} bars), cut at "
          f"{df.index[cut]}")
    print(f"orders {'match' if not bad else f'DIFFER at bars {bad}'}")
    print(f"max |column difference| before the cut "
          f"(target, funding_scale, funding_pctile) = {worst:.3e}")
    print("PASS - no decision at or before the cut moves" if good else "FAIL")

    # Sanity: the gate must actually have fired somewhere in this slice,
    # or the check above proves nothing about the funding logic.
    fired = float(pa["funding_scale"].to_numpy()[:cut].astype(float).min())
    print(f"\nsanity: min funding_scale before the cut = {fired:.2f} "
          f"({'gate fired at least once - check is meaningful' if fired < 1.0 else 'gate never fired in this slice - INVALID CHECK'})")


# --------------------------------------------------------------------- holdout


def holdout(frozen: dict | None = None) -> None:
    """Step 4. Frozen config vs kelly_regime_v4 vs buy_and_hold, both markets."""
    frozen = frozen or dict(pctile_threshold=0.90, floor=0.0, momentum_override=None)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    lo = int(DF.index.searchsorted(OOS[0]))
    active_frac = None
    for mname, market in MARKETS:
        print(f"\nHOLDOUT 2023-01-01 -> / {mname}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            m, res = measure(get_strategy(name), *OOS, market=market)
            line(f"  {name}", m, res)
            rows.append({"market": mname, "strategy": name, "final": m.final_balance,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "fills": len(res.fills), "fees": m.fees_paid,
                         "liquidated": m.liquidated})
        s = make(**frozen)
        m, res = measure(s, *OOS, market=market)
        line(f"  funding_gate_v4 {frozen}", m, res)
        rows.append({"market": mname, "strategy": "funding_gate_v4", "final": m.final_balance,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "fills": len(res.fills), "fees": m.fees_paid,
                     "liquidated": m.liquidated})
        if active_frac is None:
            active_frac = float(res.df["funding_active"].mean())
            gated_frac = float((res.df["funding_scale"] < 1.0).mean())
    print(f"\nfraction of holdout bars with the funding gate ACTIVE "
          f"(funding data covers it): {active_frac:.1%}")
    print(f"fraction of holdout bars where the gate actually FIRED "
          f"(active AND scaled): {gated_frac:.2%}")
    pd.DataFrame(rows).to_csv(OUT / "holdout.csv", index=False)
    print(f"\nwritten: {OUT / 'holdout.csv'}")


# -------------------------------------------------------------------- interval


def interval(frozen: dict | None = None) -> None:
    """Paired block-bootstrap, funding_gate_v4 minus kelly_regime_v4, on the holdout."""
    from tradebot.inference import (daily_returns, max_drawdown_from_returns,
                                    paired_bootstrap, stationary_bootstrap_indices,
                                    total_log_return)

    frozen = frozen or dict(pctile_threshold=0.90, floor=0.0, momentum_override=None)
    rows = []
    for mname, market in MARKETS:
        res_gate = run_period(make(**frozen), DF, *OOS, market=market,
                              start_balance=1_000.0, data_label=LABEL)
        res_base = run_period(get_strategy("kelly_regime_v4"), DF, *OOS, market=market,
                              start_balance=1_000.0, data_label=LABEL)
        a = daily_returns(res_gate.equity).to_numpy()
        b = daily_returns(res_base.equity).to_numpy()
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        idx = stationary_bootstrap_indices(n, 30.0, 2_000, np.random.default_rng(7))
        print(f"\n{mname}: funding_gate_v4 minus kelly_regime_v4 on the 2023+ holdout "
              f"({n} daily observations)")
        for stat_name, stat in (("Δ log growth", total_log_return),
                                ("Δ max drawdown (pp)", max_drawdown_from_returns)):
            r = paired_bootstrap(a, b, stat, indices=idx)
            mark = "▲" if r.diff.lo > 0 else ("▼" if r.diff.hi < 0 else "≈")
            print(f"  {stat_name:22s} {mark} {r.diff.point:>+7.3f} "
                  f"[{r.diff.lo:>+7.3f}, {r.diff.hi:>+7.3f}]  P(>0)={r.p_positive:.2f}")
            rows.append({"market": mname, "stat": stat_name, "gate": r.stat_a,
                         "baseline": r.stat_b, "diff": r.diff.point,
                         "lo": r.diff.lo, "hi": r.diff.hi,
                         "p_positive": r.p_positive, "significant": r.significant})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intervals.csv", index=False)
    print(f"\nwritten: {OUT / 'intervals.csv'}")


# --------------------------------------------------------------------- windows


def windows(trials: int = 40, seed: int = 42, frozen: dict | None = None) -> None:
    """Pre-registered falsification test: resampled windows over the
    funding-available period (2020-01-01 -> 2023-12-31) ONLY -- the only
    span where the gate can possibly do anything. Paired against
    kelly_regime_v4 on identical windows, R-19/R-28's design.
    """
    from tradebot.metrics import max_drawdown_pct

    frozen = frozen or dict(pctile_threshold=0.90, floor=0.0, momentum_override=None)
    contenders = [("buy_and_hold", get_strategy("buy_and_hold")),
                  ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                  ("funding_gate_v4", make(**frozen))]

    fund_lo = int(DF.index.searchsorted(FUNDING.index[0]))
    fund_hi = int(DF.index.searchsorted(FUNDING.index[-1] + pd.Timedelta(hours=8)))
    warmup = max(s.warmup for _, s in contenders) + 10
    span_lo = fund_lo + warmup
    span_hi = fund_hi
    if span_hi <= span_lo:
        raise SystemExit("funding-covered span too short for windowing")

    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        max_len_bars = span_hi - span_lo
        length = int(rng.integers(30, min(366, max_len_bars // 288)) * 288)
        start = int(rng.integers(span_lo, span_hi - length))
        specs.append((start, length))

    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window_df = DF.iloc[start - warmup: start + length]
        for mname, market in MARKETS:
            for name, strat in contenders:
                res = run_backtest(strat, window_df, market, 1_000.0,
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

    print(f"\n\n{trials} random windows (30-365 days) inside the funding-covered "
          f"span ({FUNDING.index[0]:%Y-%m-%d} -> {FUNDING.index[-1]:%Y-%m-%d}), "
          f"identical across strategies:\n")
    for mname, _ in MARKETS:
        print(f"  {mname}:")
        sub = res[res.market == mname]
        bench = sub[sub.strategy == "buy_and_hold"].set_index("trial")["return_pct"]
        for name, _ in contenders:
            g = sub[sub.strategy == name].set_index("trial")
            print(f"    {name:18s} median return {g.return_pct.median():>+8.1f}%  "
                  f"median DD {g.max_dd_pct.median():>5.1f}%  "
                  f"worst DD {g.max_dd_pct.max():>5.1f}%  "
                  f"beat hold {(g['return_pct'] > bench).mean():>5.0%}  "
                  f"liq {g.liquidated.mean():>4.0%}")
        a = sub[sub.strategy == "funding_gate_v4"].set_index("trial")
        b = sub[sub.strategy == "kelly_regime_v4"].set_index("trial")
        d_ret = (a.return_pct - b.return_pct).dropna()
        d_dd = (a.max_dd_pct - b.max_dd_pct).dropna()
        print(f"    paired funding_gate_v4 - kelly_regime_v4: "
              f"return median {d_ret.median():+.1f}pp, gate higher in "
              f"{(d_ret > 0).mean():.0%};  DD median {d_dd.median():+.1f}pp, "
              f"gate deeper in {(d_dd > 0).mean():.0%}")
        print()


# ----------------------------------------------------------------------- costs


def costs(frozen: dict | None = None) -> None:
    """Step 4 cost checks: 0.40% spot taker tier, funding charged on futures."""
    frozen = frozen or dict(pctile_threshold=0.90, floor=0.0, momentum_override=None)

    print("HOLDOUT 2023+ spot, at both taker tiers:")
    for tier, label in ((0.001, "0.10% (table assumption)"),
                        (0.004, "0.40% (Bitstamp entry tier)")):
        market = MarketSpec.spot(fee_rate=tier)
        print(f"  {label}")
        for name in ("buy_and_hold", "kelly_regime_v4"):
            line(f"    {name}", *measure(get_strategy(name), *OOS, market=market))
        line("    funding_gate_v4", *measure(make(**frozen), *OOS, market=market))

    print(f"\nHOLDOUT 2023+ futures 5x with REAL funding CHARGED "
          f"(real coverage through {FUNDING.index[-1]:%Y-%m}):")
    lo = int(DF.index.searchsorted(OOS[0]))
    for name, strat in (("buy_and_hold", get_strategy("buy_and_hold")),
                        ("kelly_regime_v4", get_strategy("kelly_regime_v4")),
                        ("funding_gate_v4", make(**frozen))):
        pre = min(lo, strat.warmup)
        raw = run_backtest(strat, DF.iloc[lo - pre:], FUTURES, 1_000.0,
                           trade_start=pre, funding=FUNDING, data_label=LABEL)
        trimmed = replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
        m = compute_metrics(trimmed)
        print(f"  {name:20s} final=${m.final_balance:>9,.0f} "
              f"DD={m.max_drawdown_pct:>5.1f}% sharpe={m.sharpe:>5.2f} "
              f"funding paid=${raw.funding_paid:>7,.0f}"
              f"{'  LIQUIDATED' if m.liquidated else ''}")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}"
          f"  (data: {LABEL})", file=sys.stderr)
    cmds = {"rates": rates, "sweep": sweep, "window_check": window_check,
            "causality": causality, "holdout": holdout, "interval": interval,
            "windows": windows, "costs": costs}
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate.py [{'|'.join(cmds)}]")
