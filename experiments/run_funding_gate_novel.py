#!/usr/bin/env python
"""Driver for the NOVEL funding branch: continuous cost-of-carry-adjusted Kelly.

See ``experiments/funding_gate_novel.py`` for the derivation. This file is
the evaluation harness: sweep, freeze, falsification stress, causality.

Splits, per this task's explicit instructions (funding data only exists
2020-2023, so the usual 2017-based inner splits don't apply here)::

    inner-train        2020-01-01 -> 2021-12-31   fit / sweep
    inner-validation    2022-01-01 -> 2022-12-31   select k, halflife
    holdout             2023-01-01 ->              NOT TOUCHED by this file

Usage::

    python experiments/run_funding_gate_novel.py sweep         # step 3, 12 configs
    python experiments/run_funding_gate_novel.py falsification # 0.10% vs 0.40% fee stress
    python experiments/run_funding_gate_novel.py causality     # lookahead probe
    python experiments/run_funding_gate_novel.py all
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

from experiments.funding_gate_novel import FundingGateNovel  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)

# --- pre-registered inner splits for this task -----------------------------
TRAIN = ("2020-01-01", "2021-12-31")
VALID = ("2022-01-01", "2022-12-31")
# The holdout (2023-01-01 ->) is never constructed or referenced below.

N_EVALUATED = 0  # every configuration this file scores, counted once
K_GRID = (0.5, 1.0, 2.0, 4.0)
HALFLIFE_GRID = (1.0, 3.0, 7.0)


def period(strategy, start, end, *, funding=None, market=FUTURES, count=False):
    """Backtest over a date range, warmed on the bars before it.

    Adapted from ``scripts/funding_study.py::_period``. ``start``/``end``
    must never be at or after 2023-01-01 in this file.
    """
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    assert pd.Timestamp(end, tz="UTC") < pd.Timestamp("2023-01-01", tz="UTC"), (
        "run_funding_gate_novel.py must not evaluate data at/after 2023-01-01"
    )
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], market, 1_000.0,
                        trade_start=pre, funding=funding, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    return compute_metrics(trimmed), raw.funding_paid, len(raw.fills)


def _row(label, split, m, funding_paid, trades):
    return {"label": label, "split": split, "final_balance": m.final_balance,
            "max_dd_pct": m.max_drawdown_pct, "sharpe": m.sharpe,
            "trades": trades, "funding_paid": funding_paid}


def _print_row(r):
    print(f"    {r['label']:34s} final=${r['final_balance']:>10,.0f} "
          f"DD={r['max_dd_pct']:>5.1f}% sharpe={r['sharpe']:>6.2f} "
          f"trades={r['trades']:>4d} funding_paid=${r['funding_paid']:>8,.0f}")


# ---------------------------------------------------------------------- sweep


def sweep(write: bool = True) -> pd.DataFrame:
    """Step 3: 12 configurations (k x halflife), both inner splits, funding-charged."""
    rows = []
    print(f"{'=' * 90}\nBASELINE  kelly_regime_v4  (ungated, real funding charged both arms)\n{'=' * 90}")
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        m, paid, trades = period(get_strategy("kelly_regime_v4"), start, end,
                                  funding=REAL_FUNDING, count=(split_name == "inner-train"))
        r = _row("kelly_regime_v4", split_name, m, paid, trades)
        rows.append(r)
        _print_row(r)

    print(f"\n{'=' * 90}\nSWEEP  funding_gate_novel  k in {K_GRID}  "
          f"halflife_days in {HALFLIFE_GRID}\n{'=' * 90}")
    for hl in HALFLIFE_GRID:
        for k in K_GRID:
            label = f"k={k:g} hl={hl:g}d"
            print(f"\n  {label}")
            for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
                strat = FundingGateNovel(funding=REAL_FUNDING, k=k, funding_halflife_days=hl)
                m, paid, trades = period(strat, start, end, funding=REAL_FUNDING,
                                          count=(split_name == "inner-train"))
                r = _row(label, split_name, m, paid, trades)
                rows.append(r)
                _print_row(r)

    df = pd.DataFrame(rows)
    print(f"\nconfigurations evaluated (distinct, counted once): {N_EVALUATED}")
    if write:
        out = ROOT / "reports" / "funding_gate_novel"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "sweep.csv", index=False)
        print(f"written: {out / 'sweep.csv'}")
    return df


def freeze(df: pd.DataFrame | None = None) -> tuple[float, float]:
    """Select (k, halflife) using inner-validation only, and report the plateau."""
    if df is None:
        df = sweep(write=False)
    val = df[(df.split == "inner-validation") & (df.label != "kelly_regime_v4")].copy()
    val["k"] = val["label"].str.extract(r"k=([\d.]+)").astype(float)
    val["hl"] = val["label"].str.extract(r"hl=([\d.]+)d").astype(float)
    base = df[(df.split == "inner-validation") & (df.label == "kelly_regime_v4")].iloc[0]

    val["sharpe_edge"] = val["sharpe"] - base["sharpe"]
    val["dd_edge"] = base["max_dd_pct"] - val["max_dd_pct"]  # positive = shallower DD
    val["final_edge"] = val["final_balance"] - base["final_balance"]

    print(f"\n{'=' * 90}\nFREEZE  selection on inner-validation only  "
          f"(baseline kelly_regime_v4: final=${base['final_balance']:,.0f} "
          f"DD={base['max_dd_pct']:.1f}% sharpe={base['sharpe']:.2f} "
          f"trades={base['trades']})\n{'=' * 90}")
    print(f"  {'k':>5s} {'hl':>5s} {'final':>12s} {'DD':>6s} {'sharpe':>7s} "
          f"{'sharpe_edge':>11s} {'dd_edge(pp)':>11s} {'final_edge':>12s} {'trades':>7s}")
    for _, r in val.sort_values(["hl", "k"]).iterrows():
        print(f"  {r.k:>5g} {r.hl:>5g} ${r.final_balance:>11,.0f} {r.max_dd_pct:>5.1f}% "
              f"{r.sharpe:>7.2f} {r.sharpe_edge:>+11.2f} {r.dd_edge:>+11.1f} "
              f"${r.final_edge:>+11,.0f} {r.trades:>7d}")

    # Naive selection (argmax cell) — shown first because it is the wrong
    # way to do this, and the wrongness has to be visible, not just avoided.
    # Picking the single best-Sharpe cell out of 12 is exactly the kind of
    # search ROUTINE.md warns manufactures fake winners: the argmax below
    # lands on k=4, hl=7d, whose own row is NOT monotonic in k (k=2 is worse
    # than both its neighbors) — a lone spike sitting on the thinnest slice
    # of the whole grid (15 validation trades). It is reported, not frozen.
    naive = val.sort_values(["sharpe_edge"], ascending=False).iloc[0]
    print(f"\n  naive argmax (best single cell): k={naive.k:g} hl={naive.hl:g}  "
          f"sharpe_edge={naive.sharpe_edge:+.2f} -- NOT used, see reasoning below")

    # Principled selection: anchor at k=1.0, the literal coefficient the
    # derivation produces (module docstring). k=0.5/2/4 exist to check
    # whether the neighborhood around that literal value is a stable region
    # rather than to be searched over for the best score. Among the three
    # halflives at k=1.0, pick the one with the best inner-validation
    # sharpe_edge (a one-dimensional choice, not a 12-cell search).
    at_k1 = val[np.isclose(val["k"], 1.0)].sort_values("sharpe_edge", ascending=False)
    print(f"\n  principled selection: hold k=1.0 fixed (the literal derivation), "
          f"choose halflife on inner-validation:")
    for _, r in at_k1.iterrows():
        print(f"    hl={r.hl:>4g}d  sharpe_edge={r.sharpe_edge:+.2f}  "
              f"dd_edge={r.dd_edge:+.1f}pp  final_edge=${r.final_edge:+,.0f}")
    frozen_hl = float(at_k1.iloc[0].hl)
    frozen_k = 1.0
    print(f"\n  FROZEN: k={frozen_k:g}  funding_halflife_days={frozen_hl:g}")

    # Plateau check: across k at the frozen halflife, is k=1.0's choice a
    # region or a lone spike? A monotonic, non-reversing sweep counts as a
    # region even if it is not flat — a spike is a value that beats BOTH
    # its neighbors by more than the grid's own noise, which is what the
    # naive argmax row above shows and this row must NOT show.
    same_hl = val[val["hl"] == frozen_hl].sort_values("k")
    print(f"\n  plateau check at hl={frozen_hl:g}d, across k in {K_GRID}:")
    for _, r in same_hl.iterrows():
        marker = " <-- frozen (k=1.0)" if abs(r.k - frozen_k) < 1e-9 else ""
        print(f"    k={r.k:>4g}  sharpe_edge={r.sharpe_edge:+.2f}  "
              f"dd_edge={r.dd_edge:+.1f}pp  final_edge=${r.final_edge:+,.0f}{marker}")
    edges = same_hl.sort_values("k")["sharpe_edge"].to_numpy()
    monotone = np.all(np.diff(edges) >= -1e-9) or np.all(np.diff(edges) <= 1e-9)
    print(f"  row is {'monotonic (a region, not a spike)' if monotone else 'NOT monotonic'} "
          f"in k at hl={frozen_hl:g}d")
    return frozen_k, frozen_hl


# --------------------------------------------------------------- falsification


def falsification(frozen: tuple[float, float] | None = None) -> None:
    """0.40% taker-fee stress: does the candidate's edge over ungated v4 invert?

    Restricted to 2020-01-01..2022-12-31 (inner-train + inner-validation
    combined). Compares default futures fee (0.05%, MarketSpec.futures()'s
    default) against a 0.40% stress tier, real funding charged on both arms
    at both fee tiers.
    """
    if frozen is None:
        frozen = freeze()
    k, hl = frozen
    start, end = "2020-01-01", "2022-12-31"

    print(f"\n{'=' * 90}\nFALSIFICATION  0.05% (default) vs 0.40% taker fee, "
          f"{start} .. {end}, real funding both arms\n{'=' * 90}")
    print(f"  frozen candidate: k={k:g} funding_halflife_days={hl:g}\n")

    results = {}
    for tier_label, fee in (("0.05% (project default)", 0.0005), ("0.40% (stress)", 0.004)):
        market = MarketSpec.futures(leverage=5.0, fee_rate=fee)
        base_m, base_paid, base_trades = period(get_strategy("kelly_regime_v4"), start, end,
                                                 funding=REAL_FUNDING, market=market)
        cand_strat = FundingGateNovel(funding=REAL_FUNDING, k=k, funding_halflife_days=hl)
        cand_m, cand_paid, cand_trades = period(cand_strat, start, end,
                                                 funding=REAL_FUNDING, market=market)
        results[tier_label] = (base_m, cand_m)
        print(f"  {tier_label}")
        _print_row(_row("kelly_regime_v4 (ungated)", "combined", base_m, base_paid, base_trades))
        _print_row(_row(f"funding_gate_novel k={k:g} hl={hl:g}", "combined",
                         cand_m, cand_paid, cand_trades))
        final_edge = cand_m.final_balance - base_m.final_balance
        log_growth_edge = (np.log(max(cand_m.final_balance, 1e-9))
                            - np.log(max(base_m.final_balance, 1e-9)))
        print(f"    edge: final_balance {final_edge:+,.0f}   "
              f"log-growth {log_growth_edge:+.4f}\n")

    (base_lo, cand_lo) = results["0.05% (project default)"]
    (base_hi, cand_hi) = results["0.40% (stress)"]
    edge_lo = cand_lo.final_balance - base_lo.final_balance
    edge_hi = cand_hi.final_balance - base_hi.final_balance
    lg_lo = np.log(max(cand_lo.final_balance, 1e-9)) - np.log(max(base_lo.final_balance, 1e-9))
    lg_hi = np.log(max(cand_hi.final_balance, 1e-9)) - np.log(max(base_hi.final_balance, 1e-9))
    inverted_final = (edge_lo > 0) != (edge_hi > 0)
    inverted_growth = (lg_lo > 0) != (lg_hi > 0)
    print(f"  final-balance edge sign:  {edge_lo:+,.0f} (0.05%)  ->  {edge_hi:+,.0f} (0.40%)  "
          f"{'INVERTED (falsified)' if inverted_final else 'stable'}")
    print(f"  log-growth edge sign:     {lg_lo:+.4f} (0.05%)  ->  {lg_hi:+.4f} (0.40%)  "
          f"{'INVERTED (falsified)' if inverted_growth else 'stable'}")


# ------------------------------------------------------------------- causality


def causality() -> None:
    """Strict lookahead probe, by hand — experiments get no CI protection.

    Same two-opposite-tampers procedure as ``experiments/run_matched_risk.py``:
    bars after a cut are multiplied by 3x in one copy and divided by 3x in
    the other (volume by 7x/÷7x); every queued order at or before the cut
    must be byte-identical between the two tampered copies, and the
    ``target`` / ``funding_drag_annualized`` columns must be identical
    before the cut (max abs diff exactly 0.0) — the column check is what
    catches a full-series statistic that a truncation-only test would miss.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    # Window and tamper zone are both confined to strictly before
    # 2023-01-01: unlike run_matched_risk.py's causality() (which is free to
    # use the tail of the full series), this file's task instructions
    # forbid slicing or evaluating ANY 2023+ data, so the window ends at the
    # holdout boundary rather than at the end of the dataset.
    boundary = int(DF.index.searchsorted("2023-01-01"))
    df = DF.iloc[boundary - 200_000: boundary].copy()
    assert df.index[-1] < pd.Timestamp("2023-01-01", tz="UTC")
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    configs = [(k, hl) for hl in HALFLIFE_GRID for k in K_GRID]
    ok = True
    for k, hl in configs:
        def decisions(frame, k=k, hl=hl):
            s = FundingGateNovel(funding=REAL_FUNDING, k=k, funding_halflife_days=hl)
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
        pa = FundingGateNovel(funding=REAL_FUNDING, k=k, funding_halflife_days=hl).prepare(up.copy())
        pb = FundingGateNovel(funding=REAL_FUNDING, k=k, funding_halflife_days=hl).prepare(down.copy())
        worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                    for c in ("target", "funding_drag_annualized"))
        good = not bad and worst == 0.0
        ok &= good
        print(f"  k={k:>4g} hl={hl:>4g}d  "
              f"orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column difference| before the cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------ main


COMMANDS = {"sweep": sweep, "freeze": freeze, "falsification": falsification,
            "causality": causality}


def main() -> None:
    if REAL_FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "all":
        df = sweep()
        frozen = freeze(df)
        falsification(frozen)
        causality()
    elif choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_novel.py "
              f"[{'|'.join(COMMANDS)}|all]")


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    main()
