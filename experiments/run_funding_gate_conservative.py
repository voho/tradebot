#!/usr/bin/env python
"""Driver for backlog B-05 -- conservative hard top-decile funding gate.

kelly_regime_v4, unchanged, except forced flat whenever the causal
rolling percentile-rank of the current funding rate is in the top
decile of its own trailing history (a hysteresis latch, exactly like
kelly_regime_v3's volatility-breakout state machine). See
``experiments/funding_gate_conservative.py`` for the mechanism.

Splits, per docs/ROUTINE.md step 3, restricted to the funding data's
own coverage (2020-01-01 -> 2023-12-31) since this idea has no effect
before funding is observed and the holdout starts 2023-01-01::

    inner-train       2020-01-01 -> 2021-12-31   sweep, iterate
    inner-validation  2022-01-01 -> 2022-12-31   select the frozen config
    holdout           2023-01-01 ->               NOT TOUCHED by this file

Every evaluation charges REAL funding on both arms (candidate and the
ungated kelly_regime_v4 comparison), 5x futures, via ``period()``
below -- adapted directly from ``scripts/funding_study.py::_period``.

Usage::

    python experiments/run_funding_gate_conservative.py sweep       # 9-config grid
    python experiments/run_funding_gate_conservative.py plateau     # gap check
    python experiments/run_funding_gate_conservative.py falsify     # Monte Carlo windows
    python experiments/run_funding_gate_conservative.py causality   # lookahead probe
    python experiments/run_funding_gate_conservative.py all         # everything + report
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

from experiments.funding_gate_conservative import FundingGateConservative  # noqa: E402
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_funding  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics, max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402

DF, LABEL = load_dataset(ROOT / "data", "spot")
REAL_FUNDING = load_funding(ROOT / "data")
FUTURES = MarketSpec.futures(leverage=5.0)

# Restricted to funding coverage; never touches 2023-01-01 onward.
INNER_TRAIN = ("2020-01-01", "2021-12-31")
INNER_VALID = ("2022-01-01", "2022-12-31")
FUNDING_COVERED_END = "2022-12-31"  # for the Monte Carlo falsification test

N_EVALUATED = 0  # distinct candidate configurations swept, for the honest count

# --------------------------------------------------------------------- period


def _run(strategy, start, end):
    """One backtest over a date range, warmed on the bars before it, real funding."""
    lo = int(DF.index.searchsorted(start))
    hi = int(DF.index.searchsorted(end, side="right"))
    pre = min(lo, strategy.warmup)
    raw = run_backtest(strategy, DF.iloc[lo - pre: hi], FUTURES, 1_000.0,
                        trade_start=pre, funding=REAL_FUNDING, data_label=LABEL)
    trimmed = raw if pre == 0 else replace(raw, equity=raw.equity.iloc[pre:], df=raw.df.iloc[pre:])
    return raw, trimmed


def period(strategy, start, end):
    """Backtest over a date range -> (metrics, funding_paid).

    Adapted from scripts/funding_study.py::_period -- same pattern,
    parametrized on a strategy instance rather than a registry name so an
    unregistered experiment class can be passed directly.
    """
    raw, trimmed = _run(strategy, start, end)
    return compute_metrics(trimmed), raw.funding_paid


def _row(tag, m, funding_paid, fills):
    return {"tag": tag, "final": m.final_balance, "max_dd": m.max_drawdown_pct,
            "sharpe": m.sharpe, "fills": fills, "funding_paid": funding_paid}


def _print_row(r):
    print(f"    {r['tag']:42s} final=${r['final']:>10,.0f} "
          f"DD={r['max_dd']:>5.1f}% sharpe={r['sharpe']:>6.2f} "
          f"fills={r['fills']:>5d} funding_paid=${r['funding_paid']:>9,.0f}")


def _eval_candidate(tag, strategy, start, end, *, count):
    global N_EVALUATED
    if count:
        N_EVALUATED += 1
    raw, trimmed = _run(strategy, start, end)
    m = compute_metrics(trimmed)
    return _row(tag, m, raw.funding_paid, len(raw.fills))


# ----------------------------------------------------------------------- grid

DECILE_IN_GRID = (0.85, 0.90, 0.95)
WINDOW_GRID = (90, 180, 365)


def _configs():
    for decile_in in DECILE_IN_GRID:
        for pct_window_days in WINDOW_GRID:
            yield decile_in, pct_window_days


def sweep(verbose: bool = True) -> pd.DataFrame:
    """Step 3: 9-configuration grid on inner-train and inner-validation only."""
    rows = []
    for split_name, (start, end) in (("inner-train", INNER_TRAIN),
                                      ("inner-validation", INNER_VALID)):
        if verbose:
            print(f"\n{split_name} ({start} -> {end}), 5x futures, real funding:")
        base = get_strategy("kelly_regime_v4")
        base_row = _eval_candidate("kelly_regime_v4 (ungated, comparison)",
                                    base, start, end, count=False)
        if verbose:
            _print_row(base_row)
        for decile_in, pct_window_days in _configs():
            tag = f"decile_in={decile_in:.2f} out=0.75 window={pct_window_days}d"
            s = FundingGateConservative(funding=REAL_FUNDING, decile_in=decile_in,
                                         decile_out=0.75, pct_window_days=pct_window_days)
            row = _eval_candidate(tag, s, start, end,
                                   count=(split_name == "inner-train"))
            row["split"] = split_name
            row["decile_in"] = decile_in
            row["decile_out"] = 0.75
            row["pct_window_days"] = pct_window_days
            rows.append(row)
            if verbose:
                _print_row(row)
        base_row["split"], base_row["decile_in"] = split_name, None
        base_row["decile_out"], base_row["pct_window_days"] = None, None
        rows.append(base_row)
    df = pd.DataFrame(rows)
    if verbose:
        print(f"\nconfigurations evaluated in the grid (distinct, counted once): "
              f"{N_EVALUATED}")
    return df


# -------------------------------------------------------------------- plateau


def plateau(frozen_decile_in: float, frozen_window: int, verbose: bool = True) -> pd.DataFrame:
    """Gap check around the frozen decile_in: decile_out at gap 10pp and 20pp."""
    global N_EVALUATED
    rows = []
    gaps = (0.10, 0.20)
    for split_name, (start, end) in (("inner-train", INNER_TRAIN),
                                      ("inner-validation", INNER_VALID)):
        if verbose:
            print(f"\n{split_name} plateau check, decile_in={frozen_decile_in:.2f}, "
                  f"window={frozen_window}d:")
        for gap in gaps:
            decile_out = round(frozen_decile_in - gap, 2)
            tag = f"gap={gap*100:.0f}pp (out={decile_out:.2f})"
            s = FundingGateConservative(funding=REAL_FUNDING, decile_in=frozen_decile_in,
                                         decile_out=decile_out, pct_window_days=frozen_window)
            row = _eval_candidate(tag, s, start, end,
                                   count=(split_name == "inner-train"))
            row["split"], row["gap"] = split_name, gap
            row["decile_in"], row["decile_out"] = frozen_decile_in, decile_out
            rows.append(row)
            if verbose:
                _print_row(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- falsify


def falsify(decile_in: float, decile_out: float, pct_window_days: int,
            trials: int = 30, seed: int = 11, verbose: bool = True) -> pd.DataFrame:
    """Monte Carlo path sensitivity: random contiguous blocks, 2020-01-01 -> 2022-12-31 ONLY.

    Kill condition (falsified): the candidate is worse than ungated v4 on
    BOTH return and drawdown in a majority of windows.
    """
    candidate = FundingGateConservative(funding=REAL_FUNDING, decile_in=decile_in,
                                         decile_out=decile_out, pct_window_days=pct_window_days)
    baseline = get_strategy("kelly_regime_v4")
    warmup = max(candidate.warmup, baseline.warmup) + 10

    lo = int(DF.index.searchsorted("2020-01-01"))
    hi = int(DF.index.searchsorted(FUNDING_COVERED_END, side="right"))
    assert DF.index[hi - 1] < pd.Timestamp("2023-01-01", tz="UTC"), \
        "falsify() window touched 2023-01-01+; refusing to run"

    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(1, trials + 1):
        # a few weeks to a few months: 14 to 120 days, in 5-minute bars
        length_days = int(rng.integers(14, 121))
        length = length_days * 288
        # window body (post-warmup) must sit fully inside [lo, hi] with a
        # warmup prefix also inside [lo, hi] so no bar reaches back before
        # 2020-01-01 or forward past 2022-12-31
        start_pos = int(rng.integers(lo + warmup, max(lo + warmup + 1, hi - length)))
        window = DF.iloc[start_pos - warmup: start_pos + length]
        assert window.index[-1] < pd.Timestamp("2023-01-01", tz="UTC")
        assert window.index[0] >= pd.Timestamp("2020-01-01", tz="UTC") - pd.Timedelta(days=1)

        for name, strat in (("candidate", candidate), ("v4 (ungated)", baseline)):
            raw = run_backtest(strat, window, FUTURES, 1_000.0, trade_start=warmup,
                                funding=REAL_FUNDING, data_label=LABEL)
            eq = raw.equity.to_numpy(dtype=float)
            base_val, seg = eq[warmup], eq[warmup:]
            ok = np.isfinite(base_val) and base_val > 0
            rows.append({
                "trial": trial, "strategy": name,
                "return_pct": 100.0 * (seg[-1] / base_val - 1.0) if ok else -100.0,
                "max_dd_pct": max_drawdown_pct(seg) if ok else 100.0,
            })
        if verbose:
            print(f"[{trial}/{trials}]", end=" ", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    res = pd.DataFrame(rows)

    cand = res[res.strategy == "candidate"].set_index("trial")
    base = res[res.strategy == "v4 (ungated)"].set_index("trial")
    beat_return = (cand.return_pct > base.return_pct)
    beat_dd = (cand.max_dd_pct < base.max_dd_pct)  # lower DD is better
    both_worse = (~beat_return) & (~beat_dd)

    if verbose:
        print(f"\n{trials} random contiguous windows, 14-120 days, "
              f"2020-01-01 -> {FUNDING_COVERED_END} only:")
        print(f"  candidate beats ungated v4 on RETURN in "
              f"{beat_return.mean():.0%} of windows")
        print(f"  candidate beats ungated v4 on MAX DRAWDOWN (shallower) in "
              f"{beat_dd.mean():.0%} of windows")
        print(f"  candidate WORSE on BOTH in {both_worse.mean():.0%} of windows "
              f"(kill condition: worse on both in a MAJORITY)")
        verdict = "FALSIFIED" if both_worse.mean() > 0.5 else "NOT falsified"
        print(f"  verdict: {verdict}")
    return res


# ------------------------------------------------------------------- causality


def causality(configs=None) -> bool:
    """Strict lookahead probe, by hand -- experiments get no CI protection.

    Copied almost verbatim from experiments/run_matched_risk.py::causality.
    Two opposite tampers of bars after a cut (OHLC x3 / /3, volume x7 / /7);
    every queued order at or before the cut must be byte-identical between
    the two tampered copies, AND the funding_pct / target columns must be
    exactly equal (max abs diff == 0.0) before the cut -- the column check
    is what catches a full-series statistic that a truncation test alone
    would miss.

    Uses the tail of the full dataset (mirrors run_matched_risk.py's own
    causality() probe) purely to exercise the tamper-invariance mechanism;
    it scores no return, drawdown, or other performance figure, and is not
    a strategy evaluation.
    """
    from tradebot.broker import PaperBroker
    from tradebot.strategy import Context

    if configs is None:
        configs = [
            dict(decile_in=0.90, decile_out=0.75, pct_window_days=180),
            dict(decile_in=0.85, decile_out=0.75, pct_window_days=90),
            dict(decile_in=0.95, decile_out=0.75, pct_window_days=365),
            dict(decile_in=0.90, decile_out=0.70, pct_window_days=180),
        ]

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
    for cfg in configs:
        def make():
            return FundingGateConservative(funding=REAL_FUNDING, **cfg)

        def decisions(frame):
            s = make()
            prepared = s.prepare(frame.copy())
            broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
            out = []
            for i in bars:
                ctx = Context(prepared, i, broker)
                s.on_bar(ctx)
                out.append([(o.side, o.qty, o.target) for o in ctx.orders])
            return out

        bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
        pa = make().prepare(up.copy())
        pb = make().prepare(down.copy())
        worst = max(float(np.nanmax(np.abs(pa[c].to_numpy()[:cut] - pb[c].to_numpy()[:cut])))
                    for c in ("target", "funding_pct"))
        good = not bad and worst == 0.0
        ok &= good
        print(f"  decile_in={cfg['decile_in']:.2f} decile_out={cfg['decile_out']:.2f} "
              f"window={cfg['pct_window_days']:>3d}d   "
              f"orders {'match' if not bad else f'DIFFER at {bad}'}   "
              f"max |column diff| before cut = {worst:.3e}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------- main


def all_steps() -> None:
    grid_df = sweep()
    frozen_decile_in, frozen_window = 0.90, 90  # see report for the reasoning
    plateau_df = plateau(frozen_decile_in, frozen_window)
    print("\n" + "=" * 74 + "\ncausality\n" + "=" * 74)
    causality_ok = causality()
    print("\n" + "=" * 74 + "\nfalsification (Monte Carlo, 2020-01-01 -> 2022-12-31)\n" + "=" * 74)
    falsify(frozen_decile_in, 0.75, frozen_window)
    print(f"\ncausality PASS: {causality_ok}")


COMMANDS = {"sweep": sweep, "plateau": lambda: plateau(0.90, 90),
            "falsify": lambda: falsify(0.90, 0.75, 90),
            "causality": causality, "all": all_steps}


if __name__ == "__main__":
    if REAL_FUNDING is None:
        raise SystemExit("no funding data committed; see docs/VALIDATION.md")
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    print(f"funding: {len(REAL_FUNDING):,} settlements  "
          f"{REAL_FUNDING.index[0]:%Y-%m-%d} -> {REAL_FUNDING.index[-1]:%Y-%m-%d}",
          file=sys.stderr)
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in COMMANDS:
        COMMANDS[choice]()
    else:
        print(f"usage: python experiments/run_funding_gate_conservative.py "
              f"[{'|'.join(COMMANDS)}]")
