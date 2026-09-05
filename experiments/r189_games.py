#!/usr/bin/env python
"""R-189: frozen ten-game intraday battery and chart integration.

Run with ``.venv/bin/python experiments/r189_games.py run --workers 4``;
then ``... report``. No parameter selection or holdout-driven retuning.
The run writes a source/data hash manifest BEFORE preparing any signals.

Protocol: explicit historical Bitstamp spot, completed 5m bars, next-open
fills, $1,000 fresh broker per cell. Online learners retain causal earlier
history; account resets do not reset their knowledge. Inner train ends
2020-12-31 23:55, validation ends 2022-12-31 23:55, retrospective holdout
starts 2023-01-01. Primary spot costs 10bp taker +1bp slippage each way.
Falsifications: 40bp taker+1bp; ETH spot; venue-matched Deribit perpetual
prices +8h funding (continuous-funding approximation), 5bp taker+1bp.
24 seeded, overlapping 120–365 day windows per market are descriptive
robustness checks, not independent hypothesis tests. Every candidate runs
every cell, including losers. No forced transactions to satisfy cadence.

Frozen promotion rule (all clauses required): primary validation and holdout
Sharpe exceed BOTH hold and Kelly v4 by >0.20; holdout profitable, shallower
drawdown than hold, 2–6 fills/calendar day; positive paired 95% lower bound
on holdout log-growth vs hold; program-trials DSR >=.95; profitable fee
stress, ETH and funded futures; >50% beta-window growth wins against BOTH
controls in BOTH markets, with no liquidation. Otherwise RESEARCH ONLY.
Registration/chart inclusion is requested by the operator and is independent
of promotion. No live trading. All ten defaults count as configurations.

Historical comparison cells use the old table's EXACT zero-slippage,
unfunded spot-proxy futures convention, separately labelled. Existing rows
are reused only after independently reproducing both controls on both
markets. They are never mixed with the funded/slippage evaluation cells.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from tradebot.broker import MarketSpec
from tradebot.data import load_ohlcv_csv, load_funding_deribit
from tradebot.engine import run_backtest
from tradebot.inference import (annualized_sharpe, daily_returns,
    deflated_sharpe_ratio, moments, paired_bootstrap, stationary_bootstrap_indices,
    total_log_return, max_drawdown_from_returns, bootstrap_interval)
from tradebot.metrics import Metrics, compute_metrics
from tradebot.registry import get_strategy
from tradebot.strategy import Strategy

OUT = ROOT / "reports/r189_games"
CANDIDATES = ("cautious_optimism", "squint_council", "normalhedge_council",
    "swap_regret_council", "blackwell_council", "minimax_council",
    "nash_council", "qre_council", "sleeping_council", "defensive_forecast")
CONTROLS = ("buy_and_hold", "kelly_regime_v4")
NAMES = CANDIDATES + CONTROLS
FILES = {"spot": "btcusd_spot_5m.csv.gz",
         "perp": "btcusdt_deribit_perp_5m.csv.gz",
         "eth": "ethusd_coinbase_spot_5m.csv.gz"}
HOLDOUT = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-08-12 00:40", tz="UTC")
PREFIX = 100 * 288 + 10
PRIOR_CONSULTATIONS = 831  # approximate inherited program count, not independent trials


class Prepared(Strategy):
    """Replay already-causal columns, delegating unchanged decision code."""

    def __init__(self, strategy, initial_bar: int | None = None):
        self.strategy = strategy
        self.name, self.warmup = strategy.name, strategy.warmup
        self.initial_bar = initial_bar

    def prepare(self, df):
        return df

    def on_bar(self, ctx):
        # The incumbent emits only target CHANGES. A fresh account must receive
        # its already-known target at the evaluation boundary, too. Historical
        # full-period cells retain the original strategy protocol unchanged.
        if (self.name == "kelly_regime_v4" and ctx.i == self.initial_bar
                and not ctx.in_market and float(ctx.bar["target"]) > 0):
            ctx.order_notional(float(ctx.bar["target"]))
            return
        self.strategy.on_bar(ctx)


def specifications():
    cells = [
        ("inner_train", "spot", None, pd.Timestamp("2020-12-31 23:55", tz="UTC"), .001, 1., False),
        ("inner_val", "spot", pd.Timestamp("2021-01-01", tz="UTC"), HOLDOUT - pd.Timedelta(minutes=5), .001, 1., False),
        ("holdout", "spot", HOLDOUT, END, .001, 1., False),
        ("fee_stress", "spot", HOLDOUT, END, .004, 1., False),
        ("eth_holdout", "eth", HOLDOUT, END, .001, 1., False),
        ("funded_val", "perp", pd.Timestamp("2021-01-01", tz="UTC"), HOLDOUT - pd.Timedelta(minutes=5), .0005, 1., True),
        ("funded_holdout", "perp", HOLDOUT, END, .0005, 1., True),
        ("full_spot", "spot", None, END, .001, 0., False),
        ("full_proxy", "spot", None, END, .0005, 0., False),
    ]
    rng = np.random.default_rng(189)
    days = (END.normalize() - HOLDOUT).days
    for k in range(24):
        length = int(rng.integers(120, 366))
        start = HOLDOUT + pd.Timedelta(days=int(rng.integers(0, days - length + 1)))
        end = start + pd.Timedelta(days=length) - pd.Timedelta(minutes=5)
        for kind, fee, funded in (("spot", .001, False), ("perp", .0005, True)):
            cells.append((f"beta_{kind}_{k:02d}", kind, start, end, fee, 1., funded))
    return cells


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT / "src/tradebot/strategies/intraday_games.py",
             ROOT / "src/tradebot/strategies/kelly_regime.py",
             ROOT / "src/tradebot/strategies/kelly_regime_v3.py",
             ROOT / "src/tradebot/strategies/kelly_regime_v4.py"]
    paths += [ROOT / "data" / f for f in FILES.values()]
    paths += [ROOT / "data/btcusdt_deribit_perp_funding_8h.csv.gz"]
    manifest = {"frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "configurations": len(CANDIDATES), "candidates": CANDIDATES,
        "protocol": __doc__, "hashes": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "cells_per_strategy": len(specifications()),
        "holdout_cells": sum(c[3] >= HOLDOUT for c in specifications()) * len(NAMES),
        "prior_consultations_approx": PRIOR_CONSULTATIONS}
    path = OUT / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["hashes"] != manifest["hashes"]:
            raise SystemExit("Frozen source/data changed: preserve this run and explicitly version a new protocol.")
        return previous
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def one_strategy(name):
    strategy = get_strategy(name)
    prepared = {}
    for kind, filename in FILES.items():
        raw = load_ohlcv_csv(ROOT / "data" / filename).loc[:END]
        prepared[kind] = strategy.prepare(raw.copy())
    funding = load_funding_deribit(ROOT / "data")
    assert funding is not None
    covered = funding.loc[pd.Timestamp("2021-01-01", tz="UTC"):END]
    expected_funding = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), END.floor("8h"), freq="8h")
    assert covered.index.equals(expected_funding), "Missing or incomplete funding coverage"
    rows, daily = [], []
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    for k, (cell, kind, start, end, fee, slip, funded) in enumerate(specifications()):
        frame = prepared[kind]
        lo = 0 if start is None else int(frame.index.searchsorted(start))
        hi = int(frame.index.searchsorted(end, side="right"))
        assert hi > lo and (start is None or frame.index[lo] == start)
        assert frame.index[hi - 1] == end, (kind, "incomplete price coverage", end)
        prefix = min(lo, PREFIX)
        sub = frame.iloc[lo - prefix:hi]
        futures = kind == "perp" or cell == "full_proxy"
        market = MarketSpec.futures(leverage=5., fee_rate=fee) if futures else MarketSpec.spot(fee_rate=fee)
        label = "spot (perp proxy)" if cell == "full_proxy" else (
            "Deribit perp + Deribit 8h funding" if funded else
            "Coinbase ETH spot" if kind == "eth" else "real")
        replay = Prepared(strategy, initial_bar=prefix if not cell.startswith("full_") else None)
        result = run_backtest(replay, sub, market, 1000., slippage_bps=slip,
            data_label=label, trade_start=prefix, funding=funding if funded else None)
        result = replace(result, equity=result.equity.iloc[prefix:], df=result.df.iloc[prefix:])
        metric = compute_metrics(result)
        days = (result.equity.index[-1] - result.equity.index[0] + pd.Timedelta(minutes=5)).total_seconds() / 86400
        completed = sum(not t.open_at_end for t in result.trades)
        fill_days = pd.Series([f.ts for f in result.fills], dtype="datetime64[ns, UTC]")
        active = fill_days.dt.normalize().nunique() if len(fill_days) else 0
        calendar_days = (result.equity.index[-1].normalize() - result.equity.index[0].normalize()).days + 1
        row = metric.as_row() | {"cell": cell, "start": str(result.equity.index[0]),
            "end": str(result.equity.index[-1]), "days": days, "fee_rate": fee,
            "slippage_bps": slip, "funding_paid": result.funding_paid,
            "funding_model": "Deribit 8h aggregation" if funded else "not charged",
            "fills": len(result.fills), "fills_per_day": len(result.fills)/days,
            "completed_round_trips": completed, "round_trips_per_day": completed/days,
            "active_days_pct": 100*active/calendar_days}
        rows.append(row)
        if not cell.startswith("beta_"):
            eq = result.equity.resample("1D").last()
            ret = daily_returns(result.equity)
            # Count the first day's PnL against the known fresh balance.
            # Historical chart intervals retain the existing repo convention.
            if not cell.startswith("full_"):
                ret.loc[eq.index[0]] = eq.iloc[0] / 1000. - 1.
            daily.extend({"cell": cell, "timestamp": str(t), "equity": float(e),
                          "return": float(ret.get(t, np.nan))} for t, e in eq.items())
        if cell == "holdout":
            pd.DataFrame([{"timestamp": str(f.ts), "side": f.side.name,
                "qty": f.qty, "price": f.price, "fee": f.fee, "kind": f.kind}
                for f in result.fills]).to_csv(out / "holdout_fills.csv", index=False)
        if k % 10 == 0:
            print(f"{name}: {k+1}/{len(specifications())} {cell} ${metric.final_balance:,.0f}, {len(result.fills)/days:.2f} fills/day", flush=True)
    pd.DataFrame(rows).to_csv(out / "cells.csv", index=False)
    pd.DataFrame(daily).to_csv(out / "daily.csv", index=False)
    print(f"{name}: complete", flush=True)
    return rows


def run(workers):
    freeze()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one_strategy, n): n for n in NAMES}
        for f in as_completed(futures):
            rows.extend(f.result())
    pd.DataFrame(rows).to_csv(OUT / "cells.csv", index=False)


def returns(name, cell):
    data = pd.read_csv(OUT / name / "daily.csv")
    data = data[data.cell == cell].set_index("timestamp")
    return data["return"].dropna()


def inference(cells):
    rows = []
    for cell, period, market in (("full_spot", "full", "spot"),
            ("full_proxy", "full", "futures"), ("holdout", "holdout", "spot"),
            ("funded_holdout", "holdout_funded", "futures")):
        bench = returns("buy_and_hold", cell)
        idx = stationary_bootstrap_indices(len(bench), 30., 2000, np.random.default_rng(7))
        for name in NAMES:
            r = returns(name, cell)
            assert r.index.equals(bench.index)
            a, b = r.to_numpy(), bench.to_numpy()
            sr = paired_bootstrap(a, b, annualized_sharpe, indices=idx)
            dd = paired_bootstrap(a, b, max_drawdown_from_returns, indices=idx)
            gr = paired_bootstrap(a, b, total_log_return, indices=idx)
            own_sr = bootstrap_interval(a, annualized_sharpe, indices=idx)
            own_dd = bootstrap_interval(a, max_drawdown_from_returns, indices=idx)
            nz = np.flatnonzero(a != 0)
            rows.append(dict(period=period, market=market, strategy=name, days=len(a),
                dead_tail_pct=100*(len(a)-1-nz[-1])/len(a) if len(nz) else 100.,
                sharpe=sr.stat_a, sharpe_lo=own_sr.lo, sharpe_hi=own_sr.hi,
                d_sharpe=sr.diff.point, d_sharpe_lo=sr.diff.lo, d_sharpe_hi=sr.diff.hi,
                p_sharpe_beats_hold=sr.p_positive, max_dd_pct=dd.stat_a,
                max_dd_lo=own_dd.lo, max_dd_hi=own_dd.hi,
                d_max_dd_pp=dd.diff.point, d_max_dd_lo=dd.diff.lo, d_max_dd_hi=dd.diff.hi,
                p_dd_deeper_than_hold=dd.p_positive, d_log_growth=gr.diff.point,
                d_log_growth_lo=gr.diff.lo, d_log_growth_hi=gr.diff.hi,
                p_growth_beats_hold=gr.p_positive))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "bootstrap.csv", index=False)
    return frame


def decisions(cells, evidence):
    def get(name, cell):
        return cells[(cells.strategy == name) & (cells.cell == cell)].iloc[0]
    trials = PRIOR_CONSULTATIONS + json.loads((OUT / "manifest.json").read_text())["holdout_cells"]
    sd = np.std([annualized_sharpe(returns(n, "inner_val").to_numpy()) for n in CANDIDATES], ddof=1)
    rows = []
    for name in CANDIDATES:
        h, v = get(name, "holdout"), get(name, "inner_val")
        r = returns(name, "holdout").to_numpy()
        skew, kurt = moments(r)
        dsr = deflated_sharpe_ratio(annualized_sharpe(r), len(r), skew, kurt, trials, sd)
        e = evidence[(evidence.strategy == name) & (evidence.period == "holdout")].iloc[0]
        checks = {"positive_holdout": h.final_balance > 1000,
            "cadence_2_to_6_fills": 2 <= h.fills_per_day <= 6,
            "lower_drawdown": h.max_drawdown_pct < get("buy_and_hold", "holdout").max_drawdown_pct,
            "growth_interval": e.d_log_growth_lo > 0,
            "program_dsr": dsr >= .95,
            "cost_stress": get(name, "fee_stress").final_balance > 1000,
            "eth": get(name, "eth_holdout").final_balance > 1000,
            "funded": get(name, "funded_holdout").final_balance > 1000}
        wins = {}
        for control in CONTROLS:
            checks[f"holdout_vs_{control}"] = h.sharpe > get(control, "holdout").sharpe + .20
            checks[f"validation_vs_{control}"] = v.sharpe > get(control, "inner_val").sharpe + .20
            for kind in ("spot", "perp"):
                a = cells[(cells.strategy == name) & cells.cell.str.startswith(f"beta_{kind}_")].set_index("cell")
                b = cells[(cells.strategy == control) & cells.cell.str.startswith(f"beta_{kind}_")].set_index("cell")
                rate = float((a.final_balance > b.final_balance.reindex(a.index)).mean())
                wins[f"beta_{kind}_beats_{control}_pct"] = 100*rate
                checks[f"beta_{kind}_vs_{control}"] = rate > .5 and not a.liquidated.any()
        rows.append({"strategy": name, "verdict": "FORWARD PAPER CANDIDATE" if all(checks.values()) else "RESEARCH ONLY",
            "holdout_balance": h.final_balance, "holdout_sharpe": h.sharpe,
            "holdout_dd_pct": h.max_drawdown_pct, "fills_per_day": h.fills_per_day,
            "round_trips_per_day": h.round_trips_per_day, "active_days_pct": h.active_days_pct,
            "fee_stress_balance": get(name, "fee_stress").final_balance,
            "eth_balance": get(name, "eth_holdout").final_balance,
            "funded_balance": get(name, "funded_holdout").final_balance,
            "program_trials_approx": trials, "trial_sharpe_sd": sd, "program_dsr": dsr,
            "failed_checks": ";".join(k for k, ok in checks.items() if not ok), **wins})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "decision.csv", index=False)
    return result


def report():
    cells = pd.read_csv(OUT / "cells.csv")
    assert len(cells) == len(NAMES)*len(specifications())
    evidence = inference(cells)
    decision = decisions(cells, evidence)
    print(decision[["strategy", "verdict", "holdout_balance", "fills_per_day", "round_trips_per_day"]].round(3).to_string(index=False))
    chart(cells, decision)
    integrate(cells, evidence)


def chart(cells, decision):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), layout="constrained")
    palette = dict(zip(CANDIDATES, plt.get_cmap("tab10").colors))
    palette.update(buy_and_hold="black", kelly_regime_v4="#777777")
    for ax, cell, title in ((axes[0, 0], "holdout", "BTC spot · 10bp fee + 1bp slippage"),
                           (axes[0, 1], "funded_holdout", "Deribit perp · 5bp fee + 1bp slippage + funding")):
        for name in NAMES:
            raw = pd.read_csv(OUT / name / "daily.csv")
            data = raw[raw.cell == cell]
            ax.plot(pd.to_datetime(data.timestamp), data.equity, color=palette[name],
                    lw=2 if name in CONTROLS else 1.25, ls="--" if name in CONTROLS else "-", label=name)
        ax.set(title=title, ylabel="Equity from $1,000 (log scale)", yscale="log")
        ax.grid(alpha=.2)
    ranks = decision.sort_values("holdout_balance")
    labels = ranks.strategy.tolist()
    colors = [palette[n] for n in labels]
    axes[1, 0].barh(labels, ranks.holdout_balance, color=colors)
    axes[1, 0].axvline(1000, color="black", lw=1, ls="--")
    axes[1, 0].set(title="Retrospective holdout result · spot", xlabel="Final balance ($)")
    axes[1, 1].barh(labels, ranks.fills_per_day, color=colors, label="Executed fills/day")
    axes[1, 1].plot(ranks.round_trips_per_day, labels, "kx", label="Completed round trips/day")
    axes[1, 1].axvspan(2, 6, color="green", alpha=.08, label="Target: 2–6 fills/day")
    axes[1, 1].set(title="Actual cadence · all calendar days", xlabel="Per day")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=8)
    fig.suptitle("R-189 · ten game-theory candidates · 2023-01-01 to 2026-08-12\nRetrospective research; registration does not imply promotion", fontsize=15)
    fig.savefig(OUT / "candidates.png", dpi=150)
    plt.close(fig)


def integrate(cells, evidence):
    from tradebot.evidence import load_evidence
    from tradebot.report import comparison_report, update_readme
    historical = pd.read_csv(ROOT / "reports/comparison.csv")
    fresh = cells[cells.cell.isin(["full_spot", "full_proxy"])].copy()
    # Verify both controls, including their complete trade count and costs.
    for _, row in fresh[fresh.strategy.isin(CONTROLS)].iterrows():
        old = historical[(historical.strategy == row.strategy) & (historical.market == row.market)].iloc[0]
        for col in ("final_balance", "num_trades", "fees_paid", "max_drawdown_pct"):
            assert np.isclose(row[col], old[col], rtol=1e-7, atol=1e-7), (row.strategy, row.market, col, row[col], old[col])
    historical = historical[~historical.strategy.isin(CANDIDATES)]
    fresh = fresh[fresh.strategy.isin(CANDIDATES)][historical.columns]
    combined = pd.concat([historical, fresh], ignore_index=True)
    evpath = ROOT / "reports/inference/bootstrap.csv"
    old_ev = pd.read_csv(evpath)
    new_ev = evidence[(evidence.period == "full") & evidence.strategy.isin(CANDIDATES)]
    old_ev = old_ev[~((old_ev.period == "full") & old_ev.strategy.isin(CANDIDATES))]
    pd.concat([old_ev, new_ev], ignore_index=True).to_csv(evpath, index=False)
    metrics = []
    for row in combined.to_dict("records"):
        row["data_label"] = row.pop("data")
        metrics.append(Metrics(**row))
    period = "2017-01-01 to 2026-08-12 (1,010,889 x 5m bars)"
    ev = load_evidence(ROOT / "reports")
    # Old adjacent-order tests concern the 27-row roster, so omit that stale
    # statement rather than reusing it against the enlarged 37-row roster.
    comparison_report(metrics, ROOT / "reports", period=period, evidence=ev, ordering={})
    update_readme(metrics, ROOT / "README.md", period=period, evidence=ev, ordering={})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "run":
        run(args.workers)
    else:
        report()
