"""Run the full comparison matrix: strategies x markets x start balances."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from tradebot import data as datamod
from tradebot.broker import MarketSpec
from tradebot.engine import BacktestResult, run_backtest
from tradebot.metrics import Metrics, compute_metrics
from tradebot.registry import available_strategies, get_strategy
from tradebot.evidence import load_evidence, ordering_counts
from tradebot.report import (
    comparison_report,
    overlay_chart,
    print_comparison,
    run_chart,
    update_readme,
)


@dataclass
class RunConfig:
    data_dir: Path = Path("data")
    out_dir: Path = Path("reports")
    # One start balance is enough: results are proportional to capital
    # (verified across all strategies; the only exceptions came from the
    # exchange minimum order size). Pass --balances to test more.
    balances: list[float] = field(default_factory=lambda: [1_000.0])
    markets: list[str] = field(default_factory=lambda: ["spot", "futures"])
    leverage: float = 5.0
    spot_fee: float = 0.001
    futures_fee: float = 0.0005
    slippage_bps: float = 0.0
    strategies: list[str] | None = None  # None = all registered
    max_bars: int | None = None  # optionally trim to the most recent N bars
    readme: Path = Path("README.md")  # comparison table target (markers inside)

    def market_specs(self) -> list[tuple[MarketSpec, str]]:
        """(spec, data kind) pairs; spot trades spot data, futures the perp."""
        out = []
        for m in self.markets:
            if m == "spot":
                out.append((MarketSpec.spot(fee_rate=self.spot_fee), "spot"))
            elif m == "futures":
                out.append((MarketSpec.futures(leverage=self.leverage,
                                               fee_rate=self.futures_fee), "perp"))
            else:
                raise ValueError(f"unknown market {m!r} (use 'spot' or 'futures')")
        return out


def run_matrix(cfg: RunConfig) -> tuple[list[Metrics], list[BacktestResult]]:
    names = cfg.strategies or sorted(available_strategies())
    specs = cfg.market_specs()

    datasets: dict[str, tuple] = {}
    for _, kind in specs:
        if kind not in datasets:
            df, label = datamod.load_dataset(cfg.data_dir, kind)
            if cfg.max_bars:
                df = df.iloc[-cfg.max_bars:]
            datasets[kind] = (df, label)

    all_metrics: list[Metrics] = []
    all_results: list[BacktestResult] = []
    charts_dir = Path(cfg.out_dir) / "charts"

    for spec, kind in specs:
        df, label = datasets[kind]
        for balance in cfg.balances:
            group_results: list[BacktestResult] = []
            for name in names:
                strategy = get_strategy(name)  # fresh instance per run
                print(f"running {name} on {spec.name} with {balance:,.0f} USD ...",
                      file=sys.stderr)
                result = run_backtest(
                    strategy, df, spec, balance,
                    slippage_bps=cfg.slippage_bps, data_label=label,
                )
                metrics = compute_metrics(result)
                run_chart(result, metrics,
                          charts_dir / f"{name}__{spec.name}__{balance:g}.png")
                all_metrics.append(metrics)
                group_results.append(result)
            title = (f"balance curves · {spec.name} · start {balance:,.0f} USD"
                     + ("" if label == "real" else f" · [{label} data]"))
            overlay_chart(group_results,
                          title,
                          charts_dir / f"_all__{spec.name}__{balance:g}.png")
            all_results.extend(group_results)

    first_df = next(iter(datasets.values()))[0]
    period = (f"{first_df.index[0]:%Y-%m-%d} to {first_df.index[-1]:%Y-%m-%d} "
              f"({len(first_df):,} x 5m bars)")
    # R-29's intervals, if they are on disk (backlog B-12). They describe
    # the full 2017-2026 history, so a trimmed run gets none rather than a
    # mismatched one; the per-market keys likewise only match a 5x futures
    # run, because that is the leverage they were measured at.
    evidence: dict = {}
    ordering: dict = {}
    if not cfg.max_bars:
        evidence = load_evidence(cfg.out_dir)
        ordering = ordering_counts(cfg.out_dir)
        if not evidence:
            print("no bootstrap intervals found under "
                  f"{Path(cfg.out_dir) / 'inference'}; the comparison table "
                  "will print point estimates only (run scripts/inference.py)",
                  file=sys.stderr)

    md = comparison_report(all_metrics, cfg.out_dir, period=period,
                           evidence=evidence, ordering=ordering)
    print_comparison(all_metrics)
    print(f"\nreport: {md}\ncharts: {charts_dir}/", file=sys.stderr)

    # The README table must always cover every registered strategy on the
    # full matrix (CI enforces it), so partial or synthetic runs skip it.
    full_run = set(names) >= set(available_strategies())
    real_data = all(m.data_label != "SYNTHETIC" for m in all_metrics)
    if full_run and real_data and not cfg.max_bars:
        if update_readme(all_metrics, cfg.readme, period=period,
                         evidence=evidence, ordering=ordering):
            print(f"updated comparison table in {cfg.readme}", file=sys.stderr)
    else:
        print("README comparison not updated (partial/synthetic/trimmed run)",
              file=sys.stderr)
    return all_metrics, all_results
