"""Run the full comparison matrix: strategies x markets x start balances."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from tradebot import data as datamod
from tradebot.broker import MarketSpec
from tradebot.engine import BacktestResult, run_backtest
from tradebot.metrics import Metrics, compute_metrics, max_drawdown_pct as _bar_max_drawdown_pct
from tradebot.multi_engine import align_frames as _ma_align_frames
from tradebot.multi_engine import load_universe as _ma_load_universe
from tradebot.multi_engine import static_hold_equity as _ma_static_hold_equity
from tradebot.multi_strategy import (
    DEFAULT_WINDOW as MULTI_ASSET_WINDOW,
    available_multi_asset_strategies,
    run_multi_asset_backtest,
)
from tradebot.registry import available_strategies, get_strategy
from tradebot.evidence import load_evidence, ordering_counts
from tradebot.report import (
    comparison_report,
    overlay_chart,
    print_comparison,
    run_chart,
    update_readme,
    update_readme_multi_asset,
)


def run_multi_asset_matrix(data_dir: Path, out_dir: Path,
                           spot_fee: float = 0.001,
                           balance: float = 1_000.0) -> list[dict]:
    """Run every registered multi-asset strategy (backlog B-32), spot only.

    Additive and independent of the single-asset matrix above: an empty
    ``available_multi_asset_strategies()`` makes this a no-op, so a run
    with no multi-asset strategy registered produces exactly the rows this
    function always would have produced -- none -- and touches nothing
    downstream. Futures are out of scope for this axis (see
    ``tradebot.multi_engine``'s module docstring): a levered multi-asset
    book needs a shared-margin/liquidation model this codebase does not
    have.
    """
    strategies = available_multi_asset_strategies()
    rows: list[dict] = []
    market = MarketSpec.spot(fee_rate=spot_fee)
    for name, cls in sorted(strategies.items()):
        strategy = cls()
        print(f"running multi-asset {name} on portfolio with {balance:,.0f} USD ...",
              file=sys.stderr)
        eq = run_multi_asset_backtest(strategy, data_dir, market, balance,
                                      window=MULTI_ASSET_WINDOW)

        frames = _ma_load_universe(strategy.instruments, data_dir)
        aligned = _ma_align_frames(frames, MULTI_ASSET_WINDOW)
        ew = _ma_static_hold_equity(aligned, strategy.instruments, market,
                                    start_balance=balance)

        rows.append({
            "name": name,
            "instruments": list(strategy.instruments),
            "description": strategy.describe(),
            "start_balance": balance,
            "final_balance": float(eq.iloc[-1]),
            "max_drawdown_pct": _bar_max_drawdown_pct(eq.to_numpy(dtype=float)),
            "ew_final_balance": float(ew.iloc[-1]),
        })
    return rows


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

    # Multi-asset strategies (backlog B-32): additive, after and independent
    # of everything above. Runs on the same "full, real-data, untrimmed"
    # gate as the single-asset README update; a no-op when
    # `available_multi_asset_strategies()` is empty, so single-asset
    # behaviour is unaffected whether or not this step runs at all.
    if full_run and real_data and not cfg.max_bars:
        ma_rows = run_multi_asset_matrix(cfg.data_dir, cfg.out_dir,
                                         spot_fee=cfg.spot_fee,
                                         balance=cfg.balances[0])
        if ma_rows and update_readme_multi_asset(ma_rows, cfg.readme,
                                                  period=f"{MULTI_ASSET_WINDOW[0]} to last bar"):
            print(f"updated multi-asset section in {cfg.readme}", file=sys.stderr)

    return all_metrics, all_results
