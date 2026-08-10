"""Markdown reporting.

The report deliberately leads with the fee tier and the cost sensitivity rather
than with a single Sharpe ratio.  At 5-minute frequency the gross edge is single
digit basis points per trade, so quoting one performance number without the cost
assumption that produced it is close to meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import replace

from ..data.schema import BTCUSD_5M, BarSpec
from ..engine.backtest import run_backtest
from ..engine.broker import FEE_TIERS, CostModel, ExecutionConfig
from ..risk import RiskConfig
from ..strategy import GameTheoreticStrategy, StrategyConfig
from . import metrics, stats
from .account import DEFAULT_DEPOSIT, DEFAULT_LEVERAGE, DIRECTIONS, format_table, simulate_account


def _run(bars, tier, execution, max_leverage):
    cost = CostModel.for_tier(tier)
    cfg = StrategyConfig(assumed_cost_bp=cost.round_trip_bp(execution))
    # Keep the risk layer's cap in step with the requested one, or the table
    # below would be produced at 2x while the header claims otherwise.
    cfg.risk = replace(cfg.risk, max_leverage=max_leverage)
    res = run_backtest(bars, GameTheoreticStrategy(cfg), costs=cost,
                       execution=execution, max_leverage=max_leverage)
    m = metrics.compute(res.returns, res.equity, res.position, res.costs,
                        bars_per_year=BTCUSD_5M.bars_per_year, n_trades=res.n_trades)
    gross = metrics.compute(res.gross_returns, np.cumprod(1.0 + res.gross_returns),
                            res.position, res.costs * 0,
                            bars_per_year=BTCUSD_5M.bars_per_year)
    return res, m, gross


def render_report(
    bars: pd.DataFrame,
    *,
    tiers: list[str] | None = None,
    execution: ExecutionConfig | None = None,
    max_leverage: float = 2.0,
    leverage: float = DEFAULT_LEVERAGE,
    deposit: float = DEFAULT_DEPOSIT,
    spec: BarSpec = BTCUSD_5M,
) -> str:
    """Render a full markdown report for one bar series."""
    tiers = tiers or list(FEE_TIERS)
    execution = execution or ExecutionConfig(entry_mode="taker", exit_mode="maker", ttl_bars=1)

    years = len(bars) / spec.bars_per_year
    lines: list[str] = []
    lines.append("# Backtest report\n")
    lines.append(f"- bars: **{len(bars):,}** (~{years:.2f} years)")
    lines.append(f"- execution: **{execution.entry_mode} in / {execution.exit_mode} out**")
    lines.append(f"- max leverage: **{max_leverage}x**\n")

    lines.append("## Sensitivity to the fee tier\n")
    lines.append("| tier | round trip | net Sharpe | gross Sharpe | CAGR | vol | max DD | trades/yr | cost drag |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    best_returns = None
    best_sharpe = -np.inf
    for tier in tiers:
        res, m, gross = _run(bars, tier, execution, max_leverage)
        rt = CostModel.for_tier(tier).round_trip_bp(execution)
        lines.append(
            f"| {tier} | {rt:.2f} bp | {m.sharpe:+.2f} | {gross.sharpe:+.2f} | {m.cagr:+.2%} | "
            f"{m.ann_vol:.2%} | {m.max_drawdown:.2%} | {m.n_trades / max(years, 1e-9):.0f} | "
            f"{m.cost_drag_annual:.2%} |"
        )
        if m.sharpe > best_sharpe:
            best_sharpe, best_returns, best_tier = m.sharpe, res.returns, tier

    lines.append("")
    lines.append(f"## What ${deposit:,.0f} at {leverage:g}x actually does\n")
    accounts = [
        simulate_account(bars, tier=tier, direction=d, sizing_mode=sm,
                         leverage=leverage, deposit=deposit,
                         execution=execution, spec=spec)
        for sm in ("robust", "fixed")
        for d in DIRECTIONS
        for tier in tiers
    ]
    lines.append(format_table(accounts, markdown=True))
    liq = [a for a in accounts if a.liquidated]
    if liq:
        lines.append(f"\n**{len(liq)} of {len(accounts)} configurations were liquidated.**")
    else:
        lines.append(
            f"\nNo liquidations; the worst bar consumed "
            f"{max(a.margin_use for a in accounts):.1%} of the distance to liquidation."
        )
    lines.append("")

    if best_returns is not None:
        st = stats.summarise(best_returns, bars_per_year=spec.bars_per_year)
        lines.append(f"## Statistics (best tier: {best_tier})\n")
        lines.append(f"- annualised Sharpe: **{st['sharpe']:+.2f}**")
        lines.append(
            f"- block-bootstrap 95% CI: [{st['sharpe_ci95'][0]:+.2f}, {st['sharpe_ci95'][1]:+.2f}]"
        )
        lines.append(f"- bootstrap p(Sharpe <= 0): **{st['bootstrap_p_value']:.4f}**")
        lines.append(f"- Newey-West t-statistic: **{st['newey_west_t']:+.2f}**")
        lines.append(f"- return skew: {st['skew']:+.2f}\n")

    lines.append(
        "> Net Sharpe falls sharply with the fee tier because the gross edge at this "
        "frequency is only a few basis points per trade.  Read any single performance "
        "number together with the round-trip cost that produced it."
    )
    return "\n".join(lines)
