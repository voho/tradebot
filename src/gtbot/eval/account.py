"""Concrete account outcomes: what a real deposit actually does.

Sharpe ratios and CAGRs hide two things that decide whether a strategy is
usable: how many dollars come back, and whether the account survives the path.
This module answers both for a specified deposit and leverage, and it is wired
into every reporting surface so the answer is always on screen rather than left
as an exercise.

Leverage is applied as a cap on gross exposure (a position may reach
``leverage`` times equity), which is how a perpetual-futures account behaves
when you set it in the UI. Two consequences are reported explicitly:

* **Returns scale roughly linearly with leverage, and so does drawdown.**  The
  strategy's edge per unit of exposure is unchanged by leverage; only the
  quantity of it changes.  Sharpe is therefore roughly invariant, which is
  exactly why quoting only a return number is misleading.
* **Liquidation is a real outcome.**  At 5x, a ~19.5% adverse move ends the
  account, and :mod:`gtbot.engine.backtest` closes the position when a bar's
  adverse extreme reaches it.  ``margin_use`` reports how close the path came:
  0.25 means the worst bar consumed a quarter of the distance to liquidation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

from ..data.schema import BTCUSD_5M, BarSpec
from ..engine.backtest import run_backtest
from ..engine.broker import CostModel, ExecutionConfig, MarginConfig
from ..risk import RiskConfig
from ..strategy import GameTheoreticStrategy, StrategyConfig
from . import metrics

#: The account the user asked about.  Kept as module constants so every report
#: quotes the same thing.
DEFAULT_DEPOSIT = 1_000.0
DEFAULT_LEVERAGE = 5.0

DIRECTIONS: dict[str, str] = {
    "both": "long/short",
    "long_only": "long-only",
}


@dataclass
class AccountResult:
    """What happened to one deposit under one configuration."""

    direction: str
    direction_label: str
    sizing_mode: str
    tier: str
    deposit: float
    leverage: float
    years: float
    final_equity: float
    profit_usd: float
    return_pct: float
    cagr: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    worst_bar_usd: float
    ann_vol: float
    sharpe: float
    trades: int
    trades_per_year: float
    #: Fraction of bars holding a position.  This strategy is flat the great
    #: majority of the time, so a leverage *cap* of 5x translates into a much
    #: smaller time-averaged exposure — which is why realised volatility stays
    #: far below what "5x" suggests.
    time_in_market: float
    #: Mean gross exposure while in a trade, as a multiple of equity.
    avg_position_when_in: float
    fees_paid_usd: float
    liquidated: bool
    margin_use: float

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_account(
    bars: pd.DataFrame,
    *,
    tier: str = "vip6",
    direction: str = "both",
    leverage: float = DEFAULT_LEVERAGE,
    deposit: float = DEFAULT_DEPOSIT,
    sizing_mode: str = "robust",
    config: StrategyConfig | None = None,
    execution: ExecutionConfig | None = None,
    margin: MarginConfig | None = None,
    spec: BarSpec = BTCUSD_5M,
) -> AccountResult:
    """Run the strategy on one deposit and report the account outcome."""
    if direction not in DIRECTIONS:
        raise KeyError(f"unknown direction {direction!r}; known: {sorted(DIRECTIONS)}")

    execution = execution or ExecutionConfig(entry_mode="taker", exit_mode="maker", ttl_bars=1)
    cost = CostModel.for_tier(tier)
    base = config or StrategyConfig()
    cfg = StrategyConfig(**{**base.__dict__})
    cfg.direction = direction
    cfg.sizing_mode = sizing_mode
    cfg.assumed_cost_bp = cost.round_trip_bp(execution)
    # The account's leverage is the single source of truth: the risk layer's own
    # cap must match it, or it silently clamps below the level being reported.
    cfg.risk = replace(base.risk, max_leverage=leverage)

    strategy = GameTheoreticStrategy(cfg, spec=spec)
    res = run_backtest(
        bars,
        strategy,
        costs=cost,
        execution=execution,
        spec=spec,
        initial_equity=deposit,
        max_leverage=leverage,
        margin=margin,
    )
    warmup = strategy.warmup
    m = metrics.compute(
        res.returns, res.equity, res.position, res.costs,
        bars_per_year=spec.bars_per_year, n_trades=res.n_trades,
    )

    active = np.abs(res.position) > 1e-12
    final = float(res.equity[-1])
    peak = np.maximum.accumulate(res.equity)
    dd_usd = float(np.max(peak - res.equity))

    # ``returns[i]`` is earned on the equity *entering* bar i, and ``equity[i]``
    # is already post-return, so dollar figures must use the previous bar's
    # equity.  Using equity[i] understates fees on every bar and — because a
    # liquidation bar ends at zero equity — reports the bar that destroyed the
    # account as costing exactly $0.
    prev_equity = np.empty_like(res.equity)
    prev_equity[0] = deposit
    prev_equity[1:] = res.equity[:-1]
    bar_pnl = res.returns * prev_equity
    worst_bar = float(np.min(bar_pnl)) if bar_pnl.size else 0.0
    fees_usd = float(np.sum(res.costs * prev_equity))

    # Warm-up bars are flat by construction, not by choice; counting them would
    # understate how much of the tradeable series is actually spent in a position.
    tradeable = max(len(res.position) - warmup, 1)

    return AccountResult(
        direction=direction,
        direction_label=DIRECTIONS[direction],
        sizing_mode=sizing_mode,
        tier=tier,
        deposit=deposit,
        leverage=leverage,
        years=m.years,
        final_equity=final,
        profit_usd=final - deposit,
        return_pct=final / deposit - 1.0,
        cagr=m.cagr,
        max_drawdown_pct=m.max_drawdown,
        max_drawdown_usd=dd_usd,
        worst_bar_usd=worst_bar,
        ann_vol=m.ann_vol,
        sharpe=m.sharpe,
        trades=res.n_trades,
        trades_per_year=res.n_trades / max(m.years, 1e-9),
        time_in_market=float(active.sum() / max(tradeable, 1)),
        avg_position_when_in=float(np.abs(res.position[active]).mean()) if active.any() else 0.0,
        fees_paid_usd=fees_usd,
        liquidated=res.liquidated_at is not None,
        margin_use=res.worst_margin_use,
    )


def format_table(results: list[AccountResult], *, markdown: bool = False) -> str:
    """Render account outcomes as a table."""
    if not results:
        return "(no results)"
    dep = results[0].deposit
    lev = results[0].leverage
    yrs = results[0].years

    head = [
        "mode", "sizing", "tier", "final $", "P&L $", "return", "CAGR",
        "max DD $", "worst bar $", "trades/yr", "in mkt", "fees $", "liq?",
    ]
    rows = []
    for r in results:
        rows.append([
            r.direction_label,
            r.sizing_mode,
            r.tier,
            f"{r.final_equity:,.0f}",
            f"{r.profit_usd:+,.0f}",
            f"{r.return_pct:+.1%}",
            f"{r.cagr:+.1%}",
            f"{r.max_drawdown_usd:,.0f}",
            f"{r.worst_bar_usd:+,.0f}",
            f"{r.trades_per_year:,.0f}",
            f"{r.time_in_market:.1%}",
            f"{r.fees_paid_usd:,.0f}",
            "YES" if r.liquidated else "no",
        ])

    title = f"${dep:,.0f} deposit at {lev:g}x leverage over {yrs:.2f} years"
    if markdown:
        out = [f"**{title}**", "", "| " + " | ".join(head) + " |",
               "|" + "|".join(["---"] * len(head)) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)

    widths = [max(len(head[i]), max(len(r[i]) for r in rows)) for i in range(len(head))]
    lines = [title, "  ".join(h.rjust(w) for h, w in zip(head, widths))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(lines)


def run_account_sweep(
    bars: pd.DataFrame,
    *,
    tiers: list[str],
    directions: list[str] | None = None,
    sizing_modes: list[str] | None = None,
    leverage: float = DEFAULT_LEVERAGE,
    deposit: float = DEFAULT_DEPOSIT,
    config: StrategyConfig | None = None,
    execution: ExecutionConfig | None = None,
    spec: BarSpec = BTCUSD_5M,
) -> list[AccountResult]:
    """Every (direction, tier) combination for one deposit."""
    directions = directions or list(DIRECTIONS)
    sizing_modes = sizing_modes or ["robust"]
    return [
        simulate_account(
            bars, tier=tier, direction=d, sizing_mode=sm, leverage=leverage,
            deposit=deposit, config=config, execution=execution, spec=spec,
        )
        for sm in sizing_modes
        for d in directions
        for tier in tiers
    ]
