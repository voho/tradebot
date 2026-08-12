"""Performance metrics for a backtest run. Final balance is the primary
comparison criterion; the rest explains how a strategy got there."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tradebot.engine import BacktestResult

BARS_PER_YEAR = 365.25 * 24 * 12  # 5m bars


@dataclass
class Metrics:
    strategy: str
    market: str
    start_balance: float
    final_balance: float
    profit: float
    profit_pct: float
    num_trades: int
    win_rate_pct: float
    best_trade: float
    worst_trade: float
    avg_trade: float
    max_drawdown_pct: float
    sharpe: float
    time_in_market_pct: float
    fees_paid: float
    liquidated: bool
    data_label: str

    def as_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "market": self.market,
            "start_balance": self.start_balance,
            "final_balance": self.final_balance,
            "profit": self.profit,
            "profit_pct": self.profit_pct,
            "num_trades": self.num_trades,
            "win_rate_pct": self.win_rate_pct,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "avg_trade": self.avg_trade,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe": self.sharpe,
            "time_in_market_pct": self.time_in_market_pct,
            "fees_paid": self.fees_paid,
            "liquidated": self.liquidated,
            "data": self.data_label,
        }


def max_drawdown_pct(equity: np.ndarray) -> float:
    """Largest peak-to-trough equity drop, as a percentage of the peak."""
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    return float(np.nanmax(dd) * 100.0)


def sharpe_ratio(equity: np.ndarray) -> float:
    """Annualized Sharpe of per-bar equity returns (rf = 0)."""
    if len(equity) < 3:
        return 0.0
    prev = equity[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(equity) / prev, 0.0)
    sd = rets.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(rets.mean() / sd * np.sqrt(BARS_PER_YEAR))


def compute_metrics(result: BacktestResult) -> Metrics:
    equity = result.equity.to_numpy(dtype=float)
    final = float(equity[-1]) if len(equity) else result.start_balance
    pnls = [t.pnl for t in result.trades]
    closed = [t for t in result.trades if not t.open_at_end]
    wins = sum(1 for t in closed if t.pnl > 0)

    # bars where a position was open: between entry and exit fills
    in_market = 0
    if len(result.equity) and result.trades:
        idx = result.equity.index
        for t in result.trades:
            start = idx.searchsorted(t.entry_ts)
            end = idx.searchsorted(t.exit_ts) if t.exit_ts is not None else len(idx)
            in_market += max(0, end - start)
    time_in_market = 100.0 * in_market / max(1, len(result.equity))

    return Metrics(
        strategy=result.strategy_name,
        market=result.market.name,
        start_balance=result.start_balance,
        final_balance=final,
        profit=final - result.start_balance,
        profit_pct=100.0 * (final / result.start_balance - 1.0),
        num_trades=len(result.trades),
        win_rate_pct=100.0 * wins / len(closed) if closed else 0.0,
        best_trade=max(pnls) if pnls else 0.0,
        worst_trade=min(pnls) if pnls else 0.0,
        avg_trade=float(np.mean(pnls)) if pnls else 0.0,
        max_drawdown_pct=max_drawdown_pct(equity),
        sharpe=sharpe_ratio(equity),
        time_in_market_pct=min(100.0, time_in_market),
        fees_paid=result.fees_paid,
        liquidated=result.liquidated,
        data_label=result.data_label,
    )
