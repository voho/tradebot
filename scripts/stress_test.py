#!/usr/bin/env python
"""Monte Carlo window stress test for the leading strategies.

The comparison table ranks strategies on ONE path: the full 2017-2026
history. That single number cannot distinguish a robust edge from a lucky
decade. This script resamples many random windows (random start, random
length) and reports the distribution of outcomes.

Design notes that make the comparison fair:

- Every window is preceded by a **warmup prefix** long enough to satisfy
  the slowest strategy (100-day regime anchors). The prefix warms
  indicators and internal state only: trading is disabled until the window
  start (``trade_start``), so short windows are not penalised for a cold
  start *and* every strategy enters the window flat, with the full
  starting balance. Measuring a leveraged strategy that was already
  liquidated inside its own prefix would score a corpse, not the window.
- All strategies see the **identical** window, and buy-and-hold is
  evaluated alongside as the per-window benchmark - the honest question
  is not "did it make money" (BTC rose) but "did it beat holding, and how
  deep did it dig".
- The RNG is seeded, so the window set is reproducible.

Usage::

    python scripts/stress_test.py                 # 40 windows, both markets
    python scripts/stress_test.py --trials 100 --markets spot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import max_drawdown_pct  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.report import BASELINE, GRID, INK, INK_2, MUTED, PAGE, SERIES, SURFACE  # noqa: E402

BARS_PER_DAY = 288
# The top three by final balance, plus the benchmark every strategy must
# beat and the structurally different survivor (champions_council) as a
# contrast — the three leaders are variants of one another, so a council
# built from unrelated members is the useful control.
STRATEGIES = ["kelly_regime_v4", "kelly_regime_v3", "kelly_regime_v2",
              "buy_and_hold", "champions_council"]
BENCHMARK = "buy_and_hold"


def evaluate(name: str, window: pd.DataFrame, eval_start: int,
             market: MarketSpec, balance: float = 1_000.0) -> dict:
    """Backtest over ``window``; the account only opens at ``eval_start``.

    Bars before ``eval_start`` warm the strategy's indicators and internal
    state but cannot trade, so every window measures a **fresh** account
    from the window start. Without that, a leveraged strategy could be
    liquidated inside the prefix and the window would score a corpse.
    """
    result = run_backtest(get_strategy(name), window, market, balance,
                          trade_start=eval_start)
    equity = result.equity.to_numpy(dtype=float)
    base = equity[eval_start]
    if not np.isfinite(base) or base <= 0:
        return {"return_pct": -100.0, "max_dd_pct": 100.0, "trades": 0,
                "liquidated": True}
    seg = equity[eval_start:]
    start_ts = window.index[eval_start]
    return {
        "return_pct": 100.0 * (seg[-1] / base - 1.0),
        "max_dd_pct": max_drawdown_pct(seg),
        "trades": sum(1 for t in result.trades if t.entry_ts >= start_ts),
        "liquidated": result.liquidated,
    }


def run(trials: int, markets: list[str], min_days: int, max_days: int,
        seed: int) -> pd.DataFrame:
    df, label = load_dataset(ROOT / "data", "spot")
    if label == "SYNTHETIC":
        raise SystemExit("real dataset required for the stress test")

    warmup = max(get_strategy(n).warmup for n in STRATEGIES) + 10
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(trials):
        length = int(rng.integers(min_days, max_days + 1) * BARS_PER_DAY)
        start = int(rng.integers(warmup, len(df) - length))
        specs.append((start, length))

    market_specs = {"spot": MarketSpec.spot(),
                    "futures": MarketSpec.futures(leverage=5.0)}
    rows = []
    for k, (start, length) in enumerate(specs, 1):
        window = df.iloc[start - warmup: start + length]
        eval_start = warmup
        print(f"[{k}/{trials}] {window.index[eval_start]:%Y-%m-%d} "
              f"+{length // BARS_PER_DAY}d", file=sys.stderr)
        for market_name in markets:
            for name in STRATEGIES:
                stats = evaluate(name, window, eval_start, market_specs[market_name])
                rows.append({
                    "trial": k, "market": market_name, "strategy": name,
                    "start": window.index[eval_start], "days": length // BARS_PER_DAY,
                    **stats,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- reporting

def summarize(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (market, name), grp in res.groupby(["market", "strategy"], sort=False):
        bench = res[(res.market == market) & (res.strategy == BENCHMARK)] \
            .set_index("trial")["return_pct"]
        beat = (grp.set_index("trial")["return_pct"] > bench).mean() * 100.0
        out.append({
            "market": market, "strategy": name,
            "median return %": grp["return_pct"].median(),
            "mean return %": grp["return_pct"].mean(),
            "profitable %": (grp["return_pct"] > 0).mean() * 100.0,
            "beat hold %": beat,
            "worst %": grp["return_pct"].min(),
            "best %": grp["return_pct"].max(),
            "median maxDD %": grp["max_dd_pct"].median(),
            "worst maxDD %": grp["max_dd_pct"].max(),
            "liquidated %": grp["liquidated"].mean() * 100.0,
        })
    return pd.DataFrame(out)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)


def charts(res: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {n: SERIES[i] for i, n in enumerate(STRATEGIES)}
    paths = []

    for market in res["market"].unique():
        sub = res[res.market == market]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        fig.patch.set_facecolor(PAGE)

        # 1. return distribution per strategy
        ax = axes[0]
        _style(ax)
        data = [sub[sub.strategy == n]["return_pct"].to_numpy() for n in STRATEGIES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                        medianprops=dict(color=INK, linewidth=2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5,
                                        markerfacecolor=MUTED, markeredgecolor="none"))
        for patch, n in zip(bp["boxes"], STRATEGIES):
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.35)
            patch.set_edgecolor(colors[n])
        for element in ("whiskers", "caps"):
            for item in bp[element]:
                item.set_color(BASELINE)
        ax.axhline(0.0, color=BASELINE, linewidth=1)
        ax.set_xticklabels([n.replace("_", "\n") for n in STRATEGIES], fontsize=8)
        ax.set_ylabel("window return %", color=MUTED, fontsize=9)
        ax.set_title("Return distribution", color=INK, fontsize=10, loc="left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # 2. head-to-head vs buy-and-hold
        ax = axes[1]
        _style(ax)
        bench = sub[sub.strategy == BENCHMARK].set_index("trial")["return_pct"]
        for n in STRATEGIES:
            if n == BENCHMARK:
                continue
            s = sub[sub.strategy == n].set_index("trial")["return_pct"]
            ax.scatter(bench.loc[s.index], s, s=42, color=colors[n], alpha=0.75,
                       edgecolors=SURFACE, linewidths=1.5, label=n, zorder=3)
        lim = [min(sub["return_pct"].min(), 0) * 1.05, sub["return_pct"].max() * 1.05]
        ax.plot(lim, lim, color=BASELINE, linewidth=1.5, zorder=2)
        ax.annotate("above line = beat holding", (lim[1], lim[1]), fontsize=7,
                    color=MUTED, ha="right", va="bottom")
        # On leverage a surviving hold can return +1,800% while a liquidated
        # one returns -98%. On a linear axis that range crushes every point
        # into the left edge, so the panel shows nothing. symlog keeps the
        # diagonal meaningful and stays linear through zero.
        if sub["return_pct"].max() > 500:
            ax.set_xscale("symlog", linthresh=100)
            ax.set_yscale("symlog", linthresh=100)
        ax.set_xlabel("buy_and_hold return %", color=MUTED, fontsize=9)
        ax.set_ylabel("strategy return %", color=MUTED, fontsize=9)
        ax.set_title("Head to head, same window", color=INK, fontsize=10, loc="left")
        ax.legend(loc="upper left", fontsize=8, labelcolor=INK_2, frameon=True,
                  facecolor=SURFACE, edgecolor="none", framealpha=0.85)

        # 3. drawdown distribution
        ax = axes[2]
        _style(ax)
        for n in STRATEGIES:
            vals = np.sort(sub[sub.strategy == n]["max_dd_pct"].to_numpy())
            ax.plot(vals, np.linspace(0, 100, len(vals)), color=colors[n],
                    linewidth=2, label=n, solid_capstyle="round")
        ax.set_xlabel("max drawdown % in window", color=MUTED, fontsize=9)
        ax.set_ylabel("% of windows below", color=MUTED, fontsize=9)
        ax.set_title("Drawdown, cumulative", color=INK, fontsize=10, loc="left")
        ax.legend(loc="lower right", fontsize=8, labelcolor=INK_2, frameon=True,
                  facecolor=SURFACE, edgecolor="none", framealpha=0.85)

        n_trials = sub["trial"].nunique()
        fig.suptitle(f"Stress test · {market} · {n_trials} random windows "
                     f"({sub['days'].min()}-{sub['days'].max()} days)",
                     color=INK, fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = out_dir / f"stress_{market}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=PAGE)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--markets", nargs="+", default=["spot", "futures"],
                    choices=["spot", "futures"])
    ap.add_argument("--min-days", type=int, default=90)
    ap.add_argument("--max-days", type=int, default=730)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "stress")
    args = ap.parse_args()

    res = run(args.trials, args.markets, args.min_days, args.max_days, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "stress_results.csv", index=False)

    summary = summarize(res)
    summary.to_csv(args.out / "stress_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(1).to_string(index=False))
    for path in charts(res, args.out):
        print(f"chart: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
