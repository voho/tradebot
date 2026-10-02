"""Render R191's saved results with readable labels; no financial evaluations.

This post-result presentation pass leaves the frozen decision/report source
unchanged. Run after r191_report.py report; all plotted values come from its
committed daily returns and bootstrap.csv.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r191_report import OUT, read
from r191_strategies import PRIMARY_NAMES

SHORT = dict(zip(PRIMARY_NAMES, ("Proximal allocation", "Downside budget",
    "Cost-gated reversion", "Sticky experts", "Robust growth")))
SHORT.update(buy_and_hold="Passive holding (1×)", kelly_regime_v4="Kelly v4")


def main():
    _, daily = read("holdout")
    boot = pd.read_csv(OUT / "bootstrap.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, cell, title in zip(axes[0], ("holdout", "funded_holdout"),
            ("BTC spot · 40bp fee + 1bp slippage", "BTC perpetual · 5bp + 1bp + funding")):
        for name in (*PRIMARY_NAMES, "buy_and_hold", "kelly_regime_v4"):
            r = daily[name, cell]
            ax.plot(pd.to_datetime(r.index), 1000 * (1 + r).cumprod(),
                label=SHORT[name], lw=1.6, ls="-" if name in PRIMARY_NAMES else "--")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set(title=title, ylabel="Account value ($), log scale", yscale="log")
        ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    for ax, cell, title in zip(axes[1], ("holdout", "funded_holdout"),
            ("Spot · difference versus matched holding", "Funded · difference versus matched holding")):
        b = boot[boot.cell == cell].set_index("strategy").loc[list(PRIMARY_NAMES)]
        y = np.arange(len(b))
        ax.hlines(y, b.d_sharpe_lo, b.d_sharpe_hi, color="tab:blue")
        ax.scatter(b.d_sharpe, y, color="tab:blue")
        for i, valid in enumerate(b.risk_valid):
            if not valid:
                ax.plot(b.d_sharpe.iloc[i], i, "rx", ms=10)
        ax.axvline(0, color="gray", lw=1)
        ax.axvline(.20, color="green", ls="--", lw=1)
        ax.set(yticks=y, yticklabels=[SHORT[n] for n in PRIMARY_NAMES],
            title=title, xlabel="Annualized daily Sharpe difference · paired 95% interval")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("R-191 · Five cost-aware allocation strategies", fontsize=17)
    fig.supxlabel("2023–2026 holdout · $1,000 start · green line: +0.20 hurdle · red ×: invalid risk match", fontsize=10)
    fig.savefig(OUT / "summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
