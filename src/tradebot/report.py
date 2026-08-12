"""Charts and comparison tables.

Every run gets one PNG: price with trade markers, equity (balance) curve
vs a hold benchmark, and drawdown, plus a results box. Every
(market, balance) group gets an overlay chart of all strategies' equity
curves, and the whole matrix lands in one comparison table
(markdown + CSV + console) sorted by final balance.

Styling follows the validated reference dataviz palette (light mode).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

import inspect  # noqa: E402
import os  # noqa: E402

from tradebot.engine import BacktestResult  # noqa: E402
from tradebot.metrics import Metrics  # noqa: E402

# --- reference palette (light mode) -----------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # fixed order, never cycled
GOOD = "#0ca30c"      # buy marker (with ▲ shape carrying the meaning)
CRITICAL = "#d03b3b"  # sell marker / drawdown


# Cap plotted points per line so decade-long 5m series render fast without
# hiding spikes: decimation keeps each bucket's min and max in time order.
MAX_PLOT_POINTS = 60_000
MAX_MARKERS = 6_000


def _decimate(idx, values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= MAX_PLOT_POINTS:
        return idx, values
    buckets = MAX_PLOT_POINTS // 2
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    keep: list[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if a == b:
            continue
        seg = values[a:b]
        lo, hi = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        keep.extend(sorted({lo, hi}))
    keep_arr = np.array(keep)
    return idx[keep_arr], values[keep_arr]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0, alpha=1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)


def _legend(ax) -> None:
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_2, frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.85)


def _mmoney(x: float) -> str:
    """_money escaped for matplotlib text ($..$ would trigger mathtext)."""
    return _money(x).replace("$", r"\$")


def _money(x: float) -> str:
    neg = x < 0
    a = abs(x)
    # pick the unit off the ROUNDED value so 999,950 -> $1.00M, not $1000.0K
    if round(a / 1e9, 2) >= 1.0:
        s = f"${a / 1e9:,.2f}B"
    elif round(a / 1e6, 2) >= 1.0:
        s = f"${a / 1e6:,.2f}M"
    elif round(a / 1e3, 1) >= 10.0:
        s = f"${a / 1e3:,.1f}K"
    else:
        s = f"${a:,.2f}" if a < 100 else f"${a:,.0f}"
    return f"-{s}" if neg else s


def run_chart(result: BacktestResult, metrics: Metrics, path: str | Path) -> Path:
    """One PNG per run: price + trades, balance curve, drawdown, results box."""
    df = result.df
    equity = result.equity
    idx = df.index

    fig, (ax_p, ax_e, ax_d) = plt.subplots(
        3, 1, figsize=(12, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.2, 1.0], "hspace": 0.12},
    )
    fig.patch.set_facecolor(PAGE)

    # -- panel 1: price with trade markers
    _style_axes(ax_p)
    ax_p.plot(*_decimate(idx, df["close"]), color=INK_2, linewidth=1.3,
              solid_joinstyle="round", solid_capstyle="round", zorder=2)
    fills = result.fills
    shown = fills
    if len(fills) > MAX_MARKERS:
        stride = -(-len(fills) // MAX_MARKERS)
        shown = fills[::stride]
        ax_p.text(0.995, 0.97, f"showing {len(shown):,} of {len(fills):,} trade markers",
                  transform=ax_p.transAxes, fontsize=7, color=INK_2,
                  va="top", ha="right")
    buys_t, buys_p, sells_t, sells_p = [], [], [], []
    for f in shown:
        (buys_t if f.side.name == "BUY" else sells_t).append(f.ts)
        (buys_p if f.side.name == "BUY" else sells_p).append(f.price)
    ax_p.scatter(buys_t, buys_p, marker="^", s=64, color=GOOD,
                 edgecolors=SURFACE, linewidths=2, zorder=3, label="buy")
    ax_p.scatter(sells_t, sells_p, marker="v", s=64, color=CRITICAL,
                 edgecolors=SURFACE, linewidths=2, zorder=3, label="sell")
    ax_p.set_ylabel("price (USD)", color=MUTED, fontsize=9)
    _legend(ax_p)

    # -- panel 2: balance curve vs hold benchmark
    _style_axes(ax_e)
    hold = result.start_balance * df["close"] / float(df["close"].iloc[0])
    ax_e.plot(*_decimate(idx, hold), color=BASELINE, linewidth=2,
              solid_joinstyle="round", solid_capstyle="round",
              label="hold benchmark (1x)", zorder=2)
    ax_e.plot(*_decimate(idx, equity), color=SERIES[0], linewidth=2,
              solid_joinstyle="round", solid_capstyle="round",
              label="strategy balance", zorder=3)
    lo = float(min(equity.min(), hold.min()))
    hi = float(max(equity.max(), hold.max()))
    if lo > 0 and hi / lo > 50:
        ax_e.set_yscale("log")
        ax_e.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_e.set_ylabel("balance (USD)", color=MUTED, fontsize=9)
    _legend(ax_e)

    box = (
        f"final balance  {_mmoney(metrics.final_balance)}\n"
        f"profit         {_mmoney(metrics.profit)} ({metrics.profit_pct:+.1f}%)\n"
        f"trades         {metrics.num_trades}   win rate {metrics.win_rate_pct:.0f}%\n"
        f"best / worst   {_mmoney(metrics.best_trade)} / {_mmoney(metrics.worst_trade)}\n"
        f"max drawdown   {metrics.max_drawdown_pct:.1f}%"
        + ("\nLIQUIDATED" if metrics.liquidated else "")
    )
    ax_e.text(0.995, 0.03, box, transform=ax_e.transAxes, fontsize=8,
              family="monospace", color=INK_2, va="bottom", ha="right",
              bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.5"))

    # -- panel 3: drawdown
    _style_axes(ax_d)
    eq = equity.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(eq)
    dd = np.where(peaks > 0, (eq - peaks) / peaks * 100.0, 0.0)
    dd_idx, dd_vals = _decimate(idx, dd)
    ax_d.fill_between(dd_idx, dd_vals, 0.0, color=CRITICAL, alpha=0.10, zorder=2)
    ax_d.plot(dd_idx, dd_vals, color=CRITICAL, linewidth=1.5, zorder=3)
    ax_d.set_ylabel("drawdown %", color=MUTED, fontsize=9)
    # percent axis needs decimals for shallow drawdowns ("-0.25", not "-0")
    ax_d.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.2f}".rstrip("0").rstrip(".")))

    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    label = "" if metrics.data_label == "real" else f"   [{metrics.data_label} data]"
    fig.suptitle(
        f"{metrics.strategy}  ·  {metrics.market}  ·  start {_mmoney(metrics.start_balance)}{label}",
        color=INK, fontsize=12, x=0.06, ha="left",
    )
    fig.autofmt_xdate(rotation=0, ha="center")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    return path


def overlay_chart(results: list[BacktestResult], title: str, path: str | Path) -> list[Path]:
    """All strategies' balance curves for one (market, start balance) group.

    The categorical palette has 8 slots and is never cycled: past 8
    strategies the group is faceted into multiple charts (…_part2.png).
    Chunk membership follows the stable run order, so a strategy keeps
    its color in every group.
    """
    path = Path(path)
    if len(results) > len(SERIES):
        chunks = [results[j:j + len(SERIES)] for j in range(0, len(results), len(SERIES))]
        return [
            _overlay_chart_single(
                chunk, f"{title} ({k + 1}/{len(chunks)})",
                path.with_stem(f"{path.stem}_part{k + 1}"))
            for k, chunk in enumerate(chunks)
        ]
    return [_overlay_chart_single(results, title, path)]


def _overlay_chart_single(results: list[BacktestResult], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(PAGE)
    _style_axes(ax)

    lo, hi = float("inf"), 0.0
    for k, res in enumerate(results):
        color = SERIES[k]
        ax.plot(*_decimate(res.equity.index, res.equity), color=color, linewidth=2,
                solid_joinstyle="round", solid_capstyle="round",
                label=res.strategy_name)
        lo = min(lo, float(res.equity.min()))
        hi = max(hi, float(res.equity.max()))
        if len(results) <= 4:  # direct end labels while they stay legible
            ax.annotate(
                f" {res.strategy_name}",
                (res.equity.index[-1], float(res.equity.iloc[-1])),
                color=INK_2, fontsize=8, va="center",
            )
    if lo > 0 and hi / lo > 50:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("balance (USD)", color=MUTED, fontsize=9)
    _legend(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(title, color=INK, fontsize=12, loc="left")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- tables


def _source_path(strategy_name: str, relative_to: str | Path | None) -> str | None:
    """Path of the strategy's source file, relative to the report location."""
    try:
        from tradebot.registry import available_strategies

        cls = available_strategies().get(strategy_name)
        if cls is None:
            return None
        src = inspect.getsourcefile(cls)
        if src is None:
            return None
        if relative_to is None:
            return src
        return os.path.relpath(src, Path(relative_to).resolve())
    except Exception:  # noqa: BLE001 - a report must never die over a link
        return None


def _strategy_doc(name: str) -> str:
    """First docstring line of the strategy class: the idea in one line."""
    try:
        from tradebot.registry import available_strategies

        cls = available_strategies().get(name)
        doc = (cls.__doc__ or "").strip() if cls else ""
        return doc.splitlines()[0].strip() if doc else ""
    except Exception:  # noqa: BLE001
        return ""


def _strategy_cell(name: str, out_dir: str | Path | None) -> str:
    parts = [f"**{name}**"]
    doc = _strategy_doc(name)
    if doc:
        parts.append(f"_{doc}_")
    link = _source_path(name, out_dir)
    if link:
        parts.append(f"[source]({link})")
    return "<br>".join(parts)


def _balance_label(balance: float) -> str:
    if balance >= 1e6 and balance % 1e6 == 0:
        return f"${balance / 1e6:g}M"
    if balance >= 1e3 and balance % 1e3 == 0:
        return f"${balance / 1e3:g}K"
    return _money(balance)


def _config_order(all_metrics: list[Metrics]) -> list[tuple[str, float]]:
    """(market, balance) columns: spot before futures, small balance first."""
    configs = {(m.market, m.start_balance) for m in all_metrics}
    return sorted(configs, key=lambda c: (0 if c[0] == "spot" else 1, c[0], c[1]))


def matrix_table(all_metrics: list[Metrics], out_dir: str | Path | None = None) -> str:
    """One row per strategy; per-config cells with the key numbers.

    Cell contents: num trades, profit, worst trade, best trade and the
    balance after the run — for every (market, start balance) config.
    Rows are ranked by the strategy's best final balance across configs.
    """
    configs = _config_order(all_metrics)
    by_key = {(m.strategy, m.market, m.start_balance): m for m in all_metrics}
    strategies = sorted(
        {m.strategy for m in all_metrics},
        key=lambda s: max((by_key[(s, mk, b)].final_balance
                           for (mk, b) in configs if (s, mk, b) in by_key),
                          default=float("-inf")),
        reverse=True,
    )

    header = ("| strategy | "
              + " | ".join(f"{mk} · {_balance_label(b)}" for mk, b in configs) + " |")
    sep = "|" + "|".join("---" for _ in range(len(configs) + 1)) + "|"
    lines = [header, sep]
    for name in strategies:
        cells = [_strategy_cell(name, out_dir)]
        for mk, b in configs:
            m = by_key.get((name, mk, b))
            if m is None:
                cells.append("—")
                continue
            parts = [
                f"trades {m.num_trades:,}",
                f"profit {_money(m.profit)}",
                f"worst {_money(m.worst_trade)}",
                f"best {_money(m.best_trade)}",
                f"**after {_money(m.final_balance)}**",
            ]
            if m.liquidated:
                parts.append("LIQUIDATED")
            cells.append("<br>".join(parts))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

TABLE_COLS = [
    ("strategy", "strategy", "{}"),
    ("final_balance", "final balance", "money"),
    ("profit", "profit", "money"),
    ("profit_pct", "profit %", "{:+.2f}%"),
    ("num_trades", "trades", "{:d}"),
    ("win_rate_pct", "win %", "{:.1f}"),
    ("best_trade", "best trade", "money"),
    ("worst_trade", "worst trade", "money"),
    ("max_drawdown_pct", "max DD %", "{:.1f}"),
    ("sharpe", "sharpe", "{:.2f}"),
    ("time_in_market_pct", "in market %", "{:.1f}"),
    ("fees_paid", "fees", "money"),
    ("liquidated", "liq.", "{}"),
]


def _fmt(value, spec: str) -> str:
    if spec == "money":
        return _money(float(value))
    if spec == "{}":
        return "yes" if value is True else ("" if value is False else str(value))
    return spec.format(value)


def markdown_table(group: list[Metrics], out_dir: str | Path | None = None) -> str:
    """One markdown table, sorted by final balance (primary criterion)."""
    rows = sorted(group, key=lambda m: m.final_balance, reverse=True)
    header = "| " + " | ".join(h for _, h, _ in TABLE_COLS) + " |"
    sep = "|" + "|".join("---" for _ in TABLE_COLS) + "|"
    lines = [header, sep]
    for m in rows:
        d = m.as_row()
        cells = [_fmt(d[k], spec) for k, _, spec in TABLE_COLS]
        link = _source_path(m.strategy, out_dir)
        if link:
            cells[0] = f"[{m.strategy}]({link})"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


README_BEGIN = "<!-- comparison:begin -->"
README_END = "<!-- comparison:end -->"


def update_readme(all_metrics: list[Metrics], readme_path: str | Path,
                  period: str = "") -> bool:
    """Splice the consolidated comparison table into the README.

    The table lands between the ``comparison:begin``/``end`` markers,
    sorted best to worst by final balance. Returns False when the README
    or its markers are missing.
    """
    readme_path = Path(readme_path)
    if not readme_path.exists():
        return False
    text = readme_path.read_text()
    if README_BEGIN not in text or README_END not in text:
        return False

    labels = ", ".join(sorted({m.data_label for m in all_metrics}))
    head = f"_Period: {period} · data: {labels}_\n\n" if period else f"_Data: {labels}_\n\n"
    table = matrix_table(all_metrics, out_dir=readme_path.parent)
    before = text.split(README_BEGIN)[0]
    after = text.split(README_END)[1]
    readme_path.write_text(
        f"{before}{README_BEGIN}\n{head}{table}\n{README_END}{after}")
    return True


def comparison_report(all_metrics: list[Metrics], out_dir: str | Path,
                      period: str = "") -> Path:
    """Write comparison.md + comparison.csv; return path to the markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_labels = {m.data_label for m in all_metrics}
    parts = ["# Strategy comparison", ""]
    if period:
        parts.append(f"Period: {period}  ")
    parts.append(f"Data: {', '.join(sorted(data_labels))}  ")
    parts.append("Ranked by **final balance** (the primary comparison criterion); "
                 "rows ordered by each strategy's best config.")
    parts.append("")
    parts.append(matrix_table(all_metrics, out_dir))
    parts.append("")
    parts.append("## Details per market and starting balance")
    parts.append("")

    groups: dict[tuple[str, float], list[Metrics]] = {}
    for m in all_metrics:
        groups.setdefault((m.market, m.start_balance), []).append(m)
    for (market, balance) in sorted(groups):
        parts.append(f"### {market} · start balance {_money(balance)}")
        parts.append("")
        parts.append(markdown_table(groups[(market, balance)], out_dir))
        parts.append("")

    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(parts))

    pd.DataFrame([m.as_row() for m in all_metrics]).sort_values(
        ["market", "start_balance", "final_balance"],
        ascending=[True, True, False],
    ).to_csv(out_dir / "comparison.csv", index=False)
    return md_path


def print_comparison(all_metrics: list[Metrics]) -> None:
    # summary matrix: final balance per strategy per config
    configs = _config_order(all_metrics)
    by_key = {(m.strategy, m.market, m.start_balance): m for m in all_metrics}
    strategies = sorted(
        {m.strategy for m in all_metrics},
        key=lambda s: max((by_key[(s, mk, b)].final_balance
                           for (mk, b) in configs if (s, mk, b) in by_key),
                          default=float("-inf")),
        reverse=True,
    )
    headers = ["strategy"] + [f"{mk} · {_balance_label(b)}" for mk, b in configs]
    rows = []
    for name in strategies:
        row = [name]
        for mk, b in configs:
            m = by_key.get((name, mk, b))
            cell = "—" if m is None else _money(m.final_balance)
            if m is not None and m.liquidated:
                cell += " (liq.)"
            row.append(cell)
        rows.append(row)
    widths = [max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(len(headers))]
    print("\n=== final balance after run (primary criterion) ===")
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))

    groups: dict[tuple[str, float], list[Metrics]] = {}
    for m in all_metrics:
        groups.setdefault((m.market, m.start_balance), []).append(m)
    for (market, balance) in sorted(groups):
        print(f"\n=== {market} · start {_money(balance)} "
              f"(ranked by final balance) ===")
        rows = sorted(groups[(market, balance)], key=lambda m: m.final_balance, reverse=True)
        table = [[_fmt(m.as_row()[k], spec) for k, _, spec in TABLE_COLS] for m in rows]
        headers = [h for _, h, _ in TABLE_COLS]
        widths = [max(len(headers[c]), *(len(r[c]) for r in table)) for c in range(len(headers))]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        for r in table:
            print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
