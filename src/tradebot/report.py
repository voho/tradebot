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
from tradebot.evidence import (BETTER, CORPSE, SAME,  # noqa: E402
                               WORSE, Evidence)
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
    """Strategy name, linked to its source file when the path resolves."""
    link = _source_path(name, out_dir)
    return f"[{name}]({link})" if link else name


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


SCALE_TOLERANCE_PCT = 1.0  # return gap that counts as "not scale-invariant"
RANK_BADGES = {1: "🥇", 2: "🥈", 3: "🥉"}
DEEP_DRAWDOWN_PCT = 50.0  # above this, flag the risk


def _outcome_badge(m: Metrics) -> str:
    """Emoji summarising a run: wiped out, profitable, or a loss."""
    if m.liquidated:
        return "💀"
    if m.final_balance > m.start_balance:
        return "🟢"
    return "🔴"


#: The market the promotion bar is stated against. README, ROUTINE and
#: LEDGER all agree: leveraged buy-and-hold is not a benchmark but a stress
#: case, and it gets liquidated — so "did it beat holding?" is a question
#: about spot, and the verdict column answers it there whichever market a
#: row's balance happens to be bolded in.
BENCHMARK_MARKET = "spot"


def matrix_table(all_metrics: list[Metrics], out_dir: str | Path | None = None,
                 evidence: dict[tuple[str, str], Evidence] | None = None,
                 ordering: dict[str, tuple[int, int]] | None = None) -> str:
    """Scannable leaderboard: one row per strategy, one balance per market.

    Results are almost perfectly proportional to the starting balance, so
    showing every start balance as its own column is duplication. The
    table reports the **smallest** start balance (the realistic retail
    case) and marks with a dagger any strategy whose return at another
    start balance differs by more than ``SCALE_TOLERANCE_PCT`` — which
    happens only where the exchange minimum order size bites. Full
    per-config numbers stay in the detail tables below.

    Rows are ranked by best final balance; the strategy's best market is
    bolded and a liquidation is marked ``!``.

    When ``evidence`` is supplied (R-29's paired bootstrap intervals, see
    :mod:`tradebot.evidence`) two columns are appended: the paired
    difference from ``buy_and_hold`` in log growth and in max drawdown,
    each with its interval. They sit *after* the observed numbers because
    the divide is real — everything to their left happened on one path,
    and only they say whether it is distinguishable from having done
    nothing. Without evidence the table renders exactly as before: the
    intervals are an enrichment, not a dependency.
    """
    by_key = {(m.strategy, m.market, m.start_balance): m for m in all_metrics}
    markets = sorted({m.market for m in all_metrics},
                     key=lambda mk: (0 if mk == "spot" else 1, mk))
    ref_balance = min(m.start_balance for m in all_metrics)
    ref = {(m.strategy, m.market): m for m in all_metrics
           if m.start_balance == ref_balance}

    strategies = sorted(
        {m.strategy for m in all_metrics},
        key=lambda s: max((ref[(s, mk)].final_balance for mk in markets
                           if (s, mk) in ref), default=float("-inf")),
        reverse=True,
    )

    def scale_sensitive(name: str) -> bool:
        """True when another start balance changes the return materially."""
        for mk in markets:
            base = ref.get((name, mk))
            if base is None:
                continue
            for (s, market, _bal), m in by_key.items():
                if s == name and market == mk and abs(
                        m.profit_pct - base.profit_pct) > SCALE_TOLERANCE_PCT:
                    return True
        return False

    evidence = evidence or {}
    # The evidence columns go last, after everything observed on this one
    # path: left of the divide is what happened, right of it is whether it
    # is distinguishable from having done nothing.
    evidence_cols = ([f"growth vs hold ({BENCHMARK_MARKET})",
                      f"max DD vs hold ({BENCHMARK_MARKET})"] if evidence else [])
    columns = ["#", "strategy", *markets, "trades", "profit", "max DD",
               *evidence_cols]
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    any_dagger = False
    for rank, name in enumerate(strategies, 1):
        best_market = max((mk for mk in markets if (name, mk) in ref),
                          key=lambda mk: ref[(name, mk)].final_balance, default=None)
        best = ref.get((name, best_market)) if best_market else None
        # Pinned to the benchmark's market, not the row's bolded one: on
        # 5x futures `buy_and_hold` is liquidated in early 2017, so a
        # verdict there would be measured against a corpse (R-22).
        ev = evidence.get((name, BENCHMARK_MARKET))
        label = _strategy_cell(name, out_dir)
        if scale_sensitive(name):
            label += " †"
            any_dagger = True
        cells = [f"{RANK_BADGES.get(rank, '')}{rank}".strip(), label]
        for mk in markets:
            m = ref.get((name, mk))
            if m is None:
                cells.append("—")
                continue
            text = _money(m.final_balance)
            if mk == best_market:
                text = f"**{text}**"
            cells.append(f"{_outcome_badge(m)} {text}")
        if best is None:
            cells += ["—", "—", "—"]
        else:
            arrow = "📈" if best.profit > 0 else "📉"
            dd = f"{best.max_drawdown_pct:.0f}%"
            if best.max_drawdown_pct >= DEEP_DRAWDOWN_PCT:
                dd += " ⚠️"
            cells += [f"{best.num_trades:,}", f"{arrow} {_money(best.profit)}", dd]
        if evidence_cols:
            cells += ([ev.growth_cell(), ev.drawdown_cell()] if ev
                      else ["—", "—"])
        lines.append("| " + " | ".join(cells) + " |")

    legend = (f"_Balances from a {_money(ref_balance)} start · bold = the "
              "strategy's better market · 🟢 profit · 🔴 loss · 💀 liquidated · "
              f"⚠️ drawdown over {DEEP_DRAWDOWN_PCT:.0f}%. Trades, profit and "
              "max drawdown describe that market._")
    if any_dagger:
        legend += ("\n_† return differs by more than "
                   f"{SCALE_TOLERANCE_PCT:g}pp at a larger starting balance — the "
                   "exchange minimum order size blocks small rebalances on a "
                   "small account. Everything else is proportional to capital._")
    if evidence:
        legend += "\n\n" + _evidence_legend(evidence, ordering)
    lines.append("")
    lines.append(legend)
    return "\n".join(lines)


def _evidence_legend(evidence: dict[tuple[str, str], Evidence],
                     ordering: dict[str, tuple[int, int]] | None) -> str:
    """Explain the two evidence columns, and what they do *not* license.

    A reader who sees a gold medal beside an interval containing zero
    needs to be told, in the same place, that the medal is the weaker of
    the two statements.
    """
    sample = next(iter(evidence.values()))
    counted = [ev for (_, market), ev in evidence.items()
               if market == BENCHMARK_MARKET and not ev.is_benchmark]
    n_better = sum(ev.growth_distinguishable and ev.d_log_growth > 0
                   for ev in counted)
    lines = [
        f"_The last two columns are the only ones that answer **\"is this "
        f"difference real?\"** Both are paired differences against "
        f"`buy_and_hold` on {BENCHMARK_MARKET} over the {sample.period} "
        f"period ({sample.days:,} daily observations), each with a 95% "
        f"stationary block-bootstrap interval — 30-day mean block, 2,000 "
        f"resamples, the identical resample applied to both strategies so "
        f"the market's own variance cancels instead of swamping the gap. "
        f"{BETTER} / {WORSE} = the interval excludes zero and the strategy is "
        f"better / worse; **{SAME} = it contains zero, so the difference from "
        f"simply holding is not established**._",
        "",
        f"_**Growth**, not Sharpe, because final balance is what this table "
        f"ranks by — and the two disagree. **{BENCHMARK_MARKET}**, because "
        f"leveraged buy-and-hold is a stress case rather than a benchmark: it "
        f"is liquidated in early 2017, and an account that cannot draw down "
        f"further is not something to draw down less than (R-22). On this "
        f"run **{n_better} of {len(counted)}** strategies are distinguishably "
        f"better than holding on growth; the drawdown column is where the "
        f"project's findings actually live._",
    ]
    if ordering:
        counts = " · ".join(
            f"**{k} of {n}** on {market}"
            for market, (k, n) in sorted(
                ordering.items(),
                key=lambda kv: (0 if kv[0] == BENCHMARK_MARKET else 1, kv[0]))
        )
        lines += [
            "",
            f"_Adjacent steps down this ranking that survive the same test: "
            f"{counts}. The order is a display convention, not a result — "
            f"read the table as buckets._",
        ]
    lines += [
        "",
        "_Regenerate with `python scripts/inference.py`; the numbers live in "
        "`reports/inference/bootstrap.csv`._",
    ]
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


EVIDENCE_COLS = ["Δ sharpe vs hold", "Δ max DD vs hold", "Δ log growth vs hold",
                 "P(growth > hold)"]


def markdown_table(group: list[Metrics], out_dir: str | Path | None = None,
                   evidence: dict[tuple[str, str], Evidence] | None = None) -> str:
    """One markdown table, sorted by final balance (primary criterion).

    With ``evidence``, four paired-bootstrap columns are appended: this is
    the detail table, so it carries the full error bars that the README's
    summary compresses into one ``vs hold`` cell.
    """
    evidence = evidence or {}
    rows = sorted(group, key=lambda m: m.final_balance, reverse=True)
    extra = EVIDENCE_COLS if evidence else []
    header = "| " + " | ".join([h for _, h, _ in TABLE_COLS] + extra) + " |"
    sep = "|" + "|".join("---" for _ in range(len(TABLE_COLS) + len(extra))) + "|"
    lines = [header, sep]
    for m in rows:
        d = m.as_row()
        cells = [_fmt(d[k], spec) for k, _, spec in TABLE_COLS]
        link = _source_path(m.strategy, out_dir)
        if link:
            cells[0] = f"[{m.strategy}]({link})"
        if evidence:
            ev = evidence.get((m.strategy, m.market))
            cells += ([ev.sharpe_cell(), ev.drawdown_cell(), ev.growth_cell(),
                       f"{ev.p_growth_beats_hold:.2f}"]
                      if ev else ["—"] * len(EVIDENCE_COLS))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


README_BEGIN = "<!-- comparison:begin -->"
README_END = "<!-- comparison:end -->"


def update_readme(all_metrics: list[Metrics], readme_path: str | Path,
                  period: str = "",
                  evidence: dict[tuple[str, str], Evidence] | None = None,
                  ordering: dict[str, tuple[int, int]] | None = None) -> bool:
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
    table = matrix_table(all_metrics, out_dir=readme_path.parent,
                         evidence=evidence, ordering=ordering)
    before = text.split(README_BEGIN)[0]
    after = text.split(README_END)[1]
    readme_path.write_text(
        f"{before}{README_BEGIN}\n{head}{table}\n{README_END}{after}")
    return True


def comparison_report(all_metrics: list[Metrics], out_dir: str | Path,
                      period: str = "",
                      evidence: dict[tuple[str, str], Evidence] | None = None,
                      ordering: dict[str, tuple[int, int]] | None = None) -> Path:
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
    if evidence:
        parts.append("")
        parts.append(
            "Every comparison against `buy_and_hold` carries a 95% paired "
            "block-bootstrap interval. A rank is not a result: see the legend "
            "under the table for how much of this ordering survives that "
            f"test. In the per-market detail tables below, {CORPSE} marks a "
            "comparison made against a `buy_and_hold` account that was "
            "liquidated early and inert for most of the period — on 5x "
            "futures it dies in January 2017, so beating it there is a "
            "statement about surviving leverage, not about edge (R-22).")
    parts.append("")
    parts.append(matrix_table(all_metrics, out_dir, evidence=evidence,
                              ordering=ordering))
    parts.append("")
    parts.append("## Details per market and starting balance")
    parts.append("")

    groups: dict[tuple[str, float], list[Metrics]] = {}
    for m in all_metrics:
        groups.setdefault((m.market, m.start_balance), []).append(m)
    for (market, balance) in sorted(groups):
        parts.append(f"### {market} · start balance {_money(balance)}")
        parts.append("")
        parts.append(markdown_table(groups[(market, balance)], out_dir,
                                    evidence=evidence))
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
