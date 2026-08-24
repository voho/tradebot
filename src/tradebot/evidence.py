"""The comparison table's error bars.

R-29 computed a 95% paired block-bootstrap interval for every registered
strategy against ``buy_and_hold`` and wrote it to
``reports/inference/bootstrap.csv``. The comparison table then went on
printing point estimates in rank order — the most confident possible way
to present numbers whose ordering is, in its decision-relevant region,
noise. This module is the wire between the two (backlog B-12).

It is deliberately a *reader*, not a calculator: nothing here computes a
statistic. :mod:`tradebot.inference` and ``scripts/inference.py`` produce
the intervals; this loads them, matches them to the runs in a report, and
refuses to match anything it cannot match exactly. Three ways a wrong
number could be printed next to a right one, and what stops each:

- **A different market.** ``bootstrap.csv`` says ``futures``; the table
  says ``futures_5x``, because the name carries the leverage. The alias
  map below is exact and one-way, so an interval measured at 5x can never
  be printed beside a run at another leverage — the cell goes blank
  instead.
- **A stale row.** A strategy whose code changed since the bootstrap ran
  would get an interval describing the old code. ``tests/test_evidence.py``
  makes a missing row a CI failure, which is the cheapest available proxy
  for freshness: adding or changing a strategy means re-running the
  inference script.
- **A dead benchmark.** On 5x futures ``buy_and_hold`` is liquidated in
  early 2017 and inert for 99.7% of the full period. A comparison against
  an account that cannot draw down because it has nothing left is the R-22
  mistake, and it is what made the futures holdout column look like a
  landslide before R-29 caught it. Every row therefore carries the
  *benchmark's* dead-tail share as well as its own, and a comparison
  against a corpse is flagged rather than scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BENCHMARK = "buy_and_hold"

#: Where ``scripts/inference.py`` writes its tables, relative to a report dir.
INFERENCE_SUBDIR = "inference"
BOOTSTRAP_CSV = "bootstrap.csv"
ORDERING_CSV = "ordering.csv"

#: The period whose intervals belong beside the full-history comparison
#: table. ``holdout`` is the other one on file, and it is the harsher
#: reading: no strategy in the table is distinguishable from holding on it.
DEFAULT_PERIOD = "full"

#: ``bootstrap.csv`` names a market by kind; the table names it by
#: leverage. Exact, and one-way on purpose — see the module docstring.
#: ``"portfolio"`` is the one multi-asset axis (backlog B-32): it carries no
#: leverage variants, so it aliases to itself.
MARKET_ALIASES = {"spot": "spot", "futures": "futures_5x", "portfolio": "portfolio"}

#: Above this share of dead trailing days the benchmark is a corpse and the
#: comparison against it is not a result. Same threshold the inference
#: script prints its own warning at.
DEAD_TAIL_PCT = 10.0

BETTER = "▲"
WORSE = "▼"
SAME = "≈"
CORPSE = "☠"


def _signed(x: float, places: int = 2) -> str:
    return f"{x:+.{places}f}"


@dataclass(frozen=True)
class Evidence:
    """One strategy's measured difference from ``buy_and_hold``.

    Every field is a *paired* quantity computed on identical bootstrap
    resamples of the two strategies, which is why the intervals are so
    much narrower than a marginal one: most of a strategy's uncertainty is
    the market's, and pairing cancels it.
    """

    strategy: str
    market: str
    period: str
    days: int
    dead_tail_pct: float
    bench_dead_tail_pct: float
    sharpe: float
    d_sharpe: float
    d_sharpe_lo: float
    d_sharpe_hi: float
    d_max_dd_pp: float
    d_max_dd_lo: float
    d_max_dd_hi: float
    d_log_growth: float
    d_log_growth_lo: float
    d_log_growth_hi: float
    p_growth_beats_hold: float

    @property
    def is_benchmark(self) -> bool:
        return self.strategy == BENCHMARK

    @property
    def benchmark_inert(self) -> bool:
        """The benchmark spent most of this period liquidated (R-22)."""
        return self.bench_dead_tail_pct > DEAD_TAIL_PCT

    @staticmethod
    def _excludes_zero(lo: float, hi: float) -> bool:
        if pd.isna(lo) or pd.isna(hi):
            return False
        return lo > 0.0 or hi < 0.0

    @property
    def growth_distinguishable(self) -> bool:
        return self._excludes_zero(self.d_log_growth_lo, self.d_log_growth_hi)

    @property
    def sharpe_distinguishable(self) -> bool:
        return self._excludes_zero(self.d_sharpe_lo, self.d_sharpe_hi)

    @property
    def drawdown_distinguishable(self) -> bool:
        return self._excludes_zero(self.d_max_dd_lo, self.d_max_dd_hi)

    @property
    def drawdown_beats_hold(self) -> bool:
        """A *shallower* drawdown than holding, and distinguishably so.

        False against an inert benchmark: an account that was liquidated
        early cannot draw down further, so drawing down less than it is
        not a finding.
        """
        return (self.drawdown_distinguishable and self.d_max_dd_pp < 0.0
                and not self.benchmark_inert)

    def _glyph(self, point: float, distinguishable: bool,
               more_is_better: bool = True) -> str:
        if self.benchmark_inert:
            return CORPSE
        if not distinguishable:
            return SAME
        better = point > 0.0 if more_is_better else point < 0.0
        return BETTER if better else WORSE

    def growth_cell(self) -> str:
        """``▲ +3.74 [+2.37, +5.03]`` — the table's own ranking criterion.

        The table ranks by final balance, so the quantity that belongs in
        an "is this real?" column is the paired difference in **log
        growth**, not the Sharpe. They disagree: `kelly_regime_v4` beats
        holding on Sharpe and is a coin flip on growth (P = 0.52).
        """
        if self.is_benchmark:
            return "benchmark"
        if pd.isna(self.d_log_growth_lo):
            return "—"
        return (f"{self._glyph(self.d_log_growth, self.growth_distinguishable)} "
                f"{_signed(self.d_log_growth)} "
                f"[{_signed(self.d_log_growth_lo)}, {_signed(self.d_log_growth_hi)}]")

    def sharpe_cell(self) -> str:
        if self.is_benchmark:
            return "benchmark"
        return (f"{self._glyph(self.d_sharpe, self.sharpe_distinguishable)} "
                f"{_signed(self.d_sharpe)} "
                f"[{_signed(self.d_sharpe_lo)}, {_signed(self.d_sharpe_hi)}]")

    def drawdown_cell(self) -> str:
        """Δ max drawdown in percentage points; **less is better**."""
        if self.is_benchmark:
            return "benchmark"
        glyph = self._glyph(self.d_max_dd_pp, self.drawdown_distinguishable,
                            more_is_better=False)
        return (f"{glyph} {_signed(self.d_max_dd_pp, 1)}pp "
                f"[{_signed(self.d_max_dd_lo, 1)}, {_signed(self.d_max_dd_hi, 1)}]")


def _bootstrap_path(report_dir: str | Path) -> Path:
    return Path(report_dir) / INFERENCE_SUBDIR / BOOTSTRAP_CSV


def _ordering_path(report_dir: str | Path) -> Path:
    return Path(report_dir) / INFERENCE_SUBDIR / ORDERING_CSV


def load_evidence(report_dir: str | Path,
                  period: str = DEFAULT_PERIOD) -> dict[tuple[str, str], Evidence]:
    """``{(strategy, table_market): Evidence}`` for one period.

    Returns an empty mapping when the file is absent or holds no rows for
    ``period`` — a report must render without it, just without error bars.
    Markets the alias map does not cover are dropped rather than guessed.
    """
    path = _bootstrap_path(report_dir)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame["period"] == period]
    if frame.empty:
        return {}

    out: dict[tuple[str, str], Evidence] = {}
    for market, block in frame.groupby("market"):
        table_market = MARKET_ALIASES.get(str(market))
        if table_market is None:
            continue
        bench = block[block["strategy"] == BENCHMARK]
        bench_dead = float(bench["dead_tail_pct"].iloc[0]) if len(bench) else 0.0
        for row in block.to_dict("records"):
            out[(str(row["strategy"]), table_market)] = Evidence(
                strategy=str(row["strategy"]),
                market=table_market,
                period=period,
                days=int(row["days"]),
                dead_tail_pct=float(row["dead_tail_pct"]),
                bench_dead_tail_pct=bench_dead,
                sharpe=float(row["sharpe"]),
                d_sharpe=float(row["d_sharpe"]),
                d_sharpe_lo=float(row["d_sharpe_lo"]),
                d_sharpe_hi=float(row["d_sharpe_hi"]),
                d_max_dd_pp=float(row["d_max_dd_pp"]),
                d_max_dd_lo=float(row["d_max_dd_lo"]),
                d_max_dd_hi=float(row["d_max_dd_hi"]),
                d_log_growth=float(row["d_log_growth"]),
                # Older bootstrap.csv files predate the growth interval:
                # report the point and blank the bar rather than invent one.
                d_log_growth_lo=float(row.get("d_log_growth_lo", float("nan"))),
                d_log_growth_hi=float(row.get("d_log_growth_hi", float("nan"))),
                p_growth_beats_hold=float(row["p_growth_beats_hold"]),
            )
    return out


def ordering_counts(report_dir: str | Path,
                    period: str = DEFAULT_PERIOD) -> dict[str, tuple[int, int]]:
    """``{table_market: (distinguishable, total)}`` for adjacent pairs.

    The table's claim is an *order*, and this is how much of that order
    survives a 95% interval. R-29's answer over both periods and both
    markets was 10 of 96, none of them separating two of the top eight.
    """
    path = _ordering_path(report_dir)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame["period"] == period]
    out: dict[str, tuple[int, int]] = {}
    for market, block in frame.groupby("market"):
        table_market = MARKET_ALIASES.get(str(market))
        if table_market is None:
            continue
        out[table_market] = (int(block["distinguishable"].sum()), len(block))
    return out
