"""The comparison table's error bars (backlog B-12).

Two kinds of test here. The first kind checks the reader: it must match a
strategy to its interval exactly, and blank the cell rather than guess
whenever it cannot. The second kind is a CI rule in the same spirit as
"every registered strategy appears in the README table" — every
registered strategy must also have a measured interval, so a new strategy
cannot slip into the table as a bare point estimate.
"""

from pathlib import Path

import pytest

from tradebot.evidence import (BETTER, CORPSE, SAME, WORSE, load_evidence,
                               ordering_counts)
from tradebot.registry import available_strategies
from tradebot.report import markdown_table, matrix_table

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

COLUMNS = ["period", "market", "strategy", "days", "dead_tail_pct", "sharpe",
           "d_sharpe", "d_sharpe_lo", "d_sharpe_hi", "d_max_dd_pp",
           "d_max_dd_lo", "d_max_dd_hi", "d_log_growth", "d_log_growth_lo",
           "d_log_growth_hi", "p_growth_beats_hold"]


def _row(strategy, market="spot", period="full", *, dead=0.0, d_sharpe=0.0,
         d_sharpe_lo=-1.0, d_sharpe_hi=1.0, d_dd=0.0, d_dd_lo=-1.0, d_dd_hi=1.0,
         d_growth=0.0, d_growth_lo=-1.0, d_growth_hi=1.0, p=0.5):
    return dict(period=period, market=market, strategy=strategy, days=3510,
                dead_tail_pct=dead, sharpe=1.0, d_sharpe=d_sharpe,
                d_sharpe_lo=d_sharpe_lo, d_sharpe_hi=d_sharpe_hi,
                d_max_dd_pp=d_dd, d_max_dd_lo=d_dd_lo, d_max_dd_hi=d_dd_hi,
                d_log_growth=d_growth, d_log_growth_lo=d_growth_lo,
                d_log_growth_hi=d_growth_hi, p_growth_beats_hold=p)


def _write(tmp_path: Path, rows, columns=None, name="bootstrap.csv") -> Path:
    import pandas as pd

    out = tmp_path / "inference"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame[columns]
    frame.to_csv(out / name, index=False)
    return tmp_path


# ------------------------------------------------------------------ reader


def test_missing_file_yields_no_evidence(tmp_path):
    assert load_evidence(tmp_path) == {}
    assert ordering_counts(tmp_path) == {}


def test_futures_is_renamed_to_the_leverage_the_table_uses(tmp_path):
    _write(tmp_path, [_row("buy_and_hold"), _row("buy_and_hold", "futures")])
    ev = load_evidence(tmp_path)
    assert set(ev) == {("buy_and_hold", "spot"), ("buy_and_hold", "futures_5x")}


def test_unknown_markets_are_dropped_not_guessed(tmp_path):
    """An interval measured elsewhere must never be printed here."""
    _write(tmp_path, [_row("buy_and_hold"), _row("buy_and_hold", "futures_3x")])
    assert set(load_evidence(tmp_path)) == {("buy_and_hold", "spot")}


def test_only_the_requested_period_is_loaded(tmp_path):
    _write(tmp_path, [_row("buy_and_hold"), _row("buy_and_hold", period="holdout")])
    assert load_evidence(tmp_path, "full")[("buy_and_hold", "spot")].period == "full"
    assert load_evidence(tmp_path, "holdout")[("buy_and_hold", "spot")].days == 3510
    assert load_evidence(tmp_path, "inner") == {}


def test_verdict_glyphs_follow_the_interval_not_the_point(tmp_path):
    _write(tmp_path, [
        _row("buy_and_hold"),
        _row("winner", d_growth=3.7, d_growth_lo=2.4, d_growth_hi=5.0),
        _row("loser", d_growth=-9.5, d_growth_lo=-11.0, d_growth_hi=-8.0),
        # a large point estimate whose interval still contains zero: the
        # single most common shape in this repo's results
        _row("noise", d_growth=0.49, d_growth_lo=-1.1, d_growth_hi=1.2),
    ])
    ev = load_evidence(tmp_path)
    assert ev[("winner", "spot")].growth_cell().startswith(f"{BETTER} +3.70")
    assert ev[("loser", "spot")].growth_cell().startswith(f"{WORSE} -9.50")
    assert ev[("noise", "spot")].growth_cell().startswith(f"{SAME} +0.49")
    assert ev[("noise", "spot")].growth_cell().endswith("[-1.10, +1.20]")
    assert not ev[("noise", "spot")].growth_distinguishable
    assert ev[("buy_and_hold", "spot")].growth_cell() == "benchmark"


def test_drawdown_verdict_inverts_because_less_is_better(tmp_path):
    _write(tmp_path, [
        _row("buy_and_hold"),
        _row("calmer", d_dd=-41.1, d_dd_lo=-54.8, d_dd_hi=-18.4),
        _row("wilder", d_dd=+16.3, d_dd_lo=+3.1, d_dd_hi=+39.1),
        _row("unclear", d_dd=-27.1, d_dd_lo=-35.8, d_dd_hi=+1.9),
    ])
    ev = load_evidence(tmp_path)
    assert ev[("calmer", "spot")].drawdown_cell().startswith(f"{BETTER} -41.1pp")
    assert ev[("calmer", "spot")].drawdown_beats_hold
    assert ev[("wilder", "spot")].drawdown_cell().startswith(f"{WORSE} +16.3pp")
    # R-29's C2: it misses by 1.9pp on the holdout, and that is a miss
    assert ev[("unclear", "spot")].drawdown_cell().startswith(f"{SAME} -27.1pp")
    assert not ev[("unclear", "spot")].drawdown_beats_hold


def test_a_dead_benchmark_gets_a_corpse_flag_not_a_verdict(tmp_path):
    """R-22: on 5x futures buy_and_hold is liquidated in early 2017."""
    _write(tmp_path, [
        _row("buy_and_hold", "futures", dead=99.7),
        _row("survivor", "futures", d_growth=6.3, d_growth_lo=4.1,
             d_growth_hi=8.0, d_dd=-65.1, d_dd_lo=-70.7, d_dd_hi=-60.0),
    ])
    ev = load_evidence(tmp_path)[("survivor", "futures_5x")]
    assert ev.benchmark_inert
    assert ev.growth_cell().startswith(f"{CORPSE} +6.30")
    # the interval excludes zero, and it still is not a win over holding
    assert ev.drawdown_distinguishable and not ev.drawdown_beats_hold


def test_a_strategys_own_flat_days_are_not_a_corpse_flag(tmp_path):
    """A shut gate is a position; a liquidated benchmark is not."""
    _write(tmp_path, [_row("buy_and_hold"), _row("gated", dead=38.7)])
    ev = load_evidence(tmp_path)[("gated", "spot")]
    assert ev.dead_tail_pct == pytest.approx(38.7)
    assert not ev.benchmark_inert


def test_an_older_file_without_growth_bounds_blanks_the_cell(tmp_path):
    """Report the missing bar as missing rather than inventing one."""
    legacy = [c for c in COLUMNS if c not in ("d_log_growth_lo", "d_log_growth_hi")]
    _write(tmp_path, [_row("buy_and_hold"), _row("older", d_growth=1.5)],
           columns=legacy)
    ev = load_evidence(tmp_path)[("older", "spot")]
    assert ev.growth_cell() == "—"
    assert not ev.growth_distinguishable


def test_ordering_counts_summarise_adjacent_pairs(tmp_path):
    import pandas as pd

    out = tmp_path / "inference"
    out.mkdir(parents=True)
    pd.DataFrame([
        {"period": "full", "market": "spot", "distinguishable": True},
        {"period": "full", "market": "spot", "distinguishable": False},
        {"period": "full", "market": "futures", "distinguishable": False},
        {"period": "holdout", "market": "spot", "distinguishable": True},
    ]).to_csv(out / "ordering.csv", index=False)
    assert ordering_counts(tmp_path) == {"spot": (1, 2), "futures_5x": (0, 1)}
    assert ordering_counts(tmp_path, "holdout") == {"spot": (1, 1)}


# -------------------------------------------------------------- the tables


def _metrics(name, final, market="spot", balance=1_000.0, dd=10.0):
    from test_report import _metrics as base

    m = base(name, final, market, balance)
    m.max_drawdown_pct = dd
    return m


def test_matrix_table_grows_evidence_columns_only_with_evidence(tmp_path):
    metrics = [_metrics("buy_and_hold", 1_500.0), _metrics("noise", 2_000.0)]
    assert "vs hold" not in matrix_table(metrics)

    _write(tmp_path, [
        _row("buy_and_hold"),
        _row("noise", d_growth=0.3, d_growth_lo=-0.4, d_growth_hi=1.0,
             d_dd=-41.1, d_dd_lo=-54.8, d_dd_hi=-18.4),
    ])
    table = matrix_table(metrics, evidence=load_evidence(tmp_path),
                         ordering={"spot": (3, 24)})
    header = table.splitlines()[0]
    # the evidence columns come last, after everything observed on the path
    assert header == ("| # | strategy | spot | trades | profit | max DD "
                      "| growth vs hold (spot) | max DD vs hold (spot) |")
    rows = [ln for ln in table.splitlines() if ln.startswith("| ")][1:]
    assert len(rows) == 2
    assert all(ln.count("|") == header.count("|") for ln in rows)
    noise = next(ln for ln in rows if "| noise " in ln)
    assert noise.endswith(f"| {SAME} +0.30 [-0.40, +1.00] "
                          f"| {BETTER} -41.1pp [-54.8, -18.4] |")
    assert "| benchmark | benchmark |" in table
    assert "3 of 24" in table and "buckets" in table
    assert "0 of 1" in table  # none distinguishably better than holding


def test_matrix_table_counts_how_many_beat_holding(tmp_path):
    metrics = [_metrics("buy_and_hold", 1_500.0), _metrics("real", 3_000.0),
               _metrics("noise", 1_600.0), _metrics("bad", 5.0)]
    _write(tmp_path, [
        _row("buy_and_hold"),
        _row("real", d_growth=3.7, d_growth_lo=2.4, d_growth_hi=5.0),
        _row("noise", d_growth=0.3, d_growth_lo=-0.4, d_growth_hi=1.0),
        _row("bad", d_growth=-9.5, d_growth_lo=-11.0, d_growth_hi=-8.0),
    ])
    table = matrix_table(metrics, evidence=load_evidence(tmp_path))
    assert "**1 of 3** strategies are distinguishably better" in table


def test_the_verdict_columns_stay_on_the_benchmarks_market(tmp_path):
    """The row's balance may be bolded on futures; the verdict is spot.

    ``buy_and_hold`` on 5x futures is a stress case, not a benchmark — it
    is liquidated in early 2017 — so the promotion question is asked where
    the promotion bar is stated, and the futures interval (which excludes
    zero, against a corpse) never reaches the summary table.
    """
    metrics = [_metrics("buy_and_hold", 66_000.0),
               _metrics("buy_and_hold", 18.0, "futures_5x"),
               _metrics("v4", 66_800.0),
               _metrics("v4", 156_000.0, "futures_5x", dd=35.0)]
    _write(tmp_path, [
        _row("buy_and_hold"), _row("buy_and_hold", "futures", dead=99.7),
        _row("v4", d_growth=0.04, d_growth_lo=-2.60, d_growth_hi=2.85,
             d_dd=-41.1, d_dd_lo=-54.8, d_dd_hi=-18.4),
        _row("v4", "futures", d_growth=6.3, d_growth_lo=4.1, d_growth_hi=8.0,
             d_dd=-65.1, d_dd_lo=-70.7, d_dd_hi=-60.0),
    ])
    table = matrix_table(metrics, evidence=load_evidence(tmp_path))
    row = next(ln for ln in table.splitlines() if "| v4 " in ln)
    assert "**$156.0K**" in row          # futures is the bolded market...
    assert f"{CORPSE} +6.30" not in row  # ...but never the judged one
    assert row.endswith(f"| {SAME} +0.04 [-2.60, +2.85] "
                        f"| {BETTER} -41.1pp [-54.8, -18.4] |")


def test_matrix_table_blanks_a_strategy_with_no_interval(tmp_path):
    metrics = [_metrics("buy_and_hold", 1_500.0), _metrics("brand_new", 1_400.0)]
    _write(tmp_path, [_row("buy_and_hold")])
    table = matrix_table(metrics, evidence=load_evidence(tmp_path))
    row = next(ln for ln in table.splitlines() if "brand_new" in ln)
    assert row.endswith("| — | — |")


def test_detail_table_carries_the_full_error_bars(tmp_path):
    metrics = [_metrics("buy_and_hold", 1_500.0), _metrics("noise", 1_400.0)]
    assert "Δ log growth vs hold" not in markdown_table(metrics)

    _write(tmp_path, [
        _row("buy_and_hold"),
        _row("noise", d_sharpe=0.21, d_sharpe_lo=-0.30, d_sharpe_hi=0.72,
             d_dd=-41.1, d_dd_lo=-54.8, d_dd_hi=-18.4,
             d_growth=0.04, d_growth_lo=-0.37, d_growth_hi=0.40, p=0.52),
    ])
    table = markdown_table(metrics, evidence=load_evidence(tmp_path))
    header = table.splitlines()[0]
    for col in ("Δ sharpe vs hold", "Δ max DD vs hold", "Δ log growth vs hold",
                "P(growth > hold)"):
        assert col in header
    row = next(ln for ln in table.splitlines() if ln.startswith("| noise "))
    assert row.count("|") == header.count("|")
    assert f"{SAME} +0.21 [-0.30, +0.72]" in row
    assert f"{BETTER} -41.1pp [-54.8, -18.4]" in row
    assert "| 0.52 |" in row


# ----------------------------------------------------- the CI rule itself


@pytest.mark.parametrize("period", ["full", "holdout"])
def test_every_registered_strategy_has_a_measured_interval(period):
    """Same rule as the README table: no strategy ships without one.

    A strategy that is in the comparison table but not in
    ``bootstrap.csv`` would be printed as a bare point estimate beside
    rows that carry an interval — the exact reading R-29 set out to stop.
    """
    evidence = load_evidence(REPORTS, period)
    assert evidence, ("reports/inference/bootstrap.csv is missing or empty - "
                      "run 'python scripts/inference.py'")
    for name in available_strategies():
        for market in ("spot", "futures_5x"):
            assert (name, market) in evidence, (
                f"strategy {name!r} has no {period} interval for {market} - "
                "run 'python scripts/inference.py' and commit "
                "reports/inference/")


def test_the_committed_intervals_carry_a_growth_bar():
    """The column the comparison table prints has to exist on disk."""
    evidence = load_evidence(REPORTS)
    ev = evidence[("kelly_regime_v4", "spot")]
    assert ev.growth_cell() != "—", (
        "bootstrap.csv predates the log-growth interval - re-run "
        "'python scripts/inference.py bootstrap'")
