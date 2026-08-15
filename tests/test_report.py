from tradebot.report import _mmoney, _money, markdown_table, matrix_table
from tradebot.metrics import Metrics


def test_money_formatting_and_unit_promotion():
    assert _money(999_950) == "$1.00M"
    assert _money(999_999_999) == "$1.00B"
    assert _money(-1_234_567) == "-$1.23M"
    assert _money(9_996) == "$10.0K"
    assert _money(1_000) == "$1,000"
    assert _money(150) == "$150"
    assert _money(3.5) == "$3.50"
    assert _money(-0.42) == "-$0.42"


def test_mmoney_escapes_mathtext_dollars():
    assert _mmoney(150) == r"\$150"


def _metrics(name: str, final: float, market: str = "spot",
             balance: float = 1_000.0, liquidated: bool = False) -> Metrics:
    return Metrics(
        strategy=name, market=market, start_balance=balance,
        final_balance=final, profit=final - balance,
        profit_pct=100.0 * (final / balance - 1.0), num_trades=3,
        win_rate_pct=50.0, best_trade=10.0, worst_trade=-5.0, avg_trade=2.0,
        max_drawdown_pct=10.0, sharpe=0.5, time_in_market_pct=40.0,
        fees_paid=1.0, liquidated=liquidated, data_label="SYNTHETIC",
    )


def test_markdown_table_sorted_by_final_balance_desc():
    table = markdown_table([_metrics("worst", 900.0), _metrics("best", 1_100.0)])
    lines = table.splitlines()
    assert lines[0].startswith("| strategy | final balance |")
    assert lines[2].startswith("| best ")
    assert lines[3].startswith("| worst ")
    assert all(line.count("|") == lines[0].count("|") for line in lines)


def test_markdown_table_links_registered_strategies(tmp_path):
    table = markdown_table([_metrics("buy_and_hold", 1_100.0)], out_dir=tmp_path)
    assert "[buy_and_hold](" in table
    assert "buy_and_hold.py)" in table


def test_matrix_table_one_row_per_strategy_with_config_cells(tmp_path):
    metrics = []
    for market, balance in (("spot", 1_000.0), ("spot", 1_000_000.0),
                            ("futures_5x", 1_000.0), ("futures_5x", 1_000_000.0)):
        metrics.append(_metrics("buy_and_hold", balance * 1.5, market, balance))
        metrics.append(_metrics("macd_cross", balance * 0.5, market, balance,
                                liquidated=(market == "futures_5x")))
    table = matrix_table(metrics, out_dir=tmp_path)
    lines = table.splitlines()

    # one column per MARKET, not per (market, balance): results are
    # proportional to capital, so start balances are not separate columns
    assert lines[0] == ("| # | strategy | spot | futures_5x "
                        "| trades | profit | max DD |")
    # header, separator, two strategy rows, blank line, legend
    assert len(lines) == 6
    # ranked by best final balance: buy_and_hold first, with a medal badge
    assert lines[2].startswith("| 🥇1 | [buy_and_hold](")
    assert "buy_and_hold.py)" in lines[2]
    assert lines[3].startswith("| 🥈2 | ")

    # one balance per cell, best market bolded exactly once per row
    assert lines[2].count("**") == 2
    # the verbose per-cell stats moved to the detail tables
    for token in ("trades 3", "worst ", "<br>"):
        assert token not in lines[2]
    # emoji encode the outcome redundantly alongside the numbers
    assert lines[2].count("🟢") == 2  # profitable in both markets
    assert "💀" in lines[3]  # the liquidated futures run
    # summary columns describe the better market, at the $1K reference
    assert lines[2].endswith("| 3 | 📈 $500 | 10% |")
    assert "$1,000 start" in lines[-1]


def test_matrix_table_flags_deep_drawdowns():
    metrics = [_metrics("risky", 1_500.0), _metrics("calm", 1_400.0)]
    metrics[0].max_drawdown_pct = 84.0
    metrics[1].max_drawdown_pct = 12.0
    table = matrix_table(metrics)
    risky = next(line for line in table.splitlines() if "risky" in line)
    calm = next(line for line in table.splitlines() if "calm" in line)
    assert "84% ⚠️" in risky
    assert "⚠️" not in calm


def test_matrix_table_flags_scale_sensitive_strategies():
    """A strategy whose return changes with account size gets a dagger."""
    metrics = [
        # proportional: same return at both start balances
        _metrics("buy_and_hold", 1_500.0), _metrics("buy_and_hold", 1_500_000.0,
                                                    balance=1_000_000.0),
        # not proportional: min order size blocks rebalances on the small account
        _metrics("universal_kelly", 1_300.0),
        _metrics("universal_kelly", 1_010_000.0, balance=1_000_000.0),
    ]
    table = matrix_table(metrics)
    rows = {line.split("|")[2].strip(): line for line in table.splitlines()
            if line.startswith("| ")}
    assert not any("†" in name for name in rows if "buy_and_hold" in name)
    assert any("†" in name for name in rows if "universal_kelly" in name)
    assert "minimum order size" in table


def test_matrix_table_missing_market_shows_dash():
    metrics = [_metrics("buy_and_hold", 1_500.0),
               _metrics("macd_cross", 800.0),
               _metrics("macd_cross", 900.0, "futures_5x")]
    table = matrix_table(metrics)
    row = next(line for line in table.splitlines() if "buy_and_hold" in line)
    assert row.count("—") == 1  # buy_and_hold has no futures run
