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

    # header: strategy + one column per config, spot before futures, $1K before $1M
    assert lines[0] == ("| strategy | spot · $1K | spot · $1M "
                        "| futures_5x · $1K | futures_5x · $1M |")
    assert len(lines) == 4  # header, separator, one row per strategy
    # ranked by best final balance: buy_and_hold first
    assert lines[2].startswith("| **buy_and_hold**")
    assert "[source](" in lines[2] and "buy_and_hold.py)" in lines[2]
    # each config cell carries the requested numbers
    assert lines[2].count("trades 3") == 4
    assert lines[2].count("**after ") == 4
    assert "profit " in lines[2] and "worst " in lines[2] and "best " in lines[2]
    assert lines[3].count("LIQUIDATED") == 2  # both futures configs


def test_matrix_table_missing_config_shows_dash():
    metrics = [_metrics("buy_and_hold", 1_500.0),
               _metrics("macd_cross", 800.0),
               _metrics("macd_cross", 900_000.0, "spot", 1_000_000.0)]
    table = matrix_table(metrics)
    row = next(line for line in table.splitlines() if "buy_and_hold" in line)
    assert row.count("—") == 1  # buy_and_hold missing the $1M config
