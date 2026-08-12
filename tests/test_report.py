from tradebot.report import _mmoney, _money, markdown_table
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


def _metrics(name: str, final: float) -> Metrics:
    return Metrics(
        strategy=name, market="spot", start_balance=1_000.0,
        final_balance=final, profit=final - 1_000.0,
        profit_pct=100.0 * (final / 1_000.0 - 1.0), num_trades=3,
        win_rate_pct=50.0, best_trade=10.0, worst_trade=-5.0, avg_trade=2.0,
        max_drawdown_pct=10.0, sharpe=0.5, time_in_market_pct=40.0,
        fees_paid=1.0, liquidated=False, data_label="SYNTHETIC",
    )


def test_markdown_table_sorted_by_final_balance_desc():
    table = markdown_table([_metrics("worst", 900.0), _metrics("best", 1_100.0)])
    lines = table.splitlines()
    assert lines[0].startswith("| strategy | final balance |")
    assert lines[2].startswith("| best ")
    assert lines[3].startswith("| worst ")
    assert all(line.count("|") == lines[0].count("|") for line in lines)
