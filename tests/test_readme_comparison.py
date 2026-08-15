"""Every registered strategy MUST be described and MUST appear in the
README comparison table — these tests make that a CI-enforced rule."""

from pathlib import Path

from tradebot.registry import available_strategies
from tradebot.report import README_BEGIN, README_END, update_readme

README = Path(__file__).resolve().parents[1] / "README.md"


def _comparison_section() -> str:
    text = README.read_text()
    assert README_BEGIN in text and README_END in text, \
        "README.md lost its comparison markers"
    return text.split(README_BEGIN)[1].split(README_END)[0]


def test_every_registered_strategy_is_in_readme_table():
    section = _comparison_section()
    for name in available_strategies():
        assert f"[{name}](" in section, (
            f"strategy {name!r} is missing from the README comparison table - "
            "run the full 'tradebot run' and commit the regenerated README")


def test_every_registered_strategy_describes_its_idea():
    for name, cls in available_strategies().items():
        doc = (cls.__doc__ or "").strip()
        assert doc, f"strategy {name!r} must have a docstring describing the idea"
        assert len(doc.splitlines()[0]) >= 10, (
            f"strategy {name!r}: first docstring line should describe the idea")


def test_update_readme_splices_between_markers(tmp_path):
    from test_report import _metrics  # reuse the metrics factory

    readme = tmp_path / "README.md"
    readme.write_text(
        f"# head\n\n{README_BEGIN}\nold table\n{README_END}\n\ntail stays\n")
    metrics = [_metrics("buy_and_hold", 2_000.0), _metrics("macd_cross", 500.0)]
    assert update_readme(metrics, readme, period="test period")

    text = readme.read_text()
    assert "old table" not in text
    assert text.startswith("# head")
    assert text.rstrip().endswith("tail stays")
    assert "test period" in text
    section = text.split(README_BEGIN)[1].split(README_END)[0]
    # sorted best to worst, and ranked with medal badges
    assert section.index("buy_and_hold") < section.index("macd_cross")
    assert "| 🥇1 | " in section and "| 🥈2 | " in section


def test_update_readme_returns_false_without_markers(tmp_path):
    from test_report import _metrics

    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    assert not update_readme([_metrics("x", 1.0)], readme)
    assert readme.read_text() == "no markers here\n"
    assert not update_readme([_metrics("x", 1.0)], tmp_path / "missing.md")
