import pytest

from tradebot.scaffold import new_strategy


def test_scaffold_generates_valid_python(tmp_path):
    path = new_strategy("ema_trend_test", strategies_dir=tmp_path)
    source = path.read_text()
    compile(source, str(path), "exec")  # syntax-valid
    assert 'name = "ema_trend_test"' in source
    assert "class EmaTrendTest(Strategy):" in source
    assert "@register" in source


def test_scaffold_rejects_bad_names(tmp_path):
    for bad in ("Bad-Name", "1abc", "CamelCase", "with space"):
        with pytest.raises(SystemExit):
            new_strategy(bad, strategies_dir=tmp_path)


def test_scaffold_refuses_existing_file(tmp_path):
    new_strategy("dupe_test", strategies_dir=tmp_path)
    with pytest.raises(SystemExit, match="already exists"):
        new_strategy("dupe_test", strategies_dir=tmp_path)


def test_scaffold_refuses_registered_name(tmp_path):
    with pytest.raises(SystemExit, match="already registered"):
        new_strategy("buy_and_hold", strategies_dir=tmp_path)
