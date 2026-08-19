"""The pinned presets.

A preset exists so that "the strategy" means one specific configuration that one
specific set of measurements was made about.  These tests pin the values, so
editing a dataclass default cannot silently redefine a preset and quietly
invalidate every published number.
"""

from __future__ import annotations

import json

import pytest

from gtbot.strategies import (
    DEFAULT_PRESET,
    DISLOCATION_V1,
    DISLOCATION_V2,
    PRESETS,
    available,
    get,
)


def test_registry_is_consistent():
    assert set(available()) == set(PRESETS)
    assert DEFAULT_PRESET in PRESETS
    assert get() is DISLOCATION_V2
    for name, preset in PRESETS.items():
        assert preset.name == name


def test_unknown_preset_is_rejected():
    with pytest.raises(KeyError):
        get("does_not_exist")


def test_shipped_preset_values_are_pinned():
    """If this fails, a default moved and the published numbers are now stale."""
    c = DISLOCATION_V2.config
    assert (c.horizon, c.max_hold) == (3, 3)
    assert c.entry_signal == 0.55
    assert c.exit_signal == 0.10
    assert c.variance_reduction is True
    assert c.adaptive_exit is True
    assert c.direction == "both"
    assert c.sizing_mode == "robust"
    assert c.learner.rule == "hedge"
    assert c.learner.eta == 0.03
    assert c.learner.prior_expert == "transient_dislocation"
    assert c.ambiguity.k_sigma == 0.5
    assert c.ambiguity.model_haircut == 0.90
    assert c.risk.max_leverage == 5.0

    e = DISLOCATION_V2.execution
    assert (e.entry_mode, e.exit_mode) == ("taker", "maker")


def test_v1_is_the_pre_improvement_baseline():
    """v1 must differ from v2 in exactly the two ablated improvements."""
    a, b = DISLOCATION_V1.config, DISLOCATION_V2.config
    assert (a.variance_reduction, a.adaptive_exit) == (False, False)
    assert (b.variance_reduction, b.adaptive_exit) == (True, True)
    differing = {
        k for k in vars(a)
        if repr(getattr(a, k)) != repr(getattr(b, k))
    }
    assert differing == {"variance_reduction", "adaptive_exit"}, differing


def test_build_returns_an_independent_configuration():
    preset = get("dislocation_v2")
    built = preset.build(direction="long_only")
    assert built.cfg.direction == "long_only"
    assert preset.config.direction == "both", "build() must not mutate the preset"
    # Nested configs must be copied too, not shared.
    built.cfg.risk = type(built.cfg.risk)(max_leverage=1.0)
    assert preset.config.risk.max_leverage == 5.0


def test_build_rejects_unknown_overrides():
    with pytest.raises(AttributeError):
        get().build(not_a_field=1)


def test_config_for_tier_tells_the_sizer_the_truth():
    preset = get()
    cheap = preset.config_for_tier("vip9")
    dear = preset.config_for_tier("retail")
    assert dear.assumed_cost_bp > cheap.assumed_cost_bp
    assert cheap.assumed_cost_bp == pytest.approx(2.05)
    assert dear.assumed_cost_bp == pytest.approx(6.65)


def test_metadata_is_complete_and_serialisable():
    for name in available():
        m = get(name).metadata
        assert m.summary and m.thesis and m.instrument
        assert m.limitations, "a preset with no stated limitations is not documented"
        assert m.provenance
        assert m.min_bars > 0
        blob = json.loads(m.to_json())
        assert blob["name"] == name


def test_metadata_records_the_simulated_data_caveat():
    """The most important caveat must not be quietly dropped."""
    text = " ".join(get().metadata.limitations).lower()
    assert "simulat" in text and "real" in text


def test_describe_mentions_the_fee_tier_requirement():
    text = get().describe()
    assert "vip6" in text
    assert "LIMITATIONS" in text
