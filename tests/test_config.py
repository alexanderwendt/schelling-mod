"""Tests for versioned run config helpers."""

import json

import pytest

from schelling_mod.config import build_default_config
from schelling_mod.config import get_active_groups
from schelling_mod.config import load_config_json
from schelling_mod.config import migrate_config
from schelling_mod.config import parse_pairwise_distances


def test_default_config_has_version_and_lotr_groups() -> None:
    """Defaults should expose current MVP config shape."""
    config = build_default_config()

    assert config["config_version"] == 1
    assert [group["name"] for group in config["groups"]] == ["Knights", "Elves", "Orcs"]
    assert config["simulation"]["seed"] == 0
    assert config["movement"]["mode"] == "best_sampled"


def test_legacy_config_migrates_pairwise_alias_to_cultural_distance() -> None:
    """Legacy Streamlit keys should load into versioned fields."""
    legacy = {
        "simulation": {"population_size": 4096, "n_neighbors": 2},
        "teams": {"1": {"population_share": 0.2, "income_mean": 1.5}},
        "pairwise_cultural_multipliers": {"1-2": 0.7},
    }

    migrated = migrate_config(legacy)

    assert migrated["config_version"] == 1
    assert migrated["simulation"]["population_size"] == 4096
    assert migrated["groups"][0]["population_share"] == 0.2
    assert migrated["groups"][0]["income_mean"] == 1.5
    assert migrated["cultural_distances"]["1-2"] == 0.7


def test_default_config_supports_two_to_five_groups() -> None:
    """MVP group count range should generate active pairs."""
    for group_count in range(2, 6):
        config = build_default_config(group_count=group_count)

        assert len(get_active_groups(config)) == group_count
        assert len(config["cultural_distances"]) == group_count * (group_count - 1) // 2


def test_group_count_outside_mvp_range_is_rejected() -> None:
    """Group counts outside 2-5 are unsupported."""
    with pytest.raises(ValueError, match="group_count must be between 2 and 5"):
        build_default_config(group_count=6)


def test_load_config_json_and_parse_pairwise_distances(tmp_path) -> None:
    """Exported/migrated JSON should parse tuple-keyed cultural distances."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"group_count": 2, "cultural_distances": {"1-2": 0.25}}), encoding="utf-8")

    config = load_config_json(path)

    assert parse_pairwise_distances(config, [1, 2]) == {(1, 2): 0.25}
