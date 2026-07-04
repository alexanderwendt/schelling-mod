"""Versioned run configuration helpers."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1
MIN_GROUP_COUNT = 2
MAX_GROUP_COUNT = 5
DEFAULT_CULTURAL_DISTANCE = 0.5
NEIGHBORHOOD_TYPES = ("moore", "von_neumann")
MOVEMENT_MODES = (
    "random_empty",
    "first_acceptable",
    "best_sampled",
    "best_available",
    "limited_distance",
)

DEFAULT_GROUPS_1A: list[dict[str, object]] = [
    {
        "id": 1,
        "name": "Knights",
        "color": "red",
        "population_share": 0.8,
        "threshold_mean": 0.40,
        "threshold_std": 0.05,
        "income_mean": 1.0,
        "income_std": 0.1,
    },
    {
        "id": 2,
        "name": "Elves",
        "color": "royalblue",
        "population_share": 0.1,
        "threshold_mean": 0.45,
        "threshold_std": 0.05,
        "income_mean": 1.2,
        "income_std": 0.1,
    },
    {
        "id": 3,
        "name": "Orcs",
        "color": "green",
        "population_share": 0.1,
        "threshold_mean": 0.50,
        "threshold_std": 0.05,
        "income_mean": 0.9,
        "income_std": 0.1,
    },
    {
        "id": 4,
        "name": "Dwarves",
        "color": "orange",
        "population_share": 0.25,
        "threshold_mean": 0.45,
        "threshold_std": 0.05,
        "income_mean": 1.1,
        "income_std": 0.1,
    },
    {
        "id": 5,
        "name": "Hobbits",
        "color": "gold",
        "population_share": 0.20,
        "threshold_mean": 0.40,
        "threshold_std": 0.05,
        "income_mean": 0.8,
        "income_std": 0.1,
    },
]


def _pair_key(first_id: int, second_id: int) -> str:
    first, second = sorted((int(first_id), int(second_id)))
    return f"{first}-{second}"


def build_pairwise_distance_defaults(group_ids: list[int]) -> dict[str, float]:
    """Return default cultural distances for every active pair."""
    distances: dict[str, float] = {}
    for index, first_id in enumerate(group_ids):
        for second_id in group_ids[index + 1:]:
            distances[_pair_key(first_id, second_id)] = DEFAULT_CULTURAL_DISTANCE
    return distances


def build_default_config(group_count: int = 3) -> dict[str, Any]:
    """Return full default run config for two to five active groups."""
    if not MIN_GROUP_COUNT <= group_count <= MAX_GROUP_COUNT:
        raise ValueError("group_count must be between 2 and 5")

    groups = copy.deepcopy(DEFAULT_GROUPS_1A[:group_count])
    if group_count != 3:
        equal_share = 1.0 / group_count
        for group in groups:
            group["population_share"] = equal_share

    group_ids = [int(group["id"]) for group in groups]
    return {
        "config_version": CONFIG_VERSION,
        "group_count": group_count,
        "simulation": {
            "population_size": 1000,
            "empty_ratio": 0.2,
            "n_neighbors": 2,
            "n_iterations": 10,
            "seed": 0,
            "neighborhood_type": "moore",
            "property_values_enabled": True,
            "density_preference_enabled": True,
        },
        "groups": groups,
        "cultural_distances": build_pairwise_distance_defaults(group_ids),
        "movement": {
            "mode": "best_sampled",
            "sample_size": 10,
            "movement_search_radius": None,
            "dissatisfied_move_probability": 1.0,
            "satisfied_move_probability": 0.0,
            "require_satisfied_destination": False,
        },
        "economics": {"affordability_enabled": True},
        "export": {"output_dir": None},
    }


def get_active_groups(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return active group dictionaries from config."""
    group_count = int(config.get("group_count", len(config.get("groups", [])) or 3))
    groups = list(config.get("groups", []))[:group_count]
    return [dict(group) for group in groups]


def parse_pairwise_distances(
    config: Mapping[str, Any],
    group_ids: list[int],
) -> dict[tuple[int, int], float]:
    """Return tuple-keyed cultural distances for active group ids."""
    raw = config.get("cultural_distances") or config.get("pairwise_cultural_multipliers") or {}
    result: dict[tuple[int, int], float] = {}
    defaults = build_pairwise_distance_defaults(group_ids)
    for key, default in defaults.items():
        first_text, second_text = key.split("-", maxsplit=1)
        value = raw.get(key, default) if isinstance(raw, Mapping) else default
        result[(int(first_text), int(second_text))] = float(value)
    return result


def _merge_section(default: dict[str, Any], loaded: Any) -> dict[str, Any]:
    merged = copy.deepcopy(default)
    if isinstance(loaded, Mapping):
        for key in merged:
            if key in loaded:
                merged[key] = loaded[key]
    return merged


def migrate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate legacy or partial config to current versioned shape."""
    if not isinstance(config, Mapping):
        return build_default_config()

    requested_group_count = int(config.get("group_count", 3))
    if "groups" in config and isinstance(config["groups"], list):
        requested_group_count = len(config["groups"])
    requested_group_count = max(MIN_GROUP_COUNT, min(MAX_GROUP_COUNT, requested_group_count))

    migrated = build_default_config(requested_group_count)
    migrated["simulation"] = _merge_section(migrated["simulation"], config.get("simulation"))
    migrated["movement"] = _merge_section(migrated["movement"], config.get("movement"))
    migrated["economics"] = _merge_section(migrated["economics"], config.get("economics"))
    migrated["export"] = _merge_section(migrated["export"], config.get("export"))

    if isinstance(config.get("groups"), list):
        defaults_by_id = {int(group["id"]): group for group in migrated["groups"]}
        groups = []
        for loaded_group in config["groups"][:requested_group_count]:
            if not isinstance(loaded_group, Mapping):
                continue
            group_id = int(loaded_group.get("id", len(groups) + 1))
            merged_group = copy.deepcopy(defaults_by_id.get(group_id, DEFAULT_GROUPS_1A[group_id - 1]))
            for key in merged_group:
                if key in loaded_group:
                    merged_group[key] = loaded_group[key]
            groups.append(merged_group)
        if groups:
            migrated["groups"] = groups
            migrated["group_count"] = len(groups)
    if isinstance(config.get("teams"), Mapping):
        teams = config["teams"]
        for group in migrated["groups"]:
            loaded_team = teams.get(str(group["id"]), {})
            if isinstance(loaded_team, Mapping):
                for key in (
                    "population_share",
                    "threshold_mean",
                    "threshold_std",
                    "income_mean",
                    "income_std",
                ):
                    if key in loaded_team:
                        group[key] = loaded_team[key]

    raw_distances = config.get("cultural_distances")
    if raw_distances is None:
        raw_distances = config.get("pairwise_cultural_multipliers")
    if isinstance(raw_distances, Mapping):
        distances = build_pairwise_distance_defaults([int(group["id"]) for group in migrated["groups"]])
        for key in distances:
            if key in raw_distances:
                distances[key] = float(raw_distances[key])
        migrated["cultural_distances"] = distances

    migrated["config_version"] = CONFIG_VERSION
    return migrated


def load_config_json(path: Path) -> dict[str, Any]:
    """Load and migrate config JSON from path."""
    with path.open("r", encoding="utf-8") as config_file:
        loaded = json.load(config_file)
    return migrate_config(loaded)
