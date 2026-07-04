"""Application entry points for the Schelling simulation."""

import argparse
import csv
import io
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap, is_color_like, to_hex
from matplotlib.figure import Figure

from .agent import Agent
from .city import City
from .config import MOVEMENT_MODES
from .config import NEIGHBORHOOD_TYPES
from .config import build_default_config
from .config import build_pairwise_distance_defaults
from .config import get_active_groups
from .config import load_config_json
from .config import migrate_config
from .config import parse_pairwise_distances
from .export import build_run_id
from .export import write_run_export
from .feature import FeatureType
from .metrics import build_metric_row
from .movement import BEST_AVAILABLE_EMPTY_CELL_WARNING_THRESHOLD
from .movement import MovementStats
from .movement import should_warn_best_available

__author__ = "Alexander Wendt"
__copyright__ = "Copyright 2024"
__credits__ = [
    "Credits to Adil Moujahid and his version from https://github.com/adilmoujahid/streamlit-schelling"
]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Alexander Wendt"
__email__ = "alexander.wendt@gmx.at"
__status__ = "Experimental"

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

CITY_MAP = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 2, 1, 2, 1],
        [1, 1, 1, 1, 2],
        [0, 0, 0, 0, 0],
        [2, 1, 2, 1, 2],
    ]
)
N_NEIGHBORS = 2
MENTAL_VALUES_STD_DEV = 0.0
MENTAL_VALUES_MAP = {1: [0, 0], 2: [0.2, 0.2], 3: [0.707, 0.707]}
TEAM_NAMES = {1: "Knights", 2: "Elves", 3: "Orcs"}
TEAMS_DISTRIBUTION = {1: 0.8, 2: 0.1, 3: 0.1}
SIMILARITY_THRESHOLD_DISTRIBUTIONS = {
    1: (0.40, 0.05),
    2: (0.45, 0.05),
    3: (0.50, 0.05),
}
INCOME_DISTRIBUTIONS = {
    1: (1.0, 0.1),
    2: (1.2, 0.1),
    3: (0.9, 0.1),
}
PAIRWISE_CULTURAL_MULTIPLIERS = {
    (1, 2): 1.2,
    (1, 3): 1.5,
    (2, 3): 2.5,
}
DENSITY_PREFERENCE_WEIGHT = 0.1
EMPTY_HOUSE_SAMPLE_SIZE = 10
CELL_COLORS = ["black", "white", "red", "royalblue", "green"]
TEAM_COLORS = {1: "red", 2: "royalblue", 3: "green"}
CONFIG_FILE_PATH = Path("schelling_streamlit_config.json")


def build_group_name_map(groups: list[dict[str, Any]]) -> dict[int, str]:
    """Return group id to display name mapping."""
    return {int(group["id"]): str(group.get("name", group["id"])) for group in groups}


def build_group_color_map(groups: list[dict[str, Any]]) -> dict[int, str]:
    """Return group id to display color mapping."""
    return {int(group["id"]): str(group.get("color", "gray")) for group in groups}


def build_pairwise_label(first_id: int, second_id: int, names: Mapping[int, str]) -> str:
    """Return user-facing pairwise cultural distance label."""
    return f"{names[first_id]} - {names[second_id]} cultural distance"


class Schelling:
    """Simulation container for city generation and agent moves."""

    def __init__(
        self,
        size,
        empty_ratio,
        n_neighbors,
        load_map: bool,
        races=2,
        mental_values_std_dev: float = MENTAL_VALUES_STD_DEV,
        teams_distribution: dict | None = None,
        similarity_threshold_distributions: Mapping[int, tuple[float, float]] | None = None,
        income_distributions: Mapping[int, tuple[float, float]] | None = None,
        pairwise_cultural_multipliers: Mapping[tuple[int, int], float] | None = None,
        property_values_enabled: bool = True,
        density_preference_enabled: bool = True,
        seed: int | None = 0,
        neighborhood_type: str = "moore",
        movement_mode: str = "best_sampled",
        sample_size: int = EMPTY_HOUSE_SAMPLE_SIZE,
        movement_search_radius: int | None = None,
        dissatisfied_move_probability: float = 1.0,
        satisfied_move_probability: float = 0.0,
        require_satisfied_destination: bool = False,
        groups: list[dict[str, Any]] | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.groups = groups or [
            {
                "id": group_id,
                "name": TEAM_NAMES.get(group_id, str(group_id)),
                "color": TEAM_COLORS.get(group_id, "gray"),
                "population_share": (teams_distribution or TEAMS_DISTRIBUTION).get(group_id, 0.0),
                "threshold_mean": SIMILARITY_THRESHOLD_DISTRIBUTIONS.get(group_id, (0.4, 0.05))[0],
                "threshold_std": SIMILARITY_THRESHOLD_DISTRIBUTIONS.get(group_id, (0.4, 0.05))[1],
                "income_mean": INCOME_DISTRIBUTIONS.get(group_id, (1.0, 0.1))[0],
                "income_std": INCOME_DISTRIBUTIONS.get(group_id, (1.0, 0.1))[1],
            }
            for group_id in range(1, races + 1)
        ]
        self.group_name_map = build_group_name_map(self.groups)
        self.group_color_map = build_group_color_map(self.groups)
        active_group_ids = [int(group["id"]) for group in self.groups]
        self.races = np.array([0, *active_group_ids])
        self.n_neighbors = n_neighbors
        self.neighborhood_type = neighborhood_type
        self.movement_mode = movement_mode
        self.sample_size = sample_size
        self.movement_search_radius = movement_search_radius
        self.dissatisfied_move_probability = dissatisfied_move_probability
        self.satisfied_move_probability = satisfied_move_probability
        self.require_satisfied_destination = require_satisfied_destination
        self.property_values_enabled = property_values_enabled
        self.density_preference_enabled = density_preference_enabled
        self.last_agent_actions: dict[int, str] = {}
        self.last_movement_stats = MovementStats()
        self.similarity_threshold_distributions = (
            similarity_threshold_distributions
            or {int(group["id"]): (float(group["threshold_mean"]), float(group["threshold_std"])) for group in self.groups}
        )
        self.income_distributions = income_distributions or {
            int(group["id"]): (float(group["income_mean"]), float(group["income_std"])) for group in self.groups
        }
        self.pairwise_cultural_multipliers = (
            pairwise_cultural_multipliers or PAIRWISE_CULTURAL_MULTIPLIERS
        )
        teams_distribution = teams_distribution or {
            int(group["id"]): float(group.get("population_share", 0.0)) for group in self.groups
        }
        teams_distribution = normalize_teams_distribution(teams_distribution)
        self.city = City(self.rng)

        if load_map:
            self.city.set_map(CITY_MAP)
        else:
            self.city.generate_map(size, empty_ratio, self.races, teams_distribution)

        self.city.instantiate_city(
            {group_id: MENTAL_VALUES_MAP.get(group_id, np.array([0.0, 0.0])) for group_id in active_group_ids},
            mental_values_std_dev,
            self.similarity_threshold_distributions,
            self.income_distributions,
        )

        log.info("City: \n%s", self.city.city)
        log.info("Team map: \n%s", self.city.get_team_map())

    @staticmethod
    def format_feature_for_log(feature, similarity_ratio: float | None = None) -> str:
        """Return a compact one-line feature description for logs."""
        agent = feature.agent
        if agent is None:
            agent_description = "empty"
        else:
            agent_description = (
                f"team={agent.team_id}, id={agent.agent_id}, "
                f"tau={agent.similarity_threshold:.2f}, income={agent.income:.2f}"
            )
            if similarity_ratio is not None:
                agent_description = f"{agent_description}, happiness={similarity_ratio:.4f}"

        return (
            f"pos={feature.position}, value={feature.property_value:.2f}, "
            f"loc={feature.location_multiplier:.2f}, {agent_description}"
        )

    @classmethod
    def format_neighborhood_for_log(cls, neighborhood: list) -> str:
        """Return a multi-line neighborhood description for logs."""
        if not neighborhood:
            return "  none"

        return "\n".join(
            f"  - {cls.format_feature_for_log(neighbor)}"
            for neighbor in neighborhood
        )

    def run(self) -> bool:
        """Run one simulation step and return whether everyone is satisfied."""
        self.last_movement_stats = MovementStats()
        occupied_positions = [
            (row, col)
            for (row, col), feature in np.ndenumerate(self.city.city)
            if feature.type == FeatureType.HOUSE and feature.agent is not None
        ]
        all_agents_happy = True

        for row, col in occupied_positions:
            feature = self.city.city[row, col]
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                current_agent: Agent = feature.agent
                current_agent.position = (row, col)
                neighborhood = self.city.get_neighbors(row, col, self.n_neighbors, self.neighborhood_type)
                satisfaction_score = current_agent.get_satisfaction_score(
                    neighborhood,
                    self.pairwise_cultural_multipliers,
                    self.city.get_neighborhood_capacity(row, col, self.n_neighbors, self.neighborhood_type),
                    self.density_preference_enabled,
                    DENSITY_PREFERENCE_WEIGHT,
                    len(neighborhood),
                )
                agent_log_line = self.format_feature_for_log(feature, satisfaction_score)
                neighborhood_log = self.format_neighborhood_for_log(neighborhood)
                current_neighbor_count = len(self.city.get_immediate_neighbors(row, col))
                is_socially_unhappy = satisfaction_score < current_agent.similarity_threshold
                must_move_for_affordability = (
                    self.property_values_enabled
                    and not current_agent.can_afford(feature.property_value)
                )
                if self.property_values_enabled:
                    is_unhappy = is_socially_unhappy or must_move_for_affordability
                else:
                    is_unhappy = is_socially_unhappy

                chosen_house_candidate = None
                move_probability = (
                    self.dissatisfied_move_probability if is_unhappy else self.satisfied_move_probability
                )
                should_attempt_move = self.rng.random() < move_probability
                if should_attempt_move:
                    chosen_house_candidate = self.city.get_empty_house_candidate(
                        current_agent,
                        self.n_neighbors,
                        self.pairwise_cultural_multipliers,
                        self.neighborhood_type,
                        self.movement_mode,
                        self.sample_size,
                        self.movement_search_radius,
                        self.require_satisfied_destination,
                        self.property_values_enabled,
                        self.density_preference_enabled,
                        DENSITY_PREFERENCE_WEIGHT,
                        self.rng,
                    )

                if is_unhappy or should_attempt_move:
                    all_agents_happy = False
                    chosen_house_position = (
                        chosen_house_candidate["position"]
                        if chosen_house_candidate is not None
                        else None
                    )
                    if chosen_house_position is not None:
                        self.city.city[chosen_house_position[0], chosen_house_position[1]].agent = (
                            feature.agent
                        )
                        self.city.city[row, col].agent = None
                        self.city.recalculate_property_values()
                        distance = abs(chosen_house_position[0] - row) + abs(chosen_house_position[1] - col)
                        self.last_movement_stats.move_count += 1
                        self.last_movement_stats.displacement_distances.append(distance)
                        self.last_agent_actions[current_agent.agent_id] = (
                            f"Moved from [{row}, {col}] to "
                            f"[{chosen_house_position[0]}, {chosen_house_position[1]}]"
                        )
                        reasons = []
                        if is_socially_unhappy:
                            reasons.append("socially unhappy")
                        if must_move_for_affordability:
                            reasons.append("cannot afford current house")
                        if not is_unhappy:
                            reasons.append("exploratory move")
                        decision_log = (
                            f"{' and '.join(reasons)}. Moved from [{row}, {col}] to "
                            f"[{chosen_house_position[0]}, {chosen_house_position[1]}]"
                        )
                    else:
                        self.last_agent_actions[current_agent.agent_id] = (
                            "Unhappy. No available destination."
                        )
                        reasons = []
                        if is_socially_unhappy:
                            reasons.append("socially unhappy")
                        if must_move_for_affordability:
                            reasons.append("cannot afford current house")
                            self.last_movement_stats.affordability_failures += 1
                        decision_log = (
                            f"{' and '.join(reasons)}. No available destination."
                        )
                else:
                    self.last_agent_actions[current_agent.agent_id] = "Happy. Stayed in place."
                    decision_log = "happy. Stayed in place."

                log.debug(
                    "Processing agent:\n  %s\nNeighborhood:\n%s\nDecision:\n  %s",
                    agent_log_line,
                    neighborhood_log,
                    decision_log,
                )
            else:
                log.debug("Feature %s is not a house or house is empty", feature)

        return all_agents_happy


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Schelling Model Modified")
    parser.add_argument(
        "-r",
        "--run_simulation",
        action="store_true",
        help="Run simulation with default parameters",
        required=False,
    )
    parser.add_argument("--config-json", help="Path to versioned run config JSON", type=Path, default=None)
    parser.add_argument("--population_size", "--population-size", help="Population size", type=int, default=None)
    parser.add_argument("--empty_ratio", "--empty-ratio", help="Empty houses ratio", type=float, default=None)
    parser.add_argument(
        "--threshold_std_dev",
        help="Standard deviation used for per-group similarity thresholds",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument("--iterations", help="Number of iterations", type=int, default=None, required=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--neighborhood-radius", type=int, default=None)
    parser.add_argument("--neighborhood-type", choices=NEIGHBORHOOD_TYPES, default=None)
    parser.add_argument("--movement-mode", choices=MOVEMENT_MODES, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--movement-search-radius", type=int, default=None)
    parser.add_argument("--dissatisfied-move-probability", type=float, default=None)
    parser.add_argument("--satisfied-move-probability", type=float, default=None)
    affordability = parser.add_mutually_exclusive_group()
    affordability.add_argument("--affordability", dest="affordability", action="store_true", default=None)
    affordability.add_argument("--no-affordability", dest="affordability", action="store_false")
    density = parser.add_mutually_exclusive_group()
    density.add_argument("--density-preference", dest="density_preference", action="store_true", default=None)
    density.add_argument("--no-density-preference", dest="density_preference", action="store_false")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def build_config_from_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build migrated run config from defaults, optional JSON, and CLI overrides."""
    config = load_config_json(args.config_json) if args.config_json else build_default_config()
    config = migrate_config(config)
    simulation = config["simulation"]
    movement = config["movement"]
    if args.population_size is not None:
        simulation["population_size"] = args.population_size
    if args.empty_ratio is not None:
        simulation["empty_ratio"] = args.empty_ratio
    if args.iterations is not None:
        simulation["n_iterations"] = args.iterations
    if args.seed is not None:
        simulation["seed"] = args.seed
    if args.neighborhood_radius is not None:
        simulation["n_neighbors"] = args.neighborhood_radius
    if args.neighborhood_type is not None:
        simulation["neighborhood_type"] = args.neighborhood_type
    if args.threshold_std_dev is not None:
        for group in config["groups"]:
            group["threshold_std"] = args.threshold_std_dev
    if args.movement_mode is not None:
        movement["mode"] = args.movement_mode
    if args.sample_size is not None:
        movement["sample_size"] = args.sample_size
    if args.movement_search_radius is not None:
        movement["movement_search_radius"] = args.movement_search_radius
    if args.dissatisfied_move_probability is not None:
        movement["dissatisfied_move_probability"] = args.dissatisfied_move_probability
    if args.satisfied_move_probability is not None:
        movement["satisfied_move_probability"] = args.satisfied_move_probability
    if args.affordability is not None:
        config["economics"]["affordability_enabled"] = args.affordability
        simulation["property_values_enabled"] = args.affordability
    if args.density_preference is not None:
        simulation["density_preference_enabled"] = args.density_preference
    if args.output_dir is not None:
        config["export"]["output_dir"] = args.output_dir
    return add_legacy_config_aliases(config)


def run_cli_simulation(args) -> None:
    """Run the non-interactive simulation mode."""
    config = build_config_from_cli_args(args)
    active_groups = get_active_groups(config)
    simulation = config["simulation"]
    movement = config["movement"]
    group_ids = [int(group["id"]) for group in active_groups]
    teams_distribution = normalize_teams_distribution({
        int(group["id"]): float(group["population_share"]) for group in active_groups
    })
    similarity_threshold_distributions = {
        int(group["id"]): (float(group["threshold_mean"]), float(group["threshold_std"]))
        for group in active_groups
    }
    income_distributions = {
        int(group["id"]): (float(group["income_mean"]), float(group["income_std"]))
        for group in active_groups
    }
    schelling = Schelling(
        simulation["population_size"],
        simulation["empty_ratio"],
        simulation["n_neighbors"],
        False,
        len(active_groups),
        MENTAL_VALUES_STD_DEV,
        teams_distribution,
        similarity_threshold_distributions=similarity_threshold_distributions,
        income_distributions=income_distributions,
        pairwise_cultural_multipliers=parse_pairwise_distances(config, group_ids),
        property_values_enabled=simulation["property_values_enabled"],
        density_preference_enabled=simulation["density_preference_enabled"],
        seed=simulation["seed"],
        neighborhood_type=simulation["neighborhood_type"],
        movement_mode=movement["mode"],
        sample_size=movement["sample_size"],
        movement_search_radius=movement["movement_search_radius"],
        dissatisfied_move_probability=movement["dissatisfied_move_probability"],
        satisfied_move_probability=movement["satisfied_move_probability"],
        require_satisfied_destination=movement["require_satisfied_destination"],
        groups=active_groups,
    )
    empty_count = len(schelling.city.get_empty_house_candidates())
    if movement["mode"] == "best_available" and should_warn_best_available(empty_count):
        raise ValueError(
            f"best_available disabled for more than {BEST_AVAILABLE_EMPTY_CELL_WARNING_THRESHOLD} valid empty cells"
        )

    metric_rows = []
    run_id = build_run_id(config)
    group_names = build_group_name_map(active_groups)
    metric_rows.append(build_metric_row(
        run_id, 0, schelling.city, group_names, schelling.pairwise_cultural_multipliers,
        simulation["n_neighbors"], simulation["neighborhood_type"], simulation["density_preference_enabled"],
        MovementStats(), None,
    ))
    convergence_iteration = None
    for iteration_index in range(int(simulation["n_iterations"])):
        all_agents_happy = schelling.run()
        schelling.city.get_mean_similarity_ratio(
            simulation["n_neighbors"],
            schelling.pairwise_cultural_multipliers,
            simulation["neighborhood_type"],
        )
        if all_agents_happy:
            convergence_iteration = iteration_index + 1
            log.info("Simulation converged after %s iteration(s).", iteration_index + 1)
        metric_rows.append(build_metric_row(
            run_id, iteration_index + 1, schelling.city, group_names, schelling.pairwise_cultural_multipliers,
            simulation["n_neighbors"], simulation["neighborhood_type"], simulation["density_preference_enabled"],
            schelling.last_movement_stats, convergence_iteration,
        ))
        if all_agents_happy:
            break
    output_dir = config.get("export", {}).get("output_dir")
    if output_dir:
        write_run_export(Path(output_dir), config, metric_rows)


def build_default_streamlit_config() -> dict[str, Any]:
    """Return the default persisted Streamlit configuration."""
    return add_legacy_config_aliases(build_default_config())


def add_legacy_config_aliases(config: Mapping[str, Any]) -> dict[str, Any]:
    """Add old Streamlit config keys for compatibility."""
    current = migrate_config(config)
    current["teams"] = {
        str(group["id"]): {
            "population_share": group["population_share"],
            "threshold_mean": group["threshold_mean"],
            "threshold_std": group["threshold_std"],
            "income_mean": group["income_mean"],
            "income_std": group["income_std"],
        }
        for group in current["groups"]
    }
    current["pairwise_cultural_multipliers"] = dict(current["cultural_distances"])
    return current


def clamp_float(value: Any, min_value: float, max_value: float, default: float) -> float:
    """Return a float constrained to a widget range."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default

    return min(max(numeric_value, min_value), max_value)


def clamp_int(value: Any, min_value: int, max_value: int, default: int) -> int:
    """Return an int constrained to a widget range."""
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return default

    return min(max(numeric_value, min_value), max_value)


def parse_bool(value: Any, default: bool) -> bool:
    """Return a bool from config-like values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes", "on"}:
            return True
        if value.lower() in {"false", "0", "no", "off"}:
            return False

    return default


def normalize_color_for_picker(value: Any, default: str = "#808080") -> str:
    """Return a hex color accepted by Streamlit color_picker."""
    if isinstance(value, str) and is_color_like(value):
        return to_hex(value)
    return default


def merge_streamlit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a loaded config with defaults, ignoring unknown or missing keys."""
    return add_legacy_config_aliases(config)


def load_streamlit_config(config_path: Path = CONFIG_FILE_PATH) -> dict[str, Any]:
    """Load persisted Streamlit parameters from disk."""
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded_config = json.load(config_file)
    except FileNotFoundError:
        return build_default_streamlit_config()
    except json.JSONDecodeError:
        log.warning("Could not parse config file: %s", config_path)
        return build_default_streamlit_config()

    if not isinstance(loaded_config, Mapping):
        return build_default_streamlit_config()

    if "config_version" in loaded_config:
        return dict(loaded_config)

    return merge_streamlit_config(loaded_config)


def save_streamlit_config(
    config: Mapping[str, Any],
    config_path: Path = CONFIG_FILE_PATH,
) -> None:
    """Persist current Streamlit parameters to disk."""
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")


def persist_streamlit_config_if_changed(
    current_config: Mapping[str, Any],
    loaded_config: Mapping[str, Any],
) -> None:
    """Write the Streamlit config file only when values actually changed."""
    if json.dumps(current_config, sort_keys=True) == json.dumps(loaded_config, sort_keys=True):
        return

    save_streamlit_config(current_config)


def normalize_teams_distribution(teams_distribution: Mapping[int, float]) -> dict[int, float]:
    """Return population shares normalized to sum to one."""
    total_share = sum(max(0.0, float(share)) for share in teams_distribution.values())
    if total_share <= 0:
        equal_share = 1.0 / max(1, len(teams_distribution))
        return {team_id: equal_share for team_id in teams_distribution}

    return {
        team_id: max(0.0, float(share)) / total_share
        for team_id, share in teams_distribution.items()
    }


def build_property_value_map(city: City) -> np.ndarray:
    """Return a numeric map of current house property values."""
    property_value_map = np.zeros(city.city.shape, dtype=float)
    for (row, col), feature in np.ndenumerate(city.city):
        if feature.type == FeatureType.HOUSE:
            property_value_map[row, col] = feature.property_value

    return property_value_map


def build_barrier_mask(city: City) -> np.ndarray:
    """Return a mask for street and barrier cells."""
    barrier_mask = np.zeros(city.city.shape, dtype=bool)
    for (row, col), feature in np.ndenumerate(city.city):
        barrier_mask[row, col] = feature.type == FeatureType.BARRIER

    return barrier_mask


def build_income_map(city: City) -> np.ndarray:
    """Return a numeric map of current resident incomes."""
    income_map = np.zeros(city.city.shape, dtype=float)
    for (row, col), feature in np.ndenumerate(city.city):
        if feature.type == FeatureType.HOUSE and feature.agent is not None:
            income_map[row, col] = feature.agent.income

    return income_map


def get_mean_resident_income(city: City) -> float:
    """Return the mean income across occupied houses."""
    incomes = [
        feature.agent.income
        for _, feature in np.ndenumerate(city.city)
        if feature.type == FeatureType.HOUSE and feature.agent is not None
    ]
    if not incomes:
        return 0.0

    return float(np.mean(incomes))


def build_selected_feature_details(
    schelling: Schelling,
    row: int,
    col: int,
) -> list[dict[str, str]]:
    """Return display rows for the selected city cell."""
    city = schelling.city
    feature = city.city[row, col]
    details = [
        {"Property": "Position", "Value": f"[{row}, {col}]"},
        {"Property": "Feature type", "Value": feature.type.name},
        {"Property": "Property value", "Value": f"{feature.property_value:.4f}"},
        {"Property": "Location multiplier", "Value": f"{feature.location_multiplier:.4f}"},
    ]

    if feature.agent is None:
        details.append({"Property": "Agent", "Value": "empty"})
        return details

    agent = feature.agent
    details.extend(
        [
            {"Property": "Team", "Value": schelling.group_name_map.get(agent.team_id, TEAM_NAMES.get(agent.team_id, str(agent.team_id)))},
            {
                "Property": "Last action",
                "Value": schelling.last_agent_actions.get(agent.agent_id, "Not run yet."),
            },
            {"Property": "Similarity threshold", "Value": f"{agent.similarity_threshold:.4f}"},
            {"Property": "Income", "Value": f"{agent.income:.4f}"},
        ]
    )

    return details


def get_streamlit_team_configuration(config: Mapping[str, Any]) -> tuple[
    dict[int, float],
    dict[int, tuple[float, float]],
    dict[int, tuple[float, float]],
    dict[tuple[int, int], float],
    list[dict[str, Any]],
    dict[str, float],
]:
    """Collect dynamic group and cultural distance settings from Streamlit."""
    config = migrate_config(config)
    teams_distribution = {}
    similarity_threshold_distributions = {}
    income_distributions = {}
    current_groups: list[dict[str, Any]] = []

    st.sidebar.subheader("Group definitions")
    group_count = st.sidebar.selectbox(
        "Group count",
        options=[2, 3, 4, 5],
        index=max(0, min(3, int(config.get("group_count", 3)) - 2)),
        help="MVP supports 2-5 fictional example groups.",
    )
    defaults = build_default_config(group_count)
    defaults_by_id = {int(group["id"]): group for group in defaults["groups"]}
    loaded_by_id = {int(group["id"]): group for group in config.get("groups", [])}

    for group_id in range(1, group_count + 1):
        group_config = {**defaults_by_id[group_id], **loaded_by_id.get(group_id, {})}
        team_name = str(group_config["name"])
        team_color = str(group_config["color"])
        st.sidebar.markdown(
            (
                f"<span style='display:inline-block;width:0.9rem;height:0.9rem;"
                f"background:{team_color};border:1px solid #666;margin-right:0.4rem;"
                f"vertical-align:-0.1rem;'></span><strong>{team_name}</strong>"
            ),
            unsafe_allow_html=True,
        )
        name = st.sidebar.text_input(
            f"Group {group_id} name",
            value=team_name,
            key=f"group_{group_id}_name",
        )
        color = st.sidebar.color_picker(
            f"{name} color",
            value=normalize_color_for_picker(team_color),
            key=f"group_{group_id}_color",
        )

        population_share = st.sidebar.number_input(
            f"{name} population share (%)",
            min_value=0.0,
            max_value=100.0,
            value=clamp_float(
                float(group_config["population_share"]) * 100,
                0.0,
                100.0,
                float(defaults_by_id[group_id]["population_share"]) * 100,
            ),
            step=1.0,
        )
        threshold_mean = st.sidebar.slider(
            f"{name} threshold mean",
            0.0,
            1.0,
            clamp_float(group_config["threshold_mean"], 0.0, 1.0, 0.4),
            0.01,
        )
        threshold_std = st.sidebar.slider(
            f"{name} threshold std dev",
            0.0,
            0.5,
            clamp_float(group_config["threshold_std"], 0.0, 0.5, 0.05),
            0.01,
        )
        income_mean = st.sidebar.number_input(
            f"{name} income mean",
            min_value=0.0,
            value=clamp_float(group_config["income_mean"], 0.0, 100.0, 1.0),
            step=0.1,
        )
        income_std = st.sidebar.number_input(
            f"{name} income std dev",
            min_value=0.0,
            value=clamp_float(group_config["income_std"], 0.0, 100.0, 0.1),
            step=0.05,
        )

        teams_distribution[group_id] = population_share / 100
        similarity_threshold_distributions[group_id] = (threshold_mean, threshold_std)
        income_distributions[group_id] = (income_mean, income_std)
        current_groups.append({
            "id": group_id,
            "name": name,
            "color": color,
            "population_share": population_share / 100,
            "threshold_mean": threshold_mean,
            "threshold_std": threshold_std,
            "income_mean": income_mean,
            "income_std": income_std,
        })

    pairwise_cultural_multipliers = {}
    current_distances = {}
    st.sidebar.header("Preferences / cultural distance")
    st.sidebar.caption("Cultural similarity is max(0, 1 - cultural distance).")
    group_names = build_group_name_map(current_groups)
    active_ids = [int(group["id"]) for group in current_groups]
    defaults = build_pairwise_distance_defaults(active_ids)
    loaded_distances = config.get("cultural_distances") or config.get("pairwise_cultural_multipliers") or {}
    for config_key, default_distance in defaults.items():
        first_text, second_text = config_key.split("-", maxsplit=1)
        first_team_id = int(first_text)
        second_team_id = int(second_text)
        config_key = f"{first_team_id}-{second_team_id}"
        label = build_pairwise_label(first_team_id, second_team_id, group_names)
        distance = st.sidebar.slider(
            label,
            0.0,
            5.0,
            clamp_float(
                loaded_distances.get(config_key, default_distance) if isinstance(loaded_distances, Mapping) else default_distance,
                0.0,
                5.0,
                default_distance,
            ),
            0.1,
        )
        pairwise_cultural_multipliers[(first_team_id, second_team_id)] = distance
        current_distances[config_key] = distance

    return (
        normalize_teams_distribution(teams_distribution),
        similarity_threshold_distributions,
        income_distributions,
        pairwise_cultural_multipliers,
        current_groups,
        current_distances,
    )


def plot_simulation_state(
    schelling: Schelling,
    mean_similarity_ratio: list[float],
    n_iterations: int,
) -> Figure:
    """Plot current city state and summary metrics."""
    plt.style.use("ggplot")
    figure = plt.figure(figsize=(16, 6), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        3,
        width_ratios=[1.5, 1.0, 1.0],
        height_ratios=[0.15, 1.0, 0.15],
        wspace=0.25,
    )
    teams_axis = figure.add_subplot(grid[:, 0])
    property_axis = figure.add_subplot(grid[1, 1])
    similarity_axis = figure.add_subplot(grid[1, 2])

    teams_axis.axis("off")
    teams_axis.set_box_aspect(1)
    teams_axis.set_title("Groups", fontsize=12)
    max_group_id = max(schelling.group_color_map) if schelling.group_color_map else 0
    cell_colors = ["black", "white"] + [
        schelling.group_color_map.get(group_id, "gray")
        for group_id in range(1, max_group_id + 1)
    ]
    cmap = ListedColormap(cell_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(cell_colors) + 0.5, 1), cmap.N)
    teams_axis.pcolor(
        schelling.city.get_team_map() + 1,
        cmap=cmap,
        norm=norm,
        edgecolors="w",
        linewidths=1,
    )
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Empty house"),
        mpatches.Patch(facecolor="black", edgecolor="black", label="Street/barrier"),
    ]
    legend_handles.extend(
        mpatches.Patch(facecolor=color, edgecolor="black", label=schelling.group_name_map[group_id])
        for group_id, color in schelling.group_color_map.items()
    )
    teams_axis.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True)

    property_axis.axis("off")
    property_axis.set_title("Housing cost / attractiveness", fontsize=12)
    property_values = np.ma.masked_array(
        build_property_value_map(schelling.city),
        mask=build_barrier_mask(schelling.city),
    )
    property_cmap = plt.get_cmap("viridis").copy()
    property_cmap.set_bad("black")
    property_mesh = property_axis.pcolor(
        property_values,
        cmap=property_cmap,
        edgecolors="w",
        linewidths=1,
    )
    figure.colorbar(property_mesh, ax=property_axis, fraction=0.046, pad=0.03)
    property_axis.legend(
        handles=[mpatches.Patch(facecolor="white", edgecolor="black", label="Street")],
        loc="lower left",
        fontsize=8,
        frameon=True,
    )

    similarity_axis.set_xlabel("Iterations")
    similarity_axis.set_xlim([0, n_iterations + 1])
    similarity_axis.set_ylim([0.0, 1])
    similarity_axis.set_title("Mean Similarity Ratio", fontsize=12)
    similarity_axis.plot(range(len(mean_similarity_ratio)), mean_similarity_ratio)
    similarity_axis.text(
        0,
        0.95,
        "Similarity Ratio: %.4f" % mean_similarity_ratio[-1],
        fontsize=10,
    )

    return figure


def metric_rows_to_csv_bytes(metric_rows: list[Mapping[str, object]]) -> bytes:
    """Return metrics rows as CSV bytes for Streamlit downloads."""
    output = io.StringIO()
    fieldnames: list[str] = []
    for row in metric_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if fieldnames:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)
    return output.getvalue().encode("utf-8")


def config_to_json_bytes(config: Mapping[str, Any]) -> bytes:
    """Return config as formatted JSON bytes for Streamlit downloads."""
    return json.dumps(migrate_config(config), indent=2, sort_keys=True).encode("utf-8")


def run_streamlit_app(args) -> None:
    """Run the Streamlit interface."""
    if args.run_simulation:
        run_cli_simulation(args)
        return

    st.set_page_config(layout="wide")
    st.title("Schelling's Model of Segregation")

    if "loaded_streamlit_config" not in st.session_state:
        st.session_state.loaded_streamlit_config = load_streamlit_config()

    config = migrate_config(st.session_state.loaded_streamlit_config)
    simulation_config = config["simulation"]
    movement_config = config["movement"]
    economics_config = config["economics"]

    st.sidebar.header("Population / groups")
    population_size = st.sidebar.slider(
        "Population Size",
        9,
        10000,
        clamp_int(simulation_config["population_size"], 9, 10000, 1000),
    )
    empty_ratio = st.sidebar.slider(
        "Empty Houses Ratio",
        0.0,
        1.0,
        clamp_float(simulation_config["empty_ratio"], 0.0, 1.0, 0.2),
    )
    st.sidebar.header("Run settings")
    n_iterations = st.sidebar.number_input(
        "Number of Iterations",
        min_value=1,
        value=clamp_int(simulation_config["n_iterations"], 1, 10000, 10),
    )
    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=2**31 - 1,
        value=clamp_int(simulation_config.get("seed"), 0, 2**31 - 1, 0),
        help="Same config and seed reproduce map generation, sampling, and movement choices.",
    )

    (
        teams_distribution,
        similarity_threshold_distributions,
        income_distributions,
        pairwise_cultural_multipliers,
        active_groups,
        cultural_distances,
    ) = get_streamlit_team_configuration(config)

    st.sidebar.header("Neighborhoods")
    n_neighbors = st.sidebar.slider(
        "Neighborhood Radius",
        1,
        5,
        clamp_int(simulation_config["n_neighbors"], 1, 5, N_NEIGHBORS),
    )
    neighborhood_type = st.sidebar.selectbox(
        "Neighborhood type",
        options=list(NEIGHBORHOOD_TYPES),
        index=list(NEIGHBORHOOD_TYPES).index(simulation_config.get("neighborhood_type", "moore")),
        help="Moore uses Chebyshev distance; von Neumann uses Manhattan distance.",
    )
    st.sidebar.caption(
        "MVP streets/barriers occupy cells but do not block line-of-sight or movement across streets."
    )

    st.sidebar.header("Movement")
    movement_mode = st.sidebar.selectbox(
        "Movement mode",
        options=list(MOVEMENT_MODES),
        index=list(MOVEMENT_MODES).index(movement_config.get("mode", "best_sampled")),
    )
    sample_size = st.sidebar.number_input(
        "Sample size",
        min_value=1,
        max_value=10000,
        value=clamp_int(movement_config.get("sample_size"), 1, 10000, EMPTY_HOUSE_SAMPLE_SIZE),
        help="Maximum sampled destinations for imperfect-information movement modes.",
    )
    movement_search_radius_value = movement_config.get("movement_search_radius")
    use_search_radius = st.sidebar.checkbox(
        "Limit movement search radius",
        value=movement_search_radius_value is not None or movement_mode == "limited_distance",
    )
    movement_search_radius = None
    if use_search_radius:
        movement_search_radius = st.sidebar.number_input(
            "Movement search radius",
            min_value=1,
            max_value=1000,
            value=clamp_int(movement_search_radius_value, 1, 1000, max(1, n_neighbors)),
        )
    dissatisfied_move_probability = st.sidebar.slider(
        "Dissatisfied move probability",
        0.0,
        1.0,
        clamp_float(movement_config.get("dissatisfied_move_probability"), 0.0, 1.0, 1.0),
        0.01,
    )
    satisfied_move_probability = st.sidebar.slider(
        "Satisfied exploratory move probability",
        0.0,
        1.0,
        clamp_float(movement_config.get("satisfied_move_probability"), 0.0, 1.0, 0.0),
        0.01,
    )
    require_satisfied_destination = st.sidebar.checkbox(
        "Require satisfied destination",
        value=parse_bool(movement_config.get("require_satisfied_destination"), False),
    )

    st.sidebar.header("Economics / attractiveness")
    property_values_enabled = st.sidebar.checkbox(
        "Enable affordability constraints",
        value=parse_bool(economics_config.get("affordability_enabled"), True),
        help="Uses abstract housing-cost / attractiveness scores, not calibrated market prices.",
    )
    density_preference_enabled = st.sidebar.checkbox(
        "Enable density preference",
        value=parse_bool(simulation_config.get("density_preference_enabled"), True),
        help="When enabled: 0.9 * social similarity + 0.1 * occupied density.",
    )

    st.sidebar.header("Export")
    st.sidebar.caption("After a run, download single-run metrics.csv and config.json.")

    group_count = len(active_groups)
    current_config = {
        "config_version": config.get("config_version", 1),
        "group_count": group_count,
        "simulation": {
            "population_size": population_size,
            "empty_ratio": empty_ratio,
            "n_neighbors": n_neighbors,
            "n_iterations": n_iterations,
            "seed": int(seed),
            "neighborhood_type": neighborhood_type,
            "property_values_enabled": property_values_enabled,
            "density_preference_enabled": density_preference_enabled,
        },
        "groups": active_groups,
        "cultural_distances": cultural_distances,
        "movement": {
            "mode": movement_mode,
            "sample_size": int(sample_size),
            "movement_search_radius": None if movement_search_radius is None else int(movement_search_radius),
            "dissatisfied_move_probability": dissatisfied_move_probability,
            "satisfied_move_probability": satisfied_move_probability,
            "require_satisfied_destination": require_satisfied_destination,
        },
        "economics": {"affordability_enabled": property_values_enabled},
        "export": config.get("export", {"output_dir": None}),
    }
    current_config = add_legacy_config_aliases(current_config)

    config_signature = json.dumps(current_config, sort_keys=True)
    if (
        "schelling" not in st.session_state
        or st.session_state.get("schelling_config_signature") != config_signature
    ):
        st.session_state.schelling = Schelling(
            population_size,
            empty_ratio,
            n_neighbors,
            False,
            group_count,
            MENTAL_VALUES_STD_DEV,
            teams_distribution,
            similarity_threshold_distributions=similarity_threshold_distributions,
            income_distributions=income_distributions,
            pairwise_cultural_multipliers=pairwise_cultural_multipliers,
            property_values_enabled=property_values_enabled,
            density_preference_enabled=density_preference_enabled,
            seed=int(seed),
            neighborhood_type=neighborhood_type,
            movement_mode=movement_mode,
            sample_size=int(sample_size),
            movement_search_radius=None if movement_search_radius is None else int(movement_search_radius),
            dissatisfied_move_probability=dissatisfied_move_probability,
            satisfied_move_probability=satisfied_move_probability,
            require_satisfied_destination=require_satisfied_destination,
            groups=active_groups,
        )
        st.session_state.mean_similarity_ratio = [
            st.session_state.schelling.city.get_mean_similarity_ratio(
                n_neighbors,
                st.session_state.schelling.pairwise_cultural_multipliers,
                neighborhood_type,
            )
        ]
        st.session_state.schelling_config_signature = config_signature
        st.session_state.metric_rows = []

    schelling = st.session_state.schelling
    mean_similarity_ratio = st.session_state.mean_similarity_ratio
    empty_count = len(schelling.city.get_empty_house_candidates())
    best_available_blocked = movement_mode == "best_available" and should_warn_best_available(empty_count)
    if best_available_blocked:
        st.sidebar.warning(
            "best_available disabled for more than "
            f"{BEST_AVAILABLE_EMPTY_CELL_WARNING_THRESHOLD} valid empty cells "
            f"(current: {empty_count})."
        )

    figure = plot_simulation_state(schelling, mean_similarity_ratio, n_iterations)

    map_column, detail_column = st.columns([5, 1.4], gap="large")
    with map_column:
        city_plot = st.pyplot(figure, use_container_width=True)
    with detail_column:
        st.subheader("Selected Cell")
        row_count, col_count = schelling.city.city.shape
        selected_row = st.number_input(
            "Row",
            min_value=0,
            max_value=row_count - 1,
            value=0,
            step=1,
        )
        selected_col = st.number_input(
            "Column",
            min_value=0,
            max_value=col_count - 1,
            value=0,
            step=1,
        )
        selected_cell_table = st.empty()
        selected_cell_table.dataframe(
            build_selected_feature_details(
                schelling,
                int(selected_row),
                int(selected_col),
            ),
            column_config={
                "Property": st.column_config.TextColumn("Property", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
            },
            hide_index=True,
        )
    progress_bar = st.progress(0)

    st.caption(
        "Empty houses are white. Property values use current neighboring incomes and fixed location multipliers."
    )
    with st.expander("Current configuration", expanded=False):
        st.json(migrate_config(current_config))
    st.dataframe(
        {
            "Metric": [
                "Property values active",
                "Density preference active",
                "Mean similarity",
                "Mean property value",
                "Mean resident income",
                "Movement mode",
                "Neighborhood type",
                "Seed",
            ],
            "Value": [
                "active" if property_values_enabled else "inactive",
                "active" if density_preference_enabled else "inactive",
                f"{mean_similarity_ratio[-1]:.4f}",
                f"{float(np.mean(build_property_value_map(schelling.city))):.4f}",
                f"{get_mean_resident_income(schelling.city):.4f}",
                movement_mode,
                neighborhood_type,
                str(seed),
            ],
        },
        hide_index=True,
    )

    if st.sidebar.button("Run Simulation", use_container_width=True, disabled=best_available_blocked):
        save_streamlit_config(current_config)
        st.session_state.loaded_streamlit_config = current_config

        schelling = Schelling(
            population_size,
            empty_ratio,
            n_neighbors,
            False,
            group_count,
            MENTAL_VALUES_STD_DEV,
            teams_distribution,
            similarity_threshold_distributions=similarity_threshold_distributions,
            income_distributions=income_distributions,
            pairwise_cultural_multipliers=pairwise_cultural_multipliers,
            property_values_enabled=property_values_enabled,
            density_preference_enabled=density_preference_enabled,
            seed=int(seed),
            neighborhood_type=neighborhood_type,
            movement_mode=movement_mode,
            sample_size=int(sample_size),
            movement_search_radius=None if movement_search_radius is None else int(movement_search_radius),
            dissatisfied_move_probability=dissatisfied_move_probability,
            satisfied_move_probability=satisfied_move_probability,
            require_satisfied_destination=require_satisfied_destination,
            groups=active_groups,
        )
        mean_similarity_ratio = [
            schelling.city.get_mean_similarity_ratio(
                n_neighbors,
                schelling.pairwise_cultural_multipliers,
                neighborhood_type,
            )
        ]
        st.session_state.schelling = schelling
        st.session_state.mean_similarity_ratio = mean_similarity_ratio
        st.session_state.schelling_config_signature = config_signature
        run_id = build_run_id(current_config)
        group_names = build_group_name_map(active_groups)
        metric_rows = [
            build_metric_row(
                run_id,
                0,
                schelling.city,
                group_names,
                schelling.pairwise_cultural_multipliers,
                n_neighbors,
                neighborhood_type,
                density_preference_enabled,
                MovementStats(),
                None,
            )
        ]

        completed_iterations = 0
        convergence_iteration = None
        for i in range(n_iterations):
            all_agents_happy = schelling.run()
            mean_similarity_ratio.append(
                schelling.city.get_mean_similarity_ratio(
                    n_neighbors,
                    schelling.pairwise_cultural_multipliers,
                    neighborhood_type,
                )
            )
            if all_agents_happy:
                convergence_iteration = i + 1
            metric_rows.append(
                build_metric_row(
                    run_id,
                    i + 1,
                    schelling.city,
                    group_names,
                    schelling.pairwise_cultural_multipliers,
                    n_neighbors,
                    neighborhood_type,
                    density_preference_enabled,
                    schelling.last_movement_stats,
                    convergence_iteration,
                )
            )
            st.session_state.mean_similarity_ratio = mean_similarity_ratio
            figure = plot_simulation_state(schelling, mean_similarity_ratio, n_iterations)

            city_plot.pyplot(figure, use_container_width=True)
            plt.close(figure)
            progress_bar.progress((i + 1.0) / n_iterations)
            completed_iterations = i + 1
            if all_agents_happy:
                st.info(f"Simulation stopped early after {completed_iterations} iteration(s): everybody is happy.")
                break

        st.session_state.metric_rows = metric_rows

        if completed_iterations == n_iterations and n_iterations > 0:
            progress_bar.progress(1.0)

        selected_cell_table.dataframe(
            build_selected_feature_details(
                schelling,
                int(selected_row),
                int(selected_col),
            ),
            column_config={
                "Property": st.column_config.TextColumn("Property", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
            },
            hide_index=True,
        )

    metric_rows = st.session_state.get("metric_rows", [])
    if metric_rows:
        st.subheader("Export")
        first_col, second_col = st.columns(2)
        with first_col:
            st.download_button(
                "Download metrics.csv",
                data=metric_rows_to_csv_bytes(metric_rows),
                file_name="metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with second_col:
            st.download_button(
                "Download config.json",
                data=config_to_json_bytes(current_config),
                file_name="config.json",
                mime="application/json",
                use_container_width=True,
            )


def main() -> None:
    """Run the application."""
    args, _ = build_parser().parse_known_args()
    log.info(args)
    run_streamlit_app(args)


if __name__ == "__main__":
    main()
