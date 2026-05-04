"""Application entry points for the Schelling simulation."""

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from .agent import Agent
from .city import City
from .feature import FeatureType

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
CELL_COLORS = ["black", "white", "red", "royalblue", "green"]
TEAM_COLORS = {1: "red", 2: "royalblue", 3: "green"}
CONFIG_FILE_PATH = Path("schelling_streamlit_config.json")


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
    ):
        self.races = np.arange(races + 1)
        self.n_neighbors = n_neighbors
        self.property_values_enabled = property_values_enabled
        self.last_agent_actions: dict[int, str] = {}
        self.similarity_threshold_distributions = (
            similarity_threshold_distributions or SIMILARITY_THRESHOLD_DISTRIBUTIONS
        )
        self.income_distributions = income_distributions or INCOME_DISTRIBUTIONS
        self.pairwise_cultural_multipliers = (
            pairwise_cultural_multipliers or PAIRWISE_CULTURAL_MULTIPLIERS
        )
        self.city = City()

        if load_map:
            self.city.set_map(CITY_MAP)
        else:
            self.city.generate_map(size, empty_ratio, self.races, teams_distribution)

        self.city.instantiate_city(
            MENTAL_VALUES_MAP,
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
                neighborhood = self.city.get_neighbors(row, col, self.n_neighbors)
                similarity_ratio = current_agent.get_similarity_ratio(
                    neighborhood,
                    self.pairwise_cultural_multipliers,
                )
                agent_log_line = self.format_feature_for_log(feature, similarity_ratio)
                neighborhood_log = self.format_neighborhood_for_log(neighborhood)
                is_socially_unhappy = similarity_ratio < current_agent.similarity_threshold
                must_move_for_affordability = (
                    self.property_values_enabled
                    and not current_agent.can_afford(feature.property_value)
                )
                if self.property_values_enabled:
                    is_unhappy = is_socially_unhappy or must_move_for_affordability
                else:
                    is_unhappy = is_socially_unhappy

                if is_unhappy:
                    all_agents_happy = False
                    if self.property_values_enabled:
                        chosen_house_position = self.city.get_best_sampled_empty_house_position(
                            current_agent,
                            self.n_neighbors,
                            self.pairwise_cultural_multipliers,
                            require_affordable=True,
                        )
                    else:
                        chosen_house_position = self.city.get_best_sampled_empty_house_position(
                            current_agent,
                            self.n_neighbors,
                            self.pairwise_cultural_multipliers,
                        )
                    if chosen_house_position is not None:
                        self.city.city[chosen_house_position[0], chosen_house_position[1]].agent = (
                            feature.agent
                        )
                        self.city.city[row, col].agent = None
                        self.city.recalculate_property_values()
                        self.last_agent_actions[current_agent.agent_id] = (
                            f"Moved from [{row}, {col}] to "
                            f"[{chosen_house_position[0]}, {chosen_house_position[1]}]"
                        )
                        reasons = []
                        if is_socially_unhappy:
                            reasons.append("socially unhappy")
                        if must_move_for_affordability:
                            reasons.append("cannot afford current house")
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
    parser.add_argument("--population_size", help="Population size", type=int, default=2500, required=False)
    parser.add_argument("--empty_ratio", help="Empty houses ratio", type=float, default=0.2, required=False)
    parser.add_argument(
        "--threshold_std_dev",
        help="Standard deviation used for per-group similarity thresholds",
        type=float,
        default=0.05,
        required=False,
    )
    parser.add_argument("--iterations", help="Number of iterations", type=int, default=10, required=False)
    return parser


def run_cli_simulation(args) -> None:
    """Run the non-interactive simulation mode."""
    similarity_threshold_distributions = {
        team_id: (distribution[0], args.threshold_std_dev)
        for team_id, distribution in SIMILARITY_THRESHOLD_DISTRIBUTIONS.items()
    }
    schelling = Schelling(
        args.population_size,
        args.empty_ratio,
        1,
        False,
        len(MENTAL_VALUES_MAP),
        MENTAL_VALUES_STD_DEV,
        TEAMS_DISTRIBUTION,
        similarity_threshold_distributions=similarity_threshold_distributions,
    )

    for iteration_index in range(args.iterations):
        all_agents_happy = schelling.run()
        schelling.city.get_mean_similarity_ratio(
            N_NEIGHBORS,
            schelling.pairwise_cultural_multipliers,
        )
        if all_agents_happy:
            log.info("Simulation converged after %s iteration(s).", iteration_index + 1)
            break


def build_default_streamlit_config() -> dict[str, Any]:
    """Return the default persisted Streamlit configuration."""
    return {
        "simulation": {
            "population_size": 1000,
            "empty_ratio": 0.2,
            "n_neighbors": N_NEIGHBORS,
            "n_iterations": 10,
            "property_values_enabled": True,
        },
        "teams": {
            str(team_id): {
                "population_share": TEAMS_DISTRIBUTION[team_id],
                "threshold_mean": threshold_distribution[0],
                "threshold_std": threshold_distribution[1],
                "income_mean": INCOME_DISTRIBUTIONS[team_id][0],
                "income_std": INCOME_DISTRIBUTIONS[team_id][1],
            }
            for team_id, threshold_distribution in SIMILARITY_THRESHOLD_DISTRIBUTIONS.items()
        },
        "pairwise_cultural_multipliers": {
            f"{first_team_id}-{second_team_id}": multiplier
            for (first_team_id, second_team_id), multiplier in PAIRWISE_CULTURAL_MULTIPLIERS.items()
        },
    }


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


def merge_streamlit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a loaded config with defaults, ignoring unknown or missing keys."""
    merged_config = build_default_streamlit_config()
    simulation_config = config.get("simulation", {})
    if isinstance(simulation_config, Mapping):
        for key in merged_config["simulation"]:
            if key in simulation_config:
                merged_config["simulation"][key] = simulation_config[key]

    team_config = config.get("teams", {})
    if isinstance(team_config, Mapping):
        for team_id, defaults in merged_config["teams"].items():
            loaded_team_config = team_config.get(team_id, {})
            if isinstance(loaded_team_config, Mapping):
                for key in defaults:
                    if key in loaded_team_config:
                        defaults[key] = loaded_team_config[key]

    pairwise_config = config.get("pairwise_cultural_multipliers", {})
    if isinstance(pairwise_config, Mapping):
        for key in merged_config["pairwise_cultural_multipliers"]:
            if key in pairwise_config:
                merged_config["pairwise_cultural_multipliers"][key] = pairwise_config[key]

    return merged_config


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
        return TEAMS_DISTRIBUTION.copy()

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
            {"Property": "Team", "Value": TEAM_NAMES.get(agent.team_id, str(agent.team_id))},
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
    dict[str, Any],
]:
    """Collect per-team backend parameters from Streamlit controls."""
    teams_distribution = {}
    similarity_threshold_distributions = {}
    income_distributions = {}
    current_config = {
        "teams": {},
        "pairwise_cultural_multipliers": {},
    }

    st.sidebar.subheader("Team Parameters")
    for team_id, team_name in TEAM_NAMES.items():
        team_color = TEAM_COLORS[team_id]
        st.sidebar.markdown(
            (
                f"<span style='display:inline-block;width:0.9rem;height:0.9rem;"
                f"background:{team_color};border:1px solid #666;margin-right:0.4rem;"
                f"vertical-align:-0.1rem;'></span><strong>{team_name}</strong>"
            ),
            unsafe_allow_html=True,
        )
        team_config = config["teams"][str(team_id)]

        population_share = st.sidebar.number_input(
            f"{team_name} population share",
            min_value=0.0,
            max_value=100.0,
            value=clamp_float(
                float(team_config["population_share"]) * 100,
                0.0,
                100.0,
                TEAMS_DISTRIBUTION[team_id] * 100,
            ),
            step=1.0,
        )
        threshold_mean = st.sidebar.slider(
            f"{team_name} threshold mean",
            0.0,
            1.0,
            clamp_float(team_config["threshold_mean"], 0.0, 1.0, 0.4),
            0.01,
        )
        threshold_std = st.sidebar.slider(
            f"{team_name} threshold std dev",
            0.0,
            0.5,
            clamp_float(team_config["threshold_std"], 0.0, 0.5, 0.05),
            0.01,
        )
        income_mean = st.sidebar.number_input(
            f"{team_name} income mean",
            min_value=0.0,
            value=clamp_float(team_config["income_mean"], 0.0, 100.0, 1.0),
            step=0.1,
        )
        income_std = st.sidebar.number_input(
            f"{team_name} income std dev",
            min_value=0.0,
            value=clamp_float(team_config["income_std"], 0.0, 100.0, 0.1),
            step=0.05,
        )

        teams_distribution[team_id] = population_share / 100
        similarity_threshold_distributions[team_id] = (threshold_mean, threshold_std)
        income_distributions[team_id] = (income_mean, income_std)
        current_config["teams"][str(team_id)] = {
            "population_share": population_share / 100,
            "threshold_mean": threshold_mean,
            "threshold_std": threshold_std,
            "income_mean": income_mean,
            "income_std": income_std,
        }

    pairwise_cultural_multipliers = {}
    st.sidebar.subheader("Cultural Distance")
    for team_pair, default_multiplier in PAIRWISE_CULTURAL_MULTIPLIERS.items():
        first_team_id, second_team_id = team_pair
        config_key = f"{first_team_id}-{second_team_id}"
        label = f"{TEAM_NAMES[first_team_id]} - {TEAM_NAMES[second_team_id]}"
        pairwise_cultural_multipliers[team_pair] = st.sidebar.slider(
            label,
            0.0,
            5.0,
            clamp_float(
                config["pairwise_cultural_multipliers"].get(config_key, default_multiplier),
                0.0,
                5.0,
                default_multiplier,
            ),
            0.1,
        )
        current_config["pairwise_cultural_multipliers"][config_key] = (
            pairwise_cultural_multipliers[team_pair]
        )

    return (
        normalize_teams_distribution(teams_distribution),
        similarity_threshold_distributions,
        income_distributions,
        pairwise_cultural_multipliers,
        current_config,
    )


def plot_simulation_state(
    schelling: Schelling,
    mean_similarity_ratio: list[float],
    n_iterations: int,
) -> Figure:
    """Plot current city state and summary metrics."""
    plt.style.use("ggplot")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.5),
        gridspec_kw={
            "width_ratios": [1.0, 1.15, 1.35],
            "wspace": 0.45,
        },
        constrained_layout=True,
    )

    teams_axis, property_axis, similarity_axis = axes

    teams_axis.axis("off")
    teams_axis.set_title("Teams", fontsize=12)
    cmap = ListedColormap(CELL_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(CELL_COLORS) + 0.5, 1), cmap.N)
    teams_axis.pcolor(
        schelling.city.get_team_map() + 1,
        cmap=cmap,
        norm=norm,
        edgecolors="w",
        linewidths=1,
    )

    property_axis.axis("off")
    property_axis.set_title("Property Value", fontsize=12)
    property_mesh = property_axis.pcolor(
        build_property_value_map(schelling.city),
        cmap="viridis",
        edgecolors="w",
        linewidths=1,
    )
    figure.colorbar(property_mesh, ax=property_axis, fraction=0.046, pad=0.03)

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


def run_streamlit_app(args) -> None:
    """Run the Streamlit interface."""
    if args.run_simulation:
        run_cli_simulation(args)
        return

    st.set_page_config(layout="wide")
    st.title("Schelling's Model of Segregation")

    if "loaded_streamlit_config" not in st.session_state:
        st.session_state.loaded_streamlit_config = load_streamlit_config()

    config = st.session_state.loaded_streamlit_config
    simulation_config = config["simulation"]

    st.sidebar.header("Simulation")
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
    n_neighbors = st.sidebar.slider(
        "Neighborhood Radius",
        1,
        5,
        clamp_int(simulation_config["n_neighbors"], 1, 5, N_NEIGHBORS),
    )
    n_iterations = st.sidebar.number_input(
        "Number of Iterations",
        min_value=1,
        value=clamp_int(simulation_config["n_iterations"], 1, 10000, 10),
    )
    property_values_enabled = st.sidebar.checkbox(
        "Use Property Values",
        value=parse_bool(simulation_config["property_values_enabled"], True),
    )
    (
        teams_distribution,
        similarity_threshold_distributions,
        income_distributions,
        pairwise_cultural_multipliers,
        team_config,
    ) = get_streamlit_team_configuration(config)
    current_config = {
        "simulation": {
            "population_size": population_size,
            "empty_ratio": empty_ratio,
            "n_neighbors": n_neighbors,
            "n_iterations": n_iterations,
            "property_values_enabled": property_values_enabled,
        },
        "teams": team_config["teams"],
        "pairwise_cultural_multipliers": team_config["pairwise_cultural_multipliers"],
    }

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
            len(MENTAL_VALUES_MAP),
            MENTAL_VALUES_STD_DEV,
            teams_distribution,
            similarity_threshold_distributions=similarity_threshold_distributions,
            income_distributions=income_distributions,
            pairwise_cultural_multipliers=pairwise_cultural_multipliers,
            property_values_enabled=property_values_enabled,
        )
        st.session_state.mean_similarity_ratio = [
            st.session_state.schelling.city.get_mean_similarity_ratio(
                n_neighbors,
                st.session_state.schelling.pairwise_cultural_multipliers,
            )
        ]
        st.session_state.schelling_config_signature = config_signature

    schelling = st.session_state.schelling
    mean_similarity_ratio = st.session_state.mean_similarity_ratio

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
    st.dataframe(
        {
            "Metric": [
                "Property values active",
                "Mean similarity",
                "Mean property value",
                "Mean resident income",
            ],
            "Value": [
                "active" if property_values_enabled else "inactive",
                f"{mean_similarity_ratio[-1]:.4f}",
                f"{float(np.mean(build_property_value_map(schelling.city))):.4f}",
                f"{get_mean_resident_income(schelling.city):.4f}",
            ],
        },
        hide_index=True,
    )

    if st.sidebar.button("Run Simulation"):
        completed_iterations = 0
        for i in range(n_iterations):
            all_agents_happy = schelling.run()
            mean_similarity_ratio.append(
                schelling.city.get_mean_similarity_ratio(
                    n_neighbors,
                    schelling.pairwise_cultural_multipliers,
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


def main() -> None:
    """Run the application."""
    args, _ = build_parser().parse_known_args()
    log.info(args)
    run_streamlit_app(args)


if __name__ == "__main__":
    main()
