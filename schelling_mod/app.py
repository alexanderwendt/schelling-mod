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
from matplotlib.colors import ListedColormap

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
TEAM_COLORS = ["white", "red", "royalblue", "green"]
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

    def run(self) -> None:
        """Run one simulation step over the city grid."""
        occupied_positions = [
            (row, col)
            for (row, col), feature in np.ndenumerate(self.city.city)
            if feature.type == FeatureType.HOUSE and feature.agent is not None
        ]

        for row, col in occupied_positions:
            feature = self.city.city[row, col]
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                current_agent: Agent = feature.agent
                log.debug("Processing feature: %s", feature)
                neighborhood = self.city.get_neighbors(row, col, self.n_neighbors)
                log.debug("Neighborhood: %s", neighborhood)
                if self.property_values_enabled:
                    is_unhappy = current_agent.is_unhappy(
                        neighborhood,
                        feature.property_value,
                        self.pairwise_cultural_multipliers,
                    )
                else:
                    is_unhappy = (
                        current_agent.get_similarity_ratio(
                            neighborhood,
                            self.pairwise_cultural_multipliers,
                        )
                        < current_agent.similarity_threshold
                    )

                if is_unhappy:
                    if self.property_values_enabled:
                        random_house_position = self.city.get_random_affordable_empty_house_position(
                            current_agent
                        )
                    else:
                        random_house_position = self.city.get_random_empty_house_position()
                    if random_house_position is not None:
                        self.city.city[random_house_position[0], random_house_position[1]].agent = (
                            feature.agent
                        )
                        self.city.city[row, col].agent = None
                        self.city.recalculate_property_values()
                        log.debug(
                            "Agent %s is unhappy. Move from %s to %s",
                            current_agent,
                            [row, col],
                            random_house_position,
                        )
                    else:
                        log.debug("Agent %s is unhappy but no affordable house exists", current_agent)
                else:
                    log.debug("Agent %s is happy. Do nothing", current_agent)
            else:
                log.debug("Feature %s is not a house or house is empty", feature)


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

    for _ in range(args.iterations):
        schelling.run()
        schelling.city.get_mean_similarity_ratio(
            N_NEIGHBORS,
            schelling.pairwise_cultural_multipliers,
        )


def build_default_streamlit_config() -> dict[str, Any]:
    """Return the default persisted Streamlit configuration."""
    return {
        "simulation": {
            "population_size": 1000,
            "empty_ratio": 0.2,
            "n_neighbors": N_NEIGHBORS,
            "n_iterations": 10,
            "mental_values_std_dev": MENTAL_VALUES_STD_DEV,
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
) -> None:
    """Plot current city state and summary metrics."""
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.axis("off")
    plt.title("Teams", fontsize=12)
    cmap = ListedColormap(TEAM_COLORS)
    plt.pcolor(schelling.city.get_team_map(), cmap=cmap, edgecolors="w", linewidths=1)

    plt.subplot(132)
    plt.axis("off")
    plt.title("Property Value", fontsize=12)
    plt.pcolor(
        build_property_value_map(schelling.city),
        cmap="viridis",
        edgecolors="w",
        linewidths=1,
    )
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(133)
    plt.xlabel("Iterations")
    plt.xlim([0, n_iterations + 1])
    plt.ylim([0.0, 1])
    plt.title("Mean Similarity Ratio", fontsize=12)
    plt.plot(range(len(mean_similarity_ratio)), mean_similarity_ratio)
    plt.text(
        0,
        0.95,
        "Similarity Ratio: %.4f" % mean_similarity_ratio[-1],
        fontsize=10,
    )


def run_streamlit_app(args) -> None:
    """Run the Streamlit interface."""
    if args.run_simulation:
        run_cli_simulation(args)
        return

    st.title("Schelling's Model of Segregation")

    config = load_streamlit_config()
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
    mental_values_std_dev = st.sidebar.slider(
        "Mental Values Std Dev",
        0.0,
        0.5,
        clamp_float(simulation_config["mental_values_std_dev"], 0.0, 0.5, MENTAL_VALUES_STD_DEV),
        0.01,
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
            "mental_values_std_dev": mental_values_std_dev,
            "property_values_enabled": property_values_enabled,
        },
        "teams": team_config["teams"],
        "pairwise_cultural_multipliers": team_config["pairwise_cultural_multipliers"],
    }
    save_streamlit_config(current_config)

    schelling = Schelling(
        population_size,
        empty_ratio,
        n_neighbors,
        False,
        len(MENTAL_VALUES_MAP),
        mental_values_std_dev,
        teams_distribution,
        similarity_threshold_distributions=similarity_threshold_distributions,
        income_distributions=income_distributions,
        pairwise_cultural_multipliers=pairwise_cultural_multipliers,
        property_values_enabled=property_values_enabled,
    )

    mean_similarity_ratio = [
        schelling.city.get_mean_similarity_ratio(
            n_neighbors,
            schelling.pairwise_cultural_multipliers,
        )
    ]

    plot_simulation_state(schelling, mean_similarity_ratio, n_iterations)

    city_plot = st.pyplot(plt)
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
                property_values_enabled,
                mean_similarity_ratio[-1],
                float(np.mean(build_property_value_map(schelling.city))),
                get_mean_resident_income(schelling.city),
            ],
        },
        hide_index=True,
    )

    if st.sidebar.button("Run Simulation"):
        for i in range(n_iterations):
            schelling.run()
            mean_similarity_ratio.append(
                schelling.city.get_mean_similarity_ratio(
                    n_neighbors,
                    schelling.pairwise_cultural_multipliers,
                )
            )
            plot_simulation_state(schelling, mean_similarity_ratio, n_iterations)

            city_plot.pyplot(plt)
            plt.close("all")
            progress_bar.progress((i + 1.0) / n_iterations)


def main() -> None:
    """Run the application."""
    args, _ = build_parser().parse_known_args()
    log.info(args)
    run_streamlit_app(args)


if __name__ == "__main__":
    main()
