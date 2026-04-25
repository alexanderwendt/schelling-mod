"""Application entry points for the Schelling simulation."""

import argparse
import logging

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


class Schelling:
    """Simulation container for city generation and agent moves."""

    def __init__(
        self,
        size,
        empty_ratio,
        similarity_threshold,
        n_neighbors,
        load_map: bool,
        races=2,
        teams_distribution: dict | None = None,
    ):
        self.races = np.arange(races + 1)
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.city = City()

        if load_map:
            self.city.set_map(CITY_MAP)
        else:
            self.city.generate_map(size, empty_ratio, self.races, teams_distribution)

        self.city.instantiate_city(
            MENTAL_VALUES_MAP,
            MENTAL_VALUES_STD_DEV,
            self.similarity_threshold,
        )

        log.info("City: \n%s", self.city.city)
        log.info("Team map: \n%s", self.city.get_team_map())

    def run(self) -> None:
        """Run one simulation step over the city grid."""
        for (row, col), value in np.ndenumerate(self.city.city):
            feature = self.city.city[row, col]
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                current_agent: Agent = feature.agent
                log.debug("Processing feature: %s", feature)
                neighborhood = self.city.get_neighbors(row, col, self.n_neighbors)
                log.debug("Neighborhood: %s", neighborhood)
                is_unhappy = current_agent.is_unhappy(neighborhood)

                if is_unhappy:
                    random_house_position = self.city.get_random_empty_house_position()
                    self.city.city[random_house_position[0], random_house_position[1]].agent = feature.agent
                    self.city.city[row, col].agent = None
                    log.debug(
                        "Agent %s is unhappy. Move from %s to %s",
                        current_agent,
                        [row, col],
                        random_house_position,
                    )
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
        "--similarity_threshold",
        help="Similarity threshold",
        type=float,
        default=0.4,
        required=False,
    )
    parser.add_argument("--iterations", help="Number of iterations", type=int, default=10, required=False)
    return parser


def run_cli_simulation(args) -> None:
    """Run the non-interactive simulation mode."""
    schelling = Schelling(
        args.population_size,
        args.empty_ratio,
        args.similarity_threshold,
        1,
        False,
        len(MENTAL_VALUES_MAP),
        TEAMS_DISTRIBUTION,
    )

    for _ in range(args.iterations):
        schelling.run()
        schelling.city.get_mean_similarity_ratio(N_NEIGHBORS)


def run_streamlit_app(args) -> None:
    """Run the Streamlit interface."""
    if args.run_simulation:
        run_cli_simulation(args)
        return

    st.title("Schelling's Model of Segregation")

    population_size = st.sidebar.slider("Population Size", 9, 10000, 1000)
    empty_ratio = st.sidebar.slider("Empty Houses Ratio", 0.0, 1.0, 0.2)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.3)
    n_iterations = st.sidebar.number_input("Number of Iterations", 10)

    schelling = Schelling(
        population_size,
        empty_ratio,
        similarity_threshold,
        N_NEIGHBORS,
        False,
        len(MENTAL_VALUES_MAP),
        TEAMS_DISTRIBUTION,
    )

    mean_similarity_ratio = [schelling.city.get_mean_similarity_ratio(N_NEIGHBORS)]

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))

    cmap = ListedColormap(["white", "red", "royalblue", "green"])
    plt.subplot(121)
    plt.axis("off")
    plt.pcolor(schelling.city.get_team_map(), cmap=cmap, edgecolors="w", linewidths=1)

    plt.subplot(122)
    plt.xlabel("Iterations")
    plt.xlim([0, n_iterations + 1])
    plt.ylim([0.0, 1])
    plt.title("Mean Similarity Ratio", fontsize=15)
    plt.text(
        1,
        0.95,
        "Similarity Ratio: %.4f" % schelling.city.get_mean_similarity_ratio(N_NEIGHBORS),
        fontsize=10,
    )

    city_plot = st.pyplot(plt)
    progress_bar = st.progress(0)

    if st.sidebar.button("Run Simulation"):
        for i in range(n_iterations + 1):
            schelling.run()
            mean_similarity_ratio.append(schelling.city.get_mean_similarity_ratio(N_NEIGHBORS))
            plt.figure(figsize=(8, 4))

            plt.subplot(121)
            plt.axis("off")
            plt.pcolor(schelling.city.get_team_map(), cmap=cmap, edgecolors="w", linewidths=1)

            plt.subplot(122)
            plt.xlabel("Iterations")
            plt.xlim([0, n_iterations + 1])
            plt.ylim([0.0, 1])
            plt.title("Mean Similarity Ratio", fontsize=15)
            plt.plot(range(1, len(mean_similarity_ratio) + 1), mean_similarity_ratio)
            plt.text(
                1,
                0.95,
                "Similarity Ratio: %.4f" % schelling.city.get_mean_similarity_ratio(N_NEIGHBORS),
                fontsize=10,
            )

            city_plot.pyplot(plt)
            plt.close("all")
            progress_bar.progress((i + 1.0) / (n_iterations + 1))


def main() -> None:
    """Run the application."""
    args, _ = build_parser().parse_known_args()
    log.info(args)
    run_streamlit_app(args)


if __name__ == "__main__":
    main()
