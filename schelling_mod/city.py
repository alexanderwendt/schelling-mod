"""City grid and neighborhood logic."""

from collections.abc import Mapping

import numpy as np

from .agent import Agent
from .feature import Feature, FeatureType
from . import utils


class City:
    """Represents the simulation world."""

    def __init__(self):
        self.raw_map = None
        self.city = None

    def set_map(self, city_map: np.ndarray) -> None:
        """Set the raw city map."""
        self.raw_map = city_map

    def generate_map(
        self,
        size: int,
        empty_ratio: float,
        races: list[int],
        teams_distribution: dict | None,
    ) -> None:
        """Generate a random city map with empty and occupied houses."""
        probabilities = [empty_ratio]
        if teams_distribution is not None:
            probabilities.extend(
                np.array(list(teams_distribution.values())) * (1 - empty_ratio)
            )
        else:
            probabilities.extend(
                [((1 - empty_ratio) / (len(races) - 1)) for _ in range(len(races) - 1)]
            )

        city_size = int(np.sqrt(size)) ** 2
        raw_map = np.random.default_rng(0).choice(races, size=city_size, p=probabilities)
        edge_length = int(np.sqrt(city_size))
        self.raw_map = np.reshape(raw_map, (edge_length, edge_length))

    def instantiate_city(
        self,
        agent_values: Mapping[int, np.ndarray],
        agent_values_std: float,
        similarity_threshold_distributions: Mapping[int, tuple[float, float]],
        income_distributions: Mapping[int, tuple[float, float]],
    ) -> None:
        """Instantiate city features and agents from the raw map."""
        self.city = np.empty(self.raw_map.shape, dtype=object)
        current_agent_id = 0
        row_count, col_count = self.raw_map.shape

        for (i, j), value in np.ndenumerate(self.raw_map):
            team_id = self.raw_map[i][j].item()
            location_multiplier = utils.calculate_location_multiplier(i, j, row_count, col_count)

            if team_id != 0:
                threshold_mean, threshold_std = similarity_threshold_distributions[team_id]
                income_mean, income_std = income_distributions[team_id]
                agent = Agent(
                    current_agent_id,
                    team_id,
                    agent_values.get(team_id),
                    agent_values_std,
                    utils.sample_normal_value(threshold_mean, threshold_std),
                    utils.sample_normal_value(income_mean, income_std),
                )
            else:
                agent = None

            self.city[i][j] = Feature(
                FeatureType.HOUSE,
                [i, j],
                agent,
                location_multiplier=location_multiplier,
            )
            current_agent_id += 1

        self.recalculate_property_values()

    def get_team_map(self) -> np.ndarray:
        """Return a map containing the current team ids."""
        current_team_map = np.zeros(self.raw_map.shape, dtype=int)
        for (i, j), value in np.ndenumerate(self.raw_map):
            if self.city[i][j].agent is not None:
                current_team_map[i][j] = self.city[i][j].agent.team_id
            else:
                current_team_map[i][j] = 0

        return current_team_map

    def get_neighbors(self, row, col, n_neighbors) -> list:
        """Return neighboring occupied house features."""
        neighborhood = []
        for i in range(row - n_neighbors, row + n_neighbors + 1):
            for j in range(col - n_neighbors, col + n_neighbors + 1):
                is_in_bounds = 0 <= i < self.city.shape[0] and 0 <= j < self.city.shape[1]
                if is_in_bounds and [i, j] != [row, col]:
                    feature = self.city[i, j]
                    if feature.agent is not None and feature.type == FeatureType.HOUSE:
                        neighborhood.append(feature)

        return neighborhood

    def get_immediate_neighbors(self, row: int, col: int) -> list:
        """Return Moore-neighborhood occupied house features."""
        return self.get_neighbors(row, col, 1)

    def get_random_empty_house_position(self) -> list[int] | None:
        """Return the position of a random empty house."""
        empty_houses = []
        for (i, j), value in np.ndenumerate(self.city):
            feature = self.city[i, j]
            if feature.type == FeatureType.HOUSE and feature.agent is None:
                empty_houses.append([i, j])

        if not empty_houses:
            return None

        return empty_houses[np.random.choice(len(empty_houses))]

    def get_affordable_empty_house_positions(self, agent: Agent) -> list[list[int]]:
        """Return all empty house positions affordable for the given agent."""
        affordable_houses = []
        for (i, j), feature in np.ndenumerate(self.city):
            if feature.type == FeatureType.HOUSE and feature.agent is None and agent.can_afford(feature.property_value):
                affordable_houses.append([i, j])

        return affordable_houses

    def get_random_affordable_empty_house_position(self, agent: Agent) -> list[int] | None:
        """Return a random affordable empty house position, if any."""
        affordable_houses = self.get_affordable_empty_house_positions(agent)
        if not affordable_houses:
            return None

        return affordable_houses[np.random.choice(len(affordable_houses))]

    def recalculate_property_values(self) -> None:
        """Recompute property values for all houses."""
        for (row, col), feature in np.ndenumerate(self.city):
            if feature.type != FeatureType.HOUSE:
                continue

            neighboring_incomes = [
                neighbor.agent.income
                for neighbor in self.get_immediate_neighbors(row, col)
                if neighbor.agent is not None
            ]
            feature.property_value = utils.calculate_property_value(
                neighboring_incomes,
                feature.location_multiplier,
            )

    def get_mean_similarity_ratio(
        self,
        n_neighbors: int,
        pairwise_multipliers: Mapping[tuple[int, int], float] | None = None,
    ) -> float:
        """Return the mean similarity ratio across all occupied houses."""
        similarities = []
        for (i, j), feature in np.ndenumerate(self.city):
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                neighborhood = self.get_neighbors(i, j, n_neighbors)
                similarities.append(
                    feature.agent.get_similarity_ratio(neighborhood, pairwise_multipliers)
                )

        return np.average(similarities)

    def __repr__(self) -> str:
        return "City"
