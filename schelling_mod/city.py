"""City grid and neighborhood logic."""

import numpy as np

from .agent import Agent
from .feature import Feature, FeatureType


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
        agent_values: dict,
        agent_values_std: float,
        similarity_threshold: float,
    ) -> None:
        """Instantiate city features and agents from the raw map."""
        self.city = np.empty(self.raw_map.shape, dtype=object)
        current_agent_id = 0

        for (i, j), value in np.ndenumerate(self.raw_map):
            team_id = self.raw_map[i][j].item()

            if team_id != 0:
                agent = Agent(
                    current_agent_id,
                    team_id,
                    agent_values.get(team_id),
                    agent_values_std,
                    similarity_threshold,
                )
            else:
                agent = None

            self.city[i][j] = Feature(FeatureType.HOUSE, [i, j], agent)
            current_agent_id += 1

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

    def get_random_empty_house_position(self) -> list[int]:
        """Return the position of a random empty house."""
        empty_houses = []
        for (i, j), value in np.ndenumerate(self.city):
            feature = self.city[i, j]
            if feature.type == FeatureType.HOUSE and feature.agent is None:
                empty_houses.append([i, j])

        return empty_houses[np.random.choice(len(empty_houses))]

    def get_mean_similarity_ratio(self, n_neighbors) -> float:
        """Return the mean similarity ratio across all occupied houses."""
        similarities = []
        for (i, j), feature in np.ndenumerate(self.city):
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                neighborhood = self.get_neighbors(i, j, n_neighbors)
                similarities.append(feature.agent.get_similarity_ratio(neighborhood))

        return np.average(similarities)

    def __repr__(self) -> str:
        return "City"
