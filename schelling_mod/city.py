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
        self.raw_map = np.array(city_map, copy=True)
        self.apply_cross_street_layout()

    def apply_cross_street_layout(self) -> None:
        """Partition the map into four blocks using a cross-shaped street."""
        if self.raw_map is None:
            return

        row_count, col_count = self.raw_map.shape
        center_row = row_count // 2
        center_col = col_count // 2
        self.raw_map[center_row, :] = -1
        self.raw_map[:, center_col] = -1

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
        self.apply_cross_street_layout()

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

            if team_id == -1:
                self.city[i][j] = Feature(
                    FeatureType.BARRIER,
                    [i, j],
                    agent=None,
                    location_multiplier=location_multiplier,
                    property_value=0.0,
                )
                continue

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
            if self.city[i][j].type == FeatureType.BARRIER:
                current_team_map[i][j] = -1
            elif self.city[i][j].agent is not None:
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

    def get_best_sampled_empty_house_position(
        self,
        agent: Agent,
        n_neighbors: int,
        pairwise_multipliers: Mapping[tuple[int, int], float] | None = None,
        sample_size: int = 10,
        require_affordable: bool = False,
    ) -> list[int] | None:
        """Sample random cells and return the best valid empty house for the agent."""
        row_count, col_count = self.city.shape
        total_cells = row_count * col_count
        candidate_count = min(sample_size, total_cells)
        sampled_indices = np.random.choice(total_cells, size=candidate_count, replace=False)

        best_position = None
        best_score = None
        best_property_value = None

        for flat_index in sampled_indices:
            row, col = np.unravel_index(flat_index, self.city.shape)
            feature = self.city[row, col]
            if feature.type != FeatureType.HOUSE or feature.agent is not None:
                continue
            if require_affordable and not agent.can_afford(feature.property_value):
                continue

            neighborhood = self.get_neighbors(row, col, n_neighbors)
            score = agent.get_similarity_ratio(neighborhood, pairwise_multipliers)

            if (
                best_position is None
                or score > best_score
                or (
                    score == best_score
                    and feature.property_value < best_property_value
                )
            ):
                best_position = [row, col]
                best_score = score
                best_property_value = feature.property_value

        return best_position

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
