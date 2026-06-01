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

    def get_neighborhood_capacity(self, row: int, col: int, n_neighbors: int) -> int:
        """Return the number of non-barrier neighbor cells around a position."""
        capacity = 0
        for i in range(row - n_neighbors, row + n_neighbors + 1):
            for j in range(col - n_neighbors, col + n_neighbors + 1):
                is_in_bounds = 0 <= i < self.city.shape[0] and 0 <= j < self.city.shape[1]
                if not is_in_bounds or [i, j] == [row, col]:
                    continue

                if self.city[i, j].type != FeatureType.BARRIER:
                    capacity += 1

        return capacity

    def get_immediate_neighbors(self, row: int, col: int) -> list:
        """Return Moore-neighborhood occupied house features."""
        return self.get_neighbors(row, col, 1)

    def get_property_value_inputs(self, row: int, col: int) -> list[float]:
        """Return occupied neighbor incomes or location fallback for empty houses."""
        values = []
        location_multiplier = self.city[row, col].location_multiplier

        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                is_in_bounds = 0 <= i < self.city.shape[0] and 0 <= j < self.city.shape[1]
                if not is_in_bounds or [i, j] == [row, col]:
                    continue

                feature = self.city[i, j]
                if feature.type != FeatureType.HOUSE:
                    continue

                if feature.agent is not None:
                    values.append(feature.agent.income)
                else:
                    values.append(location_multiplier)

        return values

    def get_best_sampled_empty_house_position(
        self,
        agent: Agent,
        n_neighbors: int,
        pairwise_multipliers: Mapping[tuple[int, int], float] | None = None,
        sample_size: int = 10,
        require_affordable: bool = False,
        density_preference_enabled: bool = False,
        density_preference_weight: float = 0.1,
    ) -> list[int] | None:
        """Sample random cells and return the best valid empty house for the agent."""
        best_candidate = self.get_best_sampled_empty_house_candidate(
            agent,
            n_neighbors,
            pairwise_multipliers,
            sample_size,
            require_affordable,
            density_preference_enabled,
            density_preference_weight,
        )
        if best_candidate is None:
            return None

        return best_candidate["position"]

    def get_best_sampled_empty_house_candidate(
        self,
        agent: Agent,
        n_neighbors: int,
        pairwise_multipliers: Mapping[tuple[int, int], float] | None = None,
        sample_size: int = 10,
        require_affordable: bool = False,
        density_preference_enabled: bool = False,
        density_preference_weight: float = 0.1,
    ) -> dict[str, float | int | list[int]] | None:
        """Sample random cells and return the best valid empty house candidate."""
        row_count, col_count = self.city.shape
        total_cells = row_count * col_count
        candidate_count = min(sample_size, total_cells)
        sampled_indices = np.random.choice(total_cells, size=candidate_count, replace=False)

        best_position = None
        best_score = None
        best_property_value = None
        best_neighbor_count = None

        for flat_index in sampled_indices:
            row, col = np.unravel_index(flat_index, self.city.shape)
            feature = self.city[row, col]
            if feature.type != FeatureType.HOUSE or feature.agent is not None:
                continue
            if require_affordable and not agent.can_afford(feature.property_value):
                continue

            neighborhood = self.get_neighbors(row, col, n_neighbors)
            score = agent.get_satisfaction_score(
                neighborhood,
                pairwise_multipliers,
                self.get_neighborhood_capacity(row, col, 1),
                density_preference_enabled,
                density_preference_weight,
                len(self.get_immediate_neighbors(row, col)),
            )
            neighbor_count = len(self.get_immediate_neighbors(row, col))

            if (
                best_position is None
                or score > best_score
                or (
                    np.isclose(score, best_score)
                    and neighbor_count > best_neighbor_count
                )
                or (
                    np.isclose(score, best_score)
                    and neighbor_count == best_neighbor_count
                    and feature.property_value < best_property_value
                )
            ):
                best_position = [row, col]
                best_score = score
                best_property_value = feature.property_value
                best_neighbor_count = neighbor_count

        if best_position is None:
            return None

        return {
            "position": best_position,
            "score": best_score,
            "neighbor_count": best_neighbor_count,
            "property_value": best_property_value,
        }

    def recalculate_property_values(self) -> None:
        """Recompute property values for all houses."""
        for (row, col), feature in np.ndenumerate(self.city):
            if feature.type != FeatureType.HOUSE:
                continue

            feature.property_value = utils.calculate_property_value(
                self.get_property_value_inputs(row, col),
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
