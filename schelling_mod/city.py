"""City grid and neighborhood logic."""

from collections.abc import Mapping

import numpy as np

from .agent import Agent
from .feature import Feature, FeatureType
from . import utils


class City:
    """Represents the simulation world."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.raw_map = None
        self.city = None
        self.rng = rng or np.random.default_rng(0)

    def set_rng(self, seed: int | None) -> None:
        """Reset city random generator from seed."""
        self.rng = np.random.default_rng(seed)

    def set_map(self, city_map: np.ndarray) -> None:
        """Set the raw city map."""
        self.raw_map = np.array(city_map, copy=True)
        self.apply_cross_street_layout()

    def apply_cross_street_layout(self) -> None:
        """Partition the map into four blocks using a cross-shaped street."""
        if self.raw_map is None:
            return

        row_count, col_count = self.raw_map.shape
        if row_count < 3 or col_count < 3:
            return
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
        raw_map = self.rng.choice(races, size=city_size, p=probabilities)
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
                    utils.sample_normal_value(threshold_mean, threshold_std, self.rng),
                    utils.sample_normal_value(income_mean, income_std, self.rng),
                    self.rng,
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

    def _is_in_neighborhood(
        self,
        row: int,
        col: int,
        origin_row: int,
        origin_col: int,
        radius: int,
        neighborhood_type: str,
    ) -> bool:
        """Return whether a cell is inside the selected neighborhood shape."""
        row_distance = abs(row - origin_row)
        col_distance = abs(col - origin_col)
        if row_distance == 0 and col_distance == 0:
            return False
        if neighborhood_type == "moore":
            return max(row_distance, col_distance) <= radius
        if neighborhood_type == "von_neumann":
            return row_distance + col_distance <= radius
        raise ValueError(f"Unsupported neighborhood_type: {neighborhood_type}")

    def get_neighbors(
        self,
        row,
        col,
        n_neighbors,
        neighborhood_type: str = "moore",
    ) -> list:
        """Return neighboring occupied house features."""
        neighborhood = []
        for i in range(row - n_neighbors, row + n_neighbors + 1):
            for j in range(col - n_neighbors, col + n_neighbors + 1):
                is_in_bounds = 0 <= i < self.city.shape[0] and 0 <= j < self.city.shape[1]
                if is_in_bounds and self._is_in_neighborhood(i, j, row, col, n_neighbors, neighborhood_type):
                    feature = self.city[i, j]
                    if feature.agent is not None and feature.type == FeatureType.HOUSE:
                        neighborhood.append(feature)

        return neighborhood

    def get_neighborhood_capacity(
        self,
        row: int,
        col: int,
        n_neighbors: int,
        neighborhood_type: str = "moore",
    ) -> int:
        """Return the number of non-barrier neighbor cells around a position."""
        capacity = 0
        for i in range(row - n_neighbors, row + n_neighbors + 1):
            for j in range(col - n_neighbors, col + n_neighbors + 1):
                is_in_bounds = 0 <= i < self.city.shape[0] and 0 <= j < self.city.shape[1]
                if not is_in_bounds or not self._is_in_neighborhood(i, j, row, col, n_neighbors, neighborhood_type):
                    continue

                if self.city[i, j].type != FeatureType.BARRIER:
                    capacity += 1

        return capacity

    def get_immediate_neighbors(self, row: int, col: int) -> list:
        """Return Moore-neighborhood occupied house features."""
        return self.get_neighbors(row, col, 1)

    def get_empty_house_candidates(
        self,
        origin: tuple[int, int] | None = None,
        movement_search_radius: int | None = None,
    ) -> list[tuple[int, int]]:
        """Return valid empty house destinations, optionally within Manhattan radius."""
        candidates = []
        for (row, col), feature in np.ndenumerate(self.city):
            if feature.type != FeatureType.HOUSE or feature.agent is not None:
                continue
            if origin is not None and movement_search_radius is not None:
                if abs(row - origin[0]) + abs(col - origin[1]) > movement_search_radius:
                    continue
            candidates.append((row, col))
        return candidates

    def get_random_empty_house_position(self) -> list[int] | None:
        """Return a seeded random empty house position for legacy callers."""
        candidates = self.get_empty_house_candidates()
        if not candidates:
            return None
        row, col = candidates[int(self.rng.integers(0, len(candidates)))]
        return [int(row), int(col)]

    def get_affordable_empty_house_positions(self, agent: Agent) -> list[list[int]]:
        """Return empty house positions affordable for the agent."""
        positions = []
        for row, col in self.get_empty_house_candidates():
            if agent.can_afford(self.city[row, col].property_value):
                positions.append([int(row), int(col)])
        return positions

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
        """Backward-compatible sampled candidate helper."""
        return self.get_empty_house_candidate(
            agent,
            n_neighbors,
            pairwise_multipliers,
            "moore",
            "best_sampled",
            sample_size,
            None,
            False,
            require_affordable,
            density_preference_enabled,
            density_preference_weight,
            self.rng,
        )

    def get_empty_house_candidate(
        self,
        agent: Agent,
        n_neighbors: int,
        pairwise_distances: Mapping[tuple[int, int], float] | None,
        neighborhood_type: str,
        movement_mode: str,
        sample_size: int,
        movement_search_radius: int | None,
        require_satisfied_destination: bool,
        require_affordable: bool,
        density_preference_enabled: bool,
        density_preference_weight: float,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float | int | list[int]] | None:
        """Select an empty destination using configured movement mode."""
        generator = rng or self.rng
        origin = tuple(agent.position) if hasattr(agent, "position") else None
        if movement_mode == "limited_distance" and movement_search_radius is None:
            movement_search_radius = n_neighbors
        candidates = self.get_empty_house_candidates(origin, movement_search_radius)
        if not candidates:
            return None

        if movement_mode == "random_empty":
            ordered = list(generator.permutation(candidates))
            for row, col in ordered:
                feature = self.city[row, col]
                if not require_affordable or agent.can_afford(feature.property_value):
                    return {"position": [int(row), int(col)], "score": 0.0, "neighbor_count": 0, "property_value": feature.property_value}
            return None

        if movement_mode == "best_sampled" or movement_mode == "limited_distance":
            sample_count = min(max(0, int(sample_size)), len(candidates))
            if sample_count == 0:
                return None
            indexes = generator.choice(len(candidates), size=sample_count, replace=False)
            candidates = [candidates[int(index)] for index in indexes]
        elif movement_mode == "first_acceptable":
            candidates = list(generator.permutation(candidates))
        elif movement_mode != "best_available":
            raise ValueError(f"Unsupported movement_mode: {movement_mode}")

        best_candidates = []
        best_key = None
        for row, col in candidates:
            feature = self.city[row, col]
            if require_affordable and not agent.can_afford(feature.property_value):
                continue
            neighbors = self.get_neighbors(row, col, n_neighbors, neighborhood_type)
            score = agent.get_satisfaction_score(
                neighbors,
                pairwise_distances,
                self.get_neighborhood_capacity(row, col, n_neighbors, neighborhood_type),
                density_preference_enabled,
                density_preference_weight,
                len(neighbors),
            )
            if require_satisfied_destination and score < agent.similarity_threshold:
                continue
            neighbor_count = len(neighbors)
            candidate = {
                "position": [int(row), int(col)],
                "score": float(score),
                "neighbor_count": int(neighbor_count),
                "property_value": float(feature.property_value),
            }
            if movement_mode == "first_acceptable":
                if score >= agent.similarity_threshold:
                    return candidate
                continue
            key = (float(score), int(neighbor_count), -float(feature.property_value))
            if best_key is None or key > best_key:
                best_key = key
                best_candidates = [candidate]
            elif key == best_key:
                best_candidates.append(candidate)

        if not best_candidates:
            return None
        return best_candidates[int(generator.integers(0, len(best_candidates)))]

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
        neighborhood_type: str = "moore",
    ) -> float:
        """Return the mean similarity ratio across all occupied houses."""
        similarities = []
        for (i, j), feature in np.ndenumerate(self.city):
            if feature.type == FeatureType.HOUSE and feature.agent is not None:
                neighborhood = self.get_neighbors(i, j, n_neighbors, neighborhood_type)
                similarities.append(
                    feature.agent.get_similarity_ratio(neighborhood, pairwise_multipliers)
                )

        return np.average(similarities)

    def __repr__(self) -> str:
        return "City"
