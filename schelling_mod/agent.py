"""Agent model for the Schelling simulation."""

import logging
from collections.abc import Mapping

import numpy as np

from . import utils

log = logging.getLogger(__name__)


def generate_values_from_distribution(
    mean_value,
    std_dev,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a two-dimensional value vector from a normal distribution."""
    generator = rng or np.random.default_rng()
    return generator.normal(loc=mean_value, scale=std_dev, size=(2,))


class Agent:
    """Represents one agent in the simulation."""

    def __init__(
        self,
        agent_id: int,
        team_id: int,
        mental_values_mean: np.ndarray,
        mental_values_std_dev: float,
        similarity_threshold: float,
        income: float,
        rng: np.random.Generator | None = None,
    ):
        log.info(
            "Init agent with agentId: %s, teamId: %s, mental_values: %s, std: %s, threshold: %s, income: %s",
            agent_id,
            team_id,
            mental_values_mean,
            mental_values_std_dev,
            similarity_threshold,
            income,
        )
        self.agent_id = agent_id
        self.team_id = team_id
        self.mental_values = generate_values_from_distribution(
            mental_values_mean,
            mental_values_std_dev,
            rng,
        )
        self.similarity_threshold = similarity_threshold
        self.income = income

    def get_similarity_ratio(
        self,
        neighborhood: list,
        pairwise_distances: Mapping[tuple[int, int], float] | None = None,
    ) -> float:
        """Return the average similarity to neighboring agents."""
        if neighborhood:
            similarity_ratio = np.average(
                [
                    (
                        utils.CULTURAL_SIMILARITY_WEIGHT
                        * utils.calculate_cultural_similarity(
                            self.team_id,
                            neighbor.agent.team_id,
                            pairwise_distances,
                        )
                    )
                    + (
                        utils.INCOME_SIMILARITY_WEIGHT
                        * utils.calculate_income_similarity(
                            self.income,
                            neighbor.agent.income,
                        )
                    )
                    for neighbor in neighborhood
                ]
            )
        else:
            similarity_ratio = 1

        return similarity_ratio

    def get_satisfaction_score(
        self,
        neighborhood: list,
        pairwise_distances: Mapping[tuple[int, int], float] | None = None,
        neighborhood_capacity: int = 0,
        density_preference_enabled: bool = False,
        density_preference_weight: float = 0.1,
        occupied_neighbor_count: int | None = None,
    ) -> float:
        """Return similarity adjusted by an optional preference for occupied neighbors."""
        similarity_ratio = self.get_similarity_ratio(neighborhood, pairwise_distances)
        if not density_preference_enabled:
            return similarity_ratio

        if neighborhood_capacity <= 0:
            occupancy_ratio = 0.0
        else:
            occupied_count = (
                len(neighborhood)
                if occupied_neighbor_count is None
                else occupied_neighbor_count
            )
            occupancy_ratio = occupied_count / neighborhood_capacity

        return (
            ((1 - density_preference_weight) * similarity_ratio)
            + (density_preference_weight * occupancy_ratio)
        )

    def can_afford(self, house_value: float) -> bool:
        """Return whether the agent can afford a given house value."""
        return house_value <= utils.calculate_affordability_limit(self.income)

    def is_unhappy(
        self,
        neighborhood: list,
        house_value: float,
        pairwise_multipliers: Mapping[tuple[int, int], float] | None = None,
    ) -> bool:
        """Return whether the agent is socially or economically unhappy."""
        return (
            self.get_similarity_ratio(neighborhood, pairwise_multipliers) < self.similarity_threshold
            or not self.can_afford(house_value)
        )

    def __repr__(self) -> str:
        return (
            f"Agent(team={self.team_id}, id={self.agent_id}, tau={self.similarity_threshold:.2f}, "
            f"income={self.income:.2f})"
        )
