"""Agent model for the Schelling simulation."""

import logging
from collections.abc import Mapping

import numpy as np

from . import utils

log = logging.getLogger(__name__)


def generate_values_from_distribution(mean_value, std_dev) -> np.ndarray:
    """Generate a two-dimensional value vector from a normal distribution."""
    return np.random.normal(loc=mean_value, scale=std_dev, size=(2,))


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
        )
        self.similarity_threshold = similarity_threshold
        self.income = income

    def get_similarity_ratio(
        self,
        neighborhood: list,
        pairwise_multipliers: Mapping[tuple[int, int], float] | None = None,
    ) -> float:
        """Return the average similarity to neighboring agents."""
        if neighborhood:
            similarity_ratio = np.average(
                [
                    utils.calculate_total_similarity(
                        self.mental_values,
                        neighbor.agent.mental_values,
                        self.income,
                        neighbor.agent.income,
                        distance_multiplier=utils.get_pairwise_cultural_multiplier(
                            self.team_id,
                            neighbor.agent.team_id,
                            pairwise_multipliers,
                        ),
                    )
                    for neighbor in neighborhood
                ]
            )
        else:
            similarity_ratio = 1

        return similarity_ratio

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
        rounded_values = [str(round(num, 1)) for num in self.mental_values]
        return (
            f"Agent(team={self.team_id}, id={self.agent_id}, tau={self.similarity_threshold:.2f}, "
            f"income={self.income:.2f}, val={rounded_values})"
        )
