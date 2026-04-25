"""Unit tests for the agent model."""

import numpy as np
import pytest

from schelling_mod.agent import Agent
from schelling_mod.feature import Feature, FeatureType


def make_neighbor(team_id: int, values: list[float]) -> Feature:
    """Build a neighboring occupied house."""
    neighbor = Agent(
        agent_id=10 + team_id,
        team_id=team_id,
        mental_values_mean=np.array(values),
        mental_values_std_dev=0.0,
        similarity_threshold=0.5,
    )
    return Feature(FeatureType.HOUSE, [0, 0], neighbor)


def test_agent_similarity_ratio_defaults_to_one_without_neighbors() -> None:
    """Agents without neighbors should be fully satisfied."""
    agent = Agent(
        agent_id=1,
        team_id=1,
        mental_values_mean=np.array([0.0, 0.0]),
        mental_values_std_dev=0.0,
        similarity_threshold=0.5,
    )

    assert agent.get_similarity_ratio([]) == 1


def test_agent_is_unhappy_when_similarity_below_threshold() -> None:
    """Agents should become unhappy below the configured threshold."""
    agent = Agent(
        agent_id=1,
        team_id=1,
        mental_values_mean=np.array([0.0, 0.0]),
        mental_values_std_dev=0.0,
        similarity_threshold=0.9,
    )
    neighborhood = [make_neighbor(team_id=2, values=[0.2, 0.2])]

    assert bool(agent.is_unhappy(neighborhood))


def test_agent_similarity_ratio_matches_neighbor_average() -> None:
    """The similarity ratio should average all neighbor scores."""
    agent = Agent(
        agent_id=1,
        team_id=1,
        mental_values_mean=np.array([0.0, 0.0]),
        mental_values_std_dev=0.0,
        similarity_threshold=0.1,
    )
    neighborhood = [
        make_neighbor(team_id=1, values=[0.0, 0.0]),
        make_neighbor(team_id=2, values=[0.2, 0.2]),
    ]

    assert agent.get_similarity_ratio(neighborhood) == pytest.approx(0.8585786438)
