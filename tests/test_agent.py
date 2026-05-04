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
        income=1.0,
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
        income=1.0,
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
        income=1.0,
    )
    neighborhood = [make_neighbor(team_id=2, values=[0.2, 0.2])]

    assert bool(agent.is_unhappy(neighborhood, house_value=1.0, pairwise_multipliers={(1, 2): 0.2}))


def test_agent_similarity_ratio_matches_neighbor_average() -> None:
    """The similarity ratio should average all neighbor scores."""
    agent = Agent(
        agent_id=1,
        team_id=1,
        mental_values_mean=np.array([0.0, 0.0]),
        mental_values_std_dev=0.0,
        similarity_threshold=0.1,
        income=1.0,
    )
    neighborhood = [
        make_neighbor(team_id=1, values=[0.0, 0.0]),
        make_neighbor(team_id=2, values=[0.2, 0.2]),
    ]

    assert agent.get_similarity_ratio(neighborhood, {(1, 2): 0.2}) == pytest.approx(0.9)


def test_agent_is_unhappy_when_house_is_not_affordable() -> None:
    """Economic dissatisfaction should also trigger unhappiness."""
    agent = Agent(
        agent_id=1,
        team_id=1,
        mental_values_mean=np.array([0.0, 0.0]),
        mental_values_std_dev=0.0,
        similarity_threshold=0.1,
        income=1.0,
    )

    assert bool(agent.is_unhappy([], house_value=15.0))


def test_agent_similarity_uses_pairwise_cultural_distance() -> None:
    """Cultural distances should reduce similarity for configured pairs."""
    agent = Agent(
        agent_id=1,
        team_id=1,
        mental_values_mean=np.array([0.0, 0.0]),
        mental_values_std_dev=0.0,
        similarity_threshold=0.1,
        income=1.0,
    )
    neighborhood = [make_neighbor(team_id=2, values=[0.2, 0.2])]

    similarity = agent.get_similarity_ratio(neighborhood, {(1, 2): 0.6})

    assert similarity == pytest.approx(0.4)
