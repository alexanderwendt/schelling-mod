"""Unit tests for city generation and neighborhood logic."""

import numpy as np
import pytest

from schelling_mod.city import City


AGENT_VALUES = {
    1: np.array([0.0, 0.0]),
    2: np.array([0.2, 0.2]),
}


def test_generate_map_creates_square_grid() -> None:
    """Generated maps should be square grids truncated to a perfect square."""
    city = City()

    city.generate_map(
        size=10,
        empty_ratio=0.2,
        races=[0, 1, 2],
        teams_distribution={1: 0.5, 2: 0.5},
    )

    assert city.raw_map.shape == (3, 3)


def test_instantiate_city_populates_agents_and_houses() -> None:
    """Instantiation should convert the raw map into feature objects."""
    city = City()
    city.set_map(np.array([[1, 0], [2, 1]]))

    city.instantiate_city(AGENT_VALUES, agent_values_std=0.0, similarity_threshold=0.4)

    assert city.city.shape == (2, 2)
    assert city.city[0, 0].agent is not None
    assert city.city[0, 1].agent is None
    assert city.city[1, 0].agent.team_id == 2


def test_get_neighbors_returns_only_occupied_adjacent_houses() -> None:
    """Neighbor lookup should ignore empty houses and the origin cell."""
    city = City()
    city.set_map(np.array([[1, 0], [2, 1]]))
    city.instantiate_city(AGENT_VALUES, agent_values_std=0.0, similarity_threshold=0.4)

    neighbors = city.get_neighbors(0, 0, 1)

    assert len(neighbors) == 2
    assert {neighbor.agent.team_id for neighbor in neighbors} == {1, 2}


def test_get_random_empty_house_position_returns_empty_cell() -> None:
    """Random empty-house selection should pick one of the empty cells."""
    city = City()
    city.set_map(np.array([[1, 0], [0, 2]]))
    city.instantiate_city(AGENT_VALUES, agent_values_std=0.0, similarity_threshold=0.4)

    empty_position = city.get_random_empty_house_position()

    assert empty_position in ([0, 1], [1, 0])
    assert city.city[empty_position[0], empty_position[1]].agent is None


def test_get_mean_similarity_ratio_averages_occupied_houses() -> None:
    """Mean similarity should be computed over occupied houses only."""
    city = City()
    city.set_map(np.array([[1, 1], [0, 0]]))
    city.instantiate_city(AGENT_VALUES, agent_values_std=0.0, similarity_threshold=0.4)

    assert city.get_mean_similarity_ratio(1) == pytest.approx(1.0)
