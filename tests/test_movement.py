"""Tests for movement helpers and candidate selection."""

import numpy as np

from schelling_mod.app import MENTAL_VALUES_MAP
from schelling_mod.app import Schelling
from schelling_mod.city import City
from schelling_mod.movement import should_warn_best_available

THRESHOLDS = {1: (0.2, 0.0), 2: (0.2, 0.0)}
INCOMES = {1: (1.0, 0.0), 2: (1.0, 0.0)}


def test_limited_distance_excludes_candidates_outside_radius() -> None:
    """Candidate enumeration should obey Manhattan movement radius."""
    city = City(rng=np.random.default_rng(1))
    city.set_map(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    city.instantiate_city(MENTAL_VALUES_MAP, 0.0, THRESHOLDS, INCOMES)

    candidates = city.get_empty_house_candidates(origin=(0, 0), movement_search_radius=1)

    assert set(candidates) == {(0, 1), (1, 0)}


def test_best_available_guard_flags_large_empty_pool() -> None:
    """Exhaustive movement has a documented guard threshold."""
    assert should_warn_best_available(2501) is True
    assert should_warn_best_available(2500) is False


def test_satisfied_agents_do_not_move_by_default() -> None:
    """Default satisfied mobility probability is zero."""
    schelling = Schelling(4, 0.75, 1, False, races=2, property_values_enabled=False, seed=1)
    moving_agent = next(feature.agent for _, feature in np.ndenumerate(schelling.city.city) if feature.agent)
    start = next(pos for pos, feature in np.ndenumerate(schelling.city.city) if feature.agent is moving_agent)

    schelling.run()

    end = next(pos for pos, feature in np.ndenumerate(schelling.city.city) if feature.agent is moving_agent)
    assert end == start
