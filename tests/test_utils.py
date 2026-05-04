"""Unit tests for utility helpers."""

import pytest

from schelling_mod.utils import calculate_affordability_limit
from schelling_mod.utils import calculate_cultural_similarity
from schelling_mod.utils import calculate_income_similarity
from schelling_mod.utils import calculate_location_multiplier
from schelling_mod.utils import calculate_property_value
from schelling_mod.utils import calculate_similarity_to_neighbor
from schelling_mod.utils import calculate_total_similarity
from schelling_mod.utils import get_pairwise_cultural_multiplier


def test_calculate_similarity_to_neighbor_returns_one_for_identical_vectors() -> None:
    """Identical vectors should have maximum similarity."""
    similarity = calculate_similarity_to_neighbor([0.0, 0.0], [0.0, 0.0])

    assert similarity == pytest.approx(1.0)


def test_calculate_similarity_to_neighbor_uses_euclidean_distance() -> None:
    """Similarity is one minus the Euclidean distance."""
    similarity = calculate_similarity_to_neighbor([0.0, 0.0], [0.2, 0.2])

    assert similarity == pytest.approx(1 - (0.08 ** 0.5))


def test_get_pairwise_cultural_multiplier_is_symmetric() -> None:
    """Pairwise cultural distances should work in both lookup directions."""
    multipliers = {(1, 2): 1.2}

    assert get_pairwise_cultural_multiplier(1, 2, multipliers) == pytest.approx(1.2)
    assert get_pairwise_cultural_multiplier(2, 1, multipliers) == pytest.approx(1.2)


def test_calculate_cultural_similarity_uses_pairwise_group_distance() -> None:
    """Cultural similarity should come directly from configured group distance."""
    distances = {(1, 2): 0.25}

    assert calculate_cultural_similarity(1, 1, distances) == pytest.approx(1.0)
    assert calculate_cultural_similarity(1, 2, distances) == pytest.approx(0.75)


def test_calculate_income_similarity_reflects_relative_difference() -> None:
    """Income similarity should decrease as the gap grows."""
    assert calculate_income_similarity(1.0, 1.0) == pytest.approx(1.0)
    assert calculate_income_similarity(1.0, 0.5) == pytest.approx(0.5)


def test_calculate_total_similarity_combines_culture_and_income() -> None:
    """Total similarity should currently use only cultural similarity."""
    similarity = calculate_total_similarity(1, 2, 1.0, 0.5, {(1, 2): 0.25})

    assert similarity == pytest.approx(0.75)


def test_calculate_property_value_uses_neighbor_income_and_location() -> None:
    """Property values should use the configured formula."""
    property_value = calculate_property_value([1.0, 3.0], location_multiplier=1.5)

    assert property_value == pytest.approx(30.0)


def test_calculate_property_value_uses_default_without_neighbors() -> None:
    """Empty neighborhoods should fall back to the default value."""
    property_value = calculate_property_value([], location_multiplier=0.5)

    assert property_value == pytest.approx(0.5)


def test_calculate_location_multiplier_stays_in_expected_range() -> None:
    """Location multipliers should stay between 0.5 and 1.5."""
    assert calculate_location_multiplier(0, 0, 3, 3) == pytest.approx(0.5)
    assert calculate_location_multiplier(2, 2, 3, 3) == pytest.approx(1.5)


def test_calculate_affordability_limit_uses_income_rule() -> None:
    """Affordability should use 60 percent of 20 years of income."""
    assert calculate_affordability_limit(2.0) == pytest.approx(24.0)
