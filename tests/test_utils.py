"""Unit tests for utility helpers."""

import pytest

from schelling_mod.utils import calculate_similarity_to_neighbor


def test_calculate_similarity_to_neighbor_returns_one_for_identical_vectors() -> None:
    """Identical vectors should have maximum similarity."""
    similarity = calculate_similarity_to_neighbor([0.0, 0.0], [0.0, 0.0])

    assert similarity == pytest.approx(1.0)


def test_calculate_similarity_to_neighbor_uses_euclidean_distance() -> None:
    """Similarity is one minus the Euclidean distance."""
    similarity = calculate_similarity_to_neighbor([0.0, 0.0], [0.2, 0.2])

    assert similarity == pytest.approx(1 - (0.08 ** 0.5))
