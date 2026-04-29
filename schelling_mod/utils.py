"""Utility helpers for similarity and economic calculations."""

from collections.abc import Mapping

import numpy as np

DEFAULT_PROPERTY_VALUE = 1.0
HOUSING_COST_RATIO = 0.6
INCOME_YEARS = 20
LOCATION_MULTIPLIER_MIN = 0.5
LOCATION_MULTIPLIER_MAX = 1.5
PROPERTY_VALUE_FACTOR = 0.5


def sample_normal_value(mean_value: float, std_dev: float) -> float:
    """Sample a non-negative scalar from a normal distribution."""
    return max(0.0, float(np.random.normal(loc=mean_value, scale=std_dev)))


def calculate_similarity_to_neighbor(
    vector1,
    vector2,
    distance_multiplier: float = 1.0,
) -> float:
    """Calculate similarity as one minus scaled Euclidean distance."""
    similarity = 1 - (np.linalg.norm(np.array(vector1) - np.array(vector2)) * distance_multiplier)
    return max(0.0, similarity)


def get_pairwise_cultural_multiplier(
    team_id: int,
    neighbor_team_id: int,
    pairwise_multipliers: Mapping[tuple[int, int], float] | None,
) -> float:
    """Return the configured cultural multiplier for a team pair."""
    if pairwise_multipliers is None or team_id == neighbor_team_id:
        return 1.0

    direct_key = (team_id, neighbor_team_id)
    reverse_key = (neighbor_team_id, team_id)
    return float(pairwise_multipliers.get(direct_key, pairwise_multipliers.get(reverse_key, 1.0)))


def calculate_income_similarity(income1: float, income2: float) -> float:
    """Calculate normalized income similarity."""
    denominator = max(income1, income2, 1.0)
    similarity = 1 - (abs(income1 - income2) / denominator)
    return max(0.0, similarity)


def calculate_total_similarity(
    vector1,
    vector2,
    income1: float,
    income2: float,
    distance_multiplier: float = 1.0,
) -> float:
    """Average cultural and income similarity."""
    cultural_similarity = calculate_similarity_to_neighbor(
        vector1,
        vector2,
        distance_multiplier=distance_multiplier,
    )
    income_similarity = calculate_income_similarity(income1, income2)
    return float(np.average([cultural_similarity, income_similarity]))


def calculate_affordability_limit(income: float) -> float:
    """Return the maximum affordable house value for an agent."""
    return HOUSING_COST_RATIO * INCOME_YEARS * income


def calculate_property_value(
    neighboring_incomes: list[float],
    location_multiplier: float,
) -> float:
    """Return a house value based on neighborhood income and location."""
    if neighboring_incomes:
        base_value = PROPERTY_VALUE_FACTOR * INCOME_YEARS * float(np.mean(neighboring_incomes))
    else:
        base_value = DEFAULT_PROPERTY_VALUE

    return base_value * location_multiplier


def calculate_location_multiplier(
    row: int,
    col: int,
    row_count: int,
    col_count: int,
) -> float:
    """Return a deterministic multiplier in the configured location range."""
    max_row = max(row_count - 1, 1)
    max_col = max(col_count - 1, 1)
    row_factor = row / max_row
    col_factor = col / max_col
    normalized_position = (row_factor + col_factor) / 2
    span = LOCATION_MULTIPLIER_MAX - LOCATION_MULTIPLIER_MIN
    return LOCATION_MULTIPLIER_MIN + (span * normalized_position)
