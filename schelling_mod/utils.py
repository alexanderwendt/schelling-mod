"""Utility helpers for similarity and economic calculations."""

from collections.abc import Mapping

import numpy as np

HOUSING_COST_RATIO = 0.5
INCOME_YEARS = 20
DEFAULT_PROPERTY_VALUE = INCOME_YEARS * HOUSING_COST_RATIO
LOCATION_MULTIPLIER_MIN = 0.5
LOCATION_MULTIPLIER_MAX = 1.5
CULTURAL_SIMILARITY_WEIGHT = 1.0
INCOME_SIMILARITY_WEIGHT = 0.0


def sample_normal_value(
    mean_value: float,
    std_dev: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Sample a non-negative scalar from a normal distribution."""
    generator = rng or np.random.default_rng()
    return max(0.0, float(generator.normal(loc=mean_value, scale=std_dev)))


def calculate_similarity_to_neighbor(
    vector1,
    vector2,
    distance_multiplier: float = 1.0,
) -> float:
    """Calculate similarity as one minus scaled Euclidean distance."""
    similarity = 1 - (np.linalg.norm(np.array(vector1) - np.array(vector2)) * distance_multiplier)
    return max(0.0, similarity)


def get_pairwise_cultural_distance(
    team_id: int,
    neighbor_team_id: int,
    pairwise_distances: Mapping[tuple[int, int], float] | None,
) -> float:
    """Return the configured cultural distance for a team pair."""
    if pairwise_distances is None or team_id == neighbor_team_id:
        return 0.0

    direct_key = (team_id, neighbor_team_id)
    reverse_key = (neighbor_team_id, team_id)
    return float(pairwise_distances.get(direct_key, pairwise_distances.get(reverse_key, 0.0)))


def get_pairwise_cultural_multiplier(
    team_id: int,
    neighbor_team_id: int,
    pairwise_multipliers: Mapping[tuple[int, int], float] | None,
) -> float:
    """Return the configured cultural distance for backward-compatible callers."""
    return get_pairwise_cultural_distance(team_id, neighbor_team_id, pairwise_multipliers)


def calculate_cultural_similarity(
    team_id: int,
    neighbor_team_id: int,
    pairwise_distances: Mapping[tuple[int, int], float] | None = None,
) -> float:
    """Calculate cultural similarity directly from configured group distance."""
    cultural_distance = get_pairwise_cultural_distance(
        team_id,
        neighbor_team_id,
        pairwise_distances,
    )
    return max(0.0, 1 - cultural_distance)


def calculate_income_similarity(income1: float, income2: float) -> float:
    """Calculate normalized income similarity."""
    denominator = max(income1, income2, 1.0)
    similarity = 1 - (abs(income1 - income2) / denominator)
    return max(0.0, similarity)


def calculate_total_similarity(
    team_id: int,
    neighbor_team_id: int,
    income1: float,
    income2: float,
    pairwise_distances: Mapping[tuple[int, int], float] | None = None,
) -> float:
    """Return weighted cultural and income similarity."""
    cultural_similarity = calculate_cultural_similarity(
        team_id,
        neighbor_team_id,
        pairwise_distances,
    )
    income_similarity = calculate_income_similarity(income1, income2)
    return (
        (CULTURAL_SIMILARITY_WEIGHT * cultural_similarity)
        + (INCOME_SIMILARITY_WEIGHT * income_similarity)
    )


def calculate_affordability_limit(income: float) -> float:
    """Return the maximum affordable house value for an agent."""
    return DEFAULT_PROPERTY_VALUE * income


def calculate_property_value(
    neighborhood_values: list[float],
    location_multiplier: float,
) -> float:
    """Return the capitalized value of neighborhood income or location value."""
    if neighborhood_values:
        mean_value = float(np.mean(neighborhood_values))
    else:
        mean_value = location_multiplier

    return DEFAULT_PROPERTY_VALUE * mean_value


def calculate_location_multiplier(
    row: int,
    col: int,
    row_count: int,
    col_count: int,
) -> float:
    """Return the highest location multiplier at the map center."""
    center_row = (row_count - 1) / 2
    center_col = (col_count - 1) / 2
    max_distance = max(
        np.linalg.norm([0 - center_row, 0 - center_col]),
        np.linalg.norm([0 - center_row, (col_count - 1) - center_col]),
        np.linalg.norm([(row_count - 1) - center_row, 0 - center_col]),
        np.linalg.norm([(row_count - 1) - center_row, (col_count - 1) - center_col]),
        1.0,
    )
    distance_from_center = np.linalg.norm([row - center_row, col - center_col])
    normalized_position = 1 - min(distance_from_center / max_distance, 1.0)
    span = LOCATION_MULTIPLIER_MAX - LOCATION_MULTIPLIER_MIN
    return LOCATION_MULTIPLIER_MIN + (span * normalized_position)
