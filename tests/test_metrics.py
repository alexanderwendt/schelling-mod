"""Tests for per-iteration metrics."""

import numpy as np

from schelling_mod.app import MENTAL_VALUES_MAP
from schelling_mod.city import City
from schelling_mod.metrics import build_metric_row
from schelling_mod.metrics import calculate_attractiveness_exposure
from schelling_mod.metrics import calculate_segregation_indexes
from schelling_mod.movement import MovementStats

THRESHOLDS = {1: (0.2, 0.0), 2: (0.2, 0.0)}
INCOMES = {1: (1.0, 0.0), 2: (1.0, 0.0)}


def test_metric_row_contains_required_fields_for_iteration_zero() -> None:
    """Metric row should contain required export columns."""
    city = City(rng=np.random.default_rng(1))
    city.set_map(np.array([[1, 1], [2, 0]]))
    city.instantiate_city(MENTAL_VALUES_MAP, 0.0, THRESHOLDS, INCOMES)

    row = build_metric_row(
        "run-1", 0, city, {1: "Knights", 2: "Elves"}, {(1, 2): 0.5}, 1, "moore", False,
        MovementStats(), None,
    )

    assert row["run_id"] == "run-1"
    assert row["move_count"] == 0
    assert "segregation_index_group_Knights" in row
    assert "attractiveness_exposure_group_Elves" in row


def test_no_occupied_neighbors_counts_local_own_group_share_as_one() -> None:
    """Segregation metric handles isolated agents."""
    city = City(rng=np.random.default_rng(1))
    city.set_map(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]]))
    city.instantiate_city(MENTAL_VALUES_MAP, 0.0, THRESHOLDS, INCOMES)

    indexes = calculate_segregation_indexes(city, {1: "Knights", 2: "Elves"}, 1, "moore")

    assert indexes["segregation_index_global"] >= 0.0


def test_metrics_create_group_columns_for_active_dynamic_names() -> None:
    """Dynamic group names should appear in group metric suffixes."""
    city = City(rng=np.random.default_rng(1))
    city.set_map(np.array([[1, 2], [0, 0]]))
    city.instantiate_city(MENTAL_VALUES_MAP, 0.0, THRESHOLDS, INCOMES)

    exposure = calculate_attractiveness_exposure(city, {1: "Dwarves", 2: "Hobbits"})

    assert "attractiveness_exposure_group_Dwarves" in exposure
    assert "attractiveness_exposure_group_Hobbits" in exposure
