"""Per-iteration metrics for Schelling runs."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np

from .city import City
from .feature import FeatureType
from .movement import MovementStats


def metric_suffix(name: str) -> str:
    """Return safe metric column suffix for group names."""
    suffix = re.sub(r"\W+", "_", str(name).strip())
    return suffix.strip("_") or "unknown"


def calculate_segregation_indexes(
    city: City,
    group_names: Mapping[int, str],
    n_neighbors: int,
    neighborhood_type: str,
) -> dict[str, float]:
    """Calculate global and per-group segregation indexes."""
    occupied = [
        feature
        for _, feature in np.ndenumerate(city.city)
        if feature.type == FeatureType.HOUSE and feature.agent is not None
    ]
    total_agents = len(occupied)
    if total_agents == 0:
        result = {"segregation_index_global": 0.0}
        for name in group_names.values():
            result[f"segregation_index_group_{metric_suffix(name)}"] = 0.0
        return result

    citywide_share = {
        group_id: sum(1 for feature in occupied if feature.agent.team_id == group_id) / total_agents
        for group_id in group_names
    }
    by_group: dict[int, list[float]] = {group_id: [] for group_id in group_names}
    all_deviations: list[float] = []

    for feature in occupied:
        row, col = feature.position
        group_id = feature.agent.team_id
        neighbors = city.get_neighbors(row, col, n_neighbors, neighborhood_type)
        if neighbors:
            same_group_count = sum(1 for neighbor in neighbors if neighbor.agent.team_id == group_id)
            local_share = same_group_count / len(neighbors)
        else:
            local_share = 1.0
        deviation = abs(local_share - citywide_share.get(group_id, 0.0))
        all_deviations.append(deviation)
        by_group.setdefault(group_id, []).append(deviation)

    result = {"segregation_index_global": float(np.mean(all_deviations))}
    for group_id, name in group_names.items():
        deviations = by_group.get(group_id, [])
        result[f"segregation_index_group_{metric_suffix(name)}"] = float(np.mean(deviations)) if deviations else 0.0
    return result


def calculate_attractiveness_exposure(
    city: City,
    group_names: Mapping[int, str],
) -> dict[str, float]:
    """Calculate mean location multiplier exposure globally and per group."""
    occupied = [
        feature
        for _, feature in np.ndenumerate(city.city)
        if feature.type == FeatureType.HOUSE and feature.agent is not None
    ]
    result = {
        "attractiveness_exposure_global": float(np.mean([f.location_multiplier for f in occupied]))
        if occupied else 0.0
    }
    for group_id, name in group_names.items():
        values = [f.location_multiplier for f in occupied if f.agent.team_id == group_id]
        result[f"attractiveness_exposure_group_{metric_suffix(name)}"] = float(np.mean(values)) if values else 0.0
    return result


def build_metric_row(
    run_id: str,
    iteration: int,
    city: City,
    group_names: Mapping[int, str],
    pairwise_distances: Mapping[tuple[int, int], float],
    n_neighbors: int,
    neighborhood_type: str,
    density_preference_enabled: bool,
    movement_stats: MovementStats,
    convergence_iteration: int | None,
) -> dict[str, object]:
    """Build one CSV-ready metric row."""
    occupied = [
        feature
        for _, feature in np.ndenumerate(city.city)
        if feature.type == FeatureType.HOUSE and feature.agent is not None
    ]
    total_agents = len(occupied)
    similarities: list[float] = []
    scores: list[float] = []
    satisfied_count = 0
    for feature in occupied:
        row, col = feature.position
        agent = feature.agent
        neighbors = city.get_neighbors(row, col, n_neighbors, neighborhood_type)
        similarity = agent.get_similarity_ratio(neighbors, pairwise_distances)
        score = agent.get_satisfaction_score(
            neighbors,
            pairwise_distances,
            city.get_neighborhood_capacity(row, col, n_neighbors, neighborhood_type),
            density_preference_enabled,
            0.1,
            len(neighbors),
        )
        similarities.append(similarity)
        scores.append(score)
        if score >= agent.similarity_threshold:
            satisfied_count += 1

    distances = movement_stats.displacement_distances
    row = {
        "run_id": run_id,
        "iteration": iteration,
        "satisfaction_percentage": (100.0 * satisfied_count / total_agents) if total_agents else 0.0,
        "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "mean_satisfaction_score": float(np.mean(scores)) if scores else 0.0,
        "move_count": movement_stats.move_count,
        "convergence_iteration": "" if convergence_iteration is None else convergence_iteration,
        "affordability_failures": movement_stats.affordability_failures,
        "mean_displacement_distance": float(np.mean(distances)) if distances else 0.0,
        "total_displacement_distance": int(sum(distances)),
    }
    row.update(calculate_segregation_indexes(city, group_names, n_neighbors, neighborhood_type))
    row.update(calculate_attractiveness_exposure(city, group_names))
    return row
