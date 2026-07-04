"""Movement mode helpers and stats."""

from __future__ import annotations

from dataclasses import dataclass, field

BEST_AVAILABLE_EMPTY_CELL_WARNING_THRESHOLD = 2500
MOVEMENT_MODES = (
    "random_empty",
    "first_acceptable",
    "best_sampled",
    "best_available",
    "limited_distance",
)


@dataclass
class MovementStats:
    """Counters collected during one simulation iteration."""

    move_count: int = 0
    affordability_failures: int = 0
    displacement_distances: list[int] = field(default_factory=list)


def should_warn_best_available(valid_empty_count: int) -> bool:
    """Return whether exhaustive search should be blocked or warned."""
    return valid_empty_count > BEST_AVAILABLE_EMPTY_CELL_WARNING_THRESHOLD
