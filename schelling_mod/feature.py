"""Feature model for city cells."""

from enum import Enum


class FeatureType(Enum):
    """Supported feature types in the city grid."""

    HOUSE = "h"
    BARRIER = "b"


class Feature:
    """Represents a city feature such as a house or barrier."""

    def __init__(
        self,
        feature_type: FeatureType,
        position: list[int],
        agent=None,
        location_multiplier: float = 1.0,
        property_value: float = 1.0,
    ):
        self.type = feature_type
        self.agent = agent
        self.position = position
        self.location_multiplier = location_multiplier
        self.property_value = property_value

    def is_empty(self) -> bool:
        """Return whether the feature has no agent."""
        return self.agent is None

    def __repr__(self) -> str:
        if self.agent is not None:
            return (
                f"({self.type.value}, pos={self.position}, value={self.property_value:.2f}, "
                f"loc={self.location_multiplier:.2f}, a={self.agent})"
            )

        return (
            f"({self.type.value}, pos={self.position}, value={self.property_value:.2f}, "
            f"loc={self.location_multiplier:.2f}, aid=None)"
        )
