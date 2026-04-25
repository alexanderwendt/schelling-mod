"""Feature model for city cells."""

from enum import Enum


class FeatureType(Enum):
    """Supported feature types in the city grid."""

    HOUSE = "h"
    BARRIER = "b"


class Feature:
    """Represents a city feature such as a house or barrier."""

    def __init__(self, feature_type: FeatureType, position: list, agent=None):
        self.type = feature_type
        self.agent = agent
        self.position = position

    def is_empty(self) -> bool:
        """Return whether the feature has no agent."""
        return self.agent is None

    def __repr__(self) -> str:
        if self.agent is not None:
            return f"({self.type.value}, pos={self.position}, a={self.agent})"

        return f"({self.type.value}, pos={self.position}, aid=None)"
