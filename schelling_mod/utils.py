"""Utility helpers for similarity calculations."""

import numpy as np


def calculate_similarity_to_neighbor(vector1, vector2) -> float:
    """Calculate similarity as one minus Euclidean distance."""
    return 1 - np.linalg.norm(np.array(vector1) - np.array(vector2))
