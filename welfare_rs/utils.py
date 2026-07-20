"""Small numeric helpers shared across ABM modules."""

from __future__ import annotations

import math

import numpy as np


def softmax(x):
    """Return numerically-stable softmax probabilities for a list of utilities."""
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def clamp(value, lo=0.0, hi=1.0):
    """Clamp a value to a closed interval."""
    return max(lo, min(hi, value))


def sigmoid(x):
    """Logistic helper for intention-style scores."""
    return 1.0 / (1.0 + math.exp(-x))


_EARTH_RADIUS_KM = 6371.0088


def haversine_km(a, b):
    """Great-circle ("as the crow flies") distance in km between two (lat, lon) points.

    Used by the recommender systems for their proximity heuristic, which the
    paper models as straight-line distance — deliberately cruder than the
    network routing used for the realised trip cost.
    """
    lat1, lon1 = a
    lat2, lon2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))
