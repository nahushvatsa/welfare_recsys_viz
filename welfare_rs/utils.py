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
