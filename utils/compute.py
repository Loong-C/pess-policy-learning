import numpy as np


def apply_floor(probs, floor):
    """Project probabilities upward to component-wise lower bounds."""
    probs = np.asarray(probs, dtype=float)
    floor = np.asarray(floor, dtype=float)
    if np.any(floor < 0):
        raise ValueError("Probability floors must be non-negative.")
    if floor.sum() > 1 + 1e-12:
        raise ValueError("Probability floors sum to more than one.")

    floored = np.maximum(probs, floor)
    excess = floored.sum() - 1.0
    if excess <= 0:
        total = floored.sum()
        return floored / total if total > 0 else np.ones_like(floored) / len(floored)

    slack = floored - floor
    slack_sum = slack.sum()
    if slack_sum <= 1e-15:
        return floor / floor.sum()
    return floored - excess * slack / slack_sum
