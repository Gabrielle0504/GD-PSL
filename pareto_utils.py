"""Exact Pareto-front utilities optimized for two and three objectives."""

from __future__ import annotations

import numpy as np

from experiment_config import NONDOMINATED_CHUNK_SIZE


def _front_on_unique_values(values: np.ndarray) -> np.ndarray:
    objective_count = values.shape[1]
    order = np.lexsort(
        tuple(values[:, column] for column in range(objective_count - 1, -1, -1))
    )
    sorted_values = values[order]
    keep_sorted = np.zeros(len(values), dtype=bool)

    if objective_count == 2:
        best_second = np.inf
        for index, second in enumerate(sorted_values[:, 1]):
            if second < best_second:
                keep_sorted[index] = True
                best_second = second
    elif objective_count == 3:
        second_coordinates = np.unique(sorted_values[:, 1])
        ranks = np.searchsorted(second_coordinates, sorted_values[:, 1]) + 1
        tree = np.full(len(second_coordinates) + 1, np.inf)
        for index, (rank, third) in enumerate(zip(ranks, sorted_values[:, 2])):
            prefix_minimum = np.inf
            cursor = int(rank)
            while cursor > 0:
                prefix_minimum = min(prefix_minimum, tree[cursor])
                cursor -= cursor & -cursor
            if prefix_minimum > third:
                keep_sorted[index] = True
            cursor = int(rank)
            while cursor < len(tree):
                tree[cursor] = min(tree[cursor], third)
                cursor += cursor & -cursor
    else:
        for start in range(0, len(values), NONDOMINATED_CHUNK_SIZE):
            rows = np.arange(start, min(start + NONDOMINATED_CHUNK_SIZE, len(values)))
            dominates = np.all(values[None, :, :] <= values[rows, None, :], axis=2)
            dominates &= np.any(values[None, :, :] < values[rows, None, :], axis=2)
            keep_sorted[rows] = ~dominates.any(axis=1)
        return keep_sorted

    keep = np.zeros(len(values), dtype=bool)
    keep[order] = keep_sorted
    return keep


def nondominated_indices(objectives: np.ndarray) -> np.ndarray:
    """Return original-order indices on the exact minimization first front."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {values.shape}")
    if not len(values):
        return np.empty(0, dtype=int)

    finite_indices = np.flatnonzero(np.isfinite(values).all(axis=1))
    if not len(finite_indices):
        return np.empty(0, dtype=int)
    unique_values, inverse = np.unique(
        values[finite_indices], axis=0, return_inverse=True
    )
    keep_unique = _front_on_unique_values(unique_values)
    keep = np.zeros(len(values), dtype=bool)
    keep[finite_indices] = keep_unique[inverse]
    return np.flatnonzero(keep)
