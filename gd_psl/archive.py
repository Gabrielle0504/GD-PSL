"""Objective-space archive construction for GD-PSL."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import NONDOMINATED_CHUNK_SIZE, NUMERICAL_EPSILON


EPS = NUMERICAL_EPSILON


def _front_on_unique_values(values: np.ndarray) -> np.ndarray:
    """Return a mask for the exact minimization front of unique rows."""
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


def as_2d(array: np.ndarray, columns: Optional[int] = None) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.size == 0:
        return np.empty((0, columns or 0), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {values.shape}")
    if columns is not None and values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns, got {values.shape[1]}")
    return values


def non_dominated_indices(objectives: np.ndarray) -> np.ndarray:
    """Return first-front indices for a minimization problem."""
    return nondominated_indices(as_2d(objectives))


def constraint_violation(constraints: Optional[np.ndarray], rows: int) -> np.ndarray:
    """Return summed positive PlatEMO constraint violations."""
    if constraints is None:
        return np.zeros(rows, dtype=float)
    values = np.asarray(constraints, dtype=float)
    if values.size == 0:
        return np.zeros(rows, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(values) != rows:
        raise ValueError("Constraint and solution row counts differ")
    values = np.where(np.isfinite(values), values, np.inf)
    return np.maximum(values, 0.0).sum(axis=1)


def standardize_base_result(
    decisions: np.ndarray,
    objectives: np.ndarray,
    constraints: Optional[np.ndarray] = None,
    feasibility_tolerance: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a finite, feasible, objective-unique nondominated archive."""
    x, f = as_2d(decisions), as_2d(objectives)
    if len(x) != len(f) or not len(x):
        raise ValueError("The base algorithm must return matching nonempty matrices")

    violation = constraint_violation(constraints, len(x))
    finite = np.isfinite(x).all(axis=1) & np.isfinite(f).all(axis=1)
    finite &= np.isfinite(violation)
    x, f, violation = x[finite], f[finite], violation[finite]
    if not len(x):
        raise ValueError("The base algorithm returned no finite solutions")

    feasible = violation <= feasibility_tolerance
    if feasible.any():
        x, f = x[feasible], f[feasible]
    else:
        best = violation.min()
        selected = violation <= best + feasibility_tolerance
        x, f = x[selected], f[selected]

    _, unique = np.unique(f, axis=0, return_index=True)
    unique.sort()
    x, f = x[unique], f[unique]
    front = non_dominated_indices(f)
    return x[front], f[front]


def objectives_to_preferences(
    objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize minimization objectives and map every row to the simplex."""
    values = as_2d(objectives)
    ideal, nadir = values.min(axis=0), values.max(axis=0)
    scale = np.where(nadir - ideal > EPS, nadir - ideal, 1.0)
    normalized = np.maximum((values - ideal) / scale, 0.0)
    totals = normalized.sum(axis=1, keepdims=True)
    preferences = np.divide(
        normalized,
        totals,
        out=np.full_like(normalized, 1.0 / values.shape[1]),
        where=totals > EPS,
    )
    return preferences, ideal, nadir


def merge_nondominated(
    base_x: np.ndarray,
    base_f: np.ndarray,
    model_x: np.ndarray,
    model_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge EA/model evaluations and retain an objective-unique first front."""
    completed_x, completed_f, sources, _ = merge_and_classify_candidates(
        base_x, base_f, model_x, model_f
    )
    return completed_x, completed_f, sources


def merge_and_classify_candidates(
    base_x: np.ndarray,
    base_f: np.ndarray,
    model_x: np.ndarray,
    model_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge once and return both the first front and every model status."""
    base_x, base_f = as_2d(base_x), as_2d(base_f)
    model_x = as_2d(model_x, base_x.shape[1])
    model_f = as_2d(model_f, base_f.shape[1])
    all_x = np.vstack((base_x, model_x))
    all_f = np.vstack((base_f, model_f))
    sources = np.r_[np.zeros(len(base_x), dtype=int), np.ones(len(model_x), dtype=int)]
    original_indices = np.arange(len(all_f))
    statuses = np.full(len(model_f), "model_dominated", dtype=object)
    finite = np.isfinite(all_x).all(axis=1) & np.isfinite(all_f).all(axis=1)
    statuses[~finite[len(base_f) :]] = "model_nonfinite"
    all_x, all_f, sources, original_indices = (
        all_x[finite],
        all_f[finite],
        sources[finite],
        original_indices[finite],
    )
    _, unique = np.unique(all_f, axis=0, return_index=True)
    unique.sort()
    duplicate = np.ones(len(all_f), dtype=bool)
    duplicate[unique] = False
    duplicate_model = original_indices[duplicate] - len(base_f)
    duplicate_model = duplicate_model[duplicate_model >= 0]
    statuses[duplicate_model] = "model_duplicate"
    all_x, all_f, sources, original_indices = (
        all_x[unique],
        all_f[unique],
        sources[unique],
        original_indices[unique],
    )
    front = non_dominated_indices(all_f)
    surviving_model = original_indices[front] - len(base_f)
    surviving_model = surviving_model[surviving_model >= 0]
    statuses[surviving_model] = "model_nondominated"
    return all_x[front], all_f[front], sources[front], statuses


def classify_model_candidates(
    base_x: np.ndarray,
    base_f: np.ndarray,
    model_x: np.ndarray,
    model_f: np.ndarray,
) -> np.ndarray:
    """Label model evaluations using finite, duplicate, and dominance checks."""
    return merge_and_classify_candidates(base_x, base_f, model_x, model_f)[3]
