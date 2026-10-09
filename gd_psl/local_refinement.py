"""Reference-free local correction for Pareto-set model decisions."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree


EPS = 1e-12


def _weighted_local_affine(
    target_preferences: np.ndarray,
    archive_preferences: np.ndarray,
    normalized_archive_decisions: np.ndarray,
    neighbor_indices: np.ndarray,
    ridge: float,
) -> np.ndarray:
    local_preferences = archive_preferences[neighbor_indices]
    local_decisions = normalized_archive_decisions[neighbor_indices]
    deltas = local_preferences[:, :, :-1] - target_preferences[:, None, :-1]
    distances = np.linalg.norm(
        local_preferences - target_preferences[:, None, :], axis=2
    )
    weights = 1.0 / np.maximum(distances, EPS)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    design = np.concatenate((np.ones((*deltas.shape[:2], 1)), deltas), axis=2)
    gram = np.einsum("nki,nkj,nk->nij", design, design, weights)
    rhs = np.einsum("nki,nkd,nk->nid", design, local_decisions, weights)
    regularizer = np.eye(design.shape[2]) * ridge
    regularizer[0, 0] = EPS
    coefficients = np.linalg.solve(gram + regularizer[None, :, :], rhs)
    return np.clip(coefficients[:, 0, :], 0.0, 1.0)


def interpolate_archive_decisions(
    target_preferences: np.ndarray,
    archive_decisions: np.ndarray,
    archive_preferences: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    neighbors: int = 32,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Generate decisions by preference-neighbor interpolation without an MLP."""
    started = time.perf_counter()
    targets = np.asarray(target_preferences, dtype=float)
    archive_x = np.asarray(archive_decisions, dtype=float)
    archive_p = np.asarray(archive_preferences, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if targets.ndim != 2 or archive_p.ndim != 2 or archive_x.ndim != 2:
        raise ValueError("Targets and archive arrays must be two-dimensional")
    if len(archive_x) != len(archive_p):
        raise ValueError("Archive decisions and preferences must have equal rows")
    if neighbors < targets.shape[1]:
        raise ValueError("neighbors must be at least the objective count")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    ranges = upper - lower
    if np.any(ranges <= 0):
        raise ValueError("Every decision variable must have a positive range")
    count = min(int(neighbors), len(archive_x))
    if count < targets.shape[1]:
        raise ValueError("The archive is too small for local interpolation")
    indices = cKDTree(archive_p).query(targets, k=count)[1]
    indices = np.asarray(indices, dtype=int)
    if indices.ndim == 1:
        indices = indices[:, None]
    normalized_archive = np.clip((archive_x - lower) / ranges, 0.0, 1.0)
    normalized = _weighted_local_affine(
        targets, archive_p, normalized_archive, indices, ridge
    )
    decisions = normalized * ranges + lower
    return decisions, {
        "strategy": "preference_neighbor_weighted_local_affine_fit",
        "neighbors": int(count),
        "ridge": float(ridge),
        "seconds": float(time.perf_counter() - started),
        "reference_pf_used": False,
        "additional_true_evaluations": 0,
    }


def refine_model_decisions(
    model_decisions: np.ndarray,
    target_preferences: np.ndarray,
    archive_decisions: np.ndarray,
    archive_preferences: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    candidate_neighbors: int = 64,
    branch_neighbors: int = 32,
    ridge: float = 1e-6,
    branch_ranks: np.ndarray | None = None,
    maximum_branches: int = 1,
) -> tuple[np.ndarray, dict]:
    """Correct model outputs with a local archive branch fit.

    The MLP output identifies a decision-space branch among nearby archive
    points. A weighted affine fit on that branch then maps the requested
    preference to a decision. This uses only the evaluated EA archive and does
    not consume an additional objective evaluation.
    """
    started = time.perf_counter()
    model_x = np.asarray(model_decisions, dtype=float)
    targets = np.asarray(target_preferences, dtype=float)
    archive_x = np.asarray(archive_decisions, dtype=float)
    archive_p = np.asarray(archive_preferences, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)

    if model_x.ndim != 2 or archive_x.ndim != 2:
        raise ValueError("Model and archive decisions must be two-dimensional")
    if targets.ndim != 2 or archive_p.ndim != 2:
        raise ValueError("Target and archive preferences must be two-dimensional")
    if len(model_x) != len(targets):
        raise ValueError("Model decisions and target preferences must have equal rows")
    if len(archive_x) != len(archive_p):
        raise ValueError("Archive decisions and preferences must have equal rows")
    if model_x.shape[1] != archive_x.shape[1]:
        raise ValueError("Model and archive decision dimensions do not match")
    if targets.shape[1] != archive_p.shape[1]:
        raise ValueError("Target and archive preference dimensions do not match")
    if candidate_neighbors < 2:
        raise ValueError("candidate_neighbors must be at least 2")
    if branch_neighbors < targets.shape[1]:
        raise ValueError("branch_neighbors must be at least the objective count")
    if branch_neighbors > candidate_neighbors:
        raise ValueError("branch_neighbors cannot exceed candidate_neighbors")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if maximum_branches < 1:
        raise ValueError("maximum_branches must be positive")
    if branch_ranks is not None and len(branch_ranks) != len(model_x):
        raise ValueError("branch_ranks must match the number of model candidates")
    ranges = upper - lower
    if np.any(ranges <= 0):
        raise ValueError("Every decision variable must have a positive range")

    diagnostics = {
        "enabled": True,
        "strategy": "mlp_branch_selection_plus_weighted_local_affine_fit",
        "candidate_neighbors": 0,
        "branch_neighbors": 0,
        "ridge": float(ridge),
        "refined_candidates": 0,
        "fallback_candidates": int(len(model_x)),
        "median_normalized_correction": 0.0,
        "p90_normalized_correction": 0.0,
        "seconds": 0.0,
        "reference_pf_used": False,
        "additional_true_evaluations": 0,
        "maximum_branches": int(maximum_branches),
    }
    if not len(model_x) or len(archive_x) < targets.shape[1]:
        diagnostics["seconds"] = time.perf_counter() - started
        return model_x.copy(), diagnostics

    normalized_model = np.clip((model_x - lower) / ranges, 0.0, 1.0)
    normalized_archive = np.clip((archive_x - lower) / ranges, 0.0, 1.0)
    candidate_count = min(int(candidate_neighbors), len(archive_x))
    branch_count = min(int(branch_neighbors), candidate_count)
    if branch_count < targets.shape[1]:
        diagnostics["seconds"] = time.perf_counter() - started
        return model_x.copy(), diagnostics

    tree = cKDTree(archive_p)
    _, candidate_indices = tree.query(targets, k=candidate_count)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.ndim == 1:
        candidate_indices = candidate_indices[:, None]

    candidate_x = normalized_archive[candidate_indices]
    model_distance = np.sum(
        np.square(candidate_x - normalized_model[:, None, :]), axis=2
    )
    anchor_position = np.argmin(model_distance, axis=1)
    if branch_ranks is not None and maximum_branches > 1:
        requested = np.asarray(branch_ranks, dtype=int) % int(maximum_branches)
        diverse_positions = np.empty((len(model_x), maximum_branches), dtype=int)
        diverse_positions[:, 0] = anchor_position
        min_distance = np.sum(
            np.square(
                candidate_x
                - candidate_x[np.arange(len(model_x)), anchor_position][:, None, :]
            ),
            axis=2,
        )
        for branch in range(1, maximum_branches):
            next_position = np.argmax(min_distance, axis=1)
            diverse_positions[:, branch] = next_position
            next_distance = np.sum(
                np.square(
                    candidate_x
                    - candidate_x[np.arange(len(model_x)), next_position][:, None, :]
                ),
                axis=2,
            )
            min_distance = np.minimum(min_distance, next_distance)
        anchor_position = diverse_positions[np.arange(len(model_x)), requested]
    anchors = candidate_x[np.arange(len(model_x)), anchor_position]
    branch_distance = np.sum(
        np.square(candidate_x - anchors[:, None, :]), axis=2
    )
    branch_positions = np.argpartition(
        branch_distance, kth=branch_count - 1, axis=1
    )[:, :branch_count]
    branch_indices = np.take_along_axis(
        candidate_indices, branch_positions, axis=1
    )

    normalized_refined = _weighted_local_affine(
        targets, archive_p, normalized_archive, branch_indices, ridge
    )
    finite = np.isfinite(normalized_refined).all(axis=1)
    normalized_refined[~finite] = normalized_model[~finite]
    refined = normalized_refined * ranges + lower

    corrections = np.linalg.norm(normalized_refined - normalized_model, axis=1)
    diagnostics.update(
        {
            "candidate_neighbors": int(candidate_count),
            "branch_neighbors": int(branch_count),
            "refined_candidates": int(np.count_nonzero(finite)),
            "fallback_candidates": int(np.count_nonzero(~finite)),
            "median_normalized_correction": float(np.median(corrections)),
            "p90_normalized_correction": float(np.quantile(corrections, 0.9)),
            "branch_selection": (
                "diverse_farthest_anchor_rotation"
                if branch_ranks is not None and maximum_branches > 1
                else "nearest_to_model_output"
            ),
            "seconds": float(time.perf_counter() - started),
        }
    )
    return refined, diagnostics
