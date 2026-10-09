"""Low-cost, reference-free validation of evaluated model candidates."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


EPS = 1e-12


def _preferences_and_radii(
    objectives: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent each normalized objective vector as radius * preference."""
    values = np.asarray(objectives, dtype=float)
    scale = np.where(nadir - ideal > EPS, nadir - ideal, 1.0)
    normalized = np.maximum((values - ideal) / scale, 0.0)
    radii = normalized.sum(axis=1)
    preferences = np.divide(
        normalized,
        radii[:, None],
        out=np.full_like(normalized, 1.0 / values.shape[1]),
        where=radii[:, None] > EPS,
    )
    return preferences, radii


def _local_models(
    base_preferences: np.ndarray,
    base_radii: np.ndarray,
    query_preferences: np.ndarray,
    neighbor_indices: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit weighted local radius models and measure two-sided agreement."""
    predictors = base_preferences[neighbor_indices, :-1]
    query_predictors = query_preferences[:, None, :-1]
    deltas = predictors - query_predictors
    neighbor_radii = base_radii[neighbor_indices]
    distances = np.linalg.norm(
        base_preferences[neighbor_indices] - query_preferences[:, None, :], axis=2
    )
    weights = 1.0 / np.maximum(distances, EPS)

    def fit(mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        active = weights if mask is None else weights * mask
        active = active / np.maximum(active.sum(axis=1, keepdims=True), EPS)
        design = np.concatenate((np.ones((*deltas.shape[:2], 1)), deltas), axis=2)
        gram = np.einsum("qki,qkj,qk->qij", design, design, active)
        rhs = np.einsum("qki,qk,qk->qi", design, neighbor_radii, active)
        regularizer = np.eye(design.shape[2]) * ridge
        regularizer[0, 0] = EPS
        coefficients = np.linalg.solve(
            gram + regularizer[None, :, :], rhs[..., None]
        )[..., 0]
        fitted = np.einsum("qki,qi->qk", design, coefficients)
        prediction = coefficients[:, 0]
        relative_fit_error = np.sqrt(
            np.sum(active * np.square(neighbor_radii - fitted), axis=1)
        ) / np.maximum(np.abs(prediction), EPS)
        return prediction, relative_fit_error

    prediction, fit_error = fit()
    if deltas.shape[2] == 0:
        return prediction, fit_error, np.full(len(prediction), np.nan)

    # Smooth neighborhoods give compatible one-sided predictions. The two
    # sides disagree when a combined fit would bridge an unsupported gap.
    covariance = np.einsum("qki,qkj->qij", deltas, deltas)
    _, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, :, -1]
    projections = np.einsum("qki,qi->qk", deltas, principal)
    left = projections < 0.0
    right = ~left
    minimum_side = max(2, base_preferences.shape[1])
    supported = (left.sum(axis=1) >= minimum_side) & (
        right.sum(axis=1) >= minimum_side
    )
    left_prediction, _ = fit(left)
    right_prediction, _ = fit(right)
    disagreement = np.abs(left_prediction - right_prediction) / np.maximum(
        np.abs(prediction), EPS
    )
    disagreement[~supported] = np.nan
    return prediction, fit_error, disagreement


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, probability)) if len(finite) else np.inf


def _inside_empirical_hull(
    base_preferences: np.ndarray,
    query_preferences: np.ndarray,
) -> np.ndarray:
    """Identify interpolation queries on 2-D/3-D preference simplexes.

    The last preference coordinate is redundant because rows sum to one. For
    more than three objectives, retain the conservative two-sided local test
    to avoid the cost of constructing a high-dimensional hull.
    """
    base = np.asarray(base_preferences, dtype=float)
    queries = np.asarray(query_preferences, dtype=float)
    intrinsic_dimension = base.shape[1] - 1
    if intrinsic_dimension == 1:
        lower = float(np.min(base[:, 0]))
        upper = float(np.max(base[:, 0]))
        tolerance = max(EPS, 1e-10 * max(1.0, upper - lower))
        return (queries[:, 0] >= lower - tolerance) & (
            queries[:, 0] <= upper + tolerance
        )
    if intrinsic_dimension != 2 or len(base) < 3:
        return np.zeros(len(queries), dtype=bool)
    try:
        hull = ConvexHull(base[:, :2])
    except QhullError:
        return np.zeros(len(queries), dtype=bool)
    residuals = (
        queries[:, :2] @ hull.equations[:, :-1].T
        + hull.equations[:, -1]
    )
    return np.max(residuals, axis=1) <= 1e-10


def validate_empirical_candidates(
    base_objectives: np.ndarray,
    target_preferences: np.ndarray,
    candidate_decisions: np.ndarray,
    candidate_objectives: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    normal_spacing: float,
    direction_match_factor: float,
    local_neighbors: int,
    error_quantile: float,
    minimum_neighbors: int,
    ridge: float,
    calibration_samples: int,
) -> dict:
    """Validate candidates against a locally calibrated empirical PF tube.

    Thresholds are obtained by leave-one-out prediction on the EA archive.
    No analytical or sampled reference Pareto front is accepted here.
    """
    base_f = np.asarray(base_objectives, dtype=float)
    targets = np.asarray(target_preferences, dtype=float)
    model_x = np.asarray(candidate_decisions, dtype=float)
    model_f = np.asarray(candidate_objectives, dtype=float)
    count = len(model_f)
    if len(targets) != count or len(model_x) != count:
        raise ValueError("Model targets, decisions, and objectives must have equal rows")
    if not 0.5 < error_quantile < 1.0:
        raise ValueError("error_quantile must be between 0.5 and 1")

    actual = np.full_like(targets, np.nan, dtype=float)
    direction_error = np.full(count, np.inf, dtype=float)
    candidate_radius = np.full(count, np.inf, dtype=float)
    predicted_radius = np.full(count, np.nan, dtype=float)
    relative_front_error = np.full(count, np.inf, dtype=float)
    local_fit_error = np.full(count, np.inf, dtype=float)
    branch_disagreement = np.full(count, np.nan, dtype=float)
    local_support = np.zeros(count, dtype=bool)
    target_base_distance = np.full(count, np.inf, dtype=float)
    coverage_improvement = np.full(count, -np.inf, dtype=float)
    effective_direction_limit = np.full(count, np.inf, dtype=float)
    inside_empirical_hull = np.zeros(count, dtype=bool)
    statuses = np.full(count, "model_nonfinite", dtype=object)
    finite = np.isfinite(model_x).all(axis=1) & np.isfinite(model_f).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    direction_limit = (
        float(direction_match_factor * normal_spacing)
        if np.isfinite(normal_spacing)
        else np.inf
    )

    front_error_limit = np.inf
    fit_error_limit = np.inf
    branch_disagreement_limit = np.inf
    neighbor_count = min(max(1, int(local_neighbors)), len(base_f))
    enough_archive = len(base_f) >= max(int(minimum_neighbors), 2)

    if len(finite_indices) and enough_archive:
        base_preferences, base_radii = _preferences_and_radii(base_f, ideal, nadir)
        finite_preferences, finite_radii = _preferences_and_radii(
            model_f[finite_indices], ideal, nadir
        )
        inside_empirical_hull[finite_indices] = _inside_empirical_hull(
            base_preferences, finite_preferences
        )
        actual[finite_indices] = finite_preferences
        candidate_radius[finite_indices] = finite_radii
        direction_error[finite_indices] = np.linalg.norm(
            finite_preferences - targets[finite_indices], axis=1
        )

        calibration_count = min(len(base_f), int(calibration_samples))
        calibration_indices = np.linspace(
            0, len(base_f) - 1, calibration_count, dtype=int
        )
        tree = cKDTree(base_preferences)
        nearest_target_distance = tree.query(targets[finite_indices], k=1)[0]
        target_base_distance[finite_indices] = nearest_target_distance
        coverage_improvement[finite_indices] = (
            nearest_target_distance - direction_error[finite_indices]
        )
        effective_direction_limit[finite_indices] = np.maximum(
            direction_limit, nearest_target_distance
        )
        calibration_k = min(neighbor_count + 1, len(base_f))
        _, calibration_neighbors = tree.query(
            base_preferences[calibration_indices], k=calibration_k
        )
        calibration_neighbors = np.atleast_2d(calibration_neighbors)
        if calibration_k > 1:
            calibration_neighbors = calibration_neighbors[:, 1:]
        calibration_prediction, calibration_fit, calibration_branch = _local_models(
            base_preferences,
            base_radii,
            base_preferences[calibration_indices],
            calibration_neighbors,
            ridge,
        )
        calibration_excess = np.maximum(
            0.0,
            (base_radii[calibration_indices] - calibration_prediction)
            / np.maximum(np.abs(calibration_prediction), EPS),
        )
        front_error_limit = max(_quantile(calibration_excess, error_quantile), EPS)
        fit_error_limit = max(_quantile(calibration_fit, error_quantile), EPS)
        # Each one-sided extrapolation carries local fit uncertainty. Adding
        # both sides' calibrated uncertainty avoids rejecting smooth points
        # merely because they fall between archive samples.
        branch_disagreement_limit = max(
            _quantile(calibration_branch, error_quantile)
            + 2.0 * fit_error_limit,
            EPS,
        )

        _, candidate_neighbors = tree.query(finite_preferences, k=neighbor_count)
        candidate_neighbors = np.asarray(candidate_neighbors, dtype=int)
        if candidate_neighbors.ndim == 1:
            candidate_neighbors = candidate_neighbors[:, None]
        prediction, fit_error, disagreement = _local_models(
            base_preferences,
            base_radii,
            finite_preferences,
            candidate_neighbors,
            ridge,
        )
        excess = (finite_radii - prediction) / np.maximum(np.abs(prediction), EPS)
        two_sided_support = np.isfinite(disagreement) & (
            disagreement <= branch_disagreement_limit + EPS
        )
        interpolation_support = (
            ~np.isfinite(disagreement)
            & inside_empirical_hull[finite_indices]
        )
        supported = (
            np.isfinite(prediction)
            & (prediction > EPS)
            & np.isfinite(fit_error)
            & (fit_error <= fit_error_limit + EPS)
            & (two_sided_support | interpolation_support)
        )
        predicted_radius[finite_indices] = prediction
        relative_front_error[finite_indices] = excess
        local_fit_error[finite_indices] = fit_error
        branch_disagreement[finite_indices] = disagreement
        local_support[finite_indices] = supported

        # A candidate is useful when it either hits an already small target
        # neighborhood or moves the empirical archive closer to a genuine
        # hole.  The latter makes tolerance scale with the hole radius instead
        # of forcing every candidate into one global median-spacing tube.
        direction_ok = (
            (direction_error[finite_indices] <= direction_limit + EPS)
            | (coverage_improvement[finite_indices] > EPS)
        )
        front_ok = excess <= front_error_limit + EPS
        statuses[finite_indices] = "model_candidate_pending"
        statuses[finite_indices[~direction_ok]] = "model_direction_mismatch"
        statuses[finite_indices[direction_ok & ~supported]] = (
            "model_unsupported_topology"
        )
        statuses[finite_indices[direction_ok & supported & ~front_ok]] = (
            "model_off_empirical_front"
        )
    elif len(finite_indices):
        statuses[finite_indices] = "model_unsupported_topology"

    eligible = statuses == "model_candidate_pending"
    return {
        "eligible": eligible,
        "statuses": statuses,
        "actual_preferences": actual,
        "direction_error": direction_error,
        "direction_limit": direction_limit,
        "effective_direction_limit": effective_direction_limit,
        "target_base_distance": target_base_distance,
        "coverage_improvement": coverage_improvement,
        "inside_empirical_hull": inside_empirical_hull,
        "candidate_radius": candidate_radius,
        "predicted_radius": predicted_radius,
        "relative_front_error": relative_front_error,
        "front_error_limit": front_error_limit,
        "local_fit_error": local_fit_error,
        "local_fit_error_limit": fit_error_limit,
        "branch_disagreement": branch_disagreement,
        "branch_disagreement_limit": branch_disagreement_limit,
        "local_support": local_support,
        "local_radius_median": predicted_radius,
        "local_radius_limit": predicted_radius * (1.0 + front_error_limit),
        "local_neighbors": neighbor_count,
        "calibration_samples": min(len(base_f), int(calibration_samples)),
        "reference_pf_used": False,
    }
