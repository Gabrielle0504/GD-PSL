"""Preference-space hole detection and auditable reason records."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .archive import EPS, as_2d


REASON_TEXT = {
    "below_gap_threshold": "geometric distance is below the hole threshold",
    "model_sample_budget_exhausted": "model evaluation budget is exhausted",
    "candidate_pool_exhausted": "the finite detection pool is exhausted",
    "model_nondominated": "the evaluated model candidate survives in the merged archive",
    "model_dominated": "the evaluated model candidate is dominated",
    "model_duplicate": "the evaluated model candidate duplicates an objective vector",
    "model_nonfinite": "the model candidate or objective vector is non-finite",
    "model_direction_mismatch": "the evaluated candidate missed its requested preference region",
    "model_unsupported_topology": "the local EA archive does not support one smooth front branch at this preference",
    "model_off_empirical_front": "the evaluated candidate lies behind the calibrated one-sided empirical-front envelope",
    "model_candidate_pending": "the candidate passed empirical diagnostics and awaits dominance filtering",
    "model_query_pending": "the selected query has not been classified",
}


def generate_simplex_candidates(
    n_objectives: int,
    pool_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a deterministic grid in 2-D or Dirichlet samples otherwise."""
    if n_objectives < 2 or pool_size < n_objectives:
        raise ValueError("The candidate pool must cover at least two objectives")
    if n_objectives == 2:
        first = np.linspace(0.0, 1.0, pool_size)
        return np.column_stack((first, 1.0 - first))
    fixed = np.vstack((np.eye(n_objectives), np.full((1, n_objectives), 1 / n_objectives)))
    random_count = max(0, pool_size - len(fixed))
    return np.vstack((fixed, rng.dirichlet(np.ones(n_objectives), size=random_count)))


def preference_spacing(preferences: np.ndarray) -> float:
    """Return the median positive nearest-neighbor distance."""
    values = as_2d(preferences)
    if len(values) < 2:
        return np.inf
    distances = cKDTree(values).query(values, k=2)[0][:, 1]
    positive = distances[distances > EPS]
    return float(np.median(positive)) if len(positive) else np.inf


def _support_scores(
    observed: np.ndarray,
    candidates: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure local gap size and whether archive points surround each query."""
    tree = cKDTree(observed)
    count = min(max(2, int(neighbors)), len(observed))
    distances, indices = tree.query(candidates, k=count)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    if distances.shape[0] != len(candidates):
        distances = distances.T
        indices = indices.T

    nearest = distances[:, 0]
    vectors = observed[indices] - candidates[:, None, :]
    lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
    directions = np.divide(
        vectors,
        lengths,
        out=np.zeros_like(vectors),
        where=lengths > EPS,
    )
    balance = 1.0 - np.clip(np.linalg.norm(directions.mean(axis=1), axis=1), 0.0, 1.0)

    if len(observed) > 1:
        observed_spacing = tree.query(observed, k=2)[0][:, 1]
        local_spacing = np.median(observed_spacing[indices], axis=1)
    else:
        local_spacing = np.full(len(candidates), np.inf)
    return nearest, balance, local_spacing


def _weighted_choice(
    indices: np.ndarray,
    scores: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or not len(indices):
        return np.empty(0, dtype=int)
    count = min(int(count), len(indices))
    weights = np.maximum(np.asarray(scores, dtype=float), 0.0) + EPS
    return np.asarray(
        rng.choice(indices, size=count, replace=False, p=weights / weights.sum()),
        dtype=int,
    )


def select_gap_preferences(
    observed_preferences: np.ndarray,
    max_samples: int,
    candidate_pool_size: int,
    threshold_factor: float,
    rng: np.random.Generator,
    return_details: bool = False,
    exhaust_budget: bool = False,
    support_neighbors: int = 32,
    minimum_support: float = 0.20,
    extreme_gap_quantile: float = 0.95,
) -> tuple:
    """Select gaps that are large enough and locally supported by the EA archive."""
    observed = as_2d(observed_preferences)
    spacing = preference_spacing(observed)
    if max_samples <= 0:
        diagnostics = {
            "normal_spacing": spacing,
            "gap_threshold": np.inf,
            "initial_max_gap": 0.0,
            "final_max_gap": 0.0,
        }
        selected = np.empty((0, observed.shape[1]))
        if not return_details:
            return selected, diagnostics
        empty = np.empty(0)
        return selected, diagnostics, {
            "candidate_preferences": selected,
            "initial_distances": empty,
            "final_distances": empty,
            "selected_indices": np.empty(0, dtype=int),
            "selected_above_threshold": np.empty(0, dtype=bool),
            "selection_order": np.empty(0, dtype=int),
            "stop_reason": "model_sample_budget_zero",
        }

    # Keep the configured detection pool larger than the FILL budget. Using
    # a pool of exactly max_samples and selecting every member would make the
    # hole-guided method identical to random preference sampling.
    pool_size = max(max_samples, candidate_pool_size)
    candidates = generate_simplex_candidates(observed.shape[1], pool_size, rng)
    initial_distances, support, local_spacing = _support_scores(
        observed, candidates, support_neighbors
    )
    distances = initial_distances.copy()
    threshold = threshold_factor * spacing if threshold_factor > 0 else 0.0
    threshold = float(threshold) if np.isfinite(threshold) else 0.0
    available = np.ones(len(candidates), dtype=bool)
    indices: list[int] = []
    above_threshold: list[bool] = []
    stop_reason = "candidate_pool_exhausted"

    if exhaust_budget:
        selection_count = min(max_samples, len(candidates))
        initial_candidate_count = len(candidates)
        expansion_rounds = 0
        supported = support >= minimum_support
        while np.count_nonzero(supported) < selection_count:
            supported_count = int(np.count_nonzero(supported))
            if supported_count == 0:
                raise ValueError(
                    "The detection pool contains no locally supported queries"
                )
            missing = selection_count - supported_count
            support_rate = supported_count / len(candidates)
            extra_count = max(
                1024,
                int(np.ceil(1.10 * missing / max(support_rate, EPS))),
            )
            extra_candidates = rng.dirichlet(
                np.ones(observed.shape[1]), size=extra_count
            )
            extra_distances, extra_support, extra_spacing = _support_scores(
                observed, extra_candidates, support_neighbors
            )
            candidates = np.vstack((candidates, extra_candidates))
            initial_distances = np.concatenate((initial_distances, extra_distances))
            distances = initial_distances.copy()
            support = np.concatenate((support, extra_support))
            local_spacing = np.concatenate((local_spacing, extra_spacing))
            supported = support >= minimum_support
            expansion_rounds += 1
        normalized_gap = np.divide(
            initial_distances,
            np.maximum(local_spacing, EPS),
            out=np.zeros_like(initial_distances),
            where=np.isfinite(local_spacing),
        )
        above_threshold = initial_distances > threshold + EPS
        hole_distances = initial_distances[above_threshold]
        extreme_limit = (
            float(np.quantile(hole_distances, extreme_gap_quantile))
            if len(hole_distances)
            else np.inf
        )
        unsupported_extreme = (
            (initial_distances > extreme_limit + EPS) & ~supported
        )
        gap_strength = 1.0 - np.exp(
            -np.maximum(normalized_gap - threshold_factor, 0.0)
        )
        selection_scores = gap_strength * (0.1 + 0.9 * support)
        selection_scores[~above_threshold | ~supported] = 0.0

        hole_indices = np.flatnonzero(selection_scores > 0.0)
        selected_indices = _weighted_choice(
            hole_indices,
            selection_scores[hole_indices],
            selection_count,
            rng,
        )
        remaining_count = selection_count - len(selected_indices)
        if remaining_count:
            remaining = np.setdiff1d(
                np.flatnonzero(supported), selected_indices, assume_unique=True
            )
            if len(remaining) < remaining_count:
                raise ValueError(
                    "The candidate pool contains too few locally supported queries; "
                    "increase candidate_pool_size instead of evaluating unsupported regions"
                )
            fallback_scores = support[remaining] * (initial_distances[remaining] + EPS)
            selected_indices = np.concatenate(
                (
                    selected_indices,
                    _weighted_choice(remaining, fallback_scores, remaining_count, rng),
                )
            )
        selected = candidates[selected_indices]
        selected_as_hole = initial_distances[selected_indices] > threshold + EPS
        covered_distances = cKDTree(
            np.vstack((observed, selected))
        ).query(candidates, k=1)[0]
        diagnostics = {
            "normal_spacing": spacing,
            "gap_threshold": threshold,
            "initial_max_gap": float(initial_distances.max()),
            "final_max_gap": float(covered_distances.max()),
            "selected_count": len(selected),
            "hole_query_count": int(np.count_nonzero(selected_as_hole)),
            "coverage_refinement_query_count": int(
                len(selected) - np.count_nonzero(selected_as_hole)
            ),
            "candidate_count": len(candidates),
            "initial_candidate_count": initial_candidate_count,
            "candidate_pool_expansion_count": len(candidates) - initial_candidate_count,
            "candidate_pool_expansion_rounds": expansion_rounds,
            "selection_strategy": "support_aware_gap_sampling",
            "support_neighbors": int(min(support_neighbors, len(observed))),
            "minimum_support": float(minimum_support),
            "extreme_gap_quantile": float(extreme_gap_quantile),
            "extreme_gap_limit": extreme_limit,
            "excluded_extreme_unsupported": int(np.count_nonzero(unsupported_extreme)),
            "excluded_low_support": int(np.count_nonzero(~supported)),
            "supported_candidate_count": int(np.count_nonzero(supported)),
            "median_candidate_support": float(np.median(support)),
        }
        if not return_details:
            return selected, diagnostics
        return selected, diagnostics, {
            "candidate_preferences": candidates,
            "initial_distances": initial_distances,
            "final_distances": covered_distances,
            "selected_indices": selected_indices,
            "selected_above_threshold": selected_as_hole,
            "selection_order": np.arange(len(selected), dtype=int),
            "support_scores": support,
            "local_spacing": local_spacing,
            "selection_scores": selection_scores,
            "eligible_candidate_mask": supported,
            "observed_preferences": observed,
            "support_neighbors": int(support_neighbors),
            "minimum_support": float(minimum_support),
            "threshold_factor": float(threshold_factor),
            "gap_threshold": threshold,
            "stop_reason": (
                "model_sample_budget_reached"
                if selection_count < len(candidates)
                else "candidate_pool_exhausted"
            ),
        }

    for _ in range(min(max_samples, len(candidates))):
        masked = np.where(available, distances, -np.inf)
        index = int(np.argmax(masked))
        gap = float(masked[index])
        if not np.isfinite(gap) or gap <= EPS:
            break
        is_hole = gap > threshold + EPS
        if not is_hole and not exhaust_budget:
            stop_reason = "gap_threshold_reached"
            break
        indices.append(index)
        above_threshold.append(is_hole)
        available[index] = False
        distances = np.minimum(distances, np.linalg.norm(candidates - candidates[index], axis=1))
    else:
        stop_reason = (
            "candidate_pool_exhausted"
            if len(indices) == len(candidates)
            else "model_sample_budget_reached"
        )

    selected = candidates[indices] if indices else np.empty((0, observed.shape[1]))
    remaining = distances[available]
    diagnostics = {
        "normal_spacing": spacing,
        "gap_threshold": threshold,
        "initial_max_gap": float(initial_distances.max()),
        "final_max_gap": float(remaining.max()) if len(remaining) else 0.0,
        "selected_count": len(indices),
        "hole_query_count": int(np.count_nonzero(above_threshold)),
        "coverage_refinement_query_count": int(len(indices) - np.count_nonzero(above_threshold)),
        "candidate_count": len(candidates),
        "selection_strategy": "exact_greedy_maximin",
    }
    if not return_details:
        return selected, diagnostics
    return selected, diagnostics, {
        "candidate_preferences": candidates,
        "initial_distances": initial_distances,
        "final_distances": distances,
        "selected_indices": np.asarray(indices, dtype=int),
        "selected_above_threshold": np.asarray(above_threshold, dtype=bool),
        "selection_order": np.arange(len(indices), dtype=int),
        "stop_reason": stop_reason,
    }


def adapt_gap_preferences(
    details: dict,
    pilot_count: int,
    pilot_success: np.ndarray,
    total_samples: int,
    success_neighbors: int,
    rng: np.random.Generator,
    local_perturbation_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Reallocate follow-up queries only to empirically successful hole regions."""
    candidates = np.asarray(details["candidate_preferences"], dtype=float)
    initial = np.asarray(details["selected_indices"], dtype=int)
    pilot_count = min(max(0, int(pilot_count)), len(initial), int(total_samples))
    pilot_indices = initial[:pilot_count]
    success = np.asarray(pilot_success, dtype=bool)
    if len(success) != pilot_count:
        raise ValueError("pilot_success must match pilot_count")

    followup_count = int(total_samples) - pilot_count
    supported_mask = np.asarray(
        details.get("eligible_candidate_mask", np.ones(len(candidates), dtype=bool)),
        dtype=bool,
    )
    threshold = float(details.get("gap_threshold", 0.0))
    hole_mask = np.asarray(details["initial_distances"], dtype=float) > threshold + EPS
    supported_remaining = np.setdiff1d(
        np.flatnonzero(supported_mask), pilot_indices, assume_unique=True
    )
    hole_remaining = np.setdiff1d(
        np.flatnonzero(supported_mask & hole_mask), pilot_indices, assume_unique=True
    )
    all_scores = np.asarray(
        details.get("selection_scores", details["initial_distances"]), dtype=float
    )
    generated_count = 0
    fallback = "none"
    if pilot_count and followup_count and np.any(success):
        remaining = hole_remaining
        count = min(max(1, int(success_neighbors)), pilot_count)
        neighbor_indices = cKDTree(candidates[pilot_indices]).query(
            candidates[remaining], k=count
        )[1]
        neighbor_indices = np.asarray(neighbor_indices, dtype=int)
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices[:, None]
        success_rate = success[neighbor_indices].mean(axis=1)
        # A query belongs to a successful region only when its nearest pilot
        # query succeeded. Failed regions receive exactly zero follow-up weight.
        successful_region = success[neighbor_indices[:, 0]]
        adaptive_scores = np.maximum(all_scores[remaining], EPS) * success_rate
        adaptive_scores[~successful_region] = 0.0
        admissible = remaining[adaptive_scores > 0.0]
        followup_indices = _weighted_choice(
            admissible,
            adaptive_scores[adaptive_scores > 0.0],
            min(followup_count, len(admissible)),
            rng,
        )

        missing = followup_count - len(followup_indices)
        if missing:
            observed = np.asarray(details["observed_preferences"], dtype=float)
            successful_centers = candidates[pilot_indices[success]]
            center_scores = np.maximum(all_scores[pilot_indices[success]], EPS)
            center_probabilities = center_scores / center_scores.sum()
            center_spacing = cKDTree(candidates).query(successful_centers, k=2)[0][:, 1]
            noise_scale = max(float(np.median(center_spacing)), EPS) * float(
                local_perturbation_scale
            )
            minimum_support = float(details.get("minimum_support", 0.0))
            support_neighbors = int(details.get("support_neighbors", 32))
            threshold_factor = float(details.get("threshold_factor", 0.0))
            generated_batches = []
            generated_distances = []
            generated_support = []
            generated_spacing = []
            generated_scores = []
            seen = {tuple(np.round(row, 12)) for row in candidates}
            attempts = 0
            while sum(len(batch) for batch in generated_batches) < missing and attempts < 24:
                needed = missing - sum(len(batch) for batch in generated_batches)
                batch_size = max(256, needed * 2)
                centers = successful_centers[
                    rng.choice(
                        len(successful_centers),
                        size=batch_size,
                        replace=True,
                        p=center_probabilities,
                    )
                ]
                noise = rng.normal(0.0, noise_scale, size=centers.shape)
                noise -= noise.mean(axis=1, keepdims=True)
                proposals = np.clip(centers + noise, EPS, None)
                proposals /= proposals.sum(axis=1, keepdims=True)
                distances, support_values, spacing_values = _support_scores(
                    observed, proposals, support_neighbors
                )
                keep = (support_values >= minimum_support) & (distances > threshold + EPS)
                kept_rows = []
                kept_indices = []
                for index in np.flatnonzero(keep):
                    key = tuple(np.round(proposals[index], 12))
                    if key not in seen:
                        seen.add(key)
                        kept_rows.append(proposals[index])
                        kept_indices.append(index)
                    if len(kept_rows) >= needed:
                        break
                if kept_rows:
                    kept_indices = np.asarray(kept_indices, dtype=int)
                    kept = np.asarray(kept_rows, dtype=float)
                    normalized_gap = distances[kept_indices] / np.maximum(
                        spacing_values[kept_indices], EPS
                    )
                    generated_batches.append(kept)
                    generated_distances.append(distances[kept_indices])
                    generated_support.append(support_values[kept_indices])
                    generated_spacing.append(spacing_values[kept_indices])
                    generated_scores.append(
                        (1.0 - np.exp(-np.maximum(normalized_gap - threshold_factor, 0.0)))
                        * support_values[kept_indices]
                    )
                attempts += 1

            generated_count = sum(len(batch) for batch in generated_batches)
            if generated_count < missing:
                raise RuntimeError(
                    "Unable to exhaust the FILL budget inside successful, supported hole regions"
                )
            appended = np.vstack(generated_batches)[:missing]
            appended_distances = np.concatenate(generated_distances)[:missing]
            appended_support = np.concatenate(generated_support)[:missing]
            appended_spacing = np.concatenate(generated_spacing)[:missing]
            appended_scores = np.concatenate(generated_scores)[:missing]
            first_new = len(candidates)
            candidates = np.vstack((candidates, appended))
            details["candidate_preferences"] = candidates
            details["initial_distances"] = np.concatenate(
                (np.asarray(details["initial_distances"]), appended_distances)
            )
            details["final_distances"] = np.concatenate(
                (np.asarray(details["final_distances"]), appended_distances)
            )
            details["support_scores"] = np.concatenate(
                (np.asarray(details["support_scores"]), appended_support)
            )
            details["local_spacing"] = np.concatenate(
                (np.asarray(details["local_spacing"]), appended_spacing)
            )
            details["selection_scores"] = np.concatenate((all_scores, appended_scores))
            details["eligible_candidate_mask"] = np.concatenate(
                (supported_mask, np.ones(missing, dtype=bool))
            )
            generated_indices = np.arange(first_new, first_new + missing, dtype=int)
            followup_indices = np.concatenate((followup_indices, generated_indices))
            generated_count = missing
    else:
        remaining = supported_remaining
        success_rate = np.zeros(len(remaining), dtype=float)
        # Strict FE accounting still needs a defined failure mode when no pilot
        # query succeeds. It remains inside the hard-supported pool and is
        # explicitly reported instead of pretending to be adaptive sampling.
        fallback = "no_pilot_success_high_support"
        adaptive_scores = np.maximum(all_scores[remaining], EPS)
        followup_indices = _weighted_choice(remaining, adaptive_scores, followup_count, rng)
    selected_indices = np.concatenate((pilot_indices, followup_indices))
    updated = dict(details)
    updated["selected_indices"] = selected_indices
    updated["selection_order"] = np.arange(len(selected_indices), dtype=int)
    threshold = float(updated.get("gap_threshold", 0.0))
    updated["selected_above_threshold"] = (
        np.asarray(updated["initial_distances"])[selected_indices] > threshold + EPS
    )
    updated["pilot_count"] = pilot_count
    updated["pilot_success_count"] = int(np.count_nonzero(success))
    updated["followup_count"] = len(followup_indices)
    updated["generated_local_followup_count"] = int(generated_count)
    updated["adaptive_fallback"] = fallback
    updated["adaptive_success_neighbors"] = min(int(success_neighbors), pilot_count)
    updated["remaining_success_rate"] = success_rate
    return candidates[selected_indices], updated


def boundary_probe_capacity(details: dict) -> int:
    """Return the number of unused low-support hole candidates."""
    support = np.asarray(details["support_scores"], dtype=float)
    distances = np.asarray(details["initial_distances"], dtype=float)
    minimum_support = float(details["minimum_support"])
    threshold = float(details["gap_threshold"])
    available = np.flatnonzero(
        (distances > threshold + EPS) & (support < minimum_support)
    )
    selected = np.asarray(details.get("selected_indices", []), dtype=int)
    if len(selected):
        available = np.setdiff1d(available, selected, assume_unique=False)
    return int(len(available))


def select_boundary_probe_preferences(
    details: dict,
    count: int,
    support_floor: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select low-support hole boundaries for fixed-budget active exploration."""
    if count <= 0:
        dimensions = np.asarray(details["candidate_preferences"]).shape[1]
        return np.empty((0, dimensions)), np.empty(0, dtype=int)
    candidates = np.asarray(details["candidate_preferences"], dtype=float)
    support = np.asarray(details["support_scores"], dtype=float)
    distances = np.asarray(details["initial_distances"], dtype=float)
    local_spacing = np.asarray(details["local_spacing"], dtype=float)
    minimum_support = float(details["minimum_support"])
    threshold = float(details["gap_threshold"])
    threshold_factor = float(details.get("threshold_factor", 0.0))
    selected = np.asarray(details.get("selected_indices", []), dtype=int)

    low_support_hole = (
        (distances > threshold + EPS)
        & (support >= support_floor)
        & (support < minimum_support)
    )
    available = np.flatnonzero(low_support_hole)
    if len(selected):
        available = np.setdiff1d(available, selected, assume_unique=False)
    fallback_used = False
    if len(available) < count:
        fallback_used = True
        fallback_pool = np.flatnonzero(
            (distances > threshold + EPS) & (support < minimum_support)
        )
        if len(selected):
            fallback_pool = np.setdiff1d(fallback_pool, selected, assume_unique=False)
        available = np.unique(np.concatenate((available, fallback_pool)))
    details["boundary_probe_fallback_used"] = bool(fallback_used)
    # Boundary probing is optional. The caller normally caps the request to
    # this pool, but truncating here keeps the selector safe for direct use.
    count = min(int(count), len(available))
    details["boundary_probe_available_count"] = int(len(available))
    if count <= 0:
        return np.empty((0, candidates.shape[1])), np.empty(0, dtype=int)

    normalized_gap = distances[available] / np.maximum(local_spacing[available], EPS)
    gap_strength = 1.0 - np.exp(
        -np.maximum(normalized_gap - threshold_factor, 0.0)
    )
    # Prefer the boundary immediately below the hard support threshold. It is
    # informative enough to test but avoids spending FE in unsupported centers.
    boundary_strength = np.clip(support[available] / minimum_support, 0.0, 1.0) ** 2
    if fallback_used:
        boundary_strength = np.maximum(boundary_strength, EPS)
    scores = gap_strength * boundary_strength
    indices = _weighted_choice(available, scores, count, rng)
    return candidates[indices], indices


def build_gap_records(
    details: dict,
    diagnostics: dict,
    model_statuses: np.ndarray,
    candidate_validation: dict | None = None,
    archive_statuses: np.ndarray | None = None,
) -> list[dict]:
    """Explain every detector candidate without benchmark-specific knowledge."""
    selected_order = details["selected_indices"].tolist()
    selected_lookup = {candidate: model for model, candidate in enumerate(selected_order)}
    selected_as_hole = details.get(
        "selected_above_threshold", np.ones(len(selected_order), dtype=bool)
    )
    threshold = float(diagnostics["gap_threshold"])
    records = []
    for candidate, preference in enumerate(details["candidate_preferences"]):
        model_index = selected_lookup.get(candidate, -1)
        if model_index >= 0:
            reason = (
                str(model_statuses[model_index])
                if model_index < len(model_statuses)
                else "model_query_pending"
            )
            phase = (
                str(details["query_phases"][model_index])
                if "query_phases" in details
                else ""
            )
            if phase == "boundary_probe":
                role = "low_support_boundary_probe"
            elif selected_as_hole[model_index]:
                role = "geometric_hole_query"
            else:
                role = "coverage_refinement_query"
        elif details["final_distances"][candidate] <= threshold + EPS:
            reason, role = "below_gap_threshold", "not_queried"
        elif details["stop_reason"] == "model_sample_budget_reached":
            reason, role = "model_sample_budget_exhausted", "not_queried"
        else:
            reason, role = "candidate_pool_exhausted", "not_queried"
        records.append(
            {
                "candidate_index": candidate,
                "model_index": model_index,
                "selected_for_model": model_index >= 0,
                "query_role": role,
                "initial_distance": float(details["initial_distances"][candidate]),
                "final_distance": float(details["final_distances"][candidate]),
                "reason_code": reason,
                "reason": REASON_TEXT[reason],
                "preference": preference.tolist(),
            }
        )
        if "support_scores" in details:
            records[-1]["support_score"] = float(details["support_scores"][candidate])
        if "selection_scores" in details:
            records[-1]["selection_score"] = float(details["selection_scores"][candidate])
        if model_index >= 0:
            phases = details.get("query_phases")
            records[-1]["allocation_phase"] = (
                str(phases[model_index])
                if phases is not None and model_index < len(phases)
                else (
                    "pilot"
                    if model_index < int(details.get("pilot_count", 0))
                    else "adaptive_followup"
                )
            )
        if model_index >= 0 and candidate_validation is not None:
            records[-1].update(
                {
                    "actual_preference": candidate_validation["actual_preferences"][model_index].tolist(),
                    "direction_error": float(candidate_validation["direction_error"][model_index]),
                    "direction_limit": float(candidate_validation["direction_limit"]),
                    "effective_direction_limit": float(
                        candidate_validation["effective_direction_limit"][model_index]
                    ),
                    "target_base_distance": float(
                        candidate_validation["target_base_distance"][model_index]
                    ),
                    "coverage_improvement": float(
                        candidate_validation["coverage_improvement"][model_index]
                    ),
                    "inside_empirical_hull": bool(
                        candidate_validation["inside_empirical_hull"][model_index]
                    ),
                    "candidate_radius": float(candidate_validation["candidate_radius"][model_index]),
                    "predicted_radius": float(candidate_validation["predicted_radius"][model_index]),
                    "relative_front_error": float(candidate_validation["relative_front_error"][model_index]),
                    "front_error_limit": float(candidate_validation["front_error_limit"]),
                    "local_fit_error": float(candidate_validation["local_fit_error"][model_index]),
                    "local_fit_error_limit": float(candidate_validation["local_fit_error_limit"]),
                    "branch_disagreement": float(candidate_validation["branch_disagreement"][model_index]),
                    "branch_disagreement_limit": float(candidate_validation["branch_disagreement_limit"]),
                    "local_support": bool(candidate_validation["local_support"][model_index]),
                }
            )
        if model_index >= 0 and archive_statuses is not None:
            archive_status = str(archive_statuses[model_index])
            records[-1].update(
                {
                    "archive_status": archive_status,
                    "archive_retained": archive_status == "model_nondominated",
                }
            )
    return records
