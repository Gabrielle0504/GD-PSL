"""Fair, reusable quality metrics for saved multi-objective archives."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .archive import nondominated_indices


def _as_vector(value: Any, n_objectives: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=float).reshape(-1)
    if len(vector) != n_objectives or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {n_objectives} finite values")
    return vector


def resolve_normalization_points(
    problem_name: str,
    problem: Any,
    n_objectives: int,
    project_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve fixed benchmark ideal/nadir points without using run outcomes."""
    root = project_root or Path(__file__).resolve().parents[1]
    family = "RE" if problem_name.lower().startswith("re") else "DTLZ_WFG"
    directory = root / "data" / family / "ideal_nadir_points"
    suffix = problem_name.upper() if family == "RE" else problem_name.lower()
    ideal_path = directory / f"ideal_point_{suffix}.dat"
    nadir_path = directory / f"nadir_point_{suffix}.dat"

    if ideal_path.exists() and nadir_path.exists():
        ideal = np.loadtxt(ideal_path, dtype=float).reshape(-1)
        nadir = np.loadtxt(nadir_path, dtype=float).reshape(-1)
        if len(ideal) == n_objectives and len(nadir) == n_objectives:
            source = f"benchmark_files:{ideal_path.relative_to(root)}|{nadir_path.relative_to(root)}"
            _validate_normalization(ideal, nadir)
            return ideal, nadir, source

    ideal_value = getattr(problem, "ideal_point", np.zeros(n_objectives, dtype=float))
    nadir_value = getattr(problem, "nadir_point", None)
    if nadir_value is None:
        raise ValueError(f"No fixed nadir point is defined for {problem_name}")
    ideal = _as_vector(ideal_value, n_objectives, "ideal point")
    nadir = _as_vector(nadir_value, n_objectives, "nadir point")
    _validate_normalization(ideal, nadir)
    return ideal, nadir, "problem_definition"


def load_benchmark_reference_front(
    problem_name: str,
    n_objectives: int,
    project_root: Path | None = None,
) -> tuple[np.ndarray, str]:
    """Load a fixed full-dimensional reference PF shipped with the project."""
    root = project_root or Path(__file__).resolve().parents[1]
    family = "RE" if problem_name.lower().startswith("re") else "DTLZ_WFG"
    suffix = problem_name.upper() if family == "RE" else problem_name.lower()
    path = root / "data" / family / "ParetoFront" / f"{suffix}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"No benchmark reference PF exists for {problem_name}: {path}")
    values = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != n_objectives:
        raise ValueError(
            f"Reference PF for {problem_name} must have {n_objectives} columns, "
            f"got {values.shape}"
        )
    values = np.unique(values[np.isfinite(values).all(axis=1)], axis=0)
    if not len(values):
        raise ValueError(f"Reference PF for {problem_name} contains no finite rows")
    return values, f"benchmark_file:{path.relative_to(root).as_posix()}"


def _validate_normalization(ideal: np.ndarray, nadir: np.ndarray) -> None:
    if not np.isfinite(ideal).all() or not np.isfinite(nadir).all():
        raise ValueError("HV normalization points must be finite")
    if np.any(nadir <= ideal):
        raise ValueError("Every HV nadir component must be greater than its ideal component")


def normalize_objectives(
    objectives: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")
    ideal = _as_vector(ideal, values.shape[1], "ideal point")
    nadir = _as_vector(nadir, values.shape[1], "nadir point")
    _validate_normalization(ideal, nadir)
    return (values - ideal) / (nadir - ideal)


def _prepare_hv_points(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(reference):
        raise ValueError("points and reference dimensions do not match")
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("reference must contain positive finite values")
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return np.empty((0, len(reference)), dtype=float)
    # The external ideal is the lower boundary of the normalized HV box.
    values = np.maximum(values, 0.0)
    values = values[np.all(values < reference, axis=1)]
    if not len(values):
        return np.empty((0, len(reference)), dtype=float)
    values = np.unique(values, axis=0)
    return values[nondominated_indices(values)]


def _exact_hypervolume_3d(points: np.ndarray, reference: np.ndarray) -> float:
    """Sweep objective 3 with an exact logarithmic-time active 2-D frontier."""
    order = np.lexsort((points[:, 1], points[:, 0], points[:, 2]))
    ordered = points[order]
    x_coordinates = np.unique(ordered[:, 0])
    active_y = np.full(len(x_coordinates), np.inf)
    fenwick = np.zeros(len(x_coordinates) + 1, dtype=int)

    def add(index: int, delta: int) -> None:
        cursor = index + 1
        while cursor < len(fenwick):
            fenwick[cursor] += delta
            cursor += cursor & -cursor

    def prefix_count(end: int) -> int:
        total = 0
        cursor = end
        while cursor > 0:
            total += int(fenwick[cursor])
            cursor -= cursor & -cursor
        return total

    def kth(order_index: int) -> int:
        if order_index <= 0 or order_index > prefix_count(len(x_coordinates)):
            return -1
        index = 0
        bit = 1 << (len(fenwick).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate < len(fenwick) and fenwick[candidate] < order_index:
                index = candidate
                order_index -= int(fenwick[candidate])
            bit >>= 1
        return index

    def predecessor(index: int) -> int:
        count = prefix_count(index)
        return kth(count) if count else -1

    def successor(index: int) -> int:
        before = prefix_count(index)
        total = prefix_count(len(x_coordinates))
        return kth(before + 1) if before < total else -1

    def next_active(index: int) -> int:
        through = prefix_count(index + 1)
        total = prefix_count(len(x_coordinates))
        return kth(through + 1) if through < total else -1

    def contribution(index: int, next_index: int) -> float:
        next_x = reference[0] if next_index < 0 else x_coordinates[next_index]
        return float((next_x - x_coordinates[index]) * (reference[1] - active_y[index]))

    area = 0.0
    volume = 0.0
    start = 0

    while start < len(ordered):
        level = ordered[start, 2]
        end = start + 1
        while end < len(ordered) and ordered[end, 2] == level:
            end += 1

        for x_value, y_value in ordered[start:end, :2]:
            position = int(np.searchsorted(x_coordinates, x_value))
            previous = predecessor(position)
            if previous >= 0 and active_y[previous] <= y_value:
                continue
            if np.isfinite(active_y[position]) and active_y[position] <= y_value:
                continue
            first = successor(position)
            if previous >= 0:
                area -= contribution(previous, first)
            removed: list[int] = []
            cursor = first
            while cursor >= 0 and active_y[cursor] >= y_value:
                following = next_active(cursor)
                area -= contribution(cursor, following)
                removed.append(cursor)
                cursor = following
            for index in removed:
                add(index, -1)
                active_y[index] = np.inf
            if np.isfinite(active_y[position]):
                # An active point at the same x coordinate was included above.
                add(position, -1)
                active_y[position] = np.inf
            active_y[position] = float(y_value)
            add(position, 1)
            following = next_active(position)
            if previous >= 0:
                area += contribution(previous, position)
            area += contribution(position, following)

        next_level = ordered[end, 2] if end < len(ordered) else reference[2]
        volume += area * (next_level - level)
        start = end
    return float(volume)


def _exact_hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    if not len(points):
        return 0.0
    if points.shape[1] == 1:
        return float(reference[0] - np.min(points[:, 0]))
    if points.shape[1] == 2:
        ordered = points[np.argsort(points[:, 0], kind="stable")]
        area = 0.0
        previous_y = reference[1]
        for x_value, y_value in ordered:
            if y_value < previous_y:
                area += (reference[0] - x_value) * (previous_y - y_value)
                previous_y = y_value
        return float(area)
    if points.shape[1] == 3:
        return _exact_hypervolume_3d(points, reference)
    levels = np.unique(points[:, -1])
    boundaries = np.concatenate((levels, [reference[-1]]))
    volume = 0.0
    for index, level in enumerate(levels):
        width = boundaries[index + 1] - level
        if width <= 0:
            continue
        active = points[points[:, -1] <= level, :-1]
        volume += _exact_hypervolume(active, reference[:-1]) * width
    return float(volume)


def hypervolume(
    normalized_objectives: np.ndarray,
    reference: np.ndarray,
    monte_carlo_samples: int = 131072,
    seed: int = 2026,
) -> tuple[float, str]:
    """Return minimization HV; exact through 3-D and Sobol-estimated above 3-D."""
    reference = np.asarray(reference, dtype=float).reshape(-1)
    points = _prepare_hv_points(normalized_objectives, reference)
    if len(reference) <= 3:
        return _exact_hypervolume(points, reference), "exact_recursive"
    if not len(points):
        return 0.0, "sobol_qmc"
    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")

    sampler = qmc.Sobol(d=len(reference), scramble=True, seed=seed)
    exponent = int(np.ceil(np.log2(monte_carlo_samples)))
    samples = sampler.random_base2(exponent)[:monte_carlo_samples] * reference
    dominated_count = 0
    for start in range(0, len(samples), 4096):
        sample_chunk = samples[start : start + 4096]
        dominated = np.any(
            np.all(points[:, None, :] <= sample_chunk[None, :, :], axis=2),
            axis=0,
        )
        dominated_count += int(np.count_nonzero(dominated))
    box_volume = float(np.prod(reference))
    return box_volume * dominated_count / len(samples), "sobol_qmc"


def archive_hypervolume_report(
    base_objectives: np.ndarray,
    completed_objectives: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    reference_value: float,
    monte_carlo_samples: int,
    seed: int,
    normalization_source: str,
) -> dict[str, Any]:
    n_objectives = np.asarray(completed_objectives).shape[1]
    reference = np.full(n_objectives, reference_value, dtype=float)
    base_normalized = normalize_objectives(base_objectives, ideal, nadir)
    completed_normalized = normalize_objectives(completed_objectives, ideal, nadir)
    base_hv, method = hypervolume(base_normalized, reference, monte_carlo_samples, seed)
    completed_hv, completed_method = hypervolume(
        completed_normalized, reference, monte_carlo_samples, seed
    )
    if method != completed_method:
        raise RuntimeError("Inconsistent HV methods for archives of the same dimension")
    return {
        "normalization_ideal": np.asarray(ideal, dtype=float).tolist(),
        "normalization_nadir": np.asarray(nadir, dtype=float).tolist(),
        "normalization_source": normalization_source,
        "reference_point": reference.tolist(),
        "method": method,
        "samples": int(monte_carlo_samples) if method == "sobol_qmc" else None,
        "seed": int(seed) if method == "sobol_qmc" else None,
        "base_stage_hv": float(base_hv),
        "completed_hv": float(completed_hv),
        "delta_hv": float(completed_hv - base_hv),
        "base_stage_is_equal_budget_baseline": False,
    }


def projected_igd_infinity(
    archive: np.ndarray,
    reference_front: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    """Return Pang's maximum pairwise projected nearest-point distance."""
    values = np.asarray(archive, dtype=float)
    reference = np.asarray(reference_front, dtype=float)
    if values.ndim != 2 or reference.ndim != 2 or values.shape[1] != reference.shape[1]:
        raise ValueError("archive and reference_front must be compatible matrices")
    values = np.unique(values[np.isfinite(values).all(axis=1)], axis=0)
    reference = np.unique(reference[np.isfinite(reference).all(axis=1)], axis=0)
    if not len(values) or not len(reference):
        raise ValueError("archive and reference_front must contain finite rows")

    pair_scores = []
    for first, second in combinations(range(values.shape[1]), 2):
        distances = cKDTree(values[:, [first, second]]).query(
            reference[:, [first, second]], k=1
        )[0]
        pair_scores.append(
            {
                "objectives": [first + 1, second + 1],
                "igd_infinity": float(np.max(distances)),
                "mean_distance": float(np.mean(distances)),
            }
        )
    if not pair_scores:
        raise ValueError("projected IGD-infinity requires at least two objectives")
    return max(item["igd_infinity"] for item in pair_scores), pair_scores


def pang_projection_references(
    problem_name: str,
    n_objectives: int,
    sample_count: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Recreate the normalized 2-D reference sets from the authors' code."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    name = problem_name.lower()
    pairs = list(combinations(range(n_objectives), 2))

    if name == "dtlz2":
        divisions = 1
        while (divisions - 1) * divisions / 2 < sample_count:
            divisions += 1
        divisions -= 1
        indices = np.asarray(list(combinations(range(1, divisions + 1), 2)), dtype=float)
        reference = np.column_stack(
            (indices[:, 0] - 1.0, divisions - indices[:, 1])
        ) / (divisions - 2.0)
        nonzero = np.sum(reference, axis=1) != 0
        row_sum = np.sum(reference[nonzero], axis=1, keepdims=True)
        row_norm = np.linalg.norm(reference[nonzero], axis=1, keepdims=True)
        reference[nonzero] = reference[nonzero] / row_norm * row_sum
        return {pair: reference.copy() for pair in pairs}

    if name == "dtlz7":
        divisions = int(np.floor(np.sqrt(sample_count)))
        gap = np.linspace(0.0, 1.0, divisions)
        first, second = np.meshgrid(gap, gap, indexing="ij")
        projected = np.column_stack(
            (first.ravel(order="F"), second.ravel(order="F"))
        )
        intervals = np.array([0.0, 0.251412, 0.631627, 0.859401])
        median = (intervals[1] - intervals[0]) / (
            intervals[3] - intervals[2] + intervals[1] - intervals[0]
        )
        lower = projected <= median
        projected[lower] = (
            projected[lower] * (intervals[1] - intervals[0]) / median + intervals[0]
        )
        projected[~lower] = (
            (projected[~lower] - median)
            * (intervals[3] - intervals[2])
            / (1.0 - median)
            + intervals[2]
        )
        reference_without_last = projected.copy()
        reference_with_last = projected[:, ::-1].copy()
        # GetReferencePointSet.m calls getDataDTLZ7(...,3) for the shared
        # reference file, including the five-objective experiments.
        author_m = 3
        a = intervals[3]
        for index in range(divisions):
            b = reference_without_last[index, 0]
            s = (b / 2.0) * (1.0 + np.sin(3.0 * np.pi * b))
            lower_bound = 2.0 * (
                author_m
                - ((author_m - 2.0) * (a / 2.0) * (1.0 + np.sin(3.0 * np.pi * a)) + s)
            )
            upper_bound = 2.0 * (author_m - s)
            start = index * divisions
            reference_with_last[start : start + divisions, 1] = np.linspace(
                lower_bound, upper_bound, divisions
            )

        for values in (reference_without_last, reference_with_last):
            minimum = np.min(values, axis=0)
            maximum = np.max(values, axis=0)
            values -= minimum
            values /= maximum - minimum
        return {
            pair: (reference_with_last if pair[1] == n_objectives - 1 else reference_without_last).copy()
            for pair in pairs
        }

    raise ValueError(f"The authors provide no projected reference set for {problem_name}")


def pang_projected_igd_infinity(
    archive: np.ndarray,
    problem_name: str,
    true_front: np.ndarray,
    sample_count: int,
) -> tuple[float, list[dict[str, Any]], int]:
    """Compute the authors' normalized maximum projected IGD distance."""
    values = np.asarray(archive, dtype=float)
    true_values = np.asarray(true_front, dtype=float)
    if values.ndim != 2 or true_values.ndim != 2 or values.shape[1] != true_values.shape[1]:
        raise ValueError("archive and true_front must be compatible matrices")
    values = np.unique(values[np.isfinite(values).all(axis=1)], axis=0)
    true_values = true_values[np.isfinite(true_values).all(axis=1)]
    if not len(values) or not len(true_values):
        raise ValueError("archive and true_front must contain finite rows")
    ideal = np.min(true_values, axis=0)
    nadir = np.max(true_values, axis=0)
    _validate_normalization(ideal, nadir)
    normalized = normalize_objectives(values, ideal, nadir)
    references = pang_projection_references(
        problem_name, values.shape[1], sample_count
    )

    pair_scores = []
    for pair, reference in references.items():
        distances = cKDTree(normalized[:, pair]).query(reference, k=1)[0]
        pair_scores.append(
            {
                "objectives": [pair[0] + 1, pair[1] + 1],
                "igd_infinity": float(np.max(distances)),
                "mean_distance": float(np.mean(distances)),
            }
        )
    return (
        max(item["igd_infinity"] for item in pair_scores),
        pair_scores,
        len(next(iter(references.values()))),
    )


def archive_igd_infinity_report(
    base_objectives: np.ndarray,
    completed_objectives: np.ndarray,
    problem_name: str,
    reference_samples: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    reference_front: np.ndarray | None,
) -> dict[str, Any]:
    """Evaluate both archives with one reporting-only projected IGD-infinity protocol."""
    if reference_front is None:
        return {
            "available": False,
            "reason": f"no shared true-PF reference is available for {problem_name}",
        }
    if reference_samples < 2:
        raise ValueError("reference_samples must be at least 2")

    name = problem_name.lower()
    if name in {"dtlz2", "dtlz7"}:
        base_value, base_pairs, actual_samples = pang_projected_igd_infinity(
            base_objectives, name, reference_front, reference_samples
        )
        completed_value, completed_pairs, _ = pang_projected_igd_infinity(
            completed_objectives, name, reference_front, reference_samples
        )
        distance_space = "true_pf_min_max_normalized_objective_space"
        reference_source = "Pang authors' problem-specific normalized 2-D reference sets"
    else:
        normalized_reference = normalize_objectives(reference_front, ideal, nadir)
        normalized_base = normalize_objectives(base_objectives, ideal, nadir)
        normalized_completed = normalize_objectives(
            completed_objectives, ideal, nadir
        )
        base_value, base_pairs = projected_igd_infinity(
            normalized_base, normalized_reference
        )
        completed_value, completed_pairs = projected_igd_infinity(
            normalized_completed, normalized_reference
        )
        actual_samples = int(len(np.asarray(reference_front)))
        distance_space = "fixed_ideal_nadir_normalized_objective_space"
        reference_source = (
            f"benchmark_file:data/RE/ParetoFront/{problem_name.upper()}.dat"
        )
    return {
        "available": True,
        "protocol": "pang_projected_2d",
        "definition": "max over objective pairs of max_z min_a ||z-a||_2",
        "distance_space": distance_space,
        "reference_source": reference_source,
        "reference_samples_requested": int(reference_samples),
        "reference_samples_actual": actual_samples,
        "base_igd_infinity": float(base_value),
        "completed_igd_infinity": float(completed_value),
        "base_pair_scores": base_pairs,
        "pair_scores": completed_pairs,
    }
