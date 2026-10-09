"""Pang--Nan--Ishibuchi large-solution-set archive baseline.

This runner is deliberately independent from ``run_fill_then_judge.py``.  It
uses PlatEMO only for evolutionary search and keeps the complete evaluation
history as an unbounded archive. The reported UEA is the unique nondominated
subset of that history, exactly as in the authors' supplied code. No Pareto-set
model, generated evaluations, or invented fixed-size subset is used.

The published paper's central reproducible protocol is therefore represented
explicitly in the artifacts: ``evaluations_*`` is the unbounded archive and
``archive_*``/``selected_*`` are the same reported UEA.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from evaluation_metrics import (
    archive_igd_infinity_report,
    hypervolume,
    load_benchmark_reference_front,
    normalize_objectives,
    resolve_normalization_points,
)
from experiment_config import (
    COMPARISON_POPULATION_SIZE,
    COMPARISON_TOTAL_FE_BUDGET,
    DEFAULT_CONFIG,
    IGD_INFINITY_REFERENCE_SAMPLES,
)
from gd_psl.platemo import reference_front
from problem_definitions import get_problem
from result_layout import run_directory
from matlab_engine_utils import start_matlab_with_retry
from pareto_utils import nondominated_indices as _shared_nondominated_indices


def _as_2d(values: np.ndarray, columns: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, columns or 0), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {array.shape}.")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns, got {array.shape[1]}.")
    return array


def nondominated_indices(objectives: np.ndarray) -> np.ndarray:
    """Return rank-one rows for minimization objectives."""
    return _shared_nondominated_indices(_as_2d(objectives))


def default_pang_population_size(n_objectives: int) -> int:
    """Use the authors' DTLZ population sizes where they are specified."""
    if n_objectives == 3:
        return 91
    if n_objectives == 5:
        return 210
    return COMPARISON_POPULATION_SIZE


def _write_matrix(path: Path, values: np.ndarray, prefix: str) -> None:
    values = _as_2d(values)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{prefix}{index + 1}" for index in range(values.shape[1])])
        writer.writerows(values.tolist())


def _run_platemo(
    engine: object,
    algorithm: str,
    problem: str,
    population_size: int,
    max_fe: int,
    n_objectives: int,
    n_dimensions: int,
    platemo_root: str,
    seed: int,
    parameters: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import matlab

    parameter_values = list(map(float, parameters))
    matlab_parameters = matlab.double([parameter_values]) if parameter_values else matlab.double([])
    _, _, _, history_x, history_f, history_fe = engine.run_platemo(
        algorithm,
        problem.upper(),
        float(population_size),
        float(max_fe),
        float(n_objectives),
        float(n_dimensions),
        platemo_root,
        matlab_parameters,
        float(seed),
        nargout=6,
    )
    return (
        np.asarray(history_x, dtype=float),
        np.asarray(history_f, dtype=float),
        np.asarray(history_fe, dtype=float).reshape(-1),
    )


def run_one(
    engine: object,
    algorithm: str,
    problem_name: str,
    problem: object,
    population_size: int,
    max_fe: int,
    search_max_fe: int,
    platemo_root: str,
    parameters: Sequence[float],
    seed: int,
    coverage_samples: int,
    report_reference: Optional[np.ndarray],
    output_dir: Path,
    plot: bool,
) -> Path:
    method_name = "PangMOEAD" if algorithm.upper() == "PANGMOEAD" else f"PangArchive_{algorithm.upper()}"
    run_dir = run_directory(
        output_dir,
        method_name,
        problem_name,
        problem.n_obj,
        seed,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history_x, history_f, history_fe = _run_platemo(
        engine, algorithm, problem_name, population_size, search_max_fe, problem.n_obj, problem.n_dim,
        platemo_root, seed, parameters,
    )
    search_seconds = time.perf_counter() - started
    history_x = _as_2d(history_x, problem.n_dim)
    history_f = _as_2d(history_f, problem.n_obj)
    if len(history_x) != len(history_f):
        raise RuntimeError("PlatEMO evaluation history has inconsistent decision/objective lengths")
    fe_used = len(history_f)
    if fe_used != max_fe:
        raise RuntimeError(
            f"Strict FE fairness requires {max_fe} evaluations, got {fe_used}"
        )
    finite = np.isfinite(history_x).all(axis=1) & np.isfinite(history_f).all(axis=1)
    history_x, history_f = history_x[finite], history_f[finite]
    history_fe = history_fe[: len(finite)][finite] if len(history_fe) >= len(finite) else np.arange(1, len(history_f) + 1)
    unique_f, unique_indices = np.unique(history_f, axis=0, return_index=True)
    unique_indices.sort(); unique_f = history_f[unique_indices]; unique_x = history_x[unique_indices]
    front_indices = nondominated_indices(unique_f)
    archive_x, archive_f = unique_x[front_indices], unique_f[front_indices]
    ideal, nadir, normalization_source = resolve_normalization_points(problem_name, problem, problem.n_obj)
    # In the authors' code UEA itself is the final large solution set. Their
    # IDSS and k-means subset-selection experiments are separate baselines.
    selected_x, selected_f = archive_x, archive_f
    normalized_archive = normalize_objectives(archive_f, ideal, nadir)
    hv, hv_method = hypervolume(normalized_archive, np.full(problem.n_obj, 1.1), seed=seed)
    igd_infinity = archive_igd_infinity_report(
        archive_f,
        archive_f,
        problem_name,
        coverage_samples,
        ideal,
        nadir,
        report_reference,
    )
    np.savez_compressed(
        run_dir / "fronts.npz", evaluations_x=history_x, evaluations_f=history_f,
        evaluations_fe=history_fe, archive_x=archive_x, archive_f=archive_f,
        selected_x=selected_x, selected_f=selected_f,
        reference_pf=(
            report_reference
            if report_reference is not None
            else np.empty((0, problem.n_obj))
        ),
    )
    # Pang et al.'s unbounded archive is the complete evaluation history, not
    # only the rank-one subset used for coverage measurements.
    _write_matrix(run_dir / "unbounded_archive.csv", np.hstack((history_x, history_f)), "value_")
    _write_matrix(run_dir / "nondominated_archive.csv", np.hstack((archive_x, archive_f)), "value_")
    _write_matrix(run_dir / "large_solution_set.csv", np.hstack((selected_x, selected_f)), "value_")
    # Remove artifacts produced by the superseded hole-query postprocessor.
    (run_dir / "coverage_queries.csv").unlink(missing_ok=True)
    summary = {
        "algorithm": algorithm, "problem": problem_name,
        "method": "pang_archive_baseline", "method_directory": method_name,
        "method_label": "Pang",
        "result_scope": "complete_historical_nondominated_archive",
        "paper": "Pang, Nan, Ishibuchi, IEEE SMC 2023, pp. 1188-1194",
        "run": seed, "population_size": population_size, "common_total_fe_budget": max_fe,
        "platemo_stop_fe": search_max_fe,
        "fe_used": int(fe_used), "evaluation_history_size": int(len(history_f)),
        "unbounded_archive_size": int(len(history_f)), "nondominated_archive_size": int(len(archive_f)),
        "large_solution_set_size": int(len(selected_f)),
        "coverage_samples": coverage_samples,
        "hypervolume": {"value": float(hv), "method": hv_method, "normalization_ideal": ideal.tolist(),
                        "normalization_nadir": nadir.tolist(), "normalization_source": normalization_source,
                        "reference_point": [1.1] * problem.n_obj},
        "igd_infinity": igd_infinity,
        "search_time_seconds": float(search_seconds),
        "reference_pf_used": bool(igd_infinity.get("available", False)),
        "protocol": "unbounded evaluation archive -> nondominated objective vectors -> exact duplicate removal -> UEA",
        "implementation_version": "pang_authors_code_sync_v1",
        "weight_perturbation": "original_uniform_weights_plus_independent_uniform_noise_every_100_generations",
        "reproduction_scope": "authors' algorithm and UEA/IGD protocol under project-selected FE, problems, and run count",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if plot:
        # All non-EA methods use the same archive comparison visualizer as
        # fill_then_judge; this keeps method panels and 3-D views comparable.
        from reporting.visualize import visualize_run

        visualize_run(run_dir, run_dir / "figures")
    igd_value = igd_infinity.get("completed_igd_infinity")
    igd_text = f"{igd_value:.6g}" if igd_value is not None else "unavailable"
    print(f"[{algorithm}/{problem_name}/run {seed}] FE={fe_used}/{max_fe}, archive={len(archive_f)}, "
          f"UEA={len(selected_f)}, IGDinf={igd_text}, "
          f"HV={hv:.8g}, artifacts={run_dir}")
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pang et al. unbounded-archive baseline.")
    parser.add_argument("--algorithms", nargs="+", default=["MOEAD"])
    parser.add_argument("--problems", nargs="+", default=["dtlz2", "dtlz7"])
    parser.add_argument("--n-objectives", type=int, default=3)
    parser.add_argument("--runs", type=int, default=DEFAULT_CONFIG.runs)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--population-size", type=int, default=None,
        help="Override population size (default: Pang's 91 for 3 objectives, 210 for 5).",
    )
    parser.add_argument(
        "--max-fe", type=int, default=COMPARISON_TOTAL_FE_BUDGET,
        help="Common total true-evaluation budget.",
    )
    parser.add_argument(
        "--platemo-max-fe", type=int, default=None,
        help="Compatibility option; strict fairness requires it to equal --max-fe.",
    )
    parser.add_argument(
        "--coverage-samples", type=int, default=IGD_INFINITY_REFERENCE_SAMPLES
    )
    parser.add_argument("--algorithm-parameters", nargs="*", type=float, default=[])
    parser.add_argument("--platemo-root", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse a run whose summary records the requested exact FE budget.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    import matlab.engine

    bridge = str(Path(__file__).resolve().parent / "matlab")
    engine = start_matlab_with_retry(matlab.engine)
    engine.addpath(bridge, nargout=0)
    if args.platemo_root:
        engine.addpath(engine.genpath(args.platemo_root), nargout=0)
    try:
        for algorithm in args.algorithms:
            for problem_name in args.problems:
                if args.n_objectives is not None and problem_name.lower().startswith("dtlz"):
                    problem = get_problem(problem_name, n_obj=args.n_objectives)
                else:
                    problem = get_problem(problem_name)
                if problem_name.lower() in {"dtlz2", "dtlz7"}:
                    report_reference = reference_front(
                        engine,
                        problem_name,
                        problem.n_obj,
                        problem.n_dim,
                        args.coverage_samples,
                        args.platemo_root,
                    )
                else:
                    report_reference, _ = load_benchmark_reference_front(
                        problem_name, problem.n_obj
                    )
                for run_offset in range(args.runs):
                    population_size = (
                        args.population_size
                        if args.population_size is not None
                        else default_pang_population_size(problem.n_obj)
                    )
                    search_max_fe = args.platemo_max_fe if args.platemo_max_fe is not None else args.max_fe
                    if search_max_fe != args.max_fe:
                        raise ValueError(
                            "--platemo-max-fe must equal --max-fe under strict FE fairness"
                        )
                    run_seed = args.seed + run_offset
                    method_name = (
                        "PangMOEAD"
                        if algorithm.upper() == "PANGMOEAD"
                        else f"PangArchive_{algorithm.upper()}"
                    )
                    expected_dir = run_directory(
                        args.output_dir,
                        method_name,
                        problem_name,
                        problem.n_obj,
                        run_seed,
                    )
                    summary_path = expected_dir / "summary.json"
                    if args.skip_completed and summary_path.exists():
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        if (
                            int(summary.get("common_total_fe_budget", -1)) == args.max_fe
                            and summary.get("implementation_version") == "pang_authors_code_sync_v1"
                        ):
                            print(f"[skip] completed run: {expected_dir}", flush=True)
                            continue
                    run_one(engine, algorithm, problem_name, problem, population_size, args.max_fe,
                            search_max_fe,
                            args.platemo_root, args.algorithm_parameters, run_seed,
                            args.coverage_samples, report_reference,
                            args.output_dir, args.plot)
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
