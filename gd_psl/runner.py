"""End-to-end orchestration for one or more GD-PSL experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

from .metrics import (
    archive_hypervolume_report,
    archive_igd_infinity_report,
    load_benchmark_reference_front,
    resolve_normalization_points,
)
from .config import COMPLETION_METHOD, ExperimentConfig
from .problems import get_problem

from .archive import (
    EPS,
    classify_model_candidates,
    merge_and_classify_candidates,
    objectives_to_preferences,
    standardize_base_result,
)
from .artifacts import method_directory_name, run_directory, save_run_artifacts
from .candidate_validation import validate_empirical_candidates
from .holes import (
    adapt_gap_preferences,
    boundary_probe_capacity,
    build_gap_records,
    select_boundary_probe_preferences,
    select_gap_preferences,
)
from .modeling import evaluate_model, synchronize, train_model
from .platemo import reference_front, run_platemo_algorithm, validate_problem_consistency


def _empty_training_history(seed: int) -> dict:
    return {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "best_epoch": None,
        "best_validation_loss": None,
        "train_loss_at_best_epoch": None,
        "generalization_gap_at_best_epoch": None,
        "epochs_trained": 0,
        "early_stopped": False,
        "converged": False,
        "stop_reason": "training_not_required",
        "convergence_rule": "no model query was selected",
        "interpretation": "training was skipped because no model query was selected",
        "training_time_seconds": 0.0,
        "train_samples": 0,
        "validation_samples": 0,
        "split_strategy": "not_applicable",
        "split_seed": seed,
        "parameter_count": 0,
    }


def _merge_completion_archive(
    base_x: np.ndarray,
    base_f: np.ndarray,
    model_x: np.ndarray,
    model_f: np.ndarray,
    validation: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge every true-evaluated proposal while retaining validation diagnostics."""
    completed_x, completed_f, sources, archive_statuses = (
        merge_and_classify_candidates(base_x, base_f, model_x, model_f)
    )
    eligible = np.asarray(validation["eligible"], dtype=bool)
    if len(eligible) != len(model_f):
        raise ValueError("Candidate validation and model evaluation counts differ")
    diagnostic_statuses = np.asarray(validation["statuses"], dtype=object).copy()
    diagnostic_statuses[eligible] = archive_statuses[eligible]
    retained = archive_statuses == "model_nondominated"
    return (
        completed_x,
        completed_f,
        sources,
        archive_statuses,
        diagnostic_statuses,
        retained,
    )


def complete_front(
    base_x: np.ndarray,
    base_f: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    problem,
    total_fe_budget: int,
    ea_fe_budget: int,
    ea_fe_used: int,
    model_fe_budget: int,
    config: ExperimentConfig,
    rng: np.random.Generator,
    seed: int,
) -> dict:
    """Detect fillable holes, adapt the query budget, and update the archive."""
    started = time.perf_counter()
    hole_started = time.perf_counter()
    observed, empirical_ideal, empirical_nadir = objectives_to_preferences(base_f)
    gap_preferences, gap_info, gap_details = select_gap_preferences(
        observed,
        model_fe_budget,
        config.candidate_pool_size,
        config.gap_threshold_factor,
        rng,
        return_details=True,
        exhaust_budget=config.exhaust_model_fe_budget,
        support_neighbors=config.hole_support_neighbors,
        minimum_support=config.hole_minimum_support,
        extreme_gap_quantile=config.hole_extreme_gap_quantile,
    )
    hole_seconds = time.perf_counter() - hole_started

    training = _empty_training_history(seed)
    generation_seconds = 0.0
    pilot_validation_seconds = 0.0
    if len(gap_preferences):
        model, _, training = train_model(
            base_x,
            observed,
            lower_bounds,
            upper_bounds,
            problem.n_obj,
            config,
            seed,
        )
        synchronize(config.device)

        requested_boundary_probe_count = min(
            len(gap_preferences) - 1,
            int(round(len(gap_preferences) * config.hole_boundary_probe_fraction)),
        )
        available_boundary_probe_count = boundary_probe_capacity(gap_details)
        boundary_probe_count = min(
            requested_boundary_probe_count, available_boundary_probe_count
        )
        completion_count = len(gap_preferences) - boundary_probe_count
        pilot_count = min(
            completion_count,
            max(1, int(np.ceil(completion_count * config.hole_pilot_fraction))),
        )
        pilot_targets = gap_preferences[:pilot_count]
        generation_started = time.perf_counter()
        pilot_x, pilot_f, pilot_refinement = evaluate_model(
            model,
            pilot_targets,
            lower_bounds,
            upper_bounds,
            problem,
            config.device,
            config.model_inference_batch_size,
            archive_decisions=base_x,
            archive_preferences=observed,
            refinement_enabled=config.local_refinement_enabled,
            refinement_candidate_neighbors=config.local_refinement_candidate_neighbors,
            refinement_branch_neighbors=config.local_refinement_branch_neighbors,
            refinement_ridge=config.local_refinement_ridge,
        )
        synchronize(config.device)
        generation_seconds += time.perf_counter() - generation_started

        pilot_validation_started = time.perf_counter()
        pilot_validation = validate_empirical_candidates(
            base_f,
            pilot_targets,
            pilot_x,
            pilot_f,
            empirical_ideal,
            empirical_nadir,
            gap_info["normal_spacing"],
            config.direction_match_factor,
            config.local_front_neighbors,
            config.local_front_error_quantile,
            config.local_front_min_neighbors,
            config.local_front_ridge,
            config.local_front_calibration_samples,
        )
        pilot_archive_status = classify_model_candidates(
            base_x, base_f, pilot_x, pilot_f
        )
        pilot_is_hole = np.asarray(
            gap_details.get("selected_above_threshold", np.ones(len(gap_preferences), dtype=bool))
        )[:pilot_count]
        pilot_success = pilot_is_hole & pilot_validation["eligible"] & (
            pilot_archive_status == "model_nondominated"
        )
        pilot_validation_seconds = time.perf_counter() - pilot_validation_started

        adaptation_started = time.perf_counter()
        gap_preferences, gap_details = adapt_gap_preferences(
            gap_details,
            pilot_count,
            pilot_success,
            completion_count,
            config.hole_success_neighbors,
            rng,
            config.hole_local_perturbation_scale,
        )
        selected_as_hole = np.asarray(
            gap_details.get("selected_above_threshold", np.ones(len(gap_preferences), dtype=bool))
        )
        gap_info["hole_query_count"] = int(np.count_nonzero(selected_as_hole))
        gap_info["coverage_refinement_query_count"] = int(
            len(selected_as_hole) - np.count_nonzero(selected_as_hole)
        )
        hole_seconds += time.perf_counter() - adaptation_started
        completion_preferences = gap_preferences
        followup_targets = completion_preferences[pilot_count:]
        generation_started = time.perf_counter()
        followup_x, followup_f, followup_refinement = evaluate_model(
            model,
            followup_targets,
            lower_bounds,
            upper_bounds,
            problem,
            config.device,
            config.model_inference_batch_size,
            archive_decisions=base_x,
            archive_preferences=observed,
            refinement_enabled=config.local_refinement_enabled,
            refinement_candidate_neighbors=config.local_refinement_candidate_neighbors,
            refinement_branch_neighbors=config.local_refinement_branch_neighbors,
            refinement_ridge=config.local_refinement_ridge,
        )
        synchronize(config.device)
        generation_seconds += time.perf_counter() - generation_started

        probe_targets, probe_indices = select_boundary_probe_preferences(
            gap_details,
            boundary_probe_count,
            config.hole_boundary_probe_support_floor,
            rng,
        )
        probe_started = time.perf_counter()
        probe_ranks = np.arange(len(probe_targets), dtype=int)
        probe_x, probe_f, probe_refinement = evaluate_model(
            model,
            probe_targets,
            lower_bounds,
            upper_bounds,
            problem,
            config.device,
            config.model_inference_batch_size,
            archive_decisions=base_x,
            archive_preferences=observed,
            refinement_enabled=config.local_refinement_enabled,
            refinement_candidate_neighbors=config.local_refinement_candidate_neighbors,
            refinement_branch_neighbors=config.local_refinement_branch_neighbors,
            refinement_ridge=config.local_refinement_ridge,
            refinement_branch_ranks=probe_ranks,
            refinement_maximum_branches=config.hole_boundary_probe_branches,
        )
        synchronize(config.device)
        generation_seconds += time.perf_counter() - probe_started
        model_x = np.vstack((pilot_x, followup_x, probe_x))
        model_f = np.vstack((pilot_f, followup_f, probe_f))
        gap_preferences = np.vstack((completion_preferences, probe_targets))
        completion_above_threshold = np.asarray(
            gap_details["selected_above_threshold"], dtype=bool
        )
        gap_details["selected_indices"] = np.concatenate(
            (np.asarray(gap_details["selected_indices"], dtype=int), probe_indices)
        )
        gap_details["selected_above_threshold"] = np.concatenate(
            (completion_above_threshold, np.ones(len(probe_targets), dtype=bool))
        )
        gap_details["selection_order"] = np.arange(len(gap_preferences), dtype=int)
        gap_details["query_phases"] = np.asarray(
            ["pilot"] * pilot_count
            + ["adaptive_followup"] * len(followup_targets)
            + ["boundary_probe"] * len(probe_targets),
            dtype=object,
        )
        gap_info["hole_query_count"] = int(np.count_nonzero(completion_above_threshold))
        gap_info["coverage_refinement_query_count"] = int(
            len(completion_above_threshold) - np.count_nonzero(completion_above_threshold)
        )
        refinement = {
            "enabled": bool(config.local_refinement_enabled),
            "strategy": "two_stage_adaptive_local_refinement",
            "seconds": float(
                pilot_refinement.get("seconds", 0.0)
                + followup_refinement.get("seconds", 0.0)
                + probe_refinement.get("seconds", 0.0)
            ),
            "pilot": pilot_refinement,
            "followup": followup_refinement,
            "boundary_probe": probe_refinement,
            "reference_pf_used": False,
            "additional_true_evaluations": 0,
        }
        gap_info.update(
            {
                "selection_strategy": "support_aware_two_stage_adaptive",
                "pilot_query_count": pilot_count,
                "pilot_success_count": int(np.count_nonzero(pilot_success)),
                "followup_query_count": len(followup_targets),
                "boundary_probe_query_count": len(probe_targets),
                "boundary_probe_requested_count": requested_boundary_probe_count,
                "boundary_probe_available_count": available_boundary_probe_count,
                "boundary_probe_shortfall": (
                    requested_boundary_probe_count - len(probe_targets)
                ),
                "boundary_probe_support_floor": config.hole_boundary_probe_support_floor,
                "boundary_probe_branches": config.hole_boundary_probe_branches,
                "boundary_probe_fallback_used": bool(
                    gap_details.get("boundary_probe_fallback_used", False)
                ),
                "pilot_success_rate": float(np.mean(pilot_success)),
                "generated_local_followup_count": int(
                    gap_details.get("generated_local_followup_count", 0)
                ),
                "adaptive_fallback": gap_details.get("adaptive_fallback", "none"),
            }
        )
    else:
        model_x = np.empty((0, len(lower_bounds)))
        model_f = np.empty((0, problem.n_obj))
        refinement = {
            "enabled": bool(config.local_refinement_enabled),
            "strategy": "not_applied_no_queries",
            "seconds": 0.0,
            "reference_pf_used": False,
            "additional_true_evaluations": 0,
        }
        gap_info.update(
            {
                "pilot_query_count": 0,
                "pilot_success_count": 0,
                "followup_query_count": 0,
                "boundary_probe_query_count": 0,
                "boundary_probe_requested_count": 0,
                "boundary_probe_available_count": 0,
                "boundary_probe_shortfall": 0,
                "pilot_success_rate": 0.0,
            }
        )

    if len(model_f) > model_fe_budget:
        raise RuntimeError("Model evaluations exceeded their FE allocation")
    if config.exhaust_model_fe_budget and len(model_f) != model_fe_budget:
        raise RuntimeError("Strict FE mode requires the complete model allocation")
    if ea_fe_used + len(model_f) > total_fe_budget:
        raise RuntimeError("EA and model evaluations exceeded the common FE budget")
    if config.exhaust_model_fe_budget and ea_fe_used + len(model_f) != total_fe_budget:
        raise RuntimeError("Strict FE mode requires the complete common FE budget")

    gap_info.update(
        {
            "training_archive_definition": "cumulative_empirical_pf_from_all_ea_evaluations",
            "training_archive_size": len(base_f),
            "exhaust_model_fe_budget": config.exhaust_model_fe_budget,
            "common_total_fe_budget": total_fe_budget,
            "ea_only_baseline_fe_budget": total_fe_budget,
            "ea_fe_budget": ea_fe_budget,
            "ea_fe_used": ea_fe_used,
            "model_fe_budget": model_fe_budget,
            "model_fe_used": len(model_f),
            "model_to_ea_fe_budget_ratio": model_fe_budget / ea_fe_budget,
            "model_to_ea_fe_actual_ratio": len(model_f) / ea_fe_used,
            "unused_total_fe": total_fe_budget - ea_fe_used - len(model_f),
            "local_refinement": refinement,
        }
    )
    validation_started = time.perf_counter()
    validation = validate_empirical_candidates(
        base_f,
        gap_preferences,
        model_x,
        model_f,
        empirical_ideal,
        empirical_nadir,
        gap_info["normal_spacing"],
        config.direction_match_factor,
        config.local_front_neighbors,
        config.local_front_error_quantile,
        config.local_front_min_neighbors,
        config.local_front_ridge,
        config.local_front_calibration_samples,
    )
    validation_seconds = (
        pilot_validation_seconds + time.perf_counter() - validation_started
    )
    merge_started = time.perf_counter()
    (
        completed_x,
        completed_f,
        sources,
        archive_statuses,
        hole_statuses,
        accepted,
    ) = _merge_completion_archive(base_x, base_f, model_x, model_f, validation)
    eligible = np.asarray(validation["eligible"], dtype=bool)
    query_phases = np.asarray(
        gap_details.get("query_phases", ["hole_fill"] * len(model_f)), dtype=object
    )
    boundary_probe_mask = query_phases == "boundary_probe"
    queried_holes = np.asarray(
        gap_details.get("selected_above_threshold", np.ones(len(model_f), dtype=bool)),
        dtype=bool,
    )
    verified_hole_mask = (
        ~boundary_probe_mask & queried_holes & eligible & accepted
    )
    verified_hole_x = model_x[verified_hole_mask]
    verified_hole_f = model_f[verified_hole_mask]
    verified_keys = {tuple(row) for row in verified_hole_f}
    completed_verified_hole = np.asarray(
        [
            bool(source == 1 and tuple(row) in verified_keys)
            for row, source in zip(completed_f, sources)
        ],
        dtype=bool,
    )
    boundary_survivor_f = model_f[boundary_probe_mask & accepted]
    boundary_keys = {tuple(row) for row in boundary_survivor_f}
    completed_boundary_probe = np.asarray(
        [
            bool(source == 1 and tuple(row) in boundary_keys)
            for row, source in zip(completed_f, sources)
        ],
        dtype=bool,
    )
    accepted_preferences = validation["actual_preferences"][accepted]
    coverage_preferences = (
        np.vstack((observed, accepted_preferences))
        if len(accepted_preferences)
        else observed
    )
    if len(gap_details["candidate_preferences"]):
        gap_details["final_distances"] = cKDTree(coverage_preferences).query(
            gap_details["candidate_preferences"], k=1
        )[0]
        gap_info["final_max_gap"] = float(gap_details["final_distances"].max())
    else:
        gap_info["final_max_gap"] = 0.0
    gap_info.update(
        {
            "direction_match_limit": validation["direction_limit"],
            "coverage_improving_model_candidates": int(
                np.count_nonzero(validation["coverage_improvement"] > EPS)
            ),
            "local_front_error_limit": validation["front_error_limit"],
            "local_front_fit_error_limit": validation["local_fit_error_limit"],
            "local_front_branch_disagreement_limit": validation[
                "branch_disagreement_limit"
            ],
            "local_front_calibration_samples": validation["calibration_samples"],
            "empirically_valid_model_candidates": int(np.count_nonzero(validation["eligible"])),
            "accepted_model_candidates": int(np.count_nonzero(accepted)),
            "successful_hole_queries": int(
                np.count_nonzero(verified_hole_mask)
            ),
            "boundary_probe_archive_survivors": int(
                np.count_nonzero(boundary_probe_mask & accepted)
            ),
            "boundary_probe_dominated": int(
                np.count_nonzero(
                    boundary_probe_mask
                    & ~accepted
                    & eligible
                    & (archive_statuses == "model_dominated")
                )
            ),
            "boundary_probe_direction_mismatch": int(
                np.count_nonzero(
                    boundary_probe_mask
                    & ~accepted
                    & (hole_statuses == "model_direction_mismatch")
                )
            ),
            "boundary_probe_unsupported_topology": int(
                np.count_nonzero(
                    boundary_probe_mask
                    & ~accepted
                    & (hole_statuses == "model_unsupported_topology")
                )
            ),
            "boundary_probe_off_empirical_front": int(
                np.count_nonzero(
                    boundary_probe_mask
                    & ~accepted
                    & (hole_statuses == "model_off_empirical_front")
                )
            ),
            "boundary_probe_rejected": int(
                np.count_nonzero(boundary_probe_mask & ~accepted)
            ),
            "boundary_probe_other_rejection": int(
                np.count_nonzero(
                    boundary_probe_mask
                    & ~accepted
                    & ~(
                        (hole_statuses == "model_direction_mismatch")
                        | (hole_statuses == "model_unsupported_topology")
                        | (hole_statuses == "model_off_empirical_front")
                        | (
                            eligible
                            & (archive_statuses == "model_dominated")
                        )
                    )
                )
            ),
            "boundary_probe_discovery_rate": float(
                np.mean(accepted[boundary_probe_mask])
                if np.any(boundary_probe_mask)
                else 0.0
            ),
            "unsupported_topology_model_candidates": int(
                np.count_nonzero(hole_statuses == "model_unsupported_topology")
            ),
            "off_empirical_front_model_candidates": int(
                np.count_nonzero(hole_statuses == "model_off_empirical_front")
            ),
            "archive_admission_rule": "finite_feasible_objective_unique_nondominated",
            "hole_diagnostics_do_not_reject_archive_solutions": True,
            "candidate_validation_reference_pf_used": False,
        }
    )
    records = build_gap_records(
        gap_details,
        gap_info,
        hole_statuses,
        validation,
        archive_statuses,
    )
    merge_seconds = time.perf_counter() - merge_started
    return {
        "observed_preferences": observed,
        "gap_preferences": gap_preferences,
        "gap_info": gap_info,
        "gap_records": records,
        "model_x": model_x,
        "model_f": model_f,
        "verified_hole_x": verified_hole_x,
        "verified_hole_f": verified_hole_f,
        "completed_x": completed_x,
        "completed_f": completed_f,
        "completed_sources": sources,
        "completed_verified_hole": completed_verified_hole,
        "completed_boundary_probe": completed_boundary_probe,
        "training_history": training,
        "completion_stage_times": {
            "hole_detection": hole_seconds,
            "model_training": training["training_time_seconds"],
            "model_generation_and_true_evaluation": generation_seconds,
            "model_local_refinement": refinement.get("seconds", 0.0),
            "candidate_validation": validation_seconds,
            "merge_filter_and_classification": merge_seconds,
            "completion_internal_total": time.perf_counter() - started,
        },
    }


def run_once(
    engine,
    algorithm: str,
    problem_name: str,
    problem,
    population_size: int,
    ea_fe_budget: int,
    total_fe_budget: int,
    model_fe_budget: int,
    report_reference: Optional[np.ndarray],
    config: ExperimentConfig,
    run_index: int,
) -> dict:
    """Run one PlatEMO search, one GD-PSL completion, and all metrics."""
    seed = config.seed + run_index
    np.random.seed(seed)
    torch.manual_seed(seed)
    method_started = time.perf_counter()
    search_started = time.perf_counter()
    raw_x, raw_f, raw_c, history_x, history_f, history_fe = run_platemo_algorithm(
        engine,
        algorithm,
        problem_name.upper(),
        population_size,
        ea_fe_budget,
        problem.n_obj,
        problem.n_dim,
        config.platemo_root,
        seed,
        config.algorithm_parameters,
    )
    final_x, final_f = standardize_base_result(
        raw_x, raw_f, raw_c, config.feasibility_tolerance
    )
    base_x, base_f = (
        standardize_base_result(history_x, history_f, None, config.feasibility_tolerance)
        if len(history_x) and len(history_f)
        else (final_x, final_f)
    )
    validate_problem_consistency(
        base_x,
        base_f,
        problem,
        config.device,
        config.consistency_sample_size,
        config.consistency_relative_tolerance,
        config.consistency_absolute_tolerance,
    )
    search_seconds = time.perf_counter() - search_started
    ea_fe_used = int(np.max(history_fe)) if len(history_fe) else ea_fe_budget
    if ea_fe_used > ea_fe_budget:
        raise RuntimeError("PlatEMO exceeded the allocated EA FE budget")

    completion_started = time.perf_counter()
    completion = complete_front(
        base_x,
        base_f,
        np.asarray(problem.lbound.cpu(), dtype=float),
        np.asarray(problem.ubound.cpu(), dtype=float),
        problem,
        total_fe_budget,
        ea_fe_budget,
        ea_fe_used,
        model_fe_budget,
        config,
        np.random.default_rng(seed),
        seed,
    )
    completion_seconds = time.perf_counter() - completion_started

    metric_started = time.perf_counter()
    ideal, nadir, normalization_source = resolve_normalization_points(
        problem_name, problem, problem.n_obj
    )
    hv = archive_hypervolume_report(
        base_f,
        completion["completed_f"],
        ideal,
        nadir,
        config.hypervolume_reference_value,
        config.hypervolume_monte_carlo_samples,
        config.hypervolume_seed,
        normalization_source,
    )
    igd = archive_igd_infinity_report(
        base_f,
        completion["completed_f"],
        problem_name,
        config.igd_infinity_reference_samples,
        ideal,
        nadir,
        report_reference,
    )
    metric_seconds = time.perf_counter() - metric_started
    reporting_pipeline_seconds = time.perf_counter() - method_started
    algorithm_seconds = search_seconds + completion_seconds
    run_dir = save_run_artifacts(
        config.output_dir,
        algorithm,
        problem_name,
        COMPLETION_METHOD,
        run_index,
        base_x,
        base_f,
        completion["model_x"],
        completion["model_f"],
        completion["verified_hole_x"],
        completion["verified_hole_f"],
        completion["completed_x"],
        completion["completed_f"],
        completion["completed_sources"],
        completion["completed_verified_hole"],
        completion["completed_boundary_probe"],
        history_x,
        history_f,
        history_fe,
        completion["observed_preferences"],
        completion["gap_preferences"],
        completion["gap_info"],
        completion["gap_records"],
        completion["training_history"],
        completion["completion_stage_times"],
        hv,
        config,
        search_seconds,
        completion_seconds,
        metric_seconds,
        reporting_pipeline_seconds,
        igd,
        report_reference,
    )
    return {
        "base_x": base_x,
        "base_f": base_f,
        "final_population_x": final_x,
        "final_population_f": final_f,
        "stage1_x": history_x,
        "stage1_f": history_f,
        "stage1_fe": history_fe,
        "completion_method": COMPLETION_METHOD,
        "search_time": search_seconds,
        "completion_time": completion_seconds,
        "metric_time": metric_seconds,
        "total_time": algorithm_seconds,
        "reporting_pipeline_time": reporting_pipeline_seconds,
        "run_dir": run_dir,
        "hypervolume": hv,
        "igd_infinity": igd,
        **completion,
    }


def _print_result(algorithm: str, problem: str, run_index: int, result: dict) -> None:
    gap, hv, igd = result["gap_info"], result["hypervolume"], result["igd_infinity"]
    print(
        f"[{algorithm}/{problem}/run {run_index + 1}] "
        f"FE={gap['ea_fe_used']}+{gap['model_fe_used']}, "
        f"archive={len(result['completed_f'])}, HV={hv['completed_hv']:.8g}, "
        f"IGDinf={igd.get('completed_igd_infinity')}, "
        f"time={result['total_time']:.3f}s, artifacts={result['run_dir']}",
        flush=True,
    )


def run_experiments(config: ExperimentConfig, *, skip_completed: bool = False) -> None:
    """Execute the configured algorithm/problem/run matrix."""
    config.validate()
    try:
        import matlab.engine
    except ImportError as error:
        raise RuntimeError("MATLAB Engine for Python is required") from error

    engine = matlab.engine.start_matlab()
    engine.addpath(str(Path(__file__).resolve().parents[1] / "matlab"), nargout=0)
    try:
        for algorithm in config.algorithms:
            for problem_name in config.problems:
                problem = (
                    get_problem(problem_name, n_obj=config.n_objectives)
                    if config.n_objectives is not None and problem_name.startswith("dtlz")
                    else get_problem(problem_name)
                )
                population = config.resolve_population_size(problem.n_dim)
                total_fe, ea_fe, model_fe = config.resolve_evaluation_budgets(problem.n_dim)
                if problem_name in {"dtlz2", "dtlz7"}:
                    report_reference = reference_front(
                        engine,
                        problem_name,
                        problem.n_obj,
                        problem.n_dim,
                        config.igd_infinity_reference_samples,
                        config.platemo_root,
                    )
                else:
                    report_reference, _ = load_benchmark_reference_front(
                        problem_name, problem.n_obj
                    )
                for run_index in range(config.runs):
                    expected = run_directory(
                        config.output_dir,
                        method_directory_name(algorithm, not config.ea_only),
                        problem_name,
                        problem.n_obj,
                        config.seed + run_index,
                    )
                    if skip_completed and (expected / "summary.json").exists():
                        print(f"[skip] completed artifacts: {expected}", flush=True)
                        continue
                    result = run_once(
                        engine,
                        algorithm,
                        problem_name,
                        problem,
                        population,
                        ea_fe,
                        total_fe,
                        model_fe,
                        report_reference,
                        config,
                        run_index,
                    )
                    _print_result(algorithm, problem_name, run_index, result)
                    if config.plot:
                        from reporting.visualize import visualize_run

                        visualize_run(
                            result["run_dir"], Path(result["run_dir"]) / "figures"
                        )
    finally:
        engine.quit()
