"""Persistence for one auditable GD-PSL run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from experiment_config import ExperimentConfig
from result_layout import method_directory_name, run_directory
from .archive import as_2d

def write_front_csv(
    path: Path,
    objectives: np.ndarray,
    sources: Optional[np.ndarray] = None,
) -> None:
    values = as_2d(objectives)
    fields = ["row", *[f"f{index + 1}" for index in range(values.shape[1])]]
    if sources is not None:
        fields.append("source")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, objectives_row in enumerate(values):
            row = {"row": index}
            row.update(
                {f"f{column + 1}": value for column, value in enumerate(objectives_row)}
            )
            if sources is not None:
                row["source"] = "model" if int(sources[index]) else "base"
            writer.writerow(row)


def write_gap_csv(path: Path, records: list[dict], dimensions: int) -> None:
    fields = [
        "candidate_index",
        "model_index",
        "selected_for_model",
        "query_role",
        "allocation_phase",
        "initial_distance",
        "final_distance",
        "support_score",
        "selection_score",
        "reason_code",
        "reason",
        "archive_status",
        "archive_retained",
        "direction_error",
        "direction_limit",
        "effective_direction_limit",
        "target_base_distance",
        "coverage_improvement",
        "inside_empirical_hull",
        "candidate_radius",
        "local_radius_median",
        "local_radius_limit",
        *[f"preference_{index + 1}" for index in range(dimensions)],
        *[f"actual_preference_{index + 1}" for index in range(dimensions)],
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields if key in record}
            row.update(
                {
                    f"preference_{index + 1}": value
                    for index, value in enumerate(record["preference"])
                }
            )
            if "actual_preference" in record:
                row.update(
                    {
                        f"actual_preference_{index + 1}": value
                        for index, value in enumerate(record["actual_preference"])
                    }
                )
            writer.writerow(row)


def _json_number(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def save_run_artifacts(
    output_dir: str,
    algorithm_name: str,
    problem_name: str,
    completion_method: str,
    run_index: int,
    base_x: np.ndarray,
    base_f: np.ndarray,
    model_x: np.ndarray,
    model_f: np.ndarray,
    verified_hole_x: np.ndarray,
    verified_hole_f: np.ndarray,
    completed_x: np.ndarray,
    completed_f: np.ndarray,
    completed_sources: np.ndarray,
    completed_verified_hole: np.ndarray,
    completed_boundary_probe: np.ndarray,
    stage1_x: np.ndarray,
    stage1_f: np.ndarray,
    stage1_fe: np.ndarray,
    observed_preferences: np.ndarray,
    gap_preferences: np.ndarray,
    gap_info: dict,
    gap_records: list[dict],
    training_history: dict,
    completion_stage_times: dict,
    hypervolume_report: dict,
    config: ExperimentConfig,
    search_time_seconds: float,
    completion_time_seconds: float,
    metric_time_seconds: float,
    method_time_seconds: float,
    igd_infinity_report: Optional[dict] = None,
    reference_front: Optional[np.ndarray] = None,
) -> Path:
    """Write arrays, tables, configuration, diagnostics, metrics, and timing."""
    uses_model = int(gap_info.get("model_fe_budget", 0)) > 0
    run_dir = run_directory(
        output_dir,
        method_directory_name(algorithm_name, uses_model),
        problem_name,
        base_f.shape[1],
        config.seed + run_index,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "fronts.npz",
        base_x=base_x,
        base_f=base_f,
        model_x=model_x,
        model_f=model_f,
        verified_hole_x=verified_hole_x,
        verified_hole_f=verified_hole_f,
        completed_x=completed_x,
        completed_f=completed_f,
        completed_sources=completed_sources,
        completed_verified_hole=completed_verified_hole,
        completed_boundary_probe=completed_boundary_probe,
        stage1_x=stage1_x,
        stage1_f=stage1_f,
        stage1_fe=stage1_fe,
        observed_preferences=observed_preferences,
        gap_preferences=gap_preferences,
        reference_pf=(
            np.asarray(reference_front, dtype=float)
            if reference_front is not None
            else np.empty((0, base_f.shape[1]))
        ),
    )
    write_front_csv(run_dir / "base_pf.csv", base_f)
    write_front_csv(run_dir / "model_candidates.csv", model_f)
    write_front_csv(run_dir / "verified_hole_pf.csv", verified_hole_f)
    write_front_csv(run_dir / "completed_pf.csv", completed_f, completed_sources)
    write_gap_csv(run_dir / "gap_regions.csv", gap_records, observed_preferences.shape[1])
    write_gap_csv(
        run_dir / "unfilled_regions.csv",
        [record for record in gap_records if record["reason_code"] != "model_nondominated"],
        observed_preferences.shape[1],
    )

    with (run_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "validation_loss"])
        writer.writerows(
            zip(
                training_history["epoch"],
                training_history["train_loss"],
                training_history["validation_loss"],
            )
        )
    training_summary = {
        key: value
        for key, value in training_history.items()
        if key not in {"epoch", "train_loss", "validation_loss"}
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    igd = igd_infinity_report or {
        "available": False,
        "reason": "IGD-infinity report was not supplied",
    }
    reason_counts: dict[str, int] = {}
    for record in gap_records:
        reason = record["reason_code"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "algorithm": algorithm_name,
        "problem": problem_name,
        "completion_method": completion_method,
        "evaluation_variant": (
            "ea_only" if int(gap_info.get("model_fe_budget", 0)) == 0 else completion_method
        ),
        "run": run_index + 1,
        "run_seed": config.seed + run_index,
        "base_nondominated": len(base_f),
        "model_candidates": len(model_f),
        "completed_nondominated": len(completed_f),
        "archive_deduplication": "exact_objective_vector",
        "model_survivors": int(np.count_nonzero(completed_sources == 1)),
        "verified_hole_survivors": int(np.count_nonzero(completed_verified_hole)),
        "boundary_probe_survivors": int(np.count_nonzero(completed_boundary_probe)),
        "gap_info": {key: _json_number(value) for key, value in gap_info.items()},
        "gap_reason_counts": reason_counts,
        "training": training_summary,
        "hypervolume": hypervolume_report,
        "igd_infinity": igd,
        "reference_pf_used": bool(igd.get("available", False)),
        "reference_information_used_for_optimization": False,
        "reference_information_used_for_reporting": bool(igd.get("available", False)),
        "problem_specific_topology_used": False,
        "interpretation": "Reasons use only runtime geometry, true evaluation, and dominance",
        "runtime_scope": (
            "algorithm_execution_total includes EA search, hole detection, model training, "
            "model generation/true evaluation, and archive update; metrics and artifacts are excluded"
        ),
        "experiment_config": config.to_dict(),
        "runtime_seconds": {
            "ea_search_and_validation": float(search_time_seconds),
            "model_completion": float(completion_time_seconds),
            **{key: float(value) for key, value in completion_stage_times.items()},
            "evaluation_metrics": float(metric_time_seconds),
            "ea_plus_model_training": float(
                search_time_seconds + completion_stage_times.get("model_training", 0.0)
            ),
            "algorithm_execution_total": float(
                search_time_seconds + completion_time_seconds
            ),
            "reporting_pipeline_total": float(method_time_seconds),
            "method_execution_total": float(search_time_seconds + completion_time_seconds),
            "total": float(search_time_seconds + completion_time_seconds),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "experiment_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return run_dir
