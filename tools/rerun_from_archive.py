"""Re-run GD-PSL completion from a saved EA archive without repeating PlatEMO."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from evaluation_metrics import (
    archive_hypervolume_report,
    archive_igd_infinity_report,
    resolve_normalization_points,
)
from experiment_config import COMPLETION_METHOD, DEFAULT_CONFIG
from gd_psl.artifacts import save_run_artifacts
from gd_psl.runner import complete_front
from problem_definitions import get_problem
from reporting.visualize import visualize_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse a saved EA archive and run only GD-PSL completion."
    )
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    source = arguments.source_run.resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    with np.load(source / "fronts.npz") as saved:
        base_x = np.asarray(saved["base_x"], dtype=float)
        base_f = np.asarray(saved["base_f"], dtype=float)
        stage1_x = np.asarray(saved["stage1_x"], dtype=float)
        stage1_f = np.asarray(saved["stage1_f"], dtype=float)
        stage1_fe = np.asarray(saved["stage1_fe"], dtype=float)
        report_reference = (
            np.asarray(saved["reference_pf"], dtype=float)
            if "reference_pf" in saved and len(saved["reference_pf"])
            else None
        )

    algorithm = str(summary["algorithm"])
    problem_name = str(summary["problem"]).lower()
    seed = int(summary.get("run_seed", 1))
    source_gap = summary.get("gap_info", {})
    total_fe = int(source_gap.get("common_total_fe_budget", 100_000))
    ea_fe = int(source_gap.get("ea_fe_used", 90_000))
    model_fe = total_fe - ea_fe
    if model_fe <= 0:
        raise ValueError("The source run leaves no FE budget for model completion")

    problem = get_problem(problem_name)
    config = replace(
        DEFAULT_CONFIG,
        algorithms=(algorithm,),
        problems=(problem_name,),
        n_objectives=None,
        runs=1,
        seed=seed,
        max_fe=total_fe,
        ea_fill_split=f"{ea_fe * 100 // total_fe}:{model_fe * 100 // total_fe}",
        output_dir=arguments.output_dir,
        device=arguments.device,
        plot=not arguments.no_plot,
    ).validate()
    if torch.device(config.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started = time.perf_counter()
    completion = complete_front(
        base_x,
        base_f,
        np.asarray(problem.lbound.cpu(), dtype=float),
        np.asarray(problem.ubound.cpu(), dtype=float),
        problem,
        total_fe,
        ea_fe,
        ea_fe,
        model_fe,
        config,
        np.random.default_rng(seed),
        seed,
    )
    completion_seconds = time.perf_counter() - started
    completion["gap_info"].update(
        {
            "ea_archive_reused": True,
            "source_run_directory": str(source),
        }
    )

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
    source_search_seconds = float(
        summary.get("runtime_seconds", {}).get(
            "ea_search_and_validation", summary.get("search_time_seconds", 0.0)
        )
    )
    run_dir = save_run_artifacts(
        config.output_dir,
        algorithm,
        problem_name,
        COMPLETION_METHOD,
        0,
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
        stage1_x,
        stage1_f,
        stage1_fe,
        completion["observed_preferences"],
        completion["gap_preferences"],
        completion["gap_info"],
        completion["gap_records"],
        completion["training_history"],
        completion["completion_stage_times"],
        hv,
        config,
        source_search_seconds,
        completion_seconds,
        metric_seconds,
        source_search_seconds + completion_seconds + metric_seconds,
        igd,
        report_reference,
    )
    if config.plot:
        visualize_run(run_dir, run_dir / "figures")
    print(
        json.dumps(
            {
                "run_directory": str(run_dir),
                "completion_seconds": completion_seconds,
                "model_training_seconds": completion["completion_stage_times"][
                    "model_training"
                ],
                "model_fe": model_fe,
                "verified_hole_fills": len(completion["verified_hole_f"]),
                "model_archive_survivors": int(
                    np.count_nonzero(completion["completed_sources"] == 1)
                ),
                "hypervolume": hv.get("completed_hv"),
                "igd_infinity": igd.get("completed_igd_infinity"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
