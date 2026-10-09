"""Run and compare the three EA/GD-PSL pairs plus PangMOEAD."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiment_config import ALLOWED_EA_FILL_SPLITS, DEFAULT_CONFIG
from evaluation_metrics import (
    load_benchmark_reference_front,
    normalize_objectives,
    projected_igd_infinity,
    resolve_normalization_points,
)
from problem_definitions import get_problem
from result_layout import problem_directory_name, run_directory
from run_fill_then_judge import run_experiments
from run_pang_archive_baseline import main as run_pang
from pareto_utils import nondominated_indices
from .visualize import visualize_run


ALGORITHMS = ("NSGAII", "NSGAIII", "MOEAD")
DTLZ_PROBLEMS = ("dtlz2", "dtlz7")
RE_PROBLEMS = ("re21", "re24", "re31", "re32", "re34", "re35", "re37")
METHODS = (
    "EA_NSGAII",
    "EA_NSGAIII",
    "EA_MOEAD",
    "GD-PSL_NSGAII",
    "GD-PSL_NSGAIII",
    "GD-PSL_MOEAD",
    "PangMOEAD",
)
METHOD_LABELS = {
    "EA_NSGAII": "NSGA-II",
    "EA_NSGAIII": "NSGA-III",
    "EA_MOEAD": "MOEA/D",
    "GD-PSL_NSGAII": "GD-PSL + NSGA-II",
    "GD-PSL_NSGAIII": "GD-PSL + NSGA-III",
    "GD-PSL_MOEAD": "GD-PSL + MOEA/D",
    "PangMOEAD": "Pang",
}


def _run_configured_experiments(arguments: argparse.Namespace) -> None:
    common = replace(
        DEFAULT_CONFIG,
        algorithms=ALGORITHMS,
        runs=1,
        seed=arguments.seed,
        population_size=arguments.population_size,
        max_fe=arguments.max_fe,
        platemo_root=arguments.platemo_root,
        device=arguments.device,
        output_dir=str(arguments.output_root),
        plot=False,
    )
    selected_dtlz = tuple(name for name in DTLZ_PROBLEMS if name in arguments.problems)
    selected_re = tuple(name for name in RE_PROBLEMS if name in arguments.problems)
    for ea_only in (True, False):
        variant = replace(
            common,
            ea_only=ea_only,
            ea_fill_split=arguments.ea_fill_split,
        )
        if selected_dtlz:
            run_experiments(
                replace(variant, problems=selected_dtlz, n_objectives=3),
                skip_completed=True,
            )
        if selected_re:
            run_experiments(
                replace(variant, problems=selected_re, n_objectives=None),
                skip_completed=True,
            )

    run_pang(
        [
            "--algorithms",
            "PangMOEAD",
            "--problems",
            *arguments.problems,
            "--n-objectives",
            "3",
            "--runs",
            "1",
            "--seed",
            str(arguments.seed),
            "--population-size",
            str(arguments.population_size),
            "--max-fe",
            str(arguments.max_fe),
            "--coverage-samples",
            str(arguments.reference_samples),
            "--platemo-root",
            arguments.platemo_root,
            "--output-dir",
            str(arguments.output_root),
            "--skip-completed",
        ]
    )


def _finite_rows(values: np.ndarray, columns: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, columns or 0), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array[np.isfinite(array).all(axis=1)]


def _method_front(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a run and derive fronts from the complete evaluation history.

    ``base_f`` and ``completed_f`` are convenient saved archives, but their
    provenance varies between runners.  The comparison figures must use the
    empirical PF accumulated over every true evaluation, so reconstruct it
    from ``stage1_f``/``evaluations_f`` and then merge true model evaluations.
    """

    def cumulative_front(values: np.ndarray) -> np.ndarray:
        finite = _finite_rows(values)
        if not len(finite):
            return finite
        unique = np.unique(finite, axis=0)
        return unique[nondominated_indices(unique)]

    with np.load(run_dir / "fronts.npz") as data:
        history_key = "evaluations_f" if "evaluations_f" in data else "stage1_f"
        history = _finite_rows(data[history_key])
        ea_archive = cumulative_front(history)

        if "completed_f" in data and "completed_sources" in data:
            completed = _finite_rows(data["completed_f"], history.shape[1])
            source_values = np.asarray(data["completed_sources"], dtype=int).reshape(-1)
            source_values = source_values[: len(data["completed_f"])]
            finite_completed = np.isfinite(np.asarray(data["completed_f"], dtype=float)).all(axis=1)
            model_points = completed[(source_values == 1) & finite_completed]
        else:
            model_points = np.empty((0, history.shape[1] if history.ndim == 2 else 0))

        # EA rows are first so exact duplicates remain classified as EA points.
        combined = np.vstack((ea_archive, model_points))
        combined_sources = np.concatenate(
            (np.zeros(len(ea_archive), dtype=int), np.ones(len(model_points), dtype=int))
        )
        if len(combined):
            _, unique_indices = np.unique(combined, axis=0, return_index=True)
            unique_indices.sort()
            combined = combined[unique_indices]
            combined_sources = combined_sources[unique_indices]
            keep = nondominated_indices(combined)
            values = combined[keep]
            sources = combined_sources[keep]
        else:
            values = combined
            sources = combined_sources
        fill = values[sources == 1]
    return values, history, ea_archive, fill


def _metric_row(
    method: str,
    run_dir: Path,
    front: np.ndarray,
    history: np.ndarray,
    fill: np.ndarray,
) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    hv = summary.get("hypervolume", {})
    igd = summary.get("igd_infinity", {})
    gap = summary.get("gap_info", {})
    runtime = summary.get("runtime_seconds", {})
    ea_seconds = runtime.get(
        "ea_search_and_validation", summary.get("search_time_seconds", "")
    )
    training_seconds = runtime.get("model_training", 0.0)
    comparison_seconds = runtime.get("ea_plus_model_training")
    if comparison_seconds is None:
        comparison_seconds = (
            float(ea_seconds) + float(training_seconds) if ea_seconds != "" else ""
        )
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "problem": str(summary.get("problem", "")).upper(),
        "seed": summary.get("run_seed", summary.get("run", "")),
        "population_size": summary.get("population_size", summary.get("experiment_config", {}).get("population_size", "")),
        "total_fe": summary.get("common_total_fe_budget", gap.get("common_total_fe_budget", "")),
        "ea_fe": gap.get("ea_fe_used", summary.get("fe_used", "")),
        "model_fe": gap.get("model_fe_used", 0),
        "history_size": len(history),
        "fill_evaluations": (
            0 if method == "PangMOEAD" else gap.get("model_fe_used", len(fill))
        ),
        "accepted_model_points": len(fill) if method.startswith("GD-PSL") else 0,
        "reported_subset_size": (
            summary.get("large_solution_set_size", "")
            if method == "PangMOEAD"
            else ""
        ),
        "archive_size": len(front),
        "result_scope": summary.get(
            "result_scope",
            "complete_historical_nondominated_archive"
            if method == "PangMOEAD"
            else (
                "complete_evaluation_history_nondominated_archive_plus_fill"
                if method.startswith("GD-PSL")
                else "complete_evaluation_history_nondominated_archive"
            ),
        ),
        "hypervolume": hv.get("completed_hv", hv.get("value", "")),
        "igd_infinity": igd.get("completed_igd_infinity", ""),
        "igd_reference": igd.get("reference_source", ""),
        "ea_search_seconds": ea_seconds,
        "model_training_seconds": training_seconds,
        "runtime_seconds": comparison_seconds,
        "runtime_definition": "EA search + model training only",
        "run_directory": str(run_dir),
    }


def _axis_limits(fronts: dict[str, np.ndarray]) -> list[tuple[float, float]]:
    all_points = np.vstack(list(fronts.values()))
    limits = []
    for objective in range(all_points.shape[1]):
        lower = min(0.0, float(np.min(all_points[:, objective])))
        upper = max(1.0, float(np.max(all_points[:, objective])))
        padding = max(0.02, 0.035 * (upper - lower))
        limits.append((lower - padding, upper + padding))
    return limits


def _draw_panel(
    axis,
    ea_archive: np.ndarray,
    fill: np.ndarray,
    objective_count: int,
    limits: list[tuple[float, float]],
    metric: dict,
    show_title: bool,
    method: str,
    best_hv: float,
) -> None:
    archive_args = dict(
        s=2.2,
        c="#2166d1",
        alpha=0.50,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    fill_args = dict(
        s=7,
        c="#d62728",
        alpha=0.88,
        edgecolors="white",
        linewidths=0.15,
        rasterized=True,
        zorder=3,
    )
    if objective_count == 3:
        axis.computed_zorder = False
        if len(ea_archive):
            axis.scatter(*ea_archive.T, depthshade=True, **archive_args)
        if len(fill):
            axis.scatter(*fill.T, depthshade=False, **fill_args)
        axis.set_zlim(*limits[2])
        axis.set_zlabel(r"$f_3$", labelpad=-2, fontsize=7)
        axis.view_init(elev=24, azim=42)
        axis.set_box_aspect((1, 1, 0.85))
    else:
        if len(ea_archive):
            axis.scatter(ea_archive[:, 0], ea_archive[:, 1], **archive_args)
        if len(fill):
            axis.scatter(fill[:, 0], fill[:, 1], **fill_args)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_xlabel(r"$f_1$", labelpad=-1, fontsize=7)
    axis.set_ylabel(r"$f_2$", labelpad=-1, fontsize=7)
    axis.tick_params(labelsize=6, pad=0)
    axis.grid(alpha=0.18)
    if show_title:
        axis.set_title(METHOD_LABELS[method], fontsize=10, fontweight="bold", pad=10)

    hv = float(metric["hypervolume"])
    igd = float(metric["igd_infinity"])
    runtime = float(metric["runtime_seconds"])
    text_method = axis.text2D if objective_count == 3 else axis.text
    text_method(
        0.5,
        -0.20,
        rf"HV: {hv:.6f}   $IGD_\infty$: {igd:.6f}   Time: {runtime:.2f}s",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold" if np.isclose(hv, best_hv) else "normal",
        color="#b22222" if np.isclose(hv, best_hv) else "#222222",
        clip_on=False,
    )
    if np.isclose(hv, best_hv):
        text_method(
            0.02,
            0.98,
            "BEST HV",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#b22222",
            fontweight="bold",
        )


def _plot_problem_row(
    normalized_fronts: dict[str, np.ndarray],
    normalized_ea_archives: dict[str, np.ndarray],
    normalized_fills: dict[str, np.ndarray],
    metrics: dict[str, dict],
    problem_label: str,
    output_path: Path,
    seed: int,
) -> None:
    objective_count = next(iter(normalized_fronts.values())).shape[1]
    displayed = {
        method: np.vstack(
            (
                normalized_ea_archives[method],
                normalized_fills[method],
            )
        )
        for method in METHODS
    }
    limits = _axis_limits(displayed)
    best_hv = max(float(metrics[method]["hypervolume"]) for method in METHODS)
    figure = plt.figure(figsize=(28, 5.2), facecolor="white")
    for column, method in enumerate(METHODS):
        projection = "3d" if objective_count == 3 else None
        axis = figure.add_subplot(1, len(METHODS), column + 1, projection=projection)
        _draw_panel(
            axis,
            normalized_ea_archives[method],
            normalized_fills[method],
            objective_count,
            limits,
            metrics[method],
            True,
            method,
            best_hv,
        )
    figure.suptitle(
        f"Final PF comparison: {problem_label} | seed={seed} | "
        "blue=EA nondominated archive, "
        "red=model-generated points retained in the final nondominated archive",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(left=0.025, right=0.99, bottom=0.17, top=0.88, wspace=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_all_problems(
    plot_data: list[dict],
    output_path: Path,
    seed: int,
) -> None:
    figure = plt.figure(figsize=(28, 38), facecolor="white")
    grid = figure.add_gridspec(
        len(plot_data),
        len(METHODS),
        left=0.045,
        right=0.99,
        bottom=0.025,
        top=0.965,
        wspace=0.08,
        hspace=0.34,
    )
    for row_index, problem_data in enumerate(plot_data):
        fronts = problem_data["fronts"]
        objective_count = problem_data["objective_count"]
        displayed = {
            method: np.vstack(
                (
                    problem_data["ea_archives"][method],
                    problem_data["fills"][method],
                )
            )
            for method in METHODS
        }
        limits = _axis_limits(displayed)
        best_hv = max(
            float(problem_data["metrics"][method]["hypervolume"])
            for method in METHODS
        )
        for column, method in enumerate(METHODS):
            projection = "3d" if objective_count == 3 else None
            axis = figure.add_subplot(grid[row_index, column], projection=projection)
            _draw_panel(
                axis,
                problem_data["ea_archives"][method],
                problem_data["fills"][method],
                objective_count,
                limits,
                problem_data["metrics"][method],
                row_index == 0,
                method,
                best_hv,
            )
            if column == 0:
                axis.text2D(
                    -0.30,
                    0.5,
                    problem_data["label"],
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                ) if objective_count == 3 else axis.text(
                    -0.30,
                    0.5,
                    problem_data["label"],
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
    figure.suptitle(
        f"Final PF comparison | rows=problems, columns=methods | seed={seed} | "
        "blue=EA nondominated archive, "
        "red=model-generated points retained in the final nondominated archive",
        fontsize=18,
        fontweight="bold",
        y=0.993,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def generate_comparisons(
    output_root: Path,
    seed: int,
    problems: Sequence[str] = (*DTLZ_PROBLEMS, *RE_PROBLEMS),
) -> list[dict]:
    rows: list[dict] = []
    plot_data: list[dict] = []
    problem_specs = [
        (name, 3 if name in DTLZ_PROBLEMS else get_problem(name).n_obj)
        for name in problems
    ]
    for problem_name, objective_count in problem_specs:
        problem = get_problem(problem_name, n_obj=objective_count) if problem_name.startswith("dtlz") else get_problem(problem_name)
        ideal, nadir, _ = resolve_normalization_points(problem_name, problem, objective_count)
        normalized_fronts: dict[str, np.ndarray] = {}
        normalized_histories: dict[str, np.ndarray] = {}
        normalized_ea_archives: dict[str, np.ndarray] = {}
        normalized_fills: dict[str, np.ndarray] = {}
        problem_metrics: dict[str, dict] = {}
        for method in METHODS:
            run_dir = run_directory(output_root, method, problem_name, objective_count, seed)
            if not (run_dir / "summary.json").exists():
                raise FileNotFoundError(f"Missing completed method run: {run_dir}")
            visualize_run(run_dir, run_dir / "figures")
            front, history, ea_archive, fill = _method_front(run_dir)
            normalized_fronts[method] = normalize_objectives(front, ideal, nadir)
            normalized_histories[method] = normalize_objectives(history, ideal, nadir)
            normalized_ea_archives[method] = normalize_objectives(
                ea_archive, ideal, nadir
            )
            normalized_fills[method] = normalize_objectives(fill, ideal, nadir)
            metric = _metric_row(method, run_dir, front, history, fill)
            problem_metrics[method] = metric

        if problem_name.startswith("re"):
            reference_front, reference_source = load_benchmark_reference_front(
                problem_name, objective_count
            )
            normalized_reference = normalize_objectives(
                reference_front, ideal, nadir
            )
            for method in METHODS:
                value, _ = projected_igd_infinity(
                    normalized_fronts[method], normalized_reference
                )
                problem_metrics[method]["igd_infinity"] = value
                problem_metrics[method]["igd_reference"] = reference_source

        problem_rows = [problem_metrics[method] for method in METHODS]
        rows.extend(problem_rows)

        comparison_dir = (
            output_root
            / "comparisons"
            / problem_directory_name(problem_name, objective_count)
            / f"seed_{seed:03d}"
        )
        _plot_problem_row(
            normalized_fronts,
            normalized_ea_archives,
            normalized_fills,
            problem_metrics,
            problem_directory_name(problem_name, objective_count),
            comparison_dir / "comparison_methods.png",
            seed,
        )
        with (comparison_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(problem_rows[0]))
            writer.writeheader()
            writer.writerows(problem_rows)
        plot_data.append(
            {
                "label": problem_directory_name(problem_name, objective_count),
                "objective_count": objective_count,
                "fronts": normalized_fronts,
                "ea_archives": normalized_ea_archives,
                "fills": normalized_fills,
                "metrics": problem_metrics,
            }
        )

    _plot_all_problems(
        plot_data,
        output_root / "comparisons" / "comparison_all_problems.png",
        seed,
    )

    summary_path = output_root / "comparisons" / "all_metrics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "unavailable_problems.json").unlink(missing_ok=True)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platemo-root", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=91)
    parser.add_argument("--max-fe", type=int, default=100_000)
    parser.add_argument(
        "--ea-fill-split",
        choices=ALLOWED_EA_FILL_SPLITS,
        default="90:10",
    )
    parser.add_argument("--reference-samples", type=int, default=50_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=(*DTLZ_PROBLEMS, *RE_PROBLEMS),
        default=[*DTLZ_PROBLEMS, *RE_PROBLEMS],
        help="Run and compare only the selected packaged problems.",
    )
    parser.add_argument("--skip-experiments", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_args(argv)
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    if not arguments.skip_experiments:
        _run_configured_experiments(arguments)
    rows = generate_comparisons(
        arguments.output_root, arguments.seed, arguments.problems
    )
    print(
        f"Completed {len(rows)} method/problem comparisons under {arguments.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
