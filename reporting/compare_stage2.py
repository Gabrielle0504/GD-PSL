"""Aggregate and visualize the registered Stage 2 seven-method comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from gd_psl.archive import nondominated_indices
from gd_psl.artifacts import problem_directory_name, run_directory
from gd_psl.metrics import normalize_objectives, resolve_normalization_points
from gd_psl.problems import get_problem


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
PROBLEMS = (
    "dtlz2",
    "dtlz7",
    "re21",
    "re24",
    "re31",
    "re32",
    "re34",
    "re35",
    "re37",
)
DEFAULT_SEEDS = tuple(range(101, 121))


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
    """Reconstruct the cumulative empirical front from saved evaluations."""

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
            raw_completed = np.asarray(data["completed_f"], dtype=float)
            source_values = np.asarray(data["completed_sources"], dtype=int).reshape(-1)
            source_values = source_values[: len(raw_completed)]
            finite_completed = np.isfinite(raw_completed).all(axis=1)
            model_points = raw_completed[(source_values == 1) & finite_completed]
        else:
            model_points = np.empty((0, history.shape[1]))

    combined = np.vstack((ea_archive, model_points))
    sources = np.concatenate(
        (np.zeros(len(ea_archive), dtype=int), np.ones(len(model_points), dtype=int))
    )
    if len(combined):
        _, unique = np.unique(combined, axis=0, return_index=True)
        unique.sort()
        combined, sources = combined[unique], sources[unique]
        front = nondominated_indices(combined)
        combined, sources = combined[front], sources[front]
    return combined, history, ea_archive, combined[sources == 1]


@dataclass(frozen=True)
class RunRecord:
    method: str
    problem: str
    objective_count: int
    seed: int
    run_dir: Path
    hv: float
    igd: float
    comparison_runtime: float
    method_runtime: float
    ea_runtime: float
    training_runtime: float
    fe_used: int
    population_size: int
    archive_size: int
    hv_signature: str
    igd_reference: str
    training_converged: Optional[bool]
    epochs_trained: Optional[int]
    best_validation_loss: Optional[float]
    training_stop_reason: str


@dataclass
class MethodPanel:
    method: str
    problem: str
    representative_seed: int
    runs: int
    median_hv: float
    q1_hv: float
    q3_hv: float
    median_igd: float
    q1_igd: float
    q3_igd: float
    median_runtime: float
    q1_runtime: float
    q3_runtime: float
    ea_archive: np.ndarray
    fill_archive: np.ndarray


def _problem_objectives(problem: str) -> int:
    return 3 if problem.startswith("dtlz") else get_problem(problem).n_obj


def _finite_float(value: object, label: str, path: Path) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Nonfinite {label} in {path}")
    return number


def _load_run(run_dir: Path, method: str, problem: str, objective_count: int) -> RunRecord:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hv = summary.get("hypervolume", {})
    igd = summary.get("igd_infinity", {})
    gap = summary.get("gap_info", {})
    runtime = summary.get("runtime_seconds", {})
    completed_hv = hv.get("completed_hv", hv.get("value"))
    completed_igd = igd.get("completed_igd_infinity")
    if completed_hv is None or completed_igd is None:
        raise ValueError(f"Missing HV or IGD-infinity in {summary_path}")

    if method == "PangMOEAD":
        seed = int(summary["run"])
        ea_seconds = _finite_float(summary["search_time_seconds"], "EA time", summary_path)
        training_seconds = 0.0
        comparison_seconds = ea_seconds
        method_seconds = ea_seconds
        fe_used = int(summary["fe_used"])
        population_size = int(summary["population_size"])
        archive_size = int(summary["nondominated_archive_size"])
        training_converged = None
        epochs_trained = None
        best_validation_loss = None
        training_stop_reason = "not_applicable"
    else:
        seed = int(summary["run_seed"])
        ea_seconds = _finite_float(
            runtime["ea_search_and_validation"], "EA time", summary_path
        )
        training_seconds = _finite_float(
            runtime.get("model_training", 0.0), "training time", summary_path
        )
        comparison_seconds = _finite_float(
            runtime.get("ea_plus_model_training", ea_seconds + training_seconds),
            "comparison time",
            summary_path,
        )
        method_seconds = _finite_float(
            runtime.get("method_execution_total", runtime.get("total", comparison_seconds)),
            "method execution time",
            summary_path,
        )
        fe_used = int(gap["ea_fe_used"]) + int(gap.get("model_fe_used", 0))
        population_size = int(summary["experiment_config"]["population_size"])
        archive_size = int(summary["completed_nondominated"])
        if method.startswith("GD-PSL"):
            training = summary["training"]
            training_converged = bool(training["converged"])
            epochs_trained = int(training["epochs_trained"])
            best_validation_loss = _finite_float(
                training["best_validation_loss"],
                "best validation loss",
                summary_path,
            )
            training_stop_reason = str(training["stop_reason"])
        else:
            training_converged = None
            epochs_trained = None
            best_validation_loss = None
            training_stop_reason = "not_applicable"

    signature_payload = {
        "ideal": hv.get("normalization_ideal"),
        "nadir": hv.get("normalization_nadir"),
        "reference": hv.get("reference_point"),
    }
    return RunRecord(
        method=method,
        problem=problem_directory_name(problem, objective_count),
        objective_count=objective_count,
        seed=seed,
        run_dir=run_dir,
        hv=_finite_float(completed_hv, "HV", summary_path),
        igd=_finite_float(completed_igd, "IGD-infinity", summary_path),
        comparison_runtime=comparison_seconds,
        method_runtime=method_seconds,
        ea_runtime=ea_seconds,
        training_runtime=training_seconds,
        fe_used=fe_used,
        population_size=population_size,
        archive_size=archive_size,
        hv_signature=json.dumps(signature_payload, sort_keys=True),
        igd_reference=str(igd.get("reference_source", "")),
        training_converged=training_converged,
        epochs_trained=epochs_trained,
        best_validation_loss=best_validation_loss,
        training_stop_reason=training_stop_reason,
    )


def load_records(
    results_root: Path,
    methods: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for method in methods:
        for problem in problems:
            objective_count = _problem_objectives(problem)
            for seed in seeds:
                directory = run_directory(
                    results_root, method, problem, objective_count, seed
                )
                if not (directory / "summary.json").is_file():
                    raise FileNotFoundError(f"Missing completed Stage 2 run: {directory}")
                record = _load_run(directory, method, problem, objective_count)
                if record.seed != seed:
                    raise ValueError(
                        f"Seed mismatch in {directory}: {record.seed} != {seed}"
                    )
                records.append(record)
    return records


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.25, 0.5, 0.75]))


def summarize_records(records: Sequence[RunRecord]) -> list[dict]:
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        groups[(record.problem, record.method)].append(record)
    rows = []
    for (problem, method), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item.seed)
        q1_hv, median_hv, q3_hv = _quantiles([item.hv for item in ordered])
        q1_igd, median_igd, q3_igd = _quantiles([item.igd for item in ordered])
        q1_time, median_time, q3_time = _quantiles(
            [item.comparison_runtime for item in ordered]
        )
        representative = min(
            ordered, key=lambda item: (abs(item.igd - median_igd), item.seed)
        )
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "problem": problem,
                "runs": len(ordered),
                "representative_seed": representative.seed,
                "representative_rule": "minimum absolute distance to median IGD-infinity",
                "median_hv": median_hv,
                "q1_hv": q1_hv,
                "q3_hv": q3_hv,
                "median_igd_infinity": median_igd,
                "q1_igd_infinity": q1_igd,
                "q3_igd_infinity": q3_igd,
                "median_runtime_seconds": median_time,
                "q1_runtime_seconds": q1_time,
                "q3_runtime_seconds": q3_time,
                "median_method_execution_seconds": float(
                    np.median([item.method_runtime for item in ordered])
                ),
                "median_ea_seconds": float(
                    np.median([item.ea_runtime for item in ordered])
                ),
                "median_model_training_seconds": float(
                    np.median([item.training_runtime for item in ordered])
                ),
                "fe_used": ",".join(
                    str(value) for value in sorted({item.fe_used for item in ordered})
                ),
                "population_size": ",".join(
                    str(value)
                    for value in sorted({item.population_size for item in ordered})
                ),
                "median_archive_size": float(
                    np.median([item.archive_size for item in ordered])
                ),
                "igd_reference": ordered[0].igd_reference,
                "training_converged_runs": (
                    sum(item.training_converged is True for item in ordered)
                    if method.startswith("GD-PSL")
                    else ""
                ),
                "median_epochs_trained": (
                    float(np.median([item.epochs_trained for item in ordered]))
                    if method.startswith("GD-PSL")
                    else ""
                ),
                "median_best_validation_loss": (
                    float(
                        np.median(
                            [item.best_validation_loss for item in ordered]
                        )
                    )
                    if method.startswith("GD-PSL")
                    else ""
                ),
                "training_stop_reasons": (
                    ";".join(
                        f"{reason}:{sum(item.training_stop_reason == reason for item in ordered)}"
                        for reason in sorted(
                            {item.training_stop_reason for item in ordered}
                        )
                    )
                    if method.startswith("GD-PSL")
                    else "not_applicable"
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_rows(records: Sequence[RunRecord]) -> list[dict]:
    return [
        {
            "method": item.method,
            "method_label": METHOD_LABELS[item.method],
            "problem": item.problem,
            "seed": item.seed,
            "hypervolume": item.hv,
            "igd_infinity": item.igd,
            "comparison_runtime_seconds": item.comparison_runtime,
            "method_execution_seconds": item.method_runtime,
            "ea_seconds": item.ea_runtime,
            "model_training_seconds": item.training_runtime,
            "fe_used": item.fe_used,
            "population_size": item.population_size,
            "archive_size": item.archive_size,
            "hv_signature": item.hv_signature,
            "igd_reference": item.igd_reference,
            "training_converged": item.training_converged,
            "epochs_trained": item.epochs_trained,
            "best_validation_loss": item.best_validation_loss,
            "training_stop_reason": item.training_stop_reason,
            "run_directory": str(item.run_dir),
        }
        for item in records
    ]


def rank_methods(summary_rows: Sequence[dict], methods: Sequence[str]) -> tuple[list[dict], list[dict]]:
    lookup = {(row["problem"], row["method"]): row for row in summary_rows}
    problems = sorted({row["problem"] for row in summary_rows})
    ranks = {method: {"hv": [], "igd": [], "runtime": []} for method in methods}
    metric_values = {"hv": [], "igd": [], "runtime": []}
    for problem in problems:
        hv = np.asarray([float(lookup[(problem, method)]["median_hv"]) for method in methods])
        igd = np.asarray(
            [float(lookup[(problem, method)]["median_igd_infinity"]) for method in methods]
        )
        runtime = np.asarray(
            [float(lookup[(problem, method)]["median_runtime_seconds"]) for method in methods]
        )
        metric_values["hv"].append(hv)
        metric_values["igd"].append(igd)
        metric_values["runtime"].append(runtime)
        for method, hv_rank, igd_rank, time_rank in zip(
            methods,
            rankdata(-hv, method="average"),
            rankdata(igd, method="average"),
            rankdata(runtime, method="average"),
        ):
            ranks[method]["hv"].append(float(hv_rank))
            ranks[method]["igd"].append(float(igd_rank))
            ranks[method]["runtime"].append(float(time_rank))
    rank_rows = [
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "ranked_problems": len(problems),
            "mean_hv_rank": float(np.mean(ranks[method]["hv"])),
            "mean_igd_infinity_rank": float(np.mean(ranks[method]["igd"])),
            "mean_runtime_rank": float(np.mean(ranks[method]["runtime"])),
        }
        for method in methods
    ]
    friedman_rows = []
    for metric, blocks in metric_values.items():
        matrix = np.asarray(blocks)
        result = friedmanchisquare(*[matrix[:, index] for index in range(len(methods))])
        friedman_rows.append(
            {
                "metric": metric,
                "problems": len(problems),
                "methods": len(methods),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return rank_rows, friedman_rows


def _holm_adjust(values: Sequence[float]) -> list[float]:
    adjusted = np.ones(len(values), dtype=float)
    running_maximum = 0.0
    for rank, index in enumerate(np.argsort(values)):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted.tolist()


def paired_tests(
    records: Sequence[RunRecord], methods: Sequence[str], alpha: float
) -> list[dict]:
    grouped: dict[tuple[str, str], dict[int, RunRecord]] = defaultdict(dict)
    for record in records:
        grouped[(record.problem, record.method)][record.seed] = record
    rows = []
    problems = sorted({record.problem for record in records})
    metrics = (
        ("hypervolume", "hv", True),
        ("igd_infinity", "igd", False),
        ("runtime", "comparison_runtime", False),
    )
    for problem in problems:
        for metric_name, attribute, higher_is_better in metrics:
            pending = []
            for left_index, left_method in enumerate(methods[:-1]):
                for right_method in methods[left_index + 1 :]:
                    left = grouped[(problem, left_method)]
                    right = grouped[(problem, right_method)]
                    seeds = sorted(set(left) & set(right))
                    left_values = np.asarray([getattr(left[seed], attribute) for seed in seeds])
                    right_values = np.asarray([getattr(right[seed], attribute) for seed in seeds])
                    differences = left_values - right_values
                    if np.all(np.abs(differences) <= 1e-15):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
                    direction = differences if higher_is_better else -differences
                    pending.append(
                        {
                            "problem": problem,
                            "metric": metric_name,
                            "left_method": left_method,
                            "right_method": right_method,
                            "paired_runs": len(seeds),
                            "left_median": float(np.median(left_values)),
                            "right_median": float(np.median(right_values)),
                            "left_wins": int(np.count_nonzero(direction > 1e-12)),
                            "ties": int(np.count_nonzero(np.abs(direction) <= 1e-12)),
                            "left_losses": int(np.count_nonzero(direction < -1e-12)),
                            "wilcoxon_p": p_value,
                            "left_is_better": bool(np.median(direction) > 0.0),
                        }
                    )
            for row, adjusted in zip(
                pending, _holm_adjust([item["wilcoxon_p"] for item in pending])
            ):
                row["holm_adjusted_p"] = adjusted
                if adjusted >= alpha:
                    row["conclusion"] = "no_significant_difference"
                elif row["left_is_better"]:
                    row["conclusion"] = "left_significantly_better"
                else:
                    row["conclusion"] = "right_significantly_better"
            rows.extend(pending)
    return rows


def _axis_text(axis, objective_count: int, *args, **kwargs):
    return (axis.text2D if objective_count == 3 else axis.text)(*args, **kwargs)


def _axis_limits(panels: Sequence[MethodPanel]) -> list[tuple[float, float]]:
    points = np.vstack(
        [
            np.vstack((panel.ea_archive, panel.fill_archive))
            for panel in panels
            if len(panel.ea_archive) + len(panel.fill_archive)
        ]
    )
    limits = []
    for objective in range(points.shape[1]):
        lower = min(0.0, float(np.min(points[:, objective])))
        upper = max(1.0, float(np.max(points[:, objective])))
        padding = max(0.02, 0.035 * (upper - lower))
        limits.append((lower - padding, upper + padding))
    return limits


def _draw_panel(axis, panel: MethodPanel, limits, show_title: bool, best_metrics) -> None:
    objective_count = panel.ea_archive.shape[1]
    archive_args = dict(
        s=2.2, c="#2166d1", alpha=0.50, linewidths=0, rasterized=True, zorder=1
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
        axis.scatter(*panel.ea_archive.T, depthshade=True, **archive_args)
        if len(panel.fill_archive):
            axis.scatter(*panel.fill_archive.T, depthshade=False, **fill_args)
        axis.set_zlim(*limits[2])
        axis.set_zlabel(r"$f_3$", labelpad=-2, fontsize=7)
        axis.view_init(elev=24, azim=42)
        axis.set_box_aspect((1, 1, 0.85))
    else:
        axis.scatter(panel.ea_archive[:, 0], panel.ea_archive[:, 1], **archive_args)
        if len(panel.fill_archive):
            axis.scatter(panel.fill_archive[:, 0], panel.fill_archive[:, 1], **fill_args)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_xlabel(r"$f_1$", labelpad=-1, fontsize=7)
    axis.set_ylabel(r"$f_2$", labelpad=-1, fontsize=7)
    axis.tick_params(labelsize=6, pad=0)
    axis.grid(alpha=0.18)
    if show_title:
        axis.set_title(METHOD_LABELS[panel.method], fontsize=9, fontweight="bold", pad=10)
    values = (
        (0.17, f"Med HV\n{panel.median_hv:.6f}", panel.median_hv, best_metrics[0]),
        (0.50, rf"Med $IGD_\infty$" + f"\n{panel.median_igd:.6f}", panel.median_igd, best_metrics[1]),
        (0.83, f"Med time\n{panel.median_runtime:.2f}s", panel.median_runtime, best_metrics[2]),
    )
    for x_position, label, value, best in values:
        is_best = np.isclose(value, best)
        _axis_text(
            axis,
            objective_count,
            x_position,
            -0.20,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.6,
            fontweight="bold" if is_best else "normal",
            color="#b22222" if is_best else "#222222",
            clip_on=False,
        )
    _axis_text(
        axis,
        objective_count,
        0.98,
        0.98,
        f"seed {panel.representative_seed}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#444444",
    )


def _save_figure(figure, stem: Path, formats: Sequence[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(stem.with_suffix(f".{extension}"), **kwargs)
    plt.close(figure)


def _build_panels(
    results_root: Path,
    summary_rows: Sequence[dict],
    methods: Sequence[str],
    problems: Sequence[str],
) -> list[list[MethodPanel]]:
    lookup = {(row["problem"], row["method"]): row for row in summary_rows}
    output = []
    for problem in problems:
        objective_count = _problem_objectives(problem)
        problem_label = problem_directory_name(problem, objective_count)
        definition = (
            get_problem(problem, n_obj=objective_count)
            if problem.startswith("dtlz")
            else get_problem(problem)
        )
        ideal, nadir, _ = resolve_normalization_points(
            problem, definition, objective_count
        )
        panels = []
        for method in methods:
            row = lookup[(problem_label, method)]
            seed = int(row["representative_seed"])
            directory = run_directory(
                results_root, method, problem, objective_count, seed
            )
            _, _, ea_archive, fill_archive = _method_front(directory)
            panels.append(
                MethodPanel(
                    method=method,
                    problem=problem_label,
                    representative_seed=seed,
                    runs=int(row["runs"]),
                    median_hv=float(row["median_hv"]),
                    q1_hv=float(row["q1_hv"]),
                    q3_hv=float(row["q3_hv"]),
                    median_igd=float(row["median_igd_infinity"]),
                    q1_igd=float(row["q1_igd_infinity"]),
                    q3_igd=float(row["q3_igd_infinity"]),
                    median_runtime=float(row["median_runtime_seconds"]),
                    q1_runtime=float(row["q1_runtime_seconds"]),
                    q3_runtime=float(row["q3_runtime_seconds"]),
                    ea_archive=normalize_objectives(ea_archive, ideal, nadir),
                    fill_archive=normalize_objectives(fill_archive, ideal, nadir),
                )
            )
        output.append(panels)
    return output


def plot_comparisons(
    problem_panels: Sequence[Sequence[MethodPanel]],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    figure = plt.figure(figsize=(28, 38), facecolor="white")
    grid = figure.add_gridspec(
        len(problem_panels),
        len(METHODS),
        left=0.045,
        right=0.99,
        bottom=0.025,
        top=0.968,
        wspace=0.08,
        hspace=0.48,
    )
    for row_index, panels in enumerate(problem_panels):
        limits = _axis_limits(panels)
        best_metrics = (
            max(panel.median_hv for panel in panels),
            min(panel.median_igd for panel in panels),
            min(panel.median_runtime for panel in panels),
        )
        for column, panel in enumerate(panels):
            objective_count = panel.ea_archive.shape[1]
            projection = "3d" if objective_count == 3 else None
            axis = figure.add_subplot(grid[row_index, column], projection=projection)
            _draw_panel(axis, panel, limits, row_index == 0, best_metrics)
            if column == 0:
                _axis_text(
                    axis,
                    objective_count,
                    -0.30,
                    0.5,
                    panel.problem,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
    figure.suptitle(
        "Stage 2 formal comparison | rows=problems, columns=methods | n=20",
        fontsize=18,
        fontweight="bold",
        y=0.992,
    )
    figure.text(
        0.5,
        0.008,
        "Blue: EA/Pang nondominated archive; red: true-evaluated GD-PSL model points retained in the final archive. "
        "Representative run is nearest the method's median IGD-infinity.",
        ha="center",
        fontsize=8,
    )
    _save_figure(figure, output_dir / "comparison_all_problems", formats, dpi=140)

    for panels in problem_panels:
        limits = _axis_limits(panels)
        best_metrics = (
            max(panel.median_hv for panel in panels),
            min(panel.median_igd for panel in panels),
            min(panel.median_runtime for panel in panels),
        )
        figure = plt.figure(figsize=(28, 5.2), facecolor="white")
        for column, panel in enumerate(panels):
            objective_count = panel.ea_archive.shape[1]
            projection = "3d" if objective_count == 3 else None
            axis = figure.add_subplot(1, len(panels), column + 1, projection=projection)
            _draw_panel(axis, panel, limits, True, best_metrics)
        figure.suptitle(
            f"Stage 2 formal comparison: {panels[0].problem} | n=20",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )
        figure.subplots_adjust(left=0.025, right=0.99, bottom=0.23, top=0.88, wspace=0.10)
        _save_figure(
            figure,
            output_dir / panels[0].problem / "comparison_methods",
            formats,
            dpi=180,
        )


def generate_report(
    results_root: Path,
    output_dir: Path,
    methods: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
    alpha: float,
    formats: Sequence[str],
) -> None:
    records = load_records(results_root, methods, problems, seeds)
    expected = len(methods) * len(problems) * len(seeds)
    if len(records) != expected:
        raise ValueError(f"Expected {expected} runs, loaded {len(records)}")
    for problem in {item.problem for item in records}:
        signatures = {item.hv_signature for item in records if item.problem == problem}
        if len(signatures) != 1:
            raise ValueError(f"HV normalization differs across methods for {problem}")
    summary_rows = summarize_records(records)
    rank_rows, friedman_rows = rank_methods(summary_rows, methods)
    test_rows = paired_tests(records, methods, alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "formal_runs.csv", _run_rows(records))
    _write_csv(output_dir / "formal_summary.csv", summary_rows)
    _write_csv(output_dir / "mean_ranks.csv", rank_rows)
    _write_csv(output_dir / "friedman_tests.csv", friedman_rows)
    _write_csv(output_dir / "paired_tests.csv", test_rows)
    panels = _build_panels(results_root, summary_rows, methods, problems)
    plot_comparisons(panels, output_dir, formats)
    print(f"Generated Stage 2 report from {len(records)} runs under {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/stage2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--problems", nargs="+", choices=PROBLEMS, default=list(PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_args(argv)
    if not 0.0 < arguments.alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    output_dir = arguments.output_dir or arguments.results_root / "comparisons"
    generate_report(
        arguments.results_root,
        output_dir,
        arguments.methods,
        arguments.problems,
        arguments.seeds,
        arguments.alpha,
        arguments.formats,
    )


if __name__ == "__main__":
    main()
