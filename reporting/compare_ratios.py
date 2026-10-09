"""Compare the five registered EA/FILL ratios from Stage 1 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from evaluation_metrics import (
    load_benchmark_reference_front,
    normalize_objectives,
    projected_igd_infinity,
)
from experiment_config import ALLOWED_EA_FILL_SPLITS


RATIOS = tuple(ALLOWED_EA_FILL_SPLITS)
ALGORITHMS = ("NSGAII", "NSGAIII", "MOEAD")
ALGORITHM_LABELS = {
    "NSGAII": "GD-PSL + NSGA-II",
    "NSGAIII": "GD-PSL + NSGA-III",
    "MOEAD": "GD-PSL + MOEA/D",
}
PROBLEMS = (
    "DTLZ2_3obj",
    "DTLZ7_3obj",
    "RE21_2obj",
    "RE24_2obj",
    "RE31_3obj",
    "RE32_3obj",
    "RE34_3obj",
    "RE35_3obj",
    "RE37_3obj",
)


@dataclass(frozen=True)
class RunRecord:
    seed: int
    run_dir: Path
    hv: float
    igd_infinity: float
    runtime_seconds: float
    ideal: np.ndarray
    nadir: np.ndarray
    igd_reference: str


@dataclass(frozen=True)
class RatioPanel:
    ratio: str
    algorithm: str
    problem: str
    representative_seed: int
    ea_archive: np.ndarray
    fill_archive: np.ndarray
    median_hv: float
    q1_hv: float
    q3_hv: float
    median_igd_infinity: float
    q1_igd_infinity: float
    q3_igd_infinity: float
    median_runtime_seconds: float
    q1_runtime_seconds: float
    q3_runtime_seconds: float
    runs: int
    igd_reference: str


def _ratio_directory(ratio: str) -> str:
    return f"ratio_{ratio.replace(':', '_')}"


def _method_directory(algorithm: str) -> str:
    return f"GD-PSL_{algorithm}"


def _finite_rows(values: np.ndarray, objective_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, objective_count), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != objective_count:
        raise ValueError(
            f"Expected an objective matrix with {objective_count} columns, got {array.shape}"
        )
    return array[np.isfinite(array).all(axis=1)]


def _runtime_seconds(summary: dict) -> float:
    runtime = summary.get("runtime_seconds", {})
    for key in ("method_execution_total", "algorithm_execution_total", "total"):
        value = runtime.get(key)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    raise ValueError("summary.json does not contain a finite algorithm runtime")


def _summary_and_bounds(run_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hv = summary.get("hypervolume", {})
    ideal = np.asarray(hv["normalization_ideal"], dtype=float)
    nadir = np.asarray(hv["normalization_nadir"], dtype=float)
    if ideal.shape != nadir.shape or ideal.ndim != 1 or np.any(nadir <= ideal):
        raise ValueError(f"Invalid normalization bounds in {summary_path}")
    return summary, ideal, nadir


def _run_record(run_dir: Path, igd_override: Optional[float] = None) -> RunRecord:
    summary, ideal, nadir = _summary_and_bounds(run_dir)
    hv = summary.get("hypervolume", {})
    igd = summary.get("igd_infinity", {})
    hv_value = float(hv["completed_hv"])
    if igd_override is None:
        igd_value = float(igd["completed_igd_infinity"])
        igd_reference = str(igd.get("reference_source", "saved shared reference"))
    else:
        igd_value = float(igd_override)
        igd_reference = (
            f"benchmark_file:data/RE/ParetoFront/{run_dir.parent.name.split('_')[0]}.dat"
        )
    seed = int(summary.get("run_seed", run_dir.name.removeprefix("seed_")))
    if not np.isfinite([hv_value, igd_value]).all():
        raise ValueError(f"Nonfinite metric in {run_dir / 'summary.json'}")
    return RunRecord(
        seed=seed,
        run_dir=run_dir,
        hv=hv_value,
        igd_infinity=igd_value,
        runtime_seconds=_runtime_seconds(summary),
        ideal=ideal,
        nadir=nadir,
        igd_reference=igd_reference,
    )


def _load_front(record: RunRecord) -> tuple[np.ndarray, np.ndarray]:
    artifact_path = record.run_dir / "fronts.npz"
    with np.load(artifact_path) as data:
        if "base_f" not in data or "completed_f" not in data:
            raise KeyError(f"Missing base_f/completed_f in {artifact_path}")
        objective_count = len(record.ideal)
        ea_archive = _finite_rows(data["base_f"], objective_count)
        raw_completed = np.asarray(data["completed_f"], dtype=float)
        if raw_completed.ndim == 1:
            raw_completed = raw_completed.reshape(1, -1)
        completed = _finite_rows(raw_completed, objective_count)
        sources = np.asarray(data["completed_sources"], dtype=int).reshape(-1)
        raw_finite = np.isfinite(raw_completed).all(axis=1)
        if len(sources) != len(raw_completed):
            raise ValueError(f"completed_sources length mismatch in {artifact_path}")
        fill_archive = completed[sources[raw_finite] == 1]
    scale = record.nadir - record.ideal
    return (ea_archive - record.ideal) / scale, (fill_archive - record.ideal) / scale


def _normalized_completed_front(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, ideal, nadir = _summary_and_bounds(run_dir)
    with np.load(run_dir / "fronts.npz") as data:
        completed = _finite_rows(data["completed_f"], len(ideal))
    return (completed - ideal) / (nadir - ideal), ideal, nadir


def _build_reference_igd_overrides(
    stage1_root: Path,
    algorithms: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, str, str, int], float]:
    overrides: dict[tuple[str, str, str, int], float] = {}
    for problem in problems:
        first_dir = (
            stage1_root
            / _ratio_directory(RATIOS[0])
            / _method_directory(algorithms[0])
            / problem
            / f"seed_{seeds[0]:03d}"
        )
        first_summary, _, _ = _summary_and_bounds(first_dir)
        if first_summary.get("igd_infinity", {}).get("completed_igd_infinity") is not None:
            continue

        problem_name = problem.split("_", 1)[0].lower()
        _, reference_ideal, reference_nadir = _summary_and_bounds(first_dir)
        raw_reference, source = load_benchmark_reference_front(
            problem_name, len(reference_ideal)
        )
        reference = normalize_objectives(
            raw_reference, reference_ideal, reference_nadir
        )
        print(
            f"Computing IGD-infinity against {len(reference)} points from {source} "
            f"({problem})...",
            flush=True,
        )
        for ratio in RATIOS:
            for algorithm in algorithms:
                for seed in seeds:
                    run_dir = (
                        stage1_root
                        / _ratio_directory(ratio)
                        / _method_directory(algorithm)
                        / problem
                        / f"seed_{seed:03d}"
                    )
                    if not (run_dir / "fronts.npz").is_file():
                        raise FileNotFoundError(f"Missing completed Stage 1 run: {run_dir}")
                    archive, ideal, nadir = _normalized_completed_front(run_dir)
                    if not np.array_equal(ideal, reference_ideal) or not np.array_equal(
                        nadir, reference_nadir
                    ):
                        raise ValueError(
                            f"Normalization bounds differ across Stage 1 runs for {problem}"
                        )
                    value, _ = projected_igd_infinity(archive, reference)
                    overrides[(ratio, algorithm, problem, seed)] = float(value)
    return overrides


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def _load_panel(
    stage1_root: Path,
    ratio: str,
    algorithm: str,
    problem: str,
    expected_seeds: Sequence[int],
    igd_overrides: dict[tuple[str, str, str, int], float],
) -> RatioPanel:
    problem_dir = (
        stage1_root / _ratio_directory(ratio) / _method_directory(algorithm) / problem
    )
    records = []
    for seed in expected_seeds:
        run_dir = problem_dir / f"seed_{seed:03d}"
        if not (run_dir / "summary.json").is_file() or not (run_dir / "fronts.npz").is_file():
            raise FileNotFoundError(f"Missing completed Stage 1 run: {run_dir}")
        records.append(
            _run_record(run_dir, igd_overrides.get((ratio, algorithm, problem, seed)))
        )

    reference_ideal = records[0].ideal
    reference_nadir = records[0].nadir
    for record in records[1:]:
        if not np.array_equal(record.ideal, reference_ideal) or not np.array_equal(
            record.nadir, reference_nadir
        ):
            raise ValueError(
                f"Normalization bounds differ across seeds for {ratio}/{algorithm}/{problem}"
            )

    q1_hv, median_hv, q3_hv = _quantiles([record.hv for record in records])
    q1_igd, median_igd, q3_igd = _quantiles(
        [record.igd_infinity for record in records]
    )
    q1_time, median_time, q3_time = _quantiles(
        [record.runtime_seconds for record in records]
    )
    representative = min(
        records,
        key=lambda record: (abs(record.igd_infinity - median_igd), record.seed),
    )
    ea_archive, fill_archive = _load_front(representative)
    return RatioPanel(
        ratio=ratio,
        algorithm=algorithm,
        problem=problem,
        representative_seed=representative.seed,
        ea_archive=ea_archive,
        fill_archive=fill_archive,
        median_hv=median_hv,
        q1_hv=q1_hv,
        q3_hv=q3_hv,
        median_igd_infinity=median_igd,
        q1_igd_infinity=q1_igd,
        q3_igd_infinity=q3_igd,
        median_runtime_seconds=median_time,
        q1_runtime_seconds=q1_time,
        q3_runtime_seconds=q3_time,
        runs=len(records),
        igd_reference=representative.igd_reference,
    )


def _axis_limits(panels: Sequence[RatioPanel]) -> list[tuple[float, float]]:
    displayed = [
        values
        for panel in panels
        for values in (panel.ea_archive, panel.fill_archive)
        if len(values)
    ]
    if not displayed:
        raise ValueError(f"No points available for {panels[0].algorithm}/{panels[0].problem}")
    all_points = np.vstack(displayed)
    limits = []
    for objective in range(all_points.shape[1]):
        lower = min(0.0, float(np.min(all_points[:, objective])))
        upper = max(1.0, float(np.max(all_points[:, objective])))
        padding = max(0.02, 0.035 * (upper - lower))
        limits.append((lower - padding, upper + padding))
    return limits


def _axis_text(axis, objective_count: int, *args, **kwargs):
    method = axis.text2D if objective_count == 3 else axis.text
    return method(*args, **kwargs)


def _draw_panel(
    axis,
    panel: RatioPanel,
    limits: list[tuple[float, float]],
    show_title: bool,
    best_hv: float,
    best_igd: float,
    best_time: float,
) -> None:
    objective_count = panel.ea_archive.shape[1]
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
        axis.set_title(f"EA:FILL {panel.ratio}", fontsize=10, fontweight="bold", pad=10)

    metrics = (
        (0.17, f"Med HV\n{panel.median_hv:.6f}", np.isclose(panel.median_hv, best_hv)),
        (
            0.50,
            rf"Med $IGD_\infty$" + f"\n{panel.median_igd_infinity:.6f}",
            np.isclose(panel.median_igd_infinity, best_igd),
        ),
        (
            0.83,
            f"Med time\n{panel.median_runtime_seconds:.2f}s",
            np.isclose(panel.median_runtime_seconds, best_time),
        ),
    )
    for x_position, label, is_best in metrics:
        _axis_text(
            axis,
            objective_count,
            x_position,
            -0.20,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
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
        fontsize=6.5,
        color="#444444",
    )


def _save_figure(figure, stem: Path, formats: Sequence[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(stem.with_suffix(f".{extension}"), **kwargs)
    plt.close(figure)


def _plot_problem(
    panels: Sequence[RatioPanel], output_stem: Path, formats: Sequence[str]
) -> None:
    objective_count = panels[0].ea_archive.shape[1]
    limits = _axis_limits(panels)
    best_hv = max(panel.median_hv for panel in panels)
    best_igd = min(panel.median_igd_infinity for panel in panels)
    best_time = min(panel.median_runtime_seconds for panel in panels)
    figure = plt.figure(figsize=(21, 5.1), facecolor="white")
    for column, panel in enumerate(panels):
        projection = "3d" if objective_count == 3 else None
        axis = figure.add_subplot(1, len(panels), column + 1, projection=projection)
        _draw_panel(axis, panel, limits, True, best_hv, best_igd, best_time)
    figure.suptitle(
        f"Stage 1 ratio comparison: {panels[0].problem} | "
        f"{ALGORITHM_LABELS[panels[0].algorithm]} | n={panels[0].runs}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.015,
        "Blue: EA nondominated archive; red: true-evaluated model points retained "
        "in the final nondominated archive. Representative run is nearest median IGD-infinity.",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.035, right=0.99, bottom=0.23, top=0.88, wspace=0.13)
    _save_figure(figure, output_stem, formats, dpi=180)


def _plot_all_problems(
    algorithm: str,
    problem_panels: Sequence[Sequence[RatioPanel]],
    output_stem: Path,
    formats: Sequence[str],
) -> None:
    figure = plt.figure(figsize=(21, 38), facecolor="white")
    grid = figure.add_gridspec(
        len(problem_panels),
        len(RATIOS),
        left=0.055,
        right=0.99,
        bottom=0.025,
        top=0.968,
        wspace=0.10,
        hspace=0.48,
    )
    for row_index, panels in enumerate(problem_panels):
        objective_count = panels[0].ea_archive.shape[1]
        limits = _axis_limits(panels)
        best_hv = max(panel.median_hv for panel in panels)
        best_igd = min(panel.median_igd_infinity for panel in panels)
        best_time = min(panel.median_runtime_seconds for panel in panels)
        for column, panel in enumerate(panels):
            projection = "3d" if objective_count == 3 else None
            axis = figure.add_subplot(grid[row_index, column], projection=projection)
            _draw_panel(
                axis,
                panel,
                limits,
                row_index == 0,
                best_hv,
                best_igd,
                best_time,
            )
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
        f"Stage 1 EA/FILL ratio comparison | {ALGORITHM_LABELS[algorithm]} | "
        "rows=problems, columns=ratios",
        fontsize=18,
        fontweight="bold",
        y=0.992,
    )
    _save_figure(figure, output_stem, formats, dpi=140)


def _panel_row(panel: RatioPanel) -> dict:
    return {
        "algorithm": panel.algorithm,
        "algorithm_label": ALGORITHM_LABELS[panel.algorithm],
        "problem": panel.problem,
        "ea_fill_ratio": panel.ratio,
        "runs": panel.runs,
        "representative_seed": panel.representative_seed,
        "representative_rule": "minimum absolute distance to median IGD-infinity",
        "igd_reference": panel.igd_reference,
        "median_hv": panel.median_hv,
        "q1_hv": panel.q1_hv,
        "q3_hv": panel.q3_hv,
        "median_igd_infinity": panel.median_igd_infinity,
        "q1_igd_infinity": panel.q1_igd_infinity,
        "q3_igd_infinity": panel.q3_igd_infinity,
        "median_runtime_seconds": panel.median_runtime_seconds,
        "q1_runtime_seconds": panel.q1_runtime_seconds,
        "q3_runtime_seconds": panel.q3_runtime_seconds,
        "ea_archive_points_in_representative_run": len(panel.ea_archive),
        "retained_fill_points_in_representative_run": len(panel.fill_archive),
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_ranks(rows: Sequence[dict]) -> list[dict]:
    rank_values = {
        ratio: {"hv": [], "igd": [], "runtime": []}
        for ratio in RATIOS
    }
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    problems = sorted({str(row["problem"]) for row in rows})
    for algorithm in algorithms:
        for problem in problems:
            group = [
                row
                for row in rows
                if row["algorithm"] == algorithm and row["problem"] == problem
            ]
            if len(group) != len(RATIOS):
                continue
            for metric, reverse in (
                ("median_hv", True),
                ("median_igd_infinity", False),
                ("median_runtime_seconds", False),
            ):
                values = np.asarray([float(row[metric]) for row in group])
                ranks = rankdata(-values if reverse else values, method="average")
                rank_key = {
                    "median_hv": "hv",
                    "median_igd_infinity": "igd",
                    "median_runtime_seconds": "runtime",
                }[metric]
                for row, rank in zip(group, ranks):
                    rank_values[row["ea_fill_ratio"]][rank_key].append(float(rank))
    return [
        {
            "ea_fill_ratio": ratio,
            "ranked_tasks": len(rank_values[ratio]["igd"]),
            "mean_hv_rank": float(np.mean(rank_values[ratio]["hv"])),
            "mean_igd_infinity_rank": float(np.mean(rank_values[ratio]["igd"])),
            "mean_runtime_rank": float(np.mean(rank_values[ratio]["runtime"])),
        }
        for ratio in RATIOS
    ]


def generate_ratio_comparisons(
    stage1_root: Path,
    output_dir: Path,
    algorithms: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
    formats: Sequence[str],
) -> list[dict]:
    rows: list[dict] = []
    igd_overrides = _build_reference_igd_overrides(
        stage1_root, algorithms, problems, seeds
    )
    for algorithm in algorithms:
        problem_panels = []
        for problem in problems:
            panels = [
                _load_panel(
                    stage1_root,
                    ratio,
                    algorithm,
                    problem,
                    seeds,
                    igd_overrides,
                )
                for ratio in RATIOS
            ]
            problem_panels.append(panels)
            rows.extend(_panel_row(panel) for panel in panels)
            _plot_problem(
                panels,
                output_dir / algorithm / problem / "comparison_ratios",
                formats,
            )
        _plot_all_problems(
            algorithm,
            problem_panels,
            output_dir / f"comparison_all_problems_{algorithm}",
            formats,
        )
    _write_csv(output_dir / "ratio_metrics.csv", rows)
    ranks = _aggregate_ranks(rows)
    _write_csv(output_dir / "ratio_mean_ranks.csv", ranks)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-root", type=Path, default=Path("results/stage1")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <stage1-root>/comparisons/ratio_ablation",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    parser.add_argument(
        "--problems", nargs="+", choices=PROBLEMS, default=list(PROBLEMS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(21, 41))
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"]
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_args(argv)
    output_dir = arguments.output_dir or (
        arguments.stage1_root / "comparisons" / "ratio_ablation"
    )
    rows = generate_ratio_comparisons(
        arguments.stage1_root,
        output_dir,
        arguments.algorithms,
        arguments.problems,
        arguments.seeds,
        arguments.formats,
    )
    print(
        f"Generated {len(rows)} ratio/problem/algorithm summaries under {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
