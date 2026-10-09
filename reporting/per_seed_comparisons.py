"""Render one multi-page PDF per seed comparing every method.

Each page contains one benchmark problem and one panel per method.  Blue points
are the evaluated EA archive and red points are additional model-accepted
solutions where the method produces them.  The figures are intentionally
read-only views of the saved ``fronts.npz`` and ``summary.json`` artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .compare_methods import _method_front


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
ALGORITHMS = ("NSGAII", "NSGAIII", "MOEAD")
RATIOS = ("50_50", "60_40", "70_30", "80_20", "90_10")
STAGE2_METHODS = (
    "EA_NSGAII",
    "EA_NSGAIII",
    "EA_MOEAD",
    "GD-PSL_NSGAII",
    "GD-PSL_NSGAIII",
    "GD-PSL_MOEAD",
    "PangMOEAD",
)
STAGE2_LABELS = {
    "EA_NSGAII": "NSGA-II",
    "EA_NSGAIII": "NSGA-III",
    "EA_MOEAD": "MOEA/D",
    "GD-PSL_NSGAII": "GD-PSL + NSGA-II",
    "GD-PSL_NSGAIII": "GD-PSL + NSGA-III",
    "GD-PSL_MOEAD": "GD-PSL + MOEA/D",
    "PangMOEAD": "Pang",
}


def _stage1_methods() -> tuple[tuple[str, str], ...]:
    methods: list[tuple[str, str]] = []
    for algorithm in ALGORITHMS:
        methods.append((f"pure_ea/EA_{algorithm}", f"{algorithm} (EA)"))
        for ratio in RATIOS:
            methods.append(
                (
                    f"ratio_{ratio}/GD-PSL_{algorithm}",
                    f"{algorithm} GD-PSL ({ratio.replace('_', ':')})",
                )
            )
    return tuple(methods)


def _stage2_methods() -> tuple[tuple[str, str], ...]:
    return tuple((method, STAGE2_LABELS[method]) for method in STAGE2_METHODS)


def _summary_metrics(run_dir: Path) -> tuple[Optional[float], Optional[float]]:
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    hv = summary.get("hypervolume", {})
    igd = summary.get("igd_infinity", {})
    try:
        hv_value = float(hv.get("completed_hv", hv.get("value")))
    except (TypeError, ValueError):
        hv_value = None
    try:
        igd_value = float(igd.get("completed_igd_infinity"))
    except (TypeError, ValueError):
        igd_value = None
    return hv_value, igd_value


def _downsample(values: np.ndarray, maximum: Optional[int] = None) -> np.ndarray:
    if maximum is None or len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[indices]


def _load_problem_runs(
    root: Path,
    problem: str,
    seed: int,
    methods: Sequence[tuple[str, str]],
) -> list[dict]:
    records: list[dict] = []
    for relative_root, label in methods:
        run_dir = root / relative_root / problem / f"seed_{seed:03d}"
        if not (run_dir / "fronts.npz").is_file():
            continue
        front, _history, ea_archive, fill = _method_front(run_dir)
        if not len(front):
            continue
        hv, igd = _summary_metrics(run_dir)
        records.append(
            {
                "label": label,
                "front": front,
                "ea": ea_archive,
                "fill": fill,
                "hv": hv,
                "igd": igd,
            }
        )
    return records


def _limits(records: Sequence[dict], objective_count: int) -> list[tuple[float, float]]:
    points = [record["front"] for record in records if len(record["front"])]
    all_values = np.vstack(points)
    result = []
    for index in range(objective_count):
        lower = float(np.nanmin(all_values[:, index]))
        upper = float(np.nanmax(all_values[:, index]))
        span = max(upper - lower, 1e-12)
        padding = max(0.02, 0.04 * span)
        result.append((lower - padding, upper + padding))
    return result


def _draw_page(records: Sequence[dict], problem: str, seed: int, stage: str) -> plt.Figure:
    objective_count = records[0]["front"].shape[1]
    columns = 6 if stage == "stage1" else 4
    rows = int(np.ceil(len(records) / columns))
    figure = plt.figure(figsize=(4.0 * columns, 3.8 * rows), facecolor="white")
    limits = _limits(records, objective_count)
    for index, record in enumerate(records):
        projection = "3d" if objective_count == 3 else None
        axis = figure.add_subplot(rows, columns, index + 1, projection=projection)
        ea = record["ea"]
        fill = record["fill"]
        if objective_count == 3:
            axis.computed_zorder = False
            if len(ea):
                axis.scatter(*ea[:, :3].T, s=3, c="#2166d1", alpha=0.38, linewidths=0, rasterized=True)
            if len(fill):
                axis.scatter(*fill[:, :3].T, s=6, c="#d62728", alpha=0.8, linewidths=0, rasterized=True)
            axis.set_zlim(*limits[2])
            axis.set_zlabel("$f_3$", fontsize=7, labelpad=-2)
            axis.view_init(elev=24, azim=42)
            axis.set_box_aspect((1, 1, 0.85))
        else:
            if len(ea):
                axis.scatter(ea[:, 0], ea[:, 1], s=4, c="#2166d1", alpha=0.42, linewidths=0, rasterized=True)
            if len(fill):
                axis.scatter(fill[:, 0], fill[:, 1], s=7, c="#d62728", alpha=0.8, linewidths=0, rasterized=True)
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_xlabel("$f_1$", fontsize=8, labelpad=-1)
        axis.set_ylabel("$f_2$", fontsize=8, labelpad=-1)
        axis.tick_params(labelsize=6, pad=0)
        axis.grid(alpha=0.18)
        metric_text = []
        if record["hv"] is not None:
            metric_text.append(f"HV {record['hv']:.4g}")
        if record["igd"] is not None:
            metric_text.append(f"IGD-inf {record['igd']:.4g}")
        title = record["label"]
        if metric_text:
            title += "\n" + " | ".join(metric_text)
        axis.set_title(title, fontsize=8.5, fontweight="bold", pad=5)

    for index in range(len(records), rows * columns):
        figure.add_subplot(rows, columns, index + 1).axis("off")
    figure.suptitle(
        f"{stage.title()} method comparison | {problem} | seed {seed}\n"
        "blue: evaluated EA archive; red: additional model-accepted solutions",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.92, wspace=0.17, hspace=0.36)
    return figure


def generate_stage(
    stage: str,
    results_root: Path,
    output_dir: Path,
    seeds: Iterable[int],
    problems: Sequence[str],
) -> int:
    methods = _stage1_methods() if stage == "stage1" else _stage2_methods()
    count = 0
    for seed in seeds:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"seed_{seed:03d}.pdf"
        pages = 0
        with PdfPages(output_path) as pdf:
            for problem in problems:
                records = _load_problem_runs(results_root, problem, seed, methods)
                if not records:
                    continue
                pdf.savefig(_draw_page(records, problem, seed, stage), bbox_inches="tight")
                plt.close("all")
                pages += 1
        if pages:
            count += 1
        else:
            output_path.unlink(missing_ok=True)
    return count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "both"), default="both")
    parser.add_argument("--stage1-root", type=Path, default=Path("results/stage1"))
    parser.add_argument("--stage2-root", type=Path, default=Path("results/stage2"))
    parser.add_argument("--output-root", type=Path, default=Path("results/per_seed_comparisons"))
    parser.add_argument("--stage1-seeds", nargs="+", type=int, default=list(range(21, 41)))
    parser.add_argument("--stage2-seeds", nargs="+", type=int, default=list(range(101, 121)))
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS), choices=PROBLEMS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_args(argv)
    generated = 0
    if arguments.stage in {"stage1", "both"}:
        generated += generate_stage(
            "stage1", arguments.stage1_root, arguments.output_root / "stage1",
            arguments.stage1_seeds, arguments.problems,
        )
    if arguments.stage in {"stage2", "both"}:
        generated += generate_stage(
            "stage2", arguments.stage2_root, arguments.output_root / "stage2",
            arguments.stage2_seeds, arguments.problems,
        )
    print(f"Generated {generated} seed PDF(s) under {arguments.output_root}")


if __name__ == "__main__":
    main()
