"""Create compact PF-completion figures for the GD-PSL paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from evaluation_metrics import normalize_objectives, resolve_normalization_points
from problem_definitions import get_problem
REPRESENTATIVE_RUNS = {
    "dtlz2": (3, 102),
    "dtlz7": (3, 104),
    "re21": (2, 110),
    "re24": (2, 117),
    "re31": (3, 115),
    "re32": (3, 115),
    "re34": (3, 114),
    "re35": (3, 117),
    "re37": (3, 115),
}

GALLERY_METHODS = (
    "GD-PSL_NSGAII",
    "GD-PSL_NSGAIII",
    "GD-PSL_MOEAD",
    "PangMOEAD",
)

GALLERY_METHOD_LABELS = {
    "GD-PSL_NSGAII": "GD-PSL + NSGA-II",
    "GD-PSL_NSGAIII": "GD-PSL + NSGA-III",
    "GD-PSL_MOEAD": "GD-PSL + MOEA/D",
    "PangMOEAD": "PangMOEAD",
}

BLUE = "#2166d1"
RED = "#d62728"


def _run_directory(results_root: Path, problem: str, objectives: int, seed: int) -> Path:
    label = f"{problem.upper()}_{objectives}obj"
    return results_root / "GD-PSL_NSGAII" / label / f"seed_{seed:03d}"


def _normalized_fronts(
    results_root: Path, problem: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objectives, seed = REPRESENTATIVE_RUNS[problem]
    run_dir = _run_directory(results_root, problem, objectives, seed)
    with np.load(run_dir / "fronts.npz") as data:
        base = np.asarray(data["base_f"], dtype=float)
        completed = np.asarray(data["completed_f"], dtype=float)
        sources = np.asarray(data["completed_sources"], dtype=int)
    definition = (
        get_problem(problem, n_obj=objectives)
        if problem.startswith("dtlz")
        else get_problem(problem)
    )
    ideal, nadir, _ = resolve_normalization_points(problem, definition, objectives)
    return (
        normalize_objectives(base, ideal, nadir),
        normalize_objectives(completed, ideal, nadir),
        sources,
    )


def _representative_seeds(
    formal_runs: Path,
    criterion: str = "median",
) -> dict[tuple[str, str], int]:
    """Choose runs by median-nearest or minimum IGD-infinity."""
    if criterion not in {"median", "minimum"}:
        raise ValueError("criterion must be 'median' or 'minimum'")
    with formal_runs.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] in GALLERY_METHODS
        ]
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        problem = row["problem"].split("_", maxsplit=1)[0].lower()
        key = (row["method"], problem)
        grouped.setdefault(key, []).append(
            (int(row["seed"]), float(row["igd_infinity"]))
        )

    expected = {
        (method, problem)
        for method in GALLERY_METHODS
        for problem in REPRESENTATIVE_RUNS
    }
    missing = sorted(expected - set(grouped))
    if missing:
        raise ValueError(f"Missing Stage-B rows for: {missing}")

    selected: dict[tuple[str, str], int] = {}
    for key, values in grouped.items():
        if criterion == "minimum":
            selected[key] = min(values, key=lambda item: (item[1], item[0]))[0]
        else:
            median = float(np.median([value for _, value in values]))
            selected[key] = min(
                values, key=lambda item: (abs(item[1] - median), item[0])
            )[0]
    return selected


def _method_normalized_fronts(
    results_root: Path,
    method: str,
    problem: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objectives = REPRESENTATIVE_RUNS[problem][0]
    run_dir = (
        results_root
        / method
        / f"{problem.upper()}_{objectives}obj"
        / f"seed_{seed:03d}"
    )
    with np.load(run_dir / "fronts.npz") as data:
        if method == "PangMOEAD":
            completed = np.asarray(data["archive_f"], dtype=float)
            base = completed
            sources = np.zeros(len(completed), dtype=int)
        else:
            base = np.asarray(data["base_f"], dtype=float)
            completed = np.asarray(data["completed_f"], dtype=float)
            sources = np.asarray(data["completed_sources"], dtype=int)

    definition = (
        get_problem(problem, n_obj=objectives)
        if problem.startswith("dtlz")
        else get_problem(problem)
    )
    ideal, nadir, _ = resolve_normalization_points(problem, definition, objectives)
    return (
        normalize_objectives(base, ideal, nadir),
        normalize_objectives(completed, ideal, nadir),
        np.asarray(sources, dtype=int),
    )


def _limits(base: np.ndarray, completed: np.ndarray) -> list[tuple[float, float]]:
    values = np.vstack((base, completed))
    output = []
    for column in range(values.shape[1]):
        lower = min(0.0, float(values[:, column].min()))
        upper = max(1.0, float(values[:, column].max()))
        padding = max(0.015, 0.025 * (upper - lower))
        output.append((lower - padding, upper + padding))
    return output


def _style_axis(axis, objectives: int, limits: list[tuple[float, float]]) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_xlabel(r"$f_1$", labelpad=-1)
    axis.set_ylabel(r"$f_2$", labelpad=-1)
    axis.tick_params(labelsize=7, pad=0)
    axis.grid(alpha=0.18)
    if objectives == 3:
        axis.set_zlim(*limits[2])
        axis.set_zlabel(r"$f_3$", labelpad=-2)
        axis.view_init(elev=24, azim=42)
        axis.set_box_aspect((1, 1, 0.84))


def _draw_row(
    figure: plt.Figure,
    grid,
    row: int,
    results_root: Path,
    problem: str,
) -> None:
    base, completed, sources = _normalized_fronts(results_root, problem)
    objectives = base.shape[1]
    limits = _limits(base, completed)
    projection = "3d" if objectives == 3 else None
    left = figure.add_subplot(grid[row, 0], projection=projection)
    right = figure.add_subplot(grid[row, 1], projection=projection)

    scatter_base = dict(s=1.5, c=BLUE, alpha=0.40, linewidths=0, rasterized=True)
    scatter_fill = dict(
        s=4.5,
        c=RED,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.10,
        rasterized=True,
    )
    if objectives == 3:
        left.scatter(*base.T, depthshade=True, **scatter_base)
        right.computed_zorder = False
        right.scatter(
            *completed[sources == 0].T,
            depthshade=True,
            zorder=1,
            **scatter_base,
        )
        right.scatter(
            *completed[sources == 1].T,
            depthshade=False,
            zorder=3,
            **scatter_fill,
        )
    else:
        left.scatter(base[:, 0], base[:, 1], zorder=1, **scatter_base)
        right.scatter(
            completed[sources == 0, 0],
            completed[sources == 0, 1],
            zorder=1,
            **scatter_base,
        )
        right.scatter(
            completed[sources == 1, 0],
            completed[sources == 1, 1],
            zorder=3,
            **scatter_fill,
        )

    _style_axis(left, objectives, limits)
    _style_axis(right, objectives, limits)
    left.set_title("EA archive", fontsize=9, pad=8)
    right.set_title("Completed archive", fontsize=9, pad=8)
    label = problem.upper()
    if objectives == 3:
        left.text2D(
            -0.17,
            0.5,
            label,
            transform=left.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    else:
        left.text(
            -0.24,
            0.5,
            label,
            transform=left.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )


def create_figure(results_root: Path, problems: tuple[str, str], output: Path) -> None:
    figure = plt.figure(figsize=(7.4, 7.1), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.995,
        bottom=0.10,
        top=0.94,
        hspace=0.22,
        wspace=0.02,
    )
    for row, problem in enumerate(problems):
        _draw_row(figure, grid, row, results_root, problem)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=5,
               label="EA archive solution"),
        Line2D([], [], marker="o", linestyle="", color=RED, markersize=5,
               label="Retained true-evaluated model solution"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.015),
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _draw_completed_panel(
    figure: plt.Figure,
    grid,
    row: int,
    column: int,
    results_root: Path,
    problem: str,
    tick_size: float = 5.5,
    title_size: float = 8.5,
) -> None:
    base, completed, sources = _normalized_fronts(results_root, problem)
    objectives = completed.shape[1]
    projection = "3d" if objectives == 3 else None
    axis = figure.add_subplot(grid[row, column], projection=projection)
    limits = _limits(base, completed)
    ea = completed[sources == 0]
    model = completed[sources == 1]
    blue = dict(s=0.9, c=BLUE, alpha=0.34, linewidths=0, rasterized=True)
    red = dict(
        s=2.6,
        c=RED,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.08,
        rasterized=True,
    )
    if objectives == 3:
        axis.computed_zorder = False
        axis.scatter(*ea.T, depthshade=True, zorder=1, **blue)
        axis.scatter(*model.T, depthshade=False, zorder=3, **red)
    else:
        axis.scatter(ea[:, 0], ea[:, 1], zorder=1, **blue)
        axis.scatter(model[:, 0], model[:, 1], zorder=3, **red)
    _style_axis(axis, objectives, limits)
    axis.tick_params(labelsize=tick_size, pad=-1)
    axis.xaxis.label.set_size(7)
    axis.yaxis.label.set_size(7)
    if objectives == 3:
        axis.zaxis.label.set_size(7)
    axis.set_title(problem.upper(), fontsize=title_size, fontweight="bold", pad=3)


def create_engineering_figure(results_root: Path, output: Path) -> None:
    problems = ("re31", "re34", "re35", "re37")
    figure = plt.figure(figsize=(7.4, 5.8), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.995,
        bottom=0.12,
        top=0.975,
        hspace=0.08,
        wspace=-0.02,
    )
    for index, problem in enumerate(problems):
        _draw_completed_panel(
            figure,
            grid,
            index // 2,
            index % 2,
            results_root,
            problem,
            tick_size=6.5,
            title_size=9.5,
        )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=5,
               label="EA archive solution"),
        Line2D([], [], marker="o", linestyle="", color=RED, markersize=5,
               label="Retained true-evaluated model solution"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.51, 0.018),
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def create_stage2_gallery(results_root: Path, output: Path) -> None:
    problems = tuple(REPRESENTATIVE_RUNS)
    figure = plt.figure(figsize=(7.4, 6.7), facecolor="white")
    grid = figure.add_gridspec(
        3,
        3,
        left=0.035,
        right=0.985,
        bottom=0.095,
        top=0.975,
        hspace=0.20,
        wspace=0.06,
    )
    for index, problem in enumerate(problems):
        _draw_completed_panel(
            figure,
            grid,
            index // 3,
            index % 3,
            results_root,
            problem,
        )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=5,
               label="EA archive solution"),
        Line2D([], [], marker="o", linestyle="", color=RED, markersize=5,
               label="Retained true-evaluated model solution"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.51, 0.012),
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _draw_compact_archive(
    figure: plt.Figure,
    slot,
    problem: str,
    base: np.ndarray,
    completed: np.ndarray,
    sources: np.ndarray,
    limits: list[tuple[float, float]],
    show_fill: bool,
    seed: int,
) -> None:
    objectives = completed.shape[1]
    axis = figure.add_subplot(slot, projection="3d" if objectives == 3 else None)
    archive = completed[sources == 0] if show_fill else completed
    fill = completed[sources == 1] if show_fill else np.empty((0, objectives))
    blue = dict(s=0.42, c=BLUE, alpha=0.30, linewidths=0, rasterized=True)
    red = dict(
        s=1.45,
        c=RED,
        alpha=0.78,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
    )
    if objectives == 3:
        axis.computed_zorder = False
        axis.scatter(*archive.T, depthshade=True, zorder=1, **blue)
        if len(fill):
            axis.scatter(*fill.T, depthshade=False, zorder=3, **red)
        axis.set_zlim(*limits[2])
        axis.set_zticks([])
        axis.view_init(elev=24, azim=42)
        axis.set_box_aspect((1, 1, 0.84))
        axis.set_proj_type("ortho")
    else:
        axis.scatter(archive[:, 0], archive[:, 1], zorder=1, **blue)
        if len(fill):
            axis.scatter(fill[:, 0], fill[:, 1], zorder=3, **red)
        for spine in axis.spines.values():
            spine.set_linewidth(0.35)
            spine.set_color("#777777")

    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(alpha=0.12, linewidth=0.3)
    axis.set_title(
        f"{problem.upper()} (s{seed})",
        fontsize=5.7,
        fontweight="bold",
        pad=1.5,
    )


def create_four_method_gallery(
    results_root: Path,
    formal_runs: Path,
    output: Path,
) -> dict[str, dict[str, int]]:
    """Plot all nine tasks for the three GD-PSL variants and PangMOEAD."""
    problems = tuple(REPRESENTATIVE_RUNS)
    seeds = _representative_seeds(formal_runs, criterion="median")
    fronts = {
        (method, problem): _method_normalized_fronts(
            results_root, method, problem, seeds[(method, problem)]
        )
        for method in GALLERY_METHODS
        for problem in problems
    }
    shared_limits = {
        problem: _limits(
            np.vstack([fronts[(method, problem)][0] for method in GALLERY_METHODS]),
            np.vstack([fronts[(method, problem)][1] for method in GALLERY_METHODS]),
        )
        for problem in problems
    }

    figure = plt.figure(figsize=(8.0, 8.15), facecolor="white")
    outer = figure.add_gridspec(
        2,
        2,
        left=0.018,
        right=0.992,
        bottom=0.065,
        top=0.965,
        hspace=0.18,
        wspace=0.08,
    )
    panel_letters = "abcd"
    for method_index, method in enumerate(GALLERY_METHODS):
        row, column = divmod(method_index, 2)
        block = outer[row, column].subgridspec(3, 3, hspace=0.13, wspace=0.025)
        for problem_index, problem in enumerate(problems):
            base, completed, sources = fronts[(method, problem)]
            _draw_compact_archive(
                figure,
                block[problem_index // 3, problem_index % 3],
                problem,
                base,
                completed,
                sources,
                shared_limits[problem],
                show_fill=method != "PangMOEAD",
                seed=seeds[(method, problem)],
            )
        bounds = outer[row, column].get_position(figure)
        figure.text(
            (bounds.x0 + bounds.x1) / 2,
            bounds.y1 + 0.014,
            f"({panel_letters[method_index]}) {GALLERY_METHOD_LABELS[method]}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )

    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=4,
               label="EA/archive solution"),
        Line2D([], [], marker="o", linestyle="", color=RED, markersize=4,
               label="Retained true-evaluated model solution (GD-PSL only)"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
        fontsize=7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)
    return {
        method: {problem: seeds[(method, problem)] for problem in problems}
        for method in GALLERY_METHODS
    }


def create_re37_projection_figure(
    results_root: Path,
    formal_runs: Path,
    output: Path,
) -> int:
    """Show the three pairwise objective projections for representative RE37."""
    method = "GD-PSL_NSGAII"
    seed = _representative_seeds(formal_runs, criterion="median")[(method, "re37")]
    base, completed, sources = _method_normalized_fronts(
        results_root, method, "re37", seed
    )
    limits = _limits(base, completed)
    ea = completed[sources == 0]
    fill = completed[sources == 1]
    pairs = ((0, 1), (0, 2), (1, 2))
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.35), facecolor="white")
    for axis, (first, second) in zip(axes, pairs):
        axis.scatter(
            ea[:, first],
            ea[:, second],
            s=0.9,
            c=BLUE,
            alpha=0.30,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            fill[:, first],
            fill[:, second],
            s=2.8,
            c=RED,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlim(*limits[first])
        axis.set_ylim(*limits[second])
        axis.set_xlabel(fr"$f_{first + 1}$", labelpad=1)
        axis.set_ylabel(fr"$f_{second + 1}$", labelpad=1)
        axis.tick_params(labelsize=6, pad=1)
        axis.grid(alpha=0.18, linewidth=0.45)
        axis.set_box_aspect(1)
    figure.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=4,
                   label="EA archive solution"),
            Line2D([], [], marker="o", linestyle="", color=RED, markersize=4,
                   label="Retained true-evaluated model solution"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=7.2,
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.985, bottom=0.22, wspace=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)
    return seed


def create_re34_state_figure(formal_runs: Path, output: Path) -> dict[str, float | int]:
    """Show the two empirical IGD-infinity states observed for GD-PSL+NSGA-II."""
    with formal_runs.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == "GD-PSL_NSGAII" and row["problem"] == "RE34_3obj"
        ]
    if len(rows) != 20:
        raise ValueError(f"Expected 20 RE34 runs, found {len(rows)}")

    seeds = np.asarray([int(row["seed"]) for row in rows])
    values = np.asarray([float(row["igd_infinity"]) for row in rows])
    ordered = np.sort(values)
    split_index = int(np.argmax(np.diff(ordered)))
    threshold = float((ordered[split_index] + ordered[split_index + 1]) / 2.0)
    lower_state = values < threshold

    figure, axis = plt.subplots(figsize=(6.2, 2.25), facecolor="white")
    axis.scatter(
        seeds[lower_state], values[lower_state], s=31, color="#17823b",
        edgecolors="white", linewidths=0.45, label=f"Lower-distance state ({lower_state.sum()}/20)",
        zorder=3,
    )
    axis.scatter(
        seeds[~lower_state], values[~lower_state], s=31, color="#e66101",
        edgecolors="white", linewidths=0.45, label=f"Higher-distance state ({(~lower_state).sum()}/20)",
        zorder=3,
    )
    axis.axhline(threshold, color="#555555", linestyle="--", linewidth=0.9)
    axis.text(
        120.2,
        threshold + 0.018,
        "largest-gap state boundary",
        color="#555555",
        fontsize=7,
        ha="right",
        va="bottom",
    )
    axis.set_xlabel("Paired seed")
    axis.set_ylabel(r"$\mathrm{IGD}_{\infty}^{\mathrm{proj}}$")
    axis.set_xticks(np.arange(101, 121, 2))
    axis.set_xlim(100.5, 120.5)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    figure.tight_layout(pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    return {
        "lower_state_count": int(lower_state.sum()),
        "higher_state_count": int((~lower_state).sum()),
        "state_boundary": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/stage2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_figures"))
    parser.add_argument(
        "--formal-stage2-runs",
        type=Path,
        default=Path("results/stage2/comparisons/formal_runs.csv"),
    )
    args = parser.parse_args()
    create_figure(
        args.results_root,
        ("dtlz2", "dtlz7"),
        args.output_dir / "pf_completion_dtlz.pdf",
    )
    create_engineering_figure(args.results_root, args.output_dir / "pf_completion_re.pdf")
    create_stage2_gallery(
        args.results_root,
        args.output_dir / "stage2_pf_gallery.pdf",
    )
    create_four_method_gallery(
        args.results_root,
        args.formal_stage2_runs,
        args.output_dir / "stage2_four_method_gallery.pdf",
    )
    create_re37_projection_figure(
        args.results_root,
        args.formal_stage2_runs,
        args.output_dir / "re37_pairwise_views.pdf",
    )
    create_re34_state_figure(
        args.formal_stage2_runs,
        args.output_dir / "re34_igd_states.pdf",
    )


if __name__ == "__main__":
    main()
