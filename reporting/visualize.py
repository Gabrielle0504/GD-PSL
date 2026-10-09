"""Visualize saved Pareto-set completion artifacts.

The script is deliberately read-only with respect to the completion pipeline:
it consumes ``fronts.npz`` and the two gap CSV files written by either
independent runner and never participates in gap selection or model judging.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pareto_utils import nondominated_indices


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


def _cumulative_nondominated_front(objectives: np.ndarray) -> np.ndarray:
    """Return the empirical PF found across all recorded minimization evaluations."""
    values = _as_2d(objectives)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return values
    values = np.unique(values, axis=0)
    return values[nondominated_indices(values)]


def _completed_cumulative_archive(
    ea_archive: np.ndarray,
    model_objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge evaluated EA/model objectives and retain a source-labelled front."""
    ea = _as_2d(ea_archive)
    model = _as_2d(model_objectives, ea.shape[1])
    values = np.vstack((ea, model))
    sources = np.concatenate((np.zeros(len(ea), dtype=int), np.ones(len(model), dtype=int)))
    finite = np.isfinite(values).all(axis=1)
    values, sources = values[finite], sources[finite]
    if not len(values):
        return values, sources

    # EA rows come first, so an objective-space duplicate is attributed to the
    # already evaluated EA archive rather than counted as a model contribution.
    _, unique_indices = np.unique(values, axis=0, return_index=True)
    unique_indices.sort()
    values, sources = values[unique_indices], sources[unique_indices]
    keep = nondominated_indices(values)
    return values[keep], sources[keep]


def _evaluations_from_file(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read evaluated objective rows and an optional true FE column."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            keys = ("evaluations_f", "evaluation_f", "all_f", "f")
            objective_key = next((key for key in keys if key in data), None)
            if objective_key is None:
                raise ValueError(f"No evaluation objective array found in {path}.")
            values = _as_2d(data[objective_key].copy())
            fe_key = next(
                (key for key in ("evaluations_fe", "evaluation_fe", "fe") if key in data),
                None,
            )
            fe = None if fe_key is None else np.asarray(data[fe_key], dtype=float).reshape(-1).copy()
        return values, fe
    if path.suffix.lower() in {".dat", ".txt"}:
        return _as_2d(np.loadtxt(path)), None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return np.empty((0, 0), dtype=float), None
    columns = sorted(
        (key for key in rows[0] if key.startswith("f") and key[1:].isdigit()),
        key=lambda key: int(key[1:]),
    )
    values = _as_2d([[float(row[key]) for key in columns] for row in rows])
    fe_key = next(
        (key for key in rows[0] if key.lower() in {"fe", "function_evaluations", "evaluation_count"}),
        None,
    )
    fe = None if fe_key is None else np.asarray([float(row[fe_key]) for row in rows], dtype=float)
    return values, fe


def _read_gap_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _preference_values(rows: list[dict[str, str]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    columns = sorted(
        (key for key in rows[0] if key.startswith("preference_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    return _as_2d([[float(row[key]) for key in columns] for row in rows])


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _set_pareto_3d_view(axis) -> None:
    """Use the trihedral view convention commonly used for DTLZ2 plots."""
    axis.view_init(elev=25, azim=45)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _plot_three_objective_report(
    base_archive: np.ndarray,
    completed_front: np.ndarray,
    completed_sources: np.ndarray,
    output: Path,
    title: str,
    ea_fe: int,
    total_fe: int,
    method_label: str = "Model",
    adds_solutions: bool = True,
    completed_verified_hole: Optional[np.ndarray] = None,
    completed_boundary_probe: Optional[np.ndarray] = None,
) -> None:
    """Compare the cumulative EA archive with the total-budget completed archive."""
    figure = plt.figure(figsize=(16, 9), facecolor="white")
    grid = figure.add_gridspec(2, 4, height_ratios=(1.22, 0.82), hspace=0.08, wspace=0.12)
    left = figure.add_subplot(grid[0, 0:2], projection="3d")
    right = figure.add_subplot(grid[0, 2:4], projection="3d")
    # Axes3D normally recomputes collection order from camera depth, which
    # can hide model points behind the denser EA cloud. This comparison uses
    # semantic layering so accepted model contributions remain inspectable.
    right.computed_zorder = False

    if len(base_archive):
        left.scatter(
            *base_archive[:, :3].T,
            s=11,
            c="#2166d1",
            alpha=0.78,
            label=f"EA cumulative empirical PF (n={len(base_archive)})",
        )
    left.set_title(f"EA cumulative archive ({ea_fe:,} FE)", pad=10)
    left.set_xlabel("$f_1$")
    left.set_ylabel("$f_2$")
    left.set_zlabel("$f_3$")
    _set_pareto_3d_view(left)
    left.legend(loc="upper right", frameon=False, markerscale=1.5)

    completed_sources = np.asarray(completed_sources, dtype=int).reshape(-1)
    base_rows = completed_sources == 0
    model_rows = completed_sources == 1
    boundary_rows = (
        np.zeros(len(model_rows), dtype=bool)
        if completed_boundary_probe is None
        else model_rows & np.asarray(completed_boundary_probe, dtype=bool).reshape(-1)
    )
    verified_rows = (
        model_rows
        if completed_verified_hole is None
        else model_rows & np.asarray(completed_verified_hole, dtype=bool).reshape(-1)
    ) & ~boundary_rows
    unverified_rows = model_rows & ~verified_rows & ~boundary_rows
    if np.any(base_rows):
        right.scatter(
            *completed_front[base_rows, :3].T,
            s=11,
            c="#2166d1",
            alpha=0.78,
            depthshade=True,
            zorder=1,
            label=f"EA archive points retained (n={np.count_nonzero(base_rows)})",
        )
    if np.any(unverified_rows):
        right.scatter(
            *completed_front[unverified_rows, :3].T,
            s=14,
            c="#f28e2b",
            alpha=0.48,
            depthshade=False,
            zorder=2,
            label=f"Other nondominated model points (n={np.count_nonzero(unverified_rows)})",
        )
    if np.any(boundary_rows):
        right.scatter(
            *completed_front[boundary_rows, :3].T,
            s=25,
            c="#2ca02c",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            depthshade=False,
            zorder=3,
            label=f"Boundary-probe discoveries (n={np.count_nonzero(boundary_rows)})",
        )
    if np.any(verified_rows):
        right.scatter(
            *completed_front[verified_rows, :3].T,
            s=25,
            c="#e45756",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            depthshade=False,
            zorder=3,
            label=f"Verified hole fills (n={np.count_nonzero(verified_rows)})",
        )
    right_title = (
        f"EA + {method_label} cumulative archive"
        if adds_solutions
        else f"{method_label} cumulative archive"
    )
    right.set_title(f"{right_title} ({total_fe:,} FE)", pad=10)
    right.set_xlabel("$f_1$")
    right.set_ylabel("$f_2$")
    right.set_zlabel("$f_3$")
    _set_pareto_3d_view(right)
    right.legend(loc="upper right", frameon=False, markerscale=1.3)

    projections = grid[1, :].subgridspec(1, 3, wspace=0.30)
    pairs = ((0, 1), (0, 2), (1, 2))
    first_projection_axis = None
    for axis_index, (x_index, y_index) in enumerate(pairs):
        axis = figure.add_subplot(projections[0, axis_index])
        if first_projection_axis is None:
            first_projection_axis = axis
        if len(base_archive):
            axis.scatter(
                base_archive[:, x_index], base_archive[:, y_index],
                s=7, c="#2166d1", alpha=0.52, label="EA cumulative archive",
            )
        if np.any(unverified_rows):
            axis.scatter(
                completed_front[unverified_rows, x_index],
                completed_front[unverified_rows, y_index],
                s=8,
                c="#f28e2b",
                alpha=0.36,
                label="Other nondominated model points",
            )
        if np.any(boundary_rows):
            axis.scatter(
                completed_front[boundary_rows, x_index],
                completed_front[boundary_rows, y_index],
                s=12,
                c="#2ca02c",
                alpha=0.82,
                label="Boundary-probe discoveries",
            )
        if np.any(verified_rows):
            axis.scatter(
                completed_front[verified_rows, x_index],
                completed_front[verified_rows, y_index],
                s=12,
                c="#e45756",
                alpha=0.82,
                label="Verified hole fills",
            )
        axis.set_xlabel(f"$f_{x_index + 1}$")
        axis.set_ylabel(f"$f_{y_index + 1}$")
        axis.grid(True, alpha=0.25)

    # Keep the legend readable while making the two main panels self-contained.
    if first_projection_axis is not None:
        first_projection_axis.legend(loc="best", frameon=False, markerscale=1.2)

    figure.subplots_adjust(top=0.92, bottom=0.08, left=0.04, right=0.96)
    figure.suptitle(title, y=0.99)
    figure.savefig(output / "ea_vs_completed_pf.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_ea_archive_report(
    archive: np.ndarray,
    output: Path,
    title: str,
) -> None:
    """Render the cumulative empirical PF without any external reference PF."""
    n_objectives = archive.shape[1]
    if n_objectives == 3:
        figure = plt.figure(figsize=(13, 10), facecolor="white")
        grid = figure.add_gridspec(2, 3, height_ratios=(1.35, 0.85), hspace=0.16, wspace=0.28)
        axis_3d = figure.add_subplot(grid[0, :], projection="3d")
        axis_3d.scatter(
            *archive[:, :3].T,
            s=14,
            c="#2166d1",
            alpha=0.85,
            label=f"Cumulative EA nondominated archive (n={len(archive)})",
        )
        axis_3d.set_xlabel("$f_1$")
        axis_3d.set_ylabel("$f_2$")
        axis_3d.set_zlabel("$f_3$")
        axis_3d.set_title(title, pad=12)
        _set_pareto_3d_view(axis_3d)
        axis_3d.legend(loc="upper right", frameon=False)

        for column, (x_index, y_index) in enumerate(((0, 1), (0, 2), (1, 2))):
            axis = figure.add_subplot(grid[1, column])
            axis.scatter(
                archive[:, x_index], archive[:, y_index],
                s=9, c="#2166d1", alpha=0.72,
            )
            axis.set_xlabel(f"$f_{x_index + 1}$")
            axis.set_ylabel(f"$f_{y_index + 1}$")
            axis.grid(True, alpha=0.24)
        figure.subplots_adjust(top=0.94, bottom=0.07, left=0.06, right=0.96)
        figure.savefig(output / "ea_archive_report.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        return

    if n_objectives == 2:
        figure, axis = plt.subplots(figsize=(8, 6.5))
        axis.scatter(
            archive[:, 0], archive[:, 1], s=24, c="#2166d1", alpha=0.85,
            label=f"Cumulative EA nondominated archive (n={len(archive)})",
        )
        axis.set_xlabel("$f_1$")
        axis.set_ylabel("$f_2$")
        axis.set_title(title)
        axis.grid(True, alpha=0.24)
        axis.legend(frameon=False)
        _save(figure, output / "ea_archive_report.png")
        return

    # A parallel-coordinate view includes every objective, unlike a truncated
    # pair grid for six- or nine-objective RE problems.
    lower = np.min(archive, axis=0)
    upper = np.max(archive, axis=0)
    ranges = np.where(upper > lower, upper - lower, 1.0)
    normalized = (archive - lower) / ranges
    x_positions = np.arange(n_objectives)
    figure, axis = plt.subplots(figsize=(max(9, n_objectives * 1.15), 6.5))
    for row in normalized:
        axis.plot(x_positions, row, color="#2166d1", alpha=0.22, linewidth=0.9)
    axis.scatter(
        np.tile(x_positions, len(normalized)), normalized.reshape(-1),
        s=4, c="#2166d1", alpha=0.18,
    )
    axis.set_xticks(x_positions, [f"$f_{index + 1}$" for index in range(n_objectives)])
    axis.set_ylabel("Per-objective normalized value")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(f"{title} (n={len(archive)})")
    axis.grid(True, axis="y", alpha=0.24)
    _save(figure, output / "ea_archive_report.png")


def _plot_ea_search_history(
    evaluations: np.ndarray,
    evaluation_fe: Optional[np.ndarray],
    archive: np.ndarray,
    output: Path,
    title: str,
) -> None:
    """Show all true evaluations separately from the final EA archive."""
    if evaluations is None or not len(evaluations):
        return
    n_objectives = evaluations.shape[1]
    color_values = (
        evaluation_fe
        if evaluation_fe is not None and len(evaluation_fe) == len(evaluations)
        else np.arange(1, len(evaluations) + 1)
    )
    if n_objectives == 3:
        figure = plt.figure(figsize=(8, 7), facecolor="white")
        axis = figure.add_subplot(111, projection="3d")
        points = axis.scatter(
            *evaluations[:, :3].T,
            s=4,
            c=color_values,
            cmap="viridis",
            alpha=0.16,
            label=f"All true evaluations (n={len(evaluations)})",
        )
        axis.scatter(
            *archive[:, :3].T,
            s=26,
            c="#e45756",
            edgecolors="white",
            linewidths=0.3,
            alpha=0.92,
            label="Final nondominated population",
        )
        axis.set_xlabel("$f_1$")
        axis.set_ylabel("$f_2$")
        axis.set_zlabel("$f_3$")
        axis.set_title(f"{title}\nAll evaluations include dominated solutions")
        _set_pareto_3d_view(axis)
        axis.legend(loc="upper right", frameon=False)
        figure.colorbar(points, ax=axis, fraction=0.045, pad=0.04, label="Function evaluations (FE)")
        figure.tight_layout()
        figure.savefig(output / "ea_search_history.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        return

    if n_objectives == 2:
        figure, axis = plt.subplots(figsize=(8, 6.5))
        points = axis.scatter(
            evaluations[:, 0], evaluations[:, 1], s=5, c=color_values,
            cmap="viridis", alpha=0.18,
            label=f"All true evaluations (n={len(evaluations)})",
        )
        axis.scatter(
            archive[:, 0], archive[:, 1], s=24, c="#e45756", alpha=0.9,
            label="Final nondominated population",
        )
        axis.set_xlabel("$f_1$")
        axis.set_ylabel("$f_2$")
        axis.set_title(f"{title}\nAll evaluations include dominated solutions")
        axis.grid(True, alpha=0.24)
        axis.legend(frameon=False)
        figure.colorbar(points, ax=axis, label="Function evaluations (FE)")
        _save(figure, output / "ea_search_history.png")
        return

    # For many objectives, show convergence of objective ranges over FE. This
    # avoids implying that a first-four-objective projection is the full PF.
    figure, axes = plt.subplots(
        n_objectives,
        1,
        figsize=(10, max(6, n_objectives * 1.65)),
        sharex=True,
        squeeze=False,
    )
    for index, axis in enumerate(axes[:, 0]):
        axis.scatter(color_values, evaluations[:, index], s=3, c="#4c78a8", alpha=0.12)
        axis.scatter(
            np.full(len(archive), np.max(color_values)), archive[:, index],
            s=9, c="#e45756", alpha=0.75,
        )
        axis.set_ylabel(f"$f_{index + 1}$")
        axis.grid(True, alpha=0.18)
    axes[-1, 0].set_xlabel("Function evaluations (FE)")
    figure.suptitle(f"{title}: all evaluations and final nondominated population")
    figure.tight_layout()
    figure.savefig(output / "ea_search_history.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_many_objective_report(
    base_archive: np.ndarray,
    completed_front: np.ndarray,
    completed_sources: np.ndarray,
    output: Path,
    title: str,
    ea_fe: int,
    total_fe: int,
    method_label: str = "Model",
    adds_solutions: bool = True,
    completed_verified_hole: Optional[np.ndarray] = None,
    completed_boundary_probe: Optional[np.ndarray] = None,
) -> None:
    """Compare cumulative archives while showing every objective dimension."""
    n_objectives = completed_front.shape[1]
    base_rows = completed_sources == 0
    model_rows = completed_sources == 1
    boundary_rows = (
        np.zeros(len(model_rows), dtype=bool)
        if completed_boundary_probe is None
        else model_rows & np.asarray(completed_boundary_probe, dtype=bool).reshape(-1)
    )
    verified_rows = (
        model_rows
        if completed_verified_hole is None
        else model_rows & np.asarray(completed_verified_hole, dtype=bool).reshape(-1)
    ) & ~boundary_rows
    unverified_rows = model_rows & ~verified_rows & ~boundary_rows
    if n_objectives == 2:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
        axes[0].scatter(
            base_archive[:, 0], base_archive[:, 1],
            s=15, c="#2166d1", alpha=0.75,
            label=f"EA cumulative empirical PF (n={len(base_archive)})",
        )
        axes[1].scatter(
            completed_front[base_rows, 0], completed_front[base_rows, 1],
            s=15, c="#2166d1", alpha=0.75,
            label=f"EA points retained (n={np.count_nonzero(base_rows)})",
        )
        if np.any(unverified_rows):
            axes[1].scatter(
                completed_front[unverified_rows, 0],
                completed_front[unverified_rows, 1],
                s=17,
                c="#f28e2b",
                alpha=0.48,
                label=f"Other nondominated model points (n={np.count_nonzero(unverified_rows)})",
            )
        if np.any(boundary_rows):
            axes[1].scatter(
                completed_front[boundary_rows, 0],
                completed_front[boundary_rows, 1],
                s=24,
                c="#2ca02c",
                alpha=0.92,
                label=f"Boundary-probe discoveries (n={np.count_nonzero(boundary_rows)})",
            )
        if np.any(verified_rows):
            axes[1].scatter(
                completed_front[verified_rows, 0], completed_front[verified_rows, 1],
                s=24, c="#e45756", alpha=0.92,
                label=f"Verified hole fills (n={np.count_nonzero(verified_rows)})",
            )
        axes[0].set_title(f"EA cumulative archive ({ea_fe:,} FE)")
        right_title = (
            f"EA + {method_label} cumulative archive"
            if adds_solutions
            else f"{method_label} cumulative archive"
        )
        axes[1].set_title(f"{right_title} ({total_fe:,} FE)")
        for axis in axes:
            axis.set_xlabel("$f_1$")
            axis.set_ylabel("$f_2$")
            axis.grid(True, alpha=0.24)
            axis.legend(frameon=False)
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output / "ea_vs_completed_pf.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        return

    lower = np.min(completed_front, axis=0)
    upper = np.max(completed_front, axis=0)
    ranges = np.where(upper > lower, upper - lower, 1.0)
    normalized_base = (base_archive - lower) / ranges
    normalized_completed = (completed_front - lower) / ranges
    x_positions = np.arange(n_objectives)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    alpha = min(0.22, max(0.018, 60.0 / max(len(completed_front), 1)))
    for row in normalized_base:
        axes[0].plot(x_positions, row, color="#2166d1", alpha=alpha, linewidth=0.75)
    for row, source, verified, boundary in zip(
        normalized_completed, completed_sources, verified_rows, boundary_rows
    ):
        color = (
            "#2166d1"
            if source == 0
            else ("#2ca02c" if boundary else ("#e45756" if verified else "#f28e2b"))
        )
        line_alpha = alpha if source == 0 else min(0.72, alpha * 4.0)
        axes[1].plot(x_positions, row, color=color, alpha=line_alpha, linewidth=0.8)
    axes[0].plot([], [], color="#2166d1", label=f"EA cumulative empirical PF (n={len(base_archive)})")
    axes[1].plot([], [], color="#2166d1", label=f"EA points retained (n={np.count_nonzero(base_rows)})")
    if np.any(unverified_rows):
        axes[1].plot([], [], color="#f28e2b", label=f"Other nondominated model points (n={np.count_nonzero(unverified_rows)})")
    if np.any(verified_rows):
        axes[1].plot([], [], color="#e45756", label=f"Verified hole fills (n={np.count_nonzero(verified_rows)})")
    if np.any(boundary_rows):
        axes[1].plot([], [], color="#2ca02c", label=f"Boundary-probe discoveries (n={np.count_nonzero(boundary_rows)})")
    axes[0].set_title(f"EA cumulative archive ({ea_fe:,} FE)")
    right_title = (
        f"EA + {method_label} cumulative archive"
        if adds_solutions
        else f"{method_label} cumulative archive"
    )
    axes[1].set_title(f"{right_title} ({total_fe:,} FE)")
    for axis in axes:
        axis.set_xticks(x_positions, [f"$f_{index + 1}$" for index in range(n_objectives)])
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, axis="y", alpha=0.24)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Per-objective normalized value (shared scale)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output / "ea_vs_completed_pf.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_final_archive(front: np.ndarray, output: Path, title: str = "Final Pareto archive") -> None:
    """Render only the final completed archive, without source overlays."""
    n_objectives = front.shape[1]
    if n_objectives == 3:
        figure = plt.figure(figsize=(7, 7), facecolor="white")
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(*front[:, :3].T, s=10, c="#2166d1", alpha=0.78)
        axis.set_xlabel("$f_1$")
        axis.set_ylabel("$f_2$")
        axis.set_zlabel("$f_3$")
        axis.set_title(title)
        _set_pareto_3d_view(axis)
        figure.tight_layout()
        figure.savefig(output / "final_archive.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    if n_objectives == 2:
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(front[:, 0], front[:, 1], s=12, c="#2166d1", alpha=0.78)
        axis.set_xlabel("$f_1$")
        axis.set_ylabel("$f_2$")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        _save(figure, output / "final_archive.png")
        return

    dimensions = min(n_objectives, 4)
    figure, axes = plt.subplots(dimensions, dimensions, figsize=(11, 11), squeeze=False)
    for row in range(dimensions):
        for column in range(dimensions):
            axis = axes[row, column]
            if row == column:
                axis.hist(front[:, row], bins=20, color="#2166d1", alpha=0.65)
            else:
                axis.scatter(front[:, column], front[:, row], s=7, c="#2166d1", alpha=0.55)
            if row == dimensions - 1:
                axis.set_xlabel(f"$f_{column + 1}$")
            if column == 0:
                axis.set_ylabel(f"$f_{row + 1}$")
    figure.suptitle(f"{title} ({n_objectives} objectives)")
    figure.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94, wspace=0.28, hspace=0.28)
    figure.savefig(output / "final_archive.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_preferences(
    observed: np.ndarray,
    selected: np.ndarray,
    unfilled: np.ndarray,
    output: Path,
) -> None:
    if observed.size == 0 and selected.size == 0 and unfilled.size == 0:
        return
    preference_dim = max((array.shape[1] for array in (observed, selected, unfilled) if array.size), default=1)
    if preference_dim <= 2:
        figure = plt.figure(figsize=(7, 6))
        axis = figure.add_subplot(111)
        if preference_dim == 1:
            axis.scatter(observed[:, 0], np.zeros(len(observed)), s=12, c="#4c78a8", label="observed")
            if len(selected):
                axis.scatter(selected[:, 0], np.ones(len(selected)), s=24, c="#f28e2b", label="selected gap")
            if len(unfilled):
                axis.scatter(unfilled[:, 0], np.full(len(unfilled), 2), s=24, c="#e15759", label="unfilled")
            axis.set_yticks([0, 1, 2], ["observed", "selected", "unfilled"])
        else:
            if len(observed):
                axis.scatter(observed[:, 0], observed[:, 1], s=12, c="#4c78a8", label="observed")
            if len(selected):
                axis.scatter(selected[:, 0], selected[:, 1], s=24, c="#f28e2b", label="selected gap")
            if len(unfilled):
                axis.scatter(unfilled[:, 0], unfilled[:, 1], s=24, c="#e15759", marker="x", label="unfilled")
            axis.set_xlabel("preference 1")
            axis.set_ylabel("preference 2")
        axis.set_title("Preference-space coverage")
        axis.legend(loc="best")
        _save(figure, output / "preference_coverage.png")
        return

    dimensions = min(preference_dim, 4)
    figure, axes = plt.subplots(dimensions, dimensions, figsize=(11, 11), squeeze=False)
    for row in range(dimensions):
        for column in range(dimensions):
            axis = axes[row, column]
            if row == column:
                if len(observed):
                    axis.hist(observed[:, row], bins=15, color="#4c78a8", alpha=0.45)
                if len(selected):
                    axis.hist(selected[:, row], bins=15, color="#f28e2b", alpha=0.45)
            else:
                if len(observed):
                    axis.scatter(observed[:, column], observed[:, row], s=5, c="#4c78a8", alpha=0.3)
                if len(selected):
                    axis.scatter(selected[:, column], selected[:, row], s=8, c="#f28e2b", alpha=0.7)
                if len(unfilled):
                    axis.scatter(unfilled[:, column], unfilled[:, row], s=8, c="#e15759", marker="x", alpha=0.7)
            if row == dimensions - 1:
                axis.set_xlabel(f"p{column + 1}")
            if column == 0:
                axis.set_ylabel(f"p{row + 1}")
    figure.suptitle("Preference-space coverage projections")
    _save(figure, output / "preference_coverage.png")


def _plot_reasons(rows: list[dict[str, str]], output: Path) -> None:
    counts = Counter(row.get("reason_code", "unknown") for row in rows)
    if not counts:
        return
    labels, values = zip(*sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color="#59a14f")
    axis.set_ylabel("number of candidate regions")
    axis.set_title("Runtime evidence for filled and unfilled regions")
    axis.tick_params(axis="x", rotation=35)
    _save(figure, output / "gap_reasons.png")


def _plot_training_loss(run_dir: Path, output: Path) -> None:
    history_path = run_dir / "training_history.csv"
    if not history_path.exists():
        return
    with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=int)
    train_loss = np.asarray([float(row["train_loss"]) for row in rows], dtype=float)
    validation_loss = np.asarray(
        [float(row["validation_loss"]) for row in rows], dtype=float
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(epochs, train_loss, color="#4c78a8", linewidth=1.8, label="Training MSE")
    axis.plot(
        epochs,
        validation_loss,
        color="#e15759",
        linewidth=1.8,
        label="Validation MSE",
    )
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best_epoch = summary.get("best_epoch")
        if best_epoch is not None:
            axis.axvline(
                int(best_epoch),
                color="#59a14f",
                linestyle="--",
                linewidth=1.2,
                label=f"Best epoch ({best_epoch})",
            )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Pareto-set model training diagnostics")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend()
    _save(figure, output / "training_loss.png")


def _visualize_pang_run(
    run_dir: Path,
    output_dir: Path,
    title: str | None = None,
) -> list[Path]:
    """Render Pang as an EA archive method without model-fill highlighting."""
    artifact = run_dir / "fronts.npz"
    with np.load(artifact) as data:
        evaluations = _as_2d(data["evaluations_f"].copy())
        archive = _as_2d(data["archive_f"].copy())
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "pf_report.png", "pf_comparison.png", "pang_archive_vs_large_set.png",
        "preference_coverage.png", "gap_reasons.png", "training_loss.png",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)

    sources = np.zeros(len(archive), dtype=int)
    algorithm = str(summary.get("algorithm", "Run"))
    problem = str(summary.get("problem", "problem")).upper()
    method_label = "Pang"
    if title is None:
        title = f"{algorithm} on {archive.shape[1]}-objective {problem}"
    ea_fe = int(summary.get("fe_used", len(evaluations)))
    total_fe = int(summary.get("common_total_fe_budget", ea_fe))
    _plot_final_archive(
        archive,
        output_dir,
        title=f"{title}: final Pang large solution set",
    )
    comparison_title = "Pang cumulative empirical PF archive"
    if archive.shape[1] == 3:
        _plot_three_objective_report(
            base_archive=archive,
            completed_front=archive,
            completed_sources=sources,
            output=output_dir,
            title=comparison_title,
            ea_fe=ea_fe,
            total_fe=total_fe,
            method_label=method_label,
            adds_solutions=False,
        )
    else:
        _plot_many_objective_report(
            base_archive=archive,
            completed_front=archive,
            completed_sources=sources,
            output=output_dir,
            title=comparison_title,
            ea_fe=ea_fe,
            total_fe=total_fe,
            method_label=method_label,
            adds_solutions=False,
        )

    hv = summary.get("hypervolume", {})
    ideal = np.asarray(hv.get("normalization_ideal", np.min(archive, axis=0)), dtype=float)
    nadir = np.asarray(hv.get("normalization_nadir", np.max(archive, axis=0)), dtype=float)
    observed_preferences = _as_2d(
        1.0 - np.clip((archive - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0), 0.0, 1.0)
    )
    observed_preferences = observed_preferences / np.maximum(
        observed_preferences.sum(axis=1, keepdims=True), 1e-15
    )
    _plot_preferences(
        observed_preferences,
        np.empty((0, archive.shape[1])),
        np.empty((0, archive.shape[1])),
        output_dir,
    )
    return sorted(output_dir.glob("*.png"))


def visualize_run(
    run_dir: Path,
    output_dir: Path,
    evaluations_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> list[Path]:
    artifact = run_dir / "fronts.npz"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing required artifact: {artifact}")
    summary_path = run_dir / "summary.json"
    method_summary = {}
    if summary_path.exists():
        try:
            method_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            method_summary = {}
    if method_summary.get("method") == "pang_archive_baseline":
        return _visualize_pang_run(run_dir, output_dir, title=title)
    with np.load(artifact) as data:
        base = _as_2d(data["base_f"].copy())
        completed = _as_2d(data["completed_f"].copy(), base.shape[1])
        sources = np.asarray(data["completed_sources"], dtype=int).copy()
        observed = _as_2d(data["observed_preferences"].copy())
        gap = _as_2d(data["gap_preferences"].copy())
        model_f = _as_2d(data["model_f"].copy(), base.shape[1])
        completed_verified_hole = (
            np.asarray(data["completed_verified_hole"], dtype=bool).reshape(-1).copy()
            if "completed_verified_hole" in data
            else None
        )
        completed_boundary_probe = (
            np.asarray(data["completed_boundary_probe"], dtype=bool).reshape(-1).copy()
            if "completed_boundary_probe" in data
            else None
        )
        saved_stage1_f = _as_2d(data["stage1_f"].copy()) if "stage1_f" in data else None
        saved_stage1_fe = (
            np.asarray(data["stage1_fe"], dtype=float).reshape(-1).copy()
            if "stage1_fe" in data
            else None
        )
    gap_rows = _read_gap_csv(run_dir / "gap_regions.csv")
    unfilled_rows = _read_gap_csv(run_dir / "unfilled_regions.csv")
    selected = _preference_values([row for row in gap_rows if row.get("selected_for_model") == "True"])
    unfilled = _preference_values(unfilled_rows)
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    if title is None:
        if summary:
            algorithm = str(summary.get("algorithm", "Run"))
            problem = str(summary.get("problem", "problem")).upper()
            title = f"{algorithm} on {completed.shape[1]}-objective {problem}"
        else:
            title = "Saved Pareto archive"
    if evaluations_path:
        evaluations, evaluation_fe = _evaluations_from_file(evaluations_path)
        evaluation_sources = None
    elif saved_stage1_f is not None and len(saved_stage1_f):
        # PlatEMO's Evaluation hook records every Stage 1 true evaluation.
        evaluations = saved_stage1_f
        evaluation_fe = saved_stage1_fe
        evaluation_sources = None
    else:
        evaluations = base
        evaluation_fe = None
        evaluation_sources = None
    output_dir.mkdir(parents=True, exist_ok=True)
    is_ea_only = summary.get("evaluation_variant") == "ea_only"
    if is_ea_only:
        for stale_name in (
            "pf_report.png",
            "pf_comparison.png",
            "preference_coverage.png",
            "gap_reasons.png",
            "training_loss.png",
        ):
            (output_dir / stale_name).unlink(missing_ok=True)
        _plot_final_archive(
            base,
            output_dir,
            title=f"{title}: final nondominated population",
        )
        cumulative_archive = _cumulative_nondominated_front(evaluations)
        _plot_ea_archive_report(
            archive=cumulative_archive,
            output=output_dir,
            title=(
                f"{title}: cumulative empirical PF from "
                f"{len(evaluations):,} true evaluations"
            ),
        )
        _plot_ea_search_history(
            evaluations=evaluations,
            evaluation_fe=evaluation_fe,
            archive=base,
            output=output_dir,
            title=f"{title}: EA-only search history",
        )
        return sorted(output_dir.glob("*.png"))

    for stale_name in ("pf_report.png", "pf_comparison.png"):
        (output_dir / stale_name).unlink(missing_ok=True)
    # The complete archive keeps every nondominated model candidate for fair
    # metrics. The saved verified mask separately identifies true hole hits.
    ea_archive = base
    completed_archive = completed
    archive_sources = sources
    gap_info = summary.get("gap_info", {}) if isinstance(summary, dict) else {}
    ea_fe = int(gap_info.get("ea_fe_used", len(evaluations)))
    model_fe = int(gap_info.get("model_fe_used", len(model_f)))
    total_fe = ea_fe + model_fe
    _plot_final_archive(
        completed_archive,
        output_dir,
        title=f"{title}: cumulative EA + model archive",
    )
    comparison_title = "Cumulative empirical PF archive: EA allocation vs total-budget EA + model"
    if completed_archive.shape[1] == 3:
        _plot_three_objective_report(
            base_archive=ea_archive,
            completed_front=completed_archive,
            completed_sources=archive_sources,
            output=output_dir,
            title=comparison_title,
            ea_fe=ea_fe,
            total_fe=total_fe,
            completed_verified_hole=completed_verified_hole,
            completed_boundary_probe=completed_boundary_probe,
        )
    else:
        _plot_many_objective_report(
            base_archive=ea_archive,
            completed_front=completed_archive,
            completed_sources=archive_sources,
            output=output_dir,
            title=comparison_title,
            ea_fe=ea_fe,
            total_fe=total_fe,
            completed_verified_hole=completed_verified_hole,
            completed_boundary_probe=completed_boundary_probe,
        )
    _plot_preferences(observed, selected if len(selected) else gap, unfilled, output_dir)
    _plot_reasons(gap_rows, output_dir)
    _plot_training_loss(run_dir, output_dir)
    return sorted(output_dir.glob("*.png"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved PF completion and hole evidence.")
    parser.add_argument("--run-dir", required=True, type=Path, help="One completed run directory containing fronts.npz.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated PNG files (default: run-dir/figures).")
    parser.add_argument(
        "--evaluations-csv",
        type=Path,
        default=None,
        help="Optional CSV/NPZ containing all evaluated objectives and an optional FE column.",
    )
    parser.add_argument("--title", default=None, help="Optional title for the left Pareto-front panel.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    destination = arguments.output_dir or arguments.run_dir / "figures"
    files = visualize_run(
        arguments.run_dir,
        destination,
        arguments.evaluations_csv,
        arguments.title,
    )
    print(f"Generated {len(files)} figure(s) in {destination}")
