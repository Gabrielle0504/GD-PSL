"""Visual evidence for detection, query roles, diagnostics, and gap contraction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

BLUE = "#2166d1"
GREEN = "#17823b"
ORANGE = "#e66101"
PURPLE = "#7b3294"
GRAY = "#777777"
SQRT3 = np.sqrt(3.0)


def simplex_xy(preferences: np.ndarray) -> np.ndarray:
    values = np.asarray(preferences, dtype=float)
    return np.column_stack((values[:, 1] + 0.5 * values[:, 2], SQRT3 * values[:, 2] / 2.0))


def _read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _vectors(rows: list[dict], prefix: str, dimensions: int) -> np.ndarray:
    output = []
    for row in rows:
        values = [row.get(f"{prefix}_{index + 1}", "") for index in range(dimensions)]
        if all(value != "" for value in values):
            output.append([float(value) for value in values])
    return np.asarray(output, dtype=float).reshape(-1, dimensions)


def _sample_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.sort(np.random.default_rng(seed).choice(length, maximum, replace=False))


def _style_simplex(axis) -> None:
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2.0], [0.0, 0.0]])
    axis.plot(triangle[:, 0], triangle[:, 1], color="#333333", lw=0.8)
    axis.text(-0.025, -0.035, r"$p_1$", fontsize=8)
    axis.text(1.008, -0.035, r"$p_2$", fontsize=8)
    axis.text(0.495, SQRT3 / 2.0 + 0.018, r"$p_3$", fontsize=8, ha="center")
    axis.set_xlim(-0.04, 1.04)
    axis.set_ylim(-0.05, SQRT3 / 2.0 + 0.05)
    axis.set_aspect("equal")
    axis.axis("off")


def create_figure(run_dir: Path, output: Path) -> dict:
    with np.load(run_dir / "fronts.npz") as data:
        observed = np.asarray(data["observed_preferences"], dtype=float)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    gap_threshold = float(summary["gap_info"]["gap_threshold"])
    records = _read_records(run_dir / "gap_regions.csv")
    selected = [row for row in records if row["selected_for_model"].lower() == "true"]
    dimensions = observed.shape[1]
    if dimensions != 3:
        raise ValueError("The evidence figure currently requires three objectives")

    candidate_preferences = _vectors(records, "preference", dimensions)
    initial_distance = np.asarray([float(row["initial_distance"]) for row in records])
    support = np.asarray([float(row["support_score"]) for row in records])
    target_preferences = _vectors(selected, "preference", dimensions)
    actual_preferences = _vectors(selected, "actual_preference", dimensions)
    roles = np.asarray(
        [
            "low_support_boundary_probe"
            if row.get("allocation_phase") == "boundary_probe"
            else (
                "geometric_hole_query"
                if float(row["initial_distance"]) > gap_threshold
                else "coverage_refinement_query"
            )
            for row in selected
        ],
        dtype=object,
    )
    empirical_pass = np.asarray(
        [row["reason_code"] in {"model_nondominated", "model_dominated", "model_duplicate"} for row in selected]
    )
    retained_in_saved_archive = np.asarray(
        [row.get("archive_retained", "").strip().lower() == "true" for row in selected]
    )
    retained = retained_in_saved_archive

    observed_xy = simplex_xy(observed)
    candidates_xy = simplex_xy(candidate_preferences)
    targets_xy = simplex_xy(target_preferences)
    actual_xy = simplex_xy(actual_preferences)
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 6.25), facecolor="white")

    sample = _sample_indices(len(candidates_xy), 8000, 31)
    scatter = axes[0, 0].scatter(
        candidates_xy[sample, 0], candidates_xy[sample, 1],
        c=initial_distance[sample], cmap="viridis", s=3, alpha=0.55,
        linewidths=0, rasterized=True,
    )
    archive_sample = _sample_indices(len(observed_xy), 2500, 32)
    axes[0, 0].scatter(
        observed_xy[archive_sample, 0], observed_xy[archive_sample, 1],
        s=2, c="#202020", alpha=0.35, linewidths=0, rasterized=True,
    )
    figure.colorbar(scatter, ax=axes[0, 0], fraction=0.042, pad=0.01, label="Nearest-archive distance")
    _style_simplex(axes[0, 0])
    axes[0, 0].set_title("(a) Runtime sparsity evidence", fontsize=9, fontweight="bold")

    axes[0, 1].scatter(
        observed_xy[archive_sample, 0], observed_xy[archive_sample, 1],
        s=2, c=GRAY, alpha=0.22, linewidths=0, rasterized=True,
    )
    role_style = {
        "geometric_hole_query": ("o", BLUE, "Geometric hole"),
        "coverage_refinement_query": ("x", PURPLE, "Coverage refinement"),
        "low_support_boundary_probe": ("^", ORANGE, "Boundary probe"),
        # Compatibility with artifacts created before explicit role naming.
        "hole_fill": ("o", BLUE, "Geometric/support query"),
    }
    for role in np.unique(roles):
        mask = roles == role
        marker, color, label = role_style.get(role, (".", "#333333", role))
        indices = _sample_indices(int(np.count_nonzero(mask)), 1400, 40 + len(label))
        points = targets_xy[mask][indices]
        axes[0, 1].scatter(
            points[:, 0], points[:, 1], marker=marker, c=color, s=10,
            alpha=0.60, linewidths=0.25, label=label, rasterized=True,
        )
    _style_simplex(axes[0, 1])
    axes[0, 1].set_title("(b) Selected target preferences", fontsize=9, fontweight="bold")
    axes[0, 1].legend(loc="upper left", fontsize=6.4, frameon=False, ncol=1)

    outcome_groups = [
        (empirical_pass & retained, GREEN, "Diagnostic pass, retained"),
        (~empirical_pass & retained, "#d62728", "Diagnostic flag, retained"),
        (empirical_pass & ~retained, GRAY, "Diagnostic pass, not retained"),
        (~empirical_pass & ~retained, ORANGE, "Diagnostic flag, not retained"),
    ]
    for group_index, (mask, color, label) in enumerate(outcome_groups):
        indices = _sample_indices(int(np.count_nonzero(mask)), 1800, 60 + group_index)
        points = actual_xy[mask][indices]
        axes[1, 0].scatter(
            points[:, 0], points[:, 1], s=6, c=color, alpha=0.52,
            linewidths=0, label=label, rasterized=True,
        )
    _style_simplex(axes[1, 0])
    axes[1, 0].set_title("(c) True-evaluated query outcomes", fontsize=9, fontweight="bold")
    axes[1, 0].legend(loc="upper left", fontsize=6.3, frameon=False)

    local_targets = target_preferences[roles != "low_support_boundary_probe"]
    before = cKDTree(observed).query(local_targets, k=1)[0]
    extreme_limit = float(summary["gap_info"].get("extreme_gap_limit", np.inf))
    local_envelope = before <= extreme_limit
    local_targets = local_targets[local_envelope]
    before = before[local_envelope]
    retained_preferences = actual_preferences[retained]
    after_support = np.vstack((observed, retained_preferences))
    after = cKDTree(after_support).query(local_targets, k=1)[0]
    for values, color, label in (
        (before, BLUE, "Before completion"),
        (after, GREEN, "After archive update"),
    ):
        ordered = np.sort(values)
        survival = 1.0 - np.arange(1, len(ordered) + 1) / len(ordered)
        axes[1, 1].plot(ordered, survival, color=color, lw=1.7, label=label)
    axes[1, 1].set_xlabel("Nearest-archive distance at queried supported targets", fontsize=7)
    axes[1, 1].set_ylabel("Fraction of queries above distance", fontsize=7)
    axes[1, 1].tick_params(labelsize=7)
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.83))
    axes[1, 1].set_title("(d) Supported-gap contraction", fontsize=9, fontweight="bold")
    axes[1, 1].text(
        0.98, 0.96,
        f"max: {before.max():.3f} $\\rightarrow$ {after.max():.3f}\n"
        f"95th pct.: {np.quantile(before, .95):.3f} $\\rightarrow$ {np.quantile(after, .95):.3f}",
        transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=7,
    )

    figure.subplots_adjust(left=0.04, right=0.985, bottom=0.055, top=0.95, wspace=0.16, hspace=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    metrics = {
        "problem": run_dir.parent.name,
        "seed": int(run_dir.name.removeprefix("seed_")),
        "queries": len(selected),
        "validation_passed": int(np.count_nonzero(empirical_pass)),
        "validation_rejected": int(np.count_nonzero(~empirical_pass)),
        "archive_retained": int(np.count_nonzero(retained)),
        "diagnostic_pass_and_retained": int(
            np.count_nonzero(empirical_pass & retained)
        ),
        "diagnostic_flag_and_retained": int(
            np.count_nonzero(~empirical_pass & retained)
        ),
        "supported_max_before": float(before.max()),
        "supported_max_after": float(after.max()),
        "supported_p95_before": float(np.quantile(before, 0.95)),
        "supported_p95_after": float(np.quantile(after, 0.95)),
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/paper_figures/gap_evidence.pdf"))
    args = parser.parse_args()
    print(json.dumps(create_figure(args.run_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
