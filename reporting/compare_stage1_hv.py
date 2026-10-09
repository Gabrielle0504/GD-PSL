"""Compare Stage 1 GD-PSL allocation ratios against equal-budget pure EAs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon

from gd_psl.config import ALLOWED_EA_FILL_SPLITS


RATIOS = tuple(ALLOWED_EA_FILL_SPLITS)
ALGORITHMS = ("NSGAII", "NSGAIII", "MOEAD")
ALGORITHM_LABELS = {
    "NSGAII": "NSGA-II",
    "NSGAIII": "NSGA-III",
    "MOEAD": "MOEA/D",
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


def _ratio_directory(ratio: str) -> str:
    return f"ratio_{ratio.replace(':', '_')}"


def _load_hv(run_dir: Path) -> float:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    value = float(summary["hypervolume"]["completed_hv"])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite HV in {summary_path}")
    return value


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    adjusted = np.ones(len(p_values), dtype=float)
    running_maximum = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted.tolist()


def _summarize(values: np.ndarray) -> tuple[float, float, float]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def _load_records(
    stage1_root: Path,
    algorithms: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
) -> tuple[list[dict], dict[tuple[str, str, str], dict[int, float]]]:
    grouped: dict[tuple[str, str, str], dict[int, float]] = {}
    rows: list[dict] = []
    variants = ("pure_ea", *RATIOS)
    for algorithm in algorithms:
        for problem in problems:
            for variant in variants:
                if variant == "pure_ea":
                    problem_dir = (
                        stage1_root / "pure_ea" / f"EA_{algorithm}" / problem
                    )
                else:
                    problem_dir = (
                        stage1_root
                        / _ratio_directory(variant)
                        / f"GD-PSL_{algorithm}"
                        / problem
                    )
                values = {}
                for seed in seeds:
                    run_dir = problem_dir / f"seed_{seed:03d}"
                    if not (run_dir / "summary.json").is_file():
                        raise FileNotFoundError(f"Missing completed run: {run_dir}")
                    values[seed] = _load_hv(run_dir)
                grouped[(algorithm, problem, variant)] = values
                array = np.asarray([values[seed] for seed in seeds])
                q1, median, q3 = _summarize(array)
                rows.append(
                    {
                        "algorithm": algorithm,
                        "algorithm_label": ALGORITHM_LABELS[algorithm],
                        "problem": problem,
                        "variant": variant,
                        "runs": len(seeds),
                        "median_hv": median,
                        "q1_hv": q1,
                        "q3_hv": q3,
                        "iqr_hv": q3 - q1,
                    }
                )
    return rows, grouped


def _paired_tests(
    grouped: dict[tuple[str, str, str], dict[int, float]],
    algorithms: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
    alpha: float,
) -> list[dict]:
    rows = []
    for ratio in RATIOS:
        ratio_rows = []
        for algorithm in algorithms:
            for problem in problems:
                pure = grouped[(algorithm, problem, "pure_ea")]
                candidate = grouped[(algorithm, problem, ratio)]
                pure_values = np.asarray([pure[seed] for seed in seeds])
                candidate_values = np.asarray([candidate[seed] for seed in seeds])
                differences = candidate_values - pure_values
                relative = 100.0 * differences / np.maximum(
                    np.abs(pure_values), np.finfo(float).tiny
                )
                if np.all(np.abs(differences) <= 1e-15):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(differences, alternative="two-sided").pvalue
                    )
                ratio_rows.append(
                    {
                        "algorithm": algorithm,
                        "algorithm_label": ALGORITHM_LABELS[algorithm],
                        "problem": problem,
                        "ea_fill_ratio": ratio,
                        "paired_runs": len(seeds),
                        "pure_ea_median_hv": float(np.median(pure_values)),
                        "gd_psl_median_hv": float(np.median(candidate_values)),
                        "median_paired_hv_difference": float(np.median(differences)),
                        "median_relative_hv_change_percent": float(np.median(relative)),
                        "wins": int(np.count_nonzero(differences > 1e-12)),
                        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                        "losses": int(np.count_nonzero(differences < -1e-12)),
                        "wilcoxon_p": p_value,
                    }
                )
        adjusted = _holm_adjust([row["wilcoxon_p"] for row in ratio_rows])
        for row, adjusted_p in zip(ratio_rows, adjusted):
            row["holm_adjusted_p"] = adjusted_p
            row["significantly_worse_than_pure_ea"] = bool(
                adjusted_p < alpha and row["median_paired_hv_difference"] < 0.0
            )
        rows.extend(ratio_rows)
    return rows


def _overview(summary_rows: Sequence[dict], test_rows: Sequence[dict]) -> list[dict]:
    summary_lookup = {
        (row["algorithm"], row["problem"], row["variant"]): row
        for row in summary_rows
    }
    ranks: dict[str, list[float]] = defaultdict(list)
    for algorithm in ALGORITHMS:
        for problem in PROBLEMS:
            medians = np.asarray(
                [summary_lookup[(algorithm, problem, ratio)]["median_hv"] for ratio in RATIOS]
            )
            for ratio, rank in zip(RATIOS, rankdata(-medians, method="average")):
                ranks[ratio].append(float(rank))
    output = []
    for ratio in RATIOS:
        selected = [row for row in test_rows if row["ea_fill_ratio"] == ratio]
        relative = np.asarray(
            [row["median_relative_hv_change_percent"] for row in selected]
        )
        output.append(
            {
                "ea_fill_ratio": ratio,
                "tasks": len(selected),
                "mean_hv_rank_among_ratios": float(np.mean(ranks[ratio])),
                "median_relative_hv_change_percent": float(np.median(relative)),
                "minimum_relative_hv_change_percent": float(np.min(relative)),
                "maximum_relative_hv_change_percent": float(np.max(relative)),
                "tasks_with_higher_median_hv": sum(
                    row["median_paired_hv_difference"] > 1e-12 for row in selected
                ),
                "tasks_with_equal_median_hv": sum(
                    abs(row["median_paired_hv_difference"]) <= 1e-12 for row in selected
                ),
                "tasks_with_lower_median_hv": sum(
                    row["median_paired_hv_difference"] < -1e-12 for row in selected
                ),
                "significantly_worse_tasks_after_holm": sum(
                    row["significantly_worse_than_pure_ea"] for row in selected
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _heatmap(
    rows: Sequence[dict],
    algorithms: Sequence[str],
    problems: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    lookup = {
        (row["algorithm"], row["problem"], row["ea_fill_ratio"]): row
        for row in rows
    }
    labels = [
        f"{ALGORITHM_LABELS[algorithm]} | {problem}"
        for algorithm in algorithms
        for problem in problems
    ]
    matrix = np.asarray(
        [
            [
                lookup[(algorithm, problem, ratio)][
                    "median_relative_hv_change_percent"
                ]
                for ratio in RATIOS
            ]
            for algorithm in algorithms
            for problem in problems
        ]
    )
    significant = np.asarray(
        [
            [
                lookup[(algorithm, problem, ratio)][
                    "significantly_worse_than_pure_ea"
                ]
                for ratio in RATIOS
            ]
            for algorithm in algorithms
            for problem in problems
        ],
        dtype=bool,
    )
    color_limit = max(0.01, float(np.quantile(np.abs(matrix), 0.98)))
    figure, axis = plt.subplots(figsize=(10.5, 14.5), facecolor="white")
    image = axis.imshow(
        matrix,
        cmap="RdBu",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="auto",
    )
    axis.set_xticks(range(len(RATIOS)), [f"EA:FILL {ratio}" for ratio in RATIOS])
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(axis="x", labelrotation=25, labelsize=9)
    axis.tick_params(axis="y", labelsize=8)
    for row_index in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row_index, column]
            marker = "*" if significant[row_index, column] else ""
            color = "white" if abs(value) > 0.55 * color_limit else "#202020"
            axis.text(
                column,
                row_index,
                f"{value:+.3f}%{marker}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
                fontweight="bold" if marker else "normal",
            )
    axis.set_title(
        "Stage 1 median HV change relative to equal-budget pure EA\n"
        "positive is better; * significantly worse after per-ratio Holm correction",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Median paired HV change (%)")
    figure.tight_layout()
    for extension in formats:
        kwargs = {"dpi": 180} if extension == "png" else {}
        figure.savefig(output_dir / f"hv_relative_change_heatmap.{extension}", **kwargs)
    plt.close(figure)


def _distribution_plot(
    rows: Sequence[dict], output_dir: Path, formats: Sequence[str]
) -> None:
    values = [
        [
            row["median_relative_hv_change_percent"]
            for row in rows
            if row["ea_fill_ratio"] == ratio
        ]
        for ratio in RATIOS
    ]
    figure, axis = plt.subplots(figsize=(9.5, 5.5), facecolor="white")
    boxes = axis.boxplot(values, labels=RATIOS, patch_artist=True, widths=0.58)
    for box in boxes["boxes"]:
        box.set_facecolor("#d9e6f2")
        box.set_edgecolor("#325d79")
    axis.axhline(0.0, color="#b22222", linewidth=1.0, linestyle="--")
    axis.set_xlabel("EA:FILL ratio")
    axis.set_ylabel("Median paired HV change relative to pure EA (%)")
    axis.set_title(
        "Stage 1 HV safeguard across 27 problem/base-EA tasks",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    for extension in formats:
        kwargs = {"dpi": 180} if extension == "png" else {}
        figure.savefig(output_dir / f"hv_relative_change_distribution.{extension}", **kwargs)
    plt.close(figure)


def generate_hv_comparison(
    stage1_root: Path,
    output_dir: Path,
    algorithms: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
    alpha: float,
    formats: Sequence[str],
) -> list[dict]:
    summary_rows, grouped = _load_records(
        stage1_root, algorithms, problems, seeds
    )
    test_rows = _paired_tests(grouped, algorithms, problems, seeds, alpha)
    overview_rows = _overview(summary_rows, test_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "hv_summary.csv", summary_rows)
    _write_csv(output_dir / "hv_paired_tests.csv", test_rows)
    _write_csv(output_dir / "hv_ratio_overview.csv", overview_rows)
    _heatmap(test_rows, algorithms, problems, output_dir, formats)
    _distribution_plot(test_rows, output_dir, formats)
    return overview_rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-root", type=Path, default=Path("results/stage1")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <stage1-root>/comparisons/hv_vs_pure_ea",
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
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"]
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_args(argv)
    output_dir = arguments.output_dir or (
        arguments.stage1_root / "comparisons" / "hv_vs_pure_ea"
    )
    overview = generate_hv_comparison(
        arguments.stage1_root,
        output_dir,
        arguments.algorithms,
        arguments.problems,
        arguments.seeds,
        arguments.alpha,
        arguments.formats,
    )
    print(
        f"Generated Stage 1 HV comparison for {len(overview)} ratios under {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
