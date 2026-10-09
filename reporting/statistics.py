"""Aggregate and statistically compare saved equal-budget hypervolume results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

from evaluation_metrics import hypervolume, normalize_objectives


def _cumulative_archive(objectives: np.ndarray) -> np.ndarray:
    """Return the unique empirical minimization PF across all supplied evaluations."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("Saved objective evaluations must be a matrix")
    values = values[np.isfinite(values).all(axis=1)]
    values = np.unique(values, axis=0)
    keep = np.ones(len(values), dtype=bool)
    for start in range(0, len(values), 256):
        chunk = values[start : start + 256]
        dominates = np.all(values[None, :, :] <= chunk[:, None, :], axis=2) & np.any(
            values[None, :, :] < chunk[:, None, :], axis=2
        )
        keep[start : start + len(chunk)] = ~dominates.any(axis=1)
    return values[keep]


def _recalculate_archive_hv(summary_path: Path, hv: dict) -> Optional[dict]:
    """Recompute HV from cumulative true-evaluation archives when artifacts allow it."""
    artifact_path = summary_path.parent / "fronts.npz"
    required = ("normalization_ideal", "normalization_nadir", "reference_point")
    if not artifact_path.exists() or any(hv.get(key) is None for key in required):
        return None
    with np.load(artifact_path) as data:
        if "stage1_f" not in data:
            return None
        stage1 = np.asarray(data["stage1_f"], dtype=float)
        model = (
            np.asarray(data["model_f"], dtype=float)
            if "model_f" in data
            else np.empty((0, stage1.shape[1]), dtype=float)
        )
    ea_archive = _cumulative_archive(stage1)
    completed_archive = _cumulative_archive(np.vstack((stage1, model)))
    ideal = np.asarray(hv["normalization_ideal"], dtype=float)
    nadir = np.asarray(hv["normalization_nadir"], dtype=float)
    reference = np.asarray(hv["reference_point"], dtype=float)
    samples = int(hv.get("samples") or 131072)
    seed = int(hv.get("seed") or 2026)
    base_hv, method = hypervolume(
        normalize_objectives(ea_archive, ideal, nadir), reference, samples, seed
    )
    completed_hv, completed_method = hypervolume(
        normalize_objectives(completed_archive, ideal, nadir), reference, samples, seed
    )
    if completed_method != method:
        raise RuntimeError("Inconsistent HV methods for cumulative archives")
    return {
        "base_stage_hv": float(base_hv),
        "completed_hv": float(completed_hv),
        "delta_hv": float(completed_hv - base_hv),
        "method": method,
        "ea_archive_size": int(len(ea_archive)),
        "completed_archive_size": int(len(completed_archive)),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(roots: Sequence[Path]) -> list[dict]:
    records = []
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("summary.json")
        for path in paths:
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            hv = summary.get("hypervolume", {})
            gap = summary.get("gap_info", {})
            runtime = summary.get("runtime_seconds", {})
            if hv.get("completed_hv") is None:
                continue
            recalculated = _recalculate_archive_hv(path, hv)
            reported_hv = recalculated or hv
            reference = hv.get("reference_point", [])
            protocol_payload = {
                key: hv.get(key)
                for key in (
                    "normalization_ideal",
                    "normalization_nadir",
                    "reference_point",
                    "method",
                    "samples",
                    "seed",
                )
            }
            protocol_json = json.dumps(protocol_payload, sort_keys=True, separators=(",", ":"))
            protocol_id = hashlib.sha256(protocol_json.encode("utf-8")).hexdigest()[:12]
            records.append(
                {
                    "problem": str(summary["problem"]).lower(),
                    "objectives": len(reference),
                    "algorithm": str(summary["algorithm"]),
                    "method": str(
                        summary.get(
                            "evaluation_variant",
                            summary.get("completion_method", "unknown"),
                        )
                    ),
                    "run": int(summary["run"]),
                    "run_seed": int(
                        summary.get(
                            "run_seed",
                            summary.get("experiment_config", {}).get("seed", 1)
                            + int(summary["run"]) - 1,
                        )
                    ),
                    "common_total_fe_budget": int(gap.get("common_total_fe_budget", -1)),
                    "completed_hv": float(reported_hv["completed_hv"]),
                    "base_stage_hv": float(reported_hv.get("base_stage_hv", np.nan)),
                    "delta_hv": float(reported_hv.get("delta_hv", np.nan)),
                    "hv_method": str(reported_hv.get("method", "unknown")),
                    "hv_archive_definition": (
                        "cumulative_empirical_pf_from_true_evaluations"
                        if recalculated is not None
                        else "saved_summary_fallback"
                    ),
                    "ea_archive_size": (
                        int(recalculated["ea_archive_size"])
                        if recalculated is not None
                        else ""
                    ),
                    "completed_archive_size": (
                        int(recalculated["completed_archive_size"])
                        if recalculated is not None
                        else ""
                    ),
                    "hv_protocol_id": protocol_id,
                    "ea_seconds": float(runtime.get("ea_search_and_validation", np.nan)),
                    "model_training_seconds": float(runtime.get("model_training", np.nan)),
                    "method_total_seconds": float(
                        runtime.get("method_execution_total", runtime.get("total", np.nan))
                    ),
                    "summary_path": str(path),
                }
            )
    return records


def summarize_runs(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        key = (
            record["problem"],
            record["objectives"],
            record["common_total_fe_budget"],
            record["hv_protocol_id"],
            record["algorithm"],
            record["method"],
        )
        groups[key].append(record)
    rows = []
    for key, group_records in sorted(groups.items()):
        values_array = np.asarray(
            [record["completed_hv"] for record in group_records], dtype=float
        )
        ea_times = np.asarray([record["ea_seconds"] for record in group_records], dtype=float)
        training_times = np.asarray(
            [record["model_training_seconds"] for record in group_records], dtype=float
        )
        total_times = np.asarray(
            [record["method_total_seconds"] for record in group_records], dtype=float
        )
        q1, q3 = np.quantile(values_array, [0.25, 0.75])
        rows.append(
            {
                "problem": key[0],
                "objectives": key[1],
                "common_total_fe_budget": key[2],
                "hv_protocol_id": key[3],
                "algorithm": key[4],
                "method": key[5],
                "runs": len(values_array),
                "median_hv": float(np.median(values_array)),
                "q1_hv": float(q1),
                "q3_hv": float(q3),
                "iqr_hv": float(q3 - q1),
                "median_ea_seconds": float(np.nanmedian(ea_times)),
                "median_model_training_seconds": float(np.nanmedian(training_times)),
                "median_method_total_seconds": float(np.nanmedian(total_times)),
            }
        )
    return rows


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def compare_to_baseline(
    records: list[dict],
    baseline_algorithm: str,
    baseline_method: str,
    alpha: float,
) -> list[dict]:
    groups = defaultdict(dict)
    for record in records:
        key = (
            record["problem"],
            record["objectives"],
            record["common_total_fe_budget"],
            record["hv_protocol_id"],
            record["algorithm"],
            record["method"],
        )
        groups[key][record["run_seed"]] = record["completed_hv"]

    pending = []
    for candidate_key, candidate_runs in sorted(groups.items()):
        problem, objectives, budget, protocol_id, algorithm, method = candidate_key
        if algorithm == baseline_algorithm and method == baseline_method:
            continue
        baseline_key = (
            problem,
            objectives,
            budget,
            protocol_id,
            baseline_algorithm,
            baseline_method,
        )
        baseline_runs = groups.get(baseline_key)
        if not baseline_runs:
            continue
        paired_ids = sorted(set(candidate_runs) & set(baseline_runs))
        if not paired_ids:
            continue
        candidate = np.asarray([candidate_runs[index] for index in paired_ids])
        baseline = np.asarray([baseline_runs[index] for index in paired_ids])
        differences = candidate - baseline
        wins = int(np.count_nonzero(differences > 1e-12))
        losses = int(np.count_nonzero(differences < -1e-12))
        ties = len(differences) - wins - losses
        if np.all(np.abs(differences) <= 1e-12):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
        pending.append(
            {
                "problem": problem,
                "objectives": objectives,
                "common_total_fe_budget": budget,
                "hv_protocol_id": protocol_id,
                "baseline_algorithm": baseline_algorithm,
                "baseline_method": baseline_method,
                "candidate_algorithm": algorithm,
                "candidate_method": method,
                "paired_runs": len(paired_ids),
                "candidate_median_hv": float(np.median(candidate)),
                "baseline_median_hv": float(np.median(baseline)),
                "median_paired_difference": float(np.median(differences)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_p": p_value,
            }
        )
    adjusted = _holm_adjust([row["wilcoxon_p"] for row in pending])
    for row, adjusted_p in zip(pending, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        difference = row["median_paired_difference"]
        if adjusted_p < alpha and difference > 0:
            conclusion = "candidate_statistically_better"
        elif adjusted_p < alpha and difference < 0:
            conclusion = "candidate_statistically_worse"
        else:
            conclusion = "no_significant_difference"
        row["conclusion"] = conclusion
    return pending


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed-archive HV using paired runs and Holm correction."
    )
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-algorithm", required=True)
    parser.add_argument("--baseline-method", default="fill_then_judge")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("hv_comparison"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_args(argv)
    if not 0 < arguments.alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    records = load_runs(arguments.roots)
    if not records:
        raise ValueError("No summary.json files containing hypervolume results were found")
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    run_fields = list(records[0])
    _write_csv(output / "hypervolume_runs.csv", records, run_fields)
    summaries = summarize_runs(records)
    _write_csv(output / "hypervolume_summary.csv", summaries, list(summaries[0]))
    comparisons = compare_to_baseline(
        records,
        arguments.baseline_algorithm,
        arguments.baseline_method,
        arguments.alpha,
    )
    comparison_fields = [
        "problem", "objectives", "common_total_fe_budget", "hv_protocol_id", "baseline_algorithm",
        "baseline_method", "candidate_algorithm", "candidate_method", "paired_runs",
        "candidate_median_hv", "baseline_median_hv", "median_paired_difference",
        "wins", "ties", "losses", "wilcoxon_p", "holm_adjusted_p", "conclusion",
    ]
    _write_csv(output / "hypervolume_comparison.csv", comparisons, comparison_fields)
    print(f"Read {len(records)} runs; wrote HV reports to {output}")


if __name__ == "__main__":
    main()
