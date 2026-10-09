"""Generate the standard visualization report for every saved run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

from .visualize import visualize_run


def visualize_batch(input_roots: Sequence[Path], output_name: str = "figures") -> list[dict]:
    """Scan result roots and render only directories containing ``fronts.npz``."""
    run_dirs = sorted({path.parent for root in input_roots for path in root.rglob("fronts.npz")})
    report: list[dict] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
        output_dir = run_dir / output_name
        generated = visualize_run(run_dir, output_dir)
        report.append(
            {
                "run_dir": str(run_dir),
                "problem": summary.get("problem", ""),
                "algorithm": summary.get("algorithm", ""),
                "base_nondominated": summary.get("base_nondominated", ""),
                "model_candidates": summary.get("model_candidates", ""),
                "completed_nondominated": summary.get("completed_nondominated", ""),
                "generated_figures": len(generated),
            }
        )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render reports for all saved fill-then-judge runs.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True, help="Result roots to scan recursively.")
    parser.add_argument("--output-name", default="figures", help="Subdirectory for generated figures.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional batch summary CSV path.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    report = visualize_batch(arguments.roots, arguments.output_name)
    if arguments.summary_csv:
        arguments.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(report[0]) if report else ["run_dir"]
        with arguments.summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report)
    print(f"Generated reports for {len(report)} run(s).")
