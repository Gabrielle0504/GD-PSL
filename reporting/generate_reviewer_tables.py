"""Generate readable LaTeX tables from the formal Stage-B results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
    "GD-PSL_NSGAII": r"\gdpsl{} + NSGA-II",
    "GD-PSL_NSGAIII": r"\gdpsl{} + NSGA-III",
    "GD-PSL_MOEAD": r"\gdpsl{} + MOEA/D",
    "PangMOEAD": "PangMOEAD",
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
PROBLEM_LABELS = {problem: problem.split("_")[0] for problem in PROBLEMS}


def _read(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["method"], row["problem"]): row for row in rows}


def _cell(
    row: dict[str, str],
    median_key: str,
    q1_key: str,
    q3_key: str,
    precision: int,
    bold: bool,
) -> str:
    median = float(row[median_key])
    q1 = float(row[q1_key])
    q3 = float(row[q3_key])
    median_text = f"{median:.{precision}f}"
    iqr_text = f"{q3 - q1:.{precision}f}"
    if bold:
        median_text = rf"\textbf{{{median_text}}}"
    return f"{median_text} ({iqr_text})"


def _stacked_cell(
    row: dict[str, str],
    median_key: str,
    q1_key: str,
    q3_key: str,
    precision: int,
    bold: bool,
) -> str:
    """Place median and IQR on separate lines to preserve a readable font size."""
    median = float(row[median_key])
    q1 = float(row[q1_key])
    q3 = float(row[q3_key])
    median_text = f"{median:.{precision}f}"
    if bold:
        median_text = rf"\textbf{{{median_text}}}"
    iqr_text = f"{q3 - q1:.{precision}f}"
    return rf"\shortstack{{{median_text}\\({iqr_text})}}"


def _table(
    lookup: dict[tuple[str, str], dict[str, str]],
    *,
    caption: str,
    label: str,
    median_key: str,
    q1_key: str,
    q3_key: str,
    higher_better: bool,
    precision: int,
) -> str:
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Problem & \shortstack{NSGA-II\\median (IQR)} & "
        r"\shortstack{NSGA-III\\median (IQR)} & "
        r"\shortstack{MOEA/D\\median (IQR)} & "
        r"\shortstack{PangMOEAD\\median (IQR)}\\",
        r"\midrule",
    ]
    best: dict[str, float] = {}
    for problem in PROBLEMS:
        values = [float(lookup[(method, problem)][median_key]) for method in METHODS]
        best[problem] = (max if higher_better else min)(values)
    baseline_methods = ("EA_NSGAII", "EA_NSGAIII", "EA_MOEAD", "PangMOEAD")
    for problem in PROBLEMS:
        cells = []
        for method in baseline_methods:
            row = lookup[(method, problem)]
            median = float(row[median_key])
            cells.append(
                _cell(row, median_key, q1_key, q3_key, precision, median == best[problem])
            )
        lines.append(PROBLEM_LABELS[problem] + " & " + " & ".join(cells) + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
            r"\begin{tabular}{cccc}",
            r"\toprule",
            r"Problem & \shortstack{\gdpsl{} + NSGA-II\\median (IQR)} & "
            r"\shortstack{\gdpsl{} + NSGA-III\\median (IQR)} & "
            r"\shortstack{\gdpsl{} + MOEA/D\\median (IQR)}\\",
            r"\midrule",
        ]
    )
    gd_methods = ("GD-PSL_NSGAII", "GD-PSL_NSGAIII", "GD-PSL_MOEAD")
    for problem in PROBLEMS:
        cells = []
        for method in gd_methods:
            row = lookup[(method, problem)]
            median = float(row[median_key])
            cells.append(
                _cell(row, median_key, q1_key, q3_key, precision, median == best[problem])
            )
        lines.append(PROBLEM_LABELS[problem] + " & " + " & ".join(cells) + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _compact_table(lookup: dict[tuple[str, str], dict[str, str]]) -> str:
    specifications = (
        ("Hypervolume $\\uparrow$", "median_hv", "q1_hv", "q3_hv", True, 5, True),
        (
            "$\\igdinf{}$ $\\downarrow$",
            "median_igd_infinity",
            "q1_igd_infinity",
            "q3_igd_infinity",
            False,
            5,
            True,
        ),
        (
            "Runtime (s) $\\downarrow$",
            "median_runtime_seconds",
            "q1_runtime_seconds",
            "q3_runtime_seconds",
            False,
            1,
            False,
        ),
    )
    lines = [
        r"\begin{table}[p]",
        r"\caption{Complete Stage-B results over 20 paired seeds: median above IQR for coverage, and median (IQR) for runtime. Bold marks the best unrounded median. GD-N2, GD-N3, and GD-MD denote \gdpsl{} with NSGA-II, NSGA-III, and MOEA/D.}",
        r"\label{tab:complete-results}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"Problem & NSGA-II & NSGA-III & MOEA/D & GD-N2 & GD-N3 & GD-MD & PangMOEAD\\",
    ]
    for section, median_key, q1_key, q3_key, higher_better, precision, stacked in specifications:
        lines.extend((r"\midrule", rf"\multicolumn{{8}}{{c}}{{\textit{{{section}}}}}\\"))
        for problem in PROBLEMS:
            medians = [float(lookup[(method, problem)][median_key]) for method in METHODS]
            best = (max if higher_better else min)(medians)
            cells = []
            for method in METHODS:
                row = lookup[(method, problem)]
                formatter = _stacked_cell if stacked else _cell
                cells.append(
                    formatter(
                        row,
                        median_key,
                        q1_key,
                        q3_key,
                        precision,
                        float(row[median_key]) == best,
                    )
                )
            lines.append(PROBLEM_LABELS[problem] + " & " + " & ".join(cells) + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/stage2/comparisons/formal_summary.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper/generated_stage2_tables.tex")
    )
    args = parser.parse_args()
    lookup = _read(args.summary)
    tables = [_compact_table(lookup)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
