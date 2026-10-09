"""Canonical artifact layout shared by every experiment method."""

from __future__ import annotations

from pathlib import Path


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def problem_directory_name(problem: str, n_objectives: int) -> str:
    return f"{safe_name(problem.upper())}_{n_objectives}obj"


def method_directory_name(algorithm: str, uses_model: bool) -> str:
    prefix = "GD-PSL" if uses_model else "EA"
    return f"{prefix}_{safe_name(algorithm.upper())}"


def run_directory(
    output_root: str | Path,
    method: str,
    problem: str,
    n_objectives: int,
    seed: int,
) -> Path:
    return (
        Path(output_root)
        / safe_name(method)
        / problem_directory_name(problem, n_objectives)
        / f"seed_{seed:03d}"
    )
