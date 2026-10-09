"""Small MATLAB/PlatEMO boundary used by the GD-PSL runner."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from .archive import as_2d


def start_matlab_with_retry(
    engine_module,
    attempts: int = 6,
    delay_seconds: float = 10.0,
):
    """Start MATLAB with bounded retries for transient license failures."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return engine_module.start_matlab()
        except engine_module.EngineError:
            if attempt == attempts:
                raise
            wait_seconds = delay_seconds * attempt
            print(
                f"MATLAB Engine startup failed ({attempt}/{attempts}); "
                f"retrying in {wait_seconds:.0f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def validate_problem_consistency(
    decisions: np.ndarray,
    matlab_objectives: np.ndarray,
    problem,
    device: str,
    sample_size: int = 32,
    relative_tolerance: float = 1e-4,
    absolute_tolerance: float = 1e-7,
) -> None:
    """Check that MATLAB and Python evaluate identical decisions equally."""
    count = min(sample_size, len(decisions))
    indices = np.linspace(0, len(decisions) - 1, count, dtype=int)
    sample = torch.as_tensor(decisions[indices], dtype=torch.float64, device=device)
    with torch.no_grad():
        actual = problem.evaluate(sample).detach().cpu().numpy()
    expected = matlab_objectives[indices]
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=relative_tolerance, atol=absolute_tolerance
    ):
        scale = np.maximum(np.abs(expected), absolute_tolerance)
        error = float(np.max(np.abs(actual - expected) / scale))
        raise ValueError(
            "MATLAB and Python problem evaluations are inconsistent "
            f"(maximum relative error {error:.3e})"
        )


def run_platemo_algorithm(
    engine,
    algorithm_name: str,
    problem_name: str,
    population_size: int,
    max_fe: int,
    n_objectives: int,
    n_dimensions: int,
    platemo_root: str,
    random_seed: int,
    algorithm_parameters: Sequence[float] = (),
) -> tuple[np.ndarray, ...]:
    """Execute one exact-budget search through ``matlab/run_platemo.m``."""
    import matlab

    values = list(map(float, algorithm_parameters))
    parameters = matlab.double([values]) if values else matlab.double([])
    result = engine.run_platemo(
        algorithm_name,
        problem_name,
        float(population_size),
        float(max_fe),
        float(n_objectives),
        float(n_dimensions),
        platemo_root,
        parameters,
        float(random_seed),
        nargout=6,
    )
    x, f, c, history_x, history_f, history_fe = result
    return (
        np.asarray(x, dtype=float),
        np.asarray(f, dtype=float),
        np.asarray(c, dtype=float),
        np.asarray(history_x, dtype=float),
        np.asarray(history_f, dtype=float),
        np.asarray(history_fe, dtype=float).reshape(-1),
    )


def run_platemo_algorithm_timed(
    engine,
    algorithm_name: str,
    problem_name: str,
    population_size: int,
    max_fe: int,
    n_objectives: int,
    n_dimensions: int,
    platemo_root: str,
    random_seed: int,
    algorithm_parameters: Sequence[float] = (),
) -> tuple[np.ndarray, ...]:
    """Execute one search and also return cumulative per-evaluation time."""
    import matlab

    values = list(map(float, algorithm_parameters))
    parameters = matlab.double([values]) if values else matlab.double([])
    result = engine.run_platemo(
        algorithm_name,
        problem_name,
        float(population_size),
        float(max_fe),
        float(n_objectives),
        float(n_dimensions),
        platemo_root,
        parameters,
        float(random_seed),
        nargout=7,
    )
    x, f, c, history_x, history_f, history_fe, history_time = result
    return (
        np.asarray(x, dtype=float),
        np.asarray(f, dtype=float),
        np.asarray(c, dtype=float),
        np.asarray(history_x, dtype=float),
        np.asarray(history_f, dtype=float),
        np.asarray(history_fe, dtype=float).reshape(-1),
        np.asarray(history_time, dtype=float).reshape(-1),
    )
def reference_front(
    engine,
    problem_name: str,
    n_objectives: int,
    n_dimensions: int,
    sample_count: int,
    platemo_root: str,
) -> np.ndarray:
    """Get the full-dimensional PlatEMO PF used for projected IGD-infinity."""
    values = engine.generate_reference_pf(
        problem_name.upper(),
        float(n_objectives),
        float(n_dimensions),
        float(sample_count),
        platemo_root,
        nargout=1,
    )
    front = as_2d(np.asarray(values, dtype=float), n_objectives)
    front = np.unique(front[np.isfinite(front).all(axis=1)], axis=0)
    if not len(front):
        raise RuntimeError("PlatEMO returned an empty reference Pareto front")
    return front
