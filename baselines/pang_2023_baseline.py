"""Standalone Pang, Nan, and Ishibuchi implementation for common-FE comparison.

This file is intentionally isolated from GD-PSL. It does not import any
project module. Given only PlatEMO, MATLAB Engine for Python, NumPy, and
optionally PyTorch/matplotlib, it runs Section IV-A of the paper:

* standard MOEA/D-PBI and MOEA/D-PBI with random weight perturbation;
* DTLZ2 and DTLZ7 with three and five objectives;
* population 91/210, the project's common FE budget, and 20 independent runs;
* an unbounded archive of every evaluated solution;
* exact duplicate removal and first-front extraction;
* projected IGD-infinity against about 50,000 true-PF points.

The following implementation details are taken from the authors' supplied
``MOEADUEA1_100.m``, ``obtainUEA.m``, ``calMaxIGD.m``, and
``GetReferencePointSet.m`` sources and are recorded in every ``summary.json``:

* perturb each time from the original weight vectors (no cumulative drift);
* rebuild neighborhoods immediately after perturbation;
* perturb independently with a continuous uniform distribution;
* perturb after 100 complete populations, before generating generation 101;
* generate one full-dimensional reference PF with PlatEMO GetOptimum, then
  project that same set onto every objective pair;
* normalize the archive with the sampled true-PF extrema;
* use the authors' problem-specific normalized two-dimensional references;
* define duplicates as exactly equal objective vectors.

Section IV-B subset selection is not implemented here because its algorithms
are delegated to references [17] and [18] and are not specified in this paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "results"
DEFAULT_COMMON_TOTAL_FE_BUDGET = 100_000
DEFAULT_FORMAL_RUNS = 20


PAPER_REPORTED_IGD_INFINITY = {
    ("MOEAD", "dtlz2", 3): 0.0385,
    ("PangMOEAD", "dtlz2", 3): 0.0231,
    ("MOEAD", "dtlz7", 3): 0.1090,
    ("PangMOEAD", "dtlz7", 3): 0.0369,
    ("MOEAD", "dtlz2", 5): 0.0378,
    ("PangMOEAD", "dtlz2", 5): 0.0099,
    ("MOEAD", "dtlz7", 5): 0.1356,
    ("PangMOEAD", "dtlz7", 5): 0.1519,
}

IMPLEMENTATION_ASSUMPTIONS = {
    "weight_perturbation_base": "original_pre_specified_weights",
    "neighborhood_after_perturbation": "recomputed_immediately",
    "perturbation_distribution": "independent_continuous_uniform",
    "first_perturbation": "before_generating_generation_101",
    "duplicate_definition": "exactly_equal_objective_vectors",
    "reference_generation": "authors' GetReferencePointSet.m construction",
    "reference_projection": "problem-specific normalized two-dimensional reference sets",
    "indicator_space": "true_pf_min_max_normalized_objective_space_euclidean_distance",
    "section_iv_b_subset_selection": "not_reproduced",
}


@dataclass(frozen=True)
class Task:
    problem: str
    objectives: int
    population: int
    evaluations: int
    variables: int


def make_task(
    problem: str,
    objectives: int,
    evaluations: int = DEFAULT_COMMON_TOTAL_FE_BUDGET,
) -> Task:
    if objectives not in {3, 5}:
        raise ValueError("The paper protocol supports only 3 or 5 objectives")
    name = problem.lower()
    if name not in {"dtlz2", "dtlz7"}:
        raise ValueError("The paper protocol supports only DTLZ2 and DTLZ7")
    population = 91 if objectives == 3 else 210
    variables = objectives + (9 if name == "dtlz2" else 19)
    if evaluations <= 0:
        raise ValueError("evaluations must be positive")
    return Task(name, objectives, population, evaluations, variables)


MATLAB_FILES = {
    "Pang2023MOEAD.m": r"""
classdef Pang2023MOEAD < ALGORITHM
    methods
        function main(Algorithm,Problem)
            type = Algorithm.ParameterSet(1);
            [W0,Problem.N] = UniformPoint(Problem.N,Problem.M);
            W = W0;
            T = ceil(Problem.N/10);
            B = Pang2023MOEAD.neighbours(W,T);
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);
            completedGenerations = 1;
            if Problem.M == 3
                magnitude = 0.1;
            elseif Problem.M == 5
                magnitude = 0.2;
            else
                error('Pang2023MOEAD:UnsupportedObjectives', ...
                      'The paper reports only 3 and 5 objectives.');
            end

            while Algorithm.NotTerminated(Population)
                if mod(completedGenerations,100) == 0
                    delta = -magnitude + 2*magnitude*rand(size(W0));
                    W = max(W0 + delta,1e-6);
                    W = W./sum(W,2);
                    B = Pang2023MOEAD.neighbours(W,T);
                end
                for i = 1 : Problem.N
                    if Problem.FE >= Problem.maxFE
                        break;
                    end
                    P = B(i,randperm(size(B,2)));
                    Offspring = OperatorGAhalf(Problem,Population(P(1:2)));
                    Z = min(Z,Offspring.obj);
                    normW = sqrt(sum(W(P,:).^2,2));
                    normP = sqrt(sum((Population(P).objs-Z).^2,2));
                    normO = sqrt(sum((Offspring.obj-Z).^2,2));
                    cosineP = sum((Population(P).objs-Z).*W(P,:),2)./normW./max(normP,1e-12);
                    cosineO = sum((Offspring.obj-Z).*W(P,:),2)./normW./max(normO,1e-12);
                    if type ~= 1
                        error('Pang2023MOEAD:UnsupportedType','The paper uses PBI (type 1).');
                    end
                    gOld = normP.*cosineP + 5*normP.*sqrt(max(0,1-cosineP.^2));
                    gNew = normO.*cosineO + 5*normO.*sqrt(max(0,1-cosineO.^2));
                    Population(P(gOld>=gNew)) = Offspring;
                end
                completedGenerations = completedGenerations + 1;
            end
        end
    end
    methods (Static, Access = private)
        function B = neighbours(W,T)
            distances = pdist2(W,W);
            [~,B] = sort(distances,2);
            B = B(:,1:T);
        end
    end
end
""",
    "PangLoggedDTLZ2.m": r"""
classdef PangLoggedDTLZ2 < DTLZ2
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ2(obj,varargin{:});
            pang_log_evaluations(Population);
        end
    end
end
""",
    "PangLoggedDTLZ7.m": r"""
classdef PangLoggedDTLZ7 < DTLZ7
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ7(obj,varargin{:});
            pang_log_evaluations(Population);
        end
    end
end
""",
    "pang_log_evaluations.m": r"""
function pang_log_evaluations(Population)
    global PANG_DECS PANG_OBJS PANG_FE PANG_COUNT;
    n = length(Population);
    if n == 0
        return;
    end
    PANG_DECS = [PANG_DECS; Population.decs]; %#ok<AGROW>
    PANG_OBJS = [PANG_OBJS; Population.objs]; %#ok<AGROW>
    PANG_FE = [PANG_FE; (PANG_COUNT + (1:n))']; %#ok<AGROW>
    PANG_COUNT = PANG_COUNT + n;
end
""",
    "pang_run_platemo.m": r"""
function [HistoryX,HistoryF,HistoryFE,FrontNo] = pang_run_platemo(algorithmName,problemName,N,maxFE,M,D,platemoRoot,seed)
    addpath(genpath(platemoRoot));
    bridgeDir = fileparts(mfilename('fullpath'));
    addpath(bridgeDir);
    rng(double(seed),'twister');

    if strcmpi(algorithmName,'MOEAD')
        algorithmSpec = {@MOEAD,1};
    elseif strcmpi(algorithmName,'PangMOEAD')
        algorithmSpec = {@Pang2023MOEAD,1};
    else
        error('pang_run_platemo:UnknownAlgorithm','Unknown algorithm: %s',algorithmName);
    end
    if strcmpi(problemName,'DTLZ2')
        problemHandle = @PangLoggedDTLZ2;
    elseif strcmpi(problemName,'DTLZ7')
        problemHandle = @PangLoggedDTLZ7;
    else
        error('pang_run_platemo:UnknownProblem','Unknown problem: %s',problemName);
    end

    global PANG_DECS PANG_OBJS PANG_FE PANG_COUNT;
    PANG_DECS = zeros(0,double(D));
    PANG_OBJS = zeros(0,double(M));
    PANG_FE = zeros(0,1);
    PANG_COUNT = 0;
    platemo('algorithm',algorithmSpec,'problem',problemHandle, ...
            'N',double(N),'maxFE',double(maxFE),'M',double(M), ...
            'D',double(D),'draw',false);
    HistoryX = PANG_DECS;
    HistoryF = PANG_OBJS;
    HistoryFE = PANG_FE;
    [FrontNo,~] = NDSort(HistoryF,1);
    FrontNo = FrontNo(:);
end
""",
    "pang_reference_pf.m": r"""
function ReferencePF = pang_reference_pf(problemName,M,D,sampleCount,platemoRoot)
    addpath(genpath(platemoRoot));
    rng(1,'twister');
    if strcmpi(problemName,'DTLZ2')
        problemHandle = @DTLZ2;
    elseif strcmpi(problemName,'DTLZ7')
        problemHandle = @DTLZ7;
    else
        error('pang_reference_pf:UnknownProblem','Unknown problem: %s',problemName);
    end
    Problem = problemHandle('M',double(M),'D',double(D));
    ReferencePF = double(Problem.GetOptimum(double(sampleCount)));
    ReferencePF = ReferencePF(all(isfinite(ReferencePF),2),:);
    ReferencePF = unique(ReferencePF,'rows','stable');
end
""",
}


@contextmanager
def matlab_bridge() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="pang_2023_") as temporary:
        root = Path(temporary)
        for name, source in MATLAB_FILES.items():
            (root / name).write_text(source.strip() + "\n", encoding="ascii")
        yield root


def _matrix(values: Any, columns: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, columns or 0), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or (columns is not None and array.shape[1] != columns):
        raise ValueError(f"Unexpected matrix shape: {array.shape}")
    return array


def _write_csv(path: Path, values: np.ndarray, headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(np.asarray(values).tolist())


def _device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested, but PyTorch reports no CUDA device")
    return "cuda" if available else "cpu"


def _nearest_distances(queries: np.ndarray, points: np.ndarray, device: str) -> np.ndarray:
    if device == "cuda":
        import torch

        query_tensor = torch.as_tensor(queries, dtype=torch.float64, device="cuda")
        point_tensor = torch.as_tensor(points, dtype=torch.float64, device="cuda")
        block_size = max(32, min(2048, 20_000_000 // max(1, len(points))))
        result = []
        with torch.inference_mode():
            for start in range(0, len(query_tensor), block_size):
                distances = torch.cdist(query_tensor[start : start + block_size], point_tensor)
                result.append(distances.min(dim=1).values.cpu().numpy())
        return np.concatenate(result)

    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(points).query(queries, k=1)
        return np.asarray(distances, dtype=float)
    except ImportError:
        result = np.empty(len(queries), dtype=float)
        for start in range(0, len(queries), 256):
            block = queries[start : start + 256]
            distances = np.linalg.norm(block[:, None, :] - points[None, :, :], axis=2)
            result[start : start + len(block)] = distances.min(axis=1)
        return result


def _author_projection_references(
    problem_name: str,
    n_objectives: int,
    sample_count: int,
) -> dict[tuple[int, int], np.ndarray]:
    pairs = list(combinations(range(n_objectives), 2))
    if problem_name == "dtlz2":
        divisions = 1
        while (divisions - 1) * divisions / 2 < sample_count:
            divisions += 1
        divisions -= 1
        indices = np.asarray(list(combinations(range(1, divisions + 1), 2)), dtype=float)
        points = np.column_stack(
            (indices[:, 0] - 1.0, divisions - indices[:, 1])
        ) / (divisions - 2.0)
        nonzero = np.sum(points, axis=1) != 0
        points[nonzero] = (
            points[nonzero]
            / np.linalg.norm(points[nonzero], axis=1, keepdims=True)
            * np.sum(points[nonzero], axis=1, keepdims=True)
        )
        return {pair: points.copy() for pair in pairs}

    if problem_name == "dtlz7":
        divisions = int(np.floor(np.sqrt(sample_count)))
        gap = np.linspace(0.0, 1.0, divisions)
        first, second = np.meshgrid(gap, gap, indexing="ij")
        points = np.column_stack(
            (first.ravel(order="F"), second.ravel(order="F"))
        )
        interval = np.array([0.0, 0.251412, 0.631627, 0.859401])
        median = interval[1] / (interval[3] - interval[2] + interval[1])
        lower = points <= median
        points[lower] = points[lower] * interval[1] / median
        points[~lower] = (
            (points[~lower] - median)
            * (interval[3] - interval[2])
            / (1.0 - median)
            + interval[2]
        )
        without_last = points.copy()
        with_last = points[:, ::-1].copy()
        a = interval[3]
        for index in range(divisions):
            b = without_last[index, 0]
            s = (b / 2.0) * (1.0 + np.sin(3.0 * np.pi * b))
            lower_bound = 2.0 * (
                3.0 - ((a / 2.0) * (1.0 + np.sin(3.0 * np.pi * a)) + s)
            )
            upper_bound = 2.0 * (3.0 - s)
            start = index * divisions
            with_last[start : start + divisions, 1] = np.linspace(
                lower_bound, upper_bound, divisions
            )
        for values in (without_last, with_last):
            minimum, maximum = np.min(values, axis=0), np.max(values, axis=0)
            values -= minimum
            values /= maximum - minimum
        return {
            pair: (with_last if pair[1] == n_objectives - 1 else without_last).copy()
            for pair in pairs
        }
    raise ValueError(f"Unsupported paper problem: {problem_name}")


def projected_igd_infinity(
    archive: np.ndarray,
    reference: np.ndarray,
    device: str,
    problem_name: str,
    sample_count: int,
) -> dict[str, Any]:
    true_ideal = np.min(reference, axis=0)
    true_nadir = np.max(reference, axis=0)
    archive = (archive - true_ideal) / (true_nadir - true_ideal)
    pair_references = _author_projection_references(
        problem_name, archive.shape[1], sample_count
    )
    pair_scores = []
    for first, second in combinations(range(archive.shape[1]), 2):
        pair_reference = pair_references[(first, second)]
        distances = _nearest_distances(pair_reference, archive[:, [first, second]], device)
        pair_scores.append({
            "objectives": [first + 1, second + 1],
            "igd_infinity": float(distances.max()),
            "mean_distance": float(distances.mean()),
        })
    report = {
        "value": max(item["igd_infinity"] for item in pair_scores),
        "pair_scores": pair_scores,
        "definition": "max over objective pairs of max_z min_a ||z-a||_2",
        "reference_source": "Pang authors' problem-specific normalized 2-D reference sets",
        "reference_samples_actual": int(len(pair_reference)),
        "device": device,
    }
    return report


def _reference_front(engine: Any, task: Task, count: int, platemo_root: Path) -> np.ndarray:
    values = engine.pang_reference_pf(
        task.problem.upper(),
        float(task.objectives),
        float(task.variables),
        float(count),
        str(platemo_root),
        nargout=1,
    )
    reference = _matrix(values, task.objectives)
    if not len(reference) or not np.isfinite(reference).all():
        raise RuntimeError("PlatEMO returned an invalid reference PF")
    if task.problem == "dtlz2":
        if not np.allclose(np.linalg.norm(reference, axis=1), 1.0, atol=1e-8, rtol=1e-8):
            raise RuntimeError("DTLZ2 reference points are not on the unit hypersphere")
        if not np.any(np.linalg.norm(reference[:, :2], axis=1) < 0.95):
            raise RuntimeError("DTLZ2 projection does not contain disk-interior points")
    return reference


def _run_search(
    engine: Any,
    algorithm: str,
    task: Task,
    platemo_root: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    history_x, history_f, history_fe, front_no = engine.pang_run_platemo(
        algorithm,
        task.problem.upper(),
        float(task.population),
        float(task.evaluations),
        float(task.objectives),
        float(task.variables),
        str(platemo_root),
        float(seed),
        nargout=4,
    )
    elapsed = time.perf_counter() - started
    return (
        _matrix(history_x, task.variables),
        _matrix(history_f, task.objectives),
        np.asarray(history_fe, dtype=float).reshape(-1),
        np.asarray(front_no, dtype=float).reshape(-1),
        elapsed,
    )


def _nondominated_archive(
    history_x: np.ndarray,
    history_f: np.ndarray,
    front_no: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(history_x).all(axis=1) & np.isfinite(history_f).all(axis=1)
    history_x = history_x[finite]
    history_f = history_f[finite]
    front_no = front_no[: len(finite)][finite]
    _, first_indices = np.unique(history_f, axis=0, return_index=True)
    first_indices.sort()
    first_front = first_indices[front_no[first_indices] == 1]
    return history_x[first_front], history_f[first_front]


def _plot_archive(path: Path, archive: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    objectives = archive.shape[1]
    if objectives == 3:
        figure = plt.figure(figsize=(12, 9))
        axis = figure.add_subplot(2, 2, 1, projection="3d")
        axis.scatter(*archive.T, s=3, alpha=0.45, color="#1769aa", rasterized=True)
        axis.set(xlabel="f1", ylabel="f2", zlabel="f3")
        axis.view_init(elev=24, azim=42)
        for panel, (first, second) in enumerate(combinations(range(3), 2), start=2):
            axis = figure.add_subplot(2, 2, panel)
            axis.scatter(archive[:, first], archive[:, second], s=2, alpha=0.35,
                         color="#1769aa", rasterized=True)
            axis.set(xlabel=f"f{first + 1}", ylabel=f"f{second + 1}")
    else:
        pairs = list(combinations(range(objectives), 2))
        figure, axes = plt.subplots(2, 5, figsize=(18, 7))
        for axis, (first, second) in zip(axes.flat, pairs):
            axis.scatter(archive[:, first], archive[:, second], s=1, alpha=0.3,
                         color="#1769aa", rasterized=True)
            axis.set(xlabel=f"f{first + 1}", ylabel=f"f{second + 1}")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_one(
    engine: Any,
    algorithm: str,
    task: Task,
    reference: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    method = "PangMOEAD" if algorithm == "PangMOEAD" else f"EA_{algorithm}"
    run_dir = (
        args.output_dir
        / method
        / f"{task.problem.upper()}_{task.objectives}obj"
        / f"seed_{seed:03d}"
    )
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    history_x, history_f, history_fe, front_no, search_seconds = _run_search(
        engine, algorithm, task, args.platemo_root, seed
    )
    if len(history_f) != task.evaluations:
        raise RuntimeError(
            f"{algorithm}/{task.problem}/{task.objectives} objectives used "
            f"{len(history_f)} FE, expected {task.evaluations}"
        )
    archive_x, archive_f = _nondominated_archive(history_x, history_f, front_no)
    metric = projected_igd_infinity(
        archive_f,
        reference,
        args.resolved_device,
        task.problem,
        args.reference_samples,
    )
    total_seconds = time.perf_counter() - started

    np.savez_compressed(
        run_dir / "fronts.npz",
        evaluations_x=history_x,
        evaluations_f=history_f,
        evaluations_fe=history_fe,
        archive_x=archive_x,
        archive_f=archive_f,
        selected_x=archive_x,
        selected_f=archive_f,
    )
    _write_csv(
        run_dir / "nondominated_archive.csv",
        np.hstack((archive_x, archive_f)),
        [*[f"x{i + 1}" for i in range(task.variables)],
         *[f"f{i + 1}" for i in range(task.objectives)]],
    )
    (run_dir / "coverage_queries.csv").unlink(missing_ok=True)
    if args.save_history_csv:
        _write_csv(
            run_dir / "unbounded_archive.csv",
            np.hstack((history_x, history_f)),
            [*[f"x{i + 1}" for i in range(task.variables)],
             *[f"f{i + 1}" for i in range(task.objectives)]],
        )
    visualization_dir = run_dir / "figures"
    if args.plot:
        _plot_archive(
            visualization_dir / "archive_pf.png",
            archive_f,
            f"{algorithm} | {task.problem.upper()} | {task.objectives} objectives | seed {seed}",
        )

    summary = {
        "method": "pang_2023_baseline",
        "paper_reproduction_section": "Section IV-A algorithm under common-FE comparison",
        "algorithm": algorithm,
        "problem": task.problem,
        "n_objectives": task.objectives,
        "n_variables": task.variables,
        "run": seed,
        "population_size": task.population,
        "full_population_equivalents": task.evaluations / task.population,
        "common_total_fe_budget": task.evaluations,
        "fe_used": len(history_f),
        "unbounded_archive_size": len(history_f),
        "nondominated_archive_size": len(archive_f),
        "large_solution_set_size": len(archive_f),
        "modified_igd_infinity": metric,
        "igd_infinity": {
            "available": True,
            "protocol": "pang_projected_2d",
            "completed_igd_infinity": metric["value"],
            "pair_scores": metric["pair_scores"],
        },
        "runtime_seconds": {
            "ea_search": search_seconds,
            "archive_and_metric": total_seconds - search_seconds,
            "method_execution_total": total_seconds,
            "total": total_seconds,
        },
        "search_device": "PlatEMO/MATLAB",
        "metric_device": args.resolved_device,
        "visualization_directory": str(visualization_dir),
        "implementation_assumptions": IMPLEMENTATION_ASSUMPTIONS,
        "implementation_version": "pang_authors_code_sync_v1",
        "protocol": "all evaluated objectives -> nondominated sorting -> exact objective deduplication -> UEA",
        "reproduction_scope": "authors' algorithm, UEA, and IGD-infinity protocol under project-selected FE, problems, and 20-run formal statistics",
        "reproduction_status": "common_fe_comparison_not_exact_paper_budget",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{algorithm}/{task.problem}/{task.objectives}obj/seed {seed}] "
        f"FE={len(history_f)}, archive={len(archive_f)}, "
        f"IGDinf={metric['value']:.8g}, time={total_seconds:.2f}s",
        flush=True,
    )
    return summary


def write_aggregate(output_dir: Path, summaries: Sequence[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        groups[(summary["algorithm"], summary["problem"], summary["n_objectives"])].append(summary)

    rows = []
    for key, runs in sorted(groups.items()):
        algorithm, problem, objectives = key
        values = np.asarray([run["modified_igd_infinity"]["value"] for run in runs])
        times = np.asarray([run["runtime_seconds"]["method_execution_total"] for run in runs])
        sizes = np.asarray([run["nondominated_archive_size"] for run in runs])
        paper_value = PAPER_REPORTED_IGD_INFINITY[key]
        mean = float(values.mean())
        comparison_budget = int(runs[0]["common_total_fe_budget"])
        paper_budget = int(runs[0]["population_size"] * 1000)
        budget_matches_paper = comparison_budget == paper_budget
        rows.append({
            "algorithm": algorithm,
            "problem": problem,
            "n_objectives": objectives,
            "runs": len(runs),
            "igd_infinity_mean": mean,
            "igd_infinity_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "igd_infinity_min": float(values.min()),
            "igd_infinity_max": float(values.max()),
            "paper_reported_mean": paper_value,
            "paper_total_fe_budget": paper_budget,
            "comparison_total_fe_budget": comparison_budget,
            "paper_budget_comparable": budget_matches_paper,
            "absolute_error_from_paper": abs(mean - paper_value) if budget_matches_paper else None,
            "relative_error_from_paper": (
                abs(mean - paper_value) / paper_value if budget_matches_paper else None
            ),
            "archive_size_mean": float(sizes.mean()),
            "runtime_seconds_mean": float(times.mean()),
        })
    _write_csv(
        output_dir / "section_iv_a_summary.csv",
        np.asarray([[row[key] for key in row] for row in rows], dtype=object),
        list(rows[0]),
    )
    (output_dir / "section_iv_a_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Pang et al. SMC 2023 Section IV-A baseline"
    )
    parser.add_argument("--platemo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--algorithms", nargs="+", choices=("MOEAD", "PangMOEAD"),
                        default=["MOEAD", "PangMOEAD"])
    parser.add_argument("--problems", nargs="+", choices=("dtlz2", "dtlz7"),
                        default=["dtlz2", "dtlz7"])
    parser.add_argument("--objectives", nargs="+", type=int, choices=(3, 5), default=[3, 5])
    parser.add_argument("--runs", type=int, default=DEFAULT_FORMAL_RUNS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-fe", type=int, default=DEFAULT_COMMON_TOTAL_FE_BUDGET)
    parser.add_argument("--reference-samples", type=int, default=50_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-history-csv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.platemo_root.is_dir():
        raise FileNotFoundError(f"PlatEMO root does not exist: {args.platemo_root}")
    if args.runs < 1 or args.reference_samples < 2:
        raise ValueError("runs must be positive and reference-samples must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.resolved_device = _device(args.device)

    import matlab.engine

    summaries = []
    with matlab_bridge() as bridge:
        engine = matlab.engine.start_matlab()
        engine.addpath(str(bridge), nargout=0)
        try:
            for objectives in args.objectives:
                for problem in args.problems:
                    task = make_task(problem, objectives, args.max_fe)
                    reference = _reference_front(
                        engine, task, args.reference_samples, args.platemo_root
                    )
                    _write_csv(
                        args.output_dir / "comparisons" / "Pang2023" / "references"
                        / f"reference_pf_{problem}_{objectives}obj.csv",
                        reference,
                        [f"f{i + 1}" for i in range(objectives)],
                    )
                    for algorithm in args.algorithms:
                        for offset in range(args.runs):
                            summaries.append(
                                run_one(engine, algorithm, task, reference, args, args.seed + offset)
                            )
        finally:
            engine.quit()
    write_aggregate(args.output_dir / "comparisons" / "Pang2023", summaries)
    print(f"Completed Pang 2023 Section IV-A baseline: {args.output_dir}")


if __name__ == "__main__":
    main()
