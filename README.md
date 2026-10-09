# GD-PSL

**Gap-Directed Pareto Set Learning for empirical Pareto-front completion**

GD-PSL is a post-evolutionary completion strategy for multi-objective
optimization. A PlatEMO optimizer first constructs a cumulative archive from
all true evaluations. GD-PSL then detects sparsely covered, archive-supported
preference regions, trains a preference-to-decision model, refines its
proposals locally, and evaluates every proposal with the true objective
functions. The reported archive is the finite, objective-unique nondominated
set over the EA archive and all evaluated model proposals.

Reference Pareto fronts are never used by search, training, query allocation,
candidate diagnostics, or archive construction. They are loaded only after a
run to calculate HV and projected IGD-infinity.

## Repository

```text
gd_psl/             GD-PSL, configuration, metrics, and problem definitions
matlab/             PlatEMO bridge, exact-FE adapters, and formal RE problems
data/               Fixed PF and normalization files for the nine-task suite
baselines/          Standalone Pang et al. reproduction
reporting/          Stage A/B aggregation and run visualization
scripts/            Formal Stage A/B launchers
tests/              Unit tests independent of a live MATLAB session
docs/               Formal protocol, pseudocode, and Pang reproduction map

run_gd_psl.py       GD-PSL and equal-budget pure-EA entry point
run_pang.py         PangMOEAD entry point for the common experiment suite
```

Generated artifacts are written under `results/` and are not committed.

## Requirements

- Python 3.10 or newer
- MATLAB and MATLAB Engine for Python
- PlatEMO
- NumPy, SciPy, PyTorch, and matplotlib
- CUDA-capable PyTorch for model training

Install the PyTorch build appropriate for the server CUDA version, then run:

```bash
python -m pip install -e .
export PLATEMO_ROOT=/workspace/PlatEMO/PlatEMO
```

MATLAB Engine is installed using MathWorks' instructions for the local MATLAB
release; this project does not fetch it from PyPI.

## Run one experiment

GD-PSL with NSGA-II:

```bash
python run_gd_psl.py \
  --algorithms NSGAII \
  --problems dtlz2 \
  --n-objectives 3 \
  --population-size 91 \
  --max-fe 91000 \
  --ea-fill-split 90:10 \
  --runs 1 --device cuda \
  --platemo-root "$PLATEMO_ROOT"
```

The equal-budget pure EA uses the same command with `--ea-only`. The common
suite Pang baseline is run with:

```bash
python run_pang.py \
  --algorithms PangMOEAD \
  --problems dtlz2 \
  --n-objectives 3 \
  --population-size 91 \
  --max-fe 91000 \
  --runs 1 \
  --platemo-root "$PLATEMO_ROOT"
```

All editable hyperparameters and their validation rules are in
`gd_psl/config.py`. Command-line options override them for one run.

## Reproduce the formal study

The formal suite contains DTLZ2, DTLZ7, RE21, RE24, RE31, RE32, RE34, RE35,
and RE37. Two-objective tasks use population 100 and 100,000 true function
evaluations (FE); three-objective tasks use population 91 and 91,000 FE. The
initial population counts toward the total budget.

Stage A evaluates each pre-specified EA/FILL split with seeds 21--40:

```bash
scripts/run_stage1_ratio.sh 90:10
scripts/run_stage1_pure_ea.sh NSGAII
python -m reporting.compare_ratios --stage1-root results/stage1
python -m reporting.compare_stage1_hv --stage1-root results/stage1
```

Stage B uses paired seeds 101--120. Run one method per server:

```bash
scripts/run_stage2_method.sh GD-PSL_NSGAII
```

Accepted method names are `EA_NSGAII`, `EA_NSGAIII`, `EA_MOEAD`,
`GD-PSL_NSGAII`, `GD-PSL_NSGAIII`, `GD-PSL_MOEAD`, and `PangMOEAD`.
After all seven methods finish:

```bash
python -m reporting.compare_stage2 --results-root results/stage2
```

The launchers accept `PROJECT_ROOT`, `PYTHON_BIN`, `PLATEMO_ROOT`, and
`OUTPUT_DIR`; their defaults reproduce the experiment-server layout. The full
frozen design is in `docs/formal_experiment_protocol.md`.

## Result contract

Each run is stored as:

```text
results/<method>/<problem>_<M>obj/seed_<seed>/
```

The key files are:

- `fronts.npz`: evaluated history, EA archive, model candidates, and completed archive;
- `summary.json`: FE use, timing, metrics, and effective configuration;
- `training_history.csv`: training and validation losses;
- `gap_regions.csv`: query roles, empirical diagnostics, and archive status.

Every true-evaluated model proposal consumes one FE, including proposals that
are nonfinite, duplicated, dominated, or diagnostically flagged. Direction,
local-support, and empirical-envelope checks guide adaptive querying and remain
in the audit record; final retention uses finite, objective-unique,
nondominated merging.

For figures, EA archive points are blue and surviving model points are red.
Pure EA and Pang figures contain blue points only. Visualize a saved run with:

```bash
python -m reporting.visualize --run-dir <run-directory>
```

## Reproducibility checks

- Equal true-FE budgets within each problem.
- Paired seeds across methods.
- Objective-vector deduplication before archive size and metrics.
- Fixed external ideal/nadir values and HV reference point `(1.1, ..., 1.1)`.
- Pang-style projected IGD-infinity with one fixed reference set per problem.
- Runtime excludes metric calculation, plotting, and artifact serialization.
- Representative PF figures use the run nearest the method-problem median
  projected IGD-infinity.

Run the local checks with:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same test suite on every push and pull request.

## Reference

Pang, L. M., Nan, Y., and Ishibuchi, H. "How to Find a Large Solution Set to
Cover the Entire Pareto Front in Evolutionary Multi-Objective Optimization."
IEEE SMC, pp. 1188--1194, 2023.

PlatEMO must be cited according to the citation instructions distributed with
the installed PlatEMO version.
