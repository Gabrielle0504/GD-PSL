# GD-PSL

**Gap-Directed Pareto Set Learning for empirical Pareto-front completion**

GD-PSL is a post-evolutionary completion strategy for multi-objective
optimization. A configurable PlatEMO algorithm first builds a cumulative
empirical Pareto archive. GD-PSL then identifies under-covered regions in
preference space, trains a preference-to-decision model, evaluates generated
decisions with the true objective functions, audits direction, support, and
front-envelope consistency, and then applies finite, objective-unique,
nondominated archive merging to every evaluated candidate.

The optimization procedure never uses a benchmark reference Pareto front.
Reference fronts and fixed normalization data are available only to the
post-run evaluation code.

## Method scope

- Base optimizers: NSGA-II, NSGA-III, MOEA/D, or another compatible PlatEMO
  algorithm.
- Completion model: a PyTorch preference-to-decision network with validation
  loss monitoring and early stopping.
- Runtime diagnostics: empirical direction agreement, local front support,
  and a one-sided empirical-front envelope. Final archive retention uses finite
  objectives, duplicate removal, and nondominated filtering.
- Primary evaluation: hypervolume (HV), IGD-infinity, and algorithm runtime.
- Baseline: an isolated reproduction of Pang, Nan, and Ishibuchi (SMC 2023).

Every model-generated decision consumes one function evaluation when its true
objectives are evaluated, regardless of whether it is retained in the final
archive.

## Repository layout

```text
gd_psl/                         GD-PSL implementation
matlab/                         PlatEMO bridge, exact-budget adapters, RE problems
baselines/                      Independent Pang baseline reproduction
data/                           Fixed benchmark fronts and normalization data
docs/                           Protocol, pseudocode, and reproduction notes
tests/                          Unit tests that do not require a live MATLAB session
reporting/                      Visualization, comparison plots, and statistics
scripts/                        Reproducible shell launchers for formal experiments
tools/                          Non-formal diagnostic utilities
results/                        Generated artifacts; ignored by Git

run_fill_then_judge.py          Main GD-PSL and pure-EA entry point
run_pang_archive_baseline.py    Project-suite Pang baseline entry point
experiment_config.py            Shared experiment and model configuration
```

## Requirements

- Python 3.10 or newer
- MATLAB with the MATLAB Engine for Python
- PlatEMO
- NumPy, SciPy, PyTorch, and matplotlib
- A CUDA-capable PyTorch installation for GD-PSL model training
- AdamW is the explicit default optimizer. `schedulefree_adamw` remains an
  optional mode, and each run records both the requested and resolved optimizer.

Install PyTorch using the command appropriate for the server CUDA version,
then install this project's remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The editable installation keeps the packaged reference fronts and MATLAB
adapters anchored to the checked-out repository. MATLAB Engine is installed
using MathWorks' instructions for the local MATLAB release; it is not fetched
from PyPI by this project.

Set the PlatEMO path either on the command line or through `PLATEMO_ROOT`:

```bash
export PLATEMO_ROOT=/workspace/PlatEMO/PlatEMO
```

## Quick start

Validate the central configuration:

```bash
python experiment_config.py
```

Run GD-PSL with NSGA-II:

```bash
python run_fill_then_judge.py \
  --algorithms NSGAII \
  --problems dtlz2 \
  --n-objectives 3 \
  --population-size 91 \
  --max-fe 91000 \
  --ea-fill-split 90:10 \
  --runs 1 \
  --device cuda \
  --platemo-root "$PLATEMO_ROOT"
```

Run the equal-budget pure EA by adding `--ea-only`:

```bash
python run_fill_then_judge.py \
  --algorithms NSGAII \
  --problems dtlz2 \
  --n-objectives 3 \
  --population-size 91 \
  --max-fe 91000 \
  --runs 1 \
  --ea-only \
  --platemo-root "$PLATEMO_ROOT"
```

Run the project-suite Pang baseline:

```bash
python run_pang_archive_baseline.py \
  --algorithms PangMOEAD \
  --problems dtlz2 \
  --n-objectives 3 \
  --population-size 91 \
  --max-fe 91000 \
  --runs 1 \
  --platemo-root "$PLATEMO_ROOT"
```

The paper-specific, standalone Pang reproduction is
`baselines/pang_2023_baseline.py`. It is intentionally independent of GD-PSL
and supports the paper's three- and five-objective DTLZ2/DTLZ7 protocol.

## Formal experiment protocol

The pre-specified two- and three-objective study uses 1,000 evaluated population
equivalents:

| Objective count | Population | Total true FE |
|---:|---:|---:|
| 2 | 100 | 100,000 |
| 3 | 91 | 91,000 |

The first stage evaluates the pre-specified splits `50:50`, `60:40`, `70:30`,
`80:20`, and `90:10` using ablation seeds 21-40. A single split is selected and
frozen before the formal comparison. The second stage compares NSGA-II,
NSGA-III, MOEA/D, their three GD-PSL variants, and PangMOEAD using 20 paired
seeds, 101-120.

The values in `DEFAULT_CONFIG` are convenient single-run defaults, not an
implicit formal protocol. Formal launch commands must pass the population and
FE values specified for each objective count above.

Run one Stage 1 allocation on an experiment server:

```bash
scripts/run_stage1_ratio.sh 80:20
```

Resume the same allocation while preserving completed seeds:

```bash
scripts/run_stage1_ratio.sh --resume 80:20
```

Run one Stage 2 method on its assigned server. The script fixes the
formal seeds to 101-120 and writes every method under `results/stage2/`:

```bash
scripts/run_stage2_method.sh GD-PSL_NSGAII
```

Accepted method names are `EA_NSGAII`, `EA_NSGAIII`, `EA_MOEAD`,
`GD-PSL_NSGAII`, `GD-PSL_NSGAIII`, `GD-PSL_MOEAD`, and `PangMOEAD`.
After all seven allocations finish, generate the 20-run statistics and PF
comparison with:

```bash
python -m reporting.compare_stage2 --results-root results/stage2
```

See [docs/formal_experiment_protocol.md](docs/formal_experiment_protocol.md)
for the complete pre-specified problem list, selection rule, timing boundary,
archive definition, and statistical analysis.
The legacy formal artifacts' resolved training optimizer is documented in
[docs/formal_optimizer_resolution.md](docs/formal_optimizer_resolution.md).

## Outputs

Runs are stored under:

```text
results/<method>/<problem>_<M>obj/seed_<seed>/
```

Important artifacts include:

- `base_pf.csv`: finite, objective-unique nondominated cumulative EA archive;
- `model_candidates.csv`: all true-evaluated model candidates;
- `completed_pf.csv`: final finite, objective-unique nondominated archive over
  the cumulative EA archive and all true-evaluated model candidates;
- `gap_regions.csv` and `unfilled_regions.csv`: hole-selection audit data;
- `training_history.csv` and `training_summary.json`: loss and stopping data;
- `summary.json`: FE usage, stage timing, metrics, and effective configuration.

For GD-PSL plots, blue points are retained EA archive points and red points
are true-evaluated model points that survive in the final nondominated archive.
Pure EA and Pang plots contain only blue archive points.

Generate a report for one saved run or recursively render a result tree:

```bash
python -m reporting.visualize --help
python -m reporting.visualize_batch --roots results/stage1
python -m reporting.compare_ratios --stage1-root results/stage1
python -m reporting.compare_stage1_hv --stage1-root results/stage1
```

The seven-method comparison and repeated-run statistical analysis are exposed
as modules so the repository root remains limited to formal run entry points:

```bash
python -m reporting.compare_methods --help
python -m reporting.statistics --help
```

The shell launchers accept `PROJECT_ROOT`, `PYTHON_BIN`, `PLATEMO_ROOT`, and
`OUTPUT_DIR` environment variables. Their defaults reproduce the experiment
server layout used in this study.

## Reproducibility rules

1. Methods compared on one problem receive the same total true FE budget.
2. The same paired seed initializes every method on that problem.
3. Objective-vector deduplication precedes archive size and metric reporting.
4. HV uses fixed external ideal/nadir values and the normalized reference
   point `(1.1, ..., 1.1)`.
5. IGD-infinity uses one fixed, full-dimensional reference set per problem.
   Packaged RE tasks use `data/RE/ParetoFront/RE*.dat`, normalized with the
   corresponding fixed ideal/nadir files before objective-pair projection.
6. Runtime includes EA search, hole processing, model training, candidate
   generation, true evaluation, and archive update; plotting and metrics are
   excluded.
7. Result figures use the run nearest the method's median IGD-infinity rather
   than a hand-picked best run.

## Tests

Run the local unit tests with:

```bash
python -m unittest discover -s tests -v
```

MATLAB/PlatEMO integration is checked separately on the experiment server.

## References

Pang, L. M., Nan, Y., and Ishibuchi, H. "How to Find a Large Solution Set to
Cover the Entire Pareto Front in Evolutionary Multi-Objective Optimization."
2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC),
pp. 1188-1194, 2023.

PlatEMO should be cited according to the citation instructions distributed
with the installed PlatEMO version.
