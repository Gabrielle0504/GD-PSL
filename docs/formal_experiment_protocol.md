# Formal GD-PSL Experiment Protocol

This document freezes the two-stage experiment design. Any change to an item
below creates a new experiment and must be recorded in the run manifest.

## 1. Benchmark tasks and true-evaluation budgets

The budget corresponds to 1,000 evaluated population equivalents. Initial
solutions count toward the total true function evaluations (FE).

| Problem | Objectives | Variables | Population | Total FE |
|---|---:|---:|---:|---:|
| RE21 | 2 | 4 | 100 | 100,000 |
| RE24 | 2 | 2 | 100 | 100,000 |
| DTLZ2 | 3 | 12 | 91 | 91,000 |
| DTLZ7 | 3 | 22 | 91 | 91,000 |
| RE31 | 3 | 3 | 91 | 91,000 |
| RE32 | 3 | 4 | 91 | 91,000 |
| RE34 | 3 | 5 | 91 | 91,000 |
| RE35 | 3 | 7 | 91 | 91,000 |
| RE37 | 3 | 4 | 91 | 91,000 |

Within one task, every method uses the same population, total FE, decision
dimension, objective count, and paired seed. The initial decision vectors are
sampled uniformly inside the problem bounds by PlatEMO. No reference-front or
feasibility-informed initialization is allowed.

## 2. Stage A: pre-specified EA/FILL ablation

Evaluate one common split for GD-PSL with each base optimizer: NSGA-II,
NSGA-III, and MOEA/D. Use 20 ablation seeds, 21-40.

| Split | Two-objective EA/FILL FE | Three-objective EA/FILL FE |
|---|---:|---:|
| 50:50 | 50,000 / 50,000 | 45,500 / 45,500 |
| 60:40 | 60,000 / 40,000 | 54,600 / 36,400 |
| 70:30 | 70,000 / 30,000 | 63,700 / 27,300 |
| 80:20 | 80,000 / 20,000 | 72,800 / 18,200 |
| 90:10 | 90,000 / 10,000 | 81,900 / 9,100 |

Primary selection rule:

1. Rank the five splits by IGD-infinity on each `(problem, base EA)` task.
2. Test the lowest-ranked split against the corresponding pure EAs. Reject it
   if its HV is significantly worse under paired seeds after Holm correction
   over the 27 task--optimizer comparisons, then continue down the rank order.
3. If admissible splits are statistically tied, select the one with fewer
   FILL evaluations; use algorithm runtime as the final tie-breaker.

The selected split is global: it cannot vary by problem or base EA. Freeze it,
the implementation, and all model hyperparameters before Stage B.

Five identical servers may execute Stage A concurrently, one split per server.
Each server runs all nine problems, three base EAs, and 20 seeds, for 540
GD-PSL runs. Pure-EA pilot runs are shared and must not be repeated per split.

## 3. Stage B: formal method comparison

Compare these seven methods with paired seeds 101-120:

1. NSGA-II
2. NSGA-III
3. MOEA/D
4. GD-PSL + NSGA-II
5. GD-PSL + NSGA-III
6. GD-PSL + MOEA/D
7. PangMOEAD

Pure EA and PangMOEAD receive the complete task budget. GD-PSL divides the
same total budget according to the single Stage-A split. Stage B contains
`9 x 7 x 20 = 1,260` independent runs.

## 4. FE and archive accounting

Every call to a true problem objective consumes one FE. A model candidate
still consumes an FE if it is infeasible, nonfinite, duplicated, dominated, or
flagged by empirical-front diagnostics. Model training and inference do not
consume FE.

Pure EA and Pang solution sets are constructed from cumulative evaluated
history:

```text
Archive = Nondominated(UniqueObjectives(Finite(AllEvaluatedSolutions)))
```

They are cumulative archives rather than final populations. GD-PSL applies the
same finite, objective-unique nondominated rule after merging the cumulative EA
archive with all true-evaluated model candidates. Runtime direction, local
support, and one-sided empirical-front-envelope checks remain in the audit and
guide pilot adaptation, but do not reject an otherwise nondominated candidate
from the reported archive.

## 5. Metrics and references

The three primary metrics are HV (higher is better), IGD-infinity (lower is
better), and algorithm runtime (lower is better).

- Normalize every method with the same fixed ideal and nadir points.
- Compute HV with the normalized reference point `(1.1, ..., 1.1)`.
- Use one fixed, full-dimensional IGD-infinity reference set per problem.
- Never rebuild the reference set separately for a method, split, seed, or
  result batch.
- Reference fronts are post-run evaluation data and are forbidden inside EA
  search, hole detection, model training, generation, diagnostics, and archive construction.

For packaged RE tasks, use the fixed full-dimensional reference fronts in
`data/RE/ParetoFront/RE*.dat`. Normalize both the evaluated archive and this
reference with the matching fixed files in `data/RE/ideal_nadir_points/`
before objective-pair projection. Record the source, hash, normalization
bounds, and row count.

## 6. Timing boundary

Algorithm runtime includes EA search, cumulative archive construction needed
by the method, hole processing, model training, model inference, true candidate
evaluation, candidate diagnostics, and final archive update. Metric calculation,
plotting, and artifact serialization are excluded and reported separately.

Record at least `ea_time`, `hole_time`, `model_training_time`, `fill_time`, and
`algorithm_time`. Run one process per server/GPU. All timing comparisons must
use identical hardware and software environments. PlatEMO algorithms execute
on the CPU; the GD-PSL neural model uses CUDA.

## 7. Statistical analysis and figures

Report median and interquartile range over 20 formal runs. Use paired Wilcoxon
signed-rank tests per problem with Holm correction, and report cross-problem
Friedman average ranks. Bold only the best median in each problem row; attach
significance markers separately.

Choose the displayed PF run by minimum absolute distance from that method's
median IGD-infinity. Pure EA and Pang plots contain blue archive points only.
GD-PSL plots show EA archive points in blue and true-evaluated model points
that survive in the final archive in red. All methods for one problem use the
same axes, limits, marker sizes, and three-dimensional camera.

## 8. Integrity checks before aggregation

- Recorded FE equals the pre-specified task budget exactly.
- Population and objective count match the task table.
- Seeds and effective configurations are present in `summary.json`.
- All expected run directories exist and contain finite metric values.
- No two methods write to the same run directory.
- Failed runs are rerun with the same method, task, and seed.
- Stage-A seeds are never reused in Stage B.
