# Pang et al. Full-Text Reproduction Map

Source: Pang, Nan, and Ishibuchi, *How to Find a Large Solution Set to Cover
the Entire Pareto Front in Evolutionary Multi-Objective Optimization*, IEEE
SMC 2023, pp. 1188--1194.

## Reproduced points

| Paper requirement | Local implementation | Status |
|---|---|---|
| Store every examined solution in an unbounded archive | `run_pang_archive_baseline.py` writes `evaluations_*` and `unbounded_archive.csv` | Reproduced |
| Remove duplicate objective vectors and dominated archive rows for the reported PF archive | `nondominated_indices()` and the archive construction in `run_one()` | Reproduced |
| PBI-based MOEA/D with penalty 5 | `matlab/PangMOEAD.m`, `type == 1`, penalty constant 5 | Reproduced |
| Perturb every weight-vector element every 100 generations | `matlab/PangMOEAD.m`, `perturbationPeriod = 100` | Reproduced |
| Perturbation range [-0.1, 0.1] for 3 objectives | `matlab/PangMOEAD.m` | Reproduced |
| Perturbation range [-0.2, 0.2] for 5 objectives | `matlab/PangMOEAD.m` | Reproduced |
| Clip negative weights to 1e-6 and normalize each row | `matlab/PangMOEAD.m` | Reproduced |
| Population 91/210 in the authors' DTLZ experiments | `baselines/pang_2023_baseline.py` | Reproduced |
| Two-objective projections for every objective pair | `pang_projected_igd_infinity()` | Reproduced |
| Euclidean nearest-reference-point distance | `pang_projected_igd_infinity()` with `cKDTree.query()` | Reproduced |
| Replace average IGD by maximum distance, Eq. (3) | `max(distances)` per pair | Reproduced |
| Take the maximum indicator over all objective-pair projections | `max(pair_scores)` | Reproduced |
| Authors' DTLZ2 quarter-disk and DTLZ7 `optimum1`/`optimum2` reference construction | `pang_projection_references()`; default `--coverage-samples 50000` | Reproduced |
| Exact objective-vector deduplication of the nondominated evaluation history to form UEA | `run_pang_archive_baseline.py` and `baselines/pang_2023_baseline.py` | Reproduced |

## Scope boundaries

The paper's exact projected indicator protocol is defined for its DTLZ2/DTLZ7
experiments. The project also stores reference fronts for RE and other DTLZ
tasks. Those tasks receive a clearly labelled `generalized_full_space`
IGD-infinity value for cross-method evaluation, but it must not be described
as the paper's projected DTLZ protocol.

Section IV-B additionally evaluates two external archive-subset algorithms
([17] distance-based selection and [18] K-means selection). They are separate
from the supplied `MOEADUEA1_100` search algorithm and are not injected into
the reported UEA. The local runner therefore does not add a project-defined
maximin or hole-selection postprocessor.

The comparison protocol intentionally differs from the paper in exactly three
experiment-design dimensions: the common FE budget, the project problem suite,
and 20 independent runs for formal statistics. These are comparison controls,
not changes to the Pang search or indicator definitions.

## Artifacts

For DTLZ2/DTLZ7, `summary.json` records every objective pair's IGD-infinity
score, the reference-point count, normalization source, and distance-space
definition. `fronts.npz` stores the complete evaluation history and the final
UEA, so the aggregate indicator can be independently recalculated.
