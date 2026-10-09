# GD-PSL Empirical Diagnostics and Archive Update

This document is the compact, implementation-aligned pseudocode for GD-PSL.
The direction, local-support, and radial-envelope checks are empirical
diagnostics. They guide pilot adaptation and label candidate outcomes; they do
not form a hard gate for the final archive.

```text
Algorithm: GD-PSL with empirical auditing
Input : problem P, base EA E, total FE budget B, EA ratio rho
Output: completed archive A, candidate audit R

1  B_EA <- floor(rho * B); B_FILL <- B - B_EA
2  H_EA <- RunPlatEMO(E, P, B_EA)
3  A_EA <- FiniteObjectiveUniqueNondominated(H_EA)
4  P_EA <- ArchiveDerivedPreferenceMap(A_EA)
5  D <- BuildSeededSimplexDetectionPool(P_EA)
6  Q_supported <- AllocateSupportedQueries(D, P_EA, B_FILL)
7  model <- TrainAndRestoreBest(P_EA, A_EA.decisions)
8  B_boundary <- AvailableBoundaryQuota(D, B_FILL)
9  Q_pilot <- PilotSubset(Q_supported, 0.2 * (B_FILL - B_boundary))
10 H_pilot <- ProposeRefineAndTrueEvaluate(Q_pilot, model, A_EA, P)
11 R_pilot <- EmpiricalAudit(H_pilot, A_EA)
12 S_pilot <- GeometricHole(R_pilot)
                   and DiagnosticPass(R_pilot)
                   and NondominatedAfterMerge(R_pilot)
13 Q_followup <- AdaptiveReallocation(
                     D, Q_pilot, S_pilot,
                     B_FILL - B_boundary - |Q_pilot|)
14 H_followup <- ProposeRefineAndTrueEvaluate(Q_followup, model, A_EA, P)
15 Q_boundary <- SelectUnusedLowSupportProbes(D, B_boundary)
16 H_boundary <- ProposeRefineAndTrueEvaluate(Q_boundary, model, A_EA, P)
17 H_fill <- H_pilot union H_followup union H_boundary
18 R <- EmpiricalAudit(H_fill, A_EA)
        // direction, local support, one-sided radial envelope
19 A <- FiniteObjectiveUniqueNondominated(A_EA union H_fill)
20 AttachArchiveStatus(R, A)
21 return A, R
```

Every decision in `H_fill` consumes one true function evaluation. Reference
Pareto fronts are absent from search, training, query allocation, diagnostics,
and archive construction; they are loaded only for post-run HV and projected
IGD-infinity measurement. A diagnostic flag and final archive retention are
independent labels, so a flagged candidate remains in the reported archive when
it is finite, objective-unique, and nondominated.
