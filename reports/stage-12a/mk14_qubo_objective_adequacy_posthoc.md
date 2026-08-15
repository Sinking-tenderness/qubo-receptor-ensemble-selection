# Stage 12A MAPK14 QUBO Objective Adequacy Diagnostic

## Scope

This is a post hoc development diagnostic over all 560 three-receptor subsets.
It reads only Stage 09 Train-696 rows and never reads Stage 11 validation scores.
The provisional v2 candidate requires independent protein-target validation.

## Frozen v1 diagnosis

| Quantity | Value |
|---|---:|
| Mean train rank correlation, -energy vs robust BEDROC | 0.941489 |
| Mean holdout rank correlation, -energy vs robust BEDROC | 0.608927 |
| Mean train-to-holdout subset rank correlation | 0.589513 |
| Mean v1 holdout delta vs direct greedy | +0.018478 |

## Surrogate comparison

| Model | Alpha | Holdout composite | Holdout rank rho | Delta vs v1 | Delta vs direct greedy | Fold wins vs direct |
|---|---:|---:|---:|---:|---:|---:|
| additive | 1e-06 | 0.933670 | 0.578424 | -0.004258 | +0.014220 | 2/4 |
| quadratic | 1 | 0.937333 | 0.588943 | -0.000595 | +0.017883 | 1/4 |

## Provisional v2

- Selected alpha: `1.0`
- Full-train subset: `MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3BV2_aligned`
- Exact penalized-QUBO subset: `MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3BV2_aligned`
- Development status: `stage12a_no_quadratic_v2_gate_retain_v1_for_external_testing`

## Interpretation

- The best quadratic surrogate improves mean holdout composite over the best additive surrogate by +0.003663, so pair terms contain real signal.
- It changes mean holdout composite versus frozen v1 by -0.000595 and wins 0/4 folds.
- The full-train v2 returns the same three-receptor subset already tested in Stage 11, so it creates no new MAPK14 confirmatory candidate.
- Retain v1 for the external-target pilot; do not spend more docking on a retuned MAPK14 objective.

## Decision boundary

This analysis was designed after the Stage 11 outcome was known and is therefore post hoc development evidence only. It reads Stage 09 Train-696 rows but no Stage 11 validation or test row. It may diagnose objective mismatch and nominate a provisional v2 QUBO for new-target testing; it cannot rescue or reinterpret the failed Stage 11 confirmatory gate, establish QUBO superiority, or establish quantum advantage.
