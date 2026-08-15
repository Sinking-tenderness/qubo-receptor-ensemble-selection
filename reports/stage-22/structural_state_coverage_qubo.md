# Stage 22: structural-state coverage QUBO

This is a post-hoc, structure-only model-design screen. No ligand label or docking result was read.

## Frozen objective

`F(S) = covered structural-state fraction + 0.15 * mean pairwise structural distance`

Primary neighborhood fraction: 0.10; sensitivity fractions: 0.05, 0.10, 0.20.

## Primary results

| Target | k | Coverage QUBO | Direct greedy | Delta | Coverage | Diversity | Restart agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| MK14 | 2 | 0.587285 | 0.587285 | 0.000000 | 0.488372 | 0.659416 | 1.000 |
| MK14 | 3 | 0.781371 | 0.781371 | 0.000000 | 0.697674 | 0.557980 | 1.000 |
| MK14 | 4 | 0.945584 | 0.945584 | 0.000000 | 0.860465 | 0.567458 | 0.906 |
| MK14 | 6 | 1.053806 | 1.053806 | 0.000000 | 0.953488 | 0.668784 | 0.406 |
| MK14 | 8 | 1.107438 | 1.100730 | 0.006708 | 1.000000 | 0.716256 | 0.031 |
| PPARG | 2 | 0.667972 | 0.667972 | 0.000000 | 0.632653 | 0.235462 | 1.000 |
| PPARG | 3 | 0.831446 | 0.831446 | 0.000000 | 0.795918 | 0.236853 | 1.000 |
| PPARG | 4 | 0.926081 | 0.926081 | 0.000000 | 0.887755 | 0.255509 | 1.000 |
| PPARG | 6 | 1.022625 | 1.009079 | 0.013546 | 0.979592 | 0.286888 | 0.062 |
| PPARG | 8 | 1.071642 | 1.055606 | 0.016037 | 0.989796 | 0.545644 | 0.062 |

## Decision

Structural coverage gate passed: `false`.
Passing authorizes only preparation of a separate small matched-docking preregistration. It does not authorize docking execution or quantum hardware.

## Interpretation boundary

This post-hoc structure-only screen can show only that a structural-state coverage QUBO is mathematically valid, reproducible, and can improve its frozen structural objective over direct greedy selection on the existing hard-gate-eligible pools. It cannot establish ligand enrichment, docking benefit, prospective generalization, quantum speedup, quantum advantage, or biological benefit. A passing gate authorizes only a separately preregistered small matched Uni-Dock experiment.
