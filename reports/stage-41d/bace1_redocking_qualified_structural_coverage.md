# Stage 41d BACE1 redocking-qualified structural coverage

## Decision

Stage 41d gives a conditional go for a new, explicitly post-hoc BACE1
development route using the 34 Stage 41c redocking-qualified receptors. This
does not rescue or reinterpret the failed Stage 41c confirmatory gate.

All five descriptive development criteria passed:

| Metric | Observed | Development criterion |
|---|---:|---:|
| Qualified receptor count | 34 | at least 30 |
| Pairwise 95th-percentile distance retention | 0.9702 | at least 0.90 |
| Structural diameter retention | 0.8438 | at least 0.80 |
| Maximum failed-to-passing distance ratio | 1.2253 | at most 1.25 |
| Maximum landmark cover-radius inflation | 1.2811 | at most 1.30 |

The thresholds are outcome-informed descriptive heuristics because Stage 41c
results and preliminary structural coverage were already known. They support a
resource decision only and are not independent confirmatory evidence.

## What was retained

The passing pool preserves 98.8% of the full-pool mean pairwise distance and
97.0% of its 95th-percentile distance. Farthest-first landmarks drawn only from
the passing pool cover all frozen structures with 4.4% to 28.1% radius
inflation across landmark budgets 4, 8, 12, and 16.

The largest structural losses are 4I1C and 4I12. Their nearest passing
representatives are 4DJW and 4DJV at standardized distances 1.2649 and 1.2112,
respectively. The reference 3L5D structure failed redocking but is structurally
represented by passing 3L59 at distance 0.5990.

## Prospective scale

The 34-receptor pool contains 1,676,115 candidate subsets across k=1 through
k=6, including 1,344,904 six-receptor subsets. A 266-ligand, three-seed
functional matrix would require 27,132 receptor-ligand-seed pairs.

## Next action

Freeze Stage 42 before functional docking. Stage 42 must identify the 34
receptors as an outcome-informed development pool, preserve the failed Stage
41c record, freeze the 266 development ligands and three seeds, and prohibit
fresh-validation or locked-test access. Its purpose is objective construction
and classical-versus-QUBO scaling analysis, not independent efficacy
validation or a quantum-advantage claim.
