# MAPK14 Uncertainty-Aware QUBO Train Gate Record

## Scope

The preregistered gate used only the 160 development-train ligands, comprising
80 actives and 80 decoys. It retained the unchanged three-seed e32 matrix. No
e64 value replaced an e32 score, and no development-validation or test score
was read.

The analysis used four-fold nested grouped-scaffold cross-validation. Every
matrix was independently min-max normalized from its current training fold.
The QUBO grid contained 576 two- or three-receptor candidates, each with a seed
stability term and at least one non-cardinality quadratic interaction.

## Result

The gate failed and validation remains unavailable.

The final train-refit QUBO used the discriminative family, three receptors,
`min_score` aggregation, and the following subset:

- `MK14_3K3J_aligned`
- `MK14_3KQ7_aligned`
- `MK14_3MPT_aligned`

Its matched linear top-k comparator selected:

- `MK14_2BAJ_aligned`
- `MK14_3K3J_aligned`
- `MK14_3KQ7_aligned`

The QUBO passed the structural non-degeneracy checks. Its final subset differed
from linear top-k, its non-cardinality quadratic coefficients had a range of
1.0, and seed-specific subset fits were sufficiently stable.

It failed every preregistered performance comparison against matched linear
top-k:

| Matrix | QUBO BEDROC | Linear BEDROC | Delta |
| --- | ---: | ---: | ---: |
| Three-seed median | 0.841 | 0.872 | -0.032 |
| seed0 | 0.832 | 0.875 | -0.043 |
| seed1 | 0.849 | 0.878 | -0.029 |
| seed2 | 0.861 | 0.872 | -0.011 |

The mean individual-seed delta was -0.0277 and the worst individual-seed delta
was -0.0429. The result is not explained by one anomalous seed.

## Interpretation

The quadratic redundancy term did change the selected subset, but the change
was harmful. It penalized the strong `2BAJ`/`3K3J`/`3KQ7` group and replaced
`2BAJ` with the less discriminative `3MPT`. Structural or score diversity alone
therefore did not imply complementary early enrichment.

The selected final configuration reached a much higher ordinary four-fold
tuning estimate than its nested OOF estimate. That gap is expected after
searching 576 configurations and confirms why the nested result, not the
non-nested tuning value, must control promotion.

The QUBO remained above the median fixed subset, reaching the 78.6th percentile
on primary BEDROC and the 83.9th percentile on mean-seed BEDROC. This is useful
but insufficient: it did not beat the fair matched linear comparator.

## Next Step

Validation remains unavailable. The next train-only diagnostic will evaluate
each frozen QUBO candidate against its own matched linear top-k across the same
folds. Its purpose is to distinguish two cases:

1. The current objective family contains a viable non-degenerate candidate but
   the tuning rule fails to identify it reliably.
2. Every current candidate loses to its linear part, meaning the pairwise
   objective itself must be redesigned before another preregistered gate.

This diagnostic cannot retroactively pass the failed gate or nominate a subset
for validation.
