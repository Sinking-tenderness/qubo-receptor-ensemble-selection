# MAPK14 Delta-Aware QUBO Train Gate Record

## Scope

The second train-only gate inherited the same 160 ligands, eight receptors,
three e32 seeds, four grouped-scaffold folds, and 576 QUBO candidates. No score
matrix or weight grid changed. Development-validation and test rows remained
unavailable.

The only algorithmic change was the inner selection order. Candidate tuning
prioritized the worst individual-seed paired BEDROC improvement over each
candidate's matched linear top-k comparator.

## Result

The delta-aware nested gate failed.

The nested OOF primary BEDROC values were:

- delta-aware QUBO: 0.831;
- matched linear top-k: 0.880; and
- single best receptor: 0.861.

The QUBO-minus-linear primary delta was -0.049. Individual-seed deltas were
-0.067, -0.050, and -0.044. The QUBO also remained below single best on the
primary matrix and every seed.

The final full-train QUBO and matched linear top-k both selected
`2BAJ + 3KQ7 + 3MPT`, so the final quadratic terms did not change the selected
subset even though the non-cardinality coefficients themselves were nonzero and
nonconstant.

## Selection Optimism

Each outer split selected a candidate with a positive inner estimate:

| Outer fold | Expected worst-seed delta | Held-out primary delta |
| ---: | ---: | ---: |
| 0 | +0.030 | -0.005 |
| 1 | +0.052 | -0.197 |
| 2 | +0.105 | 0.000 |
| 3 | +0.029 | 0.000 |

Fold 1 caused the largest failure. Its inner selector expected a robust gain but
the chosen QUBO reduced held-out primary BEDROC by about 0.197 relative to its
linear component.

## Interpretation

Optimizing paired improvement across 576 configurations did not solve the
problem. It increased selection pressure on a small data set and found
fold-specific gains that failed on unseen scaffold groups.

This result does not contradict the post hoc finding that some fixed candidates
have positive four-fold OOF deltas. It shows that the current nested selector
cannot reliably identify them from only the inner folds.

## Next Step

No further broad hyperparameter search is justified on these 160 ligands. The
candidate diagnostic already froze one deterministic candidate as best by worst
individual-seed delta. That single candidate can be evaluated without tuning
across multiple preregistered grouped-scaffold partitions.

If the fixed candidate is not stable across repeated partitions, the next step
must be a larger development-train matrix rather than another gate amendment.
If it is stable, it may be nominated for one genuinely independent development-
validation experiment, with its post hoc origin disclosed.

Validation remains unavailable and test ligands remain locked.
