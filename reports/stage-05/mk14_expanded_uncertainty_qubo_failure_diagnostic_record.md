# MAPK14 QUBO Failure Diagnostic Record

## Question

The preregistered uncertainty-aware train gate selected a non-degenerate QUBO
but lost to its matched linear top-k comparator. This post hoc development-only
diagnostic asked whether the complete frozen grid contained any QUBO candidate
that robustly improved on its own linear part.

The diagnostic did not change the failed gate and did not read development-
validation or test rows.

## Scan

All 576 previously frozen QUBO configurations were evaluated by fixed four-fold
grouped-scaffold OOF prediction. Each QUBO was compared with the top-k receptors
from its own linear coefficients, using the same target size and score
aggregation.

A diagnostic candidate had to satisfy all of the following:

- nonnegative primary median-matrix BEDROC delta;
- nonnegative mean individual-seed BEDROC delta;
- nonnegative BEDROC delta for every individual seed;
- a full-train subset different from matched linear top-k; and
- mean seed-specific subset Jaccard of at least 0.5 both across folds and after
  the full-train refit.

## Result

The candidate family contains viable train-only structures:

- 130 of 576 candidates were non-worse on primary BEDROC.
- 132 were non-worse on mean individual-seed BEDROC.
- 118 were non-worse on every individual seed.
- 530 produced a full-train subset different from matched linear top-k.
- 97 passed every diagnostic check.

Most viable candidates were three-receptor `min_score` models:

| Family | Size | Aggregation | Passing candidates |
| --- | ---: | --- | ---: |
| discriminative QUBO | 3 | min score | 61 |
| coverage QUBO | 3 | min score | 22 |
| discriminative QUBO | 2 | min/mean score | 12 |
| other | 3 | mean score | 2 |

The best candidate by worst individual-seed delta used:

- family: discriminative QUBO;
- size: 3;
- aggregation: `min_score`;
- weights: active coverage 0.0, active overlap 1.0, decoy exposure 0.5,
  redundancy 1.0, seed stability 1.5; and
- full-train subset: `2QD9 + 3KQ7 + 3MPT`.

Its matched linear subset was `2BAJ + 3KQ7 + 3MPT`. OOF primary BEDROC
increased from 0.839 to 0.889, a delta of +0.050. Mean individual-seed delta
was +0.055 and the worst individual-seed delta was +0.051.

## Interpretation

The original gate failed because its inner tuning rule maximized absolute QUBO
BEDROC. That rule could select a QUBO with a strong-looking inner score even
when its quadratic terms made the subset worse than the same model's linear
component. The candidate family itself is not empty or universally harmful.

This diagnostic is deliberately post hoc. The 97 candidates cannot be promoted
directly, and the best observed candidate cannot be called validated. The result
only supports testing a new algorithmic rule: select QUBO hyperparameters by
robust paired improvement over matched linear top-k inside each inner fold.

## Next Step

A second train-only nested gate must be preregistered before execution. Its
inner selection order will prioritize the worst individual-seed paired BEDROC
delta, then primary and mean-seed paired deltas, while retaining absolute
performance and seed-stability safeguards. The outer OOF comparison will remain
unseen until that selector is frozen.

Development-validation remains unavailable and test ligands remain locked.
