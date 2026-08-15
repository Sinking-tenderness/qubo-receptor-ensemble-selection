# MAPK14 Train-only Greedy Failure Screen

## Scope

This post hoc diagnostic uses Train-696 only. It compares deterministic
forward greedy selection with exact fixed-cardinality enumeration. No fresh
validation or locked-test rows were read. Exact enumeration is an optimization
reference; it is not evidence that a quantum device found the solution.

## Full eight-receptor QUBO objective screen

| Objective | Trials | Strict failures | Rate | Max regret | Held-out exact-better fraction |
|---|---:|---:|---:|---:|---:|
| coverage_qubo | 120 | 8 | 0.067 | 0.183322 | 0.750 |
| pair_synergy_qubo | 120 | 13 | 0.108 | 0.503135 | 0.364 |
| pair_utility_qubo | 120 | 2 | 0.017 | 0.027455 | 1.000 |

## Frozen primary-source budget-three check

| Objective | Contexts | Strict failures |
|---|---:|---:|
| coverage_qubo | 5 | 0 |
| pair_synergy_qubo | 5 | 0 |
| pair_utility_qubo | 5 | 0 |

## Receptor-pool growth screen at budget three

The full eight-receptor coefficient normalization is held fixed while the
available variable pool is restricted. This isolates search behavior from
changes in coefficient estimation.

| Objective | Pool size | Trials | Strict failures | Rate |
|---|---:|---:|---:|---:|
| coverage_qubo | 4 | 1400 | 5 | 0.004 |
| coverage_qubo | 5 | 1120 | 17 | 0.015 |
| coverage_qubo | 6 | 560 | 21 | 0.037 |
| coverage_qubo | 7 | 160 | 11 | 0.069 |
| coverage_qubo | 8 | 20 | 2 | 0.100 |
| pair_synergy_qubo | 4 | 1400 | 52 | 0.037 |
| pair_synergy_qubo | 5 | 1120 | 57 | 0.051 |
| pair_synergy_qubo | 6 | 560 | 64 | 0.114 |
| pair_synergy_qubo | 7 | 160 | 29 | 0.181 |
| pair_synergy_qubo | 8 | 20 | 4 | 0.200 |
| pair_utility_qubo | 4 | 1400 | 2 | 0.001 |
| pair_utility_qubo | 5 | 1120 | 8 | 0.007 |
| pair_utility_qubo | 6 | 560 | 12 | 0.021 |
| pair_utility_qubo | 7 | 160 | 8 | 0.050 |
| pair_utility_qubo | 8 | 20 | 2 | 0.100 |

## Direct robust-BEDROC selection screen

This second check optimizes the existing robust metric hierarchy directly,
rather than optimizing a fitted QUBO surrogate.

| Aggregation | Full-pool contexts | Strict failures | Rate |
|---|---:|---:|---:|
| mean_score | 5 | 1 | 0.200 |
| min_score | 5 | 2 | 0.400 |

| Aggregation | Pool size | Trials | Strict failures | Rate |
|---|---:|---:|---:|---:|
| mean_score | 4 | 350 | 11 | 0.031 |
| mean_score | 5 | 280 | 31 | 0.111 |
| mean_score | 6 | 140 | 31 | 0.221 |
| mean_score | 7 | 40 | 12 | 0.300 |
| mean_score | 8 | 5 | 1 | 0.200 |
| min_score | 4 | 350 | 4 | 0.011 |
| min_score | 5 | 280 | 14 | 0.050 |
| min_score | 6 | 140 | 18 | 0.129 |
| min_score | 7 | 40 | 10 | 0.250 |
| min_score | 8 | 5 | 2 | 0.400 |

## Largest full-pool objective regrets

- pair_synergy_qubo, outer_fold_2, seed0, k=3: regret=0.503135; greedy=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3KQ7_aligned; exact=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3MPT_aligned
- pair_synergy_qubo, outer_fold_2, seed0, k=4: regret=0.334059; greedy=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3KQ7_aligned+MK14_3MPT_aligned; exact=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3K3J_aligned+MK14_3MPT_aligned
- pair_synergy_qubo, outer_fold_3, seed0, k=3: regret=0.283789; greedy=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3KQ7_aligned; exact=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3MPT_aligned
- pair_synergy_qubo, full_train, seed0, k=3: regret=0.277226; greedy=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3KQ7_aligned; exact=MK14_2BAJ_aligned+MK14_2QD9_reference+MK14_3MPT_aligned
- pair_synergy_qubo, outer_fold_2, seed0, k=2: regret=0.244618; greedy=MK14_2QD9_reference+MK14_3KQ7_aligned; exact=MK14_2BAJ_aligned+MK14_2QD9_reference
- coverage_qubo, outer_fold_0, seed1, k=2: regret=0.183322; greedy=MK14_3KQ7_aligned+MK14_3MPT_aligned; exact=MK14_2BAJ_aligned+MK14_3MPT_aligned
- coverage_qubo, outer_fold_0, seed2, k=2: regret=0.169008; greedy=MK14_3KQ7_aligned+MK14_3MPT_aligned; exact=MK14_2BAJ_aligned+MK14_3MPT_aligned
- pair_synergy_qubo, full_train, seed0, k=2: regret=0.161797; greedy=MK14_2QD9_reference+MK14_3KQ7_aligned; exact=MK14_2BAJ_aligned+MK14_3MPT_aligned
- pair_synergy_qubo, outer_fold_2, seed1, k=2: regret=0.099536; greedy=MK14_2QD9_reference+MK14_3KQ7_aligned; exact=MK14_2BAJ_aligned+MK14_2QD9_reference
- coverage_qubo, outer_fold_0, primary, k=2: regret=0.095120; greedy=MK14_3KQ7_aligned+MK14_3MPT_aligned; exact=MK14_2BAJ_aligned+MK14_3MPT_aligned

## Interpretation

Strict greedy local optima observed: `true`.
Frozen pair-synergy primary-source budget-three failures: `0`.
Across outer-fold QUBO failure cases, exact selection improved held-out primary BEDROC in `12` of `21` cases (`0.571`).

A positive QUBO objective regret proves only that forward greedy missed the
exact optimum of the fitted quadratic objective. Better held-out BEDROC must
be demonstrated separately, and neither result establishes quantum advantage.
