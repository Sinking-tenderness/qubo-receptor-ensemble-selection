# Stage 10 MAPK14 Expanded16 QUBO-Greedy Screen

## Scope

This is a Train-696 diagnostic using only the audited Stage 09 Uni-Dock matrix.
Frozen objective definitions and weights were transferred without retuning.
No validation or test rows were read.

## Objective Results

| Objective | Trials | Greedy failures | Max regret | Held-out exact better than QUBO greedy | Held-out exact better than direct greedy |
|---|---:|---:|---:|---:|---:|
| coverage_qubo | 140 | 36 | 0.184734 | 19 | 16 |
| pair_synergy_qubo | 140 | 27 | 0.274998 | 16 | 18 |
| pair_utility_qubo | 140 | 1 | 0.006186 | 0 | 0 |

## Interpretation

A positive objective regret proves only that forward greedy missed the exact optimum of a frozen quadratic objective. Held-out training-fold improvement is hypothesis-generating, not independent validation. This screen does not access validation/test rows, run a quantum device, or establish QUBO or quantum advantage.
