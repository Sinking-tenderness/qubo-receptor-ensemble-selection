# Stage53 PPARA large-pool QUBO transfer

| Method | k | Full robust BEDROC20 |
|---|---:|---:|
| bedroc_linear_topk | 1 | 0.845166 |
| bedroc_nested_greedy | 1 | 0.845166 |
| bedroc_random_search | 1 | 0.845166 |
| best_single_receptor | 1 | 0.845166 |
| rank_pair_direct_greedy | 2 | 0.794656 |
| rank_pair_qubo_exact | 2 | 0.763188 |
| rank_pair_strong_classical | 2 | 0.763188 |
| coverage_linear_additive | 6 | 0.714690 |
| all_receptors | 20 | 0.693606 |
| coverage_qubo_exact | 6 | 0.678145 |
| coverage_strong_classical | 6 | 0.678145 |
| coverage_direct_greedy | 6 | 0.678145 |

## Decision

Frozen QUBO application transfer: **NO-GO**.

Solver novelty over strong classical search: **NO-GO**.

Stage53 is train-only and post-hoc because the PPARA receptor pool was filtered using Stage51 outcomes. Scaffold holdouts test internal transfer only. A positive result can justify a new preregistered independent experiment, but cannot repair Stage51, establish QUBO superiority on independent data, authorize same-data tuning, or establish quantum execution, speedup, or advantage.
