# Stage76 variable-k sampler repair

## Frozen scientific object

Stage76 reuses all 80 Stage75 CQMs without changing the pair objective, reward levels, allowed budgets, conditional quality thresholds, or three explicit constraints. Only initialization, cardinality moves, and temperature coupling change.

## Matched-budget comparison

Each stochastic method uses 8 repeats and 8192 state proposals per cell. Parallel-tempering exchanges reuse already evaluated energies and are reported separately. Classical warm-start construction cost is explicit and is not presented as a quantum speed advantage.

| Method | Exact-frontier match | Joint-tabu competitive | Frontier competitive | Frontier improvements |
|---|---:|---:|---:|---:|
| Stage75 cold global annealing | 0.650 | 0.300 | 0.287 | 0 |
| Stage76 cold adjacent annealing | 0.700 | 0.338 | 0.338 | 0 |
| Stage76 frontier-warm parallel tempering | 1.000 | 0.875 | 1.000 | 5 |

## Decision

- Stage75 CQM identity preserved: `True`.
- Standalone cold-start sampler repaired: `False`.
- Warm-start parallel-tempering fidelity gate: `True`.
- Local warm-start hardware-shaped emulation authorized: `True`.
- Cloud CQM, direct QPU, quantum scaling, and quantum advantage claims: `False`.

The frontier-warm result is a hybrid refinement route. It inherits classical fixed-k frontier information and therefore cannot be used as evidence of standalone quantum superiority.
