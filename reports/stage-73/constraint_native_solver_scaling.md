# Stage73 constraint-native solver scaling

## Question

Do the current $k=3$ constraint-native receptor-selection instances require a global quantum or hybrid solver, and when do budget-matched classical methods begin to fail as the pool and feasible region grow?

## Protocol

Nested pools are ordered only by integer quality deficit and a receptor-ID hash. Each pool is tested under the frozen quality floor, a deterministic 10% feasible-density floor, and no quality floor. Exact enumeration supplies the oracle. Single-start greedy, budgeted random feasible sampling, budgeted multistart greedy, and constraint-preserving simulated annealing are compared using deterministic work-unit accounting; cloud CQM and quantum hardware are not executed.

## Scale

- Workload cells: `276`.
- Largest candidate pool: `96`.
- Largest fixed-$k$ search space: `142880`.
- Largest full-pool frozen feasible set: `17`.
- Current exact-enumeration tractability gate: `True`.

## Full-pool frozen task

- Multistart greedy success: `1.000`.
- Constraint-preserving annealing success: `1.000`.
- Random feasible success: `1.000`.

## Hardness

- Single-start greedy failed workload cells: `39`.
- Budgeted multistart greedy failed trials: `51`.
- Constraint-preserving annealing failed trials: `67`.

## Decision

- Larger-$k$ scaling study authorized: `True`.
- Direct QPU execution authorized: `False`.
- Quantum scaling/advantage claim authorized: `False` / `False`.

The current $k=3$ instances are a logical formulation proof, not yet a quantum-scale workload. Stage73 uses oracle-feasible initialization for the budgeted stochastic methods and reports operation counts rather than claiming end-to-end solver speedup.
