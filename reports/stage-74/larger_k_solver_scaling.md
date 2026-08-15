# Stage74 larger-k constraint-native solver scaling

## Question

Does the frozen pair-redundancy CQM become computationally nontrivial when receptor cardinality grows beyond $k=3$, while retaining deterministic quality constraints?

## Protocol

The complete receptor pools and frozen Stage72 pair coefficients are reused. Cardinalities are drawn from $k\in\{3,4,6,8,10,12,16\}$ subject to pool size. Quality thresholds contain at least 1%, 10%, or 100% of fixed-$k$ subsets and are computed exactly by integer subset-sum dynamic programming. Exact enumeration is used only when $\binom{n}{k}\leq 200,000$; larger cells use an explicitly labelled pooled best-known reference.

## Scale

- Models / model-$k$ pairs / workload cells: `16` / `100` / `300`.
- Solver trials: `7620`.
- Largest state space: `662252084388541314` ($10^{17.82}$).
- Exact-oracle workload cells: `120`.
- Largest logical model: `96` variables and `4560` pair couplers.

## Solver validation and hardness

- Strong classical exact-cell success: `1.000`.
- Annealing exact-cell success: `1.000`.
- Non-exact solver disagreement: `108/180` (`0.600`).
- Annealing strict wins / classical strict wins / ties: `19` / `41` / `120`.

## Decision

- Larger-scale state-space gate: `True`.
- Exact classical validation gate: `True`.
- Solver-hardness gate: `True`.
- Explicit variable-$k$ CQM design authorized: `True`.
- Hardware-shaped sampler PoC authorized: `True`.
- Direct QPU / quantum-scaling / quantum-advantage claims: `False / False / False`.

Large cells do not have certified global optima. A pooled best-known solution is not an exact oracle, and solver disagreement is evidence of optimization difficulty rather than evidence of quantum advantage or biological benefit.
