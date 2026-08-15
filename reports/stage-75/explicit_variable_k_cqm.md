# Stage75 explicit variable-k constraint-native CQM

## Formulation

For reward level $\rho$, minimize

$$
E_\rho(x)=\sum_{i<j}(r_{ij}-\rho)x_ix_j
$$

subject to one-hot cardinality selection, $\sum_i x_i=\sum_k k y_k$, and the Stage74 balanced-10% conditional quality constraint $\sum_i d_ix_i\leq\sum_k D_k y_k$. Reward levels are frozen pair-coefficient order statistics at 10%, 25%, 50%, 75%, and 90%.

## Encoding

- CQM models: `80`.
- Maximum logical variables / quadratic couplers: `103` / `4560`.
- Explicit constraints: `3`.
- Maximum energy-identity residual: `3.908e-14`.
- Monotonic reward paths: `16/16`.
- Distinct selected budgets: `[3, 4, 6, 8, 10, 12, 16]`.

## Solver comparison

- Exact fixed-$k$ frontier cells: `20`.
- Joint classical exact-frontier match: `1.000`.
- Variable annealing exact-frontier match: `0.650`.
- Annealing competitive with joint classical: `0.300`.
- Annealing competitive with frozen fixed-$k$ frontier: `0.287`.
- Annealing wins / joint-classical wins / ties: `1` / `56` / `23`.
- Stage74 pooled frontiers refined: `11`.

## Decision

- Explicit variable-$k$ CQM freeze authorized: `True`.
- Local hardware-shaped emulation authorized: `False`.
- Cloud CQM / direct QPU / quantum claims authorized: `False / False / False`.

The reward path is an optimization trade-off study, not a biological estimate of the best receptor count. Non-exact Stage74 frontiers remain pooled best-known references and may be improved by Stage75 joint solvers.
