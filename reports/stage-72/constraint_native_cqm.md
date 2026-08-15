# Stage72 constraint-native CQM

## Question

Can the frozen Stage70 receptor-portfolio objective retain its exact solution while moving cardinality and quality constraints out of penalty-expanded BQM coefficients?

## Formulation

Stage72 minimizes

$$
\sum_{i<j}(R_{ij}-c)x_ix_j+c\binom{k}{2},
$$

subject to

$$
\sum_i x_i=k, \qquad \sum_i d_i x_i\le D.
$$

Here $c=(\min R+\max R)/2$. Because every feasible state has exactly $\binom{k}{2}$ selected pairs, the midpoint transformation preserves the original objective exactly while reducing programmable full scale.

## Result

- Exact source optima preserved: `16/16`.
- Connected feasible swap graphs: `16/16`.
- Minimum normalized-gap improvement versus Stage71: `1.74219e+06x`.
- Maximum native logical variables: `96`.
- Maximum removed slack variables: `9`.

## Noise gates

- Matched Stage71 reference $10^{-6}$: quantization `True`, Gaussian `True`.
- Stress reference $10^{-3}$: quantization `True`, Gaussian `True`.
- Constraint-native formulation freeze authorized: `True`.
- Direct QPU execution authorized: `False`.

## Boundary

This establishes a precision-robust logical constrained model, not a hardware implementation. CQM hybrid solvers or constraint-preserving gate/annealing methods still require a separate solver-scaling and embedding study. No hardware speedup or quantum advantage is claimed.
