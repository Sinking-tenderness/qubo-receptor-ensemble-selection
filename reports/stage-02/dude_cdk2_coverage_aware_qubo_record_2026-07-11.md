# Coverage-Aware Receptor QUBO

## Final QUBO Convention

The implemented QUBO uses the explicit convention:

\[
Q(x)=c+\sum_i h_i x_i+\sum_{i<j}J_{ij}x_ix_j,
\qquad x_i\in\{0,1\}.
\]

`x_i=1` means that receptor i is selected. The coefficient table is written
to `results/metrics/dude_cdk2_coverage_qubo_train.json` under
`qubo_coefficients`.

## Terms

The current objective combines:

1. Train receptor utility, using normalized train ROC-AUC.
2. Score-correlation redundancy penalty.
3. Receptor count and soft target-size penalty.
4. Active coverage reward at train Top10%.
5. Pairwise overlap penalty for active coverage sets.

For receptor i, let (A_i) be its train Top10% active set. The coverage part
is:

\[
-\lambda_c\sum_i \frac{|A_i|}{N_A}x_i
 +\lambda_o\sum_{i<j}\frac{|A_i\cap A_j|}{N_A}x_ix_j.
\]

For exactly two selected receptors, this rewards the normalized active union:

\[
|A_i\cup A_j|=|A_i|+|A_j|-|A_i\cap A_j|.
\]

For more than two receptors, the pairwise form is an approximation because
triple intersections are not represented. An exact general OR encoding would
require auxiliary variables and additional quadratization constraints.

## Current Configuration And Result

- Target size: `K=2`.
- Coverage fraction: `0.10`.
- Coverage weight: `0.50`.
- Coverage-overlap weight: `0.50`.
- Redundancy weight: `0.25`.
- Count weight: `0.10`.
- Size penalty: `1.0`.
- Utility: normalized train ROC-AUC.

The exhaustive minimum is:

```text
1AQ1 + 1JVP
```

The result agrees with the classical train-selected baseline. This is a
consistency result, not evidence of quantum advantage or biological optimality.

## Verification

All eight possible subsets of the three-receptor pool were evaluated. The
direct objective and the explicit coefficient form agreed to within
`6.7e-16`. The full test suite passed with `15 passed`.

## Scope Boundary

The QUBO is now fully constructed and executable for the current score matrix,
but the coverage and redundancy terms are still teaching-scale proxies. The
next research upgrade is to derive interaction-level features from docked
poses and assess whether they improve held-out early enrichment without
leaking test information.
