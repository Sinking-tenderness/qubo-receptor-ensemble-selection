# Stage68 quality-plateau portfolio QUBO

## Question

Can a compact QUBO reduce functionally redundant receptor choices while retaining the strong additive pair-off screening baseline within a frozen training-uncertainty floor?

## Frozen formulation

For fixed $k$, the pair-off top-$k$ mean singleton utility defines $U_k^*$. The quality constraint is

$$
\frac{1}{k}\sum_i u_i x_i \ge U_k^* - m\sigma_k,
$$

where $m\in\{0.25,0.5,1.0\}$ and $\sigma_k$ is the root-sum-square jackknife spread of the pair-off top-$k$ set divided by $k$. Within that feasible plateau, Stage68 minimizes

$$
\sum_{i<j} R_{ij}x_ix_j,
\qquad
R_{ij}=\max\left(0,\min_s \mathrm{corr}(r_{s,i},r_{s,j})\right).
$$

Only receptor rank correlations positive in all three docking seeds are penalized.

## Development result

- Selected multiplier: `0.5x`.
- Mean target BEDROC20 gain versus pair-off: `-0.000920`.
- Worst target mean gain: `-0.009346`.
- Mean stable-redundancy reduction: `0.029811`.
- Targets within 0.01 BEDROC of pair-off: `4/4`.

## Transfer and QUBO fidelity

- Leave-one-target-out mean held-target gain: `-0.004057`.
- Leave-one-target-out worst held-target gain: `-0.009346`.
- QUBO/continuous mean subset Jaccard: `1.000000`.
- QUBO mean holdout gap versus continuous certificate: `+0.000000`.
- Maximum logical variables: `105`.
- Maximum coefficient dynamic range: `1.65721e+07`.

## Decision boundary

The objective may be frozen only for preregistration on a genuinely new target if every Stage68 route gate passes. Exact MILP is retained as the strongest classical reference. This stage does not establish independent efficacy, solver speedup, quantum execution, or quantum advantage. Hardware execution remains blocked until coefficient compression and embedding are audited.

An unfrozen alternate jackknife partition changed the route decision from pass to fail. Therefore this is a protocol-specific development freeze, not evidence of partition robustness; the alternate probe is retained under `analysis/stage68_unfrozen_partition_probe_20260806`.
