# Stage71 QUBO coefficient-noise robustness

## Question

Does the frozen Stage70 logical BQM preserve its selected receptor subset after full-scale coefficient quantization or independent additive noise, and can a fixed classical annealing budget recover the unperturbed solution?

## Separation of effects

Stage71 reports two distinct quantities. Exact enumeration over every fixed-$k$ quality-feasible receptor subset measures whether perturbed coefficients change the mathematical optimum. Full-BQM simulated annealing plus steepest descent separately measures finite-budget recovery and constraint leakage. The latter is a software diagnostic, not a quantum-hardware benchmark.

## Exact landscape

- Frozen models checked: `16`.
- Unique Stage70 optima recovered exactly: `16/16`.
- Minimum normalized feasible energy gap: `1.09716e-08`.
- Maximum normalized feasible energy gap: `2.14062e-07`.

## Sampler calibration

- Zero-noise best feasible recovery: `13/16`.
- Zero-noise exact-subset recovery: `11/16`.
- Calibration gate passed: `True`.

## Noise envelope

- Quantization: largest tested full-scale step passing the project gate = `3e-09`.
- Gaussian noise: largest tested full-scale sigma passing the project gate = `1e-09`.
- Reference quantization gate passed: `False`.
- Reference Gaussian gate passed: `False`.
- Coefficient-robust logical BQM gate passed: `False`.

## Decision boundary

- Direct QPU execution authorized: `False`.
- Constraint-native reformulation authorized: `True`.
- New-target preregistration remains authorized: `True`.
- Quantum-advantage claim authorized: `False`.

The tested noise levels are project sensitivity probes, not specifications for any physical annealer. This post-hoc analysis uses four consumed development targets and cannot establish independent efficacy, embedding quality, hardware speedup, or quantum advantage.
