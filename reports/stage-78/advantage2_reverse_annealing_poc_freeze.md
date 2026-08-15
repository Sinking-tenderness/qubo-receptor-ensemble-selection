# Stage78 Advantage2 Reverse-Annealing PoC Freeze

## Purpose

Freeze a minimal, falsifiable hardware experiment from the passing Stage77 local-BQM gate. This stage performs no cloud query and no QPU sampling.

## Frozen Instances

- Canonical reward quantile: `0.5`.
- Paid-run primary set: `6` independent fixed-k BQMs.
- Hardware-resolvable positives: `2`.
- Cross-target hard negatives: `3`.
- Sub-resolution calibration diagnostic: `1`; used for hardware-only tuning and excluded from confirmation endpoints.
- Maximum logical variables / interactions: `40` / `780`.

The sub-resolution PPARG diagnostic is used only for hardware calibration. Both hardware-resolvable PPARG positives and the three target-matched negatives remain untouched until confirmation. Reward quantiles are not treated as replicates.

## Local Evidence

- Every frozen BQM has an independently checkable SciPy MILP optimum and a warm all-zero state.
- Classical controls include cold and warm simulated annealing, warm tabu, and warm steepest descent.
- Wall-clock time is recorded nowhere as a scientific endpoint.

## External Stop Boundary

The next command that cannot be completed locally is the Leap preflight: it needs a D-Wave Leap account, API token, and access to an Advantage2 Zephyr QPU. Preflight queries the current working graph and creates physical embeddings but performs no QPU sampling. Calibration and confirmation additionally require two explicit paid-execution acknowledgements.

## Claim Boundary

Stage78 preregisters a physical-hardware proof of concept. It does not authorize a quantum-advantage, scaling, biological-generalization, or end-to-end speedup claim.
