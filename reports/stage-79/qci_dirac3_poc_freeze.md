# Stage79 QCI Dirac-3 Local Move-QUBO PoC Freeze

## Frozen panel

- Instances: `6`.
- Confirmation positives: `2`.
- Confirmation negatives: `3`.
- Calibration diagnostics: `1`.
- Maximum variables: `40`.
- Maximum quadratic interactions: `780`.

The Stage78 variables already describe local receptor swaps around a frozen
classical solution. The warm solution is the all-zero move vector, so an XOR
warm-start transformation would be the identity. Stage79 therefore submits the
same local search landscape to Dirac-3 as a degree-two integer polynomial with
two levels per variable.

## Translation

The unsupported constant offset is omitted from the QCI payload. Every other
coefficient is divided by its instance maximum absolute value and rounded to
float32. All returned bit vectors will be classified by the original Stage78
float64 BQM, not by the device-reported energy. Quantized exact MILP checks
preserved the expected positive/negative role for all instances.

## External stop

Local preparation made zero QCI queries and zero device submissions. The next
step is an allocation-only preflight. It requires a QCI token but consumes no
planned device sample. Calibration and confirmation require a separate double
acknowledgement.

The frozen plan uses `4` calibration
jobs and `5` confirmation jobs,
for `600` planned Dirac-3 samples. The
protocol refuses a paid allocation and caps recorded device use at
`480` seconds.

## Claim boundary

This is a cross-hardware physical optimization proof of concept. It does not
authorize a quantum-advantage, scaling, biological-generalization,
drug-discovery, or end-to-end speedup claim.
