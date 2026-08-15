# Stage82 Lagrangian Fixed-k QUBO Gate

## Formulation

Stage82 decomposes the Stage75 variable-k CQM into fixed-k QUBOs. A frozen
quality-weight grid generates candidates, an analytically bounded cardinality
penalty enforces k, and a classical guard rejects every candidate that violates
the original conditional-quality threshold. The original Stage75 objective is
used for all final comparisons across weights and k values.

## Local gate result

- Fixed-k frontier-competitive cells: `0.4000`.
- Variable-k frontier-competitive models: `0.3125`.
- Maximum logical variables: `96`.
- Maximum QUBO interactions: `4560`.
- Minimum coefficient retention: `1.000000`.

## Decision

Limited Dirac-3 calibration authorized: `False`.
The route is a hybrid candidate-generation protocol, not an exact unconstrained
encoding and not evidence of quantum advantage. A physical run remains limited
to frozen positive and negative controls until hardware fidelity is measured.
