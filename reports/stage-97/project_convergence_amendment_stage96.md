# Stage97 Stage96 convergence amendment

Stage89 remains preserved as the original frozen claim ledger. Stage96 adds a completed hidden-matrix replay on real PPARG and BACE1 Uni-Dock matrices.

## Decision

The Stage96 adaptive docking policy gate failed. At the primary 20% task budget, QUBO exact was below the best non-QUBO policy on both targets, and exact QUBO never exceeded the classical one-swap solver on the same candidate problems.

The project therefore remains a feasibility-and-boundary study. Manuscript preparation and reproducibility packaging continue; new objective tuning, new target docking, and quantum hardware spending remain blocked until a new preregistered instance passes an independent classical-hardness gate.

## Stage96 evidence

- PPARG: QUBO exact mean BEDROC=0.649755; best non-QUBO=0.783702; mean gain=-0.133947; pass=False.
- BACE1: QUBO exact mean BEDROC=0.948285; best non-QUBO=0.965811; mean gain=-0.017526; pass=False.

## Interpretation

This is a useful negative result: it rules out the current adaptive-QUBO formulation as a demonstrated advantage on the available matrices. It does not prove that every future protein or every quantum algorithm will fail, but it does define the evidence required before reopening the experimental branch.
