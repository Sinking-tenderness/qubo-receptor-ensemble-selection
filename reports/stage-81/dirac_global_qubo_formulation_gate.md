# Stage81 Dirac Global QUBO Formulation Gate

## Direct CQM-to-BQM route

The screen evaluated `6` penalty/precision conditions over 16
canonical variable-k protein models. The best cold-start condition reached a
raw-best feasible fraction of
`0.1250` and a frontier-competitive
fraction of `0.0000`. No condition
passed the frozen feasibility and competitiveness gate.

## Conservative quality prefilter

The per-receptor inner approximation retained enough receptors in
`84/100` fixed-k cells.
It fully retained the historical frontier in
`0/100` cells;
the mean retained fraction was
`0.5410`. This route changes the
scientific feasible set too aggressively and is rejected.

## Decision

No additional Dirac-3 global-QUBO submission is authorized. The remaining
trial allocation is preserved. A future physical run requires a formulation
that preserves the original conditional-quality constraint without a
float32-dominating penalty and without discarding the historical frontier.

This is an encoding no-go result, not evidence against the Stage75 scientific
objective or against quantum optimization in general.
