# Stage66 cross-target auxiliary-variable coverage QUBO

## Scope

Four consumed historical development matrices were analyzed with no new docking,
protected-data read, or quantum-hardware job.

## Objective

The QUBO rewards multiscale active-ligand OR coverage at 5%, 10%, and 20% rank
thresholds, penalizes decoy OR exposure, and anchors selection with robust
singleton quality. Active and decoy coverage are represented by explicit binary
auxiliary variables rather than receptor-pair residuals.

## Selected historical-development candidate

- Candidate: `ms_all_any_d0p25_s0p5`
- Mean target gain over pair-off: -0.039041
- Worst-target gain over pair-off: -0.099323
- Nonnegative targets: 0/4
- Mean target gain over same-objective direct greedy: -0.000467
- Selection differences from same-objective greedy: 12

## Decision

Coverage objective freeze gate: **NO-GO**.

The same-objective classical comparison is a solver audit, not a claim that a
QUBO must beat the optimum of its own objective. Any freeze only authorizes
preregistration on a genuinely new target; it does not authorize hardware or
establish quantum advantage.
