# Stage67 BEDROC-aligned rank-bin QUBO

## Scope

Stage67 uses four consumed historical development matrices only. It performs no
new docking, protected-data read, or quantum-hardware execution.

## Question

Can a QUBO preserve the continuous best-rank signal that Stage66 lost? The
continuous reference uses exp(-20*r). Rank-bin QUBOs approximate the same
function at B=4, 8, 16, and 32 levels without tuning biological weights.

## Continuous objective ceiling

- Mean target gain over pair-off: -0.059925
- Worst-target gain: -0.140440
- Nonnegative targets: 0/4

## B=32 QUBO approximation

- Mean target gain over pair-off: -0.054872
- Worst-target gain: -0.123957
- Mean subset Jaccard versus continuous reference: 0.896726
- Mean absolute training-objective quantization error: 0.006530

## Decision

- Continuous objective supported: **NO-GO**
- B=32 rank-bin QUBO frozen: **NO-GO**

This stage tests objective fidelity, not quantum speedup or quantum advantage.
