# Stage65 cross-target pair-sign mechanism adjudication

## Scope

Stage65 uses only consumed historical development matrices from BACE1, PPARG,
PPARA, and PPARD. It performs no docking, protected-data read, or hardware job.

## Edge transfer

- Mean fold train-versus-holdout pair-residual Spearman: +0.204642
- LCB-positive edge count: 10929
- LCB-positive holdout-positive rate: 0.620734
- All-edge holdout-positive rate: 0.551644

## Preregistered positive-LCB candidate

Candidate `lcb_positive_0p25` has mean target gain
-0.007060 and worst-target gain
-0.027054 over pair-off. It is
nonnegative on 1/4 targets.

## Decision

Continue pair-residual QUBO development: **NO-GO**.

If NO-GO, the next objective must use explicit ligand/region coverage auxiliary
variables rather than another rescaling of the same pair residual.

## Boundary

This is post-hoc mechanism adjudication. It does not establish independent
efficacy, solver advantage, quantum execution, speedup, or quantum advantage.
