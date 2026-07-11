# QUBO Utility Scale Calibration

## Problem Found

ROC-AUC and BEDROC are approximately bounded between 0 and 1, while EF5%
can be greater than 1. Putting their raw values into the same QUBO changes
the relative meaning of the utility, redundancy, and size terms. In the
initial unnormalized EF5 experiment, the soft target-size constraint could
be overridden even when `K=1`.

This was a modeling-scale issue, not biological evidence.

## Fix

`solve_qubo_receptor_subset.py` and
`sensitivity_qubo_receptor_subset.py` now support:

```text
--utility-normalization none
--utility-normalization minmax
```

With `minmax`, utilities across the candidate receptor pool are mapped to
the interval [0,1]. If all candidate utilities are identical, each is set to
0.5 so that the utility term does not arbitrarily select a receptor.

## Normalized Sensitivity Results

- ROC-AUC utility: K=1 selects 1JVP; K=2 selects 1AQ1+1JVP.
- BEDROC utility: K=1 selects 1AQ1; K=2 selects 1AQ1+1JVP.
- EF5% utility: K=1 selects 1AQ1; K=2 selects 1HCL+1JVP for nonzero
  redundancy penalties.

The EF5 result differs because the train split has only six actives and the
EF5 values are coarse. It should not be interpreted as a stable biological
preference.

## Lesson For Future QUBO Design

Every objective term needs an explicit scale and interpretation. Before
comparing utility choices, we must freeze the normalization rule, choose
weights on train/validation only, and report sensitivity. Otherwise a larger
numeric metric can dominate the optimization for purely mathematical reasons.
