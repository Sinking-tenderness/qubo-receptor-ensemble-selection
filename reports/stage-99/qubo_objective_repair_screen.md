# Stage99 QUBO objective repair screen

This screen uses existing docking matrices only. Train labels are used to fit the selector inside each outer fold; test labels are used only for BEDROC evaluation.

## Frozen repair

- Percentile-normalize each receptor on outer-train rows.
- Use `exp(-20*r)` to match BEDROC alpha=20 early recognition.
- Define singleton utility as active support minus `1.5` times decoy exposure.
- Define pair utility from the mean of two receptor ranks, so a single extreme score is insufficient.
- Select k in {1,2,3} by inner group validation and compare exact QUBO against greedy plus one-swap.

## Gate

- Fixed-k=3 mean gain over the best single receptor: `0.007267`
- Adaptive-k mean gain over the best single receptor: `-0.008496`
- Adaptive-k worst-target gain: `-0.070444`
- Adaptive-k targets with gain >= 0.02: `1/5`
- Exact repair solution differs from one-swap in `5/75` fixed-k cells
- Go/No-Go: `NO-GO`
