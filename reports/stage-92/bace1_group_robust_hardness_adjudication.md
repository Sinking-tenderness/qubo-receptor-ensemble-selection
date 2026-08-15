# Stage92 BACE1 group-robust hardness adjudication

Status: `stage92_bace1_group_robust_hardness_gate_failed`.

The 34-receptor, k=6 space contains 1,344,904 subsets. Exact enumeration improved the direct greedy objective by 0.006869040, showing a real combination effect.

However, greedy plus deterministic one-swap search reached the exact solution, leaving an exact-minus-strong-greedy gap of 0.000000000. No reproducible multi-move local trap was present.

## Decision

Confirmation A and quantum hardware remain locked. This is a negative hardness result, not evidence that the docking data or receptor ensemble is invalid.

## Integrity boundary

Stage91 froze the weights but not every implementation detail. Stage92 therefore records the conventional ceil/rank/Jaccard operationalization as post-score adjudication and does not reinterpret it as fully prospective.
