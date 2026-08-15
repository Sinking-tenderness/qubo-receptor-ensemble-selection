# Stage64 cross-target uncertainty-shrunk rank-pair QUBO

## Scope

Stage64 uses only consumed development results from BACE1, PPARG, PPARA, and
PPARD. It performs no docking, reads no fresh-validation or locked-test row, and
runs no quantum hardware job.

## Baseline reconstruction

The original Stage42f rank-pair QUBO was reconstructed exactly in
96 fixed-k outer-fold cells before any
candidate comparison.

## Candidate selection

The maximin candidate is `pair_scale_0p25` with pair scale
0.25, MAD shrinkage 0.0, and sign-support
threshold 0.0.

- Mean target gain over baseline: +0.053638
- Worst target gain over baseline: +0.028266
- Mean target gain over pair-off: -0.008570
- Worst target gain over pair-off: -0.036587
- Nonnegative targets: 4/4
- Positive targets: 4/4

## Freeze decision

Uncertainty-shrunk objective frozen: **NO-GO**.

Checks:

- Non-baseline candidate selected: True
- Minimum mean target gain: True
- Minimum worst-target gain: True
- Minimum nonnegative target count: True
- Minimum mean gain over pair-off: False
- Minimum worst-target gain over pair-off: False
- Minimum nonnegative targets over pair-off: False
- Nonnegative leave-one-target-out mean gain: True
- Minimum positive leave-one-target-out targets: True
- Nonnegative leave-one-target-out mean gain over pair-off: False
- Minimum positive leave-one-target-out targets over pair-off: False

## Boundary

This is post-hoc cross-target objective development. A pass freezes only the pair
coefficient rule for a future nested-k evaluation and genuinely new target. It
does not establish independent efficacy, solver advantage, quantum execution,
speedup, or quantum advantage.
