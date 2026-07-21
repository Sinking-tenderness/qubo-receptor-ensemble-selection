# Stage 5 MAPK14 Train-696 Pair-Utility QUBO Gate

Date: 2026-07-21

## Motivation

The inherited active-overlap QUBO selected a stable, non-degenerate receptor
triple but underperformed matched linear and nested greedy early enrichment.
This follow-up tested one narrower quadratic hypothesis: directly reward the
training-fold BEDROC of each receptor pair.

An earlier CDK2 pair-BEDROC experiment had used mean-score pair utility for
both min-score and mean-score candidates and did not alter nested-CV selection.
The present implementation corrected that mismatch. Within each current
training fold, it independently calculated min-score and mean-score pair
BEDROC, then supplied only the map matching the candidate aggregation.

## Frozen Protocol

The candidate family contained only:

- normalized singleton BEDROC linear utility;
- aggregation-aligned pair BEDROC quadratic reward;
- the inherited seed-stability reward; and
- the cardinality penalty.

Active coverage, decoy exposure, active overlap, and redundancy were fixed to
zero. Pair utility weights were 0.5, 1.0, and 2.0; stability weights were 0.5
and 1.5; subset sizes were two and three; and both min-score and mean-score
aggregation were available. The complete grid therefore contained 24
candidates.

The same four outer and three inner grouped-scaffold folds were used. Every
pair utility map was fitted inside the current training fold. Acceptance
required noninferiority to both matched linear and nested greedy on primary,
mean-seed, and worst-seed BEDROC. Validation and test remained unavailable.

## Selected Candidate

The selected candidate used:

- family: `pair_utility_qubo`;
- target size: 3;
- aggregation: `min_score`;
- pair utility weight: 0.5;
- stability weight: 1.5; and
- subset: `2BAJ + 3KQ7 + 3MPT`.

All three full-train seed fits selected the same triple. The quadratic terms
were nonzero, but the final QUBO subset was identical to the matched linear
subset. Pair utility changed the QUBO subset relative to linear in two of four
outer folds, so the OOF predictions were not identical.

## OOF Results

| Method | ROC-AUC | PR-AUC | BEDROC20 | EF5% |
|---|---:|---:|---:|---:|
| Pair-utility QUBO | 0.7354 | 0.7556 | 0.9157 | 1.8286 |
| Matched linear top-k | 0.7610 | 0.7733 | 0.9180 | 1.8286 |
| Nested greedy | 0.7423 | 0.7593 | 0.9245 | 1.9429 |

Pair-utility QUBO BEDROC deltas were:

- primary versus matched linear: -0.00222;
- mean seed versus matched linear: -0.00577;
- worst seed versus matched linear: -0.01688;
- primary versus nested greedy: -0.00873;
- mean seed versus nested greedy: -0.01546; and
- worst seed versus nested greedy: -0.02883.

The selected QUBO was at the 89.3rd percentile among all 56 fixed triples on
primary and mean-seed BEDROC, and at the 83.9th percentile on worst-seed
BEDROC. This was substantially better than the previous active-overlap QUBO,
whose primary BEDROC was 0.8743, but it still did not meet either classical
comparator.

Status: `train_pair_utility_qubo_gate_failed_validation_unavailable`.

## Interpretation

Aggregation alignment made the pair objective materially more useful, raising
primary QUBO BEDROC by approximately 0.0415 relative to the previous objective.
However, raw pair BEDROC remained strongly aligned with singleton utility. The
final pair reward therefore reinforced the same receptors chosen by the linear
terms rather than adding independent combinatorial value.

The raw pair-utility objective is stopped. Its weights will not be tuned again
on Train-696, and validation remains unavailable.

A mathematically distinct follow-up would use marginal pair synergy, such as
pair ensemble BEDROC gain beyond the stronger singleton, instead of raw pair
BEDROC. That residual would explicitly remove singleton double counting. It is
not authorized by this result and would require a separate frozen protocol,
with matched linear and nested greedy retained as hard comparators.
