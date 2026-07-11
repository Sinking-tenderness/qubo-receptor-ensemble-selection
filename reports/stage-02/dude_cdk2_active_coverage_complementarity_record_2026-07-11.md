# Active Coverage Complementarity

## Definition

For each receptor, ligands are ranked by ascending Vina score. At a chosen
early fraction, the active coverage set is the active molecules appearing in
that receptor's early-ranked list. The union of two or more coverage sets
measures whether the additional receptor contributes active molecules that
the current subset did not already capture.

This is different from score correlation: two receptors may have different
scores but still capture the same active molecules.

## Top10% Results

### Train

The train split has 36 ligands, so Top10% contains four ligands:

- 1AQ1 active coverage: `A0001`, `A0003`, `A0010`
- 1HCL active coverage: `A0010`
- 1JVP active coverage: `A0010`
- Three-receptor active union: 3 of 6 actives, or 50%

Neither 1HCL nor 1JVP adds a new active molecule beyond 1AQ1 at this cutoff.
The HCL-JVP active coverage sets are identical.

### Validation and Test

Each split has only 12 ligands, so Top10% contains two ligands:

- Validation: all three receptors capture zero active molecules.
- Test: all three receptors capture the same active molecule, `A0009`.

These results are too small to support a stable complementarity conclusion.

## Relevance To QUBO

The coverage calculation provides a more task-aligned signal than score
correlation, but it is still noisy here. A future objective could reward the
union of train active coverage while penalizing overlapping active sets. The
coverage reward should be calculated on train, its threshold selected using
validation, and never tuned on test.

The script is `scripts/analyze_active_coverage.py`; its local output is
`results/metrics/dude_cdk2_active_coverage_top10.json`.
