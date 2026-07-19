# MAPK14 Fixed-QUBO Repeated-CV Record

## Scope

One post hoc development candidate was frozen before this run. No QUBO weight,
subset size, aggregation method, or candidate was selected during repeated CV.
The fixed candidate was evaluated across 20 preregistered four-fold grouped-
scaffold partitions of the same 160 development-train ligands.

Development-validation and test rows remained unavailable.

## Fixed Candidate

- Family: discriminative QUBO
- Receptors selected per fold: 3
- Aggregation: `min_score`
- Active coverage weight: 0.0
- Active overlap weight: 1.0
- Decoy exposure weight: 0.5
- Redundancy weight: 1.0
- Seed stability weight: 1.5

The full-train QUBO selected `2QD9 + 3KQ7 + 3MPT`. Its matched linear top-k
selected `2BAJ + 3KQ7 + 3MPT`.

## Result

The repeated-CV gate failed:

- Median primary BEDROC delta versus linear: -0.0236.
- Median mean-seed BEDROC delta versus linear: -0.0206.
- Lower-quartile worst-seed delta versus linear: -0.0393.
- Fraction of repeats non-worse on primary: 0.25.
- Fraction of repeats non-worse on every seed: 0.25.
- Median primary delta versus single best: -0.0088.
- Lower-quartile worst-seed delta versus single best: -0.0617.

The candidate passed structural checks. Its full-train subset differed from
linear top-k, its quadratic terms were nonconstant, and seed-specific subset
stability remained adequate. It also ranked above the median fixed subset.
Those properties did not translate into stable early-enrichment improvement.

## Interpretation

The positive result from the original four-fold post hoc diagnostic depended on
one favorable partition. Across 20 partitions, the same fixed candidate usually
lost to matched linear top-k. The evidence therefore does not support accessing
development-validation with the current 160-ligand training matrix.

Repeated folds overlap and are not independent experiments, so their fractions
are stability summaries rather than confidence intervals. Even under that
limited interpretation, a 25% non-worse rate is clearly below the frozen 75%
threshold.

## Decision

Further tuning on these 160 ligands is stopped. The next development matrix will
use all 348 scaffold-disjoint train actives and 348 group-diverse train decoys.
It will retain the current 80-active/80-decoy train panel as a strict subset so
existing ligand preparation and uniform e32 docking evidence can be audited and
reused.

The expanded panel will not include any validation or test ligand. Its docking
run requires a remote CPU instance; a GPU is not useful for official Vina 1.2.7.
