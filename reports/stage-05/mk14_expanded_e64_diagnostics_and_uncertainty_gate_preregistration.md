# MAPK14 e64 Diagnostics and Uncertainty-Aware Train Gate

## Frozen Evidence

The uniform eight-receptor e32 matrix contains 1,280 receptor-ligand pairs for
160 development-train ligands. The original all-pairs consensus rule rejected
14 pairs and stopped all label-aware analysis.

A targeted e64 diagnostic then ran three independent seeds for all 14 pairs:

- 42 of 42 Vina jobs completed successfully.
- 10 of 14 pairs passed the frozen two-of-three consensus rule at e64.
- Four pairs remained outside the rule.
- For those four pairs, the absolute e32-to-e64 median-score change was at most
  0.018 kcal/mol.
- Their maximum fixed-order pose RMSDs ranged from 7.78 to 13.74 angstrom,
  indicating distinct stochastic pose basins rather than a drifting median.

The evidence does not support a complete e64 recomputation. e64 values remain
diagnostic and no e32 matrix cell is replaced.

## Protocol Amendment

The unchanged three-seed e32 matrices are retained. Seed disagreement is now
measured as docking uncertainty rather than treated as a requirement that every
replicate converge to the minimum score.

The next analysis is restricted to the existing 80-active/80-decoy
development-train set. Development-validation scores are unavailable and test
ligands remain locked.

## Frozen Train-Only Gate

The gate uses four-fold nested grouped-scaffold cross-validation. Per-receptor
normalization is fitted independently within every training fold for the median,
minimum, and three individual-seed matrices.

Candidate QUBOs select two or three receptors. Each candidate contains a seed
stability reward and at least one non-cardinality quadratic term. Hyperparameters
are selected first by worst-seed inner-fold BEDROC, then by median-matrix BEDROC
and the frozen tie breakers.

The candidate is compared with:

- the single best receptor;
- a matched linear top-k using the same QUBO linear coefficients;
- nested exhaustive and greedy subset selection;
- all eight receptors; and
- the exact distribution of fixed subsets at the final size and aggregation.

Every acceptance check must pass. In particular, the final subset must contain
at least two receptors, differ from the matched linear top-k subset, contain
non-constant quadratic objective terms, remain stable across seed-specific
fits, and achieve no worse median, mean-seed, or worst-seed OOF BEDROC than the
matched linear comparator.

Passing this gate would only nominate a train-only candidate for a separate
validation-docking decision. It would not release or evaluate validation/test
scores and would not establish binding affinity, biological activity, general
cross-target superiority, or quantum advantage.
