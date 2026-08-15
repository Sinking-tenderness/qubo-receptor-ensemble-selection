# Stage 51b PPARA Redocking-Bias Diagnostic

Stage51b is a post-hoc, label-free technical diagnostic. It used no activity
labels, development-panel docking scores, fresh-validation rows, locked-test
rows, new docking jobs, or quantum hardware jobs. The failed Stage51
confirmatory gate remains failed.

## Main Result

Leave-one-out nearest-neighbor prediction showed that both receptor pocket
structure and cognate-ligand chemistry strongly track redocking success.
Structural distance achieved balanced accuracy 0.900 and Morgan-fingerprint
distance achieved 0.875; both permutation p-values were below 0.0001. The
chemistry-minus-structure difference was -0.025 with p = 0.603, so neither
source can be identified as the single dominant driver.

The passing cognate ligands were larger and less flexible. Heavy-atom count,
rotatable-bond count, molecular weight, TPSA, and fraction sp3 all remained
associated with outcome after Benjamini-Hochberg adjustment. The 60 ligands
formed 27 fingerprint clusters at Tanimoto similarity 0.6, and the 20 passing
receptors covered 13 of those clusters.

## Structural Coverage

The 20-receptor passing pool retained the extreme structural states: its
coverage-radius percentile among 10,000 random 20-subsets was 0.891, inside the
frozen 0.95 exploratory limit. However, its mean nearest-distance percentile
was 0.995, showing much poorer average representativeness than a random
20-subset. The stable 18-receptor pool had the same coverage radius and a mean
nearest-distance percentile of 0.997.

## Decision

All three frozen exploratory conditions passed: at least 16 stable receptors,
coverage-radius percentile at most 0.95, and at least six passing chemotype
clusters. A separate exploratory development-panel branch is therefore worth
testing. It must preserve all 20 Stage51-passing receptors, remain explicitly
post hoc, and cannot repair the failed PPARA confirmation claim.

The next branch should dock only the frozen PPARA development panel, keep the
validation and test sets sealed, and evaluate whether QUBO gains arise across
chemotypes rather than from the redocking-selected chemical families.
