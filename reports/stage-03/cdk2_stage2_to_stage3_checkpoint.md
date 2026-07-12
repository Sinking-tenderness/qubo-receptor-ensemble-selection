# CDK2: From Two-Conformer Docking to the Stage 3 Feature Path

## Scope

This checkpoint records what was learned after extending the CDK2 experiment
from one receptor to two receptors, then to eight crystal conformers. It also
defines the implemented Stage 3 entry point. It is a scientific record, not a
claim that the current QUBO objective improves virtual screening.

## 1. Two-conformer comparison

### What was done

- Receptors: CDK2 1AQ1 and a 1AQ1-aligned 1HCL structure.
- Ligands: a seeded DUD-E subset of 10 actives and 50 decoys.
- Controls held fixed: ligand preparation, Vina 1.2.7, docking box, seed rule,
  and `exhaustiveness=16`.
- Both receptors docked all 60 ligands successfully. The long table therefore
  has 120 receptor-ligand records and no missing score pairs.
- Scores were compared using correlation, per-receptor virtual-screening
  metrics, early active coverage, and two simple ensemble rules:
  minimum score and mean score.

### What was observed

| Observation | Result |
| --- | --- |
| Score correlation | Spearman 0.789; Pearson 0.810 |
| 1AQ1 ROC-AUC / PR-AUC / BEDROC | 0.644 / 0.459 / 0.653 |
| 1HCL ROC-AUC / PR-AUC / BEDROC | 0.576 / 0.285 / 0.377 |
| 1HCL new active coverage in Top1/3/6/10 | none |
| Minimum-score ensemble | effectively equal to 1AQ1 |
| Mean-score ensemble | weaker than 1AQ1 |

Because lower Vina score is ranked earlier, 1AQ1 gave the lower score for
59 of 60 ligands. The small two-conformer ensemble thus did not create useful
early active complementarity for this ligand subset.

### Scientific interpretation

The two conformers were not identical, but structural difference alone did not
make them a useful screening ensemble. This is a useful negative control:
future selection must reward complementary active coverage and decoy rejection,
not merely choose geometrically different structures or average all scores.

## 2. Expanded eight-conformer baseline

### What was done

- Expanded to eight aligned CDK2 crystal receptors: 1AQ1, 1H00, 1HCL, 1JVP,
  1Y8Y, 2C68, 2C69, and 3RKB.
- Built a DUD-E benchmark with 100 actives and 100 decoys.
- Completed all 1,600 receptor-ligand dockings with Vina 1.2.7,
  `exhaustiveness=1`, `num_modes=1`, one CPU per worker, and a common aligned
  docking box. This faster protocol was used for a screening baseline; it does
  not replace the higher-search redocking validation.
- Compared single-receptor, all-conformer score aggregation, and a fixed-size
  score-only QUBO subset under scaffold-group train/validation/test separation
  and repeated outer cross-validation.

### What was observed

| Method | ROC-AUC | PR-AUC | BEDROC(alpha=20) | EF5% |
| --- | ---: | ---: | ---: | ---: |
| Best single receptor (1AQ1) | 0.732 | 0.787 | 0.963 | 2.0 |
| All-conformer minimum score | 0.713 | 0.752 | 0.931 | 2.0 |
| All-conformer mean score | 0.702 | 0.714 | 0.856 | 2.0 |
| CV train-selected single | 0.722 | 0.769 | 0.948 | 2.0 |
| CV score-only QUBO subset | 0.722 | 0.759 | 0.925 | 2.0 |

For QUBO minus single-receptor selection, paired bootstrap intervals were:

- ROC-AUC: -0.00005, 95% CI [-0.0268, +0.0255]
- PR-AUC: -0.0101, 95% CI [-0.0391, +0.0166]
- BEDROC: -0.0227, 95% CI [-0.0913, +0.0194]

### Scientific interpretation

The candidate score-only QUBO objective is not validated. Its intervals include
zero and its point estimates are not better than the single-receptor baseline.
Accordingly, no QAOA or quantum-hardware result should be presented as a
performance claim for this objective. The reproducible pipeline is still a
valuable result: it establishes docking, score-matrix construction,
scaffold-aware evaluation, and uncertainty estimation.

## 3. Implemented Stage 3 route

The next route intentionally changes the input information rather than
retuning the failed score-only objective on the same test data.

1. **Official AF2 entry point**: download the canonical CDK2 AlphaFold DB model
   through the official API and record version, source URL, SHA-256, residue
   count, and pLDDT summary.
2. **Shared coordinate frame**: align AF2 and future MD frames to 1AQ1 using
   sequence-consistent C-alpha alignment before reusing the docking box.
3. **Pocket-level feature schema**: use the 21 residues within 5 A of 1AQ1 STU
   as a fixed reference schema. Record residue presence, side-chain centroid,
   and minimum distance to the reference ligand. These are geometry proxies,
   not interaction energies.
4. **Structure-only baseline**: cluster conformers from pocket features and
   choose medoids without ligand labels. This is a necessary classical baseline
   for later QUBO comparison.
5. **AF2 smoke docking**: dock two actives and two decoys only to verify the
   aligned box, prepared AF2 receptor, Vina outputs, and score parsing. It is
   not an AF2 screening evaluation.
6. **MD gate**: create an auditable OpenMM environment and write the MD
   protocol before generating any frames. Protonation, force field, solvent,
   ions, equilibration, production duration, frame stride, seeds, and
   replicates must be fixed in advance.
7. **Future QUBO gate**: construct any new objective on training folds only,
   then compare it against random, clustering-medoid, greedy, single-best, and
   all-conformer baselines under scaffold-group outer CV. A quantum solver is
   considered only after a classical objective shows a stable positive effect.

## 4. Stage 3 facts available now

- Official AF2 CDK2 entry: `AF-P24941-F1`, model version 6, created 2025-08-01.
- AF2 model audit: 2,398 atoms, 298 residues, mean pLDDT 88.546.
- AF2 alignment to 1AQ1: 277 sequence-matched C-alpha atoms, RMSD reduced from
  45.916 A to 0.944 A, with no reflection.
- The 21-residue pocket schema is complete in AF2; AF2 pocket C-alpha RMSD to
  1AQ1 is 0.8202 A.
- The exploratory three-cluster structural baseline selected 1H00, 1AQ1, and
  2C69 as medoids. AF2 belongs to the 1H00/1JVP/3RKB cluster but is not its
  medoid. This says nothing yet about AF2 docking performance.

## 5. Current limitations

- DUD-E decoys are presumed decoys, not experimentally confirmed inactive
  compounds; decoy and analogue bias remain possible.
- The 100A/100D benchmark is useful for a reproducible MVP but remains modest
  for EF1% and chemotype-specific claims.
- Vina score is a docking-model signal, not measured binding free energy.
- AF2 pLDDT is confidence in predicted local structure, not validation of a
  ligand-competent CDK2 pocket.
- No MD-derived conformers or ligand-contact features have yet been generated.

## Primary records

- `reports/stage-02/dude_cdk2_two_receptor_comparison_record.md`
- `reports/stage-02/cdk2_100a_100d_final_validation_2026-07-12.md`
- `reports/stage-03/af2_md_feature_protocol.md`
