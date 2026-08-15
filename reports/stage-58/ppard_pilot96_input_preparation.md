# Stage58a PPARD Pilot-96 input preparation

Stage57 passed its frozen cognate-redocking gate with 29 of 51 receptors, above the preregistered minimum of 24. All 117 attempted receptor-seed redocking batches completed, with zero pose-integrity failures and zero unresolved warning events.

Stage58a prepares only the 96 development ligands selected outcome-blind in Stage56: 48 actives and 48 decoys, balanced as 12 plus 12 in each of four outer folds. It does not read PPARD docking scores or expose fresh-validation or locked-test rows.

Each ligand is embedded deterministically with RDKit ETKDGv3 and prepared with Meeko 0.7.1. Flexible preparation is attempted first; macrocycles fall back to rigid preparation only when required, and the final PDBQT must contain no closure pseudoatoms. Per-ligand checkpoints support exact resume.

Passing the independent Stage58a input audit authorizes the frozen 29-receptor by 96-ligand by three-seed Uni-Dock pilot matrix. It does not establish enrichment, functional complementarity, QUBO superiority, or quantum advantage.
