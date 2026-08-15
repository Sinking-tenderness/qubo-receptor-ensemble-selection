# Stage58b PPARD Pilot-96 Uni-Dock production

Stage58b generates the outcome-blind development-pilot score matrix frozen in
Stage55. It docks 96 pilot ligands against every one of the 29 Stage57
cognate-redocking passes with three frozen seeds, for 87 batches and 8,352
receptor-ligand-seed rows.

The protocol is Uni-Dock 1.1.3 with the enhanced profile: exhaustiveness 1024,
max step 80, refine step 5, one pose, energy range 3, and seeds 20260801-20260803.
The common PPARD box is centered at (15.38, 2.56, 38.05) Angstrom with size
(22, 24, 24) Angstrom.

This stage only creates and independently audits the docking matrix. It does
not evaluate enrichment, change the frozen functional gate, access fresh
validation or locked test rows, fit a QUBO, or establish quantum advantage.
The next stage applies the preregistered functional-complementarity gate.
