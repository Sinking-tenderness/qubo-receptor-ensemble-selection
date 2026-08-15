# Stage61b PPARD Remaining-144 Uni-Dock production

Stage61b completes the frozen PPARD development matrix without repeating Pilot-96.

- Ligands: 144 (72 active, 72 decoy).
- Receptors: all 29 Stage57-passing conformations.
- Seeds: 20260801, 20260802, and 20260803.
- Uni-Dock: 1.1.3 enhanced profile, exhaustiveness 1024, max step 80.
- New receptor-ligand-seed jobs: 12,528.
- Full Train-240 jobs after merging Stage58b and Stage61b: 20,880.

The run is checkpointed by seed and receptor. It reads no fresh-validation or locked-test row. A successful technical audit authorizes only the frozen Stage60 nested development analysis.
