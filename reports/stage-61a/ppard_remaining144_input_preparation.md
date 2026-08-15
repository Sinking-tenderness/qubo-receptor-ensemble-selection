# Stage61a PPARD Remaining-144 input preparation

Stage61a prepares only the 144 Train-240 ligands excluded from Pilot-96.

- Source balance: 72 active and 72 decoy ligands.
- Preparation: deterministic RDKit 3D generation plus macrocycle-safe Meeko PDBQT.
- Resume: one validated checkpoint per ligand.
- Compute: CPU only; no Uni-Dock job is launched by this bundle.
- Boundary: fresh validation, locked test, and quantum hardware remain closed.

After the independent input audit passes, Stage61b may dock 144 ligands against 29 frozen receptors with three frozen seeds (12,528 receptor-ligand-seed jobs).
