# Scripts

Command-line workflow entry points will be added incrementally. Each script
should validate inputs, log failures, and produce a documented manifest.

Stage 2 ligand preparation scripts:

- `make_dude_subset.py`: create a small, seeded DUD-E active/decoy subset.
- `check_ligand_smiles.py`: audit SMILES with RDKit and write QC summaries.
- `prepare_ligand_3d_sdf.py`: generate explicit-H 3D SDF ligand files.
- `batch_prepare_ligand_pdbqt.py`: convert prepared SDF ligands to Vina PDBQT.
- `batch_vina_docking.py`: run Vina for a ligand PDBQT manifest and write a
  long docking score table.
- `evaluate_virtual_screening.py`: convert docking scores into per-ligand
  rankings and basic enrichment metrics.
- `analyze_top_hits.py`: merge rankings with ligand properties and flag
  top-ranked molecules for structural inspection.
