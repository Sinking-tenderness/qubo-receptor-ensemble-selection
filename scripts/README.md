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
- `prepare_pose_inspection.py`: extract selected Vina poses and write a PyMOL
  inspection script.
- `analyze_pose_contacts.py`: summarize simple receptor-ligand contact geometry
  for selected docked poses.
- `build_score_matrix.py`: convert long docking results into representative
  ligand-by-receptor score tables and matrices.
- `align_receptor_structure.py`: rigidly align receptor PDB coordinates to a
  shared reference frame using sequence-matched C-alpha atoms.
- `prepare_receptor.py`: select a receptor chain and alternate locations,
  invoke Meeko through the active Python environment, and audit Vina PDBQT.
- `compare_receptor_screening.py`: compare two receptor score tables and
  calculate score correlation, active coverage, and simple ensemble baselines.
- `run_receptor_ensemble_mvp.py`: run leakage-aware train/validation/test
  receptor-subset selection and held-out QUBO baseline comparison.
- `repeat_ensemble_mvp.py`: repeat the MVP over fixed stratified split seeds.
- `cross_validate_ensemble_mvp.py`: run outer stratified cross-validation,
  paired bootstrap uncertainty, and QUBO-vs-single out-of-fold comparison.
- `solve_discriminative_coverage_qubo.py`: build a fixed-weight QUBO that
  rewards active coverage and penalizes decoy exposure.
- `cross_validate_discriminative_qubo.py`: evaluate that fixed objective under
  outer cross-validation without tuning on the test fold.
- `split_ligand_scaffold.py`: create a deterministic Bemis-Murcko
  scaffold-disjoint train/validation/test split.
- `batch_vina_docking_parallel.py`: run controlled parallel Vina jobs with
  per-ligand checkpoints and resume support.
- `extend_score_matrix.py`: merge new raw docking tables into an existing
  ligand-by-receptor matrix with ID and label checks.
- `cross_validate_scaffold_ensemble.py`: run scaffold-group outer CV for
  receptor subset selection.
- `aggregate_repeated_cv.py`: aggregate repeated out-of-fold scores and paired
  bootstrap deltas.
- `batch_prepare_ligand_pdbqt_parallel.py`: prepare Meeko PDBQT files with
  resumable controlled parallel workers.
- `download_alphafold_structure.py`: download a canonical official AlphaFold
  DB model and write a source, version, SHA-256, and pLDDT audit manifest.
- `extract_pocket_features.py`: calculate aligned pocket geometry proxies from
  a reference co-crystal ligand and a candidate conformer.
- `extract_pocket_features_batch.py`: apply the pocket feature schema to an
  auditable crystal, AF2, or MD conformer manifest.
- `build_pocket_feature_matrix.py`: combine per-conformer residue geometry into
  a wide matrix for pocket clustering and later train-only selection features.
- `cluster_pocket_conformers.py`: create a structure-only pocket clustering and
  medoid baseline without using ligand activity labels.
