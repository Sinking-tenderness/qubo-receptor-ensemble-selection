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
- `run_repeated_development_scaffold_cv.py`: derive, resume, independently
  audit, and aggregate repeated development-only scaffold-CV gates.
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
- `build_openmm_system.py`: build and audit a solvated OpenMM system from a
  versioned protocol without running minimization or dynamics.
- `benchmark_openmm_platform.py`: measure fixed-step CPU, OpenCL, or CUDA OpenMM
  throughput without writing an MD trajectory.
- `run_openmm_equilibration_smoke.py`: run bounded minimization plus short
  NVT/NPT numerical checks before starting the documented MD pilot.
- `run_openmm_equilibration.py`: run resumable minimization, NVT, and NPT with
  stage checkpoints, progress records, and an auditable final state.
- `run_openmm_production.py`: run a resumable NPT production pilot as durable
  DCD chunks with metrics, checkpoints, and a final state manifest.
- `analyze_md_trajectory.py`: align chunked MD coordinates and calculate
  backbone, pocket, and per-residue trajectory quality metrics.
- `cluster_md_pocket_frames.py`: cluster aligned MD frames from invariant
  pocket-distance features and export representative medoid PDB structures.
- `align_md_medoid_receptors.py`: align MD medoids to a crystal reference and
  export audited protein-only heavy-atom structures for receptor preparation.
- `batch_prepare_md_receptors.py`: apply one audited Meeko protocol to all MD
  medoids while preserving per-receptor failures and parameterization audits.
- `run_md_receptor_docking_gate.py`: run a fixed four-ligand Vina execution
  gate across prepared MD receptors and build a parent-referenced score matrix.
- `run_vina_search_robustness.py`: compare paired exhaustiveness-4 and
  exhaustiveness-8 scores across fixed receptor-ligand cases and random seeds.
- `run_vina_warning_diagnostics.py`: rerun a fixed source warning table across
  paired search protocols and seeds without modifying the source score matrix.
- `aggregate_vina_seed_replicates.py`: combine complete seeded score matrices
  into minimum and median matrices with per-pair seed-stability warnings.
- `run_md_receptor_ligand_benchmark.py`: run a resumable audited ligand
  benchmark across a prepared receptor manifest and build a score matrix.
- `build_expanded_structural_receptor_pool.py`: integrity-gate aligned crystal,
  AF2, and MD receptors and build label-independent invariant pocket-clustering
  baselines.
- `build_development_ligand_manifest.py`: materialize a hash-pinned train plus
  validation PDBQT manifest while excluding the locked test split.
- `subset_score_matrices_to_development.py`: subset existing aggregate matrices
  by development ligand ID without parsing locked-test receptor score cells.
- `run_development_scaffold_cv_gate.py`: tune receptor-selection baselines and
  QUBO families inside nested development-only scaffold CV, refit the nominated
  family on all development ligands, and keep the test split locked.
- `audit_development_scaffold_cv_gate.py`: independently reproduce OOF metrics,
  bootstrap intervals, fold isolation, and constrained-subset invariants from a
  completed development scaffold-CV gate.
- `run_repeated_development_scaffold_cv.py`: repeat an audited development gate
  over fixed scaffold-fold seeds and aggregate feasibility, directional
  consistency, OOF rankings, and receptor-selection frequencies.
- `select_reliable_receptor_protocol.py`: apply preregistered repeated-CV
  reliability criteria to hash-pinned candidates and fail closed to a
  development-only single-best protocol when no QUBO candidate passes every
  check.
- `materialize_single_best_protocol.py`: refit the fail-closed method on the
  complete development set, fix one receptor ID and PDBQT hash, summarize
  repeated-fold selection stability, and preserve a zero-score-cell test lock.
- `release_locked_test_protocol.py`: consume explicit project-owner approval
  once, evaluate only the fixed receptor column on the locked split, calculate
  preregistered metrics and stratified bootstrap intervals, and refuse reruns.
