# Uni-Dock Macrocycle Preparation Fallback Implementation Plan

> **For the implementation worker:** use the confirmed task package and keep the existing Stage102A selection contract unchanged.

**Goal:** Make the full raw-data workflow produce Uni-Dock-compatible ligand PDBQT files when Meeko emits macrocycle closure pseudoatoms.

**Design:** Run Meeko in its normal flexible mode first. If preparation fails, produces an invalid PDBQT, or emits atom types matching `^(?:CG|G)\\d+$`, rerun the same SDF with `--rigid_macrocycles`. Reject the ligand if the final PDBQT is still invalid or still contains closure pseudoatoms. Record the preparation variant in `prepared_ligands.csv`.

**Constraints:** Preserve the selected ligand IDs, order, labels, active/decoy quotas, receptor preparation, docking box, and Uni-Dock score parsing. Do not drop affected ligands or infer scores from logs.

### Task 1: Add failing regression coverage

**Files:**
- Modify: `tests/test_full_workflow_preparation.py`

- Add a test for detecting `CG0` and `G0` atom types.
- Add a test proving flexible preparation is retried with `rigid_macrocycles=True` when closure pseudoatoms are emitted.
- Add a test proving a nonzero flexible Meeko exit also falls back to rigid preparation.

### Task 2: Implement the preparation contract

**Files:**
- Modify: `src/qubo_receptor_ensemble/preparation.py`
- Modify: `src/qubo_receptor_ensemble/experiment.py`

- Add the shared closure-pseudoatom detector using the historical `CG*/G*` rule.
- Run flexible Meeko first, remove a failed/intermediate PDBQT before retrying, and validate the rigid result.
- Record `preparation_variant` and an auditable `pdbqt_message`.
- Reject final zero-atom, missing-TORSDOF, or closure-pseudoatom PDBQT files.

### Task 3: Verify and integrate

- Run the focused preparation tests and confirm the pre-change regression fails before implementation.
- Run all full-workflow, preparation, docking-adapter, and raw-preparation tests.
- Run `compileall` and `git diff --check`.
- Inspect the diff and commit only the plan, implementation, and tests listed above.
- Push the resulting commit to `origin/dev_ylj`.
