# PPARA Pool30 Remote Reuse Implementation Plan

> **For the implementation agent:** Required subskill: use `executing-plans` to implement this plan task by task. Track each step with the checkbox syntax below.

**Goal:** Extend the PPARA Stage102A receptor pool from 15 to 30 on the Linux host while reusing only verified score tables for the original 15 receptors.

**Architecture:** A new remote-only experiment configuration writes every artifact under `results/runs/ppara_pool30_adaptive_remote`. A small audit-and-seed script compares frozen protocol fields, ligand PDBQT hashes, docking boxes, and the 15 overlapping receptor PDBQT hashes before hard-linking the 45 complete old score tables into the new run. The normal runner is then invoked with `--resume`, which docks only the missing 15 receptors and carries the complete 30-column matrix through solve and evaluation.

**Technology:** JSON configuration, Python standard library, pytest, Linux hard links, Uni-Dock on the remote host.

---

## File Responsibilities

- Create: `configs/experiments/ppara_pool30_adaptive_remote.json` - Stage102A PPARA configuration with remote absolute source and result paths.
- Create: `scripts/seed_ppara_pool30_score_reuse.py` - validates and hard-links old score tables after the destination prepare stage.
- Create: `scripts/run_ppara_pool30_remote.sh` - remote-only orchestration for prepare, audit/seed, resume docking, and persistence.
- Create: `tests/test_seed_ppara_pool30_score_reuse.py` - isolated tests for the audit and hard-link behavior.
- Create: `docs/superpowers/plans/2026-08-28-ppara-pool30-remote-reuse.md` - this implementation record.

### Task 1: Add the Failing Reuse-Audit Tests

**Files:**
- Create: `tests/test_seed_ppara_pool30_score_reuse.py`

- [ ] **Step 1: Write a test fixture with identical source/destination manifests, boxes, and 15 overlapping receptor hashes.**

```python
def test_seed_reuse_hardlinks_only_complete_verified_tables(tmp_path: Path) -> None:
    source, destination = make_matching_runs(tmp_path, overlap_ids=("R01", "R02"))
    audit = seed_verified_score_tables(source, destination)
    assert audit["linked_table_count"] == 6
    assert os.stat(source / "score_tables" / "seed_11__R01.csv").st_ino == os.stat(
        destination / "score_tables" / "seed_11__R01.csv"
    ).st_ino
```

- [ ] **Step 2: Run the focused test and confirm it fails because the audit module does not exist.**

Run: `python -m pytest -q tests/test_seed_ppara_pool30_score_reuse.py`

Expected: FAIL with an import error for `scripts.seed_ppara_pool30_score_reuse`.

- [ ] **Step 3: Add failure tests for a changed ligand PDBQT hash and an incomplete old score table.**

```python
def test_seed_reuse_rejects_ligand_hash_mismatch(tmp_path: Path) -> None:
    source, destination = make_matching_runs(tmp_path, overlap_ids=("R01",))
    write_ligand_hash(destination, "L001", "different")
    with pytest.raises(ValueError, match="ligand"):
        seed_verified_score_tables(source, destination)
```

- [ ] **Step 4: Run the focused test file and confirm both failure modes are exercised.**

Run: `python -m pytest -q tests/test_seed_ppara_pool30_score_reuse.py`

Expected: FAIL because the implementation is still absent, not because the fixtures are malformed.

### Task 2: Implement the Remote Audit-and-Seed Tool

**Files:**
- Create: `scripts/seed_ppara_pool30_score_reuse.py`
- Modify: `tests/test_seed_ppara_pool30_score_reuse.py`

- [ ] **Step 1: Implement SHA-256 comparison for prepared ligands, shared receptor PDBQTs, and canonical docking boxes.**

```python
def verify_shared_inputs(source_run: Path, destination_run: Path) -> list[str]:
    source_ligands = read_manifest(source_run / "prepared_ligands.csv")
    destination_ligands = read_manifest(destination_run / "prepared_ligands.csv")
    verify_same_ligand_hashes(source_ligands, destination_ligands)
    shared_ids = verify_receptor_prefix(source_run, destination_run)
    verify_same_box(source_run / "docking_box.json", destination_run / "docking_box.json")
    return shared_ids
```

- [ ] **Step 2: Implement table completeness checks and `os.link` creation without overwrite.**

```python
for seed in seeds:
    for receptor_id in shared_ids:
        source_table = source_scores / f"seed_{seed}__{receptor_id}.csv"
        require_complete_table(source_table, ligand_ids, receptor_id, seed)
        os.link(source_table, destination_scores / source_table.name)
```

- [ ] **Step 3: Persist `score_table_reuse_audit.json` with source/destination paths, hashes, shared IDs, and each hard link.**

- [ ] **Step 4: Run focused tests and the existing configuration tests.**

Run: `python -m pytest -q tests/test_seed_ppara_pool30_score_reuse.py tests/test_full_experiment_config.py`

Expected: PASS with all source and destination integrity checks exercised.

### Task 3: Add the Remote Pool30 Configuration

**Files:**
- Create: `configs/experiments/ppara_pool30_adaptive_remote.json`
- Modify: `tests/test_full_experiment_config.py`

- [ ] **Step 1: Write a configuration test that loads the new file with a remote data root and asserts the run-specific remote paths.**

```python
def test_ppara_pool30_remote_config_has_isolated_remote_paths() -> None:
    config = load_full_experiment_config(POOL30_CONFIG, data_root=Path("/root/autodl-tmp/qubo_data_root"))
    assert config.data["selection"]["receptor_count"] == 30
    assert str(config.paths["run_directory"]) == "/root/autodl-tmp/qubo_data_root/results/runs/ppara_pool30_adaptive_remote"
```

- [ ] **Step 2: Run the new test and confirm it fails because the configuration does not exist.**

Run: `python -m pytest -q tests/test_full_experiment_config.py -k pool30`

Expected: FAIL for missing `ppara_pool30_adaptive_remote.json`.

- [ ] **Step 3: Copy the frozen PPARA remote protocol and change only `experiment_id`, `selection.receptor_count`, and every output path beneath `paths`.**

```json
"experiment_id": "ppara-pool30-adaptive-remote",
"selection": {"receptor_count": 30},
"paths": {
  "run_directory": "/root/autodl-tmp/qubo_data_root/results/runs/ppara_pool30_adaptive_remote"
}
```

- [ ] **Step 4: Run the focused configuration test and check the stage plan on the remote host.**

Run: `python -m pytest -q tests/test_full_experiment_config.py -k pool30`

Expected: PASS. Remote preflight: `python scripts/run_experiment.py plan --config configs/experiments/ppara_pool30_adaptive_remote.json --data-root /root/autodl-tmp/qubo_data_root`.

### Task 4: Add the Remote Orchestration Script

**Files:**
- Create: `scripts/run_ppara_pool30_remote.sh`

- [ ] **Step 1: Validate Linux-only prerequisites and declare all remote paths as overridable variables.**

```bash
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/qubo_data_root}"
SOURCE_RUN="${SOURCE_RUN:-$DATA_ROOT/results/runs/ppara_adaptive_remote}"
DESTINATION_RUN="${DESTINATION_RUN:-$DATA_ROOT/results/runs/ppara_pool30_adaptive_remote}"
```

- [ ] **Step 2: Run destination `prepare`, invoke the audit-and-seed script, then run from `dock` to `persist` with `--resume`.**

```bash
python scripts/run_experiment.py run --config "$CONFIG" --data-root "$DATA_ROOT" --to prepare
python scripts/seed_ppara_pool30_score_reuse.py --source-run "$SOURCE_RUN" --destination-run "$DESTINATION_RUN"
python scripts/run_experiment.py run --config "$CONFIG" --data-root "$DATA_ROOT" --from dock --to persist --resume
```

- [ ] **Step 3: Add explicit postconditions for 30 receptors, 90 complete score tables, and a completed result manifest.**

- [ ] **Step 4: Validate shell syntax locally and execute only the remote `plan` command before launching docking.**

Run: `bash -n scripts/run_ppara_pool30_remote.sh`

Expected: exit 0. Remote preflight must report `receptor_count: 30` and the `ppara_pool30_adaptive_remote` result root.

### Task 5: Run and Report the Comparison

**Files:**
- Create remotely: `/root/autodl-tmp/qubo_data_root/results/runs/ppara_pool30_adaptive_remote/`
- Create locally after retrieval: `reports/stage-12/ppara_pool30_method_comparison.md`

- [ ] **Step 1: Launch the remote script only after preflight passes.**

- [ ] **Step 2: Verify the reuse audit shows exactly 15 shared receptors and 45 hard-linked score tables; verify exactly 45 tables were newly docked.**

- [ ] **Step 3: Run the matched outer-fold QUBO/linear/greedy/single comparison with the same adaptive k and report M=15 versus M=30.**

- [ ] **Step 4: State scaling observations as classical exact-solver runtime evidence only; do not claim quantum advantage without a corresponding quantum backend comparison.**
