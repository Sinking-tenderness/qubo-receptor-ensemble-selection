# Stage57 PPARD cognate redocking

## Scope

Stage57 prepares all 51 receptors that passed the unchanged Stage56b PPARD coordinate hard gate and then runs the preregistered three-seed Uni-Dock 1.1.3 cognate-redocking gate. No PPARD docking outcome, ligand label, fresh-validation row, locked-test row, or quantum result is included in the input bundle.

The frozen technical gate requires at least 24 of the original 51 receptors to pass. A receptor passes when at least two of three seeds recover the cognate pose at symmetry-corrected heavy-atom RMSD $\leq 2.0$ angstrom and the median RMSD is also $\leq 2.0$ angstrom. Preparation failures remain failures in the denominator and cannot be replaced.

## Existing environment

The remote script reuses `qubo-unidock-stage08` only when it imports `gemmi`, `meeko`, `numpy==1.26.4`, `openmm`, `pdbfixer`, `prody`, `rdkit`, and `scipy`, and reports Uni-Dock successfully. It creates that same environment from the bundled YAML only when the environment does not exist. It will not mutate an incomplete environment while another task may be using it.

## Run

```bash
cd /root/autodl-tmp
sha256sum stage57_ppard_cognate_redocking_input_v3.tar.gz

mkdir -p stage57_ppard_cognate_redocking_v3
tar -xzf stage57_ppard_cognate_redocking_input_v3.tar.gz \
  -C stage57_ppard_cognate_redocking_v3

cd stage57_ppard_cognate_redocking_v3
sha256sum -c bundle_manifest.sha256

AUTO_POWEROFF=1 nohup bash \
  scripts/experimental/unidock/run_stage57_ppard_cognate_redocking_remote.sh \
  > stage57_ppard_cognate_redocking.log 2>&1 &

echo $! | tee stage57.pid
tail -f stage57_ppard_cognate_redocking.log
```

The script is resumable. Re-run the same `AUTO_POWEROFF=1 nohup ...` command after an interruption. Completed preparation cases and valid redocking batches are hash-checked and reused.

## Outputs

- `/root/autodl-tmp/stage57_ppard_cognate_redocking_core_v1.tar.gz`
- `/root/autodl-tmp/stage57_ppard_cognate_redocking_diagnostics_v1.tar.gz`
- `/root/autodl-tmp/stage57_ppard_cognate_redocking_failed_runtime_v1.tar.gz` when execution exits unsuccessfully

Upload the core and diagnostics archives for local independent analysis. A passing gate authorizes only the frozen 96-ligand PPARD development pilot. It does not authorize protected validation or quantum-hardware claims.
