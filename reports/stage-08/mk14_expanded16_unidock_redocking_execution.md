# Stage 08 MAPK14 Expanded-16 Uni-Dock Redocking

## Purpose

This is a technical admission gate, not the Train-696 production matrix. It
prepares only the eight new structurally selected receptors and redocks each
cognate ligand with three paired seeds under frozen Uni-Dock 1.1.3 enhanced
settings (`exhaustiveness=1024`, `max_step=80`). Validation and test data are
not present in the bundle.

## Frozen Gate

- Receptors: 8 new additions to the audited 16-receptor structural pool.
- Seeds: `20260801`, `20260802`, and `20260803`.
- GPU redocking jobs: `8 x 3 = 24`.
- One output mode per job.
- Per-receptor pass: at least two of three top poses have RMSD at most 2 A,
  and the three-seed median RMSD is at most 2 A.
- Global pass: all eight receptors pass, with zero unresolved engine warnings
  and zero pose-integrity failures.
- Production docking remains blocked until the independent RMSD audit passes.

## Remote Setup

Extract the input archive under a new data-disk directory. Verify
`bundle_manifest.sha256`, then create the frozen environment:

```bash
cd /root/autodl-tmp
mkdir -p stage08_mk14_expanded16_redocking_v1
tar -xzf stage08_mk14_expanded16_unidock_redocking_input_v1.tar.gz \
  -C stage08_mk14_expanded16_redocking_v1
cd stage08_mk14_expanded16_redocking_v1
sha256sum -c bundle_manifest.sha256

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment/stage08_unidock_gpu.yml
conda activate qubo-unidock-stage08
```

If the environment already exists and matches the YAML, activate it directly.

## Execute

Run once in the foreground for the input-only audit:

```bash
python scripts/experimental/unidock/run_stage08_mk14_expanded16_redocking.py \
  --config configs/stage08_mk14_expanded16_unidock_redocking.json \
  --audit-only
```

Then launch the resumable run. `AUTO_POWEROFF=1` asks the instance to shut down
after either success or failure, after files are synchronized and failure
diagnostics are packaged when possible.

```bash
nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage08_mk14_expanded16_redocking_remote.sh \
  > stage08_mk14_expanded16_redocking.log 2>&1 &
echo $! | tee stage08.pid
```

Monitor without interrupting it:

```bash
tail -f stage08_mk14_expanded16_redocking.log
```

The successful run creates one core archive and one diagnostic archive under
`/root/autodl-tmp`. Return both archives and the two SHA-256 lines. Do not start
the `696 x 16 x 3` production matrix until these results have been reviewed.
