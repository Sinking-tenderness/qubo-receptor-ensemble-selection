# Stage 09 MAPK14 Train-696 Uni-Dock Production

## Scope

The final pool contains 16 independently admitted MAPK14 receptors. The
Train-696 manifest contains 348 actives and 348 decoys. Fifteen flexible
macrocycles were deterministically re-prepared with Meeko 0.7.1
`--rigid_macrocycles`; the other 681 PDBQT hashes are unchanged. The frozen
Uni-Dock 1.1.3 enhanced profile uses exhaustiveness 1024 and max-step 80.

The complete run contains 48 GPU batches and 33,408 logical receptor-ligand-
seed cells. It reads no validation or test rows and must not be mixed with the
historical CPU Vina matrices.

## Single GPU

Upload the input archive to `/root/autodl-tmp`, then verify and extract it:

```bash
cd /root/autodl-tmp
sha256sum stage09_mk14_train696_unidock113_production_input_v1.tar.gz

mkdir -p stage09_mk14_train696_unidock113_v1
tar -xzf stage09_mk14_train696_unidock113_production_input_v1.tar.gz \
  -C stage09_mk14_train696_unidock113_v1
cd stage09_mk14_train696_unidock113_v1
sha256sum -c bundle_manifest.sha256

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qubo-unidock-stage08

python scripts/experimental/unidock/run_stage09_mk14_train696_production.py \
  --config configs/stage09_mk14_train696_unidock113_production.json \
  --audit-only
```

After `status: audit_only_ok`, run all 48 batches with automatic shutdown:

```bash
nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage09_mk14_train696_production_remote.sh \
  > stage09_mk14_train696.log 2>&1 &
echo $! | tee stage09.pid
tail -f stage09_mk14_train696.log
```

The Stage 07c measured rate projects about 2.3 hours of GPU batch time on an
RTX 4090. Allow roughly 3-4 hours for input checks, pose-integrity auditing,
matrix construction, independent re-audit, and archive creation.

## Three GPUs

Three independent instances may each run one seed from the same input bundle:

```bash
nohup env SEED_IDS=seed0 PARTITION_ID=seed0 AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage09_mk14_train696_production_remote.sh \
  > stage09_seed0.log 2>&1 &
```

Use `seed1` and `seed2` on the other instances. Each produces a checkpoint
archive. Extract all three checkpoint archives over one clean input bundle,
then run the remote script with `FINALIZE_ONLY=1` to construct and independently
audit the complete matrices.
