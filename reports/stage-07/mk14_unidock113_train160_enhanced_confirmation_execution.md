# Stage 07b Uni-Dock Enhanced Confirmation Execution

## Scope

This bundle contains consumed MAPK14 Train-160 rows only. It runs four
receptors, three paired seeds, and four Uni-Dock 1.1.3 profiles. The 512/40
profile is a diagnostic recheck; 512/80, 1024/40, and 1024/80 are candidates.
Fresh validation and locked-test files are absent.

## Upload and extract

Upload `stage07b_mk14_unidock113_train160_enhanced_confirmation_input_v1.tar.gz`
to `/root/autodl-tmp`, then run:

```bash
BASE=/root/autodl-tmp/stage07b_mk14_unidock_confirmation_v1
mkdir -p "$BASE"
tar -xzf \
  /root/autodl-tmp/stage07b_mk14_unidock113_train160_enhanced_confirmation_input_v1.tar.gz \
  -C "$BASE"
cd "$BASE"
```

## Create the environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment/stage07_unidock_gpu.yml
conda activate qubo-unidock-stage07

unidock --version
python -c "import scipy; print(scipy.__version__)"
nvidia-smi
```

If the environment already exists, use:

```bash
conda env update \
  -n qubo-unidock-stage07 \
  -f environment/stage07_unidock_gpu.yml \
  --prune
conda activate qubo-unidock-stage07
```

## Audit without docking

```bash
python \
  scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py \
  --config \
  configs/stage07b_mk14_unidock113_train160_enhanced_confirmation.json \
  --audit-only
```

The audit must report 160 ligands, four receptors, four profiles, three seeds,
7680 expected pairs, zero closure pseudoatoms, and zero validation/test rows.

## Run in the background

```bash
nohup bash \
  scripts/experimental/unidock/run_stage07b_unidock_confirmation_remote.sh \
  > stage07b_unidock_confirmation.log 2>&1 &

echo $! | tee stage07b_unidock_confirmation.pid
```

To request poweroff after either success or failure, launch with:

```bash
AUTO_POWEROFF=1 nohup bash \
  scripts/experimental/unidock/run_stage07b_unidock_confirmation_remote.sh \
  > stage07b_unidock_confirmation.log 2>&1 &

echo $! | tee stage07b_unidock_confirmation.pid
```

The poweroff option is intentionally opt-in. The exit trap calls `sync` before
requesting shutdown, but the instance will also stop after a failed run.

Monitor progress with:

```bash
tail -f stage07b_unidock_confirmation.log
```

In a second terminal:

```bash
find \
  results/runs/stage07b_mk14_unidock113_train160_enhanced_confirmation \
  -name batch_summary.json | wc -l
nvidia-smi
```

There are 48 batches in total. On the same RTX 4090 class used for Stage 07,
the expected wall time is approximately 30-60 minutes. The remote script
evaluates all profiles and creates both a core archive and a warning-focused
pose archive.

## Resume after interruption

Run the same background command again. Completed batches are reused only after
their signature, score hash, output pose hashes, and pose-audit fields pass.

## Collect results

After completion:

```bash
cat \
  data/stage07b_mk14_unidock113_train160_enhanced_confirmation_result.json

cat \
  reports/stage-07/mk14_unidock113_train160_enhanced_confirmation.md

cat \
  data/stage07b_mk14_unidock113_train160_pose_diagnostics_summary.json

sha256sum \
  /root/autodl-tmp/stage07b_mk14_unidock113_train160_enhanced_confirmation_core_v1.tar.gz \
  /root/autodl-tmp/stage07b_mk14_unidock113_train160_pose_diagnostics_v1.tar.gz
```

Download and return both archives. The pose archive is intentionally small
when no warnings occur and contains complete affected batches when a warning
or atom-integrity failure is detected.
