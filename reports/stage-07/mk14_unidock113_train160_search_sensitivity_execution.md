# Stage 07 Uni-Dock Search Sensitivity Execution

## Scope

This bundle contains consumed MAPK14 Train-160 rows only. It runs four
receptors, three paired seeds, and the official Uni-Dock 1.1.3 fast, balance,
and detail profiles. Fresh validation and locked-test files are absent.

## Upload and extract

Upload `stage07_mk14_unidock113_train160_search_sensitivity_input_v1.tar.gz`
to `/root/autodl-tmp`, then run:

```bash
BASE=/root/autodl-tmp/stage07_mk14_unidock_sensitivity_v1
mkdir -p "$BASE"
tar -xzf \
  /root/autodl-tmp/stage07_mk14_unidock113_train160_search_sensitivity_input_v1.tar.gz \
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

If the environment already exists, update it instead:

```bash
conda env update \
  -n qubo-unidock-stage07 \
  -f environment/stage07_unidock_gpu.yml \
  --prune
conda activate qubo-unidock-stage07
```

## Audit without docking

```bash
python scripts/experimental/unidock/run_stage07_unidock_sensitivity.py \
  --config configs/stage07_mk14_unidock113_train160_search_sensitivity.json \
  --audit-only
```

The audit must report 160 ligands, four receptors, three profiles, three
seeds, 5760 expected pairs, zero closure pseudoatoms, and zero validation/test
rows.

## Run in the background

```bash
nohup bash \
  scripts/experimental/unidock/run_stage07_unidock_sensitivity_remote.sh \
  > stage07_unidock_sensitivity.log 2>&1 &

echo $! | tee stage07_unidock_sensitivity.pid
```

Monitor progress with:

```bash
tail -f stage07_unidock_sensitivity.log
```

In a second terminal:

```bash
find \
  results/runs/stage07_mk14_unidock113_train160_search_sensitivity \
  -name batch_summary.json | wc -l
nvidia-smi
```

There are 36 batches in total. A completed run automatically evaluates the
profiles and creates the core result archive.

## Resume after interruption

Run the same background command again. The remote script invokes the runner
with `--resume`; every completed batch is verified by signature and SHA-256
before it is reused.

## Collect results

After completion:

```bash
cat \
  data/stage07_mk14_unidock113_train160_search_sensitivity_result.json

cat \
  reports/stage-07/mk14_unidock113_train160_search_sensitivity.md

sha256sum \
  stage07_mk14_unidock113_train160_search_sensitivity_core_v1.tar.gz
```

Download and return only:

`stage07_mk14_unidock113_train160_search_sensitivity_core_v1.tar.gz`
