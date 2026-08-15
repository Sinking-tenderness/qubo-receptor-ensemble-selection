# Stage 07c Uni-Dock Warning Adjudication Execution

## Scope

This bundle contains consumed MAPK14 Train-160 rows only. It runs one new
enhanced-profile seed across four receptors and replays one known warning
batch. The total is 800 Uni-Dock pairs in five batches. Fresh validation and
locked-test files are absent.

The statistical thresholds remain unchanged. A known coordinate warning is
resolved only when all output poses pass atom and mode audits and the replay
exactly matches all 160 frozen scores and pose hashes.

## Upload and extract

Upload `stage07c_mk14_unidock113_warning_adjudication_input_v1.tar.gz` to
`/root/autodl-tmp`, then run:

```bash
BASE=/root/autodl-tmp/stage07c_mk14_unidock_warning_adjudication_v1
mkdir -p "$BASE"
tar -xzf \
  /root/autodl-tmp/stage07c_mk14_unidock113_warning_adjudication_input_v1.tar.gz \
  -C "$BASE"
cd "$BASE"
```

## Environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q '^qubo-unidock-stage07 '; then
  conda activate qubo-unidock-stage07
else
  conda env create -f environment/stage07_unidock_gpu.yml
  conda activate qubo-unidock-stage07
fi

unidock --version
python -c "import scipy; print(scipy.__version__)"
nvidia-smi
```

## Audit without docking

```bash
python \
  scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py \
  --config configs/stage07c_mk14_unidock113_warning_adjudication.json \
  --audit-only
```

The audit must report 160 ligands, four receptors, four combined seeds, 1920
frozen prior scores, 640 new-seed pairs, 160 replay pairs, zero closure
pseudoatoms, and zero validation/test rows.

## Run

```bash
AUTO_POWEROFF=1 nohup bash \
  scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication_remote.sh \
  > stage07c_warning_adjudication.log 2>&1 &

echo $! | tee stage07c_warning_adjudication.pid
```

`AUTO_POWEROFF=1` requests poweroff after success or failure. Omit it when the
instance should remain online.

Monitor with:

```bash
tail -f stage07c_warning_adjudication.log
```

In another terminal:

```bash
find \
  results/runs/stage07c_mk14_unidock113_warning_adjudication \
  -name batch_summary.json | wc -l
nvidia-smi
```

The final batch count is five. Expected wall time on an RTX 4090 is about
4-8 minutes. The same command can be rerun after interruption; completed
batches are reused only after signature, score, pose, and audit checks.

## Results

The remote script creates:

```text
/root/autodl-tmp/stage07c_mk14_unidock113_warning_adjudication_core_v1.tar.gz
/root/autodl-tmp/stage07c_mk14_unidock113_pose_diagnostics_v1.tar.gz
```

Verify and download both:

```bash
sha256sum \
  /root/autodl-tmp/stage07c_mk14_unidock113_warning_adjudication_core_v1.tar.gz \
  /root/autodl-tmp/stage07c_mk14_unidock113_pose_diagnostics_v1.tar.gz
```
