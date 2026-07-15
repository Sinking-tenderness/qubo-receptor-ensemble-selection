# Stage 4 non-MD e32 Remote Execution

## Scope

Run eight eligible non-MD CDK2 receptors against the fixed 160-ligand
development panel with three paired Vina seeds. Each seed contains 1,280
receptor-ligand jobs. The 40-ligand locked test split is absent from the input
archive and every benchmark manifest.

The input archive is:

`stage04_cdk2_non_md_80a80d_e32_inputs_v1.tar.gz`

Expected SHA-256:

`F991A0C35D030C3B1140AAC2C169B093AF7A91075B96049D3C487714D48C0400`

It contains 171 files: two manifests, eight receptor PDBQTs, 160 development
ligand PDBQTs, and the official Vina 1.2.7 Linux x86-64 executable. Vina
executable SHA-256:

`F31F774F723BBA7BBE6E9D1C47577020EEA9A8DA16424284C043D22593570644`

Release source: <https://github.com/ccsb-scripps/AutoDock-Vina/releases/tag/v1.2.7>

## Repository and Input Setup

Adjust `REPO` only if the clone is in a different directory.

```bash
set -euo pipefail

REPO=/root/autodl-tmp/qubo-receptor-ensemble-selection
BUNDLE=/root/autodl-tmp/stage04_cdk2_non_md_80a80d_e32_inputs_v1.tar.gz

cd "$REPO"
git pull --ff-only

echo "F991A0C35D030C3B1140AAC2C169B093AF7A91075B96049D3C487714D48C0400  $BUNDLE" \
  | sha256sum -c -

tar -xzf "$BUNDLE" -C "$REPO"
chmod +x environment/bin/vina_1.2.7_linux_x86_64

sha256sum \
  environment/bin/vina_1.2.7_linux_x86_64 \
  data/processed/stage04_cdk2_non_md_e32_receptor_manifest.csv \
  data/processed/stage04_cdk2_development_80a80d_pdbqt_manifest.csv

nproc
free -h
df -h "$REPO"
```

Expected manifest hashes:

- receptor manifest:
  `52D5530AAAD57B210ACF76F63CE334E1A6ACA22CB58CDA5371BB77E89F52C136`
- ligand manifest:
  `7F8A28CF7867A8442F1539A246148787093E2F4BF25C2EB569500F68C74BFE8E`

## Zero-Job Audit Gate

Run all three audits before starting docking. These commands verify every
configured input and every per-receptor/per-ligand PDBQT hash but create no
result file and start no Vina process.

```bash
for seed in 0 1 2; do
  python scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage04_cdk2_non_md_80a80d_e32_seed${seed}_benchmark_linux.json" \
    --audit-only \
    > "/tmp/stage04_seed${seed}_audit.json"
done

grep -H '"status": "audit_only_ok"' /tmp/stage04_seed*_audit.json
grep -H '"locked_test_manifest_rows": 0' /tmp/stage04_seed*_audit.json
```

There must be three matches for each `grep` command.

## Docking Runs

Each configuration already uses eight workers, four CPU threads per Vina job,
and a maximum total CPU budget of 32. Run seeds sequentially unless the machine
has at least 96 CPU threads and concurrent scaling has been benchmarked.

Use `--resume` from the first invocation. It works on a fresh run and also
reuses completed receptor score tables plus ligand checkpoints after a terminal
disconnect or instance restart, provided the persistent result directory has
not been deleted.

```bash
mkdir -p /root/autodl-tmp/stage04_run_logs

for seed in 0 1 2; do
  python scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage04_cdk2_non_md_80a80d_e32_seed${seed}_benchmark_linux.json" \
    --resume \
    2>&1 | tee "/root/autodl-tmp/stage04_run_logs/seed${seed}.log"
done
```

For a terminal-independent run, execute the same loop inside `tmux`. Do not use
`--overwrite` after a partial run; it intentionally deletes expected
checkpoints and completed receptor outputs.

## Completion Gate

```bash
python - <<'PY'
import json
from pathlib import Path

for seed in range(3):
    path = Path(
        f"results/runs/stage04_cdk2_non_md_80a80d_e32_seed{seed}_linux/summary.json"
    )
    summary = json.loads(path.read_text())
    print(
        seed,
        summary["status"],
        summary["successful_receptor_ligand_pairs"],
        summary["failed_receptor_ligand_pairs"],
        summary["search_quality_warning_count"],
    )
    assert summary["status"] in {"ok", "ok_with_search_warning"}
    assert summary["successful_receptor_ligand_pairs"] == 1280
    assert summary["failed_receptor_ligand_pairs"] == 0
PY
```

Search warnings are retained observations, not permission to replace matrix
cells. A failed pair blocks aggregation and must remain auditable.

## Three-Seed Aggregation

Run only after all three completion assertions pass.

```bash
python scripts/aggregate_vina_seed_replicates.py \
  --config configs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux.json
```

The median matrix is primary. The minimum matrix is sensitivity only. Do not
substitute selected minimum or rerun scores into the median matrix.

## Result Package

Package compact tables and summaries rather than poses and per-ligand logs.

```bash
cd "$REPO"

cat > /tmp/stage04_result_files.txt <<'EOF'
results/runs/stage04_cdk2_non_md_80a80d_e32_seed0_linux/summary.json
results/runs/stage04_cdk2_non_md_80a80d_e32_seed0_linux/receptor_runs.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_seed0_linux/representative_scores.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_seed1_linux/summary.json
results/runs/stage04_cdk2_non_md_80a80d_e32_seed1_linux/receptor_runs.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_seed1_linux/representative_scores.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_seed2_linux/summary.json
results/runs/stage04_cdk2_non_md_80a80d_e32_seed2_linux/receptor_runs.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_seed2_linux/representative_scores.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux/aggregate_scores.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux/median_score_matrix.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux/minimum_score_matrix.csv
results/runs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux/summary.json
EOF

if [ -f results/runs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux/seed_stability_warnings.csv ]; then
  echo results/runs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux/seed_stability_warnings.csv \
    >> /tmp/stage04_result_files.txt
fi

tar -czf /root/autodl-tmp/stage04_cdk2_non_md_e32_results_v1.tar.gz \
  -T /tmp/stage04_result_files.txt

sha256sum /root/autodl-tmp/stage04_cdk2_non_md_e32_results_v1.tar.gz
```

The final 40-ligand test remains closed after aggregation. The returned
development matrices must first be merged with the eight-MD development
matrices and pass the preregistered nested development gate.
