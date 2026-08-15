# Stage43 PPARG MD-96 execution

Stage43 transfers the unchanged Stage42f rank-sensitive pair QUBO to the 96-frame PPARG MD panel frozen in Stage31. Sixteen receptors reuse the independently audited Stage32 Train-160 matrix; only the remaining 80 receptors are newly prepared and docked.

## Remote execution

The remote instance must retain the eight Stage28b aligned trajectory files and the existing `qubo-receptor-md` and `qubo-unidock-stage08` environments.

```bash
cd /root/autodl-tmp
sha256sum stage43_pparg_md96_rank_sensitive_replication_input_v1.tar.gz
mkdir -p stage43_pparg_md96_v1
tar -xzf stage43_pparg_md96_rank_sensitive_replication_input_v1.tar.gz -C stage43_pparg_md96_v1
cd stage43_pparg_md96_v1
sha256sum -c bundle_manifest.sha256

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qubo-unidock-stage08

AUTO_POWEROFF=1 nohup bash scripts/experimental/unidock/run_stage43_pparg_md96_production_remote.sh \
  > stage43_pparg_md96.log 2>&1 &
echo $! | tee stage43.pid
tail -f stage43_pparg_md96.log
```

If auto-discovery cannot find the Stage28b workspace, prefix the command with `TRAJECTORY_ROOT=/root/autodl-tmp/<stage28b-workspace>`.

The run is resumable. Repeating the same command with `--resume` behavior in the wrapper reuses every batch whose protocol signature and pose audit pass.

## Expected workload

- Combined matrix: 96 receptors x 160 ligands x 3 seeds = 46,080 pairs.
- Historical reuse: 7,680 pairs from 16 Stage32 receptors.
- New GPU work: 80 receptors x 160 ligands x 3 seeds = 38,400 pairs in 240 batches.
- No fresh-validation rows, test rows, or quantum-hardware jobs are permitted.
