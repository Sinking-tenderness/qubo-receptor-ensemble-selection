# Stage41b BACE1 large-pool input preparation

This bundle prepares 49 frozen BACE1 receptors and their cognate ligands. It starts no docking job and reads no protected benchmark row. A GPU is not required.

```bash
cd /root/autodl-tmp
sha256sum stage41b_bace1_large_pool_redocking_input_v1.tar.gz
mkdir -p stage41b_bace1_large_pool_input_v1
tar -xzf stage41b_bace1_large_pool_redocking_input_v1.tar.gz \
  -C stage41b_bace1_large_pool_input_v1
cd stage41b_bace1_large_pool_input_v1
sha256sum -c bundle_manifest.sha256

nohup env AUTO_POWEROFF=1 OUTPUT_ROOT=/root/autodl-tmp \
  bash scripts/run_stage41b_bace1_large_pool_input_preparation_remote.sh \
  > stage41b_bace1_large_pool_input_preparation.log 2>&1 &

echo $! | tee stage41b.pid
tail -f stage41b_bace1_large_pool_input_preparation.log
```

Rerunning the same command resumes every case whose checkpoint and output hashes remain valid. On success, download both the core and diagnostic archives from `/root/autodl-tmp`.
