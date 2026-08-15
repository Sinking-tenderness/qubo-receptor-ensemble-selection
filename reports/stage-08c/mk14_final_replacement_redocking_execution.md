# Stage 08c MAPK14 Final Replacement Redocking

## Decision

Stage 08 admitted six of eight new receptors. Stage 08b then admitted
`MK14_3ITZ_aligned` but rejected `MK14_2BAK_aligned`. `MK14_4A9Y_aligned`
is a zero-distance structural duplicate of the failed `2BAK-AQZ` case, so it
is excluded without another GPU run. The current pool therefore contains 15
admitted receptors.

Using the frozen structural-distance rule, the final nonredundant replacement
is `MK14_1OZ1_aligned` with co-crystal ligand `FPH`. This stage runs only one
receptor with the three paired seeds, for `1 x 3 = 3` Uni-Dock jobs. It keeps
the frozen `2 A` RMSD gate and reads no validation or test rows.

## Execute

Upload `stage08c_mk14_final_replacement_input_v1.tar.gz` to
`/root/autodl-tmp`, then extract and verify it:

```bash
cd /root/autodl-tmp
mkdir -p stage08c_mk14_final_replacement_v1
tar -xzf stage08c_mk14_final_replacement_input_v1.tar.gz \
  -C stage08c_mk14_final_replacement_v1
cd stage08c_mk14_final_replacement_v1
sha256sum -c bundle_manifest.sha256

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qubo-unidock-stage08
```

Run the read-only input audit:

```bash
python scripts/experimental/unidock/run_stage08c_mk14_final_replacement_redocking.py \
  --config configs/stage08c_mk14_final_replacement_redocking.json \
  --audit-only
```

After `status: audit_only_ok`, launch the three-job run with automatic shutdown:

```bash
nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage08c_mk14_final_replacement_remote.sh \
  > stage08c_mk14_final_replacement.log 2>&1 &
echo $! | tee stage08c.pid
tail -f stage08c_mk14_final_replacement.log
```

On success, return the `core` and `diagnostics` archives printed at the end of
the log. On failure, return the `failed_diagnostics` archive. A passing
independent audit freezes the final 16-receptor pool and authorizes a separate,
preregistered Train-696 production run.
