# Stage 08b MAPK14 Replacement Redocking

## Decision

The first Stage 08 execution completed all 24 jobs with zero engine warnings
and zero pose-integrity failures. Six of eight new receptors passed the frozen
three-seed RMSD gate. `MK14_3ZSH_aligned` failed all three seeds and
`MK14_2GFS_aligned` passed only one of three, so both are permanently excluded
from this production pool. The 2 A threshold is not relaxed.

Using only the frozen structural-distance table and binary technical-admission
status, deterministic max-min refill selected:

- `MK14_3ITZ_aligned`, co-crystal ligand `P66`.
- `MK14_2BAK_aligned`, co-crystal ligand `AQZ`.

This recovery runs only `2 receptors x 3 seeds = 6` Uni-Dock jobs. It does not
rerun the completed first-round jobs and contains no validation or test rows.

## Execute

Extract into a clean directory on the data disk and verify every bundled file:

```bash
cd /root/autodl-tmp
mkdir -p stage08b_mk14_replacement_v1
tar -xzf stage08b_mk14_expanded16_replacement_input_v1.tar.gz \
  -C stage08b_mk14_replacement_v1
cd stage08b_mk14_replacement_v1
sha256sum -c bundle_manifest.sha256

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qubo-unidock-stage08
```

Run the read-only recovery audit first:

```bash
python scripts/experimental/unidock/run_stage08b_mk14_replacement_redocking.py \
  --config configs/stage08b_mk14_expanded16_replacement_redocking.json \
  --audit-only
```

After `status: audit_only_ok`, launch the six-job run:

```bash
nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage08b_mk14_replacement_remote.sh \
  > stage08b_mk14_replacement.log 2>&1 &
echo $! | tee stage08b.pid
tail -f stage08b_mk14_replacement.log
```

On success, return both the core and diagnostics archives. Only a passing
independent audit creates the final 16-receptor manifest and authorizes design
of the separate Train-696 production bundle.
