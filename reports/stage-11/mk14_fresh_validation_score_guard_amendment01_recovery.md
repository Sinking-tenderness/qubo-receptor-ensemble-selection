# Stage 11 Score-Guard Amendment 01 Recovery

## Failure Finding

The GPU environment, rigid-macrocycle preparation, and Uni-Dock execution were
healthy. Eight batches completed with 12,608 audited poses, zero pose-integrity
failures, and zero unresolved warnings. A ninth batch produced all 1,576 poses
and a normal end-of-run log, but post-processing stopped on this finite score:

- Seed: `seed1`
- Receptor: `MK14_2BAJ_aligned`
- Ligand: `MK14_decoy_L032976`
- Raw Uni-Dock score: `+172.351 kcal/mol`
- Pose count: 1

The output is syntactically complete. Its reported intermolecular energy is
strongly positive, consistent with a severe clash. The same ligand scored near
-8 kcal/mol in the eight completed receptor-seed combinations. The exception
was raised only because the original technical parser guard rejected any score
whose absolute value exceeded 100 kcal/mol.

## Amendment

Amendment 01 raises the finite-score storage guard from 100 to 1,000 kcal/mol.
It retains the raw `172.351` value and does not cap, clip, replace, discard, or
redock it with a different seed. The docking command, candidates, receptors,
ligands, labels, three seeds, search parameters, aggregation, BEDROC alpha, and
confirmatory threshold remain unchanged.

The original configuration and protocol signatures remain frozen at the 100
kcal/mol guard so the eight completed checkpoints can be reused exactly. The
recovery wrapper applies 1,000 kcal/mol only while parsing new outputs and while
performing the final finite-score audit. The amendment is written into new batch
summaries and the final Stage 11 summary.

## Resume Workload

- Reusable complete batches: 8 of 18.
- Batch to rerun after post-processing interruption: 1.
- Not-yet-run batches: 9.
- Expected GPU docking time: about 50 to 55 minutes at the observed rate.
- Allow additional time for environment checks, full pose audit, bootstrap
  evaluation, packaging, and disk synchronization.

## Remote Command

After extracting the recovery bundle and verifying its manifest:

```bash
cd /root/autodl-tmp/stage11_mk14_fresh_validation_recovery_amendment01_v1

nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage11_mk14_fresh_validation_recovery_remote.sh \
  > stage11_recovery_amendment01.log 2>&1 &

echo $! | tee stage11_recovery.pid
tail -f stage11_recovery_amendment01.log
```

The first eight batches should print `resume ok`. The interrupted ninth batch is
rerun because it has no completed score ledger or batch summary. The script then
runs the remaining batches, independent matrix audit, frozen evaluation, result
packaging, `sync`, and optional automatic shutdown.
