# Stage 42c BACE1 Train-266 production

## Scope

Stage 42c runs the frozen 266-ligand development panel against exactly the 34
Stage 41c redocking-qualified BACE1 receptors. Stage 41c remains failed because
34 receptors did not meet its preregistered minimum of 40; Stage 42c is an
explicitly outcome-informed, post-hoc development and scaling experiment.

The complete grid contains 34 receptors, 266 ligands, and three paired seeds,
for 102 GPU batches and 27,132 receptor-ligand-seed pairs. It uses Uni-Dock
1.1.3 with the frozen enhanced profile: exhaustiveness 1024, max-step 80,
refine-step 5, and one retained pose.

## Input gate

The independent Stage 42b audit verified 266 ligand identities, 133 actives and
133 decoys, 532 prepared SDF/PDBQT files, zero failed ligands, zero macrocycle
closure pseudoatoms, and no fresh-validation or locked-test structure access.
Twenty-two SDF records report MMFF94 non-convergence at the iteration cap, but
all embedded structures and final PDBQT files passed integrity and hash checks.

## Execution contract

Each receptor-seed batch contains all 266 ligands. A batch is reusable only
when its protocol signature, score hash, pose hashes, pose-integrity audit, and
warning adjudication all pass. The remote wrapper always uses `--resume`, can
partition by seed or receptor, writes checkpoint archives for incomplete runs,
and supports `AUTO_POWEROFF=1`.

Completion requires all 102 batches, 27,132 unique scores, zero unresolved
warnings, and zero pose-integrity failures. The primary matrix is the median
score across three seeds; the sensitivity matrix uses the minimum score.

## Interpretation boundary

This development matrix can determine whether a 34-receptor BACE1 pool creates
a meaningful large combinatorial selection problem. It cannot rescue Stage
41c, cannot be presented as independent target validation, and does not by
itself establish QUBO superiority or quantum advantage.
