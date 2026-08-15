# Stage 41c BACE1 large-pool cognate redocking

## Scope

This is a technical pose-recovery gate for the frozen BACE1 large pool. It runs
48 prepared receptor/cognate-ligand pairs across three frozen seeds, for 144
Uni-Dock GPU batches. The known 6DMI preparation failure remains in the
49-receptor denominator.

## Frozen protocol

- Uni-Dock 1.1.3, enhanced profile, Vina scoring
- Exhaustiveness 1024, max step 80, refine step 5
- One output mode and seeds 20260801, 20260802, 20260803
- Common box center `(24.35, 13.96, 23.58)` angstrom
- Common box size `(30, 22, 22)` angstrom
- Pose success threshold: symmetry-corrected heavy-atom RMSD at most 2.0
  angstrom without post-docking alignment

## Decision rule

At least two of three seeds and the median seed RMSD must pass for each
receptor. At least 40 of the frozen 49 receptors must pass globally. No
replacement, protocol retuning, or activity-label access is permitted after
outcomes are known.

Passing authorizes preparation and docking of the frozen 266-ligand BACE1
development panel against every passing receptor. It does not authorize an
efficacy, QUBO-superiority, or quantum-advantage claim.

## Executed result

The run completed all 144 frozen receptor-seed batches with zero unresolved
warnings, zero pose-integrity failures, and no missing or hash-mismatched
runtime evidence. Thirty-four of the frozen 49 receptors passed, below the
preregistered minimum of 40, so the Stage 41c gate failed.

Among the 34 passing receptors, 27 passed all three seeds and seven passed two
of three. Their median receptor RMSD was 0.5792 angstrom. The known 6DMI input
failure remained in the denominator as required.

Two redocking failures were close to the 2.0-angstrom threshold: 3L5F had a
median RMSD of 2.0323 angstrom and 2QK5 had 2.0619 angstrom. This does not
explain the six-receptor shortfall. A post-hoc threshold of 3.5 angstrom would
still pass only 39 of 49; reaching 40 would require approximately 4.0 angstrom,
which is not an acceptable retrospective change to the frozen gate.

The preregistered Stage 41 route therefore stops before the 266-ligand BACE1
functional matrix. The 34 stable receptors may be used only for a clearly
labelled post-hoc failure diagnostic or for designing a newly preregistered
experiment; they cannot retroactively rescue this gate.
