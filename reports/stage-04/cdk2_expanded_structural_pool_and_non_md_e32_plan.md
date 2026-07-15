# Stage 4 CDK2 Expanded Structural Pool and non-MD e32 Plan

## Scientific Decision

The latest nested development-set cross-validation gate found no stable
combination advantage in the eight AF2/MD medoids. The final mechanical fit
collapsed to the single receptor `CDK2_AF2_MD2NS_C06_F077`, and all five
preregistered acceptance rules failed. Further coefficient tuning on the same
160 development ligands is therefore stopped.

The next controlled intervention is to increase receptor information content,
not to change QUBO coefficients. Experimental CDK2 structures and the AF2
parent are added to the current MD-only pool using structure-only eligibility
and clustering rules. No ligand identity, active/decoy label, docking score,
enrichment metric, or QUBO result influences receptor inclusion.

## Two Different Eight-Receptor Pools

Two earlier receptor sets must not be conflated:

1. The Stage 2 static pool contained eight crystal receptors: 1AQ1, 1H00,
   1HCL, 1JVP, 1Y8Y, 2C68, 2C69, and 3RKB. It was docked against 200 ligands
   with `exhaustiveness=1` and one search replicate.
2. The current Stage 3 pool contains eight medoids selected from 100 correlated
   frames of one 2 ns AF2-started MD trajectory. It was docked with
   `exhaustiveness=32` and three seeds.

The Stage 4 pool combines eligible members of both structural sources. Old
Stage 2 e1 scores are retained as prior negative evidence but are not merged
with the e32 MD matrix because search-strength and replicate differences would
confound receptor effects.

## Receptor Integrity Gate

The configured candidate pool contained 17 structures:

- eight Stage 3 AF2/MD medoids;
- eight previously used crystal structures;
- the aligned AF2 parent structure.

All source PDB and prepared PDBQT files were SHA-256 verified. Eligibility
required all 21 reference pocket residues, matching residue names, a prepared
PDBQT, zero PDBQT `HETATM` records, and exactly the AutoDock atom types
`A,C,HD,N,NA,OA,SA`.

`CDK2_1H00_aligned` was excluded because reference pocket residue Gly13 is
missing. The final pool therefore contains 16 receptors:

- eight MD medoids;
- seven crystal receptors;
- one AF2 parent receptor.

The 1JVP crystal file uses numeric alternate locations `1/2`. Its existing
Meeko receptor preparation selected altloc `1`, so Stage 4 explicitly applies
the same selection during pocket feature extraction. All other candidates use
altloc `A`; atoms with blank altloc are always retained.

## Label-Independent Structural Result

For every eligible receptor, 210 pocket C-alpha pair distances and 210
side-chain-heavy centroid pair distances were calculated. Gly11 and Gly13 use
their C-alpha as the defined side-chain-centroid fallback. All 420 features
passed the preregistered 0.01 A standard-deviation threshold.

Pairwise pocket C-alpha Kabsch RMSD showed:

| Pair group | Pairs | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| non-MD vs non-MD | 28 | 0.219 A | 0.654 A | 0.861 A |
| MD vs MD | 28 | 0.439 A | 0.891 A | 1.352 A |
| MD vs non-MD | 64 | 0.647 A | 1.023 A | 1.483 A |

At `k=2`, Ward clustering separated all eight MD medoids from the seven
crystals plus AF2 parent. At `k=3`, the non-MD group remained one cluster and
the MD medoids divided into two clusters. This is evidence that adding the
non-MD group expands pocket geometry beyond the current MD-only pool. It is
not evidence of docking-score complementarity or better enrichment.

The closest pocket pair was 2C68-2C69 at 0.219 A. The farthest was
1JVP-MD-C06 at 1.483 A. All eight eligible non-MD receptors are retained for
docking rather than deleting candidates solely from geometric similarity.

## Locked Development Input

The existing scaffold split is preserved exactly:

- development: train plus validation, 160 ligands, 80 active and 80 decoy;
- locked test: 40 ligands, 20 active and 20 decoy.

The development PDBQT manifest records a fresh SHA-256 for each of the 160
files. No locked-test ligand is written into the new docking manifest. The
existing 200-row MD aggregate matrices were mechanically subset to the same
160 development IDs; receptor score cells for skipped locked rows were not
parsed, and no ranking metric was calculated.

## Paired non-MD Docking Protocol

The new calculation contains:

- receptors: 8 eligible non-MD structures;
- ligands: 160 development ligands only;
- seeds: `20260901`, `20360901`, and `20460901`;
- total receptor-ligand-seed jobs: `8 x 160 x 3 = 3,840`;
- Vina 1.2.7, common 1AQ1-aligned box;
- `exhaustiveness=32`, `num_modes=1`, `cpu=4` per job;
- eight workers with maximum total CPU 32.

The three seeds are identical to the Stage 3 MD protocol. Median across seeds
is the primary score matrix; minimum across seeds remains sensitivity only.
No single seed or selectively rerun cell may replace a primary matrix value.

## Integrity Checkpoints

- Expanded structural-pool summary SHA-256:
  `788CB991A8633A68F55C7D5F72AA404F6273D9DAED8B23E36468AB590DFAA4B6`
- Eligible 16-receptor manifest SHA-256:
  `C70185C8A2F52AA5175966F690B060320FD47B0C33B2EF12828A672FA757CB81`
- non-MD e32 receptor manifest SHA-256:
  `52D5530AAAD57B210ACF76F63CE334E1A6ACA22CB58CDA5371BB77E89F52C136`
- Development ligand manifest SHA-256:
  `7F8A28CF7867A8442F1539A246148787093E2F4BF25C2EB569500F68C74BFE8E`
- Development-ligand summary SHA-256:
  `84050DF761DDFE028D9B3AC822B05D81B6F66B47BCB0B27712529F106EF3AB3E`
- MD development primary matrix SHA-256:
  `D410983B187A50CB8AC0DECF9BA667AF49E794F10A03763D7DE29A5CBF842399`
- MD development sensitivity matrix SHA-256:
  `2A7A94C668E9B7DAA3956543DE571A8A67DBDACFABA63A7421998BF28833EA5D`
- MD development-subset summary SHA-256:
  `B842405793268B95DE5571E2671A99C4E0883EC3F8A4A34F86F0C7CD2604BF66`
- Selective remote-input archive: 170 files comprising two manifests, eight
  receptor PDBQTs, and 160 development-ligand PDBQTs; zero locked-test rows;
  SHA-256:
  `0FC508D3AB597A073B49E1F37B02839401EA18D17766CE8B155EEFCAF296135E`

Each locally generated summary was reproduced with an identical SHA-256 on a
second run in the `qubo-receptor-ensemble` environment.

## Next Gate

After all 3,840 new jobs finish:

1. Require 1,280/1,280 successful pairs in each seed run; preserve all search
   warnings and failures.
2. Aggregate the three complete non-MD runs into median-primary and
   minimum-sensitivity matrices.
3. Merge them with the already materialized eight-MD, 160-ligand development
   matrices to obtain a 16-receptor development matrix.
4. Repeat nested scaffold cross-validation with the same locked-test boundary
   and compare single-best, all-receptor, structure-cluster medoids, greedy,
   and QUBO subset selection.
5. Keep the final 40-ligand test closed unless preregistered development-gate
   criteria pass and the candidate protocol is manually reviewed.

This stage tests whether broader receptor information creates stable subset
value. It does not claim quantum advantage, biological optimality, or improved
virtual screening before the development gate is completed.
