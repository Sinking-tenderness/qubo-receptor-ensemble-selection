# Stage 3 CDK2 AF2 MD Receptor Preparation Record

## Scope

The eight aligned MD medoids were converted into rigid AutoDock Vina receptor
PDBQT files with one fixed ProDy and Meeko protocol. A single primary medoid,
C01 F005, was prepared first as a gate. The complete batch was run only after
that gate passed.

## Software and Protocol

- Python: 3.11.15
- Meeko: 0.7.1
- ProDy: 2.4.1
- Setuptools: 80.10.2 (`<81` compatibility pin)
- Receptor chain: A
- Charge model: Gasteiger
- Alternate-location rule: ProDy A
- `allow_bad_res`: false
- Expected AutoDock types: A, C, HD, N, NA, OA, SA
- Batch config SHA-256:
  `0E5382C0D4C512F0AAB42A3EF1FC9081A53D12459D16E711877B9D9770C3C553`

The `pkg_resources` message is a deprecation warning from ProDy and did not
change the successful return code. The setuptools compatibility pin remains
necessary for ProDy 2.4.1.

## C01 Pilot Gate

- Input aligned-heavy PDB SHA-256:
  `09BA60D7F991C0EA36DC89E75B9460F24F8A2AEE46E2F1936940B7175F9E4120`
- Pilot summary SHA-256:
  `9340AF5D355F5A81899618FCE2D3F937B8904E985CA9E22B2153D96F057CC13B`
- Pilot receptor PDBQT SHA-256:
  `72188E1B98B0C64A4348138FEDF6D50B3839A006A41EA85BF6FC7516D797E1DC`
- Residues before and after parameterization: 298
- Input heavy atoms: 2,398
- Prepared PDB atoms: 4,846
- Prepared PDB hydrogens: 2,448
- Receptor PDBQT atoms: 2,912
- Receptor PDBQT hydrogen-like atoms: 514
- Charge range: -0.549 to 0.345
- HETATM records: 0

The batch-generated C01 PDBQT has the same SHA-256 as the pilot PDBQT. This
confirms that the audited single-receptor command was reproduced by the batch
workflow.

## Eight-Receptor Batch Audit

- Requested receptors: 8
- Successful receptors: 8
- Failed receptors: 0
- Total measured preparation runtime: 183.815 seconds
- Unique prepared-PDB hashes: 8
- Unique receptor-PDBQT hashes: 8
- All output-file hashes matched their manifest records

Every receptor produced the same composition audit:

| Property | Value |
|---|---:|
| Protein-only heavy atoms | 2,398 |
| Prepared PDB atoms | 4,846 |
| Prepared PDB hydrogens | 2,448 |
| Receptor PDBQT atoms | 2,912 |
| Receptor residues | 298 |
| PDBQT hydrogen-like atoms | 514 |
| Minimum partial charge | -0.549 |
| Maximum partial charge | 0.345 |
| HETATM records | 0 |

Identical composition does not imply identical coordinates. All eight prepared
PDB and PDBQT hashes are distinct, preserving the selected MD geometry while
holding preparation rules constant.

## Output Integrity

- Preparation manifest SHA-256:
  `21DFAD19DB59B168C89D1369EFB540499ED68DF98E6B3EFF6C9EE26EC6A6EA78`
- Preparation summary SHA-256:
  `B3B3185C9F442CB1DE6995F75083E06526B7DB332DD7AEE4F868090AA93EB537`

## Decision

All eight PDBQT files pass the receptor-parameterization gate and may enter a
small docking execution test. The next experiment is limited to two actives and
two decoys across the eight receptors (32 receptor-ligand pairs). It verifies
the shared docking box, Vina execution, output parsing, and per-receptor score
matrix construction before the 100-active/100-decoy benchmark is expanded.

Successful receptor preparation does not establish pose quality, favorable
scores, enrichment, or biological validity.
