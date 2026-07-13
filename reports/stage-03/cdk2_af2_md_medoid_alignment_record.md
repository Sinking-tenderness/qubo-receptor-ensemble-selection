# Stage 3 CDK2 AF2 MD Medoid Alignment Record

## Scope

Eight MD cluster medoids were rigidly aligned to the 1AQ1 chain-A coordinate
frame. Existing MD hydrogens and all HETATM records were removed before any
new protonation, charge assignment, or PDBQT conversion.

## Input Integrity

- Clustering summary SHA-256:
  `E9CCB9998D3A6204602E5C1983C1F7B373E612B7B99ACEE7824B2C2ABDA164D5`
- Medoid manifest SHA-256:
  `AFEDEDDEE6CA5AF03F1E217AD01DD878BABA2E77915494ACBF744A72C7D81769`
- Reference 1AQ1 PDB SHA-256:
  `6EF163DDCF3E3298B6C0982DC8C06E89C61B9D6705FB34CED7E142A0313D764D`
- Alignment config SHA-256:
  `543265F61D5B2FDC9A7C78E846C6D08022BC3387CFFB935FFAE6FB93DD85223E`

## Structural Audit

- Medoids processed: 8
- Primary revisited candidates: 7
- Exploratory low-temporal-support candidates: 1
- Sequence-matched C-alpha atoms per alignment: 277
- Heavy protein atoms per output: 2,398
- Protein residues per output: 298
- Hydrogens per output: 0
- HETATM records per output: 0
- Residue-name mismatches in matched C-alpha atoms: 0
- Rotation determinant: 1.0 for every medoid

The 277 matched C-alpha atoms reflect residues available in both the complete
AF2-derived medoids and the experimentally unresolved portions of 1AQ1. The
missing reference residues are not silently rebuilt in 1AQ1.

## Alignment Metrics

| Medoid | Role | Global RMSD before (A) | Global RMSD after (A) | Pocket C-alpha RMSD (A) |
|---|---|---:|---:|---:|
| C00 F001 | exploratory low support | 3.082099 | 1.527985 | 1.331097 |
| C01 F005 | primary revisited | 3.105729 | 1.580139 | 1.159243 |
| C02 F031 | primary revisited | 3.046309 | 1.469348 | 1.318451 |
| C03 F041 | primary revisited | 3.130814 | 1.620530 | 1.204397 |
| C04 F066 | primary revisited | 3.135957 | 1.580387 | 1.491604 |
| C05 F074 | primary revisited | 3.074406 | 1.406226 | 1.392919 |
| C06 F077 | primary revisited | 3.153681 | 1.560344 | 1.475538 |
| C07 F095 | primary revisited | 3.068516 | 1.436778 | 1.354411 |

After rigid-body alignment, the global matched-C-alpha RMSD range is
1.406-1.621 A and the 21-residue pocket C-alpha RMSD range is 1.159-1.492 A.
These finite residual differences represent internal conformational variation,
not unremoved global translation or rotation.

## Output Integrity

- Alignment manifest SHA-256:
  `FDC78D913C126097C94EB4EEC5AFE2C27F1B95380AB989A351806EE69E0BB724`
- Deterministic alignment summary SHA-256:
  `620A9F0DEA0839D1BDFAECAFF0E9D9C9A17DF098EF23843B193B24F34D4B1220`

Two consecutive clustering and alignment regenerations produced identical
summary hashes. The hash comparison had no diff.

## Decision

All eight aligned heavy-atom PDBs pass the coordinate and composition audit.
They may be transferred to the established local receptor-preparation
environment. The next gate first parameterizes the primary C01 medoid. Only
after that single-receptor audit succeeds should the same Meeko settings be
applied to all eight structures.

This record does not claim that any medoid improves docking or enrichment.
