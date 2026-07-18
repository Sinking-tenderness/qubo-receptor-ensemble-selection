# Stage 5 MAPK14 Expanded Receptor Structural Selection Record

Date: 2026-07-18

## Purpose and Boundary

The first four-receptor development gate selected only `MK14_3KQ7_aligned`
and was identical to the single-best baseline. This follow-up expands the
receptor pool before any new validation panel is sampled. Structural selection
may read receptor coordinates and co-crystal ligand locations only. It reads
no active/decoy labels, docking scores, consumed validation rows, or locked
test rows.

The original preregistration SHA-256 is
`8927604B67E803BBB86070876DF46403572CDFE53387A7C59FF3CA537064A2AE`.

## Discovery

The RCSB query returned 265 X-ray entries for UniProt Q16539. Metadata filters
retained 183 entries and left 179 new candidates after removing the four
existing receptors. Coordinate eligibility uses proper-rotation Kabsch
alignment to 2QD9 chain A, at least 300 matched C-alpha atoms, global aligned
RMSD at most 3 A, the frozen 29-residue pocket and anchors, and a qualifying
same-chain co-crystal ligand within 6 A of the reference pocket.

## Technical Failures

The v1 result was rejected before receptor preparation. `1IAN` has 328 protein
`ATOM` records, all named `CA`, and no backbone or side-chain atoms. The first
feature implementation therefore used each C-alpha coordinate as its fallback
side-chain centroid, making a non-dockable structure appear maximally distinct.
The same audit found backbone-complete `TPO180` and `PTR182` HETATM residues in
6ZQS; the frozen protein-only preparation would silently delete both.

Amendment 01 added all-atom density, pocket heavy-atom completeness, and
polymer-like HETATM checks. Its SHA-256 is
`82D2FDE0736E30AEE3D7D5984465399778A20F1ED52BBE8DA122FCC6D9E913AC`.

The resulting v2 selection was also rejected before Vina. Meeko found that
3UVR lacks seven side-chain heavy atoms in TYR35, six in ARG220, and four in
LYS267. Deleting these residues with `allow_bad_res` would introduce a
preparation artifact; TYR35 is also adjacent to the frozen pocket. Amendment
02 therefore required every present standard amino acid to contain its full
heavy-atom template. Its SHA-256 is
`D71CB03D274AEC9B51001409E8BEA684F67E6205CB03895EFC3FB3BD72B8894F`.

Neither amendment used ligand labels, docking scores, enrichment metrics, or
manual receptor substitution.

## Final V3 Selection

The cumulative v3 gate retained 39 of 179 new candidates. Together with the
four existing receptors, 43 structures entered the 812-feature matrix and
produced 903 pairwise distances. Deterministic max-min selection gave:

| Rank | Addition | Resolution (A) | Global C-alpha RMSD (A) | Pocket completeness | Co-crystal | Minimum distance |
|---:|---|---:|---:|---:|---|---:|
| 1 | MK14_2BAJ_aligned | 2.25 | 1.645 | 96.15% | 1PP A 401 | 1.649 |
| 2 | MK14_4F9W_aligned | 2.00 | 0.802 | 100.00% | LM4 A 403 | 1.432 |
| 3 | MK14_3OCG_aligned | 2.21 | 0.948 | 100.00% | OCG A 361 | 1.426 |
| 4 | MK14_3MPT_aligned | 1.89 | 0.828 | 100.00% | 1GK A 361 | 1.361 |

2BAJ lacks pocket residue 115, which is not a frozen anchor, and remains above
the 95% pocket heavy-atom threshold. Every selected v3 addition has zero
incomplete standard residues and zero polymer-like HETATM residues. In 4F9W,
the non-protein Zn/ACT/BME/GOL records are at least 14.2 A from selected ligand
LM4 and are removed by the common preparation protocol.

The final pool is the existing `2QD9`, `1A9U`, `3K3J`, and `3KQ7` receptors
plus `2BAJ`, `4F9W`, `3OCG`, and `3MPT`.

## Verification

An independent implementation reconstructed all four max-min choices and
distances from the 43-row eligible pool and 903-row distance table. A complete
second execution left the SHA-256 values of all six core outputs unchanged.

- v3 summary SHA-256: `94221AA0FAFF078C276D96C303932E5C905CBD7B1C3E24906B3DD63DE25A24AB`
- selected manifest SHA-256: `17769E5DEF367C7C5406B7AB75C670DEA151F54F2BC2FC9078C374FBEDB9F4FB`
- independent audit SHA-256: `D0E14D798272F14AA55C806E655292D8468E99D530EBA9943C920AF751256C09`

## Decision

The v3 eight-receptor structural pool passes the label-independent selection
gate and may enter co-crystal redocking. This result establishes a reproducible
structural candidate pool only; it does not establish enrichment,
complementarity, affinity accuracy, QUBO benefit, or quantum advantage.
