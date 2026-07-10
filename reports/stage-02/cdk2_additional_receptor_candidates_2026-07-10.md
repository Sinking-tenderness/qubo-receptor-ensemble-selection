# CDK2 Additional Receptor Candidate Audit

## Purpose

Add additional CDK2 conformers in the same 1AQ1-aligned coordinate frame before
multi-receptor docking. The raw structures are retained locally and are not
committed because the repository excludes raw molecular data.

## Sources

| PDB | Chain | Experimental resolution | Local raw SHA-256 | Decision |
|---|---:|---:|---|---|
| 1JVP | P | 1.53 A | `1B784017220012AF8B55D10F318DA7A233D5839823ED60D5CB062D323A030932` | Include after pilot docking |
| 1H00 | A | 1.60 A | `4327A6EEC4F43720D3167FDE1087243540A7ACB113F914B8028B76606E1184B7` | Exclude from current baseline |

Source pages: [RCSB 1JVP](https://www.rcsb.org/structure/1JVP) and
[RCSB 1H00](https://www.rcsb.org/structure/1H00). Accessed 2026-07-10.

## Alignment

Both structures were rigidly aligned to `receptors/raw/1AQ1.pdb`, chain A,
using sequence-matched C-alpha atoms and a Kabsch rotation. This puts the
future docking box in one shared coordinate frame.

| PDB | Mobile chain | Matched C-alpha | RMSD before | RMSD after | Rotation determinant |
|---|---:|---:|---:|---:|---:|
| 1JVP | P | 263 | 1.346 A | 0.748 A | +1.0 |
| 1H00 | A | 275 | 1.447 A | 0.667 A | +1.0 |

## Preparation audit

### 1JVP: included candidate

- Selected ProDy alternate location: `1`.
- Meeko default alternate location: `1`.
- `allow_bad_res`: false.
- Prepared receptor: `receptors/prepared/1JVP_P_aligned_receptor.pdbqt`.
- 281 residues, 2752 coordinate records, 487 hydrogen-like atoms.
- No HETATM records; AutoDock types are `A,C,HD,N,NA,OA,SA`.
- Charge range: `-0.549` to `0.345`.

### 1H00: excluded from current baseline

- Alternate location A was selected.
- Meeko required `allow_bad_res` and ignored 17 residues:
  `A:9,A:12,A:15,A:25,A:34,A:50,A:51,A:73,A:74,A:75,A:96,A:150,A:162,A:164,A:178,A:278,A:297`.
- The resulting PDBQT has 261 residues rather than the 278 residues in the
  protein-only input.
- Several ignored residues are close to the kinase-domain region used by the
  ATP-site pocket definition. Therefore this is not a controlled receptor
  preparation for the current comparison and will not be docked in the
  baseline.

## Next validation

Run the same 2-active/2-decoy pilot used for 1HCL against 1JVP, using the
shared 1AQ1 box and the same ligand manifest, seed rule, and Vina version.
Only after checking output completeness and score parsing should 1JVP be
expanded to all 60 ligands.
