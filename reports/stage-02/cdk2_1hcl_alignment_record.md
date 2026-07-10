# Stage 2 Receptor Alignment Record: CDK2 1HCL to 1AQ1

## Purpose

This module adds 1HCL chain A as the first candidate receptor conformer beyond
the validated 1AQ1 single-receptor baseline. The immediate goal is to place
1HCL in the 1AQ1 coordinate frame before transferring the fixed STU-derived
docking box.

This alignment does not establish that 1HCL improves virtual-screening
enrichment. It only establishes a reproducible common coordinate frame.

## Candidate Structure Audit

- Target: human CDK2
- Candidate PDB ID: 1HCL
- Chain: A
- State: apo
- Experimental method: X-ray diffraction
- Resolution: 1.80 A
- Mutation: none
- RCSB source: `https://files.rcsb.org/download/1HCL.pdb`
- Accessed: 2026-07-10
- Local raw path: `receptors/raw/1HCL.pdb`
- Raw size: 236925 bytes
- Raw SHA-256: `4C44CC9DE5EB61B5AD197F7D3058FA59327D7EA3C721C3CA3945CC8C3D41F9AE`
- ATOM records: 2372
- HETATM records: 180
- HETATM residue types: HOH only
- Missing residues: A:37-40

The missing residues are not members of the previously defined 1AQ1 STU 5 A
pocket-residue set. Indirect conformational effects have not been excluded.

## Alignment Method

Script:

```text
scripts/align_receptor_structure.py
```

Command:

```powershell
conda run -n qubo-receptor-ensemble python `
  .\scripts\align_receptor_structure.py `
  --reference .\receptors\raw\1AQ1.pdb `
  --mobile .\receptors\raw\1HCL.pdb `
  --output .\receptors\prepared\1HCL_A_aligned_to_1AQ1.pdb `
  --summary-output .\data\processed\1HCL_A_alignment_to_1AQ1_summary.json `
  --reference-chain A `
  --mobile-chain A
```

The script matches C-alpha atoms by chain, residue number, insertion code, and
residue name. A Kabsch rigid-body transform is calculated from all matched
C-alpha atoms and applied to every coordinate record in the mobile PDB.

No bond lengths, internal coordinates, or receptor conformational differences
are optimized by this operation.

## Alignment Result

- Reference: 1AQ1 chain A
- Mobile: 1HCL chain A
- Sequence-matched C-alpha atoms: 277
- Residue-name mismatches excluded: 0
- C-alpha RMSD before alignment: 133.188 A
- C-alpha RMSD after alignment: 0.666 A
- Rotation determinant: approximately +1.000
- Output ATOM records: 2372
- Output HETATM records: 180
- Aligned output SHA-256: `D60C8A340167A35D06FA9DD6059BE062AAA7DF54056FF8BCAA373C2E328A840F`

The earlier PyMOL result of 0.474 A over 250 atoms is not contradictory.
PyMOL's default `align` procedure iteratively rejects outliers, whereas this
pipeline retains all 277 sequence-matched C-alpha atoms under a fixed rule.

## Docking Box Transfer Check

The fixed box inherited from the validated 1AQ1-STU protocol is:

```text
center_x = 0.52
center_y = 27.06
center_z = 8.97
size_x = 18
size_y = 18
size_z = 16
```

After loading the saved aligned 1HCL coordinates without running another
alignment, the crystallographic 1AQ1 STU pose lies inside the aligned 1HCL ATP
pocket. Twenty-two 1HCL residues have at least one atom within 5 A of STU:

```text
ILE10 GLY11 GLU12 GLY13 VAL18 ALA31 LYS33 VAL64 PHE80 GLU81 PHE82
LEU83 HIS84 GLN85 ASP86 LYS89 GLN131 ASN132 LEU134 ALA144 ASP145 LEU148
```

This confirms that the numerical box is transferred to the intended binding
site. It does not prove that STU binds favorably to the apo conformation or
that 1HCL will improve active/decoy ranking.

## Water Handling

The 180 crystallographic waters were transformed with the protein so the
alignment step preserves the raw structure. The first controlled two-receptor
baseline will use the same water-removal rule as 1AQ1. Conserved structural
waters may be assessed later as a separate sensitivity experiment.

## Relevance to the Score Matrix and QUBO

The aligned structure can become the second receptor column only after it is
prepared with the same protonation, charge, atom-typing, and docking rules used
for 1AQ1. A shared coordinate frame and fixed box prevent coordinate placement
from becoming an uncontrolled variable in the future ligand-by-conformer score
matrix.
