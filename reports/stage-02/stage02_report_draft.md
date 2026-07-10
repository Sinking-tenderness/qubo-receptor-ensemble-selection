# Stage 2 Report Draft: Single-Receptor Docking And Virtual Screening Baseline

## 1. Research Context

The long-term research goal is to study QUBO-guided sparse receptor conformer
ensemble selection for improving early enrichment in flexible virtual
screening.

Stage 2 does not attempt to solve the QUBO problem yet. Instead, it establishes
a reproducible single-receptor docking and screening baseline. This baseline is
needed because future conformer selection will compare many receptor
conformations using the same ligand set, docking protocol, and evaluation
metrics.

The core question in this stage is:

```text
Can we prepare one CDK2 receptor structure, dock a small active/decoy ligand
set reproducibly, evaluate ranking quality, and inspect likely false positives?
```

## 2. Target And Structure

Target:

- Protein: Cyclin-dependent kinase 2
- Gene: CDK2
- Organism: Homo sapiens
- UniProt accession: P24941

Structure:

- PDB ID: 1AQ1
- Chain: A
- Experimental method: X-ray diffraction
- Resolution: 2.00 Angstrom
- Co-crystal ligand: STU, staurosporine

The 1AQ1-STU complex was selected as the first docking system because it is a
CDK2-ligand complex with a non-covalent small-molecule inhibitor and does not
introduce ATP phosphate chemistry or metal coordination complexity.

## 3. Software And Environment

Main environment:

- Operating system: Windows + PowerShell
- Conda environment: `qubo-receptor-ensemble`
- Python: 3.11.15
- RDKit: 2026.03.1
- Meeko: 0.7.1
- ProDy: 2.4.1
- AutoDock Vina: 1.2.7
- PyMOL: used separately for visual inspection

Vina executable:

```text
environment/bin/vina_1.2.7_win.exe
```

Vina SHA256:

```text
E0C4B2715E0C1A74F6E92D0F3BE0328AC97542EAFBC111E6B1EFAD897A73CCE5
```

## 4. Data Sources And Local Data Policy

Raw receptor and ligand inputs were downloaded locally and are not committed to
Git.

Raw PDB:

- Source URL: `https://files.rcsb.org/download/1AQ1.pdb`
- Local path: `receptors/raw/1AQ1.pdb`
- SHA256: `6EF163DDCF3E3298B6C0982DC8C06E89C61B9D6705FB34CED7E142A0313D764D`

Reference ligand:

- Source URL:
  `https://models.rcsb.org/v1/1AQ1/ligand?auth_asym_id=A&auth_seq_id=299&encoding=sdf`
- Local path: `ligands/raw/1AQ1_STU_A_299.sdf`
- Raw SHA256:
  `7086CC6FF54A582A9E8ACAAD593A42C7E00D0A123C7366C4510B395FAAC98E5F`
- Mol-block SHA256:
  `92179A671812942500C2CD8E94DFFAD2324F40AEDA92776767641F8F47BD0548`

DUD-E CDK2 data:

- Actives: 474 rows
- Decoys: 27850 rows
- Actives SHA256:
  `AEBFB8188B01D7E072FC5D14716D3049C603AAA55ADD4DF5F9720DB4F66A9E3D`
- Decoys SHA256:
  `765E5D2B0366085C3D3389007506A72A8EE25BE6C155915E5CE0615468B3341C`

Raw and processed datasets, prepared molecular files, docking poses, logs, and
generated metric tables are intentionally ignored by Git. The repository stores
scripts, configs, manifests, summaries, and reports.

## 5. Receptor And Ligand Preparation

Receptor preparation:

- Input: `receptors/raw/1AQ1.pdb`
- Chain: A
- Hetero atoms, STU, and crystallographic waters were removed for the prepared
  receptor used in this baseline.
- Prepared receptor: `receptors/prepared/1AQ1_A_receptor.pdbqt`
- Receptor PDBQT atom count: 2703
- HETATM count: 0
- STU/HOH count: 0
- AutoDock atom types: A, C, HD, N, NA, OA, SA
- Charge range: -0.549 to 0.345

Reference ligand preparation:

- Input: `ligands/raw/1AQ1_STU_A_299.sdf`
- Explicit-H SDF: `ligands/prepared/1AQ1_STU_A_299_explicitH.sdf`
- Prepared ligand: `ligands/prepared/1AQ1_STU_A_299.pdbqt`
- PDBQT atom lines: 37
- TORSDOF: 2

Ligand preparation for the DUD-E subset:

- Sampled subset: 10 actives and 50 decoys
- RDKit parse OK: 60 / 60
- Unique canonical SMILES: 60
- Multi-fragment molecules: 0
- Charged ligands: 12 decoys
- Heavy atom count range: 18-37
- Molecular weight range: 249.358-513.034
- 3D SDF generation: 59 OK, 1 warning, 0 failed
- Warning ligand: `CDK2_D0036`, `MMFF94_not_converged_code_1`
- Meeko PDBQT generation: 60 OK, 0 failed
- TORSDOF range: 2-13

The warning ligand was retained and flagged instead of being silently removed.

## 6. Redocking Validation

The docking box was derived from the co-crystal STU coordinates:

| parameter | value |
|---|---:|
| center_x | 0.52 |
| center_y | 27.06 |
| center_z | 8.97 |
| size_x | 18 |
| size_y | 18 |
| size_z | 16 |

Vina parameters:

- scoring function: vina
- exhaustiveness: 16
- num_modes: 10
- seed: 20260708

Redocking result:

- Output poses: 10
- Best docking score: -13.87 kcal/mol
- Heavy-atom RMSD vs co-crystal STU pose: 0.225 Angstrom
- Heavy atoms compared: 35

Interpretation:

The redocking result supports that receptor preparation, ligand preparation,
box placement, and Vina execution can reproduce the co-crystal pose for this
single ligand and single receptor structure.

This does not prove that Vina scores are true binding free energies or that
virtual screening enrichment will be high.

## 7. Batch Docking Baseline

Batch docking used the same receptor, box, and exhaustiveness as the redocking
baseline.

Input:

- Receptor ID: `CDK2_1AQ1_A_prepared`
- Ligand subset: 10 actives and 50 decoys
- Base seed: 20260709

Batch docking result:

- Selected ligands: 60
- Successful ligands: 60
- Failed ligands: 0
- Score table rows: 599
- Pose-rank-1 rows: 60
- Pose-rank-10 rows: 59

The score table has 599 rows instead of 600 because one ligand produced only
9 modes within the Vina output constraints. This is not a docking failure.

Best-pose score summary:

| label | count | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|---:|
| active | 10 | -9.589 | 1.458 | -12.320 | -9.114 | -7.547 |
| decoy | 50 | -8.767 | 1.000 | -10.980 | -8.821 | -6.661 |

Top 10 ligands by best Vina score:

| rank | ligand_id | label | docking_score |
|---:|---|---|---:|
| 1 | CDK2_A0009 | active | -12.32 |
| 2 | CDK2_A0010 | active | -11.06 |
| 3 | CDK2_D0022 | decoy | -10.98 |
| 4 | CDK2_A0003 | active | -10.69 |
| 5 | CDK2_D0037 | decoy | -10.64 |
| 6 | CDK2_D0013 | decoy | -10.41 |
| 7 | CDK2_A0001 | active | -10.33 |
| 8 | CDK2_D0036 | decoy | -10.20 |
| 9 | CDK2_D0049 | decoy | -10.10 |
| 10 | CDK2_D0016 | decoy | -10.06 |

## 8. Virtual Screening Metrics

For ranking metrics, Vina scores were converted as:

```text
ranking_score = -docking_score
```

This is necessary because Vina scores are usually better when they are more
negative, while ranking metrics usually assume larger scores indicate stronger
positive predictions.

Point estimates:

| metric | value |
|---|---:|
| ROC-AUC | 0.644 |
| PR-AUC / average precision | 0.459 |
| BEDROC(alpha=20) | 0.653 |
| EF1% | 6.0 |
| EF5% | 4.0 |
| EF10% | 3.0 |

Bootstrap 95% confidence intervals:

| metric | mean | 95% CI low | 95% CI high |
|---|---:|---:|---:|
| ROC-AUC | 0.645 | 0.411 | 0.840 |
| PR-AUC / average precision | 0.460 | 0.134 | 0.748 |
| BEDROC(alpha=20) | 0.611 | 0.056 | 0.920 |
| EF1% | 5.821 | 0.000 | 12.000 |
| EF5% | 4.490 | 0.000 | 8.571 |
| EF10% | 3.307 | 0.714 | 6.250 |

Interpretation:

The point estimates suggest better-than-random ranking on this small teaching
subset. However, the confidence intervals are wide. EF1% is especially unstable
because 1% of 60 ligands corresponds to only the top-ranked ligand.

This baseline should be treated as a reproducible teaching result, not as a
strong research-level claim.

## 9. Top-Hit And False-Positive Analysis

Top 10:

- Actives: 4
- Decoys: 6

Top 20:

- Actives: 4
- Decoys: 16

Top-ranked decoys selected for inspection:

- `CDK2_D0022`: rank 3, high cLogP
- `CDK2_D0036`: rank 8, large ligand, high cLogP, preparation warning
- `CDK2_D0049`: rank 9, high cLogP
- `CDK2_D0016`: rank 10, formal charge -1
- `CDK2_D0042`: rank 11, high TORSDOF

Structural inspection focused on:

- Active controls: `CDK2_A0009`, `CDK2_A0010`
- Suspicious decoys: `CDK2_D0022`, `CDK2_D0036`

PyMOL visual inspection:

- `CDK2_A0009` and `CDK2_A0010` enter the STU reference pocket.
- `CDK2_D0022` and `CDK2_D0036` also enter the pocket, but show portions that
  extend outside the reference-ligand region.
- `CDK2_D0022` is visually the more suspicious decoy because it has a large
  extended region and an orientation pattern that differs from the other
  inspected ligands.

Geometry-based contact analysis:

| ligand_id | label | contact residues | polar candidates | hydrophobic candidates | possible clashes |
|---|---|---:|---:|---:|---:|
| CDK2_A0009 | active | 16 | 3 | 36 | 0 |
| CDK2_A0010 | active | 15 | 2 | 30 | 0 |
| CDK2_D0022 | decoy | 16 | 4 | 29 | 0 |
| CDK2_D0036 | decoy | 16 | 2 | 32 | 0 |

Closest residue patterns:

- `CDK2_A0009`: HIS84, LEU83, PHE80, ILE10, ASP145, GLU81
- `CDK2_A0010`: HIS84, LEU83, ILE10, ALA31, ASP145, GLN131
- `CDK2_D0022`: THR14, GLU12, ASN132, GLN131, PHE80
- `CDK2_D0036`: ASP86, LEU83, PHE82, GLU8, ASP145

Interpretation:

`CDK2_D0022` is not suspicious because it fails to enter the pocket. It is
suspicious because it occupies the pocket with a distinct extended geometry and
a shifted contact pattern. It should be treated as a candidate false positive,
not as a proven inactive molecule.

## 10. Reproducibility And Git Checkpoints

Key committed workflow checkpoints:

| commit | message |
|---|---|
| 1f76179 | feat: add target selection and redocking workflow |
| b641eb3 | feat: add ligand preparation pipeline |
| 7dacb9b | feat: add batch docking workflow |
| 2f059f9 | feat: add virtual screening metrics |
| 626960f | feat: add uncertainty metrics for screening |
| 5a140cc | feat: add top hit analysis workflow |
| 723be90 | feat: add pose inspection workflow |
| f494801 | feat: add pose contact analysis |

Generated files intentionally not committed:

- raw DUD-E files
- processed ligand CSV files
- prepared ligand SDF/PDBQT files
- prepared receptor PDBQT files
- docking poses
- Vina logs
- generated metrics CSV/JSON
- PyMOL pose-inspection files

## 11. Limitations

This stage has several important limitations:

- The ligand set is small: 10 actives and 50 decoys.
- DUD-E decoys are not experimentally confirmed inactive compounds.
- Only one receptor conformation, CDK2 1AQ1 chain A, was used.
- Vina scores are not experimental binding free energies.
- Protonation and tautomer states were not exhaustively enumerated.
- Receptor flexibility was not modeled inside a single docking run.
- Structural water, cofactors, and alternative receptor preparation choices were
  not systematically compared.
- EF1%, BEDROC, and other early-recognition metrics have wide uncertainty.
- Top-ranked decoys may be scoring artifacts, plausible binders, or dataset
  artifacts; this baseline cannot distinguish these cases definitively.

## 12. Connection To The Next Stage

This baseline prepares the interface needed for receptor conformer ensemble
selection.

The current score table has the long-table format:

```text
target_id
receptor_id
ligand_id
label
pose_rank
docking_score
status
runtime_seconds
seed
software_version
```

The next stage should repeat the same protocol for multiple CDK2 receptor
conformations:

```text
ligand_id x receptor_conformer_id -> docking_score
```

This matrix can support:

- single-conformer performance comparison
- all-conformer minimum-score baseline
- mean or consensus-score baseline
- random conformer subset baseline
- clustering-based conformer subset baseline
- greedy conformer subset baseline
- QUBO-guided sparse receptor conformer subset selection

The future QUBO objective should not optimize only a tiny-set EF1%. It should
consider early enrichment, uncertainty, conformer complementarity, redundancy,
failure rates, and structural plausibility.

## 13. Stage 2 Takeaway

Stage 2 successfully established a reproducible single-receptor docking and
virtual screening baseline for CDK2 1AQ1.

The protocol can reproduce the co-crystal STU pose, dock a small active/decoy
set, compute ranking metrics, quantify uncertainty, and inspect suspicious
top-ranked decoys at the pose/contact level.

The result is not a claim that docking is accurate. It is a controlled baseline
for the next research question:

```text
Can multiple receptor conformations, selected sparsely and explainably, improve
early enrichment while avoiding redundant or artifact-prone conformers?
```
