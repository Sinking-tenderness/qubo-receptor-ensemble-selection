# Stage 3 CDK2 AF2/MD 100A/100D E64 High-Risk Diagnostic Record

## Scope

The complete Stage 3 benchmark contains eight MD receptor conformers, 100
actives, 100 decoys, and three exhaustiveness-32 seed replicates per
receptor-ligand pair. The preregistered primary matrix is the median across the
three e32 scores; the minimum matrix is sensitivity only.

The e32 aggregate audit identified eight pairs with a minimum-to-median delta
greater than 1 kcal/mol. Those eight fixed pairs were rerun at exhaustiveness
64 with the same ligand-specific seeds. This diagnostic tests whether stronger
search reduces the observed seed instability. It does not replace any e32
matrix cell.

## Fixed Protocol

- Receptor-ligand pairs: 8
- Seeds per pair: 3
- Expected Vina runs: 24
- Successful Vina runs: 24
- Failed Vina runs: 0
- Vina: AutoDock Vina v1.2.7 Linux x86-64
- Exhaustiveness: 64
- Modes retained per run: 1
- Parallel layout: 8 workers x 4 CPUs
- Box center: `(0.52, 27.06, 8.97)` Angstrom
- Box size: `(18, 18, 16)` Angstrom
- Wall time: 957.022 seconds
- Sum of individual Vina runtimes: 6560.149 seconds

The executable, receptor manifest, ligand manifest, aggregate score table,
individual receptor PDBQT files, and individual selected ligand PDBQT files
were verified by SHA-256 before execution.

## Results

The acceptance rule required at least two negative scores, a median below zero,
a seed range no greater than 1 kcal/mol, and a minimum-to-median delta no
greater than 1 kcal/mol.

| Pair | Label | E32 median | E32 range | E64 median | E64 range | Favorable e64 seeds | Classification |
|---|---|---:|---:|---:|---:|---:|---|
| C00-D0016 | decoy | -1.082 | 10.880 | -2.965 | 4.224 | 2/3 | favorable but seed-variable |
| C00-D0023 | decoy | -5.270 | 23.972 | -5.270 | 23.972 | 2/3 | favorable but seed-variable |
| C00-D0093 | decoy | -4.917 | 3.154 | -5.601 | 0.274 | 3/3 | stable at e64 |
| C01-D0054 | decoy | -0.604 | 10.346 | -2.696 | 0.914 | 3/3 | stable at e64 |
| C03-A0088 | active | -6.571 | 9.103 | -6.571 | 9.103 | 2/3 | favorable but seed-variable |
| C03-D0016 | decoy | +7.417 | 88.905 | -3.841 | 11.391 | 2/3 | critical pair partly rescued |
| C06-D0046 | decoy | -7.084 | 1.482 | -8.457 | 1.353 | 3/3 | favorable but seed-variable |
| C07-A0048 | active | -7.337 | 1.239 | -8.715 | 0.178 | 3/3 | stable at e64 |

Three of eight pairs passed all stability thresholds. Four remained favorable
on the median but seed-variable. The critical C03-D0016 pair changed from one
negative e32 seed and a positive e32 median to two negative e64 seeds and a
negative e64 median. Its remaining positive seed and 11.391 kcal/mol range mean
that it was only partly rescued, not stabilized.

C00-D0023 and C03-A0088 reproduced all three e32 scores exactly at e64. This is
consistent with successful e64 execution that did not find a better pose for
those seed-specific searches. C06-D0046 improved only marginally and remained
above the 1 kcal/mol seed-range threshold.

## Interpretation

Increasing exhaustiveness from 32 to 64 is not a uniform solution to the seed
instability. It stabilized three selected pairs and repaired one catastrophic
seed for C03-D0016, but five of eight pairs still failed at least one stability
criterion. The evidence therefore does not justify recomputing the complete
matrix at e64.

The result also does not show that e32 is physically accurate. It supports a
narrower protocol decision: the uniform three-seed e32 median remains more
appropriate for the primary benchmark than selective e64 replacement or a
single best-seed/minimum score.

## Matrix Decision

1. Keep the original 1600-cell three-seed e32 median matrix unchanged as the
   primary score matrix.
2. Keep the original e32 minimum matrix as a prespecified sensitivity analysis.
3. Do not replace the eight diagnosed cells with e64 values.
4. Do not launch a complete e64 matrix solely on the basis of this diagnostic.
5. Preserve all 61 e32 seed warnings as reliability annotations; do not delete
   their ligands or receptors from downstream evaluation.

## Integrity

- Received archive SHA-256:
  `D52ED3B841BBE88CFC523042CD6F3CBC67EA3D9015A06D533C5F9EF4AF8883F6`
- E64 raw run table SHA-256:
  `F38C85FFC502D6CB48397440220C5B5D03F00B56CA4E1DE8EBEA23BFD996EBE0`
- E64 case summary SHA-256:
  `F3C5A229AD3D189189129F1A58CD588B3C65FF10ED096EE8C2D18D3F58A572EA`
- E64 summary SHA-256:
  `518FE15D587929082EA4FF1D0A27C79BDAAE1F2702F73D3E9B12194A0B9F8440`
- Diagnostic config SHA-256:
  `47E3EA10F99E9DEB3EDA9333B92FDDF8F80486E9565693BB6ACC397A80C81E46`
- Source e32 aggregate table SHA-256:
  `89B87DDAD1B26C50BD8AD980652089B6A14B7385B81416501C9DD5C972BC77BD`

The received archive contains the raw run table, case summary, and JSON
summary. Generated poses and Vina logs were not included in this archive and
remain remote run artifacts.

## Next Gate

The next analysis uses the unchanged full e32 median matrix with the locked,
scaffold-disjoint 100-active/100-decoy split: 60/60 train, 20/20 validation,
and 20/20 test. Receptor utility, redundancy, complementarity, ensemble
baselines, and QUBO parameters must be learned or selected from train and
validation only. The test split remains untouched until the selection rule is
fixed.

The minimum-score matrix will repeat the same evaluation as sensitivity. E64
diagnostic values will be reported only in this audit and will not enter either
screening matrix.
